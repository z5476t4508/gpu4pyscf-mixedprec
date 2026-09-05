"""Batch DF-RHF single-point driver for large XYZ collections.

Reads one molecule per ``.xyz`` file from an input directory, runs a
GPU DF-RHF single point for each, and stores the total energy plus the AO
density matrix into a per-shard HDF5 file that can be resumed after a crash.

The screening lane (the default) runs every SCF iteration with float32
contractions (``precision_mode='fp32'``, ``conv_tol=3e-5``): ~1e-4 Eh absolute
error but the molecule ranking is preserved.  Use ``--mode auto`` for the
mixed-precision lane (fp32 early iterations, fp64 tail) when full float64
accuracy is required.

Do not tighten ``--conv-tol`` below ~3e-5 in fp32 mode.  The float32 noise
floor is ~4e-5, so a tighter threshold buys no accuracy at all and instead
makes the iteration count unstable: measured on Azadirachtin (934 AO),
conv_tol=1e-5 needs 9-38 iterations run to run (up to 18.8 s) against 7-12
(max 12.2 s) at 3e-5, for the same ~3.3e-4 Eh error.

Run against a gpu4pyscf build that has ``gpu4pyscf.lib.precision`` (the
mixed-precision branch).  The released 1.8.1 wheel does not, so point
PYTHONPATH at this repository::

    PYTHONPATH=/home/tong/soft/gpu4pyscf python -m batch_rhf \
        --input-dir mols/ --output-dir results/

Charge and spin are read from the XYZ comment line (``charge=-1 spin=0``),
falling back to ``--charge`` / ``--spin``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback

import h5py
import numpy as np

SCHEMA_VERSION = 1

# status codes stored in the 'status' dataset
PENDING = 0
RUNNING = 1
COMPLETED = 2
FAILED = 3

_STATUS_NAMES = {PENDING: 'pending', RUNNING: 'running',
                 COMPLETED: 'completed', FAILED: 'failed'}

# attributes that must match exactly when resuming into an existing shard file
_CONFIG_KEYS = ('basis', 'auxbasis', 'mode', 'conv_tol', 'max_cycle',
                'default_charge', 'default_spin', 'shard_index', 'num_shards',
                'cderi_precision')


class BatchError(RuntimeError):
    '''Fatal batch-level error (bad input, incompatible resume target).'''


# --------------------------------------------------------------------------
# input parsing
# --------------------------------------------------------------------------

def parse_comment_metadata(comment):
    '''Extract ``charge=`` / ``spin=`` tokens from an XYZ comment line.

    Returns a dict with the keys that were present.  Unknown tokens are
    ignored so ordinary titles ("Tamoxifen") parse to an empty dict.
    '''
    meta = {}
    for token in comment.replace(',', ' ').split():
        if '=' not in token:
            continue
        key, _, value = token.partition('=')
        key = key.strip().lower()
        if key not in ('charge', 'spin'):
            continue
        try:
            meta[key] = int(value.strip())
        except ValueError:
            raise BatchError(
                f'invalid {key}={value!r} in XYZ comment line') from None
    return meta


def parse_xyz(path, default_charge=0, default_spin=0):
    '''Parse a single-molecule standard XYZ file.

    Layout: atom count, comment line, then one ``symbol x y z`` line per atom.
    Returns ``{'atom': <pyscf atom string>, 'natm', 'charge', 'spin'}``.
    '''
    with open(path) as f:
        lines = f.read().splitlines()

    # drop trailing blank lines; keep interior ones so a truncated file is
    # reported as a count mismatch rather than silently accepted
    while lines and not lines[-1].strip():
        lines.pop()

    if len(lines) < 2:
        raise BatchError(f'{path}: XYZ file needs an atom count and a comment line')

    try:
        natm = int(lines[0].strip())
    except ValueError:
        raise BatchError(f'{path}: first line must be the atom count, got {lines[0]!r}') from None
    if natm <= 0:
        raise BatchError(f'{path}: atom count must be positive, got {natm}')

    body = lines[2:]
    if len(body) != natm:
        raise BatchError(
            f'{path}: header declares {natm} atoms but the file has {len(body)} coordinate lines')

    atoms = []
    for lineno, line in enumerate(body, start=3):
        fields = line.split()
        if len(fields) < 4:
            raise BatchError(f'{path}:{lineno}: expected "symbol x y z", got {line!r}')
        symbol = fields[0]
        try:
            x, y, z = (float(v) for v in fields[1:4])
        except ValueError:
            raise BatchError(f'{path}:{lineno}: coordinates are not numeric: {line!r}') from None
        atoms.append(f'{symbol} {x!r} {y!r} {z!r}')

    meta = parse_comment_metadata(lines[1])
    return {
        'atom': '\n'.join(atoms),
        'natm': natm,
        'charge': meta.get('charge', default_charge),
        'spin': meta.get('spin', default_spin),
    }


def iter_xyz_files(input_dir):
    '''Return ``[(molecule_id, path), ...]`` sorted by the stable id.

    The id is the path relative to ``input_dir`` so nested directories stay
    distinguishable and the ordering is reproducible across runs.
    '''
    input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise BatchError(f'input directory does not exist: {input_dir}')

    found = {}
    for dirpath, dirnames, filenames in os.walk(input_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.lower().endswith('.xyz'):
                continue
            path = os.path.join(dirpath, name)
            mol_id = os.path.relpath(path, input_dir)
            if mol_id in found:  # pragma: no cover - os.walk cannot repeat a path
                raise BatchError(f'duplicate molecule id {mol_id!r}')
            found[mol_id] = path

    if not found:
        raise BatchError(f'no .xyz files found under {input_dir}')
    return [(mol_id, found[mol_id]) for mol_id in sorted(found)]


def shard_tasks(tasks, shard_index, num_shards):
    '''Split the stable task list into ``num_shards`` contiguous blocks.'''
    if num_shards < 1:
        raise BatchError(f'--num-shards must be >= 1, got {num_shards}')
    if not 0 <= shard_index < num_shards:
        raise BatchError(f'--shard-index must be in [0, {num_shards}), got {shard_index}')

    n = len(tasks)
    base, extra = divmod(n, num_shards)
    start = shard_index * base + min(shard_index, extra)
    stop = start + base + (1 if shard_index < extra else 0)
    return tasks[start:stop]


def fingerprint(ids):
    '''Stable digest of the shard's molecule ids, to detect changed input.'''
    digest = hashlib.sha256()
    for mol_id in ids:
        digest.update(mol_id.encode())
        digest.update(b'\0')
    return digest.hexdigest()


# --------------------------------------------------------------------------
# HDF5 storage
# --------------------------------------------------------------------------

class ShardStore:
    '''Per-shard HDF5 result file with crash-safe completion marking.

    Density matrices are stored ragged: every completed molecule appends
    ``nao*nao`` float32 values to ``dm_data`` and records its start offset.
    A task is marked ``completed`` only after its payload has been flushed, so
    an interrupted write can never be mistaken for a finished molecule.
    '''

    def __init__(self, path, ids, config):
        self.path = path
        self.ids = list(ids)
        self.config = dict(config)
        self._f = None

    def __enter__(self):
        if os.path.exists(self.path):
            self._open_existing()
        else:
            self._create()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._f is not None:
            self._f.close()
            self._f = None
        return False

    def _create(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)) or '.', exist_ok=True)
        n = len(self.ids)
        vlen_str = h5py.string_dtype(encoding='utf-8')
        f = h5py.File(self.path, 'w')
        f.attrs['schema_version'] = SCHEMA_VERSION
        f.attrs['fingerprint'] = fingerprint(self.ids)
        for key in _CONFIG_KEYS:
            f.attrs[key] = self.config[key]

        f.create_dataset('id', data=np.array(self.ids, dtype=object), dtype=vlen_str)
        f.create_dataset('status', data=np.full(n, PENDING, dtype=np.uint8))
        f.create_dataset('energy', data=np.full(n, np.nan, dtype=np.float64))
        f.create_dataset('nao', data=np.full(n, -1, dtype=np.int32))
        f.create_dataset('natm', data=np.full(n, -1, dtype=np.int32))
        f.create_dataset('dm_offset', data=np.full(n, -1, dtype=np.int64))
        f.create_dataset('converged', data=np.zeros(n, dtype=np.uint8))
        f.create_dataset('cycles', data=np.full(n, -1, dtype=np.int32))
        f.create_dataset('wall_time', data=np.full(n, np.nan, dtype=np.float64))
        f.create_dataset('error_type', shape=(n,), dtype=vlen_str)
        f.create_dataset('error_message', shape=(n,), dtype=vlen_str)
        f.create_dataset('dm_data', shape=(0,), maxshape=(None,), dtype=np.float32,
                         chunks=(1 << 16,), compression='lzf')
        f.flush()
        self._f = f

    def _open_existing(self):
        f = h5py.File(self.path, 'a')
        try:
            version = int(f.attrs.get('schema_version', -1))
            if version != SCHEMA_VERSION:
                raise BatchError(
                    f'{self.path}: schema version {version} != {SCHEMA_VERSION}; '
                    'delete the file or point --output-dir somewhere else')
            if f.attrs.get('fingerprint') != fingerprint(self.ids):
                raise BatchError(
                    f'{self.path}: the input molecule list changed since this shard was '
                    'created; delete the file or point --output-dir somewhere else')
            for key in _CONFIG_KEYS:
                stored = f.attrs.get(key)
                wanted = self.config[key]
                if isinstance(wanted, float):
                    same = stored is not None and float(stored) == wanted
                else:
                    same = stored is not None and type(wanted)(stored) == wanted
                if not same:
                    raise BatchError(
                        f'{self.path}: config mismatch for {key!r} '
                        f'(file has {stored!r}, run wants {wanted!r})')
        except Exception:
            f.close()
            raise

        # a 'running' entry means the previous run died mid-molecule
        status = f['status'][:]
        stale = status == RUNNING
        if stale.any():
            status[stale] = PENDING
            f['status'][:] = status
            f.flush()
        self._f = f

    # -- accessors ---------------------------------------------------------

    def status(self):
        return self._f['status'][:]

    def mark_running(self, index):
        self._f['status'][index] = RUNNING
        self._f.flush()

    def commit_success(self, index, energy, dm, natm, converged, cycles, wall_time):
        '''Write the payload, flush, and only then mark the task completed.'''
        f = self._f
        dm = np.ascontiguousarray(dm, dtype=np.float32)
        nao = dm.shape[0]
        flat = dm.reshape(-1)

        data = f['dm_data']
        offset = data.shape[0]
        data.resize((offset + flat.size,))
        data[offset:offset + flat.size] = flat

        f['energy'][index] = energy
        f['nao'][index] = nao
        f['natm'][index] = natm
        f['dm_offset'][index] = offset
        f['converged'][index] = 1 if converged else 0
        f['cycles'][index] = cycles
        f['wall_time'][index] = wall_time
        f.flush()

        f['status'][index] = COMPLETED
        f.flush()

    def commit_failure(self, index, error_type, error_message, wall_time):
        f = self._f
        f['error_type'][index] = error_type
        f['error_message'][index] = error_message[:4096]
        f['wall_time'][index] = wall_time
        f.flush()

        f['status'][index] = FAILED
        f.flush()


def read_result(path, mol_id):
    '''Read back one molecule's stored result (energy + trimmed AO DM).'''
    with h5py.File(path, 'r') as f:
        ids = [x.decode() if isinstance(x, bytes) else x for x in f['id'][:]]
        try:
            index = ids.index(mol_id)
        except ValueError:
            raise KeyError(f'{mol_id!r} is not in {path}') from None

        status = int(f['status'][index])
        out = {'id': mol_id, 'status': _STATUS_NAMES[status]}
        if status == COMPLETED:
            nao = int(f['nao'][index])
            offset = int(f['dm_offset'][index])
            dm = f['dm_data'][offset:offset + nao * nao]
            out.update(
                energy=float(f['energy'][index]),
                dm=np.asarray(dm, dtype=np.float32).reshape(nao, nao),
                natm=int(f['natm'][index]),
                converged=bool(f['converged'][index]),
                cycles=int(f['cycles'][index]),
                wall_time=float(f['wall_time'][index]),
            )
        elif status == FAILED:
            error_type = f['error_type'][index]
            error_message = f['error_message'][index]
            out.update(
                error_type=error_type.decode() if isinstance(error_type, bytes) else error_type,
                error_message=(error_message.decode()
                               if isinstance(error_message, bytes) else error_message),
            )
        return out


# --------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------

def preflight():
    '''Import the compute stack once, so a broken environment fails fast.

    Without this every molecule in the shard would be recorded as an
    individual failure for what is really a single setup problem.
    '''
    try:
        import cupy  # noqa: F401
        import pyscf  # noqa: F401
        from gpu4pyscf import scf  # noqa: F401
    except Exception as exc:
        raise BatchError(f'cannot import the GPU stack: {exc}') from exc

    try:
        from gpu4pyscf.lib import precision  # noqa: F401
    except ImportError as exc:
        import gpu4pyscf
        raise BatchError(
            f'gpu4pyscf at {os.path.dirname(gpu4pyscf.__file__)} has no lib.precision '
            'module, so --mode fp32/auto would silently run in float64. Point '
            'PYTHONPATH at the mixed-precision checkout of this repository.') from exc


def run_molecule(spec, basis, auxbasis, mode, conv_tol, max_cycle,
                 cderi_precision='fp64'):
    '''Run one DF-RHF single point.  Returns (e_tot, dm_float32, meta).

    The global precision mode is restored afterwards: gpu4pyscf's SCF kernel
    sets it on entry but does not reset it if the SCF raises, which would
    otherwise leak fp32 into the next molecule.
    '''
    import cupy
    import pyscf
    from gpu4pyscf import scf
    from gpu4pyscf.lib import precision

    if spec['spin'] != 0:
        raise BatchError(
            f"RHF requires a closed-shell molecule, got spin={spec['spin']}")

    try:
        precision.set_cderi_precision(cderi_precision)
        mol = pyscf.M(atom=spec['atom'], basis=basis, charge=spec['charge'],
                      spin=spec['spin'], verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
        mf.precision_mode = mode
        mf.conv_tol = conv_tol
        mf.max_cycle = max_cycle
        mf.chkfile = None
        e_tot = mf.kernel()
        dm = cupy.asnumpy(mf.make_rdm1()).astype(np.float32)
        meta = {'converged': bool(mf.converged), 'cycles': int(mf.cycles),
                'natm': int(mol.natm)}
    finally:
        precision.set_precision('fp64')
        precision.set_cderi_precision('fp64')

    return float(e_tot), dm, meta


def run_batch(tasks, store, basis, auxbasis, mode, conv_tol, max_cycle,
              default_charge, default_spin, retry_failed=True,
              cderi_precision='fp64', log=print):
    '''Run every unfinished task in the shard, committing as we go.'''
    status = store.status()
    done = 0
    failed = 0
    skipped = 0

    for index, (mol_id, path) in enumerate(tasks):
        state = status[index]
        if state == COMPLETED or (state == FAILED and not retry_failed):
            skipped += 1
            continue

        store.mark_running(index)
        t0 = time.time()
        try:
            spec = parse_xyz(path, default_charge, default_spin)
            e_tot, dm, meta = run_molecule(spec, basis, auxbasis, mode,
                                           conv_tol, max_cycle, cderi_precision)
        except ImportError:
            # an environment problem, not a property of this molecule: abort
            # instead of burning through the shard marking everything failed
            store.commit_failure(index, 'ImportError', traceback.format_exc(),
                                 time.time() - t0)
            raise
        except Exception as exc:
            wall = time.time() - t0
            store.commit_failure(index, type(exc).__name__,
                                 f'{exc}\n{traceback.format_exc()}', wall)
            failed += 1
            log(f'[{index + 1}/{len(tasks)}] {mol_id}: FAILED ({type(exc).__name__}: {exc})')
        else:
            wall = time.time() - t0
            store.commit_success(index, e_tot, dm, meta['natm'],
                                 meta['converged'], meta['cycles'], wall)
            done += 1
            flag = '' if meta['converged'] else '  [NOT CONVERGED]'
            log(f'[{index + 1}/{len(tasks)}] {mol_id}: E={e_tot:.8f}  '
                f'{wall:.2f}s  {meta["cycles"]} iters{flag}')

    return {'completed': done, 'failed': failed, 'skipped': skipped}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        description='Batch DF-RHF single points (energy + AO density matrix) for XYZ files')
    p.add_argument('--input-dir', required=True,
                   help='directory scanned recursively for .xyz files')
    p.add_argument('--output-dir', required=True,
                   help='directory for the per-shard HDF5 result files')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-svp-jkfit',
                   help='J/K fitting basis; do not substitute an RI/MP2 basis here')
    p.add_argument('--mode', default='fp32', choices=('fp32', 'auto', 'fp64'),
                   help='fp32: screening lane; auto: fp32 start then fp64 tail; fp64: reference')
    p.add_argument('--conv-tol', type=float, default=3e-5,
                   help='SCF convergence; below ~3e-5 the fp32 lane gains no accuracy '
                        'and its iteration count becomes unstable')
    p.add_argument('--max-cycle', type=int, default=50)
    p.add_argument('--cderi-precision', default='fp64', choices=('fp64', 'fp32'),
                   help='fp32 builds the CDERI tensor in float32: 1.8-2.8x faster '
                        'overall, but the pairwise energy-gap error grows from '
                        '~0.09 to ~0.9 kcal/mol. Only for coarse screening')
    p.add_argument('--charge', type=int, default=0,
                   help='default charge when the XYZ comment does not set one')
    p.add_argument('--spin', type=int, default=0,
                   help='default spin (2S) when the XYZ comment does not set one')
    p.add_argument('--shard-index', type=int, default=0)
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--no-retry-failed', action='store_true',
                   help='leave previously failed molecules alone instead of recomputing them')
    p.add_argument('--dry-run', action='store_true',
                   help='list the shard contents and exit without computing')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.cderi_precision == 'fp32' and args.mode != 'fp32':
        print(f"error: --cderi-precision fp32 cannot be combined with --mode {args.mode}; "
              'the float32 CDERI is not accurate enough for a float64 result',
              file=sys.stderr)
        return 2

    if args.mode == 'fp32' and args.conv_tol < 3e-5:
        print(f'warning: conv_tol={args.conv_tol:g} is below the fp32 noise floor (~4e-5). '
              'It buys no accuracy and makes the iteration count unstable; '
              'use --mode auto if you need tighter convergence.', file=sys.stderr)

    all_tasks = iter_xyz_files(args.input_dir)
    tasks = shard_tasks(all_tasks, args.shard_index, args.num_shards)
    ids = [mol_id for mol_id, _ in tasks]

    print(f'{len(all_tasks)} molecules total, {len(tasks)} in shard '
          f'{args.shard_index}/{args.num_shards}')

    if args.dry_run:
        for mol_id, path in tasks:
            print(f'  {mol_id}\t{path}')
        return 0

    if not tasks:
        print('nothing to do for this shard')
        return 0

    preflight()

    config = {
        'basis': args.basis,
        'auxbasis': args.auxbasis,
        'mode': args.mode,
        'conv_tol': float(args.conv_tol),
        'max_cycle': int(args.max_cycle),
        'default_charge': int(args.charge),
        'default_spin': int(args.spin),
        'shard_index': int(args.shard_index),
        'num_shards': int(args.num_shards),
        'cderi_precision': args.cderi_precision,
    }
    out_path = os.path.join(
        args.output_dir, f'results-{args.shard_index:05d}-of-{args.num_shards:05d}.h5')

    t0 = time.time()
    with ShardStore(out_path, ids, config) as store:
        stats = run_batch(tasks, store, args.basis, args.auxbasis, args.mode,
                          args.conv_tol, args.max_cycle, args.charge, args.spin,
                          retry_failed=not args.no_retry_failed,
                          cderi_precision=args.cderi_precision)
    elapsed = time.time() - t0

    print(f'\n{out_path}')
    print(f'completed {stats["completed"]}, failed {stats["failed"]}, '
          f'already done {stats["skipped"]}  ({elapsed:.1f}s)')
    return 1 if stats['failed'] else 0


if __name__ == '__main__':
    sys.exit(main())

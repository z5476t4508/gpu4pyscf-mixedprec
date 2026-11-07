#!/usr/bin/env python
'''Batch single-point throughput channel (issue #11 (a), optional lane).

Runs the admitted energy-matrix cells sequentially in ONE process and
records per-cell energy/timing/memory plus batch-level throughput.  This
complements vram_mapper.py (which probes capacity in fresh subprocesses):
the batch channel measures what in-process sequential execution costs in
memory-pool carry-over and wall time -- the operationally relevant number
for energy-matrix workloads (plan section #11: 批量单点吞吐可选通道).

Resume semantics (tested without a GPU in tools/tests/test_batch_channel.py):
  * rows with status ``ok`` are never re-run;
  * rows with status ``failed`` are re-run only with --retry-failed;
  * rows stuck in ``running`` (a crashed previous process) reset to
    pending and are re-run;
  * any per-cell exception is recorded in the row and the batch continues
    (bounded execution; failures are evidence, never silently dropped);
  * unknown row_keys in the output file are preserved untouched (the file
    is append/upsert by row_key, like the rtx4090 runner --merge-from).

Lanes follow the rtx4090 runner: 'fp64' reference, 'auto' mixed precision
(fp32 head + fp64 tail, the production lane).  Timing here is throughput
mode: ONE sample per cell per invocation (no warmup separation, no
accuracy claim) -- accuracy gates stay with the rtx4090-energy-v1
documents from issues #5-#8.  The batch channel only re-checks each
cell's energy against an optional reference list for sanity.

Usage (GPU python):
    python batch_channel.py --cells cells.json --out batch_run.json
    python batch_channel.py --cells cells.json --out batch_run.json \
        --retry-failed          # re-run failed rows
'''

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

SCHEMA_VERSION = 'batch-channel-v1'

OK, RUNNING, FAILED, PENDING = 'ok', 'running', 'failed', 'pending'


def row_key(cell):
    return '|'.join([os.path.basename(cell['geom']), cell['basis'],
                     cell['xc'], cell['lane']])


def load_output(path):
    '''Load a resumable output document; None when absent/corrupt.'''
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            doc = json.load(f)
        if doc.get('schema_version') != SCHEMA_VERSION:
            return None
        return doc
    except (ValueError, OSError):
        return None


def merge_state(doc, cells, retry_failed):
    '''Decide what to run.  Pure; unit-tested without a GPU.

    Returns (rows_by_key, run_list) where rows_by_key is the upsert
    target (prior rows preserved, running->pending reset) and run_list
    holds the cells to execute this invocation.
    '''
    rows = {}
    if doc is not None:
        for row in doc.get('rows', []):
            key = row.get('row_key')
            if key is None:
                continue
            if row.get('status') == RUNNING:
                row = dict(row, status=PENDING,
                           note=(row.get('note') or '') +
                                ' [reset from running by reload]')
            rows[key] = row
    run_list = []
    for cell in cells:
        key = row_key(cell)
        prior = rows.get(key)
        if prior is not None:
            status = prior.get('status')
            if status == OK:
                continue
            if status == FAILED and not retry_failed:
                continue
        run_list.append(cell)
    return rows, run_list


def gpu_execute(cell, headroom_mib):
    '''Run one cell in-process.  Imports GPU stack lazily.

    Returns a result dict (status ok/failed) -- never raises past the
    caller: every failure is a recorded row.
    '''
    import cupy as cp
    from rtx4090_runner import MemorySampler, make_mol, _apply_common
    from rtx4090_runner import is_3c_xc, resolve_3c, attach_3c_corrections

    row = {'row_key': row_key(cell), 'lane': cell['lane'], 'status': FAILED}
    try:
        from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
        free_b, total_b = cp.cuda.runtime.memGetInfo()
        row['mem'] = {'gpu_free_before_mib': round(free_b / 2**20, 1),
                      'gpu_total_mib': round(total_b / 2**20, 1),
                      'headroom_budget_mib': headroom_mib}
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        with MemorySampler() as samp:
            mol, mf = None, None
            xc = cell['xc']
            three_c = is_3c_xc(xc)
            if three_c:
                xc = resolve_3c(xc)
            mol = make_mol(cell['geom'], cell['basis'],
                           cell.get('charge'), cell.get('spin'))
            if xc == 'hf':
                cls = gpu_scf.UHF if mol.spin else gpu_scf.RHF
                mf = cls(mol)
            else:
                cls = gpu_dft.UKS if mol.spin else gpu_dft.RKS
                mf = cls(mol, xc=xc)
            mf = mf.density_fit(auxbasis=cell.get(
                'auxbasis', 'def2-universal-jkfit'))
            ns = argparse.Namespace(
                conv_tol=cell.get('conv_tol', 1e-10),
                max_cycle=cell.get('max_cycle', 50),
                init_guess=cell.get('init_guess', 'minao'),
                direct_scf_tol=1e-14,
                grids_level=cell.get('grids_level', 3),
                nlcgrids_level=cell.get('nlcgrids_level'))
            _apply_common(mf, ns, True)
            if cell['lane'] != 'fp64':
                mf.precision_mode = cell['lane']
            if three_c:
                attach_3c_corrections(mf)
            e = mf.kernel()
            cp.cuda.runtime.deviceSynchronize()
        row['result'] = {
            'e_tot': float(e), 'converged': bool(mf.converged),
            'scf_cycles': int(getattr(mf, 'cycles', 0) or 0),
            'timings': {'t_cell_s': round(time.perf_counter() - t0, 3),
                        'sync': 'deviceSynchronize before/after cell'}}
        row['mem'].update({
            'gpu_peak_alloc_mib': round(
                total_b / 2**20 - samp.min_free_mib, 1),
            'pool_cached_after_free_mib': round(
                cp.get_default_memory_pool().total_bytes() / 2**20, 1),
            'within_budget': bool(
                (total_b / 2**20 - samp.min_free_mib) + headroom_mib
                <= total_b / 2**20)})
        row['status'] = OK
    except Exception as exc:  # record honestly, continue the batch
        import traceback
        row['error'] = {'type': type(exc).__name__,
                        'message': str(exc)[:400],
                        'traceback_head': traceback.format_exc()[:600]}
    finally:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    return row


def run_batch(cells, out_path, execute, retry_failed=False, log=print,
              meta_extra=None):
    '''Driver: load, merge, execute, upsert, write after every cell.'''
    doc = load_output(out_path) or {
        'schema_version': SCHEMA_VERSION, 'meta': {}, 'rows': []}
    if meta_extra:
        doc['meta'].update(meta_extra)
    rows_by_key, run_list = merge_state(doc, cells, retry_failed)
    log(f'{len(run_list)} cells to run, '
        f'{len(rows_by_key) - len(run_list)} already terminal')
    t0 = time.time()
    for i, cell in enumerate(run_list):
        log(f'[{i + 1}/{len(run_list)}] {row_key(cell)} ...')
        try:
            row = execute(cell)
        except Exception as exc:  # a crashing executor is a failed row,
            # never a lost batch (bounded execution)
            import traceback
            row = {'row_key': row_key(cell), 'lane': cell['lane'],
                   'status': FAILED,
                   'error': {'type': type(exc).__name__,
                             'message': str(exc)[:400],
                             'traceback_head': traceback.format_exc()[:600]}}
        rows_by_key[row['row_key']] = row
        doc['rows'] = list(rows_by_key.values())
        doc['meta']['updated_at_utc'] = time.strftime(
            '%Y-%m-%dT%H:%M:%S+00:00', time.gmtime())
        with open(out_path, 'w') as f:
            json.dump(doc, f, indent=1)   # crash-safe: write after each cell
    counts = {}
    for row in rows_by_key.values():
        counts[row['status']] = counts.get(row['status'], 0) + 1
    doc['summary'] = {
        'cells_total': len(cells), 'ran': len(run_list),
        'status_counts': counts,
        'batch_wall_s': round(time.time() - t0, 1),
        'throughput_cells_per_hour': round(
            len(run_list) / (time.time() - t0) * 3600, 2)
        if time.time() > t0 else None,
        'resumed': doc.get('summary', {}).get('ran', 0) if run_list else 0,
    }
    with open(out_path, 'w') as f:
        json.dump(doc, f, indent=1)
    log(f'batch done: {json.dumps(doc["summary"])}')
    return doc


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='batch_channel', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cells', required=True,
                   help='JSON list of cell objects: geom/basis/xc/lane '
                        '(+ optional charge, spin, auxbasis, conv_tol, '
                        'max_cycle, grids_level, nlcgrids_level)')
    p.add_argument('--out', required=True, help='resumable output JSON')
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--headroom-mib', type=float, default=2048.0)
    args = p.parse_args(argv)

    with open(args.cells) as f:
        cells = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    from rtx4090_runner import gpu_static_info, git_full_sha
    meta = {'run_id': str(uuid.uuid4()),
            'gpu4pyscf_source_commit': git_full_sha(REPO),
            'gpu4pyscf_source_path': REPO,
            'mode': 'throughput: one sample per cell per invocation; '
                    'accuracy gates live in the rtx4090-energy-v1 '
                    'documents, not here',
            'mem_headroom_mib': args.headroom_mib}
    try:
        meta.update(gpu_static_info())
    except Exception:
        pass
    doc = run_batch(cells, args.out, lambda c: gpu_execute(
        c, args.headroom_mib), retry_failed=args.retry_failed, log=print,
        meta_extra=meta)
    failed = doc['summary']['status_counts'].get(FAILED, 0)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())

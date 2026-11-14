#!/usr/bin/env python3
"""Four-arm upstream comparison runner (control card 2026-10-18).

One invocation = ONE ARM of ONE BLOCK of ONE CELL of the balanced-blocks-v1
protocol: a fresh independent process that builds the cell from the frozen
schedule, runs its own separated warmup, then exactly ONE formal sample,
and writes an immutable sample record.  The batch driver (run_upstream_batch.py)
spawns this script 16 times per cell (4 blocks x 4 arms, Latin-square order,
seed-frozen schedule) and merges the four per-role sample sets into the
rtx4090-gradient-v1 comparison documents the operator reporter consumes.

Roles (control card §四路计算):
  upstream        O  unmodified official GPU4PySCF 1.8.1 from the read-only
                      RECORD-verified package; no precision_mode (it does
                      not exist there), no project knobs, no repo probe.
  optimized_fp64  B  our source with the complete tested opt-in stack,
                      SCF mode=fp64.
  optimized_auto  C  the identical stack and settings as B, SCF mode=auto.
  strict_fp64     R  our source with every approximate knob OFF,
                      SCF/gradient fp64 -- the accuracy anchor; the real
                      probe must show zero FP32 entry.

O-source isolation (control card §原版参考已就绪): the upstream arm scrubs
the repo root from sys.path, installs a fail-closed meta_path finder that
resolves gpu4pyscf* ONLY under the official package root, and re-verifies
after the run that every loaded gpu4pyscf module file is inside that root
and (for .py/.so files listed in the reference manifest) byte-identical to
the RECORD-verified hashes.  The shared construction (rtx4090_runner/
gradient_runner: same mol/mf/DF/grids/conv settings, same 3c D4+gCP hooks)
is imported only AFTER the guard is in place, so its own sys.path.insert
can never shadow the official package -- the guard intercepts every
gpu4pyscf import first.  run_gpu_grad_cell is never used for O (it
force-installs the mixedprec precision probe, which does not exist in the
official package); O calls _gpu_grad_point(..., scf_mode=None, grad_mode=None,
probe=None) directly, exactly as the card prescribes.

Timing protocol (control card §配对预注册): per (block, arm) process --
independent warmup separated from one formal sample; all four formal
samples per role (one per block) are kept; the Latin square balances arm
order across blocks.  Ratios are always reference_time/candidate_time.
"""

import argparse
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from types import MethodType, SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
# the shared runner modules (gradient_runner/rtx4090_runner/schema_validator)
# are imported as top-level modules from this directory by every caller
sys.path.insert(0, HERE)

SCHEMA_VERSION = 'rtx4090-gradient-v1'
SAMPLE_SCHEMA = 'upstream-comparison-sample-v1'
TIMING_PROTOCOL = 'balanced-blocks-v1'
ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')
# benchmark v1 (2026-10-30): the S route is the gradient_runner
# 'auto-strict' lane (auto SCF policy + in-run strict-fp64 reference
# matching, ALL of its verification/matching/fallback work inside the
# timed sample -- it never borrows C speedups); the S batch pairs it
# against the same optimized_fp64 arm the main schedule uses.
ROLE_AUTO_STRICT = 'auto_strict'
ROLES_S = (ROLE_AUTO_STRICT, 'optimized_fp64')
# schedule schema strings (kept in sync with upstream_schedule.py; not
# imported from it to keep the arm module import-light)
SCHEMA_V1 = 'upstream-comparison-schedule-v1'
SCHEMA_V2 = 'upstream-comparison-schedule-v2'
SCHEMA_S = 'upstream-comparison-schedule-s-v1'
ROLE_TO_LANE = {'upstream': 'fp64', 'optimized_fp64': 'fp64',
                'optimized_auto': 'auto', 'strict_fp64': 'fp64',
                ROLE_AUTO_STRICT: 'auto-strict'}
# roles that run under the complete tested opt-in stack (B/C/S); O/R run
# with every approximate knob off
STACKED_ROLES = ('optimized_fp64', 'optimized_auto', ROLE_AUTO_STRICT)


def roles_for_schedule(schedule):
    '''The comparison roles a schedule drives: the O/B/C/R quartet for
    balanced-blocks v1/v2, the S/B pair for the s-paired protocol.  A
    four-block schedule NEVER acquires S arms and an S schedule never
    acquires O/C/R -- the schema routes, nothing else.'''
    schema = schedule.get('schema_version')
    if schema in (SCHEMA_V1, SCHEMA_V2):
        return ROLES
    if schema == SCHEMA_S:
        return ROLES_S
    raise ValueError(f'unknown schedule schema {schema!r}')
# the complete tested opt-in stack (LC-103/105/115/121/122 recommended
# configuration; the exact STACK_ENV of the LC-170/202 drivers)
STACK_ENV = {
    'GPU4PYSCF_XC_GRID_BLOCKSIZE': '16384',
    'GPU4PYSCF_VV10_UWE_KERNEL': 'fp32',
    'GPU4PYSCF_DF_IP1_FUSED': '1',
    'GPU4PYSCF_DF_IP1_F32': '1',
    'GPU4PYSCF_GRAD_GRID_F32': '1',
    'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
}
OFFICIAL_ROOT = '/home/qczcode/gpu4pyscf-official-1.8.1'
OFFICIAL_CERTIFICATE = '/opt/qc-agent/upstream-reference.json'
PRECISION_MODE = {'upstream': 'official-default', 'optimized_fp64': 'fp64',
                  'optimized_auto': 'auto', 'strict_fp64': 'fp64',
                  ROLE_AUTO_STRICT: 'auto+strict'}
SAMPLE_RECORD_KEYS = ('schema_version', 'batch_id', 'schedule_sha256',
                      'cell_index', 'role', 'block', 'position', 'lane',
                      'optimization_env', 'row', 'startup', 'provenance',
                      'command', 'recorded_at_utc')


# ---------------------------------------------------------------------------
# official-package source isolation (upstream arm only)
# ---------------------------------------------------------------------------

class OfficialOnlyFinder:
    '''meta_path finder placed FIRST for the upstream arm: every import of
    gpu4pyscf or a submodule resolves ONLY under the official package root;
    a gpu4pyscf name absent from the official tree raises ImportError
    instead of falling through to sys.path / editable finders (fail-closed:
    a mixed official+repo import is a hard error, never a silent shadow).'''

    def __init__(self, official_root):
        self.root = os.path.abspath(official_root)

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'gpu4pyscf' and not fullname.startswith('gpu4pyscf.'):
            return None
        if fullname in sys.modules:
            return None
        base = os.path.join(self.root, *fullname.split('.'))
        if os.path.isdir(base):
            init = os.path.join(base, '__init__.py')
            if not os.path.isfile(init):
                raise ImportError(
                    f'{fullname}: official package directory {base} has no '
                    '__init__.py; refusing any other source')
            return importlib_spec(fullname, init, [base])
        for suffix in ('.py', '.so'):
            candidate = base + suffix
            if os.path.isfile(candidate):
                return importlib_spec(fullname, candidate)
        raise ImportError(
            f'{fullname} not present in the official reference {self.root}; '
            'refusing to import it from anywhere else (fail-closed '
            'isolation for the upstream arm)')


def importlib_spec(fullname, path, search=None):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        fullname, path, submodule_search_locations=search)
    if spec is None or spec.loader is None:
        raise ImportError(f'cannot build an import spec for {fullname} '
                          f'from {path}')
    return spec


def install_official_isolation(official_root):
    '''Prepare this process so that gpu4pyscf can ONLY come from the official
    read-only package.  Must run before ANY import of the shared runner
    modules (they insert the repo root into sys.path at import time).'''
    sys.dont_write_bytecode = True
    # refuse with NO side effects first: every precondition that can fail
    # is checked before sys.path/sys.meta_path are touched (a refusal that
    # leaves the finder half-installed would poison the process -- the
    # SP-18 lazy gate-constant import surfaced exactly this leak)
    leaked = {k: v for k, v in os.environ.items()
              if k.startswith('GPU4PYSCF_')}
    if leaked:
        raise RuntimeError(f'upstream arm refuses to run with inherited '
                           f'GPU4PYSCF_* variables: {leaked}')
    if not os.path.isdir(os.path.join(official_root, 'gpu4pyscf')):
        raise RuntimeError(f'official root {official_root} has no '
                           'gpu4pyscf/ package directory')
    contaminated = [n for n in sys.modules
                    if n == 'gpu4pyscf' or n.startswith('gpu4pyscf.')]
    if contaminated:
        raise RuntimeError('gpu4pyscf already imported before isolation '
                           f'was installed: {sorted(contaminated)}')
    # drop any repo-root path entries (PYTHONPATH, -m cwd effects); keep
    # THIS script's directory (the shared benchmark modules live there)
    root_abs = os.path.abspath(REPO)
    dropped = [p for p in sys.path if os.path.abspath(p or os.getcwd()) == root_abs]
    if dropped:
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.getcwd()) != root_abs]
    # fail-closed guard FIRST in meta_path: runs before PathFinder and any
    # editable-install finder, for every gpu4pyscf* import
    sys.meta_path.insert(0, OfficialOnlyFinder(official_root))
    # the official root is the directory CONTAINING gpu4pyscf/ (the
    # RECORD-verified package root's parent layout:
    # /home/qczcode/gpu4pyscf-official-1.8.1/gpu4pyscf/...); putting it on
    # sys.path makes 'import gpu4pyscf' find the official package
    if official_root not in sys.path:
        sys.path.insert(0, official_root)
    return {'repo_paths_dropped': dropped,
            'official_root': os.path.abspath(official_root)}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_official_provenance(official_root, manifest):
    '''After the run: every loaded gpu4pyscf module must live under the
    official root; every one of its files that the reference manifest lists
    must be byte-identical.  Also hashes the libraries actually mapped from
    the official root (the ctypes-loaded libxc.so included).  Returns the
    provenance inventory for the sample record, or raises.'''
    root_abs = os.path.abspath(official_root)
    bad, loaded = [], {}
    for name, mod in sorted(sys.modules.items()):
        if name != 'gpu4pyscf' and not name.startswith('gpu4pyscf.'):
            continue
        path = getattr(mod, '__file__', None)
        if path is None:
            continue
        path = os.path.abspath(path)
        if not path.startswith(root_abs + os.sep):
            bad.append((name, path))
        else:
            # manifest keys are relative to the official package root
            # (the directory CONTAINING gpu4pyscf/), e.g.
            # 'gpu4pyscf/df/df.py'
            loaded[os.path.relpath(path, root_abs)] = \
                sha256_file(path)
    if bad:
        raise RuntimeError(f'non-official gpu4pyscf modules loaded: {bad}')
    mismatch = []
    for rel, digest in sorted(loaded.items()):
        want = manifest['files'].get(rel)
        if want is not None and want != digest:
            mismatch.append(rel)
        if want is None and rel.endswith('.py'):
            mismatch.append(rel + ' (not in reference manifest)')
    if mismatch:
        raise RuntimeError(f'official package files drifted from the '
                            f'RECORD-verified manifest: {mismatch}')
    mapped = {}
    try:
        with open('/proc/self/maps', encoding='utf-8') as f:
            for line in f:
                path = line.rstrip('\n').rsplit(' ', 1)[-1]
                if path.startswith(root_abs + os.sep) and path not in mapped:
                    try:
                        mapped[path] = sha256_file(path)
                    except OSError:
                        mapped[path] = 'unreadable'
    except OSError:
        pass
    return {'gpu4pyscf_package_root': root_abs,
            'loaded_module_files': {rel: loaded[rel] for rel in sorted(loaded)},
            'mapped_library_files': {p: mapped[p] for p in sorted(mapped)},
            'manifest_checked_files': manifest.get('checked_files'),
            'verified_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                             time.gmtime())}


def load_official_manifest():
    with open(os.path.join(OFFICIAL_ROOT, 'reference-manifest.json'),
              encoding='utf-8') as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# settings shared by all four arms (identical construction discipline)
# ---------------------------------------------------------------------------

def namespace_for(cell, settings, role):
    '''The shared runner namespace, mirroring gradient_runner's argparse
    defaults exactly (verified against its parser) -- all four arms use
    the same numbers so the settings blocks come out identical.'''
    return SimpleNamespace(
        out='', geom=cell['geom'], basis=cell['basis'], xc=cell['xc'],
        auxbasis=cell.get('auxbasis', 'def2-universal-jkfit'),
        lane=[ROLE_TO_LANE[role]], samples=1,
        conv_tol=settings['conv_tol'], max_cycle=settings['max_cycle'],
        init_guess=settings['init_guess'],
        direct_scf_tol=settings['direct_scf_tol'],
        grids_level=settings['grids_level'],
        nlcgrids_level=settings.get('nlcgrids_level'),
        headroom_mib=settings['headroom_mib'],
        min_free_mib=cell.get('min_free_mib'),
        pilot_stage=cell.get('pilot_stage', 'S2'),
        charge=cell.get('charge'), spin=cell.get('spin'),
        dispersion=None, gcp=None, merge_from=None, energy_gate_doc=None,
        evidence_root='/data/results', no_validate=True)


def apply_3c_setup(args, energy_runner, log):
    '''gradient_runner.main's r2SCAN-3c whole-method setup, verbatim in
    effect: prescribed-basis guard, D4 dispersion block + gCP auto-fill,
    CPU D4/gCP probe.  Shared by all arms (settings-level, no GPU).'''
    if energy_runner.is_3c_xc(args.xc):
        if args.basis.lower() != energy_runner.R2SCAN3C_PRESCRIBED_BASIS:
            raise RuntimeError(
                f"r2SCAN-3c rows must use the prescribed basis "
                f"{energy_runner.R2SCAN3C_PRESCRIBED_BASIS}, got "
                f"{args.basis!r}")
        import importlib.metadata
        disp_version = importlib.metadata.version('pyscf-dispersion')
        args.dispersion = energy_runner.r2scan3c_dispersion_block(disp_version)
        args.gcp = energy_runner.R2SCAN3C_GCP_METHOD
        probe_mol = energy_runner.make_mol(args.geom, args.basis,
                                           args.charge, args.spin)
        carrier = SimpleNamespace(mol=probe_mol, scf_summary={})
        energy_runner._r2scan3c_get_dispersion(carrier)
        log('r2SCAN-3c mode: D4+gCP probe OK '
            f'(D4={carrier.scf_summary["dispersion_d4"]:.6e} '
            f'gCP={carrier.scf_summary["gcp"]:.6e} Ha)')
    libxc_xc = energy_runner.resolve_3c(args.xc) or args.xc
    from pyscf.dft import libxc as cpu_libxc
    if bool(cpu_libxc.is_nlc(libxc_xc)):
        from gpu4pyscf.dft import libxc as gpu_libxc
        if not gpu_libxc.is_nlc(libxc_xc):
            raise RuntimeError(f'{libxc_xc!r} is NLC on pyscf libxc but '
                               'not on the gpu4pyscf libxc wheel; refusing '
                               'to run')
        log(f'NLC guard OK: {libxc_xc!r} is_nlc on both libxc stacks')


# ---------------------------------------------------------------------------
# the upstream arm (O): _gpu_grad_point with probe=None, official package
# ---------------------------------------------------------------------------

def run_upstream_point(geom, basis, xc, auxbasis, args, gradient_runner,
                       energy_runner):
    '''One O-lane measurement: the shared _gpu_grad_point construction with
    scf_mode=None, grad_mode=None, probe=None -- no precision_mode is ever
    set on the official objects (it does not exist there), no probe wraps
    anything.  The 3c gradient hook is the SAME shared D4+gCP MethodType
    the production runner attaches; the official grad/rhf.py kernel
    dispatches it via base.do_disp() exactly like our tree.'''
    return gradient_runner._gpu_grad_point(geom, basis, xc, auxbasis, args,
                                           None, None, probe=None)


def _official_timing_block(gradient_runner, warmup_s, phases):
    '''The O-lane arm's aggregate timing block from the SAME measured
    per-repeat phases that sum to the sample total.  This call-site
    historically forwarded only scf/grad and dropped t_cderi_s/t_mol_s,
    emitting the pre-repair 0.0 placeholder in per-arm upstream records
    (batchv1 2026-10-30-obcr arms; the merged role docs and the stats
    tool read repeats, so they were never affected) -- the exact
    phase-misattribution class the benchmark v1 protocol bans.'''
    return gradient_runner._grad_timing_block(
        warmup_s, [sum(phases.values())],
        [phases['t_scf_s']], [phases['t_grad_s']], 'fp64',
        cderi_times=[phases['t_cderi_s']],
        mol_times=[phases['t_mol_s']])


def run_official_cell(geom, basis, xc, args, gradient_runner, energy_runner,
                      log):
    '''run_gpu_grad_cell's structure for the O lane, without the probe and
    without any precision attribute: headroom gate, cderi admission gate,
    memory sampler, one separated warmup, ONE formal sample, worst-repeat
    aggregate -- one row in the standard schema shape.'''
    import cupy as cp
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': 'fp64'}
    if args.min_free_mib is not None and \
            free_mib - args.headroom_mib < args.min_free_mib:
        return energy_runner._skipped_row(row_key, args, 'headroom-budget',
                                           free_mib, total_mib), None
    admitted, gate = energy_runner.cderi_admission_gate(
        geom, basis, args.auxbasis, args, free_mib, total_mib)
    if admitted is False:
        reason = gate['skip_reason'] or 'admission-gate'
        row = energy_runner._skipped_row(row_key, args,
                                         reason.split(':')[0], free_mib,
                                         total_mib)
        row['gate'] = gate
        return row, None
    sampler = energy_runner.MemorySampler()
    try:
        with sampler:
            try:
                log('upstream: warmup ...')
                warm = run_upstream_point(geom, basis, xc, args.auxbasis,
                                          args, gradient_runner,
                                          energy_runner)
                warmup_s = sum(warm[8].values())
            except Exception as exc:
                return energy_runner._failed_row(
                    row_key, args, exc, sampler, free_mib, total_mib,
                    'fp64', log), None
            try:
                log('upstream: formal sample ...')
                (e, converged, cycles, nlc, summary, grad, _gm, _gh,
                 phases) = run_upstream_point(geom, basis, xc, args.auxbasis,
                                              args, gradient_runner,
                                              energy_runner)
            except Exception as exc:
                return energy_runner._failed_row(
                    row_key, args, exc, sampler, free_mib, total_mib,
                    'fp64', log), None
            repeat = gradient_runner._build_grad_repeat(
                0, e, converged, cycles, grad, 'fp64', 0,
                energy_runner._e_terms(summary, nlc),
                {'switched': False, 'cycle': None, 'trigger': 'none'},
                summary, phases)
            gradient_runner.validate_grad_repeats([repeat])
    finally:
        pass
    peak_alloc = max(free_mib - (sampler.min_free_mib or free_mib), 0.0)
    within, expl = energy_runner._budget_check(peak_alloc, total_mib,
                                                args.headroom_mib)
    mol, mf = energy_runner._gpu_mol_mf(geom, basis, xc, True,
                                        args.auxbasis, args, None)
    row = {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'system': energy_runner._system_block(geom, mol),
        'method': energy_runner._method_block(mol, mf, xc, basis,
                                               args.auxbasis, True, args),
        'status': 'ok', 'skip_reason': None, 'gate': gate,
        'result': {
            'e_tot': repeat['e_tot'],
            'converged': repeat['converged'], 'scf_cycles': repeat['scf_cycles'],
            'gradient': repeat['gradient'],
            'gradient_norm': repeat['gradient_norm'],
            'gradient_mode': 'fp64',
            'repeats': [repeat],
            'repeat_evidence': 'all-repeats',
            'e_terms': repeat.get('e_terms'),
            'timings': _official_timing_block(
                gradient_runner, warmup_s, phases),
            'mem': gradient_runner._grad_mem_block(sampler, free_mib,
                                                   total_mib,
                                                   args.headroom_mib,
                                                   peak_alloc, within, expl),
            # the official package has no project probe; an empty dict is the
            # honest record (never a fabricated probe), per the control card
            'fp32_entry_hits': {},
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none',
                                 'note': 'official GPU4PySCF has no '
                                         'precision_mode / project probe'},
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    if gate is not None and gate.get('admitted') is True:
        gate['measured_gpu_peak_alloc_mib'] = round(peak_alloc, 1)
    log(f'upstream: ok E={e:.12f} conv={converged} cyc={cycles} '
        f'peak={peak_alloc:.0f}MiB')
    return row, {'warmup_s': warmup_s}


# ---------------------------------------------------------------------------
# one arm of one block
# ---------------------------------------------------------------------------

def optimization_env_of(role):
    '''The exact optimization variables this role must run under; {} for
    O/R (all approximate knobs off), the complete tested stack for B/C/S.'''
    if role in ('upstream', 'strict_fp64'):
        return {}
    if role in STACKED_ROLES:
        return dict(STACK_ENV)
    raise ValueError(role)


def check_env_discipline(role):
    '''Fail-closed knob discipline: O/R must see no GPU4PYSCF_* variables at
    all; B/C must see exactly the stack (no extras, none missing).'''
    seen = {k: v for k, v in os.environ.items() if k.startswith('GPU4PYSCF_')}
    want = optimization_env_of(role)
    if seen != want:
        raise RuntimeError(f'{role} arm environment discipline violated: '
                           f'want {want}, saw {seen}')
    return {'seen': seen}


def run_arm(role, cell, settings, batch_id, schedule_sha256, out_path,
            official_root=OFFICIAL_ROOT, source_commit=None,
            source_root=None):
    t_start = time.perf_counter()
    lines = []

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        lines.append(line)

    command = ' '.join([sys.executable or 'python', os.path.abspath(__file__)]
                       + sys.argv[1:])
    env_record = check_env_discipline(role)
    provenance = {}
    manifest = None
    if role == 'upstream':
        provenance['isolation'] = install_official_isolation(official_root)
        manifest = load_official_manifest()

    # the shared runner modules (import AFTER the official guard is in
    # place for the upstream arm; their module-level sys.path.insert of
    # the repo root is then harmless -- the guard intercepts every
    # gpu4pyscf import)
    sys.path.insert(0, HERE)
    import rtx4090_runner as energy_runner          # noqa: E402
    import gradient_runner                          # noqa: E402

    args = namespace_for(cell, settings, role)
    apply_3c_setup(args, energy_runner, log)

    if role == 'upstream':
        row, warm = run_official_cell(cell['geom'], cell['basis'],
                                      cell['xc'], args, gradient_runner,
                                      energy_runner, log)
    else:
        row = gradient_runner.run_gpu_grad_cell(
            cell['geom'], cell['basis'], cell['xc'], ROLE_TO_LANE[role],
            args, log)
        warm = None
        tr = (row.get('result') or {}).get('timings') or {}
        warm = {'warmup_s': tr.get('warmup_s')}

    provenance['gpu4pyscf_package_root'] = (
        os.path.dirname(os.path.abspath(__import__('gpu4pyscf').__file__))
        if role == 'upstream' else
        os.path.join(os.path.abspath(REPO), 'gpu4pyscf'))
    if role == 'upstream':
        provenance['official_verification'] = verify_official_provenance(
            official_root, manifest)
        provenance['reference_manifest_sha256'] = sha256_file(
            OFFICIAL_CERTIFICATE)
    else:
        # the frozen source provenance comes from the schedule's batch
        # section (the driver freezes it into the schedule at snapshot
        # time); the arm records it verbatim, never re-derives it
        provenance['source_commit'] = source_commit
        provenance['source_root'] = source_root or os.path.abspath(REPO)

    meta_base = energy_runner.build_meta(args)
    if role == 'upstream':
        # the flat source fields describe the OFFICIAL wheel, not the
        # snapshot the arm script happens to run from
        meta_base['gpu4pyscf_source_commit'] = 'official-1.8.1-wheel'
        meta_base['gpu4pyscf_source_path'] = official_root
        meta_base['libxc'] = None  # env-level libxc recorded in dependencies
    record = {
        'schema_version': SAMPLE_SCHEMA,
        'batch_id': batch_id,
        'schedule_sha256': schedule_sha256,
        'cell_index': cell['index'],
        'role': role,
        'block': cell['_block'],
        'position': cell['_position'],
        'lane': ROLE_TO_LANE[role],
        'optimization_env': env_record['seen'],
        'row': row,
        'startup': {'t_import_and_setup_s': round(
                        time.perf_counter() - t_start, 6),
                    'warmup_s': (warm or {}).get('warmup_s')},
        'provenance': provenance,
        'command': command,
        'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                         time.gmtime()),
    }
    missing = [k for k in SAMPLE_RECORD_KEYS if k not in record]
    if missing:
        raise RuntimeError(f'incomplete sample record (missing {missing})')
    meta_base['dependencies'] = dependency_versions()
    record['meta_base'] = meta_base
    out_dir = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    log_path = os.path.splitext(out_path)[0] + '.runlog.log'
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    if row.get('status') != 'ok':
        log(f'row status {row.get("status")!r} retained honestly')
        return 1
    return 0


def dependency_versions():
    '''The five unified-dependency versions at distribution level (the same
    /home/qczcode/gpu4090-src environment for all four arms, so the dicts
    are identical by construction; the driver verifies).'''
    import importlib.metadata as md
    out = {}
    for key, dist in (('pyscf', 'pyscf'), ('cupy', 'cupy-cuda12x'),
                      ('numpy', 'numpy'), ('libxc', 'gpu4pyscf-libxc-cuda12x'),
                      ('cutensor', 'cutensor-cu12')):
        try:
            out[key] = md.version(dist)
        except md.PackageNotFoundError:
            out[key] = None
    return out


# ---------------------------------------------------------------------------
# merge: 4 sample records per role -> one rtx4090-gradient-v1 comparison doc
# ---------------------------------------------------------------------------

def merge_role(samples, role, batch_id, schedule, schedule_sha256):
    '''Build one comparison document from the role's block samples.
    The expected count is PROTOCOL-ROUTED, never guessed: a v1 schedule
    is the hard-wired four-block contract exactly as before (old
    fixtures and old schedules are never reinterpreted); benchmark v1
    and the S protocol take EIGHT blocks from the schedule's own cell
    shape (two Latin squares / eight S-B pairs).'''
    schema = schedule.get('schema_version')
    if schema in (SCHEMA_V2, SCHEMA_S):
        n_blocks = len(schedule['cells'][0]['blocks'])
    else:
        n_blocks = 4
    blocks = [s['block'] for s in samples]
    if len(samples) != n_blocks or \
            sorted(blocks) != list(range(n_blocks)):
        raise ValueError(f'{role}: expected {n_blocks} blocks '
                         f'(schedule {schema}), got {blocks}')
    protocol = schedule.get('timing_protocol', TIMING_PROTOCOL)
    base = samples[0]
    deps = base['meta_base']['dependencies']
    for s in samples[1:]:
        if s['meta_base']['dependencies'] != deps:
            raise ValueError(f'{role}: dependency drift across blocks')
        for field in ('gpu_name', 'driver_version', 'cuda_version',
                      'cpu_model', 'python', 'cupy', 'pyscf'):
            if s['meta_base'].get(field) != base['meta_base'].get(field):
                raise ValueError(f'{role}: meta field {field} drifted '
                                 f'across blocks')
    rows = [s['row'] for s in samples]
    for r in rows[1:]:
        if r['system'] != rows[0]['system'] or \
                r['method'] != rows[0]['method']:
            raise ValueError(f'{role}: system/method drifted across blocks')
    repeats = []
    for block, r in enumerate(rows):
        reps = r['result']['repeats']
        if len(reps) != 1:
            raise ValueError(f'{role} block {block}: expected one repeat')
        rep = copy.deepcopy(reps[0])
        rep['index'] = block
        rep['comparison_block'] = block
        repeats.append(rep)
    lane = rows[0]['row_key']['lane']
    if lane != ROLE_TO_LANE[role]:
        raise ValueError(f'{role}: row lane {lane!r} mismatch')
    import gradient_runner as gradient_runner  # noqa: F811 (already importable)
    worst = gradient_runner.select_worst_repeat(repeats, None)
    samples_s = [r['sample_s'] for r in repeats]
    scf_s = [r['t_scf_s'] for r in repeats]
    grad_s = [r['t_grad_s'] for r in repeats]
    cderi_s = [r['t_cderi_s'] for r in repeats]
    mol_s = [r['t_mol_s'] for r in repeats]
    warmups = [s['startup'].get('warmup_s') for s in samples]
    timings = gradient_runner._grad_timing_block(
        statistics.median([w for w in warmups if w is not None]),
        samples_s, scf_s, grad_s, lane, cderi_times=cderi_s,
        mol_times=mol_s)
    timings['warmup_samples_s'] = warmups
    timings['sync'] = (
        f'{protocol}: {n_blocks} adjacent blocks with balanced arm '
        'order; each formal sample is an independent process with its '
        'own separated warmup; device-sync before/after every phase')
    result = copy.deepcopy(rows[0]['result'])
    result['repeats'] = repeats
    result['e_tot'] = worst['e_tot']
    result['converged'] = all(r['converged'] for r in repeats)
    result['scf_cycles'] = worst['scf_cycles']
    result['gradient'] = worst['gradient']
    result['gradient_norm'] = worst['gradient_norm']
    result['gradient_mode'] = gradient_runner._aggregate_gradient_mode(repeats)
    result['timings'] = timings
    row = copy.deepcopy(rows[0])
    row['result'] = result
    row['evidence'] = {
        'sample_records': [
            {'block': s['block'], 'role': role,
             'record': os.path.basename(s['_path'])}
            for s in samples]}
    meta = copy.deepcopy(base['meta_base'])
    meta['dependencies'] = deps
    meta['comparison_role'] = role
    meta['comparison_batch_id'] = batch_id
    meta['timing_protocol'] = protocol
    meta['schedule_sha256'] = schedule_sha256
    meta['optimization_env'] = optimization_env_of(role)
    meta['precision_mode'] = PRECISION_MODE[role]
    meta['gpu4pyscf_package_root'] = base['provenance'][
        'gpu4pyscf_package_root']
    if role == 'upstream':
        meta['reference_manifest_sha256'] = base['provenance'].get(
            'reference_manifest_sha256')
        meta['official_verification'] = base['provenance'].get(
            'official_verification')
    else:
        meta['source_commit'] = base['provenance'].get('source_commit')
        meta['source_root'] = base['provenance'].get('source_root')
    meta['comparison_note'] = (
        'official wheel vs source-compiled product comparison; build '
        'provenance recorded per role; differences are not attributable to '
        'any single kernel' if role == 'upstream' else
        'our source-compiled tree; see source_commit/root')
    return {'schema_version': SCHEMA_VERSION, 'meta': meta, 'rows': [row]}


# ---------------------------------------------------------------------------
# comparison document validation (structure only; no GPU)
# ---------------------------------------------------------------------------

_AUTO_HEAD_MIN_AO_CACHE = None


def _auto_head_min_ao():
    '''The SP-14 size gate from its one source of truth
    (gpu4pyscf/scf/hf.py), resolved WITHOUT importing the package:
    importing executes gpu4pyscf/__init__.py -> cupy -> a live CUDA
    context in the CALLING process, which in run_upstream_batch.py (the
    driver) is held by the driver itself and deadlocks its fail-closed
    gpu-busy gate (measured 2026-10-26T01:55:44Z: cell-0 merge -> a
    480 MiB context -> wait_gpu_idle can never pass; tools/tests/
    test_sp19_gpufree_gate_resolution.py is the red->green pin).  The
    package is located by scanning sys.path exactly like the import
    machinery would (the FIRST gpu4pyscf package wins; a winning package
    without scf/hf.py is the import-error class -> None), and the value
    is the top-level assignment -- a literal or the pyscf __config__
    idiom getattr(__config__, 'scf_hf_auto_head_min_ao', <default>),
    whose DEFAULT is taken (no PYSCF_CONFIG override exists in any run
    environment; a hypothetical larger arm-side gate yields
    above-threshold nao rows this merge still rejects = fail-closed
    direction).  Unresolvable -> None, and every caller fails CLOSED (a
    zero-hit auto row without an established gate threshold is still
    missing evidence).'''
    global _AUTO_HEAD_MIN_AO_CACHE
    if _AUTO_HEAD_MIN_AO_CACHE is None:
        try:
            import ast
            for entry in sys.path:
                base = (os.path.join(entry, 'gpu4pyscf')
                        if entry else 'gpu4pyscf')
                if not os.path.isfile(os.path.join(base, '__init__.py')):
                    continue
                hf = os.path.join(base, 'scf', 'hf.py')
                if not os.path.isfile(hf):
                    break  # this package wins the import and fails it
                for node in ast.parse(
                        open(hf, encoding='utf-8').read()).body:
                    if (isinstance(node, ast.Assign) and any(
                            isinstance(t, ast.Name)
                            and t.id == '_AUTO_HEAD_MIN_AO'
                            for t in node.targets)):
                        value = node.value
                        if (isinstance(value, ast.Call)
                                and len(value.args) == 3
                                and (getattr(value.func, 'id', None)
                                     == 'getattr'
                                     or getattr(value.func, 'attr', None)
                                     == 'getattr')):
                            value = value.args[2]
                        _AUTO_HEAD_MIN_AO_CACHE = int(
                            ast.literal_eval(value))
                        break
                break
        except Exception:
            return None
    return _AUTO_HEAD_MIN_AO_CACHE


def validate_comparison_doc(doc, role, batch_id, schedule_sha256,
                            expected_repeats=4):
    '''The load_cell contract of /opt/qc-agent/three_way_report.py as a
    local check: run it before the driver updates the index so a
    malformed merge never reaches the reporter as a source.  v1 batches
    carry four formal repeats per role, benchmark v1 carries eight (and
    the S role carries eight with its strict-match certificate on every
    repeat -- an uncertified S sample is a failure, never a pass).'''
    meta = doc['meta']
    row = doc['rows'][0]
    required = {'pyscf', 'cupy', 'numpy', 'libxc', 'cutensor'}
    if not required.issubset(meta['dependencies']):
        raise ValueError('missing runtime versions')
    if len(doc['rows']) != 1:
        raise ValueError('one row per comparison doc')
    if row['row_key']['lane'] != (
            'auto' if role == 'optimized_auto' else
            'auto-strict' if role == ROLE_AUTO_STRICT else 'fp64'):
        raise ValueError('lane mismatch')
    reps = row['result']['repeats']
    if len(reps) != expected_repeats:
        raise ValueError(f'expected {expected_repeats} formal repeats, '
                         f'got {len(reps)}')
    if meta['comparison_role'] != role or \
            meta['comparison_batch_id'] != batch_id or \
            meta['schedule_sha256'] != schedule_sha256:
        raise ValueError('unmatched comparison provenance')
    if role == 'strict_fp64':
        if row['result']['gradient_mode'] != 'fp64':
            raise ValueError('strict reference gradient_mode must be fp64')
        hits = row['result']['fp32_entry_hits']
        if any(v != 0 for v in hits.values()):
            raise ValueError(f'FP32 execution in strict reference: {hits}')
    if role == ROLE_AUTO_STRICT:
        # S charges its own certification: every formal repeat must carry
        # the auto+strict record with strict_match_verified True (an
        # incomplete certification or an adopted-displacement abort is a
        # fail-closed S sample, exactly like the standalone validator)
        for r in reps:
            sv = r.get('stability_verification')
            if not isinstance(sv, dict) or \
                    sv.get('policy') != 'auto+strict':
                raise ValueError(f'S repeat {r.get("index")!r} has no '
                                 'auto+strict stability_verification '
                                 'record')
            if sv.get('strict_match_verified') is not True:
                reason = (sv.get('strict_match_incomplete_reason')
                          or 'reason unrecorded')
                raise ValueError(
                    f'S repeat {r.get("index")!r} strict_match_verified '
                    f'is {sv.get("strict_match_verified")!r} '
                    f'({reason}): an uncertified S sample never '
                    'merges as ok')
    if role == 'optimized_auto':
        if not any(v > 0 for v in
                   row['result']['fp32_entry_hits'].values()):
            # SP-18: below the SP-14 size gate the plain-auto lane is
            # bitwise the fp64 lane by construction (fp64 from cycle 0,
            # never switches; fp64 gradient since SP-16), so an
            # all-zero fp32 record is the CORRECT evidence for exactly
            # this size class -- the gated-lane context must come from
            # the row itself (its measured nao, an unswitched policy,
            # an fp64 gradient); above-gate, absent nao, a fired switch
            # or a non-fp64 gradient is still missing evidence.
            nao = row.get('system', {}).get('nao')
            thr = _auto_head_min_ao()
            res = row['result']
            gated_lane = (
                thr is not None and isinstance(nao, int) and nao < thr
                and res['precision_switch'].get('switched') is False
                and res['gradient_mode'] == 'fp64')
            if not gated_lane:
                raise ValueError('no actual FP32 entry evidence for auto')
    if role == 'upstream':
        with open(OFFICIAL_CERTIFICATE, encoding='utf-8') as f:
            cert = json.load(f)
        import hashlib as _h
        with open(OFFICIAL_CERTIFICATE, 'rb') as f:
            cert_hash = _h.sha256(f.read()).hexdigest()
        if meta['gpu4pyscf_package_root'] != cert['package_root'] + '/gpu4pyscf' \
                or meta.get('reference_manifest_sha256') != cert_hash:
            raise ValueError('not the pinned official package')
    return True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_comparison',
        description='One arm of one balanced block of the upstream '
                    'comparison (control card 2026-10-18; benchmark v1 '
                    '2026-10-30 adds the 8-block and S protocols).')
    p.add_argument('--role', required=True,
                   choices=list(ROLES) + [ROLE_AUTO_STRICT])
    p.add_argument('--schedule', required=True,
                   help='frozen schedule JSON (upstream_schedule.py)')
    p.add_argument('--cell-index', type=int, required=True)
    p.add_argument('--block', type=int, required=True,
                   help='block index within the cell (0..N-1; N is the '
                        'schedule protocol block count)')
    p.add_argument('--out', required=True, help='sample record JSON path')
    p.add_argument('--official-root', default=OFFICIAL_ROOT)
    args = p.parse_args(argv)

    with open(args.schedule, encoding='utf-8') as f:
        raw = f.read()
    schedule = json.loads(raw)
    schedule_sha256 = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    cells = schedule['cells']
    cell = next(c for c in cells if c['index'] == args.cell_index)
    n_blocks = len(cell['blocks'])
    if not 0 <= args.block < n_blocks:
        raise RuntimeError(f'block {args.block} outside 0..{n_blocks - 1} '
                           f'of cell {args.cell_index}')
    order = cell['blocks'][args.block]
    role_short = {'upstream': 'O', 'optimized_fp64': 'B',
                  'optimized_auto': 'C', 'strict_fp64': 'R',
                  ROLE_AUTO_STRICT: 'S'}[args.role]
    if role_short not in order:
        raise RuntimeError(f'role {args.role} not scheduled in block '
                           f'{args.block} of cell {args.cell_index}')
    cell = dict(cell)
    cell['_block'] = args.block
    cell['_position'] = order.index(role_short)
    return run_arm(args.role, cell, schedule['settings'],
                   schedule['batch_id'], schedule_sha256, args.out,
                   official_root=args.official_root,
                   source_commit=schedule.get('source_commit'),
                   source_root=schedule.get('source_root'))


if __name__ == '__main__':
    sys.exit(main())

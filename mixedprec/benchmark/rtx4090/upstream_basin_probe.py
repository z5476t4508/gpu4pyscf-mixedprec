#!/usr/bin/env python3
"""Post-batch SCF-basin dm0-transfer probe (preregistration LC-206).

Characterizes the auto-lane gradient-gate FAIL rows of batch
uc-20261018-a on methoxy (the historical LC-39 UKS-branch molecule):
are the auto-lane endpoints genuine stationary points of the fp64
functional (distinct UKS branches on a multi-branch landscape), or
mixed-precision trajectory artifacts?

Stage 1 (CPU, session-side, no tool needed): the batch/A5 anchored
gradient geometry -- on methoxy_b3lyp_def2-svp the C-vs-R divergence is
collinear with the B-vs-R divergence (cos +1.000, single soft mode,
~3.7x amplified); on methoxy_wb97mv_def2-svp the landscape has >=3
distinct solutions (R at -114.9016950130, batch-C at +4.25e-7, A5
R0_AUTO at -1.283e-6 BELOW R, 19 cycles vs 11) with non-collinear
directions (cos 0.50/0.27); methoxy_m062x_def2-tzvp needs no arms (its
C lane is bitwise-clean vs R; the FAIL there is the fp64-lane VMT
knob, already named by A5).

Stage 2 (this driver): per cell, dm-producer arms run the default-init
trajectory in three configurations (F64 = R0 env + fp64 SCF = the batch
R role; AUTO = R0 env + auto SCF = the A5 R0_AUTO arm; CAUTO = STACK
env + auto SCF = the batch C role) and save their converged density
matrices; transfer arms then rerun the SCF from the OTHER lane's
converged density (F64_FROM_AUTO, F64_FROM_CAUTO, AUTO_FROM_F64,
CAUTO_FROM_F64) plus same-lane restart controls (F64_FROM_F64,
AUTO_FROM_AUTO).  A transfer arm that re-converges at its source
energy proves the source endpoint is a stationary point of the
transfer lane's functional; one that relaxes to the lane's default
basin proves the source endpoint was an artifact of its own
trajectory.

Arms run against the SAME frozen source snapshot as the batch (the
measured code), one fresh subprocess per (cell, arm, block) with a
fail-closed env discipline (R0: no GPU4PYSCF_* variables at all;
STACK: exactly the snapshot STACK_ENV).  Sample records use a distinct
schema (basin-probe-sample-v1) so they can never be mistaken for batch
index sources.  Timing is NOT a measurement here: arms run a single
point with no warmup, and the recorded phase walls are context only.
The driver never edits the batch's index or samples, never runs while
any batch driver process is alive, and refuses to start unless the
batch denominator is complete.
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time

PY = sys.executable or 'python3'
SAMPLE_SCHEMA = 'basin-probe-sample-v1'
HERE = os.path.dirname(os.path.abspath(__file__))

BATCH_DRIVER_PATTERNS = ('run_upstream_batch.py', 'upstream_fill_launcher.py',
                         'upstream_knob_isolation.py',
                         'upstream_basin_probe.py')

REQUIRED_PLAN_KEYS = ('unit', 'amendment_id', 'batch_id', 'index_path',
                      'schedule_path', 'schedule_sha256', 'snapshot_root',
                      'snapshot_commit', 'evidence_dir', 'cells', 'arms',
                      'blocks', 'seed', 'timeout_s', 'max_retries',
                      'anchor_tolerance_e', 'anchor_tolerance_dg',
                      'basin_tolerance_e', 'control_tolerance_e', 'anchors')

ARM_KEYS = ('scf_mode', 'grad_mode', 'env', 'stage')


# ---------------------------------------------------------------------------
# fail-closed guards (mirrors upstream_knob_isolation discipline)
# ---------------------------------------------------------------------------

def _proc_cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return []


def refuse_if_any_driver():
    '''One GPU workload: refuse while any batch driver, fill launcher,
    isolation driver or basin-probe driver process is alive.'''
    me = os.getpid()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == me:
            continue
        cmd = _proc_cmdline(pid)
        for part in cmd:
            base = os.path.basename(part)
            if base in BATCH_DRIVER_PATTERNS:
                raise RuntimeError(
                    f'another GPU driver process is alive (pid {pid}, '
                    f'{base}); refusing to double-drive the GPU')


def batch_complete(index_path, batch_id):
    '''True only when every pre-listed cell of the batch carries all four
    role sources (planned cells with empty sources = still running).'''
    with open(index_path, encoding='utf-8') as f:
        index = json.load(f)
    for batch in index.get('batches', []):
        if batch.get('id') != batch_id:
            continue
        cells = batch.get('cells', [])
        if not cells:
            return False
        for cell in cells:
            sources = cell.get('sources') or {}
            if not all(sources.get(r) for r in
                       ('upstream', 'optimized_fp64', 'optimized_auto',
                        'strict_fp64')):
                return False
        return True
    raise RuntimeError(f'batch {batch_id!r} not found in {index_path}')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot_inventory(snapshot_root, expected_commit):
    '''The probe must measure the SAME code as the batch: re-walk the
    read-only snapshot and require every file hash to match the batch's
    stored inventory.  Fail-closed on any drift, extra or missing file.'''
    marker = os.path.join(snapshot_root, '.upstream-batch-source-commit')
    if not os.path.isfile(marker):
        raise RuntimeError('snapshot missing its source-commit marker')
    with open(marker, encoding='utf-8') as f:
        frozen = f.read().strip()
    if frozen != expected_commit:
        raise RuntimeError(f'snapshot commit {frozen} != plan commit '
                           f'{expected_commit}; refusing to mix versions')
    inv_path = snapshot_root + '.inventory.json'
    if not os.path.isfile(inv_path):
        raise RuntimeError(f'snapshot inventory missing: {inv_path}')
    with open(inv_path, encoding='utf-8') as f:
        inventory = json.load(f)['files']
    live = {}
    for dirpath, dirnames, filenames in os.walk(snapshot_root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in ('.git', '__pycache__'))
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            live[os.path.relpath(p, snapshot_root)] = sha256_file(p)
    missing = sorted(set(inventory) - set(live))
    extra = sorted(set(live) - set(inventory))
    drifted = sorted(k for k in inventory if k in live
                     and inventory[k] != live[k])
    if missing or extra or drifted:
        raise RuntimeError(
            f'snapshot drift vs the batch inventory: {len(missing)} '
            f'missing, {len(extra)} extra, {len(drifted)} changed '
            f'(first: {(missing or extra or drifted)[:3]}); refusing to '
            'run arms against a snapshot that is not the batch snapshot')
    return len(live)


def gpu_busy():
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid',
             '--format=csv,noheader'], capture_output=True, text=True,
            timeout=30).stdout.strip()
        if out:
            return True
    except Exception:
        return True
    try:
        util = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader'], capture_output=True, text=True,
            timeout=30).stdout.strip()
        return util not in ('0 %', '[Not Supported]') and bool(util)
    except Exception:
        return True


def wait_gpu_idle(log, timeout_s=7200):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > timeout_s:
            raise RuntimeError('GPU did not go idle within '
                               f'{timeout_s}s; refusing to overlap')
        time.sleep(20)
    log('gpu idle')


# ---------------------------------------------------------------------------
# plan handling + env discipline
# ---------------------------------------------------------------------------

def load_plan(path):
    with open(path, encoding='utf-8') as f:
        plan = json.load(f)
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise RuntimeError(f'plan is missing required key {key!r}')
    for cell in plan['cells']:
        for key in ('tag', 'index'):
            if key not in cell:
                raise RuntimeError(f'plan cell missing {key!r}')
    if not plan['cells']:
        raise RuntimeError('plan carries no cells')
    arms = plan['arms']
    if not arms:
        raise RuntimeError('plan carries no arms')
    stage0 = [n for n, c in arms.items() if c['stage'] == 0]
    if not stage0:
        raise RuntimeError('plan has no stage-0 dm producers')
    for name, cfg in arms.items():
        for key in ARM_KEYS:
            if key not in cfg:
                raise RuntimeError(f'arm {name!r} missing {key!r}')
        if cfg['scf_mode'] not in ('fp64', 'auto'):
            raise RuntimeError(f'arm {name!r} scf_mode must be fp64/auto')
        if cfg['grad_mode'] not in ('fp64', 'auto'):
            raise RuntimeError(f'arm {name!r} grad_mode must be fp64/auto')
        if cfg['env'] not in ('R0', 'STACK'):
            raise RuntimeError(f'arm {name!r} env must be R0/STACK')
        if cfg['stage'] not in (0, 1):
            raise RuntimeError(f'arm {name!r} stage must be 0/1')
        dm_in = cfg.get('dm_in')
        if cfg['stage'] == 0:
            if dm_in:
                raise RuntimeError(f'stage-0 arm {name!r} cannot carry '
                                   'dm_in (producers run the default '
                                   'init trajectory)')
            if not cfg.get('dm_out'):
                raise RuntimeError(f'stage-0 arm {name!r} must set dm_out')
        else:
            if not dm_in:
                raise RuntimeError(f'stage-1 arm {name!r} must carry dm_in')
            if dm_in not in stage0:
                raise RuntimeError(
                    f'arm {name!r} dm_in {dm_in!r} is not a stage-0 '
                    'producer (cycles and cross-stage transfers are '
                    'refused; only a producer density may seed an arm)')
    stack_env = plan.get('stack_env')
    if not isinstance(stack_env, dict) or not stack_env:
        raise RuntimeError("plan 'stack_env' must be a non-empty dict of "
                           'GPU4PYSCF_* variables (the measured stack)')
    for var in stack_env:
        if not var.startswith('GPU4PYSCF_'):
            raise RuntimeError(f"plan stack_env carries non-knob variable "
                               f"{var!r}")
    for anchor, spec in plan['anchors'].items():
        for cell in plan['cells']:
            tag = cell['tag']
            if tag not in spec.get('per_cell', {}):
                raise RuntimeError(f'anchor {anchor!r} missing per_cell '
                                   f'entry for {tag!r}')
            entry = spec['per_cell'][tag]
            for key in ('path', 'sha256', 'expected_median_e'):
                if key not in entry:
                    raise RuntimeError(f'anchor {anchor!r} cell {tag!r} '
                                       f'missing {key!r}')
            if not isinstance(entry['expected_median_e'], (int, float)):
                raise RuntimeError(
                    f'anchor {anchor!r} cell {tag!r} has a non-numeric '
                    'expected_median_e')
    return plan


def arm_env(plan, arm_name):
    '''Clean subprocess environment: no inherited GPU4PYSCF_* variables,
    no PYTHONPATH; R0 arms see none, STACK arms see exactly the plan's
    frozen stack env (which the arm fail-closed-verifies equals the
    snapshot STACK_ENV).'''
    cfg = plan['arms'][arm_name]
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GPU4PYSCF_') and k != 'PYTHONPATH'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    if cfg['env'] == 'STACK':
        env.update(plan['stack_env'])
    return env


def check_env_discipline(plan, arm_name):
    '''Fail-closed: the arm process must see EXACTLY its arm's variables.'''
    cfg = plan['arms'][arm_name]
    seen = {k: v for k, v in os.environ.items()
            if k.startswith('GPU4PYSCF_')}
    want = dict(plan['stack_env']) if cfg['env'] == 'STACK' else {}
    if seen != want:
        raise RuntimeError(
            f'basin-probe arm environment discipline violated: want '
            f'{want}, saw {seen}')
    return seen


def assert_cell_matches_schedule(cell, sch_cell):
    '''The arm must measure the schedule's cell: tag and index identity
    are always required; carried identity fields are checked when
    present (the schedule is authoritative, the plan is a pointer).'''
    if sch_cell.get('tag') != cell['tag'] \
            or sch_cell.get('index') != cell['index']:
        raise RuntimeError(
            f"plan cell {cell['tag']!r}#{cell['index']} does not match "
            f"schedule cell {sch_cell.get('tag')!r}"
            f"#{sch_cell.get('index')}")
    for field in ('geom', 'basis', 'xc', 'charge', 'spin'):
        if field in cell and field in sch_cell \
                and cell[field] != sch_cell[field]:
            raise RuntimeError(
                f"plan cell {cell['tag']!r} field {field!r} disagrees "
                f"with the schedule ({cell[field]!r} vs "
                f"{sch_cell[field]!r})")


# ---------------------------------------------------------------------------
# arm mode: one (cell, arm, block) sample in a fresh process
# ---------------------------------------------------------------------------

def run_arm(plan, cell, arm_name, block, out_path):
    '''Runs inside the arm subprocess (clean env): resolves the SHARED
    RUNNERS AND gpu4pyscf FROM THE FROZEN SNAPSHOT (never the live
    repo), builds the cell exactly like the batch's arms, runs ONE
    basin point (optionally seeded with another arm's converged density)
    and optionally saves its own converged density.'''
    cfg = plan['arms'][arm_name]
    env_seen = check_env_discipline(plan, arm_name)

    snap_rtx = os.path.join(plan['snapshot_root'], 'mixedprec', 'benchmark',
                            'rtx4090')
    if not os.path.isfile(os.path.join(snap_rtx, 'upstream_comparison.py')):
        raise RuntimeError(f'snapshot runner dir not found: {snap_rtx}')
    sys.path.insert(0, snap_rtx)
    import upstream_comparison as uc          # the SNAPSHOT's copy
    import rtx4090_runner as energy_runner    # snapshot
    import gradient_runner                    # snapshot

    # the gpu4pyscf package must resolve INSIDE the frozen snapshot
    import gpu4pyscf
    gp_root = os.path.abspath(plan['snapshot_root'])
    gp_file = os.path.abspath(gpu4pyscf.__file__)
    if not gp_file.startswith(gp_root + os.sep):
        raise RuntimeError(f'gpu4pyscf resolved outside the snapshot: '
                           f'{gp_file}')
    # STACK arms: the plan's frozen stack env must equal the snapshot's
    # measured STACK_ENV (fail-closed, same discipline as the A4/A5
    # FULL configs)
    if cfg['env'] == 'STACK' and env_seen != dict(uc.STACK_ENV):
        raise RuntimeError(f'plan stack_env != snapshot STACK_ENV: '
                           f'{env_seen} != {dict(uc.STACK_ENV)}')

    with open(plan['schedule_path'], encoding='utf-8') as f:
        schedule = json.load(f)
    if sha256_file(plan['schedule_path']) != plan['schedule_sha256']:
        raise RuntimeError('schedule file does not match the plan\'s '
                           'frozen schedule_sha256')
    settings = schedule['settings']
    sch_cell = schedule['cells'][cell['index']]
    assert_cell_matches_schedule(cell, sch_cell)

    lines = []

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {msg}'
        print(line, flush=True)
        lines.append(line)

    role = 'strict_fp64' if cfg['scf_mode'] == 'fp64' else 'optimized_auto'
    args = uc.namespace_for(sch_cell, settings, role)
    uc.apply_3c_setup(args, energy_runner, log)

    dm_dir = os.path.join(plan['evidence_dir'], 'dm')
    os.makedirs(dm_dir, exist_ok=True)
    dm_in_npz = None
    if cfg.get('dm_in'):
        dm_in_npz = os.path.join(
            dm_dir, f'{cell["tag"]}_{cfg["dm_in"]}_b{block}.npz')
        if not os.path.isfile(dm_in_npz):
            raise RuntimeError(f'dm_in file missing (the producer arm for '
                               f'block {block} did not complete): '
                               f'{dm_in_npz}')
    dm_out_npz = None
    if cfg.get('dm_out'):
        dm_out_npz = os.path.join(
            dm_dir, f'{cell["tag"]}_{arm_name}_b{block}.npz')

    probe = energy_runner.PrecisionProbe()
    probe.install()
    sampler = energy_runner.MemorySampler()
    try:
        with sampler:
            result = _basin_point(
                energy_runner, gradient_runner, sch_cell['geom'],
                sch_cell['basis'], sch_cell['xc'], args, cfg['scf_mode'],
                cfg['grad_mode'], probe, sampler, dm_in_npz, dm_out_npz,
                log)
    finally:
        probe.remove()

    arm_config = {k: cfg[k] for k in ARM_KEYS}
    if cfg.get('dm_in'):
        arm_config['dm_in'] = cfg['dm_in']
    if cfg.get('dm_out'):
        arm_config['dm_out'] = True
    record = {
        'schema_version': SAMPLE_SCHEMA,
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'schedule_sha256': plan['schedule_sha256'],
        'cell_index': cell['index'],
        'cell_tag': cell['tag'],
        'arm': arm_name,
        'arm_config': arm_config,
        'optimization_env': env_seen,
        'block': block,
        'scf_mode': cfg['scf_mode'],
        'row': result,
        'arm_log': lines,
        'provenance': {
            'source_commit': schedule['source_commit'],
            'source_root': plan['snapshot_root'],
            'runners_resolved_from': snap_rtx,
            'gpu4pyscf_package_root': os.path.join(gp_root, 'gpu4pyscf'),
            'gpu4pyscf_resolved_file': gp_file,
            'plan_sha256': sha256_file(plan['_plan_path']),
            'command': ' '.join([PY, os.path.abspath(__file__)]
                                + sys.argv[1:]),
        },
        'recorded_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                        time.gmtime()),
    }
    with open(out_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=1)
    os.replace(out_path + '.tmp', out_path)
    log(f'wrote {out_path}')


def _basin_point(energy_runner, gradient_runner, geom, basis, xc, args,
                 scf_mode, grad_mode, probe, sampler, dm_in_npz,
                 dm_out_npz, log):
    '''One basin point: mol + mf + cderi exactly like the batch arms,
    one SCF (optionally seeded with dm_in), optional dm save, one
    gradient.  Single point, no warmup: phase walls are context, never
    timing claims.'''
    import cupy as cp
    import numpy as np
    from types import MethodType

    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    mol, mf = energy_runner._gpu_mol_mf(geom, basis, xc, True,
                                         args.auxbasis, args, scf_mode,
                                         probe)
    cp.cuda.runtime.deviceSynchronize()
    t_mol = time.perf_counter() - t0

    t1 = time.perf_counter()
    mf.with_df.build()
    cp.cuda.runtime.deviceSynchronize()
    t_cderi = time.perf_counter() - t1

    dm0 = None
    dm_in_info = None
    if dm_in_npz is not None:
        z = np.load(dm_in_npz)
        keys = sorted(z.files)
        blocks = [np.asarray(z[k]) for k in keys]
        dm0 = tuple(blocks) if len(blocks) > 1 else blocks[0]
        dm_in_info = {
            'path': dm_in_npz, 'sha256': sha256_file(dm_in_npz),
            'spin_blocks': keys,
            'shapes': [list(b.shape) for b in blocks],
        }
        log(f'dm0 seeded from {os.path.basename(dm_in_npz)}: '
            f'{keys} {[list(b.shape) for b in blocks]}')

    t1 = time.perf_counter()
    e = mf.kernel(dm0=dm0) if dm0 is not None else mf.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_scf = time.perf_counter() - t1
    converged = bool(mf.converged)
    cycles = int(getattr(mf, 'cycles', 0) or 0)
    summary = dict(getattr(mf, 'scf_summary', {}) or {})
    nlc = probe.last_nlc_energy if probe is not None else None

    dm_out_info = None
    if dm_out_npz is not None:
        dm = mf.make_rdm1()
        # preserve the EXACT structure make_rdm1 returned: a tuple/list
        # is saved block-per-element, a bare ndarray (2D or stacked 3D)
        # is saved as ONE block -- the load path reconstructs the same
        # shape of object, so the consumer arm feeds the SCF the same
        # dm0 layout its producer lane produced
        if isinstance(dm, (tuple, list)):
            blocks = [np.asarray(cp.asnumpy(x)) for x in dm]
        else:
            blocks = [np.asarray(cp.asnumpy(dm))]
        np.savez(dm_out_npz,
                 **{f's{i}': b for i, b in enumerate(blocks)})
        dm_out_info = {
            'path': dm_out_npz, 'sha256': sha256_file(dm_out_npz),
            'spin_blocks': [f's{i}' for i in range(len(blocks))],
            'shapes': [list(b.shape) for b in blocks],
            'was_tuple': isinstance(dm, (tuple, list)),
        }

    g = mf.nuc_grad_method()
    if energy_runner.is_3c_xc(xc):
        g.get_dispersion = MethodType(
            gradient_runner._r2scan3c_get_dispersion_grad, g)
        g.auxbasis_response = True
    if grad_mode is not None:
        g.precision_mode = grad_mode
    hits_before = probe.hits['grad_ip1'] if probe is not None else 0
    t1 = time.perf_counter()
    de = g.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_grad = time.perf_counter() - t1
    grad_hits = (probe.hits['grad_ip1'] - hits_before) \
        if probe is not None else 0
    de = cp.asnumpy(de)
    grad_mode_effective = 'fp32' if grad_hits > 0 else 'fp64'

    grad_list = [[float(x) for x in row] for row in de]
    grad_flat = [x for row in grad_list for x in row]
    grad_norm = sum(x * x for x in grad_flat) ** 0.5

    peak_alloc = max(free_mib - (sampler.min_free_mib or free_mib), 0.0)
    within = peak_alloc + args.headroom_mib <= total_mib

    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': scf_mode}
    log(f'{scf_mode}{"+" + os.path.basename(dm_in_npz) if dm_in_npz else ""}:'
        f' E={e:.12f} conv={converged} cyc={cycles} '
        f'gnorm={grad_norm:.3e} gmode={grad_mode_effective} '
        f'peak={peak_alloc:.0f}MiB')
    return {
        'row_key': row_key,
        'pilot_stage': args.pilot_stage,
        'system': energy_runner._system_block(geom, mol),
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': float(e), 'converged': converged,
            'scf_cycles': cycles,
            'gradient': grad_list,
            'gradient_norm': grad_norm,
            'gradient_mode': grad_mode_effective,
            'e_terms': energy_runner._e_terms(summary, nlc),
            'precision_switch': probe.switch_info(
                'auto' if scf_mode == 'auto' else scf_mode),
            'fp32_entry_hits': dict(probe.hits),
            'timings': {
                'protocol': 'basin-probe-v1-single-point-no-warmup',
                't_mol_s': t_mol, 't_cderi_s': t_cderi,
                't_scf_s': t_scf, 't_grad_s': t_grad},
            'mem': {
                'gpu_free_before_mib': free_mib,
                'gpu_total_mib': total_mib,
                'gpu_peak_alloc_mib': round(peak_alloc, 1),
                'headroom_budget_mib': args.headroom_mib,
                'within_budget': bool(within)},
            'dm_in': dm_in_info, 'dm_out': dm_out_info,
            'fallback_events': [], 'errors': None,
        },
    }


# ---------------------------------------------------------------------------
# driver mode
# ---------------------------------------------------------------------------

def record_matches(record, cell, arm_name, block, plan):
    try:
        return (record.get('schema_version') == SAMPLE_SCHEMA
                and record.get('arm') == arm_name
                and record.get('block') == block
                and record.get('cell_tag') == cell['tag']
                and record.get('amendment_id') == plan['amendment_id']
                and isinstance(record.get('row'), dict))
    except Exception:
        return False


def run_arm_subprocess(plan, cell, arm_name, block, out_path,
                       log, max_retries):
    cmd = [PY, os.path.abspath(__file__), '--plan', plan['_plan_path'],
           '--arm', '--cell-tag', cell['tag'], '--arm-name', arm_name,
           '--block', str(block), '--out', out_path]
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        wait_gpu_idle(log)
        log(f'arm {cell["tag"]} {arm_name} block {block} '
            f'attempt {attempts}')
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=arm_env(plan, arm_name),
                capture_output=True, text=True, timeout=plan['timeout_s'])
            rc = proc.returncode
            if rc != 0:
                log(f'  arm stderr tail: '
                    f'{(proc.stderr or "")[-800:]}')
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        log(f'arm {cell["tag"]} {arm_name} block {block} rc={rc} '
            f'wall={time.time() - t0:.0f}s')
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding='utf-8') as f:
                    record = json.load(f)
                if record_matches(record, cell, arm_name, block, plan):
                    if rc == 0:
                        return True, attempts
                    keep = f'{out_path}.attempt{attempts}.json'
                    os.replace(out_path, keep)
                    log(f'  failed attempt preserved at {keep}')
            except (OSError, ValueError) as exc:
                log(f'  malformed record: {exc!r}')
        if attempts >= max_retries:
            log(f'arm {cell["tag"]} {arm_name} block {block}: FAILED '
                f'after {attempts} attempts; honest gap in the matrix')
            return False, attempts
    return False, attempts


def driver(plan, log):
    refuse_if_any_driver()
    if not batch_complete(plan['index_path'], plan['batch_id']):
        raise RuntimeError('batch denominator incomplete; the basin probe '
                           'only runs post-batch')
    n_files = verify_snapshot_inventory(plan['snapshot_root'],
                                        plan['snapshot_commit'])
    log(f'snapshot verified: {n_files} files match the batch inventory')
    if sha256_file(plan['schedule_path']) != plan['schedule_sha256']:
        raise RuntimeError('schedule drifted vs the plan\'s frozen sha256')
    wait_gpu_idle(log)

    samples_dir = os.path.join(plan['evidence_dir'], 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    dm_dir = os.path.join(plan['evidence_dir'], 'dm')
    os.makedirs(dm_dir, exist_ok=True)

    rng = random.Random(plan['seed'])
    stage0 = sorted(n for n, c in plan['arms'].items() if c['stage'] == 0)
    stage1 = sorted(n for n, c in plan['arms'].items() if c['stage'] == 1)
    failures = []
    for cell in plan['cells']:
        for block in range(plan['blocks']):
            for stage, names in ((0, stage0), (1, stage1)):
                order = sorted(names, key=lambda n: rng.random())
                for name in order:
                    out_path = os.path.join(
                        samples_dir,
                        f'{cell["tag"]}_{name}_b{block}.json')
                    ok, _ = run_arm_subprocess(
                        plan, cell, name, block, out_path, log,
                        plan['max_retries'])
                    if not ok:
                        failures.append((cell['tag'], name, block))
        # after all blocks of a cell: every producer dm must exist
        for name in stage0:
            for block in range(plan['blocks']):
                p = os.path.join(dm_dir, f'{cell["tag"]}_{name}_b{block}.npz')
                if not os.path.isfile(p):
                    failures.append((cell['tag'], name, block, 'dm-missing'))
    if failures:
        log(f'HONEST GAPS: {failures}')
    log('driver complete')
    return failures


# ---------------------------------------------------------------------------
# mechanical analysis (no verdicts; the session writes conclusions)
# ---------------------------------------------------------------------------

def _flatten_gradient(g):
    out = []
    for part in g:
        if isinstance(part, (list, tuple)):
            out.extend(float(x) for x in part)
        else:
            out.append(float(part))
    return out


def _median_block(samples_dir, cell, arm, blocks, field_path):
    '''Component-wise/numeric median across blocks; None when a block is
    missing or lacks the field.'''
    per_block = []
    for block in range(blocks):
        p = os.path.join(samples_dir, f'{cell["tag"]}_{arm}_b{block}.json')
        if not os.path.isfile(p):
            return None
        with open(p, encoding='utf-8') as f:
            record = json.load(f)
        val = record
        for key in field_path:
            val = (val or {}).get(key)
        if val is None:
            return None
        per_block.append(val)
    if isinstance(per_block[0], list):
        flat = [_flatten_gradient(v) for v in per_block]
        return [statistics.median(col) for col in zip(*flat)]
    return statistics.median(per_block)


def _grad_dmax(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _grad_cos(d_a, d_b):
    na = sum(x * x for x in d_a) ** 0.5
    nb = sum(x * x for x in d_b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(d_a, d_b)) / (na * nb)


def classify_basin(e_arm, e_source, e_lane_default, basin_tol):
    '''Mechanical basin classification of a transfer arm: STAY (its
    source basin), SWITCH (the lane's default basin), or OTHER (neither
    within tolerance; both distances recorded).  When the source IS the
    lane's default basin (same-lane restart controls), STAY/SWITCH are
    the same statement and the plain STAY/OTHER reading applies; when
    two DISTINCT reference basins sit within tolerance of each other
    the classification is honestly DEGENERATE (the distinction is not
    resolvable at this tolerance).'''
    d_source = abs(e_arm - e_source)
    d_default = abs(e_arm - e_lane_default)
    if e_source == e_lane_default:
        if d_source <= basin_tol:
            return 'STAY', d_source, d_default
        return 'OTHER', d_source, d_default
    if d_source <= basin_tol and d_default <= basin_tol:
        return 'DEGENERATE', d_source, d_default
    if d_source <= basin_tol:
        return 'STAY', d_source, d_default
    if d_default <= basin_tol:
        return 'SWITCH', d_source, d_default
    return 'OTHER', d_source, d_default


def analyze(plan, out_json, out_md=None):
    '''Anchors (producers only), basin classification (transfer arms),
    controls.  Purely mechanical: no scientific verdicts.'''
    samples_dir = os.path.join(plan['evidence_dir'], 'samples')
    blocks = plan['blocks']
    report = {
        'amendment_id': plan['amendment_id'],
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime()),
        'cells': {}, 'gaps': [],
    }
    for cell in plan['cells']:
        tag = cell['tag']
        entry = {'anchors': {}, 'arms': {}}
        med_e, med_g, med_cy = {}, {}, {}
        for arm in plan['arms']:
            med_e[arm] = _median_block(samples_dir, cell, arm, blocks,
                                       ('row', 'result', 'e_tot'))
            g = _median_block(samples_dir, cell, arm, blocks,
                              ('row', 'result', 'gradient'))
            med_g[arm] = _flatten_gradient(g) if g else None
            med_cy[arm] = _median_block(samples_dir, cell, arm, blocks,
                                        ('row', 'result', 'scf_cycles'))
        # anchors: each stage-0 producer is compared against its frozen
        # reference (path sha256-verified; expected median energy frozen
        # in the plan at registration)
        anchor_arms = {'F64': 'batch_R', 'AUTO': 'a5_R0_AUTO',
                      'CAUTO': 'batch_C'}
        for arm, anchor in anchor_arms.items():
            if arm not in plan['arms']:
                continue
            spec = plan['anchors'][anchor]['per_cell'][tag]
            ok = {'sha256_ok': None, 'e_diff': None,
                  'e_within_tolerance': None, 'note': None}
            if med_e[arm] is None:
                ok['note'] = 'arm missing (honest gap)'
                report['gaps'].append(f'{tag}:{arm}')
                entry['anchors'][anchor] = ok
                continue
            if not os.path.isfile(spec['path']):
                ok['note'] = f'anchor source missing: {spec["path"]}'
                entry['anchors'][anchor] = ok
                continue
            sha = sha256_file(spec['path'])
            ok['sha256_ok'] = (sha == spec['sha256'])
            ok['e_diff'] = med_e[arm] - spec['expected_median_e']
            ok['e_within_tolerance'] = \
                abs(ok['e_diff']) <= plan['anchor_tolerance_e']
            if anchor == 'batch_C' and med_g[arm] is not None:
                with open(spec['path'], encoding='utf-8') as f:
                    doc = json.load(f)
                ref_g = _flatten_gradient(
                    doc['rows'][0]['result']['gradient'])
                ok['dg_vs_batch_C'] = _grad_dmax(med_g[arm], ref_g)
                ok['dg_within_tolerance'] = \
                    ok['dg_vs_batch_C'] <= plan['anchor_tolerance_dg']
            entry['anchors'][anchor] = ok
        # basin classification for stage-1 arms
        for arm, cfg in plan['arms'].items():
            if cfg['stage'] != 1:
                continue
            src = cfg['dm_in']
            lane_default = 'F64' if cfg['scf_mode'] == 'fp64' else (
                'CAUTO' if cfg['env'] == 'STACK' else 'AUTO')
            out = {'dm_in': src, 'lane_default_basin': lane_default,
                   'control': bool(cfg.get('control'))}
            if med_e[arm] is None:
                out['note'] = 'arm missing (honest gap)'
                report['gaps'].append(f'{tag}:{arm}')
                entry['arms'][arm] = out
                continue
            out['median_e'] = med_e[arm]
            out['median_cycles'] = med_cy[arm]
            tol = plan['control_tolerance_e'] if cfg.get('control') \
                else plan['basin_tolerance_e']
            cls, d_src, d_def = classify_basin(
                med_e[arm], med_e[src], med_e[lane_default], tol)
            out['classification'] = cls
            out['abs_dE_vs_source'] = d_src
            out['abs_dE_vs_lane_default'] = d_def
            out['signed_dE_vs_source'] = med_e[arm] - med_e[src]
            if med_g[arm] is not None and med_g[src] is not None:
                out['dg_vs_source'] = _grad_dmax(med_g[arm], med_g[src])
            if med_g[arm] is not None and med_g[lane_default] is not None:
                out['dg_vs_lane_default'] = \
                    _grad_dmax(med_g[arm], med_g[lane_default])
            entry['arms'][arm] = out
        # producer summaries for reference
        for arm in ('F64', 'AUTO', 'CAUTO'):
            if arm in plan['arms']:
                entry.setdefault('producers', {})[arm] = {
                    'median_e': med_e[arm],
                    'median_cycles': med_cy[arm],
                }
        report['cells'][tag] = entry
    with open(out_json + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    os.replace(out_json + '.tmp', out_json)
    if out_md:
        with open(out_md + '.tmp', 'w', encoding='utf-8') as f:
            f.write(render_report_md(plan, report))
        os.replace(out_md + '.tmp', out_md)
    return report


def render_report_md(plan, report):
    lines = [f"# Basin-probe report {report['amendment_id']}",
             f"generated {report['generated_at_utc']} "
             "(mechanical output; verdicts live in the session docs)", '']
    for tag, entry in report['cells'].items():
        lines.append(f"## {tag}")
        lines.append('### anchors (producers vs frozen references)')
        for anchor, ok in entry.get('anchors', {}).items():
            if ok.get('e_diff') is not None:
                extra = ''
                if 'dg_vs_batch_C' in ok:
                    extra = (f", dg={ok['dg_vs_batch_C']:.2e} "
                             f"(<= {plan['anchor_tolerance_dg']:.0e}: "
                             f"{ok['dg_within_tolerance']})")
                lines.append(
                    f"- {anchor}: dE={ok['e_diff']:+.3e} "
                    f"(sha_ok={ok['sha256_ok']}, "
                    f"within {plan['anchor_tolerance_e']:.0e}: "
                    f"{ok['e_within_tolerance']}){extra}")
            else:
                lines.append(f"- {anchor}: {ok.get('note')}")
        lines.append('### producers')
        for arm, p in entry.get('producers', {}).items():
            if p.get('median_e') is None:
                lines.append(f"- {arm}: MISSING (honest gap)")
                continue
            lines.append(f"- {arm}: e={p['median_e']:.10f} "
                         f"cycles={p.get('median_cycles')}")
        lines.append('### transfer arms')
        for arm, out in entry.get('arms', {}).items():
            if 'classification' not in out:
                lines.append(f"- {arm}: {out.get('note')}")
                continue
            lines.append(
                f"- {arm} (from {out['dm_in']}, lane default "
                f"{out['lane_default_basin']}): "
                f"{out['classification']} | e={out['median_e']:.10f} "
                f"cycles={out['median_cycles']} | "
                f"|dE vs source|={out['abs_dE_vs_source']:.2e}, "
                f"|dE vs lane default|={out['abs_dE_vs_lane_default']:.2e}"
                + (f", dg vs source={out['dg_vs_source']:.2e}"
                   if 'dg_vs_source' in out else '')
                + (f", dg vs lane default="
                   f"{out['dg_vs_lane_default']:.2e}"
                   if 'dg_vs_lane_default' in out else ''))
        lines.append('')
    if report['gaps']:
        lines.append(f"HONEST GAPS: {report['gaps']}")
    else:
        lines.append('no gaps: every planned arm completed')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        description='LC-206 methoxy basin probe: dm0-transfer arms '
                    'against the frozen batch snapshot')
    p.add_argument('--plan', required=True)
    p.add_argument('--arm', action='store_true',
                   help='run one arm (driver spawns these)')
    p.add_argument('--cell-tag')
    p.add_argument('--arm-name')
    p.add_argument('--block', type=int)
    p.add_argument('--out')
    p.add_argument('--driver', action='store_true')
    p.add_argument('--driver-log')
    p.add_argument('--analyze', action='store_true')
    p.add_argument('--report-json')
    p.add_argument('--report-md')
    args = p.parse_args(argv)

    plan = load_plan(args.plan)
    plan['_plan_path'] = os.path.abspath(args.plan)

    if args.arm:
        if not (args.cell_tag and args.arm_name
                and args.block is not None and args.out):
            raise RuntimeError('arm mode needs --cell-tag, --arm-name, '
                               '--block, --out')
        cell = next(c for c in plan['cells']
                    if c['tag'] == args.cell_tag)
        run_arm(plan, cell, args.arm_name, args.block, args.out)
        return 0
    if args.driver:
        log_path = args.driver_log or 'basin-probe-driver.log'
        if not os.path.isabs(log_path):
            log_path = os.path.join(plan['evidence_dir'], log_path)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'a', encoding='utf-8') as logf:
            def log(msg):
                line = (f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime())}] {msg}')
                print(line, flush=True)
                logf.write(line + '\n')
                logf.flush()
            log(f'driver start: plan {plan["_plan_path"]} '
                f'sha256 {sha256_file(plan["_plan_path"])}')
            driver(plan, log)
            log('driver exit 0')
        return 0
    if args.analyze:
        if not args.report_json:
            raise RuntimeError('analyze mode needs --report-json')
        report = analyze(plan, args.report_json, args.report_md)
        print(json.dumps({'gaps': report['gaps']}, indent=1))
        return 0
    raise RuntimeError('one of --arm / --driver / --analyze is required')


if __name__ == '__main__':
    sys.exit(main())

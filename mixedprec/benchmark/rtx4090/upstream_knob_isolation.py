#!/usr/bin/env python3
"""Post-batch knob-isolation diagnostic driver (preregistrations A4/A5).

Executes the preregistered single-knob isolation matrix for the
epinephrine precision-gate exceedance of batch uc-20261018-a
(LC-205i: epinephrine/def2-svp hybrids B/C sit below strict R by
1.932e-6 (b3lyp, gate-exceeding), 7.164e-7 (m062x), 2.491e-7
(wb97mv); B==C on every method so the auto switch is excluded; only
GPU4PYSCF_XC_GRID_BLOCKSIZE and GPU4PYSCF_SCF_VMAT_F32 touch the SCF
path for non-VV10 functionals, VV10_UWE adds itself for wb97mv).

A5 (methoxy gradient-gate family) extends the same driver with a
per-config optional SCF lane: configs may carry the reserved key
'__lane__' ('fp64', the default and the only value A4 uses, or 'auto'
for arms reproducing the batch optimized_auto role) plus, when the
plan sets gradient_attribution, a mechanical gradient-side
classification mirroring the energy one.  An A4 plan without these
keys behaves exactly as before.

The plan (cells, configs, reproduce references, decision rules) is a
separate preregistered JSON anchored in the batch dir BEFORE any arm;
this driver only executes and reports mechanically.  It never edits the
batch's index or samples, never runs while any batch process is alive,
and refuses to start unless the batch denominator is complete.

Arms run against the SAME frozen source snapshot as the batch (the
measured code), one fresh subprocess per (cell, config, block) with a
fail-closed env discipline: the subprocess must see EXACTLY the config's
GPU4PYSCF_* variables, nothing more.  Sample records use a distinct
schema (knob-isolation-sample-v1) so they can never be mistaken for
batch index sources.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

PY = sys.executable or 'python3'
SAMPLE_SCHEMA = 'knob-isolation-sample-v1'
HERE = os.path.dirname(os.path.abspath(__file__))

# the only knobs that can change a non-VV10 SCF energy (verified in
# numint.py:49-78,610-618 and scf/hf.py:300-426); the three gradient-path
# knobs cannot change e_tot by construction -- the GRAD3 arm verifies that
# empirically
SCF_PATH_KNOBS = ('GPU4PYSCF_XC_GRID_BLOCKSIZE',
                  'GPU4PYSCF_SCF_VMAT_F32',
                  'GPU4PYSCF_VV10_UWE_KERNEL')
GRAD_PATH_KNOBS = ('GPU4PYSCF_DF_IP1_FUSED', 'GPU4PYSCF_DF_IP1_F32',
                   'GPU4PYSCF_GRAD_GRID_F32')

BATCH_DRIVER_PATTERNS = ('run_upstream_batch.py', 'upstream_fill_launcher.py',
                          'upstream_knob_isolation.py')


# ---------------------------------------------------------------------------
# fail-closed guards (mirrors run_upstream_batch.py / watchdog discipline)
# ---------------------------------------------------------------------------

def _proc_cmdline(pid):
    try:
        with open(f'/proc/{pid}/cmdline', 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return []


def refuse_if_any_driver():
    '''One GPU workload: refuse while any batch driver, fill launcher or
    isolation driver process is alive (element-based argv scan; a
    launcher/watchdog argv carrying the script path is excluded by the
    caller-side pattern list choice: those two are re-arm infrastructure,
    not drivers, and their liveness is handled by the batch-completeness
    gate below).'''
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


def verify_snapshot_inventory(snapshot_root, expected_commit):
    '''The rerun must measure the SAME code as the batch: re-walk the
    read-only snapshot and require every file hash to match the batch's
    stored inventory (and the commit marker to match the schedule).
    Fail-closed on any drift, extra or missing file.'''
    marker = os.path.join(snapshot_root, '.upstream-batch-source-commit')
    if not os.path.isfile(marker):
        raise RuntimeError('snapshot missing its source-commit marker')
    with open(marker, encoding='utf-8') as f:
        frozen = f.read().strip()
    if frozen != expected_commit:
        raise RuntimeError(f'snapshot commit {frozen} != schedule commit '
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
    for key in ('amendment_id', 'batch_id', 'schedule_path',
                'snapshot_root', 'evidence_root', 'cells', 'configs',
                'blocks_per_config', 'reproduce_tolerance_ha',
                'attribution_rules'):
        if key not in plan:
            raise RuntimeError(f'plan is missing required key {key!r}')
    for cell in plan['cells']:
        for key in ('tag', 'index'):
            if key not in cell:
                raise RuntimeError(f'plan cell missing {key!r}')
    for name, cfg in plan['configs'].items():
        for var in cfg:
            if var == '__lane__':
                if cfg[var] not in ('fp64', 'auto'):
                    raise RuntimeError(
                        f"config {name!r} carries an unsupported __lane__ "
                        f"{cfg[var]!r}; only 'fp64' or 'auto' are allowed")
                continue
            if not var.startswith('GPU4PYSCF_'):
                raise RuntimeError(f'config {name!r} carries a non-knob '
                                   f'variable {var!r}')
    return plan


def knobs_of(config):
    '''The config's environment knobs only: reserved '__'-prefixed keys
    (the A5 __lane__ directive) are driver instructions, never
    environment variables.'''
    return {k: v for k, v in config.items() if not k.startswith('__')}


def arm_env(config):
    '''Clean subprocess environment: no inherited GPU4PYSCF_* variables,
    no PYTHONPATH; exactly the config's knobs.'''
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GPU4PYSCF_') and k != 'PYTHONPATH'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    env.update(knobs_of(config))
    return env


def check_env_discipline(config):
    '''Fail-closed: the arm process must see EXACTLY the config's knobs.'''
    seen = {k: v for k, v in os.environ.items()
            if k.startswith('GPU4PYSCF_')}
    if seen != knobs_of(config):
        raise RuntimeError(
            f'knob-isolation arm environment discipline violated: want '
            f'{knobs_of(config)}, saw {seen}')
    return seen


def positions_for(config_names, blocks):
    '''Deterministic rotation: config i in block b runs at position
    (i + b) % len(config_names) -- each config sees 4 distinct positions
    (energy comparisons are position-insensitive; the rotation is
    pro-forma honesty).'''
    n = len(config_names)
    return {(name, b): (i + b) % n
            for i, name in enumerate(config_names) for b in range(blocks)}


# ---------------------------------------------------------------------------
# arm mode: one (cell, config, block) sample in a fresh process
# ---------------------------------------------------------------------------

def assert_cell_matches_schedule(cell, sch_cell):
    '''Fail-closed plan/schedule agreement: every identity field the
    plan cell carries must equal the schedule cell's (plans freeze
    identity by index+tag; extra fields, when present, are checked
    too).  A disagreement means the preregistration did not freeze
    this cell; refuse to run it.'''
    if sch_cell.get('tag') != cell.get('tag') \
            or sch_cell.get('index') != cell.get('index'):
        raise RuntimeError(
            f"plan cell {cell.get('tag')!r} disagrees with schedule cell "
            f"{cell.get('index')} on tag/index; refusing to run a cell "
            'the preregistration did not freeze')
    for key in ('geom', 'basis', 'xc', 'charge', 'spin'):
        if key in cell and sch_cell.get(key) != cell[key]:
            raise RuntimeError(
                f"plan cell {cell['tag']!r} disagrees with schedule cell "
                f"{cell['index']} on {key!r}; refusing to run a cell the "
                'preregistration did not freeze')


def run_arm(plan, cell, config_name, block, position, out_path):
    '''Runs inside the arm subprocess (clean env): resolves the SHARED
    RUNNERS AND gpu4pyscf FROM THE FROZEN SNAPSHOT (never the live repo)
    by putting the snapshot's rtx4090 directory first on sys.path, then
    builds the cell exactly like the batch's optimized_fp64 arm and runs
    it with the config's knobs as the only difference from R0.'''
    config = plan['configs'][config_name]
    lane = config.get('__lane__', 'fp64')
    env_seen = check_env_discipline(config)

    snap_rtx = os.path.join(plan['snapshot_root'], 'mixedprec', 'benchmark',
                            'rtx4090')
    if not os.path.isfile(os.path.join(snap_rtx, 'upstream_comparison.py')):
        raise RuntimeError(f'snapshot runner dir not found: {snap_rtx}')
    sys.path.insert(0, snap_rtx)
    import upstream_comparison as uc          # the SNAPSHOT's copy
    import rtx4090_runner as energy_runner    # snapshot
    import gradient_runner                    # snapshot

    # tie the full-stack configs to the measured stack, fail-closed
    # (FULL = batch optimized_fp64 env; FULL_AUTO = the same stack under
    # the auto SCF lane, i.e. the batch optimized_auto env)
    if config_name in ('FULL', 'FULL_AUTO') \
            and knobs_of(config) != dict(uc.STACK_ENV):
        raise RuntimeError(f'plan {config_name} config does not equal the '
                           'snapshot STACK_ENV: '
                           f'{knobs_of(config)} != {dict(uc.STACK_ENV)}')

    with open(plan['schedule_path'], encoding='utf-8') as f:
        schedule = json.load(f)
    settings = schedule['settings']
    sch_cell = schedule['cells'][cell['index']]
    assert_cell_matches_schedule(cell, sch_cell)

    lines = []

    def log(msg):
        line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {msg}'
        print(line, flush=True)
        lines.append(line)

    args = uc.namespace_for(sch_cell, settings, 'optimized_fp64')
    uc.apply_3c_setup(args, energy_runner, log)
    row = gradient_runner.run_gpu_grad_cell(
        sch_cell['geom'], sch_cell['basis'], sch_cell['xc'], lane,
        args, log)

    record = {
        'schema_version': SAMPLE_SCHEMA,
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'schedule_sha256': sha256_file(plan['schedule_path']),
        'cell_index': cell['index'],
        'cell_tag': cell['tag'],
        'config': config_name,
        'optimization_env': env_seen,
        'block': block,
        'position': position,
        'lane': lane,
        'row': row,
        'arm_log': lines,
        'provenance': {
            'source_commit': schedule['source_commit'],
            'source_root': plan['snapshot_root'],
            'runners_resolved_from': snap_rtx,
            'gpu4pyscf_package_root': os.path.join(
                plan['snapshot_root'], 'gpu4pyscf'),
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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# driver mode
# ---------------------------------------------------------------------------

def record_matches(record, cell, config_name, block, plan):
    try:
        return (record.get('schema_version') == SAMPLE_SCHEMA
                and record.get('config') == config_name
                and record.get('block') == block
                and record.get('cell_tag') == cell['tag']
                and record.get('amendment_id') == plan['amendment_id']
                and isinstance(record.get('row'), dict))
    except Exception:
        return False


def run_arm_subprocess(plan, cell, config_name, block, position, out_path,
                       log, max_retries):
    cmd = [PY, os.path.abspath(__file__), '--plan', plan['_plan_path'],
           '--arm', '--cell-tag', cell['tag'], '--config', config_name,
           '--block', str(block), '--out', out_path]
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        wait_gpu_idle(log)
        log(f'arm {cell["tag"]} {config_name} block {block} '
            f'attempt {attempts}')
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=arm_env(plan['configs'][config_name]),
                capture_output=True, text=True, timeout=plan['timeout_s'])
            rc = proc.returncode
            if rc != 0:
                log(f'  arm stderr tail: '
                    f'{(proc.stderr or "")[-800:]}')
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        log(f'arm {cell["tag"]} {config_name} block {block} rc={rc} '
            f'wall={time.time() - t0:.0f}s')
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding='utf-8') as f:
                    record = json.load(f)
                if record_matches(record, cell, config_name, block, plan):
                    if rc == 0:
                        return True, attempts
                    keep = f'{out_path}.attempt{attempts}.json'
                    os.replace(out_path, keep)
                    log(f'  failed attempt preserved at {keep}')
            except (OSError, ValueError) as exc:
                log(f'  malformed record: {exc!r}')
        if attempts >= max_retries:
            log(f'arm {cell["tag"]} {config_name} block {block}: FAILED '
                f'after {attempts} attempts; honest gap in the matrix')
            return False, attempts
    return False, attempts


# ---------------------------------------------------------------------------
# mechanical analysis (no verdicts; the session writes conclusions)
# ---------------------------------------------------------------------------

def reproduce_gate(sample_e, reference_e, tolerance):
    '''Each formal sample must sit within tolerance of the batch's
    reference per-block energies (the plan freezes them from the
    anchored batch index).'''
    return [abs(e - r) <= tolerance
            for e, r in zip(sample_e, reference_e)]


def attribution(shifts, full_shift, rules):
    '''shifts: {config: median(e) - median(e_R0)} for the single-knob
    configs; full_shift: median(e_FULL) - median(e_R0).  Classification
    is mechanical against the preregistered thresholds:
      clean      |shift| <= rules['clean_ha']
      carries    |shift| >= rules['carry_fraction'] * |full_shift|
      partial    otherwise
    The overall label names the follow-up the rules prescribe; it is a
    classification, never a verdict.'''
    out = {}
    for name, shift in shifts.items():
        if abs(shift) <= rules['clean_ha']:
            label = 'clean'
        elif abs(shift) >= rules['carry_fraction'] * abs(full_shift):
            label = 'carries'
        else:
            label = 'partial'
        out[name] = {'shift_ha': shift, 'label': label,
                     'share_of_full': (shift / full_shift
                                       if full_shift else None)}
    carrying = [n for n, v in out.items() if v['label'] == 'carries']
    partial = [n for n, v in out.items() if v['label'] == 'partial']
    # single-knob-cause requires EXACTLY one carrying knob and NO
    # partials (the plan rules: two+ carrying or any partial routes to
    # the pair/interaction follow-up)
    if len(carrying) == 1 and not partial:
        overall = 'single-knob-cause'
    elif carrying or partial:
        overall = 'pair-or-interaction-follow-up'
    else:
        overall = 'no-single-knob-shift (interaction only, or null)'
    return {'configs': out, 'overall': overall,
            'carrying': carrying, 'partial': partial}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def median_block_energies(samples_dir, cell, config_name, blocks):
    '''Per-block e_tot of one (cell, config), block order sorted.'''
    es = []
    for block in range(blocks):
        p = os.path.join(samples_dir,
                         f'{cell["tag"]}_{config_name}_b{block}.json')
        if not os.path.isfile(p):
            return None
        with open(p, encoding='utf-8') as f:
            record = json.load(f)
        es.append(record['row']['result']['e_tot'])
    return es


def _flatten_gradient(g):
    out = []
    for part in g:
        if isinstance(part, (list, tuple)):
            out.extend(float(x) for x in part)
        else:
            out.append(float(part))
    return out


def median_block_gradient(samples_dir, cell, config_name, blocks):
    '''Component-wise median across blocks of one (cell, config)'s flat
    gradient (the reported result.gradient, worst-repeat).  None when a
    block is missing or carries no gradient.'''
    per_block = []
    for block in range(blocks):
        p = os.path.join(samples_dir,
                         f'{cell["tag"]}_{config_name}_b{block}.json')
        if not os.path.isfile(p):
            return None
        with open(p, encoding='utf-8') as f:
            record = json.load(f)
        g = ((record.get('row') or {}).get('result') or {}).get('gradient')
        if not g:
            return None
        per_block.append(_flatten_gradient(g))
    n = len(per_block[0])
    return [statistics.median(col) for col in zip(*per_block)]


def _grad_dmax(a, b):
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _grad_cos(d_a, d_b):
    na = sum(x * x for x in d_a) ** 0.5
    nb = sum(x * x for x in d_b) ** 0.5
    if na == 0 or nb == 0:
        return None
    return sum(x * y for x, y in zip(d_a, d_b)) / (na * nb)


def gradient_attribution(g_r0, g_full, per_config, rules):
    '''Mechanical gradient-side classification mirroring attribution():
    the shift metric is max|dg| vs R0; carrying uses the same
    carry_fraction rule; a knob is only named a gradient-signature
    carrier when its difference also points the same way as FULL's
    (direction cos >= 0.9, the plan's direction_rule).  Labels are
    classifications, never verdicts.'''
    full_dmax = _grad_dmax(g_full, g_r0)
    d_full = [x - y for x, y in zip(g_full, g_r0)]
    out = {}
    for name, g in per_config.items():
        d = _grad_dmax(g, g_r0)
        cos = _grad_cos([x - y for x, y in zip(g, g_r0)], d_full)
        if d <= rules.get('clean_dg', 0.0):
            label = 'clean'
        elif d >= rules['carry_fraction'] * full_dmax:
            label = 'carries'
        else:
            label = 'partial'
        if label == 'carries' and (cos is None or cos < 0.9):
            label = 'partial (direction mismatch)'
        out[name] = {'max_dg_vs_r0': d, 'cos_vs_full': cos, 'label': label,
                     'share_of_full': d / full_dmax if full_dmax else None}
    return {'full_max_dg': full_dmax, 'configs': out}


def analyze(plan, out_json, out_md=None):
    '''Mechanical report for the plan's amendment (A4 and, with its
    extra keys, A5): reproduce gates (R0 vs batch R, FULL vs batch B,
    per-block, plan tolerance) and, only when both anchors reproduce,
    the attribution classification (single-knob shifts vs the FULL shift
    under the plan's rules; A5 adds the gradient-side mirror when
    gradient_attribution is set).  Classification only -- conclusions
    are written by the session, never here.'''
    ev = plan['evidence_root']
    samples_dir = os.path.join(ev, 'samples')
    blocks = plan['blocks_per_config']
    report = {'amendment_id': plan['amendment_id'],
              'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                time.gmtime()),
              'cells': {}, 'reproduce_all_ok': True}
    for cell in plan['cells']:
        tag = cell['tag']
        entry = {}
        refs = cell.get('batch_reference_energies', {})
        grefs = cell.get('batch_reference_gradients', {})
        tol_dg = plan.get('reproduce_tolerance_dg')
        scheduled = cell.get('configs_scheduled', list(plan['configs']))
        anchors = [('R0', 'R0', 'strict_fp64_R'),
                   ('FULL', 'FULL', 'optimized_fp64_B')]
        if 'FULL_AUTO' in scheduled:
            anchors.append(('FULL_AUTO', 'FULL_AUTO', 'optimized_auto_C'))
        for label, config_name, ref_key in anchors:
            got = median_block_energies(samples_dir, cell, config_name,
                                        blocks)
            ref = refs.get(ref_key)
            if got is None or ref is None:
                entry[label] = {'status': 'missing samples'}
                report['reproduce_all_ok'] = False
                continue
            gates = reproduce_gate(got, ref, plan['reproduce_tolerance_ha'])
            anchor = {
                'status': 'reproduced' if all(gates) else 'REPRODUCE FAIL',
                'sample_e': got, 'reference_e': ref, 'per_block_ok': gates}
            if tol_dg is not None and grefs.get(ref_key):
                g_med = median_block_gradient(samples_dir, cell,
                                              config_name, blocks)
                if g_med is None:
                    anchor['status'] = 'missing gradient samples'
                    report['reproduce_all_ok'] = False
                else:
                    dg = _grad_dmax(g_med, grefs[ref_key])
                    anchor['max_dg_vs_batch_ref'] = dg
                    anchor['gradient_ok'] = dg <= tol_dg
                    if not anchor['gradient_ok']:
                        report['reproduce_all_ok'] = False
            entry[label] = anchor
            if entry[label]['status'] != 'reproduced':
                report['reproduce_all_ok'] = False
        # new-condition arms (no batch reference) are recorded, not gated
        for config_name in scheduled:
            if config_name.endswith('_AUTO') and config_name != 'FULL_AUTO':
                got = median_block_energies(samples_dir, cell, config_name,
                                            blocks)
                if got is not None:
                    entry[config_name] = {
                        'status': 'new-condition (no batch reference)',
                        'sample_e': got}
        # attribution only on reproducing anchors
        e_r0 = median_block_energies(samples_dir, cell, 'R0', blocks)
        e_full = median_block_energies(samples_dir, cell, 'FULL', blocks)
        if (report['reproduce_all_ok'] and e_r0 is not None
                and e_full is not None):
            full_shift = (statistics.median(e_full)
                          - statistics.median(e_r0))
            shifts = {}
            for config_name in scheduled:
                if config_name in ('R0', 'FULL'):
                    continue
                es = median_block_energies(samples_dir, cell, config_name,
                                          blocks)
                if es is None:
                    continue
                shifts[config_name] = (statistics.median(es)
                                       - statistics.median(e_r0))
            entry['attribution'] = attribution(
                shifts, full_shift, plan['attribution_rules'])
            entry['full_shift_ha'] = full_shift
            if plan.get('gradient_attribution'):
                g_r0 = median_block_gradient(samples_dir, cell, 'R0',
                                             blocks)
                g_full = median_block_gradient(samples_dir, cell, 'FULL',
                                                blocks)
                per_config = {}
                for config_name in scheduled:
                    if config_name in ('R0', 'FULL'):
                        continue
                    g = median_block_gradient(samples_dir, cell, config_name,
                                              blocks)
                    if g is not None:
                        per_config[config_name] = g
                if g_r0 is not None and g_full is not None and per_config:
                    entry['gradient_attribution'] = gradient_attribution(
                        g_r0, g_full, per_config, plan['attribution_rules'])
                if 'R0_AUTO' in scheduled:
                    g_auto = median_block_gradient(samples_dir, cell,
                                                   'R0_AUTO', blocks)
                    if g_auto is not None and g_r0 is not None:
                        entry['R0_AUTO']['max_dg_vs_r0'] = \
                            _grad_dmax(g_auto, g_r0)
                if 'FULL_AUTO' in scheduled:
                    g_fauto = median_block_gradient(samples_dir, cell,
                                                    'FULL_AUTO', blocks)
                    if g_fauto is not None and g_r0 is not None:
                        entry.setdefault('FULL_AUTO', {})[
                            'max_dg_vs_r0'] = _grad_dmax(g_fauto, g_r0)
        report['cells'][tag] = entry

    with open(out_json + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    os.replace(out_json + '.tmp', out_json)
    if out_md:
        lines = [f"# Knob-isolation {plan['amendment_id']} mechanical "
                 f"report ({report['generated_at_utc']})",
                 '', f"Reproduce gates all OK: {report['reproduce_all_ok']}", '']
        for tag, entry in report['cells'].items():
            lines.append(f"## {tag}")
            for anchor in ('R0', 'FULL', 'FULL_AUTO'):
                a = entry.get(anchor)
                if not a:
                    continue
                lines.append(f"- {anchor}: {a.get('status')}")
                if a.get('max_dg_vs_batch_ref') is not None:
                    lines.append(f"  - gradient reproduce max|dg| vs batch "
                                 f"ref: {a['max_dg_vs_batch_ref']:.3e} "
                                 f"({'ok' if a.get('gradient_ok') else 'FAIL'})")
            for name in sorted(k for k in entry
                              if k.endswith('_AUTO') and k != 'FULL_AUTO'):
                a = entry[name]
                if isinstance(a, dict) and a.get('status'):
                    lines.append(f"- {name}: {a['status']}"
                                 + (f", max|dg| vs R0 "
                                    f"{a['max_dg_vs_r0']:.3e}"
                                    if a.get('max_dg_vs_r0') is not None
                                    else ''))
            att = entry.get('attribution')
            if att:
                lines.append(f"- full shift {entry['full_shift_ha']:+.3e} Ha")
                for name, v in att['configs'].items():
                    lines.append(f"- {name}: {v['shift_ha']:+.3e} Ha "
                                 f"({v['label']}"
                                 + (f", share {v['share_of_full']:.0%}"
                                    if v['share_of_full'] is not None else '')
                                 + ')')
                lines.append(f"- classification: {att['overall']}")
            gatt = entry.get('gradient_attribution')
            if gatt:
                lines.append(f"- FULL gradient max|dg| vs R0: "
                             f"{gatt['full_max_dg']:.3e}")
                for name, v in gatt['configs'].items():
                    cos = v['cos_vs_full']
                    lines.append(f"- {name}: max|dg| {v['max_dg_vs_r0']:.3e} "
                                 f"({v['label']}"
                                 + (f", cos {cos:.2f}"
                                    if cos is not None else '')
                                 + ')')
            if entry.get('FULL_AUTO', {}).get('max_dg_vs_r0') is not None:
                lines.append(f"- FULL_AUTO gradient max|dg| vs R0: "
                             f"{entry['FULL_AUTO']['max_dg_vs_r0']:.3e}")
            lines.append('')
        with open(out_md + '.tmp', 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(out_md + '.tmp', out_md)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_knob_isolation',
        description='Post-batch single-knob isolation matrix '
                    '(preregistrations A4 epinephrine energy exceedance '
                    'and A5 methoxy gradient family; runs only after '
                    'batch uc-20261018-a is COMPLETE).')
    p.add_argument('--plan', required=True)
    p.add_argument('--index', default=None,
                   help='the batch upstream-effects-index.json '
                        '(completeness gate; required in driver mode '
                        'only -- arm and analyze modes do not read it)')
    p.add_argument('--max-arms', type=int, default=None,
                   help='stop after this many NEW samples (resumable)')
    p.add_argument('--arm', action='store_true',
                   help='internal: run ONE arm (spawned by the driver)')
    p.add_argument('--cell-tag', default=None)
    p.add_argument('--config', default=None)
    p.add_argument('--block', type=int, default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--analyze', action='store_true',
                   help='do not run arms; mechanically classify the '
                        'finished samples under the plan rules')
    args = p.parse_args(argv)

    plan = load_plan(args.plan)
    plan['_plan_path'] = os.path.abspath(args.plan)

    if args.arm:
        cell = next(c for c in plan['cells'] if c['tag'] == args.cell_tag)
        pos = positions_for(list(plan['configs']), plan['blocks_per_config'])
        run_arm(plan, cell, args.config, args.block,
                pos[(args.config, args.block)], args.out)
        return 0

    if args.analyze:
        report = analyze(plan,
                         os.path.join(plan['evidence_root'],
                                      'knob-isolation-report.json'),
                         os.path.join(plan['evidence_root'],
                                      'knob-isolation-report.md'))
        print(f"reproduce_all_ok={report['reproduce_all_ok']}; "
              'classification is mechanical only; conclusions are '
              'written by the session')
        return 0

    # driver mode guards: one GPU workload, batch must be complete, and
    # the snapshot must still be exactly the batch's frozen snapshot
    refuse_if_any_driver()
    if not args.index:
        raise RuntimeError(
            'driver mode requires --index (the batch completeness gate); '
            'only --arm and --analyze modes run without it')
    if not batch_complete(args.index, plan['batch_id']):
        raise RuntimeError(
            f'batch {plan["batch_id"]} is not complete; the knob-isolation '
            'rerun is preregistered to run only AFTER the whole batch '
            '(single-GPU rule); refusing')
    with open(plan['schedule_path'], encoding='utf-8') as f:
        schedule = json.load(f)
    n_files = verify_snapshot_inventory(plan['snapshot_root'],
                                        schedule['source_commit'])

    ev = plan['evidence_root']
    os.makedirs(ev, exist_ok=True)
    samples_dir = os.path.join(ev, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    log_path = os.path.join(ev, 'knob-isolation.log')
    log_file = open(log_path, 'a', encoding='utf-8')

    def log(msg):
        line = (f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] '
                f'{msg}')
        print(line, flush=True)
        log_file.write(line + '\n')
        log_file.flush()

    log(f'driver start; plan {plan["_plan_path"]} sha256 '
        f'{sha256_file(plan["_plan_path"])}')
    log(f'snapshot {plan["snapshot_root"]} verified against the batch '
        f'inventory: {n_files} files, 0 missing/0 extra/0 changed')

    config_names = list(plan['configs'])
    positions = positions_for(config_names, plan['blocks_per_config'])
    done = 0
    for cell in plan['cells']:
        # per-cell matrix: the wb97mv cell adds the VV10-only arm; the
        # others schedule the five shared configs (preregistered)
        cell_configs = cell.get('configs_scheduled', config_names)
        for config_name in cell_configs:
            for block in range(plan['blocks_per_config']):
                out_path = os.path.join(
                    samples_dir,
                    f"{cell['tag']}_{config_name}_b{block}.json")
                if os.path.isfile(out_path):
                    continue
                if args.max_arms is not None and done >= args.max_arms:
                    log(f'stopping at --max-arms {args.max_arms} '
                        '(resumable)')
                    return 0
                ok, _ = run_arm_subprocess(
                    plan, cell, config_name, block,
                    positions[(config_name, block)], out_path, log,
                    args.max_retries)
                if ok:
                    done += 1
    log(f"matrix complete; run the {plan['amendment_id']} analysis "
        'against the samples (classification only; conclusions are '
        'written by the session)')
    return 0


if __name__ == '__main__':
    sys.exit(main())

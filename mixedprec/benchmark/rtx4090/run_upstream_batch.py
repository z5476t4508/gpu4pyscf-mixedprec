#!/usr/bin/env python3
"""Detached serial driver for the four-arm upstream comparison batch
(control card 2026-10-18).

Per the card's 配对预注册与实跑 rules:
  * the source/binary snapshot is a REAL independent frozen copy (cp -a,
    including every compiled .so), made read-only, hash-inventoried --
    never live-tree .so that a later rebuild could change mid-batch;
  * one GPU workload at a time: serial arms, GPU-idle-gated before every
    launch; the driver never runs alongside another GPU process;
  * each (block, arm) = one independent process with its own warmup and
    exactly one formal sample (balanced-blocks-v1); all four formal
    samples per role are kept;
  * resume-safe: completed sample records are never re-run, failed ones
    retry up to --max-retries (default 3) and the last attempt is always
    preserved;
  * cells are pre-listed in the schedule/index denominator; a cell with
    any missing arm stays incomplete and is never dropped;
  * after each completed cell the four comparison documents are merged,
    validated against the reporter's load_cell contract, and the
    upstream-effects-index is updated ATOMICALLY (tmp + os.replace).

Run detached:  setsid nohup <python> run_upstream_batch.py ... &
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import upstream_comparison as uc                            # noqa: E402
import upstream_schedule as us                             # noqa: E402

INDEX_DEFAULT = ('/data/work/gpu4pyscf-agent/state/long-chain/'
                 'upstream-effects-index.json')
PY = '/home/qczcode/gpu4090-src/bin/python'


def log(line, path):
    stamped = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ")}] {line}'
    print(stamped, flush=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(stamped + '\n')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# frozen snapshot
# ---------------------------------------------------------------------------

def create_snapshot(source_root, snapshot_root, commit, log_path):
    '''Real independent copy of the source tree (code + compiled .so),
    read-only, with a full sha256 inventory.  Never a symlink farm.'''
    if os.path.exists(snapshot_root):
        raise RuntimeError(f'snapshot {snapshot_root} already exists; '
                           'refusing to overwrite a frozen snapshot')
    exclude = shutil.ignore_patterns('.git', '__pycache__', '*.pyc',
                                      '.pytest_cache')
    log(f'snapshot: copying {source_root} -> {snapshot_root}', log_path)
    shutil.copytree(source_root, snapshot_root, ignore=exclude)
    marker = os.path.join(snapshot_root, '.upstream-batch-source-commit')
    with open(marker, 'w', encoding='utf-8') as f:
        f.write(commit + '\n')
    inventory = {}
    for dirpath, dirnames, filenames in os.walk(snapshot_root):
        dirnames[:] = sorted(d for d in dirnames
                             if d not in ('.git', '__pycache__'))
        for fn in sorted(filenames):
            p = os.path.join(dirpath, fn)
            inventory[os.path.relpath(p, snapshot_root)] = sha256_file(p)
    inv_path = snapshot_root + '.inventory.json'
    with open(inv_path, 'w', encoding='utf-8') as f:
        json.dump({'source_commit': commit,
                   'source_root': os.path.abspath(source_root),
                   'files': inventory}, f, indent=1)
    # read-only: the measurement must not be perturbed by later edits
    for dirpath, dirnames, filenames in os.walk(snapshot_root):
        for fn in filenames:
            os.chmod(os.path.join(dirpath, fn), 0o444)
        for dn in dirnames:
            os.chmod(os.path.join(dirpath, dn), 0o555)
    os.chmod(snapshot_root, 0o555)
    log(f'snapshot: frozen {len(inventory)} files read-only, inventory '
        f'{inv_path}', log_path)
    return inv_path


def verify_snapshot(snapshot_root, commit, log_path):
    marker = os.path.join(snapshot_root, '.upstream-batch-source-commit')
    if not os.path.isfile(marker):
        raise RuntimeError('snapshot missing its source-commit marker')
    with open(marker, encoding='utf-8') as f:
        frozen = f.read().strip()
    if frozen != commit:
        raise RuntimeError(f'snapshot commit {frozen} != schedule commit '
                           f'{commit}; refusing to mix source versions')
    arm = os.path.join(snapshot_root, 'mixedprec', 'benchmark', 'rtx4090',
                       'upstream_comparison.py')
    if not os.path.isfile(arm):
        raise RuntimeError(f'snapshot arm script missing: {arm}')
    log('snapshot: verified', log_path)


# ---------------------------------------------------------------------------
# GPU idle gate (single GPU workload discipline)
# ---------------------------------------------------------------------------

def gpu_busy():
    '''True when any process holds the GPU (compute apps) or utilization
    is nonzero -- the gate the driver waits on before every arm launch.'''
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
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu',
             '--format=csv,noheader,nounits'], capture_output=True,
            text=True, timeout=30).stdout.strip()
        return int(float(out.splitlines()[0])) > 3
    except Exception:
        return True


def wait_gpu_idle(log_path, timeout_s=7200):
    t0 = time.time()
    while gpu_busy():
        if time.time() - t0 > timeout_s:
            raise RuntimeError('GPU did not go idle within '
                               f'{timeout_s}s; refusing to overlap')
        time.sleep(20)
    log('gpu idle', log_path)


# ---------------------------------------------------------------------------
# arm execution
# ---------------------------------------------------------------------------

def arm_env(role):
    '''The subprocess environment per role: no inherited GPU4PYSCF_*
    variables, no PYTHONPATH; the complete tested stack only for B/C
    (exactly the STACK_ENV the arm asserts).'''
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GPU4PYSCF_') and k != 'PYTHONPATH'}
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONUNBUFFERED'] = '1'
    if role in ('optimized_fp64', 'optimized_auto'):
        env.update(uc.STACK_ENV)
    return env


def sample_path(samples_dir, cell, role, block):
    return os.path.join(samples_dir,
                        f"{cell['tag']}_{role}_b{block}.json")


def record_matches(record, cell, role, block, batch_id, schedule_sha256):
    try:
        return (record.get('schema_version') == uc.SAMPLE_SCHEMA
                and record.get('role') == role
                and record.get('block') == block
                and record.get('cell_index') == cell['index']
                and record.get('batch_id') == batch_id
                and record.get('schedule_sha256') == schedule_sha256
                and isinstance(record.get('row'), dict))
    except Exception:
        return False


def run_arm_subprocess(snapshot_root, cell, role, block, out_path, settings,
                       log_path, max_retries):
    script = os.path.join(snapshot_root, 'mixedprec', 'benchmark',
                          'rtx4090', 'upstream_comparison.py')
    cmd = [PY, script, '--role', role, '--schedule', ARGS.schedule,
           '--cell-index', str(cell['index']), '--block', str(block),
           '--out', out_path]
    attempts = 0
    while attempts < max_retries:
        attempts += 1
        wait_gpu_idle(log_path)
        log(f"arm {cell['tag']} {role} block {block} attempt {attempts}",
            log_path)
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd, env=arm_env(role), capture_output=True, text=True,
                timeout=int(settings['timeout_s_per_arm']))
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            rc = 'timeout'
        log(f"arm {cell['tag']} {role} block {block} rc={rc} "
            f"wall={time.time() - t0:.0f}s", log_path)
        if os.path.isfile(out_path):
            try:
                with open(out_path, encoding='utf-8') as f:
                    record = json.load(f)
                if record_matches(record, cell, role, block, ARGS.batch_id,
                                  ARGS.schedule_sha256):
                    if rc == 0:
                        return True, attempts
                    # a failed attempt that still wrote a valid record:
                    # preserve it as attempt evidence, then retry
                    keep = f'{out_path}.attempt{attempts}.json'
                    os.replace(out_path, keep)
                    log(f'  failed attempt preserved at {keep}', log_path)
            except (OSError, ValueError) as exc:
                log(f'  malformed record: {exc!r}', log_path)
        if attempts >= max_retries:
            log(f"arm {cell['tag']} {role} block {block}: FAILED after "
                f"{attempts} attempts; the unit stays an honest gap",
                log_path)
            return False, attempts
    return False, attempts


# ---------------------------------------------------------------------------
# index maintenance (atomic; planned cells never dropped)
# ---------------------------------------------------------------------------

def load_or_create_index(index_path, batch_id, schedule):
    if os.path.isfile(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
    else:
        index = {'batches': []}
    batches = [b for b in index.get('batches', [])
               if isinstance(b, dict) and b.get('id') == batch_id]
    if batches:
        batch = batches[0]
        for b in index['batches']:
            if b.get('id') == batch_id and b is not batch:
                raise RuntimeError('duplicate batch entries')
    else:
        batch = {'id': batch_id, 'cells': [
            {'cell': {k: c[k] for k in
                      ('index', 'tag', 'geom', 'basis', 'xc', 'charge',
                       'spin')},
             'sources': {}} for c in schedule['cells']]}
        index.setdefault('batches', []).append(batch)
    return index, batch


def update_index(index_path, index, batch, cell, sources):
    '''Fill one cell's sources atomically; the planned denominator stays.'''
    for entry in batch['cells']:
        if entry['cell']['index'] == cell['index']:
            entry['sources'] = sources
            break
    else:
        raise RuntimeError(f"cell {cell['index']} not pre-listed in index")
    tmp = index_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=1)
    os.replace(tmp, index_path)


# ---------------------------------------------------------------------------
# merge + per-cell reporting
# ---------------------------------------------------------------------------

def merge_cell(samples_dir, out_dir, cell, schedule, batch_id,
               schedule_sha256, log_path):
    '''Load the 16 sample records, merge per role, validate, write the four
    cell_{role}.json comparison documents and the cell summary.'''
    role_docs, sources = {}, {}
    for role in uc.ROLES:
        records = []
        for block in range(4):
            path = sample_path(samples_dir, cell, role, block)
            if not os.path.isfile(path):
                log(f"merge {cell['tag']} {role}: block {block} missing",
                    log_path)
                return None
            with open(path, encoding='utf-8') as f:
                record = json.load(f)
            if not record_matches(record, cell, role, block, batch_id,
                                  schedule_sha256):
                log(f"merge {cell['tag']} {role} block {block}: record "
                    'mismatch', log_path)
                return None
            if record['row'].get('status') != 'ok':
                log(f"merge {cell['tag']} {role} block {block}: honest "
                    f"status {record['row'].get('status')!r}; cell stays "
                    'incomplete', log_path)
                return None
            record['_path'] = path
            records.append(record)
        doc = uc.merge_role(records, role, batch_id, schedule,
                            schedule_sha256)
        uc.validate_comparison_doc(doc, role, batch_id, schedule_sha256)
        role_docs[role] = doc
    # cross-role diagnostics vs the strict anchor (the reporter recomputes
    # the authoritative gates; this summary is the driver's own check)
    summary = {'cell': {k: cell[k] for k in
                        ('index', 'tag', 'geom', 'basis', 'xc', 'charge',
                         'spin')},
               'batch_id': batch_id, 'schedule_sha256': schedule_sha256,
               'roles': {}}
    import statistics as st
    ref = role_docs['strict_fp64']['rows'][0]['result']
    ref_reps = ref['repeats']
    for role, doc in role_docs.items():
        row = doc['rows'][0]
        res = row['result']
        de = dg = 0.0
        for a in res['repeats']:
            for b in ref_reps:
                de = max(de, abs(a['e_tot'] - b['e_tot']))
                ga, gb = a['gradient'], b['gradient']
                if len(ga) == len(gb):
                    dg = max(dg, max(abs(x - y) for ra, rb in zip(ga, gb)
                                     for x, y in zip(ra, rb)))
        entry = {
            'median_sample_s': st.median(r['sample_s']
                                         for r in res['repeats']),
            'median_t_scf_s': st.median(r['t_scf_s']
                                        for r in res['repeats']),
            'median_t_grad_s': st.median(r['t_grad_s']
                                         for r in res['repeats']),
            'max_dE_vs_strict': de, 'max_dg_vs_strict': dg,
            'all_converged': all(r['converged']
                                 for r in res['repeats']) and
            all(r['converged'] for r in ref_reps),
        }
        summary['roles'][role] = entry
    ref_med = summary['roles']['strict_fp64']['median_sample_s']
    for role, (num, den) in {
            'O/B': ('upstream', 'optimized_fp64'),
            'B/C': ('optimized_fp64', 'optimized_auto'),
            'O/C': ('upstream', 'optimized_auto')}.items():
        num_med = summary['roles'][num]['median_sample_s']
        den_med = summary['roles'][den]['median_sample_s']
        summary.setdefault('ratios', {})[role] = (
            round(num_med / den_med, 4) if den_med > 0 else None)
    for role, doc in role_docs.items():
        out_path = os.path.join(out_dir, f"{cell['tag']}_{role}.json")
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=1)
        sources[role] = {'path': out_path,
                         'sha256': sha256_file(out_path)}
    with open(os.path.join(out_dir, f"{cell['tag']}_summary.json"),
              'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1)
    return sources


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

ARGS = None


def main(argv=None):
    global ARGS
    p = argparse.ArgumentParser(
        prog='run_upstream_batch',
        description='Detached serial driver of the four-arm upstream '
                    'comparison batch (balanced-blocks-v1).')
    p.add_argument('--schedule', required=True)
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--snapshot-root', required=True)
    p.add_argument('--index', default=INDEX_DEFAULT)
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--max-cells', type=int, default=None,
                   help='stop after this many NEW completed cells (the '
                        'batch is cumulative; re-running continues where '
                        'it stopped)')
    p.add_argument('--no-snapshot', action='store_true',
                   help='skip snapshot creation/verification (testing '
                        'only)')
    args = p.parse_args(argv)
    ARGS = args

    with open(args.schedule, encoding='utf-8') as f:
        raw = f.read()
    schedule = json.loads(raw)
    schedule_sha256 = hashlib.sha256(raw.encode('utf-8')).hexdigest()
    us.verify_schedule(schedule)
    ARGS.schedule_sha256 = schedule_sha256
    ARGS.batch_id = schedule['batch_id']

    os.makedirs(args.evidence_root, exist_ok=True)
    samples_dir = os.path.join(args.evidence_root, 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    docs_dir = os.path.join(args.evidence_root, 'cells')
    os.makedirs(docs_dir, exist_ok=True)
    log_path = os.path.join(args.evidence_root, 'driver.log')

    if not args.no_snapshot:
        if not os.path.isdir(args.snapshot_root):
            create_snapshot(schedule['source_root'], args.snapshot_root,
                            schedule['source_commit'], log_path)
        verify_snapshot(args.snapshot_root, schedule['source_commit'],
                        log_path)

    index, batch = load_or_create_index(args.index, schedule['batch_id'],
                                        schedule)
    log(f'batch {schedule["batch_id"]}: {len(schedule["cells"])} planned '
        f'cells, schedule sha256 {schedule_sha256[:12]}', log_path)

    completed = 0
    for cell in schedule['cells']:
        done_entry = next((e for e in batch['cells']
                           if e['cell']['index'] == cell['index']
                           and e.get('sources')), None)
        if done_entry:
            log(f"cell {cell['index']} {cell['tag']}: already complete",
                log_path)
            continue
        try:
            # run the 16 (block, arm) units in the frozen balanced order
            for block in range(4):
                for position, short in enumerate(cell['blocks'][block]):
                    role = us.ROLE_NAMES[short]
                    out_path = sample_path(samples_dir, cell, role, block)
                    if os.path.isfile(out_path):
                        try:
                            with open(out_path, encoding='utf-8') as f:
                                record = json.load(f)
                            if record_matches(record, cell, role, block,
                                              schedule['batch_id'],
                                              schedule_sha256):
                                log(f"cell {cell['tag']} {role} b{block}: "
                                    'record exists (resume)', log_path)
                                continue
                        except (OSError, ValueError):
                            pass
                    ok, _ = run_arm_subprocess(
                        args.snapshot_root, cell, role, block, out_path,
                        schedule['settings'], log_path, args.max_retries)
                    if not ok:
                        log(f"cell {cell['tag']}: arm {role} block {block} "
                            'unresolvable; moving on (unit stays in '
                            'denominator)', log_path)
            sources = merge_cell(samples_dir, docs_dir, cell, schedule,
                                 schedule['batch_id'], schedule_sha256,
                                 log_path)
        except Exception as exc:
            log(f"cell {cell['tag']}: driver-level exception {exc!r}; "
                'the cell stays an honest gap and the batch continues',
                log_path)
            sources = None
        if sources:
            update_index(args.index, index, batch, cell, sources)
            completed += 1
            ratios = json.load(open(os.path.join(
                docs_dir, f"{cell['tag']}_summary.json")))['ratios']
            log(f"cell {cell['tag']}: COMPLETE {ratios}", log_path)
            if args.max_cells and completed >= args.max_cells:
                log(f'stopping after {completed} new cells (--max-cells)',
                    log_path)
                break
        else:
            log(f"cell {cell['tag']}: INCOMPLETE (honest gap, stays in "
                'denominator)', log_path)
    log('batch pass finished', log_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""LC-205 smoke: small-system 4-method O-lane compatibility check of the
official GPU4PySCF 1.8.1 wheel under the unified dependency environment.

Runs AFTER the LC-204 batch truly ends (never overlapping the GPU; the
driver's own GPU gate is reused).  Creates the frozen source/binary
snapshot first (the smoke runs FROM the snapshot -- the same frozen code
the formal batch will run), then four upstream-arm processes (warmup +
one formal sample each): b3lyp, m062x, wb97mv (NLC/VV10), r2scan3c
(D4+gCP gradient hook).  Any incompatibility is preserved as evidence for
bounded adapter debugging per the control card.
"""

import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
PY = '/home/qczcode/gpu4090-src/bin/python'
EV = ('/data/results/gpu4pyscf-agent/tasks/issue-9/'
      '2026-10-18-lc205-upstream-comparison-batch')
# the frozen snapshot lives INSIDE the evidence directory: it is first-class
# batch evidence (its sha256 inventory is anchored with the prereg amendment);
# /data/work/gpu4pyscf-agent itself is root-owned read-only
SNAP_ROOT = os.path.join(EV, 'snapshot')
SCHEDULE_OUT = os.path.join(EV, 'smoke', 'schedule.json')


def log(msg):
    print(f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ")}] {msg}', flush=True)


def main():
    sys.path.insert(0, HERE)
    import run_upstream_batch as rb
    import upstream_schedule as us

    commit = subprocess.run(['git', '-C', REPO, 'rev-parse', 'HEAD'],
                            capture_output=True, text=True).stdout.strip()
    if not commit:
        raise RuntimeError('cannot resolve HEAD; commit the LC-205 code '
                           'before freezing the snapshot')
    os.makedirs(os.path.join(EV, 'smoke'), exist_ok=True)
    if not os.path.isdir(SNAP_ROOT):
        rb.create_snapshot(REPO, SNAP_ROOT, commit,
                           os.path.join(EV, 'smoke', 'snapshot.log'))
    rb.verify_snapshot(SNAP_ROOT, commit, os.path.join(EV, 'smoke',
                                                       'snapshot.log'))

    # import the schedule builder FROM THE SNAPSHOT (frozen code path)
    sys.path.insert(0, os.path.join(SNAP_ROOT, 'mixedprec', 'benchmark',
                                    'rtx4090'))
    snap_us = __import__('upstream_schedule')

    specs = json.load(open(os.path.join(EV, 'smoke_cells.json')))
    schedule = {'schema_version': snap_us.SCHEMA, 'seed': snap_us.SEED,
                'batch_id': 'uc-20261018-a-smoke',
                'source_commit': commit, 'source_root': SNAP_ROOT,
                'timing_protocol': 'balanced-blocks-v1',
                'settings': snap_us.SETTINGS,
                'cells': snap_us.build_cells(specs)}
    snap_us.verify_schedule(schedule)
    with open(SCHEDULE_OUT, 'w', encoding='utf-8') as f:
        json.dump(schedule, f, indent=1)
    import hashlib
    log(f'smoke schedule frozen: {SCHEDULE_OUT} sha256 '
        f'{hashlib.sha256(open(SCHEDULE_OUT, "rb").read()).hexdigest()}')

    arm = os.path.join(SNAP_ROOT, 'mixedprec', 'benchmark', 'rtx4090',
                       'upstream_comparison.py')
    log_path = os.path.join(EV, 'smoke', 'smoke.log')
    rc_total = 0
    for cell in schedule['cells']:
        rb.wait_gpu_idle(log_path)
        tag = cell['tag']
        out = os.path.join(EV, 'smoke', f'{tag}_O_b0.json')
        cmd = [PY, arm, '--role', 'upstream', '--schedule', SCHEDULE_OUT,
               '--cell-index', str(cell['index']), '--block', '0',
               '--out', out]
        log(f'smoke {tag}: launching O arm')
        t0 = time.time()
        proc = None
        try:
            proc = subprocess.run(cmd, env=rb.arm_env('upstream'),
                                 capture_output=True, text=True, timeout=1200)
            rc = proc.returncode
        except subprocess.TimeoutExpired as exc:
            rc = 'timeout'
            proc = SimpleNamespace(
                stdout=(exc.stdout or b'').decode()
                if isinstance(exc.stdout, bytes) else (exc.stdout or ''),
                stderr=(exc.stderr or b'').decode()
                if isinstance(exc.stderr, bytes) else (exc.stderr or ''))
        log(f'smoke {tag}: rc={rc} wall={time.time() - t0:.0f}s')
        stdout_log = os.path.join(EV, 'smoke', f'{tag}_O_b0.stdout.log')
        with open(stdout_log, 'w', encoding='utf-8') as f:
            f.write(proc.stdout if isinstance(proc.stdout, str) else '')
            f.write('\n--- stderr ---\n')
            f.write(proc.stderr if isinstance(proc.stderr, str) else '')
        if rc != 0 or not os.path.isfile(out):
            rc_total = rc_total or 1
            log(f'smoke {tag}: FAILED (rc={rc}); evidence preserved in '
                f'{stdout_log} and the sample record dir')
        else:
            rec = json.load(open(out))
            row = rec['row']
            log(f'smoke {tag}: OK status={row["status"]} '
                f'E={row["result"]["e_tot"]:.12f} '
                f'conv={row["result"]["converged"]}')
    log(f'SMOKE COMPLETE rc_total={rc_total} '
        '(0 = the official wheel is compatible with all four methods '
        'under the unified environment)')
    return rc_total


if __name__ == '__main__':
    sys.exit(main())

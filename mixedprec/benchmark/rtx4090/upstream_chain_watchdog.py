#!/usr/bin/env python3
'''CPU-only watchdog for the chained upstream-comparison batch.

The LC-205b fill chain is instance-chained: one launcher PID waits on one
stage-1 driver PID, then launches the fill driver and exits.  If the
launcher (or the fill driver it launched) dies, the multi-day cumulative
batch would leave the GPU idle until a session follows the runbook.  This
watchdog automates that runbook WITHOUT ever touching the GPU:

  - one instance (flock on <evidence-root>/watchdog.lock);
  - every poll reads the batch index and scans /proc;
  - batch complete (all pre-listed cells carry all four role sources)
    -> exit 0;
  - ANY live run_upstream_batch.py driver (ours, or a foreign batch's --
    the box runs one GPU workload) -> healthy, nothing to do;
  - a live launcher that is not wedged -> healthy, nothing to do
    ("wedged" = no driver exists, fill-launch.json was never written, and
    this state persisted past --launcher-stuck-after-s: the launcher is
    stuck on a reused PID or otherwise failed to fire);
  - nothing drives the batch and nothing will -> re-arm the fill launcher
    in --assume-stage1-done mode (detached): it preserves the stage-1
    artifacts idempotently, refuses to double-drive, and launches the
    resume-safe fill driver on the cumulative schedule;
  - re-arms are bounded (--max-rearms) with a minimum spacing
    (--min-spacing-s); budget exhausted with the batch incomplete exits 3.

It never kills, restarts, or preempts anything; all GPU-side discipline
(GPU-idle gates, one-workload, bounded retries) stays inside the driver it
(re)launches.  Run detached:
  setsid nohup <python> upstream_chain_watchdog.py --batch-id <id> ...
      --log <watchdog structured log> >> <watchdog stdout log> 2>&1 &
'''

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

PY = '/home/qczcode/gpu4090-src/bin/python'
HERE = os.path.dirname(os.path.abspath(__file__))

DRIVER_NEEDLE = 'run_upstream_batch.py'
LAUNCHER_NEEDLE = 'upstream_fill_launcher.py'
WATCHDOG_NEEDLE = 'upstream_chain_watchdog.py'
ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


class Log:
    def __init__(self, path):
        self.path = path

    def __call__(self, line):
        stamped = f'[{utc()}] {line}'
        print(stamped, flush=True)
        with open(self.path, 'a', encoding='utf-8') as f:
            f.write(stamped + '\n')


def _cmdline_of(pid, proc='/proc'):
    try:
        with open(os.path.join(proc, str(pid), 'cmdline'), 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return None


def _runs_script(elems, script):
    '''True when an argv element actually IS the script (equal to it or
    ending in "/<script>"): a lingering shell wrapper whose -c string
    merely mentions the script does not count.'''
    return any(e == script or e.endswith(f'/{script}') for e in elems)


def scan(script, exclude=(), proc='/proc', self_pid=None,
         must_contain=()):
    '''[(pid, joined argv)] of live processes actually RUNNING `script`.
    Processes also running any script in `exclude`, and our own PID, are
    skipped; every string in `must_contain` must appear in the joined
    argv (e.g. this batch's evidence root).'''
    found = []
    me = os.getpid() if self_pid is None else self_pid
    for pid in os.listdir(proc):
        if not pid.isdigit() or int(pid) == me:
            continue
        cmd = _cmdline_of(pid, proc)
        if not cmd:
            continue
        elems = [c for c in cmd if c]
        if not _runs_script(elems, script):
            continue
        if any(_runs_script(elems, x) for x in exclude):
            continue
        joined = ' '.join(elems)
        if not all(s in joined for s in must_contain):
            continue
        found.append((int(pid), joined))
    return found


def drivers_alive(proc='/proc'):
    '''All live batch drivers (any batch; the box runs one GPU workload).'''
    return scan(DRIVER_NEEDLE, exclude=(LAUNCHER_NEEDLE, WATCHDOG_NEEDLE),
                proc=proc)


def launcher_alive_for(evidence_root, proc='/proc'):
    '''Live fill launchers for THIS evidence root (another batch's
    launcher does not count as ours).'''
    return scan(LAUNCHER_NEEDLE, exclude=(WATCHDOG_NEEDLE,), proc=proc,
                must_contain=(evidence_root,))


def batch_state(index_path, batch_id):
    '''(total, done) for the batch, or None when unreadable / unknown;
    `done` counts cells with all four role sources present.'''
    try:
        with open(index_path, encoding='utf-8') as f:
            idx = json.load(f)
    except (OSError, ValueError):
        return None
    for batch in idx.get('batches', []):
        if batch.get('id') == batch_id:
            cells = batch.get('cells', [])
            done = sum(
                1 for c in cells
                if isinstance(c.get('sources'), dict)
                and all(r in c['sources'] for r in ROLES))
            return len(cells), done
    return None


def decide(batch_complete, any_driver_alive, launcher_alive,
           launcher_wedged, rearm_count, last_rearm_age_s, max_rearms,
           min_spacing_s):
    '''Pure decision table (unit-tested).  Returns one of:
    exit_complete | none_healthy | exit_budget | none_spacing | rearm.'''
    if batch_complete:
        return 'exit_complete'
    if any_driver_alive:
        return 'none_healthy'
    if launcher_alive and not launcher_wedged:
        return 'none_healthy'
    if rearm_count >= max_rearms:
        return 'exit_budget'
    if last_rearm_age_s is not None and last_rearm_age_s < min_spacing_s:
        return 'none_spacing'
    return 'rearm'


def write_state_file(path, payload):
    tmp = f'{path}.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_chain_watchdog',
        description='CPU-only liveness watchdog for the chained '
                    'upstream-comparison batch: re-arms the fill launcher '
                    '(assume mode) when nothing drives the incomplete '
                    'batch; never touches the GPU.')
    p.add_argument('--batch-id', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--schedule', required=True,
                   help='the cumulative fill schedule')
    p.add_argument('--snapshot-root', required=True)
    p.add_argument('--launcher-script', default=os.path.join(
        HERE, 'upstream_fill_launcher.py'))
    p.add_argument('--launcher-log', default=None,
                   help='structured log of the (re-armed) launcher '
                        '(default <evidence-root>/fill-launcher.log)')
    p.add_argument('--launcher-stdout', default=None,
                   help='stdout capture for the re-armed launcher '
                        '(default <evidence-root>/fill-launcher.stdout.log)')
    p.add_argument('--log', required=True,
                   help='structured watchdog log path')
    p.add_argument('--poll-s', type=int, default=60)
    p.add_argument('--max-rearms', type=int, default=12)
    p.add_argument('--min-spacing-s', type=int, default=1800)
    p.add_argument('--launcher-stuck-after-s', type=int, default=7200)
    p.add_argument('--heartbeat-every', type=int, default=20)
    args = p.parse_args(argv)
    if args.launcher_log is None:
        args.launcher_log = os.path.join(args.evidence_root,
                                         'fill-launcher.log')
    if args.launcher_stdout is None:
        args.launcher_stdout = os.path.join(
            args.evidence_root, 'fill-launcher.stdout.log')
    log = Log(args.log)

    lock = open(os.path.join(args.evidence_root, 'watchdog.lock'), 'w',
                encoding='utf-8')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log('another watchdog holds the lock; exiting (single instance)')
        return 0
    lock.write(f'watchdog pid {os.getpid()}\n')
    lock.flush()
    log(f'watchdog up: batch {args.batch_id}; poll {args.poll_s}s; '
        f'max re-arms {args.max_rearms}; spacing {args.min_spacing_s}s; '
        f'launcher-stuck bound {args.launcher_stuck_after_s}s')

    rearm_count = 0
    last_rearm = None
    wedged_since = None
    prev_signature = None
    beats = 0
    state_path = os.path.join(args.evidence_root, 'watchdog-state.json')
    while True:
        state = batch_state(args.index, args.batch_id)
        drivers = drivers_alive()
        launchers = launcher_alive_for(args.evidence_root)
        fill_launched = os.path.isfile(os.path.join(
            args.evidence_root, 'fill-launch.json'))
        now = datetime.now(timezone.utc)

        total, done = state if state else (None, None)
        batch_complete = bool(state and total and total > 0
                               and done == total)
        launcher_wedged = bool(launchers) and not fill_launched
        wedged = bool(
            launcher_wedged and wedged_since is not None
            and (now - wedged_since).total_seconds()
            > args.launcher_stuck_after_s)
        age = (None if last_rearm is None
               else (now - last_rearm).total_seconds())
        action = decide(batch_complete, bool(drivers), bool(launchers),
                        wedged, rearm_count, age, args.max_rearms,
                        args.min_spacing_s)

        signature = (total, done, len(drivers), len(launchers),
                      fill_launched, wedged, action)
        if signature != prev_signature:
            log(f'state: cells {done}/{total}; drivers {len(drivers)}; '
                f'launchers {len(launchers)}; fill-launched {fill_launched}; '
                f'wedged {wedged}; -> {action}')
            prev_signature = signature
        beats += 1
        if beats % args.heartbeat_every == 0:
            log(f'heartbeat: re-arms {rearm_count}/{args.max_rearms}; '
                f'cells {done}/{total}')

        if action == 'exit_complete':
            log(f'batch {args.batch_id} complete ({done}/{total} cells); '
                'watchdog exiting')
            return 0
        if action == 'exit_budget':
            log(f'RE-ARM BUDGET EXHAUSTED ({rearm_count}/{args.max_rearms}) '
                f'with the batch incomplete ({done}/{total} cells); '
                'watchdog exiting nonzero -- runbook: launch the fill '
                'driver manually (command recorded in fill-launch.json '
                '/ fill-launcher.log)')
            return 3
        if action == 'rearm':
            cmd = [PY, args.launcher_script, '--assume-stage1-done',
                   '--schedule', args.schedule,
                   '--evidence-root', args.evidence_root,
                   '--snapshot-root', args.snapshot_root,
                   '--index', args.index,
                   '--log', args.launcher_log]
            log('re-arming the fill launcher (assume-stage1-done): '
                + ' '.join(cmd))
            with open(args.launcher_stdout, 'ab') as out:
                subprocess.Popen(cmd, stdout=out,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                                cwd=os.path.dirname(HERE))
            rearm_count += 1
            last_rearm = now
            wedged_since = None

        if launcher_wedged and wedged_since is None:
            wedged_since = now
            log('launcher present without fill-launch.json; starting '
                'the wedged-launcher clock')
        elif not launcher_wedged and wedged_since is not None:
            wedged_since = None
            log('wedged-launcher clock cleared')

        write_state_file(state_path, {
            'batch_id': args.batch_id,
            'cells_total': total, 'cells_done': done,
            'drivers_alive': len(drivers), 'launchers_alive': len(launchers),
            'fill_launched': fill_launched, 'launcher_wedged': wedged,
            'rearm_count': rearm_count,
            'last_rearm_utc': (None if last_rearm is None
                               else last_rearm.strftime(
                                   '%Y-%m-%dT%H:%M:%SZ')),
            'last_decision': action, 'updated_at_utc': utc(),
            'watchdog_pid': os.getpid()})
        time.sleep(args.poll_s)


if __name__ == '__main__':
    sys.exit(main())

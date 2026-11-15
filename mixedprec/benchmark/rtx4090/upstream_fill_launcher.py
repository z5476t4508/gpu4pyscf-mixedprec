#!/usr/bin/env python3
'''Chained fill launcher for the four-arm upstream comparison batch.

Normal mode (--wait-pid): wait for the running stage-1 batch driver (one
specific PID, cmdline-verified on every poll so a reused PID cannot wedge
the chain), preserve the stage-1 artifacts under
<evidence-root>/stage1-preserved/, write the timestamp-convention marker
into the shared driver.log, then launch the fill driver on the cumulative
schedule detached (same batch id, same frozen snapshot) so the single GPU
stays busy across the stage boundary without any session needing to be
alive.

Re-arm mode (--assume-stage1-done): the stage-1 driver has already exited
(e.g. the original launcher died, or the fill driver itself died mid-pass
and must be resumed); skip the wait and go straight to marker + preserve +
guarded launch.  Used by upstream_chain_watchdog.py and by the runbook.

Both modes are idempotent and fail-closed: the marker and fill-launch.json
are written only when absent (re-arms append to fill-relaunches.jsonl), the
preserve is atomic (partial dir + rename; first complete copy wins; an
interrupted preserve without an inventory is moved aside and redone from
the still-live originals), and the fill driver is only ever launched when
NO live run_upstream_batch.py process exists (double-drive refusal; the
driver's own GPU-idle gates keep the one-workload discipline).  The
launcher itself never touches the GPU.

Run detached:
  setsid nohup <python> upstream_fill_launcher.py --wait-pid <pid> ...
      --log <launcher structured log> >> <launcher stdout log> 2>&1 &
'''

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

PY = '/home/qczcode/gpu4090-src/bin/python'
HERE = os.path.dirname(os.path.abspath(__file__))

DRIVER_NEEDLE = 'run_upstream_batch.py'
LAUNCHER_NEEDLE = 'upstream_fill_launcher.py'
WATCHDOG_NEEDLE = 'upstream_chain_watchdog.py'
MARKER_TAG = '=== FILL CHAIN:'


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


def pid_cmdline(pid, proc='/proc'):
    try:
        with open(os.path.join(proc, str(pid), 'cmdline'), 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return None


def pid_is_driver(pid, proc='/proc'):
    '''cmdline-verified liveness: an exited (or reused) PID is no driver.'''
    cmd = pid_cmdline(pid, proc)
    return bool(cmd) and any(DRIVER_NEEDLE in c for c in cmd)


def find_live_driver(proc='/proc'):
    '''First live batch-driver PID, or None.  An argv element must actually
    BE the driver script (a lingering shell wrapper whose -c string merely
    mentions it does not count); launcher / watchdog processes that carry
    the driver path as an argument are excluded, as is our own PID (this
    launcher may be invoked with --driver).'''
    me = os.getpid()

    def runs(elems, script):
        return any(e == script or e.endswith(f'/{script}') for e in elems)

    for pid in os.listdir(proc):
        if not pid.isdigit() or int(pid) == me:
            continue
        cmd = pid_cmdline(pid, proc)
        if not cmd:
            continue
        elems = [c for c in cmd if c]
        if not runs(elems, DRIVER_NEEDLE):
            continue
        if runs(elems, LAUNCHER_NEEDLE) or runs(elems, WATCHDOG_NEEDLE):
            continue
        return int(pid)
    return None


def ensure_marker(evidence_root):
    '''Timestamp-convention boundary marker, written at most once.'''
    path = os.path.join(evidence_root, 'driver.log')
    line = (f'[{utc()}] === FILL CHAIN: the stage-1 driver exited; '
            'the fill driver launches next. All lines ABOVE this '
            'marker use local-CST stamps labeled Z (true UTC = stamp '
            'minus 8h); lines BELOW use true UTC ===\n')
    try:
        with open(path, encoding='utf-8') as f:
            if MARKER_TAG in f.read():
                return False
    except OSError:
        pass
    with open(path, 'a', encoding='utf-8') as f:
        f.write(line)
    return True


def preserve_stage1(evidence_root, log):
    '''Atomic, idempotent preservation of the stage-1 artifacts under
    <evidence-root>/stage1-preserved/ with a sha256 inventory (the fill
    only ever re-runs INCOMPLETE cells; the preserved copy keeps even those
    original partial records).  The first complete copy wins; a preserve
    interrupted before its inventory exists is moved aside (kept as honest
    partial evidence) and redone from the still-live originals.'''
    preserved = os.path.join(evidence_root, 'stage1-preserved')
    inventory = os.path.join(preserved, 'stage1-preserved-inventory.txt')
    stamp = utc().replace('-', '').replace(':', '')

    if os.path.isdir(preserved) and not os.path.isfile(inventory):
        aside = os.path.join(
            evidence_root, f'stage1-preserved.interrupted.{stamp}')
        log(f'existing {preserved} has no inventory (interrupted '
            f'preserve?); moving it aside to {aside} and re-copying '
            'from the live originals')
        os.rename(preserved, aside)

    if not os.path.isdir(preserved):
        tmp = os.path.join(evidence_root,
                           f'stage1-preserved.partial.{stamp}')
        for name in ('samples', 'cells'):
            src = os.path.join(evidence_root, name)
            if os.path.isdir(src):
                log(f'preserving {src} -> {os.path.join(tmp, name)}')
                shutil.copytree(src, os.path.join(tmp, name))
        driver_log = os.path.join(evidence_root, 'driver.log')
        if os.path.isfile(driver_log):
            shutil.copy2(driver_log,
                         os.path.join(tmp, 'driver-stage1.log'))
        try:
            os.rename(tmp, preserved)
        except OSError:
            log('another instance completed the preserve first; '
                'discarding this partial copy')
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        log('stage-1 artifacts already preserved; keeping the first copy')

    if not os.path.isfile(inventory):
        entries = []
        for dirpath, dirnames, filenames in os.walk(preserved):
            dirnames[:] = sorted(dirnames)
            for fn in sorted(filenames):
                full = os.path.join(dirpath, fn)
                digest = hashlib.sha256(
                    open(full, 'rb').read()).hexdigest()
                entries.append(f'{digest}  '
                               f'{os.path.relpath(full, preserved)}')
        with open(inventory, 'w', encoding='utf-8') as f:
            f.write('\n'.join(entries) + '\n')
        log(f'stage-1 artifacts preserved: {len(entries)} files '
            f'inventoried under {preserved}')
    else:
        log('preserve inventory already present')


def launch_fill(args, log, assume_mode):
    '''Guarded, detached launch of the fill driver; records the first
    launch in fill-launch.json and every later one in
    fill-relaunches.jsonl (source files are never overwritten).'''
    other = find_live_driver()
    if other is not None:
        log(f'ABORT: a live upstream batch driver exists (pid {other}); '
            'refusing to double-drive the batch')
        return 4
    fill_cmd = [PY, args.driver, '--schedule', args.schedule,
                '--evidence-root', args.evidence_root,
                '--snapshot-root', args.snapshot_root,
                '--index', args.index]
    log('launching fill driver: ' + ' '.join(fill_cmd))
    with open(args.log, 'a', encoding='utf-8') as out:
        proc = subprocess.Popen(fill_cmd, stdout=out,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                                cwd=os.path.dirname(HERE))
    record = {'launched_at_utc': utc(), 'pid': proc.pid,
              'cmd': fill_cmd, 'schedule': args.schedule,
              'waited_pid': args.wait_pid, 'assume_mode': assume_mode,
              'launcher_log': args.log}
    first = os.path.join(args.evidence_root, 'fill-launch.json')
    if not os.path.isfile(first):
        with open(first, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=1)
    else:
        with open(os.path.join(args.evidence_root, 'fill-relaunches.jsonl'),
                  'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    log(f'fill driver launched: pid {proc.pid}; launcher exiting')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_fill_launcher',
        description='Wait for the stage-1 upstream batch driver, preserve '
                    'its artifacts, then launch the cumulative fill '
                    'driver detached (no GPU use by the launcher itself).')
    p.add_argument('--wait-pid', type=int, default=None,
                   help='PID of the running stage-1 batch driver '
                        '(required without --assume-stage1-done)')
    p.add_argument('--assume-stage1-done', action='store_true',
                   help='re-arm mode: the stage-1 driver has already '
                        'exited; skip the wait and go straight to '
                        'marker + preserve + guarded launch')
    p.add_argument('--schedule', required=True)
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--snapshot-root', required=True)
    p.add_argument('--index', required=True)
    p.add_argument('--driver', default=os.path.join(
        HERE, 'run_upstream_batch.py'))
    p.add_argument('--log', required=True)
    p.add_argument('--poll-s', type=int, default=30)
    args = p.parse_args(argv)
    if not args.assume_stage1_done and args.wait_pid is None:
        p.error('--wait-pid is required without --assume-stage1-done')
    log = Log(args.log)

    if args.assume_stage1_done:
        log('re-arm mode (--assume-stage1-done): the stage-1 driver has '
            'already exited; proceeding to marker + preserve + guarded '
            'launch (the caller must have verified no live driver)')
    else:
        cmd = pid_cmdline(args.wait_pid)
        if cmd is None or not any(DRIVER_NEEDLE in c for c in cmd):
            log(f'ABORT: wait-pid {args.wait_pid} is not a live upstream '
                'batch driver (already exited? re-arm afterwards with '
                '--assume-stage1-done)')
            return 2
        log(f'launcher up; waiting for stage-1 driver pid {args.wait_pid} '
            f'(cmd: {" ".join(c for c in cmd if c)[:160]})')
        beats = 0
        while pid_is_driver(args.wait_pid):
            time.sleep(args.poll_s)
            beats += 1
            if beats % 20 == 0:
                log(f'still waiting for pid {args.wait_pid}')
        log(f'stage-1 driver pid {args.wait_pid} exited')

    # preserve the stage-1 artifacts BEFORE the boundary marker so the
    # preserved driver-stage1.log is purely the stage-1 driver's own
    # output; the marker then separates stage-1 from fill lines in the
    # live shared driver.log
    preserve_stage1(args.evidence_root, log)
    if ensure_marker(args.evidence_root):
        log('timestamp-convention boundary marker written')
    else:
        log('timestamp-convention boundary marker already present')
    return launch_fill(args, log, args.assume_stage1_done)


if __name__ == '__main__':
    sys.exit(main())

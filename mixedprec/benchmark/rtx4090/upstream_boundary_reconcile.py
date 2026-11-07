#!/usr/bin/env python3
'''Post-boundary reconciliation for the chained upstream-comparison batch.

Automates the LC-205d five-step checklist for the session that finds the
stage-1 batch complete / the cumulative fill running.  CPU-only and
report-only: every chain file is READ, never rewritten, renamed or
deleted (the launcher overwrites characterized in LC-205d are damage to
RECORD, not repair); the only writes are this tool's own report pair
under <evidence-root>.

  C1 process census   live drivers/launchers/watchdogs; exactly one fill
                      driver for this evidence root expected once the
                      boundary has passed; batch progress cross-checked
                      against the index and watchdog-state.json
  C2 launcher story   fill-launcher.log read IN FULL; the old in-memory
                      launcher (4187632 code, cmdline-fragile waiter)
                      crash evidence (Branch A: traceback captured in
                      fill-launcher.stdout.log, interrupted/partial
                      preserve asides) detected and PRESERVED as the
                      honest record
  C3 launch records   fill-launch.json + fill-relaunches.jsonl vs the
                      true launch sequence in the structured log; the
                      old code overwrites fill-launch.json
                      unconditionally, so a mismatch is reconstructed
                      from the log (the first 'fill driver launched:
                      pid N' line) and recorded in adjudication notes
  C4 boundary marker  '=== FILL CHAIN:' count in driver.log: zero ->
                      PRE_BOUNDARY; one -> clean; duplicates ->
                      first marker is the stage-1/fill boundary, lines
                      between two markers are true-UTC fill lines (the
                      second marker's "above = local-CST" caption is
                      wrong for exactly those lines, cosmetic only)
  C5 stage1 preserve  stage1-preserved/ inventory re-hashed in full and
                      cross-checked against the live originals (the
                      driver never deletes or rewrites stage-1
                      sample/cell files, so preserved == live by
                      content); the old code's damage modes detected
                      (self-referential inventory entry, refreshed
                      driver-stage1.log containing the marker)

States: PRE_BOUNDARY (no marker yet: stage-1 still running or the chain
not fired) | RECONCILED | RECONCILED_WITH_FLAGS (recoverable damage,
reconstructed records carried in adjudication notes) |
NEEDS_ATTENTION (broken chain: nothing drives an incomplete batch, a
double-drive is live, or the preserve is missing/corrupt).

Exit codes: 0 = reconciled / pre-boundary (flags allowed); 3 = needs
attention; 4 = usage / unreadable inputs.  No GPU use ever.
'''

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

INDEX_DEFAULT = ('/data/work/gpu4pyscf-agent/state/long-chain/'
                 'upstream-effects-index.json')
DRIVER_NEEDLE = 'run_upstream_batch.py'
LAUNCHER_NEEDLE = 'upstream_fill_launcher.py'
WATCHDOG_NEEDLE = 'upstream_chain_watchdog.py'
MARKER_TAG = '=== FILL CHAIN:'
ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')

LOG_LINE = re.compile(r'^\[([\dT:-]+Z)\] (.*)$')
LAUNCH_PID = re.compile(r'fill driver launched: pid (\d+);')


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


# ---------------------------------------------------------------------------
# /proc census (element-based matching: a wrapper whose -c string merely
# mentions a script is not that script; mirrors the watchdog's scans)
# ---------------------------------------------------------------------------

def _cmdline_of(pid, proc='/proc'):
    try:
        with open(os.path.join(proc, str(pid), 'cmdline'), 'rb') as f:
            return f.read().decode('utf-8', 'replace').split('\0')
    except OSError:
        return None


def _runs_script(elems, script):
    return any(e == script or e.endswith(f'/{script}') for e in elems)


def scan(script, exclude=(), proc='/proc', self_pid=None,
         must_contain=()):
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


# ---------------------------------------------------------------------------
# structured-log parsing
# ---------------------------------------------------------------------------

def read_log_entries(path):
    '''[(stamp, message)] of the launcher's structured lines; the fill
    driver's stdout is redirected into the same file, so unrelated lines
    are skipped rather than rejected.'''
    entries = []
    try:
        with open(path, encoding='utf-8') as f:
            text = f.read()
    except OSError:
        return None
    for line in text.splitlines():
        m = LOG_LINE.match(line)
        if m:
            entries.append((m.group(1), m.group(2)))
    return entries


def launch_sequence(log_entries):
    '''True ordered launch sequence [(stamp, pid)] from the launcher's
    own "fill driver launched: pid N; launcher exiting" lines.'''
    return [(stamp, int(m.group(1)))
            for stamp, msg in log_entries
            for m in [LAUNCH_PID.search(msg)] if m]


# ---------------------------------------------------------------------------
# the five checklist checks; each returns a dict with status
# PASS | FLAG | FAIL | INFO (plus evidence for the report)
# ---------------------------------------------------------------------------

def _check(id_, name, status, detail, **extra):
    out = {'id': id_, 'name': name, 'status': status, 'detail': detail}
    out.update(extra)
    return out


def check_processes(root, index_path, batch_id, proc='/proc',
                    tail_lines=5):
    '''C1: live census + progress; driver.log tail for human context.'''
    drivers = scan(DRIVER_NEEDLE,
                   exclude=(LAUNCHER_NEEDLE, WATCHDOG_NEEDLE), proc=proc)
    ours = [(p, c) for p, c in drivers if root in c]
    launchers = scan(LAUNCHER_NEEDLE, exclude=(WATCHDOG_NEEDLE,),
                     proc=proc, must_contain=(root,))
    watchdogs = scan(WATCHDOG_NEEDLE, proc=proc, must_contain=(root,))
    state = batch_state(index_path, batch_id)
    wd_state = None
    wd_path = os.path.join(root, 'watchdog-state.json')
    if os.path.isfile(wd_path):
        try:
            with open(wd_path, encoding='utf-8') as f:
                wd_state = json.load(f)
        except (OSError, ValueError):
            wd_state = 'unreadable'
    total, done = state if state else (None, None)
    complete = bool(state and total and total > 0 and done == total)

    evidence = {
        'drivers_alive': [{'pid': p, 'ours': bool(root in c),
                            'argv': c[:200]} for p, c in drivers],
        'launchers_alive': len(launchers), 'watchdogs_alive': len(watchdogs),
        'index_cells_total': total, 'index_cells_done': done,
        'watchdog_state': wd_state}
    if wd_state and isinstance(wd_state, dict):
        evidence['watchdog_last_decision'] = wd_state.get('last_decision')
        evidence['watchdog_cells'] = [wd_state.get('cells_done'),
                                      wd_state.get('cells_total')]

    # driver.log tail (the shared log spans the boundary; stamp
    # conventions differ above/below the marker, C4 classifies that)
    try:
        with open(os.path.join(root, 'driver.log'), encoding='utf-8') as f:
            tail = f.read().splitlines()[-tail_lines:]
    except OSError:
        tail = None
    evidence['driver_log_tail'] = tail

    if len(ours) > 1:
        return _check('C1', 'process census', 'FAIL',
                      f'{len(ours)} live drivers for THIS evidence root '
                      f'(pids {[p for p, _ in ours]}): double-drive in '
                      'progress -- the drivers\' own startup refusal '
                      'should collapse this within seconds; re-run and '
                      'investigate if it persists', evidence=evidence)
    if not ours and not complete:
        return _check('C1', 'process census', 'FAIL',
                      'no live batch driver and the batch is incomplete '
                      f'({done}/{total}): nothing drives the batch -- '
                      'follow the LC-205b runbook / the watchdog will '
                      're-arm if it is alive', evidence=evidence)
    status = 'PASS' if ours else 'INFO'
    detail = (f'one live batch driver (pid {ours[0][0]}) drives the '
              f'batch; {done}/{total} cells complete' if ours else
              f'no live driver (batch complete: {done}/{total} cells)')
    return _check('C1', 'process census', status, detail, evidence=evidence)


def check_launcher_story(root):
    '''C2: the launcher's structured log read in full; Branch A (old
    waiter crashed) and interrupted preserves detected and PRESERVED.'''
    entries = read_log_entries(os.path.join(root, 'fill-launcher.log'))
    if entries is None:
        return _check('C2', 'launcher story', 'FLAG',
                      'fill-launcher.log is missing/unreadable -- no '
                      'launcher record at all (was a launcher ever '
                      'started?)',
                      evidence={'structured_lines': 0,
                                'launches': [], 'old_launcher_crash': None,
                                'preserve_asides': []})
    waited = [m for _, m in entries
              if m.startswith('launcher up; waiting for stage-1 driver')]
    rearms = [m for _, m in entries if m.startswith('re-arm mode')]
    driver_exit = [m for _, m in entries
                   if 'exited' in m and 'stage-1 driver pid' in m]
    launches = launch_sequence(entries)
    aborts = [m for _, m in entries if 'ABORT' in m]
    crash = None
    stdout_path = os.path.join(root, 'fill-launcher.stdout.log')
    if os.path.isfile(stdout_path):
        try:
            with open(stdout_path, encoding='utf-8') as f:
                if 'Traceback' in f.read():
                    crash = 'traceback present in fill-launcher.stdout.log'
        except OSError:
            crash = 'fill-launcher.stdout.log unreadable'
    asides = sorted(
        os.path.basename(p) for p in
        glob.glob(os.path.join(root, 'stage1-preserved.interrupted.*')) +
        glob.glob(os.path.join(root, 'stage1-preserved.partial.*')))

    evidence = {'structured_lines': len(entries), 'waiters': len(waited),
                'rearm_mode_entries': len(rearms),
                'driver_exit_lines': driver_exit,
                'launches': [{'stamp': s, 'pid': p} for s, p in launches],
                'aborts': aborts, 'old_launcher_crash': crash,
                'preserve_asides': asides}
    flags = []
    if crash:
        flags.append('old-launcher crash evidence: ' + crash)
    if asides:
        flags.append(
            'interrupted/partial preserve asides kept as honest record: '
            + ', '.join(asides))
    status = 'FLAG' if flags else 'PASS'
    detail = (f'launcher log tells a coherent story: {len(waited)} '
              f'waiter(s), {len(rearms)} re-arm mode entrie(s), '
              f'{len(launches)} launch(es), {len(aborts)} abort(s)'
              + ('; ' + '; '.join(flags) if flags else ''))
    return _check('C2', 'launcher story', status, detail, evidence=evidence)


def check_launch_records(root, log_entries):
    '''C3: fill-launch.json + fill-relaunches.jsonl vs the log's true
    launch sequence; the old code overwrites the first, so the log is
    the authority and mismatches are reconstructed, never repaired.'''
    launches = launch_sequence(log_entries)
    if not launches:
        return _check('C3', 'launch records', 'INFO',
                      'no fill launch in the structured log yet',
                      evidence={'log_launches': []})
    first_json = None
    first_path = os.path.join(root, 'fill-launch.json')
    if os.path.isfile(first_path):
        try:
            with open(first_path, encoding='utf-8') as f:
                first_json = json.load(f)
        except (OSError, ValueError):
            first_json = 'unreadable'
    relaunches = []
    rel_path = os.path.join(root, 'fill-relaunches.jsonl')
    if os.path.isfile(rel_path):
        try:
            with open(rel_path, encoding='utf-8') as f:
                relaunches = [json.loads(x) for x in f if x.strip()]
        except (OSError, ValueError):
            relaunches = 'unreadable'

    true_first = {'launched_at_utc': launches[0][0],
                  'pid': launches[0][1]}
    evidence = {'true_launch_sequence':
                [{'stamp': s, 'pid': p} for s, p in launches],
                'fill_launch_json': first_json,
                'fill_relaunches_jsonl': relaunches}
    flags = []
    if first_json == 'unreadable' or relaunches == 'unreadable':
        flags.append('a launch-record file is present but unreadable')
    ok_first = (isinstance(first_json, dict)
                and first_json.get('pid') == true_first['pid'])
    if not ok_first:
        flags.append(
            "fill-launch.json does NOT hold the true first launch "
            "(missing or overwritten by the old launcher's unconditional "
            "write); the true first-launch record RECONSTRUCTED FROM THE "
            "STRUCTURED LOG for the adjudication notes: "
            + json.dumps(true_first))
    if isinstance(relaunches, list):
        logged_after_first = len(launches) - 1
        if len(relaunches) != logged_after_first:
            flags.append(
                f'fill-relaunches.jsonl holds {len(relaunches)} record(s) '
                f'but the log shows {logged_after_first} relaunch(es) -- '
                'the jsonl is the new-code source record; a deficit means '
                'pre-7081be8 launches (old code overwrote fill-launch.json '
                'instead of appending)')
        for i, rec in enumerate(relaunches):
            if (not isinstance(rec, dict)
                    or i + 1 >= len(launches)
                    or rec.get('pid') != launches[i + 1][1]):
                flags.append(
                    f'relaunch record #{i + 1} does not match the log '
                    'sequence (pid mismatch)')
                break
    status = 'FLAG' if flags else 'PASS'
    detail = (f'{len(launches)} fill launch(es) in the structured log; '
              'fill-launch.json holds the true first' if not flags else
              '; '.join(flags))
    return _check('C3', 'launch records', status, detail, evidence=evidence,
                  true_first_launch=true_first)


def check_marker(root):
    '''C4: boundary-marker census in the shared driver.log.'''
    path = os.path.join(root, 'driver.log')
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        return _check('C4', 'boundary marker', 'FAIL',
                      'driver.log is missing/unreadable',
                      evidence={'marker_count': 0,
                                'driver_log_missing': True})
    hits = [(i + 1, ln) for i, ln in enumerate(lines)
            if MARKER_TAG in ln]
    if not hits:
        return _check('C4', 'boundary marker', 'INFO',
                      'no boundary marker in driver.log yet: the stage-1 '
                      'driver is still running (or the chain never fired) '
                      '-- pre-boundary', evidence={'marker_count': 0})
    if len(hits) == 1:
        return _check('C4', 'boundary marker', 'PASS',
                      'exactly one boundary marker (line '
                      f'{hits[0][0]}): stage-1 lines above use local-CST '
                      'stamps labeled Z (true UTC = stamp minus 8h); '
                      'fill lines below use true UTC',
                      evidence={'marker_count': 1,
                                'marker_lines': [hits[0][1][:200]]})
    # duplicates: the old launcher appended its marker unconditionally
    flags = ('duplicate boundary markers (lines '
             + ', '.join(str(h[0]) for h in hits)
             + ') -- old-launcher overwrite damage: the FIRST marker is '
             'the stage-1/fill boundary; lines BETWEEN the two markers '
             'are true-UTC fill lines (the second marker\'s "above = '
             'local-CST" caption is wrong for exactly those lines, '
             'cosmetic only)')
    return _check('C4', 'boundary marker', 'FLAG', flags,
                  evidence={'marker_count': len(hits),
                            'marker_lines': [h[1][:200] for h in hits],
                            'rule': 'first marker = boundary; lines '
                                    'between markers are true-UTC fill '
                                    'lines'})


def _digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def check_preserve(root, marker_count):
    '''C5: stage1-preserved/ inventory re-hashed in full and
    cross-checked against the live originals (the driver never deletes
    or rewrites stage-1 sample/cell files).'''
    preserved = os.path.join(root, 'stage1-preserved')
    inventory = os.path.join(preserved, 'stage1-preserved-inventory.txt')
    if not os.path.isdir(preserved):
        if marker_count:
            return _check('C5', 'stage1 preserve', 'FAIL',
                          'no stage1-preserved/ although the boundary '
                          'has passed: the preserve never ran or was '
                          'deleted -- the fill chain contract is broken',
                          evidence={'preserved': False})
        return _check('C5', 'stage1 preserve', 'INFO',
                      'no stage1-preserved/ yet: expected pre-boundary '
                      '(the preserve is created by the launcher when the '
                      'stage-1 driver exits)',
                      evidence={'preserved': False})
    if not os.path.isfile(inventory):
        return _check('C5', 'stage1 preserve', 'FAIL',
                      'stage1-preserved/ exists WITHOUT an inventory: the '
                      'preserve was interrupted; honest partial evidence '
                      '-- do not delete, re-run the preserve (the launcher '
                      'does this idempotently on re-arm)',
                      evidence={'preserved': True, 'inventory': False})
    entries = []
    with open(inventory, encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and len(parts[0]) == 64:
                entries.append((parts[0], parts[1]))
    if not entries:
        return _check('C5', 'stage1 preserve', 'FAIL',
                      'preserve inventory is empty or unparseable',
                      evidence={'preserved': True, 'inventory': True,
                                'entries': 0})

    flags = []
    missing, mismatched, live_missing, live_diff = [], [], [], []
    self_entry = None
    for digest, rel in entries:
        p = os.path.join(preserved, rel)
        if rel == 'stage1-preserved-inventory.txt':
            self_entry = digest
            continue
        if not os.path.isfile(p):
            missing.append(rel)
            continue
        if _digest(p) != digest:
            mismatched.append(rel)
        if rel.startswith(('samples/', 'cells/')):
            live = os.path.join(root, rel)
            if not os.path.isfile(live):
                live_missing.append(rel)
            elif _digest(live) != digest:
                live_diff.append(rel)
    if self_entry is not None:
        flags.append(
            'inventory carries a self-referential entry for '
            'stage1-preserved-inventory.txt: written by the OLD launcher '
            'code over a previous inventory (its walk hashed the '
            'pre-existing file) -- the inventory is the later old-code '
            'write; the underlying preserved payload is unaffected')
    ds1 = os.path.join(preserved, 'driver-stage1.log')
    marker_in_ds1 = False
    fill_lines_after = 0
    if os.path.isfile(ds1):
        try:
            with open(ds1, encoding='utf-8') as f:
                ds1_lines = f.read().splitlines()
            for i, ln in enumerate(ds1_lines):
                if MARKER_TAG in ln:
                    marker_in_ds1 = True
                    fill_lines_after = len(ds1_lines) - i - 1
                    break
        except OSError:
            pass
    if marker_in_ds1:
        flags.append(
            'preserved driver-stage1.log CONTAINS the boundary marker '
            f'(+{fill_lines_after} line(s) after it): refreshed by the old '
            "launcher's unconditional copy2 with a later driver.log -- "
            'stage-1 content is everything ABOVE the marker; the lines '
            'from the marker on are fill lines (true UTC)')

    for label, lst in (('missing from stage1-preserved/', missing),
                       ('hash mismatch vs inventory', mismatched),
                       ('missing from the live originals', live_missing),
                       ('differs from the live original', live_diff)):
        if lst:
            flags.append(f'{len(lst)} file(s) {label}: '
                         + ', '.join(lst[:5])
                         + (' ...' if len(lst) > 5 else ''))
    if missing or mismatched or live_missing or live_diff:
        status = 'FAIL'
    elif flags:
        status = 'FLAG'
    else:
        status = 'PASS'
    detail = (f'{len(entries)} inventoried file(s); full re-hash '
              + ('clean; preserved == live originals'
                 if not (missing or mismatched or live_missing or live_diff)
                 else 'NOT clean (see flags)'))
    if flags:
        detail += '; ' + ' | '.join(flags)
    return _check('C5', 'stage1 preserve', status, detail,
                  evidence={'preserved': True, 'inventory': True,
                            'entries': len(entries),
                            'self_referential_inventory_entry':
                                self_entry is not None,
                            'marker_in_preserved_driver_log':
                                marker_in_ds1},
                  flags=flags)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def classify(checks, marker_count, batch_complete):
    '''(state, exit_code) from the five checks.'''
    has_fail = any(c['status'] == 'FAIL' for c in checks)
    has_flag = any(c['status'] == 'FLAG' for c in checks)
    if marker_count == 0:
        if batch_complete:
            # the fill cannot run without the boundary having passed;
            # a complete batch with no marker means the shared driver.log
            # was damaged (marker deleted) -- attention, not pre-boundary
            return 'NEEDS_ATTENTION', 3
        # no boundary yet: FAILs still mean attention (e.g. nothing
        # drives the incomplete batch even pre-boundary)
        return ('NEEDS_ATTENTION', 3) if has_fail else ('PRE_BOUNDARY', 0)
    if has_fail:
        return 'NEEDS_ATTENTION', 3
    return ('RECONCILED_WITH_FLAGS', 0) if has_flag else ('RECONCILED', 0)


def render_md(report):
    lines = ['# Upstream batch boundary reconciliation',
             '',
             f"- run at (true UTC): {report['run_at_utc']}",
             f"- evidence root: `{report['evidence_root']}`",
             f"- batch: {report['batch_id']}",
             f"- state: **{report['state']}**", '']
    for c in report['checks']:
        lines.append(f"## {c['id']} {c['name']} -- {c['status']}")
        lines.append('')
        lines.append(c['detail'])
        lines.append('')
    if report['adjudication_notes']:
        lines.append('## Adjudication notes (carry into stage-1 '
                     'adjudication)')
        lines.append('')
        lines.append('```json')
        lines.append(json.dumps(report['adjudication_notes'], indent=1))
        lines.append('```')
        lines.append('')
    lines.append('This tool is read-only on every chain file; nothing was '
                 'rewritten, renamed, or deleted.')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_boundary_reconcile',
        description='Read-only post-boundary reconciliation of the chained '
                    'upstream-comparison batch (LC-205d checklist, '
                    'automated). CPU-only; never modifies chain files.')
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--index', default=INDEX_DEFAULT)
    p.add_argument('--batch-id', default='uc-20261018-a')
    p.add_argument('--proc', default='/proc')
    p.add_argument('--tail-lines', type=int, default=5)
    p.add_argument('--out-json', default=None)
    p.add_argument('--out-md', default=None)
    args = p.parse_args(argv)
    root = os.path.abspath(args.evidence_root)
    if not os.path.isdir(root):
        print(f'evidence root not found: {root}', file=sys.stderr)
        return 4

    log_entries = read_log_entries(os.path.join(root, 'fill-launcher.log'))
    if log_entries is None:
        log_entries = []
    # C4 first: the marker count tells C5 whether a missing preserve is
    # the expected pre-boundary state or a broken post-boundary chain
    marker = check_marker(root)
    checks = [
        check_processes(root, args.index, args.batch_id,
                        proc=args.proc, tail_lines=args.tail_lines),
        check_launcher_story(root),
        check_launch_records(root, log_entries),
        marker,
        check_preserve(root, marker['evidence'].get('marker_count') or 0)]
    marker_count = (checks[3]['evidence'].get('marker_count') or 0)
    c1_ev = checks[0]['evidence']
    batch_complete = bool(
        c1_ev.get('index_cells_total')
        and c1_ev.get('index_cells_done') == c1_ev['index_cells_total'])
    state, code = classify(checks, marker_count, batch_complete)

    adjudication_notes = {}
    for c in checks:
        if c.get('true_first_launch') is not None:
            adjudication_notes['true_first_launch'] = c['true_first_launch']
        if c['id'] == 'C2':
            adjudication_notes['launcher_launches'] = c['evidence'][
                'launches']
            adjudication_notes['old_launcher_crash'] = c['evidence'][
                'old_launcher_crash']
            adjudication_notes['preserve_asides'] = c['evidence'][
                'preserve_asides']
        if c['id'] == 'C4' and c['evidence'].get('marker_count', 0) > 1:
            adjudication_notes['marker_rule'] = c['evidence']['rule']
        if c['id'] == 'C5' and c['status'] in ('FLAG', 'FAIL'):
            adjudication_notes['preserve_flags'] = c.get('flags', [])
    report = {'schema': 'upstream-boundary-reconciliation-v1',
              'run_at_utc': utc(), 'evidence_root': root,
              'batch_id': args.batch_id, 'state': state,
              'checks': checks, 'adjudication_notes': adjudication_notes}

    stamp = utc().replace('-', '').replace(':', '')
    out_json = args.out_json or os.path.join(
        root, f'boundary-reconciliation-{stamp}.json')
    out_md = args.out_md or os.path.join(
        root, f'boundary-reconciliation-{stamp}.md')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(render_md(report))

    print(f'state: {state}')
    for c in checks:
        print(f"  {c['id']} {c['name']}: {c['status']} -- {c['detail']}")
    print(f'report: {out_json}')
    print('       ' + out_md)
    return code


if __name__ == '__main__':
    sys.exit(main())

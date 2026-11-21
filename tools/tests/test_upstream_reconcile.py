'''Unit tests for upstream_boundary_reconcile: the automated LC-205d
post-boundary checklist.  Pure CPU; every /proc scan is pointed at a
fake proc root, every evidence tree is a temp dir, so nothing here
depends on (or can see) the real running batch.  The damage scenarios
exercised are exactly the old-launcher (4187632 in-memory code)
overwrites characterized in the LC-205d stage-boundary race audit:
duplicate marker, overwritten fill-launch.json, self-referential
inventory, refreshed driver-stage1.log, interrupted preserves, and the
fail-closed conditions (nothing driving an incomplete batch, corrupted
preserve, double-drive).'''

import hashlib
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'

spec = importlib.util.spec_from_file_location(
    'upstream_boundary_reconcile', _DIR / 'upstream_boundary_reconcile.py')
rec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rec)

BATCH = 'test-batch'
LAUNCH_PID = 555


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _fake_proc(tmp, entries):
    root = pathlib.Path(tmp) / 'proc'
    root.mkdir(parents=True, exist_ok=True)
    for pid, cmd in entries.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / 'cmdline').write_bytes(cmd)
    return root


class EvidenceBuilder:
    '''Synthetic post-boundary evidence root (clean by default).'''

    def __init__(self, tmp):
        self.tmp = pathlib.Path(tmp)
        self.ev = self.tmp / 'ev'
        self.ev.mkdir()
        (self.ev / 'samples').mkdir()
        (self.ev / 'cells').mkdir()
        self.index_path = self.tmp / 'index.json'
        self.marker_count = 1
        self.launch_pid = LAUNCH_PID
        self.launches = [(  # (stamp, pid)
            '2026-10-18T17:00:01Z', LAUNCH_PID)]
        self.write_fill_launch_json = True
        self.relaunch_records = None
        self.launcher_log_extra = []
        self.ds1_with_marker = False
        self.inventory_extra = []
        self.corrupt_preserved = None
        self.diverge_live = None
        self.cells = [{'sources': {
            'upstream': {'path': 'x', 'sha256': '0' * 64},
            'optimized_fp64': {'path': 'x', 'sha256': '0' * 64},
            'optimized_auto': {'path': 'x', 'sha256': '0' * 64},
            'strict_fp64': {'path': 'x', 'sha256': '0' * 64}}}, {}]
        self.marker_fill_lines = 1

    def build(self):
        ev = self.ev
        # shared driver.log: stage-1 (CST-labeled) lines, then marker(s),
        # then true-UTC fill lines
        stage1 = ['[2026-10-18T23:39:57Z] snapshot: verified',
                  '[2026-10-19T00:00:00Z] cell complete']
        lines = list(stage1)
        for _ in range(self.marker_count):
            lines.append('[2026-10-19T01:00:00Z] === FILL CHAIN: the '
                         'stage-1 driver exited; the fill driver launches '
                         'next. All lines ABOVE this marker use local-CST '
                         'stamps labeled Z (true UTC = stamp minus 8h); '
                         'lines BELOW use true UTC ===')
        for i in range(self.marker_fill_lines):
            lines.append(f'[2026-10-19T01:0{i}:30Z] fill arm rc=0')
        (ev / 'driver.log').write_text('\n'.join(lines) + '\n')

        # preserved copy (new-code order: preserved BEFORE the marker, so
        # driver-stage1.log is stage-1-only unless the old code refreshed
        # it afterward)
        pres = ev / 'stage1-preserved'
        (pres / 'samples').mkdir(parents=True)
        (pres / 'cells').mkdir()
        (pres / 'samples' / 's1.json').write_text('{"a": 1}')
        (ev / 'samples' / 's1.json').write_text('{"a": 1}')
        (pres / 'cells' / 'c1.json').write_text('{"b": 2}')
        (ev / 'cells' / 'c1.json').write_text('{"b": 2}')
        ds1 = list(stage1)
        if self.ds1_with_marker:
            ds1.append(lines[len(stage1)])  # the marker line
            ds1.append('[2026-10-19T01:01:30Z] fill arm rc=0')
        (pres / 'driver-stage1.log').write_text('\n'.join(ds1) + '\n')
        if self.corrupt_preserved:
            (pres / 'samples' / 's1.json').write_text('{"tampered": 1}')
        if self.diverge_live:
            (ev / 'samples' / 's1.json').write_text('{"diverged": 1}')
        entries = [f'{_sha(pres / "cells" / "c1.json")}  cells/c1.json',
                   f'{_sha(pres / "driver-stage1.log")}  '
                   'driver-stage1.log',
                   f'{_sha(pres / "samples" / "s1.json")}  samples/s1.json']
        entries.extend(self.inventory_extra)
        (pres / 'stage1-preserved-inventory.txt').write_text(
            '\n'.join(entries) + '\n')

        # launcher structured log (the fill driver's stdout lands here
        # too; the parser must skip those unstamped lines)
        log = ['[2026-10-18T16:14:40Z] launcher up; waiting for stage-1 '
               'driver pid 111 (cmd: python run_upstream_batch.py ...)',
               'unstashed fill-driver stdout line without stamp',
               '[2026-10-18T17:00:00Z] stage-1 driver pid 111 exited',
               '[2026-10-18T17:00:01Z] launching fill driver: python '
               'run_upstream_batch.py --schedule cumulative.json']
        for i, (stamp, pid) in enumerate(self.launches):
            if i:
                log.append(f'[2026-10-18T1{i}:00:0{i}Z] re-arm mode '
                           '(--assume-stage1-done): the stage-1 driver '
                           'has already exited')
                log.append(f'[2026-10-18T1{i}:00:0{i}Z] launching fill '
                           'driver: python run_upstream_batch.py')
            log.append(f'[{stamp}] fill driver launched: pid {pid}; '
                       'launcher exiting')
        log.extend(self.launcher_log_extra)
        (ev / 'fill-launcher.log').write_text('\n'.join(log) + '\n')

        if self.write_fill_launch_json:
            (ev / 'fill-launch.json').write_text(json.dumps({
                'launched_at_utc': self.launches[0][0],
                'pid': self.launch_pid, 'cmd': ['python', '...'],
                'assume_mode': False}, indent=1))
        if self.relaunch_records is not None:
            (ev / 'fill-relaunches.jsonl').write_text(
                '\n'.join(json.dumps(r) for r in self.relaunch_records)
                + '\n')

        (ev / 'watchdog-state.json').write_text(json.dumps({
            'batch_id': BATCH,
            'cells_total': len(self.cells), 'cells_done':
                sum(1 for c in self.cells if c.get('sources')),
            'drivers_alive': 1, 'launchers_alive': 0,
            'fill_launched': True, 'launcher_wedged': False,
            'rearm_count': 0, 'last_rearm_utc': None,
            'last_decision': 'none_healthy',
            'updated_at_utc': '2026-10-18T17:05:00Z',
            'watchdog_pid': 9}, indent=1))
        self.index_path.write_text(json.dumps({'batches': [
            {'id': BATCH, 'cells': self.cells}]}, indent=1))
        return self

    def build_index_only(self):
        self.index_path.write_text(json.dumps({'batches': [
            {'id': BATCH, 'cells': self.cells}]}, indent=1))
        return self

    def run_tool(self, proc, out):
        return rec.main(['--evidence-root', str(self.ev),
                        '--index', str(self.index_path),
                        '--batch-id', BATCH, '--proc', str(proc),
                        '--out-json', str(out / 'r.json'),
                        '--out-md', str(out / 'r.md')])

    def report(self, out):
        return json.loads((out / 'r.json').read_text())


class PreBoundaryTests(unittest.TestCase):

    def test_pre_boundary_clean(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.marker_count = 0
            b.build()
            # no preserve yet, no launches yet
            import shutil
            shutil.rmtree(b.ev / 'stage1-preserved')
            log = (b.ev / 'fill-launcher.log').read_text().splitlines()
            (b.ev / 'fill-launcher.log').write_text(
                '\n'.join(log[:1]) + '\n')
            (b.ev / 'fill-launch.json').unlink()
            b.cells = [{}, {}]
            b.build_index_only()
            proc = _fake_proc(tmp, {
                111: (f'python\0/x/run_upstream_batch.py\0'
                      f'--evidence-root\0{b.ev}\0--schedule\0s.json'
                      ).encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'PRE_BOUNDARY')
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C4']['status'], 'INFO')
            self.assertEqual(by_id['C1']['status'], 'PASS')

    def test_no_driver_incomplete_batch_needs_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp).build()
            proc = _fake_proc(tmp, {})  # nothing alive
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 3)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'NEEDS_ATTENTION')
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C1']['status'], 'FAIL')

    def test_complete_batch_without_marker_is_damage(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.marker_count = 0
            b.cells = [{'sources': {'upstream': 1, 'optimized_fp64': 1,
                                    'optimized_auto': 1, 'strict_fp64': 1}}]
            b.build()
            proc = _fake_proc(tmp, {})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 3)
            self.assertEqual(b.report(out)['state'], 'NEEDS_ATTENTION')


class PostBoundaryCleanTests(unittest.TestCase):

    def test_clean_boundary_reconciles(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp).build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}\0'
                             f'--schedule\0cumulative.json').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'RECONCILED')
            by_id = {c['id']: c for c in rep['checks']}
            for cid in ('C1', 'C2', 'C3', 'C4', 'C5'):
                self.assertEqual(by_id[cid]['status'], 'PASS', cid)
            # the report pair is written and the md carries the state
            self.assertIn('RECONCILED', (out / 'r.md').read_text())

    def test_complete_batch_no_driver_is_reconciled(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.cells = [{'sources': {'upstream': 1, 'optimized_fp64': 1,
                                    'optimized_auto': 1, 'strict_fp64': 1}}]
            b.build()
            proc = _fake_proc(tmp, {})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'RECONCILED')
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C1']['status'], 'INFO')

    def test_double_driver_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp).build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode(),
                556: (f'python\0/y/run_upstream_batch.py\0'
                      f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 3)
            by_id = {c['id']: c for c in b.report(out)['checks']}
            self.assertEqual(by_id['C1']['status'], 'FAIL')


class OldLauncherDamageTests(unittest.TestCase):
    '''The 4187632 in-memory overwrite modes from the LC-205d audit.'''

    def test_duplicate_marker_flagged_with_rule(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.marker_count = 2
            b.marker_fill_lines = 2
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'RECONCILED_WITH_FLAGS')
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C4']['status'], 'FLAG')
            self.assertIn('FIRST marker', by_id['C4']['detail'])
            self.assertIn('marker_rule', rep['adjudication_notes'])

    def test_overwritten_fill_launch_json_reconstructed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.launch_pid = 999  # the file holds the OLD code's record
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C3']['status'], 'FLAG')
            true_first = rep['adjudication_notes']['true_first_launch']
            self.assertEqual(true_first,
                             {'launched_at_utc': '2026-10-18T17:00:01Z',
                              'pid': LAUNCH_PID})

    def test_self_referential_inventory_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.inventory_extra = [
                '0' * 64 + '  stage1-preserved-inventory.txt']
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C5']['status'], 'FLAG')
            self.assertIn('self-referential', by_id['C5']['detail'])
            self.assertTrue(
                by_id['C5']['evidence']['self_referential_inventory_entry'])

    def test_refreshed_driver_stage1_log_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.ds1_with_marker = True
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            by_id = {c['id']: c for c in b.report(out)['checks']}
            self.assertEqual(by_id['C5']['status'], 'FLAG')
            self.assertTrue(
                by_id['C5']['evidence']['marker_in_preserved_driver_log'])

    def test_branch_a_crash_evidence_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            # the old waiter crashed mid-preserve; the watchdog re-armed
            b.launches = [('2026-10-18T17:30:00Z', LAUNCH_PID)]
            b.launcher_log_extra = [
                '[2026-10-18T17:20:00Z] existing stage1-preserved has no '
                'inventory (interrupted preserve?); moving it aside']
            b.build()
            (b.ev / 'fill-launcher.stdout.log').write_text(
                'Traceback (most recent call last):\n'
                '  File "upstream_fill_launcher.py", line 80, in alive\n'
                'OSError: /proc/111 gone\n')
            aside = b.ev / 'stage1-preserved.interrupted.20261018T171930Z'
            aside.mkdir()
            (aside / 'samples').mkdir()
            (aside / 'samples' / 'partial.json').write_text('{}')
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            rep = b.report(out)
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C2']['status'], 'FLAG')
            ev = by_id['C2']['evidence']
            self.assertIn('traceback', ev['old_launcher_crash'])
            self.assertTrue(ev['preserve_asides'])
            # read-only contract: the honest evidence still exists
            self.assertTrue(aside.is_dir())
            self.assertTrue((aside / 'samples' / 'partial.json')
                            .is_file())
            self.assertIn(aside.name,
                          rep['adjudication_notes']['preserve_asides'])

    def test_relaunch_record_deficit_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.launches = [('2026-10-18T17:00:01Z', LAUNCH_PID),
                          ('2026-10-18T18:00:00Z', 777)]
            b.relaunch_records = []  # the jsonl lost/mis-recorded it
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            by_id = {c['id']: c for c in b.report(out)['checks']}
            self.assertEqual(by_id['C3']['status'], 'FLAG')

    def test_relaunch_records_consistent_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.launches = [('2026-10-18T17:00:01Z', LAUNCH_PID),
                          ('2026-10-18T18:00:00Z', 777)]
            b.relaunch_records = [
                {'launched_at_utc': '2026-10-18T18:00:00Z', 'pid': 777,
                 'assume_mode': True}]
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 0)
            by_id = {c['id']: c for c in b.report(out)['checks']}
            self.assertEqual(by_id['C3']['status'], 'PASS')


class PreserveIntegrityTests(unittest.TestCase):

    def test_corrupted_preserve_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.corrupt_preserved = True
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 3)
            rep = b.report(out)
            self.assertEqual(rep['state'], 'NEEDS_ATTENTION')
            by_id = {c['id']: c for c in rep['checks']}
            self.assertEqual(by_id['C5']['status'], 'FAIL')

    def test_live_original_divergence_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            b = EvidenceBuilder(tmp)
            b.diverge_live = True
            b.build()
            proc = _fake_proc(tmp, {
                LAUNCH_PID: (f'python\0/x/run_upstream_batch.py\0'
                             f'--evidence-root\0{b.ev}').encode()})
            out = pathlib.Path(tmp) / 'out'
            out.mkdir()
            rc = b.run_tool(proc, out)
            self.assertEqual(rc, 3)
            by_id = {c['id']: c for c in b.report(out)['checks']}
            self.assertEqual(by_id['C5']['status'], 'FAIL')


class ParserTests(unittest.TestCase):

    def test_launch_sequence_skips_unstamped_lines(self):
        entries = rec.read_log_entries(__file__) or []
        self.assertEqual(entries, [])  # this file is not a structured log
        entries = [
            ('2026-10-18T17:00:01Z',
             'fill driver launched: pid 555; launcher exiting'),
            ('2026-10-18T17:00:02Z', 'unrelated structured line'),
            ('2026-10-18T18:00:00Z',
             'fill driver launched: pid 777; launcher exiting')]
        self.assertEqual(rec.launch_sequence(entries),
                         [('2026-10-18T17:00:01Z', 555),
                          ('2026-10-18T18:00:00Z', 777)])

    def test_missing_launcher_log_parses_empty(self):
        self.assertEqual(rec.read_log_entries('/nonexistent/x.log'), None)


if __name__ == '__main__':
    unittest.main()

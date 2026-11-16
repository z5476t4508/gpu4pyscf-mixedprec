'''Unit tests for the chained-batch liveness automation: the hardened
fill launcher (re-arm mode, cmdline-verified wait, atomic idempotent
preserve, first-wins records, double-drive refusal) and the CPU-only
chain watchdog (pure decision table, /proc scans with launcher/watchdog
exclusions, batch-index completion, single instance via flock, bounded
re-arms).  stdlib only; no GPU.  All /proc scans are either pointed at a
fake proc root or mocked, so nothing here depends on (or can see) the
real running batch.'''

import fcntl
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fl = _load('upstream_fill_launcher')
wd = _load('upstream_chain_watchdog')

FAKE_FILL_DRIVER = '''import os
out = os.environ.get('FAKE_DRIVER_OUT')
if out:
    with open(out, 'w') as f:
        f.write('ran\\n')
'''

FAKE_STAGE1_DRIVER = 'import time\ntime.sleep(3)\n'


class LauncherRearmTests(unittest.TestCase):
    '''--assume-stage1-done mode: marker, preserve, guarded launch, and
    the idempotence / first-wins guarantees for re-arms.'''

    def _evidence(self, tmp):
        ev = pathlib.Path(tmp)
        (ev / 'samples').mkdir()
        (ev / 'samples' / 's1.json').write_text('{"a": 1}')
        (ev / 'cells').mkdir()
        (ev / 'cells' / 'c1.json').write_text('{"b": 2}')
        (ev / 'driver.log').write_text(
            '[2026-10-19T00:00:00Z] stage-1 line\n')
        return ev

    def _run(self, ev, driver, fake_out, extra=()):
        os.environ['FAKE_DRIVER_OUT'] = str(fake_out)
        return fl.main(['--assume-stage1-done', *extra,
                        '--schedule', str(ev / 'sched.json'),
                        '--evidence-root', str(ev),
                        '--snapshot-root', str(ev / 'snapshot'),
                        '--index', str(ev / 'index.json'),
                        '--driver', str(driver),
                        '--log', str(ev / 'fill-launcher.log')])

    def _wait_for(self, path, timeout=15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if os.path.isfile(path):
                return True
            time.sleep(0.2)
        return False

    def test_rearm_preserves_marks_and_launches(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            driver = pathlib.Path(tmp) / 'fake_fill_driver.py'
            driver.write_text(FAKE_FILL_DRIVER)
            with mock.patch.object(fl, 'find_live_driver',
                                   return_value=None):
                rc = self._run(ev, driver, pathlib.Path(tmp) / 'out1')
            self.assertEqual(rc, 0)
            # guarded fill driver really launched (detached)
            self.assertTrue(self._wait_for(pathlib.Path(tmp) / 'out1'))
            # marker written exactly once
            text = (ev / 'driver.log').read_text()
            self.assertEqual(text.count('=== FILL CHAIN:'), 1)
            # preserve: copy + inventory + the stage-1 driver log
            pres = ev / 'stage1-preserved'
            self.assertEqual((pres / 'samples' / 's1.json').read_text(),
                             '{"a": 1}')
            self.assertEqual((pres / 'cells' / 'c1.json').read_text(),
                             '{"b": 2}')
            self.assertEqual((pres / 'driver-stage1.log').read_text(),
                             '[2026-10-19T00:00:00Z] stage-1 line\n')
            inv = (pres / 'stage1-preserved-inventory.txt').read_text()
            self.assertIn('driver-stage1.log', inv)
            # first launch recorded in fill-launch.json
            rec = json.loads((ev / 'fill-launch.json').read_text())
            self.assertTrue(rec['assume_mode'])
            self.assertIsNone(rec['waited_pid'])
            self.assertFalse((ev / 'fill-relaunches.jsonl').exists())

    def test_second_rearm_is_idempotent_first_copy_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            driver = pathlib.Path(tmp) / 'fake_fill_driver.py'
            driver.write_text(FAKE_FILL_DRIVER)
            with mock.patch.object(fl, 'find_live_driver',
                                   return_value=None):
                self.assertEqual(
                    self._run(ev, driver, pathlib.Path(tmp) / 'out1'), 0)
                self.assertTrue(self._wait_for(pathlib.Path(tmp) / 'out1'))
            # the fill driver appends to the shared driver log after the
            # first preserve; a re-arm must keep the FIRST preserved copy
            with open(ev / 'driver.log', 'a', encoding='utf-8') as f:
                f.write('[2026-10-19T00:05:00Z] fill line\n')
            with mock.patch.object(fl, 'find_live_driver',
                                   return_value=None):
                self.assertEqual(
                    self._run(ev, driver, pathlib.Path(tmp) / 'out2'), 0)
                self.assertTrue(self._wait_for(pathlib.Path(tmp) / 'out2'))
            # marker not duplicated
            self.assertEqual(
                (ev / 'driver.log').read_text().count('=== FILL CHAIN:'), 1)
            # preserved stage-1 log is the FIRST copy, not overwritten
            self.assertEqual(
                (ev / 'stage1-preserved' / 'driver-stage1.log').read_text(),
                '[2026-10-19T00:00:00Z] stage-1 line\n')
            # first launch record kept; the re-arm logged to the jsonl
            rec = json.loads((ev / 'fill-launch.json').read_text())
            self.assertTrue(rec['assume_mode'])
            lines = (ev / 'fill-relaunches.jsonl').read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertTrue(json.loads(lines[0])['assume_mode'])

    def test_interrupted_preserve_is_redone(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            driver = pathlib.Path(tmp) / 'fake_fill_driver.py'
            driver.write_text(FAKE_FILL_DRIVER)
            # a preserve interrupted before its inventory: partial dir
            partial = ev / 'stage1-preserved'
            partial.mkdir()
            (partial / 'junk.txt').write_text('partial\n')
            with mock.patch.object(fl, 'find_live_driver',
                                   return_value=None):
                rc = self._run(ev, driver, pathlib.Path(tmp) / 'out1')
            self.assertEqual(rc, 0)
            self.assertTrue(self._wait_for(pathlib.Path(tmp) / 'out1'))
            # the interrupted copy was moved aside (kept as evidence)
            aside = [d for d in ev.iterdir()
                     if d.name.startswith('stage1-preserved.interrupted.')]
            self.assertEqual(len(aside), 1)
            self.assertEqual((aside[0] / 'junk.txt').read_text(), 'partial\n')
            # and a complete preserve with inventory exists
            pres = ev / 'stage1-preserved'
            self.assertTrue((pres / 'stage1-preserved-inventory.txt')
                            .is_file())
            self.assertTrue((pres / 'samples' / 's1.json').is_file())

    def test_rearm_refuses_double_drive(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            driver = pathlib.Path(tmp) / 'fake_fill_driver.py'
            driver.write_text(FAKE_FILL_DRIVER)
            with mock.patch.object(fl, 'find_live_driver',
                                   return_value=424242):
                rc = self._run(ev, driver, pathlib.Path(tmp) / 'out')
            self.assertEqual(rc, 4)
            self.assertFalse((ev / 'fill-launch.json').exists())
            self.assertFalse(self._wait_for(pathlib.Path(tmp) / 'out',
                                           timeout=2.0))

    def test_ensure_marker_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = pathlib.Path(tmp)
            (ev / 'driver.log').write_text('plain\n')
            self.assertTrue(fl.ensure_marker(str(ev)))
            self.assertFalse(fl.ensure_marker(str(ev)))
            self.assertEqual(
                (ev / 'driver.log').read_text().count('=== FILL CHAIN:'), 1)


class LauncherWaitModeTests(unittest.TestCase):
    '''Normal --wait-pid mode: cmdline-verified wait and aborts.'''

    def _evidence(self, tmp):
        ev = pathlib.Path(tmp)
        (ev / 'samples').mkdir()
        (ev / 'cells').mkdir()
        (ev / 'driver.log').write_text('stage-1\n')
        return ev

    def test_wait_mode_follows_driver_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            fake = pathlib.Path(tmp) / 'run_upstream_batch.py'
            fake.write_text(FAKE_STAGE1_DRIVER)
            proc = subprocess.Popen(
                [sys.executable, str(fake), '--fake'])
            try:
                driver = pathlib.Path(tmp) / 'fake_fill_driver.py'
                driver.write_text(FAKE_FILL_DRIVER)
                os.environ['FAKE_DRIVER_OUT'] = str(pathlib.Path(tmp) / 'o')
                with mock.patch.object(fl, 'find_live_driver',
                                       return_value=None):
                    rc = fl.main(['--wait-pid', str(proc.pid),
                                  '--poll-s', '1',
                                  '--schedule', str(ev / 's.json'),
                                  '--evidence-root', str(ev),
                                  '--snapshot-root', str(ev / 'snapshot'),
                                  '--index', str(ev / 'i.json'),
                                  '--driver', str(driver),
                                  '--log', str(ev / 'fill-launcher.log')])
                self.assertEqual(rc, 0)
                log = (ev / 'fill-launcher.log').read_text()
                self.assertIn('launcher up; waiting', log)
                self.assertIn('exited', log)
                deadline = time.time() + 15
                while not os.path.isfile(pathlib.Path(tmp) / 'o'):
                    if time.time() > deadline:
                        self.fail('fill driver never launched')
                    time.sleep(0.2)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=10)

    def test_wait_mode_aborts_on_non_driver_pid(self):
        # PID-reuse protection: a live PID whose cmdline is NOT the
        # batch driver must abort instead of waiting on it forever
        with tempfile.TemporaryDirectory() as tmp:
            ev = self._evidence(tmp)
            rc = fl.main(['--wait-pid', str(os.getpid()),
                         '--schedule', str(ev / 's.json'),
                         '--evidence-root', str(ev),
                         '--snapshot-root', str(ev / 'snapshot'),
                         '--index', str(ev / 'i.json'),
                         '--driver', str(ev / 'fake.py'),
                         '--log', str(ev / 'fill-launcher.log')])
            self.assertEqual(rc, 2)
            self.assertIn('ABORT',
                          (ev / 'fill-launcher.log').read_text())
            # aborted before any marker or preserve
            self.assertEqual((ev / 'driver.log').read_text(), 'stage-1\n')
            self.assertFalse((ev / 'stage1-preserved').exists())


class LauncherScanTests(unittest.TestCase):

    def _fake_proc(self, tmp, entries):
        root = pathlib.Path(tmp) / 'proc'
        for pid, cmd in entries.items():
            d = root / str(pid)
            d.mkdir(parents=True)
            (d / 'cmdline').write_bytes(cmd)
        return root

    def test_find_live_driver_excludes_launcher_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._fake_proc(tmp, {
                7: b'python\0/x/run_upstream_batch.py\0--schedule\0c\0',
                8: (b'python\0/x/upstream_fill_launcher.py\0--driver\0'
                    b'/x/run_upstream_batch.py\0'),
                9: b'python\0unrelated.py\0',
                # a lingering shell wrapper whose -c string merely
                # mentions the driver script must NOT count as a driver
                10: (b'/bin/bash\0-c\0EVID=/e && setsid nohup python '
                     b'/x/run_upstream_batch.py --schedule s\0')})
            self.assertEqual(fl.find_live_driver(str(proc)), 7)

    def test_find_live_driver_none_when_only_launchers(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._fake_proc(tmp, {
                8: (b'python\0/x/upstream_fill_launcher.py\0--driver\0'
                    b'/x/run_upstream_batch.py\0')})
            self.assertIsNone(fl.find_live_driver(str(proc)))


class WatchdogDecideTests(unittest.TestCase):
    '''The pure decision table.'''

    def test_table(self):
        # batch complete always wins
        self.assertEqual(
            wd.decide(True, False, False, False, 0, None, 12, 1800),
            'exit_complete')
        # any live driver (ours or a foreign batch's) -> healthy
        self.assertEqual(
            wd.decide(False, True, False, False, 0, None, 12, 1800),
            'none_healthy')
        # live launcher that is not wedged -> healthy
        self.assertEqual(
            wd.decide(False, False, True, False, 0, None, 12, 1800),
            'none_healthy')
        # wedged launcher (stuck past the bound, no fill launched) ->
        # re-arm; the re-armed launcher guard-aborts fail-closed if a
        # driver is actually live
        self.assertEqual(
            wd.decide(False, False, True, True, 0, None, 12, 1800),
            'rearm')
        # nothing alive, budget available -> re-arm
        self.assertEqual(
            wd.decide(False, False, False, False, 0, None, 12, 1800),
            'rearm')
        # spacing not yet elapsed -> wait
        self.assertEqual(
            wd.decide(False, False, False, False, 0, 100, 12, 1800),
            'none_spacing')
        # budget exhausted with the batch incomplete -> exit nonzero
        self.assertEqual(
            wd.decide(False, False, False, False, 12, None, 12, 1800),
            'exit_budget')
        # first re-arm after spacing elapsed is allowed again
        self.assertEqual(
            wd.decide(False, False, False, False, 3, 1800, 12, 1800),
            'rearm')


class WatchdogScanTests(unittest.TestCase):

    def _fake_proc(self, tmp, entries):
        root = pathlib.Path(tmp) / 'proc'
        for pid, cmd in entries.items():
            d = root / str(pid)
            d.mkdir(parents=True)
            (d / 'cmdline').write_bytes(cmd)
        return root

    def test_driver_scan_excludes_launcher_and_watchdog_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._fake_proc(tmp, {
                10: (b'python\0/x/run_upstream_batch.py\0--schedule\0'
                     b'cumulative\0'),
                20: (b'python\0/x/upstream_fill_launcher.py\0--driver\0'
                     b'/x/run_upstream_batch.py\0'),
                30: (b'python\0/x/upstream_chain_watchdog.py\0'
                     b'--launcher-script\0/x/upstream_fill_launcher.py\0'),
                40: b'python\0unrelated.py\0',
                # lingering -c wrappers that merely mention the scripts
                50: (b'/bin/bash\0-c\0EVID=/evid && setsid nohup python '
                     b'/x/run_upstream_batch.py --schedule s\0'),
                51: (b'/bin/bash\0-c\0EVID=/evid && setsid nohup python '
                     b'/x/upstream_fill_launcher.py --wait-pid 1\0')})
            drivers = wd.drivers_alive(str(proc))
            self.assertEqual([pid for pid, _ in drivers], [10])

    def test_launcher_scan_filters_by_evidence_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = self._fake_proc(tmp, {
                20: (b'python\0/x/upstream_fill_launcher.py\0'
                     b'--evidence-root\0/evid-a\0'),
                21: (b'python\0/x/upstream_fill_launcher.py\0'
                     b'--evidence-root\0/evid-b\0'),
                30: (b'python\0/x/upstream_chain_watchdog.py\0'
                     b'--launcher-script\0/x/upstream_fill_launcher.py\0'
                     b'--evidence-root\0/evid-a\0'),
                # the wrapper for evid-a's launcher mentions the evidence
                # root but runs no launcher script itself
                52: (b'/bin/bash\0-c\0EVID=/evid-a && setsid nohup python '
                     b'/x/upstream_fill_launcher.py --wait-pid 1 '
                     b'--evidence-root $EVID\0')})
            ours = wd.launcher_alive_for('/evid-a', str(proc))
            self.assertEqual([pid for pid, _ in ours], [20])


class WatchdogBatchStateTests(unittest.TestCase):

    def _index(self, path, cells):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'batches': [
                {'id': 'b1', 'cells': cells},
                {'id': 'other', 'cells': []}]}, f)

    def test_counts_only_four_role_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.json')
            full = {'sources': {r: {'path': 'p', 'sha256': 's'}
                                for r in wd.ROLES}}
            self._index(path, [full, {'sources': {}},
                               {'sources': {'upstream': {}}}])
            self.assertEqual(wd.batch_state(path, 'b1'), (3, 1))
            self.assertIsNone(wd.batch_state(path, 'missing'))
            self.assertIsNone(wd.batch_state(
                os.path.join(tmp, 'absent.json'), 'b1'))
            with open(path, 'w', encoding='utf-8') as f:
                f.write('{corrupt')
            self.assertIsNone(wd.batch_state(path, 'b1'))


class WatchdogMainTests(unittest.TestCase):

    def _index_complete(self, path, complete=True):
        cell = {'sources': ({r: {'path': 'p', 'sha256': 's'}
                             for r in wd.ROLES} if complete
                            else {'upstream': {}})}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({'batches': [{'id': 'b1', 'cells': [cell]}]}, f)

    def _args(self, tmp, log):
        ev = os.path.join(tmp, 'ev')
        os.makedirs(ev, exist_ok=True)
        return ['--batch-id', 'b1',
                '--index', os.path.join(tmp, 'index.json'),
                '--evidence-root', ev,
                '--schedule', os.path.join(tmp, 'sched.json'),
                '--snapshot-root', os.path.join(tmp, 'snap'),
                '--log', os.path.join(tmp, log),
                '--poll-s', '0', '--max-rearms', '1',
                '--min-spacing-s', '0', '--heartbeat-every', '1000']

    def test_main_exits_zero_on_complete_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._index_complete(os.path.join(tmp, 'index.json'))
            with mock.patch.object(wd, 'drivers_alive',
                                   return_value=[]), \
                    mock.patch.object(wd, 'launcher_alive_for',
                                      return_value=[]):
                rc = wd.main(self._args(tmp, 'wd.log'))
            self.assertEqual(rc, 0)
            self.assertIn('complete', open(os.path.join(
                tmp, 'wd.log')).read())

    def test_main_rearms_once_then_exits_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._index_complete(os.path.join(tmp, 'index.json'),
                                complete=False)
            with mock.patch.object(wd, 'drivers_alive',
                                   return_value=[]), \
                    mock.patch.object(wd, 'launcher_alive_for',
                                      return_value=[]), \
                    mock.patch('subprocess.Popen') as popen:
                rc = wd.main(self._args(tmp, 'wd.log'))
            self.assertEqual(rc, 3)
            self.assertEqual(popen.call_count, 1)
            cmd = popen.call_args[0][0]
            self.assertIn('--assume-stage1-done', cmd)
            self.assertIn('upstream_fill_launcher.py',
                          ' '.join(cmd))
            log = open(os.path.join(tmp, 'wd.log')).read()
            self.assertIn('re-arming the fill launcher', log)
            self.assertIn('BUDGET EXHAUSTED', log)

    def test_main_single_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._index_complete(os.path.join(tmp, 'index.json'))
            args = self._args(tmp, 'wd.log')
            ev_root = args[args.index('--evidence-root') + 1]
            lock_path = os.path.join(ev_root, 'watchdog.lock')
            held = open(lock_path, 'w')
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with mock.patch.object(wd, 'drivers_alive',
                                       return_value=[]), \
                        mock.patch.object(wd, 'launcher_alive_for',
                                          return_value=[]):
                    rc = wd.main(args)
                self.assertEqual(rc, 0)
                self.assertIn('another watchdog holds the lock',
                              open(os.path.join(tmp, 'wd.log')).read())
            finally:
                fcntl.flock(held, fcntl.LOCK_UN)
                held.close()


if __name__ == '__main__':
    unittest.main()

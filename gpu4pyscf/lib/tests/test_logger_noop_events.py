# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''LOGGER-NOOP-01: below-level gpu0 timer chains must not create CUDA events.

timer/_timer_debug2 return the chain-consistent tuple with a None event
below their verbosity level (the event was never read there); at or above
the level behavior is unchanged (event created, recorded, synchronized,
printed). _timer_debug1 KEEPS creating events below DEBUG1: its chained
events are read by _TimingCollector.collect (gpu4pyscf/scf/jk.py) through
cp.cuda.get_elapsed_time(t1p[2], t1[2]) at any verbosity. timer_silent and
init_timer are untouched measurement entry points.
'''

import io
import unittest
from unittest import mock

from pyscf.lib import logger as pyscf_lib_logger
from gpu4pyscf.lib import logger as glogger

NOTE = pyscf_lib_logger.NOTE
TIMER_LEVEL = pyscf_lib_logger.TIMER_LEVEL
DEBUG1 = pyscf_lib_logger.DEBUG1
DEBUG2 = pyscf_lib_logger.DEBUG2
FAKE_ELAPSED_MS = 7.25


class CountingEvent:
    creations = 0
    records = 0
    synchronizes = 0

    def __init__(self):
        CountingEvent.creations += 1

    def record(self):
        CountingEvent.records += 1

    def synchronize(self):
        CountingEvent.synchronizes += 1


class _Rec:
    '''Plain stub: a bare Mock would auto-create rec.TIMER_LEVEL and break
    logger.timer's getattr(rec, 'TIMER_LEVEL', TIMER_LEVEL) default.'''
    pass


def _rec(verbose=NOTE):
    rec = _Rec()
    rec.verbose = verbose
    rec.stdout = io.StringIO()
    return rec


def _reset_counts():
    CountingEvent.creations = 0
    CountingEvent.records = 0
    CountingEvent.synchronizes = 0


def _fake_gpu_chain(cpu0=1.0, wall0=2.0):
    seed = CountingEvent()
    _reset_counts()  # the seed itself must not pollute per-call assertions
    return (cpu0, wall0, seed)


class _EventPatchMixin:
    def setUp(self):
        CountingEvent.creations = CountingEvent.records = CountingEvent.synchronizes = 0
        patcher_ev = mock.patch.object(glogger.cupy.cuda, 'Event', CountingEvent)
        patcher_gt = mock.patch.object(
            glogger.cupy.cuda, 'get_elapsed_time', lambda a, b: FAKE_ELAPSED_MS)
        patcher_ev.start()
        patcher_gt.start()
        self.addCleanup(patcher_ev.stop)
        self.addCleanup(patcher_gt.stop)


class TestTimerNoopEvents(_EventPatchMixin, unittest.TestCase):
    def test_timer_zero_events_below_level(self):
        '''verbose=NOTE < TIMER_LEVEL: no Event creation, chain tuple kept, no print.'''
        rec = _rec(NOTE)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 0)
        self.assertEqual(len(t), 3)
        self.assertIsNone(t[2])
        self.assertEqual(rec.stdout.getvalue(), '')

    def test_timer_debug2_zero_events_below_level(self):
        '''verbose=NOTE < DEBUG2: no Event creation, chain tuple kept, no print.'''
        rec = _rec(NOTE)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer_debug2(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 0)
        self.assertEqual(len(t), 3)
        self.assertIsNone(t[2])
        self.assertEqual(rec.stdout.getvalue(), '')

    def test_timer_debug1_keeps_events_below_level(self):
        '''Divergence documented by LOGGER-NOOP-01: _timer_debug1's chained
        events are consumed by _TimingCollector at ANY verbosity, so the
        below-DEBUG1 branch must keep creating and recording real events.'''
        rec = _rec(NOTE)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer_debug1(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(CountingEvent.records, 1)
        self.assertEqual(CountingEvent.synchronizes, 0)
        self.assertEqual(len(t), 3)
        self.assertIsInstance(t[2], CountingEvent)
        self.assertEqual(rec.stdout.getvalue(), '')

    def test_timing_collector_contract_at_low_verbose(self):
        '''The hidden consumer itself: _TimingCollector wraps timer_debug1
        and reads both chained events below DEBUG1 without real CUDA.'''
        from gpu4pyscf.scf import jk
        log = glogger.Logger(stdout=io.StringIO(), verbose=NOTE)
        collector = jk._TimingCollector(log.timer_debug1)
        cpu0, wall0, seed = _fake_gpu_chain()
        t1 = (cpu0, wall0, seed)
        t1 = collector.collect('j', t1, 'step1')
        t1 = collector.collect('j', t1, 'step2')
        self.assertEqual(collector.collection['j'], 2 * FAKE_ELAPSED_MS * 1e3)
        self.assertIsInstance(t1[2], CountingEvent)

    def test_timer_identity_at_level(self):
        '''verbose >= TIMER_LEVEL: event created/recorded/synchronized once and
        the full three-part line printed with the GPU elapsed time (unchanged
        branch).'''
        rec = _rec(TIMER_LEVEL)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(CountingEvent.records, 1)
        self.assertEqual(CountingEvent.synchronizes, 1)
        self.assertEqual(len(t), 3)
        out = rec.stdout.getvalue()
        self.assertIn('CPU time for msg', out)
        self.assertIn('wall time', out)
        self.assertIn('GPU time %9.2f ms' % FAKE_ELAPSED_MS, out)

    def test_timer_debug2_identity_at_level(self):
        rec = _rec(DEBUG2)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer_debug2(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(CountingEvent.synchronizes, 1)
        self.assertIn('GPU time %9.2f ms' % FAKE_ELAPSED_MS, rec.stdout.getvalue())

    def test_timer_debug1_identity_at_level(self):
        rec = _rec(DEBUG1)
        cpu0, wall0, seed = _fake_gpu_chain()
        t = glogger.timer_debug1(rec, 'msg', cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(CountingEvent.synchronizes, 1)
        self.assertIn('GPU time %9.2f ms' % FAKE_ELAPSED_MS, rec.stdout.getvalue())

    def test_chain_degradation_arity(self):
        '''gpu0 -> wall0 -> cpu-only -> gpu0 again: each hop keeps the
        star-unpack contract (t0 = logger.timer(rec, m, *t0)) and creates
        zero events below the level.'''
        rec = _rec(NOTE)
        t = _fake_gpu_chain()
        t = glogger.timer(rec, 'gpu hop', *t)
        self.assertEqual(len(t), 3)
        self.assertIsNone(t[2])
        t = glogger.timer(rec, 'wall hop', *t)
        self.assertEqual(len(t), 2)
        t = glogger.timer(rec, 'wall hop 2', *t)
        self.assertEqual(len(t), 2)
        t = (t[0],)
        t = glogger.timer(rec, 'cpu hop', *t)
        self.assertEqual(len(t), 1)
        t = (t[0], 3.0, CountingEvent())
        _reset_counts()
        t = glogger.timer(rec, 'gpu again', *t)
        self.assertEqual(len(t), 3)
        self.assertIsNone(t[2])
        self.assertEqual(CountingEvent.creations, 0)

    def test_timer_silent_untouched(self):
        '''timer_silent remains a real (silent) measurement entry point.'''
        rec = _rec(NOTE)
        cpu0, wall0, seed = _fake_gpu_chain()
        dcpu, dwall, dgpu = glogger.timer_silent(rec, cpu0, wall0, seed)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(dgpu, FAKE_ELAPSED_MS)
        self.assertEqual(rec.stdout.getvalue(), '')

    def test_init_timer_untouched(self):
        rec = _rec(NOTE)
        t = glogger.init_timer(rec)
        self.assertEqual(CountingEvent.creations, 1)
        self.assertEqual(CountingEvent.records, 1)
        self.assertEqual(len(t), 3)
        self.assertIsInstance(t[2], CountingEvent)


if __name__ == '__main__':
    unittest.main()

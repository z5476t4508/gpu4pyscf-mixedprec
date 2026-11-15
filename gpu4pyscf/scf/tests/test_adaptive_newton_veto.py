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

"""Regression tests for the CONV-UKS-DISPATCH-VETO-01 opt-in
fire-time veto on the adaptive-newton dispatcher
(GPU4PYSCF_SCF_ADAPTIVE_NEWTON_VETO, gpu4pyscf/scf/hf.py).

Provenance: entry 447 (CONV-UKS-DISPATCH-WINPRED-SCREEN-01,
2026-11-07) screened the ALREADY-CAPTURED pre-fire indicator
trajectories of entries 443/445/446 (9 WIN fired trajectories over 3
contexts of ibuprofen-573 vs 3 LOSS trajectories of phenacetin-481)
and found the WIN/LOSS information at fire time living in trajectory
SHAPE, not the fire level: F03 mean(v[7..12]) 0.4754-0.4786 vs
0.09434; F06 log-slope (ln v12 - ln v7)/5 -0.3426..-0.3271 vs
+0.03051; F08 v12-v11 -0.19475..-0.16049 vs -0.017330 (disjoint
ranges, gap >= 2x within-class spread and >= 10% of level).  F01
fire level did NOT separate (confirms the entry-445
near-identical-levels prior).  The veto thresholds are the exact
midpoints of those frozen gaps; F11 nao separated only as a cell
fingerprint and is excluded (overfit risk).

Mechanism: when BOTH the dispatcher (GPU4PYSCF_SCF_ADAPTIVE_NEWTON)
and this knob are armed, every WOULD-BE fire (cycle+1 >= min_cycle
and |ddm| >= ddm threshold) additionally evaluates the frozen 3-vote
rule on the last 6 trace entries; >= 2 LOSS votes suppresses that
dispatch (plain CDIIS keeps control to its natural convergence) and
records a veto event.  Veto-off lanes are bitwise verbatim 9e76115
(the check short-circuits on a False local inside the fire branch;
no numerical operation is added).  Continuous re-evaluation: a later
cycle whose window no longer says LOSS may still fire.

Contract tests (source-level, CPU):
  * exactly one env reader for the veto knob (home module scf.hf);
    '1'/'on' arm; ''/'0'/'off' verbatim; malformed -> warn+off;
  * veto constants are EXACTLY the entry-447 midpoint-of-gap values,
    re-derived in-test from the pinned 447 range endpoints;
  * _adaptive_veto_decision is a pure function of the trace (no env,
    no GPU): >= 2 of 3 LOSS votes with frozen >= / <= semantics;
    short (<6) or non-contiguous windows never veto; the log-slope
    vote is undecidable (not LOSS) on non-positive endpoints;
  * the veto site sits INSIDE the dispatcher fire branch, wrapped in
    `if not _vetoed:` around the VERBATIM fire body; the std
    convergence test and every tolerance stay untouched;
  * veto armed while the dispatcher knob is off warns once and is
    inert; strict fp64-reference lanes never arm;
  * home-module suppression counter counts real suppressions only;
  * absence contract: the veto knob string never appears outside the
    home module and this test file (green at the parent by
    construction).

Numerical tests (GPU, small DF-UKS wb97mv sentinel, the corpus method
class).  Trace-shape injection DISCLOSURE: the LOSS/WIN shape tests
drive mf.callback to REPLACE _adaptive_state['indicator_trace'] (pure
evidence/bookkeeping state, never computational state) with a
synthetic 6-entry window before the engineered fire cycle, so the
in-kernel suppression path runs end-to-end on the tiny sentinel
without a real grind cell; the real grind cells are the formal GPU
phase's job.  The callback writes the observable the decision reads,
nothing else:
  * default OFF: dispatcher-on + veto absent/'0' reproduces the
    9e76115 fire class bitwise (e, cycles, fire evidence);
  * engineered short fire (cycle 1): the <6-entry window guard keeps
    the fire (no veto) -- veto-on == veto-off;
  * LOSS-shaped window at an engineered cycle-6 fire: fire
    SUPPRESSED, plain CDIIS continues to natural convergence, e_tot
    BITWISE == the plain knob-off run (suppression adds nothing to
    the numerical trajectory), veto event + counter + no fired attr;
  * WIN-shaped window at the same engineered fire: fires (honest
    unconverged head), no veto event;
  * veto armed alone (dispatcher off): warns, bitwise verbatim, zero
    counters, no state;
  * strict-fp64-reference: inert, bitwise verbatim;
  * per-run state resets across kernels while the suppression
    counter accumulates.
"""
import os
import re
import sys
import unittest
import warnings

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')
KNOB = 'GPU4PYSCF_SCF_ADAPTIVE_NEWTON_VETO'
DISPATCH_KNOB = 'GPU4PYSCF_SCF_ADAPTIVE_NEWTON'
# modules that must stay veto-knob-free (absence contract; the home
# module and this test file are the only allowed occurrences)
OTHER_MODULES = [
    os.path.join(REPO, 'gpu4pyscf', 'scf', 'soscf.py'),
    os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py'),
    os.path.join(REPO, 'gpu4pyscf', 'df', 'df_jk.py'),
]

# entry-447 frozen ranges (full precision, winpredscreen01-verdict.json)
F03_WIN = [0.4753593805940451, 0.4786405852758531]
F03_LOSS = [0.09434038113570287, 0.09434038113570287]
F06_WIN = [-0.3425851982792107, -0.3271361227271979]
F06_LOSS = [0.03051159540236039, 0.03051159540236039]
F08_WIN = [-0.19474501360979996, -0.1604855771740755]
F08_LOSS = [-0.017329982233245625, -0.017329982233245625]


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


def _mk_trace(vals, start=1):
    return [(start + i, v) for i, v in enumerate(vals)]


# the measured fire-window shapes (entry 447 raw prefixes, cycles 7..12)
WIN_WINDOW = [0.5923, 0.7969, 0.7176, 0.3197, 0.3101, 0.1154]    # 573 @12
LOSS_WINDOW = [0.09416, 0.104, 0.07681, 0.0544, 0.127, 0.1097]   # 481 @12


class TestSourceContract(unittest.TestCase):
    def test_veto_parser_contract(self):
        parser = _func_src(HF_SRC, '_scf_adaptive_newton_veto')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader for the veto knob (home module)
        src = open(HF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        self.assertIn("raw in ('1', 'on')", parser)
        self.assertIn("raw not in ('', '0', 'off')", parser)
        self.assertIn('warnings.warn', parser)

    def test_threshold_constants_are_447_midpoints(self):
        from gpu4pyscf.scf import hf
        # exact midpoint-of-gap re-derivation from the pinned 447
        # ranges (F03/F06/F08 are disjoint with the gap edge at
        # lmax|wmin / wmax|lmin respectively)
        self.assertEqual(hf._ADAPTIVE_VETO_MEAN_LATE,
                         (F03_LOSS[1] + F03_WIN[0]) / 2.0)
        self.assertEqual(hf._ADAPTIVE_VETO_LOG_SLOPE,
                         (F06_WIN[1] + F06_LOSS[0]) / 2.0)
        self.assertEqual(hf._ADAPTIVE_VETO_LAST_DELTA,
                         (F08_WIN[1] + F08_LOSS[0]) / 2.0)
        # and the frozen literal values, verbatim
        self.assertEqual(repr(hf._ADAPTIVE_VETO_MEAN_LATE),
                         '0.28484988086487395')
        self.assertEqual(repr(hf._ADAPTIVE_VETO_LOG_SLOPE),
                         '-0.14831226366241876')
        self.assertEqual(repr(hf._ADAPTIVE_VETO_LAST_DELTA),
                         '-0.08890777970366057')

    def test_decision_fn_pure_and_home_module(self):
        fn = _func_src(HF_SRC, '_adaptive_veto_decision')
        self.assertIsNotNone(fn)
        # pure function of the trace: no env read, no GPU work
        self.assertNotIn('os.environ', fn)
        self.assertNotIn('cupy', fn)
        # frozen vote semantics
        self.assertIn('<= _ADAPTIVE_VETO_MEAN_LATE', fn)
        self.assertIn('>= _ADAPTIVE_VETO_LOG_SLOPE', fn)
        self.assertIn('>= _ADAPTIVE_VETO_LAST_DELTA', fn)
        # 2-of-3 rule
        self.assertIn('loss_votes >= 2', fn)

    def test_kernel_hook_contract(self):
        body = _func_src(HF_SRC, '_kernel_body')
        self.assertIsNotNone(body)
        # the VERBATIM std convergence test, exactly once, PRIOR to the
        # fire rule (unchanged by this child)
        self.assertEqual(
            body.count('if(e_diff < conv_tol and norm_gorb < '
                       'conv_tol_grad):'), 1)
        std_pos = body.index('if(e_diff < conv_tol and norm_gorb < '
                             'conv_tol_grad):')
        fire_pos = body.index('if (_adaptive_dispatch and cycle + 1 >=')
        self.assertLess(std_pos, fire_pos)
        # the veto sits INSIDE the fire branch; the original fire body
        # is guarded by the vetoed local (knob-off adds no numerical op)
        veto_pos = body.index('_adaptive_veto_decision(')
        self.assertGreater(veto_pos, fire_pos)
        guard_pos = body.index('if not _vetoed:')
        self.assertGreater(guard_pos, veto_pos)
        fired_pos = body.index("mf._adaptive_newton_fired_cycle = "
                               "cycle + 1")
        self.assertGreater(fired_pos, guard_pos)
        # suppression records the event + counter + mf marker
        self.assertIn("_adaptive_state['veto_events'].append(", body)
        self.assertIn('_note_adaptive_veto_suppress()', body)
        self.assertIn('mf._adaptive_newton_veto_cycles = ', body)
        # the veto arming local requires the dispatcher (never lonely)
        self.assertIn('_adaptive_veto_armed = ', body)

    def test_counter_helper_home_module(self):
        helper = _func_src(HF_SRC, '_note_adaptive_veto_suppress')
        self.assertIsNotNone(helper)
        self.assertIn('global _ADAPTIVE_VETO_SUPPRESSES', helper)

    def test_lonely_warn_helper(self):
        helper = _func_src(HF_SRC, '_warn_adaptive_veto_lonely')
        self.assertIsNotNone(helper)
        self.assertIn('warnings.warn', helper)
        self.assertIn('global _ADAPTIVE_VETO_LONELY_WARNED', helper)

    def test_absence_contract_scope_guard(self):
        # the veto knob string lives in the home module (+ this test
        # file) and NOWHERE else: green at the parent (zero
        # occurrences anywhere) and at HEAD (home-module-only) by
        # construction
        for path in OTHER_MODULES:
            self.assertEqual(open(path).read().count(KNOB), 0,
                             '%s must stay veto-knob-free' % path)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import hf
        self.hf = hf
        self._saved = (os.environ.get(KNOB), hf._ADAPTIVE_VETO)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.hf._ADAPTIVE_VETO = cache

    def _parse(self, raw):
        self.hf._ADAPTIVE_VETO = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.hf._scf_adaptive_newton_veto()

    def test_values(self):
        self.assertFalse(self._parse(None))
        self.assertFalse(self._parse(''))
        self.assertFalse(self._parse('0'))
        self.assertFalse(self._parse('off'))
        self.assertTrue(self._parse('1'))
        self.assertTrue(self._parse('on'))
        for bad in ('yes', 'true', '2', '01'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                val = self._parse(bad)
            self.assertFalse(val, 'value %r must be ignored' % bad)
            self.assertTrue(any(KNOB in str(x.message) for x in w),
                            'value %r must warn' % bad)


class TestDecisionArithmetic(unittest.TestCase):
    """Pure-CPU arithmetic of the frozen 3-vote rule (no GPU)."""

    @classmethod
    def setUpClass(cls):
        from gpu4pyscf.scf import hf
        cls.hf = hf

    def _decide(self, vals, start=1):
        return self.hf._adaptive_veto_decision(_mk_trace(vals, start))

    def test_measured_win_window_no_veto(self):
        veto, ev = self._decide(WIN_WINDOW, start=7)
        self.assertFalse(veto)
        self.assertEqual(ev['loss_votes'], 0)
        self.assertFalse(any(ev['votes'].values()))

    def test_measured_loss_window_vetoes(self):
        veto, ev = self._decide(LOSS_WINDOW, start=7)
        self.assertTrue(veto)
        self.assertEqual(ev['loss_votes'], 3)
        self.assertTrue(all(ev['votes'].values()))

    def test_threshold_boundary_semantics(self):
        # exact-threshold values are LOSS votes (<= for mean, >= for
        # slope/delta -- the frozen inclusive semantics); a flat
        # window at t_mean sits exactly on all three boundaries
        t_mean = self.hf._ADAPTIVE_VETO_MEAN_LATE
        vals = [t_mean] * 6
        veto, ev = self._decide(vals)
        self.assertTrue(ev['votes']['mean_late'])       # mean == T
        self.assertTrue(ev['votes']['last_delta'])      # 0 >= T_delta
        self.assertTrue(ev['votes']['log_slope_late'])  # 0 >= T_slope
        self.assertEqual(ev['loss_votes'], 3)
        self.assertTrue(veto)
        # mean just above threshold alone is not enough when the other
        # two votes still say LOSS (2-of-3 still vetoes)
        x = t_mean * 1.001
        veto2, ev2 = self._decide([x] * 5 + [x - 1e-9])
        self.assertFalse(ev2['votes']['mean_late'])     # mean > T
        self.assertTrue(ev2['votes']['last_delta'])     # ~0 >= T_delta
        self.assertTrue(ev2['votes']['log_slope_late']) # ~0 >= T_slope
        self.assertTrue(veto2)

    def test_two_of_three_mixed(self):
        t_mean = self.hf._ADAPTIVE_VETO_MEAN_LATE
        # mean + slope LOSS, delta WIN: low window with a mild
        # overall rise (slope >= T) but a steeply falling last step
        vals = [t_mean, t_mean, t_mean, t_mean, 0.30, 0.16]
        veto, ev = self._decide(vals)
        self.assertTrue(ev['votes']['mean_late'])
        self.assertTrue(ev['votes']['log_slope_late'])
        self.assertFalse(ev['votes']['last_delta'])
        self.assertTrue(veto)
        self.assertEqual(ev['loss_votes'], 2)
        # mean + delta LOSS, slope WIN: strong overall decay whose
        # last step is nearly flat
        vals = [0.5, 0.2, 0.15, 0.13, 0.12, 0.11]
        veto, ev = self._decide(vals)
        self.assertTrue(ev['votes']['mean_late'])
        self.assertFalse(ev['votes']['log_slope_late'])
        self.assertTrue(ev['votes']['last_delta'])
        self.assertTrue(veto)
        # only ONE LOSS vote (delta alone) -> no veto
        vals = [5.0, 4.0, 3.0, 2.0, 1.0, 1.05]
        veto, ev = self._decide(vals)
        self.assertEqual(ev['loss_votes'], 1)
        self.assertFalse(ev['votes']['mean_late'])
        self.assertFalse(ev['votes']['log_slope_late'])
        self.assertTrue(ev['votes']['last_delta'])
        self.assertFalse(veto)

    def test_short_or_noncontiguous_never_vetoes(self):
        for n in range(0, 6):
            veto, ev = self._decide(LOSS_WINDOW[:n])
            self.assertFalse(veto)
            self.assertEqual(ev.get('loss_votes'), 0)
        # non-contiguous cycle labels (gap): refuse to decide
        trace = [(1, LOSS_WINDOW[0]), (2, LOSS_WINDOW[1]),
                 (3, LOSS_WINDOW[2]), (5, LOSS_WINDOW[3]),
                 (6, LOSS_WINDOW[4]), (7, LOSS_WINDOW[5])]
        veto, ev = self.hf._adaptive_veto_decision(trace)
        self.assertFalse(veto)
        self.assertEqual(ev.get('loss_votes'), 0)

    def test_log_guard_undecidable_not_loss(self):
        # zero endpoints make the log-slope vote undecidable (not a
        # LOSS vote); the other two votes still decide 2-of-3
        t_mean = self.hf._ADAPTIVE_VETO_MEAN_LATE
        vals = [0.0, t_mean, t_mean, t_mean, t_mean, t_mean]
        veto, ev = self._decide(vals)
        self.assertIsNone(ev['log_slope_late'])
        self.assertFalse(ev['votes']['log_slope_late'])
        self.assertTrue(ev['votes']['mean_late'])
        self.assertTrue(ev['votes']['last_delta'])
        self.assertTrue(veto)
        # with the guard vote undecidable and only one other LOSS
        # vote, no veto
        vals = [0.0, 1.0, 1.0, 1.0, 1.0, 1.05]
        veto, ev = self._decide(vals)
        self.assertIsNone(ev['log_slope_late'])
        self.assertEqual(ev['loss_votes'], 1)   # last_delta 1.05-1.0
        self.assertFalse(veto)


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS wb97mv sentinel (the corpus method class; the
    plain-CDIIS lane is bitwise-reproducible run-to-run on this
    stack).  Trace-shape injection via mf.callback replaces ONLY the
    dispatcher's indicator_trace evidence list (bookkeeping state)
    before an engineered fire cycle, so the in-kernel veto path runs
    end-to-end on the sentinel; every knob is restored per test.'''

    ATOM = [['O', (0., 0., 0.)], ['O', (0., 0., 1.2078)]]

    def _env(self, dispatch=None, veto=None, ddm=None, min_cycle=None):
        from gpu4pyscf.scf import hf
        hf._ADAPTIVE_NEWTON = None   # force re-read
        hf._ADAPTIVE_VETO = None
        for key in (DISPATCH_KNOB, DISPATCH_KNOB + '_DDM',
                    DISPATCH_KNOB + '_MIN_CYCLE', KNOB):
            os.environ.pop(key, None)
        if dispatch is not None:
            os.environ[DISPATCH_KNOB] = dispatch
        if veto is not None:
            os.environ[KNOB] = veto
        if ddm is not None:
            os.environ[DISPATCH_KNOB + '_DDM'] = ddm
        if min_cycle is not None:
            os.environ[DISPATCH_KNOB + '_MIN_CYCLE'] = min_cycle
        return hf

    def _restore(self):
        for key in (DISPATCH_KNOB, DISPATCH_KNOB + '_DDM',
                    DISPATCH_KNOB + '_MIN_CYCLE', KNOB):
            os.environ.pop(key, None)

    def setUp(self):
        self._restore()

    tearDown = setUp

    def _run(self, dispatch=None, veto=None, ddm=None, min_cycle=None,
             strict_ref=False, max_cycle=50, inject_window=None,
             inject_from_cycle=None):
        """One fresh plain-CDIIS kernel.  inject_window REPLACES the
        indicator_trace (evidence-only injection) at the end of EVERY
        cycle >= inject_from_cycle with the 6-entry window labeled to
        end at the current cycle -- the veto re-evaluates at every
        would-be fire, so the injected observable must persist for the
        whole run (a one-shot injection would let later cycles decide
        on real values)."""
        from pyscf import gto
        from gpu4pyscf import dft
        hf = self._env(dispatch, veto, ddm, min_cycle)
        mol = gto.M(verbose=0, output='/dev/null', atom=self.ATOM,
                    basis='def2-svp', charge=0, spin=2)
        mf = dft.UKS(mol, xc='wb97mv').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 3
        mf.max_cycle = max_cycle
        if strict_ref:
            mf._strict_fp64_reference = True

        def _inject(envs):
            state = getattr(hf, '_ADAPTIVE_NEWTON_RUN', None)
            if (state is not None and inject_window is not None
                    and envs['cycle'] + 1 >= inject_from_cycle):
                # evidence-state injection only: the decision reads the
                # trace; no computational state is touched
                c = envs['cycle'] + 1
                state['indicator_trace'][:] = [
                    (c - 5 + i, v) for i, v in enumerate(inject_window)]

        if inject_window is not None:
            mf.callback = _inject
        fires0 = hf._ADAPTIVE_NEWTON_FIRES
        supp0 = hf._ADAPTIVE_VETO_SUPPRESSES
        e = mf.kernel()
        state = getattr(hf, '_ADAPTIVE_NEWTON_RUN', None)
        rec = {
            'fired': bool(state is not None and state['fired']),
            'fire_cycle': (state['fire_cycle'] if state is not None
                           else None),
            'veto_events': (list(state['veto_events'])
                            if state is not None
                            and 'veto_events' in state else None),
            'veto_armed_state': (state.get('veto_armed')
                                 if state is not None else None),
            'fires_delta': hf._ADAPTIVE_NEWTON_FIRES - fires0,
            'suppress_delta': hf._ADAPTIVE_VETO_SUPPRESSES - supp0,
            'converged': bool(mf.converged),
            'cycles': int(mf.cycles),
            'mf_fired_attr': getattr(mf, '_adaptive_newton_fired_cycle',
                                     None),
            'mf_veto_attr': getattr(mf, '_adaptive_newton_veto_cycles',
                                    None),
        }
        self._restore()
        return rec, repr(float(e))

    def test_veto_off_reproduces_parent_fire_class(self):
        # dispatcher-on + veto ABSENT: the 9e76115 fire class verbatim
        r1, e1 = self._run(dispatch='1', ddm='1.0', min_cycle='1')
        self.assertTrue(r1['fired'])
        self.assertEqual(r1['fire_cycle'], 1)
        self.assertEqual(r1['fires_delta'], 1)
        self.assertEqual(r1['suppress_delta'], 0)
        self.assertIsNone(r1['veto_events'])   # no veto key (9e76115)
        self.assertIsNone(r1['mf_veto_attr'])
        # explicit '0' is the same lane, bitwise
        r2, e2 = self._run(dispatch='1', veto='0', ddm='1.0',
                           min_cycle='1')
        self.assertEqual(e2, e1)
        self.assertTrue(r2['fired'])
        self.assertEqual(r2['fire_cycle'], 1)
        self.assertEqual(r2['suppress_delta'], 0)

    def test_short_engineered_fire_not_vetoed(self):
        # a cycle-1 fire has a <6-entry window: the guard keeps the
        # fire under the veto knob (no veto) -- veto-on == veto-off
        r_off, e_off = self._run(dispatch='1', ddm='1.0',
                                 min_cycle='1')
        r, e = self._run(dispatch='1', veto='1', ddm='1.0',
                         min_cycle='1')
        self.assertTrue(r['fired'])
        self.assertEqual(r['fire_cycle'], 1)
        self.assertEqual(r['suppress_delta'], 0)
        self.assertEqual(e, e_off)
        self.assertEqual(r['veto_armed_state'], True)

    def test_loss_window_suppresses_fire_bitwise_plain(self):
        # engineered cycle-6+ would-be fires with a LOSS-shaped window:
        # every dispatch is SUPPRESSED (continuous re-evaluation), plain
        # CDIIS continues to natural convergence, and e_tot is BITWISE
        # the plain knob-off run
        r_plain, e_plain = self._run()
        r, e = self._run(dispatch='1', veto='1', ddm='1e-12',
                         min_cycle='6', inject_window=LOSS_WINDOW,
                         inject_from_cycle=6)
        self.assertFalse(r['fired'])
        self.assertIsNone(r['fire_cycle'])
        self.assertIsNone(r['mf_fired_attr'])
        self.assertTrue(r['converged'])
        self.assertEqual(e, e_plain)   # bitwise plain trajectory
        self.assertEqual(r['cycles'], r_plain['cycles'])
        n = r_plain['cycles']
        # would-be fires at cycles 6..n-1 (the std test preempts at n):
        # every one vetoed, zero FIRES
        self.assertEqual(r['fires_delta'], 0)
        expected_cycles = list(range(6, n))
        self.assertEqual(r['suppress_delta'], len(expected_cycles))
        self.assertEqual([c for c, _ in r['veto_events']],
                         expected_cycles)
        for _, ev in r['veto_events']:
            self.assertEqual(ev['loss_votes'], 3)
        self.assertEqual(r['mf_veto_attr'], expected_cycles)

    def test_win_window_fires_under_veto(self):
        # the same engineered cycle-6 fire with a WIN-shaped window:
        # no veto, fires exactly like the veto-off dispatcher lane
        r_ref, e_ref = self._run(dispatch='1', ddm='1e-12',
                                 min_cycle='6', inject_window=WIN_WINDOW,
                                 inject_from_cycle=6)
        self.assertTrue(r_ref['fired'])
        self.assertEqual(r_ref['fire_cycle'], 6)
        r, e = self._run(dispatch='1', veto='1', ddm='1e-12',
                         min_cycle='6', inject_window=WIN_WINDOW,
                         inject_from_cycle=6)
        self.assertTrue(r['fired'])
        self.assertEqual(r['fire_cycle'], 6)
        self.assertEqual(r['mf_fired_attr'], 6)
        self.assertEqual(r['suppress_delta'], 0)
        self.assertEqual(r['veto_events'], [])
        self.assertFalse(r['converged'])   # honest unconverged head
        self.assertEqual(e, e_ref)
        self.assertEqual(r['cycles'], 6)

    def test_veto_alone_warns_and_is_inert(self):
        r_plain, e_plain = self._run()
        from gpu4pyscf.scf import hf
        hf._ADAPTIVE_VETO_LONELY_WARNED = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r, e = self._run(veto='1')
        self.assertTrue(any(KNOB in str(x.message) for x in w))
        self.assertEqual(r['fires_delta'], 0)
        self.assertEqual(r['suppress_delta'], 0)
        self.assertFalse(r['fired'])
        self.assertIsNone(r['veto_events'])
        self.assertEqual(e, e_plain)   # bitwise verbatim
        # warned once, not per kernel
        with warnings.catch_warnings(record=True) as w2:
            warnings.simplefilter('always')
            self._run(veto='1')
        self.assertFalse(any(KNOB in str(x.message) for x in w2))

    def test_strict_ref_never_arms(self):
        r_plain, e_plain = self._run()
        r, e = self._run(dispatch='1', veto='1', ddm='1.0',
                         min_cycle='1', strict_ref=True)
        self.assertFalse(r['fired'])
        self.assertIsNone(r['mf_fired_attr'])
        self.assertIsNone(r['mf_veto_attr'])
        self.assertEqual(r['fires_delta'], 0)
        self.assertEqual(r['suppress_delta'], 0)
        self.assertTrue(r['converged'])
        self.assertEqual(e, e_plain)   # bitwise verbatim

    def test_per_run_state_reset_counter_accumulates(self):
        # two suppressed kernels in one process: veto_events reset
        # per kernel (fresh lists, own cycles), the suppression counter
        # accumulates across kernels
        r1, _ = self._run(dispatch='1', veto='1', ddm='1e-12',
                          min_cycle='6', inject_window=LOSS_WINDOW,
                          inject_from_cycle=6)
        r2, _ = self._run(dispatch='1', veto='1', ddm='1e-12',
                          min_cycle='6', inject_window=LOSS_WINDOW,
                          inject_from_cycle=6)
        self.assertGreaterEqual(r1['suppress_delta'], 1)
        self.assertEqual(r2['suppress_delta'], r1['suppress_delta'])
        self.assertGreaterEqual(len(r1['veto_events']), 1)
        self.assertGreaterEqual(len(r2['veto_events']), 1)  # reset, fresh
        self.assertEqual([c for c, _ in r2['veto_events']],
                         [c for c, _ in r1['veto_events']])


if __name__ == '__main__':
    unittest.main()

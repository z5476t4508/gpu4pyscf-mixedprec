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

"""Regression tests for the CONV-UKS-ADAPTIVE-SWITCH-01 opt-in
self-limiting adaptive-newton dispatcher
(GPU4PYSCF_SCF_ADAPTIVE_NEWTON, gpu4pyscf/scf/hf.py _kernel_body).

Provenance: entry 442 (CONV-UKS-GRIND-PREDICT-01, 2026-11-07) measured
the per-cycle CDIIS indicator I1=|ddm| separating the grind class
(ibuprofen-573: 0.115 @ cycle 12, 49 plain cycles; the composed newton
wins there, entry 428 ratio 0.7173) from the easy class
(naproxen-611/testosterone-819: <= 0.00707 @ 12, 21/17 cycles; the
fixed switch loses there, entries 433/414) by 16.27x on the closed
corpus.  This knob arms the frozen midpoint rule (0.0285 @ cycle >=
12) as an additional loop-exit in the plain SCF kernel; the caller
owns the newton handoff.  Mechanism site: the corpus UKS cells run
gpu4pyscf _kernel_body (the class binds kernel/scf to the module-level
scf(); the site-packages pyscf check_convergence pivot is NOT on this
path), and the level_shift-gated extra recheck never runs on default
settings -- the fire breaks the loop directly and scf_conv is honestly
reset after the loop (the entry-414/428 fixed-k head class).

Contract tests (source-level, CPU):
  * exactly one env reader for the knob (home module scf.hf);
  * '1'/'on' arm; ''/'0'/'off' keep verbatim; malformed -> warn+off;
  * companion DDM/MIN_CYCLE params: malformed/non-finite/out-of-domain
    -> warn + frozen defaults (0.0285 / 12);
  * the _kernel_body std convergence test stays VERBATIM and prior to
    the fire rule; the fire rule and indicator record short-circuit on
    the armed local; the post-loop honest reset is fired-only;
  * strict fp64-reference lanes never arm;
  * absence contract: the knob string never appears outside the home
    module and this test file (green at the parent by construction).

Numerical tests (GPU, small DF-UKS wb97mv sentinel, the corpus method
class):
  * default OFF (absent and explicit '0'): run-to-run bitwise e_tot,
    zero fires, no mf._adaptive_newton_fired_cycle attribute;
  * armed huge threshold: fires exactly at cycle 1 (min_cycle=1),
    head returns UNCONVERGED (honest reset), counter == 1,
    indicator trace recorded;
  * armed impossible threshold: never fires, e_tot bitwise == the
    knob-off run, converged naturally, zero fires;
  * min_cycle gate: threshold 0.0 fires exactly at cycle min_cycle;
  * malformed knob value: warn + verbatim (bitwise e_tot, 0 fires);
  * malformed params: warn + frozen defaults from the parser;
  * strict-fp64-reference mf: knob armed but never engages (bitwise
    verbatim, 0 fires);
  * two armed kernels in one process: per-run state resets, the fire
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
KNOB = 'GPU4PYSCF_SCF_ADAPTIVE_NEWTON'
KNOB_DDM = KNOB + '_DDM'
KNOB_MIN = KNOB + '_MIN_CYCLE'
# modules that must stay knob-free (absence contract; the home module
# and this test file are the only allowed occurrences)
OTHER_MODULES = [
    os.path.join(REPO, 'gpu4pyscf', 'scf', 'soscf.py'),
    os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py'),
    os.path.join(REPO, 'gpu4pyscf', 'df', 'df_jk.py'),
]


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(HF_SRC, '_scf_adaptive_newton')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(HF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # accepted values + warn-and-ignore
        self.assertIn("raw in ('1', 'on')", parser)
        self.assertIn("raw not in ('', '0', 'off')", parser)
        self.assertIn('warnings.warn', parser)

    def test_param_parser_contract(self):
        parser = _func_src(HF_SRC, '_scf_adaptive_newton_params')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB_DDM, parser)
        self.assertIn("os.environ.get('%s'" % KNOB_MIN, parser)
        # frozen defaults + strict-domain warn-and-fallback
        self.assertIn('math.isfinite(val) and val > 0.0', parser)
        self.assertIn('if val < 1:', parser)
        self.assertIn('warnings.warn', parser)

    def test_counter_helper_home_module(self):
        helper = _func_src(HF_SRC, '_note_adaptive_newton_fire')
        self.assertIsNotNone(helper)
        self.assertIn('global _ADAPTIVE_NEWTON_FIRES', helper)

    def test_kernel_hook_contract(self):
        body = _func_src(HF_SRC, '_kernel_body')
        self.assertIsNotNone(body)
        # the VERBATIM std convergence test, exactly once, PRIOR to the
        # fire rule
        self.assertEqual(
            body.count('if(e_diff < conv_tol and norm_gorb < '
                       'conv_tol_grad):'), 1)
        std_pos = body.index('if(e_diff < conv_tol and norm_gorb < '
                             'conv_tol_grad):')
        fire_pos = body.index('if (_adaptive_dispatch and cycle + 1 >=')
        self.assertLess(std_pos, fire_pos)
        # the fire rule short-circuits on the armed local and never
        # touches the tolerances
        self.assertIn('float(norm_ddm) >= _adaptive_ddm_threshold', body)
        # the fire marks the run + mf attribute, then breaks
        self.assertIn("mf._adaptive_newton_fired_cycle = cycle + 1", body)
        self.assertIn('_note_adaptive_newton_fire()', body)
        # honest reset after the loop: fired-only, before the return
        reset_pos = body.index(
            'if _adaptive_dispatch and _adaptive_state[\'fired\']:')
        self.assertLess(fire_pos, reset_pos)
        self.assertLess(reset_pos, body.rindex('return scf_conv, e_tot, '
                                               'mo_energy, mo_coeff, '
                                               'mo_occ'))
        # indicator record inside the loop (engagement evidence)
        self.assertIn("_adaptive_state['indicator_trace'].append(", body)

    def test_strict_ref_guard_on_armed_local(self):
        body = _func_src(HF_SRC, '_kernel_body')
        self.assertIsNotNone(body)
        self.assertIn(
            "_adaptive_dispatch = (_scf_adaptive_newton()\n"
            "                          and not getattr(mf, "
            "'_strict_fp64_reference',\n"
            "                                          False))", body)

    def test_absence_contract_scope_guard(self):
        # the knob string lives in the home module (+ this test file)
        # and NOWHERE else: green at the parent (zero occurrences) and
        # at HEAD (home-module-only) by construction
        for path in OTHER_MODULES:
            self.assertEqual(open(path).read().count(KNOB), 0,
                             '%s must stay knob-free' % path)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import hf
        self.hf = hf
        self._saved = (os.environ.get(KNOB), hf._ADAPTIVE_NEWTON)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.hf._ADAPTIVE_NEWTON = cache

    def _parse(self, raw):
        self.hf._ADAPTIVE_NEWTON = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.hf._scf_adaptive_newton()

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

    def test_params_defaults_and_malformed(self):
        for key in (KNOB_DDM, KNOB_MIN):
            os.environ.pop(key, None)
        # frozen defaults on absent companions
        self.assertEqual(self.hf._scf_adaptive_newton_params(),
                         (0.0285, 12))
        # explicit valid values
        os.environ[KNOB_DDM] = '0.1'
        os.environ[KNOB_MIN] = '7'
        self.assertEqual(self.hf._scf_adaptive_newton_params(),
                         (0.1, 7))
        # malformed / non-finite / out-of-domain -> warn + default
        for bad_d in ('abc', '-1', '0', 'nan', 'inf'):
            os.environ[KNOB_DDM] = bad_d
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = self.hf._scf_adaptive_newton_params()
            self.assertEqual(got[0], 0.0285,
                             'DDM=%r must fall back' % bad_d)
            self.assertTrue(any(KNOB_DDM in str(x.message) for x in w),
                            'DDM=%r must warn' % bad_d)
        os.environ.pop(KNOB_DDM, None)
        for bad_m in ('x', '2.5', '0', '-3'):
            os.environ[KNOB_MIN] = bad_m
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = self.hf._scf_adaptive_newton_params()
            self.assertEqual(got[1], 12,
                             'MIN_CYCLE=%r must fall back' % bad_m)
            self.assertTrue(any(KNOB_MIN in str(x.message) for x in w),
                            'MIN_CYCLE=%r must warn' % bad_m)
        os.environ.pop(KNOB_MIN, None)


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS wb97mv sentinel (the corpus method class; the
    plain-CDIIS lane is bitwise-reproducible run-to-run on this stack
    -- entries 414/428/433 pinned bitwise-identical within-cell
    replicates).  Each _run builds a fresh mol+mf; the dispatcher
    cache is force-re-read per run and every knob is restored.'''

    ATOM = [['O', (0., 0., 0.)], ['O', (0., 0., 1.2078)]]

    def _env(self, main, ddm=None, min_cycle=None):
        from gpu4pyscf.scf import hf
        hf._ADAPTIVE_NEWTON = None   # force re-read
        for key in (KNOB, KNOB_DDM, KNOB_MIN):
            os.environ.pop(key, None)
        if main is not None:
            os.environ[KNOB] = main
        if ddm is not None:
            os.environ[KNOB_DDM] = ddm
        if min_cycle is not None:
            os.environ[KNOB_MIN] = min_cycle
        return hf

    def _restore(self):
        for key in (KNOB, KNOB_DDM, KNOB_MIN):
            os.environ.pop(key, None)

    def setUp(self):
        self._restore()

    tearDown = setUp

    def _run(self, main=None, ddm=None, min_cycle=None, strict_ref=False,
             max_cycle=50):
        """One fresh plain-CDIIS kernel.  Returns (record, e_tot_repr)
        where record carries fired/fire_cycle/indicator/converged and
        the module counter/run-state snapshots."""
        from pyscf import gto
        from gpu4pyscf import dft
        hf = self._env(main, ddm, min_cycle)
        mol = gto.M(verbose=0, output='/dev/null', atom=self.ATOM,
                    basis='def2-svp', charge=0, spin=2)
        mf = dft.UKS(mol, xc='wb97mv').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 3
        mf.max_cycle = max_cycle
        if strict_ref:
            mf._strict_fp64_reference = True
        fires0 = hf._ADAPTIVE_NEWTON_FIRES
        e = mf.kernel()
        state = getattr(hf, '_ADAPTIVE_NEWTON_RUN', None)
        rec = {
            'fired': bool(state is not None and state['fired']),
            'fire_cycle': (state['fire_cycle'] if state is not None
                           else None),
            'indicator': (list(state['indicator_trace'])
                          if state is not None else None),
            'fires_delta': hf._ADAPTIVE_NEWTON_FIRES - fires0,
            'converged': bool(mf.converged),
            'cycles': int(mf.cycles),
            'mf_fired_attr': getattr(mf, '_adaptive_newton_fired_cycle',
                                     None),
        }
        self._restore()
        return rec, repr(float(e))

    def test_default_off_bitwise_run_to_run(self):
        r1, e1 = self._run()
        r2, e2 = self._run()
        self.assertEqual(e1, e2)   # run-to-run bitwise (lane determinism)
        self.assertFalse(r1['fired'])
        self.assertEqual(r1['fires_delta'], 0)
        self.assertIsNone(r1['indicator'])
        self.assertIsNone(r1['mf_fired_attr'])
        self.assertTrue(r1['converged'])
        # explicit '0' is the same verbatim lane
        r3, e3 = self._run(main='0')
        self.assertEqual(e3, e1)
        self.assertEqual(r3['fires_delta'], 0)

    def test_armed_fires_and_honest_unconverged_head(self):
        r_off, e_off = self._run()
        # huge threshold + min_cycle 1: fires at cycle 1
        r, e = self._run(main='1', ddm='1.0', min_cycle='1')
        self.assertTrue(r['fired'])
        self.assertEqual(r['fire_cycle'], 1)
        self.assertEqual(r['mf_fired_attr'], 1)
        self.assertEqual(r['fires_delta'], 1)
        self.assertEqual(r['cycles'], 1)
        # the fire is NOT tolerance convergence: honest unconverged
        # head (the entry-414/428 fixed-k head class), e at cycle 1
        # far from the converged basin energy
        self.assertFalse(r['converged'])
        self.assertNotEqual(e, e_off)
        self.assertGreater(abs(float(e) - float(e_off)), 1e-6)
        # engagement evidence recorded
        self.assertEqual(len(r['indicator']), 1)
        self.assertEqual(r['indicator'][0][0], 1)
        self.assertGreaterEqual(r['indicator'][0][1], 1.0)

    def test_armed_never_fires_bitwise_verbatim(self):
        r_off, e_off = self._run()
        # impossible threshold: runs to natural convergence, bitwise
        # the knob-off trajectory
        r, e = self._run(main='1', ddm='1e300')
        self.assertFalse(r['fired'])
        self.assertIsNone(r['fire_cycle'])
        self.assertIsNone(r['mf_fired_attr'])
        self.assertEqual(r['fires_delta'], 0)
        self.assertTrue(r['converged'])
        self.assertEqual(r['cycles'], r_off['cycles'])
        self.assertEqual(e, e_off)
        # the full indicator trajectory was still recorded
        self.assertEqual(len(r['indicator']), r['cycles'])

    def test_min_cycle_gate(self):
        # tiny positive threshold (always crossed): the gate alone
        # decides the cycle
        r, _ = self._run(main='1', ddm='1e-12', min_cycle='3')
        self.assertTrue(r['fired'])
        self.assertEqual(r['fire_cycle'], 3)
        self.assertEqual(r['cycles'], 3)
        self.assertEqual(r['fires_delta'], 1)

    def test_malformed_knob_verbatim(self):
        r_off, e_off = self._run()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            r, e = self._run(main='bogus')
        self.assertTrue(any(KNOB in str(x.message) for x in w))
        self.assertEqual(r['fires_delta'], 0)
        self.assertFalse(r['fired'])
        self.assertEqual(e, e_off)

    def test_strict_ref_never_arms(self):
        r_off, e_off = self._run()
        r, e = self._run(main='1', ddm='1.0', min_cycle='1',
                         strict_ref=True)
        self.assertFalse(r['fired'])
        self.assertIsNone(r['mf_fired_attr'])
        self.assertEqual(r['fires_delta'], 0)
        self.assertTrue(r['converged'])
        self.assertEqual(e, e_off)   # bitwise verbatim

    def test_per_run_state_reset_counter_accumulates(self):
        # two armed kernels in one process: the per-run state resets
        # (fresh indicator/fired) and the fire counter accumulates
        r1, _ = self._run(main='1', ddm='1.0', min_cycle='1')
        r2, _ = self._run(main='1', ddm='1.0', min_cycle='1')
        self.assertEqual(r1['fire_cycle'], 1)
        self.assertEqual(r2['fire_cycle'], 1)
        self.assertEqual(len(r1['indicator']), 1)
        self.assertEqual(len(r2['indicator']), 1)   # reset, not appended
        self.assertEqual(r1['fires_delta'], 1)
        self.assertEqual(r2['fires_delta'], 1)


if __name__ == '__main__':
    unittest.main()

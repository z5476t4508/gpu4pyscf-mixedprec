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

"""Regression tests for the CONV-UKS-NEWTON-INNERCUT-01 opt-in AH
tol floor (GPU4PYSCF_SOSCF_AH_TOL_FLOOR, gpu4pyscf/scf/soscf.py).

Provenance: entry 416 (CONV-UKS-NEWTON-INTERIOR-01, 2026-11-07)
decomposed the newton interior at the 573-AO ibuprofen_anion wb97m-v
doublet: vind counts [4, 6, 18, 1] per macro with per-call cost flat
to 0.3% (~3.22 s/call); the 18-call grind macro is pure davidson
CALL-COUNT mass (70.1% of t_newton), not response-build mass
(gen_response construction is 0.30 s/macro).  The keyframe AH flags
measured in the trace: ah_conv_tol = 1e-12, ah_start_tol = 1e+09,
ah_start_cycle = 1 -- so the inner davidson always runs at
tol = min(norm_gorb**2, 1e-12), i.e. it resolves the AH eigenvalue ~7
orders below the outer conv_tol_grad target near the basin.  This knob
floors that tolerance (opt-in, default OFF); the outer keyframe /
restore machinery keeps evaluating the true gradient and stays the
accuracy authority.

Contract tests (source-level, CPU):
  * exactly one env reader for the knob (home module soscf);
  * validation band (0, 1e-4]; malformed/out-of-band -> warn + ignore;
  * the floor site guards the verbatim line
    ``ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)`` and only
    rebinds when the floor strictly loosens it (counter counts real
    engagements, not knob presence);
  * strict fp64-reference lanes never arm (flag on the newton wrapper
    OR the wrapped base mf both suppress).

Numerical tests (GPU, small UKS sentinel, same lane class as entry
416 -- DF-UKS newton, deterministic per the 416 bitwise C-arm pins):
  * default OFF: run-to-run bitwise e_tot (lane determinism), zero
    engagement counter, cache resolves 0.0;
  * armed 1e-8: converges in the same basin (|dE| <= 1e-8 Ha vs the
    knob-off value), engagement counter >= 1;
  * armed malformed value: verbatim path (zero counter), bitwise e_tot
    equal to the knob-off run.
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

SOSCF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'soscf.py')
KNOB = 'GPU4PYSCF_SOSCF_AH_TOL_FLOOR'


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(SOSCF_SRC, '_ah_tol_floor')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(SOSCF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # validation band + warn-and-ignore
        self.assertIn('0.0 < val <= 1e-4', parser)
        self.assertIn('warnings.warn', parser)

    def test_counter_helper_home_module(self):
        helper = _func_src(SOSCF_SRC, '_note_ah_tol_floor_bind')
        self.assertIsNotNone(helper)
        self.assertIn('global _AH_TOL_FLOOR_BINDS', helper)

    def test_floor_site_guards_verbatim_line(self):
        rot = _func_src(SOSCF_SRC, '_rotate_orb_cc')
        self.assertIsNotNone(rot)
        self.assertIn('ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)',
                      rot)
        self.assertIn('if _ah_tol_floor_armed(mf):', rot)
        # rebinding counts only strict loosening (real engagements)
        self.assertIn('if _ah_tol_floor() > ah_eff:', rot)
        self.assertIn('_note_ah_tol_floor_bind()', rot)

    def test_strict_ref_inert_on_both_objects(self):
        armed = _func_src(SOSCF_SRC, '_ah_tol_floor_armed')
        self.assertIsNotNone(armed)
        self.assertIn("for obj in (mf, getattr(mf, '_scf', None)):", armed)
        self.assertIn("getattr(obj, '_strict_fp64_reference', False)", armed)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB), soscf._AH_TOL_FLOOR)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.soscf._AH_TOL_FLOOR = cache

    def _parse(self, raw):
        self.soscf._AH_TOL_FLOOR = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.soscf._ah_tol_floor()

    def test_values(self):
        self.assertEqual(self._parse(None), 0.0)
        self.assertEqual(self._parse('0'), 0.0)
        self.assertEqual(self._parse('1e-8'), 1e-8)
        self.assertEqual(self._parse('1e-4'), 1e-4)   # band edge included
        for bad in ('abc', '1e-2', '-1e-8', '0.0', 'nan', ''):
            if bad == '':
                # empty string == unset semantics (verbatim path)
                self.assertEqual(self._parse(''), 0.0)
                continue
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                val = self._parse(bad)
            self.assertEqual(val, 0.0, 'value %r must be ignored' % bad)
            self.assertTrue(any(KNOB in str(x.message) for x in w),
                            'value %r must warn' % bad)

    def test_armed_helper_strict_ref(self):
        class Mf:
            pass
        self._parse('1e-8')
        mf = Mf()
        self.assertTrue(self.soscf._ah_tol_floor_armed(mf))
        mf._strict_fp64_reference = True
        self.assertFalse(self.soscf._ah_tol_floor_armed(mf))
        mf2 = Mf()
        mf2._scf = Mf()
        mf2._scf._strict_fp64_reference = True
        self.assertFalse(self.soscf._ah_tol_floor_armed(mf2))
        self._parse(None)
        self.assertFalse(self.soscf._ah_tol_floor_armed(Mf()))


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS b3lyp newton sentinel (the entry-416 lane class:
    DF-UKS newton is bitwise-reproducible there -- 3 formal C arms +
    warmup all pinned e_tot bitwise across fresh processes).  Each _run
    builds a fresh mol+mf (fresh-arm composition, no cross-run state).'''

    def _run(self, raw):
        """Head = 2 CDIIS cycles, then public newton() under the knob
        value `raw` (None = unset).  Returns (e_tot_repr, counter_delta,
        vind_calls, converged)."""
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.scf import soscf
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        soscf._AH_TOL_FLOOR = None   # force re-read
        counter0 = soscf._AH_TOL_FLOOR_BINDS
        mol = gto.M(verbose=0, output='/dev/null',
                    atom=[['O', (0., 0., 0.)],
                          ['O', (0., 0., 1.2078)]],
                    basis='def2-svp', charge=0, spin=2)
        try:
            mf = dft.UKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.max_cycle = 2
            mf.conv_check = False
            mf.kernel()
            mfn = mf.newton()
            mfn.max_cycle = 20
            mfn.conv_tol_grad = 1e-5
            vind_calls = [0]
            orig_gen_response = mfn.gen_response

            def gen_response_wrap(*a, **kw):
                vind = orig_gen_response(*a, **kw)

                def vind_wrap(dm1):
                    vind_calls[0] += 1
                    return vind(dm1)
                return vind_wrap
            mfn.gen_response = gen_response_wrap
            try:
                e = mfn.kernel()
            finally:
                del mfn.gen_response
            return (repr(float(e)),
                    soscf._AH_TOL_FLOOR_BINDS - counter0,
                    vind_calls[0],
                    bool(getattr(mfn, 'converged', False)))
        finally:
            mol.stdout.close()

    def setUp(self):
        from gpu4pyscf.scf import soscf
        self._saved = (os.environ.get(KNOB), soscf._AH_TOL_FLOOR,
                       soscf._AH_TOL_FLOOR_BINDS)

    def tearDown(self):
        from gpu4pyscf.scf import soscf
        raw, cache, binds = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        soscf._AH_TOL_FLOOR = cache
        soscf._AH_TOL_FLOOR_BINDS = binds

    def test_default_off_bitwise_and_armed(self):
        # (1) default OFF: bitwise run-to-run, zero engagements
        e1, d1, v1, c1 = self._run(None)
        self.assertTrue(c1, 'knob-off newton must converge')
        self.assertEqual(d1, 0, 'knob OFF must not engage')
        e2, d2, v2, c2 = self._run(None)
        self.assertTrue(c2)
        self.assertEqual(d2, 0)
        self.assertEqual(e1, e2, 'knob-off lane must be bitwise '
                         'run-to-run (416 determinism class)')
        # (2) armed 1e-8: converges in the same basin within 1e-8 Ha,
        # engages >= 1 keyframe
        e3, d3, v3, c3 = self._run('1e-8')
        self.assertTrue(c3, 'armed newton must converge under cap')
        self.assertGreaterEqual(d3, 1, 'armed knob must engage the floor')
        self.assertLessEqual(abs(float(e3) - float(e1)), 1e-8,
                             'armed newton must stay in the same basin')
        # the floor cannot explode davidson applications at this
        # sentinel (the 573 cell adjudicates the count cut)
        self.assertLessEqual(v3, v1 + 2,
                             'armed vind calls exploded: %d vs %d'
                             % (v3, v1))
        # (3) malformed value: verbatim path, bitwise with (1)
        e4, d4, v4, c4 = self._run('not-a-float')
        self.assertTrue(c4)
        self.assertEqual(d4, 0, 'malformed value must be ignored')
        self.assertEqual(e4, e1, 'malformed value must be bitwise '
                         'verbatim')
        self.assertEqual(v4, v1)


if __name__ == '__main__':
    print('Regression tests for the SOSCF AH tol floor knob')
    unittest.main()

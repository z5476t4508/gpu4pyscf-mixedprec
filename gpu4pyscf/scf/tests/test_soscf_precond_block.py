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

"""Regression tests for the CONV-UKS-NEWTON-PRECOND-01 opt-in
block/Schur AH-davidson preconditioner (GPU4PYSCF_SOSCF_PRECOND,
gpu4pyscf/scf/soscf.py).

Provenance: entry 416 (CONV-UKS-NEWTON-INTERIOR-01, 2026-11-07)
decomposed the newton interior at the 573-AO ibuprofen_anion wb97m-v
doublet: vind counts [4, 6, 18, 1] per macro with per-call cost flat
(~3.22 s/call); the 18-call grind macro is pure davidson CALL-COUNT
mass at near-degenerate rotations, the class the deployed DIAGONAL
preconditioner under-resolves.  Entry 417 (INNERCUT) then closed the
TRUNCATION axis NEGATIVE: relaxing the inner tolerance collapses the
grind macro but stalls the quadratic endgame (tolerance change alters
the acceptance event).  This knob attacks the same count mass by a
different mechanism -- a block/Schur (Sylvester-class) solve of the
diagonal-block AH problem from the per-spin MO fock blocks
(fvva/fooa, exactly as gen_g_hop_uhf derives them) -- at UNCHANGED
tolerance, so the acceptance event is untouched by construction.

Contract tests (source-level):
  * exactly one env reader for the knob (home module soscf);
  * accepted value exactly 'block'; unset/0/off keep the verbatim
    diagonal path; anything else warns + is ignored;
  * the precond site keeps the verbatim diagonal solve lines and the
    verbatim AH tol line (tol UNCHANGED contract); the block path is
    guarded by a None-check on block_pc (OFF path instruction-
    identical) and the builder only runs behind _precond_block_armed;
  * the builder warns + returns None for non-UHF layouts (mo_occ.ndim
    != 2);
  * strict fp64-reference lanes never arm (flag on the newton wrapper
    OR the wrapped base mf both suppress).

Math tests (GPU):
  * _apply_block_precond with identity Schur factors reduces exactly
    to the deployed diagonal solve x/(h_diag - e_eff) including the
    1e-8 lindep clamp (block generalization consistency);

Numerical tests (GPU, small UKS sentinel, same lane class as entry
416 -- DF-UKS newton bitwise-reproducible there):
  * default OFF: run-to-run bitwise e_tot (lane determinism), zero
    engagement counter, cache resolves off;
  * armed 'block': converges in the same basin (|dE| <= 1e-8 Ha vs
    the knob-off value), engagement counter >= 1, vind calls not
    exploding;
  * armed malformed value: verbatim path (zero counter), bitwise
    e_tot equal to the knob-off run;
  * armed + strict fp64-reference flag on the newton wrapper:
    verbatim path, bitwise e_tot, zero engagements.
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
KNOB = 'GPU4PYSCF_SOSCF_PRECOND'


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(SOSCF_SRC, '_soscf_precond')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(SOSCF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # accepted value + warn-and-ignore
        self.assertIn("_PRECOND_BLOCK = 'block' if raw == 'block' else False",
                      parser)
        self.assertIn('warnings.warn', parser)

    def test_counter_helper_home_module(self):
        helper = _func_src(SOSCF_SRC, '_note_precond_block_engaged')
        self.assertIsNotNone(helper)
        self.assertIn('global _PRECOND_BLOCK_BUILDS', helper)

    def test_precond_site_guards_verbatim_lines(self):
        rot = _func_src(SOSCF_SRC, '_rotate_orb_cc')
        self.assertIsNotNone(rot)
        # the deployed diagonal solve stays verbatim
        self.assertIn('hdiagd = h_diag-(e-mf.ah_level_shift)', rot)
        self.assertIn('hdiagd[abs(hdiagd)<1e-8] = 1e-8', rot)
        # the verbatim AH tol line stays (tol UNCHANGED contract)
        self.assertIn('ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)',
                      rot)
        # the block path is None-guarded (OFF path instruction-identical)
        self.assertIn('if block_pc is not None:', rot)
        self.assertIn('_apply_block_precond(block_pc, x, e,', rot)
        # the builder only runs behind the armed check + real engagements
        # counted
        self.assertIn('if _precond_block_armed(mf):', rot)
        self.assertIn('_build_block_precond_uhf(mf, mo_coeff, mo_occ,', rot)
        self.assertIn('_note_precond_block_engaged()', rot)

    def test_strict_ref_inert_on_both_objects(self):
        armed = _func_src(SOSCF_SRC, '_precond_block_armed')
        self.assertIsNotNone(armed)
        self.assertIn("for obj in (mf, getattr(mf, '_scf', None)):", armed)
        self.assertIn("getattr(obj, '_strict_fp64_reference', False)", armed)

    def test_builder_layout_guard(self):
        builder = _func_src(SOSCF_SRC, '_build_block_precond_uhf')
        self.assertIsNotNone(builder)
        self.assertIn("getattr(mo_occ, 'ndim', 1) != 2", builder)
        # runtime guard: non-UHF layout warns and returns None before
        # touching cupy (mf/mo_coeff/fock_ao unused on that path)
        import numpy
        from gpu4pyscf.scf import soscf
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            out = soscf._build_block_precond_uhf(
                None, None, numpy.zeros(5), None, None)
        self.assertIsNone(out)
        self.assertTrue(any(KNOB in str(x.message) for x in w),
                        'non-UHF layout must warn')

    def test_apply_clamp_contract(self):
        apply_src = _func_src(SOSCF_SRC, '_apply_block_precond')
        self.assertIsNotNone(apply_src)
        # the deployed lindep clamp semantics on the denominator
        self.assertIn('denom[abs(denom) < 1e-8] = 1e-8', apply_src)
        self.assertIn('e_eff = e - level_shift', apply_src)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB), soscf._PRECOND_BLOCK)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.soscf._PRECOND_BLOCK = cache

    def _parse(self, raw):
        self.soscf._PRECOND_BLOCK = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.soscf._soscf_precond()

    def test_values(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('off'), False)
        self.assertEqual(self._parse('block'), 'block')
        for bad in ('diag', 'Block', 'blocks', 'jacobi', 'schur', '1'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                val = self._parse(bad)
            self.assertIs(val, False, 'value %r must be ignored' % bad)
            self.assertTrue(any(KNOB in str(x.message) for x in w),
                            'value %r must warn' % bad)

    def test_armed_helper_strict_ref(self):
        class Mf:
            pass
        self._parse('block')
        mf = Mf()
        self.assertTrue(self.soscf._precond_block_armed(mf))
        mf._strict_fp64_reference = True
        self.assertFalse(self.soscf._precond_block_armed(mf))
        mf2 = Mf()
        mf2._scf = Mf()
        mf2._scf._strict_fp64_reference = True
        self.assertFalse(self.soscf._precond_block_armed(mf2))
        self._parse(None)
        self.assertFalse(self.soscf._precond_block_armed(Mf()))


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestApplyMath(unittest.TestCase):
    def test_identity_factors_reduce_to_diagonal(self):
        """With identity Schur factors (diagonal fock blocks) the block
        solve must equal the deployed diagonal solve elementwise,
        including the 1e-8 clamp."""
        import cupy as cp
        import numpy
        from gpu4pyscf.scf import soscf
        numpy.random.seed(20261107)
        nvir, nocc = 7, 5
        lv = cp.asarray(numpy.random.uniform(0.1, 2.0, nvir))
        lo = cp.asarray(numpy.random.uniform(-2.0, -0.1, nocc))
        x = cp.asarray(numpy.random.uniform(-1.0, 1.0, nvir * nocc))
        e = 0.37
        shift = 1e-8
        blocks = [(cp.eye(nvir), lv, cp.eye(nocc), lo, nvir, nocc, 0)]
        got = soscf._apply_block_precond(blocks, x, e, shift)
        # deployed diagonal reference: x / (hdiag - (e - shift)), clamped
        hdiag = (lv[:, None] - lo[None, :]).ravel()
        hdiagd = hdiag - (e - shift)
        hdiagd[abs(hdiagd) < 1e-8] = 1e-8
        want = x / hdiagd
        cp.testing.assert_allclose(got, want, rtol=0, atol=0)
        # clamp branch really exercised by a near-degenerate pair
        lv2 = cp.asarray(numpy.array([1.0, 1.0 + 5e-9]))
        lo2 = cp.asarray(numpy.array([0.0]))
        blocks2 = [(cp.eye(2), lv2, cp.eye(1), lo2, 2, 1, 0)]
        e2 = 1.0   # denominator ~ -5e-9 -> clamped to 1e-8
        got2 = soscf._apply_block_precond(blocks2, cp.ones(2), e2, 0.0)
        denom = cp.asarray(numpy.array([1.0 - 1.0, 1.0 + 5e-9 - 1.0]))
        denom[abs(denom) < 1e-8] = 1e-8
        cp.testing.assert_allclose(got2, 1.0 / denom, rtol=0, atol=0)


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS b3lyp newton sentinel (the entry-416 lane class:
    DF-UKS newton is bitwise-reproducible there -- 3 formal C arms +
    warmup all pinned e_tot bitwise across fresh processes).  Each _run
    builds a fresh mol+mf (fresh-arm composition, no cross-run
    state).'''

    def _run(self, raw, strict_ref=False):
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
        soscf._PRECOND_BLOCK = None   # force re-read
        counter0 = soscf._PRECOND_BLOCK_BUILDS
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
            if strict_ref:
                mfn._strict_fp64_reference = True
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
                    soscf._PRECOND_BLOCK_BUILDS - counter0,
                    vind_calls[0],
                    bool(getattr(mfn, 'converged', False)))
        finally:
            mol.stdout.close()

    def setUp(self):
        from gpu4pyscf.scf import soscf
        self._saved = (os.environ.get(KNOB), soscf._PRECOND_BLOCK,
                       soscf._PRECOND_BLOCK_BUILDS)

    def tearDown(self):
        from gpu4pyscf.scf import soscf
        raw, cache, builds = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        soscf._PRECOND_BLOCK = cache
        soscf._PRECOND_BLOCK_BUILDS = builds

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
        # (2) armed 'block': converges in the same basin within 1e-8 Ha,
        # engages >= 1 keyframe; tol UNCHANGED so the accepted step is
        # solved to the same accuracy in (expected fewer) expansions
        e3, d3, v3, c3 = self._run('block')
        self.assertTrue(c3, 'armed newton must converge under cap')
        self.assertGreaterEqual(d3, 1, 'armed knob must build the block '
                                'precond at >= 1 keyframe')
        self.assertLessEqual(abs(float(e3) - float(e1)), 1e-8,
                             'armed newton must stay in the same basin')
        # the block precond cannot explode davidson applications at this
        # sentinel (the 573 cell adjudicates the count cut)
        self.assertLessEqual(v3, v1 + 2,
                             'armed vind calls exploded: %d vs %d'
                             % (v3, v1))
        # (3) malformed value: verbatim path, bitwise with (1)
        e4, d4, v4, c4 = self._run('jacobi')
        self.assertTrue(c4)
        self.assertEqual(d4, 0, 'malformed value must be ignored')
        self.assertEqual(e4, e1, 'malformed value must be bitwise '
                         'verbatim')
        self.assertEqual(v4, v1)
        # (4) armed + strict fp64-reference flag: verbatim + bitwise
        e5, d5, v5, c5 = self._run('block', strict_ref=True)
        self.assertTrue(c5)
        self.assertEqual(d5, 0, 'strict fp64-reference lanes never arm')
        self.assertEqual(e5, e1, 'strict-ref armed run must be bitwise '
                         'verbatim')
        self.assertEqual(v5, v1)


if __name__ == '__main__':
    print('Regression tests for the SOSCF block preconditioner knob')
    unittest.main()

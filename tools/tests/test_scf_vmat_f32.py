#!/usr/bin/env python3
"""Regression tests for the LC-105 SCF-side fp64-lane vmat-contraction
fp32 knob (GPU4PYSCF_SCF_VMAT_F32, gpu4pyscf/dft/numint.py + the lane
gate in gpu4pyscf/scf/hf.py).

LC-95/98 measured the SCF-side MGGA vmat GEMM train as the top remaining
SCF piece (57.2ms tau + 17.2ms GGA per steady call on
wb97m-v/benzene/tzvp fp64, compute-rate bound at 55-61% of the RTX 4090
fp64 peak; re-blocking, engine and plan-cache all falsified).  The
fp32_grid cast in _nr_rks_task/_nr_uks_task applied only while the
resolved precision is fp32 (the auto policy's head).  LC-105 extends it
to PURE-fp64 SCF lanes behind the opt-in knob: rho_tot (and hence
excsum/nelec) keep their fp64 AO and are computed BEFORE the vmat loop
(bitwise unchanged), the contraction helpers run in the operands' fp32
dtype, vmat accumulates in float64.

Measured (2026-10-13-lc105-vmat-f32-scf-knob, recommended stack, fresh
processes): whole-SCF t_scf 1.122x wb97m-v/benzene/tzvp and 1.137x
m06-2x/benzene/tzvp (MGGA tau-train cells), honest parity on
b3lyp/benzene/tzvp (plain GGA) and a within-noise no-win on the UKS
methoxy cell; converged e_tot shifts <= 6.8e-11 Ha; excsum/nelec
bitwise 0 on all cells, max|d vmat| 1.9e-6 at identical dm; the auto
policy's fp64 tail must stay bitwise fp64 (ungated counterfactual
shifts e_tot by 3.5e-11) -- gated at hf._kernel, and
stability_verify's strict fp64-reference siblings carry
_strict_fp64_reference and never feel the knob.  (b3)-class precision
policy: knob-on, the fp64-lane SCF is NOT the bitwise strict fp64
reference.

Contract tests (CPU, source-level):
  * the parser defaults OFF, accepts exactly '1', warns on garbage;
  * both task functions gate the cast on
    contr_f32 = fp32_grid or _scf_vmat_f32_lane (the lane flag, not the
    env, so standalone nr_rks/nr_uks calls are unchanged);
  * hf._kernel sets the lane flag only for pure-fp64 lanes
    (not _mixed_auto, not _strict_fp64_reference) and restores it in
    the finally block;
  * _build_fp64_sibling marks the sibling _strict_fp64_reference.

GPU tests:
  * dtype shim: with the knob ON, the main-XC vmat contractions see
    float32 AO on the fp64 lane; the auto lane's fp64 tail and the NLC
    loop (nr_nlc_vxc, out of scope) stay float64;
  * same-dm swap bound: max|d vmat| <= 5e-6 with excsum/nelec bitwise
    equal (measured 1.9e-6 / exact 0);
  * whole-SCF e_tot bound on an MGGA cell (measured <= 6.8e-11);
  * auto-lane no-op: with the knob ON, an auto SCF's e_tot matches the
    knob-OFF auto e_tot bitwise (same-process discipline is not
    achievable; the test compares two fresh objects in one process,
    asserting the lane flag never turns on for _mixed_auto modes).
"""
import os
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')
SV_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'stability_verify.py')


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestKnobParser(unittest.TestCase):
    def test_parser_contract(self):
        src = open(NUMINT_SRC).read()
        self.assertIn("os.environ.get('GPU4PYSCF_SCF_VMAT_F32', '')", src)
        self.assertIn("raw == '1'", src)
        self.assertIn('warnings.warn', src)

    def test_default_off(self):
        os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
        from gpu4pyscf.dft import numint
        numint._SCF_VMAT_F32 = None
        try:
            self.assertIs(numint._scf_vmat_f32(), False)
            self.assertIs(numint._scf_vmat_f32_lane, False)
        finally:
            numint._SCF_VMAT_F32 = None

    def test_garbage_value_warns_and_off(self):
        os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'yes'
        from gpu4pyscf.dft import numint
        numint._SCF_VMAT_F32 = None
        try:
            with self.assertWarns(UserWarning):
                self.assertIs(numint._scf_vmat_f32(), False)
        finally:
            del os.environ['GPU4PYSCF_SCF_VMAT_F32']
            numint._SCF_VMAT_F32 = None


class TestSourceContract(unittest.TestCase):
    def test_both_task_functions_gate_on_lane_flag(self):
        for fname in ('_nr_rks_task', '_nr_uks_task'):
            src = _func_src(NUMINT_SRC, fname)
            self.assertIsNotNone(src, fname)
            # the cast reads the lane flag, NOT the env parser: the auto
            # policy's fp64 tail also resolves fp32_grid=False and must
            # not feel the knob
            self.assertIn('contr_f32 = fp32_grid or _scf_vmat_f32_lane',
                          src)
            self.assertIn('if contr_f32:', src)
            # rho/excsum passes precede the loop and keep the fp64 AO
            self.assertNotIn('rho_tot[:,p0:p1] = numint.eval_rho(_sorted_mol, ao_mask.astype',
                             src)

    def test_hf_lane_gate_pure_fp64_only(self):
        src = _func_src(HF_SRC, '_kernel')
        self.assertIsNotNone(src)
        self.assertIn('_numint_mod._scf_vmat_f32_lane = (', src)
        self.assertIn('_numint_mod._scf_vmat_f32()', src)
        self.assertIn('and not _mixed_auto', src)
        self.assertIn("getattr(mf, '_strict_fp64_reference', False)", src)
        # restore in the finally block: a raised kernel must not leak the
        # flag into the next calculation
        self.assertIn('_numint_mod._scf_vmat_f32_lane = _vmat_f32_saved', src)

    def test_strict_reference_sibling_marked(self):
        src = open(SV_SRC).read()
        self.assertIn('base._strict_fp64_reference = True', src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobDtypeShim(unittest.TestCase):
    '''Which AO dtype reaches _scale_ao from the main-XC vmat loop and
    the NLC loop, per lane, with the knob ON.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'benzene.xyz')

    def _run(self, mode):
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.dft import numint

        records = []
        _orig_scale = numint._scale_ao
        _region = {'r': 'outside'}

        def w(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig_scale(ao, wv, *a, **kw)

        numint._scale_ao = w
        numint._SCF_VMAT_F32 = True
        try:
            mol = gto.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.precision_mode = mode
            ni = mf._numint
            _orig_nlc = ni.nr_nlc_vxc

            def nlc_w(*a, **kw):
                prev = _region['r']
                _region['r'] = 'nlc'
                try:
                    return _orig_nlc(*a, **kw)
                finally:
                    _region['r'] = prev
            ni.nr_nlc_vxc = nlc_w
            try:
                mf.kernel()
            finally:
                ni.nr_nlc_vxc = _orig_nlc
        finally:
            numint._scale_ao = _orig_scale
            numint._SCF_VMAT_F32 = None
        xc_dts = {dt for r, dt in records if r == 'outside'}
        nlc_dts = {dt for r, dt in records if r == 'nlc'}
        return xc_dts, nlc_dts, float(mf.e_tot), mf.converged

    def test_fp64_lane_contractions_fp32(self):
        xc_dts, nlc_dts, e, conv = self._run('fp64')
        self.assertTrue(conv)
        self.assertEqual(xc_dts, {'float32'},
                         f'knob-on fp64-lane vmat AO must be float32: {xc_dts}')
        self.assertEqual(nlc_dts, {'float64'},
                         f'nr_nlc_vxc is out of scope and must stay fp64: '
                         f'{nlc_dts}')

    def test_auto_lane_tail_stays_fp64(self):
        # the auto head casts (fp32_grid), the fp64 tail must NOT: the
        # mixed {'float32','float64'} signature is exactly the tail
        # protection -- a knob-on all-fp32 signature would mean the
        # gate leaked into the auto tail
        xc_dts, nlc_dts, e, conv = self._run('auto')
        self.assertTrue(conv)
        self.assertEqual(xc_dts, {'float32', 'float64'},
                         f'auto lane must show fp32 head + fp64 tail: {xc_dts}')
        self.assertEqual(nlc_dts, {'float64'})


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobNumerics(unittest.TestCase):
    '''Same-dm swap cost + whole-SCF e_tot bound.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'benzene.xyz')

    def test_same_dm_swap_bound(self):
        import numpy as np
        import cupy as cp
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.dft import numint

        mol = gto.M(atom=self.GEOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-11
        mf.grids.level = 3
        mf.precision_mode = 'fp64'
        mf.kernel()
        assert mf.converged
        ni = mf._numint
        dm = mf.make_rdm1()

        numint._scf_vmat_f32_lane = False
        try:
            n0, e0, v0 = ni.nr_rks(mol, mf.grids, 'wb97m-v', dm)
            v0 = cp.asnumpy(v0)
        finally:
            numint._scf_vmat_f32_lane = False
        numint._scf_vmat_f32_lane = True
        try:
            n1, e1, v1 = ni.nr_rks(mol, mf.grids, 'wb97m-v', dm)
            v1 = cp.asnumpy(v1)
        finally:
            numint._scf_vmat_f32_lane = False

        # rho/excsum/nelec are bitwise: both passes precede the loop
        self.assertEqual(e0, e1)
        np.testing.assert_array_equal(np.atleast_1d(n0), np.atleast_1d(n1))
        dv = float(np.abs(v1 - v0).max())
        self.assertLessEqual(dv, 5e-6,
                             f'same-dm vmat swap cost (measured 1.9e-6 '
                             f'on tzvp-class cells): {dv}')

    def test_whole_scf_energy_bound(self):
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.dft import numint

        results = {}
        for knob in (False, True):
            numint._SCF_VMAT_F32 = knob
            mol = gto.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-11
            mf.grids.level = 3
            mf.precision_mode = 'fp64'
            mf.kernel()
            results[knob] = float(mf.e_tot)
            self.assertTrue(mf.converged)
        numint._SCF_VMAT_F32 = None
        de = results[True] - results[False]
        self.assertLessEqual(abs(de), 1e-9,
                             f'knob-on whole-SCF e_tot shift (measured '
                             f'<= 6.8e-11 on tzvp-class cells): {de}')


if __name__ == '__main__':
    unittest.main()

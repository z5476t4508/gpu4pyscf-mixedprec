#!/usr/bin/env python3
"""Regression tests for the LC-101 fp64-lane XC grid-contraction fp32
knob (GPU4PYSCF_GRAD_GRID_F32, gpu4pyscf/grad/rks.py).

LC-87 measured the last open >=0.05s fp64-gradient-lane lever: the XC
grid contractions run fp64 on the fp64 lane while the auto lane runs the
identical shapes fp32 (_tau_grad_dot_ 0.133s vs 0.018s,
_gga_grad_sum_ 0.090s vs 0.019s, nlc _gga_grad_sum_ 0.089s vs 0.012s at
wb97mv/benzene/tzvp@16384).  LC-101 adds the opt-in knob: with
GPU4PYSCF_GRAD_GRID_F32=1 the fp64-lane gradient casts the block AO to
float32 for the contraction calls only (LC-88's landed design: rho keeps
its fp64 AO and buffers -- bitwise-identical density -- and the
contraction helpers run in the operands' fp32 dtype with dtype-sized
buffers; dm_mask einsum and the exc1 accumulator stay float64).

Measured (2026-10-13-lc101-grad-grid-f32, same-orbital A/B, recommended
stack): t_grad 1.56x wb97mv/benzene/tzvp, 1.43x m062x, 2.47x
r2scan3c/vitaminC/mtzvpp (298 AO), 1.16x at default blocksize; honest
0.805x on b3lyp (plain GGA: no tau/NLC GEMM win to pay for the per-block
AO cast); max|dg| swap cost 1.3e-7..2.3e-6 Eh/Bohr.  (b3)-class
precision policy: knob-on the fp64 lane is NOT the bitwise strict fp64
reference.

Contract tests (CPU, source-level):
  * the parser defaults OFF, accepts exactly '1', warns on garbage;
  * _get_exc_task computes contr_f32 = fp32_grid or knob and casts ONLY
    the contraction AO (rho keeps fp32_grid-only suppression) in all
    three xctype branches;
  * get_nlc_exc's LC-88 cast fires on fp32_grid or the knob, gated to
    the RKS path (nset == 1);
  * the UKS _get_exc_task surface carries NO knob (LC-89 closed it:
    slower on all four measured configs).

Numerical tests (GPU):
  * dtype shim: with the knob ON under a fp64 resolved gradient, the
    contraction calls see float32 AO from both the main-XC and the NLC
    loops; with the knob OFF they stay float64;
  * same-orbital bound: knob on vs off at one converged fp64 SCF,
    max|dg| <= 5e-6 (measured 4.5e-7 on benzene/tzvp-class cells);
  * knob OFF: fp64-lane repeats stay at the atomic nondeterminism floor
    (the default path is byte-equivalent to pre-change).
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

RKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'rks.py')
UKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'uks.py')


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestKnobParser(unittest.TestCase):
    def test_parser_contract(self):
        src = open(RKS_SRC).read()
        self.assertIn("os.environ.get('GPU4PYSCF_GRAD_GRID_F32', '')", src)
        self.assertIn("raw == '1'", src)
        self.assertIn('warnings.warn', src)

    def test_default_off(self):
        os.environ.pop('GPU4PYSCF_GRAD_GRID_F32', None)
        from gpu4pyscf.grad import rks as rks_grad_mod
        rks_grad_mod._GRAD_GRID_F32 = None   # reset the module cache
        try:
            self.assertIs(rks_grad_mod._grad_grid_f32(), False)
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None

    def test_garbage_value_warns_and_off(self):
        os.environ['GPU4PYSCF_GRAD_GRID_F32'] = 'yes'
        from gpu4pyscf.grad import rks as rks_grad_mod
        rks_grad_mod._GRAD_GRID_F32 = None
        try:
            with self.assertWarns(UserWarning):
                self.assertIs(rks_grad_mod._grad_grid_f32(), False)
        finally:
            del os.environ['GPU4PYSCF_GRAD_GRID_F32']
            rks_grad_mod._GRAD_GRID_F32 = None


class TestSourceContract(unittest.TestCase):
    def test_get_exc_task_contraction_gate(self):
        src = _func_src(RKS_SRC, '_get_exc_task')
        self.assertIsNotNone(src)
        self.assertIn('contr_f32 = fp32_grid or _grad_grid_f32()', src)
        # rho keeps the fp32-mode-only buffer suppression: the density
        # must be bitwise identical on the knob-on fp64 lane
        self.assertIn('_rbuf = (lambda b: None) if fp32_grid else (lambda b: b)',
                      src)
        # all three xctype branches cast the contraction AO via the
        # same single gate
        self.assertEqual(src.count('ao_contr = ao_mask.astype(cupy.float32)'), 3,
                         'one contraction-AO cast per xctype branch')
        self.assertEqual(src.count('if contr_f32 and not fp32_grid:'), 3)

    def test_nlc_loop_knob_gate(self):
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        self.assertIsNotNone(src)
        self.assertIn('contr_f32 = (fp32_grid or _grad_grid_f32()) and nset == 1',
                      src)
        self.assertEqual(src.count('ao_mask.astype(cupy.float32)'), 1,
                         'exactly one AO cast in the NLC gradient loop')

    def test_nlc_loop_uks_path_protected(self):
        # the knob fires only on the RKS NLC path (nset == 1); the UKS
        # NLC path (nset == 2) keeps the LC-88 fp32-mode-only cast
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        self.assertIn('nset == 1', src)

    def test_uks_surface_has_no_knob(self):
        # LC-89 measured the UKS _get_exc_task fp32 alignment SLOWER on
        # all four configs: the UKS surface must not grow a knob cast
        uks_src = open(UKS_SRC).read()
        self.assertNotIn('_grad_grid_f32', uks_src)
        task = _func_src(UKS_SRC, '_get_exc_task')
        self.assertNotIn('astype(cupy.float32)', task)

    def test_accumulator_dtype_unchanged(self):
        src = _func_src(RKS_SRC, '_get_exc_task')
        self.assertIn("exc1_ao = cupy.zeros((nao,3))", src)
        self.assertIn(
            "exc1_ao[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T",
            src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobDtypeShim(unittest.TestCase):
    '''dtype shim: which AO dtype reaches _gga_grad_sum_/_tau_grad_dot_
    from the main-XC and NLC loops, per knob state on the fp64 lane.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'methanol.xyz')

    def _run(self, knob_on):
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        records = []
        _orig_gga = rks_grad_mod._gga_grad_sum_
        _orig_tau = rks_grad_mod._tau_grad_dot_
        _orig_nlc = rks_grad_mod.get_nlc_exc
        _region = {'r': 'outside'}

        def rec(name, fn):
            def w(ao, wv, *a, **kw):
                records.append((name, _region['r'], str(ao.dtype)))
                return fn(ao, wv, *a, **kw)
            return w

        def region_wrap(region, orig):
            def w(*a, **kw):
                prev = _region['r']
                _region['r'] = region
                try:
                    return orig(*a, **kw)
                finally:
                    _region['r'] = prev
            return w

        rks_grad_mod._gga_grad_sum_ = rec('gga', _orig_gga)
        rks_grad_mod._tau_grad_dot_ = rec('tau', _orig_tau)
        rks_grad_mod.get_nlc_exc = region_wrap('nlc', _orig_nlc)
        rks_grad_mod.get_exc = region_wrap('xc', rks_grad_mod.get_exc)
        rks_grad_mod._GRAD_GRID_F32 = knob_on
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method()
            g.precision_mode = 'fp64'
            g.kernel()
        finally:
            rks_grad_mod._gga_grad_sum_ = _orig_gga
            rks_grad_mod._tau_grad_dot_ = _orig_tau
            rks_grad_mod.get_nlc_exc = _orig_nlc
            rks_grad_mod._GRAD_GRID_F32 = None
        return records

    def test_knob_on_contractions_are_fp32(self):
        records = self._run(True)
        xc_dts = {dt for name, region, dt in records
                  if region == 'xc' and name in ('gga', 'tau')}
        self.assertTrue(xc_dts, 'the main-XC loop must call the helpers')
        self.assertEqual(xc_dts, {'float32'},
                         f'knob-on fp64-lane XC AO must be float32: {xc_dts}')
        nlc_dts = {dt for name, region, dt in records
                   if region == 'nlc' and name == 'gga'}
        self.assertTrue(nlc_dts, 'the NLC loop must call _gga_grad_sum_')
        self.assertEqual(nlc_dts, {'float32'},
                         f'knob-on fp64-lane NLC AO must be float32: {nlc_dts}')

    def test_knob_off_contractions_are_fp64(self):
        records = self._run(False)
        xc_dts = {dt for name, region, dt in records
                  if region == 'xc' and name in ('gga', 'tau')}
        self.assertTrue(xc_dts)
        self.assertEqual(xc_dts, {'float64'},
                         f'knob-off fp64-lane XC AO must stay float64: {xc_dts}')
        nlc_dts = {dt for name, region, dt in records
                   if region == 'nlc' and name == 'gga'}
        self.assertTrue(nlc_dts)
        self.assertEqual(nlc_dts, {'float64'},
                         f'knob-off fp64-lane NLC AO must stay float64: {nlc_dts}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobNumerics(unittest.TestCase):
    '''Same-orbital A/B: the knob swap alone at one converged fp64 SCF.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'benzene.xyz')

    def test_knob_swap_same_orbital_bound(self):
        import numpy as np
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-11
        mf.grids.level = 3
        mf.kernel()
        assert mf.converged
        g = mf.nuc_grad_method()
        g.precision_mode = 'fp64'
        g.kernel()   # warm

        rks_grad_mod._GRAD_GRID_F32 = False
        try:
            g_off = np.asarray(g.kernel()).copy()
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None
        rks_grad_mod._GRAD_GRID_F32 = True
        try:
            g_on = np.asarray(g.kernel()).copy()
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None

        dg = np.abs(g_on - g_off).max()
        # measured 1.3e-7..2.3e-6 across the LC-101 probe cells; the
        # bound is the conservative family band
        self.assertLessEqual(dg, 5e-6,
                             'the grid-contraction fp32 swap must stay '
                             'far below the gradient gate')

    def test_knob_off_fp64_lane_at_floor(self):
        # default OFF: fp64-lane repeats must stay at the atomic
        # nondeterminism floor (the default path is byte-equivalent to
        # pre-change; measured 1.4e-14 on benzene/tzvp)
        import numpy as np
        import pyscf
        from gpu4pyscf.dft import RKS

        mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-11
        mf.grids.level = 3
        mf.kernel()
        g = mf.nuc_grad_method()
        g.precision_mode = 'fp64'
        g0 = np.asarray(g.kernel()).copy()
        g1 = np.asarray(g.kernel()).copy()
        self.assertLessEqual(np.abs(g1 - g0).max(), 1e-13,
                             'knob-off fp64-lane repeats must stay at the '
                             'atomic nondeterminism floor')


if __name__ == '__main__':
    unittest.main()

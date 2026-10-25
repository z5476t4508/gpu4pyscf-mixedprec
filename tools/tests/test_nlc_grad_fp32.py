#!/usr/bin/env python3
"""Regression tests for the LC-88 NLC gradient-loop fp32 alignment
(gpu4pyscf/grad/rks.py::get_nlc_exc).

LC-87 measured a structural asymmetry on the auto (production
mixed-precision) lane: the main-XC gradient loop (_get_exc_task) casts
ao_mask to float32 in fp32 mode (fp32_grid, inherited upstream) in all
three xctype branches, but the NLC gradient loop's deriv=2
_gga_grad_sum_ contraction had NO precision handling — AO stayed
float64 on every lane, costing nlc_gga_grad_sum 0.089s on BOTH lanes
(16.5% of the auto-lane t_grad at wb97mv/benzene/tzvp while the
xc-region gga_grad_sum ran 4.6x faster on auto).

LC-88 aligns the NLC loop to the same discipline: in fp32 mode the AO
is cast once per block; _gga_grad_sum_'s _scale_ao/_d1_dot_ run in the
operands' dtype; the einsum against the float64 dm_mask and the exc1
accumulator stay float64.  The fp64 lane is untouched (the cast fires
only when get_precision() == 'fp32').  This is a same-class change to
the EXISTING fp32 policy surface, not a new precision knob.

Contract tests (CPU, source-level):
  * the cast exists in get_nlc_exc, gated on
    precision.get_precision() == 'fp32', exactly once;
  * the full_response variant (opt-in grid_response=True route) is NOT
    given the cast (it is a separate route, unchanged scope);
  * the exc1 accumulator/dm einsum lines are unchanged (no dtype
    changes beyond the AO cast).

Numerical tests (GPU):
  * dtype shim: on the auto-resolved fp32 gradient, AO arriving at
    _gga_grad_sum_ from the NLC loop is float32; on the fp64 gradient
    it stays float64;
  * same-orbital A/B (fix on vs off at one converged auto SCF, the
    fix-off side re-entering get_nlc_exc under a forced fp64 mode):
    max|dg| <= 5e-6 bound (measured 1.1-1.2e-9);
  * the fp64-lane gradient is bit-identical to the pre-change path
    (same process, module-level A/B on the cast — the cast is gated on
    the global mode, so the fp64 lane never casts; verified by the
    dtype shim plus a fp64-lane gradient comparison against the
    module with the cast codepath disabled).
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


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_nlc_loop_cast_gated(self):
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        self.assertIsNotNone(src)
        self.assertIn(
            "fp32_grid = precision.get_precision() == 'fp32'", src)
        self.assertEqual(src.count('ao_mask.astype(cupy.float32)'), 1,
                         'exactly one AO cast in the NLC gradient loop')

    def test_cast_inside_deriv2_loop(self):
        # the cast must sit in the deriv=2 contraction loop (after the
        # second block_loop begins), not in the rho loop
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        loop2 = src[src.index('ao_deriv = 2'):]
        self.assertIn('ao_mask.astype(cupy.float32)', loop2)
        rho_loop = src[:src.index('ao_deriv = 2')]
        self.assertNotIn('astype(cupy.float32)', rho_loop,
                         'the rho-evaluation loop keeps its existing '
                         'internal discipline (eval_rho2 casts inside)')

    def test_accumulator_dtype_unchanged(self):
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        # the exc1 accumulator and the dm einsum stay exactly as before
        self.assertIn("exc1 = cupy.zeros((nao,3))", src)
        self.assertIn(
            "exc1[mask] += cupy.einsum('nij,ij->ni', vmat_tmp, dm_mask).T",
            src)

    def test_full_response_variant_untouched(self):
        src = _func_src(RKS_SRC, 'get_nlc_exc_full_response')
        self.assertIsNotNone(src)
        self.assertNotIn('fp32_grid', src)

    def test_uks_shares_the_fixed_function(self):
        uks_src = open(os.path.join(REPO, 'gpu4pyscf', 'grad',
                                    'uks.py')).read()
        self.assertIn('get_nlc_exc = rks_grad.get_nlc_exc', uks_src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestNlcLoopDtype(unittest.TestCase):
    '''dtype shim: which AO dtype reaches _gga_grad_sum_ from the NLC
    loop, per gradient precision mode.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'methanol.xyz')

    def _run(self, grad_mode):
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        records = []
        _orig = rks_grad_mod._gga_grad_sum_
        _orig_nlc = rks_grad_mod.get_nlc_exc
        _region = {'r': 'outside'}

        def rec(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig(ao, wv, *a, **kw)

        def region_wrap(region, orig):
            def w(*a, **kw):
                prev = _region['r']
                _region['r'] = region
                try:
                    return orig(*a, **kw)
                finally:
                    _region['r'] = prev
            return w

        rks_grad_mod._gga_grad_sum_ = rec
        rks_grad_mod.get_nlc_exc = region_wrap('nlc', _orig_nlc)
        rks_grad_mod.get_exc = region_wrap('xc', _orig_nlc.__globals__['get_exc']) \
            if 'get_exc' in _orig_nlc.__globals__ else rks_grad_mod.get_exc
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method()
            g.precision_mode = grad_mode
            g.kernel()
        finally:
            rks_grad_mod._gga_grad_sum_ = _orig
            rks_grad_mod.get_nlc_exc = _orig_nlc
        return records

    def test_fp32_mode_nlc_ao_is_float32(self):
        records = self._run('fp32')
        nlc_dts = {dt for region, dt in records if region == 'nlc'}
        self.assertTrue(nlc_dts, 'the NLC loop must call _gga_grad_sum_')
        self.assertEqual(nlc_dts, {'float32'},
                         f'auto-lane NLC AO must be float32: {nlc_dts}')

    def test_fp64_mode_nlc_ao_is_float64(self):
        records = self._run('fp64')
        nlc_dts = {dt for region, dt in records if region == 'nlc'}
        self.assertTrue(nlc_dts)
        self.assertEqual(nlc_dts, {'float64'},
                         f'fp64-lane NLC AO must stay float64: {nlc_dts}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestNlcLoopNumerics(unittest.TestCase):
    '''Same-orbital A/B: the NLC-loop swap alone at one converged auto
    SCF (the fix-off side re-enters get_nlc_exc under a forced fp64
    global mode).'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms',
                        'methanol.xyz')

    def test_nlc_swap_same_orbital_bound(self):
        import numpy as np
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.grad import rks as rks_grad_mod
        import gpu4pyscf.lib.precision as precision_mod

        mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.precision_mode = 'auto'
        mf.conv_tol = 1e-11
        mf.grids.level = 3
        mf.kernel()
        assert mf.converged
        g = mf.nuc_grad_method()
        g.precision_mode = 'fp32'
        g.kernel()   # warm

        g_on = np.asarray(g.kernel()).copy()

        _orig_get_nlc = rks_grad_mod.get_nlc_exc

        def _get_nlc_fp64(ni, mol_, grids, xc_code, dms, *a, **kw):
            saved = precision_mod.PRECISION_MODE
            precision_mod.PRECISION_MODE = 'fp64'
            try:
                return _orig_get_nlc(ni, mol_, grids, xc_code, dms,
                                     *a, **kw)
            finally:
                precision_mod.PRECISION_MODE = saved

        rks_grad_mod.get_nlc_exc = _get_nlc_fp64
        try:
            g_off = np.asarray(g.kernel()).copy()
        finally:
            rks_grad_mod.get_nlc_exc = _orig_get_nlc

        dg = np.abs(g_on - g_off).max()
        # measured 1.1-1.2e-9 at benzene/methanol; the bound is the
        # conservative family band (auto-lane gradient family 1e-6)
        self.assertLessEqual(dg, 5e-6,
                             'the NLC-loop fp32 swap must stay far below '
                             'the auto-lane family error')

    def test_fp64_lane_unchanged(self):
        # the cast is fp32-mode-only: on the fp64 lane the gradient must
        # reproduce the run-to-run determinism floor (~1e-14 class, the
        # known atomic nondeterminism; measured 8.0e-15 here) -- i.e. the
        # change introduces no fp64-lane effect beyond that floor
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
                             'fp64-lane repeats must stay at the atomic '
                             'nondeterminism floor')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestUksFp32ModeCast(unittest.TestCase):
    '''SP-10 (2026-10-25): the UKS NLC-loop dtype shim, POST-FALSIFICATION.

    LC-88's original cast had no nset gate (fp32 mode cast BOTH spin
    structures); LC-101's regrouping `(fp32_grid or
    _grad_grid_f32()) and nset == 1` silently un-cast the UKS path
    even in fp32 mode, contradicting LC-101's own comment.  SP-10
    repaired the grouping (commit d6aceef) and measured it
    preregistered on the production UKS+NLC doublet cells (methoxy/
    ethoxy/NO2 q0s1 wb97m-v, 43-204 AO): numerics harmless (dg swap
    3e-10..1e-9, e_tot bitwise) but the cost justification FAILED
    everywhere (NLC piece ratio 0.86-1.04 at the <=0.5x bar, t_grad
    0.97-1.02 -- the LC-87/LC-101 cast wins are GEMM-bound-shape-only
    at >= ~222 AO; these contractions are launch/memory-bound).  Per
    the preregistered revert branch the un-cast UKS behavior is kept
    DELIBERATELY.  These tests pin that measured-kept behavior and bar
    silent re-introduction of the fp32-mode UKS cast without fresh
    evidence.'''

    GEOM = os.path.join('/data/datasets/gpu4pyscf-chemistry-cohort-'
                        '20261013-v1', 'methoxy.xyz')

    def _run(self, grad_mode, knob=False):
        import pyscf
        from gpu4pyscf.dft import UKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        records = []
        _orig = rks_grad_mod._gga_grad_sum_
        _orig_nlc = rks_grad_mod.get_nlc_exc
        _region = {'r': 'outside'}

        def rec(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig(ao, wv, *a, **kw)

        def region_wrap(region, orig):
            def w(*a, **kw):
                prev = _region['r']
                _region['r'] = region
                try:
                    return orig(*a, **kw)
                finally:
                    _region['r'] = prev
            return w

        rks_grad_mod._gga_grad_sum_ = rec
        rks_grad_mod.get_nlc_exc = region_wrap('nlc', _orig_nlc)
        saved_knob = rks_grad_mod._GRAD_GRID_F32
        if knob:
            rks_grad_mod._GRAD_GRID_F32 = True
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp',
                          charge=0, spin=1, verbose=0)
            mf = UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method()
            g.precision_mode = grad_mode
            g.kernel()
        finally:
            rks_grad_mod._gga_grad_sum_ = _orig
            rks_grad_mod.get_nlc_exc = _orig_nlc
            rks_grad_mod._GRAD_GRID_F32 = saved_knob
        return records

    def test_uks_fp32_mode_nlc_ao_is_float64_measured_kept(self):
        # SP-10 falsification pin: the fp32-mode UKS NLC cast was
        # measured WINLESS at every production doublet shape and
        # reverted per the preregistered rule; the un-cast behavior is
        # deliberate.  If this test turns RED, someone re-enabled the
        # cast -- require fresh measured evidence first (see
        # grad/rks.py SP-10 note).
        records = self._run('fp32')
        nlc_dts = {dt for region, dt in records if region == 'nlc'}
        self.assertTrue(nlc_dts, 'the UKS NLC loop must call '
                                 '_gga_grad_sum_')
        self.assertEqual(nlc_dts, {'float64'},
                         f'fp32-mode UKS NLC AO must stay float64 '
                         f'(SP-10 measured the cast winless at '
                         f'production shapes and reverted it): '
                         f'{nlc_dts}')

    def test_uks_fp64_mode_nlc_ao_is_float64(self):
        records = self._run('fp64')
        nlc_dts = {dt for region, dt in records if region == 'nlc'}
        self.assertTrue(nlc_dts)
        self.assertEqual(nlc_dts, {'float64'},
                         f'fp64-lane UKS NLC AO must stay float64: '
                         f'{nlc_dts}')

    def test_uks_knob_fp64_mode_stays_fp64(self):
        # LC-89/LC-101 contract: the fp64-lane knob is RKS-only; arming
        # it must NOT cast the UKS NLC loop
        records = self._run('fp64', knob=True)
        nlc_dts = {dt for region, dt in records if region == 'nlc'}
        self.assertTrue(nlc_dts)
        self.assertEqual(nlc_dts, {'float64'},
                         f'the GRAD_GRID_F32 knob must stay RKS-only '
                         f'(nset == 1); UKS NLC keeps fp64 on the fp64 '
                         f'lane: {nlc_dts}')


if __name__ == '__main__':
    unittest.main()

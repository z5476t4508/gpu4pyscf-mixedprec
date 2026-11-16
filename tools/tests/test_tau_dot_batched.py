#!/usr/bin/env python3
"""Focused regressions for the TAU-BATCH-01 opt-in batched/fused _tau_dot
(gpu4pyscf/dft/numint.py, knob GPU4PYSCF_TAU_DOT_BATCHED).

Measured basis (m3next01A, 2026-11-02): tau_dot is the top FP64 op of the
B-lane nr_rks body (58.2% at phenacetin r2scan3c def2-mtzvpp 327 AO).
The knob swaps the per-component train [3 scale passes + 3 separate
contraction-engine GEMMs, beta=1] for [1 broadcast scale + 1 cuBLAS
strided-batched GEMM + 1 reduction]. Accumulation order and kernel
selection differ: numerically equivalent under the project error gates,
NOT bitwise (user program 2026-11-04 reopens this class explicitly; the
LC-98 stacked-tau concatenation -- 3x FLOP penalty, 2.03x slower -- stays
closed and is not re-proposed).

Contract tests:
  * default OFF: knob unset -> legacy path runs (fused never engaged);
  * armed: fused vs legacy max|diff| within dtype-class bands on synthetic
    (4,nao,ngrids) stacks, fp64 AND fp32, symmetric and asymmetric
    operands, including the out= reuse form;
  * parser: non-'0'/'false'/'False' arms; '0'/'false'/'False' disarm;
  * inert while _strict_fp64_reference_active (the strict fp64 reference
    sibling must keep the plain-default solution);
  * shape guard: a 3-component stack under an armed knob behaves exactly
    like the legacy path (loud IndexError from bra[3]), never a silent
    fused mis-computation;
  * end-to-end: tiny MGGA SCF, knob off vs on, |dE| <= 1e-10 Ha with cycle
    counts reported (equal expected; a drift is reported, not asserted
    away).
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy as np
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

from gpu4pyscf.dft import numint  # noqa: E402


def _reset_knob():
    numint._TAU_DOT_BATCHED = None


class _KnobEnv:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self._saved = os.environ.pop('GPU4PYSCF_TAU_DOT_BATCHED', None)
        _reset_knob()
        if self.value is not None:
            os.environ['GPU4PYSCF_TAU_DOT_BATCHED'] = self.value
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ['GPU4PYSCF_TAU_DOT_BATCHED'] = self._saved
        else:
            os.environ.pop('GPU4PYSCF_TAU_DOT_BATCHED', None)
        _reset_knob()


def _synth_stack(nao, ngrids, dtype, seed=0):
    rng = np.random.default_rng(seed)
    ao = rng.standard_normal((4, nao, ngrids)).astype(dtype)
    wv = np.abs(rng.standard_normal(ngrids))
    return cp.asarray(ao), cp.asarray(wv)


class TestKnobParser(unittest.TestCase):
    def test_default_off(self):
        with _KnobEnv(None):
            self.assertFalse(numint._tau_dot_batched_enabled())

    def test_armed_by_one(self):
        with _KnobEnv('1'):
            self.assertTrue(numint._tau_dot_batched_enabled())

    def test_armed_by_any_nonfalse_string(self):
        with _KnobEnv('on'):
            self.assertTrue(numint._tau_dot_batched_enabled())

    def test_disarmed_by_zero_and_false(self):
        for off in ('0', 'false', 'False'):
            with _KnobEnv(off):
                self.assertFalse(numint._tau_dot_batched_enabled())

    def test_inert_under_strict_reference(self):
        saved = numint._strict_fp64_reference_active
        numint._strict_fp64_reference_active = True
        try:
            with _KnobEnv('1'):
                self.assertFalse(numint._tau_dot_batched_enabled())
        finally:
            numint._strict_fp64_reference_active = saved


class TestFusedEquivalence(unittest.TestCase):
    def _pair(self, nao, ngrids, dtype, ket_seed=None):
        bra, wv = _synth_stack(nao, ngrids, dtype, seed=nao)
        ket = bra if ket_seed is None else _synth_stack(
            nao, ngrids, dtype, seed=ket_seed)[0]
        with _KnobEnv(None):
            ref = numint._tau_dot(bra, ket, wv)
        with _KnobEnv('1'):
            fused = numint._tau_dot(bra, ket, wv)
        scale = max(float(cp.abs(ref).max()), 1.0)
        return float(cp.abs(fused - ref).max()), scale

    def test_fp64_symmetric(self):
        for nao, ngrids in ((31, 512), (112, 4096), (216, 1024)):
            diff, scale = self._pair(nao, ngrids, np.float64)
            self.assertLessEqual(
                diff, 1e-12 * scale,
                f'nao={nao} ngrids={ngrids} diff={diff:.3e}')

    def test_fp64_asymmetric(self):
        diff, scale = self._pair(48, 2048, np.float64, ket_seed=99)
        self.assertLessEqual(diff, 1e-12 * scale)

    def test_fp32_class(self):
        for nao, ngrids in ((31, 512), (112, 2048)):
            diff, scale = self._pair(nao, ngrids, np.float32)
            self.assertLessEqual(
                diff / scale, 1e-5,
                f'nao={nao} ngrids={ngrids} rel={diff / scale:.3e}')

    def test_out_reuse_form(self):
        bra, wv = _synth_stack(48, 1024, np.float64, seed=1)
        ket = _synth_stack(48, 1024, np.float64, seed=2)[0]
        with _KnobEnv(None):
            ref = numint._tau_dot(bra, ket, wv)
        with _KnobEnv('1'):
            out = cp.zeros_like(ref)
            ret = numint._tau_dot(bra, ket, wv, out=out)
        self.assertIs(ret, out)
        diff = float(cp.abs(out - ref).max())
        self.assertLessEqual(diff, 1e-12 * max(float(cp.abs(ref).max()), 1.0))

    def test_default_off_engages_legacy_only(self):
        # with the knob unset, monkeypatching the fused entry point must
        # not change the result: the legacy train is what runs
        bra, wv = _synth_stack(24, 256, np.float64, seed=3)
        with _KnobEnv(None):
            def _must_not_run(*a, **kw):
                raise AssertionError('fused path engaged with knob unset')
            saved = numint._tau_dot_batched
            numint._tau_dot_batched = _must_not_run
            try:
                numint._tau_dot(bra, bra, wv)
            finally:
                numint._tau_dot_batched = saved

    def test_guard_routes_three_component_stack_to_legacy(self):
        # a 3-component stack cannot take the fused path; under an armed
        # knob it must behave exactly like the legacy path -- which indexes
        # bra[3] and fails loudly -- never a silent fused mis-computation
        bra, wv = _synth_stack(24, 256, np.float64, seed=4)
        with _KnobEnv(None):
            with self.assertRaises(IndexError):
                numint._tau_dot(bra[:3], bra[:3], wv)
        with _KnobEnv('1'):
            with self.assertRaises(IndexError):
                numint._tau_dot(bra[:3], bra[:3], wv)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestTinyMGGASCF(unittest.TestCase):
    def test_knob_on_energy_invariance(self):
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        results = {}
        for knob in ('0', '1'):
            with _KnobEnv(knob):
                mol = gto.M(atom='H 0 0 0; H 0 0 0.74; O 0 0 1.30',
                            basis='sto-3g', verbose=0)
                mf = RKS(mol, xc='r2SCAN').density_fit(
                    auxbasis='def2-universal-jkfit')
                mf.conv_tol = 1e-12
                mf.grids.level = 2
                e = mf.kernel()
                results[knob] = (e, getattr(mf, 'cycles', None))
        e0, c0 = results['0']
        e1, c1 = results['1']
        self.assertTrue(mf.converged)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        if c0 != c1:
            print(f'TAU-BATCH-01 tiny MGGA cycle drift reported: '
                  f'{c0} -> {c1} (not asserted away)')


if __name__ == '__main__':
    unittest.main()

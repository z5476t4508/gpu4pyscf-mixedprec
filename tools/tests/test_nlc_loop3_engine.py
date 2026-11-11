#!/usr/bin/env python3
"""Focused regressions for the C2-IMPL-01 opt-in cuTensor engine of the
nr_nlc_vxc loop-3 GEMM (gpu4pyscf/dft/numint.py, knob
GPU4PYSCF_NLC_LOOP3_ENGINE).

Measured basis (entry 301, C2-ENGINE-01 Phase A, 2026-11-04): at the roster
NLC loop-3 shapes the cuTensor fp64 contraction measured 1.289x over the
production same-shape cuBLAS call at all three ibuprofen shape classes,
parity at testosterone, BITWISE-identical output with per-call plan
creation inside the timed window.  This implementation hoists the
descriptors/plan per (shape, dtype, pointer-alignment class) behind the
opt-in selector.  Engineering probe (synthetic shapes, never adjudicated):
create_plan must receive estimateWorkspaceSize as ws_limit; ws_limit=0
selects a slower reassociated algorithm that is NOT bitwise.

Contract tests:
  * default OFF: knob unset / '' / 'cublas' -> 'cublas' (production path
    verbatim; the cuTensor entry point must never be engaged);
  * 'cutensor' arms; any other value fails closed with ValueError;
  * inert while _strict_fp64_reference_active;
  * engine equivalence: _nlc_loop3_cutensor_matmul is an exact fp64
    contraction within the reassociation class (rel <= 1e-12, engineering
    probe 2 measured 8.4e-16..2.0e-15) vs a.dot(b.T) on synthetic fp64
    stacks, incl. cache reuse across calls, two interleaved shapes,
    ao[0]-style slice operands (element-aligned pointer class under the
    canonical-itemsize requirement) and the out= form; one plan per
    (shape, dtype) regardless of pool placement (the attempt-1
    alignment-class keying fragmented under pool churn and tripped the
    fail-closed cap inside a real SCF);
  * empty grid block returns zeros (beta=0 never leaves garbage);
  * fail-closed: cuTensor import failure raises RuntimeError naming the
    knob (no silent cuBLAS fallback); non-contiguous/mismatched operands
    raise; the plan cache refuses to grow past its cap;
  * end-to-end: tiny wb97mv (NLC) SCF, knob off vs on, |dE| <= 1e-10 Ha
    with cycle counts reported (bitwise expected at matched shapes;
    reported, not asserted away).
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
    numint._NLC_LOOP3_ENGINE = None
    numint._NLC_LOOP3_CUTENSOR = None


class _KnobEnv:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self._saved = os.environ.pop('GPU4PYSCF_NLC_LOOP3_ENGINE', None)
        _reset_knob()
        if self.value is not None:
            os.environ['GPU4PYSCF_NLC_LOOP3_ENGINE'] = self.value
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ['GPU4PYSCF_NLC_LOOP3_ENGINE'] = self._saved
        else:
            os.environ.pop('GPU4PYSCF_NLC_LOOP3_ENGINE', None)
        _reset_knob()


class TestKnobParser(unittest.TestCase):
    def test_default_off(self):
        with _KnobEnv(None):
            self.assertEqual(numint._nlc_loop3_engine(), 'cublas')

    def test_explicit_cublas_disarms(self):
        with _KnobEnv('cublas'):
            self.assertEqual(numint._nlc_loop3_engine(), 'cublas')

    def test_empty_string_disarms(self):
        with _KnobEnv(''):
            self.assertEqual(numint._nlc_loop3_engine(), 'cublas')

    def test_cutensor_arms(self):
        with _KnobEnv('cutensor'):
            self.assertEqual(numint._nlc_loop3_engine(), 'cutensor')

    def test_garbage_fails_closed(self):
        for bad in ('cudnn', 'Cutensor', 'CUTENSOR', '1', 'true'):
            with self.assertRaises(ValueError, msg=bad):
                with _KnobEnv(bad):
                    numint._nlc_loop3_engine()

    def test_inert_under_strict_reference(self):
        saved = numint._strict_fp64_reference_active
        numint._strict_fp64_reference_active = True
        try:
            with _KnobEnv('cutensor'):
                self.assertEqual(numint._nlc_loop3_engine(), 'cublas')
        finally:
            numint._strict_fp64_reference_active = saved


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestEngineEquivalence(unittest.TestCase):
    # the canonical-itemsize plan is an exact fp64 contraction whose
    # accumulation order differs from cuBLAS only at the reassociation
    # level (engineering probe 2: rel 8.4e-16..2.0e-15); the entry-301
    # card records rel<=1e-12 as the acceptable kernel-equivalence class
    REL_TOL = 1e-12
    SHAPES = ((31, 512), (96, 2048), (61, 1025), (200, 4096))

    def _rel(self, got, ref):
        return float(cp.abs(got - ref).max() /
                     max(float(cp.abs(ref).max()), 1e-300))

    def _pair(self, nao, ngr, seed):
        rng = np.random.default_rng(seed)
        a = cp.asarray(rng.standard_normal((nao, ngr)))
        b = cp.asarray(rng.standard_normal((nao, ngr)))
        buf = cp.empty((nao, nao))
        with _KnobEnv('cutensor'):
            got = numint._nlc_loop3_cutensor_matmul(a, b, out=buf)
            # second call with identical buffers: pure plan-cache hit
            got2 = numint._nlc_loop3_cutensor_matmul(a, b, out=buf)
            n_plans = len(numint._NLC_LOOP3_CUTENSOR['plans'])
            ws = numint._NLC_LOOP3_CUTENSOR['ws']
        ref = a.dot(b.T)
        cp.cuda.runtime.deviceSynchronize()
        self.assertLessEqual(self._rel(got, ref), self.REL_TOL)
        self.assertLessEqual(self._rel(got2, ref), self.REL_TOL)
        self.assertEqual(got.dtype, ref.dtype)
        return n_plans, (0 if ws is None else int(ws.size))

    def test_rel_vs_cublas(self):
        for i, (nao, ngr) in enumerate(self.SHAPES):
            n_plans, ws = self._pair(nao, ngr, seed=nao + i)
            self.assertEqual(n_plans, 1,
                             'identical buffers must reuse one plan')
            self.assertGreater(ws, 0)

    def test_fresh_out_allocations_one_plan(self):
        # the production loop allocates a fresh out (and _scale_ao a fresh
        # aow) every block/cycle; the canonical-itemsize key must collapse
        # all pool placements of one shape into exactly ONE plan (the
        # attempt-1 alignment-class keying fragmented here and tripped the
        # cap inside one real SCF)
        rng = np.random.default_rng(42)
        a = cp.asarray(rng.standard_normal((96, 2048)))
        b = cp.asarray(rng.standard_normal((96, 2048)))
        with _KnobEnv('cutensor'):
            for _ in range(6):
                got = numint._nlc_loop3_cutensor_matmul(a, b)
                self.assertLessEqual(self._rel(got, a.dot(b.T)),
                                     self.REL_TOL)
            n_plans = len(numint._NLC_LOOP3_CUTENSOR['plans'])
        self.assertEqual(n_plans, 1)

    def test_two_shapes_interleaved_cache(self):
        with _KnobEnv('cutensor'):
            arrays = []
            for nao, ngr in ((48, 1024), (96, 2048)):
                rng = np.random.default_rng(nao)
                a = cp.asarray(rng.standard_normal((nao, ngr)))
                b = cp.asarray(rng.standard_normal((nao, ngr)))
                buf = cp.empty((nao, nao))
                arrays.append((a, b, buf))
            for a, b, buf in arrays:   # first pass: both plans created
                numint._nlc_loop3_cutensor_matmul(a, b, out=buf)
            n_after_first = len(numint._NLC_LOOP3_CUTENSOR['plans'])
            for a, b, buf in arrays:   # second pass: pure cache hits
                numint._nlc_loop3_cutensor_matmul(a, b, out=buf)
                self.assertLessEqual(self._rel(buf, a.dot(b.T)),
                                     self.REL_TOL)
            self.assertEqual(len(numint._NLC_LOOP3_CUTENSOR['plans']),
                             n_after_first)
            self.assertEqual(n_after_first, 2)

    def test_slice_operand_alignment_class(self):
        # ao[0]-style slice: odd ngrids makes the component-plane offset an
        # 8-byte alignment class (77*2045*8 % 512 == 200), unlike the
        # pool-aligned fresh arrays -- the canonical requirement must
        # serve both under one plan
        rng = np.random.default_rng(7)
        stack = cp.asarray(rng.standard_normal((4, 77, 2045)))
        a = stack[0]
        b = cp.asarray(rng.standard_normal((77, 2045)))
        with _KnobEnv('cutensor'):
            got = numint._nlc_loop3_cutensor_matmul(a, b)
            n_plans = len(numint._NLC_LOOP3_CUTENSOR['plans'])
        cp.cuda.runtime.deviceSynchronize()
        self.assertLessEqual(self._rel(got, a.dot(b.T)), self.REL_TOL)
        self.assertEqual(n_plans, 1)

    def test_out_form_and_empty_block(self):
        rng = np.random.default_rng(8)
        a = cp.asarray(rng.standard_normal((40, 1024)))
        b = cp.asarray(rng.standard_normal((40, 1024)))
        with _KnobEnv('cutensor'):
            out = cp.empty((40, 40))
            ret = numint._nlc_loop3_cutensor_matmul(a, b, out=out)
            self.assertIs(ret, out)
            empty = numint._nlc_loop3_cutensor_matmul(
                cp.empty((5, 0)), cp.empty((5, 0)))
        cp.cuda.runtime.deviceSynchronize()
        self.assertLessEqual(self._rel(out, a.dot(b.T)), self.REL_TOL)
        self.assertTrue(cp.array_equal(empty, cp.zeros((5, 5))))

    def test_fail_closed_on_layout(self):
        rng = np.random.default_rng(9)
        a = cp.asarray(rng.standard_normal((40, 1024)))
        b = cp.asarray(rng.standard_normal((40, 1024)))
        b_nc = cp.asfortranarray(b)          # non-C-contiguous operand
        with _KnobEnv('cutensor'):
            with self.assertRaises(RuntimeError):
                numint._nlc_loop3_cutensor_matmul(a, b_nc)
            with self.assertRaises(RuntimeError):   # dtype/shape mismatch
                numint._nlc_loop3_cutensor_matmul(a, b.astype(np.float32))

    def test_plan_cache_cap_fail_closed(self):
        saved_cap = numint._NLC_LOOP3_PLAN_CAP
        numint._NLC_LOOP3_PLAN_CAP = 1
        try:
            with _KnobEnv('cutensor'):
                rng = np.random.default_rng(1)
                a = cp.asarray(rng.standard_normal((48, 1024)))
                b = cp.asarray(rng.standard_normal((48, 1024)))
                buf = cp.empty((48, 48))
                numint._nlc_loop3_cutensor_matmul(a, b, out=buf)  # 1 plan
                numint._nlc_loop3_cutensor_matmul(a, b, out=buf)  # cache hit
                a2 = cp.asarray(rng.standard_normal((64, 512)))
                b2 = cp.asarray(rng.standard_normal((64, 512)))
                buf2 = cp.empty((64, 64))
                with self.assertRaises(RuntimeError):
                    numint._nlc_loop3_cutensor_matmul(a2, b2, out=buf2)
        finally:
            numint._NLC_LOOP3_PLAN_CAP = saved_cap


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestFailClosedCutensorUnavailable(unittest.TestCase):
    def test_import_failure_raises_no_silent_fallback(self):
        saved = sys.modules.get('cupyx.cutensor', None)
        sys.modules['cupyx.cutensor'] = None   # import ... -> ImportError
        try:
            with _KnobEnv('cutensor'):
                rng = np.random.default_rng(3)
                a = cp.asarray(rng.standard_normal((24, 512)))
                b = cp.asarray(rng.standard_normal((24, 512)))
                with self.assertRaises(RuntimeError) as cm:
                    numint._nlc_loop3_cutensor_matmul(a, b)
                self.assertIn('GPU4PYSCF_NLC_LOOP3_ENGINE', str(cm.exception))
                self.assertIn('cuTensor', str(cm.exception))
        finally:
            if saved is None:
                sys.modules.pop('cupyx.cutensor', None)
            else:
                sys.modules['cupyx.cutensor'] = saved


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestTinyNLCSCF(unittest.TestCase):
    def _run(self):
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        mol = gto.M(atom='H 0 0 0; H 0 0 0.74; O 0 0 1.30',
                    basis='sto-3g', verbose=0)
        mf = RKS(mol, xc='wb97mv').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-12
        mf.grids.level = 2
        mf.nlcgrids.level = 2
        e = mf.kernel()
        return e, getattr(mf, 'cycles', None), bool(mf.converged)

    def test_knob_on_energy_invariance(self):
        results = {}
        for knob in (None, 'cutensor'):
            with _KnobEnv(knob):
                results[str(knob)] = self._run()
        e0, c0, conv0 = results['None']
        e1, c1, conv1 = results['cutensor']
        self.assertTrue(conv0 and conv1)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        if c0 != c1:
            print(f'C2-IMPL-01 tiny NLC SCF cycle drift reported: '
                  f'{c0} -> {c1} (not asserted away)')

    def test_default_off_never_engages_engine(self):
        def _must_not_run(*a, **kw):
            raise AssertionError('cutensor engine engaged with knob unset')
        saved = numint._nlc_loop3_cutensor_matmul
        numint._nlc_loop3_cutensor_matmul = _must_not_run
        try:
            with _KnobEnv(None):
                e, cycles, conv = self._run()
        finally:
            numint._nlc_loop3_cutensor_matmul = saved
        self.assertTrue(conv)


if __name__ == '__main__':
    unittest.main()

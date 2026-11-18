#!/usr/bin/env python3
"""Regression tests for the LC-86 ip1-f32-on-fp64-lane knob
(gpu4pyscf/df/grad/rhf.py + gpu4pyscf/df/grad/uhf.py,
GPU4PYSCF_DF_IP1_F32).

The DF-gradient ip1 integral kernel exists in two variants:
sum_ejk_int3c2e_ip1 (double) and sum_ejk_int3c2e_ip1_f32 (float
interior, float pseudo-DM input, double ejk/ejk_aux accumulators).
Upstream selects the f32 variant only when the global precision mode is
'fp32'.  LC-83's post-knob decomposition named the fused ip1 kernel as
the largest remaining addressable gradient piece (0.117s = 14.3% of
t_grad at wb97mv/benzene/tzvp in the both-knobs state), so this opt-in
knob measures the precision/cost trade of running the f32 kernel on the
fp64 lane -- a precision-policy prototype for the pending user decision
(LC-45 (b2) family), NOT a default.

The knob must preserve the existing fp32 discipline exactly:
  * only the integral kernel and its pseudo-DM/dm+auxvec INPUTS go
    float32;
  * the j3c@dm contractions, the j2c^-1 metric solve and the atomic
    ejk/ejk_aux accumulation targets stay float64 (the metric solve
    amplifies float32 relative error by the auxiliary-basis condition
    number straight into the gradient);
  * lanes that already resolve the gradient to fp32 (the 'auto'
    policy's gradient resolution) must be structurally unaffected;
  * the upstream nuclear-attraction ip1 caller (grad/rhf.py fakemol,
    charges as 'auxiliary basis') is NOT part of the knob.

Contract tests (CPU, source-level):
  * the knob parser: unset/empty/other -> False (default OFF, byte-for-
    byte upstream kernel selection); '1' -> True; a warning on any other
    non-empty value; cached after first read;
  * all three DF ip1 kernel-selection sites consult _ip1_f32() with the
    OR 'precision.get_precision() == "fp32" or _ip1_f32()';
  * the pre-contraction section (metric solve) stays float64;
  * the nuclear-attraction path (grad/rhf.py) stays ungated.

Numerical tests (GPU, small cells):
  * default (knob unset) selects the double kernel in fp64 mode;
  * knob on selects the f32 kernel in fp64 mode (selection shim) at the
    closed-shell J+K path AND at the UKS J+K path;
  * knob-on gradient matches knob-off at the same converged orbitals on
    the J+K hybrid path (bound set from the measured LC-86 evidence:
    fp32 pseudo-DM cast ~1e-7 relative through one kernel, well below
    the 3e-5 gate);
  * the J-only path (pure functional) selects the f32 kernel with the
    knob on (its dm/auxvec inputs are cast, metric solve stays fp64).
"""
import os
import re
import sys
import unittest
import warnings

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

RHF_SRC = os.path.join(REPO, 'gpu4pyscf', 'df', 'grad', 'rhf.py')
UHF_SRC = os.path.join(REPO, 'gpu4pyscf', 'df', 'grad', 'uhf.py')
GRAD_RHF_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'rhf.py')


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestKnobParser(unittest.TestCase):
    def setUp(self):
        import gpu4pyscf.df.grad.rhf as rhf_mod
        self.rhf_mod = rhf_mod
        self._saved = rhf_mod._IP1_F32
        rhf_mod._IP1_F32 = None

    def tearDown(self):
        self.rhf_mod._IP1_F32 = self._saved

    def _with_env(self, value):
        old = os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
        if value is not None:
            os.environ['GPU4PYSCF_DF_IP1_F32'] = value
        try:
            self.rhf_mod._IP1_F32 = None
            return self.rhf_mod._ip1_f32()
        finally:
            if value is not None:
                os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = old

    def test_default_off(self):
        self.assertFalse(self._with_env(None))
        self.assertFalse(self._with_env(''))

    def test_value_1_enables(self):
        self.assertTrue(self._with_env('1'))

    def test_other_values_warn_and_fall_back(self):
        for bad in ('0', 'true', 'yes', '2', 'fp32'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.assertFalse(self._with_env(bad))
            self.assertTrue(any('GPU4PYSCF_DF_IP1_F32' in str(x.message)
                                for x in w), bad)

    def test_cached_after_first_read(self):
        self._with_env('1')
        os.environ['GPU4PYSCF_DF_IP1_F32'] = '0'
        try:
            self.assertTrue(self.rhf_mod._ip1_f32())
        finally:
            os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)

    def test_independent_of_fused_knob_cache(self):
        # the two knobs are separate caches; flipping one must not
        # disturb the other's cached value
        self.rhf_mod._IP1_F32 = None
        self.rhf_mod._IP1_FUSED = None
        old = os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
        oldf = os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        try:
            os.environ['GPU4PYSCF_DF_IP1_F32'] = '1'
            self.assertTrue(self.rhf_mod._ip1_f32())
            self.assertFalse(self.rhf_mod._ip1_fused())
        finally:
            os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = old
            if oldf is not None:
                os.environ['GPU4PYSCF_DF_IP1_FUSED'] = oldf


class TestCallSiteContract(unittest.TestCase):
    SELECT = 'fp32 = precision.get_precision() == \'fp32\' or _ip1_f32()'

    def test_rhf_jk_site_gated(self):
        src = _func_src(RHF_SRC, '_jk_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertIn(self.SELECT, src)
        # the f32 branch must still cast the pseudo-DM (the kernel reads
        # float dm; a missing cast would reinterpret bits)
        self.assertIn('dm_tensor.astype(cp.float32)', src)

    def test_rhf_j_only_site_gated(self):
        src = _func_src(RHF_SRC, '_j_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertIn(self.SELECT, src)
        # the f32 branch must still cast dm and auxvec
        self.assertIn('dm_kern = cp.asarray(dm, dtype=cp.float32', src)
        self.assertIn('auxvec_kern = cp.asarray(auxvec, dtype=cp.float32)',
                      src)

    def test_uhf_jk_site_gated(self):
        src = _func_src(UHF_SRC, '_jk_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertIn(self.SELECT, src)
        self.assertIn('dm_tensor.astype(cp.float32)', src)
        # uhf imports the parser from the closed-shell module
        uhf_imports = open(UHF_SRC).read()
        self.assertIn('_ip1_f32', uhf_imports.split('def ')[0])

    def test_metric_solve_stays_fp64(self):
        # the pre-contraction machinery (j3c build, dm contraction,
        # metric solve) must show no knob-driven dtype change: the knob
        # must appear in exactly ONE executable selection line per site
        for path, func in ((RHF_SRC, '_jk_energy_per_atom'),
                           (UHF_SRC, '_jk_energy_per_atom'),
                           (RHF_SRC, '_j_energy_per_atom')):
            src = _func_src(path, func)
            self.assertEqual(src.count(self.SELECT), 1,
                             f'{path}:{func} selects via the knob exactly once')
            self.assertIn('_gen_metric_solver', src)

    def test_nuclear_attraction_path_untouched(self):
        # grad/rhf.py's hcore ip1 call (fakemol, charges) is upstream,
        # precision-ungated and stays that way
        src = open(GRAD_RHF_SRC).read()
        self.assertNotIn('_ip1_f32', src)
        self.assertNotIn('GPU4PYSCF_DF_IP1_F32', src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKernelSelection(unittest.TestCase):
    '''Selection shims: which kernel symbol the DF gradient calls, in
    fp64 precision mode, knob off vs on.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')
    UKS_GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methoxy.xyz')

    def _run_grad_counting(self, xc, knob, use_uks=False):
        import pyscf
        from gpu4pyscf.dft import RKS, UKS
        from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys
        import gpu4pyscf.df.grad.rhf as rhf_mod
        from gpu4pyscf.lib import precision

        self.assertEqual(precision.get_precision(), 'fp64',
                         'test must run in the default fp64 mode')
        rhf_mod._IP1_F32 = knob
        counts = {'f64': 0, 'f32': 0}
        _o64 = libvhf_rys.sum_ejk_int3c2e_ip1
        _o32 = libvhf_rys.sum_ejk_int3c2e_ip1_f32

        def shim64(*a, **kw):
            counts['f64'] += 1
            return _o64(*a, **kw)

        def shim32(*a, **kw):
            counts['f32'] += 1
            return _o32(*a, **kw)

        libvhf_rys.sum_ejk_int3c2e_ip1 = shim64
        libvhf_rys.sum_ejk_int3c2e_ip1_f32 = shim32
        old = os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
        try:
            geom = self.UKS_GEOM if use_uks else self.GEOM
            mol = pyscf.M(atom=geom, basis='def2-svp', verbose=0,
                          spin=1 if use_uks else 0)
            if use_uks:
                mol.spin = 1
                mf = UKS(mol, xc=xc).density_fit(
                    auxbasis='def2-universal-jkfit')
            else:
                mf = RKS(mol, xc=xc).density_fit(
                    auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            mf.nuc_grad_method().kernel()
        finally:
            libvhf_rys.sum_ejk_int3c2e_ip1 = _o64
            libvhf_rys.sum_ejk_int3c2e_ip1_f32 = _o32
            rhf_mod._IP1_F32 = None
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = old
        return counts

    def test_hybrid_default_uses_double_kernel(self):
        counts = self._run_grad_counting('b3lyp', knob=None)
        self.assertGreater(counts['f64'], 0)
        self.assertEqual(counts['f32'], 0)

    def test_hybrid_knob_uses_f32_kernel(self):
        counts = self._run_grad_counting('b3lyp', knob=True)
        self.assertGreater(counts['f32'], 0)
        # the nuclear-attraction caller (grad/rhf.py) legitimately keeps
        # the double kernel: it is not part of the knob
        self.assertGreaterEqual(counts['f64'], 1)

    def test_uks_hybrid_knob_uses_f32_kernel(self):
        counts = self._run_grad_counting('b3lyp', knob=True, use_uks=True)
        self.assertGreater(counts['f32'], 0)
        self.assertGreaterEqual(counts['f64'], 1)

    def test_pure_functional_j_only_knob_uses_f32_kernel(self):
        # pure functional: k_factor == 0 -> the J-only path; the f32
        # kernel must be selected there too (dm + auxvec casts)
        counts = self._run_grad_counting('lda', knob=True)
        self.assertGreater(counts['f32'], 0)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobNumerics(unittest.TestCase):
    '''Same-orbital A/B on the J+K hybrid path: one converged SCF, the
    gradient evaluated twice with the knob off/on.  The delta is the
    kernel-swap cost alone (fp32 pseudo-DM cast + fp32 interior +
    fp32-in, fp64-out accumulators) at identical inputs.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def test_hybrid_gradient_knob_delta(self):
        import pyscf
        from gpu4pyscf.dft import RKS
        import gpu4pyscf.df.grad.rhf as rhf_mod
        old = os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            rhf_mod._IP1_F32 = False
            g0 = mf.nuc_grad_method().kernel()
            rhf_mod._IP1_F32 = True
            g1 = mf.nuc_grad_method().kernel()
        finally:
            rhf_mod._IP1_F32 = None
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = old
        dg = abs(g1 - g0).max()
        # bound from the LC-86 measured evidence (preregistration P1
        # band 1e-7..5e-6); assert the conservative structural bound --
        # a single-kernel fp32 swap must stay far below the 3e-5 gate
        self.assertLessEqual(dg, 1e-5,
                             'ip1-f32 kernel swap must stay far below the '
                             '3e-5 gradient gate')

    def test_j_only_gradient_knob_delta(self):
        import pyscf
        from gpu4pyscf.dft import RKS
        import gpu4pyscf.df.grad.rhf as rhf_mod
        old = os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='lda').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            rhf_mod._IP1_F32 = False
            g0 = mf.nuc_grad_method().kernel()
            rhf_mod._IP1_F32 = True
            g1 = mf.nuc_grad_method().kernel()
        finally:
            rhf_mod._IP1_F32 = None
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = old
        dg = abs(g1 - g0).max()
        self.assertLessEqual(dg, 1e-5)


if __name__ == '__main__':
    unittest.main()

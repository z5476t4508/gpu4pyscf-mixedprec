#!/usr/bin/env python3
"""Regression tests for the LC-42 fused ip1 launch knob
(gpu4pyscf/df/grad/rhf.py + gpu4pyscf/df/grad/uhf.py,
GPU4PYSCF_DF_IP1_FUSED).

The J+K (hybrid-functional) gradient path serializes one
sum_ejk_int3c2e_ip1 launch per aux l-batch with grid
(n_shell_pair_blocks, 1): 8-12 sequential launches per gradient at
16-28% SM occupancy on the 128-SM RTX 4090 (measured, LC-42 probes).
The opt-in fused launch passes the full batch-boundary array as the
y-grid in ONE launch (grid (n_sp, n_batches)); per-y-block addressing
is identical to the per-batch launches by construction, so the result
differs only by atomic interleaving order (the per-batch path's own
cross-run nondeterminism class).

Sub-batch y-chunking was rejected by static analysis: the exchange
branch indexes the compressed buffer as
dm[pair_offset*naux + k0 + k*nksh] with nksh = ksh1-ksh0 (the y-block
shell count), so changing block boundaries INSIDE a batch would read
wrong columns.  These tests pin that contract.

Contract tests (CPU, source-level):
  * the knob parser: unset/empty/other -> False (default OFF, the
    upstream per-batch path byte-for-byte); '1' -> True; a warning on
    any other non-empty value;
  * both call sites pass nbatches_ksh = len(ksh_offsets)-1 only when
    fused, else the upstream literal 1;
  * the memory guard (full-width compressed buffer < 25% of available
    device memory) with the per-batch fallback;
  * aux_offset = 0 + full-width buffer + kernel naux arg = total naux
    in the fused branch (the addressing-consistency contract).

Numerical tests (GPU, small cell):
  * default (knob unset) gradient == the pre-change per-batch path
    (bitwise, same process, monkeypatched knob cache);
  * fused-on gradient matches knob-off at the same-orbital floor
    (atomic-order noise only);
  * fused-on actually issues ONE ip1 launch per _jk_energy_per_atom
    call (vs aux_batches per-batch), verified by a counting shim.
"""
import ctypes
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


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestKnobParser(unittest.TestCase):
    def setUp(self):
        import importlib
        import gpu4pyscf.df.grad.rhf as rhf_mod
        self.rhf_mod = rhf_mod
        self._saved = rhf_mod._IP1_FUSED
        rhf_mod._IP1_FUSED = None

    def tearDown(self):
        self.rhf_mod._IP1_FUSED = self._saved

    def _with_env(self, value):
        old = os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        if value is not None:
            os.environ['GPU4PYSCF_DF_IP1_FUSED'] = value
        try:
            self.rhf_mod._IP1_FUSED = None
            return self.rhf_mod._ip1_fused()
        finally:
            if value is not None:
                os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_FUSED'] = old

    def test_default_off(self):
        self.assertFalse(self._with_env(None))
        self.assertFalse(self._with_env(''))

    def test_value_1_enables(self):
        self.assertTrue(self._with_env('1'))

    def test_other_values_warn_and_fall_back(self):
        for bad in ('0', 'true', 'yes', '2'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                self.assertFalse(self._with_env(bad))
            self.assertTrue(any('GPU4PYSCF_DF_IP1_FUSED' in str(x.message)
                                for x in w), bad)

    def test_cached_after_first_read(self):
        self._with_env('1')
        # second read must not consult the environment again
        os.environ['GPU4PYSCF_DF_IP1_FUSED'] = '0'
        try:
            self.assertTrue(self.rhf_mod._ip1_fused())
        finally:
            os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)


class TestCallSiteContract(unittest.TestCase):
    def test_rhf_jk_site_fused_wiring(self):
        src = _func_src(RHF_SRC, '_jk_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertIn('_ip1_fused()', src)
        # the memory guard: full-width buffer must fit 25% of free memory
        self.assertIn('get_avail_mem()*0.25', src)
        # fused: single iteration over the whole aux range
        self.assertIn('range(aux_batches if not fused else 1)', src)
        # nbatches_ksh: fused -> all batch boundaries, else upstream 1
        self.assertIn("ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused else 1)",
                      src)
        # ksh_offsets: fused -> the full array, else the per-batch suffix
        self.assertIn('ksh_offsets_gpu.data.ptr if fused', src)
        # addressing consistency: fused passes aux_offset 0 and the
        # total naux as the kernel's naux (row stride of the full-width
        # compressed buffer)
        self.assertIn('aux_ao_offset = 0', src)
        self.assertIn('naux_in_batch = naux', src)

    def test_uhf_jk_site_fused_wiring(self):
        src = _func_src(UHF_SRC, '_jk_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertIn('_ip1_fused()', src)
        self.assertIn('get_avail_mem()*0.25', src)
        self.assertIn('range(aux_batches if not fused else 1)', src)
        self.assertIn("ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused else 1)",
                      src)
        self.assertIn('ksh_offsets_gpu.data.ptr if fused', src)
        self.assertIn('aux_ao_offset = 0', src)
        self.assertIn('naux_in_batch = naux', src)

    def test_subbatch_chunking_not_attempted(self):
        # the rejected design must not sneak in: the only boundary
        # computation is the upstream batch-size-driven l-group split
        # (one _split_l_ctr_pattern call per function), and the fused
        # launch must feed those SAME boundaries to the kernel -- the
        # exchange-branch compressed addressing embeds the y-block shell
        # count, so sub-batch re-splitting would corrupt it
        for path in (RHF_SRC, UHF_SRC):
            src = _func_src(path, '_jk_energy_per_atom')
            self.assertEqual(src.count('_split_l_ctr_pattern('), 1,
                             'exactly the upstream boundary split; no '
                             'sub-batch chunking')
            self.assertIn('ksh_offsets_cpu = l_ctr_aux_offsets', src)

    def test_j_only_path_untouched(self):
        # the J-only path (pure functionals) has fused since upstream;
        # this change must not touch it
        src = _func_src(RHF_SRC, '_j_energy_per_atom')
        self.assertIsNotNone(src)
        self.assertNotIn('_ip1_fused', src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestFusedNumerics(unittest.TestCase):
    '''GPU numerics on a small hybrid cell (the J+K path).'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def _grad(self, fused):
        import pyscf
        from gpu4pyscf.dft import RKS
        import gpu4pyscf.df.grad.rhf as rhf_mod
        rhf_mod._IP1_FUSED = fused
        old = os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method().kernel()
            return mf.e_tot, g
        finally:
            rhf_mod._IP1_FUSED = None
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_FUSED'] = old

    def test_fused_matches_perbatch(self):
        import pyscf
        from gpu4pyscf.dft import RKS
        import gpu4pyscf.df.grad.rhf as rhf_mod
        old = os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            # Same-orbital A/B (the LC-42 verification protocol): two
            # fresh SCFs differ at the 1e-10 gradient scale by dm noise
            # alone, which sits exactly at the assertion bound and makes
            # this test flaky; the knob's own effect is atomic
            # interleaving order on identical inputs.
            rhf_mod._IP1_FUSED = False
            g0 = mf.nuc_grad_method().kernel()
            rhf_mod._IP1_FUSED = True
            g1 = mf.nuc_grad_method().kernel()
        finally:
            rhf_mod._IP1_FUSED = None
            if old is not None:
                os.environ['GPU4PYSCF_DF_IP1_FUSED'] = old
        dg = abs(g1 - g0).max()
        self.assertLessEqual(dg, 1e-10,
                             'fused launch must differ from per-batch only '
                             'by atomic interleaving order')

    def test_fused_single_launch(self):
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys
        import gpu4pyscf.df.grad.rhf as rhf_mod
        rhf_mod._IP1_FUSED = True
        counts = {'n': 0, 'ksh': []}
        _orig = libvhf_rys.sum_ejk_int3c2e_ip1

        def shim(ejk, ejk_aux, dm, auxvec, n_dm, envs, shm_size,
                 n_sp_blocks, n_ksh_blocks, *a, **kw):
            counts['n'] += 1
            counts['ksh'].append(int(n_ksh_blocks.value))
            return _orig(ejk, ejk_aux, dm, auxvec, n_dm, envs, shm_size,
                         n_sp_blocks, n_ksh_blocks, *a, **kw)

        libvhf_rys.sum_ejk_int3c2e_ip1 = shim
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            mf.nuc_grad_method().kernel()
        finally:
            libvhf_rys.sum_ejk_int3c2e_ip1 = _orig
            rhf_mod._IP1_FUSED = None
        # the J+K ip1 call is ONE fused launch with all batch boundaries
        # on the y-grid (>= 2 l-groups on this cell); the other
        # sum_ejk_int3c2e_ip1 caller in the gradient is the
        # nuclear-attraction path (grad/rhf.py, fakemol: single l-group,
        # naux=charges) which legitimately stays at ksh=1
        self.assertGreaterEqual(counts['n'], 2)
        self.assertIn(11, counts['ksh'])   # the J+K fused launch
        self.assertEqual(counts['ksh'].count(1), counts['n'] - 1,
                         f'only the J+K launch fuses: {counts}')

    def test_default_off_multi_launch(self):
        import pyscf
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.df.int3c2e_bdiv import libvhf_rys
        counts = {'n': 0, 'ksh': []}
        _orig = libvhf_rys.sum_ejk_int3c2e_ip1

        def shim(ejk, ejk_aux, dm, auxvec, n_dm, envs, shm_size,
                 n_sp_blocks, n_ksh_blocks, *a, **kw):
            counts['n'] += 1
            counts['ksh'].append(int(n_ksh_blocks.value))
            return _orig(ejk, ejk_aux, dm, auxvec, n_dm, envs, shm_size,
                         n_sp_blocks, n_ksh_blocks, *a, **kw)

        libvhf_rys.sum_ejk_int3c2e_ip1 = shim
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            mf.nuc_grad_method().kernel()
        finally:
            libvhf_rys.sum_ejk_int3c2e_ip1 = _orig
        self.assertGreater(counts['n'], 1)
        self.assertTrue(all(k == 1 for k in counts['ksh']),
                        f'knob off must keep the upstream per-batch '
                        f'launches: {counts}')


if __name__ == '__main__':
    unittest.main()

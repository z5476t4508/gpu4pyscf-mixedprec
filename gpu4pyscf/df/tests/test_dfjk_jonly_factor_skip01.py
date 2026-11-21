# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
'''DFJK-JONLY-FACTOR-SKIP-01 focused regression (2026-11-19).

The df_jk.get_jk entry calls factorize_dm(dms, hermi) unconditionally
at the parent sha, but the J contraction consumes ONLY dm_sparse
(gathered from dms_3d) and the cderi blocks -- the factorization
output feeds the K branch and the nocc==0 early return exclusively.
At a J-only entry (with_k=False) the factorization is dead work
whenever the dm arrives without mo_coeff/factor_l tags:
_DFHF.get_veff's pure-functional UKS branch passes dm[0]+dm[1] (the
spin sum strips every tag), so decompose_rdm1_svd (hermi=1) pays one
full dm eigendecomposition per call whose result the J path never
reads.  The waste fires exactly at pre-built-cderi pure-functional
open-shell SCFs -- the steady-restart protocol whose driver builds
with_df BEFORE kernel() (the e724/e726/e727/e735/e742 depth lineage
at the frozen tamoxifen_anion r2scan3c UKS 687 cell).  The tagged
pure-RKS branch and every hybrid branch (J+K in one call) are exempt;
entries 325/385 measured those exempt branches.

The fix skips the factorization when with_k is False (nao/nspin
derived from dms, the zero-dm early return preserved by the
exact-zero count_nonzero twin, the warmup/syrk K-side guards inert at
factor_l=None).  No knob, no default-precision object touched: the J
computation is verbatim, only provably-unread work is dropped (the
B2D default-on pure-engineering class).

Evidence in this file:
- value identity: J-only vj BITWISE == the full-route vj at the
  untagged spin-summed dm (the waste case), a tagged dm, hermi=0,
  and the 4-d TDDFT-shaped dm (the shared dm_sparse/cderi code path
  proves the skip cannot change the J values);
- the zero-dm early return preserved (exact-zero dm short-circuits;
  a nonzero dm never does);
- fail-closed engagement: decompose_rdm1_svd call count == 0 across a
  whole pre-built-cderi pure-functional UKS SCF (the eigh the parent
  sha paid per cycle), with a positive control that the counter
  itself fires on a direct untagged factorization;
- knob compositions: the K-side syrk and eager-plan-warmup knobs
  armed at a J-only entry stay inert (no counter bump, no crash, the
  J values bitwise unchanged).
'''
import os
import unittest

import numpy
import pyscf
import cupy as cp
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000'''

O2 = '''
O       0.0    0.0    0.0
O       0.0    0.0    1.2'''


def setUpModule():
    global mol
    mol = pyscf.M(atom=WATER, basis='def2-svp',
                  output='/dev/null', verbose=0)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _rand_dm(nao, nocc, hermi=1, seed=7):
    '''A deterministic untagged symmetric psd (or nonsymmetric) dm.'''
    rng = numpy.random.default_rng(seed)
    c = rng.standard_normal((nao, nocc))
    if hermi == 1:
        dm = 2.0 * c @ c.T
        dm = 0.5 * (dm + dm.T)
    else:
        dm = 2.0 * c @ rng.standard_normal((nocc, nao))
    return cp.asarray(dm)


class TestJOnlyValueIdentity(unittest.TestCase):
    '''The J-only skip is value-neutral by construction: the J values
    must be BITWISE identical to the full-route (with_k=True) vj of
    the same dm -- both routes share the dm_sparse gather and the
    cderi block contraction; the factorization is the only thing the
    skip removes.'''

    @classmethod
    def setUpClass(cls):
        cls.mf = dft.RKS(mol).density_fit()
        cls.mf.xc = 'pbe'
        cls.mf.with_df.build()
        cls.nao = mol.nao

    def _assert_identity(self, dm, hermi):
        vj_only, vk_only = df_jk.get_jk(
            self.mf.with_df, dm, hermi, with_j=True, with_k=False)
        vj_full, vk_full = df_jk.get_jk(
            self.mf.with_df, dm, hermi, with_j=True, with_k=True)
        self.assertIsNone(vk_only)
        self.assertIsNotNone(vk_full)
        self.assertTrue(bool(cp.array_equal(vj_only, vj_full)))
        self.assertEqual(vj_only.shape, dm.shape)

    def test_untagged_spinsum_hermi1(self):
        # the waste case: an untagged spin-summed dm, hermi=1 -- the
        # parent sha paid one discarded eigh per such call.
        dm = _rand_dm(self.nao, 5, hermi=1)
        self._assert_identity(dm, 1)

    def test_untagged_nonsym_hermi0(self):
        # the response/TDDFT-shaped untagged caller, hermi=0 (the
        # parent sha paid one discarded SVD per such call).
        dm = _rand_dm(self.nao, 5, hermi=0)
        self._assert_identity(dm, 0)

    def test_tagged_dm_hermi1(self):
        # the tagged pure-RKS branch shape (the parent fast path; the
        # skip must be value-neutral there too).
        dm = _rand_dm(self.nao, 5, hermi=1)
        nocc = 5
        mo_coeff = cp.asarray(numpy.random.default_rng(11)
                              .standard_normal((self.nao, nocc)))
        mo_occ = cp.ones(nocc)
        dm = df_jk.tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
        self._assert_identity(dm, 1)

    def test_4d_tddft_shape(self):
        # the 4-d (TDDFT/hessian) J-only entry: nspin is derived from
        # dms.ndim with the skip (the assert preserved), the J branch
        # runs the same dm_sparse path.
        dm2 = _rand_dm(self.nao, 5, hermi=1)
        dm = cp.asarray(numpy.stack([dm2.get(), dm2.get()]))[None]
        vj_only, vk_only = df_jk.get_jk(
            self.mf.with_df, dm, 1, with_j=True, with_k=False)
        vj_full, vk_full = df_jk.get_jk(
            self.mf.with_df, dm, 1, with_j=True, with_k=True)
        self.assertEqual(vj_only.shape, dm.shape)
        self.assertTrue(bool(cp.array_equal(vj_only, vj_full)))

    def test_zero_dm_early_return(self):
        # the exact-zero twin of the nocc==0 early return: an all-zero
        # dm short-circuits (vj is the zero input itself, same shape);
        # a nonzero dm never takes the shortcut.
        dm0 = cp.zeros((self.nao, self.nao))
        vj, vk = df_jk.get_jk(
            self.mf.with_df, dm0, 1, with_j=True, with_k=False)
        self.assertIs(vj, dm0)
        dm0_3d = cp.zeros((2, self.nao, self.nao))
        vj3, vk3 = df_jk.get_jk(
            self.mf.with_df, dm0_3d, 1, with_j=True, with_k=False)
        self.assertIs(vj3, dm0_3d)
        dm = _rand_dm(self.nao, 5, hermi=1)
        vj_nz, _ = df_jk.get_jk(
            self.mf.with_df, dm, 1, with_j=True, with_k=False)
        self.assertIsNot(vj_nz, dm)
        self.assertFalse(bool(cp.array_equal(vj_nz, dm)))


class TestPureUKSEngagement(unittest.TestCase):
    '''Fail-closed engagement: across a WHOLE pre-built-cderi
    pure-functional UKS SCF, decompose_rdm1_svd (the eigh/SVD entry
    the discarded factorization would take) must fire ZERO times.
    The SCF runs the steady-restart driver order verbatim:
    with_df.build() BEFORE kernel().'''

    def test_decompose_count_zero_at_pure_uks_scf(self):
        calls = [0]
        orig = df_jk.decompose_rdm1_svd
        def counted(dm, hermi=0):
            calls[0] += 1
            return orig(dm, hermi)
        df_jk.decompose_rdm1_svd = counted
        try:
            # positive control: a direct untagged factorization does
            # fire the counter (the parent sha's per-call route).
            df_jk.factorize_dm(_rand_dm(24, 4, hermi=1), 1)
            self.assertEqual(calls[0], 1)
            calls[0] = 0

            mol_o2 = pyscf.M(atom=O2, basis='def2-svp', spin=2,
                             output='/dev/null', verbose=0)
            mf = dft.UKS(mol_o2).density_fit()
            mf.xc = 'pbe'
            mf.conv_tol = 1e-10
            mf.grids.level = 2
            # the steady-driver order: the cderi is BUILT before the
            # kernel, so every J-only veff call enters the module
            # get_jk (not the df.py j_engine fast path).
            mf.with_df.build()
            mf.kernel()
            self.assertTrue(mf.converged)
            self.assertTrue(numpy.isfinite(mf.e_tot))
            self.assertTrue(mf.e_tot < 0.0)
            # the skip: the discarded per-cycle factorization is GONE
            self.assertEqual(calls[0], 0)
        finally:
            df_jk.decompose_rdm1_svd = orig


class TestKnobCompositions(unittest.TestCase):
    '''The K-side opt-in knobs composed with a J-only entry stay
    inert (their guards read factor_l/vk/buf, all None at the skip):
    no counter bump, no crash, J values bitwise unchanged.'''

    def test_syrk_and_warmup_armed_at_jonly(self):
        mfr = dft.RKS(mol).density_fit()
        mfr.xc = 'pbe'
        mfr.with_df.build()
        dm = _rand_dm(mol.nao, 5, hermi=1)
        vj_ref, _ = df_jk.get_jk(
            mfr.with_df, dm, 1, with_j=True, with_k=False)

        n_syrk_before = df_jk._SCF_JK_KACC_SYRK_CALLS
        env_syrk = os.environ.get('GPU4PYSCF_SCF_JK_KACC_SYRK')
        env_warm = os.environ.get('GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP')
        df_jk._SCF_JK_KACC_SYRK = None  # reset the cached parse
        try:
            os.environ['GPU4PYSCF_SCF_JK_KACC_SYRK'] = '1'
            os.environ['GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP'] = '1'
            # a fresh DF object so the warmup latch is not already set
            mf2 = dft.RKS(mol).density_fit()
            mf2.xc = 'pbe'
            mf2.with_df.build()
            vj_arm, vk_arm = df_jk.get_jk(
                mf2.with_df, dm, 1, with_j=True, with_k=False)
        finally:
            if env_syrk is None:
                os.environ.pop('GPU4PYSCF_SCF_JK_KACC_SYRK', None)
            else:
                os.environ['GPU4PYSCF_SCF_JK_KACC_SYRK'] = env_syrk
            if env_warm is None:
                os.environ.pop('GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP', None)
            else:
                os.environ['GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP'] = env_warm
            df_jk._SCF_JK_KACC_SYRK = None
        self.assertTrue(bool(cp.array_equal(vj_arm, vj_ref)))
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, n_syrk_before)
        self.assertIsNone(vk_arm)


if __name__ == '__main__':
    print('run via unittest')

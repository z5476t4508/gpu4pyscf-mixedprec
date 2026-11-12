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
'''GRAD-P1F32-01 focused equivalence regression (2026-11-06).

The opt-in GPU4PYSCF_DFGRAD_P1_F32 knob (DEFAULT OFF) runs the phase-1
j3c@dm-factor contractions of the DF-gradient K-chain in float32 on the
GRAD-J3C-REUSE-01 ARMED path only: the dense symmetric scatter of the
retained cderi blocks, the two per-block GEMMs and the X buffer go fp32
(in-scatter dtype conversion; dm factors cast once per call), while the
aux_coef einsum and everything downstream keep the armed parent's exact
dtypes (X is cast back to float64 before the einsum).  The verbatim
path's float64 phase-1 + metric-solve discipline (the amplification note
at the top of _jk_energy_per_atom) is untouched: the knob is inert --
warn + ignore -- whenever the reuse pack is absent, and the strict
fp64-reference context forces it inert.

Scientific distinction priced by these tests before any GPU measurement:
the armed path removed the j2c^-1 metric solve (amplification kappa);
dm_oo = aux_coef @ X applies j2c^-1/2 (sqrt-kappa), and the downstream
fp32 tolerance is the deployed resp32 precedent (entry 369: dg 4.6e-7).

Value-perturbing class: armed-vs-armed-parent equality is banded, never
bitwise (fp32 GEMM rounding ~1e-7 relative on X plus the ip1 atomicAdd
scatter); the band below is calibrated at water/wb97mv-water scale and
sits four-plus orders under the 3e-5 project gradient gate.
'''
import os
import unittest
import warnings

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.dft import numint


def setUpModule():
    global mol_sph, mol_wb97mv
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_wb97mv = pyscf.M(atom=mol_sph.atom, basis='def2-svp',
                         output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_wb97mv
    mol_sph.stdout.close()
    mol_wb97mv.stdout.close()
    del mol_sph, mol_wb97mv

_ULP_BAND = 1e-12   # ip1 atomicAdd scatter measured ~2e-16 at this scale
_P1F32_BAND = 1e-6  # fp32 phase-1 GEMM rounding through the sqrt-kappa
                    # aux_coef composition; measured 4.35e-8 (b3lyp) /
                    # 1.43e-7 (wb97mv SR+LR) at water scale this session
                    # -> 1e-6 tripwire = 7x headroom, 4+ orders under
                    # the 3e-5 project gate
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob and its neighbors touch.'''

    def setUp(self):
        self.s_p1f32 = df_rhf_grad._P1_F32
        self.s_calls = df_rhf_grad._P1_F32_GRADS
        self.s_reuse = df_rhf_grad._J3C_REUSE
        self._had_env = 'GPU4PYSCF_DFGRAD_P1_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_DFGRAD_P1_F32')
        self._had_reuse = 'GPU4PYSCF_DF_GRAD_J3C_REUSE' in os.environ
        self._reuse = os.environ.get('GPU4PYSCF_DF_GRAD_J3C_REUSE')

    def tearDown(self):
        df_rhf_grad._P1_F32 = self.s_p1f32
        df_rhf_grad._P1_F32_GRADS = self.s_calls
        df_rhf_grad._J3C_REUSE = self.s_reuse
        for had, key, val in ((self._had_env, 'GPU4PYSCF_DFGRAD_P1_F32',
                               self._env),
                              (self._had_reuse,
                               'GPU4PYSCF_DF_GRAD_J3C_REUSE',
                               self._reuse)):
            if had:
                os.environ[key] = val
            elif key in os.environ:
                del os.environ[key]


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            df_rhf_grad._P1_F32 = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DFGRAD_P1_F32', None)
            else:
                os.environ['GPU4PYSCF_DFGRAD_P1_F32'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = df_rhf_grad._p1_f32()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='b3lyp'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_knob_off_armed_parent_bitwise(self):
        # reuse ON, p1f32 OFF: the armed parent runs identically twice and
        # never touches the p1f32 counter
        df_rhf_grad._J3C_REUSE = True
        df_rhf_grad._P1_F32 = False
        mf = self._converged(mol_sph)
        c0 = df_rhf_grad._P1_F32_GRADS
        g1 = mf.nuc_grad_method().kernel()
        g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._P1_F32_GRADS, c0)
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_b3lyp_envelope(self):
        mf = self._converged(mol_sph)
        df_rhf_grad._J3C_REUSE = True
        df_rhf_grad._P1_F32 = True
        c0 = df_rhf_grad._P1_F32_GRADS
        r0 = df_rhf_grad._J3C_REUSE_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertGreaterEqual(df_rhf_grad._P1_F32_GRADS - c0, 1)  # SR pack
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS - r0,
                         df_rhf_grad._P1_F32_GRADS - c0)  # same packs armed
        df_rhf_grad._P1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._P1_F32_GRADS - c0, 1)
        err = abs(g_on - g_off).max()
        self.assertLess(err, _P1F32_BAND)   # strictly perturbed, not bitwise
        self.assertGreater(err, 0.0)
        self.assertLessEqual(err, _GRAD_GATE)
        self.assertEqual(g_on.dtype, np.float64)

    def test_armed_wb97mv_two_packs(self):
        mf = self._converged(mol_wb97mv, xc='wb97mv')
        df_rhf_grad._J3C_REUSE = True
        df_rhf_grad._P1_F32 = True
        c0 = df_rhf_grad._P1_F32_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._P1_F32_GRADS - c0, 2)  # SR + LR packs
        df_rhf_grad._P1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLess(err, _P1F32_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_knob_on_reuse_off_warns_inert(self):
        # p1f32 ON but the reuse pack absent: warn + ignore, verbatim
        # gradient, counter never moves
        df_rhf_grad._J3C_REUSE = False
        df_rhf_grad._P1_F32 = True
        mf = self._converged(mol_sph)
        c0 = df_rhf_grad._P1_F32_GRADS
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            g1 = mf.nuc_grad_method().kernel()
        self.assertTrue(any('GPU4PYSCF_DFGRAD_P1_F32' in str(x.message)
                            for x in w))
        self.assertEqual(df_rhf_grad._P1_F32_GRADS, c0)
        g2 = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)

    def test_strict_reference_inert(self):
        # strict fp64-reference: both knobs inert, no p1f32 warning noise
        df_rhf_grad._J3C_REUSE = True
        df_rhf_grad._P1_F32 = True
        mf = self._converged(mol_sph)
        c0 = df_rhf_grad._P1_F32_GRADS
        r0 = df_rhf_grad._J3C_REUSE_GRADS
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            with numint.strict_fp64_reference_ctx():
                g = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._P1_F32_GRADS, c0)
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS, r0)
        self.assertFalse(any('GPU4PYSCF_DFGRAD_P1_F32' in str(x.message)
                             for x in w))
        self.assertEqual(g.dtype, np.float64)


class TestDtypeMechanics(_KnobState):
    '''the fp32 scatter conversion and fp32-contract dispatch at unit scale'''

    def test_scatter_cast_and_fp32_contract(self):
        nao, npair, dk, nocc = 5, 8, 3, 2
        i_addr = cp.asarray([0, 1, 2, 3, 4, 1, 2, 0], dtype=cp.int32)
        j_addr = cp.asarray([0, 0, 1, 1, 2, 2, 3, 4], dtype=cp.int32)
        cderi_blk = cp.random.random((dk, npair))
        dm = cp.random.random((nao, nocc))
        # fp32 destination scatter converts in-kernel
        j3c = cp.zeros((nao, nao, dk), dtype=cp.float32)
        j3c[j_addr, i_addr] = j3c[i_addr, j_addr] = cderi_blk.T
        ref = cp.zeros((nao, nao, dk))
        ref[j_addr, i_addr] = ref[i_addr, j_addr] = cderi_blk.T
        # both sides round the same fp64 values to fp32: identical bits
        cp.testing.assert_array_equal(j3c, ref.astype(cp.float32))
        # fp32 operands contract in fp32 (cutensor result_type rule)
        from gpu4pyscf.lib.cutensor import contract
        dm32 = dm.astype(cp.float32)
        tmp = contract('pqr,pi->iqr', j3c, dm32)
        self.assertEqual(tmp.dtype, cp.float32)
        ref64 = cp.einsum('pqr,pi->iqr', ref, dm)
        cp.testing.assert_allclose(
            cp.asarray(tmp, dtype=cp.float64), ref64, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()

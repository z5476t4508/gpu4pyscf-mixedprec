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
'''GRAD-J3C-REUSE-01 focused equivalence regression (2026-11-06).

The opt-in GPU4PYSCF_DF_GRAD_J3C_REUSE knob (DEFAULT OFF) lets the
DF-gradient K-chain consume the SCF-built cderi retained across the
phase boundary -- dm_oo = (dm_factor_l . cderi . dm_factor_r) .
aux_coef^T, the exact associativity of the verbatim j3c re-evaluation +
fresh j2c metric route (differences: the _decompose_j2c discarded
subspace + fp64 reassociation rounding only) -- instead of re-evaluating
plain fp64 j3c blocks per aux batch and re-solving the metric.  These
tests precede any GPU measurement: parse mapping, knob-off identity
(counter-exact never-armed + ULP-band value equality), armed engagement
(SR holder and the wb97mv LR RSH-DF holder), strict fp64-reference
inertness, capture fail-closed classes, DF retention lifecycle, and the
dm_oo identity envelope at debug scale.

Bitwise note (same class as test_df_resp_f32_impl01): gradient calls
carry the ip1-kernel atomicAdd ULP run-to-run scatter (~2e-16 measured
at water scale), so armed-vs-verbatim equality is banded, not bitwise;
the band here is dominated by the reconstruction difference (target
<= 1e-9 Eh/Bohr at water scale, four orders under the 3e-5 project gate).
'''
import os
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.df import df as df_mod
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.dft import numint


def setUpModule():
    global mol_sph, mol_wb97mv, mol_o2
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_wb97mv = pyscf.M(atom=mol_sph.atom, basis='def2-svp',
                         output='/dev/null', verbose=0)
    mol_o2 = pyscf.M(atom='O 0 0 0; O 0 0 1.2078', basis='def2-svp',
                     spin=2, output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_wb97mv, mol_o2
    mol_sph.stdout.close()
    mol_wb97mv.stdout.close()
    mol_o2.stdout.close()
    del mol_sph, mol_wb97mv, mol_o2

_ULP_BAND = 1e-12      # ip1 atomicAdd scatter measured ~2e-16 at this scale
_REUSE_BAND = 1e-8     # reconstruction (reassociation + discarded subspace);
                       # measured <=1e-9 RKS water, 1.7e-9 UKS O2 wb97mv
_GRAD_GATE = 3e-5      # the project gradient gate (Eh/Bohr), the hard bound


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob and its neighbors touch.'''

    def setUp(self):
        self.s_reuse = df_rhf_grad._J3C_REUSE
        self.s_calls = df_rhf_grad._J3C_REUSE_GRADS
        self._had_env = 'GPU4PYSCF_DF_GRAD_J3C_REUSE' in os.environ
        self._env = os.environ.get('GPU4PYSCF_DF_GRAD_J3C_REUSE')

    def tearDown(self):
        df_rhf_grad._J3C_REUSE = self.s_reuse
        df_rhf_grad._J3C_REUSE_GRADS = self.s_calls
        if self._had_env:
            os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = self._env
        elif 'GPU4PYSCF_DF_GRAD_J3C_REUSE' in os.environ:
            del os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE']


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            df_rhf_grad._J3C_REUSE = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_J3C_REUSE', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = df_rhf_grad._j3c_reuse()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='b3lyp'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_knob_off_verbatim(self):
        df_rhf_grad._J3C_REUSE = False
        mf = self._converged(mol_sph)
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        g1 = mf.nuc_grad_method().kernel()
        g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS, c0)
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_rhf_envelope(self):
        mf = self._converged(mol_sph)
        df_rhf_grad._J3C_REUSE = True
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        g_on = mf.nuc_grad_method().kernel()
        n_engaged = df_rhf_grad._J3C_REUSE_GRADS - c0
        self.assertGreaterEqual(n_engaged, 1)   # SR pack at minimum
        df_rhf_grad._J3C_REUSE = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS - c0, n_engaged)
        err = abs(g_on - g_off).max()
        self.assertLessEqual(err, _REUSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_armed_rks_wb97mv_lr_holder(self):
        mf = self._converged(mol_wb97mv, xc='wb97mv')
        wdf = mf.with_df
        self.assertIsNotNone(wdf._cderi)             # SR pack
        self.assertTrue(wdf._rsh_df)                 # LR RSH-DF object exists
        lr_holder = next(iter(wdf._rsh_df.values()))
        self.assertIsNotNone(lr_holder._cderi)
        # both holder branches capture on the post-SCF state
        omega_key = next(iter(wdf._rsh_df))
        omega_lr = float(omega_key)
        pack_sr = df_rhf_grad._capture_grad_j3c_reuse(mf, None)
        pack_lr = df_rhf_grad._capture_grad_j3c_reuse(mf, omega_lr)
        self.assertIsNotNone(pack_sr)
        self.assertIsNotNone(pack_lr)
        self.assertIs(pack_lr[0], lr_holder._cderi[0])
        df_rhf_grad._J3C_REUSE = True
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertGreaterEqual(df_rhf_grad._J3C_REUSE_GRADS - c0, 1)
        # the retained pack survives the armed call (no reset happened)
        self.assertIsNotNone(wdf._cderi)
        self.assertIsNotNone(lr_holder._cderi)
        df_rhf_grad._J3C_REUSE = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLessEqual(err, _REUSE_BAND)

    def test_strict_reference_inert(self):
        mf = self._converged(mol_sph)
        df_rhf_grad._J3C_REUSE = True
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        with numint.strict_fp64_reference_ctx():
            g = mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS, c0)
        self.assertEqual(g.dtype, np.float64)


class TestUKSTwin(_KnobState):
    '''GRAD-J3C-REUSE-01 UKS twin (uhf.py): spin-stacked reuse branch.'''

    def _converged_uks(self, xc='wb97mv'):
        mf = dft.UKS(mol_o2, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_armed_uks_envelope(self):
        mf = self._converged_uks()
        df_rhf_grad._J3C_REUSE = True
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertGreaterEqual(df_rhf_grad._J3C_REUSE_GRADS - c0, 1)
        self.assertIsNotNone(mf.with_df._cderi)   # armed: no reset
        df_rhf_grad._J3C_REUSE = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLessEqual(err, _REUSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_knob_off_uks_counter(self):
        mf = self._converged_uks('b3lyp')
        df_rhf_grad._J3C_REUSE = False
        c0 = df_rhf_grad._J3C_REUSE_GRADS
        mf.nuc_grad_method().kernel()
        self.assertEqual(df_rhf_grad._J3C_REUSE_GRADS, c0)


class TestCaptureFailClosed(_KnobState):
    '''_capture_grad_j3c_reuse inert classes on synthetic holders.'''

    def _mf(self, wdf):
        return SimpleNamespace(with_df=wdf, mol=mol_sph)

    def _good_pack(self):
        nao = mol_sph.nao
        nF, npair, naux = 3, 5, 4
        pair = cp.arange(npair, dtype=cp.int32)
        cderi = cp.zeros((nF, npair))
        aux_coef = cp.zeros((naux, nF))
        wdf = SimpleNamespace(
            _cderi=[cderi], _cderi_aux_coef=aux_coef,
            _cderi_idx=(pair, None), _rsh_df={})
        return wdf, cderi, pair, aux_coef

    def test_armed_shape_passes(self):
        wdf, cderi, pair, aux_coef = self._good_pack()
        pack = df_rhf_grad._capture_grad_j3c_reuse(self._mf(wdf), None)
        self.assertIsNotNone(pack)
        self.assertIs(pack[0], cderi)
        self.assertIs(pack[2], aux_coef)
        self.assertEqual(pack[3], mol_sph.nao)

    def test_inert_classes(self):
        # no with_df
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(
            SimpleNamespace(mol=mol_sph), None))
        wdf, cderi, pair, aux_coef = self._good_pack()
        # cderi missing
        w = SimpleNamespace(**vars(wdf)); w._cderi = None
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # aux_coef missing (pre-change DF objects)
        w = SimpleNamespace(**vars(wdf)); w._cderi_aux_coef = None
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # host-resident cderi
        w = SimpleNamespace(**vars(wdf))
        w._cderi = [np.zeros((3, 5))]
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # fp32-stored cderi
        w = SimpleNamespace(**vars(wdf))
        w._cderi = [cp.zeros((3, 5), dtype=cp.float32)]
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # row/pair shape mismatch
        w = SimpleNamespace(**vars(wdf))
        w._cderi_idx = (cp.arange(6, dtype=cp.int32), None)
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # aux_coef/F mismatch
        w = SimpleNamespace(**vars(wdf))
        w._cderi_aux_coef = cp.zeros((4, 4))
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # addresses outside nao^2
        w = SimpleNamespace(**vars(wdf))
        w._cderi_idx = (cp.asarray(
            [mol_sph.nao*mol_sph.nao, 1, 2, 3, 4], dtype=cp.int32), None)
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(self._mf(w), None))
        # omega>0 with no RSH-DF object
        self.assertIsNone(df_rhf_grad._capture_grad_j3c_reuse(
            self._mf(wdf), 0.3))


class TestDFRetentionLifecycle(_KnobState):

    def test_build_retains_reset_clears(self):
        wdf = df_mod.DF(mol_sph, 'def2-universal-jkfit')
        self.assertIsNone(wdf._cderi_aux_coef)
        wdf.build()
        self.assertIsNotNone(wdf._cderi_aux_coef)
        cderi = wdf._cderi[0] if isinstance(wdf._cderi, list) else wdf._cderi
        self.assertEqual(wdf._cderi_aux_coef.shape[1], cderi.shape[0])
        self.assertEqual(wdf._cderi_aux_coef.dtype, cp.float64)
        wdf.reset()
        self.assertIsNone(wdf._cderi_aux_coef)
        self.assertIsNone(wdf._cderi)


if __name__ == '__main__':
    unittest.main()

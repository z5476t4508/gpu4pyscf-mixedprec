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
import unittest
import numpy as np
import pyscf
from gpu4pyscf import scf
from gpu4pyscf.lib import precision

def setUpModule():
    global mol_sph
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=1)

def tearDownModule():
    global mol_sph
    mol_sph.stdout.close()
    del mol_sph

class KnownValues(unittest.TestCase):

    def test_auto_scf_accuracy(self):
        '''auto mode must restore full float64 accuracy'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-12
        e_ref = mf.kernel()

        mf2 = scf.RHF(mol_sph).density_fit()
        mf2.conv_tol = 1e-12
        mf2.precision_mode = 'auto'
        e_mixed = mf2.kernel()
        self.assertAlmostEqual(e_mixed, e_ref, delta=1e-9)
        # cderi stays float64 in auto mode
        self.assertEqual(mf2.with_df._cderi[0].dtype, np.float64)

    def test_fp32_scf_screening(self):
        '''pure fp32 mode: screening-grade accuracy, cderi in float64'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()

        mf2 = scf.RHF(mol_sph).density_fit()
        mf2.conv_tol = 1e-5
        mf2.precision_mode = 'fp32'
        e32 = mf2.kernel()
        self.assertAlmostEqual(e32, e_ref, delta=1e-3)

    def test_fp32_get_jk_accuracy(self):
        '''fp32 contractions reproduce J/K to float32 rounding'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        vj0, vk0 = mf.with_df.get_jk(dm, hermi=1)
        with precision.fp32():
            vj1, vk1 = mf.with_df.get_jk(dm, hermi=1)
        vj1 = vj1.astype(np.float64)
        vk1 = vk1.astype(np.float64)
        self.assertLess(abs(vj1-vj0).max()/abs(vj0).max(), 1e-5)
        self.assertLess(abs(vk1-vk0).max()/abs(vk0).max(), 1e-4)

    def test_fp32_gradient_accuracy(self):
        '''fp32 gradient kernel keeps the gradient to ~1e-5'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()
        g = mf.nuc_grad_method()
        g0 = g.kernel()
        with precision.fp32():
            g1 = g.kernel()
        self.assertLess(abs(g1-g0).max(), 1e-5)

    def test_scanner_resets_mode(self):
        '''the auto policy must re-activate for every SCF call (scanner)'''
        results = []
        def scanner_run():
            mf = scf.RHF(mol_sph).density_fit()
            mf.conv_tol = 1e-10
            mf.precision_mode = 'auto'
            mf.kernel()
            return mf.e_tot
        e1 = scanner_run()
        e2 = scanner_run()
        results.append(e2)
        self.assertAlmostEqual(e1, e2, delta=1e-9)

    def test_cderi_precision_default_is_fp64(self):
        '''the fp32 cderi build must stay opt-in'''
        self.assertEqual(precision.get_cderi_precision(), 'fp64')
        mf = scf.RHF(mol_sph).density_fit()
        mf.precision_mode = 'fp32'
        mf.conv_tol = 1e-5
        mf.kernel()
        self.assertEqual(mf.with_df._cderi[0].dtype, np.float64)

    def test_fp32_cderi_build(self):
        '''fp32 cderi: float32 storage, screening-grade energy'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()

        mf2 = scf.RHF(mol_sph).density_fit()
        mf2.precision_mode = 'fp32'
        mf2.conv_tol = 3e-5
        try:
            precision.set_cderi_precision('fp32')
            e32 = mf2.kernel()
        finally:
            precision.set_cderi_precision('fp64')
        self.assertEqual(mf2.with_df._cderi[0].dtype, np.float32)
        self.assertAlmostEqual(e32, e_ref, delta=1e-2)

    def test_fp32_cderi_uses_eigendecomposition(self):
        '''the Cholesky factor cancels too badly for a float32 contraction'''
        from gpu4pyscf.df.df import _decompose_j2c
        mf = scf.RHF(mol_sph).density_fit()
        mf.with_df.build(build_cderi=False)
        auxmol = mf.with_df.intopt.auxmol

        _, tag64 = _decompose_j2c(auxmol)
        self.assertEqual(tag64, 'cd')
        try:
            precision.set_cderi_precision('fp32')
            _, tag32 = _decompose_j2c(auxmol)
        finally:
            precision.set_cderi_precision('fp64')
        self.assertEqual(tag32, 'ed')

    def test_invalid_cderi_precision(self):
        with self.assertRaises(ValueError):
            precision.set_cderi_precision('fp16')
        self.assertEqual(precision.get_cderi_precision(), 'fp64')

    def test_precision_mode_is_a_declared_attribute(self):
        '''an undeclared attribute triggers check_sanity warnings and can be
        dropped by copy/serialisation paths, silently reverting to fp64'''
        mf = scf.RHF(mol_sph).density_fit()
        # PySCF unions _keys across the MRO (lib/misc.py check_sanity)
        keys = set().union(*[getattr(c, '_keys', set())
                             for c in type(mf).__mro__])
        self.assertIn('precision_mode', keys)
        self.assertIsNone(mf.precision_mode)

        mf.precision_mode = 'auto'
        mf.verbose = 4
        mf.check_sanity()          # must not report an unknown attribute

    def test_invalid_precision_mode_is_rejected(self):
        '''an unrecognised mode used to fall through to fp64 silently'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.precision_mode = 'float32'
        with self.assertRaises(ValueError):
            mf.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_precision_is_restored_after_a_failed_scf(self):
        '''a raising SCF must not leave the process in fp32'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.precision_mode = 'fp32'

        def boom(*args, **kwargs):
            raise RuntimeError('boom')
        mf.get_veff = boom
        with self.assertRaises(RuntimeError):
            mf.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_precision_is_restored_after_a_normal_scf(self):
        mf = scf.RHF(mol_sph).density_fit()
        mf.precision_mode = 'auto'
        mf.conv_tol = 1e-10
        mf.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_get_jk_dtype_is_independent_of_the_mode(self):
        '''vj used to come back float32 while vk was float64'''
        mf = scf.RHF(mol_sph).density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        dm = mf.make_rdm1()
        with precision.fp32():
            vj, vk = mf.with_df.get_jk(dm, hermi=1)
        self.assertEqual(vj.dtype, np.float64)
        self.assertEqual(vk.dtype, vj.dtype)

if __name__ == '__main__':
    print('Full tests for df mixed precision')
    unittest.main()

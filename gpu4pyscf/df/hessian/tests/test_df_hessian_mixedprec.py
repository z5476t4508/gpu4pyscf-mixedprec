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

from gpu4pyscf import dft
from gpu4pyscf import scf
from gpu4pyscf.lib import precision


def setUpModule():
    global mol, mf, hess_ref
    mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                  basis='def2-svp', output='/dev/null', verbose=1)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    hess_ref = mf.Hessian().kernel()


def tearDownModule():
    global mol, mf, hess_ref
    mol.stdout.close()
    del mol, mf, hess_ref


class KnownValues(unittest.TestCase):

    def tearDown(self):
        mf.precision_mode = None
        precision.set_precision('fp64')

    def test_auto_selects_fp32(self):
        '''a Hessian has no accuracy left to recover by spending float64 on
        the CPHF solve, so 'auto' means float32 here -- as it does on a
        gradient'''
        mf.precision_mode = 'auto'
        self.assertEqual(mf.Hessian()._resolve_precision(), 'fp32')

    def test_hessian_object_overrides_the_mean_field_setting(self):
        mf.precision_mode = 'auto'
        h = mf.Hessian()
        h.precision_mode = 'fp64'
        self.assertEqual(h._resolve_precision(), 'fp64')

    def test_rejects_an_unknown_mode(self):
        h = mf.Hessian()
        h.precision_mode = 'half'
        self.assertRaises(ValueError, h.kernel)

    def test_fp32_cphf_accuracy(self):
        '''the float32 CPHF J/K must stay far inside anything a frequency
        calculation would notice'''
        mf.precision_mode = 'auto'
        hess = mf.Hessian().kernel()
        self.assertLess(np.abs(hess - hess_ref).max(), 1e-6)

        n = 3 * mol.natm
        eig = np.linalg.eigvalsh(hess.transpose(0, 2, 1, 3).reshape(n, n))
        eig_ref = np.linalg.eigvalsh(
            hess_ref.transpose(0, 2, 1, 3).reshape(n, n))
        self.assertLess(np.abs(eig - eig_ref).max(), 1e-6)

    def test_default_lane_is_unchanged(self):
        '''an unset precision_mode must leave the float64 path alone'''
        hess = mf.Hessian().kernel()
        self.assertLess(np.abs(hess - hess_ref).max(), 1e-10)

    def test_global_mode_is_restored(self):
        '''kernel() mutates process-global state; a raise must not leak it'''
        mf.precision_mode = 'fp32'
        h = mf.Hessian()
        h.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

        h.solve_mo1 = lambda *a, **kw: 1 / 0
        self.assertRaises(ZeroDivisionError, h.kernel)
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_uhf_fp32_cphf(self):
        '''the CPHF J/K is shared with UHF, which uses the nspin=2 branch'''
        umf = scf.UHF(mol).density_fit()
        umf.conv_tol = 1e-12
        umf.kernel()
        ref = umf.Hessian().kernel()

        umf.precision_mode = 'auto'
        self.assertLess(np.abs(umf.Hessian().kernel() - ref).max(), 1e-6)

    def test_rks_fp32_cphf_response(self):
        '''_nr_rks_fxc_mo_task carries the CPHF's XC response and branches on
        the functional type -- LDA, GGA and MGGA each take a different path,
        and a hybrid additionally exercises the float32 K build'''
        for xc in ('LDA,VWN', 'PBE', 'r2scan', 'B3LYP'):
            with self.subTest(xc=xc):
                rks = dft.RKS(mol, xc=xc).density_fit()
                rks.conv_tol = 1e-12
                rks.kernel()
                ref = rks.Hessian().kernel()

                rks.precision_mode = 'auto'
                hess = rks.Hessian().kernel()
                self.assertLess(np.abs(hess - ref).max(), 1e-6)

                n = 3 * mol.natm
                eig = np.linalg.eigvalsh(hess.transpose(0, 2, 1, 3).reshape(n, n))
                eig_ref = np.linalg.eigvalsh(
                    ref.transpose(0, 2, 1, 3).reshape(n, n))
                self.assertLess(np.abs(eig - eig_ref).max(), 1e-6)


if __name__ == '__main__':
    print('Full tests for the mixed-precision DF Hessian')
    unittest.main()

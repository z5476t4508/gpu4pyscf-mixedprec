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


HARTREE2WAVENUMBER = 219474.6313632


def frequencies(hess):
    '''harmonic frequencies in cm^-1, sign kept for imaginary modes'''
    mass = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -.5
    n = 3 * mol.natm
    h = hess.transpose(0, 2, 1, 3).reshape(n, n) * mass[:, None] * mass[None, :]
    ev = np.linalg.eigvalsh(h) / 1822.888486209
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER


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
        '''The DFT lane covers make_h1's grid loop as well as the CPHF, and
        each functional type takes a different branch of both: LDA, GGA and
        MGGA differ, and a hybrid additionally exercises the float32 K build.

        The bound is on frequencies rather than on Hessian elements. Elements
        are a proxy, and a misleading one here: the float32 make_h1 moves
        max|dH| on this molecule to ~1e-5 while the frequencies it produces
        move by at most 0.031 cm^-1 (PBE), and on Tamoxifen/def2-SVP -- 57
        atoms, where it is worth 4.05x -- by 0.001 cm^-1. 0.1 cm^-1 leaves
        3x margin over the worst measured value and still catches a real
        regression: bracketing partial_hess_elec too, which was rejected,
        moved Tamoxifen by 0.131-0.181 cm^-1.
        '''
        for xc in ('LDA,VWN', 'PBE', 'r2scan', 'B3LYP'):
            with self.subTest(xc=xc):
                rks = dft.RKS(mol, xc=xc).density_fit()
                rks.conv_tol = 1e-12
                rks.kernel()
                ref = rks.Hessian().kernel()

                rks.precision_mode = 'auto'
                hess = rks.Hessian().kernel()

                vib = slice(6, None)
                nu = frequencies(hess)[vib]
                nu_ref = frequencies(ref)[vib]
                self.assertLess(np.abs(nu - nu_ref).max(), 0.1)
                # a loose element bound, to catch gross breakage rather than
                # benign precision loss
                self.assertLess(np.abs(hess - ref).max(), 1e-4)


if __name__ == '__main__':
    print('Full tests for the mixed-precision DF Hessian')
    unittest.main()

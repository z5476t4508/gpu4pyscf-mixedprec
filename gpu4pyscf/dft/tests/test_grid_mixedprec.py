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
'''Mixed precision on the DFT quadrature grid.

Unlike the DF J/K contractions, the grid work carries essentially no
cancellation (rho is a sum of like-signed terms), so float32 keeps ~1e-7
relative accuracy -- below the quadrature error of the grid itself.
'''
import unittest
import numpy as np
import cupy
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.lib import precision

def setUpModule():
    global mol, mol_open
    mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                  basis='def2-svp', output='/dev/null', verbose=1)
    mol_open = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                       basis='def2-svp', charge=1, spin=1,
                       output='/dev/null', verbose=1)

def tearDownModule():
    global mol, mol_open
    mol.stdout.close()
    mol_open.stdout.close()
    del mol, mol_open


class KnownValues(unittest.TestCase):

    def test_contract_rho_f32_matches_f64(self):
        '''the float32 kernel must agree with the double one to fp32 rounding'''
        rng = cupy.random.default_rng(0)
        bra = cupy.asarray(rng.standard_normal((40, 2048)))
        ket = cupy.asarray(rng.standard_normal((40, 2048)))
        ref = numint._contract_rho(bra, ket)

        out = cupy.empty(2048)
        got = numint._contract_rho(bra.astype(cupy.float32),
                                   ket.astype(cupy.float32), rho=out)
        self.assertEqual(got.dtype, np.float64)
        self.assertLess(float(cupy.abs(got-ref).max()/cupy.abs(ref).max()), 1e-5)

    def test_scale_ao_rejects_the_double_kernel_for_fp32(self):
        '''GDFTscale_ao takes an is_real flag, not a dtype: float32 input used
        to be read as complex128 and silently corrupted'''
        rng = cupy.random.default_rng(0)
        ao = cupy.asarray(rng.standard_normal((4, 30, 512)))
        wv = cupy.asarray(rng.standard_normal((4, 512)))
        ref = numint._scale_ao(ao, wv)
        got = numint._scale_ao(ao.astype(cupy.float32), wv.astype(cupy.float32))
        self.assertEqual(got.dtype, np.float32)
        self.assertLess(float(cupy.abs(got.astype(cupy.float64)-ref).max()
                              / cupy.abs(ref).max()), 1e-5)

    def test_grid_default_stays_float64(self):
        mf = dft.RKS(mol, xc='pbe').density_fit()
        mf.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_fp32_grid_accuracy_mgga(self):
        '''meta-GGA: the fp32 grid must stay far inside the quadrature error'''
        mf = dft.RKS(mol, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()
        dm = mf.make_rdm1()
        ni = mf._numint
        n0, exc0, v0 = ni.nr_rks(mol, mf.grids, 'r2scan', dm)
        with precision.fp32():
            n1, exc1, v1 = ni.nr_rks(mol, mf.grids, 'r2scan', dm)
        self.assertLess(abs(n1-n0), 1e-5)
        self.assertLess(abs(exc1-exc0), 1e-4)
        self.assertLess(float(cupy.abs(v1-v0).max()), 1e-4)
        self.assertEqual(v1.dtype, np.float64)

    def test_fp32_grid_accuracy_gga(self):
        mf = dft.RKS(mol, xc='pbe').density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        dm = mf.make_rdm1()
        ni = mf._numint
        n0, exc0, v0 = ni.nr_rks(mol, mf.grids, 'pbe', dm)
        with precision.fp32():
            n1, exc1, v1 = ni.nr_rks(mol, mf.grids, 'pbe', dm)
        self.assertLess(abs(n1-n0), 1e-5)
        self.assertLess(abs(exc1-exc0), 1e-4)
        self.assertEqual(v1.dtype, np.float64)

    def test_auto_recovers_float64_energy_mgga(self):
        '''the auto policy must land on the float64 answer for r2SCAN'''
        mf = dft.RKS(mol, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()

        mf2 = dft.RKS(mol, xc='r2scan').density_fit()
        mf2.conv_tol = 1e-10
        mf2.precision_mode = 'auto'
        e_auto = mf2.kernel()
        self.assertAlmostEqual(e_auto, e_ref, delta=1e-8)

    def test_auto_recovers_float64_energy_gga(self):
        mf = dft.RKS(mol, xc='pbe').density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()

        mf2 = dft.RKS(mol, xc='pbe').density_fit()
        mf2.conv_tol = 1e-10
        mf2.precision_mode = 'auto'
        e_auto = mf2.kernel()
        self.assertAlmostEqual(e_auto, e_ref, delta=1e-8)

    def test_uks_fp32_grid_accuracy(self):
        '''the open-shell grid path must match too (alpha/beta separately)'''
        mf = dft.UKS(mol_open, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        dm = mf.make_rdm1()
        ni = mf._numint
        n0, exc0, v0 = ni.nr_uks(mol_open, mf.grids, 'r2scan', dm)
        with precision.fp32():
            n1, exc1, v1 = ni.nr_uks(mol_open, mf.grids, 'r2scan', dm)
        self.assertLess(float(abs(np.asarray(n1)-np.asarray(n0)).max()), 1e-5)
        self.assertLess(float(abs(np.asarray(exc1)-np.asarray(exc0)).max()), 1e-4)
        self.assertLess(float(cupy.abs(v1[0]-v0[0]).max()), 1e-4)
        self.assertLess(float(cupy.abs(v1[1]-v0[1]).max()), 1e-4)
        self.assertEqual(v1[0].dtype, np.float64)

    def test_uks_auto_recovers_float64_energy(self):
        mf = dft.UKS(mol_open, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        e_ref = mf.kernel()

        mf2 = dft.UKS(mol_open, xc='r2scan').density_fit()
        mf2.conv_tol = 1e-10
        mf2.precision_mode = 'auto'
        e_auto = mf2.kernel()
        self.assertAlmostEqual(e_auto, e_ref, delta=1e-8)

    def test_fp32_gradient_accuracy(self):
        '''the fp32 grid gradient must stay far under a geometry optimiser's
        convergence threshold (geomeTRIC's default is 3e-4 Eh/Bohr)'''
        mf = dft.RKS(mol, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        g = mf.nuc_grad_method()
        g.auxbasis_response = True
        ref = g.kernel()
        with precision.fp32():
            got = g.kernel()
        self.assertLess(float(np.abs(got-ref).max()), 1e-5)

    def test_gradient_default_stays_float64(self):
        '''an SCF run under auto must leave the gradient in float64'''
        mf = dft.RKS(mol, xc='r2scan').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto'
        mf.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')


if __name__ == '__main__':
    print('Tests for mixed precision on the DFT grid')
    unittest.main()

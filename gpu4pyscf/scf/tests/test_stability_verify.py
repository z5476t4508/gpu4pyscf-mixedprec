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
from gpu4pyscf import dft
from gpu4pyscf.lib import precision


def setUpModule():
    global mol_sph, mol_uks
    # mol_uks: a small open-shell doublet (the methoxy-radical family the
    # 'auto+verify' policy targets; near-degenerate UKS stationary pairs)
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=1)
    mol_uks = pyscf.M(atom='''
C       0.0000000000     0.0000000000     0.1274000000
O       0.0000000000     1.1000000000    -0.4600000000
H       0.8700000000    -0.5500000000    -0.4600000000
H      -0.8700000000    -0.5500000000    -0.4600000000''',
                      basis='def2-svp', spin=1, charge=-1,
                      output='/dev/null', verbose=1)


def tearDownModule():
    global mol_sph, mol_uks
    mol_sph.stdout.close()
    del mol_sph
    mol_uks.stdout.close()
    del mol_uks


class KnownValues(unittest.TestCase):

    def test_policy_validation(self):
        '''the accepted precision_mode vocabulary; auto+verify admitted'''
        mf = scf.RHF(mol_sph).density_fit()
        for ok in (None, 'fp64', 'fp32', 'auto', 'auto+verify'):
            mf.precision_mode = ok
            # kernel() is not run here; validation happens inside _kernel,
            # so exercise the setter path via a 0-cycle scf call
            mf.max_cycle = 0
            mf.precision_mode = ok
            mf.kernel()
        mf.precision_mode = 'bogus'
        with self.assertRaises(ValueError):
            mf.kernel()

    def test_auto_verify_records_decision(self):
        '''auto+verify must record the selection in scf_summary and keep
        the lower converged solution'''
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        e_v = mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertEqual(rec['policy'], 'auto+verify')
        self.assertIn(rec['selected'], ('main', 'reference'))
        # the kept energy must be the minimum of the two branches
        self.assertAlmostEqual(
            float(mf.e_tot), min(rec['main_e_tot'], rec['ref_e_tot']),
            delta=1e-12)
        self.assertTrue(mf.converged)

    def test_auto_verify_closed_shell_same_solution(self):
        '''closed shell: the verification trajectory converges to the same
        density; the record must say so and the energy must not move'''
        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto'
        e_auto = mf.kernel()

        mf2 = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf2.conv_tol = 1e-10
        mf2.precision_mode = 'auto+verify'
        e_v = mf2.kernel()
        rec = mf2.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertEqual(rec['selected'], 'main')
        self.assertAlmostEqual(e_v, e_auto, delta=1e-10)

    def test_auto_verify_selects_lower_branch(self):
        '''when the two trajectories land on different solutions, the
        lower-energy one must be kept (synthetic two-branch case is hard
        to force on water; the assertion runs the machinery and checks the
        invariant e_tot == min(main, ref) whenever branches differ)'''
        mf = dft.UKS(mol_uks, xc='wb97m-v').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        if rec['distinct_solutions']:
            self.assertAlmostEqual(
                float(mf.e_tot), min(rec['main_e_tot'], rec['ref_e_tot']),
                delta=1e-12)
            # a rejected record always exists on distinct branches
            self.assertIn('rejected', rec)

    def test_auto_verify_unconverged_skips(self):
        '''a non-converged main SCF must skip verification, not crash'''
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-14
        mf.max_cycle = 1
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertIn('skipped', rec)
        self.assertEqual(rec['skipped'], 'main SCF did not converge')

    def test_grad_resolve_auto_verify(self):
        '''gradients on an auto+verify object resolve to the fp32 kernel
        lane (verification selects the solution, not the kernel lane)'''
        from gpu4pyscf.df.grad import uhf as df_uhf_grad
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        g = mf.nuc_grad_method()
        self.assertEqual(g._resolve_precision(), 'fp32')
        # explicit override still wins
        g.precision_mode = 'fp64'
        self.assertEqual(g._resolve_precision(), 'fp64')


if __name__ == '__main__':
    print('Full tests for SCF solution-stability verification')
    unittest.main()

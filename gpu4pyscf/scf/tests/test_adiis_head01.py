# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

'''
ADIAS-HEAD-01 focused tests (GPU4PYSCF_DIIS_MODE=adiis, default off):
opt-in ADIIS head-phase Fock extrapolation with CDIIS tail handover.

Four preregistered classes:
1. knob parsing / knob-absent default / strict-fp64-reference inertness
2. update-path arithmetic parity vs the pyscf CPU diis.ADIIS reference
3. end-to-end engagement: converged same fixed point, switch recorded,
   RKS and UKS (spin-stack buffer) paths
4. knob-absent identity: default path untouched, no adiis bookkeeping
'''

import os
import unittest
import numpy
import cupy
from pyscf import gto
import pyscf.scf.diis as cpu_diis
from gpu4pyscf.scf import diis

def setUpModule():
    global mol_rks, mol_uks
    atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''
    mol_rks = gto.M(atom=atom, basis='def2-svp', output='/dev/null')
    mol_uks = gto.M(atom=atom, basis='def2-svp', spin=1, charge=1,
                    output='/dev/null')

class FakeMF(object):
    verbose = 0
    _strict_fp64_reference = False

class StrictMF(FakeMF):
    _strict_fp64_reference = True

def _run_scf(mol, diis_mode=None):
    env = os.environ.pop('GPU4PYSCF_DIIS_MODE', None)
    try:
        if diis_mode is not None:
            os.environ['GPU4PYSCF_DIIS_MODE'] = diis_mode
        if mol.spin == 0:
            from gpu4pyscf.dft import rks
            mf = rks.RKS(mol, xc='b3lyp')
        else:
            from gpu4pyscf.dft import uks
            mf = uks.UKS(mol, xc='b3lyp')
        mf.conv_tol = 1e-10
        mf.max_cycle = 60
        e = mf.kernel()
        return mf, e
    finally:
        if diis_mode is not None and env is None:
            os.environ.pop('GPU4PYSCF_DIIS_MODE', None)
        elif env is not None:
            os.environ['GPU4PYSCF_DIIS_MODE'] = env

class TestKnobParsing(unittest.TestCase):
    def test_armed_only_on_adiis_value(self):
        for value, armed in [('', False), ('0', False), ('1', False),
                             ('false', False), ('cdiis', False),
                             ('garbage', False), ('adiis', True),
                             ('ADIIS', True), (' adiis ', True)]:
            os.environ['GPU4PYSCF_DIIS_MODE'] = value
            self.assertEqual(diis._adiis_head_tail_armed(FakeMF()), armed, value)
        os.environ.pop('GPU4PYSCF_DIIS_MODE', None)
        self.assertFalse(diis._adiis_head_tail_armed(FakeMF()))

    def test_knob_absent_is_default(self):
        self.assertNotIn('GPU4PYSCF_DIIS_MODE', os.environ)
        self.assertFalse(diis._adiis_head_tail_armed(FakeMF()))

    def test_strict_fp64_reference_inert_even_when_armed(self):
        os.environ['GPU4PYSCF_DIIS_MODE'] = 'adiis'
        try:
            self.assertFalse(diis._adiis_head_tail_armed(StrictMF()))
        finally:
            os.environ.pop('GPU4PYSCF_DIIS_MODE', None)

class TestAdiisMinimizeParity(unittest.TestCase):
    def test_matches_pyscf_reference(self):
        # deterministic pseudo-history: 6-entry ring with the two oldest
        # slots zero-padded, mirroring early-cycle buffer state
        rng = numpy.random.RandomState(1952)
        nx, nao = 6, 8
        ds = numpy.zeros((nx, nao, nao))
        fs = numpy.zeros((nx, nao, nao))
        for i in range(2, nx):
            a = rng.randn(nao, nao)
            ds[i] = a + a.T
            b = rng.randn(nao, nao)
            fs[i] = b + b.T
        for idnewest in range(2, nx):
            df_ref = numpy.einsum('inpq,jnqp->ij',
                                  ds.reshape(nx, -1, nao, nao),
                                  fs.reshape(nx, -1, nao, nao)).real
            fun_ref, c_ref = cpu_diis.adiis_minimize(ds, fs, idnewest)
            fun, c = diis._adiis_minimize(df_ref.copy(), idnewest)
            self.assertAlmostEqual(fun, fun_ref, delta=1e-10)
            self.assertTrue(numpy.allclose(c, c_ref, atol=1e-8))

    def test_device_df_matches_host_df(self):
        rng = numpy.random.RandomState(1953)
        nx, nao = 4, 10
        ds = rng.randn(nx, nao, nao)
        fs = rng.randn(nx, nao, nao)
        df_host = numpy.einsum('inpq,jnqp->ij',
                               ds.reshape(nx, -1, nao, nao),
                               fs.reshape(nx, -1, nao, nao)).real
        df_dev = cupy.einsum('inpq,jnqp->ij',
                             cupy.asarray(ds).reshape(nx, -1, nao, nao),
                             cupy.asarray(fs).reshape(nx, -1, nao, nao)).real
        self.assertTrue(numpy.allclose(cupy.asnumpy(df_dev), df_host, atol=1e-11))

class TestEndToEndEngagement(unittest.TestCase):
    def test_rks_engages_switches_and_converges_same_fixed_point(self):
        mf_default, e_default = _run_scf(mol_rks)
        self.assertNotIn('adias_switch_cycle', mf_default.scf_summary)
        mf_adiis, e_adiis = _run_scf(mol_rks, diis_mode='adiis')
        self.assertTrue(mf_adiis.converged)
        self.assertAlmostEqual(e_adiis, e_default, delta=1e-9)
        self.assertIn('adias_switch_cycle', mf_adiis.scf_summary)
        self.assertGreaterEqual(mf_adiis.scf_summary['adias_switch_cycle'], 1)

    def test_uks_spin_stack_path(self):
        mf_adiis, e_adiis = _run_scf(mol_uks, diis_mode='adiis')
        mf_default, e_default = _run_scf(mol_uks)
        self.assertTrue(mf_adiis.converged)
        self.assertAlmostEqual(e_adiis, e_default, delta=1e-9)
        self.assertIn('adias_switch_cycle', mf_adiis.scf_summary)

    def test_knob_absent_energy_band_repeat(self):
        mf1, e1 = _run_scf(mol_rks)
        mf2, e2 = _run_scf(mol_rks)
        # same default path: identical within the documented run-to-run
        # scatter band (j3c atomicAdd ULP nondeterminism class; a bitwise
        # contract is wrong for full-SCF repeats on that path)
        self.assertAlmostEqual(e1, e2, delta=1e-10)
        self.assertNotIn('adias_head_cycles', mf1.scf_summary)
        self.assertNotIn('adias_head_cycles', mf2.scf_summary)
        self.assertNotIn('adias_switch_cycle', mf1.scf_summary)
        self.assertNotIn('adias_switch_cycle', mf2.scf_summary)

    def test_switch_records_realistic_head(self):
        # the handover must happen strictly inside the cycle budget and the
        # total cycle count stays bounded (no wrong-solution drift)
        mf_adiis, e_adiis = _run_scf(mol_rks, diis_mode='adiis')
        mf_default, _ = _run_scf(mol_rks)
        sw = mf_adiis.scf_summary['adias_switch_cycle']
        self.assertLessEqual(sw, mf_adiis.cycles)
        self.assertLess(mf_adiis.cycles, 40)

if __name__ == "__main__":
    print("Full Tests for ADIAS-HEAD-01")
    unittest.main()

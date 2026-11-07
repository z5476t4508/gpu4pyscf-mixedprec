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
'''DFJK-RESPGEMM-IMPL-01 focused equivalence regression (2026-11-06).

The opt-in GPU4PYSCF_DFGRAD_RESP_F32 knob (DEFAULT OFF) runs the phase-2
response contractions of the DF-gradient J+K pseudo-DM build in float32
(BENCH-03 V1 basis, entry 368) while phase-1, the j2c metric solve, the
J-part build and the auxbasis-response contraction keep the deployed
float64 discipline.  These tests precede any GPU measurement: parse
mapping, knob-off identity (counter-exact op-sequence identity + ULP-band
value equality + dtype), armed engagement (exactly one counter hit per
gradient call, both RHF/RKS and UHF/UKS twins), strict fp64-reference
inertness, composition with the deployed ip1-f32 kernel lane, and
whole-gradient subprocess arms.

Bitwise note (measured 2026-11-06, water/def2-svp/B3LYP): repeated
gradient calls are NOT bitwise reproducible even with zero code change
-- the sum_ejk_int3c2e_ip1 atomicAdd accumulation has a documented ULP
run-to-run scatter (the LC-42 comment in df/grad/rhf.py; ~2e-16 measured
here).  The knob-off identity claim is therefore delivered as: the armed
branch provably never executes (counter delta exactly 0), the value
equality holds to the ULP band (<= 1e-12, three orders above the
measured scatter), and the returned dtype is unchanged.
'''
import os
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.dft import numint

def setUpModule():
    global mol_sph, mol_o2
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_o2 = pyscf.M(atom='O 0 0 0; O 0 0 1.2078', basis='def2-svp',
                     spin=2, output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_o2
    mol_sph.stdout.close()
    mol_o2.stdout.close()
    del mol_sph, mol_o2

_ULP_BAND = 1e-12      # ip1 atomicAdd scatter measured ~2e-16 at this scale
_GRAD_GATE = 3e-5      # the project gradient gate (Eh/Bohr), the hard bound

class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob and its neighbors touch.'''

    def setUp(self):
        self.s_resp = df_rhf_grad._RESP_F32
        self.s_ip1f32 = df_rhf_grad._IP1_F32
        self.s_calls = df_rhf_grad._RESP_F32_CALLS
        self.s_prec = numint.precision.get_precision()
        self._env = os.environ.get('GPU4PYSCF_DFGRAD_RESP_F32')
        self._had_env = 'GPU4PYSCF_DFGRAD_RESP_F32' in os.environ

    def tearDown(self):
        df_rhf_grad._RESP_F32 = self.s_resp
        df_rhf_grad._IP1_F32 = self.s_ip1f32
        df_rhf_grad._RESP_F32_CALLS = self.s_calls
        numint.precision.set_precision(self.s_prec)
        if self._had_env:
            os.environ['GPU4PYSCF_DFGRAD_RESP_F32'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_DFGRAD_RESP_F32', None)

class RespF32Parse(_KnobState):

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_DFGRAD_RESP_F32', None)
        else:
            os.environ['GPU4PYSCF_DFGRAD_RESP_F32'] = raw
        df_rhf_grad._RESP_F32 = None
        return df_rhf_grad._resp_f32()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('1'), True)
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)

def _rks_grad(mol, xc='b3lyp'):
    mf = dft.RKS(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged
    return mf, mf.nuc_grad_method()

class KnobOffIdentity(_KnobState):
    '''Knob-off = the armed branch never executes; values equal to the
    ULP band; dtype unchanged (see the module docstring bitwise note).'''

    @classmethod
    def setUpClass(cls):
        cls.mf, cls.grad = _rks_grad(mol_sph)

    def test_knob_off_counter_zero_ulp_stable(self):
        df_rhf_grad._RESP_F32 = False
        c0 = df_rhf_grad._RESP_F32_CALLS
        g0 = self.grad.kernel()
        g1 = self.grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 0)
        self.assertEqual(g0.dtype, np.float64)
        self.assertEqual(g1.dtype, np.float64)
        self.assertLessEqual(np.abs(g1 - g0).max(), _ULP_BAND)

    def test_invalid_knob_value_behaves_off(self):
        df_rhf_grad._RESP_F32 = None
        os.environ['GPU4PYSCF_DFGRAD_RESP_F32'] = 'bogus'
        with self.assertWarns(UserWarning):
            self.assertIs(df_rhf_grad._resp_f32(), False)
        c0 = df_rhf_grad._RESP_F32_CALLS
        g = self.grad.kernel()
        df_rhf_grad._RESP_F32 = False
        g0 = self.grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 0)
        self.assertLessEqual(np.abs(g - g0).max(), _ULP_BAND)

class ArmedEquivalence(_KnobState):
    '''Armed: exactly one engagement counter hit per gradient call
    (B3LYP is a global hybrid -> a single jk_energy_per_atom call),
    float64 return, banded numerics; composition with the deployed
    ip1-f32 kernel lane; strict-reference inertness.'''

    @classmethod
    def setUpClass(cls):
        cls.mf, cls.grad = _rks_grad(mol_sph)

    def test_armed_engages_once_bands_and_dtype(self):
        df_rhf_grad._RESP_F32 = False
        g0 = self.grad.kernel()
        df_rhf_grad._RESP_F32 = True
        c0 = df_rhf_grad._RESP_F32_CALLS
        g1 = self.grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 1)
        self.assertEqual(g1.dtype, np.float64)
        d = np.abs(g1 - g0).max()
        print('armed resp-f32 max|dg| vs knob-off (water/def2-svp/b3lyp):', d)
        self.assertLessEqual(d, _GRAD_GATE)

    def test_armed_stable_across_calls(self):
        df_rhf_grad._RESP_F32 = True
        g1 = self.grad.kernel()
        g2 = self.grad.kernel()
        # both armed: same fp32 chain, only the ip1 ULP scatter separates
        self.assertLessEqual(np.abs(g2 - g1).max(), _ULP_BAND)

    def test_armed_composes_with_ip1_f32_lane(self):
        # single-variable discipline: on the fp32-kernel lane the armed
        # delta is compared against THAT lane's knob-off run
        df_rhf_grad._IP1_F32 = True
        df_rhf_grad._RESP_F32 = False
        gi0 = self.grad.kernel()
        df_rhf_grad._RESP_F32 = True
        c0 = df_rhf_grad._RESP_F32_CALLS
        gi1 = self.grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 1)
        self.assertLessEqual(np.abs(gi1 - gi0).max(), _GRAD_GATE)

    def test_strict_reference_ctx_inert(self):
        df_rhf_grad._RESP_F32 = False
        g0 = self.grad.kernel()
        df_rhf_grad._RESP_F32 = True
        c0 = df_rhf_grad._RESP_F32_CALLS
        with numint.strict_fp64_reference_ctx():
            g1 = self.grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 0)
        self.assertLessEqual(np.abs(g1 - g0).max(), _ULP_BAND)

class UhfTwin(_KnobState):
    '''The uhf.py twins behind UKS gradients share the arming helper,
    the counter and the fp64 auxresp discipline (one engagement per
    gradient call; the spin-stacked response chain casts once).'''

    def test_uks_armed_engages_once_and_bands(self):
        mf = dft.UKS(mol_o2, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.kernel()
        self.assertTrue(mf.converged)
        grad = mf.nuc_grad_method()
        df_rhf_grad._RESP_F32 = False
        g0 = grad.kernel()
        df_rhf_grad._RESP_F32 = True
        c0 = df_rhf_grad._RESP_F32_CALLS
        g1 = grad.kernel()
        self.assertEqual(df_rhf_grad._RESP_F32_CALLS - c0, 1)
        self.assertEqual(g1.dtype, np.float64)
        d = np.abs(g1 - g0).max()
        print('armed resp-f32 UKS twin max|dg| (O2/def2-svp/b3lyp):', d)
        self.assertLessEqual(d, _GRAD_GATE)

_E2E_GRAD = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_DFGRAD_RESP_F32'] = {knob!r}
import numpy as np
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as df_rhf_grad
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol, xc='b3lyp').density_fit()
mf.conv_tol = 1e-12
mf.kernel()
c0 = df_rhf_grad._RESP_F32_CALLS
g = mf.nuc_grad_method().kernel()
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      df_rhf_grad._RESP_F32_CALLS - c0, g.dtype.str,
      float(np.abs(g).max()))
print('GRADHEX', g.tobytes().hex())
"""

class EndToEndSubprocess(unittest.TestCase):
    '''Fresh-subprocess whole-gradient arms (env-parsed arming, the
    deployment-realistic path); the SCF itself is untouched by the knob
    (cross-process energy band 1e-10 per the documented cross-process
    noise floor), the gradient band is the project gate.'''

    def _run_arm(self, knob):
        import subprocess, tempfile
        # rhf.py sits one level deeper than dft/numint.py (df/grad/), so
        # the repo root is four dirnames up -- three would resolve the
        # package dir itself onto sys.path and the empty site-packages
        # gpu4pyscf namespace dir would shadow the repo
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(df_rhf_grad.__file__)))))
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(_E2E_GRAD.format(repo=repo, knob=knob))
            path = f.name
        try:
            out = subprocess.check_output(
                [sys.executable, path], timeout=900, cwd=repo)
        finally:
            os.unlink(path)
        lines = out.decode().splitlines()
        meta = [l for l in lines if l.startswith('E2EJSON')][-1]
        e_hex, conv, served, dt, gmax = meta.split()[1:]
        grad_hex = [l for l in lines if l.startswith('GRADHEX')][-1].split()[1]
        return (float.fromhex(e_hex), conv == 'True', int(served), dt,
                np.frombuffer(bytes.fromhex(grad_hex), dtype=np.float64))

    def test_env_armed_vs_off(self):
        e0, conv0, s0, dt0, g0 = self._run_arm('0')
        e1, conv1, s1, dt1, g1 = self._run_arm('1')
        self.assertTrue(conv0 and conv1)
        self.assertEqual(s0, 0)
        self.assertEqual(s1, 1)
        self.assertEqual(dt1, '<f8')
        self.assertLessEqual(abs(e1 - e0), 1e-10)   # SCF untouched by knob
        d = np.abs(g1 - g0).max()
        print('armed resp-f32 subprocess max|dg|:', d)
        self.assertLessEqual(d, _GRAD_GATE)

if __name__ == '__main__':
    unittest.main()

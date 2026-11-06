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
'''SCFJK-KBUILD-IMPL-01 focused equivalence regression (2026-11-05).

The opt-in GPU4PYSCF_SCF_JK_F32 knob (DEFAULT OFF, parse mirroring
GPU4PYSCF_SCF_VMAT_F32/GPU4PYSCF_SCF_RHO_F32) arms the EXISTING fp32
branch of df_jk.get_jk (fp32 MO factors and cderi operands, fp32 K
accumulation, fp64 upcast on return -- the exact discipline the auto
policy's fp32 head runs) on the SCF head of pure-fp64 lanes; 'headtail'
flips the lane off at the same scf.hf single trigger as the vmat/rho
knobs, so the convergence tail keeps the exact fp64 K map.  These tests
precede any GPU measurement: parse mapping, knob-off fp64 pass-through
(bitwise unchanged lanes), auto-lane inertness of the new counter,
lane-armed engagement with banded numerics, strict-reference inertness,
and whole-SCF subprocess arms.
'''
import os
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
from gpu4pyscf.lib import precision

def setUpModule():
    global mol_sph, mol_benz
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_benz = pyscf.M(atom='''
C      1.3970 0 0
C      0.6985 1.2098 0
C     -0.6985 1.2098 0
C     -1.3970 0 0
C     -0.6985 -1.2098 0
C      0.6985 -1.2098 0
H      2.4810 0 0
H      1.2405 2.1486 0
H     -1.2405 2.1486 0
H     -2.4810 0 0
H     -1.2405 -2.1486 0
H      1.2405 -2.1486 0''',
                       basis='def2-svp', output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_benz
    mol_sph.stdout.close()
    mol_benz.stdout.close()
    del mol_sph, mol_benz

def _bitwise_equal(a, b):
    a, b = cp.asarray(a), cp.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())

def _df_hf_dm(mol):
    '''Converged DF-HF on mol; returns (dfobj, mo-tagged dm) so the
    direct get_jk arms reproduce the per-cycle call shape (mo-marked
    dm, dm_factor_mode 0).'''
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.with_df, cp.asarray(mf.make_rdm1())

class ScfJkF32Parse(unittest.TestCase):
    def setUp(self):
        self._saved = df_jk._SCF_JK_F32
        self._had_env = 'GPU4PYSCF_SCF_JK_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_SCF_JK_F32')

    def tearDown(self):
        df_jk._SCF_JK_F32 = self._saved
        if self._had_env:
            os.environ['GPU4PYSCF_SCF_JK_F32'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)
        else:
            os.environ['GPU4PYSCF_SCF_JK_F32'] = raw
        df_jk._SCF_JK_F32 = None
        return df_jk._scf_jk_f32()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('1'), True)
        self.assertIs(self._parse('headtail'), 'headtail')
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)

class ScfJkF32Wrapper(unittest.TestCase):
    '''Direct df_jk.get_jk arms at fixed dm: knob-off determinism,
    lane-armed numerics bands, counter engagement.'''

    def setUp(self):
        self._saved_lane = df_jk._scf_jk_f32_lane
        self._saved_prec = precision.get_precision()

    def tearDown(self):
        df_jk._scf_jk_f32_lane = self._saved_lane
        precision.set_precision(self._saved_prec)

    def test_knob_off_runs_bitwise_deterministic(self):
        dfobj, dm = _df_hf_dm(mol_benz)
        precision.set_precision('fp64')
        df_jk._scf_jk_f32_lane = False
        c0 = df_jk._scf_jk_f32_head_calls
        vj0, vk0 = df_jk.get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True)
        vj1, vk1 = df_jk.get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(df_jk._scf_jk_f32_head_calls, c0)
        self.assertTrue(_bitwise_equal(vj0, vj1))
        self.assertTrue(_bitwise_equal(vk0, vk1))
        self.assertEqual(vk0.dtype, cp.float64)

    def test_lane_armed_engages_and_bands(self):
        dfobj, dm = _df_hf_dm(mol_benz)
        precision.set_precision('fp64')
        df_jk._scf_jk_f32_lane = False
        vj0, vk0 = df_jk.get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True)
        df_jk._scf_jk_f32_lane = True
        c0 = df_jk._scf_jk_f32_head_calls
        vj1, vk1 = df_jk.get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True)
        served = df_jk._scf_jk_f32_head_calls - c0
        self.assertEqual(served, 1)
        # returned dtype independent of the precision mode; the fp32
        # K-transform accumulation is upcast on return (banded, never
        # bitwise -- fp32 operand rounding ~1e-7 RELATIVE)
        self.assertEqual(vj1.dtype, cp.float64)
        self.assertEqual(vk1.dtype, cp.float64)
        self.assertFalse(_bitwise_equal(vk1, vk0))
        self.assertGreater(float(cp.abs(vk1 - vk0).max()), 0)
        self.assertLessEqual(float(cp.abs(vk1 - vk0).max()), 1e-5)
        self.assertLessEqual(float(cp.abs(vj1 - vj0).max()), 1e-5)

    def test_auto_lane_calls_do_not_count(self):
        # the auto policy's fp32 head keeps its deployed first branch
        # (precision gate alone); the new counter counts ONLY lane-armed
        # engagements (knob-off and auto-lane deltas are exactly 0)
        dfobj, dm = _df_hf_dm(mol_benz)
        precision.set_precision('fp32')
        df_jk._scf_jk_f32_lane = False
        c0 = df_jk._scf_jk_f32_head_calls
        vj1, vk1 = df_jk.get_jk(dfobj, dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(vk1.dtype, cp.float64)
        self.assertEqual(df_jk._scf_jk_f32_head_calls, c0)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = {vmat_knob!r}
os.environ['GPU4PYSCF_SCF_RHO_F32'] = {rho_knob!r}
os.environ['GPU4PYSCF_SCF_JK_F32'] = {jk_knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol).density_fit(); mf.xc = {xc!r}; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
c0 = df_jk._scf_jk_f32_head_calls
mf.kernel()
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      df_jk._scf_jk_f32_head_calls - c0)
"""

_E2E_ARM_UKS = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_JK_F32'] = {jk_knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
mol = pyscf.M(atom=\"\"\"O 0 0 0
H 0 0 0.97\"\"\", basis='def2-svp', spin=1,
              output='/dev/null', verbose=0)
mf = dft.UKS(mol).density_fit(); mf.xc = {xc!r}; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
c0 = df_jk._scf_jk_f32_head_calls
mf.kernel()
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      df_jk._scf_jk_f32_head_calls - c0)
"""

class EndToEndSCF(unittest.TestCase):
    '''Whole-SCF arms, one fresh subprocess per knob setting (the
    in-process sequential-SCF bitwise instability documented by
    test_xc_ao_carryover01 applies here too; the honest band across
    fresh subprocesses is 1e-10).'''

    def _run_arm(self, script, **fmt):
        import subprocess, tempfile
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(df_jk.__file__))))
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(script.format(repo=repo, **fmt))
            path = f.name
        try:
            out = subprocess.check_output(
                [sys.executable, path], timeout=600, cwd=repo)
        finally:
            os.unlink(path)
        line = [l for l in out.decode().splitlines()
                if l.startswith('E2EJSON')][-1]
        e_hex, conv, served = line.split()[1:]
        return float.fromhex(e_hex), conv == 'True', int(served)

    def _run_rks(self, jk_knob, vmat_knob='0', rho_knob='0', xc='b3lyp'):
        return self._run_arm(_E2E_ARM, jk_knob=jk_knob, vmat_knob=vmat_knob,
                             rho_knob=rho_knob, xc=xc)

    def test_headtail_and_full_knobs_vs_off(self):
        e_off, conv_off, served_off = self._run_rks('0')
        e_ht, conv_ht, served_ht = self._run_rks('headtail')
        e_full, conv_full, served_full = self._run_rks('1')
        self.assertTrue(conv_off and conv_ht)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        self.assertGreater(served_full, served_ht)
        # headtail restores the true fp64 fixed point (strict band)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)
        # '1' runs the f32 K-transform through the convergence tail: the
        # LC-115 fixed-point displacement class, MEASURED HERE at water
        # scale already -- the fp32 K noise (fixed-dm band ~2e-6 on vk)
        # feeds DIIS every cycle and the run does not converge at
        # conv_tol=1e-10 within the 50-cycle cap (measured 2026-11-05:
        # unconverged, final energy displaced 2e-7..3.5e-6 across fresh
        # runs -- no stable band exists for an unconverged trajectory).
        # Class assertion: '1' either converges inside the 1e-6 Ha
        # science gate, or is honestly non-convergent with its
        # displacement contained at the head-noise scale (<1e-4, a
        # diagnostic containment against gross wrongness, NOT a gate);
        # it is never silently strict-close.  The head/tail flip is
        # REQUIRED for strict convergence -- only 'headtail' is the
        # adoption candidate.
        if conv_full:
            self.assertLessEqual(abs(e_full - e_off), 1e-6)
        else:
            self.assertFalse(conv_full)
            self.assertLessEqual(abs(e_full - e_off), 1e-4)

    def test_headtail_arms_alongside_deployed_vmat_rho_headtail(self):
        # the production stack arms the vmat/rho knobs; the jk knob must
        # compose with them (all three flip at the same trigger)
        e_off, conv_off, served_off = self._run_rks(
            '0', vmat_knob='headtail', rho_knob='headtail')
        e_ht, conv_ht, served_ht = self._run_rks(
            'headtail', vmat_knob='headtail', rho_knob='headtail')
        self.assertTrue(conv_off and conv_ht)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)

    def test_mgga_headtail_vs_off(self):
        e_off, conv_off, _ = self._run_rks('0', xc='m062x')
        e_ht, conv_ht, served_ht = self._run_rks('headtail', xc='m062x')
        self.assertTrue(conv_off and conv_ht)
        self.assertGreater(served_ht, 0)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)

    def test_uks_headtail_vs_off(self):
        e_off, conv_off, served_off = self._run_arm(
            _E2E_ARM_UKS, jk_knob='0', xc='b3lyp')
        e_ht, conv_ht, served_ht = self._run_arm(
            _E2E_ARM_UKS, jk_knob='headtail', xc='b3lyp')
        self.assertTrue(conv_off and conv_ht)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        # UKS cold-start band: the fp32 K head steers the open-shell
        # iteration into a nearby stationary point (measured 2.9e-7 with
        # FASTER convergence, 8 vs 11 cycles; the fixed-dm operator band
        # is 4.6e-7 relative; warm-starting headtail from the knob-off
        # converged dm stays at the same solution, -2.2e-10 -- the
        # fixed point is NOT displaced, this is trajectory/basin
        # steering).  The closed-shell strict band is asserted in the
        # RKS arms; the UKS arm asserts the 1e-6 Ha science gate.
        self.assertLessEqual(abs(e_ht - e_off), 1e-6)

class LaneGating(unittest.TestCase):
    '''In-process kernel-gate assertions: the strict fp64-reference
    sibling and the default lane never arm the lane flag.'''

    def setUp(self):
        self._saved_knob = df_jk._SCF_JK_F32
        self._saved_lane = df_jk._scf_jk_f32_lane

    def tearDown(self):
        df_jk._SCF_JK_F32 = self._saved_knob
        df_jk._scf_jk_f32_lane = self._saved_lane

    def _mf(self):
        mf = dft.RKS(mol_sph).density_fit()
        mf.xc = 'b3lyp'
        mf.grids.level = 2
        mf.conv_tol = 1e-10
        return mf

    def test_strict_reference_sibling_inert(self):
        df_jk._SCF_JK_F32 = 'headtail'   # in-process cached-global arm
        mf = self._mf()
        mf._strict_fp64_reference = True
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(df_jk._scf_jk_f32_head_calls - c0, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)

    def test_default_lane_never_engages_in_process(self):
        df_jk._SCF_JK_F32 = False
        mf = self._mf()
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(df_jk._scf_jk_f32_head_calls - c0, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)

    def test_headtail_rearms_per_kernel_and_converges(self):
        df_jk._SCF_JK_F32 = 'headtail'
        mf = self._mf()
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        served1 = df_jk._scf_jk_f32_head_calls - c0
        self.assertTrue(mf.converged)
        self.assertGreater(served1, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)  # tail flipped + restored
        mf.kernel()   # re-armed at the next _kernel call
        served2 = df_jk._scf_jk_f32_head_calls - c0 - served1
        self.assertGreater(served2, 0)

if __name__ == '__main__':
    unittest.main()

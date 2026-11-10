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
'''SCFJK-WINDOW-01 focused equivalence regression (2026-11-06).

The opt-in GPU4PYSCF_SCF_JK_F32_ARM_AFTER=N companion knob (DEFAULT 0 =
arm at kernel entry, the SCFJK-KBUILD-IMPL-01 semantics) delays the
arming of the GPU4PYSCF_SCF_JK_F32=headtail jk lane: the fp64 prefix
(the init get_veff and cycles 1..N) is bitwise the knob-off
trajectory, the lane arms once at the end of cycle N, and the
e_diff < switch-tolerance tail flip is unchanged.  These tests precede
any GPU measurement: parse mapping (including the headtail-only combo
gate), never-armed bitwise-prefix equivalence, arm-timing engagement,
strict/default-lane inertness, and whole-SCF subprocess arms (RKS
strict band, UKS 1e-6 Ha science gate, composition with the deployed
vmat/rho headtail knobs).
'''
import os
import sys
import unittest
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk

def setUpModule():
    global mol_sph
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph
    mol_sph.stdout.close()
    del mol_sph

class ScfJkF32ArmAfterParse(unittest.TestCase):
    def setUp(self):
        self._saved_knob = df_jk._SCF_JK_F32
        self._saved_delay = df_jk._SCF_JK_F32_ARM_AFTER
        self._had_env = 'GPU4PYSCF_SCF_JK_F32_ARM_AFTER' in os.environ
        self._env = os.environ.get('GPU4PYSCF_SCF_JK_F32_ARM_AFTER')

    def tearDown(self):
        df_jk._SCF_JK_F32 = self._saved_knob
        df_jk._SCF_JK_F32_ARM_AFTER = self._saved_delay
        if self._had_env:
            os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', None)

    def _parse(self, raw, knob='headtail'):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', None)
        else:
            os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = raw
        df_jk._SCF_JK_F32 = knob
        df_jk._SCF_JK_F32_ARM_AFTER = None
        return df_jk._scf_jk_f32_arm_after()

    def test_parse_mapping(self):
        self.assertEqual(self._parse(None), 0)
        self.assertEqual(self._parse(''), 0)
        self.assertEqual(self._parse('0'), 0)
        self.assertEqual(self._parse('2'), 2)

    def test_parse_rejects_garbage(self):
        with self.assertWarns(UserWarning):
            self.assertEqual(self._parse('bogus'), 0)
        with self.assertWarns(UserWarning):
            self.assertEqual(self._parse('-1'), 0)

    def test_combo_headtail_only(self):
        # the delay is defined for the 'headtail' knob value only;
        # any other value warns and ignores the delay (arm at entry)
        self.assertEqual(self._parse('3', knob='headtail'), 3)
        with self.assertWarns(UserWarning):
            self.assertEqual(self._parse('3', knob='0'), 0)
        with self.assertWarns(UserWarning):
            self.assertEqual(self._parse('3', knob='1'), 0)

class LaneGating(unittest.TestCase):
    '''In-process kernel-gate assertions: the strict fp64-reference
    sibling and the knob-off lane never arm through the delayed
    window, and the delayed arm engages fewer calls than entry arming
    at the same system.'''

    def setUp(self):
        self._saved_knob = df_jk._SCF_JK_F32
        self._saved_delay = df_jk._SCF_JK_F32_ARM_AFTER
        self._saved_lane = df_jk._scf_jk_f32_lane

    def tearDown(self):
        df_jk._SCF_JK_F32 = self._saved_knob
        df_jk._SCF_JK_F32_ARM_AFTER = self._saved_delay
        df_jk._scf_jk_f32_lane = self._saved_lane

    def _mf(self):
        mf = dft.RKS(mol_sph).density_fit()
        mf.xc = 'b3lyp'
        mf.grids.level = 2
        mf.conv_tol = 1e-10
        return mf

    def test_strict_reference_sibling_inert_under_delay(self):
        df_jk._SCF_JK_F32 = 'headtail'
        df_jk._SCF_JK_F32_ARM_AFTER = 2
        mf = self._mf()
        mf._strict_fp64_reference = True
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(df_jk._scf_jk_f32_head_calls - c0, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)

    def test_knob_off_lane_never_arms_under_delay_env(self):
        # knob off + delay set: neither the entry gate nor the delayed
        # latch may arm anything
        df_jk._SCF_JK_F32 = False
        df_jk._SCF_JK_F32_ARM_AFTER = 2
        mf = self._mf()
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(df_jk._scf_jk_f32_head_calls - c0, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)

    def test_delayed_arm_engages_fewer_calls_than_entry(self):
        df_jk._SCF_JK_F32 = 'headtail'
        df_jk._SCF_JK_F32_ARM_AFTER = 0
        mf = self._mf()
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        served_entry = df_jk._scf_jk_f32_head_calls - c0
        self.assertFalse(df_jk._scf_jk_f32_lane)  # tail flipped
        self.assertGreater(served_entry, 0)
        df_jk._SCF_JK_F32_ARM_AFTER = 2
        mf2 = self._mf()
        c1 = df_jk._scf_jk_f32_head_calls
        mf2.kernel()
        self.assertTrue(mf2.converged)
        served_delayed = df_jk._scf_jk_f32_head_calls - c1
        self.assertFalse(df_jk._scf_jk_f32_lane)
        # the delayed window arms a strict subset of the entry window
        # at the same system (same cycles -> 2 fewer armed calls: the
        # init get_veff and cycle 1 stay fp64; never more)
        self.assertGreater(served_delayed, 0)
        self.assertLess(served_delayed, served_entry)

    def test_delay_beyond_max_cycle_never_arms(self):
        df_jk._SCF_JK_F32 = 'headtail'
        df_jk._SCF_JK_F32_ARM_AFTER = 50     # > any water cycle count
        mf = self._mf()
        c0 = df_jk._scf_jk_f32_head_calls
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(df_jk._scf_jk_f32_head_calls - c0, 0)
        self.assertFalse(df_jk._scf_jk_f32_lane)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = {vmat_knob!r}
os.environ['GPU4PYSCF_SCF_RHO_F32'] = {rho_knob!r}
os.environ['GPU4PYSCF_SCF_JK_F32'] = {jk_knob!r}
os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = {arm_after!r}
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
os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = {arm_after!r}
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

    def _run_rks(self, jk_knob, arm_after='0', vmat_knob='0',
                 rho_knob='0', xc='b3lyp'):
        return self._run_arm(_E2E_ARM, jk_knob=jk_knob,
                             arm_after=arm_after, vmat_knob=vmat_knob,
                             rho_knob=rho_knob, xc=xc)

    def test_delayed_window_vs_off_and_vs_entry(self):
        e_off, conv_off, served_off = self._run_rks('0')
        e_entry, conv_entry, served_entry = self._run_rks(
            'headtail', arm_after='0')
        e_win, conv_win, served_win = self._run_rks(
            'headtail', arm_after='1')
        self.assertTrue(conv_off and conv_entry and conv_win)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_entry, 0)
        # the delayed window arms a strict subset of the entry window
        self.assertGreater(served_win, 0)
        self.assertLess(served_win, served_entry)
        # head/tail restores the true fp64 fixed point (strict band);
        # the delayed window shares the same tail discipline
        self.assertLessEqual(abs(e_entry - e_off), 1e-10)
        self.assertLessEqual(abs(e_win - e_off), 1e-10)

    def test_delay_beyond_max_cycle_is_knob_off(self):
        # ARM_AFTER larger than any cycle count: the lane never arms,
        # the whole run is the bitwise knob-off code path
        e_off, conv_off, served_off = self._run_rks('0')
        e_never, conv_never, served_never = self._run_rks(
            'headtail', arm_after='50')
        self.assertTrue(conv_off and conv_never)
        self.assertEqual(served_never, 0)
        self.assertLessEqual(abs(e_never - e_off), 1e-10)

    def test_delayed_window_composes_with_deployed_vmat_rho(self):
        # the production stack arms the vmat/rho knobs at entry; the
        # delayed jk window must compose with them (the jk lane alone
        # carries the delay; the tail flip stays shared)
        e_off, conv_off, served_off = self._run_rks(
            '0', vmat_knob='headtail', rho_knob='headtail')
        e_win, conv_win, served_win = self._run_rks(
            'headtail', arm_after='1',
            vmat_knob='headtail', rho_knob='headtail')
        self.assertTrue(conv_off and conv_win)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_win, 0)
        self.assertLessEqual(abs(e_win - e_off), 1e-10)

    def test_uks_delayed_window_vs_off(self):
        e_off, conv_off, served_off = self._run_arm(
            _E2E_ARM_UKS, jk_knob='0', arm_after='0', xc='b3lyp')
        e_win, conv_win, served_win = self._run_arm(
            _E2E_ARM_UKS, jk_knob='headtail', arm_after='1', xc='b3lyp')
        self.assertTrue(conv_off and conv_win)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_win, 0)
        # UKS band: the fp32 K window can steer the open-shell
        # iteration into a nearby stationary point (the Phase-A UKS
        # basin-steering class of SCFJK-KBUILD-IMPL-01); the science
        # gate is the 1e-6 Ha acceptance bound, not the closed-shell
        # strict band
        self.assertLessEqual(abs(e_win - e_off), 1e-6)

if __name__ == '__main__':
    unittest.main()

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
'''RHO-F32-LANE-B-01 focused equivalence regression (2026-11-05).

The opt-in GPU4PYSCF_SCF_RHO_F32 knob (DEFAULT OFF, parse mirroring
GPU4PYSCF_SCF_VMAT_F32) extends the headtail discipline to the rho
contraction on pure-fp64 lanes: in armed head cycles the mo-path rho
block loop consumes the f32 twin and _eval_rho2 runs its existing fp32
branch (fp64 rho output accumulation); the exact-fp64 tail keeps rho
bitwise fp64.  These tests precede any GPU measurement: parse mapping,
knob-off fp64 pass-through (bitwise unchanged lanes), auto-lane
inertness of the new counter, lane-armed engagement with banded
numerics, strict-reference inertness, and whole-SCF subprocess arms.
'''
import os
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid

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

def _nr_fixed_dm(mol, dm, xc='b3lyp'):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = 2
    grids.build()
    dms = cp.asarray(dm)
    nelec, excsum, vmat = numint.nr_rks(ni, mol, grids, xc, dms, hermi=1)
    return nelec, excsum, vmat, grids

class RhoF32Parse(unittest.TestCase):
    def setUp(self):
        self._saved = numint._SCF_RHO_F32
        self._had_env = 'GPU4PYSCF_SCF_RHO_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_SCF_RHO_F32')

    def tearDown(self):
        numint._SCF_RHO_F32 = self._saved
        if self._had_env:
            os.environ['GPU4PYSCF_SCF_RHO_F32'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_SCF_RHO_F32', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_RHO_F32', None)
        else:
            os.environ['GPU4PYSCF_SCF_RHO_F32'] = raw
        numint._SCF_RHO_F32 = None
        return numint._scf_rho_f32()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('1'), True)
        self.assertIs(self._parse('headtail'), 'headtail')
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)

class RhoF32Wrapper(unittest.TestCase):
    '''Direct unit tests of _ao_f32_for_rho's lane routing.'''

    def setUp(self):
        self._saved_lane = numint._scf_rho_f32_lane
        self._saved_prec = numint.precision.get_precision()

    def tearDown(self):
        numint._scf_rho_f32_lane = self._saved_lane
        numint.precision.set_precision(self._saved_prec)

    def _block(self):
        ni = dft.numint.NumInt()
        grids = gen_grid.Grids(mol_sph)
        grids.level = 1
        grids.build()
        ni.build(mol_sph, grids.coords)
        opt = ni.gdftopt
        for ao, idx, weight, coords in numint._block_loop(
                ni, opt._sorted_mol, grids, opt._sorted_mol.nao, 1,
                max_memory=None, loop_blksize=numint.MIN_BLK_SIZE):
            return ao, grids, opt
        raise AssertionError('no block')

    def test_knob_off_fp64_pass_through_identical_object(self):
        numint.precision.set_precision('fp64')
        numint._scf_rho_f32_lane = False
        ao, grids, opt = self._block()
        c0 = numint._rho_f32_head_blocks
        out = numint._ao_f32_for_rho(
            ao, grids, opt, numint.MIN_BLK_SIZE, 0, ao.shape[0])
        self.assertIs(out, ao)
        self.assertEqual(numint._rho_f32_head_blocks, c0)

    def test_lane_armed_serves_f32_twin_and_counts(self):
        numint.precision.set_precision('fp64')
        numint._scf_rho_f32_lane = True
        ao, grids, opt = self._block()
        c0 = numint._rho_f32_head_blocks
        out = numint._ao_f32_for_rho(
            ao, grids, opt, numint.MIN_BLK_SIZE, 0, ao.shape[0])
        self.assertEqual(out.dtype, cp.float32)
        self.assertTrue(_bitwise_equal(out, ao.astype(cp.float32)))
        self.assertEqual(numint._rho_f32_head_blocks, c0 + 1)

    def test_auto_lane_twin_serves_do_not_count(self):
        # the auto policy's fp32 head keeps its deployed first branch;
        # the new counter counts ONLY lane-B servings (knob-off and
        # auto-lane deltas are exactly 0)
        numint.precision.set_precision('fp32')
        numint._scf_rho_f32_lane = False
        ao, grids, opt = self._block()
        c0 = numint._rho_f32_head_blocks
        out = numint._ao_f32_for_rho(
            ao, grids, opt, numint.MIN_BLK_SIZE, 0, ao.shape[0])
        self.assertEqual(out.dtype, cp.float32)
        self.assertEqual(numint._rho_f32_head_blocks, c0)

class RhoF32LaneArmed(unittest.TestCase):
    '''Fixed-dm nr_rks: knob-off determinism, armed numerics bands,
    counter engagement (mo-path blocks served).'''

    def setUp(self):
        self._saved_lane = numint._scf_rho_f32_lane
        self._saved_prec = numint.precision.get_precision()

    def tearDown(self):
        numint._scf_rho_f32_lane = self._saved_lane
        numint.precision.set_precision(self._saved_prec)

    def test_knob_off_runs_bitwise_deterministic(self):
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint.precision.set_precision('fp64')
        numint._scf_rho_f32_lane = False
        c0 = numint._rho_f32_head_blocks
        n0, e0, v0, _ = _nr_fixed_dm(mol_benz, dm)
        n1, e1, v1, _ = _nr_fixed_dm(mol_benz, dm)
        self.assertEqual(numint._rho_f32_head_blocks, c0)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertTrue(_bitwise_equal(v1, v0))

    def test_lane_armed_engages_and_bands(self):
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint.precision.set_precision('fp64')
        numint._scf_rho_f32_lane = False
        n0, e0, v0, _ = _nr_fixed_dm(mol_benz, dm)
        numint._scf_rho_f32_lane = True
        c0 = numint._rho_f32_head_blocks
        n1, e1, v1, _ = _nr_fixed_dm(mol_benz, dm)
        served = numint._rho_f32_head_blocks - c0
        self.assertGreater(served, 0)
        # fp64 rho output accumulation + fp64 vmat add_sparse: banded,
        # never bitwise (the head contraction itself runs f32, ~1e-7
        # RELATIVE accuracy -- the documented _eval_rho2 f32 scale)
        self.assertLessEqual(abs(n1 - n0), 1e-7 * max(1.0, abs(n0)))
        self.assertLessEqual(abs(e1 - e0), 1e-7 * max(1.0, abs(e0)))
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-5)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = {vmat_knob!r}
os.environ['GPU4PYSCF_SCF_RHO_F32'] = {rho_knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol); mf.xc = {xc!r}; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
c0 = numint._rho_f32_head_blocks
mf.kernel()
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      numint._rho_f32_head_blocks - c0)
"""

class EndToEndSCF(unittest.TestCase):
    '''Whole-SCF arms, one fresh subprocess per knob setting (the
    in-process sequential-SCF bitwise instability documented by
    test_xc_ao_carryover01 applies here too; the honest band across
    fresh subprocesses is 1e-10).'''

    def _run_arm(self, rho_knob, vmat_knob='0', xc='b3lyp'):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            f.write(_E2E_ARM.format(
                repo=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(numint.__file__)))),
                rho_knob=rho_knob, vmat_knob=vmat_knob, xc=xc))
            path = f.name
        try:
            out = subprocess.check_output(
                [sys.executable, path], timeout=600,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(numint.__file__)))))
        finally:
            os.unlink(path)
        line = [l for l in out.decode().splitlines()
                if l.startswith('E2EJSON')][-1]
        e_hex, conv, served = line.split()[1:]
        return float.fromhex(e_hex), conv == 'True', int(served)

    def test_headtail_and_full_knobs_vs_off(self):
        e_off, conv_off, served_off = self._run_arm('0')
        e_ht, conv_ht, served_ht = self._run_arm('headtail')
        e_full, conv_full, served_full = self._run_arm('1')
        self.assertTrue(conv_off and conv_ht and conv_full)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        self.assertGreater(served_full, 0)
        self.assertGreaterEqual(served_full, served_ht)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)
        # '1' runs the f32 rho through the convergence tail: the LC-115
        # fixed-point displacement class (measured here at water scale,
        # ~3e-7) -- inside the 1e-6 Ha science gate but NOT the strict
        # band; headtail above restores the true fp64 fixed point.
        # This is why only 'headtail' is the adoption candidate.
        self.assertLessEqual(abs(e_full - e_off), 1e-6)

    def test_headtail_arms_alongside_deployed_vmat_headtail(self):
        # the production stack arms the vmat knob; the rho knob must
        # compose with it (both flip at the same trigger)
        e_off, conv_off, served_off = self._run_arm(
            '0', vmat_knob='headtail')
        e_ht, conv_ht, served_ht = self._run_arm(
            'headtail', vmat_knob='headtail')
        self.assertTrue(conv_off and conv_ht)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)

    def test_mgga_headtail_vs_off(self):
        e_off, conv_off, served_off = self._run_arm('0', xc='m062x')
        e_ht, conv_ht, served_ht = self._run_arm('headtail', xc='m062x')
        self.assertTrue(conv_off and conv_ht)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_ht, 0)
        self.assertLessEqual(abs(e_ht - e_off), 1e-10)

class LaneGating(unittest.TestCase):
    '''In-process kernel-gate assertions: the strict fp64-reference
    sibling and the default lane never arm the lane flag.'''

    def setUp(self):
        self._saved_knob = numint._SCF_RHO_F32
        self._saved_lane = numint._scf_rho_f32_lane

    def tearDown(self):
        numint._SCF_RHO_F32 = self._saved_knob
        numint._scf_rho_f32_lane = self._saved_lane

    def test_strict_reference_sibling_inert(self):
        numint._SCF_RHO_F32 = 'headtail'   # in-process cached-global arm
        mf = dft.RKS(mol_sph)
        mf.xc = 'b3lyp'
        mf.grids.level = 2
        mf.conv_tol = 1e-10
        mf._strict_fp64_reference = True
        c0 = numint._rho_f32_head_blocks
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(numint._rho_f32_head_blocks - c0, 0)
        self.assertFalse(numint._scf_rho_f32_lane)

    def test_default_lane_never_engages_in_process(self):
        numint._SCF_RHO_F32 = False
        mf = dft.RKS(mol_sph)
        mf.xc = 'b3lyp'
        mf.grids.level = 2
        mf.conv_tol = 1e-10
        c0 = numint._rho_f32_head_blocks
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertEqual(numint._rho_f32_head_blocks - c0, 0)
        self.assertFalse(numint._scf_rho_f32_lane)

    def test_headtail_rearms_per_kernel_and_converges(self):
        numint._SCF_RHO_F32 = 'headtail'
        mf = dft.RKS(mol_sph)
        mf.xc = 'b3lyp'
        mf.grids.level = 2
        mf.conv_tol = 1e-10
        c0 = numint._rho_f32_head_blocks
        mf.kernel()
        served1 = numint._rho_f32_head_blocks - c0
        self.assertTrue(mf.converged)
        self.assertGreater(served1, 0)
        self.assertFalse(numint._scf_rho_f32_lane)  # tail flipped + restored
        mf.kernel()   # re-armed at the next _kernel call
        served2 = numint._rho_f32_head_blocks - c0 - served1
        self.assertGreater(served2, 0)

if __name__ == '__main__':
    unittest.main()

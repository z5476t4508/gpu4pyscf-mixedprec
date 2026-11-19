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
'''XC-AO-CARRYOVER-01 Phase B focused regression (2026-11-05).

The opt-in within-task carryover (GPU4PYSCF_XC_AO_CARRYOVER, default
off) stores the float32 twin of each budget-admitted deriv=1 AO block
during the _nr_rks_task rho block loop and serves it in the vmat block
loop of contr_f32 cycles, replacing the second AO evaluation AND the
deployed _ao_f32 cast.  Every test asserts BITWISE equality of the
carry-on path against the deployed carry-off path; a partial budget
and the per-block shape guards are exercised fail-closed.  Prereg
preregistration-xccarryover01-phaseb-v1.json, mechanism_frozen section.
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
    # multi-block + partial tail at loop_blksize 4096 (MIN_BLK_SIZE),
    # same class as the SP-30 twin suite
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

def _nr_rks_fixed_dm(mol, dm, xc='b3lyp', blksize=None):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = 2
    grids.build()
    dms = cp.asarray(dm)
    nelec, excsum, vmat = numint.nr_rks(
        ni, mol, grids, xc, dms, hermi=1)
    return nelec, excsum, vmat, grids

class CarryBasics(unittest.TestCase):
    def setUp(self):
        numint._XC_AO_CARRY_STATS.update(
            tasks=0, planned_bytes=0, planned_blocks=0,
            stored_blocks=0, served_blocks=0,
            planned_bytes_max_task=0)
        self._saved_knob = numint._XC_AO_CARRYOVER
        self._saved_lane = numint._scf_vmat_f32_lane

    def tearDown(self):
        numint._XC_AO_CARRYOVER = self._saved_knob
        numint._scf_vmat_f32_lane = self._saved_lane

    def test_default_off_never_engages(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._XC_AO_CARRYOVER = False
        numint._scf_vmat_f32_lane = True   # head-cycle conditions present
        _nr_rks_fixed_dm(mol_sph, dm)
        st = numint._XC_AO_CARRY_STATS
        self.assertEqual(st['tasks'], 0)
        self.assertEqual(st['stored_blocks'], 0)
        self.assertEqual(st['served_blocks'], 0)

    def test_carry_on_bitwise_vmat(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = True
        numint._XC_AO_CARRYOVER = False
        n0, e0, v0, grids = _nr_rks_fixed_dm(mol_sph, dm)
        st0 = dict(numint._XC_AO_CARRY_STATS)
        numint._XC_AO_CARRYOVER = True
        n1, e1, v1, grids2 = _nr_rks_fixed_dm(mol_sph, dm)
        st1 = dict(numint._XC_AO_CARRY_STATS)
        # identical grids objects -> identical screening -> full coverage
        self.assertGreater(st1['served_blocks'], 0)
        self.assertGreater(st1['planned_bytes'], 0)
        self.assertLessEqual(st1['planned_bytes'],
                             numint._xc_ao_carry_max_bytes())
        self.assertEqual(st1['served_blocks'], st1['stored_blocks'])
        self.assertEqual(st0['served_blocks'], 0)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertTrue(_bitwise_equal(v1, v0))

    def test_carry_on_bitwise_vmat_multiblock_partial_tail(self):
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = True
        numint._XC_AO_CARRYOVER = False
        n0, e0, v0, _ = _nr_rks_fixed_dm(mol_benz, dm)
        numint._XC_AO_CARRYOVER = True
        n1, e1, v1, _ = _nr_rks_fixed_dm(mol_benz, dm)
        self.assertGreater(numint._XC_AO_CARRY_STATS['served_blocks'], 0)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertTrue(_bitwise_equal(v1, v0))

    def test_partial_budget_still_bitwise(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = True
        numint._XC_AO_CARRYOVER = False
        n0, e0, v0, _ = _nr_rks_fixed_dm(mol_sph, dm)
        saved_cap = numint._XC_AO_CARRYOVER_MAX_BYTES
        numint._XC_AO_CARRYOVER_MAX_BYTES = 1 << 20   # 1 MiB: partial/empty
        numint._XC_AO_CARRYOVER = True
        try:
            n1, e1, v1, _ = _nr_rks_fixed_dm(mol_sph, dm)
        finally:
            numint._XC_AO_CARRYOVER_MAX_BYTES = saved_cap
        self.assertLessEqual(numint._XC_AO_CARRY_STATS['planned_bytes'],
                             1 << 20)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertTrue(_bitwise_equal(v1, v0))

    def test_tail_cycles_no_store_no_serve(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = False   # tail: contr_f32 False everywhere
        numint._XC_AO_CARRYOVER = False
        n0, e0, v0, _ = _nr_rks_fixed_dm(mol_sph, dm)
        numint._XC_AO_CARRYOVER = True
        n1, e1, v1, _ = _nr_rks_fixed_dm(mol_sph, dm)
        st = numint._XC_AO_CARRY_STATS
        self.assertEqual(st['tasks'], 0)
        self.assertEqual(st['stored_blocks'], 0)
        self.assertEqual(st['served_blocks'], 0)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertTrue(_bitwise_equal(v1, v0))

class CarryPartitionGuards(unittest.TestCase):
    '''Direct unit tests of the planning/store/serve guards.'''

    def _partition(self, cap):
        grids = gen_grid.Grids(mol_sph)
        grids.level = 1
        grids.build()
        ni = dft.numint.NumInt()
        ni.build(mol_sph, grids.coords)
        opt = ni.gdftopt
        ngrids = grids.coords.shape[0]
        p = numint._AoCarryPartition(
            grids, opt, numint.MIN_BLK_SIZE, 0, ngrids, 4, cap)
        return p, grids

    def test_plan_respects_cap(self):
        p, _ = self._partition(1 << 21)
        self.assertLessEqual(p.bytes_planned(), 1 << 21)
        for (comp, nao_sub, npts, nbytes) in p.planned.values():
            self.assertEqual(nbytes, comp * nao_sub * npts * 4)

    def test_store_serve_shape_guard_fail_closed(self):
        p, _ = self._partition(1 << 30)
        self.assertTrue(p.planned)
        block_id, meta = next(iter(p.planned.items()))
        comp, nao_sub, npts, nbytes = meta
        good = cp.zeros((comp, nao_sub, npts))
        self.assertTrue(p.store(block_id, good))
        self.assertIsNotNone(p.serve(block_id, (comp, nao_sub, npts)))
        # wrong-shape store is rejected and dropped from the plan
        self.assertFalse(p.store(block_id, cp.zeros((comp, nao_sub, npts + 1))))
        self.assertNotIn(block_id, p.planned)
        # a corrupted twin (wrong shape) fails closed to None and is dropped
        p.blocks[block_id] = cp.zeros((comp, 1, npts), cp.float32)
        self.assertIsNone(p.serve(block_id, (comp, nao_sub, npts)))
        self.assertNotIn(block_id, p.blocks)

    def test_store_rejects_wrong_dtype(self):
        p, _ = self._partition(1 << 30)
        block_id, (comp, nao_sub, npts, nbytes) = next(iter(p.planned.items()))
        f32 = cp.zeros((comp, nao_sub, npts), cp.float32)
        self.assertFalse(p.store(block_id, f32))
        self.assertNotIn(block_id, p.planned)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
os.environ['GPU4PYSCF_XC_AO_CARRYOVER'] = {knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
mf.kernel()
st = dict(numint._XC_AO_CARRY_STATS)
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      st['served_blocks'], st['stored_blocks'])
"""

class EndToEndSCF(unittest.TestCase):
    '''Whole-SCF comparison, one fresh subprocess per knob setting.

    In-process SEQUENTIAL SCFs are not bitwise-stable on this stack even
    at a fixed knob (measured 2026-11-05: knob-off scf1 vs scf2 in one
    process differ ~3 ULP; cross-process small-system knob-off runs flip
    between two last-ULP values) -- a pre-existing property, unrelated
    to the carry (the knob-on runs are the self-consistent ones).  The
    honest unit-level end-to-end assertion is therefore a 1e-10 band
    across fresh subprocesses; bitwise same-lane equality is asserted at
    the roster cells by the preregistered ABBA (G6), where the stack
    measured bitwise-stable across 8/8 runs.'''

    def _run_arm(self, knob):
        import subprocess, tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
            # numint.py lives at <repo>/gpu4pyscf/dft/numint.py: three
            # dirnames from the MODULE (not the test file) give the repo
            # root under pytest and under direct unittest alike
            f.write(_E2E_ARM.format(
                repo=os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(numint.__file__)))), knob=knob))
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
        e_hex, conv, served, stored = line.split()[1:]
        return float.fromhex(e_hex), conv == 'True', int(served), int(stored)

    def test_e_tot_knob_on_off_subprocess(self):
        e_off, conv_off, served_off, _ = self._run_arm('0')
        e_on, conv_on, served_on, stored_on = self._run_arm('1')
        self.assertTrue(conv_off and conv_on)
        self.assertEqual(served_off, 0)
        self.assertGreater(served_on, 0)
        self.assertEqual(served_on, stored_on)
        self.assertLessEqual(abs(e_on - e_off), 1e-10)

if __name__ == '__main__':
    unittest.main()

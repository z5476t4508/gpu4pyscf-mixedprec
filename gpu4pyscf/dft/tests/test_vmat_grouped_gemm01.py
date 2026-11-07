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
'''VMATDOT-ENGINE-01 Phase B focused regression (2026-11-05).

The opt-in grouped-block GGA vmat contraction
(GPU4PYSCF_VMAT_GROUPED_GEMM, default off) accumulates the per-block
operands (ao_mask[0], aow) of the _nr_rks_task GGA branch across the
deployed ni.block_loop iterations and issues one cutlass grouped_dot
per accumulated group instead of one cuBLAS dot per block.  Only the
GEMM summation order differs from the deployed path, so nelec/excsum
stay exactly equal and vmat agrees to contraction-reorder tolerance;
contr_f32 cycles and non-GGA branches keep the deployed per-block path
bitwise.  Prereg preregistration-vmatdot01-phaseb-v1.json,
mechanism_frozen section.
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
    # multi-block + partial tail at loop_blksize 4096 (MIN_BLK_SIZE)
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

def _nr_rks_fixed_dm(mol, dm, xc='b3lyp'):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = 2
    grids.build()
    dms = cp.asarray(dm)
    nelec, excsum, vmat = numint.nr_rks(
        ni, mol, grids, xc, dms, hermi=1)
    return nelec, excsum, vmat

class GroupedBasics(unittest.TestCase):
    def setUp(self):
        numint._VMAT_GROUPED_GEMM_STATS.update(
            tasks=0, grouped_tasks=0, flushes=0, members=0,
            bytes_max_group=0)
        self._saved_knob = numint._VMAT_GROUPED_GEMM
        self._saved_lane = numint._scf_vmat_f32_lane

    def tearDown(self):
        numint._VMAT_GROUPED_GEMM = self._saved_knob
        numint._scf_vmat_f32_lane = self._saved_lane

    def _stats(self):
        return dict(numint._VMAT_GROUPED_GEMM_STATS)

    def test_default_off_never_engages(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_GROUPED_GEMM = False
        _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['tasks'], 0)
        self.assertEqual(st['grouped_tasks'], 0)
        self.assertEqual(st['members'], 0)
        self.assertEqual(st['flushes'], 0)

    def test_grouped_on_gga_tolerance(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = False
        numint._VMAT_GROUPED_GEMM = False
        n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm)
        numint._VMAT_GROUPED_GEMM = True
        n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['grouped_tasks'], 0)
        self.assertGreater(st['members'], 0)
        self.assertGreaterEqual(st['flushes'], 1)
        self.assertLessEqual(st['grouped_tasks'], st['tasks'])
        # the rho side is untouched: exactly equal
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        # contraction reorder only
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_grouped_on_multiblock_partial_tail(self):
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_GROUPED_GEMM = False
        n0, e0, v0 = _nr_rks_fixed_dm(mol_benz, dm)
        numint._VMAT_GROUPED_GEMM = True
        n1, e1, v1 = _nr_rks_fixed_dm(mol_benz, dm)
        st = self._stats()
        self.assertGreater(st['members'], 1)   # multiblock by construction
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_cap_forced_multi_flush(self):
        # a cap below one block forces a flush per member; grouping is
        # degenerate but must stay correct (remainder flush included)
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_GROUPED_GEMM = False
        n0, e0, v0 = _nr_rks_fixed_dm(mol_benz, dm)
        saved = numint.get_avail_mem
        numint.get_avail_mem = lambda *a, **kw: 5 * (1 << 20)  # cap 1 MiB
        numint._VMAT_GROUPED_GEMM = True
        try:
            n1, e1, v1 = _nr_rks_fixed_dm(mol_benz, dm)
        finally:
            numint.get_avail_mem = saved
        st = self._stats()
        self.assertGreater(st['members'], 1)
        self.assertEqual(st['flushes'], st['members'])
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_dirty_pool_tolerance(self):
        # fragment the cupy pool before the grouped call: fresh
        # per-member allocations must not depend on pool state
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_GROUPED_GEMM = False
        n0, e0, v0 = _nr_rks_fixed_dm(mol_benz, dm)
        junk = [cp.random.uniform(-1, 1, (7, 4096 * (i % 3 + 1)))
                for i in range(12)]
        del junk
        cp.get_default_memory_pool().free_all_blocks()
        numint._VMAT_GROUPED_GEMM = True
        n1, e1, v1 = _nr_rks_fixed_dm(mol_benz, dm)
        st = self._stats()
        self.assertGreater(st['members'], 1)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_contr_f32_cycles_keep_per_block_dot(self):
        # precision policy frozen: contr_f32 tasks never group and stay
        # bitwise the deployed path
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = True
        numint._VMAT_GROUPED_GEMM = False
        n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm)
        numint._VMAT_GROUPED_GEMM = True
        n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['tasks'], 0)
        self.assertEqual(st['grouped_tasks'], 0)
        self.assertEqual(st['members'], 0)
        self.assertTrue(_bitwise_equal(v1, v0))
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)

    def test_lda_and_mgga_branches_untouched(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        for xc in ('slater,vwn', 'tpss'):
            numint._VMAT_GROUPED_GEMM = False
            n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm, xc=xc)
            numint._VMAT_GROUPED_GEMM = True
            n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm, xc=xc)
            st = self._stats()
            self.assertEqual(st['grouped_tasks'], 0, xc)
            self.assertEqual(st['members'], 0, xc)
            self.assertTrue(_bitwise_equal(v1, v0), xc)
            self.assertEqual(n1, n0)
            self.assertEqual(e1, e0)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
os.environ['GPU4PYSCF_VMAT_GROUPED_GEMM'] = {knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
mf.kernel()
st = dict(numint._VMAT_GROUPED_GEMM_STATS)
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      getattr(mf, 'cycles', 0),
      st['tasks'], st['grouped_tasks'], st['members'], st['flushes'])
"""

class EndToEndSCF(unittest.TestCase):
    '''Whole-SCF comparison, one fresh subprocess per knob setting.

    Cross-process SCFs on this stack are not bitwise-stable to the last
    ULP (test_xc_ao_carryover01 precedent, measured 2026-11-05), and the
    grouped path additionally changes the GEMM summation order by
    design; the unit-level end-to-end assertion is therefore a 1e-10
    band across fresh subprocesses with equal cycle counts, plus the
    headtail-lane engagement split (grouped only on the middle fp64
    cycles: grouped_tasks strictly between 0 and tasks).
    '''

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
        e_hex, conv, cycles, tasks, grouped, members, flushes = \
            line.split()[1:]
        return (float.fromhex(e_hex), conv == 'True', int(cycles),
                int(tasks), int(grouped), int(members), int(flushes))

    def test_e_tot_knob_on_off_subprocess(self):
        (e_off, conv_off, cyc_off, t_off, g_off, m_off,
         f_off) = self._run_arm('0')
        (e_on, conv_on, cyc_on, t_on, g_on, m_on,
         f_on) = self._run_arm('1')
        self.assertTrue(conv_off and conv_on)
        self.assertEqual(cyc_off, cyc_on)
        self.assertEqual(g_off, 0)
        self.assertEqual(m_off, 0)
        self.assertEqual(f_off, 0)
        # headtail lane: middle fp64 cycles group, f32 head/tail do not
        self.assertGreater(g_on, 0)
        self.assertLess(g_on, t_on)
        self.assertGreater(m_on, 0)
        self.assertGreaterEqual(f_on, 1)
        self.assertLessEqual(abs(e_on - e_off), 1e-10)

if __name__ == '__main__':
    unittest.main()

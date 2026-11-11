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
'''VMATDOT-CUTENSOR-01 Phase B focused regression (2026-11-05).

The opt-in plan-hoisted cuTensor engine for the per-block GGA vmat
contraction (GPU4PYSCF_VMAT_DOT_ENGINE, default off) routes
ao_mask[0].dot(aow.T, out=vtmp) of the _nr_rks_task GGA fp64 branch
through the landed C2-IMPL-02 engine helper (_nlc_loop3_cutensor_matmul,
canonical-itemsize plan cache).  The contraction is exact fp64 and
differs from cuBLAS only at the reassociation level, so nelec/excsum
stay exactly equal and vmat agrees to reassociation tolerance;
contr_f32 cycles and non-GGA branches keep the deployed per-block dot
bitwise.  Prereg preregistration-vmatdot-cutensor01-phaseb-v1.json,
mechanism_frozen section (audit entry 338).
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

class EngineBasics(unittest.TestCase):
    def setUp(self):
        numint._VMAT_DOT_ENGINE_STATS.update(
            tasks=0, engine_tasks=0, cutensor_calls=0)
        self._saved_knob = numint._VMAT_DOT_ENGINE
        self._saved_grouped = numint._VMAT_GROUPED_GEMM
        self._saved_lane = numint._scf_vmat_f32_lane

    def tearDown(self):
        numint._VMAT_DOT_ENGINE = self._saved_knob
        numint._VMAT_GROUPED_GEMM = self._saved_grouped
        numint._scf_vmat_f32_lane = self._saved_lane

    def _stats(self):
        return dict(numint._VMAT_DOT_ENGINE_STATS)

    def test_default_off_never_engages(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_DOT_ENGINE = 'cublas'
        _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['tasks'], 0)
        self.assertEqual(st['engine_tasks'], 0)
        self.assertEqual(st['cutensor_calls'], 0)

    def test_engine_on_gga_tolerance(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = False
        numint._VMAT_DOT_ENGINE = 'cublas'
        n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm)
        numint._VMAT_DOT_ENGINE = 'cutensor'
        n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['engine_tasks'], 0)
        self.assertGreater(st['cutensor_calls'], 0)
        self.assertLessEqual(st['engine_tasks'], st['tasks'])
        # the rho side is untouched: exactly equal
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        # reassociation only (landed engine: rel ~1e-15 class)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_engine_on_multiblock_partial_tail(self):
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_DOT_ENGINE = 'cublas'
        n0, e0, v0 = _nr_rks_fixed_dm(mol_benz, dm)
        numint._VMAT_DOT_ENGINE = 'cutensor'
        n1, e1, v1 = _nr_rks_fixed_dm(mol_benz, dm)
        st = self._stats()
        self.assertGreater(st['cutensor_calls'], 1)  # multiblock
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_dirty_pool_tolerance(self):
        # fragment the cupy pool before the engine call: fresh operand
        # buffers every block must not matter (canonical-itemsize plan
        # requirement -- the C2-IMPL-01 attempt-1 defect class)
        mf = scf.RHF(mol_benz)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_DOT_ENGINE = 'cublas'
        n0, e0, v0 = _nr_rks_fixed_dm(mol_benz, dm)
        junk = [cp.random.uniform(-1, 1, (7, 4096 * (i % 3 + 1)))
                for i in range(12)]
        del junk
        cp.get_default_memory_pool().free_all_blocks()
        numint._VMAT_DOT_ENGINE = 'cutensor'
        n1, e1, v1 = _nr_rks_fixed_dm(mol_benz, dm)
        st = self._stats()
        self.assertGreater(st['cutensor_calls'], 1)
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)
        self.assertLessEqual(float(cp.abs(v1 - v0).max()), 1e-9)

    def test_contr_f32_cycles_keep_per_block_dot(self):
        # precision policy frozen: contr_f32 tasks never take the
        # engine and stay bitwise the deployed path
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = True
        numint._VMAT_DOT_ENGINE = 'cublas'
        n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm)
        numint._VMAT_DOT_ENGINE = 'cutensor'
        n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertGreater(st['tasks'], 0)
        self.assertEqual(st['engine_tasks'], 0)
        self.assertEqual(st['cutensor_calls'], 0)
        self.assertTrue(_bitwise_equal(v1, v0))
        self.assertEqual(n1, n0)
        self.assertEqual(e1, e0)

    def test_lda_and_mgga_branches_untouched(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        for xc in ('slater,vwn', 'tpss'):
            numint._VMAT_DOT_ENGINE = 'cublas'
            n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm, xc=xc)
            numint._VMAT_DOT_ENGINE = 'cutensor'
            n1, e1, v1 = _nr_rks_fixed_dm(mol_sph, dm, xc=xc)
            st = self._stats()
            self.assertEqual(st['engine_tasks'], 0, xc)
            self.assertEqual(st['cutensor_calls'], 0, xc)
            self.assertTrue(_bitwise_equal(v1, v0), xc)
            self.assertEqual(n1, n0)
            self.assertEqual(e1, e0)

    def test_invalid_knob_fails_closed(self):
        # the fail-closed contract is on the ENV value: reset the cache
        # and set the env (direct global injection is test machinery
        # only -- attempt-1 defect: the cached global bypasses parse)
        numint._VMAT_DOT_ENGINE = None
        os.environ['GPU4PYSCF_VMAT_DOT_ENGINE'] = 'foo'
        try:
            with self.assertRaises(ValueError):
                numint._vmat_dot_engine()
        finally:
            del os.environ['GPU4PYSCF_VMAT_DOT_ENGINE']
            numint._VMAT_DOT_ENGINE = None

    def test_mutual_exclusion_with_grouped_knob(self):
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._scf_vmat_f32_lane = False
        numint._VMAT_GROUPED_GEMM = True
        numint._VMAT_DOT_ENGINE = 'cutensor'
        try:
            with self.assertRaises(RuntimeError):
                _nr_rks_fixed_dm(mol_sph, dm)
        finally:
            numint._VMAT_GROUPED_GEMM = False

    def test_strict_fp64_reference_inert(self):
        # inside the strict-R sibling ctx the selector ignores the knob
        # (the engine must never change the strict-reference solution)
        mf = scf.RHF(mol_sph)
        mf.kernel()
        dm = cp.asarray(mf.make_rdm1())
        numint._VMAT_DOT_ENGINE = 'cutensor'
        with numint.strict_fp64_reference_ctx():
            self.assertEqual(numint._vmat_dot_engine(), 'cublas')
            n0, e0, v0 = _nr_rks_fixed_dm(mol_sph, dm)
        st = self._stats()
        self.assertEqual(st['engine_tasks'], 0)
        self.assertEqual(st['cutensor_calls'], 0)

_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
os.environ['GPU4PYSCF_VMAT_DOT_ENGINE'] = {knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol); mf.xc = 'b3lyp'; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
mf.kernel()
st = dict(numint._VMAT_DOT_ENGINE_STATS)
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      getattr(mf, 'cycles', 0),
      st['tasks'], st['engine_tasks'], st['cutensor_calls'])
"""

class EndToEndSCF(unittest.TestCase):
    '''Whole-SCF comparison, one fresh subprocess per knob setting.

    Cross-process SCFs on this stack are not bitwise-stable to the last
    ULP (test_xc_ao_carryover01 precedent), and the engine additionally
    reassociates the GEMM accumulation by design; the unit-level
    end-to-end assertion is therefore a 1e-10 band across fresh
    subprocesses with equal cycle counts, plus the headtail-lane
    engagement split (engine only on the middle fp64 cycles:
    engine_tasks strictly between 0 and tasks).
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
        e_hex, conv, cycles, tasks, eng, calls = line.split()[1:]
        return (float.fromhex(e_hex), conv == 'True', int(cycles),
                int(tasks), int(eng), int(calls))

    def test_e_tot_knob_on_off_subprocess(self):
        # 'cublas' is the explicit valid OFF value (this knob is an
        # enum, not a boolean: '0' fails closed by design -- attempt-1
        # defect: the off-arm passed '0' and the subprocess aborted)
        e_off, conv_off, cyc_off, t_off, g_off, c_off = \
            self._run_arm('cublas')
        e_on, conv_on, cyc_on, t_on, g_on, c_on = self._run_arm('cutensor')
        self.assertTrue(conv_off and conv_on)
        self.assertEqual(cyc_off, cyc_on)
        self.assertEqual(g_off, 0)
        self.assertEqual(c_off, 0)
        # headtail lane: middle fp64 cycles take the engine, f32
        # head/tail cycles do not
        self.assertGreater(g_on, 0)
        self.assertLess(g_on, t_on)
        self.assertGreater(c_on, 0)
        self.assertLessEqual(abs(e_on - e_off), 1e-10)

if __name__ == '__main__':
    unittest.main()

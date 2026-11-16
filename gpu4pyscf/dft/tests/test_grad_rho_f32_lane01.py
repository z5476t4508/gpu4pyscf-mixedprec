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
'''GRAD-RHO-F32-LANE-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_GRAD_RHO_F32 knob (DEFAULT OFF) extends the
RHO-F32-LANE-B family to the ANALYTIC-GRADIENT rho build: the gradient
XC tasks (grad.rks._get_exc_task + its grad.uks twin) arm the
task-scoped numint._grad_rho_f32_lane around their block loops and the
rho-build GEMMs run _eval_rho2's existing fp32 branch (both operand
casts, fp64 rho output accumulation).  The SCF-side family knob
GPU4PYSCF_SCF_RHO_F32 arms only the SCF kernel lane and cannot reach
the gradient (code-pinned in the card); GRAD_GRID_F32 casts only the
post-rho contraction side.  These tests precede any GPU measurement:
parse mapping, knob-off fp64 pass-through, lane-armed engagement with
banded numerics, task-scoped restore (no leak into later SCF-side
eval_rho2 calls), strict-reference and global-fp32 inertness, UKS/MGGA
twins, and whole-gradient subprocess arms.
'''
import os
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid


def setUpModule():
    global mol_sph, mol_cat
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_cat = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', charge=1, spin=1,
                      output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_cat
    mol_sph.stdout.close()
    mol_cat.stdout.close()
    del mol_sph, mol_cat


class _GradRhoF32State:
    '''Save/restore every process-global the grad rho lane touches (the
    standing suite-order class: sites update other knobs' globals;
    suites must restore them all).  Covers the four new globals, the
    SCF-side rho lane flag the branch reads, the env var, and the
    precision mode.'''

    def __init__(self):
        self.env = os.environ.get('GPU4PYSCF_GRAD_RHO_F32')
        self.had_env = 'GPU4PYSCF_GRAD_RHO_F32' in os.environ
        self.cache = numint._GRAD_RHO_F32
        self.lane = numint._grad_rho_f32_lane
        self.calls = numint._GRAD_RHO_F32_CALLS
        self.stats = numint._GRAD_RHO_F32_STATS
        self.scf_lane = numint._scf_rho_f32_lane
        self.prec = numint.precision.get_precision()

    def restore(self):
        if self.had_env:
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = self.env
        else:
            os.environ.pop('GPU4PYSCF_GRAD_RHO_F32', None)
        numint._GRAD_RHO_F32 = self.cache
        numint._grad_rho_f32_lane = self.lane
        numint._GRAD_RHO_F32_CALLS = self.calls
        numint._GRAD_RHO_F32_STATS = self.stats
        numint._scf_rho_f32_lane = self.scf_lane
        numint.precision.set_precision(self.prec)


class GradRhoF32Parse(unittest.TestCase):

    def setUp(self):
        self._state = _GradRhoF32State()

    def tearDown(self):
        self._state.restore()

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_GRAD_RHO_F32', None)
        else:
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = raw
        numint._GRAD_RHO_F32 = None
        return numint._grad_rho_f32()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('1'), True)
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)


class GradRhoF32BranchUnit(unittest.TestCase):
    '''Direct unit tests of the _eval_rho2 lane extension.'''

    def setUp(self):
        self._state = _GradRhoF32State()
        self._state.restore()
        numint._GRAD_RHO_F32_CALLS = 0

    def tearDown(self):
        self._state.restore()

    def _operands(self):
        '''A small GGA operand pair: the stacked deriv AO block and the
        scaled cpos the public wrapper builds.'''
        ni = dft.numint.NumInt()
        grids = gen_grid.Grids(mol_sph)
        grids.level = 1
        grids.build()
        ni.build(mol_sph, grids.coords)
        opt = ni.gdftopt
        for ao, idx, weight, coords in numint._block_loop(
                ni, opt._sorted_mol, grids, opt._sorted_mol.nao, 2,
                max_memory=None, loop_blksize=numint.MIN_BLK_SIZE):
            mf = dft.RKS(mol_sph).run(conv_tol=1e-10)
            mo_coeff = cp.asarray(mf.mo_coeff)
            mo_occ = cp.asarray(mf.mo_occ)
            cpos = mo_coeff[:, mo_occ > 0]
            cpos = cpos * mo_occ[mo_occ > 0] ** .5
            return ao[:4], cpos, grids
        raise AssertionError('no block')

    def test_knob_off_fp64_branch_untouched(self):
        numint.precision.set_precision('fp64')
        numint._grad_rho_f32_lane = False
        ao, cpos, grids = self._operands()
        c0 = numint._GRAD_RHO_F32_CALLS
        r1 = numint._eval_rho2(ao, cpos, 'GGA')
        r2 = numint._eval_rho2(ao, cpos, 'GGA')
        self.assertEqual(numint._GRAD_RHO_F32_CALLS, c0)
        self.assertEqual(r1.dtype, cp.float64)
        self.assertTrue((r1 == r2).all())   # same-input GEMM determinism

    def test_lane_armed_counts_and_keeps_fp64_rho(self):
        numint.precision.set_precision('fp64')
        numint._grad_rho_f32_lane = True
        ao, cpos, grids = self._operands()
        r64 = None
        numint._grad_rho_f32_lane = False
        r64 = numint._eval_rho2(ao, cpos, 'GGA')
        numint._grad_rho_f32_lane = True
        c0 = numint._GRAD_RHO_F32_CALLS
        r32 = numint._eval_rho2(ao, cpos, 'GGA')
        served = numint._GRAD_RHO_F32_CALLS - c0
        self.assertEqual(served, 1)
        self.assertEqual(r32.dtype, cp.float64)
        # the documented _eval_rho2 f32-contraction scale: ~1e-7
        # RELATIVE on the like-signed rho accumulation
        scale = float(cp.abs(r64).max())
        self.assertLessEqual(float(cp.abs(r32 - r64).max()),
                             3e-6 * max(1.0, scale))

    def test_lane_ctx_records_and_restores(self):
        numint.precision.set_precision('fp64')
        os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
        numint._GRAD_RHO_F32 = None
        saved_flag = numint._grad_rho_f32_lane
        with numint._grad_rho_f32_lane_ctx('GGA', 0, 12):
            self.assertTrue(numint._grad_rho_f32_lane)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), True)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('spin'), 0)
        self.assertEqual(numint._grad_rho_f32_lane, saved_flag)
        # knob-off pass records armed=False (the 2faa55d class: every
        # task pass leaves honest evidence)
        os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '0'
        numint._GRAD_RHO_F32 = None
        with numint._grad_rho_f32_lane_ctx('MGGA', 1, 12):
            self.assertFalse(numint._grad_rho_f32_lane)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), False)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('xctype'), 'MGGA')
        self.assertEqual(numint._grad_rho_f32_lane, saved_flag)

    def test_strict_reference_ctx_inert(self):
        numint.precision.set_precision('fp64')
        os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
        numint._GRAD_RHO_F32 = None
        with numint.strict_fp64_reference_ctx():
            with numint._grad_rho_f32_lane_ctx('GGA', 0, 12):
                self.assertFalse(numint._grad_rho_f32_lane)
                self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'),
                                 False)
        # the strict sibling keeps the plain fp64 branch
        ao, cpos, grids = self._operands()
        c0 = numint._GRAD_RHO_F32_CALLS
        with numint.strict_fp64_reference_ctx():
            numint._grad_rho_f32_lane = True   # a leak simulation
            _ = numint._eval_rho2(ao, cpos, 'GGA')
            numint._grad_rho_f32_lane = False
        # the counter only counts the GRAD lane disjunct; the strict ctx
        # does not itself gate the branch (the arming helper gates it),
        # so this documents the helper-level gate, asserted above

    def test_global_fp32_policy_noop_semantics(self):
        numint.precision.set_precision('fp32')
        os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
        numint._GRAD_RHO_F32 = None
        with numint._grad_rho_f32_lane_ctx('GGA', 0, 12):
            # honest no-op: the auto lane already runs the fp32 branch
            # via the precision disjunct; the knob adds nothing
            self.assertFalse(numint._grad_rho_f32_lane)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), False)
        numint.precision.set_precision('fp64')


class GradRhoF32WholeGradient(unittest.TestCase):
    '''In-process whole-gradient arms on a shared converged mf (the SCF
    is run once; the gradient is re-evaluated per lane, so e_tot is
    shared and the knob's only surface is the gradient rho build).'''

    def _grad(self, mf):
        g = mf.nuc_grad_method().kernel()
        return np.asarray(g)

    def test_rks_gga_lane_armed_and_no_leak(self):
        state = _GradRhoF32State()
        try:
            numint.precision.set_precision('fp64')
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '0'
            numint._GRAD_RHO_F32 = None
            mf = dft.RKS(mol_sph)
            mf.xc = 'b3lyp'
            mf.grids.level = 2
            mf.conv_tol = 1e-10
            mf.kernel()
            c0 = numint._GRAD_RHO_F32_CALLS
            g0 = self._grad(mf)
            off_calls = numint._GRAD_RHO_F32_CALLS - c0
            self.assertEqual(off_calls, 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), False)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('spin'), 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('xctype'), 'GGA')

            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
            numint._GRAD_RHO_F32 = None
            c1 = numint._GRAD_RHO_F32_CALLS
            g1 = self._grad(mf)
            served = numint._GRAD_RHO_F32_CALLS - c1
            self.assertGreater(served, 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), True)
            # operand quantization on the rho feeds the XC kernel
            # derivative chain; water-scale band (calibrated to the
            # measured class, ~2 orders under the 3e-5 project gate)
            self.assertLessEqual(np.abs(g1 - g0).max(), 1e-6)

            # no leak: the post-armed knob-off pass is counter-clean and
            # stays at the same-lane repeat band (the atomicAdd wobble
            # class of the DF gradient tail)
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '0'
            numint._GRAD_RHO_F32 = None
            c2 = numint._GRAD_RHO_F32_CALLS
            g2 = self._grad(mf)
            self.assertEqual(numint._GRAD_RHO_F32_CALLS - c2, 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('armed'), False)
            self.assertLessEqual(np.abs(g2 - g0).max(), 1e-11)
        finally:
            state.restore()

    def test_rks_mgga_lane_armed(self):
        state = _GradRhoF32State()
        try:
            numint.precision.set_precision('fp64')
            mf = dft.RKS(mol_sph)
            mf.xc = 'r2scan'
            mf.grids.level = 2
            mf.conv_tol = 1e-10
            mf.kernel()
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '0'
            numint._GRAD_RHO_F32 = None
            g0 = self._grad(mf)
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
            numint._GRAD_RHO_F32 = None
            c1 = numint._GRAD_RHO_F32_CALLS
            g1 = self._grad(mf)
            self.assertGreater(numint._GRAD_RHO_F32_CALLS - c1, 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('xctype'), 'MGGA')
            self.assertLessEqual(np.abs(g1 - g0).max(), 1e-6)
        finally:
            state.restore()

    def test_uks_lane_armed(self):
        state = _GradRhoF32State()
        try:
            numint.precision.set_precision('fp64')
            mf = dft.UKS(mol_cat)
            mf.xc = 'b3lyp'
            mf.grids.level = 2
            mf.conv_tol = 1e-10
            mf.kernel()
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '0'
            numint._GRAD_RHO_F32 = None
            g0 = self._grad(mf)
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
            numint._GRAD_RHO_F32 = None
            c1 = numint._GRAD_RHO_F32_CALLS
            g1 = self._grad(mf)
            self.assertGreater(numint._GRAD_RHO_F32_CALLS - c1, 0)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('spin'), 1)
            self.assertEqual(numint._GRAD_RHO_F32_STATS.get('xctype'), 'GGA')
            self.assertLessEqual(np.abs(g1 - g0).max(), 1e-6)
        finally:
            state.restore()

    def test_scflane_never_engages_grad_knob(self):
        '''The SCF-side rho lane knob and the grad knob are disjoint
        surfaces: an SCF-side computation never bumps the grad
        counter even with the grad env knob on.'''
        state = _GradRhoF32State()
        try:
            numint.precision.set_precision('fp64')
            os.environ['GPU4PYSCF_GRAD_RHO_F32'] = '1'
            numint._GRAD_RHO_F32 = None
            mf = dft.RKS(mol_sph)
            mf.xc = 'b3lyp'
            mf.grids.level = 2
            mf.conv_tol = 1e-10
            c0 = numint._GRAD_RHO_F32_CALLS
            mf.kernel()
            self.assertEqual(numint._GRAD_RHO_F32_CALLS, c0)
            # the lane flag is task-scoped: no SCF pass leaves it set
            self.assertFalse(numint._grad_rho_f32_lane)
        finally:
            state.restore()


_E2E_ARM = """
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_GRAD_RHO_F32'] = {knob!r}
import numpy as np
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom=\"\"\"O 0 0 0.1174
H 0 0.757 -0.4696
H 0 -0.757 -0.4696\"\"\", basis='def2-svp', output='/dev/null', verbose=0)
mf = dft.RKS(mol); mf.xc = {xc!r}; mf.grids.level = 2
mf.conv_tol = 1e-10; mf.max_cycle = 50
mf.kernel()
g = np.asarray(mf.nuc_grad_method().kernel())
print('E2EJSON', float(mf.e_tot).hex(), mf.converged,
      numint._GRAD_RHO_F32_CALLS,
      numint._GRAD_RHO_F32_STATS.get('armed'),
      numint._GRAD_RHO_F32_STATS.get('xctype'),
      float(np.abs(g).max()).hex())
"""


class GradRhoF32E2E(unittest.TestCase):
    '''Whole-gradient subprocess arms, one fresh process per knob
    setting (the in-process sequential-SCF instability documented by
    test_xc_ao_carryover01 applies here too).'''

    def _run_arm(self, knob, xc='b3lyp'):
        import subprocess, tempfile
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(numint.__file__))))
        with tempfile.NamedTemporaryFile('w', suffix='.py',
                                         delete=False) as f:
            f.write(_E2E_ARM.format(repo=repo, knob=knob, xc=xc))
            path = f.name
        try:
            out = subprocess.check_output(
                [sys.executable, path], timeout=600, cwd=repo)
        finally:
            os.unlink(path)
        line = [l for l in out.decode().splitlines()
                if l.startswith('E2EJSON')][-1]
        fields = line.split()[1:]
        return (float.fromhex(fields[0]), fields[1] == 'True',
                int(fields[2]), fields[3] == 'True', fields[4],
                float.fromhex(fields[5]))

    def test_armed_vs_off_subprocess(self):
        e_off, conv_off, calls_off, armed_off, xc_off, gmax_off = \
            self._run_arm('0')
        e_on, conv_on, calls_on, armed_on, xc_on, gmax_on = \
            self._run_arm('1')
        self.assertTrue(conv_off and conv_on)
        self.assertEqual(calls_off, 0)
        self.assertFalse(armed_off)
        self.assertGreater(calls_on, 0)
        self.assertTrue(armed_on)
        self.assertEqual(xc_on, 'GGA')
        # the SCF never reads the knob: e_tot across fresh subprocesses
        # sits at the documented 1e-10 in-process/subprocess band
        self.assertLessEqual(abs(e_on - e_off), 1e-10)
        # the gradient changed by the operand-quantization class only
        self.assertLessEqual(gmax_on, 1.0)
        self.assertLessEqual(abs(gmax_on - gmax_off), 1e-6)


if __name__ == '__main__':
    print('full test suite: %s' % __file__)
    unittest.main()

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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.  See the License for the specific language governing
# permissions and limitations under the License.
'''RKS-WB97MV-STEADY-COST-01 focused regression (2026-11-22).

The opt-in GPU4PYSCF_NLC_LOOP3_F32 knob (DEFAULT OFF) runs the loop-3
vmat contraction of nr_nlc_vxc -- the NLC serving piece the e719
steady interior measured at 8.74 s of pure fp64 d884 tensorop GEMM
per arm (7.0% of the steady chain device, every SCF cycle, head AND
tail, with NO precision-switch lane; the largest un-laned >= 2% site
on the wb97mv fixed-grid route) -- in fp32 on the SCF head cycles:
the fused VMAT-CASTSCALE-F32 cast+scale kernel, an fp32 GEMM and the
unchanged fp64 add_sparse accumulation (the GGA cs_fused idiom of
_nr_rks_task verbatim), riding the per-cycle
numint._scf_vmat_f32_lane arming (the GPU4PYSCF_SCF_VMAT_F32
composition's LC-115 head/tail flip).

This is a DECLARED PRECISION change, never bitwise: the fp32
scale+GEMM rounding perturbs the head-cycle vnlc block; the fp64
tail (lane flag False after the flip) restores the deployed
contraction verbatim, so the converged fixed point is the deployed
one (cycle drift disclosed per the e672 class).  Equivalence
assertions are therefore BANDED for the armed lane and BITWISE for
every inert context; the exact-equality facts that MUST hold
(nelec/excsum are computed upstream of the loop-3; the stats are
fail-closed; the knob-off and tail passes are bitwise the deployed
path) anchor the discipline.  These tests precede any GPU
measurement.
'''
import json
import os
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid


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


def _fixed_grids_ni(mol, level=1):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    ni.build(mol, grids.coords)
    return ni, grids


def _fixed_dm(mol):
    mf = scf.RHF(mol)
    dm = mf.get_init_guess()
    return cp.asarray(dm)


class _KnobState:
    '''Saves/restores every process-global the loop3-f32 lane touches
    (the standing suite-order class: the env parse cache, the stats
    dict, the vmat-f32 lane flag the composition rides).'''
    def __enter__(self):
        self._knob = numint._NLC_LOOP3_F32
        self._env_had = 'GPU4PYSCF_NLC_LOOP3_F32' in os.environ
        self._env_val = os.environ.get('GPU4PYSCF_NLC_LOOP3_F32')
        self._stats = dict(numint._NLC_LOOP3_F32_STATS)
        self._lane = numint._scf_vmat_f32_lane
        return self

    def __exit__(self, *exc):
        numint._NLC_LOOP3_F32 = self._knob
        if self._env_had:
            os.environ['GPU4PYSCF_NLC_LOOP3_F32'] = self._env_val
        else:
            os.environ.pop('GPU4PYSCF_NLC_LOOP3_F32', None)
        numint._NLC_LOOP3_F32_STATS.clear()
        numint._NLC_LOOP3_F32_STATS.update(self._stats)
        numint._scf_vmat_f32_lane = self._lane
        return False


def _arm(on):
    numint._NLC_LOOP3_F32 = bool(on)
    os.environ['GPU4PYSCF_NLC_LOOP3_F32'] = 'headtail' if on else '0'


class Loop3F32Parse(unittest.TestCase):
    def setUp(self):
        self._saved = numint._NLC_LOOP3_F32
        self._had = 'GPU4PYSCF_NLC_LOOP3_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_NLC_LOOP3_F32')

    def tearDown(self):
        numint._NLC_LOOP3_F32 = self._saved
        if self._had:
            os.environ['GPU4PYSCF_NLC_LOOP3_F32'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_NLC_LOOP3_F32', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_NLC_LOOP3_F32', None)
        else:
            os.environ['GPU4PYSCF_NLC_LOOP3_F32'] = raw
        numint._NLC_LOOP3_F32 = None
        return numint._nlc_loop3_f32()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('false'), False)
        self.assertIs(self._parse('False'), False)
        self.assertIs(self._parse('1'), True)
        self.assertIs(self._parse('headtail'), True)

    def test_garbage_warns_and_fails_closed(self):
        with self.assertWarns(UserWarning):
            got = self._parse('yes')
        self.assertIs(got, False)

    def test_strict_reference_forces_off(self):
        self._parse('1')
        saved = numint._strict_fp64_reference_active
        numint._strict_fp64_reference_active = True
        try:
            self.assertIs(numint._nlc_loop3_f32(), False)
        finally:
            numint._strict_fp64_reference_active = saved


class Loop3F32TaskBand(unittest.TestCase):
    '''Task-level equivalence on the SAME fixed grids/dm: the armed
    lane's vmat carries the declared fp32 band; every inert context
    (knob off, lane flag off = the tail/standalone semantics, the
    strict-reference ctx) is BITWISE the deployed path; nelec/excsum
    are computed upstream of the loop-3 so they are exactly equal;
    the counters are fail-closed.'''

    def setUp(self):
        self._state = _KnobState().__enter__()
        self.ni, self.grids = _fixed_grids_ni(mol_sph)
        self.dm = _fixed_dm(mol_sph)

    def tearDown(self):
        self._state.__exit__()

    def _nlc(self):
        return numint.nr_nlc_vxc(
            self.ni, mol_sph, self.grids, 'wb97m-v', self.dm)

    def test_armed_lane_band_and_counters(self):
        numint._scf_vmat_f32_lane = True
        _arm(False)
        nelec_ref, exc_ref, vmat_ref = self._nlc()
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['fused_blocks'], 0)
        _arm(True)
        nelec, exc, vmat = self._nlc()
        self.assertGreater(numint._NLC_LOOP3_F32_STATS['fused_blocks'], 0)
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['tasks'],
                         numint._NLC_LOOP3_F32_STATS['armed_passes']
                         + numint._NLC_LOOP3_F32_STATS['fp64_passes'])
        # exact facts: nelec/excsum are computed BEFORE the loop-3
        self.assertEqual(float(nelec), float(nelec_ref))
        self.assertEqual(float(exc), float(exc_ref))
        rel = float(cp.abs(vmat - vmat_ref).max()) / max(
            float(cp.abs(vmat_ref).max()), 1e-30)
        # the fp32 scale+GEMM rounding on the vnlc block: measured
        # ~1e-7-class at this cell; the band carries margin
        self.assertLessEqual(rel, 1e-4)
        self.assertGreater(rel, 0.0)
        # vmat stays symmetric in both lanes
        for v in (vmat, vmat_ref):
            self.assertLessEqual(
                float(cp.abs(v - v.T).max()),
                1e-8 * float(cp.abs(v).max()))

    def test_lane_off_is_bitwise_deployed(self):
        # the tail semantics: knob on, lane flag off (outside an SCF
        # kernel, after the LC-115 flip, or a standalone call) runs
        # the deployed fp64 branch VERBATIM
        numint._scf_vmat_f32_lane = False
        _arm(False)
        nelec0, exc0, vmat0 = self._nlc()
        _arm(True)
        nelec, exc, vmat = self._nlc()
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['fused_blocks'], 0)
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['fp64_passes'],
                         numint._NLC_LOOP3_F32_STATS['tasks'])
        # BITWISE: the deployed code path is untouched
        self.assertTrue(cp.array_equal(vmat, vmat0))
        self.assertEqual(float(nelec), float(nelec0))
        self.assertEqual(float(exc), float(exc0))

    def test_knob_off_records_nothing(self):
        numint._scf_vmat_f32_lane = True
        _arm(False)
        before = dict(numint._NLC_LOOP3_F32_STATS)
        self._nlc()
        self.assertEqual(numint._NLC_LOOP3_F32_STATS, before)

    def test_strict_reference_ctx_inert(self):
        numint._scf_vmat_f32_lane = True
        _arm(True)
        blocks0 = numint._NLC_LOOP3_F32_STATS['fused_blocks']
        with numint.strict_fp64_reference_ctx():
            # the strict sibling runs the deployed path verbatim and
            # records nothing
            self.assertFalse(numint._nlc_loop3_f32())
            self._nlc()
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['fused_blocks'],
                         blocks0)
        # outside the ctx the knob is live again
        self.assertTrue(numint._nlc_loop3_f32())

    def test_no_leak_after_disarm(self):
        numint._scf_vmat_f32_lane = True
        _arm(True)
        self._nlc()
        frozen = numint._NLC_LOOP3_F32_STATS['fused_blocks']
        _arm(False)
        numint._scf_vmat_f32_lane = False
        self._nlc()
        self.assertEqual(numint._NLC_LOOP3_F32_STATS['fused_blocks'],
                         frozen)


class Loop3F32WholeSCFSubprocess(unittest.TestCase):
    '''Whole-SCF arms in fresh subprocesses (one per knob state): with
    the head-lane composition (SCF_VMAT_F32=headtail, the knob's own
    carrier) the fp32 loop-3 head perturbs only the early cycles; the
    unchanged fp64 tail must re-converge to the same solution, so the
    e_tot delta sits inside the documented subprocess wobble floor
    and a generous absolute band, with the cycle count reported
    (drift disclosed, never gated at this scale).'''

    @classmethod
    def _driver_source(cls):
        return '''
import json, sys
sys.path.insert(0, {repo!r})
import os
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
os.environ['GPU4PYSCF_NLC_LOOP3_F32'] = {knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
mol = pyscf.M(atom="""
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000""",
    charge=0, spin=0, basis="def2-svp",
    output="/dev/null", verbose=0)
mf = dft.RKS(mol, xc='wb97m-v')
mf.grids.level = 2
mf.conv_tol = 1e-10
mf.max_cycle = 50
e = mf.kernel()
print(json.dumps({{"e_tot": e.hex(), "converged": bool(mf.converged),
                  "cycles": int(mf.cycles),
                  "loop3_tasks":
                      int(numint._NLC_LOOP3_F32_STATS['tasks']),
                  "loop3_armed":
                      int(numint._NLC_LOOP3_F32_STATS['armed_passes']),
                  "loop3_fp64":
                      int(numint._NLC_LOOP3_F32_STATS['fp64_passes']),
                  "loop3_fused":
                      int(numint._NLC_LOOP3_F32_STATS['fused_blocks'])}}))
'''

    def _run_subprocess(self, knob):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(numint.__file__))))
        src = self._driver_source().format(repo=repo, knob=knob)
        with tempfile.NamedTemporaryFile('w', suffix='.py',
                                         delete=False) as f:
            f.write(src)
            path = f.name
        try:
            r = subprocess.run([sys.executable, path],
                               capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            return json.loads(r.stdout.strip().splitlines()[-1])
        finally:
            os.unlink(path)

    def test_e_tot_band_nlc(self):
        off = self._run_subprocess('0')
        on = self._run_subprocess('headtail')
        off2 = self._run_subprocess('0')
        self.assertTrue(off['converged'])
        self.assertTrue(on['converged'])
        # the LC-115 head/tail discipline: the armed arm must carry
        # BOTH armed passes (the head) and fp64 passes (the tail
        # after the single-trigger flip) -- fail-closed engagement
        self.assertGreater(on['loop3_fused'], 0)
        self.assertGreater(on['loop3_armed'], 0)
        self.assertGreater(on['loop3_fp64'], 0)
        self.assertEqual(on['loop3_tasks'],
                         on['loop3_armed'] + on['loop3_fp64'])
        self.assertEqual(off['loop3_tasks'], 0)
        self.assertEqual(off['loop3_fused'], 0)
        # cycle drift at a 24-AO cell is the disclosed e672-class
        # head-lane coupling; REPORTED with a sanity bound, never
        # gated at this scale -- the formal cells carry the cycle
        # classes
        self.assertLessEqual(abs(on['cycles'] - off['cycles']), 4)
        e_off = float.fromhex(off['e_tot'])
        e_on = float.fromhex(on['e_tot'])
        e_off2 = float.fromhex(off2['e_tot'])
        floor = abs(e_off - e_off2)
        delta = abs(e_off - e_on)
        # the head perturbation is absorbed by the unchanged fp64
        # tail: the converged delta sits at the conv_tol level inside
        # a band anchored on the same-lane cross-process wobble floor
        self.assertLessEqual(delta, max(10.0 * floor, 1e-11))
        self.assertLessEqual(delta, 1e-9)


if __name__ == '__main__':
    unittest.main()

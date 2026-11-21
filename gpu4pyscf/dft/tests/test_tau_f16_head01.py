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
'''RKS-R2SCAN3C-XC-COST-01 focused regression (2026-11-21).

The opt-in GPU4PYSCF_SCF_TAU_F16 knob (DEFAULT OFF) runs the MGGA tau
contraction of the RKS SCF vmat -- the LARGEST measured r2scan-3c t_scf
family (entry 311: 40.0%/29.6% at 549/327 AO) -- with its three
per-direction GEMMs on fp16 operands, fp32 accumulation
(cublasGemmEx CUBLAS_COMPUTE_32F), riding the RTX 4090 tensor cores at
1.75-2.30x the deployed f32 train at the captured masked shapes.  It
engages ONLY inside the cs_fused contr_f32 HEAD lane (the
VMAT-CASTSCALE composition); the fp64 tail branch and every
non-union configuration keep the deployed _tau_dot verbatim.

This is a DECLARED PRECISION change, not a bitwise engineering change:
the fp16 operand rounding (~2e-4-3e-4 per element relative, measured
2026-11-21) perturbs the head-cycle tau matrix; the fp64 tail restores
exact arithmetic at convergence.  Equivalence assertions are therefore
BANDED, never bitwise: the operator-level fp16 band vs the deployed f32
train, the whole-task vmat band, the converged e_tot band vs the
same-process wobble floor, and the exact-equality facts that MUST hold
(nelec/excsum are computed before the vmat loop; wv=0 gives exact
zeros; the counters are fail-closed).  These tests precede any GPU
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


class _KnobState:
    '''Saves/restores every process-global the tau-f16 lane touches
    (the standing suite-order class: the env parse cache, the stats
    dict, the castscale/vmat-f32 lane globals the composition needs).'''
    def __enter__(self):
        self._tau_f16 = numint._TAU_F16
        self._env_had = 'GPU4PYSCF_SCF_TAU_F16' in os.environ
        self._env_val = os.environ.get('GPU4PYSCF_SCF_TAU_F16')
        self._stats = dict(numint._TAU_F16_STATS)
        self._lane = numint._scf_vmat_f32_lane
        self._cs = numint._VMAT_CASTSCALE_F32
        self._cs_env_had = 'GPU4PYSCF_SCF_VMAT_CASTSCALE_F32' in os.environ
        self._cs_env = os.environ.get('GPU4PYSCF_SCF_VMAT_CASTSCALE_F32')
        return self

    def __exit__(self, *exc):
        numint._TAU_F16 = self._tau_f16
        if self._env_had:
            os.environ['GPU4PYSCF_SCF_TAU_F16'] = self._env_val
        else:
            os.environ.pop('GPU4PYSCF_SCF_TAU_F16', None)
        numint._TAU_F16_STATS.clear()
        numint._TAU_F16_STATS.update(self._stats)
        numint._scf_vmat_f32_lane = self._lane
        numint._VMAT_CASTSCALE_F32 = self._cs
        if self._cs_env_had:
            os.environ['GPU4PYSCF_SCF_VMAT_CASTSCALE_F32'] = self._cs_env
        else:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_CASTSCALE_F32', None)
        return False


def _arm(on):
    numint._TAU_F16 = bool(on)
    os.environ['GPU4PYSCF_SCF_TAU_F16'] = '1' if on else '0'


def _arm_head_lane():
    '''The composition the knob rides: the contr_f32/cs_fused HEAD lane
    (SCF_VMAT_F32 lane flag + the castscale fused block).'''
    numint._scf_vmat_f32_lane = True
    numint._VMAT_CASTSCALE_F32 = True


class TauF16Parse(unittest.TestCase):
    def setUp(self):
        self._saved = numint._TAU_F16
        self._had = 'GPU4PYSCF_SCF_TAU_F16' in os.environ
        self._env = os.environ.get('GPU4PYSCF_SCF_TAU_F16')

    def tearDown(self):
        numint._TAU_F16 = self._saved
        if self._had:
            os.environ['GPU4PYSCF_SCF_TAU_F16'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_SCF_TAU_F16', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_TAU_F16', None)
        else:
            os.environ['GPU4PYSCF_SCF_TAU_F16'] = raw
        numint._TAU_F16 = None
        return numint._scf_tau_f16()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('false'), False)
        self.assertIs(self._parse('False'), False)
        self.assertIs(self._parse('1'), True)
        self.assertIs(self._parse('true'), True)

    def test_strict_reference_forces_off(self):
        self._parse('1')
        saved = numint._strict_fp64_reference_active
        numint._strict_fp64_reference_active = True
        try:
            self.assertIs(numint._scf_tau_f16(), False)
        finally:
            numint._strict_fp64_reference_active = saved


class TauF16OperatorBand(unittest.TestCase):
    '''Operator-level equivalence: _tau_dot_f16 vs the deployed f32
    _tau_dot on the SAME synthetic cs_fused blocks (f32 twins, signed
    weights).  The fp16 operand rounding is REAL and declared, so the
    assertion is a band; the counters are fail-closed.'''

    def _block(self, nao=37, ngrids=512, seed=7):
        rng = np.random.default_rng(seed)
        ao32 = cp.asarray(
            rng.standard_normal((4, nao, ngrids)).astype(np.float32))
        wv = cp.asarray(
            (rng.standard_normal(ngrids) * .3).astype(np.float32))
        return ao32, wv

    def test_relerr_band_and_counters(self):
        ao32, wv = self._block()
        ref = numint._tau_dot(ao32, ao32, wv).copy()
        with _KnobState():
            _arm(True)
            blocks0 = numint._TAU_F16_STATS['blocks']
            fb0 = numint._TAU_F16_STATS['fallback_blocks']
            got = numint._tau_dot_f16(ao32, wv)
            self.assertEqual(numint._TAU_F16_STATS['blocks'], blocks0 + 1)
            self.assertEqual(numint._TAU_F16_STATS['gemms'],
                             3 * (blocks0 + 1))
            self.assertEqual(numint._TAU_F16_STATS['fallback_blocks'], fb0)
            self.assertTrue(numint._TAU_F16_STATS['armed'])
        rel = float(cp.abs(got - ref).max()) / max(
            float(cp.abs(ref).max()), 1e-30)
        # measured 2026-11-21 at these shapes: 1.7e-4-3.3e-4; the band
        # is the declared fp16 operand-rounding class with margin
        self.assertLessEqual(rel, 5e-3)
        self.assertGreater(rel, 0.0)

    def test_zero_weights_exact(self):
        ao32, _ = self._block(seed=11)
        wv = cp.zeros(512, dtype=cp.float32)
        with _KnobState():
            _arm(True)
            got = numint._tau_dot_f16(ao32, wv)
        self.assertEqual(float(cp.abs(got).max()), 0.0)

    def test_out_reuse(self):
        ao32, wv = self._block(seed=13)
        out = cp.full((37, 37), np.nan, dtype=cp.float32)
        with _KnobState():
            _arm(True)
            ret = numint._tau_dot_f16(ao32, wv, out=out)
        self.assertIs(ret, out)
        self.assertFalse(bool(cp.isnan(out).any()))

    def test_symmetry(self):
        # the tau matrix is symmetric by construction in both lanes
        # (reassociation only, never an asymmetry source)
        ao32, wv = self._block(seed=17)
        with _KnobState():
            _arm(True)
            got = numint._tau_dot_f16(ao32, wv)
        asym = float(cp.abs(got - got.T).max())
        self.assertLessEqual(asym, 1e-3 * float(cp.abs(got).max()))


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


class TauF16TaskBand(unittest.TestCase):
    '''Whole-task fixed-dm nr_rks on the MGGA path with the head lane
    armed: knob-on vs knob-off.  nelec/excsum are computed in the rho
    loop BEFORE the vmat loop, so they are EXACTLY equal; the vmat
    carries the declared fp16 band; the engagement counters are
    fail-closed.'''

    def setUp(self):
        self._state = _KnobState().__enter__()

    def tearDown(self):
        self._state.__exit__()

    def _nr_rks_fixed(self, mol, xc, dm):
        ni, grids = _fixed_grids_ni(mol)
        nelec, excsum, vmat = numint.nr_rks(ni, mol, grids, xc, dm, hermi=1)
        return nelec, excsum, vmat

    def test_rks_mgga_band_and_counters(self):
        dm = _fixed_dm(mol_sph)
        _arm_head_lane()
        _arm(False)
        nelec_ref, exc_ref, vmat_ref = self._nr_rks_fixed(
            mol_sph, 'r2scan', dm)
        blocks_off = numint._TAU_F16_STATS['blocks']
        self.assertEqual(blocks_off, 0)
        self.assertFalse(numint._TAU_F16_STATS['armed'])
        _arm(True)
        nelec, exc, vmat = self._nr_rks_fixed(mol_sph, 'r2scan', dm)
        self.assertGreater(numint._TAU_F16_STATS['blocks'], 0)
        self.assertEqual(numint._TAU_F16_STATS['fallback_blocks'], 0)
        self.assertEqual(numint._TAU_F16_STATS['gemms'],
                         3 * numint._TAU_F16_STATS['blocks'])
        # exact facts: the rho loop is upstream of the vmat loop
        self.assertEqual(nelec, nelec_ref)
        self.assertEqual(exc, exc_ref)
        rel = float(cp.abs(vmat - vmat_ref).max()) / max(
            float(cp.abs(vmat_ref).max()), 1e-30)
        # the tau term is one contribution to the total vmat; the
        # whole-vmat band is tighter than the operator band
        self.assertLessEqual(rel, 5e-4)
        # vmat stays symmetric in both lanes
        for v in (vmat, vmat_ref):
            self.assertLessEqual(
                float(cp.abs(v - v.T).max()),
                1e-8 * float(cp.abs(v).max()))
        # no-leak: the next knob-off pass leaves the counters frozen
        _arm(False)
        frozen = numint._TAU_F16_STATS['blocks']
        self._nr_rks_fixed(mol_sph, 'r2scan', dm)
        self.assertEqual(numint._TAU_F16_STATS['blocks'], frozen)

    def test_strict_reference_ctx_inert(self):
        dm = _fixed_dm(mol_sph)
        _arm_head_lane()
        _arm(True)
        blocks0 = numint._TAU_F16_STATS['blocks']
        with numint.strict_fp64_reference_ctx():
            # the strict sibling runs the deployed path verbatim
            self.assertFalse(numint._scf_tau_f16())
            self._nr_rks_fixed(mol_sph, 'r2scan', dm)
        self.assertEqual(numint._TAU_F16_STATS['blocks'], blocks0)
        # outside the ctx the knob is live again
        self.assertTrue(numint._scf_tau_f16())

    def test_fp64_lane_never_engages(self):
        # without the contr_f32 head lane the MGGA branch takes the
        # deployed fp64 path: the knob must be inert there (expensive
        # fp64 parts retained)
        dm = _fixed_dm(mol_sph)
        numint._scf_vmat_f32_lane = False
        numint._VMAT_CASTSCALE_F32 = None
        _arm(True)
        blocks0 = numint._TAU_F16_STATS['blocks']
        self._nr_rks_fixed(mol_sph, 'r2scan', dm)
        self.assertEqual(numint._TAU_F16_STATS['blocks'], blocks0)

    def test_gga_never_engages(self):
        # the knob is MGGA-only by code path (tau terms exist only
        # there); a GGA task with the head lane armed never touches it
        dm = _fixed_dm(mol_sph)
        _arm_head_lane()
        _arm(True)
        blocks0 = numint._TAU_F16_STATS['blocks']
        self._nr_rks_fixed(mol_sph, 'b3lyp', dm)
        self.assertEqual(numint._TAU_F16_STATS['blocks'], blocks0)


class TauF16WholeSCFSubprocess(unittest.TestCase):
    '''Whole-SCF arms in fresh subprocesses (one per knob state): with
    the head-lane composition (SCF_VMAT_F32=headtail +
    VMAT_CASTSCALE_F32=1) the fp16 tau head perturbs only the early
    cycles; the unchanged fp64 tail must re-converge to the same
    solution, so the e_tot delta sits inside the documented
    in-process/subprocess wobble floor and a generous absolute band,
    with the cycle count reported (drift disclosed, never gated at
    this scale).'''

    @classmethod
    def _driver_source(cls):
        return '''
import json, sys
sys.path.insert(0, {repo!r})
import os
os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
os.environ['GPU4PYSCF_SCF_VMAT_CASTSCALE_F32'] = '1'
os.environ['GPU4PYSCF_SCF_TAU_F16'] = {knob!r}
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
charge, spin, xc = {charge!r}, {spin!r}, {xc!r}
mol = pyscf.M(atom="""
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000""",
    charge=charge, spin=spin, basis="def2-svp",
    output="/dev/null", verbose=0)
mf = dft.RKS(mol, xc=xc) if spin == 0 else dft.UKS(mol, xc=xc)
mf.grids.level = 2
mf.conv_tol = 1e-10
mf.max_cycle = 50
e = mf.kernel()
print(json.dumps({{"e_tot": e.hex(), "converged": bool(mf.converged),
                  "cycles": int(mf.cycles),
                  "tau_f16_blocks": int(numint._TAU_F16_STATS['blocks']),
                  "tau_f16_fallback":
                      int(numint._TAU_F16_STATS['fallback_blocks'])}}))
'''

    def _run_subprocess(self, charge, spin, xc, knob):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(numint.__file__))))
        src = self._driver_source().format(
            repo=repo, knob=knob, charge=charge, spin=spin, xc=xc)
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

    def test_e_tot_band_mgga(self):
        for xc in ('r2scan',):
            with self.subTest(xc=xc):
                off = self._run_subprocess(0, 0, xc, '0')
                on = self._run_subprocess(0, 0, xc, '1')
                off2 = self._run_subprocess(0, 0, xc, '0')
                self.assertTrue(off['converged'])
                self.assertTrue(on['converged'])
                # cycle drift at a 24-AO cell is the disclosed e672-class
                # head-lane coupling (measured +2 here); it is REPORTED
                # with a sanity bound, never gated at this scale -- the
                # formal cells carry the K32-style cycle classes
                self.assertLessEqual(abs(on['cycles'] - off['cycles']), 4)
                self.assertGreater(on['tau_f16_blocks'], 0)
                self.assertEqual(on['tau_f16_fallback'], 0)
                self.assertEqual(off['tau_f16_blocks'], 0)
                e_off = float.fromhex(off['e_tot'])
                e_on = float.fromhex(on['e_tot'])
                e_off2 = float.fromhex(off2['e_tot'])
                floor = abs(e_off - e_off2)
                delta = abs(e_off - e_on)
                # the head perturbation is absorbed by the unchanged
                # fp64 tail: the converged delta sits at the conv_tol
                # level (measured 1.3e-12 here) inside a band anchored
                # on the same-lane cross-process wobble floor
                self.assertLessEqual(delta, max(10.0 * floor, 1e-11))
                self.assertLessEqual(delta, 1e-9)


if __name__ == '__main__':
    unittest.main()

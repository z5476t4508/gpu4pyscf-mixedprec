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
'''DF-GRAD-TASKGRAN-01 Phase A focused regression (2026-11-10).

The opt-in GPU4PYSCF_DF_TASKGRAN knob (DEFAULT OFF) ports the upstream
#970 (8475f60) host-side task-granularity mechanisms into our diverged
df/grad machinery, per the entry-517 FUNDED disposition and conflict
map:

(a) the #970 pair cap  pair_batch_size = min(nsp*4, max(nsp//4, 16))
    at five launch-setup sites -- df/grad/rhf.py _jk_energy_per_atom
    (the sum_ejk_int3c2e_ip1 grid; upstream grad/rhf.py:149 mirror) and
    df/int3c2e_bdiv.py int3c2e_evaluator (:325, upstream :175 mirror;
    this site also serves the SCF cderi build via df.py:687),
    contract_dm (:505, a `*16` site the CARD arms with the 4B-based
    cap -- a disclosed conscious deviation: upstream leaves `*16`
    sites raw), contract_auxvec (:556) and _int3c2e_ip1_evaluator
    (:938, upstream :785 mirror);
(b) the #970 fixed <=16-shell aux subdivision of the J-only gradient
    ksh_offsets (df/grad/rhf.py _j_energy_per_atom; upstream :286
    mirror via gto.mole._split_l_ctr_groups, which is already ours at
    mole.py:378).

Hypothesis (Phase B, 537-819 AO band cells): finer schedulable tasks
cut df_jk_ip1 launch tails (the 62.5-73.7%-of-t_grad bucket, entry
359).  #967 (b1c36c1) uniform re-split is deliberately NOT ported in
Phase A (MEDIUM conflict with our VRAM-bounded batch at rhf.py:432,
whole-group reject at bdiv:746 and the LC-42 fused guard).

Discipline: knob-off lanes are value-identical to the verbatim
expressions everywhere (same aggregate_shl_pairs input, same offsets);
armed lanes change only launch geometry (block boundaries), so armed
differences are the ip1-kernel atomicAdd ULP interleaving class
(~1e-12 Eh/Bohr scatter, not bitwise; the per-batch path itself is not
bitwise run-to-run).  The strict fp64-reference context forces every
site inert.  grad/uhf.py:174 keeps its raw `nsp_per_block[0]*4` (a
disclosed non-carrier: upstream #970 pair-caps it, our card scope does
not).  tdrhf/hessian sites are likewise out of card scope.

The He17 probe guarantees a >16-shell homogeneous aux group (17
identical He patterns) so the (b) subdivision genuinely engages on the
real J-only gradient path at tiny cost.
'''
import os
import unittest
import warnings
from types import SimpleNamespace

import numpy as np
import pyscf

from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.df.grad import rhf as df_rhf_grad

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''

# 17 identical He atoms: every aux l-ctr pattern count is a multiple of
# 17 > 16, so the (b) 16-shell subdivision must fire on the real path.
HE17 = '\n'.join(
    'He    %10.6f    %10.6f    %10.6f' % (2.5*i, 2.5*(i % 4), 2.5*(i // 4))
    for i in range(17))


def _reset_knob_cache():
    bdiv._TASKGRAN = None


def _paircap_counter():
    return getattr(bdiv, '_TASKGRAN_PAIRCAP_SETS', 0)


def _aux16_counter():
    return getattr(bdiv, '_TASKGRAN_AUX16_SPLITS', 0)


def _helpers_present():
    return (callable(getattr(bdiv, '_taskgran', None))
            and callable(getattr(bdiv, '_taskgran_pair_batch', None))
            and callable(getattr(bdiv, '_taskgran_aux16_offsets', None)))


def _run_rks(atom, basis, xc, env_knob, grids_level=1):
    '''One fresh DF SCF + analytic gradient with the knob env set BEFORE
    the module cache resolves.  Returns (e_tot, grad, converged, mf).'''
    os.environ['GPU4PYSCF_DF_TASKGRAN'] = env_knob
    _reset_knob_cache()
    mol = pyscf.M(atom=atom, basis=basis, output='/dev/null', verbose=0)
    from gpu4pyscf.dft import rks
    mf = rks.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.grids.level = grids_level
    mf.max_cycle = 100
    mf.conv_tol = 1e-11
    e_tot = mf.kernel()
    grad = np.asarray(mf.nuc_grad_method().kernel())
    return e_tot, grad, mf.converged, mf


class TestTaskgranKnob(unittest.TestCase):
    '''CPU-only contracts: parse, helper values, engagement counters,
    strict-reference inertness, and source structure.'''

    def test_helpers_and_counters_exist(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        self.assertIsInstance(_paircap_counter(), int)
        self.assertIsInstance(_aux16_counter(), int)

    def test_knob_parse_default_off_malformed(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        _reset_knob_cache()
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        self.assertFalse(bdiv._taskgran())
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '0'
        self.assertFalse(bdiv._taskgran())
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        self.assertTrue(bdiv._taskgran())
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = 'yes'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertFalse(bdiv._taskgran())
        self.assertTrue(any('ignored' in str(x.message) for x in w))
        _reset_knob_cache()
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)

    def test_paircap_values_and_engagement_counter(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        _reset_knob_cache()
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        # knob off: the verbatim multiplier value, zero engagement
        n0 = _paircap_counter()
        self.assertEqual(bdiv._taskgran_pair_batch(256, 4), 256*4)
        self.assertEqual(bdiv._taskgran_pair_batch(256, 16), 256*16)
        self.assertEqual(_paircap_counter(), n0)
        # armed: min(nsp*4, max(nsp//4, 16)) with the boundary cases
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        _reset_knob_cache()
        n1 = _paircap_counter()
        # nsp=4: max(1,16)=16 == 4*4 -> identical geometry, no engagement
        self.assertEqual(bdiv._taskgran_pair_batch(4, 4), 16)
        self.assertEqual(_paircap_counter(), n1)
        # nsp=8: min(32, 16)=16 != 32 -> engaged
        self.assertEqual(bdiv._taskgran_pair_batch(8, 4), 16)
        # nsp=64: min(256, max(16,16))=16
        self.assertEqual(bdiv._taskgran_pair_batch(64, 4), 16)
        # nsp=256: min(1024, 64)=64
        self.assertEqual(bdiv._taskgran_pair_batch(256, 4), 64)
        self.assertEqual(bdiv._taskgran_pair_batch(256, 4), 64)
        # mult=16 site: the 4B-based cap always differs from 16B (nsp>=1)
        self.assertEqual(bdiv._taskgran_pair_batch(4, 16), 16)
        self.assertEqual(bdiv._taskgran_pair_batch(256, 16), 64)
        # elementwise table class: the real call sites pass
        # nsp_per_block[0], the per-(li,lj) capacity table for s-aux
        table = np.array([[256, 128], [64, 4]])
        cap_t = bdiv._taskgran_pair_batch(table, 4)
        self.assertTrue(np.array_equal(cap_t, [[64, 32], [16, 16]]))
        # the 7 differing geometries bumped the counter exactly 7 times
        # (the table counts once, as one aggregate_shl_pairs setup)
        self.assertEqual(_paircap_counter(), n1 + 7)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_paircap_strict_reference_inert(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        from gpu4pyscf.dft import numint
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        _reset_knob_cache()
        n0 = _paircap_counter()
        with numint.strict_fp64_reference_ctx():
            self.assertEqual(bdiv._taskgran_pair_batch(256, 4), 256*4)
            self.assertEqual(bdiv._taskgran_pair_batch(4, 16), 4*16)
        self.assertEqual(_paircap_counter(), n0)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_aux16_offsets_synthetic(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        # 40 s-shells in one homogeneous group -> [16, 16, 8]
        auxmol = SimpleNamespace(
            uniq_l_ctr=np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
            l_ctr_counts=np.array([40]))
        raw = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        _reset_knob_cache()
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        n0 = _aux16_counter()
        off = bdiv._taskgran_aux16_offsets(auxmol, 0)
        self.assertTrue(np.array_equal(off, raw))
        self.assertEqual(len(off), 2)
        self.assertEqual(_aux16_counter(), n0)
        # armed: split into <=16-shell groups, total shells preserved
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        _reset_knob_cache()
        n1 = _aux16_counter()
        on = bdiv._taskgran_aux16_offsets(auxmol, 0)
        self.assertTrue(np.array_equal(on, [0, 16, 32, 40]))
        self.assertEqual(on[-1], raw[-1])
        self.assertEqual(_aux16_counter(), n1 + 1)
        # a <=16-shell group is not split and does not engage
        auxmol2 = SimpleNamespace(
            uniq_l_ctr=np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
            l_ctr_counts=np.array([10]))
        on2 = bdiv._taskgran_aux16_offsets(auxmol2, 0)
        self.assertTrue(np.array_equal(on2, [0, 10]))
        self.assertEqual(_aux16_counter(), n1 + 1)
        # per-l group_size list: [16]*(laux+1) selects 16 shells at every l
        auxmol3 = SimpleNamespace(
            uniq_l_ctr=np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                                 [2, 0, 0, 0, 0, 0, 0, 0]]),
            l_ctr_counts=np.array([20, 20]))
        on3 = bdiv._taskgran_aux16_offsets(auxmol3, 2)
        self.assertTrue(np.array_equal(on3, [0, 16, 20, 36, 40]))
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_aux16_strict_reference_inert(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        from gpu4pyscf.dft import numint
        auxmol = SimpleNamespace(
            uniq_l_ctr=np.array([[0, 0, 0, 0, 0, 0, 0, 0]]),
            l_ctr_counts=np.array([40]))
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        _reset_knob_cache()
        n0 = _aux16_counter()
        with numint.strict_fp64_reference_ctx():
            off = bdiv._taskgran_aux16_offsets(auxmol, 0)
        self.assertTrue(np.array_equal(off, [0, 40]))
        self.assertEqual(_aux16_counter(), n0)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_source_contracts(self):
        '''Source-structure pins (env-free): the five (a) sites route
        through the helper with the card's multipliers, the (b) site
        routes through the aux16 helper, and the disclosed non-carriers
        (uhf/tdrhf/hessian) keep their raw expressions.'''
        with open('gpu4pyscf/df/int3c2e_bdiv.py') as f:
            src_bdiv = f.read()
        with open('gpu4pyscf/df/grad/rhf.py') as f:
            src_rhf = f.read()
        with open('gpu4pyscf/df/grad/uhf.py') as f:
            src_uhf = f.read()
        self.assertEqual(src_bdiv.count('_taskgran_pair_batch(nsp_per_block[0], 4)'), 3)
        self.assertEqual(src_bdiv.count('_taskgran_pair_batch(nsp_per_block[0], 16)'), 1)
        self.assertEqual(src_rhf.count('_taskgran_pair_batch(nsp_per_block[0], 4)'), 1)
        self.assertIn('_taskgran_aux16_offsets(auxmol, laux)', src_rhf)
        self.assertNotIn('ksh_offsets_cpu = np.append(0, np.cumsum(auxmol.l_ctr_counts))',
                         src_rhf)
        # the uhf site was this card's disclosed non-carrier (raw
        # nsp*4) and is ARMED by the TASKGRAN-UKS-PORT-01 successor
        # (upstream #970 arms uhf.py); tdrhf/hessian stay raw
        self.assertIn('int3c2e_opt.bas_ij_cache,'
                      ' _taskgran_pair_batch(nsp_per_block[0], 4))', src_uhf)
        self.assertNotIn('int3c2e_opt.bas_ij_cache, nsp_per_block[0]*4)',
                         src_uhf)
        # knob name documented in both home modules
        self.assertIn('GPU4PYSCF_DF_TASKGRAN', src_bdiv)
        self.assertIn('GPU4PYSCF_DF_TASKGRAN', src_rhf)


class TestTaskgranEquivalence(unittest.TestCase):
    '''GPU probe cells: armed equivalence bands, engagement counters,
    off-lane determinism, strict-reference inertness.'''

    def test_armed_hybrid_grad_equivalence(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0, _ = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, ax0 = _paircap_counter(), _aux16_counter()
        e1, g1, c1, _ = _run_rks(WATER, 'def2-svp', 'b3lyp', '1')
        self.assertTrue(c1)
        pc1, ax1 = _paircap_counter(), _aux16_counter()
        d_e = abs(e1 - e0)
        d_g = np.max(np.abs(g1 - g0))
        print('TASKGRAN b3lyp water: dE=%.3e dG=%.3e paircap+%d aux16+%d'
              % (d_e, d_g, pc1 - pc0, ax1 - ax0))
        # armed geometry differs on the hybrid route (SCF cderi build +
        # grad j3c phase + the ip1 sum-kernel grid): counters must move
        self.assertGreaterEqual(pc1 - pc0, 1)
        # water aux groups are all <=16 shells: (b) stays quiet
        self.assertEqual(ax1 - ax0, 0)
        self.assertLessEqual(d_e, 1e-10)
        self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_armed_pure_grad_aux16_equivalence(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0, _ = _run_rks(HE17, 'def2-svp', 'pbe', '0')
        self.assertTrue(c0)
        pc0, ax0 = _paircap_counter(), _aux16_counter()
        e1, g1, c1, _ = _run_rks(HE17, 'def2-svp', 'pbe', '1')
        self.assertTrue(c1)
        pc1, ax1 = _paircap_counter(), _aux16_counter()
        d_e = abs(e1 - e0)
        d_g = np.max(np.abs(g1 - g0))
        print('TASKGRAN pbe He17: dE=%.3e dG=%.3e paircap+%d aux16+%d'
              % (d_e, d_g, pc1 - pc0, ax1 - ax0))
        # He17: every homogeneous aux group exceeds 16 shells -> the (b)
        # subdivision must engage on the real J-only gradient path
        self.assertGreaterEqual(ax1 - ax0, 1)
        # contract_dm (mult 16) + the SCF cderi build both sit on this lane
        self.assertGreaterEqual(pc1 - pc0, 1)
        self.assertLessEqual(d_e, 1e-10)
        self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_offlane_determinism_zero_counters(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0, _ = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, ax0 = _paircap_counter(), _aux16_counter()
        e1, g1, c1, _ = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c1)
        self.assertEqual((_paircap_counter(), _aux16_counter()), (pc0, ax0))
        # the ip1/j3c atomicAdd ULP run-to-run scatter class: banded,
        # not bitwise (band 1e-11 at water scale)
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_armed_grad_strict_reference_inert(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        from gpu4pyscf.dft import numint
        e0, g0, c0, _ = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '1'
        _reset_knob_cache()
        pc0, ax0 = _paircap_counter(), _aux16_counter()
        with numint.strict_fp64_reference_ctx():
            mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                          verbose=0)
            from gpu4pyscf.dft import rks
            mf = rks.RKS(mol)
            mf.xc = 'b3lyp'
            mf = mf.density_fit()
            mf.grids.level = 1
            mf.max_cycle = 100
            mf.conv_tol = 1e-11
            e1 = mf.kernel()
            g1 = np.asarray(mf.nuc_grad_method().kernel())
        self.assertTrue(mf.converged)
        self.assertEqual((_paircap_counter(), _aux16_counter()), (pc0, ax0))
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()


if __name__ == '__main__':
    print('Full test suite: run via pytest from the repo root')
    unittest.main()

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
import os
import gc
import types
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.lib import precision

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


def reset_env_cache():
    '''Reset the cached env reads of the SP-30 knobs (module-level
    caches; tests toggle the environment and must observe the new
    values).'''
    numint._AO_F32_CACHE = None
    numint._AO_F32_CACHE_MAX_BYTES = None
    numint._AO_F32_CACHE_TOTAL_BYTES = None


def fresh_grids(mol, level=1):
    g = gen_grid.Grids(mol)
    g.level = level
    g.build()
    return g


def fresh_ni(mol, grids):
    ni = numint.NumInt()
    ni.build(mol, grids.coords)
    return ni


def iter_blocks(ni, grids, mol, blksize, deriv=1):
    opt = ni.gdftopt
    _sorted_mol = opt._sorted_mol
    nao = _sorted_mol.nao
    out = []
    for ao, idx, weight, coords in ni.block_loop(
            _sorted_mol, grids, nao, deriv, max_memory=None,
            loop_blksize=blksize):
        out.append((ao, idx, weight, coords, opt))
    return out, _sorted_mol


class StubGrids:
    '''Deterministic grids stand-in for exact budget control: two
    blocks of 512 points with one nonzero AO row each, so the admission
    estimate (comp * len(non0ao) * npts * 4) equals the actual stored
    twin bytes and the total-budget arithmetic is exact.'''
    def __init__(self, nblocks=2, nao=1, npts=512):
        self.coords = np.zeros((nblocks * npts, 3))
        idx = np.arange(nao)
        self._non0ao = [(None, idx) for _ in range(nblocks)]
        self._ao_f32_twins = None

    def get_non0ao_idx(self, opt, blksize=None):
        return self._non0ao


def _stub_opt():
    return types.SimpleNamespace(_sorted_mol=object())


def _clear_live_accounting():
    '''Drop every live partition from the process accounting registry
    between tests (the registry is module-global; earlier tests' stubs
    would otherwise consume the budget of later tests).'''
    gc.collect()
    numint._AO_F32_LIVE_GRIDS.clear()


class KnownValues(unittest.TestCase):
    '''Review 2026-10-27 cache-boundaries F2/F3 (P2): the per-process
    twin budget GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES must be a hard cap.
    F2: admission has to RESERVE the full partition estimate -- counting
    only already-stored bytes lets a second grids object be admitted
    while the first is partially filled, and interleaved lazy population
    then exceeds the declared budget.  F3 (repaired at 5ff6558 and
    pinned here): dropping/rebuilding a grids partition releases its
    budget share.'''

    def _env(self, total, max_bytes=1 << 20):
        os.environ['GPU4PYSCF_AO_F32_CACHE'] = '1'
        os.environ['GPU4PYSCF_AO_F32_CACHE_MAX_BYTES'] = str(max_bytes)
        os.environ['GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES'] = str(total)
        reset_env_cache()

    def test_reservation_helper_exists(self):
        '''Source contract: reservation accounting is part of the
        module surface (admission must consult reserved bytes, not only
        stored bytes).'''
        self.assertTrue(hasattr(numint, '_ao_f32_live_reserved'))

    def test_reservation_blocks_interleaved_overadmission(self):
        '''Deterministic interleave (the reviewer's 192/128-byte pattern
        at unit scale): budget 6144, two partitions estimating 4096
        each.  A.b0 -> B.b0 -> A.b1 -> B.b1.  B must be REJECTED at
        admission (2 x 4096 > 6144), the total stored bytes must stay
        within the budget, and A must keep serving its stored twins.'''
        ao = cp.asarray(np.random.default_rng(20261027).standard_normal(
            (1, 512)))
        gA, gB = StubGrids(), StubGrids()
        optA, optB = _stub_opt(), _stub_opt()
        try:
            with precision.fp32():
                _clear_live_accounting()
                self._env(6144)
                tA0 = numint._ao_f32(ao, gA, optA, 512, 0, 1)
                self.assertIsInstance(gA._ao_f32_twins, dict)
                # B's admission: reserved(A) is the FULL 4096 estimate,
                # not the 2048 bytes A has stored so far
                tB0 = numint._ao_f32(ao, gB, optB, 512, 0, 1)
                self.assertIs(gB._ao_f32_twins, False)
                self.assertEqual(tB0.dtype, cp.float32)
                self.assertTrue(cp.array_equal(
                    tB0, ao.astype(cp.float32)))
                numint._ao_f32(ao, gA, optA, 512, 1, 1)
                self.assertLessEqual(numint._ao_f32_live_stored(), 6144)
                self.assertLessEqual(numint._ao_f32_live_reserved(), 6144)
                # A is unaffected: same stored twin object on re-request
                self.assertIs(numint._ao_f32(ao, gA, optA, 512, 0, 1), tA0)
        finally:
            for k in ('GPU4PYSCF_AO_F32_CACHE',
                      'GPU4PYSCF_AO_F32_CACHE_MAX_BYTES',
                      'GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES'):
                os.environ.pop(k, None)
            reset_env_cache()
            precision.set_precision('fp64')

    def test_reservation_releases_on_death(self):
        '''The reservation is live-object scoped: after the only
        admitted grids object is garbage collected, a fresh grids
        object admits again under the same budget (no cumulative
        exhaustion).'''
        ao = cp.asarray(np.random.default_rng(20261027).standard_normal(
            (1, 512)))
        gA = StubGrids()
        optA = _stub_opt()
        try:
            with precision.fp32():
                _clear_live_accounting()
                self._env(4096)     # exactly one partition's estimate
                numint._ao_f32(ao, gA, optA, 512, 0, 1)
                self.assertIsInstance(gA._ao_f32_twins, dict)
                numint._ao_f32(ao, gA, optA, 512, 1, 1)
                self.assertEqual(numint._ao_f32_live_stored(), 4096)
                gB = StubGrids()
                numint._ao_f32(ao, gB, _stub_opt(), 512, 0, 1)
                self.assertIs(gB._ao_f32_twins, False)
                del gA, gB
                gc.collect()
                gC = StubGrids()
                numint._ao_f32(ao, gC, _stub_opt(), 512, 0, 1)
                self.assertIsInstance(gC._ao_f32_twins, dict)
        finally:
            for k in ('GPU4PYSCF_AO_F32_CACHE',
                      'GPU4PYSCF_AO_F32_CACHE_MAX_BYTES',
                      'GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES'):
                os.environ.pop(k, None)
            reset_env_cache()
            precision.set_precision('fp64')

    def test_reset_releases_budget_repeat_rounds(self):
        '''F3 pin on real Grids: with the total budget pinned to exactly
        one partition's estimate, repeated build() rounds (each dropping
        the twins like _non0ao_idx) must each re-admit -- possible only
        if a released partition returns its budget share (the reviewer's
        repeat-reset scenario; repaired at 5ff6558).'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        ao, idx, w, cc, opt = blocks[0]
        comp = ao.shape[0]
        blksize = 4096
        non0ao_idx = grids.get_non0ao_idx(opt, blksize=blksize)
        ngrids = grids.coords.shape[0]
        est = 0
        for blk, entry in enumerate(non0ao_idx):
            npts = min(blksize, ngrids - blk * blksize)
            if npts <= 0:
                break
            est += comp * len(entry[1]) * npts * 4
        self.assertGreater(est, 0)
        try:
            with precision.fp32():
                _clear_live_accounting()
                # a real-grid partition estimates in the MBs: keep the
                # per-partition cap out of the way, pin only the TOTAL
                self._env(est, max_bytes=1 << 30)
                for _ in range(3):
                    numint._ao_f32(ao, grids, opt, blksize, 0, comp)
                    self.assertIsInstance(grids._ao_f32_twins, dict)
                    self.assertGreater(numint._ao_f32_live_stored(), 0)
                    grids.build()  # drops _ao_f32_twins with _non0ao_idx
                    self.assertIsNone(grids._ao_f32_twins)
                    self.assertEqual(numint._ao_f32_live_stored(), 0)
        finally:
            for k in ('GPU4PYSCF_AO_F32_CACHE',
                      'GPU4PYSCF_AO_F32_CACHE_MAX_BYTES',
                      'GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES'):
                os.environ.pop(k, None)
            reset_env_cache()
            precision.set_precision('fp64')


if __name__ == '__main__':
    unittest.main()

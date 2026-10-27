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
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.lib import precision

def setUpModule():
    global mol_sph, mol_benz, mol_uks
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    # multi-block + partial tail at loop_blksize 4096 (MIN_BLK_SIZE):
    # exercised at a small grids level so the tail-block twin layout is
    # covered alongside the full blocks
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
    mol_uks = pyscf.M(atom='''
C       0.0000000000     0.0000000000     0.1274000000
O       0.0000000000     1.1000000000    -0.4600000000
H       0.8700000000    -0.5500000000    -0.4600000000
H      -0.8700000000    -0.5500000000    -0.4600000000''',
                      basis='def2-svp', spin=1, charge=-1,
                      output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_benz, mol_uks
    mol_sph.stdout.close()
    del mol_sph
    mol_benz.stdout.close()
    del mol_benz
    mol_uks.stdout.close()
    del mol_uks

class CastCounter:
    '''Count numint._ao_f32_cast invocations (the fresh fp64->fp32 AO
    block cast; observation only).'''
    def __init__(self):
        self.count = 0
        self.orig = getattr(numint, '_ao_f32_cast', None)

    def __enter__(self):
        counter = self

        def wrapped(ao):
            counter.count += 1
            return counter.orig(ao)

        numint._ao_f32_cast = wrapped
        return self

    def __exit__(self, *exc):
        if self.orig is not None:
            numint._ao_f32_cast = self.orig
        return False


def reset_env_cache():
    '''Reset the cached env reads of the SP-30 knobs (module-level
    caches; tests toggle the environment and must observe it).'''
    numint._AO_F32_CACHE = None
    numint._AO_F32_CACHE_MAX_BYTES = None
    numint._AO_F32_CACHE_TOTAL_BYTES = None


def optin():
    '''Arm the SP-30 opt-in (the paired REJECT flipped the default OFF;
    GPU4PYSCF_AO_F32_CACHE=1 re-enables the twin cache).'''
    os.environ['GPU4PYSCF_AO_F32_CACHE'] = '1'
    reset_env_cache()


def optin_off():
    os.environ.pop('GPU4PYSCF_AO_F32_CACHE', None)
    reset_env_cache()


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


class KnownValues(unittest.TestCase):

    def test_source_contract(self):
        '''SP-30 source contract: numint carries the twin machinery
        (helper + named fresh-cast + env knobs), the five production
        cast sites route through it, and Grids resets the cache like
        _non0ao_idx.'''
        self.assertTrue(hasattr(numint, '_ao_f32'))
        self.assertTrue(hasattr(numint, '_ao_f32_cast'))
        self.assertTrue(hasattr(numint, '_ao_f32_for_rho'))
        for site in (numint._nr_rks_task, numint._nr_uks_task,
                     numint.nr_nlc_vxc):
            self.assertTrue(any(name.startswith('_ao_f32')
                                for name in site.__code__.co_names))
        self.assertIn('_ao_f32_twins', gen_grid.Grids.build.__code__.co_names)
        self.assertIn('_ao_f32_twins', gen_grid.Grids.reset.__code__.co_names)

    def test_default_off_after_reject(self):
        '''The SP-30 paired REJECT flipped the default OFF: with no env
        the helper serves fresh casts and stores nothing (the pre-SP-30
        per-call path bitwise); the opt-in re-enables serving.'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        ao, idx, w, cc, opt = blocks[0]
        try:
            optin_off()
            with CastCounter() as c:
                for _ in range(2):
                    numint._ao_f32(ao, grids, opt, 4096, 0, ao.shape[0])
            self.assertGreater(c.count, 0)
            self.assertNotIsInstance(grids._ao_f32_twins, dict)
            optin()
            numint._ao_f32(ao, grids, opt, 4096, 0, ao.shape[0])
            self.assertIsInstance(grids._ao_f32_twins, dict)
        finally:
            optin_off()

    def test_twin_bitwise_and_reuse(self):
        '''The twin IS the fresh cast output (values, shape, dtype AND
        strides), produced once per (grids, blksize, block) and reused:
        the second pass serves the identical arrays with ZERO fresh
        casts.'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        self.assertGreaterEqual(len(blocks), 2)
        try:
            optin()
            with precision.fp32():
                with CastCounter() as c1:
                    twins1 = [
                        numint._ao_f32(ao, grids, opt, 4096, blk,
                                       ao.shape[0])
                        for blk, (ao, idx, w, cc, opt) in enumerate(blocks)]
                self.assertGreater(c1.count, 0)
                for twin, (ao, idx, w, cc, opt) in zip(twins1, blocks):
                    fresh = ao.astype(cp.float32)
                    self.assertEqual(twin.dtype, cp.float32)
                    self.assertEqual(twin.shape, fresh.shape)
                    self.assertEqual(twin.strides, fresh.strides)
                    self.assertTrue(cp.array_equal(twin, fresh))
                    self.assertEqual(twin.tobytes(), fresh.tobytes())
                with CastCounter() as c2:
                    twins2 = [
                        numint._ao_f32(ao, grids, opt, 4096, blk,
                                       ao.shape[0])
                        for blk, (ao, idx, w, cc, opt) in enumerate(blocks)]
                self.assertEqual(c2.count, 0)
                for a, b in zip(twins1, twins2):
                    self.assertIs(a, b)
        finally:
            precision.set_precision('fp64')

    def test_multiblock_tail_eval_rho2_bitwise(self):
        '''Block-level numerical equivalence incl. the partial tail:
        on a multi-block cell (full 4096-point blocks + a tail block,
        screened nao_sub varying per block), _eval_rho2 fed the twin
        equals _eval_rho2 fed the fp64 ao (its internal per-call cast)
        EXACTLY.'''
        grids = fresh_grids(mol_benz)
        ni = fresh_ni(mol_benz, grids)
        blocks, _ = iter_blocks(ni, grids, mol_benz, 4096)
        self.assertGreaterEqual(len(blocks), 3)
        tail_pts = blocks[-1][2].size
        self.assertNotEqual(tail_pts, 4096)
        nao = ni.gdftopt._sorted_mol.nao
        rng = np.random.default_rng(20261026)
        nocc = 13
        try:
            optin()
            with precision.fp32():
                for blk, (ao, idx, w, cc, opt) in enumerate(blocks):
                    nao_sub = len(idx)
                    cpos = cp.asarray(
                        rng.standard_normal((nao_sub, nocc)))
                    twin = numint._ao_f32(ao, grids, opt, 4096, blk,
                                          ao.shape[0])
                    rho_twin = numint._eval_rho2(twin, cpos.copy(), 'GGA')
                    rho_ref = numint._eval_rho2(ao, cpos.copy(), 'GGA')
                    self.assertEqual(rho_twin.tobytes(), rho_ref.tobytes())
        finally:
            precision.set_precision('fp64')

    def test_veff_bitwise_mirror_vs_killswitch(self):
        '''veff-level: under an explicit fp32 context (the auto head's
        arithmetic), get_veff with the twin cache on (default) is
        bitwise identical to the kill-switch per-call cast path, on an
        NLC functional (exercises nr_rks both loops + nr_nlc_vxc).'''
        dm = scf.RHF(mol_sph).density_fit().get_init_guess()
        e = {}
        try:
            for label, env in (('optin', '1'), ('kill', '0')):
                if env is None:
                    os.environ.pop('GPU4PYSCF_AO_F32_CACHE', None)
                else:
                    os.environ['GPU4PYSCF_AO_F32_CACHE'] = env
                reset_env_cache()
                mf = dft.RKS(mol_sph, xc='wb97mv').density_fit()
                mf.grids.level = 1
                mf.nlcgrids.level = 1
                with precision.fp32():
                    e[label] = cp.asnumpy(mf.get_veff(mol_sph, dm))
        finally:
            os.environ.pop('GPU4PYSCF_AO_F32_CACHE', None)
            reset_env_cache()
            precision.set_precision('fp64')
        self.assertEqual(e['optin'].tobytes(), e['kill'].tobytes())

    def test_auto_scf_bitwise_invariance(self):
        '''End-to-end: the auto policy's converged energy is bitwise
        identical with the twin cache on (default) vs the kill-switch
        (the per-call cast path), on RKS-wb97mv (NLC, all five sites)
        and UKS-wb97mv (the UKS sites).'''
        e = {}
        for label, env in (('optin', '1'), ('kill', '0')):
            if env is None:
                os.environ.pop('GPU4PYSCF_AO_F32_CACHE', None)
            else:
                os.environ['GPU4PYSCF_AO_F32_CACHE'] = env
            reset_env_cache()
            for key, scf_fn in (('rks', dft.RKS), ('uks', dft.UKS)):
                mf = scf_fn(mol_uks if key == 'uks' else mol_sph,
                            xc='wb97mv'
                            ).density_fit(auxbasis='def2-universal-jkfit')
                mf.conv_tol = 1e-11
                mf.grids.level = 1
                mf.nlcgrids.level = 1
                mf.precision_mode = 'auto'
                e[f'{label}_{key}'] = mf.kernel()
        os.environ.pop('GPU4PYSCF_AO_F32_CACHE', None)
        reset_env_cache()
        self.assertEqual(e['optin_rks'], e['kill_rks'])
        self.assertEqual(e['optin_uks'], e['kill_uks'])

    def test_killswitch_and_caps(self):
        '''All three fail-closed guards restore the per-call cast:
        kill-switch ('0'), a per-grids MAX_BYTES smaller than the
        twin estimate, a TOTAL_BYTES budget already exhausted.'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        ao, idx, w, cc, opt = blocks[0]
        try:
            with precision.fp32():
                for env, val in (('GPU4PYSCF_AO_F32_CACHE_MAX_BYTES',
                                  '1024'),
                                 ('GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES',
                                  '1024')):
                    os.environ[env] = val
                    os.environ['GPU4PYSCF_AO_F32_CACHE'] = '1'
                    reset_env_cache()
                    with CastCounter() as c:
                        for _ in range(2):
                            numint._ao_f32(ao, grids, opt, 4096, 0,
                                           ao.shape[0])
                    self.assertGreater(c.count, 0)
                    self.assertNotIsInstance(grids._ao_f32_twins, dict)
                    del os.environ[env]
                reset_env_cache()
        finally:
            for k in ('GPU4PYSCF_AO_F32_CACHE',
                      'GPU4PYSCF_AO_F32_CACHE_MAX_BYTES',
                      'GPU4PYSCF_AO_F32_CACHE_TOTAL_BYTES'):
                os.environ.pop(k, None)
            reset_env_cache()
            precision.set_precision('fp64')

    def test_fp64_plain_never_stores(self):
        '''A plain fp64 veff (no vmat-f32 lane knob) never stores twins
        and is bitwise unaffected by twins an fp32 context built.'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        ao, idx, w, cc, opt = blocks[0]
        optin()
        numint._ao_f32(ao, grids, opt, 4096, 0, ao.shape[0])
        self.assertIsInstance(grids._ao_f32_twins, dict)
        grids._ao_f32_twins = None
        mf = dft.RKS(mol_sph, xc='wb97mv').density_fit()
        mf.grids.level = 1
        mf.nlcgrids.level = 1
        dm = scf.RHF(mol_sph).density_fit().get_init_guess()
        v0 = cp.asnumpy(mf.get_veff(mol_sph, dm))
        self.assertNotIsInstance(getattr(mf.grids, '_ao_f32_twins', None),
                                 dict)
        v1 = cp.asnumpy(mf.get_veff(mol_sph, dm))
        self.assertEqual(v0.tobytes(), v1.tobytes())

    def test_budget_counts_live_grids_only(self):
        '''The process budget bounds LIVE twins: a grids object
        discarded with its twins releases its budget share, so a later
        fresh grids on the same process admits again (the batch-1
        discovery: the naive cumulative counter rejected every sample
        after warmup+sample-1).'''
        import gc
        try:
            optin()
            with precision.fp32():
                for _ in range(3):
                    grids = fresh_grids(mol_sph)
                    ni = fresh_ni(mol_sph, grids)
                    blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
                    ao, idx, w, cc, opt = blocks[0]
                    t = numint._ao_f32(ao, grids, opt, 4096, 0,
                                       ao.shape[0])
                    self.assertIsInstance(grids._ao_f32_twins, dict)
                    del grids, ni, blocks, ao, t
                    gc.collect()
                self.assertEqual(numint._ao_f32_live_stored(), 0)
        finally:
            precision.set_precision('fp64')

    def test_invalidation_on_rebuild(self):
        '''A grids rebuild drops the twins (a twin can never serve a
        rebuilt grid); the next fp32 request rebuilds fresh.'''
        grids = fresh_grids(mol_sph)
        ni = fresh_ni(mol_sph, grids)
        blocks, _ = iter_blocks(ni, grids, mol_sph, 4096)
        ao, idx, w, cc, opt = blocks[0]
        try:
            optin()
            with precision.fp32():
                t1 = numint._ao_f32(ao, grids, opt, 4096, 0, ao.shape[0])
                self.assertIsInstance(grids._ao_f32_twins, dict)
                grids.build()
                self.assertIsNone(grids._ao_f32_twins)
                t2 = numint._ao_f32(ao, grids, opt, 4096, 0, ao.shape[0])
                self.assertIsNot(t1, t2)
        finally:
            precision.set_precision('fp64')


if __name__ == '__main__':
    unittest.main()

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
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
from gpu4pyscf.df.df import DF
from gpu4pyscf.lib import precision
from gpu4pyscf.lib import cupy_helper

def setUpModule():
    global mol_sph, mol_benz, mol_uks
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    # naux 558 > blksize 256: exercises full blocks AND the 46-column
    # tail block whose cast layout differs from the strided fast path
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
    '''Count df_jk._cast_cderi invocations (observation only).'''
    def __init__(self):
        self.count = 0
        self.orig = df_jk._cast_cderi

    def __enter__(self):
        counter = self

        def wrapped(cderi):
            counter.count += 1
            return counter.orig(cderi)

        df_jk._cast_cderi = wrapped
        return self

    def __exit__(self, *exc):
        df_jk._cast_cderi = self.orig
        return False


def reset_env_cache():
    '''Reset the cached env reads of the SP-28 knobs (module-level caches;
    tests toggle the environment and must observe the new values).'''
    import gpu4pyscf.df.df as df_mod
    df_mod._CDERI_F32_CACHE = None
    df_mod._CDERI_F32_CACHE_MAX_BYTES = None


class KnownValues(unittest.TestCase):

    def _fresh_df(self):
        mf = scf.RHF(mol_sph).density_fit(auxbasis='def2-universal-jkfit')
        mf.with_df.build()
        return mf.with_df

    def _two_dms(self, dfobj):
        nao = dfobj.nao
        rng = np.random.default_rng(20261026)
        dm1 = rng.standard_normal((nao, nao))
        dm1 = dm1 + dm1.T
        dm2 = rng.standard_normal((nao, nao))
        dm2 = dm2 + dm2.T
        return dm1, dm2

    def test_source_contract(self):
        '''SP-28 source contract: the DF class carries the fp32 mirror
        machinery (guarded builder + loop_f32 + invalidation hooks) and
        df_jk.get_jk consults it in its fp32 branch.'''
        self.assertTrue(hasattr(DF, '_get_cderi_f32_blocks'))
        self.assertTrue(hasattr(DF, 'loop_f32'))
        self.assertIn('_cderi_f32', DF.__init__.__code__.co_names)
        self.assertIn('_cderi_f32', DF.reset.__code__.co_names)

    def test_cast_elimination_and_reuse(self):
        '''The mechanism: with the mirror admitted, the SECOND and later
        fp32 get_jk calls on the same DF object run ZERO _cast_cderi
        calls (the per-call re-cast of the unchanged tensor is gone) and
        reuse the same mirror arrays (no rebuild).'''
        dfobj = self._fresh_df()
        dm1, dm2 = self._two_dms(dfobj)
        try:
            with precision.fp32():
                with CastCounter() as c1:
                    dfobj.get_jk(dm1, hermi=1)
                self.assertIsNotNone(dfobj._cderi_f32)
                blocks = next(iter(dfobj._cderi_f32.values()))
                mirror_ids = [id(a) for a in blocks]
                with CastCounter() as c2:
                    dfobj.get_jk(dm2, hermi=1)
                self.assertEqual(c2.count, 0)
                self.assertEqual(
                    [id(a) for a in next(iter(dfobj._cderi_f32.values()))],
                    mirror_ids)
                # first call builds the cache outside _cast_cderi
                self.assertEqual(c1.count, 0)
        finally:
            precision.set_precision('fp64')

    def test_bitwise_mirror_vs_killswitch(self):
        '''The design invariant: vj/vk from the mirror path are bitwise
        identical to the per-call cast path (kill-switch), for two
        different density matrices (J and K both).'''
        dfobj = self._fresh_df()
        dm1, dm2 = self._two_dms(dfobj)
        try:
            with precision.fp32():
                os.environ['GPU4PYSCF_CDERI_F32_CACHE'] = '0'
                reset_env_cache()
                vj0_1, vk0_1 = dfobj.get_jk(dm1, hermi=1)
                vj0_2, vk0_2 = dfobj.get_jk(dm2, hermi=1)
                del os.environ['GPU4PYSCF_CDERI_F32_CACHE']
                reset_env_cache()
                vj1_1, vk1_1 = dfobj.get_jk(dm1, hermi=1)
                vj1_2, vk1_2 = dfobj.get_jk(dm2, hermi=1)
        finally:
            os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            reset_env_cache()
            precision.set_precision('fp64')
        for a, b in ((vj0_1, vj1_1), (vk0_1, vk1_1),
                     (vj0_2, vj1_2), (vk0_2, vk1_2)):
            a = np.asarray(a); b = np.asarray(b)
            self.assertEqual(a.dtype, b.dtype)
            self.assertTrue(np.array_equal(a, b))
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_killswitch_and_cap_guard(self):
        '''Both fail-closed guards restore the per-call cast path: the
        kill-switch env ('0') and a mirror-bytes cap smaller than the
        tensor.'''
        dfobj = self._fresh_df()
        dm1, _ = self._two_dms(dfobj)
        try:
            with precision.fp32():
                os.environ['GPU4PYSCF_CDERI_F32_CACHE'] = '0'
                reset_env_cache()
                with CastCounter() as c:
                    dfobj.get_jk(dm1, hermi=1)
                    dfobj.get_jk(dm1, hermi=1)
                self.assertGreater(c.count, 0)
                self.assertIsNone(dfobj._cderi_f32)
                del os.environ['GPU4PYSCF_CDERI_F32_CACHE']

                os.environ['GPU4PYSCF_CDERI_F32_CACHE_MAX_BYTES'] = '1024'
                reset_env_cache()
                with CastCounter() as c2:
                    dfobj.get_jk(dm1, hermi=1)
                    dfobj.get_jk(dm1, hermi=1)
                self.assertGreater(c2.count, 0)
                self.assertIsNone(dfobj._cderi_f32)
        finally:
            os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE_MAX_BYTES', None)
            reset_env_cache()
            precision.set_precision('fp64')

    def test_fp64_lane_never_builds_mirror(self):
        '''The fp64 context never builds the mirror and is bitwise
        unaffected (same dfobj, fp64 calls before and after a mirror
        was built by an fp32 context produce identical vj/vk).'''
        dfobj = self._fresh_df()
        dm1, _ = self._two_dms(dfobj)
        vj_ref, vk_ref = dfobj.get_jk(dm1, hermi=1)
        self.assertIsNone(dfobj._cderi_f32)
        try:
            with precision.fp32():
                dfobj.get_jk(dm1, hermi=1)
            self.assertIsNotNone(dfobj._cderi_f32)
        finally:
            precision.set_precision('fp64')
        vj2, vk2 = dfobj.get_jk(dm1, hermi=1)
        self.assertEqual(np.asarray(vj_ref).tobytes(),
                         np.asarray(vj2).tobytes())
        self.assertEqual(np.asarray(vk_ref).tobytes(),
                         np.asarray(vk2).tobytes())

    def test_invalidation_on_rebuild(self):
        '''A cderi rebuild drops the mirror (the mirror can never serve
        a rebuilt tensor); the next fp32 call rebuilds it fresh.'''
        dfobj = self._fresh_df()
        dm1, _ = self._two_dms(dfobj)
        try:
            with precision.fp32():
                dfobj.get_jk(dm1, hermi=1)
                self.assertIsNotNone(dfobj._cderi_f32)
                old_id = id(next(iter(dfobj._cderi_f32.values()))[0])
                dfobj.reset()
                self.assertIsNone(getattr(dfobj, '_cderi_f32', None))
                dfobj.build()
                dfobj.get_jk(dm1, hermi=1)
                self.assertIsNotNone(dfobj._cderi_f32)
                self.assertNotEqual(
                    id(next(iter(dfobj._cderi_f32.values()))[0]), old_id)
        finally:
            precision.set_precision('fp64')

    def test_fill_symmetric_f32_bitwise(self):
        '''K-side primitive: fill_symmetric from an fp32 tril equals the
        fp64 fill + cast bitwise (pure scatter), in the production
        column-major layout (cderi.T of a C-contiguous (naux,npairs)
        tensor, df.py loop()) and the row-major (npairs,naux) layout.'''
        import cupy as cp
        nao = 24
        npairs = nao * (nao + 1) // 2
        naux = 6
        rng = np.random.default_rng(20261026)
        tril = cp.asarray(rng.standard_normal((naux, npairs)))
        pair_idx = cp.asarray(
            np.concatenate([np.arange(i, nao) + i * nao
                            for i in range(nao)]).astype(np.int32))
        p0, p1 = 1, 5
        for a_dev in (tril.T, cp.ascontiguousarray(tril.T)):
            self.assertEqual(a_dev.shape, (npairs, naux))
            out64 = cp.zeros((nao, nao, naux))
            cupy_helper.fill_symmetric(a_dev, pair_idx, nao, p0, p1,
                                       out=out64)
            cast_then_fill = out64.astype(cp.float32)
            out32 = cp.zeros((nao, nao, naux), dtype=cp.float32)
            cupy_helper.fill_symmetric(
                a_dev.astype(cp.float32), pair_idx, nao, p0, p1,
                out=out32)
            self.assertTrue(cp.array_equal(cast_then_fill, out32))

    def test_bitwise_multiblock_tail(self):
        '''The tail-block layout parity: on a cell with naux 558 > blksize
        256 (full strided blocks + a 46-column tail), the mirror path is
        bitwise identical to the per-call cast path for J and K.'''
        mf = scf.RHF(mol_benz).density_fit(auxbasis='def2-universal-jkfit')
        dfobj = mf.with_df
        dfobj.build()
        self.assertGreater(dfobj.naux, 256)
        nao = dfobj.nao
        rng = np.random.default_rng(20261026)
        dm1 = rng.standard_normal((nao, nao))
        dm1 = dm1 + dm1.T
        try:
            with precision.fp32():
                os.environ['GPU4PYSCF_CDERI_F32_CACHE'] = '0'
                reset_env_cache()
                vj0, vk0 = dfobj.get_jk(dm1, hermi=1)
                del os.environ['GPU4PYSCF_CDERI_F32_CACHE']
                reset_env_cache()
                vj1, vk1 = dfobj.get_jk(dm1, hermi=1)
                self.assertIsNotNone(dfobj._cderi_f32)
        finally:
            os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            reset_env_cache()
            precision.set_precision('fp64')
        self.assertEqual(np.asarray(vj0).tobytes(),
                         np.asarray(vj1).tobytes())
        self.assertEqual(np.asarray(vk0).tobytes(),
                         np.asarray(vk1).tobytes())

    def test_auto_scf_bitwise_invariance(self):
        '''End-to-end: the auto policy's converged energy is bitwise
        identical with the mirror on (default) vs the kill-switch (the
        per-call cast path), on RHF and UKS.'''
        e = {}
        for label, env in (('mirror', None), ('kill', '0')):
            if env is None:
                os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            else:
                os.environ['GPU4PYSCF_CDERI_F32_CACHE'] = env
            reset_env_cache()
            for key, scf_fn in (('rhf', scf.RHF), ('uks', dft.UKS)):
                mf = scf_fn(mol_uks if key == 'uks' else mol_sph
                            ).density_fit(auxbasis='def2-universal-jkfit')
                mf.conv_tol = 1e-11
                mf.precision_mode = 'auto'
                e[f'{label}_{key}'] = mf.kernel()
        os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
        reset_env_cache()
        self.assertEqual(e['mirror_rhf'], e['kill_rhf'])
        self.assertEqual(e['mirror_uks'], e['kill_uks'])


if __name__ == '__main__':
    unittest.main()

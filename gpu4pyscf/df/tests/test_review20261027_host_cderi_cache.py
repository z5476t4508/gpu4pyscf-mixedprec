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
from gpu4pyscf.df.df import DF
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
    '''Reset the cached env reads of the SP-28 knobs (module-level
    caches; tests toggle the environment and must observe the new
    values).'''
    import gpu4pyscf.df.df as df_mod
    df_mod._CDERI_F32_CACHE = None
    df_mod._CDERI_F32_CACHE_MAX_BYTES = None


class KnownValues(unittest.TestCase):
    '''Review 2026-10-27 cache-boundaries F1 (P1): a host-resident
    CDERI (use_gpu_memory=False, or automatic host placement) must be
    REJECTED from the SP-28 fp32 mirror cache.  The cached astype
    product of a NumPy tensor stays on the host, but get_jk would then
    bypass DF.loop()'s double-buffered host upload path and feed
    fill_symmetric/contract host blocks where device pointers are
    required (AttributeError on .data.ptr), breaking the previously
    working host fallback.'''

    def _fresh_host_df(self):
        dfobj = DF(mol_sph, auxbasis='def2-universal-jkfit')
        dfobj.use_gpu_memory = False
        dfobj.build()
        c0 = dfobj._cderi[0]
        self.assertIsInstance(c0, np.ndarray)
        self.assertNotIsInstance(c0, cp.ndarray)
        return dfobj

    def _two_dms(self, nao):
        rng = np.random.default_rng(20261027)
        dm1 = rng.standard_normal((nao, nao))
        dm1 = dm1 + dm1.T
        dm2 = rng.standard_normal((nao, nao))
        dm2 = dm2 + dm2.T
        return dm1, dm2

    def test_host_cderi_rejected_from_f32_cache(self):
        '''Unit contract: _get_cderi_f32_blocks returns None (fall back
        to the original loop() host path) when the device CDERI is not
        a CuPy array.'''
        dfobj = self._fresh_host_df()
        try:
            with precision.fp32():
                blocks = dfobj._get_cderi_f32_blocks(256)
            self.assertIsNone(blocks)
            self.assertIsNone(dfobj._cderi_f32)
        finally:
            precision.set_precision('fp64')

    def test_host_cderi_get_jk_bitwise_cache_on_off(self):
        '''fp32 get_jk on a host-resident DF: the cache-enabled run must
        complete (served by the original loop() host upload path) and be
        bitwise identical to the kill-switch run (the per-call cast
        path), for J and K and two density matrices.'''
        dfobj = self._fresh_host_df()
        dm1, dm2 = self._two_dms(dfobj.nao)
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
                self.assertIsNone(dfobj._cderi_f32)
        finally:
            os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            reset_env_cache()
            precision.set_precision('fp64')
        for a, b in ((vj0_1, vj1_1), (vk0_1, vk1_1),
                     (vj0_2, vj1_2), (vk0_2, vk1_2)):
            a = np.asarray(a); b = np.asarray(b)
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(a.tobytes(), b.tobytes())

    def test_host_cderi_fp32_scf_bitwise_cache_on_off(self):
        '''Host-resident SCF regression: a full-fp32 RHF run whose DF
        keeps CDERI on the host must complete with the cache enabled
        and give the bitwise-identical converged energy to the
        kill-switch run.'''
        e = {}
        for label, env in (('kill', '0'), ('on', None)):
            if env is None:
                os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
            else:
                os.environ['GPU4PYSCF_CDERI_F32_CACHE'] = env
            reset_env_cache()
            mf = scf.RHF(mol_sph).density_fit(
                auxbasis='def2-universal-jkfit')
            mf.with_df.use_gpu_memory = False
            mf.conv_tol = 1e-9
            # the kernel sets the precision itself from precision_mode
            # (scf.hf._kernel saves/restores the ambient value), so the
            # fp32 J/K lane is engaged by the mode, not the context
            mf.precision_mode = 'fp32'
            try:
                e[label] = mf.kernel()
            finally:
                precision.set_precision('fp64')
            self.assertIsInstance(mf.with_df._cderi[0], np.ndarray)
            self.assertNotIsInstance(mf.with_df._cderi[0], cp.ndarray)
            self.assertIsNone(mf.with_df._cderi_f32)
        os.environ.pop('GPU4PYSCF_CDERI_F32_CACHE', None)
        reset_env_cache()
        self.assertEqual(e['kill'], e['on'])


if __name__ == '__main__':
    unittest.main()

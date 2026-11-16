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
'''DFJK-PHASE-01 Phase B focused regression (2026-11-05).

The opt-in contraction plan/descriptor reuse cache
(GPU4PYSCF_CONTRACT_PLAN_CACHE, default off) must execute the SAME plan
object as the per-call build and produce BITWISE-equal results for every
(shape, stride, dtype) operand family it serves, including the four df_jk
get_jk patterns, alpha/beta variants (beta=1 accumulate), stride-variant
and tail-shaped operands, repeated calls and a dirty memory pool.
Prereg preregistration-dfjk-phase01-v1.json phase_b_gate; mechanism
frozen in phasea-source-audit.md C1.
'''
import os
import unittest
import numpy as np
import cupy as cp
from gpu4pyscf.lib import cutensor

KNOB = 'GPU4PYSCF_CONTRACT_PLAN_CACHE'


def _bitwise_equal(x, y):
    # byte-image comparison; assertEqual on byte blobs hangs difflib
    return (x.shape == y.shape and x.dtype == y.dtype
            and x.tobytes() == y.tobytes())


class PlanCacheTest(unittest.TestCase):

    def setUp(self):
        self._saved = os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()

    def tearDown(self):
        if self._saved is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = self._saved
        cutensor._plan_cache.clear()

    def _cases(self):
        '''(a, b, pattern, out, alpha, beta) families covering the df_jk
        get_jk contracts plus stride/layout variants and tail shapes.
        '''
        rng = np.random.default_rng(20261105)
        nao, nocc, naux, npairs = 24, 9, 40, 300
        cderi = cp.asarray(rng.standard_normal((naux, nao, nao)))
        cderi_tril = cp.asarray(rng.standard_normal((naux, npairs)))
        dm_sparse = cp.asarray(rng.standard_normal((1, npairs)))
        factor_l = cp.asarray(rng.standard_normal((1, nao, nocc)))

        out_j1 = cp.empty((1, naux))
        out_j2 = cp.empty((1, npairs))
        out_k1 = cp.empty((1, nao, nocc, naux))
        out_k2 = cp.empty((1, nao, nao))   # 'nij' with i=j=nao (vk)
        cases = [
            # jk J pair
            ('np,Lp->nL', dm_sparse, cderi_tril, None, 1.0, 0.0),
            ('nL,Lp->np', cp.asarray(rng.standard_normal((1, naux))),
             cderi_tril, out_j2.copy(), 1.0, 1.0),
            # jk K pair (mode 0); the second operand family is the
            # rhok x rhok self-contraction (both (n,nao,nocc,L))
            ('Lij,njk->nikL', cderi, factor_l, out_k1, 1.0, 0.0),
            ('nikL,njkL->nij', out_k1.copy(), out_k1.copy(),
             out_k2.copy(), 1.0, 1.0),
            # stride variants: F-contiguous operand, transposed view
            ('Lij,njk->nikL', cp.asfortranarray(cderi), factor_l,
             None, 1.0, 0.0),
            ('Lij,njk->nikL', cderi.transpose(0, 2, 1), factor_l,
             None, 1.0, 0.0),
            # tail-shaped (smaller L) and fp32
            ('Lij,njk->nikL', cderi[:17], factor_l, None, 1.0, 0.0),
            ('np,Lp->nL', dm_sparse.astype(cp.float32),
             cderi_tril.astype(cp.float32), None, 1.0, 0.0),
            # alpha != 1
            ('np,Lp->nL', dm_sparse, cderi_tril, out_j1.copy(),
             -1.5, 0.0),
        ]
        return cases

    def test_knob_unset_cache_inert(self):
        for pattern, a, b, out, alpha, beta in self._cases():
            cutensor.contract(pattern, a, b, alpha=alpha, beta=beta,
                              out=out)
        self.assertEqual(len(cutensor._plan_cache), 0)
        self.assertFalse(cutensor._plan_cache_enabled())

    def test_cache_on_bitwise_all_families(self):
        cases = self._cases()
        # snapshot the pristine out templates: the reference pass
        # consumes them (beta accumulation), so every later pass must
        # start from the same bytes
        templates = [c[3].copy() if c[3] is not None else None
                     for c in cases]
        os.environ[KNOB] = '1'
        self.assertTrue(cutensor._plan_cache_enabled())
        refs = []
        for i, (pattern, a, b, _, alpha, beta) in enumerate(cases):
            cutensor._plan_cache.clear()
            out = templates[i].copy() if templates[i] is not None else None
            refs.append(cutensor.contract(pattern, a, b, alpha=alpha,
                                          beta=beta, out=out))
        # cache is now warm for every family; two full hit passes must
        # reproduce the reference bytes exactly
        for rep in range(2):
            for i, (pattern, a, b, _, alpha, beta) in enumerate(cases):
                out = (templates[i].copy() if templates[i] is not None
                       else None)
                got = cutensor.contract(pattern, a, b, alpha=alpha,
                                        beta=beta, out=out)
                self.assertTrue(_bitwise_equal(refs[i], got),
                                f'rep={rep} pattern={pattern} '
                                f'a.strides={a.strides} not bitwise')
        self.assertGreater(len(cutensor._plan_cache), 0)
        # knob-off reference pass in the same process: the uncached
        # per-call build must agree with the cached results bitwise too
        os.environ.pop(KNOB, None)
        for i, (pattern, a, b, _, alpha, beta) in enumerate(cases):
            out = templates[i].copy() if templates[i] is not None else None
            got = cutensor.contract(pattern, a, b, alpha=alpha,
                                    beta=beta, out=out)
            self.assertTrue(_bitwise_equal(refs[i], got),
                            f'uncached pattern={pattern} not bitwise')

    def test_cache_hits_reuse_entries(self):
        os.environ[KNOB] = '1'
        pattern, a, b = 'np,Lp->nL', *self._cases()[0][1:3]
        cutensor.contract(pattern, a, b)
        n_after_first = len(cutensor._plan_cache)
        self.assertEqual(n_after_first, 1)
        for _ in range(5):
            cutensor.contract(pattern, a, b)
        self.assertEqual(len(cutensor._plan_cache), n_after_first)

    def test_dirty_pool_between_calls(self):
        os.environ[KNOB] = '1'
        pattern, a, b, _, alpha, beta = self._cases()[0]
        ref = cutensor.contract(pattern, a, b, alpha=alpha, beta=beta)
        for _ in range(4):
            junk = [cp.random.random(2 ** 20) for _ in range(3)]
            del junk
            got = cutensor.contract(pattern, a, b, alpha=alpha,
                                    beta=beta)
            self.assertTrue(_bitwise_equal(ref, got))

    def test_cap_fail_closed(self):
        os.environ[KNOB] = '1'
        cases = self._cases()
        saved_cap = cutensor._PLAN_CACHE_MAX_ENTRIES
        cutensor._PLAN_CACHE_MAX_ENTRIES = 0
        try:
            for pattern, a, b, out, alpha, beta in cases:
                cutensor.contract(pattern, a, b, alpha=alpha, beta=beta,
                                  out=out)
            self.assertEqual(len(cutensor._plan_cache), 0)
        finally:
            cutensor._PLAN_CACHE_MAX_ENTRIES = saved_cap
        # uncapped still works after restoring the cap
        pattern, a, b, _, alpha, beta = cases[0]
        cutensor.contract(pattern, a, b, alpha=alpha, beta=beta)
        self.assertEqual(len(cutensor._plan_cache), 1)

    def test_knob_off_values(self):
        for v in ('0', 'false', 'False'):
            os.environ[KNOB] = v
            self.assertFalse(cutensor._plan_cache_enabled())
        os.environ[KNOB] = '1'
        self.assertTrue(cutensor._plan_cache_enabled())

    def test_df_get_jk_bitwise(self):
        '''Block-identity check at the mechanism site: a small hybrid
        DF get_jk (J+K, hermi=1) must return bitwise-identical vj/vk
        with the cache on vs off, over repeated calls (warm cache).
        '''
        import pyscf
        from gpu4pyscf.df import df as df_mod
        mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
        dfobj = df_mod.DF(mol, auxbasis='def2-universal-jkfit')
        dm = cp.asarray(np.linalg.inv(mol.intor('int1e_ovlp'))
                        [:mol.nao, :mol.nao])
        dm = .5*(dm + dm.T)
        os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = '1'
        for rep in range(3):
            vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True,
                                    with_k=True)
            self.assertTrue(_bitwise_equal(vj0, vj1), f'vj rep={rep}')
            self.assertTrue(_bitwise_equal(vk0, vk1), f'vk rep={rep}')
        self.assertGreater(len(cutensor._plan_cache), 0)


if __name__ == '__main__':
    unittest.main()

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
'''WARMUP-PRELOAD-01 Phase B focused regression (2026-11-05).

The opt-in eager first-use plan-cache population
(GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP, default off; implies the contraction
plan cache while enabled) must warm the exact keys the first df_jk
get_jk block loop will produce -- no contraction execution, no device
compute paid twice -- so that every first-call contraction is served
from a prebuilt plan entry.  Results must stay bitwise; the knob-off
path must stay byte-identical (inert machinery).  Prereg
preregistration-warmup-preload01-v1.json instrument.focused_regression;
mechanism frozen in phasea-source-audit.md M1.
'''
import json
import os
import subprocess
import sys
import unittest
import numpy as np
import cupy as cp
from gpu4pyscf.lib import cutensor
from gpu4pyscf.lib.cupy_helper import tag_array

KNOB = 'GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP'
CACHE_KNOB = 'GPU4PYSCF_CONTRACT_PLAN_CACHE'
PATTERNS = ('np,Lp->nL', 'nL,Lp->np', 'Lij,njk->nikL', 'nikL,njkL->nij')


def _bitwise_equal(x, y):
    # byte-image comparison; assertEqual on byte blobs hangs difflib
    return (x.shape == y.shape and x.dtype == y.dtype
            and x.tobytes() == y.tobytes())


def _small_df():
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
    dm = .5 * (dm + dm.T)
    # tag with a rank factorization the way the SCF path does
    # (factor_r=None -> dm_factor_mode 0, the warmup's v1 scope)
    w, v = cp.linalg.eigh(dm)
    mask = abs(w) > 1e-8
    factor_l = v[:, mask] * cp.sqrt(w[mask])[:, None]
    dm = tag_array(dm.copy(), factor_l=factor_l, factor_r=None)
    return dfobj, dm


class EagerPlanWarmupTest(unittest.TestCase):

    def setUp(self):
        self._saved = {k: os.environ.pop(k, None)
                       for k in (KNOB, CACHE_KNOB)}
        cutensor._plan_cache.clear()
        cutensor._reset_warmup_stats()

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        cutensor._plan_cache.clear()
        cutensor._reset_warmup_stats()

    def test_knob_semantics(self):
        # off: no warmup, no implied cache
        self.assertFalse(cutensor._eager_warmup_enabled())
        self.assertFalse(cutensor._plan_cache_enabled())
        # on: warmup armed and the plan cache implied active
        os.environ[KNOB] = '1'
        self.assertTrue(cutensor._eager_warmup_enabled())
        self.assertTrue(cutensor._plan_cache_enabled())
        for v in ('0', 'false', 'False'):
            os.environ[KNOB] = v
            self.assertFalse(cutensor._eager_warmup_enabled())
        # the standalone cache knob is untouched by the new knob being off
        os.environ.pop(KNOB, None)
        os.environ[CACHE_KNOB] = '1'
        self.assertTrue(cutensor._plan_cache_enabled())
        self.assertFalse(cutensor._eager_warmup_enabled())

    def test_knob_off_machinery_inert(self):
        dfobj, dm = _small_df()
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(len(cutensor._plan_cache), 0)
        self.assertEqual(cutensor._warmup_stats['built'], 0)
        self.assertEqual(cutensor._warmup_stats['host_ms'], 0.0)

    def test_knob_on_engagement_and_bitwise(self):
        dfobj, dm = _small_df()
        os.environ.pop(KNOB, None)
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = '1'
        cutensor._plan_cache.clear()
        vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        # engagement: all four jk contraction families warmed, on host
        self.assertGreater(cutensor._warmup_stats['built'], 0)
        self.assertGreater(cutensor._warmup_stats['host_ms'], 0.0)
        for pat in PATTERNS:
            self.assertGreaterEqual(
                cutensor._warmup_stats['patterns'].get(pat, 0), 1,
                f'pattern {pat} not warmed')
        # every first-call contraction hit a warmed entry: the
        # contraction loop added no new cache keys
        n_after = len(cutensor._plan_cache)
        # bitwise identity knob-on vs knob-off
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj not bitwise')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk not bitwise')
        # latch: a second call does not re-warm and stays bitwise (the
        # latched pass-through may record a sub-millisecond elapsed)
        built_before = cutensor._warmup_stats['built']
        ms_before = cutensor._warmup_stats['host_ms']
        vj2, vk2 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(cutensor._warmup_stats['built'], built_before)
        self.assertLess(cutensor._warmup_stats['host_ms'] - ms_before, 1.0)
        self.assertEqual(len(cutensor._plan_cache), n_after)
        self.assertTrue(_bitwise_equal(vj0, vj2), 'vj rep not bitwise')
        self.assertTrue(_bitwise_equal(vk0, vk2), 'vk rep not bitwise')

    def test_warmed_keys_exact(self):
        '''The warm step must insert entries under the keys the real
        contractions produce: after a knob-on call, replaying the same
        contracts through contraction() must add ZERO new cache keys.
        '''
        dfobj, dm = _small_df()
        os.environ[KNOB] = '1'
        vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        n_after_warm = len(cutensor._plan_cache)
        vj2, vk2 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(len(cutensor._plan_cache), n_after_warm)
        self.assertTrue(_bitwise_equal(vj1, vj2))

    def test_new_shapes_fail_open(self):
        dfobj, dm = _small_df()
        os.environ[KNOB] = '1'
        dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        built_before = cutensor._warmup_stats['built']
        # a stacked-DM call has shapes the latch never warmed: it must
        # fail open to the per-call build path and stay bitwise vs the
        # knob-off reference for the same input
        dm2 = cp.asarray(np.stack([dm.get(), dm.get()]))
        os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()
        vj0, vk0 = dfobj.get_jk(dm2, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = '1'
        cutensor._plan_cache.clear()
        vj1, vk1 = dfobj.get_jk(dm2, hermi=1, with_j=True, with_k=True)
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj new shapes')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk new shapes')
        # the latch held: no re-warm on the second dfobj call
        self.assertEqual(cutensor._warmup_stats['built'], built_before)

    def test_fp32_knob_inert(self):
        from gpu4pyscf.lib import precision
        old = precision.get_precision()
        dfobj, dm = _small_df()
        try:
            precision.set_precision('fp32')
            os.environ.pop(KNOB, None)
            cutensor._plan_cache.clear()
            vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
            os.environ[KNOB] = '1'
            cutensor._plan_cache.clear()
            cutensor._reset_warmup_stats()
            vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
            self.assertEqual(cutensor._warmup_stats['built'], 0,
                             'knob must stay inert under fp32')
            self.assertTrue(_bitwise_equal(vj0, vj1), 'vj fp32')
            self.assertTrue(_bitwise_equal(vk0, vk1), 'vk fp32')
        finally:
            precision.set_precision(old)

    def test_warmup_fails_closed(self):
        dfobj, dm = _small_df()
        os.environ[KNOB] = '1'
        # break ONLY the warmup entry point; the real contraction path
        # (per-call _build_plan_entry) must stay fully functional
        orig = cutensor.warm_contraction

        def boom(*args, **kwargs):
            raise RuntimeError('forced warmup failure')

        cutensor.warm_contraction = boom
        try:
            vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        finally:
            cutensor.warm_contraction = orig
        self.assertGreaterEqual(cutensor._warmup_stats['failures'], 1)
        os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj fail-closed')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk fail-closed')

    def test_small_b3lyp_scf_subprocess_pairs(self):
        '''End-to-end knob-on vs knob-off small b3lyp SCF in fresh
        subprocesses per setting: |dE| <= 1e-10 (cross-process ULP
        precedent), identical cycles, converged both.
        '''
        code = '''
import os, json, sys
os.environ[{knob!r}] = sys.argv[1]
import numpy as np, cupy as cp
import pyscf
from gpu4pyscf.df import df as df_mod
from gpu4pyscf.dft import rks
mol = pyscf.M(atom=
"O 0 0 0.1174; H -0.757 0 -0.4696; H 0.757 0 -0.4696",
              basis="def2-svp", output="/dev/null", verbose=0)
mf = rks.RKS(mol).density_fit(auxbasis="def2-universal-jkfit")
mf.xc = "b3lyp"
mf.conv_tol = 1e-10
e = mf.kernel()
cp.cuda.get_current_stream().synchronize()
from gpu4pyscf.lib import cutensor
print("SCFJSON " + json.dumps(dict(
    e=float(e), cycles=int(getattr(mf, "cycles", 0) or 0),
    converged=bool(mf.converged),
    built=int(cutensor._warmup_stats["built"]),
    host_ms=float(cutensor._warmup_stats["host_ms"]))))
'''
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        results = {}
        for setting in ('0', '1'):
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith('GPU4PYSCF_') and k != 'PYTHONPATH'}
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env['PYTHONUNBUFFERED'] = '1'
            env['PYTHONPATH'] = repo_root
            res = subprocess.run(
                [sys.executable, '-c', code.format(knob=KNOB), setting],
                capture_output=True, text=True, env=env, timeout=900)
            self.assertEqual(res.returncode, 0,
                             f'setting={setting} stderr={res.stderr[-2000:]}')
            payload = None
            for line in res.stdout.splitlines():
                if line.startswith('SCFJSON '):
                    payload = json.loads(line[8:])
            self.assertIsNotNone(payload, f'setting={setting} no SCFJSON')
            results[setting] = payload
        off, on = results['0'], results['1']
        self.assertTrue(off['converged'] and on['converged'])
        self.assertEqual(off['cycles'], on['cycles'])
        self.assertLessEqual(abs(off['e'] - on['e']), 1e-10)
        self.assertGreater(on['built'], 0)
        self.assertGreater(on['host_ms'], 0.0)
        self.assertEqual(off['built'], 0)


if __name__ == '__main__':
    unittest.main()

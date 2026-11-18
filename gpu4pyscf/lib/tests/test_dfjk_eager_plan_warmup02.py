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
'''WARMUP-PRELOAD-02 Phase B focused regression (2026-11-05).

The jbuild-SCOPED eager plan warmup
(GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP=jbuild, default off) must warm ONLY
the 'np,Lp->nL' j_build keys at the real first-call shapes -- the
entry-332 per-family cold-cost table's single expensive family
(j_build 629/653 ms vs 1-4 ms for the other three) -- so the warm
step stops over-paying for near-free families while every first-call
j_build contraction is served from a prebuilt entry.  Results stay
bitwise; the knob-off path stays byte-identical; the '1' value keeps
the WARMUP-PRELOAD-01 four-family behavior verbatim (reproduction
artifact).  Prereg preregistration-warmup-preload02-v1.json
instrument.focused_regression; distinction in phasea-distinction-note.md.
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
JBUILD = 'np,Lp->nL'
OTHER_PATTERNS = ('nL,Lp->np', 'Lij,njk->nikL', 'nikL,njkL->nij')
ALL_PATTERNS = (JBUILD,) + OTHER_PATTERNS


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


def _cache_keys_for(pattern):
    # plan-cache keys are tuples whose first member is the pattern
    return sum(1 for k in cutensor._plan_cache if k[0] == pattern)


class ScopedEagerWarmupTest(unittest.TestCase):

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

    def test_scope_semantics(self):
        # off: no warmup, no implied cache
        self.assertEqual(cutensor._eager_warmup_scope(), 'off')
        self.assertFalse(cutensor._eager_warmup_enabled())
        self.assertFalse(cutensor._plan_cache_enabled())
        # scoped value: enabled with the jbuild scope and implied cache
        os.environ[KNOB] = 'jbuild'
        self.assertEqual(cutensor._eager_warmup_scope(), 'jbuild')
        self.assertTrue(cutensor._eager_warmup_enabled())
        self.assertTrue(cutensor._plan_cache_enabled())
        # '1' keeps the four-family WARMUP-PRELOAD-01 behavior
        os.environ[KNOB] = '1'
        self.assertEqual(cutensor._eager_warmup_scope(), 'all')
        self.assertTrue(cutensor._eager_warmup_enabled())
        for v in ('0', 'false', 'False'):
            os.environ[KNOB] = v
            self.assertEqual(cutensor._eager_warmup_scope(), 'off')
            self.assertFalse(cutensor._eager_warmup_enabled())
        # the standalone cache knob is untouched by the warmup knob off
        os.environ.pop(KNOB, None)
        os.environ[CACHE_KNOB] = '1'
        self.assertTrue(cutensor._plan_cache_enabled())
        self.assertFalse(cutensor._eager_warmup_enabled())
        self.assertEqual(cutensor._eager_warmup_scope(), 'off')

    def test_knob_off_machinery_inert(self):
        dfobj, dm = _small_df()
        dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(len(cutensor._plan_cache), 0)
        self.assertEqual(cutensor._warmup_stats['built'], 0)
        self.assertEqual(cutensor._warmup_stats['host_ms'], 0.0)

    def test_jbuild_scope_warms_only_j_build(self):
        dfobj, dm = _small_df()
        os.environ.pop(KNOB, None)
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = 'jbuild'
        cutensor._plan_cache.clear()
        vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        # scoped engagement: ONLY the j_build pattern is warmed
        self.assertGreaterEqual(
            cutensor._warmup_stats['patterns'].get(JBUILD, 0), 1)
        for pat in OTHER_PATTERNS:
            self.assertEqual(
                cutensor._warmup_stats['patterns'].get(pat, 0), 0,
                f'scope leaked into {pat}')
        self.assertGreater(cutensor._warmup_stats['built'], 0)
        self.assertGreater(cutensor._warmup_stats['host_ms'], 0.0)
        # every first-call j_build contraction hit a warmed entry: the
        # cache holds exactly the warmed j_build keys (a contraction
        # miss would have inserted an additional np,Lp->nL key)
        self.assertEqual(_cache_keys_for(JBUILD),
                         cutensor._warmup_stats['patterns'][JBUILD])
        # bitwise identity knob-on vs knob-off
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj not bitwise')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk not bitwise')
        # latch: a second call does not re-warm, adds no new keys of
        # any pattern, and stays bitwise
        built_before = cutensor._warmup_stats['built']
        ms_before = cutensor._warmup_stats['host_ms']
        keys_before = len(cutensor._plan_cache)
        vj2, vk2 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertEqual(cutensor._warmup_stats['built'], built_before)
        self.assertLess(cutensor._warmup_stats['host_ms'] - ms_before, 1.0)
        self.assertEqual(len(cutensor._plan_cache), keys_before)
        self.assertTrue(_bitwise_equal(vj0, vj2), 'vj rep not bitwise')
        self.assertTrue(_bitwise_equal(vk0, vk2), 'vk rep not bitwise')

    def test_v1_value_reproduction(self):
        '''The '1' value must keep the closed four-family behavior of
        7563a07 verbatim (all four patterns warmed) and stay bitwise.
        '''
        dfobj, dm = _small_df()
        os.environ.pop(KNOB, None)
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = '1'
        cutensor._plan_cache.clear()
        vj1, vk1 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        for pat in ALL_PATTERNS:
            self.assertGreaterEqual(
                cutensor._warmup_stats['patterns'].get(pat, 0), 1,
                f'v1 pattern {pat} not warmed')
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj v1 not bitwise')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk v1 not bitwise')

    def test_new_shapes_fail_open(self):
        dfobj, dm = _small_df()
        os.environ[KNOB] = 'jbuild'
        dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        built_before = cutensor._warmup_stats['built']
        # a stacked-DM call has shapes the latch never warmed: it must
        # fail open to the per-call build path and stay bitwise vs the
        # knob-off reference for the same input
        dm2 = cp.asarray(np.stack([dm.get(), dm.get()]))
        os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()
        vj0, vk0 = dfobj.get_jk(dm2, hermi=1, with_j=True, with_k=True)
        os.environ[KNOB] = 'jbuild'
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
            os.environ[KNOB] = 'jbuild'
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
        os.environ[KNOB] = 'jbuild'
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
        self.assertEqual(cutensor._warmup_stats['patterns'].get(JBUILD, 0), 0)
        os.environ.pop(KNOB, None)
        cutensor._plan_cache.clear()
        vj0, vk0 = dfobj.get_jk(dm, hermi=1, with_j=True, with_k=True)
        self.assertTrue(_bitwise_equal(vj0, vj1), 'vj fail-closed')
        self.assertTrue(_bitwise_equal(vk0, vk1), 'vk fail-closed')

    def test_small_b3lyp_scf_subprocess_pairs(self):
        '''End-to-end knob-on(jbuild) vs knob-off small b3lyp SCF in
        fresh subprocesses per setting: |dE| <= 1e-10 (cross-process
        ULP precedent), identical cycles, converged both, and the
        scoped arm's warm stats restricted to np,Lp->nL.
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
    host_ms=float(cutensor._warmup_stats["host_ms"]),
    patterns=dict(cutensor._warmup_stats["patterns"]))))
'''
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))))
        results = {}
        for setting in ('0', 'jbuild'):
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
        off, on = results['0'], results['jbuild']
        self.assertTrue(off['converged'] and on['converged'])
        self.assertEqual(off['cycles'], on['cycles'])
        self.assertLessEqual(abs(off['e'] - on['e']), 1e-10)
        self.assertGreater(on['built'], 0)
        self.assertGreater(on['host_ms'], 0.0)
        self.assertEqual(off['built'], 0)
        self.assertGreaterEqual(on['patterns'].get('np,Lp->nL', 0), 1)
        for pat in ('nL,Lp->np', 'Lij,njk->nikL', 'nikL,njkL->nij'):
            self.assertEqual(on['patterns'].get(pat, 0), 0,
                             f'scope leaked into {pat} in subprocess')


if __name__ == '__main__':
    unittest.main()

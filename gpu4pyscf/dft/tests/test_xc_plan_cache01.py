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
'''XC-VMAT-STAGEHOST-01 focused regression (2026-11-24).

The opt-in contraction-plan reuse inside the SCF XC tasks
(GPU4PYSCF_SCF_XC_PLAN_CACHE, default off) runs _nr_rks_task/_nr_uks_task
in lib.cutensor.scoped_plan_cache().  It must be inert when unset, keep
the process-wide cache contract untouched, and give BITWISE-equal
nelec/excsum/vmat to the per-call build on both the plain fp64 path and
a contr_f32 (vmat-f32 lane) cycle; the scoped cache key carries the
operand-pointer alignment classes.
'''
import os
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.lib import cutensor

KNOB = 'GPU4PYSCF_SCF_XC_PLAN_CACHE'


def setUpModule():
    global mol_sph, mol_benz, mol_open
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
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
    mol_open = pyscf.M(atom='O 0 0 0; O 0 0 1.3', basis='def2-svp', spin=2,
                       output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_benz, mol_open
    for mol in (mol_sph, mol_benz, mol_open):
        mol.stdout.close()
    mol_sph = mol_benz = mol_open = None


def _bitwise(x, y):
    # byte-image comparison; assertEqual on byte blobs hangs difflib
    return (x.shape == y.shape and x.dtype == y.dtype
            and x.tobytes() == y.tobytes())


def _reset():
    numint._XC_PLAN_CACHE = None
    numint._XC_PLAN_CACHE_STATS.update(tasks=0)
    numint._XC_GRID_BLKSIZE = None
    cutensor._PLAN_CACHE_STATS.update(hits=0, misses=0)
    cutensor._plan_cache.clear()


_DM_CACHE = {}


def _dm(mol, xc='b3lyp'):
    key = (id(mol), xc)
    if key not in _DM_CACHE:
        mf = dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = 2
        mf.conv_tol = 1e-9
        mf.kernel()
        _DM_CACHE[key] = cp.asarray(mf.make_rdm1())
    return _DM_CACHE[key]


def _nr_rks_fixed_dm(mol, dm, xc, level=2):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    nelec, excsum, vmat = numint.nr_rks(ni, mol, grids, xc, cp.asarray(dm),
                                        hermi=1)
    return nelec, excsum, vmat


_KEYS = (KNOB, 'GPU4PYSCF_CONTRACT_PLAN_CACHE',
         'GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP', 'GPU4PYSCF_XC_GRID_BLOCKSIZE')


class EnvBase(unittest.TestCase):
    def setUp(self):
        # snapshot every knob this file touches and restore it exactly:
        # a leaked value would follow the process into later test files
        self._env = {key: os.environ.get(key) for key in _KEYS}
        for key in _KEYS:
            self.assertFalse(os.environ.get(key), key)
        self._lane = numint._scf_vmat_f32_lane
        self._strict = numint._strict_fp64_reference_active

    def tearDown(self):
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        numint._scf_vmat_f32_lane = self._lane
        numint._strict_fp64_reference_active = self._strict
        _reset()


class KnobSemantics(EnvBase):
    def test_default_off_never_engages(self):
        numint._XC_PLAN_CACHE = False
        sentinel = []
        orig = numint.scoped_plan_cache

        def _boom():
            sentinel.append('entered')
            raise AssertionError('scope entered with knob off')
        numint.scoped_plan_cache = _boom
        try:
            _nr_rks_fixed_dm(mol_benz, _dm(mol_benz), 'm062x')
        finally:
            numint.scoped_plan_cache = orig
        self.assertEqual(sentinel, [])
        self.assertEqual(numint._XC_PLAN_CACHE_STATS['tasks'], 0)
        self.assertEqual(cutensor._PLAN_CACHE_STATS['hits'], 0)
        self.assertEqual(cutensor._PLAN_CACHE_STATS['misses'], 0)
        self.assertEqual(len(cutensor._plan_cache), 0)
        # the task functions stay unwrapped: other source-contract tests
        # read their __code__ (the SP-30 _ao_f32 site check)
        for site in (numint._nr_rks_task, numint._nr_uks_task):
            self.assertFalse(hasattr(site, '__wrapped__'))
            self.assertIn('_xc_plan_cache_scope', site.__code__.co_names)

    def test_env_knob_parsing_and_strict_inertness(self):
        for value in ('0', 'false', 'False'):
            numint._XC_PLAN_CACHE = None
            os.environ[KNOB] = value
            self.assertFalse(numint._xc_plan_cache_enabled())
        numint._XC_PLAN_CACHE = None
        os.environ[KNOB] = '1'
        self.assertTrue(numint._xc_plan_cache_enabled())
        numint._strict_fp64_reference_active = True
        self.assertFalse(numint._xc_plan_cache_enabled())
        numint._strict_fp64_reference_active = False
        # inert under the strict sibling: the task runs, no scope entered
        numint._strict_fp64_reference_active = True
        _nr_rks_fixed_dm(mol_sph, _dm(mol_sph), 'm062x')
        self.assertEqual(numint._XC_PLAN_CACHE_STATS['tasks'], 0)
        self.assertEqual(cutensor._PLAN_CACHE_STATS['hits']
                         + cutensor._PLAN_CACHE_STATS['misses'], 0)
        self.assertEqual(len(cutensor._plan_cache), 0)


class FixedDmBitwise(EnvBase):
    def _check(self, mol, xc, lane, blksize=None, reps=6,
               expect_contractions=True):
        dm = _dm(mol)
        if blksize is not None:
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = str(blksize)
        numint._scf_vmat_f32_lane = lane
        numint._XC_PLAN_CACHE = False
        cutensor._plan_cache.clear()
        cutensor._PLAN_CACHE_STATS.update(hits=0, misses=0)
        numint._XC_PLAN_CACHE_STATS.update(tasks=0)
        ref = _nr_rks_fixed_dm(mol, dm, xc)
        self.assertEqual(len(cutensor._plan_cache), 0)
        self.assertEqual(cutensor._PLAN_CACHE_STATS['hits']
                         + cutensor._PLAN_CACHE_STATS['misses'], 0)
        os.environ[KNOB] = '1'
        numint._XC_PLAN_CACHE = None
        stats = []
        for rep in range(reps):
            got = _nr_rks_fixed_dm(mol, dm, xc)
            self.assertEqual(repr(ref[0]), repr(got[0]), f'rep={rep} nelec')
            self.assertEqual(repr(ref[1]), repr(got[1]), f'rep={rep} excsum')
            self.assertTrue(_bitwise(ref[2], got[2]),
                            f'rep={rep} vmat bytes differ')
            stats.append((cutensor._PLAN_CACHE_STATS['hits'],
                          cutensor._PLAN_CACHE_STATS['misses']))
        self.assertEqual(numint._XC_PLAN_CACHE_STATS['tasks'], reps)
        # the first armed call builds every plan (members of one block
        # already share entries); the pool state needs a few calls to
        # settle the operand alignment classes, after which every call
        # must be all hits -- the raw progression is pinned in the
        # assertion messages
        if expect_contractions:
            self.assertGreater(stats[0][1], 0, msg=str(stats))
            self.assertGreater(stats[0][0], 0, msg=str(stats))
            for rep in range(1, reps):
                self.assertGreaterEqual(stats[rep][1], stats[rep - 1][1],
                                        msg=str(stats))
                self.assertGreater(stats[rep][0], stats[rep - 1][0],
                                   msg=str(stats))
            for rep in range(reps // 2, reps):
                self.assertEqual(stats[rep][1], stats[reps // 2][1],
                                 msg=str(stats))
        else:
            self.assertEqual(stats[-1], (0, 0))
        self.assertEqual(len(cutensor._plan_cache), stats[-1][1])
        # knob back off in the same process: the per-call build agrees
        os.environ.pop(KNOB, None)
        numint._XC_PLAN_CACHE = None
        again = _nr_rks_fixed_dm(mol, dm, xc)
        self.assertTrue(_bitwise(ref[2], again[2]))
        self.assertEqual(cutensor._PLAN_CACHE_STATS, {'hits': stats[-1][0],
                                                      'misses': stats[-1][1]})

    def test_mgga_plain_fp64(self):
        self._check(mol_benz, 'm062x', False)

    def test_mgga_vmat_f32_lane(self):
        # contr_f32 cycle: the vmat loop casts the AO blocks to fp32, so
        # the scoped cache serves an fp32 operand family
        self._check(mol_benz, 'm062x', True)

    def test_mgga_loop_blksize_16384(self):
        # the funded composition's grid partition (coarse loop blocks)
        self._check(mol_benz, 'm062x', False, blksize=16384)
        self._check(mol_benz, 'm062x', True, blksize=16384)

    def test_gga_vmat_f32_lane(self):
        # GGA keeps the raw cuBLAS vmat dot: no task contraction, no
        # engagement, and the knob must not perturb the values
        self._check(mol_sph, 'b3lyp', True, expect_contractions=False)

    def test_uks_fixed_dm_bitwise(self):
        mf = dft.UKS(mol_open)
        mf.xc = 'm062x'
        mf.grids.level = 2
        mf.conv_tol = 1e-9
        mf.kernel()
        dms = cp.asarray(mf.make_rdm1())
        ni = dft.numint.NumInt()
        grids = gen_grid.Grids(mol_open)
        grids.level = 2
        grids.build()
        numint._XC_PLAN_CACHE = False
        ref = numint.nr_uks(ni, mol_open, grids, 'm062x', dms)
        os.environ[KNOB] = '1'
        numint._XC_PLAN_CACHE = None
        for rep in range(2):
            got = numint.nr_uks(ni, mol_open, grids, 'm062x', dms)
            self.assertEqual(repr(ref[0]), repr(got[0]))
            self.assertEqual(repr(ref[1]), repr(got[1]))
            for v0, v1 in zip(ref[2], got[2]):
                self.assertTrue(_bitwise(v0, v1), f'rep={rep}')
        self.assertEqual(numint._XC_PLAN_CACHE_STATS['tasks'], 2)
        self.assertGreater(cutensor._PLAN_CACHE_STATS['hits'], 0)

    def test_scf_e_tot(self):
        # Full-SCF twin: the pool layout changes between the two runs and
        # cuBLAS kernel selection is sensitive to operand alignment
        # (documented GEMM class), so the gate is the ulp floor rather
        # than bitwise; fixed-dm nr_rks above carries the bitwise gate.
        os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '16384'

        def run():
            mf = dft.RKS(mol_benz)
            mf.xc = 'm062x'
            mf.grids.level = 2
            mf.conv_tol = 1e-9
            return float(mf.kernel()), mf.converged
        numint._XC_PLAN_CACHE = False
        e0, c0 = run()
        os.environ[KNOB] = '1'
        numint._XC_PLAN_CACHE = None
        e1, c1 = run()
        self.assertTrue(c0 and c1)
        self.assertTrue(abs(e0 - e1) <= 1e-10,
                        f'e_tot knob off={e0!r} on={e1!r}')
        self.assertGreater(numint._XC_PLAN_CACHE_STATS['tasks'], 0)
        self.assertGreater(cutensor._PLAN_CACHE_STATS['hits'],
                           cutensor._PLAN_CACHE_STATS['misses'])


class ScopeSemantics(EnvBase):
    def _operands(self, dtype=cp.float64, n=48, g=4104):
        rng = np.random.default_rng(20261124)
        a = cp.asarray(rng.standard_normal((n, g)), dtype=dtype)
        b = cp.asarray(rng.standard_normal((n, g)), dtype=dtype)
        return a, b

    def test_outside_scope_inert(self):
        a, b = self._operands()
        cutensor.contract('ig,jg->ij', a, b)
        cutensor.contract('ig,jg->ij', a, b)
        self.assertFalse(cutensor._plan_cache_enabled())
        self.assertEqual(len(cutensor._plan_cache), 0)
        self.assertEqual(cutensor._PLAN_CACHE_STATS,
                         {'hits': 0, 'misses': 0})

    def test_hit_inside_then_inert_outside(self):
        a, b = self._operands()
        with cutensor.scoped_plan_cache():
            self.assertTrue(cutensor._plan_cache_enabled())
            cutensor.contract('ig,jg->ij', a, b)
            cutensor.contract('ig,jg->ij', a, b)
        self.assertEqual(cutensor._PLAN_CACHE_STATS, {'hits': 1, 'misses': 1})
        self.assertEqual(len(cutensor._plan_cache), 1)
        cutensor.contract('ig,jg->ij', a, b)
        self.assertEqual(cutensor._PLAN_CACHE_STATS, {'hits': 1, 'misses': 1})
        self.assertEqual(len(cutensor._plan_cache), 1)

    def test_reentrant_and_exception_safe(self):
        self.assertFalse(cutensor._scoped_plan_cache_active())
        with cutensor.scoped_plan_cache():
            with cutensor.scoped_plan_cache():
                self.assertTrue(cutensor._scoped_plan_cache_active())
            self.assertTrue(cutensor._scoped_plan_cache_active())
        self.assertFalse(cutensor._scoped_plan_cache_active())
        with self.assertRaises(ZeroDivisionError):
            with cutensor.scoped_plan_cache():
                1 // 0
        self.assertFalse(cutensor._scoped_plan_cache_active())
        self.assertFalse(cutensor._plan_cache_enabled())

    def test_process_wide_knob_untouched(self):
        # the pre-existing GPU4PYSCF_CONTRACT_PLAN_CACHE path keeps its
        # base key (no alignment prefix) and its own hit/miss accounting
        a, b = self._operands()
        os.environ['GPU4PYSCF_CONTRACT_PLAN_CACHE'] = '1'
        cutensor.contract('ig,jg->ij', a, b)
        cutensor.contract('ig,jg->ij', a, b)
        self.assertEqual(cutensor._PLAN_CACHE_STATS, {'hits': 1, 'misses': 1})
        self.assertEqual(len(cutensor._plan_cache), 1)
        key = next(iter(cutensor._plan_cache))
        self.assertEqual(key[0], 'ig,jg->ij')
        os.environ.pop('GPU4PYSCF_CONTRACT_PLAN_CACHE')
        with cutensor.scoped_plan_cache():
            cutensor.contract('ig,jg->ij', a, b)
        # scoped lookup does not alias the process-wide entry
        self.assertEqual(len(cutensor._plan_cache), 2)

    def test_alignment_class_key(self):
        # same shape/strides/dtype at several pointer offsets: every
        # distinct alignment class is its own entry, a repeated offset
        # hits, and every hit reproduces the per-call build at the SAME
        # pointers bitwise (fp64 and fp32, beta=1 accumulate included)
        for dtype, itemsize in ((cp.float64, 8), (cp.float32, 4)):
            n, g = 40, 520
            total = 4 * n * g
            base = cp.zeros(total + 1024, dtype=dtype)
            base += cp.arange(base.size, dtype=dtype)
            offs = [0, 8, 16, 64, 128, 256, 128, 8]
            expect_miss = len(set(offs))
            cutensor._plan_cache.clear()
            cutensor._PLAN_CACHE_STATS.update(hits=0, misses=0)
            per_call = cutensor._plan_cache_enabled
            fill = cp.ones((n, n), dtype=dtype)
            with cutensor.scoped_plan_cache():
                for k, off_bytes in enumerate(offs):
                    off = off_bytes // itemsize
                    a = base[off:off + n * g].reshape(n, g)
                    b = base[off + n * g:off + 2 * n * g].reshape(n, g)
                    out = base[off + 2 * n * g:off + 2 * n * g + n * n]\
                        .reshape(n, n)
                    # per-call build and cached call share the pointers:
                    # the result bytes are ferried to the host in between
                    cutensor._plan_cache_enabled = (lambda: False)
                    try:
                        cutensor.contract('ig,jg->ij', a, b, out=out)
                    finally:
                        cutensor._plan_cache_enabled = per_call
                    ref = cp.asnumpy(out)
                    cutensor.contract('ig,jg->ij', a, b, out=out)
                    self.assertTrue(np.array_equal(ref, cp.asnumpy(out)),
                                    f'{dtype} offset {off_bytes} B')
                    # accumulate form: out prefilled, beta = 1
                    out[:] = fill
                    cutensor._plan_cache_enabled = (lambda: False)
                    try:
                        cutensor.contract('ig,jg->ij', a, b, beta=1., out=out)
                    finally:
                        cutensor._plan_cache_enabled = per_call
                    ref2 = cp.asnumpy(out)
                    out[:] = fill
                    cutensor.contract('ig,jg->ij', a, b, beta=1., out=out)
                    self.assertTrue(np.array_equal(ref2, cp.asnumpy(out)),
                                    f'{dtype} offset {off_bytes} B beta=1')
                    self.assertEqual(
                        cutensor._PLAN_CACHE_STATS['misses'],
                        len(set(offs[:k + 1])), f'k={k} misses')
            n_entries = len(cutensor._plan_cache)
            self.assertEqual(n_entries, expect_miss)
            # one miss + one accumulate hit per new class, two hits per
            # repeated offset
            self.assertEqual(cutensor._PLAN_CACHE_STATS,
                             {'hits': 2 * len(offs) - expect_miss,
                              'misses': expect_miss})
            for key in cutensor._plan_cache:
                self.assertEqual(key[1][0], 'ig,jg->ij')


if __name__ == '__main__':
    unittest.main()

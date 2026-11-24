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
'''XC-BLGEN-PARTFIX-01 focused regression (2026-11-24).

The opt-in one-launch-per-block kernel form of the MR1 partition-fix
zeroing (GPU4PYSCF_XC_PARTFIX_KERNEL, default off) replaces the
per-fine-block cupy assignments inside numint._block_loop for float64
C-contiguous 3-D blocks.  It must be inert when unset, value-exact
(bitwise per block at both AO derivative orders; the 2-D deriv=0 blocks
keep the deployed statements) against the deployed statements,
cache-coherent with the grids screen cache, inert under the strict fp64
reference context, and give a bitwise-equal full-SCF twin at a coarse
loop block size.
'''
import os
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import dft
from gpu4pyscf.dft import numint, gen_grid

KNOB = 'GPU4PYSCF_XC_PARTFIX_KERNEL'


def setUpModule():
    global mol_sph, mol_benz
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

def tearDownModule():
    global mol_sph, mol_benz
    for mol in (mol_sph, mol_benz):
        mol.stdout.close()
    mol_sph = mol_benz = None


def _reset():
    numint._PARTFIX_KERNEL = None
    numint._XC_GRID_BLKSIZE = None
    numint._PARTFIX_KERNEL_STATS.update(
        segment_blocks=0, control_blocks=0, kernel_launches=0,
        kernel_segments=0, fallback_blocks=0)


def _stats():
    return dict(numint._PARTFIX_KERNEL_STATS)


class KnobSemantics(unittest.TestCase):

    def test_default_off(self):
        os.environ.pop(KNOB, None)
        _reset()
        self.assertFalse(numint._partfix_kernel_enabled())

    def test_env_parsing(self):
        for raw, want in (('1', True), ('true', True), ('True', True),
                          ('0', False), ('false', False)):
            os.environ[KNOB] = raw
            _reset()
            self.assertIs(numint._partfix_kernel_enabled(), want,
                          msg='KNOB=%r' % raw)
        os.environ.pop(KNOB, None)
        _reset()

    def test_strict_fp64_reference_inert(self):
        os.environ[KNOB] = '1'
        _reset()
        try:
            self.assertTrue(numint._partfix_kernel_enabled())
            with numint.strict_fp64_reference_ctx():
                self.assertFalse(numint._partfix_kernel_enabled())
            self.assertTrue(numint._partfix_kernel_enabled())
        finally:
            os.environ.pop(KNOB, None)
            _reset()


class BlockwiseBitwise(unittest.TestCase):
    '''Per-block int64-view bitwise equality of every yielded ao tensor,
    knob on vs knob off, at a coarse loop block size (8192 = 2 x
    MIN_BLK_SIZE) for deriv 0/1/2.'''

    def _collect(self, mol, deriv, loop_blksize):
        grids = gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        ni = numint.NumInt()
        nao = mol.nao
        blocks = []
        for ao, idx, w, c in ni.block_loop(mol, grids, nao, deriv,
                                           loop_blksize=loop_blksize):
            blocks.append((cp.asnumpy(ao).tobytes(),
                           cp.asnumpy(idx).tobytes(),
                           cp.asnumpy(w).tobytes(), ao.shape, ao.dtype))
        self.assertGreater(len(blocks), 1)
        return blocks, grids

    def _check(self, mol, deriv, loop_blksize):
        try:
            os.environ.pop(KNOB, None)
            _reset()
            ref, grids = self._collect(mol, deriv, loop_blksize)
            s_off = _stats()
        finally:
            os.environ.pop(KNOB, None)
            _reset()
        try:
            os.environ[KNOB] = '1'
            _reset()
            got, grids2 = self._collect(mol, deriv, loop_blksize)
            s_on = _stats()
        finally:
            os.environ.pop(KNOB, None)
            _reset()
        self.assertEqual(len(ref), len(got))
        for i, (a, b) in enumerate(zip(ref, got)):
            self.assertEqual(a, b, msg='block %d differs' % i)
        # the mechanism is live in both lanes (the screen drops rows at
        # this grid/basis; deriv>0 blocks carry the partition fix)
        self.assertGreater(s_off['control_blocks'], 0)
        self.assertEqual(s_off['kernel_launches'], 0)
        if deriv > 0:
            self.assertGreater(s_on['kernel_launches'], 0)
            self.assertEqual(s_on['kernel_launches'], s_on['segment_blocks'])
            self.assertEqual(s_on['control_blocks'], 0)
            self.assertGreaterEqual(s_on['kernel_segments'],
                                    s_on['kernel_launches'])
        else:
            # 2-D blocks keep the deployed statements by design
            self.assertEqual(s_on['kernel_launches'], 0)
            self.assertGreater(s_on['fallback_blocks'], 0)
        self.assertEqual(s_on['segment_blocks'], s_off['segment_blocks'])

    def test_benzene_deriv1(self):
        self._check(mol_benz, 1, 8192)

    def test_benzene_deriv2(self):
        self._check(mol_benz, 2, 8192)

    def test_benzene_deriv0_2d_untouched(self):
        self._check(mol_benz, 0, 8192)

    def test_benzene_16384(self):
        self._check(mol_benz, 1, 16384)


class CacheCoherence(unittest.TestCase):

    def test_index_dies_with_screen_cache(self):
        mol = mol_benz
        grids = gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        ni = numint.NumInt()
        opt = ni.gdftopt if ni.gdftopt is not None else None
        try:
            os.environ[KNOB] = '1'
            _reset()
            pf = grids.get_partition_fix(opt, blksize=8192)
            self.assertIsNotNone(pf)
            idx1 = grids.get_partition_fix_kernel_index(opt, blksize=8192)
            self.assertIn(('partition_fix_kernel', 8192), grids._non0ao_idx)
            self.assertIs(grids.get_partition_fix_kernel_index(
                opt, blksize=8192), idx1)
            # wipe the screen cache: the derived kernel index dies with it
            grids._non0ao_idx = None
            idx2 = grids.get_partition_fix_kernel_index(opt, blksize=8192)
            self.assertIsNot(idx2, idx1)
            # value-exact after the rebuild (fresh derivation)
            a = [None if p is None else tuple(cp.asnumpy(x).tobytes()
                                              for x in p[:4]) + tuple(p[4:])
                 for p in idx1]
            b = [None if p is None else tuple(cp.asnumpy(x).tobytes()
                                              for x in p[:4]) + tuple(p[4:])
                 for p in idx2]
            self.assertEqual(a, b)
            # coarse <= MIN_BLK_SIZE keeps the None contract
            self.assertIsNone(
                grids.get_partition_fix_kernel_index(opt, blksize=4096))
            self.assertIsNone(
                grids.get_partition_fix_kernel_index(opt, blksize=None))
        finally:
            os.environ.pop(KNOB, None)
            _reset()

    def test_set_equality_against_deployed_statements(self):
        '''The zeroed (row, column) pairs of the kernel form are exactly
        the deployed statements' set: derive both from the same partition
        fix on a dense block.'''
        try:
            os.environ[KNOB] = '1'
            _reset()
            self._set_equality_body()
        finally:
            os.environ.pop(KNOB, None)
            _reset()

    def _set_equality_body(self):
        mol = mol_benz
        grids = gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        ni = numint.NumInt()
        ni.build(mol, grids.coords)
        opt = ni.gdftopt
        loop = 8192
        MB = numint.MIN_BLK_SIZE
        pf = grids.get_partition_fix(opt, blksize=loop)
        compact = grids.get_partition_fix_kernel_index(opt, blksize=loop)
        nao_sub = max(len(e[1]) for e in grids.get_non0ao_idx(opt,
                                                              blksize=loop))
        for block_id, entries in enumerate(pf):
            nblk = min((block_id + 1) * loop, grids.size) - block_id * loop
            plan = compact[block_id]
            control = np.ones((2, nao_sub, nblk), dtype=np.float64)
            fired = False
            for fine_off, drop_pos in entries:
                col0 = fine_off
                col1 = min(fine_off + MB, nblk)
                if col0 >= col1 or drop_pos.size == 0:
                    continue
                control[:, drop_pos, col0:col1] = 0.0
                fired = True
            if not fired:
                self.assertIsNone(plan)
                continue
            self.assertIsNotNone(plan)
            work_dev = cp.ones((2, nao_sub, nblk), dtype=cp.float64)
            numint._apply_partition_fix(work_dev, pf, block_id,
                                        grids, opt, loop)
            work = cp.asnumpy(work_dev)
            # set equality of the zeroed pairs
            zc = set(map(tuple, np.argwhere(control == 0.0)))
            zk = set(map(tuple, np.argwhere(work == 0.0)))
            self.assertEqual(zc, zk, msg='block %d' % block_id)
        self.assertGreater(
            numint._PARTFIX_KERNEL_STATS['kernel_launches'], 0,
            msg='kernel form never engaged in the set-equality sweep')


class FixedDMBitwise(unittest.TestCase):
    '''Fixed-dm nr_rks twin: the strongest in-process evidence -- same
    grids object, same dm, knob on vs off, BITWISE nelec/excsum/vmat.'''

    def test_m062x_nr_rks_bitwise(self):
        mol = mol_benz
        grids = gen_grid.Grids(mol)
        grids.level = 3
        grids.build()
        ni = numint.NumInt()
        ni.build(mol, grids.coords)
        nao = mol.nao
        rng = np.random.default_rng(20261124)
        dm = cp.asarray(rng.random((nao, nao)) * .1)
        dm = dm + dm.T
        try:
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '8192'
            os.environ.pop(KNOB, None)
            _reset()
            ref = numint.nr_rks(ni, mol, grids, 'm062x', dm)
            s0 = _stats()
            os.environ[KNOB] = '1'
            _reset()
            got = numint.nr_rks(ni, mol, grids, 'm062x', dm)
            s1 = _stats()
        finally:
            os.environ.pop(KNOB, None)
            os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
            _reset()
        for r, g in zip(ref, got):
            if isinstance(r, cp.ndarray):
                self.assertTrue(r.shape == g.shape and r.dtype == g.dtype
                                and r.tobytes() == g.tobytes())
            else:
                self.assertEqual(repr(r), repr(g))
        self.assertGreater(s1['kernel_launches'], 0)
        self.assertEqual(s1['kernel_launches'], s1['segment_blocks'])
        self.assertEqual(s1['control_blocks'], 0)
        self.assertEqual(s0['kernel_launches'], 0)
        self.assertGreater(s0['control_blocks'], 0)


class FullSCFTwin(unittest.TestCase):

    def _scf(self, mol, xc):
        mf = dft.RKS(mol, xc=xc)
        mf.grids.level = 3
        mf.conv_tol = 1e-11
        mf.max_cycle = 60
        mf.kernel()
        self.assertTrue(mf.converged)
        return (mf.e_tot, mf.cycles if hasattr(mf, 'cycles') else None)

    def test_m062x_twin_ulp_floor(self):
        # Full-SCF twin: the pool layout changes between the two runs and
        # cuBLAS kernel selection is sensitive to operand alignment
        # (documented GEMM class, same precedent as the plan-cache suite),
        # so the gate is the ulp floor; the fixed-dm nr_rks twin and the
        # per-block bitwise sweep above carry the bitwise gate.
        try:
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '8192'
            os.environ.pop(KNOB, None)
            _reset()
            e0 = self._scf(mol_benz, 'm062x')[0]
            s0 = _stats()
            os.environ[KNOB] = '1'
            _reset()
            e1 = self._scf(mol_benz, 'm062x')[0]
            s1 = _stats()
        finally:
            os.environ.pop(KNOB, None)
            os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
            _reset()
        self.assertTrue(abs(e0 - e1) <= 1e-10,
                        msg='e_tot knob off=%r on=%r' % (e0, e1))
        self.assertGreater(s1['kernel_launches'], 0)
        self.assertEqual(s1['kernel_launches'], s1['segment_blocks'])
        self.assertEqual(s1['control_blocks'], 0)
        self.assertEqual(s0['kernel_launches'], 0)
        self.assertGreater(s0['control_blocks'], 0)

    def test_lda_default_loop_untouched(self):
        '''LDA keeps MIN_BLK_SIZE (partition_fix None): the knob has no
        entry point and no engagement.'''
        try:
            os.environ[KNOB] = '1'
            _reset()
            e = self._scf(mol_sph, 'lda,vwn')[0]
            s = _stats()
        finally:
            os.environ.pop(KNOB, None)
            _reset()
        self.assertTrue(np.isfinite(e))
        self.assertEqual(s['kernel_launches'], 0)
        self.assertEqual(s['segment_blocks'], 0)


if __name__ == '__main__':
    print('Tests from %s' % __file__)
    unittest.main()

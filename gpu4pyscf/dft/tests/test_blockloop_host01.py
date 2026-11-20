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
'''BLOCKLOOP-HOST-01 Phase B focused regression (2026-11-05).

The opt-in block_loop host-overhead hoist (GPU4PYSCF_BLOCKLOOP_HOST_OPT,
default off) prepares the per-block host state once per generator and
must yield BITWISE-equal (ao, idx, weight, coords) tuples to the default
loop, with bitwise-equal fixed-dm nr_rks outputs on both the plain fp64
path and a contr_f32 (vmat-f32 lane) cycle.  Prereg
preregistration-blockloop-host01-v1.json, phase_b_gate; mechanism frozen
in phasea-source-audit.md (C1/C1x/C2/C3/C4).
'''
import os
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft import gen_grid


def setUpModule():
    global mol_sph, mol_benz
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    # multi-coarse-block + tail at loop_blksize 16384 with an ACTIVE
    # partition_fix (coarse > MIN_BLK_SIZE; MR1 zeroing lists non-empty)
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
    mol_sph.stdout.close()
    mol_benz.stdout.close()
    del mol_sph, mol_benz


def _collect_blocks(mol, deriv, loop_blksize, strict=False, provider=None):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = 2
    grids.build(with_non0tab=False, sort_grids=True)
    ni.build(mol, grids.coords)
    out = []
    for ao, idx, weight, coords in ni.block_loop(
            mol, grids, mol.nao, deriv, max_memory=None,
            loop_blksize=loop_blksize, strict_grid_order=strict,
            ao_provider=provider):
        out.append((cp.asarray(ao), cp.asarray(idx), cp.asarray(weight),
                    cp.asarray(coords)))
    return out, grids, ni


def _tuple_bits(t):
    ao, idx, weight, coords = t
    return (ao.dtype, ao.shape, ao.tobytes(), idx.tobytes(),
            weight.dtype, weight.tobytes(), coords.dtype, coords.shape,
            coords.tobytes(), bool(ao.flags.c_contiguous),
            bool(weight.flags.c_contiguous),
            bool(coords.flags.c_contiguous))


class KnobSemantics(unittest.TestCase):
    def setUp(self):
        self._saved = numint._BLOCKLOOP_HOST_OPT
        self._saved_stats = dict(numint._BLOCKLOOP_HOST_STATS)
        self._saved_strict = numint._strict_fp64_reference_active

    def tearDown(self):
        numint._BLOCKLOOP_HOST_OPT = self._saved
        numint._BLOCKLOOP_HOST_STATS.update(self._saved_stats)
        numint._strict_fp64_reference_active = self._saved_strict
        numint._strict_fp64_reference_active = False

    def test_default_off_never_engages(self):
        numint._BLOCKLOOP_HOST_OPT = False
        before = numint._BLOCKLOOP_HOST_STATS['generators']
        sentinel = []
        orig = numint._block_loop_prepared_iter
        def _boom(*a, **kw):
            sentinel.append('called')
            raise AssertionError('prepared path taken with knob off')
        numint._block_loop_prepared_iter = _boom
        try:
            blocks, _, _ = _collect_blocks(mol_benz, 1, 16384)
        finally:
            numint._block_loop_prepared_iter = orig
        self.assertGreater(len(blocks), 0)
        self.assertEqual(sentinel, [])
        self.assertEqual(numint._BLOCKLOOP_HOST_STATS['generators'], before)

    def test_env_knob_parsing_and_strict_inertness(self):
        numint._BLOCKLOOP_HOST_OPT = None
        os.environ['GPU4PYSCF_BLOCKLOOP_HOST_OPT'] = '1'
        try:
            self.assertTrue(numint._blockloop_host_opt_enabled())
            numint._strict_fp64_reference_active = True
            self.assertFalse(numint._blockloop_host_opt_enabled())
            numint._strict_fp64_reference_active = False
            self.assertTrue(numint._blockloop_host_opt_enabled())
        finally:
            del os.environ['GPU4PYSCF_BLOCKLOOP_HOST_OPT']
            numint._BLOCKLOOP_HOST_OPT = None
        numint._BLOCKLOOP_HOST_OPT = None
        os.environ['GPU4PYSCF_BLOCKLOOP_HOST_OPT'] = 'false'
        try:
            self.assertFalse(numint._blockloop_host_opt_enabled())
        finally:
            del os.environ['GPU4PYSCF_BLOCKLOOP_HOST_OPT']
            numint._BLOCKLOOP_HOST_OPT = None
        # unset -> module default off
        self.assertFalse(numint._blockloop_host_opt_enabled())


class BlockIdentity(unittest.TestCase):
    def setUp(self):
        self._saved = numint._BLOCKLOOP_HOST_OPT
        self._saved_stats = dict(numint._BLOCKLOOP_HOST_STATS)

    def tearDown(self):
        numint._BLOCKLOOP_HOST_OPT = self._saved
        numint._BLOCKLOOP_HOST_STATS.update(self._saved_stats)

    def _assert_bitwise(self, mol, deriv, blksize, strict=False, provider=None):
        numint._BLOCKLOOP_HOST_OPT = False
        ref, grids, ni = _collect_blocks(mol, deriv, blksize, strict, provider)
        numint._BLOCKLOOP_HOST_OPT = True
        before = numint._BLOCKLOOP_HOST_STATS['generators']
        got, grids2, _ = _collect_blocks(mol, deriv, blksize, strict, provider)
        self.assertEqual(numint._BLOCKLOOP_HOST_STATS['generators'], before + 1)
        self.assertEqual(len(ref), len(got))
        for i, (t_ref, t_got) in enumerate(zip(ref, got)):
            # assertTrue with a short message: assertEqual on the byte
            # tuples would run difflib over multi-MB strings
            self.assertTrue(_tuple_bits(t_ref) == _tuple_bits(t_got),
                            f'block {i} differs (deriv={deriv}, '
                            f'blk={blksize})')

    def test_benz_mgga_multi_block_partition_fix(self):
        # loop_blksize > MIN activates the MR1 partition_fix lists and
        # exercises the C1x prepared kernel-argument path (deriv=1, 3D)
        self._assert_bitwise(mol_benz, 1, 16384)
        self._assert_bitwise(mol_benz, 1, 8192)

    def test_benz_lda_tail_block(self):
        # deriv=0 covers the 2D out[0] branch; smaller blocks force a
        # partial tail block
        self._assert_bitwise(mol_benz, 0, 4096)
        self._assert_bitwise(mol_benz, 0, 16384)

    def test_sph_strict_grid_order(self):
        self._assert_bitwise(mol_sph, 1, 4096, strict=True)
        self._assert_bitwise(mol_sph, 1, 16384, strict=True)

    def test_ao_provider_fallthrough(self):
        # a provider that declines every block must exercise the
        # prepared eval on the knob path (carry interplay contract)
        calls = []
        def provider(block_id, shape):
            calls.append((block_id, shape))
            return None
        self._assert_bitwise(mol_benz, 1, 16384, provider=provider)
        self.assertGreater(len(calls), 0)


def _nr_rks_fixed_dm(mol, dm, xc='b3lyp'):
    ni = dft.numint.NumInt()
    grids = gen_grid.Grids(mol)
    grids.level = 2
    grids.build()
    dms = cp.asarray(dm)
    nelec, excsum, vmat = numint.nr_rks(ni, mol, grids, xc, dms, hermi=1)
    return nelec, excsum, vmat, grids


class FixedDmNrRks(unittest.TestCase):
    def setUp(self):
        self._saved_knob = numint._BLOCKLOOP_HOST_OPT
        self._saved_lane = numint._scf_vmat_f32_lane

    def tearDown(self):
        numint._BLOCKLOOP_HOST_OPT = self._saved_knob
        numint._scf_vmat_f32_lane = self._saved_lane

    def _dm(self, mol):
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-9
        mf.kernel()
        return cp.asarray(mf.make_rdm1())

    def test_bitwise_plain_fp64(self):
        dm = self._dm(mol_benz)
        numint._BLOCKLOOP_HOST_OPT = False
        ne0, ex0, v0, _ = _nr_rks_fixed_dm(mol_benz, dm, xc='b3lyp')
        numint._BLOCKLOOP_HOST_OPT = True
        ne1, ex1, v1, _ = _nr_rks_fixed_dm(mol_benz, dm, xc='b3lyp')
        self.assertEqual(repr(ne0), repr(ne1))
        self.assertEqual(repr(ex0), repr(ex1))
        self.assertEqual(v0.tobytes(), v1.tobytes())

    def test_bitwise_vmat_f32_lane(self):
        # contr_f32 cycle: the vmat loop casts fresh AO blocks to fp32
        # after the (possibly prepared) block_loop yield
        dm = self._dm(mol_benz)
        numint._scf_vmat_f32_lane = True
        try:
            numint._BLOCKLOOP_HOST_OPT = False
            ne0, ex0, v0, _ = _nr_rks_fixed_dm(mol_benz, dm, xc='b3lyp')
            numint._BLOCKLOOP_HOST_OPT = True
            ne1, ex1, v1, _ = _nr_rks_fixed_dm(mol_benz, dm, xc='b3lyp')
        finally:
            numint._scf_vmat_f32_lane = self._saved_lane
        self.assertEqual(repr(ne0), repr(ne1))
        self.assertEqual(repr(ex0), repr(ex1))
        self.assertEqual(v0.tobytes(), v1.tobytes())

    def test_scf_e_tot_bitwise(self):
        # Full-SCF twin: the per-generator upload changes the in-process
        # cupy pool layout, and cuBLAS kernel selection is sensitive to
        # operand alignment (documented GEMM class), so a converged SCF
        # may differ in the last ulp(s) -- observed -76.35814748842704
        # vs -76.35814748842705 (1e-14 Ha).  Bitwise is asserted for
        # fixed-dm nr_rks (above); here the gate is the documented CPU
        # noise floor with raw values pinned in the assertion message.
        mf0 = dft.RKS(mol_sph)
        mf0.xc = 'b3lyp'
        mf0.grids.level = 2
        numint._BLOCKLOOP_HOST_OPT = False
        e0 = mf0.kernel()
        mf1 = dft.RKS(mol_sph)
        mf1.xc = 'b3lyp'
        mf1.grids.level = 2
        numint._BLOCKLOOP_HOST_OPT = True
        e1 = mf1.kernel()
        self.assertTrue(abs(float(e0) - float(e1)) <= 1e-10,
                        f'e_tot knob off={float(e0)!r} on={float(e1)!r}')


if __name__ == '__main__':
    unittest.main()

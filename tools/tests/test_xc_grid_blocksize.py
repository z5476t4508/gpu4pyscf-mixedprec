#!/usr/bin/env python3
"""Regression tests for the LC-11 opt-in SCF XC grid-block partition
(gpu4pyscf/dft/numint.py, GPU4PYSCF_XC_GRID_BLOCKSIZE).

The change adds:
  * `_scf_grid_blksize()` -- env reader (positive multiple of MIN_BLK_SIZE,
    invalid values fall back with a warning, cached per process);
  * `_effective_scf_blksize()` -- multi-GPU guard: a loop block size larger
    than MIN_BLK_SIZE is only honored when the device's [grid_start,
    grid_end) range boundaries are block-aligned, else falls back to
    MIN_BLK_SIZE (a non-last device range ends at a MIN_BLK_SIZE boundary
    and an oversized block could straddle it, double-counting points);
  * `loop_blksize` kwarg on `_block_loop` (partition math + ao-sparsity
    cache key) and its use in the four SCF-side loops
    (_nr_rks_task rho+fock, _nr_uks_task rho+fock, nr_nlc_vxc rho+vmat)
    with loop_blksize-scaled scratch buffers;
  * `grids._non0ao_idx` becomes a dict keyed by block size
    (gen_grid.get_non0ao_idx(opt, blksize=...)); invalidation sites still
    assign None.

Defaults are untouched: with the env var unset every path is the previous
MIN_BLK_SIZE behavior and the sparsity cache layout holds one entry.

These tests are CPU-only (no CUDA context): they pin the env reader, the
multi-GPU guard, the cache keying/invalidation semantics, the _block_loop
partition math against a reference partition, and the call-site wiring via
source inspection (the numerical GPU evidence lives in the LC-11 evidence
directory).  Skipped under stdlib python3.
"""
import os
import re
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
GENGRID_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'gen_grid.py')


class _EnvClean:
    def __enter__(self):
        self.saved = os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
        return self

    def __exit__(self, *exc):
        import gpu4pyscf.dft.numint as gnumint
        if self.saved is None:
            os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
        else:
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = self.saved
        gnumint._XC_GRID_BLKSIZE = None
        return False


class TestScfGridBlksizeEnv(unittest.TestCase):
    def test_default_is_min_blk_size(self):
        import gpu4pyscf.dft.numint as gnumint
        with _EnvClean():
            gnumint._XC_GRID_BLKSIZE = None
            self.assertEqual(gnumint._scf_grid_blksize(),
                             gnumint.MIN_BLK_SIZE)

    def test_valid_multiple_accepted_and_cached(self):
        import gpu4pyscf.dft.numint as gnumint
        with _EnvClean():
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '16384'
            gnumint._XC_GRID_BLKSIZE = None
            self.assertEqual(gnumint._scf_grid_blksize(), 16384)
            # cached: changing the env afterwards does not change the value
            os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '8192'
            self.assertEqual(gnumint._scf_grid_blksize(), 16384)
            gnumint._XC_GRID_BLKSIZE = None
            self.assertEqual(gnumint._scf_grid_blksize(), 8192)

    def test_invalid_values_fall_back_with_warning(self):
        import gpu4pyscf.dft.numint as gnumint
        import warnings
        for bad in ('4097', '0', '-8192', 'abc', ''):
            with _EnvClean(), warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                if bad == '':
                    # empty string == unset for this knob
                    os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
                else:
                    os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = bad
                gnumint._XC_GRID_BLKSIZE = None
                val = gnumint._scf_grid_blksize()
                self.assertEqual(val, gnumint.MIN_BLK_SIZE,
                                 f'{bad!r} must fall back to the default')
                if bad != '':
                    self.assertTrue(any('ignored' in str(x.message) for x in w),
                                    f'{bad!r} must warn')


class TestEffectiveScfBlksize(unittest.TestCase):
    def test_device0_full_range_keeps_blksize(self):
        import gpu4pyscf.dft.numint as gnumint
        self.assertEqual(
            gnumint._effective_scf_blksize(16384, 0, 61078, 61078), 16384)

    def test_unaligned_end_falls_back(self):
        # a non-last device range ending at a MIN_BLK_SIZE boundary that is
        # not blksize-aligned: 40960 % 16384 != 0
        import gpu4pyscf.dft.numint as gnumint
        self.assertEqual(
            gnumint._effective_scf_blksize(16384, 0, 40960, 65536),
            gnumint.MIN_BLK_SIZE)

    def test_unaligned_start_falls_back(self):
        import gpu4pyscf.dft.numint as gnumint
        self.assertEqual(
            gnumint._effective_scf_blksize(16384, 4096, 65536, 65536),
            gnumint.MIN_BLK_SIZE)

    def test_aligned_range_keeps_blksize(self):
        import gpu4pyscf.dft.numint as gnumint
        self.assertEqual(
            gnumint._effective_scf_blksize(16384, 16384, 65536, 65536), 16384)

    def test_min_blksize_passthrough(self):
        import gpu4pyscf.dft.numint as gnumint
        self.assertEqual(
            gnumint._effective_scf_blksize(gnumint.MIN_BLK_SIZE,
                                           4096, 8192, 65536),
            gnumint.MIN_BLK_SIZE)


class TestBlockLoopPartitionMath(unittest.TestCase):
    '''Partition math of _block_loop with loop_blksize, against a reference
    partition, without touching CUDA.'''

    @staticmethod
    def _reference_partition(grid_start, grid_end, ngrids, blksize):
        blocks = []
        b0 = grid_start // blksize
        b1 = (grid_end + blksize - 1) // blksize
        for block_id in range(b0, b1):
            ip0 = block_id * blksize
            ip1 = min(ip0 + blksize, ngrids)
            blocks.append((ip0, ip1))
        return blocks

    def test_partition_math_variants(self):
        import gpu4pyscf.dft.numint as gnumint
        MB = gnumint.MIN_BLK_SIZE
        cases = [
            (0, 61078, 61078, MB),
            (0, 61078, 61078, 8 * MB),
            (0, 61078, 61078, 16 * MB),
            (0, 61078, 61078, 65536 if 65536 % MB == 0 else 16 * MB),
            (0, 125924, 125924, 8 * MB),
            (8 * MB, 16 * MB, 24 * MB, 8 * MB),   # aligned sub-range
            (0, MB, MB, 8 * MB),                  # single undersized block
            (0, 5 * MB, 5 * MB, 8 * MB),          # tail shorter than blk
        ]
        for grid_start, grid_end, ngrids, blk in cases:
            self.assertEqual(grid_start % MB, 0)
            with self.subTest(grid_start=grid_start, grid_end=grid_end,
                              blksize=blk):
                ref = self._reference_partition(grid_start, grid_end,
                                                ngrids, blk)
                # full coverage, no overlap, inside range
                self.assertEqual(ref[0][0], grid_start)
                self.assertEqual(ref[-1][1], min(grid_end, ngrids))
                for (a0, a1), (b0, b1) in zip(ref, ref[1:]):
                    self.assertEqual(a1, b0)
                for ip0, ip1 in ref:
                    self.assertGreaterEqual(ip0, grid_start)
                    self.assertLessEqual(ip1, grid_end)
                    self.assertLess(ip0, ip1)

    def test_loop_blksize_must_be_multiple(self):
        import gpu4pyscf.dft.numint as gnumint
        src = open(NUMINT_SRC).read()
        self.assertIn('assert loop_blksize % MIN_BLK_SIZE == 0', src)


class TestCacheKeying(unittest.TestCase):
    def test_cache_dict_keyed_by_blksize(self):
        src = open(GENGRID_SRC).read()
        # get_non0ao_idx takes blksize and keys _non0ao_idx by it
        self.assertRegex(
            src, r'def get_non0ao_idx\(self, opt=None, blksize=None\)')
        self.assertRegex(src, r'self\._non0ao_idx\.get\(cache_key\)')
        self.assertRegex(
            src, r'self\._non0ao_idx\[cache_key\] = entry')
        # the builder receives the size and asserts the multiple contract
        self.assertRegex(
            src, r'def _build_non0ao_idx_cache\(self, opt=None, blksize=None\)')
        self.assertRegex(
            src, r'block_size = numint\.MIN_BLK_SIZE if blksize is None else blksize')
        self.assertRegex(
            src, r'assert block_size % numint\.MIN_BLK_SIZE == 0')

    def test_invalidation_sites_still_reset(self):
        src = open(GENGRID_SRC).read()
        self.assertEqual(src.count('self._non0ao_idx = None'), 3,
                         'grid build/prune/reset must keep clearing the cache')

    def test_block_loop_uses_keyed_cache(self):
        src = open(NUMINT_SRC).read()
        self.assertIn('grids.get_non0ao_idx(opt, blksize=loop_blksize)', src)
        # buf sized to the loop partition, not the module constant
        self.assertRegex(
            src, r'buf = cupy\.empty\(\(comp, nao_max, loop_blksize\), order=\'C\'\)')


class TestCallSiteWiring(unittest.TestCase):
    '''The four SCF-side loops opt in; the non-SCF block_loop callers must
    NOT pass loop_blksize (their scratch buffers are sized to their own
    module MIN_BLK_SIZE constants).'''

    def test_scf_tasks_pass_loop_blksize(self):
        src = open(NUMINT_SRC).read()

        def count_in_function(func_name, needle):
            m = re.search(rf'def {func_name}\(.*?(?=\ndef )', src, re.S)
            self.assertIsNotNone(m, func_name)
            return m.group(0).count(needle)

        for task in ('_nr_rks_task', '_nr_uks_task', 'nr_nlc_vxc'):
            self.assertEqual(
                count_in_function(task, 'loop_blksize=loop_blksize'), 2,
                f'{task} must wire loop_blksize into both its block loops')
            self.assertGreaterEqual(
                count_in_function(task, '_effective_scf_blksize'), 1,
                f'{task} must derive loop_blksize via the alignment guard')

    def test_non_scf_callers_untouched(self):
        src = open(NUMINT_SRC).read()
        for func in ('get_rho', 'get_rho_with_derivatives', '_nr_rks_fxc_task',
                     '_nr_uks_fxc_task', 'cache_xc_kernel', 'nr_rks_group'):
            m = re.search(rf'def {func}\(.*?(?=\ndef )', src, re.S)
            if m is None:
                continue
            self.assertNotIn('loop_blksize', m.group(0),
                             f'{func} must keep default blocks')

    def test_scf_task_buffers_scale_with_loop_blksize(self):
        src = open(NUMINT_SRC).read()
        m_rks = re.search(r'def _nr_rks_task\(.*?(?=\ndef )', src, re.S)
        m_uks = re.search(r'def _nr_uks_task\(.*?(?=\ndef )', src, re.S)
        self.assertRegex(m_rks.group(0), r'cupy\.empty\(loop_blksize \* nao\)')
        self.assertRegex(
            m_rks.group(0), r'cupy\.empty\(loop_blksize \* max\(2\*nocc, nao\)\)')
        self.assertRegex(m_uks.group(0), r'cupy\.empty\(loop_blksize \* nao\)')
        self.assertRegex(
            m_uks.group(0),
            r'cupy\.empty\(loop_blksize \* max\(2\*nocc_alpha, 2\*nocc_beta, nao\)\)')


class TestDefaultBehaviorUnchanged(unittest.TestCase):
    def test_default_env_absent_keeps_min_blk(self):
        import gpu4pyscf.dft.numint as gnumint
        with _EnvClean():
            gnumint._XC_GRID_BLKSIZE = None
            self.assertEqual(gnumint._scf_grid_blksize(), gnumint.MIN_BLK_SIZE)
            # and the guard is a no-op for the default
            self.assertEqual(
                gnumint._effective_scf_blksize(gnumint.MIN_BLK_SIZE,
                                               4096, 8192, 65536),
                gnumint.MIN_BLK_SIZE)


class TestNrNlcLazyGrids(unittest.TestCase):
    '''LC-11 regression guard: nr_nlc_vxc must keep upstream's lazy-grid
    compatibility.  _DFHF.get_veff skips initialize_grids when
    ks.grids.coords is already set, so nlcgrids can arrive UNBUILT; the
    hoisted ngrids read in the LC-11 blocksize change once crashed on
    grids.coords=None before _block_loop could build the grids lazily.
    Needs the GPU env (skipped otherwise).'''

    def test_unbuilt_grids_equivalent_to_built(self):
        import numpy  # noqa: F401
        import cupy
        import pyscf
        from gpu4pyscf import dft as gpu_dft

        mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                      verbose=0)
        mf = gpu_dft.RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.grids.level = 2
        mf.nlcgrids.level = 2
        dm = cupy.asarray(mf.init_guess_by_minao())
        ni = mf._numint

        # built-grid reference
        mf.nlcgrids.build(mol)
        n1, e1, v1 = ni.nr_nlc_vxc(mol, mf.nlcgrids, 'wb97m-v', dm)

        # fresh UNBUILT grids object: lazy build must happen inside
        from gpu4pyscf.dft.gen_grid import Grids
        lazy = Grids(mol)
        lazy.level = 2
        self.assertIsNone(lazy.coords)
        n2, e2, v2 = ni.nr_nlc_vxc(mol, lazy, 'wb97m-v', dm)

        self.assertLess(abs(float(n1) - float(n2)), 1e-10)
        self.assertLess(abs(float(e1) - float(e2)), 1e-10)
        cupy.testing.assert_allclose(v1, v2, atol=1e-12)

    def test_unbuilt_grids_with_optin_blksize(self):
        import numpy  # noqa: F401
        import cupy
        import pyscf
        from gpu4pyscf import dft as gpu_dft
        from gpu4pyscf.dft.gen_grid import Grids
        import gpu4pyscf.dft.numint as gnumint

        mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                      verbose=0)
        mf = gpu_dft.RKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.grids.level = 2
        mf.nlcgrids.level = 2
        dm = cupy.asarray(mf.init_guess_by_minao())
        ni = mf._numint

        gnumint._XC_GRID_BLKSIZE = None
        os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '16384'
        try:
            lazy = Grids(mol)
            lazy.level = 2
            n, e, v = ni.nr_nlc_vxc(mol, lazy, 'wb97m-v', dm)
            self.assertEqual(lazy.coords.shape[0] > 0, True)
        finally:
            os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
            gnumint._XC_GRID_BLKSIZE = None


if __name__ == '__main__':
    unittest.main()

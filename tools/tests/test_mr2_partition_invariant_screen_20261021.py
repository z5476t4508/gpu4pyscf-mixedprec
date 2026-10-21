#!/usr/bin/env python3
"""MR-2 (2026-10-21, named follow-up to the MR1 guard-repro) regression
tests: partition-invariant AO-screen construction.

The MR1 guard-repro measured an intermittent coarse-size screen-kernel
evaluation that dropped a shell a covered fine block keeps (3/40 fresh
processes on dopamine/b3lyp/def2-tzvp, 0/40 on the control arm; fail-closed
under MR1's guard, retry-clean).  MR-2 removes the coarse kernel evaluation
entirely: gen_grid.Grids._screen_shl_mask evaluates the kernel ONLY at
MIN_BLK_SIZE and derives any coarser mask as the exact boolean union of the
covered fine masks, so the two partitions cannot disagree by construction.

Red->green: test_kernel_only_called_at_default_size is RED pre-MR2 (a
GDFTscreen_index call with block_size=16384 occurs) and GREEN post-MR2.
The union/superset tests verify the derivation arithmetic on real grids;
the guard-liveness test proves the MR1 fail-closed defense is retained for
foreign/corrupted coarse caches; the cache-reset test proves the mask memo
dies with the screening cache (the MR1 stale-structure class).

These tests are GPU tests (a CUDA context is created); they need the
gpu4pyscf source env.  The epinephrine fixed-dm pair runs in ~15 s total on
the RTX 4090.  The separate preregistered firing-class elimination
measurement (2 arm commands x 40 fresh processes, prediction 0/40) lives in
the MR-2 evidence directory, not here.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    import pyscf  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy/pyscf)')

EPINEPHRINE_XYZ = """
O 1.7394 -2.1169 -1.0894
O -2.2941 1.0781 -1.7979
O -3.6584 0.5842 0.5722
N 2.6359 1.0243 0.7030
C 1.6787 -1.1447 -0.0373
C 0.2684 -0.6866 0.1208
C 2.6376 0.0193 -0.3576
C -0.3658 -0.0099 -0.9212
C -0.4164 -0.9343 1.3105
C -1.6849 0.4191 -0.7732
C -1.7353 -0.5053 1.4585
C -2.3696 0.1713 0.4166
C 3.5645 2.1013 0.3950
H 2.0210 -1.6511 0.8741
H 2.3808 0.4742 -1.3225
H 3.6478 -0.3931 -0.4831
H 0.1501 0.1801 -1.8589
H 0.0640 -1.4598 2.1315
H 2.9013 0.5888 1.5858
H -2.2571 -0.7050 2.3907
H 2.6646 -2.4067 -1.1652
H 3.2862 2.6124 -0.5325
H 4.5925 1.7346 0.3078
H 3.5401 2.8441 1.1985
H -3.2008 1.2997 -1.5231
H -3.9690 0.3259 1.4570
"""


def _scrub_gpu_env():
    for k in list(os.environ):
        if k.startswith('GPU4PYSCF_'):
            os.environ.pop(k)


class _EnvRestore:
    def __enter__(self):
        self.saved = {k: v for k, v in os.environ.items()
                      if k.startswith('GPU4PYSCF_')}
        _scrub_gpu_env()
        return self

    def __exit__(self, *exc):
        import gpu4pyscf.dft.numint as gnumint
        _scrub_gpu_env()
        os.environ.update(self.saved)
        gnumint._XC_GRID_BLKSIZE = None
        return False


def _built_grids(mol, level=3):
    from gpu4pyscf import dft as gpu_dft
    grids = gpu_dft.gen_grid.Grids(mol)
    grids.level = level
    grids.build(mol)
    return grids


def _opt(grids):
    import gpu4pyscf.dft.numint as gnumint
    return gnumint._GDFTOpt.from_mol(grids.mol)


def _shell_sets(grids, opt, blksize):
    '''per-block shell-index sets: build (if needed) then read the
    processed screen-cache entry -- (non0shl_idx, sections) -- directly.'''
    import numpy as np
    import gpu4pyscf.dft.numint as gnumint
    grids.get_non0ao_idx(opt, blksize=blksize)  # ensures the cache entry
    key = gnumint.MIN_BLK_SIZE if blksize is None else blksize
    _, (non0shl_idx, sections), _, _ = grids._non0ao_idx[key]
    idx = np.asarray(non0shl_idx)
    return [set(s.tolist()) for s in np.split(idx, np.asarray(sections))]


class TestDerivedScreenIsExactUnion(unittest.TestCase):
    '''The derived coarse screen must be the exact boolean union of the
    covered fine (default-partition) screens -- on real grids.'''

    def _check(self, mol, label):
        import gpu4pyscf.dft.numint as gnumint
        with _EnvRestore():
            grids = _built_grids(mol)
            opt = _opt(grids)
            fine = _shell_sets(grids, opt, blksize=None)  # MIN_BLK_SIZE
            MB = gnumint.MIN_BLK_SIZE
            for coarse_size in (8192, 16384, 32768):
                coarse = _shell_sets(grids, opt, blksize=coarse_size)
                ratio = coarse_size // MB
                self.assertEqual(len(coarse),
                                 (grids.size + coarse_size - 1) // coarse_size)
                for cb, cset in enumerate(coarse):
                    union = set()
                    for f in range(cb * ratio, min((cb + 1) * ratio,
                                                   len(fine))):
                        union |= fine[f]
                        # superset: the guard condition, now by construction
                        self.assertTrue(
                            fine[f] <= cset,
                            f'{label}: coarse {coarse_size} block {cb} '
                            f'missing shells of fine block {f}')
                    # exact union: the derivation adds nothing either
                    self.assertEqual(
                        cset, union,
                        f'{label}: coarse {coarse_size} block {cb} != '
                        f'union of covered fine screens')

    def test_small(self):
        self._check(pyscf.M(atom='H 0 0 0; F 0 0 0.95',
                            basis='def2-svp', verbose=0), 'HF')

    def test_epinephrine(self):
        self._check(pyscf.M(atom=EPINEPHRINE_XYZ, basis='def2-svp',
                            charge=0, spin=0, verbose=0), 'epinephrine')


class TestKernelOnlyAtDefaultSize(unittest.TestCase):
    '''RED pre-MR2 (a GDFTscreen_index call with block_size=16384 occurs),
    GREEN post-MR2: the kernel is invoked only ever at MIN_BLK_SIZE, no
    matter how many coarse partitions are built from the same grid.'''

    def test_no_coarse_kernel_calls(self):
        import gpu4pyscf.dft.gen_grid as ggen_grid
        import gpu4pyscf.dft.numint as gnumint
        with _EnvRestore():
            mol = pyscf.M(atom=EPINEPHRINE_XYZ, basis='def2-svp',
                          charge=0, spin=0, verbose=0)
            grids = _built_grids(mol)
            opt = _opt(grids)
            # gen_grid binds its own library object (load_library returns a
            # fresh instance per caller); shadow the symbol ON THAT OBJECT so
            # the counter sees exactly the calls gen_grid makes.
            orig = ggen_grid.libgdft.GDFTscreen_index
            calls = []

            def counting(*args):
                calls.append(int(args[5]))  # block_size positional
                return orig(*args)

            ggen_grid.libgdft.GDFTscreen_index = counting
            try:
                # a FRESH cache so the counters see every build
                grids._non0ao_idx = None
                _shell_sets(grids, opt, blksize=None)
                n_fine_calls = len(calls)
                _shell_sets(grids, opt, blksize=8192)
                _shell_sets(grids, opt, blksize=16384)
                _shell_sets(grids, opt, blksize=32768)
                grids.get_partition_fix(opt, blksize=16384)
            finally:
                del ggen_grid.libgdft.GDFTscreen_index
            self.assertEqual(n_fine_calls, 1)
            self.assertEqual(set(calls), {gnumint.MIN_BLK_SIZE},
                             f'kernel called at non-default sizes: {calls}')


class TestMaskMemoCache(unittest.TestCase):
    '''The raw-mask memo lives inside _non0ao_idx, so a screening-relevant
    external reset (AO_THRESHOLD change + grids._non0ao_idx = None, the
    MR1 stale-structure class) must rebuild it.'''

    def test_dies_with_screen_cache(self):
        import gpu4pyscf.dft.numint as gnumint
        mol = pyscf.M(atom='''
C 0.000 1.397 0
C 1.209 0.699 0
C 1.209 -0.699 0
C 0.000 -1.397 0
C -1.209 -0.699 0
C -1.209 0.699 0
H 0.000 2.481 0
H 2.147 1.241 0
H 2.147 -1.241 0
H 0.000 -2.481 0
H -2.147 -1.241 0
H -2.147 1.241 0
''', basis='def2-svp', verbose=0)
        with _EnvRestore():
            grids = _built_grids(mol)
            opt = _opt(grids)
            saved_thr = gnumint.AO_THRESHOLD
            try:
                n_default = sum(len(s) for s in
                                _shell_sets(grids, opt, blksize=16384))
                # the default screen must actually screen something here,
                # else the strict inequality below cannot discriminate
                n_coarse = len(_shell_sets(grids, opt, blksize=16384))
                nbas = len(mol.ao_loc_nr()) - 1
                self.assertLess(n_default, nbas * n_coarse)
                # screening-relevant change + external reset
                gnumint.AO_THRESHOLD = 1e-300
                grids._non0ao_idx = None
                n_unscreened = sum(len(s) for s in
                                   _shell_sets(grids, opt, blksize=16384))
            finally:
                gnumint.AO_THRESHOLD = saved_thr
                grids._non0ao_idx = None
        # the rebuilt mask reflects the new threshold: a stale memo would
        # have returned exactly the old n_default
        self.assertGreater(n_unscreened, n_default)


class TestGuardStillFailsClosed(unittest.TestCase):
    '''The MR1 fail-closed guard is retained as defense-in-depth: a foreign
    or corrupted coarse cache (a shell kept by a fine block missing from
    the coarse block) must raise, never run silently wrong.'''

    def test_corrupted_coarse_raises(self):
        import numpy as np
        with _EnvRestore():
            mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                          verbose=0)
            grids = _built_grids(mol)
            opt = _opt(grids)
            fine = _shell_sets(grids, opt, blksize=None)
            coarse = _shell_sets(grids, opt, blksize=16384)
            self.assertEqual(coarse[0], fine[0] | fine[1] | fine[2] | fine[3])
            # corrupt the CACHED coarse processed entry: remove one AO from
            # coarse block 0 (the guard reads the AO-idx lists).  Post-MR2
            # the coarse set is the union of the covered fine sets, so the
            # removed AO is kept by some fine block -> guard must fire.
            cache = grids._non0ao_idx
            entry = cache[16384]
            (idx, idx_sections), rest = entry[0], entry[1:]
            idx = np.asarray(idx)
            idx_sections = np.asarray(idx_sections)
            assert idx_sections[0] > 0  # block 0 non-empty (HF def2-svp)
            idx_corrupt = idx[1:]
            sections_corrupt = np.clip(idx_sections - 1, 0, None)
            cache[16384] = ((idx_corrupt, sections_corrupt),) + rest
            cache.pop(('partition_fix', 16384), None)
            try:
                with self.assertRaises(RuntimeError):
                    grids.get_partition_fix(opt, blksize=16384)
            finally:
                grids._non0ao_idx = None  # drop the corruption entirely


class TestBlocksizePhysicsInvariance(unittest.TestCase):
    '''MR-2 must not move the physics: the MR1 epinephrine fixed-dm floor
    holds through the derived screen (P3).'''

    def test_fixed_dm_rks_invariance(self):
        import gpu4pyscf.dft.numint as gnumint
        from gpu4pyscf import dft as gpu_dft
        mol = pyscf.M(atom=EPINEPHRINE_XYZ, basis='def2-svp',
                      charge=0, spin=0, verbose=0)
        with _EnvRestore():
            mf = gpu_dft.RKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.max_cycle = 50
            mf.direct_scf_tol = 1e-14
            mf.init_guess = 'minao'
            mf.grids.level = 3
            mf.grids.build(mol)
            dm = cp.asarray(mf.init_guess_by_minao())
            ni = mf._numint
            gnumint._XC_GRID_BLKSIZE = None
            nA, eA, vA = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
            gnumint._XC_GRID_BLKSIZE = 16384
            nB, eB, vB = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
        self.assertLessEqual(abs(float(nB) - float(nA)), 1e-11)
        self.assertLessEqual(abs(float(eB) - float(eA)), 1e-11)
        self.assertLessEqual(float(cp.abs(vA - vB).max()), 1e-9)


if __name__ == '__main__':
    unittest.main()

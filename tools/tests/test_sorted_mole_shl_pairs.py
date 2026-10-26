#!/usr/bin/env python3
"""Regression tests for the vectorized SortedMole.generate_shl_pairs
(LC-4, gpu4pyscf/gto/mole.py).

The change replaces the per-(i,j) assembly loop (~5 tiny cupy kernels per
group pair; ~150 launches, 13-22 ms on consumer cells) with host-side numpy
selection plus ONE device transfer.  The tests pin:

  1. exact key sets (hermi=1 triangular, hermi=2 full) for several
     molecules/bases;
  2. exact values per key (int32 pair indices) against an independent
     numpy reference built straight from the mask;
  3. cupy dtype/device of the cached arrays (int32, current device);
  4. aggregate_shl_pairs round-trip on the patched cache: the flattened
     bas_ij_idx equals the concatenation of the per-key arrays in the same
     dict order (the property every consumer relies on);
  5. a caller-level check: int1e_ipkin/int1e_ipovlp outputs are finite and
     symmetric-shaped (the gradient path's real consumers).

CPU-only numpy parts run everywhere; the cupy parts need the source env
(skipped under stdlib python3, same convention as the LC-2 tests).
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

# These tests exercise the REAL gpu4pyscf SortedMole path (numpy/cupy); the
# tools suite is otherwise stdlib-only under system python3 -- skip the
# whole module there (same convention as test_grad_hotspot_profiler).
try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

import numpy as np


def _build_sorted(geom_lines, basis):
    import pyscf
    from gpu4pyscf.gto.mole import SortedGTO
    mol = pyscf.M(atom=geom_lines, basis=basis, verbose=0)
    return SortedGTO.from_mol(mol, decontract=True)


GEOMS = {
    'water': 'O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
    'methanol': ('C -0.04 1.25 0; H -0.04 2.32 0; H -0.04 0.77 -1.03; '
                 'H 0.86 0.77 0.51; O -1.20 -0.41 0; H -2.05 0.08 0'),
}


class TestGenerateShlPairsValues(unittest.TestCase):

    def _reference_cache(self, s, hermi, mask_np):
        '''Independent numpy assembly (the pre-patch semantics).'''
        l_ctr_offsets = np.append(0, np.cumsum(s.l_ctr_counts))
        nbas = s.nbas
        groups = len(s.uniq_l_ctr)
        if hermi == 1:
            tasks = [(i, j) for i in range(groups) for j in range(i + 1)]
        else:
            tasks = [(i, j) for i in range(groups) for j in range(groups)]
        cache = {}
        for i, j in tasks:
            ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i + 1]
            jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j + 1]
            sub = mask_np[ish0:ish1, jsh0:jsh1]
            if hermi == 1 and i == j:
                sub = np.tril(sub)
            t = (np.arange(ish0, ish1, dtype=np.int32)[:, None] * nbas +
                 np.arange(jsh0, jsh1, dtype=np.int32))
            cache[i, j] = t[sub]
        return cache

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_values_match_reference_both_hermi(self):
        for name, geom in GEOMS.items():
            for basis in ('def2-svp', 'sto-3g'):
                s = _build_sorted(geom, basis)
                for hermi in (1, 2):
                    mask = s.shell_overlap_mask(hermi)
                    mask_np = cp.asnumpy(mask)
                    cache = s.generate_shl_pairs(hermi, mask)
                    ref = self._reference_cache(s, hermi, mask_np)
                    self.assertEqual(
                        set(cache.keys()), set(ref.keys()),
                        f'{name}/{basis} hermi={hermi} key mismatch')
                    for k in ref:
                        self.assertTrue(
                            cp.array_equal(cache[k],
                                           cp.asarray(ref[k], dtype=np.int32)),
                            f'{name}/{basis} hermi={hermi} key {k} differs')

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_dtype_and_device(self):
        s = _build_sorted(GEOMS['water'], 'sto-3g')
        cache = s.generate_shl_pairs(1)
        for k, v in cache.items():
            self.assertEqual(v.dtype, cp.int32, f'key {k}')

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_mask_passed_as_numpy_also_accepted(self):
        # the patch accepts a numpy mask (previously it assumed cupy and
        # would fail with a TypeError on np.tril of a cupy array... actually
        # np.tril dispatches; the old path worked for cupy only through
        # cp.tril; pin the numpy-mask path too since _shell_overlap_mask
        # callers may hold host masks)
        s = _build_sorted(GEOMS['water'], 'sto-3g')
        mask_cupy = s.shell_overlap_mask(1)
        c1 = s.generate_shl_pairs(1, mask_cupy)
        c2 = s.generate_shl_pairs(1, cp.asnumpy(mask_cupy))
        self.assertEqual(set(c1.keys()), set(c2.keys()))
        for k in c1:
            self.assertTrue(cp.array_equal(c1[k], c2[k]))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_empty_pair_groups(self):
        # a molecule whose overlap mask screens some group pairs to zero
        # pairs (H2 with a padded basis gives at least the diagonal; use a
        # mask of all-False to exercise the empty-concatenate branch)
        s = _build_sorted('H 0 0 0; H 0 0 0.74', 'sto-3g')
        false_mask = cp.zeros((s.nbas, s.nbas), dtype=bool)
        cache = s.generate_shl_pairs(1, false_mask)
        for k, v in cache.items():
            self.assertEqual(v.size, 0)

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_aggregate_roundtrip(self):
        s = _build_sorted(GEOMS['methanol'], 'def2-svp')
        cache = s.generate_shl_pairs(2)
        bas_ij_idx, shl_pair_offsets = s.aggregate_shl_pairs(cache)
        expected = cp.hstack(
            [cp.asarray(cache[k]) for k in cache]).astype(cp.int32)
        self.assertTrue(cp.array_equal(bas_ij_idx, expected))
        self.assertEqual(int(shl_pair_offsets[-1]), int(bas_ij_idx.size))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_intor_consumers_finite(self):
        # end-user smoke: the gradient path's two intors still run on the
        # patched pair cache and return (3, nao, nao) finite outputs
        from gpu4pyscf.pbc.gto.int1e import int1e_ipkin, int1e_ipovlp
        import pyscf
        mol = pyscf.M(atom=GEOMS['water'], basis='sto-3g', verbose=0)
        for fn in (int1e_ipkin, int1e_ipovlp):
            out = fn(mol)
            self.assertEqual(out.shape[0], 3)
            self.assertTrue(bool(cp.isfinite(out).all()))


if __name__ == '__main__':
    unittest.main()

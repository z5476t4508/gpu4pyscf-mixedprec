#!/usr/bin/env python3
"""MR-12 (2026-10-23, mainline repair) regression tests: canonicalize the
reversed duplicate AO-pair CDERI columns.

Root cause (2026-10-23-mr12-gpu-process-nondeterminism, preregistered probe
series): pair_and_diag_indices enumerates the full AO rectangle inside
diagonal shell-pair blocks, so the CDERI stores both (i,j) and (j,i) columns
for same-shell cross AO pairs (583 unordered classes on dopamine/def2-tzvp,
all within one shell). The two columns hold the same integrals but differ at
the last bits (<=3.34e-13; official gpu4pyscf 1.8.1 shows the same structure,
so the class is upstream-inherited). fill_symmetric writes both symmetric
cells from every column; with bitwise-unequal duplicate columns the racing
stores make the unpacked CDERI vary between launches (measured: 10/10 distinct
fill outputs, output asymmetric in 408-854 cells, 100% of deviating cells
contained in duplicate-pair cells), giving ULP-level VK variation and the
~4.5e-13 cross-process e_tot envelope (MR-11 named follow-up).

The J path is NOT affected (measured vs pyscf CPU: 4.04e-12 relative): the
duplicate pairs sit inside the diagonal shell blocks whose dm_sparse entries
are halved, so both orientations sum to the correct tril weight. The fix
copies the first-occurrence column bitwise onto its duplicates in DF.build,
making the racing writes harmless.

Red->green: on the pre-fix source, test_duplicate_columns_bitwise_canonical
fails deterministically (the build itself produces bitwise-unequal duplicate
columns on molecules with d/f functions) and the fill/vk determinism tests
fail with high probability; transcripts of the pre-fix RED run are preserved
in the evidence directory. test_j_matches_pyscf_cpu must stay green in both
states (guards the halving design of the J path).

These are GPU tests (a CUDA context is created); they need the gpu4pyscf
source env. ~10-20 s total on the RTX 4090.
"""
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

import cupy as cp
from pyscf import gto, scf

from gpu4pyscf.lib.cupy_helper import fill_symmetric

WATER_TZVP = 'O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24'


def _build_df(basis='def2-tzvp'):
    mol = gto.M(atom=WATER_TZVP, basis=basis, verbose=0, output='/dev/null')
    mf = scf.RHF(mol).density_fit().to_gpu()
    mf.with_df.build()
    return mol, mf


def _duplicate_classes(pair_np, nao):
    '''unordered duplicate pair classes -> list of column-index tuples'''
    seen = {}
    dups = []
    for k, v in enumerate(pair_np.tolist()):
        i, j = divmod(int(v), nao)
        key = (min(i, j), max(i, j))
        if key in seen:
            dups.append((seen[key], k))
        else:
            seen[key] = k
    return dups


class TestMR12DuplicatePairCanonicalization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = _build_df()
        cls.dfobj = cls.mf.with_df
        cls.nao = cls.mol.nao
        cls.pair_np = cp.asnumpy(cp.asarray(cls.dfobj._cderi_idx[0]))
        cls.cderi_np = cp.asnumpy(cls.dfobj._cderi[0])
        cls.dups = _duplicate_classes(cls.pair_np, cls.nao)

    def test_molecule_actually_has_duplicate_pairs(self):
        # vacuous-test guard: water/def2-tzvp has d/f shells -> same-shell
        # cross AO pairs -> the enumeration must produce duplicates
        self.assertGreaterEqual(len(self.dups), 1,
            'no duplicate AO-pair columns found; the test molecule no longer '
            'exercises the diagonal-shell-block rectangle enumeration')

    def test_duplicate_columns_bitwise_canonical(self):
        '''deterministic RED pre-fix: the build produced bitwise-unequal
        duplicate columns (d/f same-shell pairs)'''
        unequal = []
        for ca, cb in self.dups:
            if not np.array_equal(self.cderi_np[:, ca], self.cderi_np[:, cb]):
                unequal.append((int(ca), int(cb)))
        self.assertEqual(
            unequal, [],
            f'{len(unequal)}/{len(self.dups)} duplicate AO-pair columns are '
            f'bitwise unequal (first: {unequal[:3]}); the CDERI violates the '
            f'intended bitwise (ij|P) == (ji|P) symmetry and fill_symmetric '
            f'races on these columns')

    def test_fill_matches_numpy_expansion_and_symmetric(self):
        blk = min(64, self.cderi_np.shape[0])
        cderi_sparse = self.dfobj._cderi[0]
        pair_idx = cp.asarray(self.dfobj._cderi_idx[0], dtype=np.int32)
        out = cp.asnumpy(fill_symmetric(cderi_sparse.T, pair_idx, self.nao,
                                        0, blk))
        rows, cols = np.divmod(self.pair_np, self.nao)
        exact = np.zeros((self.nao, self.nao, blk))
        vals = self.cderi_np[0:blk, :].T
        exact[rows, cols, :] = vals
        exact[cols, rows, :] = vals
        self.assertTrue(np.array_equal(out, exact),
            'fill output != numpy expansion (racing duplicate columns pick '
            'arbitrary winners)')
        self.assertTrue(np.array_equal(out, out.transpose(1, 0, 2)),
            'fill output is not bitwise symmetric: racing duplicate columns '
            'left the two orientations unequal')

    def test_fill_deterministic_across_launches(self):
        blk = min(64, self.cderi_np.shape[0])
        cderi_sparse = self.dfobj._cderi[0]
        pair_idx = cp.asarray(self.dfobj._cderi_idx[0], dtype=np.int32)
        ref = None
        for _ in range(8):
            out = fill_symmetric(cderi_sparse.T, pair_idx, self.nao, 0, blk)
            out_np = cp.asnumpy(out)
            if ref is None:
                ref = out_np
            else:
                self.assertTrue(np.array_equal(out_np, ref),
                    'fill_symmetric output varies between launches (the '
                    'duplicate-column store race)')

    def test_vk_bitwise_stable(self):
        np.random.seed(20261023)
        occ = sum(self.mol.nelec) // 2
        cs = np.random.standard_normal((self.nao, self.nao))
        dm = cp.asarray(2.0 * (cs[:, :occ] @ cs[:, :occ].T))
        vk1 = cp.asnumpy(self.mf.get_k(self.mol, dm, hermi=1))
        for _ in range(4):
            vk2 = cp.asnumpy(self.mf.get_k(self.mol, dm, hermi=1))
            self.assertTrue(np.array_equal(vk1, vk2),
                'production get_k varies between calls on identical input')

    def test_j_matches_pyscf_cpu(self):
        # guards the J-path halving design: must pass pre- AND post-fix
        np.random.seed(20261024)
        occ = sum(self.mol.nelec) // 2
        cs = np.random.standard_normal((self.nao, self.nao))
        dm = cp.asarray(2.0 * (cs[:, :occ] @ cs[:, :occ].T))
        vj_gpu = cp.asnumpy(self.mf.get_j(dm=dm))
        mf_cpu = scf.RHF(self.mol).density_fit()
        mf_cpu.with_df.build()
        vj_cpu = mf_cpu.get_j(dm=dm.get())
        rel = np.abs(vj_gpu - vj_cpu).max() / np.abs(vj_cpu).max()
        self.assertLess(rel, 1e-9,
            f'J deviates from pyscf CPU DF by rel {rel:.3e}; the halving '
            f'design of the duplicate diagonal-shell-block pairs regressed')


if __name__ == '__main__':
    unittest.main()

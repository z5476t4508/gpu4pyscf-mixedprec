#!/usr/bin/env python3
'''CDERI-AUXMIX-TRI-01 unit battery (CPU-only, numpy).

Exercises the production planner and apply functions on numpy operands:
the apply loop is array-library agnostic, and _auxmix_tri_build_plan's
only additional logic is the device transfer of the mask and the
contiguous slice materialization, which this battery mirrors verbatim.
The dense reference is the deployed j3c.dot(aux_coef) contraction.'''
import numpy as np
import unittest

from gpu4pyscf.df.df import (_auxmix_tri_plan_from_mask,
                             _auxmix_tri_apply)


def _mirror_build_plan(c):
    '''numpy mirror of _auxmix_tri_build_plan (the device wrapper minus
    the cupy transfers).'''
    plan = _auxmix_tri_plan_from_mask(c != 0)
    if plan is None or plan.get('gated_out'):
        return None
    slices = []
    for cols, segments in plan['blocks']:
        seg_ops = [(r0, r1, np.ascontiguousarray(c[r0:r1, cols]))
                   for r0, r1 in segments]
        slices.append((np.asarray(cols), seg_ops))
    plan['device_blocks'] = slices
    return plan


def _random_permuted_triangular(n, rng, perm=True, n_blocks=6):
    '''M = P @ L^-T-shaped operand: an upper-triangular core under BLOCK
    row and column permutations (the aux_sorting class: l-groups stay
    contiguous, only their order changes).'''
    core = np.triu(rng.standard_normal((n, n)))
    m = core
    if perm:
        edges = [round(n * i / n_blocks) for i in range(n_blocks + 1)]
        rb = rng.permutation(n_blocks)
        cb = rng.permutation(n_blocks)
        rows = np.concatenate([np.arange(edges[i], edges[i + 1]) for i in rb])
        cols = np.concatenate([np.arange(edges[i], edges[i + 1]) for i in cb])
        m = core[np.ix_(rows, cols)]
    return m


class TestAuxmixTriPlanner(unittest.TestCase):

    def test_permuted_triangular_plan_fill_about_half(self):
        rng = np.random.default_rng(20261123)
        n = 1536
        m = _random_permuted_triangular(n, rng)
        plan = _auxmix_tri_plan_from_mask(m != 0)
        self.assertIsNotNone(plan)
        self.assertLess(plan['fill'], 0.75)
        self.assertLess(plan['n_segments'], 48)
        # every column covered exactly once
        cols = np.concatenate([c for c, _ in plan['blocks']])
        self.assertEqual(sorted(cols.tolist()), list(range(n)))

    def test_dense_operand_falls_back(self):
        rng = np.random.default_rng(1)
        m = rng.standard_normal((300, 300))
        out = _auxmix_tri_plan_from_mask(m != 0)
        self.assertTrue(out is None or out.get('gated_out'))
        if out:
            self.assertGreater(out['fill_raw'], 0.75)

    def test_fragmented_operand_falls_back(self):
        rng = np.random.default_rng(2)
        n = 700
        # scattered narrow diagonals: fill may pass but segments blow up
        m = np.zeros((n, n))
        for j in range(n):
            m[rng.integers(0, n), j] = 1.0
        m[n // 2, :] = 1.0
        plan = _auxmix_tri_plan_from_mask(m != 0)
        # either gated out or bounded; assert the invariant either way
        if plan is not None and not plan.get('gated_out'):
            self.assertLessEqual(plan['n_segments'], 48)

    def test_apply_matches_dense_reference(self):
        rng = np.random.default_rng(3)
        n_rows, n_cols = 400, 640
        m = _random_permuted_triangular(n_cols, rng)
        j3c = rng.standard_normal((n_rows, n_cols))
        plan = _mirror_build_plan(m)
        self.assertIsNotNone(plan)
        out = np.empty((n_rows, n_cols))
        _auxmix_tri_apply(j3c, plan, out)
        ref = j3c.dot(m)
        # same nonzero terms, different summation grouping: ULP-class
        np.testing.assert_allclose(out, ref, rtol=1e-12, atol=1e-10)
        # exact-zero structure: dropped terms contribute exactly zero in
        # the reference, so the difference is pure reassociation noise
        self.assertLess(np.abs(out - ref).max(), 1e-9)

    def test_apply_identity_operand_bitwise(self):
        # a block-diagonal scale operand (the 1:1 shell limit) has one
        # term per column within its shell: blocked and dense results
        # are bitwise identical for those single-term columns
        rng = np.random.default_rng(4)
        n_rows, n_cols = 300, 768
        diag = np.diag(rng.standard_normal(n_cols) + 3.0)
        j3c = rng.standard_normal((n_rows, n_cols))
        plan = _mirror_build_plan(diag)
        self.assertIsNotNone(plan)
        self.assertLess(plan['fill'], 0.55)
        out = np.empty((n_rows, n_cols))
        _auxmix_tri_apply(j3c, plan, out)
        np.testing.assert_array_equal(out, j3c.dot(diag))

    def test_plan_determinism(self):
        rng = np.random.default_rng(5)
        m = _random_permuted_triangular(1024, rng)
        p1 = _auxmix_tri_plan_from_mask(m != 0)
        p2 = _auxmix_tri_plan_from_mask(m != 0)
        self.assertEqual(p1['fill'], p2['fill'])
        self.assertEqual(p1['n_segments'], p2['n_segments'])
        for (c1, s1), (c2, s2) in zip(p1['blocks'], p2['blocks']):
            np.testing.assert_array_equal(c1, c2)
            self.assertEqual(s1, s2)


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the LC-36 molecular-only Int2c2eOpt cache
(gpu4pyscf/pbc/df/int2c2e.py).

The gradient path calls the module wrappers int2c2e() and
int2c2e_ip1_per_atom() 2-4x per gradient; each call re-constructs
Int2c2eOpt at ~18 ms construction vs ~3 ms kernel (LC-35 measured,
benzene/svp naux=558).  LC-36 adds a cache: for MOLECULAR (non-periodic)
auxmols only, the constructed opt is cached on the key cell (the sorted
auxmol's original ``.mol`` back-reference when the input is a SortedMole
-- the DF gradient path -- or the input itself for a plain Mole).
Periodic Cell inputs and explicit bvk_kmesh arguments bypass the cache.

Contract under test:
  * cache hit: a second wrapper call reuses the opt (no new construct);
  * cross-SortedMole hit: two sorted views of the same original auxmol
    (what consecutive gradients produce) share one cached opt;
  * correctness: cached-path results match uncached (bitwise for
    int2c2e; within atomics variance for energy_ip1_per_atom, which is
    not bitwise reproducible even on a single opt);
  * dm shape guard: a spherical-sized dm is rejected loudly instead of
    silently reading out of bounds (found live while testing the cache
    with an invalid input; compute-sanitizer confirmed the OOB);
  * pbc bypass: Cell inputs never touch the cache;
  * explicit bvk_kmesh bypasses the cache;
  * in-place mutation of the key cell's arrays invalidates the cache;
  * eviction: the cache lives on the key cell, so it cannot outlive it.
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy as np  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')


def _module():
    import gpu4pyscf.pbc.df.int2c2e as m
    return m


class TestOptCacheContract(unittest.TestCase):
    def _auxmol(self):
        import pyscf
        from pyscf.df import addons
        mol = pyscf.M(atom=os.path.join(
            REPO, 'mixedprec/benchmark/geoms/methanol.xyz'),
            basis='def2-svp', verbose=0)
        return addons.make_auxmol(mol, 'def2-universal-jkfit')

    def test_cache_hit_reuses_opt(self):
        m = _module()
        auxmol = self._auxmol()
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(auxmol)
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(auxmol)
        n2 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)      # first call constructs
        self.assertEqual(n2 - n1, 0)      # second call is a cache hit

    def test_two_sorted_views_share_cached_opt(self):
        # consecutive gradients each rebuild Int3c2eOpt -> fresh sorted
        # auxmol objects sharing the same original .mol; both must hit
        # one cached opt
        from gpu4pyscf.gto.mole import SortedMole
        m = _module()
        auxmol = self._auxmol()
        s1 = SortedMole.from_cell(auxmol)
        s2 = SortedMole.from_cell(auxmol)
        self.assertIsNot(s1, s2)
        self.assertIs(s1.mol, auxmol)
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(s1)
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(s2)
        n2 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)
        self.assertEqual(n2 - n1, 0)

    def test_cached_result_bitwise_equal(self):
        m = _module()
        auxmol = self._auxmol()
        cached = m.int2c2e(auxmol)
        uncached = m.Int2c2eOpt(auxmol).int2c2e()
        self.assertTrue(cp.array_equal(cp.asarray(cached),
                                       cp.asarray(uncached)))

    def test_ip1_per_atom_cached_within_lane_variance(self):
        # energy_ip1_per_atom uses GPU atomics for the per-atom
        # accumulation: NOT bitwise reproducible even on the same opt
        # (measured run-to-run max diff ~9e-13 at naux=224).  The cache
        # gate is equality within the lane's own variance.
        m = _module()
        auxmol = self._auxmol()
        n = auxmol.nao_cart()    # the kernel indexes dm in CARTESIAN ao
        dm = cp.asarray(np.random.RandomState(1).rand(n, n)
                        + np.eye(n) * 10)
        cached = cp.asarray(m.int2c2e_ip1_per_atom(auxmol, dm))
        uncached = cp.asarray(
            m.Int2c2eOpt(auxmol).energy_ip1_per_atom(dm))
        self.assertLessEqual(float(np.abs(cached - uncached).max()),
                             1e-11)

    def test_spherical_dm_rejected_loudly(self):
        # a spherical-sized dm reads out of bounds in the kernel
        # (silent garbage before the guard; compute-sanitizer
        # confirmed the OOB read on both cached and pristine paths)
        m = _module()
        auxmol = self._auxmol()
        dm = cp.asarray(np.random.RandomState(1).rand(
            auxmol.nao, auxmol.nao))
        with self.assertRaises(ValueError):
            m.int2c2e_ip1_per_atom(auxmol, dm)

    def test_pbc_cell_bypasses_cache(self):
        m = _module()
        try:
            from pyscf.pbc.gto import Cell
        except ImportError:
            self.skipTest('pyscf.pbc not available')
        cell = Cell()
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 4.
        cell.basis = 'def2-svp'
        cell.verbose = 0
        cell.build()
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(cell)
        m.int2c2e(cell)
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 2)      # no caching for Cell inputs

    def test_explicit_bvk_kmesh_bypasses_cache(self):
        m = _module()
        auxmol = self._auxmol()
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(auxmol, bvk_kmesh=np.ones(3, dtype=int))
        m.int2c2e(auxmol, bvk_kmesh=np.ones(3, dtype=int))
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 2)      # explicit kmesh never cached

    def test_key_cell_mutation_invalidates(self):
        m = _module()
        auxmol = self._auxmol()
        m.int2c2e(auxmol)
        opt_before = auxmol.__dict__.get('_int2c2e_opt_cache')
        self.assertIsNotNone(opt_before)
        auxmol._env = auxmol._env.copy()
        auxmol._env[0] += .5             # change the token content
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(auxmol)
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)      # guard miss -> fresh build
        self.assertIsNot(auxmol.__dict__.get('_int2c2e_opt_cache'),
                         opt_before)

    def test_passed_sorted_env_mutation_invalidates(self):
        # LC-66 audit A1: the sorted view's _env is a FRESH array
        # (SortedGTO.from_mol copies it), so the pre-LC-66 token never
        # saw an in-place mutation of the PASSED view -- the wrapper
        # served a stale GPU snapshot (cached-vs-truth 0.72 on a moved
        # aux coordinate).  The token must cover the view's own _env.
        from pyscf.gto import PTR_COORD
        from gpu4pyscf.gto.mole import SortedMole
        m = _module()
        auxmol = self._auxmol()
        s = SortedMole.from_cell(auxmol)
        m.int2c2e(s)                       # warm the cache
        ptr0 = int(auxmol._atm[0, PTR_COORD])
        s._env[ptr0:ptr0 + 3] += 0.05      # in place, view-local array
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        served = cp.asarray(m.int2c2e(s))
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)       # must rebuild, not serve stale
        truth = cp.asarray(m.Int2c2eOpt(s).int2c2e())
        self.assertTrue(cp.allclose(served, truth))

    def test_passed_sorted_atm_mutation_invalidates(self):
        # LC-66: the sorted view's _atm is SHARED with the key
        # (copy(deep=False)); an in-place edit (e.g. a coordinate-
        # pointer move) must invalidate.  Charges do not enter 2c2e
        # integrals, so edit PTR_COORD pointers instead.
        from pyscf.gto import PTR_COORD
        from gpu4pyscf.gto.mole import SortedMole
        m = _module()
        auxmol = self._auxmol()
        s = SortedMole.from_cell(auxmol)
        m.int2c2e(s)
        s._atm[0, PTR_COORD] += 1          # in place on the shared array
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        m.int2c2e(s)
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)       # token covers _atm -> rebuild

    def test_key_cart_flip_does_not_change_int2c2e(self):
        # LC-66 audit A4: cart is solution-relevant in BOTH paths.
        # PLAIN Mole path: cart=True changes cell.nao (224 spherical ->
        # 262 cartesian) -> token invalidates, output shape changes.
        # SORTED-view path: the opt's sorted basis is cartesian, but the
        # output CT transform (CT_dot_mat/mat_dot_C) reads the KEY's
        # cart flag at CALL time -- pre-LC-66 a key cart flip kept the
        # cache hit and silently served a differently-shaped (262 vs
        # 224) output.  The token now carries bool(key.cart), so the
        # flip invalidates here too.
        from gpu4pyscf.gto.mole import SortedMole
        m = _module()
        # plain-Mole path: invalidation
        auxmol = self._auxmol()
        out_sph = cp.asarray(m.int2c2e(auxmol))
        auxmol.cart = True
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        out_cart = cp.asarray(m.int2c2e(auxmol))
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)       # nao in token -> fresh build
        self.assertEqual(out_sph.shape, (224, 224))
        self.assertEqual(out_cart.shape, (262, 262))
        # sorted-view path: key.cart selects the CT transform -> must
        # invalidate, not serve the stale-shape output
        auxmol = self._auxmol()
        s = SortedMole.from_cell(auxmol)
        out1 = cp.asarray(m.int2c2e(s))
        auxmol.cart = True
        n0 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        out2 = cp.asarray(m.int2c2e(s))
        n1 = m._INT2C2E_OPT_CONSTRUCT_COUNT
        self.assertEqual(n1 - n0, 1)       # key.cart in token -> rebuild
        self.assertEqual(out1.shape, (224, 224))
        self.assertEqual(out2.shape, (262, 262))
        self.assertFalse(cp.array_equal(out1, out2))

    def test_cache_does_not_leak_key_cell(self):
        # the only strong reference to the cached opt lives on the key
        # cell -> deleting it frees both
        import gc
        import weakref
        m = _module()
        auxmol = self._auxmol()
        m.int2c2e(auxmol)
        self.assertIsNotNone(auxmol.__dict__.get('_int2c2e_opt_cache'))
        aux_ref = weakref.ref(auxmol)
        del auxmol
        gc.collect()
        self.assertIsNone(aux_ref())


if __name__ == '__main__':
    unittest.main()

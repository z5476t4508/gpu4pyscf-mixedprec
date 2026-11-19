#!/usr/bin/env python3
"""Regression tests for _screen_bas_pairs_qcond (LC-6,
gpu4pyscf/df/int3c2e_bdiv.py) and the Int3c2eOpt.build switch.

The change replaces, inside Int3c2eOpt.build only, the
_cache_q_cond_and_non0pairs call (whose _group_by_split_points tail
reorders pairs by descending q-cond for the JK engines' early-termination
screening -- ordering that build immediately discards via cp.sort) with a
lean screen: host-side pair assembly in PBCsort_pair_ij's exact tiled
order, the same int2e_qcond_estimator kernel call, and the equivalent
threshold mask q > log(precision).  The tests pin:

  1. bit-identical bas_ij_cache (keys, values, order) vs the reference
     _cache_q_cond_and_non0pairs + iter_pair_by_l + cp.sort path across
     molecules/bases, tril True/False, and default cutoffs;
  2. h-function (generally-contracted) input: group exclusion for
     l > LMAX behaves identically on both paths;
  3. _sort_pair_ij_order reproduces the device kernel's enumeration
     order exactly (against a python transcription of the CUDA loop,
     including partial tiles);
  4. determinism across repeated builds;
  5. gradient-path consumer smoke: _grad_nuc_without_ecp and a full
     RKS gradient produce finite results; gradient bit-equivalence vs
     the reference build path is pinned by the measured matrix, not here
     (CPU-runnable subset only in the tools suite);
  6. empty/degenerate group handling (single shell pair).

The reference path (scf.jk._cache_q_cond_and_non0pairs) is imported and
exercised directly, so any drift in either implementation fails loudly.

Needs the gpu4pyscf source env (numpy/cupy); skipped under stdlib python3
(same convention as test_sorted_mole_shl_pairs).
"""
import os
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

import numpy as np
import cupy as cp

from pyscf import gto

from gpu4pyscf.gto.mole import SortedMole, RysIntEnvVars
from gpu4pyscf.df.int3c2e_bdiv import (
    Int3c2eOpt, conc_env, _conc_locs, _screen_bas_pairs_qcond,
    _sort_pair_ij_order)
from gpu4pyscf.scf.jk import (
    _cache_q_cond_and_non0pairs, _scale_sp_ctr_coeff)

GEOMS = {
    'water': 'O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24',
    'methanol': ('C -0.04 1.25 0; H -0.04 2.32 0; H -0.04 0.77 -1.03; '
                 'H 0.86 0.77 0.51; O -1.20 -0.41 0; H -2.05 0.08 0'),
}
HFUNC_BASIS = {'H': [[0, [1.0, 1.0]], [5, [2.0, 1.0]]]}


def _make_envs(geom, basis):
    mol = gto.M(atom=geom, basis=basis, verbose=0)
    smol = SortedMole.from_mol(mol, decontract=True)
    fakemol = gto.fakemol_for_charges(mol.atom_coords())
    _atm, _bas, _env = conc_env(
        smol._atm, smol._bas, _scale_sp_ctr_coeff(smol),
        fakemol._atm, fakemol._bas, _scale_sp_ctr_coeff(fakemol))
    envs = RysIntEnvVars.new(
        smol.natm, smol.nbas, _atm, _bas, _env,
        _conc_locs(smol.ao_loc, fakemol.ao_loc))
    return mol, smol, fakemol, envs


def _reference_cache(smol, envs, cutoff=1e-14, tril=True):
    raw = _cache_q_cond_and_non0pairs(smol, envs, cutoff, tril=tril)
    return {k: cp.sort(v[0]) for k, v in smol.iter_pair_by_l(raw)}


def _new_cache(smol, envs, cutoff=1e-14, tril=True):
    raw = _screen_bas_pairs_qcond(smol, envs, cutoff, tril=tril)
    return {k: cp.sort(v) for k, v in smol.iter_pair_by_l(raw)}


def _assert_cache_equal(tc, ref, new):
    tc.assertEqual(set(ref.keys()), set(new.keys()))
    for k in ref:
        a, b = cp.asnumpy(ref[k]), cp.asnumpy(new[k])
        tc.assertEqual(a.shape, b.shape, f'group {k} shape differs')
        tc.assertTrue(np.array_equal(a, b), f'group {k} values differ')


class TestScreenBasPairsBitIdentity(unittest.TestCase):
    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_water_svp_tril(self):
        _, smol, _, envs = _make_envs(GEOMS['water'], 'def2-svp')
        _assert_cache_equal(self, _reference_cache(smol, envs),
                            _new_cache(smol, envs))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_methanol_svp_tril(self):
        _, smol, _, envs = _make_envs(GEOMS['methanol'], 'def2-svp')
        _assert_cache_equal(self, _reference_cache(smol, envs),
                            _new_cache(smol, envs))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_methanol_svp_notril(self):
        _, smol, _, envs = _make_envs(GEOMS['methanol'], 'def2-svp')
        _assert_cache_equal(self, _reference_cache(smol, envs, tril=False),
                            _new_cache(smol, envs, tril=False))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_water_sto3g(self):
        _, smol, _, envs = _make_envs(GEOMS['water'], 'sto-3g')
        _assert_cache_equal(self, _reference_cache(smol, envs),
                            _new_cache(smol, envs))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_loose_cutoff(self):
        # a much looser threshold screens more pairs; the equivalence of
        # the keep predicate must hold at any cutoff
        _, smol, _, envs = _make_envs(GEOMS['water'], 'def2-svp')
        _assert_cache_equal(self, _reference_cache(smol, envs, 1e-8),
                            _new_cache(smol, envs, 1e-8))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_hfunc_lmax_exclusion(self):
        # generally-contracted H basis with an l=5 component: groups with
        # l > LMAX are excluded from pair_keys on BOTH paths
        _, smol, _, envs = _make_envs(GEOMS['water'], HFUNC_BASIS)
        _assert_cache_equal(self, _reference_cache(smol, envs),
                            _new_cache(smol, envs))
        uniq_l = smol.uniq_l_ctr[:, 0]
        self.assertTrue((uniq_l > 4).any(), 'test needs an l>LMAX group')

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_determinism(self):
        _, smol, _, envs = _make_envs(GEOMS['methanol'], 'def2-svp')
        c1 = _new_cache(smol, envs)
        c2 = _new_cache(smol, envs)
        _assert_cache_equal(self, c1, c2)


class TestSortPairOrder(unittest.TestCase):
    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_order_matches_kernel_transcription(self):
        # python transcription of sort_pair_ij_kernel
        # (lib/pbc/supmol_sr_estimator.cu) for arbitrary partial tiles
        def kernel_order(nish, njsh, nbas, tile=4):
            out = []
            for i_tile in range((nish + tile - 1) // tile):
                nish_rem = min(tile, nish - i_tile * tile)
                tile2 = nish_rem * tile
                for n in range(nish_rem * njsh):
                    j_tile = n // tile2
                    ijr = n - j_tile * tile2
                    ir = ijr // tile
                    jr = ijr - ir * tile
                    njsh_rem = njsh - j_tile * tile
                    if njsh_rem < tile:
                        ir = ijr // njsh_rem
                        jr = ijr - njsh_rem * ir
                    out.append(i_tile * tile + ir)  # i index (local)
            return out
        # The kernel enumerates (i, j) pairs; transcribe the full pair
        # index for a set of sizes covering partial i and j tiles.
        for nish, njsh in [(1, 1), (3, 5), (4, 4), (5, 3), (7, 9),
                           (8, 12), (13, 4), (10, 10)]:
            nbas = 20
            host = _sort_pair_ij_order(0, nish, 0, njsh, nbas, tile=4)
            # transcribe kernel exactly
            ref = []
            for i_tile in range((nish + 3) // 4):
                nish_rem = min(4, nish - i_tile * 4)
                tile2 = nish_rem * 4
                for n in range(nish_rem * njsh):
                    j_tile = n // tile2
                    ijr = n - j_tile * tile2
                    ir = ijr // 4
                    jr = ijr - ir * 4
                    njsh_rem = njsh - j_tile * 4
                    if njsh_rem < 4:
                        ir = ijr // njsh_rem
                        jr = ijr - njsh_rem * ir
                    i = i_tile * 4 + ir
                    j = j_tile * 4 + jr
                    if i >= nish:
                        break
                    ref.append(i * nbas + j)
            self.assertTrue(np.array_equal(
                host, np.asarray(ref, dtype=np.int64)),
                f'order mismatch at nish={nish}, njsh={njsh}')

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_identity_vs_device_kernel(self):
        # direct device check: run PBCsort_pair_ij on a group and compare
        import ctypes
        from gpu4pyscf.pbc.scf.rsjk import libpbc
        for nish, njsh, nbas in [(3, 5, 17), (7, 9, 23), (4, 4, 30)]:
            ish = cp.arange(nish, dtype=cp.uint32)
            jsh = cp.arange(njsh, dtype=cp.uint32)
            buf = cp.empty(nish * njsh, dtype=cp.int64)
            kern = libpbc.PBCsort_pair_ij
            kern.restype = ctypes.c_int
            err = kern(
                ctypes.cast(buf.data.ptr, ctypes.c_void_p),
                ctypes.cast(ish.data.ptr, ctypes.c_void_p),
                ctypes.cast(jsh.data.ptr, ctypes.c_void_p),
                ctypes.c_int(nish), ctypes.c_int(njsh),
                ctypes.c_int(nbas), ctypes.c_int(4))
            self.assertEqual(err, 0)
            dev = cp.asnumpy(buf)
            host = _sort_pair_ij_order(0, nish, 0, njsh, nbas, tile=4)
            self.assertTrue(np.array_equal(dev, host),
                            f'device/host order mismatch at {nish}x{njsh}')


class TestInt3c2eOptBuild(unittest.TestCase):
    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_build_matches_reference(self):
        mol, smol, fakemol, envs = _make_envs(GEOMS['methanol'], 'def2-svp')
        ref = _reference_cache(smol, envs)
        opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                         fakemol).build()
        _assert_cache_equal(self, ref, dict(opt.bas_ij_cache))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_build_dtypes(self):
        mol, _, fakemol, _ = _make_envs(GEOMS['water'], 'def2-svp')
        opt = Int3c2eOpt(
            SortedMole.from_mol(mol, decontract=True), fakemol).build()
        for k, v in opt.bas_ij_cache.items():
            self.assertEqual(v.dtype, cp.uint32, f'group {k} dtype')
        # iter_pair_by_l ordering: keys sorted by li*8+lj (angular
        # momenta from uniq_l_ctr, not group indices)
        smol = SortedMole.from_mol(mol, decontract=True)
        uniq_l = smol.uniq_l_ctr[:, 0]
        keys = list(opt.bas_ij_cache.keys())
        weights = [uniq_l[i] * 8 + uniq_l[j] for i, j in keys]
        self.assertEqual(weights, sorted(weights))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_single_shell_pair_mol(self):
        # minimal mol: one atom, one shell -> single (0,0) group, tests
        # degenerate tile boundaries (nish=njsh=1)
        mol = gto.M(atom='H 0 0 0', basis='sto-3g', spin=1, verbose=0)
        smol = SortedMole.from_mol(mol, decontract=True)
        fakemol = gto.fakemol_for_charges(mol.atom_coords())
        _atm, _bas, _env = conc_env(
            smol._atm, smol._bas, _scale_sp_ctr_coeff(smol),
            fakemol._atm, fakemol._bas, _scale_sp_ctr_coeff(fakemol))
        envs = RysIntEnvVars.new(
            smol.natm, smol.nbas, _atm, _bas, _env,
            _conc_locs(smol.ao_loc, fakemol.ao_loc))
        _assert_cache_equal(self, _reference_cache(smol, envs),
                            _new_cache(smol, envs))
        opt = Int3c2eOpt(SortedMole.from_mol(mol, decontract=True),
                         fakemol).build()
        self.assertTrue(all(cp.asnumpy(v).size >= 0
                            for v in opt.bas_ij_cache.values()))

    @unittest.skipUnless(HAS_GPU, 'needs the gpu4pyscf source env')
    def test_gradient_consumer_smoke(self):
        # the real consumer: _grad_nuc_without_ecp on a converged RKS
        sys.path.insert(0, os.path.join(REPO, 'mixedprec', 'benchmark',
                                        'rtx4090'))
        from gpu4pyscf import dft as gpu_dft
        from gpu4pyscf.grad import rhf as g_rhf
        mol = gto.M(atom=GEOMS['water'], basis='def2-svp', verbose=0)
        mf = gpu_dft.RKS(mol, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.max_cycle = 50
        mf.verbose = 0
        mf.grids.level = 3
        e = mf.kernel()
        self.assertTrue(mf.converged)
        dm0 = mf.make_rdm1()
        de = g_rhf._grad_nuc_without_ecp(mol, dm0)
        self.assertTrue(np.all(np.isfinite(de)))
        g = mf.nuc_grad_method().kernel()
        self.assertTrue(np.all(np.isfinite(g)))


if __name__ == '__main__':
    unittest.main()

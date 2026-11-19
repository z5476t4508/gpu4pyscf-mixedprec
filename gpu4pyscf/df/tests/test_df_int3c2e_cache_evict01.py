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
'''CDERI-B2D-CACHE-EVICT-01 focused unit battery: the bounded-retention
repair of the identity-keyed Int3c2eOpt cache.

Covers, at a small water cell: (a) keep-latest-per-variant eviction under
in-place set_geom_ geometry evolution (the entry count stays constant --
the entry-606 CACHE_CONT_DEFECT retention slope is repaired); (b) true
weak-key death (fresh Mole + gc -> the WeakKeyDictionary key drops); (c)
within-geometry repeat e2e hits (the e594 4->3 build win preserved: 0 new
Int3c2eOpt builds) with bitwise-identical energy; (d) the range-coulomb
context fail-closed discipline (an opt built outside the context is never
served inside it; the context entry lands in its own bounded variant slot
and the outside entry is still served after the context exits).
'''
import gc
import os
import unittest
import numpy as np
import pyscf
from pyscf.df.addons import make_auxmol
from gpu4pyscf import dft as gpu_dft
from gpu4pyscf.df import int3c2e_bdiv as ib

WATER = '''O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000'''


def setUpModule():
    global mol_sph
    mol_sph = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                      verbose=0)


def tearDownModule():
    global mol_sph
    mol_sph.stdout.close()
    del mol_sph


class CountedBuilds(unittest.TestCase):
    '''Wrap Int3c2eOpt.build with a class-level counter and isolate the
    module-level cache for every test.'''

    def setUp(self):
        self._orig_build = ib.Int3c2eOpt.build
        counter = [0]
        self._counter = counter
        outer = self._orig_build

        def counted(opt, *a, **kw):
            counter[0] += 1
            return outer(opt, *a, **kw)

        ib.Int3c2eOpt.build = counted
        ib._INT3C2E_OPT_CACHE.clear()

    def tearDown(self):
        ib.Int3c2eOpt.build = self._orig_build
        ib._INT3C2E_OPT_CACHE.clear()

    def n_builds(self):
        return self._counter[0]

    def cache_entries(self):
        return sum(len(v) for v in ib._INT3C2E_OPT_CACHE.values())

    def cache_keys(self):
        return len(list(ib._INT3C2E_OPT_CACHE.keys()))


class TestCacheMechanisms(CountedBuilds):
    def test_same_geometry_identity_hit(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        auxmol = make_auxmol(mol, 'def2-universal-jkfit')
        o1 = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        o2 = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        self.assertIs(o1, o2)
        self.assertEqual(self.n_builds(), 1)
        self.assertEqual(self.cache_entries(), 1)
        # the per-key value is the keep-latest dict (EVICT-01 structure)
        for value in ib._INT3C2E_OPT_CACHE.values():
            self.assertIsInstance(value, dict)

    def test_geometry_ladder_keeps_latest(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        auxmol = make_auxmol(mol, 'def2-universal-jkfit')
        base = mol.atom_coords().copy()
        opts = []
        for i in range(4):
            mol.set_geom_(base + i * np.array([0.01, 0.005, 0.02]),
                          unit='Bohr')
            opts.append(ib._molecular_int3c2e_opt(mol, auxmol,
                                                  decontract=True))
        # every step is a new opt (token miss, fail-closed) but the entry
        # count NEVER grows (keep-latest per variant)
        self.assertEqual(len({id(o) for o in opts}), 4)
        self.assertEqual(self.n_builds(), 4)
        self.assertEqual(self.cache_entries(), 1)

    def test_variants_coexist_bounded(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        auxmol = make_auxmol(mol, 'def2-universal-jkfit')
        o_dec = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        o_plain = ib._molecular_int3c2e_opt(mol, auxmol)
        self.assertIsNot(o_dec, o_plain)
        self.assertEqual(self.cache_entries(), 2)
        # a geometry step replaces BOTH variants, still 2 entries total
        mol.set_geom_(mol.atom_coords() + np.array([0.01, 0.005, 0.02]),
                      unit='Bohr')
        ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        ib._molecular_int3c2e_opt(mol, auxmol)
        self.assertEqual(self.cache_entries(), 2)

    def test_weak_key_death_fires(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        auxmol = make_auxmol(mol, 'def2-universal-jkfit')
        ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        self.assertEqual(self.cache_keys(), 1)
        del mol, auxmol
        gc.collect()
        # the value-graph no longer holds a strong ref to the key Mole:
        # the WeakKeyDictionary entry dies with its Mole
        self.assertEqual(self.cache_keys(), 0)
        self.assertEqual(self.cache_entries(), 0)

    def test_range_coulomb_context_fail_closed(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        auxmol = make_auxmol(mol, 'def2-universal-jkfit')
        o_out = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        with mol.with_range_coulomb(0.5):
            o_in = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        # the context opt is a fresh build (never the outside one served)
        self.assertIsNot(o_in, o_out)
        self.assertEqual(self.n_builds(), 2)
        # the context entry lands in its own bounded variant slot; the
        # context-free entry is still served after the context exits
        o_post = ib._molecular_int3c2e_opt(mol, auxmol, decontract=True)
        self.assertIs(o_post, o_out)
        self.assertEqual(self.cache_entries(), 2)


class TestE2EValueAndBuilds(CountedBuilds):
    def _e2e(self, mol):
        mf = gpu_dft.RKS(mol, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.grids.level = 3
        e = mf.kernel()
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method()
        de = np.asarray(g.kernel())
        return e, de

    def test_repeat_e2e_hits_cache_and_is_bitwise(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
        e1, g1 = self._e2e(mol)
        n_after_first = self.n_builds()
        # three cache sites served the first e2e: plain get_hcore, the
        # DF.build decontract opt (shared with the grad jk site), and the
        # grad_nuc decontract+fakemol opt
        self.assertEqual(n_after_first, 3)
        self.assertEqual(self.cache_entries(), 3)
        e2, g2 = self._e2e(mol)
        # the e594 within-geometry win is intact: 0 new builds on repeat
        self.assertEqual(self.n_builds(), n_after_first)
        self.assertEqual(self.cache_entries(), 3)
        self.assertEqual(e1, e2)
        # gradients: ULP-class agreement (atomicAdd run-to-run noise, the
        # measured parent floor; never bitwise-gated)
        self.assertLessEqual(float(np.max(np.abs(g1 - g2))), 1e-10)

        # a rigid-translation geometry step rebuilds all three variants
        # (fail-closed token miss) but the entry count stays bounded at 3;
        # the fresh opts serve the CORRECT (translation-invariant) energy
        # -- a stale-geometry opt serve would break this by >= 1e-3 Ha
        mol.set_geom_(mol.atom_coords() + np.array([0.01, 0.005, 0.02]),
                      unit='Bohr')
        e3, g3 = self._e2e(mol)
        self.assertEqual(self.n_builds(), n_after_first + 3)
        self.assertEqual(self.cache_entries(), 3)
        self.assertLessEqual(abs(e3 - e1), 1e-8)
        self.assertLessEqual(float(np.max(np.abs(g3 - g1))), 1e-8)


if __name__ == '__main__':
    unittest.main()

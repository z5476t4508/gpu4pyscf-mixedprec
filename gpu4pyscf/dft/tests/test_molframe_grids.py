# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

'''
Tests for the opt-in molecular-frame grid scheme (GPU4PYSCF_GRID_MOLFRAME=1):
Lebedev table symmetry precondition, principal-frame equivariance, grid
covariance under rigid rotation, degenerate-inertia fallback and an SCF-level
rotation-invariance smoke test.
'''
import ctypes
import itertools
import os
import unittest

import numpy
import numpy as np
from scipy.spatial import cKDTree

import pyscf
from pyscf import lib
from pyscf.dft.gen_grid import LEBEDEV_NGRID
from gpu4pyscf.dft import gen_grid
from gpu4pyscf.dft.gen_grid import _principal_frame

# A small CHON molecule with distinct inertia eigenvalues
formamide = '''
C        0.6294039867    0.0666002420    0.0000000000
O        0.6294039867   -1.1438700580    0.0000000000
N       -0.5968767777    0.6167941116    0.0000000000
H       -0.5968767777    1.6232726579    0.0000000000
H        1.4303220679    0.6765931160    0.0000000000
H       -1.4688073034   -0.0227394558    0.8875000000
'''

methanol = '''
C        -0.3612156264    0.0995801247    0.0000000000
O         0.8963305567   -0.5287602899    0.0000000000
H        -0.3612156264    1.1979611340    0.0000000000
H        -0.9709244945   -0.2662565683    0.8875000000
H        -0.9709244945   -0.2662565683   -0.8875000000
H         1.7306420145   -0.0347215761    0.0000000000
'''

def rotmat(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)

def transformed_mol(mol, R, pivot):
    coords = np.asarray(mol.atom_coords(), dtype=float)
    new = (coords - pivot) @ R.T + pivot
    atom_str = '; '.join(
        f'{sym} {x:.15f} {y:.15f} {z:.15f}'
        for sym, (x, y, z) in zip(mol.elements, new))
    return pyscf.M(atom=atom_str, basis=mol.basis, unit='Bohr',
                   charge=mol.charge, spin=mol.spin,
                   output='/dev/null', verbose=0)

def signed_perms():
    mats = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, p in enumerate(perm):
                M[i, p] = signs[i]
            mats.append(M)
    return mats

def sorted_gap(A, B):
    '''Max elementwise gap between two (n,4) arrays after lexicographic
    sorting on coordinates rounded to 9 decimals (multiset comparison).'''
    if A.shape != B.shape:
        return np.inf
    key = lambda X: np.lexsort(tuple(np.round(X[:, k], 9)
                                     for k in range(X.shape[1]-1, -1, -1)))
    return float(np.abs(A[key(A)] - B[key(B)]).max())


class TestLebedevB3Symmetry(unittest.TestCase):
    def test_tables_b3_invariant(self):
        # Design precondition of the molecular-frame scheme: the
        # MakeAngularGrid (node, weight) tables must be invariant as sets
        # under all 48 signed permutations, so that any eigenvector
        # sign/order discrepancy between a geometry and its rotated copy
        # leaves the placed grid set unchanged.
        # Note LEBEDEV_NGRID lists n=1, but MakeAngularGrid does not
        # initialize a buffer for it (upstream quirk; unreachable from the
        # standard prune paths, which emit n >= 14).  Sizes the library
        # actually initializes are identified by their normalized weights
        # summing to 1.
        libdft = lib.load_library('libdft')
        checked = 0
        for n in sorted(set(int(x) for x in LEBEDEV_NGRID)):
            G = np.empty((n, 4))
            libdft.MakeAngularGrid(G.ctypes.data_as(ctypes.c_void_p),
                                   ctypes.c_int(n))
            if not np.isfinite(G).all() or abs(G[:, 3].sum() - 1) > 1e-9:
                continue
            checked += 1
            for M in signed_perms():
                Gp = np.hstack([G[:, :3] @ M.T, G[:, 3:4]])
                gap = sorted_gap(Gp, G)
                self.assertLess(gap, 1e-10, f'n={n}, perm={M.tolist()}')
        self.assertGreaterEqual(checked, 30)


class TestPrincipalFrame(unittest.TestCase):
    def test_frame_well_defined(self):
        mol = pyscf.M(atom=formamide, basis='def2-svp', output='/dev/null',
                      verbose=0)
        A = _principal_frame(mol)
        self.assertIsNotNone(A)
        self.assertAlmostEqual(abs(np.linalg.det(A)), 1.0, 12)
        np.testing.assert_allclose(A @ A.T, np.eye(3), atol=1e-12)

    def test_frame_equivariant_mod_signed_perm(self):
        # A(rotated mol) = D @ A(mol) @ U.T for some signed diagonal D (det +1,
        # both frames right-handed and U proper); equivalently
        # A2 @ U @ A1.T is a signed identity up to floating-point noise.
        mol = pyscf.M(atom=formamide, basis='def2-svp', output='/dev/null',
                      verbose=0)
        A1 = _principal_frame(mol)
        for axis, angle in [((1, 2, 3), 0.7), ((-2, 1, 0.5), 1.9),
                            ((0.3, -0.8, 2.0), 2.8)]:
            U = rotmat(axis, angle)
            pivot = np.array([100.0, -50.0, 25.0])
            mol2 = transformed_mol(mol, U, pivot)
            A2 = _principal_frame(mol2)
            M = A2 @ U @ A1.T
            signs = []
            for i in range(3):
                e_i = np.zeros(3)
                e_i[i] = 1.0
                if abs(M[i] - e_i).max() < 1e-9:
                    signs.append(1)
                elif abs(M[i] + e_i).max() < 1e-9:
                    signs.append(-1)
                else:
                    self.fail(f'row {i} of A2@U@A1.T is not +-e_i: {M[i]}')
            self.assertEqual(numpy.prod(signs), 1)

    def test_degenerate_fallbacks(self):
        # Spherical top (Td): all three inertia eigenvalues equal, no unique
        # axis at all.  Linear molecules (all mass on the axis: no in-plane
        # information) and single atoms stay in the fallback class too.
        # Note the planar-square symmetric top (D4h) NO LONGER falls back:
        # the tie-breaker resolves its in-plane axes at k=4 (see
        # TestSymmetricTopFrames), with a benign octahedral jump class.
        tetrahedral_h4 = pyscf.M(
            atom='H 1 1 1; H 1 -1 -1; H -1 1 -1; H -1 -1 1',
            basis='def2-svp', output='/dev/null', verbose=0)
        self.assertIsNone(_principal_frame(tetrahedral_h4))
        self.assertIsNone(_principal_frame(
            pyscf.M(atom='H 0 0 0; H 0 0 0.9',
                    basis='def2-svp', output='/dev/null', verbose=0)))
        self.assertIsNone(_principal_frame(
            pyscf.M(atom='He 0 0 0', basis='def2-svp',
                    output='/dev/null', verbose=0)))


# synthetic exact symmetric tops
d3h_triangle = '''
C        2.0        0.0                   0.0
C       -1.0        1.7320508075688772    0.0
C       -1.0       -1.7320508075688772    0.0
'''

d2d_pattern = '''
C         1.0        0.0        0.5
C         0.0        1.0       -0.5
C        -1.0        0.0        0.5
C         0.0       -1.0       -0.5
'''

d6h_ring = '''
C         1.5        0.0                   0.0
C         0.75       1.299038105676658     0.0
C        -0.75       1.299038105676658     0.0
C        -1.5        0.0                   0.0
C        -0.75      -1.299038105676658     0.0
C         0.75      -1.299038105676658     0.0
'''


class TestSymmetricTopFrames(unittest.TestCase):
    def _mol(self, atom_str):
        return pyscf.M(atom=atom_str, basis='def2-svp', output='/dev/null',
                       verbose=0)

    def test_k4_class_resolved_others_fall_back(self):
        # D2d/D4h/C4v tops resolve via the 4th in-plane moment; the 3/5/6-fold
        # classes keep the lab-frame fallback: their ambiguity class contains
        # non-octahedral rotations, and any deterministic selection is exactly
        # covariant for only 1/3 (3/6-fold) or 1/5 (5-fold) of uniform
        # rotations - the winding obstruction documented in
        # _symtop_inplane_axes.  Better an honest fallback than an
        # unpredictably orientation-dependent frame.
        A = _principal_frame(self._mol(d2d_pattern))
        self.assertIsNotNone(A)
        self.assertAlmostEqual(abs(numpy.linalg.det(A)), 1.0, 12)
        numpy.testing.assert_allclose(A @ A.T, numpy.eye(3), atol=1e-12)
        h4_square = self._mol('H 1 0 0; H -1 0 0; H 0 1 0; H 0 -1 0')
        A = _principal_frame(h4_square)
        self.assertIsNotNone(A)
        numpy.testing.assert_allclose(A @ A.T, numpy.eye(3), atol=1e-12)
        # the excluded classes
        self.assertIsNone(_principal_frame(self._mol(d3h_triangle)))
        self.assertIsNone(_principal_frame(self._mol(d6h_ring)))

    def test_symtop_equivariance_signed_perm(self):
        # The 4-fold class is the fully benign one: every member of its
        # ambiguity class {0, 90, 180, 270 degrees in-plane} is an octahedral
        # signed permutation, so the frame is equivariant modulo the 48 for
        # EVERY rigid rotation, cut crossings included.
        perms = signed_perms()
        for atom_str in (d2d_pattern,
                         'H 1 0 0; H -1 0 0; H 0 1 0; H 0 -1 0'):
            mol = self._mol(atom_str)
            A1 = _principal_frame(mol)
            self.assertIsNotNone(A1)
            for axis, angle in [((1, 2, 3), 0.7), ((-2, 1, 0.5), 1.9),
                                ((0.3, -0.8, 2.0), 2.8),
                                ((0, 0, 1), numpy.deg2rad(47.0))]:
                U = rotmat(axis, angle)
                pivot = numpy.array([100.0, -50.0, 25.0])
                mol2 = transformed_mol(mol, U, pivot)
                A2 = _principal_frame(mol2)
                M = A2 @ U @ A1.T
                self.assertTrue(
                    any(numpy.abs(M - D).max() < 1e-9 for D in perms),
                    f'axis={axis}, angle={angle}: M={M.tolist()}')

    def test_d4h_jump_benign_at_grid_level(self):
        # k=4's jump class is a 90-degree table rotation, which IS an
        # octahedral signed permutation: even past the branch cut (46 deg
        # about the unique axis) the placed grid set stays exactly covariant.
        # This is the design's benign-by-construction property, verified at
        # the grid-set level.
        mol = self._mol(d2d_pattern)
        U = rotmat((0, 0, 1), numpy.deg2rad(46.0))  # past the 45-deg cut
        pivot = numpy.array([100.0, -50.0, 25.0])
        mol2 = transformed_mol(mol, U, pivot)
        g1 = self._grids(mol)
        g2 = self._grids(mol2)
        self.assertIsNotNone(g1._molframe_applied)
        self.assertIsNotNone(g2._molframe_applied)
        keep1 = numpy.asarray(g1.atm_idx.get()) != -1
        keep2 = numpy.asarray(g2.atm_idx.get()) != -1
        self.assertEqual(keep1.sum(), keep2.sum())
        c1 = numpy.asarray(g1.coords.get())[keep1]
        c2 = numpy.asarray(g2.coords.get())[keep2]
        expected = (c1 - pivot) @ U.T + pivot
        tree = cKDTree(expected)
        dist, _ = tree.query(c2)
        self.assertLess(dist.max(), 1e-9)

    def test_symtop_grids_covariant_under_rotation(self):
        mol = self._mol(d2d_pattern)
        U = rotmat((1, 2, 3), 0.7)
        pivot = numpy.array([100.0, -50.0, 25.0])
        mol2 = transformed_mol(mol, U, pivot)
        g1 = self._grids(mol)
        g2 = self._grids(mol2)
        self.assertIsNotNone(g1._molframe_applied)
        self.assertIsNotNone(g2._molframe_applied)
        self.assertEqual(g1.size, g2.size)
        keep1 = numpy.asarray(g1.atm_idx.get()) != -1
        keep2 = numpy.asarray(g2.atm_idx.get()) != -1
        self.assertEqual(keep1.sum(), keep2.sum())
        w1 = numpy.asarray(g1.weights.get())[keep1]
        w2 = numpy.asarray(g2.weights.get())[keep2]
        c1 = numpy.asarray(g1.coords.get())[keep1]
        c2 = numpy.asarray(g2.coords.get())[keep2]
        expected = (c1 - pivot) @ U.T + pivot
        tree = cKDTree(expected)
        dist, idx = tree.query(c2)
        self.assertLess(dist.max(), 1e-9)
        numpy.testing.assert_allclose(w2, w1[idx], rtol=1e-12, atol=1e-13)

    def test_excluded_classes_stay_lab_frame(self):
        # the 3/6-fold cells must not get a frame: knob-on builds lab-frame
        # grids and records the fallback
        for atom_str in (d3h_triangle, d6h_ring):
            mol = self._mol(atom_str)
            g = self._grids(mol)
            self.assertIsNone(g._molframe_applied)

    def _grids(self, mol):
        old = os.environ.get('GPU4PYSCF_GRID_MOLFRAME')
        os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '1'
        try:
            grids = gen_grid.Grids(mol)
            grids.level = 3
            grids.build()
        finally:
            if old is None:
                os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            else:
                os.environ['GPU4PYSCF_GRID_MOLFRAME'] = old
        return grids


class TestMolecularFrameGrids(unittest.TestCase):
    def _build(self, mol, knob):
        old = os.environ.get('GPU4PYSCF_GRID_MOLFRAME')
        if knob:
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '1'
        else:
            os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
        try:
            grids = gen_grid.Grids(mol)
            grids.level = 3
            grids.build()
        finally:
            if old is None:
                os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            else:
                os.environ['GPU4PYSCF_GRID_MOLFRAME'] = old
        return grids

    def test_grids_covariant_under_rotation(self):
        mol = pyscf.M(atom=formamide, basis='def2-svp', output='/dev/null',
                      verbose=0)
        U = rotmat((1, 2, 3), 0.7)
        pivot = np.array([100.0, -50.0, 25.0])
        mol2 = transformed_mol(mol, U, pivot)

        g1 = self._build(mol, knob=True)
        g2 = self._build(mol2, knob=True)
        self.assertIsNotNone(g1._molframe_applied)
        self.assertIsNotNone(g2._molframe_applied)
        self.assertEqual(g1.size, g2.size)

        # padding rows are marked atm_idx == -1; real points carry the atom
        # index whatever their (possibly zero) Becke weight
        keep1 = np.asarray(g1.atm_idx.get()) != -1
        keep2 = np.asarray(g2.atm_idx.get()) != -1
        self.assertEqual(keep1.sum(), keep2.sum())
        w1 = np.asarray(g1.weights.get())[keep1]
        w2 = np.asarray(g2.weights.get())[keep2]
        c1 = np.asarray(g1.coords.get())[keep1]
        c2 = np.asarray(g2.coords.get())[keep2]
        # the rotated grid must reproduce the original grid rigidly rotated
        # about the same pivot as transformed_mol
        expected = (c1 - pivot) @ U.T + pivot
        tree = cKDTree(expected)
        dist, idx = tree.query(c2)
        self.assertLess(dist.max(), 1e-9)
        # weights agree to absolute Becke-kernel floating-point noise (the
        # kernel recomputes interatomic distances in rotated coordinates);
        # near-zero weights need an absolute tolerance
        np.testing.assert_allclose(w2, w1[idx], rtol=1e-12, atol=1e-13)

    def test_labframe_grids_not_covariant(self):
        # the defect the knob repairs: knob-off grids are lab-frame and a
        # generic rotation moves the quadrature set
        mol = pyscf.M(atom=formamide, basis='def2-svp', output='/dev/null',
                      verbose=0)
        U = rotmat((1, 2, 3), 0.7)
        pivot = np.array([100.0, -50.0, 25.0])
        mol2 = transformed_mol(mol, U, pivot)
        g1 = self._build(mol, knob=False)
        g2 = self._build(mol2, knob=False)
        self.assertIsNone(g1._molframe_applied)
        keep1 = np.asarray(g1.atm_idx.get()) != -1
        keep2 = np.asarray(g2.atm_idx.get()) != -1
        c1 = np.asarray(g1.coords.get())[keep1]
        c2 = np.asarray(g2.coords.get())[keep2]
        expected = (c1 - pivot) @ U.T + pivot
        tree = cKDTree(expected)
        dist, _ = tree.query(c2)
        self.assertGreater(dist.max(), 1e-3)

    def test_knob_off_never_touches_frame(self):
        # the default path must be untouched by the new code
        mol = pyscf.M(atom=formamide, basis='def2-svp', output='/dev/null',
                      verbose=0)
        os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
        orig = gen_grid._principal_frame

        def boom(mol):
            raise AssertionError('frame computed with knob off')
        gen_grid._principal_frame = boom
        try:
            grids = gen_grid.Grids(mol)
            grids.level = 3
            grids.build()
            self.assertIsNone(grids._molframe_applied)
        finally:
            gen_grid._principal_frame = orig

    def test_knob_parsing(self):
        orig = os.environ.get('GPU4PYSCF_GRID_MOLFRAME')
        try:
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '1'
            self.assertTrue(gen_grid._molframe_enabled())
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '0'
            self.assertFalse(gen_grid._molframe_enabled())
            os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            self.assertFalse(gen_grid._molframe_enabled())
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = 'yes'
            with self.assertWarns(UserWarning):
                self.assertFalse(gen_grid._molframe_enabled())
        finally:
            if orig is None:
                os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            else:
                os.environ['GPU4PYSCF_GRID_MOLFRAME'] = orig


class TestEnergyRotationInvariance(unittest.TestCase):
    def test_wb97mv_rotation_invariance(self):
        # SCF-level smoke test of the whole pipeline (grids + nlcgrids) on
        # the fp64 lane: the knob makes a rigid rotation change the energy by
        # floating-point noise only
        from gpu4pyscf import dft
        mol = pyscf.M(atom=methanol, basis='def2-svp', charge=0, spin=0,
                      output='/dev/null', verbose=0)
        U = rotmat((1, 2, 3), 0.7)
        pivot = numpy.array([100.0, -50.0, 25.0])
        mol2 = transformed_mol(mol, U, pivot)

        energies = {}
        for label, m in (('id', mol), ('rot', mol2)):
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '1'
            try:
                mf = dft.RKS(m, xc='wb97m-v').density_fit()
                mf.conv_tol = 1e-10
                mf.precision_mode = 'fp64'
                mf.kernel()
            finally:
                os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            self.assertTrue(mf.converged)
            energies[label] = mf.e_tot
        self.assertLess(abs(energies['rot'] - energies['id']), 1e-11)

    def test_wb97mv_symtop_rotation_invariance(self):
        # cyclobutane (D2d, resolved via the 4th in-plane moment): the
        # tie-breaker extends the knob's rotation invariance to the 4-fold
        # symmetric-top class, both for a generic rotation and for a
        # rotation about the molecule's own unique axis past the 45-degree
        # branch cut (the jump class is octahedral-benign, so the energy
        # stays invariant there too - the design's marquee property).
        from gpu4pyscf import dft
        path = ('/data/work/gpu4pyscf-agent/inventory/'
                'gpu4pyscf-g2-fallback-20261017-v1/cyclobutane.xyz')
        with open(path) as f:
            lines = f.read().splitlines()
        n = int(lines[0].split()[0])
        atoms = [(p[0], numpy.array([float(p[1]), float(p[2]), float(p[3])]))
                 for p in (l.split() for l in lines[2:2 + n])]
        atom_str = '; '.join(f'{s} {x:.15f} {y:.15f} {z:.15f}'
                             for s, (x, y, z) in atoms)
        mol = pyscf.M(atom=atom_str, basis='def2-svp', charge=0, spin=0,
                      output='/dev/null', verbose=0)
        # the unique axis for the past-cut in-plane rotation
        coords = numpy.asarray(mol.atom_coords(), dtype=float)
        masses = numpy.asarray(mol.atom_mass_list(), dtype=float)
        com = masses @ coords / masses.sum()
        p = coords - com
        gyt = numpy.einsum('i,ij,ik->jk', masses, p, p)
        inertia = numpy.trace(gyt) * numpy.eye(3) - gyt
        evals, evecs = numpy.linalg.eigh(inertia)
        deg = numpy.diff(evals) <= 1e-6 * abs(inertia).max()
        iu = 0 if deg[1] else 2
        u = evecs[:, iu]
        pivot = numpy.array([100.0, -50.0, 25.0])

        energies = {}
        for label, U in (('id', numpy.eye(3)),
                         ('rot', rotmat((1, 2, 3), 0.7)),
                         ('inplane_cut', rotmat(u, numpy.deg2rad(46.0)))):
            mol_r = transformed_mol(mol, U, pivot) if label != 'id' else mol
            os.environ['GPU4PYSCF_GRID_MOLFRAME'] = '1'
            try:
                mf = dft.RKS(mol_r, xc='wb97m-v').density_fit()
                mf.conv_tol = 1e-10
                mf.precision_mode = 'fp64'
                mf.kernel()
            finally:
                os.environ.pop('GPU4PYSCF_GRID_MOLFRAME', None)
            self.assertTrue(mf.converged)
            self.assertIsNotNone(mf.grids._molframe_applied)
            energies[label] = mf.e_tot
        self.assertLess(abs(energies['rot'] - energies['id']), 1e-11)
        self.assertLess(abs(energies['inplane_cut'] - energies['id']), 1e-11)


if __name__ == '__main__':
    unittest.main()

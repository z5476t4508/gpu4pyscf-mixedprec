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
import os
import unittest

import numpy as np
import pyscf
import pytest
from pyscf.hessian import thermo

from gpu4pyscf import dft
from gpu4pyscf import scf
from gpu4pyscf.lib import precision

VITAMIN_C = os.path.join(os.path.dirname(__file__),
                         '..', '..', '..', 'tests', '020_Vitamin_C.xyz')


def setUpModule():
    global mol, mf, hess_ref
    mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                  basis='def2-svp', output='/dev/null', verbose=1)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-12
    mf.kernel()
    hess_ref = mf.Hessian().kernel()


def tearDownModule():
    global mol, mf, hess_ref
    mol.stdout.close()
    del mol, mf, hess_ref


HARTREE2WAVENUMBER = 219474.6313632


def frequencies(hess):
    '''harmonic frequencies in cm^-1, sign kept for imaginary modes'''
    mass = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -.5
    n = 3 * mol.natm
    h = hess.transpose(0, 2, 1, 3).reshape(n, n) * mass[:, None] * mass[None, :]
    ev = np.linalg.eigvalsh(h) / 1822.888486209
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER


class KnownValues(unittest.TestCase):

    def tearDown(self):
        mf.precision_mode = None
        precision.set_precision('fp64')

    def test_auto_selects_fp32(self):
        '''a Hessian has no accuracy left to recover by spending float64 on
        the CPHF solve, so 'auto' means float32 here -- as it does on a
        gradient'''
        mf.precision_mode = 'auto'
        self.assertEqual(mf.Hessian()._resolve_precision(), 'fp32')

    def test_hessian_object_overrides_the_mean_field_setting(self):
        mf.precision_mode = 'auto'
        h = mf.Hessian()
        h.precision_mode = 'fp64'
        self.assertEqual(h._resolve_precision(), 'fp64')

    def test_rejects_an_unknown_mode(self):
        h = mf.Hessian()
        h.precision_mode = 'half'
        self.assertRaises(ValueError, h.kernel)

    def test_fp32_cphf_accuracy(self):
        '''the float32 CPHF J/K must stay far inside anything a frequency
        calculation would notice'''
        mf.precision_mode = 'auto'
        hess = mf.Hessian().kernel()
        self.assertLess(np.abs(hess - hess_ref).max(), 1e-6)

        n = 3 * mol.natm
        eig = np.linalg.eigvalsh(hess.transpose(0, 2, 1, 3).reshape(n, n))
        eig_ref = np.linalg.eigvalsh(
            hess_ref.transpose(0, 2, 1, 3).reshape(n, n))
        self.assertLess(np.abs(eig - eig_ref).max(), 1e-6)

    def test_default_lane_is_unchanged(self):
        '''an unset precision_mode must leave the float64 path alone'''
        hess = mf.Hessian().kernel()
        self.assertLess(np.abs(hess - hess_ref).max(), 1e-10)

    def test_global_mode_is_restored(self):
        '''kernel() mutates process-global state; a raise must not leak it'''
        mf.precision_mode = 'fp32'
        h = mf.Hessian()
        h.kernel()
        self.assertEqual(precision.get_precision(), 'fp64')

        h.solve_mo1 = lambda *a, **kw: 1 / 0
        self.assertRaises(ZeroDivisionError, h.kernel)
        self.assertEqual(precision.get_precision(), 'fp64')

    def test_uhf_fp32_cphf(self):
        '''the CPHF J/K is shared with UHF, which uses the nspin=2 branch'''
        umf = scf.UHF(mol).density_fit()
        umf.conv_tol = 1e-12
        umf.kernel()
        ref = umf.Hessian().kernel()

        umf.precision_mode = 'auto'
        self.assertLess(np.abs(umf.Hessian().kernel() - ref).max(), 1e-6)

    def test_rks_fp32_cphf_response(self):
        '''The DFT lane covers make_h1's grid loop as well as the CPHF, and
        each functional type takes a different branch of both: LDA, GGA and
        MGGA differ, and a hybrid additionally exercises the float32 K build.

        The bound is on frequencies rather than on Hessian elements. Elements
        are a proxy, and a misleading one here: the float32 make_h1 moves
        max|dH| on this molecule to ~1e-5 while the frequencies it produces
        move by at most 0.031 cm^-1 (PBE), and on Tamoxifen/def2-SVP -- 57
        atoms, where it is worth 4.05x -- by 0.001 cm^-1. 0.1 cm^-1 leaves
        3x margin over the worst measured value and still catches a real
        regression: bracketing partial_hess_elec too, which was rejected,
        moved Tamoxifen by 0.131-0.181 cm^-1.

        KNOWN LIMIT OF THIS TEST -- read before trusting it. Water frequencies
        are themselves only a proxy for a real molecule's. The first port of
        _get_vxc_deriv2_task passed every assertion here while moving Tamoxifen
        by 5.579 cm^-1 rms 0.475 -- 55x over this bound. Three atoms give 3
        vibrations and a grid a fraction the size, so a float32 reduction over
        grid blocks barely accumulates. A change to the grid lane is not
        verified until it has cleared mixedprec/step6l on Tamoxifen; green here
        means "not grossly broken", not "accurate".
        '''
        for xc in ('LDA,VWN', 'PBE', 'r2scan', 'B3LYP'):
            with self.subTest(xc=xc):
                rks = dft.RKS(mol, xc=xc).density_fit()
                rks.conv_tol = 1e-12
                rks.kernel()
                ref = rks.Hessian().kernel()

                rks.precision_mode = 'auto'
                hess = rks.Hessian().kernel()

                vib = slice(6, None)
                nu = frequencies(hess)[vib]
                nu_ref = frequencies(ref)[vib]
                self.assertLess(np.abs(nu - nu_ref).max(), 0.1)
                # a loose element bound, to catch gross breakage rather than
                # benign precision loss
                self.assertLess(np.abs(hess - ref).max(), 1e-4)

    @pytest.mark.slow
    def test_vitamin_c_sentinel(self):
        '''The molecule the water tests above cannot replace.

        Measured separation between the shipped lane and a known-bad change
        (bracketing _get_exc_deriv2 in float32 unported), by
        mixedprec/step7a_sentinel_adequacy.py:

                        lane        known-bad   separation
            max|dnu|    0.0011      0.0374      35x
            rms         0.0003      0.0091      31x
            dZPE        3.6e-06     1.2e-04     34x    kcal/mol
            dSvib       1.7e-05     9.3e-04     54x    cal/mol/K

        On water the same two runs give 0.0004 and 0.0005 -- a separation of
        1x. Water does not distinguish a good change from a bad one at all, so
        no threshold on it can work; that is why this test exists rather than
        another assertion on the 3-atom case.

        Each bound below is the geometric mean of the two measured columns, so
        it sits ~6x above the lane and ~6x below the known-bad change instead
        of being picked to pass. S_vib separates best because it weights the
        soft modes where a Hessian error is amplified (dw ~ dH/2*mu*w), and it
        is also what a user consumes; max|dnu| is kept but is the weakest of
        the four, since a max over modes grows with mode count.

        Frequencies come from pyscf's harmonic_analysis, which projects out
        translations and rotations. Do not replace it with eigvalsh(...)[6:]:
        this geometry has 11 imaginary modes, so sorting pulls contaminated
        rotations into the compared set (on Tamoxifen that inflated the same
        measurement by 1.27x).
        '''
        mol_c = pyscf.M(atom=VITAMIN_C, basis='def2-svp', output='/dev/null',
                        verbose=1)
        try:
            mf_c = dft.RKS(mol_c, xc='r2scan').density_fit()
            mf_c.conv_tol = 1e-10
            ref = mf_c.Hessian().kernel()

            mf_c.precision_mode = 'auto'
            hess = mf_c.Hessian().kernel()
            mf_c.precision_mode = None

            nu_ref = _freq(mol_c, ref)
            nu = _freq(mol_c, hess)
            self.assertLess(np.abs(nu - nu_ref).max(), 0.0064)
            self.assertLess(np.sqrt(((nu - nu_ref) ** 2).mean()), 0.0017)

            zpe_ref, s_ref = _thermo(nu_ref)
            zpe, s = _thermo(nu)
            self.assertLess(abs(zpe - zpe_ref) * 627.5095, 2.1e-5)
            self.assertLess(abs(s - s_ref) * 627.5095 * 1000, 1.3e-4)
        finally:
            mol_c.stdout.close()


def _freq(mol, hess):
    '''vibrational frequencies, cm^-1, translations/rotations projected out'''
    hess = np.asarray(getattr(hess, 'get', lambda: hess)())
    res = thermo.harmonic_analysis(mol, hess, imaginary_freq=False)
    return np.asarray(res['freq_wavenumber']).real


def _thermo(freq, temperature=298.15):
    '''ZPE (Eh) and S_vib (Eh/K); imaginary modes dropped, so only the
    difference between two lanes on the same geometry is meaningful'''
    nu_au = freq[freq > 0] / thermo.nist.HARTREE2WAVENUMBER
    kt = thermo.nist.BOLTZMANN / thermo.nist.HARTREE2J * temperature
    x = nu_au / kt
    s_vib = (x / np.expm1(x) - np.log1p(-np.exp(-x))).sum() * kt / temperature
    return .5 * nu_au.sum(), s_vib


if __name__ == '__main__':
    print('Full tests for the mixed-precision DF Hessian')
    unittest.main()

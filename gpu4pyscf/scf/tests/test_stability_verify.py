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
import unittest
import unittest.mock as mock
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pyscf
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.scf import stability_verify


def setUpModule():
    global mol_sph, mol_uks, mol_methoxy
    # mol_uks: a small open-shell doublet (the methoxy-radical family the
    # 'auto+verify' policy targets; near-degenerate UKS stationary pairs)
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=1)
    mol_uks = pyscf.M(atom='''
C       0.0000000000     0.0000000000     0.1274000000
O       0.0000000000     1.1000000000    -0.4600000000
H       0.8700000000    -0.5500000000    -0.4600000000
H      -0.8700000000    -0.5500000000    -0.4600000000''',
                      basis='def2-svp', spin=1, charge=-1,
                      output='/dev/null', verbose=1)
    # the benchmark UKS sentinel whose two-basin wb97m-v split was
    # measured in issue #7 (dm max|dP| 0.35, beta occupied-space swap,
    # auto branch lower by 1.28e-6 Ha) and whose m062x/tzvp sibling
    # produced the fp32 fixed-point displacement (LC-28)
    mol_methoxy = pyscf.M(atom='''
O       0.0000000000     0.0000000000     0.0000000000
C       0.0000000000     0.0000000000     1.3680000000
H       0.0000000000     1.0276700000     1.7313110000
H      -0.8899880000    -0.5138350000     1.7313110000
H       0.8899880000    -0.5138350000     1.7313110000''',
                          basis='def2-svp', spin=1, charge=0,
                          output='/dev/null', verbose=1)


def tearDownModule():
    global mol_sph, mol_uks, mol_methoxy
    mol_sph.stdout.close()
    del mol_sph
    mol_uks.stdout.close()
    del mol_uks
    mol_methoxy.stdout.close()
    del mol_methoxy


class KnownValues(unittest.TestCase):

    def test_policy_validation(self):
        '''the accepted precision_mode vocabulary; auto+verify admitted'''
        mf = scf.RHF(mol_sph).density_fit()
        for ok in (None, 'fp64', 'fp32', 'auto', 'auto+verify'):
            mf.precision_mode = ok
            # kernel() is not run here; validation happens inside _kernel,
            # so exercise the setter path via a 0-cycle scf call
            mf.max_cycle = 0
            mf.precision_mode = ok
            mf.kernel()
        mf.precision_mode = 'bogus'
        with self.assertRaises(ValueError):
            mf.kernel()

    def test_auto_verify_records_decision(self):
        '''auto+verify must record the selection in scf_summary and keep
        a verified stationary solution'''
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        e_v = mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertEqual(rec['policy'], 'auto+verify')
        self.assertIn(rec['selected'], ('main', 'reference'))
        # the kept energy must be the adopted stationary point: the raw
        # main energy when nothing fired, the refined/adopted stationary
        # solution otherwise (never a merely-lower displaced point)
        if not rec.get('distinct_solutions') and not rec.get('fp32_displacement_detected'):
            self.assertAlmostEqual(float(mf.e_tot), rec['main_e_tot'],
                                   delta=1e-12)
        else:
            self.assertTrue(rec.get('adopted_stationarity_verified'))
            cand = rec.get('main_refined_e_tot', rec['main_e_tot'])
            self.assertAlmostEqual(
                float(mf.e_tot), min(cand, rec['ref_e_tot']), delta=1e-10)
            self.assertIn('rejected', rec)
        self.assertTrue(mf.converged)

    def test_auto_verify_closed_shell_same_solution(self):
        '''closed shell: the verification trajectory converges to the same
        density; the record must say so and the energy must not move'''
        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto'
        e_auto = mf.kernel()

        mf2 = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf2.conv_tol = 1e-10
        mf2.precision_mode = 'auto+verify'
        e_v = mf2.kernel()
        rec = mf2.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertEqual(rec['selected'], 'main')
        # the dm test never fired: no refinement was run or recorded
        self.assertNotIn('refinement', rec)
        self.assertNotIn('adopted_stationarity_verified', rec)
        self.assertAlmostEqual(e_v, e_auto, delta=1e-10)

    def test_auto_verify_selects_lower_branch(self):
        '''when the two trajectories land on different solutions, the
        lower STATIONARY one must be kept; on the measured split cell the
        refinement must recognize the split and verify stationarity'''
        mf = dft.UKS(mol_methoxy, xc='wb97m-v').density_fit()
        mf.conv_tol = 1e-10
        mf.init_guess = 'minao'
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        # the measured wb97m-v split on this cell: dm max|dP| 0.35 fires
        # the distinctness test, and the fp64 refinement from the auto
        # density must stay in the auto basin (a genuine two-basin split)
        self.assertTrue(rec['distinct_solutions'])
        self.assertEqual(rec['refinement'], 'distinct')
        self.assertFalse(rec['fp32_displacement_detected'])
        self.assertTrue(rec['adopted_stationarity_verified'])
        self.assertIn(rec['selected'], ('main', 'reference'))
        # the refined auto-basin stationary point is the measured lower
        # solution (-1.28e-6 Ha below the fp64-lane branch): it must be
        # adopted, preserving the honest lane comparison
        self.assertEqual(rec['selected'], 'main')
        self.assertAlmostEqual(
            float(mf.e_tot), rec['main_refined_e_tot'], delta=1e-12)
        self.assertGreater(rec['dE_ref_minus_main'], 1e-7)
        self.assertIn('rejected', rec)
        # both solutions' orbital gradients at the fp64 floor
        self.assertLess(rec['orbital_grad_norm_main_refined'], 1e-5)
        self.assertLess(rec['orbital_grad_norm_ref'], 1e-5)
        # the occupied-space overlap record: the measured beta swap has
        # min singular value ~0.26, far below 1
        ov = rec['occ_overlap_min_main_vs_ref']
        if isinstance(ov, dict):
            self.assertLess(ov['beta'], 0.5)
            self.assertGreater(ov['alpha'], 0.99)

    def test_auto_verify_refinement_fallback(self):
        '''if the bounded fp64 refinement does not converge, the policy
        falls back to the raw two-trajectory rule and honestly flags
        adopted_stationarity_verified=False'''
        mf = dft.UKS(mol_methoxy, xc='wb97m-v').density_fit()
        mf.conv_tol = 1e-10
        mf.init_guess = 'minao'
        mf.precision_mode = 'auto+verify'
        with mock.patch.object(stability_verify,
                               'stability_verify_refine_max_cycle', 0):
            mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertTrue(rec['distinct_solutions'])
        self.assertEqual(rec['refinement'], 'non_converged')
        self.assertIs(rec['adopted_stationarity_verified'], False)
        self.assertIsNone(rec['fp32_displacement_detected'])
        # LC-64: the exact unusability reason is recorded
        self.assertEqual(rec['refine_unusable'], 'not_converged')
        # raw rule: main is lower on this cell -> kept unrefined
        self.assertEqual(rec['selected'], 'main')
        self.assertAlmostEqual(float(mf.e_tot), rec['main_e_tot'],
                               delta=1e-12)
        self.assertIn('rejected', rec)

    def test_auto_verify_same_solution_displacement(self):
        '''the fp32-displacement case (measured on m062x/methoxy/
        def2-tzvp, LC-28/29): the dm test fires but the refinement
        climbs back to the reference basin -- the policy must detect
        the displacement, adopt the STATIONARY reference point and
        preserve the displaced point's record'''
        mol = pyscf.M(atom='''
O       0.0000000000     0.0000000000     0.0000000000
C       0.0000000000     0.0000000000     1.3680000000
H       0.0000000000     1.0276700000     1.7313110000
H      -0.8899880000    -0.5138350000     1.7313110000
H       0.8899880000    -0.5138350000     1.7313110000''',
                      basis='def2-tzvp', spin=1, charge=0,
                      output='/dev/null', verbose=1)
        try:
            mf = dft.UKS(mol, xc='m062x').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.init_guess = 'minao'
            mf.grids.level = 3
            mf.precision_mode = 'auto+verify'
            e = mf.kernel()
            rec = mf.scf_summary['stability_verification']
            # the dm test fired on the fp32 displacement itself
            self.assertGreater(rec['dm_max_abs_diff'],
                               stability_verify.stability_verify_dm_tol)
            self.assertEqual(rec['refinement'], 'same_solution')
            self.assertFalse(rec['distinct_solutions'])
            self.assertTrue(rec['fp32_displacement_detected'])
            self.assertTrue(rec['adopted_stationarity_verified'])
            self.assertEqual(rec['selected'], 'reference')
            # the adopted point is the stationary saddle, which sits
            # 4.09e-8 Ha ABOVE the displaced auto point (LC-28: the
            # fp32 point slid downhill along the unstable mode)
            self.assertAlmostEqual(float(e), rec['ref_e_tot'], delta=1e-12)
            self.assertGreater(e - rec['main_e_tot'], 0)
            self.assertLess(e - rec['main_e_tot'], 1e-7)
            self.assertAlmostEqual(rec['auto_point_displacement_e'],
                                   rec['main_e_tot']
                                   - rec['main_refined_e_tot'],
                                   delta=1e-15)
            # the displaced point's elevated gradient is preserved
            self.assertGreater(rec['orbital_grad_norm_main'],
                               2 * rec['orbital_grad_norm_ref'])
            self.assertLess(rec['orbital_grad_norm_main_refined'],
                            rec['orbital_grad_norm_ref'])
            # occupied spaces agree (no branch swap on this cell)
            ov = rec['occ_overlap_min_main_vs_ref']
            self.assertGreater(min(ov['alpha'], ov['beta']), 0.99)
        finally:
            mol.stdout.close()
            del mol

    def test_auto_verify_unconverged_skips(self):
        '''a non-converged main SCF must skip verification, not crash'''
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-14
        mf.max_cycle = 1
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertIn('skipped', rec)
        self.assertEqual(rec['skipped'], 'main SCF did not converge')

    def test_sibling_carries_instance_dispersion_hook(self):
        '''_build_fp64_sibling must carry instance-attached dispersion
        hooks (the r2SCAN-3c runner's D4+gCP MethodType attach).  Before
        LC-57b the sibling silently dropped them: measured on every 3c
        verify record (benzene/vitaminC/tamoxifen), ref_e_tot was low by
        exactly the D4+gCP sum, and an adopted solution would carry a
        dispersion-free energy.'''
        from types import MethodType

        def _fake_get_dispersion(mf_self, disp=None, with_3body=None,
                                 verbose=None):
            mf_self.scf_summary['dispersion'] = -0.0525
            return -0.0525

        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        # reproduce the runner's attach shape exactly: bound MethodType
        # on the FINAL object (attach_3c_corrections) + do_disp override
        mf.get_dispersion = MethodType(_fake_get_dispersion, mf)
        mf.do_disp = lambda: True
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        # same-solution cell: record energies must agree to the SCF
        # noise floor -- before the fix ref_e_tot was high by +0.0525
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], rec['main_e_tot'],
                               delta=1e-8)
        # the kept energy still contains the dispersion piece
        self.assertAlmostEqual(float(mf.e_tot),
                               rec['main_e_tot'], delta=1e-8)
        self.assertAlmostEqual(mf.scf_summary['dispersion'], -0.0525,
                               delta=1e-12)

    def test_sibling_rebinds_hook_to_sibling_object(self):
        '''the carried get_dispersion must be rebound to the SIBLING, not
        left bound to the main object: energy_tot reads the dispersion
        from mf.scf_summary of the object it runs on, and the sibling's
        e_tot is what verify records compare/adopt.'''
        from types import MethodType
        seen = {}

        def _fake_get_dispersion(mf_self, disp=None, with_3body=None,
                                 verbose=None):
            seen[id(mf_self)] = True
            mf_self.scf_summary['dispersion'] = -0.0525
            return -0.0525

        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.get_dispersion = MethodType(_fake_get_dispersion, mf)
        mf.do_disp = lambda: True
        mf.kernel()
        # the sibling (a distinct object from mf) must have invoked the
        # hook on itself: its id is recorded and differs from mf's
        self.assertTrue(any(i != id(mf) for i in seen))
        rec = mf.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], rec['main_e_tot'],
                               delta=1e-8)

    def test_grad_resolve_auto_verify(self):
        '''gradients on an auto+verify object resolve to the fp32 kernel
        lane (verification selects the solution, not the kernel lane)'''
        from gpu4pyscf.df.grad import uhf as df_uhf_grad
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        g = mf.nuc_grad_method()
        self.assertEqual(g._resolve_precision(), 'fp32')
        # explicit override still wins
        g.precision_mode = 'fp64'
        self.assertEqual(g._resolve_precision(), 'fp64')


class _FakeSol:
    '''minimal solution stand-in for the occupied-space overlap helper'''
    def __init__(self, mo_coeff, mo_occ, ovlp):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._ovlp = ovlp

    def get_ovlp(self):
        return self._ovlp


class SiblingCompletenessTests(unittest.TestCase):
    '''The verify sibling must carry EVERY energy-affecting instance knob
    of the main object.  LC-57b found the runner's instance-attached
    dispersion hooks dropped (silent D4+gCP omission in ref_e_tot); LC-58
    found the SAME defect class on the native upstream API: mf.disp
    (native d3bj/d4, sibling ref high by exactly +e_disp = 5.7388e-4 Ha on
    water/b3lyp/d3bj) and mf.nlc overrides (nlc='vv10' on a non-NLC
    functional CRASHED the sibling with AssertionError in df_jk.get_veff).
    The four-family parity audit below is the mechanism-agnostic check
    that would have caught either defect on any family: the sibling's
    reference energy must reproduce an independently constructed
    plain-fp64 lane of the same method.'''

    def _independent_fp64(self, build):
        mf = build()
        mf.precision_mode = 'fp64'
        mf.conv_tol = 1e-10
        return mf.kernel()

    def test_sibling_carries_native_disp(self):
        '''mf.disp = 'd3bj' (native upstream dispersion API): the verify
        reference must include the dispersion sum.  Before LC-58 the
        sibling dropped mf.disp and ref_e_tot was high by exactly e_disp.'''
        def build():
            mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
            mf.disp = 'd3bj'
            return mf
        e_ind = self._independent_fp64(build)
        mf = build()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], e_ind, delta=1e-9)
        # the correction itself must be nonzero on this cell (the audit
        # must not pass vacuously through a zero correction)
        self.assertLess(mf.scf_summary['dispersion'], -1e-4)

    def test_sibling_carries_nlc_override_off(self):
        '''nlc=0 forces VV10 off on an NLC functional: the sibling must
        apply the same override (state carry, not the old bound-method
        copy).'''
        def build():
            mf = dft.RKS(mol_sph, xc='wb97m-v').density_fit()
            mf.nlc = 0
            return mf
        e_ind = self._independent_fp64(build)
        mf = build()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], e_ind, delta=1e-9)

    def test_sibling_carries_nlc_override_vv10_no_crash(self):
        '''nlc='vv10' forces VV10 on a non-NLC functional.  Before LC-58
        the sibling kept do_nlc bound to the main object while its local
        nlc state stayed '' -> AssertionError in df_jk.get_veff: a
        user-reachable auto+verify CRASH.  The sibling must carry the
        override and reproduce the independent lane.'''
        def build():
            mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
            mf.nlc = 'vv10'
            return mf
        e_ind = self._independent_fp64(build)
        mf = build()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], e_ind, delta=1e-9)

    def test_sibling_energy_parity_all_four_families(self):
        '''Mechanism-agnostic completeness audit across the four in-scope
        method families (+ UKS): for each family the auto+verify
        reference energy must match an independently constructed
        plain-fp64 lane of the SAME method to the SCF noise floor.  Any
        dropped correction/knob (the LC-57b dispersion hooks, the LC-58
        native disp/nlc state) breaks this parity on its family.'''
        from types import MethodType
        from pyscf.dispersion import dftd4, gcp

        # r2SCAN-3c family: runner-style instance hooks with REAL D4+gCP
        # (deterministic geometry-only corrections; same hook on both
        # lanes, so the parity test is about the CARRY, not the params)
        def _d4_gcp_hook(mf, disp=None, with_3body=None, verbose=None):
            e_d4 = float(dftd4.DFTD4Dispersion(
                mf.mol, xc='r2scan-3c').get_dispersion()['energy'])
            e_gcp = float(gcp.GCP(
                mf.mol, method='r2scan3c').get_counterpoise()['energy'])
            return e_d4 + e_gcp

        def attach_3c(mf):
            mf.nlc = 0
            mf.get_dispersion = MethodType(_d4_gcp_hook, mf)
            mf.do_disp = lambda: True
            return mf

        families = [
            ('b3lyp', lambda: dft.RKS(mol_sph, xc='b3lyp').density_fit()),
            ('m062x', lambda: dft.RKS(mol_sph, xc='m062x').density_fit()),
            ('wb97m-v', lambda: dft.RKS(mol_sph, xc='wb97m-v').density_fit()),
            ('r2scan-3c', lambda: attach_3c(
                dft.RKS(mol_sph, xc='r2scan').density_fit())),
            ('b3lyp-uks', lambda: dft.UKS(mol_uks, xc='b3lyp').density_fit()),
        ]
        for name, build in families:
            with self.subTest(family=name):
                e_ind = self._independent_fp64(build)
                mf = build()
                mf.conv_tol = 1e-10
                mf.precision_mode = 'auto+verify'
                mf.kernel()
                rec = mf.scf_summary['stability_verification']
                self.assertFalse(rec['distinct_solutions'],
                                 msg=f'{name}: unexpected branch split')
                self.assertAlmostEqual(rec['ref_e_tot'], e_ind, delta=1e-8,
                                       msg=f'{name}: sibling reference '
                                           f'energy parity broken')


class OccOverlapTests(unittest.TestCase):
    '''_occ_overlap_min: same occupied space -> ~1; a swapped occupied
    orbital -> << 1 (the measured beta swap signature)'''

    def _rks_pair(self):
        n = 4
        c = np.eye(n)
        occ = np.array([2., 2., 0., 0.])
        s = np.eye(n)
        same = _FakeSol(c, occ, s)
        # occupied space {0,1} vs {0,2}: one orbital swapped
        swapped = _FakeSol(c[:, [0, 2, 1, 3]], occ, s)
        return same, swapped

    def test_rks_same_space_is_one(self):
        same, _ = self._rks_pair()
        self.assertAlmostEqual(stability_verify._occ_overlap_min(same, same),
                               1.0, delta=1e-12)

    def test_rks_swapped_orbital_is_small(self):
        same, swapped = self._rks_pair()
        self.assertLess(stability_verify._occ_overlap_min(same, swapped),
                        0.1)

    def test_uks_per_spin(self):
        n = 4
        c = np.eye(n)
        s = np.eye(n)
        occ = np.array([[1., 1., 0., 0.]] * 2)
        same = _FakeSol([c, c], occ, s)
        # beta occupied space swapped, alpha unchanged
        cb = c[:, [0, 2, 1, 3]]
        other = _FakeSol([c, cb], occ, s)
        ov = stability_verify._occ_overlap_min(same, other)
        self.assertIsInstance(ov, dict)
        self.assertAlmostEqual(ov['alpha'], 1.0, delta=1e-12)
        self.assertLess(ov['beta'], 0.1)

    def test_mismatched_occupation_returns_none(self):
        n = 4
        c = np.eye(n)
        s = np.eye(n)
        a = _FakeSol(c, np.array([2., 2., 0., 0.]), s)
        b = _FakeSol(c, np.array([2., 0., 0., 0.]), s)
        self.assertIsNone(stability_verify._occ_overlap_min(a, b))


class FailClosedReferenceGates(unittest.TestCase):
    '''LC-64 (independent review 2026-10-07): an unconverged or nonfinite
    reference/refinement is NOT a stationary point.  Before the fix an
    unconverged lower-energy reference could be adopted over a converged
    main solution while the record claimed
    adopted_stationarity_verified=True.  Fail-closed semantics: never
    compare against, conclude same-solution from, or adopt such a
    candidate; keep the converged main solution and flag the exact
    reason.  These tests mock the sibling construction (no SCF cost) and
    exercise the selection logic directly.'''

    def _point(self, converged, energy):
        dm = np.eye(2)
        return SimpleNamespace(
            mol=Mock(), converged=converged, e_tot=energy, mo_coeff=dm,
            mo_energy=np.ones(2), mo_occ=np.ones(2), scf_summary={},
            cycles=2, init_guess='minao',
            kernel=Mock(return_value=energy),
            make_rdm1=Mock(return_value=dm),
            get_init_guess=Mock(return_value=dm))

    def _run(self, main, reference, refinement, dm_diffs):
        with mock.patch.object(
                stability_verify, '_build_fp64_sibling',
                side_effect=[reference, refinement]), \
             mock.patch.object(
                stability_verify, '_orbital_grad_norm',
                return_value=0.), \
             mock.patch.object(
                stability_verify, '_occ_overlap_min', return_value=1.), \
             mock.patch.object(
                stability_verify, '_max_abs_dm_diff',
                side_effect=dm_diffs):
            return stability_verify.verify_solution_stability(main)

    def test_unconverged_reference_not_adopted(self):
        '''the independent-review scenario: converged main, UNCONVERGED
        lower-energy reference, converged same-solution refinement.
        Pre-fix the reference was adopted, overwriting the main state
        with converged=False while claiming verification.'''
        main = self._point(True, -1.)
        reference = self._point(False, -2.)
        refinement = self._point(True, -1.)
        record = self._run(main, reference, refinement, [.1, 0., .1])
        self.assertTrue(main.converged,
                        'an unconverged reference replaced a converged '
                        'main solution')
        self.assertEqual(main.e_tot, -1.)
        self.assertIs(record['adopted_stationarity_verified'], False)
        self.assertEqual(record['selected'], 'main')
        self.assertEqual(record['verification_incomplete_reason'],
                         'reference_not_converged')

    def test_nonfinite_reference_energy_not_adopted(self):
        main = self._point(True, -1.)
        reference = self._point(True, float('nan'))
        refinement = self._point(True, -1.)
        record = self._run(main, reference, refinement, [.1, 0., .1])
        self.assertTrue(main.converged)
        self.assertEqual(main.e_tot, -1.)
        self.assertIs(record['adopted_stationarity_verified'], False)
        self.assertEqual(record['verification_incomplete_reason'],
                         'reference_energy_nonfinite')

    def test_unconverged_reference_never_same_solution(self):
        '''the same-density early return must be UNREACHABLE with an
        unconverged reference: the fail-closed reference gate fires
        before any density comparison (mocked to 0.0 here)'''
        main = self._point(True, -1.)
        reference = self._point(False, -2.)
        refinement = self._point(True, -1.)
        record = self._run(main, reference, refinement, [0., 0., 0.])
        self.assertIs(record['distinct_solutions'], None)
        self.assertEqual(record['selected'], 'main')
        self.assertIs(record['dm_max_abs_diff'], None)
        self.assertIs(record['adopted_stationarity_verified'], False)

    def test_nonfinite_dm_comparison_keeps_main(self):
        main = self._point(True, -1.)
        reference = self._point(True, -1.5)
        refinement = self._point(True, -1.)
        record = self._run(main, reference, refinement,
                           [float('inf'), 0., 0.])
        self.assertTrue(main.converged)
        self.assertEqual(main.e_tot, -1.)
        self.assertIs(record['adopted_stationarity_verified'], False)
        self.assertEqual(record['verification_incomplete_reason'],
                         'dm_comparison_nonfinite')

    def test_nonconverged_refinement_falls_back_honestly(self):
        '''refinement did not converge: raw two-trajectory rule may adopt
        the CONVERGED reference (lower here) but must never claim
        stationarity verification, and records the unusability reason'''
        main = self._point(True, -1.)
        reference = self._point(True, -1.5)
        refinement = self._point(False, -1.)
        record = self._run(main, reference, refinement, [.5, .5, .5])
        self.assertEqual(record['refinement'], 'non_converged')
        self.assertEqual(record['refine_unusable'], 'not_converged')
        self.assertIs(record['adopted_stationarity_verified'], False)
        # fallback rule: reference is lower AND converged -> adoptable
        self.assertEqual(record['selected'], 'reference')
        self.assertEqual(main.e_tot, -1.5)
        self.assertTrue(main.converged)

    def test_nonfinite_refinement_energy_falls_back(self):
        main = self._point(True, -1.)
        reference = self._point(True, -0.5)
        refinement = self._point(True, float('nan'))
        record = self._run(main, reference, refinement, [.5, .5, .5])
        self.assertEqual(record['refinement'], 'non_converged')
        self.assertEqual(record['refine_unusable'], 'energy_nonfinite')
        self.assertIs(record['adopted_stationarity_verified'], False)
        # fallback rule: main is lower -> kept
        self.assertEqual(record['selected'], 'main')
        self.assertEqual(main.e_tot, -1.)

    def test_nonfinite_main_energy_skips_verification(self):
        '''a converged main SCF with nonfinite e_tot skips verification
        with an explicit record instead of crashing the kernel'''
        main = self._point(True, float('nan'))
        with mock.patch.object(stability_verify, '_build_fp64_sibling'):
            record = stability_verify.verify_solution_stability(main)
        self.assertEqual(record['skipped'],
                         'main SCF energy is nonfinite')

    def test_adopt_refuses_unconverged_or_nonfinite(self):
        '''structural tripwire: _adopt itself refuses an unconverged or
        nonfinite source so a future call-site regression fails loudly'''
        mf = self._point(True, -1.)
        for bad in (self._point(False, -2.), self._point(True, float('nan'))):
            with self.assertRaises(RuntimeError):
                stability_verify._adopt(mf, bad)


class CustomGridCarryTests(unittest.TestCase):
    '''LC-64 (independent review 2026-10-07): the fp64 sibling must carry
    the full grid-DEFINITION configuration, not only grids.level.  A
    custom atom_grid=(20,74)/prune=None silently reverted to class
    defaults in the sibling (sibling.grids.atom_grid == {} != (20,74)),
    breaking the IDENTICAL-settings reference comparison.'''

    def test_custom_grid_definition_survives_sibling_construction(self):
        mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
                      output='/dev/null', verbose=0)
        try:
            mf = dft.RKS(mol, xc='wb97m-v').density_fit()
            mf.grids.atom_grid = (20, 74)
            mf.grids.prune = None
            mf.nlcgrids.atom_grid = (10, 50)
            mf.nlcgrids.prune = None
            sibling = stability_verify._build_fp64_sibling(mf)
            self.assertEqual(sibling.grids.atom_grid, (20, 74))
            self.assertIs(sibling.grids.prune, None)
            self.assertEqual(sibling.nlcgrids.atom_grid, (10, 50))
            self.assertIs(sibling.nlcgrids.prune, None)
            # radii/quadrature selectors are carried as well
            self.assertEqual(sibling.grids.level, mf.grids.level)
            self.assertIs(sibling.grids.radi_method, mf.grids.radi_method)
            self.assertIs(sibling.grids.radii_adjust,
                          mf.grids.radii_adjust)
            self.assertIs(sibling.grids.becke_scheme,
                          mf.grids.becke_scheme)
        finally:
            mol.stdout.close()
            del mol

    def test_grid_definition_dict_is_copied_by_value(self):
        '''mutation of the main object's atom_grid dict after sibling
        construction must not reach the sibling (no shared mutable
        container)'''
        mol = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
                      output='/dev/null', verbose=0)
        try:
            mf = dft.RKS(mol, xc='b3lyp').density_fit()
            mf.grids.atom_grid = {'H': (20, 74)}
            sibling = stability_verify._build_fp64_sibling(mf)
            self.assertEqual(sibling.grids.atom_grid, {'H': (20, 74)})
            mf.grids.atom_grid['H'] = (5, 10)
            self.assertEqual(sibling.grids.atom_grid, {'H': (20, 74)},
                             'sibling shares the mutable atom_grid dict')
        finally:
            mol.stdout.close()
            del mol

    def test_custom_grid_reference_parity(self):
        '''the sibling reference energy on CUSTOM grids must match an
        independently constructed plain-fp64 lane with the SAME custom
        grids (the identical-settings contract).  Pre-fix the sibling
        reverted to default grids and ref_e_tot was off by the grid
        difference.  The default-vs-custom energy gap assert proves the
        custom grids actually changed the quadrature (no vacuous
        parity).'''
        def build():
            mf = dft.RKS(mol_sph, xc='wb97m-v').density_fit()
            mf.conv_tol = 1e-10
            mf.grids.atom_grid = (20, 74)
            mf.grids.prune = None
            mf.nlcgrids.atom_grid = (10, 50)
            mf.nlcgrids.prune = None
            return mf

        e_custom = self._independent_fp64(build)
        e_default = self._independent_fp64(
            lambda: dft.RKS(mol_sph, xc='wb97m-v').density_fit())
        self.assertGreater(abs(e_custom - e_default), 1e-7,
                           'custom grids did not change the energy; the '
                           'parity test would be vacuous')

        mf = build()
        mf.precision_mode = 'auto+verify'
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertFalse(rec['distinct_solutions'])
        self.assertAlmostEqual(rec['ref_e_tot'], e_custom, delta=1e-9,
                               msg='sibling reference energy does not '
                                   'match the custom-grid fp64 lane')

    @staticmethod
    def _independent_fp64(build):
        mf = build()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'fp64'
        return mf.kernel()

    def test_gradient_after_incomplete_verification(self):
        '''gradients on an incomplete-verification object run on the
        KEPT (converged) solution without crashing: the dm test is
        forced to fire and the refinement is bound to 0 cycles, so the
        record honestly says adopted_stationarity_verified=False'''
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = 'auto+verify'
        with mock.patch.object(
                stability_verify, '_max_abs_dm_diff',
                return_value=0.5), \
             mock.patch.object(
                stability_verify, 'stability_verify_refine_max_cycle', 0):
            mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertEqual(rec['refinement'], 'non_converged')
        self.assertIs(rec['adopted_stationarity_verified'], False)
        self.assertTrue(mf.converged)
        g = mf.nuc_grad_method().kernel()
        self.assertTrue(np.all(np.isfinite(g)))
        self.assertEqual(g.shape, (mol_uks.natm, 3))


if __name__ == '__main__':
    print('Full tests for SCF solution-stability verification')
    unittest.main()

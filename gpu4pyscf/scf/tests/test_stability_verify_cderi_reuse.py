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

import cupy
import numpy as np
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.lib import precision
from gpu4pyscf.scf import stability_verify


def setUpModule():
    global mol_sph, mol_methoxy
    # closed-shell control and the measured UKS two-basin split cell
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=1)
    mol_methoxy = pyscf.M(atom='''
O       0.0000000000     0.0000000000     0.0000000000
C       0.0000000000     0.0000000000     1.3680000000
H       0.0000000000     1.0276700000     1.7313110000
H      -0.8899880000    -0.5138350000     1.7313110000
H       0.8899880000    -0.5138350000     1.7313110000''',
                          basis='def2-svp', spin=1, charge=0,
                          output='/dev/null', verbose=1)


def tearDownModule():
    global mol_sph, mol_methoxy
    mol_sph.stdout.close()
    del mol_sph
    mol_methoxy.stdout.close()
    del mol_methoxy


class CderiReuseDecisionTests(unittest.TestCase):
    '''_shareable_cderi: the sharing preconditions, checked as pure logic
    on stand-in DF objects (no SCF cost).  Any refusal reason falls back
    to the sibling's own fresh build -- the pre-MR-8 behaviour.'''

    def _dfobj(self, tensors, cderi_idx=(np.array([0]), np.array([0.])),
               rsh=None):
        return SimpleNamespace(
            _cderi=tensors, _cderi_idx=cderi_idx,
            _rsh_df=rsh if rsh is not None else {})

    def test_built_fp64_list_is_shareable(self):
        # the production storage form: a per-device LIST of float64
        # tensors (df._cholesky_eri returns a list; df_jk indexes by
        # device id)
        obj = self._dfobj([cupy.zeros(3, dtype=cupy.float64)])
        self.assertIsNone(stability_verify._shareable_cderi(obj))

    def test_built_fp64_dict_is_shareable(self):
        # the deprecated cholesky_eri_gpu form: device-keyed dict
        obj = self._dfobj({0: cupy.zeros(3, dtype=cupy.float64)})
        self.assertIsNone(stability_verify._shareable_cderi(obj))

    def test_built_fp64_array_is_shareable(self):
        obj = self._dfobj(cupy.zeros(3, dtype=cupy.float64))
        self.assertIsNone(stability_verify._shareable_cderi(obj))

    def test_unbuilt_is_refused(self):
        self.assertEqual(
            stability_verify._shareable_cderi(self._dfobj(None)),
            'cderi_not_built')
        # tensor present but the pair index missing (half-built state)
        obj = SimpleNamespace(_cderi=[cupy.zeros(1)], _cderi_idx=None,
                              _rsh_df={})
        self.assertEqual(stability_verify._shareable_cderi(obj),
                         'cderi_not_built')

    def test_fp32_tensor_is_refused(self):
        obj = self._dfobj([cupy.zeros(3, dtype=cupy.float32)])
        self.assertEqual(stability_verify._shareable_cderi(obj),
                        'tensor_not_fp64')
        # a float32 tensor in a range-separated sub-object is caught too
        rsh_obj = self._dfobj([cupy.zeros(3, dtype=cupy.float32)])
        obj = self._dfobj([cupy.zeros(3, dtype=cupy.float64)],
                           rsh={'0.300000': rsh_obj})
        self.assertEqual(stability_verify._shareable_cderi(obj),
                        'tensor_not_fp64')

    def test_process_cderi_fp32_flag_is_refused(self):
        # the process-global screening knob alone refuses sharing even
        # with an fp64 tensor stored (belt and suspenders)
        obj = self._dfobj({0: cupy.zeros(3, dtype=cupy.float64)})
        precision.set_cderi_precision('fp32')
        try:
            self.assertEqual(stability_verify._shareable_cderi(obj),
                             'process_cderi_not_fp64')
        finally:
            precision.set_cderi_precision('fp64')


class SiblingSharesDfObjectTests(unittest.TestCase):
    '''MR-8: when the main run's CDERI is a built fp64 tensor, the strict
    fp64 sibling reuses the main's DF OBJECT verbatim (density_fit's
    documented with_df= API) instead of building a duplicate 3c2e
    Cholesky; on any refusal the sibling builds its own as before.'''

    def _converged_main(self, xc='b3lyp', mol=None, mode='auto'):
        mf = dft.RKS(mol or mol_sph, xc=xc).density_fit()
        mf.conv_tol = 1e-10
        mf.precision_mode = mode
        mf.kernel()
        self.assertTrue(mf.converged)
        self.assertIsNotNone(mf.with_df._cderi)
        return mf

    def test_sibling_reuses_main_df_object(self):
        mf = self._converged_main()
        stats0 = dict(stability_verify.sibling_df_reuse_stats)
        sibling = stability_verify._build_fp64_sibling(mf)
        self.assertIs(sibling.with_df, mf.with_df,
                      'sibling rebuilt a DF object instead of reusing '
                      'the main run\'s fp64 CDERI')
        self.assertEqual(
            stability_verify.sibling_df_reuse_stats['shared'],
            stats0['shared'] + 1)
        # the sibling is still the strict fp64 reference
        self.assertEqual(sibling.precision_mode, 'fp64')
        self.assertTrue(sibling._strict_fp64_reference)

    def test_sibling_refuses_under_process_cderi_fp32(self):
        mf = self._converged_main()
        stats0 = dict(stability_verify.sibling_df_reuse_stats)
        precision.set_cderi_precision('fp32')
        try:
            sibling = stability_verify._build_fp64_sibling(mf)
        finally:
            precision.set_cderi_precision('fp64')
        self.assertIsNot(sibling.with_df, mf.with_df,
                         'an fp32-screening process must not share the '
                         'DF object with the strict fp64 sibling')
        self.assertEqual(
            stability_verify.sibling_df_reuse_stats['built'],
            stats0['built'] + 1)
        # the fresh sibling carries the same auxbasis (direct_scf_tol when
        # the main object has one, matching the guarded copy in the code)
        self.assertEqual(sibling.with_df.auxbasis, mf.with_df.auxbasis)
        tol_main = getattr(mf.with_df, 'direct_scf_tol', 'unset')
        tol_sib = getattr(sibling.with_df, 'direct_scf_tol', 'unset')
        if tol_main != 'unset':
            self.assertEqual(tol_sib, tol_main)

    def test_sibling_refuses_fp32_stored_tensor(self):
        mf = self._converged_main()
        saved = mf.with_df._cderi
        saved_idx = mf.with_df._cderi_idx
        try:
            mf.with_df._cderi = [v.astype(cupy.float32) for v in saved]
            sibling = stability_verify._build_fp64_sibling(mf)
            self.assertIsNot(sibling.with_df, mf.with_df)
        finally:
            mf.with_df._cderi = saved
            mf.with_df._cderi_idx = saved_idx

    def test_sibling_refuses_unbuilt_df(self):
        # a main whose DF never built a tensor (no kernel run): the old
        # fresh-build path, not sharing an empty object
        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit()
        sibling = stability_verify._build_fp64_sibling(mf)
        self.assertIsNot(sibling.with_df, mf.with_df)
        self.assertEqual(sibling.with_df.auxbasis, mf.with_df.auxbasis)

    def test_deterministic_cderi_build_bitwise(self):
        '''the sharing premise, measured directly: two independent DF
        builds from the same mol/auxbasis reproduce the CDERI bitwise, so
        a shared tensor is exactly the tensor the sibling would have
        built (and exactly what the plain default lane builds).'''
        from gpu4pyscf.df import df as dfmod
        df1 = dfmod.DF(mol_sph)
        df1.build()
        df2 = dfmod.DF(mol_sph)
        df2.build()
        self.assertEqual(len(df1._cderi), len(df2._cderi))
        for k in range(len(df1._cderi)):
            self.assertTrue(
                cupy.array_equal(df1._cderi[k], df2._cderi[k]),
                msg=f'device {k}: CDERI build is not deterministic; '
                    'sharing would not be bitwise-neutral')


class SharedSiblingBitwiseParityTests(unittest.TestCase):
    '''The soundness core of MR-8: with sharing enabled (default) the
    verification trajectories must be numerically IDENTICAL to the
    fresh-build sibling -- reference energies bitwise equal, decisions
    identical -- on both a closed-shell cell and the measured UKS
    two-basin split cell (where the reference is ADOPTED, so any sharing
    defect would move the returned solution).'''

    def _run_auto_strict(self, xc, mol, disable_sharing):
        mf = (dft.UKS(mol, xc=xc) if mol is mol_methoxy
              else dft.RKS(mol, xc=xc)).density_fit()
        mf.conv_tol = 1e-10
        mf.init_guess = 'minao'
        mf.precision_mode = 'auto+strict'
        ctx = (mock.patch.object(stability_verify, '_shareable_cderi',
                                 return_value='disabled_by_test')
               if disable_sharing else
               mock.patch.object(stability_verify, '_shareable_cderi',
                                 return_value=None))
        with ctx:
            e = mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertIs(rec['strict_match_verified'], True)
        return float(e), rec

    def test_shared_matches_fresh_bitwise(self):
        '''Sharing must not change the verification outcome.  On the
        closed-shell cell the trajectories reproduce BITWISE.  On the
        wb97m-v UKS cell bitwise equality is not attainable even
        fresh-vs-fresh (measured 2026-10-22: two DISABLED-sharing runs
        differ by 3e-14 in ref_e_tot and the auto main energy itself is
        not bitwise run-to-run -- cross-run GPU nondeterminism on the
        range-separated path, not a sharing effect), so the honest
        assertion there is the SCF noise floor (<= 1e-12, ~5 orders
        below the 1e-6 gate) plus IDENTICAL decisions; a genuinely wrong
        shared tensor would move the energies by orders of magnitude
        more than either.'''
        for name, xc, mol, tol in (
                ('closed-shell b3lyp/water', 'b3lyp', mol_sph, 0.0),
                ('UKS split wb97m-v/methoxy', 'wb97m-v', mol_methoxy, 1e-12)):
            with self.subTest(cell=name):
                e_shared, rec_shared = self._run_auto_strict(
                    xc, mol, disable_sharing=False)
                e_fresh, rec_fresh = self._run_auto_strict(
                    xc, mol, disable_sharing=True)
                if tol == 0.0:
                    self.assertEqual(rec_shared['ref_e_tot'],
                                     rec_fresh['ref_e_tot'])
                    self.assertEqual(e_shared, e_fresh)
                else:
                    self.assertAlmostEqual(rec_shared['ref_e_tot'],
                                           rec_fresh['ref_e_tot'], delta=tol)
                    self.assertAlmostEqual(e_shared, e_fresh, delta=tol)
                self.assertEqual(rec_shared['distinct_solutions'],
                                 rec_fresh['distinct_solutions'])
                self.assertEqual(rec_shared['selected'], rec_fresh['selected'])
                self.assertEqual(rec_shared['strict_match_verified'],
                                 rec_fresh['strict_match_verified'])
                self.assertAlmostEqual(rec_shared['main_e_tot'],
                                       rec_fresh['main_e_tot'], delta=tol)

    def test_shared_route_runs_and_uses_sharing(self):
        # the full auto+strict kernel path with sharing active: the stats
        # record the reuse and the gradient still runs on the adopted
        # solution through the shared DF object afterwards
        mf = dft.UKS(mol_methoxy, xc='wb97m-v').density_fit()
        mf.conv_tol = 1e-10
        mf.init_guess = 'minao'
        mf.precision_mode = 'auto+strict'
        stats0 = dict(stability_verify.sibling_df_reuse_stats)
        mf.kernel()
        rec = mf.scf_summary['stability_verification']
        self.assertIs(rec['strict_match_verified'], True)
        self.assertEqual(rec['selected'], 'reference')
        self.assertGreater(
            stability_verify.sibling_df_reuse_stats['shared'], stats0['shared'])
        g = mf.nuc_grad_method().kernel()
        self.assertTrue(np.all(np.isfinite(g)))


if __name__ == '__main__':
    print('MR-8 tests: strict-fp64 sibling CDERI reuse')
    unittest.main()

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
'''GRAD-J3C-REUSE-JONLY-01 focused regression (2026-11-10).

The opt-in GPU4PYSCF_DF_GRAD_J3C_REUSE knob (DEFAULT OFF) is extended to
the pure-functional J-only DF-gradient route: when the retained-cderi
pack is armed and k_factor == 0 (hyb == 0 functionals, e.g. r2scan3c),
the early return to _j_energy_per_atom (which re-evaluates the full 3c
stream via contract_dm + a fresh j2c build + CD/ED metric solve) is
bypassed and the J derivative flows through the joint machinery, where
every exchange contribution is identically zero -- the same
production-verified pack construction hybrids use (GRAD-J3C-REUSE-01).

Equivalence basis (Phase-A probe, 63-AO r2scan cell): armed k_factor==0
joint route vs verbatim J-only max|dg| = 5.4e-13 Eh/Bohr (the shared
metric-route reassociation class), e_tot bitwise (the SCF path is
untouched).  At small cells the pack construction can be SLOWER than
the verbatim J-only path (0.17 vs 0.13 s/call at 63 AO) -- the knob
stays opt-in; the band-cell benefit is Phase B's paired measurement.

Tests: parse/default-off, knob-off verbatim (J-only route never armed),
armed engagement (both counters, pure MGGA + pure GGA cells), armed
equivalence envelope, admission-failure verbatim (no cderi retained),
hybrid control (J-only counter stays 0; the deployed K-chain route
unchanged), j_factor==0 / hermi==2 guards, strict-fp64-reference
inertness, malformed-knob warn, and env-free source-structure
contracts.  Gradient equality is banded, not bitwise (the ip1-kernel
atomicAdd ULP run-to-run scatter class; band 1e-9 Eh/Bohr at water
scale, four orders under the 3e-5 project gate).
'''
import inspect
import os
import unittest
import warnings

import numpy as np
import pyscf

from gpu4pyscf.df.grad import rhf as df_rhf_grad

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''


def _reset_module_caches():
    df_rhf_grad._J3C_REUSE = None


def _jonly_counter():
    return getattr(df_rhf_grad, '_J3C_REUSE_JONLY_GRADS', 0)


def _kchain_counter():
    return getattr(df_rhf_grad, '_J3C_REUSE_GRADS', 0)


def _run_rks(xc, env_knob, pre_build, grids_level=1):
    '''One fresh SCF + analytic gradient with the knob env set BEFORE the
    module cache resolves; returns (e_tot, grad, dm-ref, mf).'''
    os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = env_knob
    _reset_module_caches()
    mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                  verbose=0)
    from gpu4pyscf.dft import rks
    mf = rks.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.grids.level = grids_level
    mf.max_cycle = 100
    mf.conv_tol = 1e-11
    e_tot = mf.kernel()
    if pre_build:
        mf.with_df.build()
    grad = np.asarray(mf.nuc_grad_method().kernel())
    return e_tot, grad, mf


class TestParseAndDefaultOff(unittest.TestCase):
    '''Env-free contracts: the knob still parses exactly as before and the
    new J-only counter/helper exist in the home module.'''

    def test_default_off_and_malformed_warn(self):
        _reset_module_caches()
        os.environ.pop('GPU4PYSCF_DF_GRAD_J3C_REUSE', None)
        self.assertFalse(df_rhf_grad._j3c_reuse())
        _reset_module_caches()
        os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = 'x'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertFalse(df_rhf_grad._j3c_reuse())
        self.assertTrue(any('ignored' in str(x.message) for x in w))
        _reset_module_caches()
        os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = '0'
        self.assertFalse(df_rhf_grad._j3c_reuse())
        _reset_module_caches()

    def test_jonly_counter_and_helper_exist(self):
        self.assertIsInstance(_jonly_counter(), int)
        self.assertTrue(callable(
            getattr(df_rhf_grad, '_note_j3c_reuse_jonly_engaged', None)))
        n0 = _jonly_counter()
        df_rhf_grad._note_j3c_reuse_jonly_engaged()
        self.assertEqual(_jonly_counter(), n0 + 1)
        # the helper touches the home-module namespace only (no fork class)
        import gpu4pyscf.df.grad.uhf as df_uhf_grad
        self.assertFalse(hasattr(df_uhf_grad, '_J3C_REUSE_JONLY_GRADS'))


class TestKnobOffVerbatim(unittest.TestCase):
    '''Knob OFF: the pure-functional gradient takes the verbatim J-only
    early return; neither reuse counter moves; run-to-run determinism is
    the banded ULP class (never bitwise for j3c/atomicAdd paths).'''

    def test_off_counters_and_determinism(self):
        # counters are cumulative process-lifetime evidence counters:
        # assert deltas around THIS test's runs, never absolute zeros
        n_j0 = _jonly_counter()
        n_k0 = _kchain_counter()
        e1, g1, _ = _run_rks('r2scan', '0', pre_build=True)
        e2, g2, _ = _run_rks('r2scan', '0', pre_build=True)
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertEqual(_kchain_counter(), n_k0)
        self.assertAlmostEqual(e1, e2, delta=1e-11)
        self.assertLessEqual(np.abs(g1 - g2).max(), 1e-11)


class TestArmedRoute(unittest.TestCase):
    '''Armed + cderi retained: the pure-functional gradient flows through
    the joint pack construction; both counters move; the gradient equals
    the knob-off verbatim within the reassociation band; e_tot is
    unaffected (the SCF path is untouched).'''

    def test_armed_pure_mgga(self):
        e_off, g_off, _ = _run_rks('r2scan', '0', pre_build=True)
        n_j0 = _jonly_counter()
        n_k0 = _kchain_counter()
        e_on, g_on, _ = _run_rks('r2scan', '1', pre_build=True)
        self.assertEqual(_jonly_counter() - n_j0, 1)
        self.assertGreaterEqual(_kchain_counter() - n_k0, 1)
        self.assertAlmostEqual(e_on, e_off, delta=1e-11)
        self.assertLessEqual(np.abs(g_on - g_off).max(), 1e-9)

    def test_armed_pure_gga(self):
        e_off, g_off, _ = _run_rks('pbe', '0', pre_build=True)
        n_j0 = _jonly_counter()
        e_on, g_on, _ = _run_rks('pbe', '1', pre_build=True)
        self.assertEqual(_jonly_counter() - n_j0, 1)
        self.assertAlmostEqual(e_on, e_off, delta=1e-11)
        self.assertLessEqual(np.abs(g_on - g_off).max(), 1e-9)

    def test_admission_failure_verbatim(self):
        # knob ON but no with_df.build(): the pure-functional SCF never
        # retains cderi, capture fails closed, the verbatim J-only early
        # return runs (counters unchanged; gradient == knob-off).
        e_off, g_off, _ = _run_rks('r2scan', '0', pre_build=True)
        n_j0 = _jonly_counter()
        n_k0 = _kchain_counter()
        e_on, g_on, _ = _run_rks('r2scan', '1', pre_build=False)
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertEqual(_kchain_counter(), n_k0)
        self.assertAlmostEqual(e_on, e_off, delta=1e-11)
        self.assertLessEqual(np.abs(g_on - g_off).max(), 1e-11)


class TestHybridControl(unittest.TestCase):
    '''Hybrid control: the armed route is the DEPLOYED K-chain behavior
    (entry 388); the J-only counter must stay 0 there (k_factor != 0).'''

    def test_hybrid_jonly_counter_zero(self):
        e_off, g_off, _ = _run_rks('b3lyp', '0', pre_build=True)
        n_j0 = _jonly_counter()
        n_k0 = _kchain_counter()
        e_on, g_on, mf = _run_rks('b3lyp', '1', pre_build=True)
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertGreaterEqual(_kchain_counter(), n_k0 + 1)
        self.assertAlmostEqual(e_on, e_off, delta=1e-11)
        self.assertLessEqual(np.abs(g_on - g_off).max(), 1e-6)


class TestGuards(unittest.TestCase):
    '''j_factor == 0 and hermi == 2 with the pack armed keep the verbatim
    early return (the fall-through is a pure-functional J route only).'''

    def test_jfactor0_hermi2_verbatim(self):
        os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = '1'
        _reset_module_caches()
        mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                      verbose=0)
        from gpu4pyscf.dft import rks
        mf = rks.RKS(mol)
        mf.xc = 'r2scan'
        mf = mf.density_fit()
        mf.grids.level = 1
        mf.max_cycle = 100
        mf.conv_tol = 1e-11
        mf.kernel()
        mf.with_df.build()
        g = mf.nuc_grad_method()
        dm = mf.make_rdm1()
        n_j0 = _jonly_counter()
        out0 = g.jk_energy_per_atom(dm, j_factor=0, k_factor=0, hermi=1)
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertTrue(np.allclose(np.asarray(out0), 0.0))
        out2 = g.jk_energy_per_atom(dm, j_factor=1, k_factor=0, hermi=2)
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertTrue(np.allclose(np.asarray(out2), 0.0))
        _reset_module_caches()

    def test_strict_reference_never_arms(self):
        # the capture wrapper refuses the strict fp64-reference context
        # (dft.numint.strict_fp64_reference_ctx, the stability_verify
        # sibling discipline): under it the pack is absent and the
        # verbatim J-only early return runs; counters must not move.
        from gpu4pyscf.dft import numint
        if not hasattr(numint, 'strict_fp64_reference_ctx'):
            self.skipTest('strict-reference context manager drifted')
        os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = '1'
        _reset_module_caches()
        mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                      verbose=0)
        from gpu4pyscf.dft import rks
        mf = rks.RKS(mol)
        mf.xc = 'r2scan'
        mf = mf.density_fit()
        mf.grids.level = 1
        mf.max_cycle = 100
        mf.conv_tol = 1e-11
        mf.kernel()
        mf.with_df.build()
        n_j0 = _jonly_counter()
        n_k0 = _kchain_counter()
        with numint.strict_fp64_reference_ctx():
            np.asarray(mf.nuc_grad_method().kernel())
        self.assertEqual(_jonly_counter(), n_j0)
        self.assertEqual(_kchain_counter(), n_k0)
        _reset_module_caches()


class TestSourceContracts(unittest.TestCase):
    '''Env-free structural contracts: the guard ordering at the k_factor==0
    early return, the engagement hook, and the _j_energy_per_atom body
    untouched (absence contract: the J-only kernel path is NOT modified).'''

    def test_early_return_guard_structure(self):
        src = inspect.getsource(df_rhf_grad._jk_energy_per_atom)
        self.assertIn('if j3c_reuse is None or j_factor == 0:', src)
        self.assertIn('_note_j3c_reuse_jonly_engaged()', src)
        # the verbatim early return is retained before the fall-through
        self.assertIn('return _j_energy_per_atom(int3c2e_opt, dm, hermi,',
                      src)

    def test_j_energy_per_atom_untouched(self):
        src = inspect.getsource(df_rhf_grad._j_energy_per_atom)
        self.assertNotIn('j3c_reuse', src)
        self.assertIn('contract_dm', src)          # verbatim 3c stream stays
        self.assertIn('_gen_metric_solver', src)   # verbatim metric stays


if __name__ == '__main__':
    unittest.main()

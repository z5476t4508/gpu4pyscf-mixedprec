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
'''DFJK-RSOMEGA-DEFENSE-01 focused regression (2026-11-10).

Defensive one-line mirror of upstream gpu4pyscf #947 (fix 4c901bb) in
_DFHF.get_k (gpu4pyscf/df/df_jk.py): the ``lr_factor != sr_factor``
composite branch forwards ``omega=abs(omega)`` to the inner DF get_jk
call instead of the raw omega.  The sr_factor == 0 "Only LR" branch
stays raw, mirroring upstream tip exactly.

Semantics basis (measured at parent 70d5aac, water/def2-svp, source
env, /data/tmp/gpu4pyscf-agent/execution/rsomega01/probe_parent*.py):

* _check_rsh_factors (gpu4pyscf/scf/jk.py) normalizes every
  sr_factor != 0, omega != 0 triple to omega = -abs(omega) (libcint
  kernel convention), so the raw line forwarded a NEGATIVE omega for
  every explicit-factors call regardless of the caller's sign.
* The DF int3c2e layer is sign-sensitive (int3c2e_bdiv evaluator
  short_range=omega<0): with_df.get_jk(omega=-w) builds the
  short-range cderi (measured K_neg ~= K_full - K_pos to 1.2e-5, the
  DF sum-rule scatter), NOT the long-range operator.
* Therefore at parent the composite branch evaluated the wrong
  operator class for every lr_factor != sr_factor call: e.g.
  get_k(omega=-w) (factors None -> intended pure SR per the
  _check_rsh_factors contract) returned the LONG-RANGE operator
  (measured max|K_api_neg - K_pos| = 1.0e-5 vs max|K_api_neg -
  (K_full - K_pos)| = 8.46), and get_k(+0.3, 1.0, 0.15) composed
  0.15*K_full + 0.85*K_SR instead of the documented
  lr*erf(|w|r)/r + sr*erfc(|w|r)/r operator (measured 7.19 off).
* Corpus impact NONE, now mechanism-pinned (correcting the entry-517
  phrasing): the deployed wb97m_v SCF path never enters the composite
  branch.  MRO puts _DFHF before RKS, so RKS/UKS DF objects use
  _DFHF.get_veff, whose hybrid branch composes
  hyb*K_full + (alpha-hyb)*get_k(omega=abs(omega)) -- the sr_factor==0
  LR-only branch (df_jk.py:364 already carried abs()).  Deployed-lane
  byte-identity is pinned here by bitwise operator tests and by
  parent-pinned wb97m-v RKS/UKS SCF energies.
* Bonus reach: _DFHF.get_veff's GHF alpha==0 lane (df_jk.py:411)
  requests the SR operator via get_k(omega=-omega); at parent that
  lane received the LR operator (same inversion class); the mirror
  fixes it.  Out of corpus (no alpha==0 RSH in the benchmark set).

Adjacent raw sites deliberately NOT changed (mirror-only, upstream tip
parity): the sr_factor == 0 branch (df_jk.py:128), the UKS get_veff
site (:332, callers pass omega >= 0 from rsh_and_hybrid_coeff), and
the GHF get_veff sites (:411-420, explicit sign requests by design).

Focused RED at parent 70d5aac: 3 honest failures (source contract on
the abs() token, factors-None SR inversion, explicit-factors operator
composition) + contracts-by-design green (sign symmetry via
normalization equalization, LR-only branch identity, deployed SCF
energy pins), disclosed below per test.
'''
import inspect
import unittest

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import rks, uks

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''

W = 0.3  # wb97m_v omega
E_RKS_WB97MV_PARENT = -76.32546770238513  # parent 70d5aac pin, bitwise x2 runs
E_UKS_WB97MV_PARENT = -75.87203437817283  # parent 70d5aac pin, bitwise x2 runs


def _build():
    '''Small DF cell + init-guess dm; returns (mol, mf, dm).'''
    mol = pyscf.M(atom=WATER, basis='def2-svp', verbose=0)
    mf = rks.RKS(mol).density_fit()
    dm = cp.asarray(mf.get_init_guess())
    return mol, mf, dm


def _dmax(a, b):
    return float(cp.abs(a - b).max())


class TestSourceContracts(unittest.TestCase):
    '''Env-free structure pins. The abs() token contract is RED at
    parent; the raw-branch and get_veff pins are green by design
    (mirror-only scope).'''

    def test_get_k_composite_branch_carries_abs(self):
        src = inspect.getsource(df_jk._DFHF.get_k)
        self.assertIn(
            'vklr = self.get_jk(mol, dm, hermi, False, True, '
            'omega=abs(omega))[1]', src)

    def test_sr_zero_branch_stays_raw_upstream_parity(self):
        src = inspect.getsource(df_jk._DFHF.get_k)
        self.assertIn('if sr_factor == 0:', src)
        self.assertIn(
            'vk = self.get_jk(mol, dm, hermi, False, True, '
            'omega=omega)[1]', src)

    def test_deployed_get_veff_abs_site_untouched(self):
        src = inspect.getsource(df_jk._DFHF.get_veff)
        self.assertIn('self.get_k(mol, dm, hermi, omega=abs(omega))', src)


class TestOperatorSemantics(unittest.TestCase):
    '''Numerical operator contracts at the DF level.'''

    def test_factors_none_negative_omega_yields_sr(self):
        '''get_k(omega=-w) (no factors) must evaluate the SR operator
        K_full - K_LR per the _check_rsh_factors contract.  RED at
        parent: returns the LR operator instead (measured 8.46 off
        at this cell; 1.0e-5 agreement with K_pos).'''
        mol, mf, dm = _build()
        k_full = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True)[1]
        k_lr = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True,
                                 omega=W)[1]
        k_api_neg = mf.get_k(mol, dm, 1, -W)
        self.assertLessEqual(_dmax(k_api_neg, k_full - k_lr), 1e-12)

    def test_explicit_factors_compose_erf_contract(self):
        '''get_k(+w, lr=1.0, sr=0.15) must compose the documented
        operator sr*K_full + (lr-sr)*K_LR (same op order as the
        implementation).  RED at parent: composes with K_SR instead
        (measured 7.19 off).'''
        mol, mf, dm = _build()
        k_full = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True)[1]
        k_lr = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True,
                                 omega=W)[1]
        vk = mf.get_k(mol, dm, 1, W, 1.0, 0.15)
        ref = 0.15 * k_full + (1.0 - 0.15) * k_lr
        self.assertLessEqual(_dmax(vk, ref), 1e-12)

    def test_explicit_factors_sign_symmetry_bitwise(self):
        '''get_k(-w, lr, sr) == get_k(+w, lr, sr) bitwise.  GREEN AT
        PARENT BY DESIGN: _check_rsh_factors normalizes both signs to
        -abs(omega) before the branch, equalizing the calls; pinned as
        a regression contract (the fix must preserve it while making
        the shared value the correct one).'''
        mol, mf, dm = _build()
        k_pos = mf.get_k(mol, dm, 1, W, 1.0, 0.15)
        k_neg = mf.get_k(mol, dm, 1, -W, 1.0, 0.15)
        self.assertTrue(cp.array_equal(k_pos, k_neg))

    def test_lr_only_branch_unchanged_bitwise(self):
        '''The sr_factor == 0 LR-only branch (the branch the deployed
        wb97m_v SCF path actually exercises via get_veff's
        omega=abs(omega) call) is untouched: get_k(+w) == the inner
        DF K bitwise.  GREEN AT PARENT BY DESIGN (corpus byte-identity
        contract).'''
        mol, mf, dm = _build()
        k_lr = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True,
                                 omega=W)[1]
        k_api_pos = mf.get_k(mol, dm, 1, W)
        self.assertTrue(cp.array_equal(k_api_pos, k_lr))
        k_full = mf.with_df.get_jk(dm, 1, with_j=False, with_k=True)[1]
        k_full_2 = mf.get_jk(mol, dm, 1, False, True)[1]
        self.assertTrue(cp.array_equal(k_full, k_full_2))


class TestDeployedLanesByteIdentical(unittest.TestCase):
    '''Deployed wb97m-v SCF lanes pinned to parent energies.  GREEN AT
    PARENT BY DESIGN (the live path is _DFHF.get_veff, which never
    enters the patched branch); guards that the one-line mirror leaves
    default lanes byte-identical (parent pins measured bitwise x2).'''

    def test_rks_wb97mv_energy_pinned(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                      verbose=0)
        mf = rks.RKS(mol)
        mf.xc = 'wb97m_v'
        mf.grids.level = 1
        mf = mf.density_fit()
        e = mf.kernel()
        self.assertLessEqual(abs(e - E_RKS_WB97MV_PARENT), 1e-12)

    def test_uks_wb97mv_energy_pinned(self):
        mol = pyscf.M(atom=WATER, basis='def2-svp', spin=1, charge=1,
                      output='/dev/null', verbose=0)
        mf = uks.UKS(mol)
        mf.xc = 'wb97m_v'
        mf.grids.level = 1
        mf = mf.density_fit()
        e = mf.kernel()
        self.assertLessEqual(abs(e - E_UKS_WB97MV_PARENT), 1e-12)


if __name__ == '__main__':
    print('Full Tests for DFJK RS omega defense')
    unittest.main()

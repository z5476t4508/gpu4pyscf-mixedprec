# Copyright 2025-2026 The PySCF Developers. All Rights Reserved.
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

'''J3C-RTCACHE-01 Phase B focused regression (opt-in GPU4PYSCF_DF_J_3C_CACHE,
default off).  Numerical equivalence of the cached-table path against the
unmodified on-the-fly kernel path at the contract_dm / contract_auxvec seams,
end-to-end get_j, and a full DF-SCF; knob-off must leave the engine unlatched.
RED (pre-implementation): the knob cannot engage, so the engagement
assertions fail.  done != acceptance; review_pending.'''

import os
import numpy as np
import pyscf
from pyscf import lib
from gpu4pyscf.df import j_engine_3c2e
from gpu4pyscf.df.df_jk import get_j as df_get_j

TOL = 1e-10

FIXTURES = {
    'water_def2svp': dict(
        atom='''O 0 0 0.12; H 0 0.76 -0.45; H 0 -0.76 -0.45''',
        basis='def2-svp', auxbasis='def2-universal-jkfit'),
    'mixed_l_general': dict(
        atom='''C1 1.3 .2 .3; C2 .19 .1 1.1''',
        basis={'C1': [[3, [1.5, 1.], [.9, 1.]], [4, [2., 1.]]],
               'C2': 'ccpvdz'},
        auxbasis={'C1': '''C S
      2.9917624900           1.0000000000
C P
      3.3490545000           1.0000000
C P
      0.5769010900           1.0000000000
C D
      0.1995412500           1.0000000000''',
                  'C2': ('unc-weigend', [[0, [.5, 1.]], [1, [.8, 1.]], [3, [.9, 1]]])}),
}


def _engine_pair(name):
    fx = FIXTURES[name]
    mol = pyscf.M(atom=fx['atom'], basis=fx['basis'], verbose=0)
    auxmol = mol.copy()
    auxmol.basis = fx['auxbasis']
    auxmol.build(False, False)
    dm = pyscf.scf.RHF(mol).init_guess_by_minao()
    rng = np.random.default_rng(20261105)
    dm = dm + 1e-3 * (rng.random(dm.shape) + rng.random(dm.shape).T)
    dm = 0.5 * (dm + dm.T)
    return mol, auxmol, dm


def _seam_results(mol, auxmol, dm):
    '''Knob-off (on-the-fly kernel) reference results at both seams.'''
    opt = j_engine_3c2e.Int3c2eOpt(mol, auxmol).build()
    assert opt._j3c_cache_state == 'unlatched'
    dm_cart = opt.mol.apply_C_mat_CT(dm)
    rhoj = opt.contract_dm(dm_cart, hermi=1)
    rhoj = opt.auxmol.apply_CT_dot(rhoj, axis=-1)
    # feed a deterministic aux-space vector through the backward seam
    rng = np.random.default_rng(7)
    v_in = rng.random(rhoj.shape)
    v_in = 0.5 * (v_in + v_in[::-1]) * 1e-2
    auxvec_cart = opt.auxmol.apply_C_dot(v_in, axis=-1)
    vj = opt.contract_auxvec(auxvec_cart)
    assert opt._j3c_cache_state in ('unlatched', 'off')
    assert opt._j3c_table is None
    return rhoj, vj


def _cached_results(mol, auxmol, dm):
    opt = j_engine_3c2e.Int3c2eOpt(mol, auxmol).build()
    dm_cart = opt.mol.apply_C_mat_CT(dm)
    rhoj = opt.contract_dm(dm_cart, hermi=1)   # first call latches + fills
    rhoj = opt.auxmol.apply_CT_dot(rhoj, axis=-1)
    rng = np.random.default_rng(7)
    v_in = rng.random(rhoj.shape)
    v_in = 0.5 * (v_in + v_in[::-1]) * 1e-2
    auxvec_cart = opt.auxmol.apply_C_dot(v_in, axis=-1)
    vj = opt.contract_auxvec(auxvec_cart)
    return opt, rhoj, vj


def test_j3c_table_cache_seam_equivalence():
    for name in FIXTURES:
        mol, auxmol, dm = _engine_pair(name)
        rhoj_ref, vj_ref = _seam_results(mol, auxmol, dm)
        os.environ['GPU4PYSCF_DF_J_3C_CACHE'] = '1'
        try:
            opt, rhoj, vj = _cached_results(mol, auxmol, dm)
        finally:
            del os.environ['GPU4PYSCF_DF_J_3C_CACHE']
        # RED guard: the knob must actually engage (no vacuous pass)
        assert opt._j3c_cache_state == 'ready', opt._j3c_cache_state
        assert opt._j3c_table is not None
        assert opt._j3c_fill_wall_s is not None
        d_rhoj = abs(rhoj.get() - rhoj_ref.get()).max()
        d_vj = abs(vj.get() - vj_ref.get()).max()
        scale_rhoj = abs(rhoj_ref.get()).max()
        scale_vj = abs(vj_ref.get()).max()
        assert d_rhoj <= TOL * max(1., scale_rhoj), (name, d_rhoj)
        assert d_vj <= TOL * max(1., scale_vj), (name, d_vj)
        print(f'{name}: dm_xyz={opt._j3c_table.shape[0]} naux={opt._j3c_table.shape[1]} '
              f'fill={opt._j3c_fill_wall_s*1000:.1f} ms '
              f'd_rhoj={d_rhoj:.2e} d_vj={d_vj:.2e}')


def test_j3c_table_cache_default_off():
    mol, auxmol, dm = _engine_pair('water_def2svp')
    os.environ.pop('GPU4PYSCF_DF_J_3C_CACHE', None)
    opt, rhoj, vj = _cached_results(mol, auxmol, dm)
    assert opt._j3c_cache_state == 'off'
    assert opt._j3c_table is None   # default path untouched


def test_j3c_table_cache_get_j_and_scf():
    from gpu4pyscf.dft import RKS
    mol = pyscf.M(atom=FIXTURES['water_def2svp']['atom'],
                  basis='def2-svp', verbose=0)
    pyscf.scf.RHF(mol).init_guess_by_minao()

    def run(knob):
        os.environ.pop('GPU4PYSCF_DF_J_3C_CACHE', None)
        if knob:
            os.environ['GPU4PYSCF_DF_J_3C_CACHE'] = '1'
        try:
            mf = RKS(mol).density_fit(auxbasis='def2-universal-jkfit')
            mf.xc = 'b3lyp'
            mf.conv_tol = 1e-11
            cycles = []
            def cb(locs):
                if locs.get('e_tot') is not None:
                    cycles.append(locs['e_tot'])
            mf.callback = cb
            e = mf.kernel()
            vj = df_get_j(mf.with_df, mf.make_rdm1(), hermi=1)
            state = getattr(mf.with_df.j_engine, '_j3c_cache_state', 'n/a')
        finally:
            os.environ.pop('GPU4PYSCF_DF_J_3C_CACHE', None)
        return e, len(cycles), vj, state

    e0, n0, vj0, s0 = run(False)
    e1, n1, vj1, s1 = run(True)
    assert s0 in ('off', 'n/a')
    assert s1 == 'ready', s1
    assert abs(e1 - e0) <= 1e-9, abs(e1 - e0)
    assert n1 == n0, (n0, n1)
    assert abs(vj1.get() - vj0.get()).max() <= 1e-9
    print(f'scf dE={e1-e0:.2e} cycles={n0}/{n1} '
          f'get_j d={abs(vj1.get()-vj0.get()).max():.2e}')


if __name__ == '__main__':
    test_j3c_table_cache_seam_equivalence()
    test_j3c_table_cache_default_off()
    test_j3c_table_cache_get_j_and_scf()
    print('J3C table cache focused tests PASS')

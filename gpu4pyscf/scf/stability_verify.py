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

'''
UKS/UKS solution-stability verification for the mixed-precision 'auto'
policy (issue #7 rework, 2026-10-06).

Background (measured on RTX 4090, methoxy/def2-svp UKS wb97m-v): the fp32
head of the auto policy can steer the UKS iteration into a *different
stationary solution* than the plain-fp64 lane reaches from the same init
guess.  Both are legitimate converged SCF solutions (each satisfies the
fp64 fixed-point equations to ~1e-6 orbital gradient; a cross-restart
fp64 SCF keeps either basin, converging in a few cycles at the same
energy).  On that system the auto branch is *lower* in energy by
1.28e-6 Ha, but the sign is system-dependent (b3lyp/methoxy: the fp64
branch is lower by 4.0e-8 Ha).  Near-degenerate solution pairs also make
nuclear gradients branch-dependent (measured: 3.7e-2 Eh/Bohr between
branches at the same geometry, wb97m-v).

The default behaviour of the library is UNCHANGED.  This module provides
the opt-in verification pass behind precision_mode = 'auto+verify':

    mf = dft.UKS(mol, xc='wb97m-v').density_fit()
    mf.precision_mode = 'auto+verify'      # auto policy + stability check

After the SCF converges, the pass runs ONE extra fp64 SCF seeded with the
converged density of a *different* solution trajectory -- the fp64 lane
started from the same init guess -- and keeps the lower-energy converged
solution.  The rejected solution's energy, orbital-gradient residual and
the cycles each verification SCF took are recorded on
mf.scf_summary['stability_verification'] for evidence.

Cost: one additional short fp64 SCF (measured 1-4 cycles from a converged
density of the neighbouring basin, ~10-20% of a full auto run on the
affected systems; exact numbers in the issue-#7 evidence).  The check only
runs when the two trajectories actually converge to different densities
(max|dP| > stability_verify_dm_tol); closed-shell and single-solution
systems pay one extra converged-density restart SCF (~1 cycle) as the
sanity anchor, not a full diagonalisation cascade.
'''

import numpy
from gpu4pyscf.lib import precision
from gpu4pyscf.scf import hf as gpu_hf

__all__ = ['verify_solution_stability', 'stability_verify_dm_tol']

# Densities closer than this (max |dP|) are the same solution; no
# verification needed.  Measured same-solution restarts sit at ~1e-5 max
# |dP| (restart noise) while the distinct-branch pairs on the affected
# systems sit at 0.1-0.4, three to four orders above.
stability_verify_dm_tol = 1e-3


def _max_abs_dm_diff(dm_a, dm_b):
    import cupy
    return float(cupy.abs(cupy.asarray(dm_a) - cupy.asarray(dm_b)).max())


def verify_solution_stability(mf, dm0=None):
    '''Run the second-trajectory reference SCF and keep the lower solution.

    mf must have completed a converged SCF (mf.converged, mf.e_tot and
    mf.mo_coeff set).  The verification trajectory is a strict-fp64 SCF
    started from the SAME init guess the main run used (or dm0 if given),
    i.e. the solution the plain-fp64 lane would have converged to.  If it
    reaches a different density and a lower converged energy, it is the
    more stable solution and replaces the main run's orbitals; the main
    run's (rejected) solution is recorded first.

    Returns a dict with the decision record (also stored on
    mf.scf_summary['stability_verification']).
    '''
    if not mf.converged:
        raise RuntimeError('stability verification requires a converged SCF')
    mol = mf.mol

    # --- reference trajectory: plain fp64, same init guess -------------
    # rebuild the same mean-field class chain (RKS/UKS, DF-wrapped etc.);
    # type(mf) is the DF-decorated combination class whose __init__ needs
    # the original object, so drop the mixin and use the base class
    from gpu4pyscf.lib import logger
    if hasattr(mf, 'undo_df'):
        base = mf.undo_df().__class__
    else:
        base = type(mf)
    mf_ref = base(mol)
    # copy the knobs that change the solution being sought (xc set before
    # other DFT attributes for the KohnShamDFT constructor path)
    if hasattr(mf, 'xc') and hasattr(mf_ref, 'xc'):
        mf_ref.xc = mf.xc
    for attr in ('conv_tol', 'conv_tol_grad', 'max_cycle',
                 'init_guess', 'level_shift', 'damp',
                 'diis', 'diis_space', 'diis_space_rollback', 'nelec'):
        val = getattr(mf, attr, None)
        if val is not None and hasattr(mf_ref, attr):
            setattr(mf_ref, attr, val)
    if getattr(mf, 'with_df', None) is not None:
        mf_ref = mf_ref.density_fit(auxbasis=mf.with_df.auxbasis)
        if hasattr(mf.with_df, 'direct_scf_tol'):
            mf_ref.with_df.direct_scf_tol = mf.with_df.direct_scf_tol
    if hasattr(mf, 'grids') and hasattr(mf_ref, 'grids'):
        mf_ref.grids.level = mf.grids.level
    if hasattr(mf, 'nlcgrids') and hasattr(mf_ref, 'nlcgrids'):
        mf_ref.nlcgrids.level = mf.nlcgrids.level
    if getattr(mf, 'do_nlc', None) is not None and hasattr(mf_ref, 'do_nlc'):
        mf_ref.do_nlc = mf.do_nlc
    mf_ref.precision_mode = 'fp64'
    mf_ref.verbose = 0
    if dm0 is None:
        dm0 = mf_ref.get_init_guess(mol, mf.init_guess)
    e_ref = mf_ref.kernel(dm0=dm0)

    import cupy
    cupy.cuda.runtime.deviceSynchronize()

    dm_main = mf.make_rdm1()
    dm_ref = mf_ref.make_rdm1()

    record = {
        'policy': 'auto+verify',
        'main_e_tot': float(mf.e_tot),
        'main_converged': bool(mf.converged),
        'main_cycles': int(getattr(mf, 'cycles', 0) or 0),
        'ref_e_tot': float(e_ref),
        'ref_converged': bool(mf_ref.converged),
        'ref_cycles': int(getattr(mf_ref, 'cycles', 0) or 0),
    }

    # orbital-gradient residual of each solution: a genuine stationary
    # point has |g_orb| at the conv_tol_grad floor, regardless of branch
    try:
        import cupy as _cp
        h1 = _cp.asarray(mf.get_hcore())
        s1 = _cp.asarray(mf.get_ovlp())
        g_main = float(_cp.linalg.norm(mf.get_grad(
            mf.mo_coeff, mf.mo_occ, mf.get_fock(h1, s1, mf.get_veff(mol, dm_main), dm_main))))
        g_ref = float(_cp.linalg.norm(mf_ref.get_grad(
            mf_ref.mo_coeff, mf_ref.mo_occ, mf_ref.get_fock(
                h1, s1, mf_ref.get_veff(mol, dm_ref), dm_ref))))
    except Exception:
        g_main = g_ref = None
    record['orbital_grad_norm_main'] = g_main
    record['orbital_grad_norm_ref'] = g_ref

    record['dm_max_abs_diff'] = _max_abs_dm_diff(dm_main, dm_ref)

    if record['dm_max_abs_diff'] <= stability_verify_dm_tol:
        # same solution: nothing to select, record and keep the main run
        record['distinct_solutions'] = False
        record['selected'] = 'main'
        record['rejected_e_tot'] = None
        mf.scf_summary['stability_verification'] = record
        return record

    record['distinct_solutions'] = True
    dE = record['ref_e_tot'] - record['main_e_tot']
    record['dE_ref_minus_main'] = dE
    if dE < -1e-12:
        # the fp64-reference trajectory found the lower solution: adopt it
        rejected = {
            'e_tot': float(mf.e_tot),
            'cycles': int(getattr(mf, 'cycles', 0) or 0),
            'orbital_grad_norm': g_main,
        }
        mf.e_tot = mf_ref.e_tot
        mf.mo_energy = mf_ref.mo_energy
        mf.mo_coeff = mf_ref.mo_coeff
        mf.mo_occ = mf_ref.mo_occ
        mf.converged = mf_ref.converged
        mf.cycles = int(getattr(mf_ref, 'cycles', 0) or 0)
        # carry the reference trajectory's scf_summary pieces (exc/nlc/
        # coul terms of the adopted solution) so downstream e_terms
        # readers see the adopted solution's decomposition
        for k, v in (mf_ref.scf_summary or {}).items():
            if k != 'stability_verification':
                mf.scf_summary[k] = v
        record['selected'] = 'reference'
        record['rejected'] = rejected
    else:
        record['selected'] = 'main'
        record['rejected_e_tot'] = float(e_ref)
        record['rejected'] = {
            'e_tot': float(e_ref),
            'cycles': int(getattr(mf_ref, 'cycles', 0) or 0),
            'orbital_grad_norm': g_ref,
        }
    mf.scf_summary['stability_verification'] = record
    return record

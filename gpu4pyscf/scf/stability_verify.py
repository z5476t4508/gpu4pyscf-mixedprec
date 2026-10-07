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
policy (issue #7 rework, 2026-10-06; refinement distinctness test
2026-10-07, LC-29).

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

A second failure mode (measured 2026-10-07, m062x/methoxy/def2-tzvp UKS,
LC-28): near a saddle the fp32 head can converge to a point that is only
a fixed point of the *fp32* kernels -- displaced ~1.2e-3 in density along
the unstable mode's downhill side, 4.1e-8 Ha below the saddle, orbital
gradient 8.9e-6 vs the fp64 floor 3.4e-6.  A raw density difference
between the two trajectories cannot distinguish this fp32 displacement
from a genuine branch split (both sit just above the 1e-3 dm tolerance).

The default behaviour of the library is UNCHANGED.  This module provides
the opt-in verification pass behind precision_mode = 'auto+verify':

    mf = dft.UKS(mol, xc='wb97m-v').density_fit()
    mf.precision_mode = 'auto+verify'      # auto policy + stability check

After the SCF converges, the pass runs ONE strict-fp64 reference SCF
seeded with the init guess the main run used (the solution the plain
fp64 lane would have converged to).  If the two densities differ by more
than stability_verify_dm_tol, a SECOND bounded strict-fp64 SCF refines
from the auto solution's own density (the cross-restart test used to
diagnose the issue-#7 multiplicity):

* the refinement climbs back to the reference density -> the trajectories
  found the SAME solution and the auto point was an fp32 displacement;
  the stationary reference solution is adopted and the displacement
  (energy and density) is recorded;
* the refinement stays in a distinct basin -> a genuine two-basin split;
  the adopted solution is the LOWER of the two *stationary* points
  (refined auto vs reference), with the full decision record;
* the bounded refinement does not converge -> honest fallback to the
  raw two-trajectory comparison, flagged
  adopted_stationarity_verified = False.

All decisions are recorded on mf.scf_summary['stability_verification']
(both energies, cycles, orbital-gradient residuals, density differences,
occupied-space overlap SVD minima, selected/rejected).

Cost (measured on the methoxy/methanol benchmark cells, median wall time
vs a plain auto run): the reference SCF always -- a FULL strict-fp64 SCF
from the same init guess (9-12 cycles; this is what discovers the
fp64-lane basin, so it cannot be replaced by any restart from the main
run's density: cross-restarts preserve the basin in both directions,
measured in the issue-#7 diagnosis); the refinement SCF only when the
dm test fires (1-8 cycles from the auto solution's converged density,
bounded at stability_verify_refine_max_cycle).  Total auto+verify/auto:
1.97-2.64x on the 6-cell methoxy UKS matrix + closed-shell controls
(old two-trajectory policy: 1.65-2.34x).
'''

import numpy
from gpu4pyscf.scf import hf as gpu_hf

__all__ = ['verify_solution_stability', 'stability_verify_dm_tol',
           'stability_verify_refine_max_cycle']

# Densities closer than this (max |dP|) are the same solution; no
# verification needed.  Measured same-solution restarts sit at ~1e-5 max
# |dP| (restart noise) while the distinct-branch pairs on the affected
# systems sit at 0.1-0.4, three to four orders above.  The fp32
# displacement measured on the m062x/tzvp saddle cell (1.16e-3) also
# exceeds this tolerance -- that is exactly the case the refinement
# cross-check below disambiguates.
stability_verify_dm_tol = 1e-3

# Cycle bound for the refinement SCF (from the auto solution's density).
# Measured climbs back to the stationary point: 1-4 cycles when the auto
# point was already fp64-stationary (issue #7 cross-restarts), 8 cycles
# from the displaced point on the m062x/tzvp saddle cell (CPU fp64).
stability_verify_refine_max_cycle = 12


def _max_abs_dm_diff(dm_a, dm_b):
    import cupy
    return float(cupy.abs(cupy.asarray(dm_a) - cupy.asarray(dm_b)).max())


def _build_fp64_sibling(mf):
    '''A strict-fp64 SCF object with the solution-defining knobs of mf.

    Used for both verification trajectories (reference from the init
    guess, refinement from a converged density); type(mf) is the
    DF-decorated combination class whose __init__ needs the original
    object, so drop the mixin and use the base class.
    '''
    from types import MethodType
    mol = mf.mol
    if hasattr(mf, 'undo_df'):
        base = mf.undo_df().__class__(mol)
    else:
        base = type(mf)(mol)
    # xc set before other DFT attributes (KohnShamDFT constructor path)
    if hasattr(mf, 'xc') and hasattr(base, 'xc'):
        base.xc = mf.xc
    # Energy-affecting instance state of the SCF/DFT object itself.
    # 'disp'/'disp_with_3body' select the NATIVE dispersion correction
    # (upstream API, e.g. mf.disp = 'd3bj' or 'd4:wb97x-3c'; check_disp
    # reads mf.disp): dropping them makes the sibling's reference energy
    # omit the dispersion sum -- measured on water/b3lyp/d3bj (LC-58),
    # ref_e_tot was high by exactly +e_disp = 5.7388e-4 Ha.  'nlc' is the
    # three-state NLC knob read by the class do_nlc ('' = follow xc,
    # 'vv10' = force on, 0/False = force off): dropping it diverges the
    # sibling whenever the user overrides it (the old code instead
    # copied the BOUND method mf.do_nlc, which consults the main
    # object's state -- while energy_elec separately reads the
    # sibling-local nlc string).
    for attr in ('conv_tol', 'conv_tol_grad', 'max_cycle',
                 'init_guess', 'level_shift', 'damp',
                 'diis', 'diis_space', 'diis_space_rollback', 'nelec',
                 'disp', 'disp_with_3body', 'nlc'):
        val = getattr(mf, attr, None)
        if val is not None and hasattr(base, attr):
            setattr(base, attr, val)
    if getattr(mf, 'with_df', None) is not None:
        base = base.density_fit(auxbasis=mf.with_df.auxbasis)
        if hasattr(mf.with_df, 'direct_scf_tol'):
            base.with_df.direct_scf_tol = mf.with_df.direct_scf_tol
    if hasattr(mf, 'grids') and hasattr(base, 'grids'):
        base.grids.level = mf.grids.level
    if hasattr(mf, 'nlcgrids') and hasattr(base, 'nlcgrids'):
        base.nlcgrids.level = mf.nlcgrids.level
    # Instance-attached dispersion hooks (e.g. the r2SCAN-3c benchmark
    # runner's D4+gCP MethodType attach) are NOT carried by class
    # reconstruction -- the exact attach-after-final-transform trap the
    # runner documents.  Without this copy the sibling's e_tot omits the
    # geometry-only correction sum: measured on every 3c verify record
    # (benzene/vitaminC/tamoxifen, LC-39/40/57), ref_e_tot was low by
    # exactly D4+gCP, and an adopted solution would carry a
    # dispersion-free energy.  Rebind bound methods to the sibling so
    # scf_summary recording targets the sibling, not the main object.
    gd = mf.__dict__.get('get_dispersion')
    if gd is not None:
        base.get_dispersion = MethodType(getattr(gd, '__func__', gd), base)
    dd = mf.__dict__.get('do_disp')
    if dd is not None:
        base.do_disp = dd
    # instance-level do_nlc override (rare; the class method reads the
    # copied nlc state): carry only what the instance actually set
    dn = mf.__dict__.get('do_nlc')
    if dn is not None:
        if callable(dn):
            base.do_nlc = MethodType(getattr(dn, '__func__', dn), base)
        else:
            base.do_nlc = dn
    base.precision_mode = 'fp64'
    base.verbose = 0
    return base


def _orbital_grad_norm(mf_obj, dm):
    '''|g_orb| of mf_obj's converged solution (fp64 fixed-point residual).'''
    import cupy as cp
    mol = mf_obj.mol
    h1 = cp.asarray(mf_obj.get_hcore())
    s1 = cp.asarray(mf_obj.get_ovlp())
    return float(cp.linalg.norm(mf_obj.get_grad(
        mf_obj.mo_coeff, mf_obj.mo_occ,
        mf_obj.get_fock(h1, s1, mf_obj.get_veff(mol, dm), dm))))


def _occ_overlap_min(mf_a, mf_b):
    '''Min singular value of the occupied-subspace overlap between two
    converged solutions, per spin (UKS: {'alpha','beta'}; RKS: scalar).

    ~1 = same occupied space; << 1 = different branch (measured beta
    occupied-space swap on the wb97m-v/svp split: 0.2576).
    '''
    import cupy as cp
    s = cp.asarray(mf_a.get_ovlp())
    ca, oa = mf_a.mo_coeff, mf_a.mo_occ
    cb, ob = mf_b.mo_coeff, mf_b.mo_occ
    if getattr(oa, 'ndim', 1) == 2:
        out = {}
        for k, spin in enumerate(('alpha', 'beta')):
            ca_o = cp.asarray(ca[k])[:, cp.asarray(oa[k]) > 0]
            cb_o = cp.asarray(cb[k])[:, cp.asarray(ob[k]) > 0]
            if ca_o.shape[1] == 0 or ca_o.shape[1] != cb_o.shape[1]:
                out[spin] = None
            else:
                m = ca_o.T @ s @ cb_o
                out[spin] = float(cp.linalg.svd(m, compute_uv=False).min())
        return out
    ca_o = cp.asarray(ca)[:, cp.asarray(oa) > 0]
    cb_o = cp.asarray(cb)[:, cp.asarray(ob) > 0]
    if ca_o.shape[1] == 0 or ca_o.shape[1] != cb_o.shape[1]:
        return None
    m = ca_o.T @ s @ cb_o
    return float(cp.linalg.svd(m, compute_uv=False).min())


def _adopt(mf, src):
    '''Replace mf's solution with src's (orbitals, energy, e_terms pieces).'''
    mf.e_tot = src.e_tot
    mf.mo_energy = src.mo_energy
    mf.mo_coeff = src.mo_coeff
    mf.mo_occ = src.mo_occ
    mf.converged = src.converged
    mf.cycles = int(getattr(src, 'cycles', 0) or 0)
    # carry the adopted trajectory's scf_summary pieces (exc/nlc/coul
    # terms of the adopted solution) so downstream e_terms readers see
    # the adopted solution's decomposition
    for k, v in (src.scf_summary or {}).items():
        if k != 'stability_verification':
            mf.scf_summary[k] = v


def verify_solution_stability(mf, dm0=None):
    '''Run the verification trajectories and keep the verified solution.

    mf must have completed a converged SCF (mf.converged, mf.e_tot and
    mf.mo_coeff set).  Verification runs a strict-fp64 reference SCF from
    the SAME init guess the main run used (or dm0 if given).  If the two
    trajectories converge to densities differing by more than
    stability_verify_dm_tol, a bounded strict-fp64 refinement SCF from
    the main run's density decides whether the difference is an fp32
    fixed-point displacement (same solution; adopt the stationary
    reference point) or a genuine two-basin split (adopt the lower of
    the two stationary points, refined auto vs reference).  If that
    refinement does not converge within
    stability_verify_refine_max_cycle cycles, the raw two-trajectory
    lower-energy rule is used and the record flags
    adopted_stationarity_verified = False.

    Returns a dict with the decision record (also stored on
    mf.scf_summary['stability_verification']).
    '''
    if not mf.converged:
        raise RuntimeError('stability verification requires a converged SCF')
    mol = mf.mol
    import cupy

    # --- reference trajectory: plain fp64, same init guess -------------
    mf_ref = _build_fp64_sibling(mf)
    if dm0 is None:
        dm0 = mf_ref.get_init_guess(mol, mf.init_guess)
    e_ref = mf_ref.kernel(dm0=dm0)

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
        g_main = _orbital_grad_norm(mf, dm_main)
        g_ref = _orbital_grad_norm(mf_ref, dm_ref)
    except Exception:
        g_main = g_ref = None
    record['orbital_grad_norm_main'] = g_main
    record['orbital_grad_norm_ref'] = g_ref
    try:
        record['occ_overlap_min_main_vs_ref'] = _occ_overlap_min(mf, mf_ref)
    except Exception:
        record['occ_overlap_min_main_vs_ref'] = None

    record['dm_max_abs_diff'] = _max_abs_dm_diff(dm_main, dm_ref)

    if record['dm_max_abs_diff'] <= stability_verify_dm_tol:
        # same solution: nothing to select, record and keep the main run
        record['distinct_solutions'] = False
        record['selected'] = 'main'
        record['rejected_e_tot'] = None
        mf.scf_summary['stability_verification'] = record
        return record

    # --- the dm test fired: disambiguate fp32 displacement vs split ----
    # bounded strict-fp64 refinement from the auto solution's density
    # (the cross-restart test that diagnosed the issue-#7 multiplicity)
    mf_refine = _build_fp64_sibling(mf)
    mf_refine.max_cycle = stability_verify_refine_max_cycle
    record['refine_max_cycle'] = int(mf_refine.max_cycle)
    e_refined = mf_refine.kernel(dm0=dm_main)

    cupy.cuda.runtime.deviceSynchronize()

    dm_refined = mf_refine.make_rdm1()
    try:
        g_refined = _orbital_grad_norm(mf_refine, dm_refined)
    except Exception:
        g_refined = None
    record['main_refined_e_tot'] = float(e_refined)
    record['main_refined_converged'] = bool(mf_refine.converged)
    record['main_refined_cycles'] = int(getattr(mf_refine, 'cycles', 0) or 0)
    record['orbital_grad_norm_main_refined'] = g_refined
    record['dm_max_abs_diff_refined_vs_ref'] = _max_abs_dm_diff(dm_refined, dm_ref)
    record['dm_max_abs_diff_refined_vs_main'] = _max_abs_dm_diff(dm_refined, dm_main)
    try:
        record['occ_overlap_min_refined_vs_ref'] = _occ_overlap_min(mf_refine, mf_ref)
    except Exception:
        record['occ_overlap_min_refined_vs_ref'] = None
    record['dE_ref_minus_main'] = float(e_ref) - float(mf.e_tot)

    if mf_refine.converged and record['dm_max_abs_diff_refined_vs_ref'] <= stability_verify_dm_tol:
        # SAME solution: the auto lane's fp32 fixed point was displaced
        # within the reference basin; the refinement climbed back to the
        # stationary point.  Adopt the stationary reference solution and
        # record the displacement honestly.
        record['refinement'] = 'same_solution'
        record['distinct_solutions'] = False
        record['fp32_displacement_detected'] = True
        record['auto_point_displacement_e'] = float(mf.e_tot) - float(e_refined)
        record['displaced_auto_point'] = {
            'e_tot': float(mf.e_tot),
            'cycles': int(getattr(mf, 'cycles', 0) or 0),
            'orbital_grad_norm': g_main,
            'note': ('fp32 fixed-point displacement, not a distinct '
                     'stationary solution; refined back to the reference '
                     'basin stationary point'),
        }
        record['rejected_e_tot'] = None
        _adopt(mf, mf_ref)
        record['selected'] = 'reference'
        record['adopted_stationarity_verified'] = True
    elif mf_refine.converged:
        # GENUINE split: the refinement stayed in a distinct basin; both
        # the refined auto point and the reference are strict-fp64
        # stationary points.  Keep the lower one.
        record['refinement'] = 'distinct'
        record['distinct_solutions'] = True
        record['fp32_displacement_detected'] = False
        dE = float(e_ref) - float(e_refined)
        record['dE_ref_minus_main'] = dE
        if dE > 1e-12:
            # the refined auto-basin stationary point is lower: adopt it
            record['rejected'] = {
                'e_tot': float(e_ref),
                'cycles': int(getattr(mf_ref, 'cycles', 0) or 0),
                'orbital_grad_norm': g_ref,
            }
            record['rejected_e_tot'] = float(e_ref)
            _adopt(mf, mf_refine)
            record['selected'] = 'main'
        else:
            record['rejected'] = {
                'e_tot': float(e_refined),
                'cycles': int(getattr(mf_refine, 'cycles', 0) or 0),
                'orbital_grad_norm': g_refined,
            }
            record['rejected_e_tot'] = float(e_refined)
            _adopt(mf, mf_ref)
            record['selected'] = 'reference'
        record['adopted_stationarity_verified'] = True
    else:
        # bounded refinement did not converge: fall back to the raw
        # two-trajectory comparison.  The adopted point's stationarity
        # on the fp64 surface is NOT verified on this run.
        record['refinement'] = 'non_converged'
        record['distinct_solutions'] = True
        record['fp32_displacement_detected'] = None
        record['adopted_stationarity_verified'] = False
        if record['dE_ref_minus_main'] < -1e-12:
            rejected = {
                'e_tot': float(mf.e_tot),
                'cycles': int(getattr(mf, 'cycles', 0) or 0),
                'orbital_grad_norm': g_main,
            }
            _adopt(mf, mf_ref)
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

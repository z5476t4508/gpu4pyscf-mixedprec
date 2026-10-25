#!/usr/bin/env python3
"""LC-211 production-reachable rung classification + ladder interior probe
(CPU fp64 pyscf).

LC-210 left two named open edges on the methoxy/wb97m-v/def2-svp
multi-basin landscape:

  rung    -- the vsap/1e default-guess SCFs land at
             -114.9016981566 = 1.851e-6 BELOW R0_AUTO (a deep rung NOT on
             the mode-following v1-ladder: dm-nearest R0_AUTO 0.287,
             1.066/0.980 from the t=0.30/t=0.50 endpoints).  Its curvature
             is unclassified.  This unit re-runs both guess SCFs (fresh
             objects, NO dm0 seed -- identical to the LC-210 reachability
             probe), SAVES the rung dms, proves own-basin stationarity
             with fresh-object stay controls, corroborates vsap vs 1e as
             the same (or distinct) solution, and builds the dense
             NLC-inclusive internal-stability Hessian at the mandatory
             rung (vsap), classifying minimum / saddle_index_N from the
             negative-eigenvalue count.
  probe   -- the LC-209 descent grid had no t=0.6; the path energy
             descends to the t=0.50 full-Hessian MINIMUM and rises at
             t=0.7, bracketing the path bottom in the unmeasured (0.5,
             0.7) interior.  This unit reruns the LC-209/210 descent
             machinery at t=0.60 on the R line (loaded LC-208 dense-H-R
             direction, one-column validation at this run's seeded state,
             hdiag informational per the 2026-10-20 erratum, Davidson
             fallback), saves the endpoint dm, and places its energy on
             the bracket.  ADAPTIVE: its dense Hessian is built ONLY if
             the endpoint lands below the t=0.50 minimum by more than the
             preregistered tol (a new deepest point demands curvature);
             an in-bracket landing is recorded energy-only.

Adaptive dense builds (all preregistered, fail-closed):
  rung_1e       -- only if vsap/1e are NOT the same rung (energy or dm)
  interior t0.60 -- only if it lands below the t=0.50 minimum by > tol
Spectra are built only at states that passed their gate (rung
reproduction vs the LC-210 anchor; probe convergence + in-place stay);
misses are recorded, never silently accepted.

The fast-stage facts (refs, rungs, corroboration, probe, placement) are
anchored to disk BEFORE any dense build so a session ceiling loses only
resumable Hessian columns (npz checkpoints every 100).

Everything here is mechanical CPU measurement (pyscf 2.14.0, the
independent implementation); verdicts live in the session docs.
numpy/pyscf/scipy are imported lazily inside the numeric paths so the
module stays importable stdlib-only for the unit tests.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream_softmode_analysis as base     # sibling, stdlib-safe
import upstream_softmode_nlc as nlc           # sibling, stdlib-safe
import upstream_descent_map as dm             # sibling, stdlib-safe
import upstream_ladder_classify as lc         # sibling, stdlib-safe

SAMPLE_SCHEMA = 'rung-classify-v1'

REQUIRED_PLAN_KEYS = (
    'unit', 'amendment_id', 'batch_id', 'evidence_dir', 'cell',
    'settings', 'family_dms', 'ladder_endpoint_dms', 'rungs',
    'rung_dense_mandatory', 'interior_probe', 'adaptive', 'thresholds',
    'spectra_ckpt_dir',
)
REQUIRED_SETTINGS_KEYS = base.REQUIRED_SETTINGS_KEYS + ('nlcgrids_level',)

DEFAULT_THRESHOLDS = {
    'reproduce_tol': 1e-9,      # |e_rerun - LC-210 anchor| for a rung
                                # reproduction (spectra only on reproduced)
    'one_column_rel': 1e-6,     # ||hop(v1)*2 - H v1|| / ||H v1|| for the
                                # loaded direction (hdiag informational
                                # only, the 2026-10-20 erratum)
    'basin_energy': 1e-8,       # |E - ref| for a destination match
    'new_dm_distance': 0.05,    # min AO dm distance for new-basin class
}
DEFAULT_ADAPTIVE = {
    'same_rung_e': 1e-9,        # |E_vsap - E_1e| for same-rung energy
    'same_rung_dm': 0.05,       # rung-rung AO dm distance for same rung
    'interior_dense_below': 1e-10,  # the t=0.60 endpoint must sit below
                                    # the t=0.50 minimum by more than
                                    # this for its dense Hessian
}


def validate_rung_plan(plan):
    '''Fail-closed plan checks before any numeric work (stdlib only).'''
    problems = []
    try:
        base.verify_inputs(plan)
    except (RuntimeError, KeyError, TypeError) as e:
        problems.append(str(e))
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            problems.append(f'missing plan key {key!r}')
    if any(p.startswith('missing plan key') for p in problems):
        return problems
    for key in REQUIRED_SETTINGS_KEYS:
        if key not in plan['settings']:
            problems.append(f'missing settings key {key!r}')

    fam = plan.get('family_dms', {})
    for p in base.FAMILY_POINTS:
        if p not in fam:
            problems.append(f'family_dms missing {p!r}')
        else:
            for key in ('path', 'anchor_e', 'anchor_source'):
                if key not in fam[p]:
                    problems.append(f'family point {p!r} missing {key!r}')

    led = plan.get('ladder_endpoint_dms', {})
    if not isinstance(led, dict) or not led:
        problems.append('ladder_endpoint_dms must be a non-empty object')
    else:
        for label, ent in led.items():
            if not isinstance(ent, dict):
                problems.append(f'ladder endpoint {label!r} must be an '
                                'object')
                continue
            for key in ('path', 'anchor_e_final', 'anchor_source'):
                if key not in ent:
                    problems.append(f'ladder endpoint {label!r} missing '
                                    f'{key!r}')
            if not isinstance(ent.get('anchor_e_final'),
                               (int, float)):
                problems.append(f'ladder endpoint {label!r} '
                                'anchor_e_final must be numeric')

    rungs = plan.get('rungs', [])
    if (not isinstance(rungs, list) or len(rungs) < 2):
        problems.append('rungs must be a list of at least two guesses '
                        '(the corroboration pair)')
    else:
        guesses = [r.get('guess') for r in rungs]
        if len(set(guesses)) != len(guesses) or any(not g for g in guesses):
            problems.append('rung guesses must be non-empty and unique')
        for r in rungs:
            if r.get('guess') not in lc.SUPPORTED_GUESSES:
                problems.append(f'rung guess {r.get("guess")!r} is not '
                                f'one of {lc.SUPPORTED_GUESSES}')
            if not isinstance(r.get('anchor_e_tot'), (int, float)):
                problems.append(f'rung {r.get("guess")!r} missing numeric '
                                'anchor_e_tot (the LC-210 report value)')
            if not isinstance(r.get('anchor_source'), str):
                problems.append(f'rung {r.get("guess")!r} missing '
                                'anchor_source')
    mandatory = plan.get('rung_dense_mandatory')
    if mandatory not in [r.get('guess') for r in rungs if
                         isinstance(r, dict)]:
        problems.append(f'rung_dense_mandatory {mandatory!r} is not one '
                        'of the plan rungs')

    ip = plan.get('interior_probe', {})
    if not isinstance(ip, dict):
        problems.append('interior_probe must be an object')
    else:
        if ip.get('point') not in base.FAMILY_POINTS:
            problems.append(f'interior_probe point {ip.get("point")!r} '
                            'is not a family point')
        if ip.get('sigma') not in (1, -1):
            problems.append('interior_probe sigma must be 1 or -1')
        t = ip.get('t')
        if not (isinstance(t, (int, float)) and t > 0):
            problems.append('interior_probe t must be positive')
        ck = ip.get('hessian_ckpt')
        if not isinstance(ck, str) or not ck:
            problems.append('interior_probe.hessian_ckpt must be a path '
                            'string')
        elif not os.path.isfile(ck):
            problems.append(f'interior_probe hessian ckpt missing: {ck}')
        br = ip.get('bracket_anchors')
        if not isinstance(br, dict):
            problems.append('interior_probe.bracket_anchors must be an '
                            'object')
        else:
            for side in ('lower', 'upper'):
                if side not in br:
                    problems.append(f'bracket_anchors missing {side!r}')
                elif not isinstance(br[side].get('e'), (int, float)):
                    problems.append(f'bracket_anchors {side!r} e must be '
                                    'numeric')
            if isinstance(br.get('lower'), dict) and \
                    isinstance(br.get('upper'), dict):
                if br['lower'].get('e') > br['upper'].get('e'):
                    problems.append('bracket lower (deeper) energy must '
                                    'be <= upper energy')
                for side in ('lower', 'upper'):
                    if not isinstance(br[side].get('label'), str):
                        problems.append(f'bracket_anchors {side!r} '
                                        "missing 'label'")

    ad = plan.get('adaptive', {})
    if not isinstance(ad, dict):
        problems.append('adaptive must be an object')
    else:
        for key, value in DEFAULT_ADAPTIVE.items():
            if key in ad and not (isinstance(ad[key], (int, float))
                                  and ad[key] > 0):
                problems.append(f'adaptive {key!r} must be positive')

    th = plan.get('thresholds', {})
    if not isinstance(th, dict):
        problems.append('thresholds must be an object')
    else:
        for key, value in DEFAULT_THRESHOLDS.items():
            if key in th and not (isinstance(th[key], (int, float))
                                  and th[key] > 0):
                problems.append(f'threshold {key!r} must be positive')

    if not isinstance(plan.get('spectra_ckpt_dir'), str):
        problems.append('spectra_ckpt_dir must be a string')
    if plan.get('spectra_points') is not None:
        problems.append('rung classify takes no spectra_points; spectra '
                        'are built at gated states only')
    return problems


def same_rung_corroboration(d_e, dm_distance, adaptive):
    '''Pure same-rung decision for the vsap/1e pair (stdlib-testable).'''
    e_ok = abs(float(d_e)) <= float(adaptive['same_rung_e'])
    dm_ok = float(dm_distance) <= float(adaptive['same_rung_dm'])
    return {
        'dE': float(d_e),
        'dm_distance': float(dm_distance),
        'same_rung_e': bool(e_ok),
        'same_rung_dm': bool(dm_ok),
        'same_rung': bool(e_ok and dm_ok),
    }


def interior_placement(e_final, lower_e, upper_e, below_tol):
    '''Place the t=0.60 endpoint on the (t0.50, t0.70) bracket and decide
    the adaptive dense-H branch (stdlib-testable).  lower_e is the DEEPER
    (more negative) t=0.50 minimum; upper_e the t=0.70 rise.'''
    e = float(e_final)
    lo, hi = float(lower_e), float(upper_e)
    return {
        'diff_vs_lower': e - lo,
        'diff_vs_upper': e - hi,
        'in_bracket': bool(lo - 1e-12 <= e <= hi + 1e-12),
        'below_lower': bool(e < lo - float(below_tol)),
        'interior_dense_branch': bool(e < lo - float(below_tol)),
    }


def _spin_dm_distance(dm_a, dm_b):
    import numpy as np
    return float(sum(np.linalg.norm(np.asarray(dm_a[k])
                                    - np.asarray(dm_b[k]))
                     for k in range(2)))


def _save_dm_npz(path, dm_final):
    '''Atomic npz save in the established (2,nao,nao) storage form.'''
    import numpy as np
    arr = np.asarray(dm_final)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise RuntimeError(f'rung dm is not the 2-spin storage form: '
                           f'{arr.shape}')
    with open(path + '.tmp', 'wb') as f:
        np.savez(f, dm=arr)
    os.replace(path + '.tmp', path)
    return path


def _spectrum_at(state, label, ckpt_dir, log):
    '''Dense NLC stability Hessian + classification + interim json at a
    gated state (mirrors the LC-210 loop).'''
    ck = os.path.join(ckpt_dir, f'dense-H-{label}.npz')
    spec = nlc.dense_stability_nlc(state, log, ckpt=ck)
    cls = lc.classify_spectrum(spec['eigenvalues'].tolist())
    summary = nlc.spectrum_summary(spec)
    summary['classification'] = cls
    interim = {
        'label': label,
        'generated_at_utc': base.utc_now(),
        'ckpt_path': ck,
        'ckpt_sha256': base.sha256_file(ck),
        'classification': cls,
        'softest_10': summary['softest_10'],
        'operator_asymmetry': summary['operator_asymmetry'],
        'davidson_softest3': summary['davidson_softest3'],
    }
    ipath = os.path.join(ckpt_dir, f'filelevel-eigh-{label}-interim.json')
    with open(ipath + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(interim, f, indent=1)
    os.replace(ipath + '.tmp', ipath)
    log(f'spectrum {label}: {cls["classification"]} '
        f'lambda1={cls["lambda1"]:+.6e} n_negative={cls["n_negative"]}')
    return summary, cls


def run(plan, out_json, out_md, log):
    import numpy as np

    problems = validate_rung_plan(plan)
    if problems:
        raise RuntimeError('plan validation failed:\n  '
                           + '\n  '.join(problems))
    cell, settings = plan['cell'], plan['settings']
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(plan.get('thresholds', {}))
    adaptive = dict(DEFAULT_ADAPTIVE)
    adaptive.update(plan.get('adaptive', {}))
    ckpt_dir = plan['spectra_ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    dm_dir = os.path.join(plan['evidence_dir'], 'dm')
    os.makedirs(dm_dir, exist_ok=True)

    mol, mf = nlc.build_mol_mf_nlc(cell, settings)
    log(f'cell {cell["tag"]}: nao={mol.nao_nr()} nelec={mol.nelec} '
        f'spin={cell["spin"]}')

    t_start = time.time()

    # ---- references: family dms + LC-210 ladder endpoint dms ----------
    dms = {}
    for p in base.FAMILY_POINTS:
        raw = base.dm_from_npz(plan['family_dms'][p]['path'])
        if isinstance(raw, tuple) and len(raw) == 2:
            dms[p] = raw
        elif hasattr(raw, 'ndim') and raw.ndim == 3 and raw.shape[0] == 2:
            dms[p] = (raw[0], raw[1])
        else:
            raise RuntimeError(f'family point {p} density is not a '
                               f'2-spin UKS dm: {type(raw)} '
                               f'{getattr(raw, "shape", None)}')
    inputs = {p: {
        'path': plan['family_dms'][p]['path'],
        'sha256': base.sha256_file(plan['family_dms'][p]['path']),
        'anchor_e': plan['family_dms'][p]['anchor_e'],
        'anchor_source': plan['family_dms'][p]['anchor_source'],
    } for p in base.FAMILY_POINTS}
    ladder_dms = {}
    ladder_inputs = {}
    for label, ent in plan['ladder_endpoint_dms'].items():
        raw = base.dm_from_npz(ent['path'])
        if isinstance(raw, tuple) and len(raw) == 2:
            ladder_dms[label] = raw
        elif hasattr(raw, 'ndim') and raw.ndim == 3 and raw.shape[0] == 2:
            ladder_dms[label] = (raw[0], raw[1])
        else:
            raise RuntimeError(f'ladder endpoint {label} density is not '
                               f'a 2-spin UKS dm')
        ladder_inputs[label] = {
            'path': ent['path'],
            'sha256': base.sha256_file(ent['path']),
            'anchor_e_final': ent['anchor_e_final'],
            'anchor_source': ent['anchor_source'],
        }

    cpu_refs = {}
    for p in base.FAMILY_POINTS:
        e = base.energy_at(mf, dms[p])
        cpu_refs[p] = e
        log(f'cpu ref {p}: E={e:.12f} '
            f'dE_vs_gpu_anchor={e - inputs[p]["anchor_e"]:+.3e}')
    ladder_refs = {}
    for label, dmm in ladder_dms.items():
        e = base.energy_at(mf, dmm)
        ladder_refs[label] = e
        log(f'cpu ref {label}: E={e:.12f} '
            f'dE_vs_lc210_e_final={e - ladder_inputs[label]["anchor_e_final"]:+.3e}')

    # ---- rungs: default-guess SCFs (NO dm0), saved dm, stay control ---
    refs = dict(cpu_refs)
    refs.update(ladder_refs)
    ref_dms = dict(dms)
    ref_dms.update(ladder_dms)
    rungs = {}
    for r in plan['rungs']:
        guess = r['guess']
        mf_g = nlc.build_mol_mf_nlc(cell, settings)[1]
        mf_g.init_guess = guess
        t0 = time.time()
        e = mf_g.kernel()
        wall = time.time() - t0
        dm_g = mf_g.make_rdm1()
        label = f'rung_{guess}'
        dm_path = os.path.join(dm_dir, f'{label}.npz')
        _save_dm_npz(dm_path, dm_g)
        dm_distances = {q: _spin_dm_distance(dm_g, ref_dms[q])
                        for q in ref_dms}
        cls = dm.classify_destination(bool(mf_g.converged), float(e),
                                      refs, dm_distances, thresholds)
        repro = lc.classify_reproduction(mf_g.converged, e,
                                         r['anchor_e_tot'],
                                         thresholds['reproduce_tol'])
        mf_stay = nlc.build_mol_mf_nlc(cell, settings)[1]
        stay = base.seeded_scf(mf_stay, dm_g, f'{label}-stay', log)
        rec = {
            'guess': guess,
            'label': label,
            'e_tot': float(e),
            'converged': bool(mf_g.converged),
            'cycles': int(getattr(mf_g, 'cycles', 0) or 0),
            'wall_s': wall,
            'anchor_e_tot': float(r['anchor_e_tot']),
            'anchor_source': r['anchor_source'],
            'diff': repro['diff'],
            'reproduced': repro['reproduced'],
            'stay': {
                'e_tot': stay['e_tot'],
                'converged': stay['converged'],
                'cycles': stay['cycles'],
                'dE_vs_rerun': stay['e_tot'] - float(e),
            },
            'dm_path': dm_path,
            'dm_sha256': base.sha256_file(dm_path),
            'dm_distances': dm_distances,
            'energy_diffs': {q: float(e) - refs[q] for q in refs},
            'classification': cls,
        }
        rungs[guess] = {
            'record': rec,
            'state': mf_g,
            'dm': dm_g,
        }
        flag = '' if rec['reproduced'] else ' NOT-REPRODUCED (spectrum will be SKIPPED)'
        log(f'rung {guess}: E={e:.12f} diff_vs_LC210={rec["diff"]:+.3e} '
            f'cyc={rec["cycles"]} stay_cyc={stay["cycles"]} '
            f'stay_dE={rec["stay"]["dE_vs_rerun"]:+.3e}{flag}')

    # ---- rung corroboration (vsap vs 1e) ------------------------------
    g0, g1 = [r['guess'] for r in plan['rungs']][:2]
    corr = same_rung_corroboration(
        rungs[g1]['record']['e_tot'] - rungs[g0]['record']['e_tot'],
        _spin_dm_distance(rungs[g0]['dm'], rungs[g1]['dm']),
        adaptive)
    log(f'rung corroboration {g0}/{g1}: dE={corr["dE"]:+.3e} '
        f'dm_dist={corr["dm_distance"]:.6f} same_rung={corr["same_rung"]}')

    # ---- interior probe: LC-209 descent machinery at t=0.60 -----------
    ip = plan['interior_probe']
    mf_p = nlc.build_mol_mf_nlc(cell, settings)[1]
    stay_p = base.seeded_scf(mf_p, dms[ip['point']], ip['point'], log)
    if not stay_p['converged']:
        raise RuntimeError(f'stay control at {ip["point"]} did not '
                           'converge; refusing to descend from a '
                           'non-stationary seed')
    H, w, v, asym = dm.load_ckpt_spectrum(ip['hessian_ckpt'])
    v1 = v[:, 0]
    lambda1 = float(w[0])
    g, hop, hdiag, _dims = nlc.gen_g_hop_uhf_nlc(
        mf_p, mf_p.mo_coeff, mf_p.mo_occ)
    rel, hdiag_dev = dm.one_column_validation(hop, hdiag, H, v1)
    e_dav, v_dav0 = lc._davidson_softest(g, hop, hdiag)
    cos_dav = base.cosine(v_dav0, v1)
    validated = rel <= thresholds['one_column_rel']
    source = 'loaded_dense_validated' if validated else 'davidson_fallback'
    if not validated:
        log(f'WARNING: loaded-H validation failed (one_col={rel:.2e}); '
            f'descending along the Davidson softest instead')
    direction = {
        'ckpt_path': ip['hessian_ckpt'],
        'ckpt_sha256': base.sha256_file(ip['hessian_ckpt']),
        'dim': int(H.shape[0]),
        'asymmetry': asym,
        'lambda1_dense': lambda1,
        'one_column_rel': rel,
        'hdiag_dev': hdiag_dev,
        'orbital_gradient_norm': float(np.linalg.norm(g)),
        'davidson_softest3': e_dav.tolist(),
        'cos_davison_dense': cos_dav,
        'validated': bool(validated),
        'direction_source': source,
    }
    log(f'direction {ip["point"]}: lambda1={lambda1:+.6e} '
        f'one_col_rel={rel:.2e} validated={validated} '
        f'davidson1={e_dav[0]:+.6e} cos={cos_dav}')
    v_dir = v1 if validated else v_dav0

    mo_p, mo_occ_p = mf_p.mo_coeff, mf_p.mo_occ
    x1a, x1b = base.split_x(v_dir, mo_occ_p[0], mo_occ_p[1])
    sigma = ip['sigma']
    t_probe = ip['t']
    plabel = (f'ladder_{ip["point"]}'
              f'{"p" if sigma == 1 else "m"}1_t{t_probe:.2f}')
    dma = base.rotated_density(mo_p[0], mo_occ_p[0], x1a, sigma * t_probe)
    dmb = base.rotated_density(mo_p[1], mo_occ_p[1], x1b, sigma * t_probe)
    mf_new = nlc.build_mol_mf_nlc(cell, settings)[1]
    scf_p = base.seeded_scf(mf_new, (dma, dmb), plabel, log)
    dm_probe = mf_new.make_rdm1()
    probe_dm_path = os.path.join(dm_dir, f'{plabel}.npz')
    _save_dm_npz(probe_dm_path, dm_probe)
    mf_pstay = nlc.build_mol_mf_nlc(cell, settings)[1]
    stay_probe = base.seeded_scf(mf_pstay, dm_probe, f'{plabel}-stay', log)
    br = ip['bracket_anchors']
    placement = interior_placement(
        scf_p['e_tot'], br['lower']['e'], br['upper']['e'],
        adaptive['interior_dense_below'])
    probe_rec = {
        'label': plabel,
        'point': ip['point'],
        'sigma': sigma,
        't': t_probe,
        'e_final': scf_p['e_tot'],
        'converged': scf_p['converged'],
        'cycles': scf_p['cycles'],
        'stay': {
            'e_tot': stay_probe['e_tot'],
            'converged': stay_probe['converged'],
            'cycles': stay_probe['cycles'],
            'dE_vs_rerun': stay_probe['e_tot'] - scf_p['e_tot'],
        },
        'dm_path': probe_dm_path,
        'dm_sha256': base.sha256_file(probe_dm_path),
        'bracket': {
            'lower_label': br['lower']['label'],
            'lower_e': br['lower']['e'],
            'upper_label': br['upper']['label'],
            'upper_e': br['upper']['e'],
            **placement,
        },
        'probe_gate': bool(scf_p['converged'] and stay_probe['converged']),
    }
    log(f'interior probe {plabel}: E={scf_p["e_tot"]:.12f} '
        f'diff_vs_t050={placement["diff_vs_lower"]:+.3e} '
        f'in_bracket={placement["in_bracket"]} '
        f'below_lower={placement["below_lower"]} '
        f'gate={probe_rec["probe_gate"]}')

    # ---- anchor the fast-stage facts BEFORE any dense build -----------
    facts = {
        'schema_version': SAMPLE_SCHEMA + '-facts',
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'cpu_family_refs': cpu_refs,
        'ladder_refs': ladder_refs,
        'stay_control_origin': stay_p,
        'direction_determination': direction,
        'rungs': {g: rungs[g]['record'] for g in rungs},
        'rung_corroboration': corr,
        'interior_probe': probe_rec,
        'operational_so_far': {
            'wall_s': time.time() - t_start,
            'gpu': 'none (CPU-only; no cupy import)',
            'batch_writes': 'none',
        },
    }
    facts_path = os.path.join(plan['evidence_dir'],
                              'rung-facts.json')
    with open(facts_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=1)
    os.replace(facts_path + '.tmp', facts_path)
    log(f'rung facts anchored: {facts_path}')

    # ---- dense spectra at gated states (adaptive, preregistered) ------
    spectra_summaries = {}
    classifications = {}
    build_queue = []
    mand = plan['rung_dense_mandatory']
    if rungs[mand]['record']['reproduced']:
        build_queue.append((f'rung_{mand}', rungs[mand]['state'],
                            'rung_mandatory'))
    else:
        log(f'WARNING: mandatory rung {mand} NOT reproduced; its '
            'spectrum is SKIPPED (recorded, never silently accepted)')
    other = g1 if mand == g0 else g0
    if not corr['same_rung'] and rungs[other]['record']['reproduced']:
        build_queue.append((f'rung_{other}', rungs[other]['state'],
                            'rung_adaptive_distinct'))
    if placement['interior_dense_branch'] and probe_rec['probe_gate']:
        build_queue.append((plabel, mf_new, 'interior_adaptive_deeper'))
    elif placement['interior_dense_branch']:
        log(f'WARNING: interior probe {plabel} below the t=0.50 minimum '
            'but gate failed (convergence/stay); spectrum SKIPPED')
    for label, state, why in build_queue:
        log(f'dense build {label} ({why})')
        summary, cls = _spectrum_at(state, label, ckpt_dir, log)
        classifications[label] = cls
        spectra_summaries[label] = summary

    wall = time.time() - t_start
    report = {
        'schema_version': SAMPLE_SCHEMA,
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'operator': 'LC-210 reachability rerun (rung dm save + stays) + '
                    'LC-208 dense NLC-inclusive stability Hessian + '
                    'LC-209 descent machinery interior probe (CPU fp64 '
                    'pyscf 2.14.0)',
        'inputs': inputs,
        'ladder_inputs': ladder_inputs,
        'cell': cell,
        'settings': settings,
        'thresholds': thresholds,
        'adaptive': adaptive,
        'cpu_family_refs': cpu_refs,
        'ladder_refs': ladder_refs,
        'stay_control_origin': stay_p,
        'direction_determination': direction,
        'rungs': {g: rungs[g]['record'] for g in rungs},
        'rung_corroboration': corr,
        'interior_probe': probe_rec,
        'build_queue': [{'label': l, 'why': w} for l, _s, w in build_queue],
        'spectra': spectra_summaries,
        'classifications': classifications,
        'operational': {
            'wall_s': wall,
            'gpu': 'none (CPU-only; no cupy import)',
            'batch_writes': 'none',
        },
    }
    with open(out_json + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    os.replace(out_json + '.tmp', out_json)
    if out_md:
        with open(out_md + '.tmp', 'w', encoding='utf-8') as f:
            f.write(render_md(report))
        os.replace(out_md + '.tmp', out_md)
    return report


def render_md(report):
    lines = [
        f"# {report['unit']}",
        '',
        f"Generated {report['generated_at_utc']} | "
        f"batch {report['batch_id']} | schema {report['schema_version']}",
        '',
        '## CPU references (family + LC-210 ladder endpoints)',
        '',
        '| point | E | dE vs anchor |', '|---|---|---|',
    ]
    for p, e in report['cpu_family_refs'].items():
        a = report['inputs'][p]['anchor_e']
        lines.append(f"| {p} | {e:.12f} | {e - a:+.3e} |")
    for p, e in report['ladder_refs'].items():
        a = report['ladder_inputs'][p]['anchor_e_final']
        lines.append(f"| {p} | {e:.12f} | {e - a:+.3e} |")
    d = report['direction_determination']
    lines += [
        '',
        '## Interior-probe direction (loaded LC-208 dense-H)',
        '',
        f"- point {report['interior_probe']['point']} sigma "
        f"{report['interior_probe']['sigma']:+d}: lambda1="
        f"{d['lambda1_dense']:+.6e}, one_col_rel={d['one_column_rel']:.2e}, "
        f"validated={d['validated']} ({d['direction_source']})",
        '',
        '## Rungs (default-guess SCFs, no dm0)',
        '',
        '| guess | E | diff vs LC-210 | cyc | stay cyc | stay dE | '
        'reproduced | class |',
        '|---|---|---|---|---|---|---|---|',
    ]
    for g, r in report['rungs'].items():
        lines.append(
            f"| {g} | {r['e_tot']:.12f} | {r['diff']:+.3e} | "
            f"{r['cycles']} | {r['stay']['cycles']} | "
            f"{r['stay']['dE_vs_rerun']:+.3e} | {r['reproduced']} | "
            f"{r['classification']['class']} |")
    c = report['rung_corroboration']
    lines += [
        '',
        f"Rung corroboration: dE={c['dE']:+.3e} "
        f"(<= {report['adaptive']['same_rung_e']:.0e}: {c['same_rung_e']}), "
        f"dm distance={c['dm_distance']:.6f} "
        f"(<= {report['adaptive']['same_rung_dm']}: {c['same_rung_dm']}) "
        f"-> same rung: {c['same_rung']}",
        '',
        '## Interior probe (t=0.60 on the R line)',
        '',
    ]
    p = report['interior_probe']
    b = p['bracket']
    lines += [
        f"- E = {p['e_final']:.12f} (converged={p['converged']}, "
        f"{p['cycles']} cycles; stay {p['stay']['cycles']} cycles, "
        f"dE {p['stay']['dE_vs_rerun']:+.3e})",
        f"- vs {b['lower_label']} ({b['lower_e']:.12f}): "
        f"{b['diff_vs_lower']:+.3e}; vs {b['upper_label']} "
        f"({b['upper_e']:.12f}): {b['diff_vs_upper']:+.3e}",
        f"- in_bracket={b['in_bracket']} below_lower={b['below_lower']} "
        f"interior_dense_branch={b['interior_dense_branch']}",
        '',
        '## Curvature classifications (dense NLC Hessian, adaptive queue)',
        '',
    ]
    if report['classifications']:
        lines += ['| label | classification | lambda1 | n_negative | '
                  'asym | davidson1 |', '|---|---|---|---|---|---|']
        for label, cls in report['classifications'].items():
            ss = report['spectra'][label]
            lines.append(
                f"| {label} | {cls['classification']} | "
                f"{cls['lambda1']:+.6e} | {cls['n_negative']} | "
                f"{ss['operator_asymmetry']:.2e} | "
                f"{ss['davidson_softest3'][0]:+.6e} |")
    else:
        lines.append('(no gated state reached a dense build)')
    lines += [
        '',
        f"Wall {report['operational']['wall_s']:.1f} s, CPU-only, "
        f"zero GPU, zero batch writes.",
        '',
    ]
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='LC-211 production-reachable rung classification + '
                    'ladder interior probe (CPU fp64 pyscf)')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md')
    ap.add_argument('--log')
    args = ap.parse_args(argv)

    # plain json load (the LC-210 convention): this tool's own
    # validate_rung_plan is the fail-closed schema gate; base.load_plan
    # enforces the softmode profile schema which does not apply here
    with open(args.plan, encoding='utf-8') as f:
        plan = json.load(f)

    def log(msg):
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        line = f'[{stamp}] {msg}'
        print(line, flush=True)
        if args.log:
            with open(args.log, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    log(f'plan {args.plan} sha256={base.sha256_file(args.plan)}')
    report = run(plan, args.out_json, args.out_md, log)
    log(f'done: {len(report["rungs"])} rungs, '
        f'{len(report["classifications"])} classified, wall '
        f'{report["operational"]["wall_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())

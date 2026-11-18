#!/usr/bin/env python3
"""LC-210 wb97mv ladder-point curvature classification + production
reachability (CPU fp64 pyscf).

The LC-209 descent map measured a one-parameter ladder of new UKS
solutions on the methoxy/wb97m-v/def2-svp multi-basin landscape,
descending at least 2.228e-6 Ha below R0_AUTO (deepest measured endpoint
R_+1 t=0.50 = -114.901698537713), reachable only by mode-following
(explicit rotated-dm seeded SCFs).  This unit closes the two named
follow-ups:

  endpoints -- rerun the LC-209 descent machinery at the plan's target
               amplitudes on the R line (sigma=+1): exact exponential
               rotation along the loaded LC-208 dense-H softest mode
               (one-column validated at this run's seeded state, hdiag
               informational per the 2026-10-20 erratum, Davidson
               fallback direction on validation failure), seeded SCF on
               a fresh object, SAVE the endpoint dm as npz, and prove
               the npz round-trip + own-basin stationarity with a
               fresh-object stay control (1-cycle class)
  curvature -- dense NLC-inclusive internal-stability Hessian (the
               LC-65/LC-208 gen_g_hop_uhf_nlc clone, stability() x2
               convention) at each REPRODUCED endpoint state,
               column-resumable npz checkpoints, dense exact eigh;
               classification = count of negative eigenvalues
               (minimum / saddle_index_N)
  reach    -- production-trajectory reachability: fresh UKS objects
               with init_guess in {minao, atom, huckel, vsap, 1e}
               (NO dm0 seed), batch settings; destination classified
               by energy match against the four family CPU references
               AND the reproduced ladder endpoints, with AO dm-distance
               corroboration; plus the below-R0_AUTO flag (the deep-rung
               reachability question)

The endpoint-facts block (reruns + stays + reachability) is written to
disk BEFORE the dense builds so a session ceiling loses nothing but
Hessian columns, which resume from the npz checkpoints.

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

SAMPLE_SCHEMA = 'ladder-classify-v1'

REQUIRED_PLAN_KEYS = (
    'unit', 'amendment_id', 'batch_id', 'evidence_dir', 'cell',
    'settings', 'family_dms', 'descent', 'reachability', 'thresholds',
    'spectra_ckpt_dir',
)
REQUIRED_SETTINGS_KEYS = base.REQUIRED_SETTINGS_KEYS + ('nlcgrids_level',)

SUPPORTED_GUESSES = ('minao', 'atom', 'huckel', 'vsap', '1e')

DEFAULT_THRESHOLDS = {
    'reproduce_tol': 1e-9,      # |e_rerun - LC-209 e_final| for a
                                # reproduction (the classification is
                                # only built on reproduced points)
    'one_column_rel': 1e-6,    # ||hop(v1)*2 - H v1|| / ||H v1|| for the
                                # loaded direction (hdiag informational
                                # only, the 2026-10-20 erratum)
    'basin_energy': 1e-8,      # |E - ref| for a destination match
    'new_dm_distance': 0.05,   # min AO dm distance for new-basin class
}


def validate_ladder_plan(plan):
    '''Fail-closed plan checks before any numeric work (stdlib only).'''
    problems = []
    try:
        base.verify_inputs(plan)
    except (RuntimeError, KeyError, TypeError) as e:
        problems.append(str(e))
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            problems.append(f'missing plan key {key!r}')
    for key in REQUIRED_SETTINGS_KEYS:
        if key not in plan['settings']:
            problems.append(f'missing settings key {key!r}')
    fam = plan.get('family_dms', {})
    for p in base.FAMILY_POINTS:
        if p not in fam:
            problems.append(f'family_dms missing {p!r} (needed for '
                            'reachability classification)')
        else:
            for key in ('path', 'anchor_e', 'anchor_source'):
                if key not in fam[p]:
                    problems.append(f'family point {p!r} missing {key!r}')

    desc = plan.get('descent', {})
    if not isinstance(desc, dict):
        problems.append('descent must be an object')
    else:
        if desc.get('point') not in base.FAMILY_POINTS:
            problems.append(f'descent point {desc.get("point")!r} is not '
                            'a family point')
        if desc.get('sigma') not in (1, -1):
            problems.append('descent sigma must be exactly 1 or -1')
        ck = desc.get('hessian_ckpt')
        if not isinstance(ck, str) or not ck:
            problems.append('descent.hessian_ckpt must be a path string')
        elif not os.path.isfile(ck):
            problems.append(f'descent hessian ckpt missing: {ck}')
        targets = desc.get('targets')
        if (not isinstance(targets, list) or not targets):
            problems.append('descent.targets must be a non-empty list')
        else:
            labels = [t.get('label') for t in targets]
            if len(set(labels)) != len(labels) or any(not l for l in labels):
                problems.append('target labels must be non-empty and '
                                'unique')
            ts = [t.get('t') for t in targets]
            if any(not (isinstance(t, (int, float)) and t > 0)
                   for t in ts):
                problems.append('target amplitudes t must be positive '
                                'numbers')
            elif sorted(ts) != list(ts) or len(set(ts)) != len(ts):
                problems.append('target amplitudes must be unique and '
                                'ascending')
            for t in targets:
                if not isinstance(t.get('anchor_e_final'), (int, float)):
                    problems.append(f'target {t.get("label")!r} missing '
                                    'numeric anchor_e_final')
                if not isinstance(t.get('anchor_source'), str):
                    problems.append(f'target {t.get("label")!r} missing '
                                    'anchor_source')

    reach = plan.get('reachability', {})
    if not isinstance(reach, dict):
        problems.append('reachability must be an object')
    else:
        guesses = reach.get('guesses')
        if (not isinstance(guesses, list) or not guesses
                or not all(isinstance(g, str) for g in guesses)):
            problems.append('reachability.guesses must be a non-empty '
                            'list of strings')
        else:
            if len(set(guesses)) != len(guesses):
                problems.append('reachability.guesses must be unique')
            for g in guesses:
                if g not in SUPPORTED_GUESSES:
                    problems.append(
                        f'reachability guess {g!r} is not one of '
                        f'{SUPPORTED_GUESSES}')

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
        problems.append('ladder classify takes no spectra_points; '
                        'spectra are built at the reproduced endpoints')
    return problems


# ---------------------------------------------------------------------------
# pure classification logic (stdlib-only; unit-tested)
# ---------------------------------------------------------------------------

def classify_reproduction(converged, e_final, anchor_e, tol):
    '''Endpoint-rerun reproduction record: converged AND within tol of the
    LC-209 anchor e_final (the curvature classification is only built on
    reproduced points; misses are recorded, never silently accepted).'''
    diff = abs(float(e_final) - float(anchor_e))
    return {
        'converged': bool(converged),
        'e_final': float(e_final),
        'anchor_e_final': float(anchor_e),
        'diff': diff,
        'reproduced': bool(converged) and diff <= tol,
    }


def classify_spectrum(eigenvalues):
    '''Curvature classification from a dense-H eigenvalue list: the count
    of negative eigenvalues is the saddle index; zero negatives = minimum.'''
    evals = [float(w) for w in eigenvalues]
    n_negative = sum(1 for w in evals if w < 0.0)
    return {
        'dim': len(evals),
        'n_negative': n_negative,
        'lambda1': min(evals) if evals else None,
        'is_minimum': n_negative == 0,
        'classification': ('minimum' if n_negative == 0
                           else f'saddle_index_{n_negative}'),
    }


def below_reference(e_final, ref_e, tol):
    '''True when e_final sits below ref_e by more than tol (the deep-rung
    reachability flag).'''
    return bool(float(e_final) < float(ref_e) - tol)


def reachability_summary(rows):
    '''Aggregate the reachability probe rows: class counts, convergence,
    any below-R0_AUTO landing (rows carry the flag from the numeric
    layer), and the lowest reached energy.'''
    classes = {}
    for r in rows:
        c = r['classification']['class']
        classes[c] = classes.get(c, 0) + 1
    return {
        'n_runs': len(rows),
        'all_converged': all(r['converged'] for r in rows) if rows else False,
        'class_counts': classes,
        'any_below_r0_auto': any(r.get('below_r0_auto')
                                 for r in rows) if rows else False,
        'min_e_tot': (min(r['e_tot'] for r in rows) if rows else None),
        'guesses': [r['guess'] for r in rows],
    }


# ---------------------------------------------------------------------------
# numeric layer (numpy/pyscf imported lazily)
# ---------------------------------------------------------------------------

def _davidson_softest(g, hop, hdiag):
    '''pyscf Davidson softest-3 record + the softest vector (the LC-209
    fallback-direction convention: pairs sorted by eigenvalue so the
    fallback is the SOFTEST root's own vector).'''
    import numpy as np
    from pyscf import lib as pyscf_lib

    def hop2(x):
        return hop(x).real * 2

    hdiag2 = np.asarray(hdiag, dtype=float) * 2

    def precond(dx, e, x0):
        hd = hdiag2 - e
        hd[abs(hd) < 1e-8] = 1e-8
        return dx / hd

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag2[g != 0]
    e_dav, v_dav = pyscf_lib.davidson(hop2, x0, precond, tol=1e-9,
                                      nroots=3, verbose=0)
    pairs = sorted(
        [(float(e), np.asarray(vv).ravel())
         for e, vv in zip(np.asarray(e_dav).ravel(), v_dav)],
        key=lambda kv: kv[0])
    e_dav = np.asarray([pe for pe, _ in pairs[:3]])
    return e_dav, pairs[0][1]


def _save_dm_npz(path, dm_final):
    '''Atomic npz save in the established (2,nao,nao) storage form.'''
    import numpy as np
    arr = np.asarray(dm_final)
    if arr.ndim != 3 or arr.shape[0] != 2:
        raise RuntimeError(f'endpoint dm is not the 2-spin storage form: '
                           f'{arr.shape}')
    with open(path + '.tmp', 'wb') as f:
        np.savez(f, dm=arr)
    os.replace(path + '.tmp', path)
    return path


def run(plan, out_json, out_md, log):
    import numpy as np

    problems = validate_ladder_plan(plan)
    if problems:
        raise RuntimeError('plan validation failed:\n  '
                            + '\n  '.join(problems))
    cell, settings = plan['cell'], plan['settings']
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(plan.get('thresholds', {}))
    desc = plan['descent']
    ckpt_dir = plan['spectra_ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    dm_dir = os.path.join(plan['evidence_dir'], 'dm')
    os.makedirs(dm_dir, exist_ok=True)

    mol, mf = nlc.build_mol_mf_nlc(cell, settings)
    log(f'cell {cell["tag"]}: nao={mol.nao_nr()} nelec={mol.nelec} '
        f'spin={cell["spin"]}')

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

    cpu_refs = {}
    for p in base.FAMILY_POINTS:
        e = base.energy_at(mf, dms[p])
        cpu_refs[p] = e
        log(f'cpu ref {p}: E={e:.12f} '
            f'dE_vs_gpu_anchor={e - inputs[p]["anchor_e"]:+.3e}')

    t_start = time.time()

    # descent point stay control (fresh object) -- the rotation source
    mf_p = nlc.build_mol_mf_nlc(cell, settings)[1]
    stay = base.seeded_scf(mf_p, dms[desc['point']], desc['point'], log)
    if not stay['converged']:
        raise RuntimeError(f'stay control at {desc["point"]} did not '
                           'converge; refusing to descend from a '
                           'non-stationary seed')

    # loaded direction + validation at THIS run's state (LC-209 fixed
    # convention: one-column check decides, hdiag informational only)
    H, w, v, asym = dm.load_ckpt_spectrum(desc['hessian_ckpt'])
    v1 = v[:, 0]
    lambda1 = float(w[0])
    g, hop, hdiag, _dims = nlc.gen_g_hop_uhf_nlc(
        mf_p, mf_p.mo_coeff, mf_p.mo_occ)
    rel, hdiag_dev = dm.one_column_validation(hop, hdiag, H, v1)
    e_dav, v_dav0 = _davidson_softest(g, hop, hdiag)
    cos_dav = base.cosine(v_dav0, v1)
    validated = rel <= thresholds['one_column_rel']
    source = 'loaded_dense_validated' if validated else 'davidson_fallback'
    if not validated:
        log(f'WARNING: loaded-H validation failed (one_col={rel:.2e}); '
            f'descending along the Davidson softest instead')
    direction = {
        'ckpt_path': desc['hessian_ckpt'],
        'ckpt_sha256': base.sha256_file(desc['hessian_ckpt']),
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
    log(f'direction {desc["point"]}: lambda1={lambda1:+.6e} '
        f'one_col_rel={rel:.2e} validated={validated} '
        f'davidson1={e_dav[0]:+.6e} cos={cos_dav}')
    v_dir = v1 if validated else v_dav0

    mo_p, mo_occ_p = mf_p.mo_coeff, mf_p.mo_occ
    x1a, x1b = base.split_x(v_dir, mo_occ_p[0], mo_occ_p[1])
    sigma = desc['sigma']

    # endpoint reruns (rotated seeds -> fresh-object seeded SCF -> saved dm
    # -> fresh-object stay control from the saved dm)
    endpoints = {}
    for tgt in desc['targets']:
        label, t = tgt['label'], tgt['t']
        dma = base.rotated_density(mo_p[0], mo_occ_p[0], x1a, sigma * t)
        dmb = base.rotated_density(mo_p[1], mo_occ_p[1], x1b, sigma * t)
        dm_rot = (dma, dmb)
        mf_new = nlc.build_mol_mf_nlc(cell, settings)[1]
        scf = base.seeded_scf(mf_new, dm_rot, label, log)
        dm_final = mf_new.make_rdm1()
        repro = classify_reproduction(
            scf['converged'], scf['e_tot'], tgt['anchor_e_final'],
            thresholds['reproduce_tol'])
        repro.update({
            't': t,
            'sigma': sigma,
            'point': desc['point'],
            'cycles': scf['cycles'],
            'anchor_source': tgt['anchor_source'],
        })
        dm_path = os.path.join(dm_dir, f'{label}.npz')
        _save_dm_npz(dm_path, dm_final)
        mf_stay = nlc.build_mol_mf_nlc(cell, settings)[1]
        stay2 = base.seeded_scf(mf_stay, dm_final, f'{label}-stay', log)
        repro['stay'] = {
            'e_tot': stay2['e_tot'],
            'converged': stay2['converged'],
            'cycles': stay2['cycles'],
            'dE_vs_rerun': stay2['e_tot'] - scf['e_tot'],
        }
        repro['dm_path'] = dm_path
        repro['dm_sha256'] = base.sha256_file(dm_path)
        endpoints[label] = {
            'reproduction': repro,
            'state': mf_new,
            'dm_final': dm_final,
        }
        flag = '' if repro['reproduced'] else ' NOT-REPRODUCED (spectrum will be SKIPPED)'
        log(f'endpoint {label}: E={scf["e_tot"]:.12f} '
            f'diff_vs_LC209={repro["diff"]:+.3e} '
            f'stay_cyc={stay2["cycles"]} '
            f'stay_dE={repro["stay"]["dE_vs_rerun"]:+.3e}{flag}')

    # reachability: default-guess trajectories vs family + ladder refs
    refs = dict(cpu_refs)
    ref_dms = dict(dms)
    for label, ep in endpoints.items():
        if ep['reproduction']['reproduced']:
            refs[label] = ep['reproduction']['e_final']
            ref_dms[label] = ep['dm_final']
    reach_rows = []
    for guess in plan['reachability']['guesses']:
        mf_g = nlc.build_mol_mf_nlc(cell, settings)[1]
        mf_g.init_guess = guess
        t0 = time.time()
        e = mf_g.kernel()
        wall = time.time() - t0
        dm_g = mf_g.make_rdm1()
        dm_distances = {
            q: float(sum(np.linalg.norm(
                np.asarray(dm_g[k]) - np.asarray(ref_dms[q][k]))
                for k in range(2)))
            for q in ref_dms}
        cls = dm.classify_destination(
            bool(mf_g.converged), float(e), refs, dm_distances,
            thresholds)
        row = {
            'guess': guess,
            'e_tot': float(e),
            'converged': bool(mf_g.converged),
            'cycles': int(getattr(mf_g, 'cycles', 0) or 0),
            'wall_s': wall,
            'dm_distances': dm_distances,
            'classification': cls,
            'below_r0_auto': below_reference(
                float(e), cpu_refs['R0_AUTO'], thresholds['basin_energy']),
        }
        reach_rows.append(row)
        log(f'reachability {guess}: E={e:.12f} conv={mf_g.converged} '
            f'cyc={row["cycles"]} class={cls["class"]} '
            f'below_r0_auto={row["below_r0_auto"]}')
    reach_summary = reachability_summary(reach_rows)

    # anchor the fast-stage facts BEFORE the dense builds
    facts = {
        'schema_version': SAMPLE_SCHEMA + '-endpoints',
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'cpu_family_refs': cpu_refs,
        'stay_control_origin': stay,
        'direction_determination': direction,
        'endpoints': {label: ep['reproduction']
                      for label, ep in endpoints.items()},
        'reachability': {'rows': reach_rows, 'summary': reach_summary},
        'operational_so_far': {
            'wall_s': time.time() - t_start,
            'gpu': 'none (CPU-only; no cupy import)',
            'batch_writes': 'none',
        },
    }
    facts_path = os.path.join(plan['evidence_dir'],
                              'ladder-endpoint-facts.json')
    with open(facts_path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=1)
    os.replace(facts_path + '.tmp', facts_path)
    log(f'endpoint facts anchored: {facts_path}')

    # dense NLC-inclusive spectra at the REPRODUCED endpoints only
    spectra_summaries = {}
    classifications = {}
    for label, ep in endpoints.items():
        if not ep['reproduction']['reproduced']:
            log(f'WARNING: {label} not reproduced; Hessian SKIPPED '
                '(recorded, never silently accepted)')
            continue
        ck = os.path.join(ckpt_dir, f'dense-H-{label}.npz')
        spec = nlc.dense_stability_nlc(ep['state'], log, ckpt=ck)
        cls = classify_spectrum(spec['eigenvalues'].tolist())
        classifications[label] = cls
        spectra_summaries[label] = nlc.spectrum_summary(spec)
        spectra_summaries[label]['classification'] = cls
        interim = {
            'label': label,
            'generated_at_utc': base.utc_now(),
            'ckpt_path': ck,
            'ckpt_sha256': base.sha256_file(ck),
            'classification': cls,
            'softest_10': spectra_summaries[label]['softest_10'],
            'operator_asymmetry': spectra_summaries[label][
                'operator_asymmetry'],
            'davidson_softest3': spectra_summaries[label][
                'davidson_softest3'],
        }
        ipath = os.path.join(ckpt_dir, f'filelevel-eigh-{label}'
                                        '-interim.json')
        with open(ipath + '.tmp', 'w', encoding='utf-8') as f:
            json.dump(interim, f, indent=1)
        os.replace(ipath + '.tmp', ipath)
        log(f'spectrum {label}: {cls["classification"]} '
            f'lambda1={cls["lambda1"]:+.6e} '
            f'n_negative={cls["n_negative"]}')

    wall = time.time() - t_start
    report = {
        'schema_version': SAMPLE_SCHEMA,
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'operator': 'LC-209 descent rerun (endpoint dm save) + LC-208 '
                    'dense NLC-inclusive stability Hessian + default-'
                    'guess reachability (CPU fp64 pyscf 2.14.0)',
        'inputs': inputs,
        'descent': {
            'point': desc['point'],
            'sigma': sigma,
            'targets': desc['targets'],
        },
        'cell': cell,
        'settings': settings,
        'thresholds': thresholds,
        'cpu_family_refs': cpu_refs,
        'stay_control_origin': stay,
        'direction_determination': direction,
        'endpoints': {label: ep['reproduction']
                      for label, ep in endpoints.items()},
        'reachability': {'guesses': plan['reachability']['guesses'],
                         'rows': reach_rows,
                         'summary': reach_summary},
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
        '## CPU family references',
        '',
        '| point | E_cpu | anchor | dE |',
        '|---|---|---|---|',
    ]
    for p, e in report['cpu_family_refs'].items():
        a = report['inputs'][p]['anchor_e']
        lines.append(f'| {p} | {e:.12f} | {a:.12f} | {e - a:+.3e} |')
    d = report['direction_determination']
    lines += ['', '## Loaded direction (descent origin)', '',
              f"- point {report['descent']['point']} sigma "
              f"{report['descent']['sigma']:+d}: lambda1="
              f"{d['lambda1_dense']:+.6e} one_col_rel={d['one_column_rel']:.2e} "
              f"validated={d['validated']} source={d['direction_source']} "
              f"davidson1={d['davidson_softest3'][0]:+.6e} "
              f"cos={d['cos_davison_dense']}",
              '', '## Endpoint reruns (LC-209 reproduction + stays)', '',
              '| label | t | E_rerun | diff vs LC-209 | cycles | '
              'reproduced | stay cycles | stay dE | dm sha256 |',
              '|---|---|---|---|---|---|---|---|---|']
    for label, r in report['endpoints'].items():
        lines.append(
            f"| {label} | {r['t']:.2f} | {r['e_final']:.12f} "
            f"| {r['diff']:+.3e} | {r['cycles']} | {r['reproduced']} "
            f"| {r['stay']['cycles']} | {r['stay']['dE_vs_rerun']:+.3e} "
            f"| {r['dm_sha256'][:12]}.. |")
    lines += ['', '## Reachability (default-guess trajectories)', '',
              '| guess | E | converged | cycles | class | below R0_AUTO |',
              '|---|---|---|---|---|---|']
    for r in report['reachability']['rows']:
        lines.append(
            f"| {r['guess']} | {r['e_tot']:.12f} | {r['converged']} "
            f"| {r['cycles']} | {r['classification']['class']} "
            f"| {r['below_r0_auto']} |")
    s = report['reachability']['summary']
    lines += ['', f"summary: {s['n_runs']} runs, all_converged="
              f"{s['all_converged']}, any_below_r0_auto="
              f"{s['any_below_r0_auto']}, classes={s['class_counts']}",
              '', '## Endpoint curvature (dense NLC-inclusive eigh)', '',
              '| label | classification | lambda1 | n_negative | dim | '
              'asym | davidson1 |', '|---|---|---|---|---|---|---|']
    for label, cls in report['classifications'].items():
        ss = report['spectra'][label]
        lines.append(
            f"| {label} | {cls['classification']} "
            f"| {cls['lambda1']:+.6e} | {cls['n_negative']} "
            f"| {cls['dim']} | {ss['operator_asymmetry']:.2e} "
            f"| {ss['davidson_softest3'][0]:+.6e} |")
    lines += ['',
              f"Wall {report['operational']['wall_s']:.1f} s, CPU-only, "
              f"zero batch writes.", '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='LC-210 ladder-point curvature classification + '
                    'production reachability (CPU fp64 pyscf)')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md')
    ap.add_argument('--log')
    ap.add_argument('--ckpt-every', type=int, default=100)
    args = ap.parse_args(argv)

    with open(args.plan, encoding='utf-8') as f:
        plan = json.load(f)

    def log(msg):
        stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        line = f'[{stamp}] {msg}'
        print(line, flush=True)
        if args.log:
            with open(args.log, 'a', encoding='utf-8') as f:
                f.write(line + '\n')

    problems = validate_ladder_plan(plan)
    if problems:
        for p in problems:
            log(f'PLAN PROBLEM: {p}')
        return 2
    report = run(plan, args.out_json, args.out_md, log)
    log(f'done: {len(report["endpoints"])} endpoints, '
        f'{len(report["classifications"])} classified, '
        f'{len(report["reachability"]["rows"])} reachability runs, '
        f'wall {report["operational"]["wall_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())

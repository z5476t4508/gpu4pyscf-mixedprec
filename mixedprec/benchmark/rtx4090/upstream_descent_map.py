#!/usr/bin/env python3
"""LC-209 wb97mv saddle-descent basin-connectivity map (CPU fp64 pyscf).

Follows the soft NEGATIVE modes of the two lane-saddle solutions of the
methoxy/wb97m-v/def2-svp UKS multi-basin landscape (batch uc-20261018-a
FAIL rows; LC-208 curvature characterization) with exact exponential
orbital rotations and seeded SCFs, and maps where each descent flows --
the terminal basin(s) of the only measured minimum R0_AUTO (-1.283e-6
below R) versus any new basin.  This is the "unstable-direction search"
named as future work in the issue-7 deferred record.

Points and directions:

  stay      -- fresh seeded SCF at each descent point's dm (the LC-206
               P6 STAY reproduction); CPU-side reference energies for
               all four family dms (direct evaluation, one shared
               object -- the destination classification baseline)
  direction -- the softest eigenvector of the LC-208 dense-H checkpoint
               matrix per point (loaded, NOT rebuilt), validated against
               a fresh one-column operator application at THIS run's
               seeded state (hop(v1) vs H v1) plus an hdiag cross-check
               and a pyscf Davidson-3 consistency record; on validation
               failure the Davidson softest vector is the recorded
               fallback direction
  descent   -- per direction sign and amplitude t: exact exponential
               rotation of the point's canonical MOs along +-t*v1 (the
               LC-207/208 profile-scan machinery), a single-point energy
               at the rotated dm (the descent profile + the local
               quadratic ratio), and a seeded SCF from the rotated dm on
               a FRESH object (e_tot/cycles/converged)
  classify  -- destination by energy match to the four CPU family
               references (<= 1e-8 -> that basin) with AO density
               distance corroboration; new-basin candidate when far
               from all four by BOTH energy and dm distance

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

SAMPLE_SCHEMA = 'descent-map-v1'

# self-contained (the base key list carries a 'profile' block this tool
# does not use; the descent grid replaces it)
REQUIRED_PLAN_KEYS = (
    'unit', 'amendment_id', 'batch_id', 'evidence_dir', 'cell',
    'settings', 'family_dms', 'descent_points', 'amplitudes',
    'directions', 'hessian_ckpt', 'thresholds',
)
REQUIRED_SETTINGS_KEYS = base.REQUIRED_SETTINGS_KEYS + ('nlcgrids_level',)

DEFAULT_AMPLITUDES = (0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70)
DEFAULT_DIRECTIONS = (1, -1)
DEFAULT_THRESHOLDS = {
    'basin_energy': 1e-8,        # |E_final - E_cpu_ref| for a basin match
    'new_dm_distance': 0.05,    # min AO dm distance for a new-basin candidate
    'one_column_rel': 1e-6,     # ||hop(v1)*2 - H v1|| / ||H v1|| validation
    # informational only (see the 2026-10-20 erratum): pyscf's gen hdiag
    # is a PRECONDITIONER diagonal, NOT diag(2*hop), for this UKS+NLC
    # operator -- LC-208's own dense builds measured hdiag_vs_dense_dev
    # = 7.98e-01 at every point; the loaded-H validation is the
    # one-column check above
    'hdiag_dev_info_only': True,
}


def validate_descent_plan(plan):
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
                            'destination classification)')
        else:
            for key in ('path', 'anchor_e', 'anchor_source'):
                if key not in fam[p]:
                    problems.append(f'family point {p!r} missing {key!r}')
    if 'descent_points' in plan:
        pts = plan['descent_points']
        if not isinstance(pts, list) or not pts:
            problems.append('descent_points must be a non-empty list')
        else:
            for p in pts:
                if p not in base.FAMILY_POINTS:
                    problems.append(f'unknown descent point {p!r}')
                if p not in plan.get('hessian_ckpt', {}):
                    problems.append(
                        f'descent point {p!r} has no hessian_ckpt entry')
    if 'amplitudes' in plan:
        amps = plan['amplitudes']
        if not isinstance(amps, list) or not amps:
            problems.append('amplitudes must be a non-empty list')
        elif any(not (isinstance(a, (int, float)) and a > 0)
                 for a in amps):
            problems.append('amplitudes must be positive numbers')
        elif sorted(amps) != list(amps) or len(set(amps)) != len(amps):
            problems.append('amplitudes must be unique and ascending')
    if 'directions' in plan:
        dirs = plan['directions']
        if not isinstance(dirs, list) or not dirs:
            problems.append('directions must be a non-empty list')
        elif sorted(dirs) != [-1, 1]:
            problems.append('directions must be exactly [1, -1]')
    if 'thresholds' in plan:
        th = plan['thresholds']
        if not isinstance(th, dict):
            problems.append('thresholds must be an object')
        else:
            for key, value in DEFAULT_THRESHOLDS.items():
                if key in th and not (isinstance(th[key], (int, float))
                                      and th[key] > 0):
                    problems.append(f'threshold {key!r} must be positive')
    if 'hessian_ckpt' in plan:
        ck = plan['hessian_ckpt']
        if not isinstance(ck, dict) or not ck:
            problems.append('hessian_ckpt must be a non-empty object')
        else:
            for p, entry in ck.items():
                if p not in base.FAMILY_POINTS:
                    problems.append(f'unknown hessian_ckpt point {p!r}')
                if not isinstance(entry, dict) or 'path' not in entry:
                    problems.append(
                        f'hessian_ckpt[{p!r}] must carry a path')
                elif not os.path.isfile(entry['path']):
                    problems.append(f'hessian ckpt missing for {p!r}: '
                                    f'{entry["path"]}')
    if plan.get('spectra_points') is not None:
        problems.append('descent map takes no spectra_points; '
                        'hessians come from hessian_ckpt')
    return problems


def load_ckpt_spectrum(path):
    '''Load a COMPLETE dense-H checkpoint and eigendecompose it.'''
    import numpy as np
    z = np.load(path)
    H = np.asarray(z['H'], dtype=float)
    n_done = int(z['n_done'])
    dim = H.shape[0]
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise RuntimeError(f'checkpoint {path} is not a square matrix')
    if n_done != dim:
        raise RuntimeError(f'checkpoint {path} incomplete: {n_done}/{dim} '
                           'columns; the descent map requires the fully '
                           'built LC-208 matrix')
    asym = float(np.abs(H - H.T).max())
    w, v = np.linalg.eigh(H)
    return H, w, v, asym


def one_column_validation(hop, hdiag, H, v1):
    '''Fresh-operator validation of the loaded direction at THIS state:
    hop(v1)*2 must reproduce H @ v1 (same operator, same MO basis) and
    hdiag*2 must match diag(H).  Returns the two diagnostics.'''
    import numpy as np
    hv = hop(np.asarray(v1, dtype=float)).real * 2
    ref = H @ v1
    ref_norm = float(np.linalg.norm(ref))
    rel = (float(np.linalg.norm(hv - ref)) / ref_norm
           if ref_norm > 0 else float('inf'))
    hdiag_dev = float(np.abs(np.asarray(hdiag, dtype=float) * 2
                             - np.diag(H)).max())
    return rel, hdiag_dev


def classify_destination(converged, e_final, cpu_refs, dm_distances,
                         thresholds):
    '''Destination classification (pure, stdlib-testable).

    basin_<p>   -- |e_final - cpu_refs[p]| <= basin_energy
    new_candidate -- farther than basin_energy from every reference by
                  energy AND farther than new_dm_distance by dm distance
    unclassified -- energy far but dm close to some family point (record
                  as-is; the report carries every number either way)
    not_converged -- the seeded SCF did not converge (honest record)
    '''
    diffs = {p: abs(e_final - refs) for p, refs in cpu_refs.items()}
    nearest = min(diffs, key=diffs.get)
    if not converged:
        return {'class': 'not_converged', 'nearest_by_energy': nearest}
    if diffs[nearest] <= thresholds['basin_energy']:
        smallest_dm = min(dm_distances, key=dm_distances.get)
        return {
            'class': 'basin_' + nearest,
            'nearest_by_energy': nearest,
            'energy_diff': diffs[nearest],
            'dm_smallest_matches': bool(smallest_dm == nearest),
        }
    if all(dist > thresholds['new_dm_distance']
           for dist in dm_distances.values()):
        return {'class': 'new_candidate', 'nearest_by_energy': nearest,
                'energy_diff': diffs[nearest]}
    return {'class': 'unclassified', 'nearest_by_energy': nearest,
            'energy_diff': diffs[nearest]}


def summarize_lines(runs, thresholds):
    '''Per (point, direction) line summary: the tipping amplitude (first
    t whose final energy leaves the origin's CPU reference by more than
    basin_energy) and the terminal basin of the largest amplitude.'''
    lines = {}
    for r in runs:
        key = (r['point'], r['sigma'])
        lines.setdefault(key, []).append(r)
    out = {}
    for (p, sigma), recs in lines.items():
        recs = sorted(recs, key=lambda r: r['t'])
        origin_ref = recs[0]['e_cpu_origin']
        tipped = [r['t'] for r in recs
                  if r['converged']
                  and abs(r['e_final'] - origin_ref) > thresholds['basin_energy']]
        out[f'{p}_{sigma:+d}'] = {
            'tipping_t': min(tipped) if tipped else None,
            'terminal': (recs[-1]['classification']['class']
                         if recs else None),
            'final_classes': {f"t={r['t']}": r['classification']['class']
                              for r in recs},
        }
    return out


def run(plan, out_json, out_md, log):
    import numpy as np

    problems = validate_descent_plan(plan)
    if problems:
        raise RuntimeError('plan validation failed:\n  '
                            + '\n  '.join(problems))
    cell, settings = plan['cell'], plan['settings']
    amplitudes = plan['amplitudes']
    directions = plan['directions']
    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(plan.get('thresholds', {}))

    mol, mf = nlc.build_mol_mf_nlc(cell, settings)
    log(f'cell {cell["tag"]}: nao={mol.nao_nr()} nelec={mol.nelec} '
        f'spin={cell["spin"]}')

    dms = {}
    for p in base.FAMILY_POINTS:
        raw = base.dm_from_npz(plan['family_dms'][p]['path'])
        # npz stores (2,nao,nao); normalize to the 2-spin tuple form the
        # rotation/classification machinery consumes (LC-208 as_ukp_spin2)
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

    # CPU-side family references (one shared object, direct evaluation)
    cpu_refs = {}
    for p in base.FAMILY_POINTS:
        e = base.energy_at(mf, dms[p])
        cpu_refs[p] = e
        log(f'cpu ref {p}: E={e:.12f} '
            f'dE_vs_gpu_anchor={e - inputs[p]["anchor_e"]:+.3e}')

    t_start = time.time()
    stay_controls = {}
    direction_determination = {}
    descent_runs = []
    e_cpu_origin = {}

    for p in plan['descent_points']:
        # fresh object per point: the stay control IS the state provider
        mf_p = nlc.build_mol_mf_nlc(cell, settings)[1]
        stay = base.seeded_scf(mf_p, dms[p], p, log)
        stay_controls[p] = stay
        e_cpu_origin[p] = stay['e_tot']
        if not stay['converged']:
            raise RuntimeError(f'stay control at {p} did not converge; '
                               'refusing to descend from a non-stationary '
                               'seed')

        # loaded direction + validation at THIS run's state
        ck = plan['hessian_ckpt'][p]['path']
        H, w, v, asym = load_ckpt_spectrum(ck)
        v1 = v[:, 0]
        lambda1 = float(w[0])
        g, hop, hdiag, _dims = nlc.gen_g_hop_uhf_nlc(
            mf_p, mf_p.mo_coeff, mf_p.mo_occ)
        rel, hdiag_dev = one_column_validation(hop, hdiag, H, v1)

        # Davidson consistency + fallback direction
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
        # davidson does not guarantee root ordering: pair and sort so the
        # fallback direction is the SOFTEST root's own vector
        pairs = sorted(
            [(float(e), np.asarray(vv).ravel())
             for e, vv in zip(np.asarray(e_dav).ravel(), v_dav)],
            key=lambda kv: kv[0])
        e_dav = np.asarray([pe for pe, _ in pairs[:3]])
        v_dav0 = pairs[0][1]
        cos_dav = base.cosine(v_dav0, v1)

        validated = rel <= thresholds['one_column_rel']
        source = 'loaded_dense_validated' if validated else 'davidson_fallback'
        if not validated:
            log(f'WARNING {p}: loaded-H validation failed '
                f'(one_col={rel:.2e}); descending along the Davidson '
                f'softest instead')
        direction_determination[p] = {
            'ckpt_path': ck,
            'ckpt_sha256': base.sha256_file(ck),
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
        log(f'direction {p}: lambda1={lambda1:+.6e} one_col_rel={rel:.2e} '
            f'hdiag_dev={hdiag_dev:.2e} validated={validated} '
            f'davidson1={e_dav[0]:+.6e} cos={cos_dav}')
        v_dir = v1 if validated else v_dav0
        lambda1_eff = (lambda1 if validated else float(e_dav[0]))

        mo_p, mo_occ_p = mf_p.mo_coeff, mf_p.mo_occ
        x1a, x1b = base.split_x(v_dir, mo_occ_p[0], mo_occ_p[1])
        for sigma in directions:
            for t in amplitudes:
                dma = base.rotated_density(mo_p[0], mo_occ_p[0], x1a,
                                           sigma * t)
                dmb = base.rotated_density(mo_p[1], mo_occ_p[1], x1b,
                                           sigma * t)
                dm_rot = (dma, dmb)
                e_rot = base.energy_at(mf, dm_rot)
                quad_pred = 0.5 * lambda1_eff * t * t
                mf_new = nlc.build_mol_mf_nlc(cell, settings)[1]
                scf = base.seeded_scf(
                    mf_new, dm_rot, f'{p} sigma={sigma:+d} t={t}', log)
                dm_final = mf_new.make_rdm1()
                dm_distances = {
                    q: float(sum(np.linalg.norm(
                        np.asarray(dm_final[k])
                        - np.asarray(dms[q][k])) for k in range(2)))
                    for q in base.FAMILY_POINTS}
                cls = classify_destination(
                    scf['converged'], scf['e_tot'], cpu_refs,
                    dm_distances, thresholds)
                rec = {
                    'point': p,
                    'sigma': sigma,
                    't': t,
                    'e_rot': e_rot,
                    'dE_rot_vs_stay': e_rot - stay['e_tot'],
                    'quad_ratio_rot': (
                        (e_rot - stay['e_tot']) / quad_pred
                        if quad_pred != 0 else None),
                    'converged': scf['converged'],
                    'e_final': scf['e_tot'],
                    'cycles': scf['cycles'],
                    'dE_final_vs_stay': scf['e_tot'] - stay['e_tot'],
                    'e_cpu_origin': stay['e_tot'],
                    'dm_distances': dm_distances,
                    'energy_diffs_vs_cpu_refs': {
                        q: abs(scf['e_tot'] - cpu_refs[q])
                        for q in base.FAMILY_POINTS},
                    'classification': cls,
                }
                descent_runs.append(rec)
                log(f'descent {p} sigma={sigma:+d} t={t:.2f}: '
                    f'dE_rot={rec["dE_rot_vs_stay"]:+.3e} '
                    f'ratio={rec["quad_ratio_rot"] if rec["quad_ratio_rot"] is not None else float("nan"):+.3f} '
                    f'E_final={scf["e_tot"]:.12f} '
                    f'class={cls["class"]}')

    wall = time.time() - t_start
    report = {
        'schema_version': SAMPLE_SCHEMA,
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'operator': 'descent along loaded LC-208 dense-H soft modes '
                    '(exact exponential rotations + seeded SCF, '
                    'CPU fp64 pyscf 2.14.0)',
        'inputs': inputs,
        'cell': cell,
        'settings': settings,
        'descent_points': plan['descent_points'],
        'amplitudes': amplitudes,
        'directions': directions,
        'thresholds': thresholds,
        'cpu_family_refs': cpu_refs,
        'stay_controls': stay_controls,
        'direction_determination': direction_determination,
        'descent_runs': descent_runs,
        'line_summary': summarize_lines(descent_runs, thresholds),
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
    lines += ['', '## Stay controls', '',
              '| point | E | cycles | converged |', '|---|---|---|---|']
    for p, s in report['stay_controls'].items():
        lines.append(f"| {p} | {s['e_tot']:.12f} | {s['cycles']} | "
                     f"{s['converged']} |")
    lines += ['', '## Direction determination', '',
              '| point | lambda1 | one_col_rel | hdiag_dev | validated | '
              'source | davidson1 |', '|---|---|---|---|---|---|---|']
    for p, dd in report['direction_determination'].items():
        lines.append(
            f"| {p} | {dd['lambda1_dense']:+.6e} | {dd['one_column_rel']:.2e} "
            f"| {dd['hdiag_dev']:.2e} | {dd['validated']} "
            f"| {dd['direction_source']} "
            f"| {dd['davidson_softest3'][0]:+.6e} |")
    lines += ['', '## Descent runs', '',
              '| point | sigma | t | dE_rot | quad_ratio | E_final | '
              'cycles | class |', '|---|---|---|---|---|---|---|---|']
    for r in report['descent_runs']:
        lines.append(
            f"| {r['point']} | {r['sigma']:+d} | {r['t']:.2f} "
            f"| {r['dE_rot_vs_stay']:+.3e} "
            f"| {r['quad_ratio_rot'] if r['quad_ratio_rot'] is not None else 'n/a'} "
            f"| {r['e_final']:.12f} | {r['cycles']} "
            f"| {r['classification']['class']} |")
    lines += ['', '## Line summary', '',
              '| line | tipping_t | terminal |', '|---|---|---|']
    for k, s in report['line_summary'].items():
        lines.append(f"| {k} | {s['tipping_t']} | {s['terminal']} |")
    lines += ['',
              f"Wall {report['operational']['wall_s']:.1f} s, CPU-only, "
              f"zero batch writes.", '']
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='LC-209 saddle-descent basin-connectivity map '
                    '(CPU fp64 pyscf; loads the LC-208 dense-H '
                    'checkpoints)')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md')
    ap.add_argument('--log')
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

    problems = validate_descent_plan(plan)
    if problems:
        for p in problems:
            log(f'PLAN PROBLEM: {p}')
        return 2
    report = run(plan, args.out_json, args.out_md, log)
    log(f'done: {len(report["descent_runs"])} descent runs, '
        f'wall {report["operational"]["wall_s"]:.1f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())

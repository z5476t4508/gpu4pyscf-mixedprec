#!/usr/bin/env python3
"""benchv1_independent_recompute.py -- independent recomputation of the
benchmark-v1 first-evaluation headline figures from RAW per-arm evidence.

Motivation (card 2026-10-30 BENCH-EXEC-01 exit-processing continuity, the
independent_recompute pattern of batches b3-b9): benchmark_v1_stats.py is a
single code path computing every figure the daily comparisons and the
effects-index dataset will cite.  This tool recomputes the same figures from
DIFFERENT files -- the raw per-arm sample records under samples/, which the
stats tool never opens (it reads only the merged cells/{tag}_{role}.json
docs) -- and cross-checks:

  1. pin integrity   -- every sample's batch_id / schedule_sha256 / block /
                        role must match the frozen schedule on disk;
  2. merge fidelity  -- the merged role doc's repeats equal the raw per-arm
                        records block-by-block (catches merge-layer defects
                        and the t_cderi aggregate class repaired at d898e96);
  3. gates           -- per-case all-pairs |dE|/max|dg| vs strict R repeats,
                        convergence, strict-R FP32-free (mirror of the
                        contract, reimplemented here);
  4. statistics      -- per-case medians, O_to_B / B_to_C / O_to_C per-case
                        speedups, cohort geomean (primary), median of
                        speedups, sum-of-medians ratio, actual-SCF-FP32 /
                        gated-fallback counts, and the hierarchical paired
                        bootstrap (seed 20261029) reimplemented;
  5. vs official     -- optional field-by-field agreement against the
                        official stats document (--stats-json); ANY
                        disagreement is reported and fails the run.

--mode s (schedule-s / batch benchv1-20261030-obcr-S) applies the same
pattern to the S-paired auto-strict batch: roles S(auto_strict)/
B(optimized_fp64), the auto+strict certificate re-verified repeat-by-
repeat from the RAW per-arm records, per-case case_ok + B/S medians,
cohort b_to_s geomean/median/total ratio and per-method coverage,
cross-checked field-by-field against the official
benchmark-v1-s-stats-v1 document.

Read-only: no evidence file is ever written or modified.  CPU-only,
stdlib-only, no CUDA context -- safe during a live batch.
--allow-partial labels the run NON-CITABLE for live monitoring; the default
(full coverage) mode is the post-batch cross-check.

Exit codes: 0 = coverage complete (or explicitly allowed), zero
merge-fidelity mismatches, zero pin defects, and (when compared) full
agreement with the official stats document; 1 otherwise.  Scientific gate
FAILURES never change the exit code by themselves -- they are honest
measurements, not recompute defects.
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time

SCHEMA = 'benchv1-independent-recompute-v1'
SCHEMA_S = 'benchv1-independent-recompute-s-v1'
SCHEDULE_SCHEMA = 'upstream-comparison-schedule-v2'
BATCH_ID = 'benchv1-20261030-obcr'
S_SCHEDULE_SCHEMA = 'upstream-comparison-schedule-s-v1'
S_BATCH_ID = 'benchv1-20261030-obcr-S'

# contract gpu4pyscf-benchmark-v1-20261029 (mirrored values, never relaxed)
ENERGY_GATE_HA = 1e-6
GRAD_GATE_HA_BOHR = 3e-5
BOOTSTRAP_DRAWS = 10000
BOOTSTRAP_SEED = 20261029

ROLE_LONG = {'O': 'upstream', 'B': 'optimized_fp64',
             'C': 'optimized_auto', 'R': 'strict_fp64'}
S_ROLE_LONG = {'S': 'auto_strict', 'B': 'optimized_fp64'}
GRAD_HIT_KEYS = ('grad_ip1',)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def scf_fp32_actual(hits):
    """Actual-SCF-FP32 from measured entry counters (grad-only hits never
    qualify -- SP-22 class)."""
    if not hits:
        return False
    return any(v > 0 for k, v in hits.items()
               if k not in GRAD_HIT_KEYS)


# ---------------------------------------------------------------------------
# raw evidence loading + pin verification
# ---------------------------------------------------------------------------

def load_raw_case(cell, evidence_root, sched_sha, defects, gaps,
                  role_long=None, batch_id=None):
    """Read the raw per-arm records for every role of one cell.

    Returns {short: {'repeats': [...], 'hits': {...}} or None when the
    coverage is incomplete}.  Present-but-wrong records append to defects
    and drop that sample (fail-closed per record); simply ABSENT files
    append to gaps -- incomplete coverage is a monitoring state, not a
    defect, and is only allowed under --allow-partial."""
    if role_long is None:
        role_long = ROLE_LONG
    if batch_id is None:
        batch_id = BATCH_ID
    tag = cell['tag']
    n_blocks = len(cell['blocks'])
    out = {}
    for short, role in role_long.items():
        reps, hits = [], None
        for k in range(n_blocks):
            path = os.path.join(evidence_root, 'samples',
                                f'{tag}_{role}_b{k}.json')
            if not os.path.isfile(path):
                continue  # absent file: coverage gap, recorded below
            try:
                with open(path, encoding='utf-8') as f:
                    rec = json.load(f)
            except (OSError, ValueError) as exc:
                defects.append(f'{tag}/{role}/b{k}: unreadable ({exc})')
                continue
            if rec.get('batch_id') != batch_id:
                defects.append(
                    f'{tag}/{role}/b{k}: batch_id '
                    f"{rec.get('batch_id')!r} != {batch_id!r}")
                continue
            if rec.get('schedule_sha256') != sched_sha:
                defects.append(f'{tag}/{role}/b{k}: schedule pin mismatch')
                continue
            if rec.get('role') != role:
                defects.append(
                    f'{tag}/{role}/b{k}: role field '
                    f"{rec.get('role')!r}")
                continue
            if rec.get('block') != k:
                defects.append(
                    f'{tag}/{role}/b{k}: block field {rec.get('block')!r}')
                continue
            res = ((rec.get('row') or {}).get('result')) or {}
            rr = res.get('repeats') or []
            if len(rr) != 1:
                defects.append(
                    f'{tag}/{role}/b{k}: {len(rr)} repeats '
                    '(expected 1 per arm)')
                continue
            reps.append(rr[0])
            if hits is None:
                hits = res.get('fp32_entry_hits')
            elif res.get('fp32_entry_hits') != hits:
                defects.append(
                    f'{tag}/{role}/b{k}: fp32_entry_hits differ between '
                    'arms of the same cell')
        if len(reps) == n_blocks:
            out[short] = {'repeats': reps, 'hits': hits}
        else:
            out[short] = None
            gaps.append(f'{tag}/{role}: {len(reps)}/{n_blocks} samples')
    return out


# ---------------------------------------------------------------------------
# merge fidelity: raw per-arm records vs the merged role documents the
# stats tool consumes
# ---------------------------------------------------------------------------

FIDELITY_FIELDS = ('sample_s', 'e_tot', 't_mol_s', 't_cderi_s',
                   't_scf_s', 't_grad_s', 'scf_cycles', 'converged')
# the S mode additionally block-compares the carried-through auto+strict
# certificate (the merge layer must never alter it)
S_FIDELITY_FIELDS = FIDELITY_FIELDS + ('stability_verification',)


def check_merge_fidelity(cell, raw, evidence_root, mismatches,
                         role_long=None, fields=None,
                         timing_fields=('t_cderi_s',)):
    """Block-by-block comparison of the merged role documents (what the
    stats tool consumes) against the raw per-arm records.  Defaults keep
    the obcr behavior; the S mode passes its role pair, the certificate
    field and the t_scf_s/t_grad_s aggregates the s-route stats read."""
    if role_long is None:
        role_long = ROLE_LONG
    if fields is None:
        fields = FIDELITY_FIELDS
    tag = cell['tag']
    checked = 0
    for short, role in role_long.items():
        entry = raw.get(short)
        path = os.path.join(evidence_root, 'cells', f'{tag}_{role}.json')
        if entry is None or not os.path.isfile(path):
            continue
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        merged = ((((doc.get('rows') or [{}])[0]).get('result'))
                  or {}).get('repeats')
        if not isinstance(merged, list) or len(merged) != len(entry['repeats']):
            n_merged = len(merged) if isinstance(merged, list) \
                else 'non-list'
            mismatches.append(
                f'{tag}/{role}: merged repeats {n_merged} != raw '
                f"{len(entry['repeats'])}")
            continue
        checked += 1
        for i, (m, r) in enumerate(zip(merged, entry['repeats'])):
            for field in fields:
                if m.get(field) != r.get(field):
                    mismatches.append(
                        f'{tag}/{role}/rep{i}: merged {field}='
                        f"{m.get(field)!r} != raw {r.get(field)!r}")
        tim = ((((doc.get('rows') or [{}])[0]).get('result'))
               or {}).get('timings') or {}
        raw_s = [r['sample_s'] for r in entry['repeats']]
        if tim.get('samples_s') != raw_s:
            mismatches.append(
                f'{tag}/{role}: merged timings.samples_s != raw sample_s')
        # the merge layer rounds aggregates to 6 decimals (us); compare
        # each aggregate median within that rounding, exactly elsewhere
        for field in timing_fields:
            vals = [r.get(field) for r in entry['repeats']
                    if r.get(field) is not None]
            if not vals:
                continue
            med = statistics.median(vals)
            if tim.get(field) is None or abs(tim[field] - med) > 1e-6:
                mismatches.append(
                    f'{tag}/{role}: merged timings.{field}='
                    f"{tim.get(field)!r} != raw median {med!r}")
    return checked


# ---------------------------------------------------------------------------
# measurement (independent reimplementation of the contract semantics)
# ---------------------------------------------------------------------------

def measure_case(cell, raw):
    out = {'tag': cell['tag'], 'molecule': cell.get('molecule'),
           'method': cell['xc'], 'basis': cell['basis'],
           'roles': {}, 'gates': {}, 'gate_pass': False}
    ref = raw.get('R')
    ref_reps = ref['repeats'] if ref else None
    for short in 'OBRC':
        entry = raw.get(short)
        if entry is None:
            out['roles'][short] = {'status': 'missing'}
            out['gates'][short] = {'pass': False, 'reason': 'missing'}
            continue
        reps = entry['repeats']
        row = {'status': 'ok',
               'median_sample_s': statistics.median(
                   r['sample_s'] for r in reps),
               'median_t_cderi_s': statistics.median(
                   r['t_cderi_s'] for r in reps),
               'n_repeats': len(reps),
               'all_converged': all(r['converged'] for r in reps),
               'scf_fp32_actual': scf_fp32_actual(entry['hits']),
               'repeats_sample_s': [r['sample_s'] for r in reps]}
        out['roles'][short] = row
        if short == 'R':
            fp32_free = not entry['hits'] or all(
                v == 0 for v in entry['hits'].values())
            ok = bool(row['all_converged'] and fp32_free)
            out['gates'][short] = {
                'pass': ok,
                'reason': None if ok else
                'strict R not converged' if not row['all_converged']
                else 'FP32 entry in strict R'}
            continue
        if ref_reps is None:
            out['gates'][short] = {'pass': False,
                                   'reason': 'no strict R reference'}
            continue
        de = dg = 0.0
        for a in reps:
            for b in ref_reps:
                de = max(de, abs(a['e_tot'] - b['e_tot']))
                ga, gb = a['gradient'], b['gradient']
                if len(ga) == len(gb):
                    dg = max(dg, max(abs(x - y)
                                     for ra, rb in zip(ga, gb)
                                     for x, y in zip(ra, rb)))
        reasons = []
        if not row['all_converged']:
            reasons.append('nonconverged repeat')
        if de > ENERGY_GATE_HA:
            reasons.append(f'max|dE| {de:.3e} > {ENERGY_GATE_HA:g}')
        if dg > GRAD_GATE_HA_BOHR:
            reasons.append(f'max|dg| {dg:.3e} > {GRAD_GATE_HA_BOHR:g}')
        out['gates'][short] = {'pass': not reasons,
                               'reason': '; '.join(reasons) if reasons
                               else None,
                               'max_dE_vs_R': de, 'max_dg_vs_R': dg}
    auto = out['roles'].get('C') or {}
    if auto.get('status') == 'ok':
        out['auto_class'] = ('actual_scf_fp32'
                             if auto['scf_fp32_actual']
                             else 'gated_fallback_fp64')
    out['gate_pass'] = all(g['pass'] for g in out['gates'].values())
    return out


def summarize(cases, num, den):
    if not cases:
        return None
    speedups, num_sum, den_sum = [], 0.0, 0.0
    for c in cases:
        a, b = c['roles'].get(num), c['roles'].get(den)
        if (not a or not b or a.get('status') != 'ok'
                or b.get('status') != 'ok'
                or b['median_sample_s'] <= 0):
            return None
        speedups.append(a['median_sample_s'] / b['median_sample_s'])
        num_sum += a['median_sample_s']
        den_sum += b['median_sample_s']
    return {'n': len(speedups),
            'primary_geomean': geomean(speedups),
            'median_speedup': statistics.median(speedups),
            'total_time_ratio': num_sum / den_sum,
            'per_case_speedups': speedups}


def bootstrap_ci(cases, num, den, draws=None, seed=None):
    """Protocol §5 hierarchical paired bootstrap, reimplemented: molecule
    clusters with replacement (each pick carries all its method cells),
    then paired block resampling within each case."""
    if draws is None:
        draws = BOOTSTRAP_DRAWS
    if seed is None:
        seed = BOOTSTRAP_SEED
    by_mol = {}
    for c in cases:
        by_mol.setdefault(c['molecule'], []).append(c)
    mol_ids = sorted(by_mol)
    if not mol_ids:
        return None
    rng = random.Random(seed)
    draws_out = []
    for _ in range(draws):
        picked = [rng.choice(mol_ids) for _ in mol_ids]
        resampled = [c for m in picked for c in by_mol[m]]
        speedups = []
        for c in resampled:
            n_rep = c['roles'][num]['n_repeats']
            idx = [rng.randrange(n_rep) for _ in range(n_rep)]
            num_med = statistics.median(
                c['roles'][num]['repeats_sample_s'][i] for i in idx)
            den_med = statistics.median(
                c['roles'][den]['repeats_sample_s'][i] for i in idx)
            if den_med <= 0:
                continue
            speedups.append(num_med / den_med)
        if speedups:
            draws_out.append(geomean(speedups))
    if not draws_out:
        return None
    draws_out.sort()
    return (draws_out[int(0.025 * len(draws_out))],
            draws_out[min(len(draws_out) - 1,
                          int(0.975 * len(draws_out)))])


# ---------------------------------------------------------------------------
# S-route measurement (independent reimplementation of the auto-strict
# contract semantics, from RAW per-arm records)
# ---------------------------------------------------------------------------

def measure_s_case(cell, raw):
    """S-paired per-case measurement recomputed from the raw per-arm
    records -- never the merged docs the stats tool reads.  The
    auto+strict certificate is re-verified repeat-by-repeat here: an
    uncertified or nonconverged repeat is a failed case, never a pass.
    b_to_s_ratio < 1 means the certified route COSTS more than plain
    optimized FP64: the honest direction, never inverted."""
    out = {'tag': cell['tag'], 'molecule': cell.get('molecule'),
           'method': cell['xc'], 'basis': cell['basis'], 'roles': {}}
    for short, role in S_ROLE_LONG.items():
        entry = raw.get(short)
        if entry is None:
            out['roles'][short] = {'status': 'missing'}
            continue
        reps = entry['repeats']
        row = {'status': 'ok',
               'median_sample_s': statistics.median(
                   r['sample_s'] for r in reps),
               'median_t_scf_s': statistics.median(
                   r['t_scf_s'] for r in reps),
               'median_t_grad_s': statistics.median(
                   r['t_grad_s'] for r in reps),
               'n_repeats': len(reps),
               'all_converged': all(r['converged'] for r in reps)}
        if role == 'auto_strict':
            certs = [r.get('stability_verification') or {}
                     for r in reps]
            row['n_certified_repeats'] = sum(
                1 for c in certs
                if c.get('policy') == 'auto+strict' and
                c.get('strict_match_verified') is True)
            row['strict_match_all_verified'] = (
                row['n_certified_repeats'] == len(reps))
        out['roles'][short] = row
    s = out['roles'].get('S') or {}
    b = out['roles'].get('B') or {}
    reasons = []
    if s.get('status') != 'ok':
        reasons.append('S arm missing')
    elif not s.get('strict_match_all_verified'):
        reasons.append(f"S certificate incomplete "
                       f"({s.get('n_certified_repeats')}/"
                       f"{s.get('n_repeats')} repeats)")
    if b.get('status') != 'ok':
        reasons.append('B arm missing')
    if s.get('status') == 'ok' and not s.get('all_converged'):
        reasons.append('S nonconverged repeat')
    if b.get('status') == 'ok' and not b.get('all_converged'):
        reasons.append('B nonconverged repeat')
    out['case_ok'] = not reasons
    out['reason'] = '; '.join(reasons) if reasons else None
    if out['case_ok'] and b['median_sample_s'] > 0:
        out['b_to_s_ratio'] = (b['median_sample_s'] /
                               s['median_sample_s'])
    return out


def build_s_report(schedule_path, evidence_root, allow_partial):
    sched_sha = sha256_file(schedule_path)
    with open(schedule_path, encoding='utf-8') as f:
        schedule = json.load(f)
    if schedule.get('schema_version') != S_SCHEDULE_SCHEMA:
        raise SystemExit(
            f'refusing: schedule schema {schedule.get("schema_version")!r} '
            f'!= {S_SCHEDULE_SCHEMA!r} (--mode s expects the s-paired '
            f'schedule)')
    if schedule.get('batch_id') != S_BATCH_ID:
        raise SystemExit(
            f'refusing: schedule batch_id {schedule.get("batch_id")!r} '
            f'!= {S_BATCH_ID!r}')
    defects, gaps, mismatches, measured = [], [], [], []
    fidelity_docs = 0
    for cell in schedule['cells']:
        raw = load_raw_case(cell, evidence_root, sched_sha, defects,
                            gaps, role_long=S_ROLE_LONG,
                            batch_id=S_BATCH_ID)
        fidelity_docs += check_merge_fidelity(
            cell, raw, evidence_root, mismatches, role_long=S_ROLE_LONG,
            fields=S_FIDELITY_FIELDS,
            timing_fields=('t_scf_s', 't_grad_s'))
        measured.append(measure_s_case(cell, raw))
    planned = len(schedule['cells'])
    ok_cases = [c for c in measured if c['case_ok']]
    ratios = [c['b_to_s_ratio'] for c in ok_cases]
    s_sum = sum(c['roles']['S']['median_sample_s'] for c in ok_cases)
    b_sum = sum(c['roles']['B']['median_sample_s'] for c in ok_cases)
    by_method = {}
    for c in measured:
        by_method.setdefault(c['method'], []).append(c)
    method_diag = {}
    for m, cases in sorted(by_method.items()):
        m_ok = [c for c in cases if c['case_ok']]
        m_ratios = [c['b_to_s_ratio'] for c in m_ok]
        method_diag[m] = {
            'planned': len(cases), 'certified_ok': len(m_ok),
            'b_to_s_geomean': geomean(m_ratios),
        }
    full_coverage = all(
        c['roles'][r].get('status') == 'ok'
        for c in measured for r in 'SB')
    return {
        'schema': SCHEMA_S,
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime()),
        'independent': True,
        'mode': 's',
        'citable': bool(full_coverage and not allow_partial
                        and not defects and not mismatches),
        'allow_partial': bool(allow_partial),
        'schedule_path': schedule_path,
        'schedule_sha256': sched_sha,
        'batch_id': S_BATCH_ID,
        'source_commit': schedule.get('source_commit'),
        'counts': {
            'planned': planned,
            'certified_ok': len(ok_cases),
            'failed_or_missing': planned - len(ok_cases),
        },
        'full_coverage': bool(full_coverage),
        'full_cohort_available': len(ok_cases) == planned,
        'b_to_s': {
            'n': len(ratios),
            'primary_geomean': geomean(ratios),
            'median_ratio': (statistics.median(ratios)
                             if ratios else None),
            'total_time_ratio': (b_sum / s_sum) if s_sum > 0 else None,
        },
        'method_coverage': method_diag,
        'merge_fidelity': {'role_docs_checked': fidelity_docs,
                           'mismatches': mismatches},
        'coverage_gaps': gaps,
        'pin_defects': defects,
        'per_case': measured,
    }


# ---------------------------------------------------------------------------
# comparison against the official stats document
# ---------------------------------------------------------------------------

REL_TOL = 1e-12


def _close(a, b):
    if a is None or b is None:
        return a is b
    if a == b:
        return True
    return abs(a - b) <= REL_TOL * max(abs(a), abs(b))


def compare_with_stats(report, stats_path, disagreements):
    with open(stats_path, encoding='utf-8') as f:
        official = json.load(f)
    proto = official.get('protocol') or {}
    if proto.get('schedule_sha256') not in (None, report['schedule_sha256']):
        disagreements.append(
            f"protocol.schedule_sha256 {proto.get('schedule_sha256')}"
            f" != recomputed {report['schedule_sha256']}")
    ob = proto.get('bootstrap') or {}
    if ob.get('draws') not in (None, BOOTSTRAP_DRAWS) or \
            ob.get('seed') not in (None, BOOTSTRAP_SEED):
        disagreements.append(
            f"protocol.bootstrap {ob} != contract "
            f"{{'draws': {BOOTSTRAP_DRAWS}, 'seed': {BOOTSTRAP_SEED}}}")
    for key, mine in report['counts'].items():
        theirs = (official.get('counts') or {}).get(key)
        if theirs != mine:
            disagreements.append(
                f'counts.{key}: official {theirs} != recomputed {mine}')
    for label in ('O_to_B', 'B_to_C', 'O_to_C'):
        mine = report['comparisons'].get(label) or {}
        theirs = (official.get('comparisons') or {}).get(label) or {}
        for field in ('n', 'primary_geomean', 'median_speedup',
                      'total_time_ratio'):
            if not _close(mine.get(field), theirs.get(field)):
                disagreements.append(
                    f'comparisons.{label}.{field}: official '
                    f"{theirs.get(field)!r} != recomputed {mine.get(field)!r}")
        mc, oc = mine.get('ci95_bootstrap'), theirs.get('ci95_bootstrap')
        if (mc is None) != (oc is None) or (
                mc is not None and not (
                    _close(mc[0], oc[0]) and _close(mc[1], oc[1]))):
            disagreements.append(
                f'comparisons.{label}.ci95_bootstrap: official {oc} '
                f'!= recomputed {mc}')
    theirs_cases = {c.get('tag'): c for c in official.get('per_case', [])}
    for c in report['per_case']:
        t = theirs_cases.get(c['tag'])
        if t is None:
            disagreements.append(f"per_case {c['tag']}: absent officially")
            continue
        if t.get('gate_pass') != c['gate_pass']:
            disagreements.append(
                f"per_case {c['tag']}.gate_pass: official "
                f"{t.get('gate_pass')} != recomputed {c['gate_pass']}")
        for short in 'OBRC':
            a, b = c['roles'].get(short), (t.get('roles') or {}).get(short)
            if (a is None) != (b is None):
                disagreements.append(
                    f"per_case {c['tag']}.roles.{short}: presence differs")
                continue
            if a and b and not _close(a.get('median_sample_s'),
                                      b.get('median_sample_s')):
                disagreements.append(
                    f"per_case {c['tag']}.roles.{short}.median_sample_s: "
                    f"official {b.get('median_sample_s')!r} != recomputed "
                    f"{a.get('median_sample_s')!r}")
    return official


def compare_with_stats_s(report, stats_path, disagreements):
    """Field-by-field agreement against the official benchmark-v1-s-stats
    document: protocol pins, counts, b_to_s figures, per-method coverage
    and every per-case case_ok/certificate/median figure."""
    with open(stats_path, encoding='utf-8') as f:
        official = json.load(f)
    proto = official.get('protocol') or {}
    if proto.get('schedule_sha256') not in (None, report['schedule_sha256']):
        disagreements.append(
            f"protocol.schedule_sha256 {proto.get('schedule_sha256')}"
            f" != recomputed {report['schedule_sha256']}")
    if proto.get('source_commit') not in (None, report['source_commit']):
        disagreements.append(
            f"protocol.source_commit {proto.get('source_commit')}"
            f" != recomputed {report['source_commit']}")
    for key, mine in report['counts'].items():
        theirs = (official.get('counts') or {}).get(key)
        if theirs != mine:
            disagreements.append(
                f'counts.{key}: official {theirs} != recomputed {mine}')
    mine_b = report['b_to_s']
    theirs_b = official.get('b_to_s') or {}
    for field in ('n', 'primary_geomean', 'median_ratio',
                  'total_time_ratio'):
        if not _close(mine_b.get(field), theirs_b.get(field)):
            disagreements.append(
                f'b_to_s.{field}: official {theirs_b.get(field)!r} '
                f"!= recomputed {mine_b.get(field)!r}")
    theirs_mc = official.get('method_coverage') or {}
    for m, mine_m in report['method_coverage'].items():
        theirs_m = theirs_mc.get(m)
        if theirs_m is None:
            disagreements.append(f'method_coverage.{m}: absent officially')
            continue
        for field in ('planned', 'certified_ok'):
            if theirs_m.get(field) != mine_m.get(field):
                disagreements.append(
                    f'method_coverage.{m}.{field}: official '
                    f'{theirs_m.get(field)} != recomputed '
                    f'{mine_m.get(field)}')
        if not _close(mine_m.get('b_to_s_geomean'),
                      theirs_m.get('b_to_s_geomean')):
            disagreements.append(
                f"method_coverage.{m}.b_to_s_geomean: official "
                f"{theirs_m.get('b_to_s_geomean')!r} != recomputed "
                f"{mine_m.get('b_to_s_geomean')!r}")
    theirs_cases = {c.get('tag'): c for c in official.get('per_case', [])}
    for c in report['per_case']:
        t = theirs_cases.get(c['tag'])
        if t is None:
            disagreements.append(f"per_case {c['tag']}: absent officially")
            continue
        if t.get('case_ok') != c['case_ok']:
            disagreements.append(
                f"per_case {c['tag']}.case_ok: official "
                f"{t.get('case_ok')} != recomputed {c['case_ok']}")
        if (c.get('reason') or None) != (t.get('reason') or None):
            disagreements.append(
                f"per_case {c['tag']}.reason: official "
                f"{t.get('reason')!r} != recomputed {c.get('reason')!r}")
        if not _close(c.get('b_to_s_ratio'), t.get('b_to_s_ratio')):
            disagreements.append(
                f"per_case {c['tag']}.b_to_s_ratio: official "
                f"{t.get('b_to_s_ratio')!r} != recomputed "
                f"{c.get('b_to_s_ratio')!r}")
        for short in 'SB':
            a = c['roles'].get(short)
            b = (t.get('roles') or {}).get(short)
            if (a is None) != (b is None):
                disagreements.append(
                    f"per_case {c['tag']}.roles.{short}: "
                    f'presence differs')
                continue
            if a is None:
                continue
            for field in ('median_sample_s', 'median_t_scf_s',
                          'median_t_grad_s'):
                if not _close(a.get(field), b.get(field)):
                    disagreements.append(
                        f'per_case {c["tag"]}.roles.{short}.{field}: '
                        f'official {b.get(field)!r} != recomputed '
                        f'{a.get(field)!r}')
            for field in ('status', 'n_repeats', 'all_converged',
                          'n_certified_repeats',
                          'strict_match_all_verified'):
                if a.get(field) != b.get(field):
                    disagreements.append(
                        f'per_case {c["tag"]}.roles.{short}.{field}: '
                        f'official {b.get(field)!r} != recomputed '
                        f'{a.get(field)!r}')
    return official

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_report(schedule_path, evidence_root, allow_partial,
                 mode='obcr'):
    if mode == 's':
        return build_s_report(schedule_path, evidence_root,
                              allow_partial)
    sched_sha = sha256_file(schedule_path)
    with open(schedule_path, encoding='utf-8') as f:
        schedule = json.load(f)
    if schedule.get('schema_version') != SCHEDULE_SCHEMA:
        raise SystemExit(
            f'refusing: schedule schema {schedule.get("schema_version")!r} '
            f'!= {SCHEDULE_SCHEMA!r} (S batches use their own recompute run)')
    if schedule.get('batch_id') != BATCH_ID:
        raise SystemExit(
            f'refusing: schedule batch_id {schedule.get("batch_id")!r} '
            f'!= {BATCH_ID!r}')
    defects, gaps, mismatches, measured = [], [], [], []
    fidelity_docs = 0
    for cell in schedule['cells']:
        raw = load_raw_case(cell, evidence_root, sched_sha, defects, gaps)
        fidelity_docs += check_merge_fidelity(
            cell, raw, evidence_root, mismatches)
        measured.append(measure_case(cell, raw))
    planned = len(schedule['cells'])
    complete = [c for c in measured if c['gate_pass']]
    full_coverage = all(
        c['roles'][s].get('status') == 'ok'
        for c in measured for s in 'OBRC')
    comparisons = {}
    for label, num, den in (('O_to_B', 'O', 'B'),
                            ('B_to_C', 'B', 'C'),
                            ('O_to_C', 'O', 'C')):
        entry = {}
        summ = summarize(complete, num, den)
        if summ is not None:
            entry.update(summ)
            entry.pop('per_case_speedups', None)
            ci = bootstrap_ci(complete, num, den)
            entry['ci95_bootstrap'] = list(ci) if ci else None
        comparisons[label] = entry
    for c in measured:
        for short in 'OBRC':
            c['roles'].get(short, {}).pop('repeats_sample_s', None)
    report = {
        'schema': SCHEMA,
        'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                          time.gmtime()),
        'independent': True,
        'mode': 'obcr',
        'citable': bool(full_coverage and not allow_partial
                        and not defects and not mismatches),
        'allow_partial': bool(allow_partial),
        'schedule_path': schedule_path,
        'schedule_sha256': sched_sha,
        'batch_id': BATCH_ID,
        'source_commit': schedule.get('source_commit'),
        'counts': {
            'planned': planned,
            'gate_pass': len(complete),
            'gate_fail_or_missing': planned - len(complete),
            'actual_scf_fp32_C': sum(
                1 for c in measured
                if c.get('auto_class') == 'actual_scf_fp32'),
            'gated_fallback_C': sum(
                1 for c in measured
                if c.get('auto_class') == 'gated_fallback_fp64'),
        },
        'full_coverage': bool(full_coverage),
        'full_cohort_available': len(complete) == planned,
        'comparisons': comparisons,
        'merge_fidelity': {'role_docs_checked': fidelity_docs,
                           'mismatches': mismatches},
        'coverage_gaps': gaps,
        'pin_defects': defects,
        'per_case': measured,
    }
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='Independent recompute of benchmark-v1 O/B/C/R (or '
                    '--mode s: S-paired auto-strict) figures from raw '
                    'per-arm samples (read-only).')
    ap.add_argument('--schedule', required=True)
    ap.add_argument('--evidence-root', required=True)
    ap.add_argument('--out-json')
    ap.add_argument('--stats-json',
                    help='official benchmark_v1_stats document to '
                         'cross-check field-by-field')
    ap.add_argument('--mode', choices=('obcr', 's'), default='obcr',
                    help='obcr: schedule-v2 O/B/C/R batch (default); '
                         's: schedule-s S-paired auto-strict batch')
    ap.add_argument('--allow-partial', action='store_true',
                    help='run on incomplete coverage for live monitoring; '
                         'output is labeled non-citable')
    args = ap.parse_args(argv)

    report = build_report(args.schedule, args.evidence_root,
                          args.allow_partial, mode=args.mode)
    disagreements = []
    official = None
    if args.stats_json:
        if args.mode == 's':
            official = compare_with_stats_s(report, args.stats_json,
                                            disagreements)
        else:
            official = compare_with_stats(report, args.stats_json,
                                          disagreements)
        report['vs_official_stats'] = {
            'path': args.stats_json,
            'official_generated_at': (
                official.get('generated_at_utc')
                if official else None),
            'disagreements': disagreements,
            'verdict': 'AGREE' if not disagreements else 'DISAGREE',
        }
    ok = ((report['full_coverage'] or args.allow_partial)
          and not report['pin_defects']
          and not report['merge_fidelity']['mismatches']
          and not disagreements)
    report['verdict'] = ('OK' if ok else 'DEFECTS_FOUND')
    if args.out_json:
        tmp = args.out_json + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
            f.write('\n')
        os.replace(tmp, args.out_json)
    counts = report['counts']
    if args.mode == 's':
        print(f"independent recompute (s): planned {counts['planned']} / "
              f"certified_ok {counts['certified_ok']} / "
              f"fail-or-missing {counts['failed_or_missing']}")
        e = report['b_to_s']
        if e.get('primary_geomean') is None:
            print('  b_to_s: unavailable')
        else:
            print(f"  b_to_s: geomean {e['primary_geomean']:.4f} "
                  f"median {e['median_ratio']:.4f} "
                  f"total {e['total_time_ratio']:.4f} (n={e['n']})")
    else:
        print(f"independent recompute: planned {counts['planned']} / "
              f"gate_pass {counts['gate_pass']} / "
              f"fail-or-missing {counts['gate_fail_or_missing']} "
              f"(FP32-subset {counts['actual_scf_fp32_C']}, "
              f"gated-fallback {counts['gated_fallback_C']})")
        for label, e in report['comparisons'].items():
            if e.get('primary_geomean') is None:
                print(f'  {label}: unavailable')
            else:
                print(f"  {label}: geomean {e['primary_geomean']:.4f} "
                      f"[{e['ci95_bootstrap'][0]:.4f}, "
                      f"{e['ci95_bootstrap'][1]:.4f}] "
                      f"median {e['median_speedup']:.4f} "
                      f"total {e['total_time_ratio']:.4f}")
    for g in report['coverage_gaps'][:10]:
        print(f'COVERAGE GAP: {g}')
    for d in report['pin_defects'][:10]:
        print(f'PIN DEFECT: {d}')
    for m in report['merge_fidelity']['mismatches'][:10]:
        print(f'MERGE MISMATCH: {m}')
    for d in disagreements[:10]:
        print(f'STATS DISAGREE: {d}')
    if not ok:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())

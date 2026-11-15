#!/usr/bin/env python3
"""Benchmark-v1 planned-denominator statistics and report (2026-10-30
resume card, sealed package benchmark-v1-20261029).

CPU-only, stdlib-only.  Consumes the frozen v2 schedule, the batch
driver's per-cell comparison documents and the package's public AO
inventory; emits the protocol §5/§6 statistics and the fixed-template
report.

Contract invariants this module enforces (each has a CPU test):

  * PLANNED DENOMINATOR: every schedule cell stays visible forever.
    A missing document, a scientific failure or an OOM keeps the case in
    the denominator with its reason; full-cohort headline and acceptance
    become unavailable, and only an explicitly labeled common-valid
    diagnostic subset may carry statistics.
  * ALL-FOUR-ROLES GATES BEFORE CLASSIFICATION: every loaded cell is
    measured against the strict R reference for ALL of O/B/C before the
    auto lane is classified actual-FP32 vs gated-fallback (the B01
    failure-visibility lesson -- a gated auto row never hides an
    upstream or optimized-fp64 gate failure).
  * PRIMARY METRIC: geometric mean of per-case paired speedups over the
    whole planned cohort; per-case time is the median of the eight
    formal samples.  Legal gated FP64 fallback stays in the accuracy and
    performance denominators; the actual-SCF-FP32 subset is a separate
    diagnostic.  Secondary: median of speedups, sum-of-medians ratio.
  * UNCERTAINTY: 10000-draw hierarchical paired bootstrap, seed 20261029
    -- molecule clusters resampled with replacement (all four methods of
    a selected molecule travel together), paired blocks resampled with
    replacement WITHIN each case (one block index list shared by every
    role of that case), percentile 95% interval.
  * PERFORMANCE CLAIM: only when all planned cases pass the scientific
    gates, complete paired data exist, and the primary B_to_C lower 95%
    bound exceeds 1.  No invented percentage target.
  * S SEPARATE: the S (auto-strict) batch reports its own costs and
    B_to_S pairing; it never borrows C speedups and its coverage is
    whatever was actually measured.
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

SCHEMA = 'benchmark-v1-stats-v1'
ENERGY_GATE_HA = 1e-6
GRAD_GATE_HA_BOHR = 3e-5
BOOTSTRAP_SEED = 20261029
BOOTSTRAP_DRAWS = 10000
AO_STRATA = (('<250', 0, 250), ('250-349', 250, 350), ('>=350', 350,
                                                       float('inf')))
ROLE_SHORT = {'upstream': 'O', 'optimized_fp64': 'B',
              'optimized_auto': 'C', 'strict_fp64': 'R'}
SHORT_ROLE = {v: k for k, v in ROLE_SHORT.items()}
GRAD_HIT_KEYS = ('grad_ip1',)
# schedule routing (kept in sync with upstream_schedule/upstream_comparison
# -- not imported to stay stdlib-only).  The reporter refuses anything
# it was not built for: legacy v1 four-block batches never pass through
# this tool, and an S schedule never renders as an O/B/C/R cohort.
V2_SCHEMA = 'upstream-comparison-schedule-v2'
S_SCHEMA = 'upstream-comparison-schedule-s-v1'
S_ROLES = (('auto_strict', 'S'), ('optimized_fp64', 'B'))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_cell_doc(evidence_root, cell, role):
    '''One role's comparison document for one cell, or None when the
    driver never produced it (missing stays missing -- never guessed).'''
    path = os.path.join(evidence_root, 'cells',
                        f"{cell['tag']}_{role}.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def scf_fp32_actual(row):
    '''Actual-SCF-FP32 classification from MEASURED entry counters: any
    fp32 entry outside the gradient-phase counters says an fp32 kernel
    executed on the SCF path; grad-only hits never qualify (SP-22: the
    plain-auto gradient resolves fp64 by design).'''
    hits = (row.get('result') or {}).get('fp32_entry_hits') or {}
    return any(v > 0 for k, v in hits.items()
               if k not in GRAD_HIT_KEYS)


def ao_of(ao_inventory, molecule, basis):
    for m in ao_inventory['molecules']:
        if m['id'] == molecule:
            return m['nao'].get(basis)
    return None


def ao_stratum(nao):
    if nao is None:
        return 'unknown'
    for label, lo, hi in AO_STRATA:
        if lo <= nao < hi:
            return label
    return 'unknown'


# ---------------------------------------------------------------------------
# per-case measurement against the strict reference
# ---------------------------------------------------------------------------

def measure_case(cell, evidence_root, role_docs):
    '''Per-role status/timing/gate measurement for ONE cell.  Every role
    present is measured against ALL strict-R repeats BEFORE any
    classification; a missing or failed role is recorded as such and
    blocks the case's gate_pass, never silently skipped.'''
    out = {'tag': cell['tag'], 'molecule': cell.get('molecule'),
           'method': cell['xc'], 'basis': cell['basis'],
           'roles': {}, 'gates': {}}
    ref_doc = role_docs.get('strict_fp64')
    ref_reps = (ref_doc['rows'][0]['result']['repeats']
                if ref_doc else None)
    for role, short in ROLE_SHORT.items():
        doc = role_docs.get(role)
        if doc is None:
            out['roles'][short] = {'status': 'missing'}
            out['gates'][short] = {'pass': False, 'reason': 'missing'}
            continue
        res = doc['rows'][0]['result']
        reps = res['repeats']
        entry = {
            'status': 'ok',
            'median_sample_s': statistics.median(
                r['sample_s'] for r in reps),
            'median_t_scf_s': statistics.median(
                r['t_scf_s'] for r in reps),
            'median_t_grad_s': statistics.median(
                r['t_grad_s'] for r in reps),
            'median_t_cderi_s': statistics.median(
                r['t_cderi_s'] for r in reps),
            'n_repeats': len(reps),
            'all_converged': all(r['converged'] for r in reps),
            'scf_fp32_actual': scf_fp32_actual(doc['rows'][0]),
        }
        out['roles'][short] = entry
        if role == 'strict_fp64':
            hits = res.get('fp32_entry_hits') or {}
            fp32_free = all(v == 0 for v in hits.values())
            out['gates'][short] = {
                'pass': bool(entry['all_converged'] and fp32_free),
                'reason': None if entry['all_converged'] and fp32_free
                else 'strict R not converged' if not
                entry['all_converged'] else 'FP32 entry in strict R'}
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
        if not entry['all_converged']:
            reasons.append('nonconverged repeat')
        if de > ENERGY_GATE_HA:
            reasons.append(f'max|dE| {de:.3e} > {ENERGY_GATE_HA:g}')
        if dg > GRAD_GATE_HA_BOHR:
            reasons.append(f'max|dg| {dg:.3e} > {GRAD_GATE_HA_BOHR:g}')
        out['gates'][short] = {
            'pass': not reasons,
            'reason': '; '.join(reasons) if reasons else None,
            'max_dE_vs_R': de, 'max_dg_vs_R': dg}
    # auto classification LAST, after all four roles were measured
    auto = out['roles'].get('C') or {}
    if auto.get('status') == 'ok':
        out['auto_class'] = ('actual_scf_fp32'
                             if auto['scf_fp32_actual']
                             else 'gated_fallback_fp64')
    out['gate_pass'] = all(g['pass'] for g in out['gates'].values())
    return out


def compare_case(measured, num_short, den_short):
    '''Paired speedup for one case: baseline/candidate per-case medians.
    Only defined when both roles are ok; the caller keeps the case in
    the denominator either way.'''
    roles = measured['roles']
    num, den = roles.get(num_short), roles.get(den_short)
    if not num or not den or num.get('status') != 'ok' or \
            den.get('status') != 'ok':
        return None
    if den['median_sample_s'] <= 0:
        return None
    return num['median_sample_s'] / den['median_sample_s']


# ---------------------------------------------------------------------------
# cohort statistics
# ---------------------------------------------------------------------------

def geomean(values):
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summarize_comparison(cases, num_short, den_short):
    '''Primary/secondary statistics for one comparison over the case
    list.  Every case in the list must have both roles ok -- the planned
    denominator decision (all 64 vs labeled subset) is the CALLER's;
    this function never hides a case.  An EMPTY list (a stratum whose
    cases are all missing/failed, or a zero-pass cohort) returns None:
    the statistics are undefined, not zero, and the missing cases stay
    in the caller's planned denominator.'''
    if not cases:
        return None
    speedups, num_sum, den_sum = [], 0.0, 0.0
    for c in cases:
        s = compare_case(c, num_short, den_short)
        if s is None or s <= 0:
            return None
        speedups.append(s)
        num_sum += c['roles'][num_short]['median_sample_s']
        den_sum += c['roles'][den_short]['median_sample_s']
    return {
        'n': len(speedups),
        'primary_geomean': geomean(speedups),
        'median_speedup': statistics.median(speedups),
        'total_time_ratio': (num_sum / den_sum) if den_sum > 0 else None,
        'per_case_speedups': speedups,
    }


def bootstrap_ci(cases, num_short, den_short, molecules,
                 draws=None, seed=None):
    '''Hierarchical paired bootstrap (protocol §5): resample molecule
    clusters with replacement (a selected molecule carries ALL its
    method cells), then resample the paired blocks with replacement
    within each case -- one block index list per case shared by every
    role, preserving the pairing.  draws/seed default to the module
    constants RESOLVED AT CALL TIME (never bound as def-time defaults;
    a pinned default argument would silently ignore contract changes).
    Returns (lo, hi) percentile 95% of the primary geomean, or None if
    the input is not complete.'''
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
    stats_draws = []
    for _ in range(draws):
        picked = [rng.choice(mol_ids) for _ in mol_ids]
        resampled = [c for m in picked for c in by_mol[m]]
        speedups = []
        for c in resampled:
            n_rep = c['roles'][num_short]['n_repeats']
            idx = [rng.randrange(n_rep) for _ in range(n_rep)]
            num_med = statistics.median(
                c['roles'][num_short]['repeats_sample_s'][i]
                for i in idx)
            den_med = statistics.median(
                c['roles'][den_short]['repeats_sample_s'][i]
                for i in idx)
            if den_med <= 0:
                continue
            speedups.append(num_med / den_med)
        if speedups:
            stats_draws.append(geomean(speedups))
    if not stats_draws:
        return None
    stats_draws.sort()
    lo = stats_draws[int(0.025 * len(stats_draws))]
    hi = stats_draws[min(len(stats_draws) - 1,
                         int(0.975 * len(stats_draws)))]
    return lo, hi


def attach_repeats_samples(measured, role_docs):
    '''Bootstrap needs the raw per-block sample times per role; keep them
    off the per-case report rows but beside them for the resampler.'''
    for short, role in SHORT_ROLE.items():
        doc = role_docs.get(role)
        if doc is not None and short in measured['roles'] and \
                measured['roles'][short].get('status') == 'ok':
            reps = doc['rows'][0]['result']['repeats']
            measured['roles'][short]['repeats_sample_s'] = [
                r['sample_s'] for r in reps]


def build_report(schedule, evidence_root, cohort, ao_inventory,
                 generated_at=None):
    '''The full planned-denominator report for one O/B/C/R v2 batch.'''
    cells = schedule['cells']
    measured_cases = []
    for cell in cells:
        role_docs = {role: load_cell_doc(evidence_root, cell, role)
                     for role in ROLE_SHORT}
        measured = measure_case(cell, evidence_root, role_docs)
        attach_repeats_samples(measured, role_docs)
        measured['nao'] = ao_of(ao_inventory, cell.get('molecule'),
                                cell['basis'])
        measured['ao_stratum'] = ao_stratum(measured['nao'])
        measured_cases.append(measured)

    planned = len(cells)
    complete = [c for c in measured_cases if c['gate_pass']]
    incomplete = [c for c in measured_cases if not c['gate_pass']]
    full_cohort = (len(complete) == planned)

    comparisons = {}
    basis_for_claim = complete if full_cohort else None
    for label, (num, den) in {'O_to_B': ('O', 'B'),
                              'B_to_C': ('B', 'C'),
                              'O_to_C': ('O', 'C')}.items():
        entry = {}
        summ = summarize_comparison(complete, num, den)
        if summ is not None:
            entry.update(summ)
            entry.pop('per_case_speedups', None)
            ci = bootstrap_ci(complete, num, den,
                              sorted({c['molecule'] for c in complete}))
            entry['ci95_bootstrap'] = (list(ci)
                                       if ci else None)
        comparisons[label] = entry

    claim = None
    if full_cohort:
        btc = comparisons.get('B_to_C') or {}
        ci = btc.get('ci95_bootstrap')
        claim = {
            'full_cohort': True,
            'all_planned_pass': True,
            'paired_data_complete': bool(summarize_comparison(
                complete, 'B', 'C')),
            'lower_ci95': ci[0] if ci else None,
            'mixed_precision_speedup_established': bool(
                ci and ci[0] > 1.0),
        }
    report = {
        'schema': SCHEMA,
        'generated_at_utc': generated_at or
        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'protocol': {
            'dataset': 'gpu4pyscf-benchmark-v1-20261029',
            'schedule_sha256': schedule.get('_sha256'),
            'source_commit': schedule.get('source_commit'),
            'timing_protocol': schedule.get('timing_protocol'),
            'blocks_per_role_per_case': len(cells[0]['blocks']),
            'energy_gate_Ha': ENERGY_GATE_HA,
            'gradient_gate_Eh_Bohr': GRAD_GATE_HA_BOHR,
            'bootstrap': {'draws': BOOTSTRAP_DRAWS,
                          'seed': BOOTSTRAP_SEED},
        },
        'counts': {
            'planned': planned,
            'gate_pass': len(complete),
            'gate_fail_or_missing': len(incomplete),
            'actual_scf_fp32_C': sum(
                1 for c in measured_cases
                if c.get('auto_class') == 'actual_scf_fp32'),
            'gated_fallback_C': sum(
                1 for c in measured_cases
                if c.get('auto_class') == 'gated_fallback_fp64'),
        },
        'full_cohort_available': full_cohort,
        'comparisons': comparisons,
        'performance_claim': claim,
        'per_case': measured_cases,
    }
    if not full_cohort:
        report['diagnostic_common_valid_subset'] = {
            'note': ('合格子集，非全集验收 -- full-cohort headline '
                     'unavailable; every failure/missing case stays in '
                     'the planned denominator above'),
            'n': len(complete),
        }
    # frozen diagnostic strata (protocol §5: diagnostics only, never a
    # replacement for the whole-cohort primary metric)
    strata = {'method': {}, 'ao': {}}
    for key, field in (('method', 'method'), ('ao', 'ao_stratum')):
        groups = {}
        for c in measured_cases:
            groups.setdefault(c[field], []).append(c)
        for group, cases in sorted(groups.items()):
            entry = {'planned': len(cases),
                     'gate_pass': sum(1 for c in cases if c['gate_pass'])}
            for label, num, den in (('O_to_B', 'O', 'B'),
                                    ('B_to_C', 'B', 'C'),
                                    ('O_to_C', 'O', 'C')):
                ok_cases = [c for c in cases if c['gate_pass']]
                summ = summarize_comparison(ok_cases, num, den)
                entry[label] = (round(summ['primary_geomean'], 4)
                                if summ else None)
            strata[key][group] = entry
    report['strata'] = strata
    return report


# ---------------------------------------------------------------------------
# S route (auto-strict, paired against B in its own batch)
# ---------------------------------------------------------------------------

def measure_s_case(cell, evidence_root, role_docs):
    '''Per-case S/B measurement.  S is the auto-strict route: its
    sample_s already contains its own verification/matching/fallback
    work, and its accuracy evidence is the per-repeat auto+strict
    certificate -- re-checked HERE independently of the merge layer
    (an uncertified or nonconverged repeat is a failed case, never a
    pass).  b_to_s_ratio < 1 means the certified route COSTS more than
    plain optimized FP64: the honest direction, never inverted.'''
    out = {'tag': cell['tag'], 'molecule': cell.get('molecule'),
           'method': cell['xc'], 'basis': cell['basis'], 'roles': {}}
    for role, short in S_ROLES:
        doc = role_docs.get(role)
        if doc is None:
            out['roles'][short] = {'status': 'missing'}
            continue
        res = doc['rows'][0]['result']
        reps = res['repeats']
        entry = {
            'status': 'ok',
            'median_sample_s': statistics.median(
                r['sample_s'] for r in reps),
            'median_t_scf_s': statistics.median(
                r['t_scf_s'] for r in reps),
            'median_t_grad_s': statistics.median(
                r['t_grad_s'] for r in reps),
            'n_repeats': len(reps),
            'all_converged': all(r['converged'] for r in reps),
        }
        if role == 'auto_strict':
            certs = [r.get('stability_verification') or {}
                     for r in reps]
            entry['n_certified_repeats'] = sum(
                1 for c in certs
                if c.get('policy') == 'auto+strict' and
                c.get('strict_match_verified') is True)
            entry['strict_match_all_verified'] = (
                entry['n_certified_repeats'] == len(reps))
        out['roles'][short] = entry
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


def build_s_report(schedule, evidence_root, cohort, ao_inventory,
                   generated_at=None):
    '''The S-route report for one s-paired batch: per-case S/B costs with
    S charging its own certification, whole-planned-cohort visibility
    (missing/uncertified cases stay with reasons), B_to_S pairing over
    the certified subset only, per-method coverage diagnostics.  S never
    borrows C speedups and no performance claim is derived here.'''
    if schedule.get('schema_version') != S_SCHEMA:
        raise ValueError(
            f's-route report requires schema {S_SCHEMA!r}, got '
            f'{schedule.get("schema_version")!r}')
    cells = schedule['cells']
    measured_cases = []
    for cell in cells:
        role_docs = {role: load_cell_doc(evidence_root, cell, role)
                     for role, _ in S_ROLES}
        measured = measure_s_case(cell, evidence_root, role_docs)
        measured['nao'] = ao_of(ao_inventory, cell.get('molecule'),
                                cell['basis'])
        measured['ao_stratum'] = ao_stratum(measured['nao'])
        measured_cases.append(measured)

    planned = len(cells)
    ok_cases = [c for c in measured_cases if c['case_ok']]
    ratios = [c['b_to_s_ratio'] for c in ok_cases]
    s_sum = sum(c['roles']['S']['median_sample_s'] for c in ok_cases)
    b_sum = sum(c['roles']['B']['median_sample_s'] for c in ok_cases)
    by_method = {}
    for c in measured_cases:
        by_method.setdefault(c['method'], []).append(c)
    method_diag = {}
    for m, cases in sorted(by_method.items()):
        m_ok = [c for c in cases if c['case_ok']]
        m_ratios = [c['b_to_s_ratio'] for c in m_ok]
        method_diag[m] = {
            'planned': len(cases), 'certified_ok': len(m_ok),
            'b_to_s_geomean': geomean(m_ratios),
        }
    return {
        'schema': 'benchmark-v1-s-stats-v1',
        'generated_at_utc': generated_at or
        time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'protocol': {
            'dataset': 'gpu4pyscf-benchmark-v1-20261029',
            'schedule_sha256': schedule.get('_sha256'),
            'source_commit': schedule.get('source_commit'),
            'timing_protocol': schedule.get('timing_protocol'),
            'blocks_per_role_per_case': len(cells[0]['blocks']),
        },
        'counts': {'planned': planned, 'certified_ok': len(ok_cases),
                   'failed_or_missing': planned - len(ok_cases)},
        'b_to_s': {
            'note': ('S 成本含其自身验证/匹配/回退开销，绝不借用 C 加速；'
                     '比值<1 表示认证路线比纯优化 FP64 更贵（诚实方向）'),
            'n': len(ratios),
            'primary_geomean': geomean(ratios),
            'median_ratio': (statistics.median(ratios)
                             if ratios else None),
            'total_time_ratio': (b_sum / s_sum) if s_sum > 0 else None,
        },
        'method_coverage': method_diag,
        'per_case': measured_cases,
    }


def render_s_md(report):
    '''Fixed-template markdown for the S-route report.'''
    lines = []
    p = report['protocol']
    c = report['counts']
    lines.append('# Benchmark v1 S 路线（auto-strict）报告（固定模板）')
    lines.append('')
    lines.append(
        f"- 协议/数据集版本：{p['dataset']}；运行时间 "
        f"{report['generated_at_utc']}；schedule sha256 "
        f"{(p['schedule_sha256'] or '')[:16]}…；源码 "
        f"{p['source_commit'][:12]}；每角色每项 "
        f"{p['blocks_per_role_per_case']} 个正式 block")
    lines.append(
        f"- 计划 {c['planned']} 项；证书齐全/完成 {c['certified_ok']}；"
        f"失败或缺失 {c['failed_or_missing']}")
    b = report['b_to_s']
    gm = b['primary_geomean']
    gm_txt = f'{gm:.4f}×' if gm is not None else '不可用'
    md_txt = b['median_ratio']
    md_txt = f'{md_txt:.4f}×' if md_txt is not None else '不可用'
    tt = b['total_time_ratio']
    tt_txt = f'{tt:.4f}×' if tt is not None else '不可用'
    lines.append(
        f"- B→S 比值（合格子集 n={b['n']}）：几何均值 {gm_txt}；"
        f"中位 {md_txt}；总耗时比 {tt_txt}")
    lines.append(f"- {b['note']}")
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# markdown rendering (protocol §6 fixed template)
# ---------------------------------------------------------------------------

def render_md(report, s_report=None):
    lines = []
    p = report['protocol']
    lines.append('# Benchmark v1 对比报告（固定模板）')
    lines.append('')
    lines.append(
        f"- 协议/数据集版本：{p['dataset']}；运行时间 "
        f"{report['generated_at_utc']}；schedule sha256 "
        f"{(p['schedule_sha256'] or '')[:16]}…；源码 {p['source_commit'][:12]}；"
        f"每角色每项 {p['blocks_per_role_per_case']} 个正式 block")
    c = report['counts']
    lines.append(
        f"- 计划 {c['planned']} 项；完成/过门 {c['gate_pass']}；失败或缺失 "
        f"{c['gate_fail_or_missing']}；实际 SCF FP32 {c['actual_scf_fp32_C']}"
        f"；门控回退 {c['gated_fallback_C']}")
    if not report['full_cohort_available']:
        lines.append('- **全集主成绩不可用**：存在失败/缺失项，全部保留在分母；'
                     '下方统计仅为合格子集诊断，非全集验收。')
    lines.append('')
    for label, name in (('O_to_B', 'O→B（官方→优化FP64-SCF栈）'),
                        ('B_to_C', 'B→C（优化栈 auto 混合精度）'),
                        ('O_to_C', 'O→C（整体方案）')):
        e = report['comparisons'].get(label) or {}
        ci = e.get('ci95_bootstrap')
        ci_txt = (f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else '不可用')
        gm = e.get('primary_geomean')
        gm_txt = f'{gm:.4f}×' if gm is not None else '不可用'
        md_txt = e.get('median_speedup')
        md_txt = f'{md_txt:.4f}×' if md_txt is not None else '不可用'
        tt = e.get('total_time_ratio')
        tt_txt = f'{tt:.4f}×' if tt is not None else '不可用'
        lines.append(
            f'- {name}：几何均值 {gm_txt}；95% 区间 {ci_txt}；'
            f'中位加速 {md_txt}；总耗时比 {tt_txt}（n={e.get("n")}）')
    claim = report.get('performance_claim')
    if claim:
        lo = claim['lower_ci95']
        lo_txt = f'{lo:.4f}' if lo is not None else '不可用'
        lines.append(
            f'- 性能结论：B→C 95% 下界 {lo_txt}'
            f'{" > 1：混合精度整体加速成立" if claim[
                "mixed_precision_speedup_established"] else "：未满足成立条件"}')
    else:
        lines.append('- 性能结论：不可裁定（存在失败/缺失项）')
    if s_report:
        lines.append('')
        lines.append('- S（auto-strict 验证路线）单列：见 s_route 节；'
                     '其成本为实测验证/匹配/回退开销，不借用 C 加速。')
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        prog='benchmark_v1_stats',
        description='Planned-denominator statistics for the benchmark '
                    'v1 O/B/C/R batch (CPU-only).')
    p.add_argument('--schedule', required=True)
    p.add_argument('--evidence-root', required=True)
    p.add_argument('--ao-inventory', default=os.path.join(
        here, 'benchv1_cohort', 'ao-inventory.json'))
    p.add_argument('--cohort-json', default=os.path.join(
        here, 'benchv1_cohort', 'cohort.json'))
    p.add_argument('--out-json', required=True)
    p.add_argument('--out-md', default=None)
    p.add_argument('--mode', choices=('obcr', 's'), default='obcr',
                   help='obcr: the 64-case four-role cohort (schedule '
                        'schema v2 only); s: the s-paired auto-strict '
                        'batch (schedule schema s only).  Legacy v1 '
                        'four-block schedules are refused -- they keep '
                        'their own historical reporting path.')
    args = p.parse_args(argv)

    with open(args.schedule, encoding='utf-8') as f:
        schedule = json.load(f)
    schedule['_sha256'] = sha256_file(args.schedule)
    schema = schedule.get('schema_version')
    if args.mode == 'obcr' and schema != V2_SCHEMA:
        raise SystemExit(
            f'--mode obcr requires schedule schema {V2_SCHEMA!r}, '
            f'got {schema!r}')
    if args.mode == 's' and schema != S_SCHEMA:
        raise SystemExit(
            f'--mode s requires schedule schema {S_SCHEMA!r}, '
            f'got {schema!r}')
    with open(args.cohort_json, encoding='utf-8') as f:
        cohort = json.load(f)
    with open(args.ao_inventory, encoding='utf-8') as f:
        ao_inventory = json.load(f)

    if args.mode == 's':
        report = build_s_report(schedule, args.evidence_root, cohort,
                                ao_inventory)
        md = render_s_md(report)
    else:
        report = build_report(schedule, args.evidence_root, cohort,
                              ao_inventory)
        md = render_md(report)
    tmp = args.out_json + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    os.replace(tmp, args.out_json)
    print(md)
    if args.out_md:
        with open(args.out_md, 'w', encoding='utf-8') as f:
            f.write(md)
    return 0


if __name__ == '__main__':
    sys.exit(main())

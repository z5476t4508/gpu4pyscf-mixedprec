"""CPU-only tests for benchv1_independent_recompute (BENCH-EXEC-01
independent-recompute harness).  stdlib only; no CUDA, no GPU, no network;
all fixtures synthetic."""

import json
import math
import os
import random
import statistics
import sys
import tempfile
import unittest

REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(
    REPO, 'mixedprec', 'benchmark', 'rtx4090'))

import benchv1_independent_recompute as ir  # noqa: E402

ROLE_BASE_S = {'O': 4.0, 'B': 2.0, 'C': 1.9, 'R': 3.0}
GRAD = [[0.01, -0.02, 0.03], [0.0, 0.01, 0.0]]
ZERO_HITS = {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0}


def _repeat(sample_s, e, converged=True):
    return {'index': 0, 'e_tot': e, 'converged': converged,
            'scf_cycles': 9, 'gradient': GRAD, 'gradient_norm': 0.01,
            'gradient_mode': 'fp64', 'grad_ip1_hits': 0,
            'precision_switch': {'switched': False},
            'sample_s': sample_s, 't_mol_s': 0.01, 't_cderi_s': 0.2,
            't_scf_s': sample_s * 0.6, 't_grad_s': sample_s * 0.3}


def _write_cohort(root, cells_spec, sched_sha, hits_by=None,
                  mutate_sample=None, mutate_merged=None,
                  drop=None, e_shift=None):
    """cells_spec: [(tag, molecule, method)] with 8 blocks per role."""
    sdir = os.path.join(root, 'samples')
    cdir = os.path.join(root, 'cells')
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    for tag, _mol, _meth in cells_spec:
        for short, role in ir.ROLE_LONG.items():
            reps = []
            for k in range(8):
                s = ROLE_BASE_S[short] + 0.1 * k
                e = -100.0
                rec_hits = dict(ZERO_HITS)
                if hits_by and (tag, short) in hits_by:
                    rec_hits.update(hits_by[(tag, short)])
                if e_shift and (tag, short, k) in e_shift:
                    e += e_shift[(tag, short, k)]
                rep = _repeat(round(s, 6), e)
                if mutate_sample and (tag, short, k) in mutate_sample:
                    mutate_sample[(tag, short, k)](rep)
                rec = {'batch_id': ir.BATCH_ID,
                       'schedule_sha256': sched_sha, 'role': role,
                       'block': k,
                       'row': {'result': {'repeats': [rep],
                                          'fp32_entry_hits': rec_hits}}}
                path = os.path.join(sdir, f'{tag}_{role}_b{k}.json')
                if drop and (tag, short, k) in drop:
                    continue
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(rec, f)
                reps.append(rep)
            merged_reps = [json.loads(json.dumps(r)) for r in reps]
            tim = {'samples_s': [r['sample_s'] for r in reps],
                   't_cderi_s': statistics.median(
                       r['t_cderi_s'] for r in reps)}
            doc = {'rows': [{'result': {'repeats': merged_reps,
                                        'timings': tim}}]}
            if mutate_merged and (tag, short) in mutate_merged:
                mutate_merged[(tag, short)](doc)
            with open(os.path.join(cdir, f'{tag}_{role}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(doc, f)


def _write_schedule(root, cells_spec, n_cells_planned=None):
    spec = list(cells_spec)
    if n_cells_planned is not None and n_cells_planned > len(spec):
        spec = spec + [('phantom%d' % i, 'phantom_mol%d' % i, 'b3lyp')
                       for i in range(n_cells_planned - len(spec))]
    schedule = {
        'schema_version': ir.SCHEDULE_SCHEMA, 'seed': 20261029,
        'batch_id': ir.BATCH_ID, 'source_commit': 'deadbee',
        'source_root': '/r', 'timing_protocol': 'balanced-blocks-v2',
        'settings': {},
        'cells': [{'index': i, 'tag': tag, 'molecule': mol,
                   'basis': 'def2-svp', 'xc': meth, 'charge': 0,
                   'spin': 0, 'geom': '/g', 'geometry_sha256': '0' * 64,
                   'blocks': [list('ORCB')] * 8}
                  for i, (tag, mol, meth) in enumerate(spec)]}
    path = os.path.join(root, 'schedule.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(schedule, f)
    return path


def _cohort(tmp, cells_spec=(('t0', 'molA', 'b3lyp'),
                             ('t1', 'molB', 'm062x')), **kw):
    sched_path = _write_schedule(tmp, cells_spec)
    sha = ir.sha256_file(sched_path)
    _write_cohort(tmp, cells_spec, sha, **kw)
    return sched_path


def _run(sched, root, out, **flags):
    argv = ['--schedule', sched, '--evidence-root', root,
            '--out-json', out]
    for k, v in flags.items():
        argv += ['--' + k.replace('_', '-')] + \
            ([] if v is True else [v])
    return ir.main(argv)


class TestFullCohort(unittest.TestCase):

    def test_matches_hand_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            out = os.path.join(tmp, 'recompute.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertTrue(rep['citable'])
            self.assertTrue(rep['full_coverage'])
            self.assertEqual(rep['counts'], {
                'planned': 2, 'gate_pass': 2, 'gate_fail_or_missing': 0,
                'actual_scf_fp32_C': 0, 'gated_fallback_C': 2})
            med = {r: statistics.median(
                ROLE_BASE_S[r] + 0.1 * k for k in range(8))
                for r in 'OBRC'}
            for c in rep['per_case']:
                for short in 'OBRC':
                    self.assertAlmostEqual(
                        c['roles'][short]['median_sample_s'],
                        med[short], places=12)
            hand = {}
            for label, num, den in (('O_to_B', 'O', 'B'),
                                    ('B_to_C', 'B', 'C'),
                                    ('O_to_C', 'O', 'C')):
                sp = [med[num] / med[den]] * 2
                hand[label] = {
                    'n': 2,
                    'primary_geomean': math.exp(
                        sum(math.log(v) for v in sp) / 2),
                    'median_speedup': statistics.median(sp),
                    'total_time_ratio': med[num] * 2 / (med[den] * 2)}
                for f, v in hand[label].items():
                    self.assertAlmostEqual(
                        rep['comparisons'][label][f], v, places=12,
                        msg=f'{label}.{f}')

    def test_bootstrap_matches_inline_mirror(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            out1 = os.path.join(tmp, 'a.json')
            out2 = os.path.join(tmp, 'b.json')
            self.assertEqual(_run(sched, tmp, out1), 0)
            self.assertEqual(_run(sched, tmp, out2), 0)
            r1 = json.load(open(out1))
            r2 = json.load(open(out2))
            for label in ('O_to_B', 'B_to_C', 'O_to_C'):
                self.assertEqual(
                    r1['comparisons'][label]['ci95_bootstrap'],
                    r2['comparisons'][label]['ci95_bootstrap'],
                    'bootstrap must be deterministic')

    def test_merge_fidelity_counts_role_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            # 2 cells x 4 roles compared, zero mismatches
            self.assertEqual(
                rep['merge_fidelity']['role_docs_checked'], 8)
            self.assertEqual(rep['merge_fidelity']['mismatches'], [])


class TestGatesAndDenominator(unittest.TestCase):

    def test_precision_failure_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(
                tmp, e_shift={('t0', 'C', 3): 1e-5})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['counts']['gate_pass'], 1)
            self.assertEqual(
                rep['counts']['gate_fail_or_missing'], 1)
            case = next(c for c in rep['per_case'] if c['tag'] == 't0')
            self.assertFalse(case['gate_pass'])
            self.assertIn('max|dE|', case['gates']['C']['reason'])
            # geomean over the one remaining case, never over a
            # hand-picked replacement
            self.assertEqual(rep['comparisons']['O_to_B']['n'], 1)
            # scientific failure is NOT a recompute defect
            self.assertEqual(rep['verdict'], 'OK')

    def test_nonconverged_repeat_fails_gate(self):
        conv = {('t0', 'B', 5)}

        def mut(rep):
            rep['converged'] = False
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, mutate_sample={
                ('t0', 'B', 5): mut})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            case = next(c for c in rep['per_case'] if c['tag'] == 't0')
            self.assertIn('nonconverged repeat',
                          case['gates']['B']['reason'])

    def test_strict_fp32_entry_fails_R_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(
                tmp, hits_by={('t0', 'R'): {'df_jk': 2}})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            case = next(c for c in rep['per_case'] if c['tag'] == 't0')
            self.assertEqual(case['gates']['R']['reason'],
                             'FP32 entry in strict R')
            self.assertFalse(case['gate_pass'])

    def test_scf_fp32_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(
                tmp, hits_by={('t0', 'C'): {'df_jk': 1},
                              ('t1', 'C'): {'grad_ip1': 3}})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['counts']['actual_scf_fp32_C'], 1)
            self.assertEqual(rep['counts']['gated_fallback_C'], 1)


class TestFailClosed(unittest.TestCase):

    def test_missing_arm_refused_without_allow_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, drop={('t1', 'C', 7)})
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertFalse(rep['full_coverage'])
            self.assertFalse(rep['citable'])

    def test_allow_partial_runs_but_is_non_citable(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, drop={('t1', 'C', 7)})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(
                _run(sched, tmp, out, allow_partial=True), 0)
            rep = json.load(open(out))
            self.assertTrue(rep['allow_partial'])
            self.assertFalse(rep['citable'])
            case = next(c for c in rep['per_case'] if c['tag'] == 't1')
            self.assertEqual(case['roles']['C']['status'], 'missing')

    def test_schedule_pin_defect_fails_even_with_allow_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            victim = os.path.join(
                tmp, 'samples', 't0_optimized_auto_b4.json')
            rec = json.load(open(victim))
            rec['schedule_sha256'] = 'f' * 64
            json.dump(rec, open(victim, 'w'))
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(
                _run(sched, tmp, out, allow_partial=True), 0)
            rep = json.load(open(out))
            self.assertTrue(any('schedule pin mismatch' in d
                                for d in rep['pin_defects']))

    def test_block_pin_defect_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            victim = os.path.join(
                tmp, 'samples', 't1_strict_fp64_b2.json')
            rec = json.load(open(victim))
            rec['block'] = 6
            json.dump(rec, open(victim, 'w'))
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertTrue(any('block field' in d
                                for d in rep['pin_defects']))

    def test_merge_fidelity_mismatch_detected(self):
        def mut(doc):
            doc['rows'][0]['result']['repeats'][2]['sample_s'] = 999.0
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, mutate_merged={('t0', 'B'): mut})
            out = os.path.join(tmp, 'r.json')
            rc = _run(sched, tmp, out)
            self.assertNotEqual(rc, 0)
            rep = json.load(open(out))
            self.assertTrue(any('sample_s' in m for m in
                                rep['merge_fidelity']['mismatches']))

    def test_merged_cderi_aggregate_mismatch_detected(self):
        # the d898e96 class: merged aggregate must equal the raw median
        def mut(doc):
            doc['rows'][0]['result']['timings']['t_cderi_s'] = 0.0
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, mutate_merged={('t0', 'O'): mut})
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertTrue(any('t_cderi_s' in m for m in
                                rep['merge_fidelity']['mismatches']))

    def test_merged_cderi_six_decimal_rounding_not_flagged(self):
        # live-data class 2026-10-29T20:4xZ: the merge layer rounds the
        # aggregate to 6 decimals (raw median 0.1788675 -> 0.178868);
        # that is serialization rounding, not a fidelity defect
        def mut(doc):
            doc['rows'][0]['result']['timings']['t_cderi_s'] = 0.2000005
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp, mutate_merged={('t0', 'O'): mut})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['merge_fidelity']['mismatches'], [])

    def test_foreign_schedule_schema_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _write_schedule(tmp, [('t0', 'm', 'b3lyp')])
            d = json.load(open(sched))
            d['schema_version'] = 'upstream-comparison-schedule-s-v1'
            d['batch_id'] = 'benchv1-20261030-obcr-S'
            json.dump(d, open(sched, 'w'))
            with self.assertRaises(SystemExit):
                ir.build_report(sched, tmp, False)


class TestVsOfficialStats(unittest.TestCase):

    def _official_from_report(self, rep, path):
        official = {
            'schema': 'benchmark-v1-stats-v1',
            'generated_at_utc': '2026-10-30T00:00:00Z',
            'protocol': {
                'schedule_sha256': rep['schedule_sha256'],
                'bootstrap': {'draws': ir.BOOTSTRAP_DRAWS,
                              'seed': ir.BOOTSTRAP_SEED}},
            'counts': dict(rep['counts']),
            'comparisons': {k: {f: v for f, v in e.items()}
                            for k, e in rep['comparisons'].items()},
            'per_case': [{'tag': c['tag'], 'gate_pass': c['gate_pass'],
                          'roles': {s: {'median_sample_s':
                                        c['roles'][s]['median_sample_s']}
                                    for s in 'OBRC'
                                    if c['roles'][s]['status'] == 'ok'}}
                          for c in rep['per_case']]}
        json.dump(official, open(path, 'w'))

    def test_agree(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            recompute = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, recompute), 0)
            rep = json.load(open(recompute))
            stats = os.path.join(tmp, 'stats.json')
            self._official_from_report(rep, stats)
            out = os.path.join(tmp, 'r2.json')
            self.assertEqual(
                _run(sched, tmp, out, stats_json=stats), 0)
            rep2 = json.load(open(out))
            self.assertEqual(
                rep2['vs_official_stats']['verdict'], 'AGREE')
            self.assertEqual(
                rep2['vs_official_stats']['disagreements'], [])

    def test_disagree_on_geomean(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            recompute = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, recompute), 0)
            rep = json.load(open(recompute))
            stats = os.path.join(tmp, 'stats.json')
            self._official_from_report(rep, stats)
            official = json.load(open(stats))
            official['comparisons']['B_to_C'][
                'primary_geomean'] = 1.5
            json.dump(official, open(stats, 'w'))
            out = os.path.join(tmp, 'r2.json')
            self.assertNotEqual(
                _run(sched, tmp, out, stats_json=stats), 0)
            rep2 = json.load(open(out))
            self.assertEqual(
                rep2['vs_official_stats']['verdict'], 'DISAGREE')
            self.assertTrue(any(
                'B_to_C.primary_geomean' in d for d in
                rep2['vs_official_stats']['disagreements']))

    def test_disagree_on_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            recompute = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, recompute), 0)
            rep = json.load(open(recompute))
            stats = os.path.join(tmp, 'stats.json')
            self._official_from_report(rep, stats)
            official = json.load(open(stats))
            official['counts']['gate_pass'] = 0
            json.dump(official, open(stats, 'w'))
            out = os.path.join(tmp, 'r2.json')
            self.assertNotEqual(
                _run(sched, tmp, out, stats_json=stats), 0)
            rep2 = json.load(open(out))
            self.assertTrue(any('counts.gate_pass' in d for d in
                                rep2['vs_official_stats']
                                ['disagreements']))

    def test_disagree_on_ci(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _cohort(tmp)
            recompute = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, recompute), 0)
            rep = json.load(open(recompute))
            stats = os.path.join(tmp, 'stats.json')
            self._official_from_report(rep, stats)
            official = json.load(open(stats))
            official['comparisons']['O_to_B'][
                'ci95_bootstrap'] = [1.0, 1.1]
            json.dump(official, open(stats, 'w'))
            out = os.path.join(tmp, 'r2.json')
            self.assertNotEqual(
                _run(sched, tmp, out, stats_json=stats), 0)
            rep2 = json.load(open(out))
            self.assertTrue(any('ci95_bootstrap' in d for d in
                                rep2['vs_official_stats']
                                ['disagreements']))


# ---------------------------------------------------------------------------
# --mode s (schedule-s S-paired auto-strict batch)
# ---------------------------------------------------------------------------

S_BASE_S = {'S': 3.8, 'B': 1.9}  # b_to_s ~0.5: S costs ~2x B (honest)


def _cert(strict_match_verified=True):
    return {'policy': 'auto+strict', 'main_e_tot': -100.0,
            'main_converged': True, 'ref_e_tot': -100.0,
            'ref_converged': True, 'distinct_solutions': False,
            'selected': 'main',
            'strict_match_verified': strict_match_verified}


def _s_repeat(sample_s, certified=True, converged=True):
    rep = _repeat(sample_s, -100.0, converged=converged)
    rep['stability_verification'] = _cert(strict_match_verified=certified)
    return rep


def _write_s_cohort(root, cells_spec, sched_sha,
                    mutate_sample=None, mutate_merged=None,
                    uncertified=None, drop=None):
    """cells_spec: [(tag, molecule, method)] with 8 blocks per role;
    `uncertified` = {(tag, k)} raw S blocks whose certificate fails."""
    sdir = os.path.join(root, 'samples')
    cdir = os.path.join(root, 'cells')
    os.makedirs(sdir, exist_ok=True)
    os.makedirs(cdir, exist_ok=True)
    for tag, _mol, _meth in cells_spec:
        for short, role in ir.S_ROLE_LONG.items():
            reps = []
            for k in range(8):
                s = S_BASE_S[short] + 0.1 * k
                if role == 'auto_strict':
                    certified = not (uncertified and (tag, k)
                                     in uncertified)
                    rep = _s_repeat(round(s, 6),
                                    certified=certified)
                else:
                    rep = _repeat(round(s, 6), -100.0)
                    rep['stability_verification'] = None
                if mutate_sample and (tag, short, k) in mutate_sample:
                    mutate_sample[(tag, short, k)](rep)
                rec = {'batch_id': ir.S_BATCH_ID,
                       'schedule_sha256': sched_sha, 'role': role,
                       'block': k,
                       'row': {'result': {'repeats': [rep],
                                          'fp32_entry_hits': dict(
                                              ZERO_HITS)}}}
                if drop and (tag, short, k) in drop:
                    continue
                path = os.path.join(sdir, f'{tag}_{role}_b{k}.json')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(rec, f)
                reps.append(rep)
            merged_reps = [json.loads(json.dumps(r)) for r in reps]
            tim = {'samples_s': [r['sample_s'] for r in reps],
                   't_scf_s': statistics.median(
                       r['t_scf_s'] for r in reps),
                   't_grad_s': statistics.median(
                       r['t_grad_s'] for r in reps)}
            doc = {'rows': [{'result': {'repeats': merged_reps,
                                        'timings': tim}}]}
            if mutate_merged and (tag, short) in mutate_merged:
                mutate_merged[(tag, short)](doc)
            with open(os.path.join(cdir, f'{tag}_{role}.json'), 'w',
                      encoding='utf-8') as f:
                json.dump(doc, f)


def _write_s_schedule(root, cells_spec):
    schedule = {
        'schema_version': ir.S_SCHEDULE_SCHEMA, 'seed': 20261029,
        'batch_id': ir.S_BATCH_ID, 'source_commit': 'deadbee',
        'source_root': '/r', 'timing_protocol': 's-paired-v1',
        'settings': {},
        'cells': [{'index': i, 'tag': tag, 'molecule': mol,
                   'basis': 'def2-svp', 'xc': meth, 'charge': 0,
                   'spin': 0, 'geom': '/g', 'geometry_sha256': '0' * 64,
                   'blocks': [list('SB')] * 8}
                  for i, (tag, mol, meth) in enumerate(cells_spec)]}
    path = os.path.join(root, 'schedule-s.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(schedule, f)
    return path


def _s_cohort(tmp, cells_spec=(('t0', 'molA', 'b3lyp'),
                               ('t1', 'molB', 'm062x')), **kw):
    sched_path = _write_s_schedule(tmp, cells_spec)
    sha = ir.sha256_file(sched_path)
    _write_s_cohort(tmp, cells_spec, sha, **kw)
    return sched_path


class TestSMode(unittest.TestCase):

    def test_s_matches_hand_computed(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(tmp)
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out, mode='s'), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['mode'], 's')
            self.assertEqual(rep['schema'], ir.SCHEMA_S)
            self.assertTrue(rep['citable'])
            self.assertTrue(rep['full_coverage'])
            self.assertEqual(rep['counts'], {
                'planned': 2, 'certified_ok': 2,
                'failed_or_missing': 0})
            med = {r: statistics.median(
                S_BASE_S[r] + 0.1 * k for k in range(8))
                for r in 'SB'}
            hand_ratio = med['B'] / med['S']
            for c in rep['per_case']:
                self.assertTrue(c['case_ok'])
                self.assertIsNone(c['reason'])
                self.assertAlmostEqual(
                    c['b_to_s_ratio'], hand_ratio, places=12)
                s_row = c['roles']['S']
                self.assertEqual(s_row['n_certified_repeats'], 8)
                self.assertTrue(s_row['strict_match_all_verified'])
                self.assertAlmostEqual(
                    s_row['median_sample_s'], med['S'], places=12)
                self.assertAlmostEqual(
                    c['roles']['B']['median_sample_s'], med['B'],
                    places=12)
            b = rep['b_to_s']
            self.assertEqual(b['n'], 2)
            self.assertAlmostEqual(
                b['primary_geomean'], hand_ratio, places=12)
            self.assertAlmostEqual(
                b['median_ratio'], hand_ratio, places=12)
            self.assertAlmostEqual(
                b['total_time_ratio'], hand_ratio, places=12)
            for m in ('b3lyp', 'm062x'):
                mc = rep['method_coverage'][m]
                self.assertEqual(mc['planned'], 1)
                self.assertEqual(mc['certified_ok'], 1)
                self.assertAlmostEqual(
                    mc['b_to_s_geomean'], hand_ratio, places=12)
            # 2 cells x 2 roles compared, zero mismatches
            self.assertEqual(
                rep['merge_fidelity']['role_docs_checked'], 4)
            self.assertEqual(rep['merge_fidelity']['mismatches'], [])

    def test_s_mode_schedule_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            obcr_sched = _cohort(tmp)
            with self.assertRaises(SystemExit):
                ir.build_report(obcr_sched, tmp, False, mode='s')
        with tempfile.TemporaryDirectory() as tmp:
            s_sched = _s_cohort(tmp)
            with self.assertRaises(SystemExit):
                ir.build_report(s_sched, tmp, False)  # default obcr

    def test_s_certificate_incomplete_fails_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(tmp, uncertified={('t0', 3)})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out, mode='s'), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['counts'], {
                'planned': 2, 'certified_ok': 1,
                'failed_or_missing': 1})
            case = next(c for c in rep['per_case'] if c['tag'] == 't0')
            self.assertFalse(case['case_ok'])
            self.assertEqual(
                case['reason'],
                'S certificate incomplete (7/8 repeats)')
            self.assertNotIn('b_to_s_ratio', case)
            # honest failure stays in the denominator: ratio over the
            # certified subset only, never a hand-picked replacement
            self.assertEqual(rep['b_to_s']['n'], 1)
            self.assertTrue(rep['citable'])  # full coverage, honest FAIL

    def test_s_merge_fidelity_flags_mutated_certificate(self):
        def _mutate(doc):
            doc['rows'][0]['result']['repeats'][0][
                'stability_verification']['strict_match_verified'] = False
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(
                tmp, mutate_merged={('t0', 'S'): _mutate})
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(_run(sched, tmp, out, mode='s'), 0)
            rep = json.load(open(out))
            self.assertEqual(rep['verdict'], 'DEFECTS_FOUND')
            self.assertTrue(any(
                'stability_verification' in m for m in
                rep['merge_fidelity']['mismatches']))

    def test_s_pin_defect_wrong_batch_id(self):
        def _mutate(rec):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(
                tmp, mutate_sample={('t0', 'S', 2): _mutate})
            # rewrite one raw record's batch pin (the fixture callback
            # cannot reach the envelope, so edit the file directly)
            p = os.path.join(tmp, 'samples', 't0_auto_strict_b2.json')
            rec = json.load(open(p))
            rec['batch_id'] = 'some-other-batch'
            json.dump(rec, open(p, 'w'))
            out = os.path.join(tmp, 'r.json')
            self.assertNotEqual(_run(sched, tmp, out, mode='s'), 0)
            rep = json.load(open(out))
            self.assertTrue(any('batch_id' in d for d in
                                rep['pin_defects']))

    def test_s_allow_partial_labels_non_citable(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(
                tmp, drop={('t1', 'S', 5), ('t1', 'B', 5)})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(
                _run(sched, tmp, out, mode='s', allow_partial=True), 0)
            rep = json.load(open(out))
            self.assertFalse(rep['citable'])
            self.assertTrue(rep['allow_partial'])
            self.assertFalse(rep['full_coverage'])
            self.assertTrue(any('t1/auto_strict' in g for g in
                                rep['coverage_gaps']))
            # without the flag the incomplete coverage must fail the run
            self.assertNotEqual(_run(sched, tmp, out, mode='s'), 0)

    def test_s_vs_official_agree_and_disagree(self):
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(tmp)
            recompute = os.path.join(tmp, 'r.json')
            self.assertEqual(
                _run(sched, tmp, recompute, mode='s'), 0)
            rep = json.load(open(recompute))
            # official-shaped s stats document built from the recompute
            # (round-trip): agreement must be exact
            official = {
                'schema': 'benchmark-v1-s-stats-v1',
                'generated_at_utc': '2026-10-31T00:00:00Z',
                'protocol': {
                    'schedule_sha256': rep['schedule_sha256'],
                    'source_commit': rep['source_commit']},
                'counts': dict(rep['counts']),
                'b_to_s': {k: v for k, v in rep['b_to_s'].items()},
                'method_coverage': json.loads(json.dumps(
                    rep['method_coverage'])),
                'per_case': json.loads(json.dumps(rep['per_case']))}
            stats = os.path.join(tmp, 'stats.json')
            json.dump(official, open(stats, 'w'))
            out = os.path.join(tmp, 'r2.json')
            self.assertEqual(
                _run(sched, tmp, out, mode='s', stats_json=stats), 0)
            rep2 = json.load(open(out))
            self.assertEqual(
                rep2['vs_official_stats']['verdict'], 'AGREE')
            # mutate one headline figure and one per-case certificate:
            # both must be named and must fail the run
            official['b_to_s']['primary_geomean'] = 1.5
            official['per_case'][0]['roles']['S'][
                'n_certified_repeats'] = 3
            json.dump(official, open(stats, 'w'))
            self.assertNotEqual(
                _run(sched, tmp, out, mode='s', stats_json=stats), 0)
            rep3 = json.load(open(out))
            self.assertEqual(
                rep3['vs_official_stats']['verdict'], 'DISAGREE')
            disagreements = rep3['vs_official_stats']['disagreements']
            self.assertTrue(any('b_to_s.primary_geomean' in d
                                for d in disagreements))
            self.assertTrue(any('n_certified_repeats' in d
                                for d in disagreements))

    def test_s_nonconverged_repeat_fails_case(self):
        def _mutate(rep):
            rep['converged'] = False
        with tempfile.TemporaryDirectory() as tmp:
            sched = _s_cohort(
                tmp, mutate_sample={('t1', 'B', 6): _mutate})
            out = os.path.join(tmp, 'r.json')
            self.assertEqual(_run(sched, tmp, out, mode='s'), 0)
            rep = json.load(open(out))
            case = next(c for c in rep['per_case'] if c['tag'] == 't1')
            self.assertFalse(case['case_ok'])
            self.assertEqual(case['reason'], 'B nonconverged repeat')


if __name__ == '__main__':
    unittest.main()

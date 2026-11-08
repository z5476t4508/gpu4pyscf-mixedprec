'''CPU-only executor tests for benchmark v1 (2026-10-30 resume card,
task BENCH-EXEC-01): the 8-block two-Latin-square schedule protocol, the
S-paired schedule, the geometry/source freeze, the gradient-runner CDERI
aggregate repair, the 8-block merge/validate contracts, the S-role
strict-match gate and the planned-denominator statistics module.  No
GPU, no pyscf imports -- only stdlib structures and the pure functions
of the executor modules.'''

import hashlib
import importlib.util
import json
import os
import pathlib
import random
import shutil
import statistics
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


us = _load('upstream_schedule')
gr = _load('gradient_runner')
uc = _load('upstream_comparison')


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _write_geom(path, natm=4):
    lines = [f'{natm}', 'test fixture']
    for i in range(natm):
        lines.append(f'C {0.1 * i:.3f} 0.0 0.0')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def _cohort_fixture(tmp, molecules=('mol_a', 'mol_b'),
                    methods=('b3lyp', 'm062x')):
    '''A miniature benchmark-cohort-v1 package: 2 molecules x 2 methods
    with per-file frozen geometry hashes, same shape as the sealed
    cohort.json.'''
    geom_dir = os.path.join(tmp, 'inputs')
    os.makedirs(geom_dir, exist_ok=True)
    cells, mols = [], []
    for m in molecules:
        fname = f'{m}.xyz'
        digest = _write_geom(os.path.join(geom_dir, fname))
        mols.append({'id': m, 'nao': {'def2-svp': 199}})
        for xc in methods:
            cells.append({
                'tag': f'{m}_{xc}_def2-svp', 'molecule': m,
                'geom': f'inputs/{fname}', 'geometry_sha256': digest,
                'xc': xc, 'basis': 'def2-svp', 'charge': 0, 'spin': 0})
    cohort = {'schema': 'benchmark-cohort-v1',
              'status': 'input_frozen_measurement_not_started',
              'molecules': mols, 'cells': cells}
    path = os.path.join(tmp, 'cohort.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cohort, f)
    return path, geom_dir


def _build_v2(tmp, batch_id='bench-v1-test'):
    cohort_path, geom_dir = _cohort_fixture(tmp)
    specs = us.load_cohort_cells(cohort_path, geom_dir)
    schedule = {
        'schema_version': us.SCHEMA_V2, 'seed': us.SEED_V2,
        'batch_id': batch_id, 'source_commit': 'c0ffee', 'source_root':
        tmp, 'timing_protocol': 'balanced-blocks-v2',
        'settings': us.SETTINGS_V2,
        'cells': us.build_cells_v2(specs, us.SEED_V2)}
    return schedule, cohort_path, geom_dir


# ---------------------------------------------------------------------------
# schedule v2: structure and freeze
# ---------------------------------------------------------------------------

class TestScheduleV2(unittest.TestCase):

    def test_eight_blocks_two_latin_squares(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, _, _ = _build_v2(tmp)
            self.assertEqual(len(schedule['cells']), 4)
            for cell in schedule['cells']:
                blocks = cell['blocks']
                self.assertEqual(len(blocks), 8)
                for half in (blocks[:4], blocks[4:]):
                    for row in half:
                        self.assertEqual(sorted(row), sorted(us.ROLES))
                    for pos in range(4):
                        col = [b[pos] for b in half]
                        self.assertEqual(sorted(col), sorted(us.ROLES))
            self.assertTrue(us.verify_schedule_v2(schedule))

    def test_seed_frozen_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            s1, _, _ = _build_v2(tmp)
            s2, _, _ = _build_v2(tmp)
            self.assertEqual(
                [c['blocks'] for c in s1['cells']],
                [c['blocks'] for c in s2['cells']])
            s1['seed'] = 20261030
            with self.assertRaises(ValueError):
                us.verify_schedule_v2(s1)

    def test_geometry_tamper_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, cohort_path, geom_dir = _build_v2(tmp)
            self.assertTrue(us.verify_schedule_v2(schedule))
            victim = schedule['cells'][0]['geom']
            with open(victim, 'a', encoding='utf-8') as f:
                f.write('C 9.9 9.9 9.9\n')
            with self.assertRaises(ValueError):
                us.verify_schedule_v2(schedule)

    def test_cohort_status_and_hash_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            cohort_path, geom_dir = _cohort_fixture(tmp)
            with open(cohort_path, encoding='utf-8') as f:
                cohort = json.load(f)
            cohort['status'] = 'something-else'
            with open(cohort_path, 'w', encoding='utf-8') as f:
                json.dump(cohort, f)
            with self.assertRaises(ValueError):
                us.load_cohort_cells(cohort_path, geom_dir)
            # geometry drift vs the frozen hash refuses at load too
            with tempfile.TemporaryDirectory() as tmp2:
                cohort_path2, geom_dir2 = _cohort_fixture(tmp2)
                first = sorted(os.listdir(geom_dir2))[0]
                with open(os.path.join(geom_dir2, first), 'a') as f:
                    f.write('H 0.0 0.0 0.1\n')
                with self.assertRaises(ValueError):
                    us.load_cohort_cells(cohort_path2, geom_dir2)

    def test_v1_legacy_still_supported_and_never_reinterpreted(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, 'sched-v1.json')
            rc = us.main(['--out', out, '--batch-id', 'legacy',
                         '--source-commit', 'x', '--source-root', tmp])
            self.assertEqual(rc, 0)
            with open(out, encoding='utf-8') as f:
                v1 = json.load(f)
            self.assertEqual(v1['schema_version'], us.SCHEMA)
            self.assertEqual(len(v1['cells'][0]['blocks']), 4)
            self.assertTrue(us.verify_dispatch(v1))
            # an 8-block schedule must NOT validate under the v1 rules
            v2, _, _ = _build_v2(tmp)
            with self.assertRaises(ValueError):
                us.verify_schedule(v2)
            self.assertTrue(us.verify_dispatch(v2))

    def test_s_schedule_pair_balance(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, _, _ = _build_v2(tmp)
            s = us.build_schedule_s(schedule['cells'], 'bench-v1-test-S',
                                    'c0ffee', tmp)
            self.assertEqual(s['schema_version'], us.SCHEMA_S)
            self.assertEqual(len(s['cells']), 4)
            for cell in s['cells']:
                self.assertEqual(len(cell['blocks']), 8)
                leads = sum(1 for b in cell['blocks'] if b[0] == 'S')
                self.assertEqual(leads, 4)
                for b in cell['blocks']:
                    self.assertEqual(sorted(b), sorted(us.S_ROLES))
            self.assertTrue(us.verify_dispatch(s))
            # S schedule never passes under the O/B/C/R v2 verifier
            with self.assertRaises(ValueError):
                us.verify_schedule_v2(s, roles=us.ROLES)


# ---------------------------------------------------------------------------
# gradient_runner: CDERI aggregate repair
# ---------------------------------------------------------------------------

class TestCderiAggregate(unittest.TestCase):

    def test_aggregate_is_real_sample_statistic(self):
        cderi = [1.0, 2.0, 3.0, 4.0]
        block = gr._grad_timing_block(9.0, [10.0] * 4, [5.0] * 4,
                                      [2.0] * 4, 'fp64',
                                      cderi_times=cderi,
                                      mol_times=[1.0] * 4)
        self.assertAlmostEqual(block['t_cderi_s'],
                               statistics.median(cderi))
        self.assertAlmostEqual(block['t_mol_s'], 1.0)

    def test_legacy_call_keeps_placeholder_with_note(self):
        block = gr._grad_timing_block(9.0, [10.0], [5.0], [2.0], 'fp64')
        self.assertEqual(block['t_cderi_s'], 0.0)
        self.assertIn('t_cderi_note', block)

    def test_repeat_phases_sum_to_sample(self):
        phases = {'t_mol_s': 1.0, 't_cderi_s': 2.0, 't_scf_s': 5.0,
                  't_grad_s': 2.0}
        rep = gr._build_grad_repeat(0, -1.0, True, 9,
                                    [[0.0, 0.0, 0.0]], 'fp64', 0, None,
                                    {'switched': False}, {}, phases)
        self.assertAlmostEqual(rep['sample_s'],
                               rep['t_mol_s'] + rep['t_cderi_s'] +
                               rep['t_scf_s'] + rep['t_grad_s'])
        self.assertAlmostEqual(rep['t_cderi_s'], 2.0)

    def test_aggregate_medians_match_repeat_values(self):
        phases = [{'t_mol_s': 1.0, 't_cderi_s': 2.0 + 0.1 * i,
                   't_scf_s': 5.0, 't_grad_s': 2.0} for i in range(8)]
        reps = [gr._build_grad_repeat(
            i, -1.0, True, 9, [[0.0, 0.0, 0.0]], 'fp64', 0, None,
            {'switched': False}, {}, p) for i, p in enumerate(phases)]
        block = gr._grad_timing_block(
            9.0, [r['sample_s'] for r in reps],
            [r['t_scf_s'] for r in reps], [r['t_grad_s'] for r in reps],
            'fp64', cderi_times=[r['t_cderi_s'] for r in reps],
            mol_times=[r['t_mol_s'] for r in reps])
        self.assertAlmostEqual(block['t_cderi_s'],
                               statistics.median(
                                   [r['t_cderi_s'] for r in reps]))


class TestOfficialArmTimingBlock(unittest.TestCase):
    '''The O-lane arm's per-record timing block must carry the MEASURED
    cderi/mol phases (2026-10-30 fix): run_official_cell built it via a
    bare _grad_timing_block call that dropped phases['t_cderi_s'] /
    phases['t_mol_s'], so every per-arm upstream record in batch
    benchv1-20261030-obcr shows the misleading pre-repair 0.0
    placeholder while the repeat next to it holds the real measurement
    (merged docs and stats read repeats and were never affected).'''

    PHASES = {'t_mol_s': 0.005018, 't_cderi_s': 0.179483,
              't_scf_s': 1.475106, 't_grad_s': 0.294441}

    def test_block_carries_measured_cderi_and_mol(self):
        block = uc._official_timing_block(gr, 4.367548, self.PHASES)
        self.assertAlmostEqual(block['t_cderi_s'],
                               self.PHASES['t_cderi_s'])
        self.assertAlmostEqual(block['t_mol_s'], self.PHASES['t_mol_s'])
        self.assertNotIn('t_cderi_note', block)

    def test_phase_sums_match_sample_total(self):
        block = uc._official_timing_block(gr, 4.367548, self.PHASES)
        total = (block['t_mol_s'] + block['t_cderi_s'] +
                 block['t_scf_s'] + block['t_grad_s'])
        self.assertAlmostEqual(total, block['median_s'], places=5)

    def test_old_inline_form_would_misattribute(self):
        # red state of the pre-fix call-site: scf/grad only, cderi dropped
        block = gr._grad_timing_block(
            4.367548, [sum(self.PHASES.values())],
            [self.PHASES['t_scf_s']], [self.PHASES['t_grad_s']], 'fp64')
        self.assertEqual(block['t_cderi_s'], 0.0)
        self.assertIn('t_cderi_note', block)

    def test_run_official_cell_uses_the_helper(self):
        # guard the call-site itself: run_official_cell cannot be executed
        # CPU-only (it opens the CUDA device first), so the regression
        # guard is structural -- the bare inline form is what regressed
        import inspect
        src = inspect.getsource(uc.run_official_cell)
        self.assertIn('_official_timing_block(', src)
        self.assertNotIn('gradient_runner._grad_timing_block(', src)


# ---------------------------------------------------------------------------
# 8-block merge + validate + S-role gate
# ---------------------------------------------------------------------------

def _sample_record(role, block, e=-100.0, sample_s=10.0, converged=True,
                   grad=None, hits=None, sv_record=None):
    short = {'upstream': 'O', 'optimized_fp64': 'B',
             'optimized_auto': 'C', 'strict_fp64': 'R',
             'auto_strict': 'S'}[role]
    if grad is None:
        grad = [[0.01, 0.0, 0.0]]
    if hits is None:
        hits = {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0,
                'other': 0}
    return {
        'schema_version': uc.SAMPLE_SCHEMA, 'batch_id': 'b',
        'schedule_sha256': 's' * 64, 'cell_index': 0, 'role': role,
        'block': block, 'position': 0,
        'lane': uc.ROLE_TO_LANE[role], 'optimization_env': {},
        'startup': {'warmup_s': 1.0},
        'meta_base': {'dependencies': {'pyscf': '2', 'cupy': '13',
                                       'numpy': '1', 'libxc': '6',
                                       'cutensor': '2'},
                      'gpu_name': 'G', 'driver_version': 'd',
                      'cuda_version': 'c', 'cpu_model': 'm',
                      'python': 'p', 'cupy': '13', 'pyscf': '2'},
        'provenance': {'gpu4pyscf_package_root': '/x'},
        'row': {'row_key': {'lane': uc.ROLE_TO_LANE[role]},
                'system': {'nao': 199},
                'method': {'xc': 'b3lyp', 'basis': 'def2-svp'},
                'result': {'repeats': [{
                    'index': 0, 'e_tot': e, 'converged': converged,
                    'scf_cycles': 9, 'gradient': grad,
                    'gradient_norm': 0.01, 'gradient_mode': 'fp64',
                    'grad_ip1_hits': 0,
                    'precision_switch': {'switched': False},
                    'stability_verification': sv_record,
                    'sample_s': sample_s, 't_mol_s': 1.0,
                    't_cderi_s': 2.0, 't_scf_s': 5.0,
                    't_grad_s': 2.0}],
                    'fp32_entry_hits': hits,
                    'gradient_mode': 'fp64',
                    'precision_switch': {'switched': False,
                                         'cycle': None,
                                         'trigger': 'none'}}},
        '_path': f'/fake/{role}-{block}.json'}


def _v2_schedule(n_cells=1, n_blocks=8):
    return {
        'schema_version': us.SCHEMA_V2, 'seed': us.SEED_V2,
        'batch_id': 'b', 'source_commit': 'c', 'source_root': '/r',
        'timing_protocol': 'balanced-blocks-v2', 'settings':
        us.SETTINGS_V2,
        'cells': [{'index': 0, 'tag': 't0', 'geom': '/g', 'basis':
                   'def2-svp', 'xc': 'b3lyp', 'charge': 0, 'spin': 0,
                   'blocks': [list('OBCR')] * n_blocks,
                   'geometry_sha256': '0' * 64}] * n_cells}


def _s_schedule(n_cells=1):
    return {
        'schema_version': us.SCHEMA_S, 'seed': us.SEED_S,
        'batch_id': 'b-S', 'source_commit': 'c', 'source_root': '/r',
        'timing_protocol': 's-paired-v1', 'settings': us.SETTINGS_V2,
        'cells': [{'index': 0, 'tag': 't0', 'geom': '/g', 'basis':
                   'def2-svp', 'xc': 'b3lyp', 'charge': 0, 'spin': 0,
                   'blocks': [['S', 'B'], ['B', 'S']] * 4,
                   'geometry_sha256': '0' * 64}] * n_cells}


SV_OK = {'policy': 'auto+strict', 'strict_match_verified': True}
SV_FAIL = {'policy': 'auto+strict', 'strict_match_verified': False,
           'strict_match_incomplete_reason': 'reference diverged'}


class TestMergeValidateV2(unittest.TestCase):

    def test_merge_eight_blocks(self):
        schedule = _v2_schedule()
        import tempfile as _tf
        from unittest import mock
        with _tf.TemporaryDirectory() as cdir:
            cert_path = os.path.join(cdir, 'cert.json')
            with open(cert_path, 'w', encoding='utf-8') as f:
                json.dump({'package_root': '/cert_root'}, f)
            cert_hash = hashlib.sha256(
                open(cert_path, 'rb').read()).hexdigest()
            for role in uc.ROLES:
                samples = [_sample_record(role, b) for b in range(8)]
                for s in samples:
                    s['provenance'] = {
                        'gpu4pyscf_package_root':
                            '/cert_root/gpu4pyscf',
                        'reference_manifest_sha256': cert_hash}
                doc = uc.merge_role(samples, role, 'b', schedule,
                                    's' * 64)
                self.assertEqual(
                    len(doc['rows'][0]['result']['repeats']), 8)
                timings = doc['rows'][0]['result']['timings']
                self.assertEqual(timings['samples_s'], [10.0] * 8)
                # the CDERI aggregate survives the merge as a real
                # statistic
                self.assertAlmostEqual(timings['t_cderi_s'], 2.0)
                self.assertIn('balanced-blocks-v2', timings['sync'])
                self.assertEqual(doc['meta']['timing_protocol'],
                                 'balanced-blocks-v2')
                with mock.patch.object(uc, 'OFFICIAL_CERTIFICATE',
                                       cert_path):
                    uc.validate_comparison_doc(doc, role, 'b', 's' * 64,
                                               expected_repeats=8)

    def test_merge_rejects_four_blocks_under_v2(self):
        schedule = _v2_schedule()
        samples = [_sample_record('upstream', b) for b in range(4)]
        with self.assertRaises(ValueError):
            uc.merge_role(samples, 'upstream', 'b', schedule, 's' * 64)

    def test_v1_merge_still_four(self):
        schedule = _v2_schedule(n_blocks=4)
        schedule['schema_version'] = us.SCHEMA
        schedule['timing_protocol'] = 'balanced-blocks-v1'
        schedule['seed'] = us.SEED
        schedule['settings'] = us.SETTINGS
        samples = [_sample_record('upstream', b) for b in range(4)]
        doc = uc.merge_role(samples, 'upstream', 'b', schedule, 's' * 64)
        self.assertEqual(len(doc['rows'][0]['result']['repeats']), 4)

    def test_s_role_requires_strict_match_on_every_repeat(self):
        schedule = _s_schedule()
        samples = [_sample_record(uc.ROLE_AUTO_STRICT, b, sv_record=SV_OK)
                   for b in range(8)]
        doc = uc.merge_role(samples, uc.ROLE_AUTO_STRICT, 'b-S',
                            schedule, 's' * 64)
        uc.validate_comparison_doc(doc, uc.ROLE_AUTO_STRICT, 'b-S',
                                   's' * 64, expected_repeats=8)
        # one uncertified repeat fails the whole merge validation
        samples[3] = _sample_record(uc.ROLE_AUTO_STRICT, 3,
                                    sv_record=SV_FAIL)
        doc = uc.merge_role(samples, uc.ROLE_AUTO_STRICT, 'b-S',
                            schedule, 's' * 64)
        with self.assertRaises(ValueError):
            uc.validate_comparison_doc(doc, uc.ROLE_AUTO_STRICT, 'b-S',
                                       's' * 64, expected_repeats=8)

    def test_roles_route_by_schema(self):
        self.assertEqual(uc.roles_for_schedule(_v2_schedule()), uc.ROLES)
        self.assertEqual(uc.roles_for_schedule(_s_schedule()),
                         uc.ROLES_S)
        bad = _v2_schedule()
        bad['schema_version'] = 'unknown'
        with self.assertRaises(ValueError):
            uc.roles_for_schedule(bad)

    def test_optimization_env_of_s_gets_stack(self):
        self.assertEqual(uc.optimization_env_of(uc.ROLE_AUTO_STRICT),
                         dict(uc.STACK_ENV))
        self.assertEqual(uc.optimization_env_of('upstream'), {})
        self.assertEqual(uc.optimization_env_of('strict_fp64'), {})


# ---------------------------------------------------------------------------
# planned-denominator statistics
# ---------------------------------------------------------------------------

bvs = _load('benchmark_v1_stats')


def _cell_doc(role, blocks, e, sample_s, grad=None, converged=True,
              auto_hits=False):
    if grad is None:
        grad = [[0.0, 0.0, 0.0]]
    reps = []
    for b in range(blocks):
        reps.append({'index': b, 'comparison_block': b, 'e_tot': e,
                     'converged': converged, 'gradient': grad,
                     'sample_s': sample_s + 0.01 * b, 't_scf_s':
                     sample_s / 2, 't_grad_s': sample_s / 4,
                     't_cderi_s': sample_s / 8, 't_mol_s': sample_s / 8})
    hits = {'df_jk': 5 if auto_hits else 0, 'eval_rho': 0,
            'xc_grid': 0, 'grad_ip1': 3 if auto_hits else 0, 'other': 0}
    return {'rows': [{'result': {'repeats': reps,
                                 'fp32_entry_hits': hits}}]}


def _stats_fixture(tmp, n_mol=2, n_method=2, blocks=8,
                   fail_cell=None, miss_cell=None, o_fail_cell=None):
    ev = os.path.join(tmp, 'ev')
    os.makedirs(os.path.join(ev, 'cells'), exist_ok=True)
    molecules = [f'mol{i}' for i in range(n_mol)]
    methods = ['b3lyp', 'm062x'][:n_method]
    cells = []
    for m in molecules:
        for xc in methods:
            cells.append({'tag': f'{m}_{xc}', 'molecule': m, 'xc': xc,
                          'basis': 'def2-svp', 'blocks':
                          [list('OBCR')] * blocks})
    schedule = {'cells': cells, 'source_commit': 'c0ffee',
                'timing_protocol': 'balanced-blocks-v2', '_sha256':
                'f' * 64}
    aoi = {'molecules': [{'id': m, 'nao': {'def2-svp': 199 if i == 0
                                           else 300}}
                         for i, m in enumerate(molecules)]}
    for cell in cells:
        tag = cell['tag']
        auto_hits = (tag == 'mol0_b3lyp')  # one actual-FP32 auto cell
        for role, e, s in (('upstream', -100.0, 10.0),
                           ('optimized_fp64', -100.0, 8.0),
                           ('optimized_auto', -100.0, 6.0),
                           ('strict_fp64', -100.0, 12.0)):
            if miss_cell == tag:
                continue
            role_hits = auto_hits and role == 'optimized_auto'
            doc = _cell_doc(role, blocks, e, s, auto_hits=role_hits)
            if fail_cell == tag and role == 'optimized_auto':
                doc = _cell_doc(role, blocks, -100.0 + 1e-3, s,
                                auto_hits=role_hits)
            if o_fail_cell == tag and role == 'upstream':
                doc = _cell_doc(role, blocks, -100.0 + 1e-3, s)
            path = os.path.join(ev, 'cells', f'{tag}_{role}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
    return schedule, ev, aoi


class TestBenchmarkV1Stats(unittest.TestCase):

    def test_full_cohort_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(tmp)
            rep = bvs.build_report(schedule, ev, {}, aoi)
            self.assertTrue(rep['full_cohort_available'])
            self.assertEqual(rep['counts']['planned'], 4)
            self.assertEqual(rep['counts']['gate_pass'], 4)
            self.assertEqual(rep['counts']['actual_scf_fp32_C'], 1)
            self.assertEqual(rep['counts']['gated_fallback_C'], 3)
            # hand-computed values: each role's per-case median is
            # s + 0.01*median(range(8)) = s + 0.035, so every per-case
            # speedup is (s_num+0.035)/(s_den+0.035)
            self.assertAlmostEqual(
                rep['comparisons']['O_to_B']['primary_geomean'],
                10.035 / 8.035, places=10)
            self.assertAlmostEqual(
                rep['comparisons']['B_to_C']['primary_geomean'],
                8.035 / 6.035, places=10)
            self.assertTrue(
                rep['performance_claim'][
                    'mixed_precision_speedup_established'])
            self.assertIn('strata', rep)
            self.assertIn('b3lyp', rep['strata']['method'])
            self.assertIn('<250', rep['strata']['ao'])

    def test_missing_case_blocks_full_cohort_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(
                tmp, miss_cell='mol1_m062x')
            rep = bvs.build_report(schedule, ev, {}, aoi)
            self.assertFalse(rep['full_cohort_available'])
            self.assertIsNone(rep['performance_claim'])
            self.assertEqual(rep['counts']['planned'], 4)
            self.assertEqual(rep['counts']['gate_fail_or_missing'], 1)
            self.assertIn('diagnostic_common_valid_subset', rep)
            md = bvs.render_md(rep)
            self.assertIn('全集主成绩不可用', md)

    def test_empty_stratum_and_zero_pass_cohort_do_not_crash(self):
        # dry-run regression (2026-10-30, live partial record): a stratum
        # whose cases are ALL missing/failed, and a zero-pass complete
        # set, must yield None statistics and a renderable report --
        # statistics.median([]) crashed the whole build before
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(
                tmp, n_mol=2, n_method=2,
                fail_cell='mol0_b3lyp', miss_cell='mol1_b3lyp')
            rep = bvs.build_report(schedule, ev, {}, aoi)
            self.assertFalse(rep['full_cohort_available'])
            self.assertEqual(rep['counts']['planned'], 4)
            self.assertEqual(rep['counts']['gate_pass'], 2)
            self.assertEqual(rep['counts']['gate_fail_or_missing'], 2)
            # the passing m062x subset still gets diagnostic statistics
            self.assertEqual(rep['comparisons']['O_to_B']['n'], 2)
            # the all-failed/missing b3lyp stratum: None, not a crash
            self.assertEqual(
                rep['strata']['method']['b3lyp']['gate_pass'], 0)
            self.assertIsNone(
                rep['strata']['method']['b3lyp']['O_to_B'])
            self.assertIsNotNone(
                rep['strata']['method']['m062x']['O_to_B'])
            self.assertIn('全集主成绩不可用', bvs.render_md(rep))
        with tempfile.TemporaryDirectory() as tmp:
            # every cell missing: complete == [] at the top-level call
            schedule, ev, aoi = _stats_fixture(
                tmp, n_mol=1, n_method=1, miss_cell='mol0_b3lyp')
            rep = bvs.build_report(schedule, ev, {}, aoi)
            self.assertEqual(rep['counts']['gate_pass'], 0)
            self.assertEqual(rep['comparisons']['B_to_C'], {})
            self.assertIsNone(rep['performance_claim'])
            self.assertIn('不可裁定', bvs.render_md(rep))

    def test_scientific_failure_stays_in_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(
                tmp, fail_cell='mol1_b3lyp')
            rep = bvs.build_report(schedule, ev, {}, aoi)
            self.assertFalse(rep['full_cohort_available'])
            case = next(c for c in rep['per_case']
                        if c['tag'] == 'mol1_b3lyp')
            self.assertFalse(case['gates']['C']['pass'])
            self.assertIn('max|dE|', case['gates']['C']['reason'])

    def test_gate_checked_for_all_roles_before_gated_classification(self):
        # B01 regression class: a gated auto cell (no SCF FP32) must
        # still surface an upstream gate failure -- classification never
        # short-circuits the per-role gates
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(
                tmp, o_fail_cell='mol1_b3lyp')
            rep = bvs.build_report(schedule, ev, {}, aoi)
            case = next(c for c in rep['per_case']
                        if c['tag'] == 'mol1_b3lyp')
            self.assertFalse(case['gates']['O']['pass'])
            self.assertFalse(case['gate_pass'])
            self.assertFalse(rep['full_cohort_available'])

    def test_bootstrap_deterministic_and_seeded(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(tmp)
            rep = bvs.build_report(schedule, ev, {}, aoi)
            ci1 = rep['comparisons']['B_to_C']['ci95_bootstrap']
            rep2 = bvs.build_report(schedule, ev, {}, aoi)
            ci2 = rep2['comparisons']['B_to_C']['ci95_bootstrap']
            self.assertEqual(ci1, ci2)
            self.assertLessEqual(ci1[0], ci1[1])
            # a different seed must be able to differ (probability of
            # identity across 10000 draws is negligible)
            schedule3 = json.loads(json.dumps(schedule))
            ev3 = os.path.join(tmp, 'ev3')
            shutil.copytree(ev, ev3)
            old = bvs.BOOTSTRAP_SEED
            bvs.BOOTSTRAP_SEED = 20261101
            try:
                rep3 = bvs.build_report(schedule3, ev3, {}, aoi)
            finally:
                bvs.BOOTSTRAP_SEED = old
            ci3 = rep3['comparisons']['B_to_C']['ci95_bootstrap']
            self.assertNotEqual(ci1, ci3)

    def test_total_time_ratio_is_sum_of_medians(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _stats_fixture(tmp)
            rep = bvs.build_report(schedule, ev, {}, aoi)
            e = rep['comparisons']['O_to_B']
            self.assertAlmostEqual(e['total_time_ratio'],
                                   10.035 / 8.035, places=10)


def _s_cell_doc(role, blocks, sample_s, certified=True, converged=True):
    reps = []
    for b in range(blocks):
        r = {'index': b, 'e_tot': -100.0, 'converged': converged,
             'gradient': [[0.0, 0.0, 0.0]],
             'sample_s': sample_s + 0.01 * b, 't_scf_s': sample_s / 2,
             't_grad_s': sample_s / 4, 't_cderi_s': sample_s / 8,
             't_mol_s': sample_s / 8}
        if role == 'auto_strict':
            r['stability_verification'] = (dict(SV_OK) if certified
                                           else dict(SV_FAIL))
        reps.append(r)
    return {'rows': [{'result': {'repeats': reps}}]}


def _s_stats_fixture(tmp, uncertified_cell=None, miss_cell=None):
    ev = os.path.join(tmp, 'ev-s')
    os.makedirs(os.path.join(ev, 'cells'), exist_ok=True)
    molecules = ['mol0', 'mol1']
    methods = ['b3lyp', 'm062x']
    cells = []
    for m in molecules:
        for xc in methods:
            cells.append({'index': 0, 'tag': f'{m}_{xc}', 'molecule': m,
                          'geom': '/g', 'basis': 'def2-svp', 'xc': xc,
                          'charge': 0, 'spin': 0,
                          'blocks': [['S', 'B'], ['B', 'S']] * 4,
                          'geometry_sha256': '0' * 64})
    schedule = {'schema_version': us.SCHEMA_S, 'seed': us.SEED_S,
                'batch_id': 'b-S', 'source_commit': 'c0ffee',
                'source_root': '/r', 'timing_protocol': 's-paired-v1',
                'settings': us.SETTINGS_V2, '_sha256': 'f' * 64,
                'cells': cells}
    aoi = {'molecules': [{'id': m, 'nao': {'def2-svp': 199}}
                         for m in molecules]}
    for cell in cells:
        tag = cell['tag']
        for role, s in (('auto_strict', 12.0), ('optimized_fp64', 8.0)):
            if role == 'auto_strict' and miss_cell == tag:
                continue
            doc = _s_cell_doc(role, 8, s,
                              certified=(tag != uncertified_cell))
            path = os.path.join(ev, 'cells', f'{tag}_{role}.json')
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(doc, f)
    return schedule, ev, aoi


class TestBenchmarkV1SStats(unittest.TestCase):

    def test_s_report_certified_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _s_stats_fixture(tmp)
            rep = bvs.build_s_report(schedule, ev, {}, aoi)
            self.assertEqual(rep['counts'],
                             {'planned': 4, 'certified_ok': 4,
                              'failed_or_missing': 0})
            # hand-computed: per-case median = s + 0.035, so every
            # b_to_s ratio is (8.035)/(12.035) -- S charges more
            self.assertAlmostEqual(
                rep['b_to_s']['primary_geomean'],
                8.035 / 12.035, places=10)
            self.assertAlmostEqual(
                rep['b_to_s']['total_time_ratio'],
                8.035 / 12.035, places=10)
            self.assertEqual(
                rep['method_coverage']['b3lyp']['certified_ok'], 2)
            md = bvs.render_s_md(rep)
            self.assertIn('B→S', md)
            self.assertIn('绝不借用 C 加速', md)

    def test_s_uncertified_and_missing_stay_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            schedule, ev, aoi = _s_stats_fixture(
                tmp, uncertified_cell='mol0_b3lyp',
                miss_cell='mol1_b3lyp')
            rep = bvs.build_s_report(schedule, ev, {}, aoi)
            self.assertEqual(rep['counts']['certified_ok'], 2)
            self.assertEqual(rep['counts']['failed_or_missing'], 2)
            # statistics only over the certified subset, unchanged
            self.assertAlmostEqual(
                rep['b_to_s']['primary_geomean'],
                8.035 / 12.035, places=10)
            uncert = next(c for c in rep['per_case']
                          if c['tag'] == 'mol0_b3lyp')
            self.assertFalse(uncert['case_ok'])
            self.assertIn('certificate incomplete (0/8',
                          uncert['reason'])
            missing = next(c for c in rep['per_case']
                           if c['tag'] == 'mol1_b3lyp')
            self.assertIn('S arm missing', missing['reason'])
            md = bvs.render_s_md(rep)
            self.assertIn('失败或缺失 2', md)

    def test_s_report_and_cli_refuse_wrong_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                bvs.build_s_report(_v2_schedule(), tmp, {}, {})
            s_schedule, ev, aoi = _s_stats_fixture(tmp)
            v2_path = os.path.join(tmp, 'v2.json')
            with open(v2_path, 'w', encoding='utf-8') as f:
                json.dump(_v2_schedule(), f)
            s_path = os.path.join(tmp, 's.json')
            with open(s_path, 'w', encoding='utf-8') as f:
                json.dump(s_schedule, f)
            out = os.path.join(tmp, 'o.json')
            with self.assertRaises(SystemExit):
                bvs.main(['--schedule', v2_path, '--evidence-root', ev,
                          '--out-json', out, '--mode', 's'])
            with self.assertRaises(SystemExit):
                bvs.main(['--schedule', s_path, '--evidence-root', ev,
                          '--out-json', out])
            self.assertFalse(os.path.exists(out))
            # the correct pairing writes both outputs and exits 0
            rc = bvs.main(['--schedule', s_path, '--evidence-root', ev,
                           '--out-json', out,
                           '--out-md', out + '.md', '--mode', 's'])
            self.assertEqual(rc, 0)
            self.assertTrue(os.path.exists(out))
            self.assertTrue(os.path.exists(out + '.md'))


if __name__ == '__main__':
    unittest.main()

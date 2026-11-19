'''CPU-only tests for benchv1_index_append.py (post-batch effects-index
append + daily-comparison figures): driver-alive guard, upstream
completeness/hash-pin refusals, stats denominator checks, the full
append pipeline with idempotence and post-write re-verification, the
dry-run no-write contract, honest-figure rendering for incomplete
cohorts, and --mode s (separate auto-strict S/B append: own stats
schema, two-role pins, own daily schema/dataset, cross-mode schema
routing refusals).  No GPU, no pyscf, stdlib only.'''

import hashlib
import importlib.util
import json
import os
import pathlib
import shutil
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


app = _load('benchv1_index_append')


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def make_cell(root, tag):
    '''One merged cell: four role docs + a summary doc in cells/.'''
    os.makedirs(os.path.join(root, 'cells'), exist_ok=True)
    paths = {}
    for role in app.ROLES:
        p = os.path.join(root, 'cells', f'{tag}_{role}.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'tag': tag, 'role': role, 'e_tot': -1.0}, f)
        paths[role] = p
    with open(os.path.join(root, 'cells', f'{tag}_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'tag': tag}, f)
    return paths


def make_up_index(tmp, tags):
    cells = []
    for i, tag in enumerate(tags):
        role_paths = make_cell(tmp, tag)
        cells.append({
            'cell': {'index': i, 'tag': tag},
            'sources': {r: {'path': role_paths[r],
                            'sha256': sha(role_paths[r])}
                        for r in app.ROLES},
        })
    idx = {'batches': [{'id': 'benchv1-20261030-obcr', 'cells': cells}]}
    p = os.path.join(tmp, 'up-index.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(idx, f)
    return p


def make_stats(tmp, planned=2, passed=2, full_cohort=True, geomean=1.5,
               schema=app.STATS_SCHEMA):
    if passed > 0:
        comps = {
            label: {'n': passed, 'primary_geomean': geomean,
                    'median_speedup': geomean,
                    'total_time_ratio': geomean,
                    'ci95_bootstrap': [geomean - 0.03, geomean + 0.03]}
            for label in app.COMPARISON_LABELS}
    else:
        comps = {label: {} for label in app.COMPARISON_LABELS}
    stats = {
        'schema': schema,
        'generated_at_utc': '2026-10-31T00:00:00Z',
        'protocol': {'dataset': 'gpu4pyscf-benchmark-v1-20261029',
                     'schedule_sha256': 'a' * 64,
                     'source_commit': 'deadbeef',
                     'blocks_per_role_per_case': 8,
                     'bootstrap': {'draws': 10000, 'seed': 20261029}},
        'counts': {'planned': planned, 'gate_pass': passed,
                   'gate_fail_or_missing': planned - passed,
                   'actual_scf_fp32_C': 0,
                   'gated_fallback_C': passed},
        'full_cohort_available': full_cohort,
        'comparisons': comps,
        'performance_claim': None,
    }
    p = os.path.join(tmp, 'stats.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    with open(os.path.join(tmp, 'stats.md'), 'w', encoding='utf-8') as f:
        f.write('# stats\n')
    return p


def make_ds_index(tmp):
    p = os.path.join(tmp, 'ds-index.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump({'datasets': [{'id': 'prior-dataset',
                                 'label': 'prior',
                                 'recorded_at': '2026-01-01T00:00:00Z',
                                 'sources': []}]}, f)
    return p


def make_cell_s(root, tag):
    '''One merged S-batch cell: two role docs + a summary doc.'''
    os.makedirs(os.path.join(root, 'cells'), exist_ok=True)
    paths = {}
    for role in app.S_ROLES:
        p = os.path.join(root, 'cells', f'{tag}_{role}.json')
        with open(p, 'w', encoding='utf-8') as f:
            json.dump({'tag': tag, 'role': role, 'e_tot': -1.0}, f)
        paths[role] = p
    with open(os.path.join(root, 'cells', f'{tag}_summary.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'tag': tag}, f)
    return paths


def make_up_index_s(tmp, tags, roles=app.S_ROLES):
    cells = []
    for i, tag in enumerate(tags):
        role_paths = make_cell_s(tmp, tag)
        cells.append({
            'cell': {'index': i, 'tag': tag},
            'sources': {r: {'path': role_paths[r],
                            'sha256': sha(role_paths[r])}
                        for r in roles},
        })
    idx = {'batches': [{'id': 'benchv1-20261030-obcr-S', 'cells': cells}]}
    p = os.path.join(tmp, 'up-index-s.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(idx, f)
    return p


def make_stats_s(tmp, planned=2, certified=2, geomean=0.98,
                 schema=app.S_STATS_SCHEMA):
    if certified > 0:
        b_to_s = {'note': 's-fixture', 'n': certified,
                  'primary_geomean': geomean,
                  'median_ratio': geomean,
                  'total_time_ratio': geomean - 0.01}
    else:
        b_to_s = {'note': 's-fixture', 'n': 0, 'primary_geomean': None,
                  'median_ratio': None, 'total_time_ratio': None}
    stats = {
        'schema': schema,
        'generated_at_utc': '2026-10-31T00:00:00Z',
        'protocol': {'dataset': 'gpu4pyscf-benchmark-v1-20261029',
                     'schedule_sha256': 'b' * 64,
                     'source_commit': 's0urce',
                     'blocks_per_role_per_case': 8},
        'counts': {'planned': planned, 'certified_ok': certified,
                   'failed_or_missing': planned - certified},
        'b_to_s': b_to_s,
        'method_coverage': {},
        'per_case': [],
    }
    p = os.path.join(tmp, 'stats-s.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(stats, f)
    with open(os.path.join(tmp, 'stats-s.md'), 'w', encoding='utf-8') as f:
        f.write('# s stats\n')
    return p


def make_proc_root(tmp, markers):
    '''A synthetic /proc with one live process per marker string.'''
    root = os.path.join(tmp, 'proc')
    os.makedirs(os.path.join(root, '99'), exist_ok=True)
    with open(os.path.join(root, '99', 'cmdline'), 'wb') as f:
        f.write('\0'.join(markers).encode() + b'\0')
    return root


class TestGuard(unittest.TestCase):

    def test_refuses_while_driver_alive(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = make_proc_root(
                tmp, ['/x/python', '/y/run_upstream_batch.py', '--schedule'])
            with self.assertRaises(SystemExit) as cm:
                app.guard_no_driver(proc)
            self.assertIn('batch driver alive', str(cm.exception))

    def test_clean_proc_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = make_proc_root(
                tmp, ['/x/python', '/y/unrelated.py'])
            self.assertIsNone(app.guard_no_driver(proc))


class TestVerifyUpstreamBatch(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.tags = ['molA_b3lyp_def2-svp', 'molB_m062x_def2-svp']

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_happy_path_summary(self):
        idx = make_up_index(self.tmp, self.tags)
        s = app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 2)
        self.assertEqual(s['cells'], 2)
        self.assertEqual(s['role_pins'], 8)
        self.assertEqual(s['tags'], self.tags)

    def test_refuses_missing_batch(self):
        idx = make_up_index(self.tmp, self.tags)
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'no-such-batch', 2)
        self.assertIn('not present', str(cm.exception))

    def test_refuses_incomplete_cell_count(self):
        idx = make_up_index(self.tmp, self.tags)
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 64)
        self.assertIn('NOT complete', str(cm.exception))

    def test_refuses_broken_pin_hash(self):
        idx = make_up_index(self.tmp, self.tags)
        victim = os.path.join(self.tmp, 'cells',
                              f'{self.tags[0]}_upstream.json')
        with open(victim, 'w', encoding='utf-8') as f:
            json.dump({'tampered': True}, f)
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 2)
        self.assertIn('hash mismatch', str(cm.exception))

    def test_refuses_missing_role_pin(self):
        idx = make_up_index(self.tmp, self.tags)
        with open(idx, encoding='utf-8') as f:
            d = json.load(f)
        del d['batches'][0]['cells'][0]['sources']['strict_fp64']
        with open(idx, 'w', encoding='utf-8') as f:
            json.dump(d, f)
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 2)
        self.assertIn('lacks a strict_fp64 pin', str(cm.exception))

    def test_refuses_unmerged_placeholder_cell(self):
        idx = make_up_index(self.tmp, self.tags)
        with open(idx, encoding='utf-8') as f:
            d = json.load(f)
        d['batches'][0]['cells'][1]['sources'] = {}
        with open(idx, 'w', encoding='utf-8') as f:
            json.dump(d, f)
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 2)
        self.assertIn('not merged yet', str(cm.exception))

    def test_refuses_pinned_file_deleted(self):
        idx = make_up_index(self.tmp, self.tags)
        os.remove(os.path.join(self.tmp, 'cells',
                               f'{self.tags[1]}_optimized_auto.json'))
        with self.assertRaises(SystemExit) as cm:
            app.verify_upstream_batch(idx, 'benchv1-20261030-obcr', 2)
        self.assertIn('pinned file missing', str(cm.exception))


class TestStatsChecks(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_happy_path_figures(self):
        p = make_stats(self.tmp)
        stats, figs = app.load_and_check_stats(p, 2)
        self.assertEqual(figs['O_to_B']['n'], 2)
        self.assertEqual(figs['B_to_C']['primary_geomean'], 1.5)

    def test_refuses_wrong_schema(self):
        p = make_stats(self.tmp, schema='some-other-schema')
        with self.assertRaises(SystemExit) as cm:
            app.load_and_check_stats(p, 2)
        self.assertIn('schema', str(cm.exception))

    def test_matching_denominator_passes_and_mismatch_refuses(self):
        p = make_stats(self.tmp, planned=64, passed=64)
        stats, _ = app.load_and_check_stats(p, 64)
        self.assertEqual(stats['counts']['planned'], 64)
        p2 = make_stats(self.tmp, planned=2)
        with self.assertRaises(SystemExit) as cm2:
            app.load_and_check_stats(p2, 64)
        self.assertIn('different cohort', str(cm2.exception))

    def test_missing_stats_file(self):
        with self.assertRaises(SystemExit) as cm:
            app.load_and_check_stats(
                os.path.join(self.tmp, 'absent.json'), 2)
        self.assertIn('not found', str(cm.exception))


class TestFullPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, 'ev')
        os.makedirs(self.root)
        with open(os.path.join(self.root, 'schedule-v2.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'schema': 'upstream-comparison-schedule-v2'}, f)
        # make_up_index writes cells/ under the batch-obcr subtree so
        # collect_sources() finds them at the real layout
        self.batch = os.path.join(self.root, 'batch-obcr')
        os.makedirs(self.batch)
        with open(os.path.join(self.batch, 'driver.log'), 'w',
                  encoding='utf-8') as f:
            f.write('[log]\n')
        self.tags = ['molA_b3lyp_def2-svp', 'molB_m062x_def2-svp']
        self.up = make_up_index(self.batch, self.tags)
        self.stats = make_stats(self.root, planned=2, passed=2)
        self.ds = make_ds_index(self.tmp)
        self.clean_proc = make_proc_root(self.tmp, ['/x/python',
                                                    '/y/idle.py'])
        self.daily = os.path.join(self.root,
                                  'benchv1-daily-comparisons.json')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _argv(self, extra=()):
        return ['--evidence-root', self.root,
                '--stats-json', self.stats,
                '--up-index', self.up,
                '--ds-index', self.ds,
                '--daily-out', self.daily,
                '--require-cells', '2',
                '--stats-tool-commit', 'testtoolsha'] + list(extra)

    def test_append_writes_daily_and_dataset(self):
        rc = app.main(self._argv(), proc_root=self.clean_proc)
        self.assertEqual(rc, 0)
        d = json.load(open(self.daily, encoding='utf-8'))
        self.assertEqual(d['schema'], app.DAILY_SCHEMA)
        self.assertEqual(d['counts']['planned'], 2)
        self.assertEqual(d['comparisons']['O_to_C']['n'], 2)
        self.assertEqual(d['arms_source_commit'], 'deadbeef')
        self.assertEqual(d['stats_tool_commit'], 'testtoolsha')
        self.assertIn('NOT new acceptance',
                      d['legacy_reference']['note'])
        self.assertIn('kept separate', d['auto_strict'])
        idx = json.load(open(self.ds, encoding='utf-8'))
        self.assertEqual(idx['datasets'][-1]['id'],
                         'benchv1-20261030-obcr-eval')
        entry = idx['datasets'][-1]
        # schedule + stats + daily (3) + stats.md + driver.log (2;
        # launch log + PHASE_RECORD absent in this fixture) +
        # 2 cells * (4 roles + summary) (10)
        self.assertEqual(len(entry['sources']), 15)
        for s in entry['sources']:
            self.assertEqual(sha(s['path']), s['sha256'])
        self.assertIn('planned 2, gate-pass 2', entry['label'])
        self.assertIn('deadbeef', entry['label'])
        # prior dataset untouched
        self.assertEqual(idx['datasets'][0]['id'], 'prior-dataset')

    def test_idempotent_second_append_refuses(self):
        app.main(self._argv(), proc_root=self.clean_proc)
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(), proc_root=self.clean_proc)
        self.assertIn('already present', str(cm.exception))

    def test_live_driver_refuses_real_append(self):
        proc = make_proc_root(
            self.tmp, ['/x/python', '/y/run_upstream_batch.py'])
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(), proc_root=proc)
        self.assertIn('batch driver alive', str(cm.exception))
        self.assertFalse(os.path.exists(self.daily))

    def test_dry_run_writes_nothing_and_does_not_guard(self):
        proc = make_proc_root(
            self.tmp, ['/x/python', '/y/run_upstream_batch.py'])
        rc = app.main(self._argv(['--dry-run']), proc_root=proc)
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.daily))
        idx = json.load(open(self.ds, encoding='utf-8'))
        self.assertEqual(len(idx['datasets']), 1)

    def test_incomplete_cohort_figures_honest(self):
        root2 = os.path.join(self.tmp, 'ev2')
        os.makedirs(os.path.join(root2, 'batch-obcr', 'cells'))
        with open(os.path.join(root2, 'schedule-v2.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'schema': 'upstream-comparison-schedule-v2'}, f)
        stats2 = make_stats(root2, planned=2, passed=0,
                            full_cohort=False)
        rc = app.main(['--evidence-root', root2,
                       '--stats-json', stats2,
                       '--up-index', self.up,
                       '--ds-index', self.ds,
                       '--require-cells', '2',
                       '--dataset-id', 'partial-x'],
                      proc_root=self.clean_proc)
        self.assertEqual(rc, 0)
        d = json.load(open(os.path.join(
            root2, 'benchv1-daily-comparisons.json'), encoding='utf-8'))
        self.assertFalse(d['full_cohort_available'])
        self.assertIsNone(
            d['comparisons']['O_to_B']['primary_geomean'])
        entry = json.load(open(self.ds, encoding='utf-8'))['datasets'][-1]
        self.assertIn('UNAVAILABLE', entry['label'])
        self.assertIn('undef', entry['label'])
        line = app.daily_line(d)
        self.assertIn('undefined (no complete pairs)', line)
        self.assertIn('UNAVAILABLE', line)

    def test_backup_created(self):
        app.main(self._argv(), proc_root=self.clean_proc)
        self.assertTrue(os.path.exists(
            self.ds + '.backup-pre-benchv1-20261030-obcr-eval'))


class TestSMode(unittest.TestCase):
    '''--mode s: the separate auto-strict S/B append (own stats schema,
    two-role pins, own daily schema and dataset id; never mixed into the
    O/B/C/R figures).'''

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.root = os.path.join(self.tmp, 'ev-s')
        os.makedirs(self.root)
        with open(os.path.join(self.root, 'schedule-s.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'schema_version':
                       'upstream-comparison-schedule-s-v1'}, f)
        self.batch = os.path.join(self.root, 'batch-s')
        os.makedirs(self.batch)
        with open(os.path.join(self.batch, 'driver.log'), 'w',
                  encoding='utf-8') as f:
            f.write('[s log]\n')
        self.tags = ['molA_b3lyp_def2-svp', 'molB_m062x_def2-svp']
        self.up = make_up_index_s(self.batch, self.tags)
        self.stats = make_stats_s(self.tmp, planned=2, certified=2)
        self.ds = make_ds_index(self.tmp)
        self.clean_proc = make_proc_root(self.tmp, ['/x/python',
                                                    '/y/idle.py'])
        self.daily = os.path.join(self.root,
                                  'benchv1-daily-comparisons-s.json')

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _argv(self, extra=()):
        # no --schedule/--batch-id/--dataset-id/--daily-out: the s-mode
        # defaults must resolve to schedule-s.json / -obcr-S / -s-eval /
        # benchv1-daily-comparisons-s.json on their own
        return ['--mode', 's',
                '--evidence-root', self.root,
                '--stats-json', self.stats,
                '--up-index', self.up,
                '--ds-index', self.ds,
                '--require-cells', '2',
                '--stats-tool-commit', 'testtoolsha'] + list(extra)

    def test_s_append_writes_daily_and_dataset(self):
        rc = app.main(self._argv(), proc_root=self.clean_proc)
        self.assertEqual(rc, 0)
        d = json.load(open(self.daily, encoding='utf-8'))
        self.assertEqual(d['schema'], app.S_DAILY_SCHEMA)
        self.assertEqual(d['counts'],
                         {'planned': 2, 'certified_ok': 2,
                          'failed_or_missing': 0})
        self.assertEqual(d['b_to_s']['n'], 2)
        self.assertEqual(d['b_to_s']['primary_geomean'], 0.98)
        self.assertEqual(d['arms_source_commit'], 's0urce')
        self.assertEqual(d['stats_tool_commit'], 'testtoolsha')
        self.assertIn('绝不借用', d['note'])
        # the S daily never carries O/B/C/R comparison keys
        self.assertNotIn('comparisons', d)
        self.assertNotIn('legacy_reference', d)
        idx = json.load(open(self.ds, encoding='utf-8'))
        entry = idx['datasets'][-1]
        self.assertEqual(entry['id'], 'benchv1-20261030-s-eval')
        # schedule-s + stats + daily (3) + stats.md + driver.log (2;
        # launch log + PHASE_RECORD absent in this fixture) +
        # 2 cells * (2 roles + summary) (6)
        self.assertEqual(len(entry['sources']), 11)
        for s in entry['sources']:
            self.assertEqual(sha(s['path']), s['sha256'])
        self.assertIn('certified_ok 2', entry['label'])
        self.assertIn('B_to_S geomean 0.9800', entry['label'])
        self.assertIn('never borrows C', entry['label'])
        # prior dataset untouched
        self.assertEqual(idx['datasets'][0]['id'], 'prior-dataset')

    def test_s_idempotent_second_append_refuses(self):
        app.main(self._argv(), proc_root=self.clean_proc)
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(), proc_root=self.clean_proc)
        self.assertIn('already present', str(cm.exception))

    def test_s_live_driver_refuses_real_append(self):
        proc = make_proc_root(
            self.tmp, ['/x/python', '/y/run_upstream_batch.py'])
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(), proc_root=proc)
        self.assertIn('batch driver alive', str(cm.exception))
        self.assertFalse(os.path.exists(self.daily))

    def test_s_dry_run_writes_nothing_and_does_not_guard(self):
        proc = make_proc_root(
            self.tmp, ['/x/python', '/y/run_upstream_batch.py'])
        rc = app.main(self._argv(['--dry-run']), proc_root=proc)
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(self.daily))
        idx = json.load(open(self.ds, encoding='utf-8'))
        self.assertEqual(len(idx['datasets']), 1)

    def test_schema_routing_refuses_cross_mode_documents(self):
        # s mode refuses an obcr stats document ...
        obcr_stats = make_stats(self.tmp, planned=2, passed=2)
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(['--stats-json', obcr_stats]),
                     proc_root=self.clean_proc)
        self.assertIn('stats schema', str(cm.exception))
        # ... and obcr mode refuses an s stats document (against a
        # proper obcr-mode fixture, so the stats check is what fires)
        root_obcr = os.path.join(self.tmp, 'ev-obcr')
        os.makedirs(root_obcr)
        with open(os.path.join(root_obcr, 'schedule-v2.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'schema': 'upstream-comparison-schedule-v2'}, f)
        up_obcr = make_up_index(os.path.join(root_obcr, 'batch-obcr'),
                                self.tags)
        with self.assertRaises(SystemExit) as cm:
            app.main(['--evidence-root', root_obcr,
                      '--stats-json', self.stats,
                      '--up-index', up_obcr,
                      '--ds-index', self.ds,
                      '--require-cells', '2',
                      '--dataset-id', 'x-obcr'],
                     proc_root=self.clean_proc)
        self.assertIn('stats schema', str(cm.exception))
        self.assertFalse(os.path.exists(
            os.path.join(root_obcr, 'benchv1-daily-comparisons.json')))

    def test_s_verify_requires_auto_strict_pin(self):
        up_b_only = make_up_index_s(self.batch, self.tags,
                                    roles=('optimized_fp64',))
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(['--up-index', up_b_only]),
                     proc_root=self.clean_proc)
        self.assertIn('auto_strict', str(cm.exception))

    def test_s_verify_refuses_obcr_four_role_batch_entry(self):
        # the obcr batch id must NOT be appended-from in s mode: its
        # cells carry no auto_strict pins
        up_obcr = make_up_index(os.path.join(self.tmp, 'batch-obcr'),
                                self.tags)
        with self.assertRaises(SystemExit) as cm:
            app.main(self._argv(['--up-index', up_obcr]),
                     proc_root=self.clean_proc)
        self.assertIn('benchv1-20261030-obcr-S', str(cm.exception))
        self.assertIn('not present', str(cm.exception))

    def test_s_no_certified_pairs_renders_undefined(self):
        stats0 = make_stats_s(self.tmp, planned=2, certified=0)
        self.daily = os.path.join(self.tmp, 'ev2',
                                  'benchv1-daily-comparisons-s.json')
        root2 = os.path.join(self.tmp, 'ev2')
        os.makedirs(root2)
        with open(os.path.join(root2, 'schedule-s.json'), 'w',
                  encoding='utf-8') as f:
            json.dump({'schema_version':
                     'upstream-comparison-schedule-s-v1'}, f)
        rc = app.main(['--mode', 's',
                       '--evidence-root', root2,
                       '--stats-json', stats0,
                       '--up-index', self.up,
                       '--ds-index', self.ds,
                       '--require-cells', '2',
                       '--dataset-id', 's-partial'],
                      proc_root=self.clean_proc)
        self.assertEqual(rc, 0)
        d = json.load(open(self.daily, encoding='utf-8'))
        self.assertEqual(d['counts']['certified_ok'], 0)
        self.assertIsNone(d['b_to_s']['primary_geomean'])
        entry = json.load(open(self.ds, encoding='utf-8'))['datasets'][-1]
        self.assertIn('undef', entry['label'])
        line = app.daily_line_s(d)
        self.assertIn('undefined (no certified pairs)', line)


if __name__ == '__main__':
    unittest.main()

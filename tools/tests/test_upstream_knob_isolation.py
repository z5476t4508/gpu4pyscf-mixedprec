#!/usr/bin/env python3
"""Tests for the post-batch knob-isolation driver (preregistration A4).

stdlib only; no GPU, no cupy import.  The /proc scan is exercised both
hermetically (mocked cmdlines) and once for real (a live sleeping
process whose argv carries the driver script name, the watchdog-test
pattern).  The arm-execution path itself (snapshot imports +
run_gpu_grad_cell) is NOT tested here -- it needs the GPU and is only
legal after batch uc-20261018-a completes; what is tested is everything
around it: plan validation, env discipline, planning, guards, and the
mechanical reproduce/attribution analysis.
"""

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ki = _load('upstream_knob_isolation')
uc = _load('upstream_comparison')            # stdlib-only top level

STACK_ENV = dict(uc.STACK_ENV)


def make_plan(**overrides):
    plan = {
        'amendment_id': 'A4-test',
        'batch_id': 'uc-test',
        'schedule_path': '/nonexistent-schedule.json',
        'snapshot_root': '/nonexistent-snapshot',
        'evidence_root': '/nonexistent-evidence',
        'cells': [{'tag': 'cell_a', 'index': 0}],
        'configs': {'R0': {}, 'FULL': dict(STACK_ENV),
                    'BLK': {'GPU4PYSCF_XC_GRID_BLOCKSIZE': '16384'},
                    'VMT': {'GPU4PYSCF_SCF_VMAT_F32': 'headtail'},
                    'GRAD3': {'GPU4PYSCF_DF_IP1_FUSED': '1'}},
        'blocks_per_config': 4,
        'reproduce_tolerance_ha': 1e-11,
        'attribution_rules': {'clean_ha': 1e-11, 'carry_fraction': 0.5},
    }
    plan.update(overrides)
    return plan


class PlanValidationTests(unittest.TestCase):

    def test_load_plan_rejects_missing_key(self):
        plan = make_plan()
        del plan['attribution_rules']
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            with open(p, 'w') as f:
                json.dump(plan, f)
            with self.assertRaisesRegex(RuntimeError, 'attribution_rules'):
                ki.load_plan(p)

    def test_load_plan_rejects_non_knob_variable(self):
        plan = make_plan(configs={'R0': {'CUDA_VISIBLE_DEVICES': '0'}})
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            with open(p, 'w') as f:
                json.dump(plan, f)
            with self.assertRaisesRegex(RuntimeError, 'non-knob'):
                ki.load_plan(p)

    def test_plan_full_config_equals_batch_stack(self):
        # the FULL config IS the measured batch stack (ties the plan to
        # upstream_comparison.STACK_ENV)
        self.assertEqual(make_plan()['configs']['FULL'], STACK_ENV)


class EnvDisciplineTests(unittest.TestCase):

    def test_arm_env_exact(self):
        config = {'GPU4PYSCF_SCF_VMAT_F32': 'headtail'}
        # clear=True: the discipline check must see EXACTLY the config
        # (the ambient environment may itself carry knob variables, and
        # a stray GPU4PYSCF_* outside the config must fail closed)
        with mock.patch.dict(os.environ,
                             {'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
                              'PYTHONPATH': '/somewhere',
                              'PATH': '/usr/bin'},
                             clear=True):
            seen = ki.check_env_discipline(config)
            env = ki.arm_env(config)
        self.assertEqual(seen, config)
        # cleaned subprocess env: exact knobs, no PYTHONPATH
        self.assertEqual({k: v for k, v in env.items()
                          if k.startswith('GPU4PYSCF_')}, config)
        self.assertNotIn('PYTHONPATH', env)

    def test_arm_env_extra_variable_fails_closed(self):
        with mock.patch.dict(os.environ,
                             {'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
                              'GPU4PYSCF_DF_IP1_FUSED': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'discipline'):
                ki.check_env_discipline(
                    {'GPU4PYSCF_SCF_VMAT_F32': 'headtail'})

    def test_arm_env_missing_variable_fails_closed(self):
        with mock.patch.dict(os.environ, {'GPU4PYSCF_DF_IP1_FUSED': '1'}):
            with self.assertRaisesRegex(RuntimeError, 'discipline'):
                ki.check_env_discipline(
                    {'GPU4PYSCF_SCF_VMAT_F32': 'headtail'})


class PlanningTests(unittest.TestCase):

    def test_positions_rotation_distinct(self):
        pos = ki.positions_for(['R0', 'FULL', 'BLK', 'VMT', 'GRAD3'], 4)
        for name in ('R0', 'FULL', 'BLK', 'VMT', 'GRAD3'):
            values = [pos[(name, b)] for b in range(4)]
            self.assertEqual(len(set(values)), 4, name)
            self.assertTrue(all(0 <= v < 5 for v in values), name)


class GuardTests(unittest.TestCase):

    def _fake_procs(self, entries):
        def fake_cmdline(pid):
            return entries.get(int(pid), [])
        return fake_cmdline

    def test_refuses_live_batch_driver(self):
        entries = self._fake_procs(
            {7: ['python', '/x/run_upstream_batch.py', '--schedule',
                 's']})
        with mock.patch.object(ki, '_proc_cmdline',
                               side_effect=entries), \
             mock.patch('os.listdir', return_value=['7', '99']):
            with self.assertRaisesRegex(RuntimeError, 'double-drive'):
                ki.refuse_if_any_driver()

    def test_refuses_live_fill_launcher(self):
        entries = self._fake_procs(
            {8: ['python', '/x/upstream_fill_launcher.py', '--wait-pid',
                 '1']})
        with mock.patch.object(ki, '_proc_cmdline',
                               side_effect=entries), \
             mock.patch('os.listdir', return_value=['8']):
            with self.assertRaisesRegex(RuntimeError, 'double-drive'):
                ki.refuse_if_any_driver()

    def test_no_refusal_when_no_driver(self):
        entries = self._fake_procs(
            {9: ['python', '-c', 'sleep 5 run_upstream_batch.py '
                                  'mentioned in a -c string']})
        with mock.patch.object(ki, '_proc_cmdline',
                               side_effect=entries), \
             mock.patch('os.listdir', return_value=['9']):
            # a -c string merely mentioning the script is not an
            # element-based driver argv
            ki.refuse_if_any_driver()

    def test_live_proc_scan_refuses_real_batch_driver(self):
        # real-process variant (the watchdog-test pattern): a live
        # sleeper whose argv element IS a run_upstream_batch.py path
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, 'run_upstream_batch.py')
            with open(script, 'w') as f:
                f.write('import time\n'
                        'import sys\n'
                        'sys.argv = sys.argv\n'
                        'time.sleep(30)\n')
            proc = subprocess.Popen(
                [sys.executable, script, '--schedule', 'x'])
            try:
                deadline = time.time() + 10
                while proc.poll() is None and time.time() < deadline:
                    try:
                        with open(f'/proc/{proc.pid}/cmdline', 'rb') as f:
                            ready = b'run_upstream_batch.py' in f.read()
                    except OSError:
                        ready = False
                    if ready:
                        break
                    time.sleep(0.1)
                self.assertTrue(ready)
                with self.assertRaisesRegex(RuntimeError,
                                            'double-drive'):
                    ki.refuse_if_any_driver()
            finally:
                proc.kill()
                proc.wait()

    def test_batch_complete_gate(self):
        def cell(n, filled):
            return {'cell': {'tag': f'c{n}'},
                    'sources': ({'upstream': {'p': 1},
                                 'optimized_fp64': {'p': 1},
                                 'optimized_auto': {'p': 1},
                                 'strict_fp64': {'p': 1}}
                                if filled else {})}
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'index.json')
            with open(p, 'w') as f:
                json.dump({'batches': [{'id': 'uc-test',
                                       'cells': [cell(0, True),
                                                 cell(1, True)]}]}, f)
            self.assertTrue(ki.batch_complete(p, 'uc-test'))
            with open(p, 'w') as f:
                json.dump({'batches': [{'id': 'uc-test',
                                       'cells': [cell(0, True),
                                                 cell(1, False)]}]}, f)
            self.assertFalse(ki.batch_complete(p, 'uc-test'))
            with open(p, 'w') as f:
                json.dump({'batches': [{'id': 'other'}]}, f)
            with self.assertRaisesRegex(RuntimeError, 'not found'):
                ki.batch_complete(p, 'uc-test')


    def test_snapshot_inventory_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            snap = os.path.join(tmp, 'snapshot')
            os.makedirs(os.path.join(snap, 'pkg'))
            with open(os.path.join(snap, '.upstream-batch-source-commit'),
                      'w') as f:
                f.write('c0ffee\n')
            for rel, body in (('pkg/a.py', 'AAA'), ('pkg/b.so', 'BBB'),
                              ('run.py', 'RRR')):
                with open(os.path.join(snap, rel), 'w') as f:
                    f.write(body)
            inventory = {'source_commit': 'c0ffee', 'source_root': 'x',
                         'files': {
                             '.upstream-batch-source-commit':
                                 ki.sha256_file(os.path.join(
                                     snap,
                                     '.upstream-batch-source-commit')),
                                 'pkg/a.py': ki.sha256_file(
                                     os.path.join(snap, 'pkg/a.py')),
                                 'pkg/b.so': ki.sha256_file(
                                     os.path.join(snap, 'pkg/b.so')),
                                 'run.py': ki.sha256_file(
                                     os.path.join(snap, 'run.py'))}}
            with open(snap + '.inventory.json', 'w') as f:
                json.dump(inventory, f)
            self.assertEqual(
                ki.verify_snapshot_inventory(snap, 'c0ffee'), 4)
            # drift fails closed
            with open(os.path.join(snap, 'pkg/a.py'), 'w') as f:
                f.write('CHANGED')
            with self.assertRaisesRegex(RuntimeError, 'drift'):
                ki.verify_snapshot_inventory(snap, 'c0ffee')
            # commit mismatch fails closed
            with self.assertRaisesRegex(RuntimeError, 'mix versions'):
                ki.verify_snapshot_inventory(snap, 'deadbee')
            # extra file fails closed
            with open(os.path.join(snap, 'pkg/a.py'), 'w') as f:
                f.write('AAA')
            with open(os.path.join(snap, 'pkg/extra.py'), 'w') as f:
                f.write('NEW')
            with self.assertRaisesRegex(RuntimeError, 'drift'):
                ki.verify_snapshot_inventory(snap, 'c0ffee')


class AnalysisTests(unittest.TestCase):

    def test_reproduce_gate(self):
        ref = [-100.0, -100.0, -100.0, -100.0]
        self.assertEqual(ki.reproduce_gate(
            [-100.0 + 5e-13] * 4, ref, 1e-11), [True] * 4)
        self.assertEqual(ki.reproduce_gate(
            [-100.0 + 5e-11, -100.0, -100.0, -100.0], ref, 1e-11),
            [False, True, True, True])

    def test_attribution_single_knob_cause(self):
        rules = {'clean_ha': 1e-11, 'carry_fraction': 0.5}
        out = ki.attribution(
            {'BLK': -1.8e-6, 'VMT': -3e-12, 'GRAD3': 0.0}, -1.93e-6, rules)
        self.assertEqual(out['overall'], 'single-knob-cause')
        self.assertEqual(out['carrying'], ['BLK'])
        self.assertEqual(out['configs']['BLK']['label'], 'carries')
        self.assertEqual(out['configs']['VMT']['label'], 'clean')
        self.assertAlmostEqual(out['configs']['BLK']['share_of_full'],
                               -1.8e-6 / -1.93e-6)

    def test_attribution_pair_follow_up(self):
        rules = {'clean_ha': 1e-11, 'carry_fraction': 0.5}
        out = ki.attribution(
            {'BLK': -1.0e-6, 'VMT': -1.0e-6, 'GRAD3': 1e-13}, -1.93e-6,
            rules)
        self.assertEqual(out['overall'], 'pair-or-interaction-follow-up')
        self.assertEqual(sorted(out['carrying']), ['BLK', 'VMT'])
        # partial regime: 0.466 share is below the carry fraction but
        # above clean -- also routes to the follow-up branch
        out2 = ki.attribution(
            {'BLK': -1.0e-6, 'VMT': -0.9e-6, 'GRAD3': 1e-13}, -1.93e-6,
            rules)
        self.assertEqual(out2['overall'],
                         'pair-or-interaction-follow-up')
        self.assertEqual(out2['carrying'], ['BLK'])
        self.assertEqual(out2['partial'], ['VMT'])

    def test_attribution_all_clean_interaction(self):
        rules = {'clean_ha': 1e-11, 'carry_fraction': 0.5}
        out = ki.attribution(
            {'BLK': 3e-12, 'VMT': -4e-12, 'GRAD3': 0.0}, -1.93e-6, rules)
        self.assertEqual(out['overall'],
                         'no-single-knob-shift (interaction only, or null)')
        self.assertEqual(out['carrying'], [])


class AnalyzeEndToEndTests(unittest.TestCase):
    '''The mechanical report on synthetic samples: R0/FULL reproduce the
    plan's batch references, one single knob carries, the rest stay
    clean; the classification follows, and nothing else is claimed.'''

    def _write_samples(self, tmp, plan, energies):
        samples = os.path.join(tmp, 'samples')
        os.makedirs(samples, exist_ok=True)
        for (tag, config), es in energies.items():
            for block, e in enumerate(es):
                rec = {'schema_version': ki.SAMPLE_SCHEMA,
                       'amendment_id': plan['amendment_id'],
                       'config': config, 'block': block, 'cell_tag': tag,
                       'row': {'result': {'e_tot': e}}}
                with open(os.path.join(
                        samples, f'{tag}_{config}_b{block}.json'),
                        'w') as f:
                    json.dump(rec, f)

    def test_analyze_full_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_r = [-630.7161527941] * 4
            ref_b = [-630.7161547262] * 4
            plan = make_plan(
                evidence_root=tmp,
                cells=[{'tag': 'cell_a', 'index': 0,
                        'configs_scheduled':
                            ['R0', 'FULL', 'BLK', 'VMT', 'GRAD3'],
                        'batch_reference_energies': {
                            'strict_fp64_R': ref_r,
                            'optimized_fp64_B': ref_b}}],
                schedule_path=os.path.join(tmp, 'sched.json'))
            with open(plan['schedule_path'], 'w') as f:
                json.dump({'settings': {}, 'source_commit': 'x',
                           'cells': [{'tag': 'cell_a', 'index': 0}]}, f)
            # R0 reproduces R; FULL reproduces B; BLK carries 0.93 of the
            # shift; VMT/GRAD3 clean
            energies = {
                ('cell_a', 'R0'): [e + 3e-13 for e in ref_r],
                ('cell_a', 'FULL'): [e + 2e-13 for e in ref_b],
                ('cell_a', 'BLK'): [-630.7161527941 - 1.8e-6] * 4,
                ('cell_a', 'VMT'): [-630.7161527941 - 2e-12] * 4,
                ('cell_a', 'GRAD3'): [-630.7161527941] * 4}
            self._write_samples(tmp, plan, energies)
            report = ki.analyze(plan, os.path.join(tmp, 'report.json'),
                                os.path.join(tmp, 'report.md'))
            self.assertTrue(report['reproduce_all_ok'])
            att = report['cells']['cell_a']['attribution']
            self.assertEqual(att['overall'], 'single-knob-cause')
            self.assertEqual(att['carrying'], ['BLK'])
            self.assertTrue(os.path.isfile(os.path.join(tmp, 'report.md')))
            with open(os.path.join(tmp, 'report.md')) as f:
                body = f.read()
            self.assertIn('single-knob-cause', body)
            self.assertIn('reproduce', body)

    def test_analyze_stops_at_reproduce_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ref_r = [-630.7161527941] * 4
            plan = make_plan(
                evidence_root=tmp,
                cells=[{'tag': 'cell_a', 'index': 0,
                        'batch_reference_energies': {
                            'strict_fp64_R': ref_r,
                            'optimized_fp64_B': ref_r}}],
                schedule_path=os.path.join(tmp, 'sched.json'))
            with open(plan['schedule_path'], 'w') as f:
                json.dump({'settings': {}, 'source_commit': 'x',
                           'cells': [{'tag': 'cell_a', 'index': 0}]}, f)
            # R0 does NOT reproduce (5e-12 > 1e-11? no: 5e-12 < 1e-11 --
            # use 5e-11 to force the failure)
            energies = {('cell_a', 'R0'): [e - 5e-11 for e in ref_r],
                        ('cell_a', 'FULL'): list(ref_r)}
            self._write_samples(tmp, plan, energies)
            report = ki.analyze(plan, os.path.join(tmp, 'report.json'))
            self.assertFalse(report['reproduce_all_ok'])
            # no attribution is computed from non-reproducing anchors
            self.assertNotIn('attribution', report['cells']['cell_a'])


if __name__ == '__main__':
    unittest.main()

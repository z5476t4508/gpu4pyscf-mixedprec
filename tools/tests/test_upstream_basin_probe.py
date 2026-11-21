#!/usr/bin/env python3
"""Tests for the LC-206 basin-probe driver (methoxy dm0-transfer arms).

stdlib only; no GPU, no cupy import.  What is tested is everything
around the arm-execution path: plan validation (dm dependency graph,
env names, anchor structure), env discipline, basin classification,
anchor comparison, record matching, and the driver's seeded ordering.
The arm path itself (snapshot imports + _basin_point) needs the GPU
and is only legal post-batch.
"""

import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bp = _load('upstream_basin_probe')

STACK_ENV = {
    'GPU4PYSCF_XC_GRID_BLOCKSIZE': '16384',
    'GPU4PYSCF_VV10_UWE_KERNEL': 'fp32',
    'GPU4PYSCF_DF_IP1_FUSED': '1',
    'GPU4PYSCF_DF_IP1_F32': '1',
    'GPU4PYSCF_GRAD_GRID_F32': '1',
    'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
}


def make_plan(**overrides):
    plan = {
        'unit': 'LC-206-test',
        'amendment_id': 'lc206-test',
        'batch_id': 'uc-test',
        'index_path': '/nonexistent-index.json',
        'schedule_path': '/nonexistent-schedule.json',
        'schedule_sha256': '0' * 64,
        'snapshot_root': '/nonexistent-snapshot',
        'snapshot_commit': 'c' * 40,
        'evidence_dir': '/nonexistent-evidence',
        'cells': [{'tag': 'cell_a', 'index': 0}],
        'arms': {
            'F64': {'scf_mode': 'fp64', 'grad_mode': 'fp64', 'env': 'R0',
                    'stage': 0, 'dm_out': True},
            'AUTO': {'scf_mode': 'auto', 'grad_mode': 'auto', 'env': 'R0',
                     'stage': 0, 'dm_out': True},
            'CAUTO': {'scf_mode': 'auto', 'grad_mode': 'auto',
                      'env': 'STACK', 'stage': 0, 'dm_out': True},
            'BF64': {'scf_mode': 'fp64', 'grad_mode': 'fp64',
                     'env': 'STACK', 'stage': 0, 'dm_out': True},
            'F64_FROM_AUTO': {'scf_mode': 'fp64', 'grad_mode': 'fp64',
                              'env': 'R0', 'stage': 1, 'dm_in': 'AUTO'},
            'F64_FROM_CAUTO': {'scf_mode': 'fp64', 'grad_mode': 'fp64',
                               'env': 'R0', 'stage': 1, 'dm_in': 'CAUTO'},
            'AUTO_FROM_F64': {'scf_mode': 'auto', 'grad_mode': 'auto',
                              'env': 'R0', 'stage': 1, 'dm_in': 'F64'},
            'CAUTO_FROM_F64': {'scf_mode': 'auto', 'grad_mode': 'auto',
                               'env': 'STACK', 'stage': 1,
                               'dm_in': 'F64'},
            'F64_FROM_F64': {'scf_mode': 'fp64', 'grad_mode': 'fp64',
                             'env': 'R0', 'stage': 1, 'dm_in': 'F64',
                             'control': True},
            'AUTO_FROM_AUTO': {'scf_mode': 'auto', 'grad_mode': 'auto',
                               'env': 'R0', 'stage': 1, 'dm_in': 'AUTO',
                               'control': True},
        },
        'blocks': 3,
        'seed': 20261026,
        'timeout_s': 1200,
        'max_retries': 3,
        'anchor_tolerance_e': 1e-12,
        'anchor_tolerance_dg': 1e-7,
        'basin_tolerance_e': 1e-10,
        'control_tolerance_e': 1e-11,
        'stack_env': dict(STACK_ENV),
        'anchors': {
            'batch_R': {'per_cell': {
                'cell_a': {'path': '/nonexistent-R.json',
                           'sha256': 'a' * 64,
                           'expected_median_e': -114.9016950130196}}},
            'a5_R0_AUTO': {'per_cell': {
                'cell_a': {'path': '/nonexistent-A.json',
                           'sha256': 'b' * 64,
                           'expected_median_e': -114.9016962956088}}},
            'batch_C': {'per_cell': {
                'cell_a': {'path': '/nonexistent-C.json',
                           'sha256': 'c' * 64,
                           'expected_median_e': -114.9016945880608}}},
            'batch_B': {'per_cell': {
                'cell_a': {'path': '/nonexistent-B.json',
                           'sha256': 'd' * 64,
                           'expected_median_e': -114.90169501}}},
        },
    }
    plan.update(overrides)
    return plan


def write_plan(tmpdir, plan):
    p = os.path.join(tmpdir, 'plan.json')
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(plan, f)
    return p


class TestPlanValidation(unittest.TestCase):

    def test_valid_plan_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_plan(tmp, make_plan())
            loaded = bp.load_plan(p)
            self.assertEqual(loaded['amendment_id'], 'lc206-test')

    def test_missing_required_key_refused(self):
        for key in ('arms', 'anchors', 'snapshot_root', 'blocks'):
            plan = make_plan()
            del plan[key]
            with tempfile.TemporaryDirectory() as tmp:
                p = write_plan(tmp, plan)
                with self.assertRaisesRegex(
                        RuntimeError, f"missing required key '{key}'"):
                    bp.load_plan(p)

    def test_stage0_with_dm_in_refused(self):
        plan = make_plan()
        plan['arms']['F64']['dm_in'] = 'AUTO'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'cannot carry dm_in'):
                bp.load_plan(write_plan(tmp, plan))

    def test_stage0_without_dm_out_refused(self):
        plan = make_plan()
        del plan['arms']['AUTO']['dm_out']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'must set dm_out'):
                bp.load_plan(write_plan(tmp, plan))

    def test_dm_in_must_reference_producer(self):
        # a stage-1 arm seeded by another stage-1 (consumer) arm is a
        # cross-stage transfer: refused
        plan = make_plan()
        plan['arms']['BAD'] = {'scf_mode': 'fp64', 'grad_mode': 'fp64',
                               'env': 'R0', 'stage': 1,
                               'dm_in': 'F64_FROM_AUTO'}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'not a stage-0'):
                bp.load_plan(write_plan(tmp, plan))

    def test_stage1_without_dm_in_refused(self):
        plan = make_plan()
        del plan['arms']['F64_FROM_AUTO']['dm_in']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'must carry dm_in'):
                bp.load_plan(write_plan(tmp, plan))

    def test_bad_env_refused(self):
        plan = make_plan()
        plan['arms']['F64']['env'] = 'MAGIC'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'env must be'):
                bp.load_plan(write_plan(tmp, plan))

    def test_bad_sc_mode_refused(self):
        plan = make_plan()
        plan['arms']['F64']['scf_mode'] = 'fp16'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'scf_mode'):
                bp.load_plan(write_plan(tmp, plan))

    def test_no_stage0_refused(self):
        plan = make_plan()
        for name in ('F64', 'AUTO', 'CAUTO', 'BF64'):
            del plan['arms'][name]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'no stage-0'):
                bp.load_plan(write_plan(tmp, plan))

    def test_stack_env_must_be_knobs(self):
        plan = make_plan(stack_env={'CUDA_VISIBLE_DEVICES': '0'})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'non-knob'):
                bp.load_plan(write_plan(tmp, plan))

    def test_anchor_missing_cell_refused(self):
        plan = make_plan()
        plan['cells'].append({'tag': 'cell_b', 'index': 1})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError,
                                        "missing per_cell entry"):
                bp.load_plan(write_plan(tmp, plan))

    def test_anchor_missing_field_refused(self):
        plan = make_plan()
        del plan['anchors']['batch_R']['per_cell']['cell_a']['sha256']
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "missing 'sha256'"):
                bp.load_plan(write_plan(tmp, plan))

    def test_anchor_non_numeric_e_refused(self):
        plan = make_plan()
        plan['anchors']['batch_R']['per_cell']['cell_a'][
            'expected_median_e'] = 'not-a-number'
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'non-numeric'):
                bp.load_plan(write_plan(tmp, plan))


class TestEnvDiscipline(unittest.TestCase):

    def test_arm_env_r0_is_clean(self):
        plan = make_plan()
        with mock.patch.dict(os.environ,
                             {'GPU4PYSCF_XC_GRID_BLOCKSIZE': '999',
                              'PYTHONPATH': '/somewhere'},
                             clear=False):
            env = bp.arm_env(plan, 'F64')
        self.assertNotIn('GPU4PYSCF_XC_GRID_BLOCKSIZE', env)
        self.assertNotIn('PYTHONPATH', env)
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'], '1')

    def test_arm_env_stack_is_exact(self):
        plan = make_plan()
        with mock.patch.dict(os.environ,
                             {'GPU4PYSCF_XC_GRID_BLOCKSIZE': '999'},
                             clear=False):
            env = bp.arm_env(plan, 'CAUTO')
        for k, v in STACK_ENV.items():
            self.assertEqual(env[k], v)
        leaked = {k: v for k, v in env.items()
                  if k.startswith('GPU4PYSCF_')} == STACK_ENV
        self.assertTrue(leaked)

    def test_check_env_discipline_r0(self):
        plan = make_plan()
        with mock.patch.dict(os.environ, {'GPU4PYSCF_DF_IP1_F32': '1'},
                             clear=False):
            with self.assertRaisesRegex(RuntimeError, 'discipline'):
                bp.check_env_discipline(plan, 'F64')
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
            seen = bp.check_env_discipline(plan, 'F64')
        self.assertEqual(seen, {})

    def test_check_env_discipline_stack(self):
        plan = make_plan()
        os_env = dict(STACK_ENV)
        os_env['GPU4PYSCF_SCF_VMAT_F32'] = 'full'      # wrong value
        with mock.patch.dict(os.environ, os_env, clear=False):
            with self.assertRaisesRegex(RuntimeError, 'discipline'):
                bp.check_env_discipline(plan, 'CAUTO')
        with mock.patch.dict(os.environ, dict(STACK_ENV), clear=False):
            seen = bp.check_env_discipline(plan, 'CAUTO')
        self.assertEqual(seen, dict(STACK_ENV))


class TestBasinClassification(unittest.TestCase):

    def test_stay(self):
        cls, d_src, d_def = bp.classify_basin(-114.9016962956,
                                              -114.9016962956,
                                              -114.9016950130, 1e-10)
        self.assertEqual(cls, 'STAY')
        self.assertEqual(d_src, 0.0)

    def test_switch(self):
        cls, d_src, d_def = bp.classify_basin(-114.9016950130,
                                              -114.9016962956,
                                              -114.9016950130, 1e-10)
        self.assertEqual(cls, 'SWITCH')
        self.assertEqual(d_def, 0.0)

    def test_other(self):
        cls, d_src, d_def = bp.classify_basin(-114.9016, -114.9016962956,
                                              -114.9016950130, 1e-10)
        self.assertEqual(cls, 'OTHER')
        self.assertGreater(d_src, 1e-10)
        self.assertGreater(d_def, 1e-10)

    def test_degenerate_when_basins_within_tol(self):
        # source and lane-default basins are DISTINCT references but sit
        # within tolerance of each other: the STAY/SWITCH distinction
        # is not resolvable and the classification is degenerate
        cls, d_src, d_def = bp.classify_basin(
            -114.9016950130, -114.9016950130,
            -114.9016950130 + 5e-11, 1e-10)
        self.assertEqual(cls, 'DEGENERATE')
        self.assertLessEqual(d_src, 1e-10)
        self.assertLessEqual(d_def, 1e-10)


class TestAnalyze(unittest.TestCase):

    @staticmethod
    def _sample(tag, arm, block, e, cycles=11, grad=None):
        if grad is None:
            grad = [[0.01, 0.02, 0.03]] * 5
        return {
            'schema_version': 'basin-probe-sample-v1',
            'amendment_id': 'lc206-test',
            'batch_id': 'uc-test',
            'cell_tag': tag,
            'arm': arm,
            'block': block,
            'row': {'result': {'e_tot': e, 'converged': True,
                                'scf_cycles': cycles,
                                'gradient': grad}},
        }

    def _write_samples(self, tmpdir, tag, samples):
        sdir = os.path.join(tmpdir, 'samples')
        os.makedirs(sdir, exist_ok=True)
        for s in samples:
            p = os.path.join(sdir, f'{tag}_{s["arm"]}_b{s["block"]}.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(s, f)

    def _anchor_file(self, tmpdir, name, e, grad=None):
        p = os.path.join(tmpdir, f'{name}.json')
        doc = {'rows': [{'result': {'e_tot': e, 'gradient': grad
                                    or [[0.01, 0.02, 0.03]] * 5}}]}
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        return p

    def test_analyze_full_cycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            e_R, e_A, e_C = -114.9016950130, -114.9016962956, \
                -114.9016945881
            e_B = -114.9016950
            for anchor, e in (('batch_R', e_R), ('a5_R0_AUTO', e_A),
                              ('batch_C', e_C), ('batch_B', e_B)):
                path = self._anchor_file(tmp, anchor, e)
                entry = plan['anchors'][anchor]['per_cell']['cell_a']
                entry['path'] = path
                entry['sha256'] = bp.sha256_file(path)
                entry['expected_median_e'] = e
            plan['evidence_dir'] = tmp
            tag = 'cell_a'
            samples = []
            for block in range(3):
                samples.append(self._sample(tag, 'F64', block, e_R, 11))
                samples.append(self._sample(tag, 'AUTO', block, e_A, 19))
                samples.append(self._sample(tag, 'CAUTO', block, e_C, 11))
                samples.append(self._sample(tag, 'BF64', block, e_B, 11))
                # marquee: fp64 SCF from the auto density STAYS at e_A
                samples.append(self._sample(tag, 'F64_FROM_AUTO', block,
                                            e_A, 3))
                # the batch-C basin also stays
                samples.append(self._sample(tag, 'F64_FROM_CAUTO', block,
                                            e_C, 3))
                # auto-from-R stays at e_R
                samples.append(self._sample(tag, 'AUTO_FROM_F64', block,
                                            e_R, 2))
                samples.append(self._sample(tag, 'CAUTO_FROM_F64', block,
                                            e_R, 2))
                samples.append(self._sample(tag, 'F64_FROM_F64', block,
                                            e_R, 1))
                samples.append(self._sample(tag, 'AUTO_FROM_AUTO', block,
                                            e_A, 2))
            self._write_samples(tmp, tag, samples)
            out_json = os.path.join(tmp, 'report.json')
            out_md = os.path.join(tmp, 'report.md')
            report = bp.analyze(plan, out_json, out_md)
            cell = report['cells'][tag]
            self.assertEqual(report['gaps'], [])
            for anchor in ('batch_R', 'a5_R0_AUTO', 'batch_C'):
                ok = cell['anchors'][anchor]
                self.assertTrue(ok['sha256_ok'])
                self.assertTrue(ok['e_within_tolerance'])
            arms = cell['arms']
            self.assertEqual(arms['F64_FROM_AUTO']['classification'],
                             'STAY')
            self.assertEqual(arms['F64_FROM_CAUTO']['classification'],
                             'STAY')
            self.assertEqual(arms['AUTO_FROM_F64']['classification'],
                             'STAY')
            self.assertEqual(arms['CAUTO_FROM_F64']['classification'],
                             'STAY')
            self.assertEqual(arms['F64_FROM_F64']['classification'],
                             'STAY')
            self.assertEqual(arms['AUTO_FROM_AUTO']['classification'],
                             'STAY')
            self.assertEqual(arms['F64_FROM_AUTO']['dm_in'], 'AUTO')
            self.assertEqual(
                arms['F64_FROM_AUTO']['lane_default_basin'], 'F64')
            self.assertEqual(
                arms['CAUTO_FROM_F64']['lane_default_basin'], 'CAUTO')
            self.assertEqual(
                arms['AUTO_FROM_F64']['lane_default_basin'], 'AUTO')
            # LC-207 BF64 producer: batch_B anchor with the dg check
            ok_b = cell['anchors']['batch_B']
            self.assertTrue(ok_b['sha256_ok'])
            self.assertTrue(ok_b['e_within_tolerance'])
            self.assertIn('dg_vs_batch_B', ok_b)
            self.assertTrue(ok_b['dg_within_tolerance'])
            self.assertEqual(cell['producers']['BF64']['median_e'], e_B)
            with open(out_md, encoding='utf-8') as f:
                md = f.read()
            self.assertIn('F64_FROM_AUTO', md)
            self.assertIn('STAY', md)
            self.assertIn('batch_B', md)
            self.assertIn('dg=', md)

    def test_analyze_relax_records_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            e_R, e_A, e_C = -114.9016950130, -114.9016962956, \
                -114.9016945881
            e_B = -114.9016950
            for anchor, e in (('batch_R', e_R), ('a5_R0_AUTO', e_A),
                              ('batch_C', e_C), ('batch_B', e_B)):
                path = self._anchor_file(tmp, anchor, e)
                entry = plan['anchors'][anchor]['per_cell']['cell_a']
                entry['path'] = path
                entry['sha256'] = bp.sha256_file(path)
                entry['expected_median_e'] = e
            plan['evidence_dir'] = tmp
            tag = 'cell_a'
            samples = []
            for block in range(3):
                samples.append(self._sample(tag, 'F64', block, e_R, 11))
                samples.append(self._sample(tag, 'AUTO', block, e_A, 19))
                samples.append(self._sample(tag, 'CAUTO', block, e_C, 11))
                samples.append(self._sample(tag, 'BF64', block, e_B, 11))
                # the auto basin RELAXES under fp64 SCF back to e_R
                samples.append(self._sample(tag, 'F64_FROM_AUTO', block,
                                            e_R, 9))
                for arm in ('F64_FROM_CAUTO', 'AUTO_FROM_F64',
                            'CAUTO_FROM_F64', 'F64_FROM_F64',
                            'AUTO_FROM_AUTO'):
                    samples.append(self._sample(tag, arm, block, e_R, 3))
            self._write_samples(tmp, tag, samples)
            report = bp.analyze(plan, os.path.join(tmp, 'r.json'))
            arms = report['cells'][tag]['arms']
            self.assertEqual(arms['F64_FROM_AUTO']['classification'],
                             'SWITCH')

    def test_analyze_gap_is_honest(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            plan['evidence_dir'] = tmp
            tag = 'cell_a'
            # only the F64 producer ran; everything else is a gap
            self._write_samples(
                tmp, tag, [self._sample(tag, 'F64', 0, -114.9)])
            report = bp.analyze(plan, os.path.join(tmp, 'r.json'))
            self.assertIn('cell_a:AUTO', report['gaps'])
            self.assertIn('cell_a:F64_FROM_AUTO', report['gaps'])
            arms = report['cells'][tag]['arms']
            self.assertEqual(arms['F64_FROM_AUTO'].get('note'),
                             'arm missing (honest gap)')

    def test_analyze_anchor_sha_mismatch_refused_equality(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan()
            e_R = -114.9016950130
            path = self._anchor_file(tmp, 'batch_R', e_R)
            entry = plan['anchors']['batch_R']['per_cell']['cell_a']
            entry['path'] = path
            entry['sha256'] = 'f' * 64          # deliberately wrong
            entry['expected_median_e'] = e_R
            plan['evidence_dir'] = tmp
            tag = 'cell_a'
            samples = [self._sample(tag, 'F64', b, e_R)
                       for b in range(3)]
            self._write_samples(tmp, tag, samples)
            report = bp.analyze(plan, os.path.join(tmp, 'r.json'))
            ok = report['cells'][tag]['anchors']['batch_R']
            self.assertFalse(ok['sha256_ok'])


class TestRecordAndSchedule(unittest.TestCase):

    def test_record_matches(self):
        plan = make_plan()
        cell = plan['cells'][0]
        good = {'schema_version': 'basin-probe-sample-v1',
                'arm': 'F64', 'block': 1, 'cell_tag': 'cell_a',
                'amendment_id': 'lc206-test', 'row': {'result': {}}}
        self.assertTrue(bp.record_matches(good, cell, 'F64', 1, plan))
        bad = dict(good, amendment_id='other')
        self.assertFalse(bp.record_matches(bad, cell, 'F64', 1, plan))
        bad = dict(good, block=2)
        self.assertFalse(bp.record_matches(bad, cell, 'F64', 1, plan))
        bad = dict(good, arm='AUTO')
        self.assertFalse(bp.record_matches(bad, cell, 'F64', 1, plan))
        self.assertTrue(bp.record_matches(bad, cell, 'AUTO', 1, plan))

    def test_schedule_cell_identity(self):
        cell = {'tag': 'cell_a', 'index': 3, 'basis': 'def2-svp'}
        sch = {'tag': 'cell_a', 'index': 3, 'basis': 'def2-svp'}
        bp.assert_cell_matches_schedule(cell, sch)
        with self.assertRaisesRegex(RuntimeError, 'does not match'):
            bp.assert_cell_matches_schedule(cell, dict(sch, index=4))
        with self.assertRaisesRegex(RuntimeError, 'disagrees'):
            bp.assert_cell_matches_schedule(
                cell, dict(sch, basis='def2-tzvp'))

    def test_refuse_if_any_driver_live_process(self):
        # a live process whose argv carries a driver script name must be
        # refused (hermetic /proc scan via mock)
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, '-c',
             'import time; time.sleep(30)'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            with mock.patch.object(
                    bp, '_proc_cmdline',
                    side_effect=lambda pid: (
                        ['python', '/x/upstream_knob_isolation.py']
                        if pid == str(proc.pid) else [])):
                with self.assertRaisesRegex(RuntimeError,
                                            'refusing to double-drive'):
                    bp.refuse_if_any_driver()
        finally:
            proc.kill()
            proc.wait()

    def test_relative_driver_log_resolves_inside_evidence_dir(self):
        # regression: a bare relative --driver-log used to crash with
        # makedirs('') -- it must resolve inside the plan's evidence
        # dir; the driver then fails closed on the nonexistent batch
        # index, but only AFTER the log file was created and the
        # start line written
        with tempfile.TemporaryDirectory() as tmp:
            plan = make_plan(evidence_dir=tmp)
            p = write_plan(tmp, plan)
            with self.assertRaises((RuntimeError, FileNotFoundError,
                                    OSError)):
                bp.main(['--plan', p, '--driver',
                         '--driver-log', 'probe-driver.log'])
            log_path = os.path.join(plan['evidence_dir'],
                                    'probe-driver.log')
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, encoding='utf-8') as f:
                self.assertIn('driver start', f.read())

    def test_flatten_and_cos(self):
        g = bp._flatten_gradient([[0.1, 0.2], [0.3]])
        self.assertEqual(g, [0.1, 0.2, 0.3])
        cos = bp._grad_cos([1.0, 0.0], [2.0, 0.0])
        self.assertAlmostEqual(cos, 1.0)
        cos = bp._grad_cos([1.0, 0.0], [0.0, 3.0])
        self.assertAlmostEqual(cos, 0.0)
        self.assertIsNone(bp._grad_cos([0.0, 0.0], [1.0, 1.0]))


if __name__ == '__main__':
    unittest.main()

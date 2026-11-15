#!/usr/bin/env python3
"""Tests for the LC-210 ladder-point classification tool.

stdlib only; no numpy, no pyscf, no GPU.  Tested: the ladder plan
validation layer (keys, descent targets, guess whitelist, checkpoint
existence, fail-closed family coverage, no-spectra_points contract),
the endpoint-reproduction record, the spectrum classification logic
(minimum / saddle index), the below-reference flag, the reachability
summary aggregation, and the MD rendering.  The numeric paths
(numpy/pyscf) are exercised by the actual CPU run, like the
LC-207/208/209 tools.
"""

import importlib.util
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


lc_ = _load('upstream_ladder_classify')


def make_plan(**overrides):
    plan = {
        'unit': 'LC-210-test',
        'amendment_id': 'lc210-test',
        'batch_id': 'uc-test',
        'evidence_dir': '/nonexistent',
        'cell': {'tag': 'methoxy_wb97mv_def2-svp', 'index': 175,
                 'geom': '/nonexistent/methoxy.xyz', 'basis': 'def2-svp',
                 'xc': 'wb97mv', 'charge': 0, 'spin': 1},
        'settings': {'auxbasis': 'def2-universal-jkfit',
                     'grids_level': 3, 'nlcgrids_level': 3,
                     'conv_tol': 1e-10, 'direct_scf_tol': 1e-14,
                     'max_cycle': 50, 'init_guess': 'minao'},
        'family_dms': {
            'R': {'path': '/nonexistent-R.npz',
                  'anchor_e': -114.90169501301963,
                  'anchor_source': 'batch_R'},
            'R0_AUTO': {'path': '/nonexistent-A.npz',
                        'anchor_e': -114.90169629560877,
                        'anchor_source': 'a5_R0_AUTO'},
            'C': {'path': '/nonexistent-C.npz',
                  'anchor_e': -114.90169458806088,
                  'anchor_source': 'batch_C'},
            'B': {'path': '/nonexistent-B.npz',
                  'anchor_e': -114.90169501301972,
                  'anchor_source': 'batch_B'},
        },
        'descent': {
            'point': 'R', 'sigma': 1,
            'hessian_ckpt': '/nonexistent-ckpt-R.npz',
            'targets': [
                {'label': 'ladder_t03', 't': 0.30,
                 'anchor_e_final': -114.901697105879,
                 'anchor_source': 'lc209'},
                {'label': 'ladder_t05', 't': 0.50,
                 'anchor_e_final': -114.901698537713,
                 'anchor_source': 'lc209'},
            ],
        },
        'reachability': {'guesses': ['minao', 'atom']},
        'spectra_ckpt_dir': '/nonexistent-ckpt',
        'thresholds': {'reproduce_tol': 1e-9, 'one_column_rel': 1e-6,
                       'basin_energy': 1e-8, 'new_dm_distance': 0.05},
    }
    plan.update(overrides)
    return plan


class PlanValidation(unittest.TestCase):

    def test_valid_plan_problems_are_only_missing_files(self):
        # geom/dm files do not exist in the test fixture; every other
        # layer must be silent (the LC-209 test convention)
        with tempfile.TemporaryDirectory() as td:
            plan = make_plan()
            ck = os.path.join(td, 'dense-H-R.npz')
            with open(ck, 'wb') as f:
                f.write(b'x')
            plan['descent']['hessian_ckpt'] = ck
            problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(problems)
        self.assertTrue(all(p.startswith('softmode plan inputs missing')
                            for p in problems), problems)

    def test_missing_keys_reported(self):
        plan = make_plan()
        del plan['descent']
        del plan['reachability']
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('descent' in p for p in problems))
        self.assertTrue(any('reachability' in p for p in problems))

    def test_family_coverage_fail_closed(self):
        plan = make_plan()
        del plan['family_dms']['C']
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('family_dms missing' in p for p in problems))

    def test_descent_point_and_sigma(self):
        plan = make_plan()
        plan['descent']['point'] = 'X'
        plan['descent']['sigma'] = 2
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('descent point' in p for p in problems))
        self.assertTrue(any('sigma' in p for p in problems))

    def test_hessian_ckpt_must_exist(self):
        plan = make_plan()   # ckpt path does not exist
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('hessian ckpt missing' in p
                            for p in problems))

    def test_targets_form(self):
        plan = make_plan()
        plan['descent']['targets'] = [
            {'label': 'a', 't': 0.5, 'anchor_e_final': -1.0,
             'anchor_source': 'x'},
            {'label': 'a', 't': 0.3, 'anchor_e_final': -1.0,
             'anchor_source': 'x'},
        ]
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('labels' in p for p in problems))
        plan['descent']['targets'] = [
            {'label': 'a', 't': 0.5, 'anchor_e_final': -1.0,
             'anchor_source': 'x'},
            {'label': 'b', 't': 0.3, 'anchor_e_final': -1.0,
             'anchor_source': 'x'},
        ]
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('ascending' in p for p in problems))
        plan['descent']['targets'] = [
            {'label': 'a', 't': 0.5, 'anchor_source': 'x'},
        ]
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('anchor_e_final' in p for p in problems))

    def test_guess_whitelist_and_uniqueness(self):
        plan = make_plan()
        plan['reachability']['guesses'] = ['minao', 'chkfile']
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any("guess 'chkfile'" in p for p in problems))
        plan['reachability']['guesses'] = ['minao', 'minao']
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('unique' in p for p in problems))
        plan['reachability']['guesses'] = []
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('guesses' in p for p in problems))

    def test_thresholds_positive(self):
        plan = make_plan()
        plan['thresholds']['reproduce_tol'] = -1e-9
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('reproduce_tol' in p for p in problems))

    def test_no_spectra_points_contract(self):
        plan = make_plan()
        plan['spectra_points'] = ['R']
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('spectra_points' in p for p in problems))

    def test_spectra_ckpt_dir_string(self):
        plan = make_plan()
        plan['spectra_ckpt_dir'] = 12
        problems = lc_.validate_ladder_plan(plan)
        self.assertTrue(any('spectra_ckpt_dir' in p for p in problems))


class ClassificationLogic(unittest.TestCase):

    def test_reproduction_hit_and_miss(self):
        hit = lc_.classify_reproduction(True, -114.901698537713,
                                        -114.901698537713, 1e-9)
        self.assertTrue(hit['reproduced'])
        self.assertEqual(hit['diff'], 0.0)
        miss = lc_.classify_reproduction(
            True, -114.901698537, -114.901698538, 1e-9)
        self.assertFalse(miss['reproduced'])
        self.assertAlmostEqual(miss['diff'], 1e-9, delta=1e-14)
        unconv = lc_.classify_reproduction(
            False, -114.901698537713, -114.901698537713, 1e-9)
        self.assertFalse(unconv['reproduced'])
        self.assertFalse(unconv['converged'])

    def test_spectrum_minimum(self):
        cls = lc_.classify_spectrum([1.0e-4, 2.0e-4, 5.0, 7.0])
        self.assertEqual(cls['n_negative'], 0)
        self.assertTrue(cls['is_minimum'])
        self.assertEqual(cls['classification'], 'minimum')
        self.assertAlmostEqual(cls['lambda1'], 1.0e-4)

    def test_spectrum_saddle_index(self):
        cls = lc_.classify_spectrum([-8.1e-5, -1.7e-5, 1.3e-5, 5.0])
        self.assertEqual(cls['n_negative'], 2)
        self.assertFalse(cls['is_minimum'])
        self.assertEqual(cls['classification'], 'saddle_index_2')
        self.assertAlmostEqual(cls['lambda1'], -8.1e-5)

    def test_spectrum_empty(self):
        cls = lc_.classify_spectrum([])
        self.assertEqual(cls['dim'], 0)
        self.assertTrue(cls['is_minimum'])
        self.assertIsNone(cls['lambda1'])

    def test_below_reference(self):
        # 2.2e-6 below -> flagged
        self.assertTrue(lc_.below_reference(-114.9016985, -114.9016963,
                                            1e-8))
        # 1e-7 below (beyond the 1e-8 tolerance) -> flagged
        self.assertTrue(lc_.below_reference(-114.9016964, -114.9016963,
                                            1e-8))
        # above the reference -> never flagged
        self.assertFalse(lc_.below_reference(-114.9016962, -114.9016963,
                                             1e-8))
        # 5e-9 below (inside the 1e-8 tolerance) -> NOT below
        self.assertFalse(lc_.below_reference(-114.901696305,
                                              -114.9016963, 1e-8))

    def test_reachability_summary(self):
        rows = [
            {'guess': 'minao', 'converged': True, 'e_tot': -114.9016950,
             'classification': {'class': 'basin_R'}, 'below_r0_auto': False},
            {'guess': 'atom', 'converged': True, 'e_tot': -114.9016964,
             'classification': {'class': 'basin_R0_AUTO'},
             'below_r0_auto': False},
            {'guess': 'huckel', 'converged': True, 'e_tot': -114.9016985,
             'classification': {'class': 'new_candidate'},
             'below_r0_auto': True},
        ]
        s = lc_.reachability_summary(rows)
        self.assertEqual(s['n_runs'], 3)
        self.assertTrue(s['all_converged'])
        self.assertTrue(s['any_below_r0_auto'])
        self.assertEqual(s['class_counts']['basin_R'], 1)
        self.assertEqual(s['class_counts']['new_candidate'], 1)
        self.assertAlmostEqual(s['min_e_tot'], -114.9016985)

        s2 = lc_.reachability_summary([])
        self.assertEqual(s2['n_runs'], 0)
        self.assertFalse(s2['all_converged'])
        self.assertFalse(s2['any_below_r0_auto'])
        self.assertIsNone(s2['min_e_tot'])


class Rendering(unittest.TestCase):

    def _report(self):
        return {
            'unit': 'LC-210-test', 'amendment_id': 'lc210-test',
            'batch_id': 'uc-test',
            'generated_at_utc': '2026-10-20T00:00:00Z',
            'schema_version': 'ladder-classify-v1',
            'inputs': {p: {'anchor_e': -114.901695}
                       for p in ('R', 'R0_AUTO', 'C', 'B')},
            'cpu_family_refs': {'R': -114.901695027205,
                                'R0_AUTO': -114.901696309862,
                                'C': -114.901694602346,
                                'B': -114.901695027239},
            'descent': {'point': 'R', 'sigma': 1, 'targets': []},
            'direction_determination': {
                'lambda1_dense': -1.669252e-05,
                'one_column_rel': 3.9e-09, 'validated': True,
                'direction_source': 'loaded_dense_validated',
                'davidson_softest3': [-1.669250e-05, 1.0e-3, 2.0e-3],
                'cos_davison_dense': 1.0},
            'endpoints': {
                'ladder_t05': {
                    't': 0.5, 'e_final': -114.901698537713,
                    'diff': 1.2e-13, 'cycles': 9, 'reproduced': True,
                    'stay': {'cycles': 1, 'dE_vs_rerun': -1e-13},
                    'dm_sha256': 'abcdef0123456789' * 3}},
            'reachability': {
                'guesses': ['minao'],
                'rows': [{'guess': 'minao',
                          'e_tot': -114.901695027205,
                          'converged': True, 'cycles': 10,
                          'classification': {'class': 'basin_R'},
                          'below_r0_auto': False}],
                'summary': lc_.reachability_summary([
                    {'guess': 'minao', 'converged': True,
                     'e_tot': -114.901695027205,
                     'classification': {'class': 'basin_R'},
                     'below_r0_auto': False}])},
            'classifications': {
                'ladder_t05': lc_.classify_spectrum(
                    [1.0e-4, 2.0e-4, 5.0])},
            'spectra': {
                'ladder_t05': {
                    'operator_asymmetry': 1e-12,
                    'davidson_softest3': [1.0e-4, 2.0e-4, 3.0e-4]}},
            'operational': {'wall_s': 1234.5},
        }

    def test_md_sections_and_values(self):
        md = lc_.render_md(self._report())
        for section in ('## CPU family references',
                        '## Loaded direction',
                        '## Endpoint reruns',
                        '## Reachability',
                        '## Endpoint curvature'):
            self.assertIn(section, md)
        self.assertIn('| ladder_t05 | minimum |', md)
        self.assertIn('| minao | -114.901695027205 |', md)
        self.assertIn('any_below_r0_auto=False', md)
        self.assertIn('loaded_dense_validated', md)
        self.assertIn('Wall 1234.5 s', md)

    def test_md_handles_no_classifications(self):
        report = self._report()
        report['classifications'] = {}
        report['spectra'] = {}
        md = lc_.render_md(report)
        self.assertIn('## Endpoint curvature', md)


class Interpreter(unittest.TestCase):

    def test_clean_interpreter_import(self):
        # the module must stay importable with a bare stdlib interpreter
        res = subprocess.run(
            [sys.executable, '-S', '-I',
             str(_DIR / 'upstream_ladder_classify.py'), '--help'],
            capture_output=True, text=True, timeout=60)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertIn('--plan', res.stdout)


if __name__ == '__main__':
    unittest.main()

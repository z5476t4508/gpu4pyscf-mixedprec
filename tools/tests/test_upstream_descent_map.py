#!/usr/bin/env python3
"""Tests for the LC-209 saddle-descent basin-connectivity map tool.

stdlib only; no numpy, no pyscf, no GPU.  Tested: the descent plan
validation layer (keys, amplitude/direction grids, hessian checkpoints,
fail-closed family coverage), the destination classification logic
(basin / new-candidate / unclassified / not-converged), the per-line
tipping summary, the plan-side contract of the loaded-H validation
thresholds, and the MD rendering.  The numeric paths (numpy/pyscf) are
exercised by the actual CPU run, like the LC-207/LC-208 tools.
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


dm_ = _load('upstream_descent_map')


def make_plan(**overrides):
    plan = {
        'unit': 'LC-209-test',
        'amendment_id': 'lc209-test',
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
        'descent_points': ['C', 'R'],
        'amplitudes': list(dm_.DEFAULT_AMPLITUDES),
        'directions': [1, -1],
        'hessian_ckpt': {
            'C': {'path': '/nonexistent-H-C.npz'},
            'R': {'path': '/nonexistent-H-R.npz'},
        },
        'thresholds': dict(dm_.DEFAULT_THRESHOLDS),
    }
    plan.update(overrides)
    return plan


class TestPlanValidation(unittest.TestCase):
    def test_valid_plan_problems_are_only_missing_files(self):
        problems = dm_.validate_descent_plan(make_plan())
        # geom/dm/ckpt files do not exist in the test fixture; every
        # other layer must be silent
        self.assertTrue(problems)
        self.assertTrue(all(('missing' in p or 'no such' in p.lower()
                             or 'not a file' in p.lower() or 'exist' in p)
                            or p.startswith('softmode plan inputs missing')
                            or 'hessian ckpt missing' in p
                            for p in problems), problems)
        self.assertTrue(any('hessian ckpt missing' in p for p in problems))
        self.assertTrue(any('softmode plan inputs missing' in p
                            for p in problems))

    def test_missing_key_reported(self):
        plan = make_plan()
        del plan['amplitudes']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("missing plan key 'amplitudes'" in p
                            for p in problems))

    def test_missing_nlcgrids_reported(self):
        plan = make_plan()
        del plan['settings']['nlcgrids_level']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("missing settings key 'nlcgrids_level'" in p
                            for p in problems))

    def test_family_point_coverage_required(self):
        plan = make_plan()
        del plan['family_dms']['B']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("family_dms missing 'B'" in p
                            for p in problems))

    def test_descent_point_needs_ckpt(self):
        plan = make_plan()
        del plan['hessian_ckpt']['R']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("descent point 'R' has no hessian_ckpt"
                            in p for p in problems))

    def test_unknown_descent_point(self):
        plan = make_plan()
        plan['descent_points'] = ['C', 'Q']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("unknown descent point 'Q'" in p
                            for p in problems))

    def test_amplitudes_must_be_positive_unique_ascending(self):
        for bad in ([0.05, 0.05, 0.1], [0.2, 0.1], [-0.1, 0.2],
                    [], ['big']):
            plan = make_plan()
            plan['amplitudes'] = bad
            problems = dm_.validate_descent_plan(plan)
            self.assertTrue(any('amplitudes' in p for p in problems),
                            bad)

    def test_directions_must_be_both_signs(self):
        for bad in ([1], [], [1, 2], [1, -1, 1]):
            plan = make_plan()
            plan['directions'] = bad
            problems = dm_.validate_descent_plan(plan)
            self.assertTrue(any('directions' in p for p in problems), bad)

    def test_spectra_points_rejected(self):
        plan = make_plan()
        plan['spectra_points'] = ['R']
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any('takes no spectra_points' in p
                            for p in problems))

    def test_negative_threshold_rejected(self):
        plan = make_plan()
        plan['thresholds'] = {'basin_energy': -1e-8}
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("threshold 'basin_energy'" in p
                            for p in problems))

    def test_hdiag_informational_only(self):
        # the 2026-10-20 erratum: hdiag is a preconditioner diagonal
        # for this operator (LC-208 measured 0.798 dev at every point);
        # validation is the one-column check, not hdiag
        self.assertTrue(dm_.DEFAULT_THRESHOLDS.get('hdiag_dev_info_only'))
        self.assertNotIn('hdiag_dev', dm_.DEFAULT_THRESHOLDS)

    def test_ckpt_entry_needs_path(self):
        plan = make_plan()
        plan['hessian_ckpt']['C'] = {'sha256': 'x'}
        problems = dm_.validate_descent_plan(plan)
        self.assertTrue(any("hessian_ckpt['C'] must carry a path" in p
                            for p in problems))

    def test_ckpt_files_existence_checked(self):
        with tempfile.TemporaryDirectory() as td:
            ck = os.path.join(td, 'dense-H-C.npz')
            with open(ck, 'wb') as f:
                f.write(b'x')
            plan = make_plan()
            plan['hessian_ckpt']['C'] = {'path': ck}
            problems = dm_.validate_descent_plan(plan)
            self.assertFalse(any(
                "hessian ckpt missing for 'C'" in p for p in problems))


class TestDefaults(unittest.TestCase):
    def test_default_amplitudes(self):
        self.assertEqual(list(dm_.DEFAULT_AMPLITUDES),
                         [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70])
        self.assertEqual(sorted(set(dm_.DEFAULT_AMPLITUDES)),
                         list(dm_.DEFAULT_AMPLITUDES))

    def test_default_directions(self):
        self.assertEqual(dm_.DEFAULT_DIRECTIONS, (1, -1))

    def test_default_thresholds_positive(self):
        for key, value in dm_.DEFAULT_THRESHOLDS.items():
            self.assertGreater(value, 0, key)


class TestClassifyDestination(unittest.TestCase):
    refs = {'R': -114.90169501301963,
            'R0_AUTO': -114.90169629560877,
            'C': -114.90169458806088,
            'B': -114.90169501301972}
    dists = {'R': 0.9, 'R0_AUTO': 0.02, 'C': 0.85, 'B': 0.88}
    th = dict(dm_.DEFAULT_THRESHOLDS)

    def test_basin_match_with_dm_corroboration(self):
        cls = dm_.classify_destination(
            True, self.refs['R0_AUTO'] + 1e-11, self.refs,
            self.dists, self.th)
        self.assertEqual(cls['class'], 'basin_R0_AUTO')
        self.assertTrue(cls['dm_smallest_matches'])
        self.assertLessEqual(cls['energy_diff'], self.th['basin_energy'])

    def test_basin_match_energy_only(self):
        dists = dict(self.dists)
        dists['R'] = 0.01     # dm-closest disagrees with the energy match
        cls = dm_.classify_destination(
            True, self.refs['R0_AUTO'] + 1e-11, self.refs, dists, self.th)
        self.assertEqual(cls['class'], 'basin_R0_AUTO')
        self.assertFalse(cls['dm_smallest_matches'])

    def test_not_converged(self):
        cls = dm_.classify_destination(
            False, self.refs['R0_AUTO'], self.refs, self.dists, self.th)
        self.assertEqual(cls['class'], 'not_converged')
        self.assertEqual(cls['nearest_by_energy'], 'R0_AUTO')

    def test_new_candidate_far_from_everything(self):
        dists = {k: 0.5 for k in self.dists}
        cls = dm_.classify_destination(
            True, -114.9017000, self.refs, dists, self.th)
        self.assertEqual(cls['class'], 'new_candidate')
        self.assertGreater(cls['energy_diff'], self.th['basin_energy'])

    def test_unclassified_energy_far_dm_close(self):
        dists = dict(self.dists)   # R0_AUTO dm-close (0.02) ...
        cls = dm_.classify_destination(
            True, self.refs['R0_AUTO'] - 1e-6, self.refs, dists, self.th)
        # ... but 1e-6 off in energy: not a basin match, and not far
        # enough by dm distance to be a new candidate
        self.assertEqual(cls['class'], 'unclassified')
        self.assertEqual(cls['nearest_by_energy'], 'R0_AUTO')

    def test_nearest_picks_minimum_difference(self):
        cls = dm_.classify_destination(
            True, self.refs['C'] + 5e-12, self.refs,
            {k: 0.5 for k in self.dists}, self.th)
        self.assertEqual(cls['class'], 'basin_C')


class TestSummarizeLines(unittest.TestCase):
    def _run(self, point, sigma, t, e_final, origin, cls, conv=True):
        return {'point': point, 'sigma': sigma, 't': t,
                'converged': conv, 'e_final': e_final,
                'e_cpu_origin': origin,
                'classification': {'class': cls}}

    def test_tipping_detected(self):
        origin = -114.90169458806088
        runs = [self._run('C', 1, 0.05, origin, origin, 'basin_C'),
                self._run('C', 1, 0.10, origin, origin, 'basin_C'),
                self._run('C', 1, 0.15, origin - 3e-6, origin,
                          'basin_R0_AUTO'),
                self._run('C', 1, 0.20, origin - 3e-6, origin,
                          'basin_R0_AUTO')]
        s = dm_.summarize_lines(runs, dict(dm_.DEFAULT_THRESHOLDS))
        line = s['C_+1']
        self.assertEqual(line['tipping_t'], 0.15)
        self.assertEqual(line['terminal'], 'basin_R0_AUTO')
        self.assertEqual(line['final_classes']['t=0.05'], 'basin_C')

    def test_no_tip_line(self):
        origin = -114.90169458806088
        runs = [self._run('C', -1, t, origin, origin, 'basin_C')
                for t in (0.05, 0.10)]
        s = dm_.summarize_lines(runs, dict(dm_.DEFAULT_THRESHOLDS))
        self.assertIsNone(s['C_-1']['tipping_t'])
        self.assertEqual(s['C_-1']['terminal'], 'basin_C')

    def test_not_converged_does_not_tip_but_is_terminal(self):
        origin = -114.90169458806088
        runs = [self._run('C', 1, 0.05, origin, origin, 'basin_C'),
                self._run('C', 1, 0.10, 0.0, origin, 'not_converged',
                          conv=False)]
        s = dm_.summarize_lines(runs, dict(dm_.DEFAULT_THRESHOLDS))
        self.assertIsNone(s['C_+1']['tipping_t'])
        self.assertEqual(s['C_+1']['terminal'], 'not_converged')

    def test_lines_are_per_point_and_sigma(self):
        origin = -114.90169458806088
        runs = [self._run('C', 1, 0.05, origin, origin, 'basin_C'),
                self._run('R', -1, 0.05, origin, origin, 'basin_C')]
        s = dm_.summarize_lines(runs, dict(dm_.DEFAULT_THRESHOLDS))
        self.assertIn('C_+1', s)
        self.assertIn('R_-1', s)


class TestRenderMd(unittest.TestCase):
    def test_render_contains_tables(self):
        report = {
            'unit': 'LC-209 test', 'generated_at_utc': '2026-10-20T00:00:00Z',
            'batch_id': 'uc-test', 'schema_version':
                dm_.SAMPLE_SCHEMA,
            'cpu_family_refs': {'R': -1.0},
            'inputs': {'R': {'anchor_e': -1.0}},
            'stay_controls': {'R': {'e_tot': -1.0, 'cycles': 1,
                                    'converged': True}},
            'direction_determination': {
                'R': {'lambda1_dense': -1.6e-5, 'one_column_rel': 1e-13,
                      'hdiag_dev': 1e-14, 'validated': True,
                      'direction_source': 'loaded_dense_validated',
                      'davidson_softest3': [-1.6e-5]}},
            'descent_runs': [
                {'point': 'R', 'sigma': 1, 't': 0.05,
                 'dE_rot_vs_stay': -2e-8, 'quad_ratio_rot': 1.2,
                 'e_final': -1.0, 'cycles': 2,
                 'classification': {'class': 'basin_R'}},
                {'point': 'R', 'sigma': -1, 't': 0.5,
                 'dE_rot_vs_stay': -4e-6, 'quad_ratio_rot': None,
                 'e_final': -1.000001, 'cycles': 12,
                 'classification': {'class': 'basin_R0_AUTO'}}],
            'line_summary': {'R_+1': {'tipping_t': None,
                                      'terminal': 'basin_R'}},
            'operational': {'wall_s': 12.3},
        }
        md = dm_.render_md(report)
        self.assertIn('## CPU family references', md)
        self.assertIn('## Stay controls', md)
        self.assertIn('## Direction determination', md)
        self.assertIn('## Descent runs', md)
        self.assertIn('## Line summary', md)
        self.assertIn('basin_R0_AUTO', md)
        self.assertIn('n/a', md)


class TestCleanInterpreterImport(unittest.TestCase):
    def test_module_imports_without_numpy(self):
        code = ('import sys; sys.path.insert(0, %r); '
                'import upstream_descent_map as m; '
                'print(m.SAMPLE_SCHEMA); print("cupy" in sys.modules, '
                '"pyscf" in sys.modules, "numpy" in sys.modules)'
                ) % str(_DIR)
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout.strip().splitlines()
        self.assertEqual(out[0], dm_.SAMPLE_SCHEMA)
        self.assertEqual(out[1], 'False False False', out)


if __name__ == '__main__':
    unittest.main()

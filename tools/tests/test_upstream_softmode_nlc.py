#!/usr/bin/env python3
"""Tests for the LC-208 NLC-inclusive soft-mode analysis tool.

stdlib only; no numpy, no pyscf, no GPU.  Tested: the NLC plan
validation layer (nlcgrids level, spectra points, profile lines),
checkpoint/segment bookkeeping, the quadratic-model prediction
arithmetic, the extended ladder rows, and the fail-closed NLC guard's
plan-side contract.  The numeric paths (numpy/pyscf) are exercised by
the actual CPU run, like the LC-207 tool.
"""

import importlib.util
import json
import math
import os
import pathlib
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


sn = _load('upstream_softmode_nlc')


def make_plan(**overrides):
    plan = {
        'unit': 'LC-208-test',
        'amendment_id': 'lc208-test',
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
                  'anchor_source': 'batch_R (LC-206 stage1)'},
            'R0_AUTO': {'path': '/nonexistent-A.npz',
                        'anchor_e': -114.90169629560877,
                        'anchor_source': 'a5_R0_AUTO (LC-206 stage1)'},
            'C': {'path': '/nonexistent-C.npz',
                  'anchor_e': -114.90169458806088,
                  'anchor_source': 'batch_C (LC-206 stage1)'},
            'B': {'path': '/nonexistent-B.npz',
                  'anchor_e': -114.90169501301972,
                  'anchor_source': 'batch_B (LC-208 anchor)'},
        },
        'profile': {'t_ref_source': 'test',
                    'fracs': [0.0, 0.5, -0.5, 1.0, -1.0]},
        'spectra_points': ['R', 'R0_AUTO', 'C', 'B'],
        'spectra_ckpt_dir': '/nonexistent-ckpt',
        'profile_lines': ['v1', 'kappa_C', 'kappa_R0_AUTO'],
    }
    plan.update(overrides)
    return plan


class TestNlcPlanValidation(unittest.TestCase):

    def _write_plan(self, plan):
        f = tempfile.NamedTemporaryFile('w', suffix='.json',
                                        delete=False)
        json.dump(plan, f)
        f.close()
        self.addCleanup(os.unlink, f.name)
        return f.name

    def test_valid_plan_loads(self):
        path = self._write_plan(make_plan())
        plan = sn.load_nlc_plan(path)
        self.assertEqual(plan['amendment_id'], 'lc208-test')
        self.assertEqual(plan['settings']['nlcgrids_level'], 3)
        self.assertEqual(plan['spectra_points'],
                         ['R', 'R0_AUTO', 'C', 'B'])

    def test_missing_nlcgrids_level_refused(self):
        plan = make_plan()
        del plan['settings']['nlcgrids_level']
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'nlcgrids_level'):
            sn.load_nlc_plan(path)

    def test_missing_spectra_points_refused(self):
        plan = make_plan()
        del plan['spectra_points']
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'spectra_points'):
            sn.load_nlc_plan(path)

    def test_empty_spectra_points_refused(self):
        plan = make_plan(spectra_points=[])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'spectra_points'):
            sn.load_nlc_plan(path)

    def test_duplicate_spectra_points_refused(self):
        plan = make_plan(spectra_points=['R', 'R', 'C'])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'unique list'):
            sn.load_nlc_plan(path)

    def test_spectra_points_without_R_refused(self):
        plan = make_plan(spectra_points=['C', 'B'])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, "'R'"):
            sn.load_nlc_plan(path)

    def test_non_family_spectra_point_refused(self):
        plan = make_plan(spectra_points=['R', 'ZZZ'])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'not a.*family point'):
            sn.load_nlc_plan(path)

    def test_missing_profile_lines_refused(self):
        plan = make_plan()
        del plan['profile_lines']
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'profile_lines'):
            sn.load_nlc_plan(path)

    def test_empty_profile_lines_refused(self):
        plan = make_plan(profile_lines=[])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'profile_lines'):
            sn.load_nlc_plan(path)

    def test_profile_lines_without_v1_refused(self):
        plan = make_plan(profile_lines=['kappa_C'])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'v1'):
            sn.load_nlc_plan(path)

    def test_unknown_profile_line_refused(self):
        plan = make_plan(profile_lines=['v1', 'kappa_R'])
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'kappa_R'):
            sn.load_nlc_plan(path)

    def test_missing_ckpt_dir_refused(self):
        plan = make_plan()
        del plan['spectra_ckpt_dir']
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'spectra_ckpt_dir'):
            sn.load_nlc_plan(path)

    def test_base_plan_validation_still_applies(self):
        plan = make_plan()
        del plan['family_dms']['C']
        path = self._write_plan(plan)
        with self.assertRaisesRegex(RuntimeError, 'family_dms'):
            sn.load_nlc_plan(path)


class TestColumnsToRun(unittest.TestCase):

    def test_fresh_build_all_columns(self):
        self.assertEqual(sn.columns_to_run(0, 4), [0, 1, 2, 3])

    def test_resume_skips_done(self):
        self.assertEqual(sn.columns_to_run(2, 5), [2, 3, 4])

    def test_complete_build_empty(self):
        self.assertEqual(sn.columns_to_run(5, 5), [])

    def test_segment_caps(self):
        self.assertEqual(sn.columns_to_run(0, 10, segment=4), [0, 1, 2, 3])
        self.assertEqual(sn.columns_to_run(4, 10, segment=4), [4, 5, 6, 7])
        self.assertEqual(sn.columns_to_run(8, 10, segment=4), [8, 9])

    def test_invalid_states_refused(self):
        with self.assertRaises(RuntimeError):
            sn.columns_to_run(-1, 4)
        with self.assertRaises(RuntimeError):
            sn.columns_to_run(6, 4)
        with self.assertRaises(RuntimeError):
            sn.columns_to_run(0, 4, segment=0)
        with self.assertRaises(RuntimeError):
            sn.columns_to_run(0, 4, segment=-3)


class TestCkptDue(unittest.TestCase):

    def test_zero_never_triggers(self):
        self.assertFalse(sn.ckpt_due(0, 100))

    def test_multiples_trigger(self):
        self.assertTrue(sn.ckpt_due(100, 100))
        self.assertTrue(sn.ckpt_due(200, 100))
        self.assertFalse(sn.ckpt_due(150, 100))

    def test_nonpositive_every_refused(self):
        with self.assertRaises(RuntimeError):
            sn.ckpt_due(100, 0)
        with self.assertRaises(RuntimeError):
            sn.ckpt_due(100, -2)


class TestQuadPrediction(unittest.TestCase):

    def test_single_mode_pure_alignment(self):
        # kappa = (3, 4), single mode v = (0.6, 0.8), lambda = 2
        q = sn.quad_prediction([2.0], [[0.6, 0.8]], [3.0, 4.0])
        # (kappa.v)^2 = (1.8+3.2)^2 = 25; dE = 2*25/2 = 25
        self.assertAlmostEqual(q, 25.0, places=12)

    def test_orthogonal_modes_add(self):
        # two orthonormal modes with lambdas 4 and -2
        evals = [4.0, -2.0]
        rows = [[1.0, 0.0], [0.0, 1.0]]
        q = sn.quad_prediction(evals, rows, [2.0, 3.0])
        # 4*4/2 + (-2)*9/2 = 8 - 9 = -1
        self.assertAlmostEqual(q, -1.0, places=12)

    def test_zero_displacement(self):
        self.assertEqual(sn.quad_prediction([3.0], [[1.0]], [0.0]), 0.0)

    def test_matches_direct_quadratic_form(self):
        # random 3x3 symmetric via explicit sum (same math, both paths)
        evals = [0.5, -1.25, 3.0]
        # orthonormal rows: identity is fine (axis-aligned eigenbasis)
        rows = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        kappa = [1.5, -2.0, 0.5]
        q = sn.quad_prediction(evals, rows, kappa)
        direct = 0.5 * sum(l * k * k for l, k in zip(evals, kappa))
        self.assertAlmostEqual(q, direct, places=12)


class TestLadderRowsNlc(unittest.TestCase):

    def test_sorted_with_ratios(self):
        rows = sn.ladder_rows_nlc([
            {'label': 'C', 'dE': 4.25e-7, 't_proj': 0.03,
             'quad_pred': 4.0e-7},
            {'label': 'R0_AUTO', 'dE': -1.28e-6, 't_proj': -0.05,
             'quad_pred': -1.30e-6},
            {'label': 'B', 'dE': 0.0, 't_proj': 0.0, 'quad_pred': 0.0},
        ])
        self.assertEqual([r['label'] for r in rows],
                         ['B', 'C', 'R0_AUTO'])
        by = {r['label']: r for r in rows}
        self.assertAlmostEqual(by['C']['dE_over_t2'],
                               4.25e-7 / 0.0009, places=20)
        self.assertAlmostEqual(by['C']['dE_over_quad'],
                               4.25e-7 / 4.0e-7, places=12)
        self.assertAlmostEqual(by['R0_AUTO']['dE_over_quad'],
                               -1.28e-6 / -1.30e-6, places=12)
        self.assertIsNone(by['B']['dE_over_t2'])
        self.assertIsNone(by['B']['dE_over_quad'])

    def test_missing_t_and_quad_handled(self):
        rows = sn.ladder_rows_nlc([
            {'label': 'X', 'dE': 1e-7, 't_proj': 0.0, 'quad_pred': None},
        ])
        self.assertIsNone(rows[0]['dE_over_t2'])
        self.assertIsNone(rows[0]['dE_over_quad'])
        self.assertEqual(rows[0]['quad_pred'], None)


class TestFailClosedNlcGuardContract(unittest.TestCase):

    def test_build_mol_mf_nlc_exists_and_is_lazy(self):
        # the guard itself needs pyscf (run-time fail-closed); here we
        # only pin the plan-side contract: the guard name exists and the
        # module imports stdlib-only
        self.assertTrue(callable(sn.build_mol_mf_nlc))
        self.assertTrue(callable(sn.gen_g_hop_uhf_nlc))
        self.assertTrue(callable(sn.dense_stability_nlc))

    def test_module_imports_without_numpy(self):
        # re-import in a clean interpreter: no numpy/pyscf at module scope
        import subprocess
        mod_path = str(_DIR / 'upstream_softmode_nlc.py')
        code = ('import importlib.util;'
                f'spec = importlib.util.spec_from_file_location('
                f'"sn2", {mod_path!r});'
                'm = importlib.util.module_from_spec(spec);'
                'spec.loader.exec_module(m); print("OK")')
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('OK', r.stdout)

    def test_base_tool_reexported(self):
        # the NLC tool shares the LC-207 arithmetic surface via base
        for name in ('t_grid', 'sym_curvature', 'odd_part',
                     'curvature_series', 'spectral_weight', 'cosine',
                     'interp_line', 'sha256_file'):
            self.assertTrue(hasattr(sn.base, name), name)


if __name__ == '__main__':
    unittest.main()

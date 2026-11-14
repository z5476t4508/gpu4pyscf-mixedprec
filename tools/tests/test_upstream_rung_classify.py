#!/usr/bin/env python3
"""Tests for the LC-211 rung-classification + interior-probe tool.

stdlib only; no numpy, no pyscf, no GPU.  Tested: the plan validation
layer (keys, rung guess whitelist + anchors, mandatory-rung contract,
ladder endpoint form, interior probe point/sigma/t/ckpt/bracket order,
adaptive positivity, no-spectra_points contract, fail-closed family
coverage), the same-rung corroboration logic, the interior-placement
and adaptive dense-branch logic, and the MD rendering.  The numeric
paths (numpy/pyscf) are exercised by the actual CPU run, like the
LC-207..210 tools.
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


rc_ = _load('upstream_rung_classify')


def make_plan(**overrides):
    plan = {
        'unit': 'LC-211-test',
        'amendment_id': 'lc211-test',
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
        'ladder_endpoint_dms': {
            'ladder_Rp1_t0.30': {
                'path': '/nonexistent-t03.npz',
                'anchor_e_final': -114.90169710587256,
                'anchor_source': 'LC-210 report'},
            'ladder_Rp1_t0.50': {
                'path': '/nonexistent-t05.npz',
                'anchor_e_final': -114.90169853771255,
                'anchor_source': 'LC-210 report'},
        },
        'rungs': [
            {'guess': 'vsap', 'anchor_e_tot': -114.90169815656608,
             'anchor_source': 'LC-210 report reachability row vsap'},
            {'guess': '1e', 'anchor_e_tot': -114.90169815656654,
             'anchor_source': 'LC-210 report reachability row 1e'},
        ],
        'rung_dense_mandatory': 'vsap',
        'interior_probe': {
            'point': 'R', 'sigma': 1, 't': 0.6,
            'hessian_ckpt': '/nonexistent-ckpt-R.npz',
            'bracket_anchors': {
                'lower': {'label': 'ladder_Rp1_t0.50',
                          'e': -114.90169853771255},
                'upper': {'label': 'ladder_Rp1_t0.70',
                          'e': -114.9016976025828},
            },
        },
        'adaptive': {'same_rung_e': 1e-9, 'same_rung_dm': 0.05,
                     'interior_dense_below': 1e-10},
        'thresholds': {'reproduce_tol': 1e-9, 'one_column_rel': 1e-6,
                       'basin_energy': 1e-8, 'new_dm_distance': 0.05},
        'spectra_ckpt_dir': '/nonexistent-ckpt',
    }
    plan.update(overrides)
    return plan


class PlanValidation(unittest.TestCase):

    def test_valid_plan_problems_are_only_missing_files(self):
        # geom/dm files do not exist in the test fixture; every other
        # layer must be silent (the LC-209/210 test convention)
        with tempfile.TemporaryDirectory() as td:
            plan = make_plan()
            ck = os.path.join(td, 'dense-H-R.npz')
            with open(ck, 'wb') as f:
                f.write(b'x')
            plan['interior_probe']['hessian_ckpt'] = ck
            problems = rc_.validate_rung_plan(plan)
        self.assertTrue(problems)
        self.assertTrue(all(p.startswith('softmode plan inputs missing')
                            for p in problems), problems)

    def test_missing_keys_reported(self):
        plan = make_plan()
        del plan['rungs']
        del plan['interior_probe']
        del plan['ladder_endpoint_dms']
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('rungs' in p for p in problems))
        self.assertTrue(any('interior_probe' in p for p in problems))
        self.assertTrue(any('ladder_endpoint_dms' in p for p in problems))

    def test_family_coverage_fail_closed(self):
        plan = make_plan()
        del plan['family_dms']['C']
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('family_dms missing' in p for p in problems))

    def test_ladder_endpoint_form(self):
        plan = make_plan()
        plan['ladder_endpoint_dms']['ladder_Rp1_t0.30'] = {
            'path': '/x.npz', 'anchor_source': 'y'}
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('anchor_e_final' in p for p in problems))
        plan['ladder_endpoint_dms']['ladder_Rp1_t0.30'] = {
            'path': '/x.npz', 'anchor_e_final': 'not-a-number',
            'anchor_source': 'y'}
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('anchor_e_final' in p for p in problems))
        plan['ladder_endpoint_dms'] = {}
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('ladder_endpoint_dms' in p for p in problems))

    def test_rungs_form(self):
        plan = make_plan()
        plan['rungs'][0]['guess'] = 'chkfile'
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any("guess 'chkfile'" in p for p in problems))
        plan = make_plan()
        del plan['rungs'][1]
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('at least two' in p for p in problems))
        plan = make_plan()
        plan['rungs'][1]['guess'] = 'vsap'
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('unique' in p for p in problems))
        plan = make_plan()
        del plan['rungs'][0]['anchor_e_tot']
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('anchor_e_tot' in p for p in problems))
        plan = make_plan()
        del plan['rungs'][1]['anchor_source']
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('anchor_source' in p for p in problems))

    def test_mandatory_rung_must_be_in_rungs(self):
        plan = make_plan()
        plan['rung_dense_mandatory'] = 'minao'
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('rung_dense_mandatory' in p for p in problems))

    def test_interior_probe_form(self):
        plan = make_plan()
        plan['interior_probe']['point'] = 'X'
        plan['interior_probe']['sigma'] = 0
        plan['interior_probe']['t'] = -0.1
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('interior_probe point' in p for p in problems))
        self.assertTrue(any('sigma' in p for p in problems))
        self.assertTrue(any('t must be positive' in p for p in problems))

    def test_hessian_ckpt_must_exist(self):
        plan = make_plan()   # ckpt path does not exist
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('hessian ckpt missing' in p
                            for p in problems))

    def test_bracket_order_fail_closed(self):
        plan = make_plan()
        plan['interior_probe']['bracket_anchors']['lower']['e'] = -1.0
        plan['interior_probe']['bracket_anchors']['upper']['e'] = -2.0
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('lower' in p and '<=' in p for p in problems))
        plan['interior_probe']['bracket_anchors']['lower'] = {
            'e': -2.0}
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any("missing 'label'" in p for p in problems))

    def test_adaptive_positive(self):
        plan = make_plan()
        plan['adaptive']['interior_dense_below'] = 0
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('interior_dense_below' in p for p in problems))

    def test_thresholds_positive(self):
        plan = make_plan()
        plan['thresholds']['reproduce_tol'] = -1e-9
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('reproduce_tol' in p for p in problems))

    def test_no_spectra_points_contract(self):
        plan = make_plan()
        plan['spectra_points'] = ['R']
        problems = rc_.validate_rung_plan(plan)
        self.assertTrue(any('no spectra_points' in p for p in problems))


class SameRungCorroboration(unittest.TestCase):

    def test_same_rung_when_both_clauses_pass(self):
        ad = {'same_rung_e': 1e-9, 'same_rung_dm': 0.05}
        c = rc_.same_rung_corroboration(4.6e-13, 1.2e-6, ad)
        self.assertTrue(c['same_rung'])
        self.assertTrue(c['same_rung_e'])
        self.assertTrue(c['same_rung_dm'])
        self.assertAlmostEqual(c['dE'], 4.6e-13)

    def test_distinct_when_energy_far(self):
        ad = {'same_rung_e': 1e-9, 'same_rung_dm': 0.05}
        c = rc_.same_rung_corroboration(-1.85e-6, 1e-6, ad)
        self.assertFalse(c['same_rung'])
        self.assertTrue(c['same_rung_dm'])
        self.assertFalse(c['same_rung_e'])

    def test_distinct_when_dm_far(self):
        ad = {'same_rung_e': 1e-9, 'same_rung_dm': 0.05}
        c = rc_.same_rung_corroboration(1e-13, 0.2874, ad)
        self.assertFalse(c['same_rung'])
        self.assertFalse(c['same_rung_dm'])
        self.assertTrue(c['same_rung_e'])


class InteriorPlacement(unittest.TestCase):

    def test_in_bracket_above_minimum(self):
        # t=0.60 between the t=0.50 minimum and the t=0.70 rise
        p = rc_.interior_placement(-114.9016981, -114.9016985377,
                                   -114.9016976026, 1e-10)
        self.assertTrue(p['in_bracket'])
        self.assertFalse(p['below_lower'])
        self.assertFalse(p['interior_dense_branch'])
        self.assertAlmostEqual(p['diff_vs_lower'], 4.377e-7, places=12)
        self.assertAlmostEqual(p['diff_vs_upper'], -4.974e-7, places=12)

    def test_below_minimum_fires_adaptive_branch(self):
        p = rc_.interior_placement(-114.9016989, -114.9016985377,
                                   -114.9016976026, 1e-10)
        self.assertTrue(p['below_lower'])
        self.assertTrue(p['interior_dense_branch'])
        # below the deeper bracket edge = outside the [lo, hi] interval
        self.assertFalse(p['in_bracket'])

    def test_above_upper_out_of_bracket(self):
        p = rc_.interior_placement(-114.9016970, -114.9016985377,
                                   -114.9016976026, 1e-10)
        self.assertFalse(p['in_bracket'])
        self.assertFalse(p['below_lower'])
        self.assertFalse(p['interior_dense_branch'])

    def test_tolerance_edge(self):
        # exactly at the below_tol boundary: NOT below (strict inequality)
        p = rc_.interior_placement(-114.9016985377 - 1e-10,
                                   -114.9016985377, -114.9016976026,
                                   1e-10)
        self.assertFalse(p['below_lower'])


class RenderMD(unittest.TestCase):

    def test_render_md_smoke(self):
        report = {
            'unit': 'LC-211-test', 'amendment_id': 'lc211-test',
            'batch_id': 'uc-test', 'schema_version': 'rung-classify-v1',
            'generated_at_utc': '2026-10-20T00:00:00Z',
            'cpu_family_refs': {'R': -114.901695013020},
            'ladder_refs': {'ladder_Rp1_t0.50': -114.901698537713},
            'inputs': {'R': {'anchor_e': -114.901695013020}},
            'ladder_inputs': {'ladder_Rp1_t0.50': {
                'anchor_e_final': -114.901698537713}},
            'direction_determination': {
                'lambda1_dense': -1.669252e-05, 'one_column_rel': 7.3e-9,
                'validated': True,
                'direction_source': 'loaded_dense_validated'},
            'rungs': {'vsap': {
                'e_tot': -114.901698156566, 'diff': 4.6e-13,
                'cycles': 11,
                'stay': {'cycles': 1, 'dE_vs_rerun': -1.2e-12},
                'reproduced': True,
                'classification': {'class': 'new_candidate'}}},
            'rung_corroboration': {
                'dE': 4.6e-13, 'dm_distance': 1.2e-6,
                'same_rung_e': True, 'same_rung_dm': True,
                'same_rung': True},
            'adaptive': {'same_rung_e': 1e-9, 'same_rung_dm': 0.05},
            'interior_probe': {
                'point': 'R', 'sigma': 1,
                'e_final': -114.9016981, 'converged': True,
                'cycles': 12,
                'stay': {'cycles': 1, 'dE_vs_rerun': -1e-12},
                'bracket': {
                    'lower_label': 'ladder_Rp1_t0.50',
                    'lower_e': -114.901698537713,
                    'upper_label': 'ladder_Rp1_t0.70',
                    'upper_e': -114.901697602583,
                    'diff_vs_lower': 4.4e-7, 'diff_vs_upper': -5.0e-7,
                    'in_bracket': True, 'below_lower': False,
                    'interior_dense_branch': False}},
            'classifications': {
                'rung_vsap': {'classification': 'minimum',
                              'lambda1': 7.568575e-05, 'n_negative': 0}},
            'spectra': {'rung_vsap': {
                'operator_asymmetry': 1.3e-14,
                'davidson_softest3': [7.568575e-05]}},
            'operational': {'wall_s': 123.4},
        }
        md = rc_.render_md(report)
        self.assertIn('# LC-211-test', md)
        self.assertIn('Rung corroboration', md)
        self.assertIn('same rung: True', md)
        self.assertIn('Interior probe (t=0.60', md)
        self.assertIn('in_bracket=True', md)
        self.assertIn('| rung_vsap | minimum |', md)
        self.assertIn('Wall 123.4 s', md)

    def test_render_md_no_classifications(self):
        report = {
            'unit': 'u', 'amendment_id': 'a', 'batch_id': 'b',
            'schema_version': 'rung-classify-v1',
            'generated_at_utc': 't',
            'cpu_family_refs': {}, 'ladder_refs': {},
            'inputs': {}, 'ladder_inputs': {},
            'direction_determination': {
                'lambda1_dense': 0.0, 'one_column_rel': 0.0,
                'validated': False, 'direction_source': 'none'},
            'rungs': {}, 'rung_corroboration': {'dE': 0.0,
                                                'dm_distance': 0.0,
                                                'same_rung_e': False,
                                                'same_rung_dm': False,
                                                'same_rung': False},
            'adaptive': {'same_rung_e': 1e-9, 'same_rung_dm': 0.05},
            'interior_probe': {
                'point': 'R', 'sigma': 1, 'e_final': 0.0,
                'converged': True, 'cycles': 1,
                'stay': {'cycles': 1, 'dE_vs_rerun': 0.0},
                'bracket': {
                    'lower_label': 'lo', 'lower_e': -2.0,
                    'upper_label': 'hi', 'upper_e': -1.0,
                    'diff_vs_lower': 0.0, 'diff_vs_upper': 0.0,
                    'in_bracket': True, 'below_lower': False,
                    'interior_dense_branch': False}},
            'classifications': {}, 'spectra': {},
            'operational': {'wall_s': 1.0},
        }
        md = rc_.render_md(report)
        self.assertIn('(no gated state reached a dense build)', md)


class CleanImport(unittest.TestCase):

    def test_clean_interpreter_import(self):
        # the module must stay importable without numpy/pyscf (the
        # lazy-import contract used by the tools suite)
        code = ('import importlib.util, sys, pathlib; '
                f'spec = importlib.util.spec_from_file_location('
                f'"rc", pathlib.Path(r"{_DIR}") / "upstream_rung_classify.py"); '
                'm = importlib.util.module_from_spec(spec); '
                'spec.loader.exec_module(m); '
                'assert m.SAMPLE_SCHEMA == "rung-classify-v1"; '
                'print("ok")')
        r = subprocess.run([sys.executable, '-c', code],
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('ok', r.stdout)


if __name__ == '__main__':
    unittest.main()

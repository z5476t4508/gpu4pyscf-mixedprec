#!/usr/bin/env python3
"""Tests for the LC-207 soft-mode analysis tool (pure-logic surface).

stdlib only; no numpy, no pyscf, no GPU.  Tested: plan validation,
input verification, scan-grid construction, symmetric curvature /
odd-part arithmetic, curvature series from +- pairs, spectral weight,
the quadratic ladder, and line interpolation.  The numeric paths
(numpy/pyscf/scipy) are exercised by the actual CPU run, like the
basin-probe arm path is exercised by the GPU run.
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


sa = _load('upstream_softmode_analysis')


def make_plan(**overrides):
    plan = {
        'unit': 'LC-207-test',
        'amendment_id': 'lc207-test',
        'batch_id': 'uc-test',
        'evidence_dir': '/nonexistent',
        'cell': {'tag': 'methoxy_b3lyp_def2-svp', 'index': 173,
                 'geom': '/nonexistent/methoxy.xyz', 'basis': 'def2-svp',
                 'xc': 'b3lyp', 'charge': 0, 'spin': 1},
        'settings': {'auxbasis': 'def2-universal-jkfit',
                     'grids_level': 3, 'conv_tol': 1e-10,
                     'direct_scf_tol': 1e-14, 'max_cycle': 50,
                     'init_guess': 'minao'},
        'family_dms': {
            'R': {'path': '/nonexistent-R.npz',
                  'anchor_e': -114.95999760644713,
                  'anchor_source': 'batch_R (LC-206 stage1)'},
            'R0_AUTO': {'path': '/nonexistent-A.npz',
                        'anchor_e': -114.95999756618325,
                        'anchor_source': 'a5_R0_AUTO (LC-206 stage1)'},
            'C': {'path': '/nonexistent-C.npz',
                  'anchor_e': -114.95999755265314,
                  'anchor_source': 'batch_C (LC-206 stage1)'},
            'B': {'path': '/nonexistent-B.npz',
                  'anchor_e': -114.95999760254054,
                  'anchor_source': 'batch_B (LC-207 anchor)'},
        },
        'profile': {'t_ref_source': 't_proj of C on v1',
                    'fracs': list(sa.PROFILE_FRACS)},
    }
    plan.update(overrides)
    return plan


class TestPlanValidation(unittest.TestCase):

    def test_valid_plan_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(make_plan(), f)
            loaded = sa.load_plan(p)
            self.assertEqual(loaded['amendment_id'], 'lc207-test')

    def _refused(self, plan, msg):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(plan, f)
            with self.assertRaisesRegex(RuntimeError, msg):
                sa.load_plan(p)

    def test_missing_top_key_refused(self):
        plan = make_plan()
        del plan['settings']
        self._refused(plan, 'missing required key')

    def test_missing_cell_field_refused(self):
        plan = make_plan()
        del plan['cell']['spin']
        self._refused(plan, "cell missing 'spin'")

    def test_missing_settings_field_refused(self):
        plan = make_plan()
        del plan['settings']['conv_tol']
        self._refused(plan, "settings missing 'conv_tol'")

    def test_missing_family_point_refused(self):
        plan = make_plan()
        del plan['family_dms']['B']
        self._refused(plan, "missing 'B'")

    def test_family_point_missing_field_refused(self):
        plan = make_plan()
        del plan['family_dms']['C']['anchor_e']
        self._refused(plan, "missing 'anchor_e'")

    def test_family_point_non_numeric_anchor_refused(self):
        plan = make_plan()
        plan['family_dms']['R']['anchor_e'] = 'minus one fourteen'
        self._refused(plan, 'non-numeric anchor_e')

    def test_missing_profile_key_refused(self):
        plan = make_plan()
        del plan['profile']['fracs']
        self._refused(plan, "profile missing 'fracs'")


class TestInputVerification(unittest.TestCase):

    def test_missing_dm_files_refused(self):
        plan = make_plan()
        with tempfile.TemporaryDirectory() as tmp:
            geom = os.path.join(tmp, 'methoxy.xyz')
            with open(geom, 'w', encoding='utf-8') as f:
                f.write('5\n\nC 0 0 0\n')
            for p in plan['family_dms'].values():
                p['path'] = os.path.join(tmp, 'missing.npz')
            with self.assertRaisesRegex(RuntimeError, 'inputs missing'):
                sa.verify_inputs(plan)

    def test_existing_inputs_pass(self):
        plan = make_plan()
        with tempfile.TemporaryDirectory() as tmp:
            geom = os.path.join(tmp, 'methoxy.xyz')
            with open(geom, 'w', encoding='utf-8') as f:
                f.write('5\n\nC 0 0 0\n')
            plan['cell']['geom'] = geom
            for p in plan['family_dms'].values():
                p['path'] = os.path.join(tmp, 'dm.npz')
                with open(p['path'], 'wb') as f:
                    f.write(b'x')
            self.assertTrue(sa.verify_inputs(plan))


class TestGridAndArithmetic(unittest.TestCase):

    def test_t_grid_sorted_unique_with_single_zero(self):
        ts = sa.t_grid(0.08)
        self.assertEqual(ts, sorted(ts))
        self.assertEqual(ts.count(0.0), 1)
        # symmetric
        for t in ts:
            if t != 0.0:
                self.assertIn(-t, ts)
        # fractions of the reference amplitude
        self.assertAlmostEqual(max(ts), 1.25 * 0.08)

    def test_t_grid_inserts_zero_when_absent(self):
        ts = sa.t_grid(1.0, fracs=(0.5, -0.5, 1.0))
        self.assertIn(0.0, ts)

    def test_sym_curvature_recovers_lambda(self):
        lam = -1.7e-5
        t = 0.02
        e0 = -114.96
        # E(t) = E0 + lam*t^2/2 + quartic-free; the E-difference
        # arithmetic carries a cancellation floor (|E| ~ 1e2, so the
        # curvature recovers to a relative 1e-5, not absolutely)
        ep = e0 + lam * t * t / 2
        em = e0 + lam * t * t / 2
        self.assertAlmostEqual(sa.sym_curvature(em, e0, ep, t), lam,
                               delta=abs(lam) * 1e-5)

    def test_sym_curvature_odd_terms_cancel(self):
        # a purely odd (linear) contribution must NOT enter the
        # symmetric curvature
        e0 = -114.96
        t = 0.03
        lin = 1e-7
        c4 = 5e-3
        ep = e0 + lin * t + c4 * t ** 4
        em = e0 - lin * t + c4 * t ** 4
        lam = sa.sym_curvature(em, e0, ep, t)
        # only the symmetric quartic part remains: E_sym(t) = c4*t^4
        # -> curvature 2*c4*t^2, to the cancellation floor
        want = 2 * c4 * t * t
        self.assertAlmostEqual(lam, want, delta=abs(want) * 1e-5)

    def test_odd_part(self):
        ep, em = -114.96 + 3e-9, -114.96 + 1e-9
        self.assertAlmostEqual(sa.odd_part(em, ep), 1e-9)

    def test_curvature_series_pairs(self):
        lam = -2e-5
        e0 = -114.96
        prof = {0.0: e0}
        for t in (0.01, 0.02, 0.05):
            prof[t] = e0 + lam * t * t / 2
            prof[-t] = e0 + lam * t * t / 2
        series = sa.curvature_series(prof)
        self.assertEqual([t for t, _ in series], [0.01, 0.02, 0.05])
        for t, l in series:
            self.assertAlmostEqual(l, lam, delta=abs(lam) * 1e-5)

    def test_curvature_series_skips_unpaired(self):
        prof = {0.0: -114.96, 0.02: -114.960000001,
                -0.02: -114.960000001, 0.05: -114.96}
        series = sa.curvature_series(prof)
        self.assertEqual([t for t, _ in series], [0.02])

    def test_spectral_weight(self):
        self.assertAlmostEqual(sa.spectral_weight(0.9), 0.81)
        self.assertAlmostEqual(sa.spectral_weight(-1.0), 1.0)

    def test_ladder_rows_sorted_with_ratio(self):
        rows = sa.ladder_rows([
            {'label': 'C', 'dE': 5.38e-8, 't_proj': 0.08},
            {'label': 'B', 'dE': 3.9e-9, 't_proj': 0.02},
            {'label': 'R0_AUTO', 'dE': 4.0e-8, 't_proj': 0.07},
        ])
        self.assertEqual([r['label'] for r in rows],
                         ['B', 'R0_AUTO', 'C'])
        for r in rows:
            self.assertAlmostEqual(r['dE_over_t2'],
                                   r['dE'] / (r['t_proj'] ** 2))
        # a zero-amplitude row keeps a None ratio, never a division
        rows = sa.ladder_rows([{'label': 'X', 'dE': 0.0,
                                't_proj': 0.0}])
        self.assertIsNone(rows[0]['dE_over_t2'])

    def test_ladder_rows_handle_missing_t(self):
        rows = sa.ladder_rows([{'label': 'X', 'dE': 1e-8,
                                't_proj': None}])
        self.assertIsNone(rows[0]['dE_over_t2'])

    def test_interp_line_exact_and_interpolated(self):
        prof = {0.0: -114.96, 0.02: -114.960000004,
                -0.02: -114.960000004, 0.04: -114.960000010}
        self.assertAlmostEqual(sa.interp_line(prof, 0.02),
                               -114.960000004)
        mid = sa.interp_line(prof, 0.03)
        self.assertAlmostEqual(mid, -114.960000007)
        self.assertIsNone(sa.interp_line(prof, 0.1))

    def test_sha256_file_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'f.txt')
            with open(p, 'wb') as f:
                f.write(b'hello')
            import hashlib
            self.assertEqual(sa.sha256_file(p),
                             hashlib.sha256(b'hello').hexdigest())


if __name__ == '__main__':
    unittest.main()

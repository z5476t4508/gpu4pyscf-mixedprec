#!/usr/bin/env python3
"""CPU-only unit tests for the LC-2 gradient hotspot profiler
(mixedprec/benchmark/rtx4090/grad_hotspot_profiler.py).

Covers validate_document rules V1-V7 (adversarial: each rule violated in
isolation must be rejected) and the GradientPatcher install/restore
double-counting guard (wrapping the DF rks class must not double-wrap the
rhf class's distinct dict entry; restore must return the original objects).
No GPU is touched: the patcher is exercised against the real module
hierarchy with a stub timer.
"""
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
RTX = os.path.abspath(os.path.join(HERE, '..', '..', 'mixedprec',
                                   'benchmark', 'rtx4090'))
sys.path.insert(0, RTX)

import grad_hotspot_profiler as ghp  # noqa: E402

# The patcher tests exercise the REAL gpu4pyscf gradient class hierarchy,
# which needs the source env (numpy/cupy).  The tools suite is otherwise
# stdlib-only and runs under system python3; skip there, same convention as
# the issue-9 smoke-document tests.
try:
    import numpy  # noqa: F401
    HAS_SOURCE_ENV = True
except ImportError:
    HAS_SOURCE_ENV = False


def _bucket(t_s, calls=1):
    return {'t_s': t_s, 'calls': calls}


def _lane(**kw):
    '''A compliant lane record (auto lane with measured fp32 entries).'''
    lane = {
        'lane': 'auto',
        'scf': {'e_tot': -1.0, 't_scf_s': 1.0, 'cycles': 9,
                'converged': True},
        'baseline': {'samples_s': [0.5, 0.5, 0.5], 'median_s': 0.5,
                     'run_to_run_grad_var_max': 1e-13},
        'instrumented': {
            'samples_s': [0.5, 0.5, 0.5], 'median_s': 0.5, 'per_sample': []},
        'buckets': {
            'grad.hcore_e1': _bucket(0.10),
            'grad.e2_total': _bucket(0.30),
            'grad.xc_grid': _bucket(0.15),
            'grad.df_jk_ip1': _bucket(0.14),
            'grad.nuc': _bucket(0.001),
        },
        'derived': {
            'unattributed_s': 0.05, 'unattributed_frac': 0.09,
            'e2_other_s': 0.01, 'e2_other_frac_of_e2': 0.03,
            'sync_serialisation_overhead': 1.0,
        },
        'fp32_entry': {'grad_ip1_hits': 6, 'grad_mode_effective': 'fp32',
                       'note': 'measured'},
        'equiv_max_abs_diff': 1e-13,
        'equiv_tolerance': 1e-8,
        'equiv_observation_only': True,
    }
    lane.update(kw)
    return lane


def _doc(cells=None):
    return {
        'schema_version': 'grad-hotspot-attribution-v1',
        'meta': {'gpu_name': 'RTX 4090',
                 'gpu4pyscf_source_commit': 'deadbeef',
                 'timestamp_utc': '2026-10-06T00:00:00Z'},
        'cells': cells if cells is not None else [
            {'cell': 'b3lyp_methanol_svp', 'xc': 'b3lyp',
             'geom': 'methanol.xyz', 'basis': 'def2-svp', 'nao': 48,
             'lanes': [_lane()]},
        ],
    }


class TestValidateDocument(unittest.TestCase):

    def test_compliant_document_passes(self):
        ok, issues = ghp.validate_document(_doc())
        self.assertTrue(ok, issues)
        self.assertEqual(issues, [])

    def test_v1_wrong_schema(self):
        doc = _doc()
        doc['schema_version'] = 'something-else'
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V1' for i in issues))

    def test_v1_missing_meta(self):
        doc = _doc()
        del doc['meta']['gpu4pyscf_source_commit']
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V1' for i in issues))

    def test_v2_too_few_instrumented_samples(self):
        lane = _lane()
        lane['instrumented']['samples_s'] = [0.5, 0.5]
        ok, issues = ghp.validate_document(_doc())
        # fix: build the doc with the modified lane
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V2' and
                            'instrumented' in i['detail'] for i in issues))

    def test_v2_missing_baseline(self):
        lane = _lane()
        lane['baseline']['samples_s'] = []
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V2' and 'baseline' in i['detail']
                            for i in issues))

    def test_v3_unattributed_gate(self):
        lane = _lane()
        lane['derived']['unattributed_frac'] = 0.11
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V3' for i in issues))

    def test_v3_boundary_10pct_passes(self):
        lane = _lane()
        lane['derived']['unattributed_frac'] = 0.10
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertTrue(ok, issues)

    def test_v4_wb97mv_requires_nlc_bucket(self):
        lane = _lane()
        lane['buckets'].pop('grad.nlc_vv10', None)
        doc = _doc(cells=[{'cell': 'wb97mv_benzene_svp', 'xc': 'wb97m-v',
                           'geom': 'benzene.xyz', 'basis': 'def2-svp',
                           'nao': 114, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V4' and 'nlc' in i['detail']
                            for i in issues))

    def test_v4_r2scan3c_requires_disp_bucket(self):
        lane = _lane()
        doc = _doc(cells=[{'cell': 'r2scan3c_benzene_s1', 'xc': 'r2scan3c',
                           'geom': 'benzene.xyz', 'basis': 'def2-mtzvpp',
                           'nao': 144, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V4' and 'disp' in i['detail']
                            for i in issues))

    def test_v4_missing_df_jk_bucket(self):
        lane = _lane()
        lane['buckets'].pop('grad.df_jk_ip1')
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V4' and 'df_jk' in i['detail']
                            for i in issues))

    def test_v4_e2_other_exploded(self):
        lane = _lane()
        lane['derived']['e2_other_frac_of_e2'] = 0.30
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V4' and 'e2_other' in i['detail']
                            for i in issues))

    def test_v5_fixed_tolerance_would_misflag_genuinely_noisy_lane(self):
        # the OLD fixed 1e-8 semantics would reject a lane whose own
        # uninstrumented repeats vary by 2e-7 (r2scan3c auto, measured);
        # the variance-aware rule must accept it
        lane = _lane()
        lane['baseline']['run_to_run_grad_var_max'] = 2.0e-7
        lane['equiv_max_abs_diff'] = 1.9e-7
        lane['equiv_tolerance'] = 3.0 * 2.0e-7
        lane['equiv_observation_only'] = True
        doc = _doc(cells=[{'cell': 'r2scan3c_benzene_s1', 'xc': 'r2scan3c',
                           'geom': 'benzene.xyz', 'basis': 'def2-mtzvpp',
                           'nao': 144, 'lanes': [lane]}])
        # give the 3c lane its disp bucket
        doc['cells'][0]['lanes'][0]['buckets']['grad.disp_d4_gcp'] = \
            _bucket(0.001)
        ok, issues = ghp.validate_document(doc)
        self.assertTrue(ok, issues)

    def test_v5_rejects_diff_beyond_variance(self):
        lane = _lane()
        lane['baseline']['run_to_run_grad_var_max'] = 1e-13
        lane['equiv_max_abs_diff'] = 5e-7
        lane['equiv_tolerance'] = 1e-8
        lane['equiv_observation_only'] = False
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V5' for i in issues))

    def test_v6_auto_lane_zero_fp32_hits_is_an_issue(self):
        # honesty rule: an auto lane that executed ZERO fp32 gradient-kernel
        # entries must be flagged (the measured lane may legitimately be
        # fp64 -- e.g. the UKS hybrid path -- but the document must say so
        # through an issue, forcing an explicit explanation rather than a
        # silent assumption)
        lane = _lane()
        lane['fp32_entry'] = {'grad_ip1_hits': 0,
                              'grad_mode_effective': 'fp64',
                              'note': 'measured'}
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V6' for i in issues))

    def test_v6_missing_fp32_entry_evidence(self):
        lane = _lane()
        lane['fp32_entry'] = {'note': 'oops'}
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V6' for i in issues))

    def test_v7_sync_overhead_required(self):
        lane = _lane()
        lane['derived']['sync_serialisation_overhead'] = None
        doc = _doc(cells=[{'cell': 'c', 'xc': 'b3lyp', 'geom': 'g',
                           'basis': 'b', 'nao': 1, 'lanes': [lane]}])
        ok, issues = ghp.validate_document(doc)
        self.assertFalse(ok)
        self.assertTrue(any(i['rule'] == 'V7' for i in issues))


@unittest.skipUnless(HAS_SOURCE_ENV,
                     'needs the gpu4pyscf source env (numpy/cupy)')
class TestGradientPatcher(unittest.TestCase):
    '''Install/restore against the REAL gpu4pyscf gradient class hierarchy
    with a stub timer (no GPU calls are made: nothing invokes the wrapped
    functions).'''

    def _stub_timer(self):
        timer = mock.Mock()
        calls = []

        def wrap(key, fn):
            def wrapper(*a, **kw):
                calls.append(key)
                return fn(*a, **kw)
            wrapper.__wrapped__ = fn
            return wrapper
        timer.wrap = wrap
        timer.calls = calls
        return timer

    def test_install_and_restore_roundtrip(self):
        from gpu4pyscf.df.grad import rhf as dg_rhf, uhf as dg_uhf
        from gpu4pyscf.df.grad import rks as dg_rks, uks as dg_uks
        from gpu4pyscf.grad import rhf as g_rhf
        originals = {
            'rhf_jk': dg_rhf.Gradients.__dict__['jk_energy_per_atom'],
            'uhf_jk': dg_uhf.Gradients.__dict__['jk_energy_per_atom'],
            'rks_jk': dg_rks.Gradients.__dict__['jk_energy_per_atom'],
            'uks_jk': dg_uks.Gradients.__dict__['jk_energy_per_atom'],
            'hcore': g_rhf.GradientsBase.__dict__['_hcore_energy'],
            'nuc': g_rhf.GradientsBase.__dict__['grad_nuc'],
        }
        # the four DF classes hold DISTINCT dict entries (rks/uks alias the
        # same FUNCTION OBJECT as rhf/uhf but in their own class dicts); the
        # patcher must wrap each dict entry around the ORIGINAL object, not
        # a previously wrapped one
        self.assertIs(dg_rks.Gradients.__dict__['jk_energy_per_atom'],
                      dg_rhf.Gradients.__dict__['jk_energy_per_atom'])
        self.assertIs(dg_uks.Gradients.__dict__['jk_energy_per_atom'],
                      dg_uhf.Gradients.__dict__['jk_energy_per_atom'])

        timer = self._stub_timer()
        patcher = ghp.GradientPatcher(timer)
        patcher.install()
        try:
            # every patched entry is a NEW object distinct from originals
            self.assertIsNot(dg_rks.Gradients.jk_energy_per_atom,
                             originals['rhf_jk'])
            self.assertIsNot(dg_uhf.Gradients.jk_energy_per_atom,
                             originals['uhf_jk'])
            # the wrapped entry delegates to the ORIGINAL function (no
            # double wrapping: rks's wrapper wraps the same original object
            # rhf's wrapper wraps -- calling through rks counts once)
            self.assertIs(dg_rks.Gradients.jk_energy_per_atom.__wrapped__,
                          originals['rhf_jk'])
        finally:
            patcher.restore()
        self.assertIs(dg_rhf.Gradients.__dict__['jk_energy_per_atom'],
                      originals['rhf_jk'])
        self.assertIs(dg_rks.Gradients.__dict__['jk_energy_per_atom'],
                      originals['rks_jk'])
        self.assertIs(g_rhf.GradientsBase.__dict__['_hcore_energy'],
                      originals['hcore'])
        self.assertIs(g_rhf.GradientsBase.__dict__['grad_nuc'],
                      originals['nuc'])

    def test_restore_is_idempotent(self):
        from gpu4pyscf.df.grad import rhf as dg_rhf
        orig = dg_rhf.Gradients.__dict__['energy_ee']
        timer = self._stub_timer()
        patcher = ghp.GradientPatcher(timer)
        patcher.install()
        patcher.restore()
        patcher.restore()  # second restore must be a no-op
        self.assertIs(dg_rhf.Gradients.__dict__['energy_ee'], orig)

    def test_module_patches_restored(self):
        from gpu4pyscf.grad import rks as g_rks, uks as g_uks
        orig_rks = g_rks.__dict__['get_exc']
        orig_uks = g_uks.__dict__['_get_denlc']
        timer = self._stub_timer()
        patcher = ghp.GradientPatcher(timer)
        patcher.install()
        try:
            self.assertIsNot(g_rks.get_exc, orig_rks)
            self.assertIsNot(g_uks._get_denlc, orig_uks)
        finally:
            patcher.restore()
        self.assertIs(g_rks.__dict__['get_exc'], orig_rks)
        self.assertIs(g_uks.__dict__['_get_denlc'], orig_uks)


if __name__ == '__main__':
    unittest.main()

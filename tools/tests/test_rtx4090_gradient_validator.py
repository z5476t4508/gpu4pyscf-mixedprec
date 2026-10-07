'''Unit tests for the rtx4090-gradient-v1 schema validator (issue #12).

The gradient contract (mixedprec/benchmark/rtx4090/schema_validator.py,
rules G1-G5 plus the gradient variants of R1/R2/R3) gets:
  * a compliant multi-lane document that must pass, and
  * targeted adversarial mutations that must be rejected with the right
    rule tag -- a fabricated pass over the gradient gate, an ineffective
    fp32 gradient lane (grad_ip1 == 0 on 'auto'), a contaminated fp64
    gradient lane (grad_ip1 > 0), a gradient delta recomputed from the
    stored gradients that does not match the stored number, a delta on a
    group with no fp64 reference (wrong tier), a missing t_grad_s, a
    silent pass on a cell whose energy gate has a known failure.

All tests are pure CPU: the validator is deliberately dependency-free.
'''

import importlib.util
import json
import pathlib
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sv = _load('schema_validator')

TINY_SHA = '0123456789abcdef' * 4

# a small 2-atom "gradient": reference and lanes differ by known deltas
G_REF = [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]
# auto / fp32-grad lanes sit 1e-6 Eh/Bohr away from the reference on one
# component (inside the 3e-5 gate); a fabricated 1e-3 delta is far outside
G_LANE_SMALL = [[0.01 + 1e-6, -0.02, 0.03], [-0.01, 0.02, -0.03]]
G_LANE_HUGE = [[0.01 + 1e-3, -0.02, 0.03], [-0.01, 0.02, -0.03]]


def base_meta():
    return {
        'gpu_name': 'NVIDIA GeForce RTX 4090',
        'gpu_total_mem_mib': 24564, 'driver_version': '595.58.03',
        'cuda_version': '12.8', 'cpu_model': 'test cpu', 'cpu_threads': 24,
        'ram_gb': 64.0, 'python': '3.12.3', 'cupy': '13.4.1',
        'pyscf': '2.14.0', 'gpu4pyscf_source_commit': '0' * 40,
        'gpu4pyscf_source_path': '/data/work/gpu4pyscf-mixedprec',
        'libxc': '7.0.0', 'mem_headroom_mib': 2048,
        'timestamp_utc': '2026-10-04T00:00:00Z', 'run_id': 'test-run',
    }


def _grad_norm(g):
    return max((x * x + y * y + z * z) ** 0.5 for x, y, z in g)


def _maxdg(g_lane, g_ref):
    return max(abs(g_lane[i][j] - g_ref[i][j])
               for i in range(len(g_ref)) for j in range(3))


def _rmsdg(g_lane, g_ref):
    n = len(g_ref) * 3
    return (sum((g_lane[i][j] - g_ref[i][j]) ** 2
                for i in range(len(g_ref))
                for j in range(3)) / n) ** 0.5


def base_row(lane, xc='b3lyp', grad=G_REF, grad_mode='fp64',
             grad_ip1=0, energy_gate=None):
    '''One ok gradient row.  grad_mode/grad_ip1 describe the gradient
    kernel lane: fp64 rows run no fp32 kernel, the auto/fp32-grad rows
    must show grad_ip1 > 0.'''
    hits = {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': grad_ip1}
    repeats = [
        {'index': i, 'e_tot': -114.5, 'converged': True, 'scf_cycles': 11,
         'gradient': grad, 'gradient_norm': _grad_norm(grad),
         'gradient_mode': grad_mode, 'grad_ip1_hits': grad_ip1,
         'sample_s': 1.0 + i * 0.1, 't_scf_s': 0.6, 't_grad_s': 0.4,
         'precision_switch': {'switched': False, 'cycle': None,
                              'trigger': 'none'}}
        for i in range(3)]
    return {
        'row_key': {'geom': 'methanol.xyz', 'basis': 'def2-svp', 'xc': xc,
                    'lane': lane},
        'pilot_stage': 'S0',
        'system': {'geom_file': 'methanol.xyz', 'geom_sha256': TINY_SHA,
                   'natm': 2, 'nao': 48, 'charge': 0, 'spin': 0, 'nelec': 18,
                   'cart': False, 'geom_units': 'angstrom'},
        'method': {'xc': xc, 'basis': 'def2-svp',
                   'xc_definition': {'xc': xc, 'libxc_decomposition':
                                     [{'id': 106, 'factor': 0.8}]},
                   'auxbasis': 'def2-universal-jkfit', 'with_df': True,
                   'grids': {'level': 3, 'atom_grid': None},
                   'nlcgrids': {'level': None, 'atom_grid': None},
                   'dispersion': None, 'gcp': None, 'conv_tol': 1e-10,
                   'conv_tol_grad': None, 'direct_scf_tol': 1e-14,
                   'init_guess': 'minao', 'max_cycle': 50,
                   'open_shell': False},
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': -114.5, 'converged': True, 'scf_cycles': 11,
            'gradient': grad, 'gradient_norm': _grad_norm(grad),
            'gradient_mode': grad_mode,
            'repeats': repeats,
            'repeat_evidence': 'all-repeats',
            'e_terms': {'e1': -185.0, 'coul': 55.0, 'exc': -14.0,
                        'nlc': None, 'dispersion': None, 'gcp': None},
            'timings': {'warmup_s': 1.0, 'samples_s': [1.0, 1.1, 1.2],
                        'median_s': 1.1, 'min_s': 1.0, 'max_s': 1.2,
                        'end_to_end_s': 1.1, 't_cderi_s': 0.5,
                        't_scf_s': 0.6, 't_grad_s': 0.4,
                        't_xc_grid_s': None, 't_jk_s': None,
                        't_nlc_s': None, 't_disp_s': None,
                        'sync': 'device-sync before/after each phase'},
            'mem': {'gpu_free_before_mib': 24000.0,
                    'gpu_total_mib': 24564.0, 'gpu_peak_alloc_mib': 500.0,
                    'host_rss_peak_mib': 1200.0,
                    'headroom_budget_mib': 2048.0, 'within_budget': True},
            'fp32_entry_hits': hits,
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'fallback_events': [], 'errors': None,
            **({'energy_gate_reference': energy_gate}
               if energy_gate is not None else {})},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [],
        'evidence': {'raw_log': f'/tmp/x@{TINY_SHA}',
                     'result_json': f'/tmp/y@{TINY_SHA}',
                     'command': 'python gradient_runner.py --out x.json'},
    }


def compliant_grad_doc(xc='b3lyp'):
    '''fp64 + auto + fp32-grad + auto-fp64-grad + cpu-fp64 lanes of one
    cell, all consistent: the auto/fp32-grad lanes sit 1e-6 Eh/Bohr from
    the reference (inside the gate), the cpu sentinel 1e-8 (historical
    implementation level, below the note threshold).'''
    rows = [base_row('fp64', xc, G_REF, 'fp64', 0),
            base_row('auto', xc, G_LANE_SMALL, 'fp32', 12),
            base_row('fp32-grad', xc, G_LANE_SMALL, 'fp32', 12),
            base_row('auto-fp64-grad', xc, G_LANE_SMALL, 'fp64', 0),
            base_row('cpu-fp64', xc, G_LANE_SMALL, 'fp64', 0)]
    by_lane = {r['row_key']['lane']: r for r in rows}
    # auto SCF hit the fp32 df_jk branch during its head
    by_lane['auto']['result']['fp32_entry_hits']['df_jk'] = 6
    by_lane['auto']['result']['precision_switch'] = {
        'switched': True, 'cycle': 5, 'trigger': 'dE'}
    for lane in ('auto', 'fp32-grad', 'auto-fp64-grad'):
        acc = by_lane[lane]['accuracy']
        acc['max_abs_dg_vs_gpu_fp64'] = _maxdg(G_LANE_SMALL, G_REF)
        acc['rms_dg_vs_gpu_fp64'] = _rmsdg(G_LANE_SMALL, G_REF)
        acc['dE_vs_gpu_fp64'] = 1e-12
        acc['dE_vs_cpu_fp64'] = 1e-11
        acc['verdict'] = 'pass'
    # the cpu sentinel: tier-(2) only -- no same-implementation reference
    # for its own row, so its verdict is no_reference (converged sentinel,
    # never gated); the cross-implementation gradient delta is reported as
    # the diagnostic other lanes' G4 notes compare against
    acc = by_lane['cpu-fp64']['accuracy']
    acc['max_abs_dg_vs_gpu_fp64'] = None
    acc['rms_dg_vs_gpu_fp64'] = None
    acc['dE_vs_gpu_fp64'] = None
    acc['dE_vs_cpu_fp64'] = None      # this row IS the cpu reference
    acc['verdict'] = 'no_reference'
    # the fp64 reference row passes on convergence alone; its own deltas
    # are null (G2 checks the keys exist, G3 checks reference self-delta
    # is null/zero)
    ref_acc = by_lane['fp64']['accuracy']
    ref_acc['max_abs_dg_vs_gpu_fp64'] = None
    ref_acc['rms_dg_vs_gpu_fp64'] = None
    ref_acc['verdict'] = 'pass'
    return {'schema_version': 'rtx4090-gradient-v1', 'meta': base_meta(),
            'rows': rows}


def _fix(doc, lane, **kw):
    doc = json.loads(json.dumps(doc))
    for row in doc['rows']:
        if row['row_key']['lane'] == lane:
            for k, v in kw.items():
                if k.startswith('result.'):
                    row['result'][k.split('.', 1)[1]] = v
                elif k.startswith('accuracy.'):
                    row['accuracy'][k.split('.', 1)[1]] = v
                elif k.startswith('timings.'):
                    row['result']['timings'][k.split('.', 1)[1]] = v
                else:
                    row[k] = v
    return doc


def _tags(issues):
    return {(it['rule'], it['tag']) for it in issues}


class CompliantGradientDocumentTests(unittest.TestCase):
    def test_compliant_document_passes(self):
        issues = sv.validate_gradient_document(
            compliant_grad_doc(), evidence_root=None)
        self.assertEqual(issues, [], msg=json.dumps(issues, indent=1))

    def test_dispatch_on_schema_version(self):
        doc = compliant_grad_doc()
        self.assertEqual(sv.validate_any(doc, evidence_root=None), [])
        energy = {'schema_version': 'rtx4090-energy-v1'}
        # energy contract must NOT accept gradient lanes
        grad_only = compliant_grad_doc()
        issues = sv.validate_any(grad_only, evidence_root=None)
        self.assertEqual(issues, [])
        self.assertNotEqual(energy.get('schema_version'),
                            grad_only['schema_version'])


class GradientRuleTests(unittest.TestCase):
    def test_fabricated_pass_over_gate(self):
        '''A lane 1e-3 Eh/Bohr away from the reference labeled pass must
        fail the gradient gate (R2 grad-gate-exceeded), not slip through.'''
        doc = compliant_grad_doc()
        # swap the auto lane's gradient to the huge delta, keep the verdict
        doc = _fix(doc, 'auto', **{'result.gradient': G_LANE_HUGE,
                                   'result.gradient_norm':
                                       _grad_norm(G_LANE_HUGE),
                                   'accuracy.max_abs_dg_vs_gpu_fp64':
                                       _maxdg(G_LANE_HUGE, G_REF),
                                   'accuracy.rms_dg_vs_gpu_fp64':
                                       _rmsdg(G_LANE_HUGE, G_REF)})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('R2', 'grad-gate-exceeded'), _tags(issues))

    def test_gate_number_must_match_stored_gradients(self):
        '''The stored max|dg| must be recomputable from the stored
        gradient and the reference row's stored gradient (G3).'''
        doc = _fix(compliant_grad_doc(), 'auto',
                   **{'accuracy.max_abs_dg_vs_gpu_fp64': 1e-9})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G3', 'grad-delta-inconsistent'), _tags(issues))

    def test_pass_without_delta_number(self):
        '''A passing non-reference lane with no numeric max|dg| must be
        rejected (R2 grad-pass-without-reference).'''
        doc = _fix(compliant_grad_doc(), 'auto',
                   **{'accuracy.max_abs_dg_vs_gpu_fp64': None})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        tags = _tags(issues)
        self.assertIn(('R2', 'grad-pass-without-reference'), tags)
        self.assertIn(('G3', 'grad-no-reference'), tags)

    def test_ineffective_auto_gradient_lane(self):
        '''An ok 'auto' gradient row with grad_ip1 == 0 never executed the
        fp32 gradient kernel -- the same wheel-shadowing regression as
        energy rule R3, on the gradient side.'''
        doc = _fix(compliant_grad_doc(), 'auto',
                   **{'result.fp32_entry_hits':
                      {'df_jk': 6, 'eval_rho': 0, 'xc_grid': 0,
                       'grad_ip1': 0}})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        tags = _tags(issues)
        self.assertIn(('R3', 'grad-lane-ineffective'), tags)
        self.assertIn(('R3', 'grad-lane-ineffective-pass'), tags)

    def test_contaminated_fp64_gradient_lane(self):
        '''An fp64 gradient row showing grad_ip1 > 0 executed the fp32
        kernel although it claims fp64 kernels -- a mislabeled lane.'''
        doc = _fix(compliant_grad_doc(), 'fp64',
                   **{'result.fp32_entry_hits':
                      {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                       'grad_ip1': 12}})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('R3', 'grad-lane-contaminated'), _tags(issues))

    def test_gradient_mode_unrecorded(self):
        doc = _fix(compliant_grad_doc(), 'fp64',
                   **{'result.gradient_mode': None})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('R3', 'grad-mode-unrecorded'), _tags(issues))

    def test_missing_t_grad_s(self):
        '''The gradient phase must be timed separately from the SCF
        (R1 grad-phase-timing).'''
        doc = _fix(compliant_grad_doc(), 'fp64', **{'timings.t_grad_s':
                                                    None})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('R1', 'grad-phase-timing'), _tags(issues))

    def test_missing_gradient_payload(self):
        doc = _fix(compliant_grad_doc(), 'fp64', **{'result.gradient':
                                                    None})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G1', 'gradient-missing'), _tags(issues))

    def test_gradient_shape_mismatch(self):
        doc = _fix(compliant_grad_doc(), 'fp64',
                   **{'result.gradient': [[0.0, 0.0, 0.0]]})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G1', 'gradient-shape'), _tags(issues))

    def test_gradient_norm_inconsistent(self):
        doc = _fix(compliant_grad_doc(), 'fp64',
                   **{'result.gradient_norm': 42.0})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G1', 'gradient-norm-inconsistent'), _tags(issues))

    def test_delta_without_fp64_reference(self):
        '''A group with no ok fp64 reference must not carry a numeric
        same-implementation delta -- it could only have been computed
        against the wrong tier (G3).'''
        doc = compliant_grad_doc()
        for row in doc['rows']:
            if row['row_key']['lane'] == 'fp64':
                row['status'] = 'error'
                row['result'] = {'errors': {'type': 'RuntimeError',
                                            'message': 'boom'}}
                row['accuracy']['verdict'] = 'fail'
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G3', 'grad-delta-without-reference'), _tags(issues))

    def test_grad_impl_unexplained(self):
        '''A cpu-sentinel gradient deviation beyond the note threshold
        without a named note (G4).'''
        doc = _fix(compliant_grad_doc(), 'auto',
                   **{'accuracy.max_abs_dg_vs_cpu_fp64': 1e-4,
                      'accuracy.impl_agreement_note': None})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G4', 'grad-impl-unexplained'), _tags(issues))

    def test_silent_pass_on_failed_energy_gate(self):
        '''A cell whose energy gate has a known failure (the inherited
        #7 methoxy UKS cell) must not show a silent gradient pass: the
        verdict needs an explicit provisional note (G5).'''
        gate = {'source_issue': 7, 'document':
                '/data/results/.../wb97mv_methoxy_svp.json',
                'gate_failed_cells': ['methoxy.xyz/def2-svp/wb97m-v'],
                'failed_cell_worst_dE_ha': -1.283e-6}
        doc = compliant_grad_doc(xc='wb97m-v')
        doc = _fix(doc, 'auto',
                   **{'result.energy_gate_reference': gate})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('G5', 'grad-pass-with-failed-energy-gate'),
                      _tags(issues))

    def test_provisional_note_satisfies_g5(self):
        '''The same cell WITH the provisional note passes G5: gradient
        validation is explicitly pending on the energy fix, never a
        fabricated acceptance.'''
        gate = {'source_issue': 7, 'document':
                '/data/results/.../wb97mv_methoxy_svp.json',
                'gate_failed_cells': ['methoxy.xyz/def2-svp/wb97m-v'],
                'failed_cell_worst_dE_ha': -1.283e-6}
        doc = compliant_grad_doc(xc='wb97m-v')
        doc = _fix(doc, 'auto',
                   **{'result.energy_gate_reference': gate,
                      'accuracy.verdict_note':
                          'provisional: energy gate of this cell has an '
                          'unresolved failure (issue #7 -1.283e-6 Ha); '
                          'gradient validation pending'})
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertNotIn(('G5', 'grad-pass-with-failed-energy-gate'),
                         _tags(issues))

    def test_unknown_gradient_lane_rejected(self):
        '''Energy-only lanes (fp32-screen, df-vs-conv) are not gradient
        lanes; a document using them under the gradient schema is
        rejected.'''
        doc = compliant_grad_doc()
        doc['rows'][0]['row_key']['lane'] = 'fp32-screen'
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('structure', 'lane-enum'), _tags(issues))

    def test_r2_shared_semantics_still_apply(self):
        '''A non-ok row without skip_reason/errors is still rejected by
        the shared R2 semantics.'''
        doc = compliant_grad_doc()
        row = doc['rows'][0]
        row['status'] = 'error'
        row['result'] = {'mem': {'gpu_free_before_mib': 1.0,
                                 'gpu_total_mib': 2.0}}
        row['accuracy']['verdict'] = 'fail'
        row['skip_reason'] = None
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertIn(('R2', 'non-ok-unexplained'), _tags(issues))


class GradientValidatorCLITests(unittest.TestCase):
    def test_cli_dispatches_on_schema_version(self):
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'grad.json'
            path.write_text(json.dumps(compliant_grad_doc()))
            rc = sv.main([str(path), '--evidence-root', ''])
            self.assertEqual(rc, 0)


if __name__ == '__main__':
    unittest.main()

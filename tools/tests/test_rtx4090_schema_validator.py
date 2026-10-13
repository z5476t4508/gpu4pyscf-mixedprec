"""Unit tests for the rtx4090-energy-v1 schema validator (issue #4).

Every mandatory rule of docs/consumer-gpu-plan.md section 4 gets:
  * a compliant document that must pass, and
  * targeted violations (one per rule, plus adversarial corners) that must
    be rejected with the right rule tag.

Adversarial cases are built from the compliant one by minimal mutations,
so a rejection is attributable to exactly one defect (no accidental
double-failures masking each other).
"""

import importlib.util
import json
import os
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

TINY_LOG = 'tiny log line\n'
TINY_SHA = '0123456789abcdef' * 4


def base_meta():
    return {
        'gpu_name': 'NVIDIA GeForce RTX 4090',
        'gpu_total_mem_mib': 24564, 'driver_version': '595.58.03',
        'cuda_version': '12.8', 'cpu_model': 'test cpu', 'cpu_threads': 24,
        'ram_gb': 64.0, 'python': '3.12.3', 'cupy': '13.4.1',
        'pyscf': '2.14.0', 'gpu4pyscf_source_commit': '0' * 40,
        'gpu4pyscf_source_path': '/data/work/gpu4pyscf-mixedprec',
        'libxc': '7.0.0', 'mem_headroom_mib': 2048,
        'timestamp_utc': '2026-10-03T00:00:00Z', 'run_id': 'test-run',
    }


def base_row(lane='fp64', xc='hf', basis='def2-svp', geom='methanol.xyz'):
    return {
        'row_key': {'geom': geom, 'basis': basis, 'xc': xc, 'lane': lane},
        'pilot_stage': 'S0',
        'system': {'geom_file': geom, 'geom_sha256': TINY_SHA, 'natm': 6,
                   'nao': 48, 'charge': 0, 'spin': 0, 'nelec': 18,
                   'cart': False, 'geom_units': 'angstrom'},
        'method': {'xc': xc, 'basis': basis,
                   'xc_definition': None,
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
            'e_terms': {'e1': -185.0, 'coul': 55.0, 'exc': 0.0, 'nlc': None,
                        'dispersion': None, 'gcp': None},
            'timings': {'warmup_s': 1.0, 'samples_s': [1.0, 1.1, 1.2],
                        'median_s': 1.1, 'min_s': 1.0, 'max_s': 1.2,
                        'end_to_end_s': 1.1, 't_cderi_s': 0.5,
                        't_scf_s': 0.6, 't_xc_grid_s': None, 't_jk_s': None,
                        't_nlc_s': None, 't_disp_s': None,
                        'sync': 'device-sync before/after each sample'},
            'mem': {'gpu_free_before_mib': 24000.0,
                    'gpu_total_mib': 24564.0, 'gpu_peak_alloc_mib': 500.0,
                    'host_rss_peak_mib': 1200.0, 'headroom_budget_mib': 2048.0,
                    'within_budget': True},
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                'grad_ip1': 0},
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': 'pass'},
        'cross_program': [],
        'evidence': {'raw_log': f'/tmp/x@{TINY_SHA}',
                     'result_json': f'/tmp/y@{TINY_SHA}',
                     'command': 'python rtx4090_runner.py --out x.json'},
    }


def compliant_doc():
    doc = {'schema_version': 'rtx4090-energy-v1', 'meta': base_meta(),
           'rows': [base_row('fp64'), base_row('auto'),
                    base_row('cpu-fp64'), base_row('df-vs-conv')]}
    by_lane = {r['row_key']['lane']: r for r in doc['rows']}
    # the auto lane actually hit fp32 branches and sits inside the gate
    by_lane['auto']['result']['fp32_entry_hits'] = {
        'df_jk': 6, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0}
    by_lane['auto']['result']['precision_switch'] = {
        'switched': True, 'cycle': 5, 'trigger': 'dE'}
    by_lane['auto']['accuracy']['dE_vs_gpu_fp64'] = 1.1e-13
    by_lane['auto']['accuracy']['dE_vs_cpu_fp64'] = 9e-11
    # the cpu sentinel agrees with the gpu reference to historical level
    by_lane['cpu-fp64']['accuracy']['dE_vs_gpu_fp64'] = 2e-11
    # the DF-approximation delta lives ONLY on the df-vs-conv row
    by_lane['df-vs-conv']['result']['e_tot'] = -114.48
    by_lane['df-vs-conv']['accuracy']['dE_df_approximation'] = 0.02
    return doc


def _fix(doc, lane, **kw):
    '''Mutate one row in a deep copy; return the copy.'''
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
                elif k.startswith('mem.'):
                    row['result']['mem'][k.split('.', 1)[1]] = v
                elif k.startswith('row_key.'):
                    row['row_key'][k.split('.', 1)[1]] = v
                else:
                    row[k] = v
    return doc


class CompliantDocumentTests(unittest.TestCase):
    def test_compliant_document_passes(self):
        doc = compliant_doc()
        # make the tier-(2) field benign on the cpu lane (self-reference)
        issues = sv.validate_document(doc, evidence_root=None)
        self.assertEqual(issues, [], f'unexpected issues: {issues}')

    def test_rows_key_must_be_nonempty_list(self):
        issues = sv.validate_document({'schema_version':
                                       'rtx4090-energy-v1', 'rows': []})
        self.assertTrue(any(i['tag'] == 'rows' for i in issues))

    def test_schema_version_enforced(self):
        doc = compliant_doc()
        doc['schema_version'] = 'rtx5090-energy-v1'
        self.assertTrue(any(i['tag'] == 'schema-version'
                            for i in sv.validate_document(doc)))

    def test_missing_meta_field(self):
        doc = compliant_doc()
        del doc['meta']['gpu4pyscf_source_commit']
        self.assertTrue(any(i['tag'] == 'meta-field'
                            for i in sv.validate_document(doc)))

    def test_duplicate_row_key(self):
        doc = compliant_doc()
        doc['rows'].append(json.loads(json.dumps(doc['rows'][0])))
        self.assertTrue(any(i['tag'] == 'duplicate-row-key'
                            for i in sv.validate_document(doc)))


class Rule1TimingTests(unittest.TestCase):
    def test_two_samples_rejected(self):
        doc = _fix(compliant_doc(), 'fp64', **{'timings.samples_s':
                                               [1.0, 1.1]})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('samples-count', tags)

    def test_warmup_mixed_into_samples_is_not_detectable_but_stats_are(self):
        doc = _fix(compliant_doc(), 'fp64', **{'timings.median_s': 3.0})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('sample-statistics', tags)

    def test_missing_warmup(self):
        doc = _fix(compliant_doc(), 'fp64', **{'timings.warmup_s': None})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('warmup-missing', tags)

    def test_missing_end_to_end(self):
        doc = _fix(compliant_doc(), 'fp64', **{'timings.end_to_end_s': None})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('end-to-end', tags)

    def test_gpu_lane_without_device_sync(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'timings.sync': 'no sync at all'})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('sync-evidence', tags)

    def test_cpu_lane_sync_wording_accepted(self):
        doc = _fix(compliant_doc(), 'cpu-fp64',
                   **{'timings.sync': 'cpu lane: wall-clock samples'})
        issues = sv.validate_document(doc)
        self.assertFalse(any(i['tag'] == 'sync-evidence' for i in issues))


class Rule2StatusTests(unittest.TestCase):
    def test_skipped_row_needs_reason(self):
        doc = _fix(compliant_doc(), 'fp64', status='skipped',
                   skip_reason=None)
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('skip-reason-missing', tags)
        self.assertIn('non-ok-unexplained', tags)

    def test_oom_row_needs_errors(self):
        doc = _fix(compliant_doc(), 'fp64', status='oom')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('errors-missing', tags)

    def test_pass_with_exceeded_gate(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'accuracy.dE_vs_gpu_fp64': 5e-6})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('gate-exceeded', tags)

    def test_pass_without_convergence(self):
        doc = _fix(compliant_doc(), 'fp64', **{'result.converged': False})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('pass-not-converged', tags)

    def test_pass_on_skipped_row(self):
        doc = _fix(compliant_doc(), 'fp64', status='skipped',
                   skip_reason='headroom-budget',
                   **{'accuracy.verdict': 'pass'})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('pass-on-non-ok', tags)

    def test_fp64_reference_self_delta(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'accuracy.dE_vs_gpu_fp64': 1e-9})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('fp64-self-delta', tags)


class Rule3LaneEffectivenessTests(unittest.TestCase):
    def test_auto_lane_zero_hits_fails(self):
        doc = _fix(compliant_doc(), 'auto', **{'result.fp32_entry_hits':
                                               {'df_jk': 0, 'eval_rho': 0,
                                                'xc_grid': 0, 'grad_ip1': 0}})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('lane-ineffective', tags)

    def test_auto_lane_zero_hits_and_pass_doubly_flagged(self):
        doc = _fix(compliant_doc(), 'auto', **{'result.fp32_entry_hits':
                                               {'df_jk': 0, 'eval_rho': 0,
                                                'xc_grid': 0, 'grad_ip1': 0},
                                               'accuracy.verdict': 'pass'})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('lane-ineffective', tags)
        self.assertIn('lane-ineffective-pass', tags)

    def test_fp64_lane_zero_hits_is_fine(self):
        doc = _fix(compliant_doc(), 'fp64', **{'result.fp32_entry_hits':
                                               {'df_jk': 0, 'eval_rho': 0,
                                                'xc_grid': 0, 'grad_ip1': 0}})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertNotIn('lane-ineffective', tags)


class Rule4MethodCompletenessTests(unittest.TestCase):
    def test_r2scan3c_missing_gcp(self):
        doc = _fix(compliant_doc(), 'fp64', xc='r2scan-3c',
                   **{'row_key.xc': 'r2scan-3c'})
        doc['rows'][0]['method']['xc'] = 'r2scan-3c'
        doc['rows'][0]['method']['basis'] = 'def2-mTZVPP'
        doc['rows'][0]['row_key']['basis'] = 'def2-mTZVPP'
        doc['rows'][0]['result']['e_terms']['dispersion'] = -0.01
        doc['rows'][0]['result']['e_terms']['gcp'] = None
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('r2scan3c-terms-missing', tags)

    def test_r2scan3c_wrong_basis(self):
        doc = _fix(compliant_doc(), 'fp64', xc='r2scan-3c',
                   **{'row_key.xc': 'r2scan-3c'})
        doc['rows'][0]['method']['xc'] = 'r2scan-3c'
        doc['rows'][0]['result']['e_terms']['dispersion'] = -0.01
        doc['rows'][0]['result']['e_terms']['gcp'] = -0.02
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('r2scan3c-basis', tags)

    def test_omegab97mv_missing_nlc(self):
        doc = _fix(compliant_doc(), 'fp64', xc='wb97m-v',
                   **{'row_key.xc': 'wb97m-v'})
        doc['rows'][0]['method']['xc'] = 'wb97m-v'
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('nlc-missing', tags)

    def test_omegab97mv_with_nlc_passes(self):
        doc = _fix(compliant_doc(), 'fp64', xc='wb97m-v',
                   **{'row_key.xc': 'wb97m-v'})
        doc['rows'][0]['method']['xc'] = 'wb97m-v'
        doc['rows'][0]['result']['e_terms']['nlc'] = -0.02
        issues = sv.validate_document(doc)
        self.assertFalse(any(i['tag'] == 'nlc-missing' for i in issues))

    def test_dft_row_without_xc_definition_rejected(self):
        doc = _fix(compliant_doc(), 'fp64', xc='b3lyp',
                   **{'row_key.xc': 'b3lyp'})
        doc['rows'][0]['method']['xc'] = 'b3lyp'
        # method.xc_definition is None in the fixture -> DFT rows must carry
        # the resolved variant contract
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('xc-definition-missing', tags)

    def test_dft_row_with_xc_definition_accepted(self):
        doc = _fix(compliant_doc(), 'fp64', xc='b3lyp',
                   **{'row_key.xc': 'b3lyp'})
        doc['rows'][0]['method']['xc'] = 'b3lyp'
        doc['rows'][0]['method']['xc_definition'] = {
            'xc': 'b3lyp', 'libxc_decomposition': [{'id': 402,
                                                    'factor': 1.0}],
            'hyb_coeff': 0.2, 'rsh': {'omega': 0.0, 'alpha': 0.0,
                                      'beta': 0.2},
            'nlc_params': None}
        issues = sv.validate_document(doc)
        self.assertFalse(any(i['tag'] == 'xc-definition-missing'
                             for i in issues))

    def test_dft_row_with_empty_decomposition_rejected(self):
        doc = _fix(compliant_doc(), 'fp64', xc='b3lyp',
                   **{'row_key.xc': 'b3lyp'})
        doc['rows'][0]['method']['xc'] = 'b3lyp'
        doc['rows'][0]['method']['xc_definition'] = {
            'xc': 'b3lyp', 'libxc_decomposition': None,
            'hyb_coeff': 0.2, 'rsh': None, 'nlc_params': None}
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('xc-definition-missing', tags)

    def test_hf_row_exempt_from_xc_definition(self):
        # the compliant fixture is all-hf with xc_definition=None and passes
        issues = sv.validate_document(compliant_doc())
        self.assertFalse(any(i['tag'] == 'xc-definition-missing'
                             for i in issues))


class Rule5TierIsolationTests(unittest.TestCase):
    def test_df_approximation_on_precision_lane_rejected(self):
        doc = _fix(compliant_doc(), 'auto',
                   **{'accuracy.dE_df_approximation': 0.01})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('df-approx-misplaced', tags)

    def test_cross_program_without_matched_flag(self):
        doc = compliant_doc()
        doc['rows'][0]['cross_program'] = [{
            'program': 'g16', 'method_key': 'RB3LYP/6-31G*',
            'e_tot': -115.7, 'dE': 1.2}]
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('cross-program', tags)


class Rule7MemoryTests(unittest.TestCase):
    def test_mem_block_missing(self):
        doc = _fix(compliant_doc(), 'fp64', **{'result.mem': None})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('mem-block-missing', tags)

    def test_mem_field_missing(self):
        doc = json.loads(json.dumps(compliant_doc()))
        del doc['rows'][0]['result']['mem']['gpu_peak_alloc_mib']
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('mem-field', tags)

    def test_budget_breach_without_explanation(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'mem.within_budget': False})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('budget-breach-unexplained', tags)

    def test_budget_breach_with_explanation_accepted(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'mem.within_budget': False})
        doc['rows'][0]['result']['mem']['budget_explanation'] = \
            'measured peak 23100 MiB + headroom 2048 exceeds total; ' \
            'recorded as measured, escalation logged'
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertNotIn('budget-breach-unexplained', tags)

    def test_skipped_row_needs_measured_free(self):
        doc = _fix(compliant_doc(), 'fp64', status='skipped',
                   skip_reason='headroom-budget')
        doc['rows'][0]['result']['mem'] = {'gpu_total_mib': 24564.0}
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('mem-block-skipped', tags)


class Rule8ImplAgreementTests(unittest.TestCase):
    def test_unexplained_cpu_deviation(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'accuracy.dE_vs_cpu_fp64': 1e-7})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('impl-unexplained', tags)

    def test_explained_cpu_deviation_accepted(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'accuracy.dE_vs_cpu_fp64': 1e-7,
                      'accuracy.impl_agreement_note':
                          'libxc version differs from CPU build (3.0 vs '
                          '7.0); screening defaults matched'})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertNotIn('impl-unexplained', tags)

    def test_tiny_cpu_deviation_needs_no_note(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'accuracy.dE_vs_cpu_fp64': 1e-11})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertNotIn('impl-unexplained', tags)


class Rule6EvidenceTests(unittest.TestCase):
    def test_bad_evidence_format(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'evidence': {'raw_log': 'no-at-sign',
                                   'result_json': f'/tmp/y@{TINY_SHA}',
                                   'command': 'cmd'}})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('evidence-format', tags)

    def test_evidence_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            ev = os.path.join(td, 'raw.json')
            with open(ev, 'w') as f:
                f.write('content')
            import hashlib
            real = hashlib.sha256(b'content').hexdigest()
            doc = _fix(compliant_doc(), 'fp64',
                       **{'evidence': {
                           'raw_log': f'{ev}@{"f" * 64}',
                           'result_json': f'{ev}@{real}',
                           'command': 'cmd'}})
            tags = {i['tag'] for i in sv.validate_document(doc,
                                                           evidence_root=td)}
            self.assertIn('evidence-hash-mismatch', tags)

    def test_evidence_outside_root_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            ev = os.path.join(td, 'raw.json')
            with open(ev, 'w') as f:
                f.write('content')
            import hashlib
            real = hashlib.sha256(b'content').hexdigest()
            doc = _fix(compliant_doc(), 'fp64',
                       **{'evidence': {
                           'raw_log': f'{ev}@{real}',
                           'result_json': f'{ev}@{real}',
                           'command': 'cmd'}})
            tags = {i['tag'] for i in sv.validate_document(
                doc, evidence_root='/data/results')}
            self.assertIn('evidence-outside-root', tags)

    def test_missing_repro_command(self):
        doc = _fix(compliant_doc(), 'fp64',
                   **{'evidence': {'raw_log': f'/tmp/x@{TINY_SHA}',
                                   'result_json': f'/tmp/y@{TINY_SHA}',
                                   'command': ''}})
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('repro-command', tags)


class Rule9GridStabilityTests(unittest.TestCase):
    '''The optional grid_stability block (issue #6 meta-GGA grid check):
    internally consistent when present, absent is fine, and every stored
    delta must be recomputable from the stored energies.'''

    def _gs_doc(self, **gs):
        doc = compliant_doc()
        doc['grid_stability'] = gs
        return doc

    def _entries(self, levels=(3, 4, 5), e=( -100.0, -100.000001, -100.0000011)):
        return [{'level': lv, 'ngrids': 1000 * lv, 'e_tot': ee,
                 'converged': True, 'scf_cycles': 10, 't_s': 1.0}
                for lv, ee in zip(levels, e)]

    def test_absent_block_is_compliant(self):
        self.assertEqual(sv.validate_document(compliant_doc()), [])

    def test_consistent_block_passes(self):
        rd = _load('rtx4090_runner')
        entries = [
            {'level': 3, 'ngrids': 3000, 'e_tot': -100.0, 'converged': True,
             'scf_cycles': 10, 't_s': 1.0},
            {'level': 4, 'ngrids': 4000, 'e_tot': -100.000001,
             'converged': True, 'scf_cycles': 11, 't_s': 1.3},
            {'level': 5, 'ngrids': 5000, 'e_tot': -100.0000011,
             'converged': True, 'scf_cycles': 11, 't_s': 1.7}]
        block = rd.assemble_grid_stability(entries, 3)
        doc = self._gs_doc(**block)
        self.assertEqual(sv.validate_document(doc), [])

    def test_production_level_must_be_scanned(self):
        doc = self._gs_doc(
            production_level=3,
            entries=self._entries(levels=(4, 5), e=(-100.0, -100.000001)),
            classification='converging', note='grid error recorded')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-production', tags)

    def test_empty_entries_rejected(self):
        doc = self._gs_doc(production_level=3, entries=[],
                           classification='single-level', note='x')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-entries', tags)

    def test_bad_classification_rejected(self):
        doc = self._gs_doc(production_level=3,
                           entries=self._entries(levels=(3, 4),
                                                 e=(-100.0, -100.000001)),
                           classification='converged', note='x')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-classification', tags)

    def test_missing_note_rejected(self):
        doc = self._gs_doc(production_level=3,
                           entries=self._entries(levels=(3, 4),
                                                 e=(-100.0, -100.000001)),
                           classification='converging', note='')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-note', tags)

    def test_transcribed_delta_mismatch_rejected(self):
        # the stored deltas disagree with the stored energies: transcription,
        # not measurement
        entries = self._entries(levels=(3, 4), e=(-100.0, -100.000001))
        doc = self._gs_doc(production_level=3,
                           entries=[{**entries[0], 'dE_vs_production': 0.0,
                                     'dE_vs_prev_level': None},
                                    {**entries[1],
                                     'dE_vs_prev_level': -1e-9}],
                           classification='converging', note='x')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-delta', tags)

    def test_entry_without_energy_needs_error_record(self):
        entries = self._entries(levels=(3, 4), e=(-100.0, -100.000001))
        entries.append({'level': 6, 'ngrids': None, 'e_tot': None,
                        'converged': False, 'scf_cycles': None, 't_s': None})
        doc = self._gs_doc(production_level=3, entries=entries,
                           classification='converging', note='x')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-entry', tags)

    def test_failed_level_with_error_record_accepted(self):
        entries = self._entries(levels=(3, 4), e=(-100.0, -100.000001))
        entries.append({'level': 6, 'ngrids': None, 'e_tot': None,
                        'converged': False, 'scf_cycles': None, 't_s': None,
                        'error': 'OutOfMemoryError: ...'})
        doc = self._gs_doc(production_level=3, entries=entries,
                           classification='converging', note='x')
        # the failed level carries no energy, so it is exempt from delta
        # recomputation; the block stays compliant
        self.assertEqual(sv.validate_document(doc), [])

    def test_grid_block_never_gates_rows(self):
        # a wildly not-converging scan must not fail any row verdict: the
        # precision gate compares lanes at identical grids (R9 is diagnostic)
        rd = _load('rtx4090_runner')
        entries = [
            {'level': 3, 'ngrids': 3000, 'e_tot': -100.0, 'converged': True,
             'scf_cycles': 10, 't_s': 1.0},
            {'level': 4, 'ngrids': 4000, 'e_tot': -99.9, 'converged': True,
             'scf_cycles': 12, 't_s': 1.3},
            {'level': 5, 'ngrids': 5000, 'e_tot': -99.7, 'converged': True,
             'scf_cycles': 12, 't_s': 1.7}]
        block = rd.assemble_grid_stability(entries, 3)
        self.assertEqual(block['classification'], 'not-converging')
        doc = self._gs_doc(**block)
        self.assertEqual(sv.validate_document(doc), [])


class Rule9NlcgridsStabilityTests(unittest.TestCase):
    '''The optional nlcgrids_stability block (issue #7 VV10 grid check):
    same internal-consistency contract as grid_stability.'''

    def _doc(self, **gs):
        doc = compliant_doc()
        doc['nlcgrids_stability'] = gs
        return doc

    def test_absent_block_is_compliant(self):
        self.assertEqual(sv.validate_document(compliant_doc()), [])

    def test_consistent_block_passes(self):
        rd = _load('rtx4090_runner')
        entries = [
            {'level': 2, 'ngrids': 2000, 'e_tot': -100.0, 'converged': True,
             'scf_cycles': 10, 't_s': 1.0},
            {'level': 3, 'ngrids': 3000, 'e_tot': -100.0000001,
             'converged': True, 'scf_cycles': 10, 't_s': 1.2}]
        block = rd.assemble_grid_stability(entries, 3, kind='nlcgrids')
        self.assertIn('nlcgrids', block['note'])
        doc = self._doc(**block)
        self.assertEqual(sv.validate_document(doc), [])

    def test_production_level_missing_rejected(self):
        doc = self._doc(production_level=4, entries=[
            {'level': 2, 'ngrids': 2000, 'e_tot': -100.0, 'converged': True,
             'scf_cycles': 10, 't_s': 1.0}],
                        classification='single-level', note='x')
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('grid-stability-production', tags)


class Rule10NlcConsistencyTests(unittest.TestCase):
    '''R10 (issue #7): NLC-functional rows must carry the VV10 term on
    every ok lane, and the NLC piece must agree with the fp64 reference
    within the precision gate.  Built on a wb97m-v version of the compliant
    document (xc_definition satisfied via a decomposition stub).'''

    def _nlc_doc(self):
        doc = compliant_doc()
        for row in doc['rows']:
            row['row_key']['xc'] = 'wb97m-v'
            row['method']['xc'] = 'wb97m-v'
            row['method']['xc_definition'] = {
                'xc': 'wb97m-v',
                'libxc_decomposition': [{'id': 531, 'factor': 1.0}],
                'hyb_coeff': 1.0, 'rsh': {'omega': 0.3, 'alpha': 1.0,
                                          'beta': -0.85},
                'nlc_params': [[6.0, 0.01, 1.0]]}
            # R4 requires a non-null NLC term on wb97m-v rows
            row['result']['e_terms']['nlc'] = -0.123456789
        return doc

    def test_consistent_nlc_terms_pass(self):
        doc = self._nlc_doc()
        self.assertEqual(sv.validate_document(doc), [])

    def test_null_nlc_term_rejected(self):
        doc = self._nlc_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['fp64']['result']['e_terms']['nlc'] = None
        tags = {i['tag'] for i in sv.validate_document(doc)}
        # R4 (per-row method completeness) and R10 (consistency) both fire
        self.assertIn('nlc-missing', tags)
        self.assertIn('nlc-term-missing', tags)

    def test_lane_nlc_disagreement_rejected(self):
        doc = self._nlc_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        # 2e-6 Ha off the fp64 reference on the auto lane: the total energy
        # may still be within its own gate; the NLC piece must not be
        by_lane['auto']['result']['e_terms']['nlc'] = -0.123456789 + 2e-6
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('nlc-inconsistent', tags)

    def test_cpu_sentinel_nlc_disagreement_rejected(self):
        doc = self._nlc_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['cpu-fp64']['result']['e_terms']['nlc'] = \
            -0.123456789 - 5e-6
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('nlc-inconsistent', tags)

    def test_nlc_agreement_within_gate_accepted(self):
        doc = self._nlc_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['auto']['result']['e_terms']['nlc'] = -0.123456789 + 1e-9
        self.assertEqual(sv.validate_document(doc), [])

    def test_non_nlc_functional_untouched(self):
        # m062x rows carry null nlc terms legitimately -- R10 must not fire
        doc = compliant_doc()
        for row in doc['rows']:
            row['row_key']['xc'] = 'm062x'
            row['method']['xc'] = 'm062x'
            row['method']['xc_definition'] = {
                'xc': 'm062x',
                'libxc_decomposition': [{'id': 47, 'factor': 1.0}],
                'hyb_coeff': 1.0, 'rsh': None, 'nlc_params': None}
        self.assertEqual(sv.validate_document(doc), [])

    def test_vv10_marker_detected(self):
        self.assertTrue(sv._is_nlc_xc('wb97m-v'))
        self.assertTrue(sv._is_nlc_xc('WB97X-V'))
        self.assertTrue(sv._is_nlc_xc('hf,vv10'))
        self.assertFalse(sv._is_nlc_xc('m062x'))
        self.assertFalse(sv._is_nlc_xc('b3lyp'))
        self.assertFalse(sv._is_nlc_xc(None))


class Rule11ThreeCConsistencyTests(unittest.TestCase):
    '''R11 (issue #8): r2SCAN-3c rows carry deterministic D4/gCP
    corrections evaluated by the same pyscf-dispersion library on every
    lane; the pieces must agree across lanes.  Built on an r2scan3c
    version of the compliant document (R4 satisfied via def2-mtzvpp basis
    + non-null dispersion/gcp terms + r2SCAN xc_definition stub).'''

    def _3c_doc(self):
        doc = compliant_doc()
        for row in doc['rows']:
            row['row_key']['xc'] = 'r2scan3c'
            row['row_key']['basis'] = 'def2-mtzvpp'
            row['method']['xc'] = 'r2scan3c'
            row['method']['basis'] = 'def2-mtzvpp'
            row['method']['xc_definition'] = {
                'xc': 'r2scan',
                'xc_alias_resolved': 'r2scan3c',
                'libxc_decomposition': [{'id': 724, 'factor': 1.0}],
                'hyb_coeff': 0.0, 'rsh': None, 'nlc_params': None}
            row['method']['dispersion'] = {
                'model': 'D4',
                'params': {'s6': 1.0, 's8': 0.0, 'a1': 0.42, 'a2': 5.65,
                           's9': 2.0, 'atm': True, 'ga': 2.0, 'gc': 1.0},
                'library': 'pyscf-dispersion',
                'library_version': '1.5.0'}
            row['method']['gcp'] = 'r2scan3c'
            # deterministic corrections, identical on every lane
            row['result']['e_terms']['dispersion'] = -5.052325715379687e-04
            row['result']['e_terms']['gcp'] = 4.582416768759153e-03
        return doc

    def test_consistent_corrections_pass(self):
        doc = self._3c_doc()
        self.assertEqual(sv.validate_document(doc), [])

    def test_lane_dispersion_disagreement_rejected(self):
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['auto']['result']['e_terms']['dispersion'] += 3e-6
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('correction-inconsistent', tags)

    def test_lane_gcp_disagreement_rejected(self):
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['cpu-fp64']['result']['e_terms']['gcp'] -= 2e-5
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('correction-inconsistent', tags)

    def test_df_vs_conv_lane_disagreement_rejected(self):
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['df-vs-conv']['result']['e_terms']['dispersion'] += 1e-3
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('correction-inconsistent', tags)

    def test_fp32_screen_lane_disagreement_rejected(self):
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['fp32-screen'] = json.loads(json.dumps(by_lane['auto']))
        by_lane['fp32-screen']['row_key']['lane'] = 'fp32-screen'
        by_lane['fp32-screen']['result']['e_terms']['gcp'] += 5e-6
        doc['rows'].append(by_lane['fp32-screen'])
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('correction-inconsistent', tags)

    def test_bitwise_agreement_accepted(self):
        # deterministic corrections agree to the last stored digit
        doc = self._3c_doc()
        self.assertEqual(sv.validate_document(doc), [])

    def test_near_bitwise_difference_within_gate_accepted(self):
        # 1e-12 Ha differences (e.g. a cross-stack rounding of the same
        # deterministic library call) are far inside the consistency bound
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['auto']['result']['e_terms']['dispersion'] += 1e-12
        by_lane['cpu-fp64']['result']['e_terms']['gcp'] -= 1e-13
        self.assertEqual(sv.validate_document(doc), [])

    def test_missing_terms_flagged_by_r4_not_r11(self):
        # a lane with a NULL correction term is R4's per-row completeness
        # violation; R11 skips nulls (no reference to compare against) and
        # must not double-report
        doc = self._3c_doc()
        by_lane = {r['row_key']['lane']: r for r in doc['rows']}
        by_lane['auto']['result']['e_terms']['gcp'] = None
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertIn('r2scan3c-terms-missing', tags)
        self.assertNotIn('correction-inconsistent', tags)

    def test_missing_fp64_reference_no_r11_comparison(self):
        # group without an ok fp64 lane: no reference, no comparison (the
        # total-energy verdict already covers the missing-reference case)
        doc = self._3c_doc()
        doc['rows'] = [r for r in doc['rows']
                       if r['row_key']['lane'] != 'fp64']
        for row in doc['rows']:
            row['accuracy']['dE_vs_gpu_fp64'] = None
            row['accuracy']['verdict'] = 'no_reference'
        tags = {i['tag'] for i in sv.validate_document(doc)}
        self.assertNotIn('correction-inconsistent', tags)

    def test_non_3c_functional_untouched(self):
        # plain r2SCAN rows (no composite alias) carry no corrections and
        # must not trigger R11 even with dispersed-looking e_terms
        doc = compliant_doc()
        for row in doc['rows']:
            row['row_key']['xc'] = 'r2scan'
            row['method']['xc'] = 'r2scan'
            row['method']['xc_definition'] = {
                'xc': 'r2scan',
                'libxc_decomposition': [{'id': 724, 'factor': 1.0}],
                'hyb_coeff': 0.0, 'rsh': None, 'nlc_params': None}
            row['result']['e_terms']['dispersion'] = -1e-3
            row['result']['e_terms']['gcp'] = 2e-3
        self.assertEqual(sv.validate_document(doc), [])

    def test_r2scan3c_marker_detection(self):
        self.assertTrue(runner_module().is_3c_xc('r2scan3c'))
        self.assertTrue(runner_module().is_3c_xc('R2SCAN-3C'))
        self.assertFalse(runner_module().is_3c_xc('r2scan'))
        self.assertFalse(runner_module().is_3c_xc('wb97m-v'))
        self.assertFalse(runner_module().is_3c_xc(None))


def runner_module():
    return _load('rtx4090_runner')


class CliTests(unittest.TestCase):
    def test_cli_exit_codes(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, 'good.json')
            with open(good, 'w') as f:
                json.dump(compliant_doc(), f)
            bad = os.path.join(td, 'bad.json')
            with open(bad, 'w') as f:
                json.dump(_fix(compliant_doc(), 'fp64', status='skipped'),
                          f)
            self.assertEqual(sv.main([good, '--evidence-root', '']), 0)
            self.assertEqual(sv.main([bad, '--evidence-root', '']), 1)
            self.assertEqual(sv.main([os.path.join(td, 'nope.json')]), 2)


if __name__ == '__main__':
    unittest.main()

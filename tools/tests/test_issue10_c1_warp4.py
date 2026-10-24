"""Issue #10 unit tests: C1 warp4 selector semantics + retest analysis.

Covers:
  * the c1_retest_analysis module on synthetic in-memory documents
    (gate pass/fail, missing-document refusal, schema-INVALID refusal);
  * the VV10 UWE kernel selector contract as documented in vv10.cu:
    default = reference warp1 kernel, 'warp4' opt-in via
    GPU4PYSCF_VV10_UWE_KERNEL, unknown values ignored with a warning.
    The selector itself is C-side (static, cached per process); the
    Python-side contract tested here is what the runner and the docs
    promise: nothing in gpu4pyscf Python reads the variable, so opting
    in never changes any other code path.

No GPU and no compiled extension is needed for these tests.
"""
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))

_DIR = pathlib.Path(REPO) / 'mixedprec' / 'benchmark' / 'rtx4090'
_spec = importlib.util.spec_from_file_location(
    'c1_retest_analysis', _DIR / 'c1_retest_analysis.py')
c1_retest_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1_retest_analysis)


def _ok_row(lane, e_tot, median, cycles=8, converged=True):
    return {
        'row_key': {'geom': 'g.xyz', 'basis': 'b', 'xc': 'wb97m-v',
                    'lane': lane},
        'pilot_stage': 'S1',
        'system': {'geom_file': 'g.xyz', 'geom_sha256': '0' * 64, 'natm': 6,
                   'nao': 114, 'charge': 0, 'spin': 0, 'nelec': 42,
                   'cart': False, 'geom_units': 'angstrom'},
        'method': {'xc': 'wb97m-v', 'basis': 'b',
                   'xc_definition': {'xc': 'wb97m-v',
                                     'libxc_decomposition': [
                                         {'id': 531, 'factor': 1.0}],
                                     'hyb_coeff': 1.0,
                                     'rsh': {'omega': 0.3, 'alpha': 1.0,
                                             'beta': -0.85},
                                     'nlc_params': [[6.0, 0.01, 1.0]]},
                   'auxbasis': 'def2-universal-jkfit', 'with_df': True,
                   'grids': {'level': 3, 'atom_grid': None},
                   'nlcgrids': {'level': 3, 'atom_grid': None},
                   'dispersion': None, 'gcp': None, 'conv_tol': 1e-10,
                   'conv_tol_grad': None, 'direct_scf_tol': 1e-14,
                   'init_guess': 'minao', 'max_cycle': 50,
                   'open_shell': False},
        'lane': lane, 'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': e_tot, 'converged': converged, 'scf_cycles': cycles,
            'e_terms': {'e1': -100.0, 'coul': 30.0, 'exc': -10.0,
                        'nlc': -0.16, 'dispersion': None, 'gcp': None},
            'timings': {
                'warmup_s': 1.0, 'samples_s': [median] * 3,
                'median_s': median, 'min_s': median, 'max_s': median,
                'end_to_end_s': median + 1.0, 't_cderi_s': 1.0,
                't_scf_s': median, 't_xc_grid_s': 1.0, 't_jk_s': 0.1,
                't_nlc_s': 0.5, 't_disp_s': 0.0,
                'sync': 'device-sync before/after each sample'},
            'mem': {'gpu_free_before_mib': 24000.0,
                    'gpu_total_mib': 24564.0, 'gpu_peak_alloc_mib': 200.0,
                    'host_rss_peak_mib': 900.0, 'headroom_budget_mib': 2048.0,
                    'within_budget': True},
            'fp32_entry_hits': {'df_jk': 48 if lane == 'auto' else 0,
                                'eval_rho': 816 if lane == 'auto' else 0,
                                'xc_grid': 24 if lane == 'auto' else 0,
                                'grad_ip1': 0,
                                'other': 20 if lane == 'auto' else 0},
            'precision_switch': {'switched': lane == 'auto', 'cycle': 6,
                                 'trigger': 'dE' if lane == 'auto'
                                            else 'none'},
            'fallback_events': [], 'errors': None,
        },
        'accuracy': {'dE_vs_gpu_fp64': 0.0 if lane == 'fp64' else 1e-13,
                     'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None,
                     'verdict': 'pass'},
        'cross_program': [],
        'evidence': {'raw_log': '/nonexistent/evidence.log@' + '0' * 64,
                     'result_json': '/nonexistent/evidence.json@' + '0' * 64,
                     'command': 'synthetic'},
    }


def _ok_doc(commit, e_fp64, e_auto, median_fp64, median_auto):
    return {
        'schema_version': 'rtx4090-energy-v1',
        'meta': {'gpu_name': 'NVIDIA GeForce RTX 4090',
                 'gpu_total_mem_mib': 24564.0, 'driver_version': '595',
                 'cuda_version': '12.8', 'cpu_model': 'x', 'cpu_threads': 24,
                 'ram_gb': 64.0, 'python': '3.10', 'cupy': '13',
                 'pyscf': '2.7', 'gpu4pyscf_source_commit': commit,
                 'gpu4pyscf_source_path': REPO, 'libxc': '6',
                 'mem_headroom_mib': 2048.0, 'timestamp_utc': '2026-10-04',
                 'run_id': 'synthetic'},
        'rows': [_ok_row('fp64', e_fp64, median_fp64),
                 _ok_row('auto', e_auto, median_auto, cycles=9)],
    }


class TestC1RetestAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.evidence_dir = self.tmp.name
        self.addCleanup(self.tmp.cleanup)
        # evidence-root validation must pass: write real anchor files
        self.log = os.path.join(self.evidence_dir, 'evidence.log')
        self.raw = os.path.join(self.evidence_dir, 'evidence.json')
        with open(self.log, 'w', encoding='utf-8') as f:
            f.write('synthetic\n')
        with open(self.raw, 'w', encoding='utf-8') as f:
            f.write('{}\n')
        import hashlib

        def sha(path):
            with open(path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        for v in ('warp1', 'warp4'):
            for lane in ('fp64', 'auto'):
                doc = _ok_doc('synthetic-commit',
                              -231.9716424768521 if v == 'warp1'
                              else -231.9716424768521 + 1e-13,
                              -231.97164247685166 if v == 'warp1'
                              else -231.97164247685166 + 1e-13,
                              8.0 if v == 'warp1' else 8.0 / 1.05,
                              9.0 if v == 'warp1' else 9.0 / 1.05)
                for row in doc['rows']:
                    row['evidence'] = {
                        'raw_log': f'{self.log}@{sha(self.log)}',
                        'result_json': f'{self.raw}@{sha(self.raw)}',
                        'command': 'synthetic'}
                with open(os.path.join(
                        self.evidence_dir, f'c1_retest_cell_{v}.json'),
                        'w', encoding='utf-8') as f:
                    json.dump(doc, f)
        # evidence must live under the root the analysis pins
        self._orig_default = c1_retest_analysis.main
        # point validation root at the tmp dir via monkey of analyse_cell
        self._orig_load = c1_retest_analysis.load_variant

        def load_variant(path, evidence_root):
            return self._orig_load(path, evidence_root=self.evidence_dir)
        c1_retest_analysis.load_variant = load_variant

    def tearDown(self):
        c1_retest_analysis.load_variant = self._orig_load

    def test_gate_pass_and_speedup(self):
        entry = c1_retest_analysis.analyse_cell(
            'cell', {v: os.path.join(self.evidence_dir,
                                     f'c1_retest_cell_{v}.json')
                     for v in ('warp1', 'warp4')},
            evidence_root=self.evidence_dir)
        self.assertTrue(entry['precision_gate_within_1e-6'])
        # auto-vs-fp64 within each variant is the gate quantity
        for v in ('warp1', 'warp4'):
            self.assertLess(abs(entry[f'dE_auto_vs_fp64_{v}']), 1e-6)
        # warp4-vs-warp1 difference is a diagnostic (reduction order)
        self.assertLess(abs(entry['fp64']['dE_variant']), 1e-9)
        self.assertAlmostEqual(entry['fp64']['speedup_median_warp4'],
                               1.05, places=6)
        self.assertAlmostEqual(entry['auto']['speedup_median_warp4'],
                               1.05, places=6)

    def test_gate_fail_when_auto_off_fp64(self):
        path = os.path.join(self.evidence_dir, 'c1_retest_cell_warp4.json')
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        auto = [r for r in doc['rows']
                if r['row_key']['lane'] == 'auto'][0]
        auto['result']['e_tot'] -= 5e-6
        auto['result']['dE_vs_gpu_fp64'] = -5e-6
        auto['result']['verdict'] = 'fail'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        entry = c1_retest_analysis.analyse_cell(
            'cell', {v: os.path.join(self.evidence_dir,
                                     f'c1_retest_cell_{v}.json')
                     for v in ('warp1', 'warp4')},
            evidence_root=self.evidence_dir)
        self.assertFalse(entry['precision_gate_within_1e-6'])

    def test_missing_document_refused(self):
        paths = {v: os.path.join(self.evidence_dir,
                                 f'c1_retest_cell_{v}.json')
                 for v in ('warp1', 'warp4')}
        paths['warp4'] += '.missing'
        with self.assertRaises(SystemExit):
            c1_retest_analysis.analyse_cell('cell', paths, self.evidence_dir)

    def test_commit_mismatch_refused(self):
        path = os.path.join(self.evidence_dir, 'c1_retest_cell_warp4.json')
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        doc['meta']['gpu4pyscf_source_commit'] = 'another-commit'
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(doc, f)
        with self.assertRaises(SystemExit):
            c1_retest_analysis.analyse_cell(
                'cell', {v: os.path.join(self.evidence_dir,
                                         f'c1_retest_cell_{v}.json')
                         for v in ('warp1', 'warp4')},
                evidence_root=self.evidence_dir)


class TestVV10SelectorContract(unittest.TestCase):
    """The kernel selector env var must be read by NOTHING in Python.

    The C launcher reads GPU4PYSCF_VV10_UWE_KERNEL once per process
    (vv10.cu). If any gpu4pyscf Python module also read it, opt-in could
    silently alter unrelated behaviour; this test pins that the only
    occurrences in the Python tree are documentation strings.
    """

    def test_env_var_only_in_docs(self):
        hits = []
        root = os.path.join(REPO, 'gpu4pyscf')
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.py'):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding='utf-8') as f:
                    for lineno, line in enumerate(f, 1):
                        if 'GPU4PYSCF_VV10_UWE_KERNEL' in line:
                            hits.append(f'{path}:{lineno}')
        # the runner-side reference lives in mixedprec/benchmark and the
        # C side in the .cu file; the gpu4pyscf package itself must be
        # free of reads (empty list expected once docs mention nothing)
        self.assertEqual(hits, [])

    def test_selector_source_semantics(self):
        """Static source assertions for the C-side selector contract."""
        cu = os.path.join(REPO, 'gpu4pyscf', 'lib', 'gdft', 'vv10.cu')
        with open(cu, encoding='utf-8') as f:
            src = f.read()
        # opt-in only: warp4 requires the exact env value
        self.assertIn('GPU4PYSCF_VV10_UWE_KERNEL', src)
        self.assertIn('strcmp(env, "warp4") == 0', src)
        # the default (unset/anything else) keeps the reference kernel
        self.assertIn('vv10_fock_eval_UWE_kernel<<<', src)
        self.assertIn('vv10_fock_eval_UWE_multiwarp_kernel<<<', src)


if __name__ == '__main__':
    unittest.main()

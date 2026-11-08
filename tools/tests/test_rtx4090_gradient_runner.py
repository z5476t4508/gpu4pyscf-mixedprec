'''Unit tests for the rtx4090-gradient-v1 runner's pure functions
(issue #12): lane specification, derived gradient columns, the homotopy
decomposition bookkeeping, energy-gate document reading and the provisional
verdict notes.  No GPU/pyscf needed -- only stdlib structures.'''

import importlib.util
import json
import pathlib
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gr = _load('gradient_runner')
sv = _load('schema_validator')


def _row(lane, e=-114.5, grad=None, converged=True, status='ok'):
    if grad is None:
        grad = [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]
    return {
        'row_key': {'geom': 'methanol.xyz', 'basis': 'def2-svp',
                    'xc': 'b3lyp', 'lane': lane},
        'pilot_stage': 'S0',
        'system': {'geom_file': 'methanol.xyz', 'geom_sha256': 'x' * 64,
                   'natm': 2, 'nao': 48, 'charge': 0, 'spin': 0,
                   'nelec': 18, 'cart': False, 'geom_units': 'angstrom'},
        'method': {'xc': 'b3lyp', 'basis': 'def2-svp',
                   'xc_definition': {'xc': 'b3lyp', 'libxc_decomposition':
                                     [{'id': 106, 'factor': 0.8}]},
                   'auxbasis': 'def2-universal-jkfit', 'with_df': True,
                   'grids': {'level': 3, 'atom_grid': None},
                   'nlcgrids': {'level': None, 'atom_grid': None},
                   'dispersion': None, 'gcp': None, 'conv_tol': 1e-10,
                   'conv_tol_grad': None, 'direct_scf_tol': 1e-14,
                   'init_guess': 'minao', 'max_cycle': 50,
                   'open_shell': False},
        'status': status, 'skip_reason': None,
        'result': {
            'e_tot': e, 'converged': converged, 'scf_cycles': 11,
            'gradient': grad,
            'gradient_norm': gr._gradient_norm(grad),
            'gradient_mode': 'fp64' if lane in ('fp64', 'auto-fp64-grad',
                                                'cpu-fp64') else 'fp32',
            'repeats': [
                {'index': i, 'e_tot': e, 'converged': converged,
                 'scf_cycles': 11, 'gradient': grad,
                 'gradient_norm': gr._gradient_norm(grad),
                 'gradient_mode': 'fp64' if lane in (
                     'fp64', 'auto-fp64-grad', 'cpu-fp64') else 'fp32',
                 'grad_ip1_hits': 0 if lane in (
                     'fp64', 'auto-fp64-grad', 'cpu-fp64') else 4,
                 'sample_s': 1.0 + i * 0.1, 't_scf_s': 0.6, 't_grad_s': 0.4,
                 'precision_switch': {'switched': False, 'cycle': None,
                                      'trigger': 'none'}}
                for i in range(3)],
            'repeat_evidence': 'all-repeats',
            'e_terms': {'e1': -185.0, 'coul': 55.0, 'exc': -14.0,
                        'nlc': None, 'dispersion': None, 'gcp': None},
            'timings': {'warmup_s': 1.0, 'samples_s': [1.0, 1.1, 1.2],
                        'median_s': 1.1, 'min_s': 1.0, 'max_s': 1.2,
                        'end_to_end_s': 1.1, 't_cderi_s': 0.5,
                        't_scf_s': 0.6, 't_grad_s': 0.4,
                        'sync': 'device-sync before/after each phase'},
            'mem': {'gpu_free_before_mib': 24000.0,
                    'gpu_total_mib': 24564.0, 'gpu_peak_alloc_mib': 500.0,
                    'host_rss_peak_mib': 1200.0,
                    'headroom_budget_mib': 2048.0, 'within_budget': True},
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                'grad_ip1': 0 if lane in (
                                    'fp64', 'auto-fp64-grad', 'cpu-fp64')
                                else 12},
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [],
        'evidence': {'raw_log': '/tmp/x@' + '0' * 64,
                     'result_json': '/tmp/y@' + '0' * 64,
                     'command': 'python gradient_runner.py --out x.json'},
    }


G_REF = [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]
G_SMALL = [[0.01 + 1e-6, -0.02, 0.03], [-0.01, 0.02, -0.03]]
G_HUGE = [[0.01 + 1e-3, -0.02, 0.03], [-0.01, 0.02, -0.03]]


def _cell_rows(auto_grad=G_SMALL, fp32g_grad=G_SMALL,
               autofp64_grad=G_SMALL, cpu_grad=G_SMALL, e_auto=-114.5):
    return [_row('fp64'),
            _row('auto', e=e_auto, grad=auto_grad),
            _row('fp32-grad', grad=fp32g_grad),
            _row('auto-fp64-grad', grad=autofp64_grad),
            _row('cpu-fp64', grad=cpu_grad)]


class LaneSpecTests(unittest.TestCase):
    def test_lane_spec_covers_all_lanes(self):
        for lane in gr.GRAD_LANES:
            self.assertIn(lane, gr.LANE_SPEC)
            scf_mode, grad_mode = gr.LANE_SPEC[lane]
            self.assertIsNone(scf_mode) if lane == 'cpu-fp64' else \
                self.assertIn(scf_mode, ('fp64', 'auto', 'auto+verify',
                                         'auto+strict'))

    def test_auto_verify_lane_spec(self):
        '''The issue-7 diagnosed route as a gradient lane: auto+verify SCF
        (stability verification selects the solution) with fp32 gradient
        kernels (the verification resolves the solution, not the kernel
        lane).'''
        self.assertEqual(gr.LANE_SPEC['auto-verify'], ('auto+verify',
                                                       None))

    def test_auto_strict_lane_spec(self):
        '''The LC-76 strict fp64-reference matching route as a gradient
        lane: auto+strict SCF (reference adopted unconditionally on a
        distinct result, or fail-closed) with fp32 gradient kernels.'''
        self.assertEqual(gr.LANE_SPEC['auto-strict'], ('auto+strict',
                                                       None))
        self.assertIn('auto-strict', gr.AUTO_SCF_LANES)
        self.assertIn('auto-strict', gr.FP32_GRAD_LANES)

    def test_homotopy_decomposition_lanes(self):
        '''The three non-reference GPU lanes differ exactly one factor at
        a time: auto = (auto SCF, fp32 grad); fp32-grad = (fp64 SCF,
        fp32 grad); auto-fp64-grad = (auto SCF, fp64 grad).'''
        self.assertEqual(gr.LANE_SPEC['auto'], ('auto', None))
        self.assertEqual(gr.LANE_SPEC['fp32-grad'], ('fp64', 'fp32'))
        self.assertEqual(gr.LANE_SPEC['auto-fp64-grad'], ('auto', 'fp64'))
        self.assertEqual(gr.LANE_SPEC['fp64'], ('fp64', 'fp64'))


class DeriveGradTests(unittest.TestCase):
    def test_compliant_cell(self):
        rows = gr.derive_grad(_cell_rows())
        by_lane = {r['row_key']['lane']: r for r in rows}
        for lane in ('auto', 'fp32-grad', 'auto-fp64-grad'):
            acc = by_lane[lane]['accuracy']
            self.assertAlmostEqual(acc['max_abs_dg_vs_gpu_fp64'], 1e-6)
            self.assertEqual(acc['verdict'], 'pass')
        self.assertEqual(by_lane['fp64']['accuracy']['verdict'], 'pass')
        self.assertIsNone(by_lane['fp64']['accuracy']
                          ['max_abs_dg_vs_gpu_fp64'])
        self.assertEqual(by_lane['cpu-fp64']['accuracy']['verdict'],
                         'no_reference')

    def test_gate_failure_is_honest(self):
        rows = gr.derive_grad(_cell_rows(auto_grad=G_HUGE))
        by_lane = {r['row_key']['lane']: r for r in rows}
        self.assertEqual(by_lane['auto']['accuracy']['verdict'], 'fail')
        self.assertAlmostEqual(
            by_lane['auto']['accuracy']['max_abs_dg_vs_gpu_fp64'], 1e-3)

    def test_unconverged_is_fail(self):
        rows = _cell_rows()
        for r in rows:
            if r['row_key']['lane'] == 'auto':
                # an unconverged repeat is the failure evidence under the
                # all-repeat contract: the top-level flag is derived from
                # the repeats, so the repeat must carry it
                r['result']['converged'] = False
                for rep in r['result']['repeats']:
                    rep['converged'] = False
        rows = gr.derive_grad(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        self.assertEqual(by_lane['auto']['accuracy']['verdict'], 'fail')

    def test_missing_reference_is_no_reference(self):
        rows = _cell_rows()
        rows = [r for r in rows if r['row_key']['lane'] != 'fp64']
        rows = gr.derive_grad(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        for lane in ('auto', 'fp32-grad', 'auto-fp64-grad'):
            self.assertEqual(by_lane[lane]['accuracy']['verdict'],
                             'no_reference', lane)

    def test_cpu_delta_diagnostic_and_note(self):
        '''A large cpu deviation gets the named two-implementation note
        (G4) but never changes the verdict.'''
        rows = _cell_rows(cpu_grad=G_HUGE)
        rows = gr.derive_grad(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        note = by_lane['auto']['accuracy']['impl_agreement_note']
        self.assertIsNotNone(note)
        self.assertIn('implementation', note)
        self.assertEqual(by_lane['auto']['accuracy']['verdict'], 'pass')

    def test_provisional_note_on_failed_energy_cell(self):
        '''G5: a passing gradient verdict on a cell whose energy gate
        failed gets the explicit provisional note.'''
        failed = {'methanol.xyz/def2-svp/b3lyp':
                  'issue-7 auto lane dE_vs_gpu_fp64 = -1.283e-06 Ha'}
        rows = gr.derive_grad(_cell_rows(), gate_failed_cells=failed)
        by_lane = {r['row_key']['lane']: r for r in rows}
        for lane in ('auto', 'fp32-grad', 'auto-fp64-grad'):
            note = by_lane[lane]['accuracy'].get('verdict_note')
            self.assertIsNotNone(note, lane)
            self.assertIn('provisional', note)
        # a FAILING verdict gets no provisional note (it is already a
        # fail; nothing is being provisionally accepted)
        rows = gr.derive_grad(_cell_rows(auto_grad=G_HUGE),
                              gate_failed_cells=failed)
        by_lane = {r['row_key']['lane']: r for r in rows}
        self.assertIsNone(by_lane['auto']['accuracy'].get('verdict_note'))

    def test_energy_delta_reported_not_gated(self):
        '''The auto lane's SCF energy delta vs the fp64 reference is
        reported (dE_vs_gpu_fp64) but the gradient verdict uses the
        gradient gate, not the energy gate.'''
        rows = gr.derive_grad(_cell_rows(e_auto=-114.5 + 5e-6))
        by_lane = {r['row_key']['lane']: r for r in rows}
        acc = by_lane['auto']['accuracy']
        self.assertAlmostEqual(acc['dE_vs_gpu_fp64'], 5e-6)
        self.assertEqual(acc['verdict'], 'pass')


class EnergyGateDocTests(unittest.TestCase):
    def test_reads_failed_auto_cells_only(self):
        doc = {'schema_version': 'rtx4090-energy-v1', 'issue': 7,
               'rows': [
                   {'row_key': {'geom': 'methoxy.xyz', 'basis': 'def2-svp',
                                'xc': 'wb97m-v', 'lane': 'auto'},
                    'status': 'ok',
                    'accuracy': {'dE_vs_gpu_fp64': -1.283e-6,
                                 'verdict': 'fail'}},
                   {'row_key': {'geom': 'benzene.xyz', 'basis': 'def2-svp',
                                'xc': 'wb97m-v', 'lane': 'auto'},
                    'status': 'ok',
                    'accuracy': {'dE_vs_gpu_fp64': 2.0e-12,
                                 'verdict': 'pass'}},
                   {'row_key': {'geom': 'x.xyz', 'basis': 'def2-svp',
                                'xc': 'wb97m-v', 'lane': 'fp32-screen'},
                    'status': 'ok',
                    'accuracy': {'dE_vs_gpu_fp64': 1e-5,
                                 'verdict': 'fail'}}]}
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / 'e.json'
            path.write_text(json.dumps(doc))
            failed = gr.read_energy_gate_documents(
                [str(path)], lambda m: None)
        self.assertEqual(len(failed), 1)
        self.assertIn('methoxy.xyz/def2-svp/wb97m-v', failed)
        # the fp32-screen honest fail is NOT an auto-lane energy failure:
        # that lane's conv_tol floor is a documented screening behavior
        self.assertNotIn('x.xyz/def2-svp/wb97m-v', failed)

    def test_unreadable_document_ignored(self):
        failed = gr.read_energy_gate_documents(
            ['/nonexistent/doc.json'], lambda m: None)
        self.assertEqual(failed, {})


class RunnerIntegrationTests(unittest.TestCase):
    def test_derived_document_validates(self):
        '''The runner's own derive output must satisfy the gradient
        schema end to end (fixtures only; no GPU).'''
        failed = {'methanol.xyz/def2-svp/b3lyp':
                  'issue-7 auto lane dE_vs_gpu_fp64 = -1.283e-06 Ha'}
        rows = gr.derive_grad(_cell_rows(), gate_failed_cells=failed)
        for row in rows:
            if '/'.join(k for k in
                        [row['row_key']['geom'], row['row_key']['basis'],
                         row['row_key']['xc']]) in failed:
                row.setdefault('result', {})['energy_gate_reference'] = {
                    'source_issues': 'issue-7 (inherited)',
                    'gate_failed_cells': [
                        'methanol.xyz/def2-svp/b3lyp'],
                    'note': failed['methanol.xyz/def2-svp/b3lyp']}
        doc = {'schema_version': sv.GRADIENT_SCHEMA_VERSION,
               'meta': {
                   'gpu_name': 'NVIDIA GeForce RTX 4090',
                   'gpu_total_mem_mib': 24564,
                   'driver_version': 'x', 'cuda_version': '12.8',
                   'cpu_model': 'x', 'cpu_threads': 24, 'ram_gb': 64.0,
                   'python': '3.12.3', 'cupy': '13.4.1',
                   'pyscf': '2.14.0', 'gpu4pyscf_source_commit': '0' * 40,
                   'gpu4pyscf_source_path': '/x', 'libxc': '7.0.0',
                   'mem_headroom_mib': 2048,
                   'timestamp_utc': '2026-10-04T00:00:00Z',
                   'run_id': 't'},
               'rows': rows}
        issues = sv.validate_gradient_document(doc, evidence_root=None)
        self.assertEqual(issues, [], msg=json.dumps(issues, indent=1))


    def test_ineffective_fp32_grad_lane_is_fail(self):
        '''The UKS hybrid structural gap: an fp32-grad row whose
        gradient_mode measured fp64 (zero grad_ip1 hits) never ran the
        mixed-precision gradient -- verdict fail with the gap named,
        never a silent fp64 fallback presented as a pass.  (SP-22 pin
        update: the plain-auto lane is NOT in this class anymore -- its
        gradient resolves fp64 BY DESIGN at every nao, so a zero-grad-
        hit auto row passes with its mixed precision evidenced by the
        SCF head; see test_sp22_fp64grad_everywhere for the pinned
        semantics.)'''
        rows = _cell_rows()
        for r in rows:
            if r['row_key']['lane'] in ('auto', 'fp32-grad'):
                r['result']['gradient_mode'] = 'fp64'
                r['result']['fp32_entry_hits']['grad_ip1'] = 0
        rows = gr.derive_grad(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        acc = by_lane['fp32-grad']['accuracy']
        self.assertEqual(acc['verdict'], 'fail', 'fp32-grad')
        self.assertIn('ineffective', acc['verdict_note'])
        self.assertIn('df/grad/uhf.py', acc['verdict_note'])
        # (SP-22) the same zero-grad-hit signature on the plain-auto
        # lane is the by-design fp64 gradient: honest pass
        self.assertEqual(by_lane['auto']['accuracy']['verdict'], 'pass')
        # auto-fp64-grad keeps its honest pass (it never requested fp32)
        self.assertEqual(by_lane['auto-fp64-grad']['accuracy']['verdict'],
                         'pass')


class StrictLaneTests(unittest.TestCase):
    '''LC-76: the auto-strict lane's verdict is fail-closed on an
    incomplete (or missing) strict-match certification in ANY repeat --
    the lane claims a CERTIFIED fp64-lane match, never a silent
    selection.'''

    @staticmethod
    def _strict_cell(verified, reason=None):
        rows = _cell_rows()
        strict = _row('auto-strict', grad=G_SMALL)
        for r in strict['result']['repeats']:
            if verified is None:
                r['stability_verification'] = None
            else:
                sv = {'policy': 'auto+strict',
                      'strict_match_verified': verified}
                if not verified:
                    sv['strict_match_incomplete_reason'] = reason
                r['stability_verification'] = sv
        return rows + [strict]

    def test_strict_certified_passes(self):
        rows = gr.derive_grad(self._strict_cell(True))
        strict = next(r for r in rows if r['row_key']['lane'] == 'auto-strict')
        self.assertEqual(strict['accuracy']['verdict'], 'pass')

    def test_strict_incomplete_fails_closed(self):
        rows = gr.derive_grad(
            self._strict_cell(False, 'reference_not_converged'))
        strict = next(r for r in rows if r['row_key']['lane'] == 'auto-strict')
        self.assertEqual(strict['accuracy']['verdict'], 'fail')
        self.assertIn('reference_not_converged',
                      strict['accuracy']['verdict_note'])
        self.assertIn('fail-closed', strict['accuracy']['verdict_note'])

    def test_strict_missing_record_fails_closed(self):
        rows = gr.derive_grad(self._strict_cell(None))
        strict = next(r for r in rows if r['row_key']['lane'] == 'auto-strict')
        self.assertEqual(strict['accuracy']['verdict'], 'fail')
        self.assertIn('no record', strict['accuracy']['verdict_note'])

    def test_strict_one_bad_repeat_fails_row(self):
        '''a single uncertified repeat among certified ones still fails
        the row (all-repeat discipline)'''
        rows = self._strict_cell(True)
        rows[-1]['result']['repeats'][1]['stability_verification'] = {
            'policy': 'auto+strict', 'strict_match_verified': False,
            'strict_match_incomplete_reason': 'dm_comparison_nonfinite'}
        rows = gr.derive_grad(rows)
        strict = next(r for r in rows if r['row_key']['lane'] == 'auto-strict')
        self.assertEqual(strict['accuracy']['verdict'], 'fail')


if __name__ == '__main__':
    unittest.main()

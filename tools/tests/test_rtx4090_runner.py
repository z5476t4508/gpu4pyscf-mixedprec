"""Unit tests for the RTX 4090 runner's GPU-free logic (issue #4).

The runner's derive/upsert/attach_cross_program functions and the fp32-hit
call-site mapping are pure and tested here without a GPU; the full GPU
smoke is the separate methanol run recorded as evidence.
"""

import importlib.util
import json
import pathlib
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'

_spec = importlib.util.spec_from_file_location(
    'rtx4090_runner', _DIR / 'rtx4090_runner.py')
runner = importlib.util.module_from_spec(_spec)
# the module imports pyscf lazily (inside functions), and schema_validator
# eagerly but stdlib-only, so exec is safe without a GPU
_spec.loader.exec_module(runner)


def _row(lane, e=None, status='ok', converged=True):
    return {
        'row_key': {'geom': 'methanol.xyz', 'basis': 'def2-svp',
                    'xc': 'hf', 'lane': lane},
        'pilot_stage': 'S0', 'status': status, 'skip_reason': None,
        'result': {'e_tot': e, 'converged': converged, 'scf_cycles': 11,
                   'e_terms': {}, 'timings': {}, 'mem': {},
                   'fp32_entry_hits': {}, 'precision_switch': {},
                   'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }


class DeriveTests(unittest.TestCase):
    def test_precision_gate_against_gpu_fp64(self):
        rows = [_row('fp64', e=-100.0), _row('auto', e=-100.0 + 5e-7),
                _row('cpu-fp64', e=-100.0 + 1e-10)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertEqual(by['auto']['accuracy']['verdict'], 'pass')
        self.assertAlmostEqual(by['auto']['accuracy']['dE_vs_gpu_fp64'],
                               5e-7)
        self.assertAlmostEqual(by['auto']['accuracy']['dE_vs_cpu_fp64'],
                               5e-7 - 1e-10)

    def test_precision_gate_failure(self):
        rows = [_row('fp64', e=-100.0), _row('auto', e=-100.0 + 5e-6)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertEqual(by['auto']['accuracy']['verdict'], 'fail')

    def test_no_reference_is_not_pass(self):
        rows = [_row('auto', e=-100.0)]     # fp64 lane absent
        runner.derive(rows)
        self.assertEqual(rows[0]['accuracy']['verdict'], 'no_reference')

    def test_unconverged_is_fail_even_with_reference(self):
        rows = [_row('fp64', e=-100.0), _row('auto', e=-100.0,
                                             converged=False)]
        runner.derive(rows)
        self.assertEqual(rows[1]['accuracy']['verdict'], 'fail')

    def test_cpu_lane_does_not_reference_itself(self):
        rows = [_row('fp64', e=-100.0), _row('cpu-fp64', e=-100.0 + 2e-7)]
        runner.derive(rows)
        acc = rows[1]['accuracy']
        self.assertIsNone(acc['dE_vs_cpu_fp64'])
        self.assertAlmostEqual(acc['dE_vs_gpu_fp64'], 2e-7)

    def test_df_vs_conv_delta_lives_only_there(self):
        rows = [_row('fp64', e=-100.0), _row('df-vs-conv', e=-100.0 + 0.02),
                _row('auto', e=-100.0)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertAlmostEqual(by['df-vs-conv']['accuracy']
                                ['dE_df_approximation'], 0.02)
        # the DF approximation delta never reaches the precision gate
        self.assertIsNone(by['auto']['accuracy'].get('dE_df_approximation'))
        self.assertIsNone(by['df-vs-conv']['accuracy']['dE_vs_gpu_fp64'])
        self.assertEqual(by['df-vs-conv']['accuracy']['verdict'], 'pass')

    def test_skipped_and_oom_verdicts(self):
        rows = [_row('fp64', e=-100.0), _row('auto', status='skipped'),
                _row('auto', e=None, status='oom')]
        # upsert semantics: keep all three distinct lanes by keying one out
        rows[2]['row_key']['lane'] = 'fp32-screen'
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertEqual(by['auto']['accuracy']['verdict'], 'skipped')
        self.assertEqual(by['fp32-screen']['accuracy']['verdict'], 'fail')

    def test_fp64_reference_row_passes_on_convergence_alone(self):
        rows = [_row('fp64', e=-100.0)]
        runner.derive(rows)
        self.assertEqual(rows[0]['accuracy']['verdict'], 'pass')
        self.assertIsNone(rows[0]['accuracy']['dE_vs_gpu_fp64'])

    def test_fp32_screen_deviation_gets_named_attribution(self):
        # the pure-fp32 lane's |dE| vs cpu-fp64 is its own noise floor; R8
        # demands a named note, and the tier-(1) verdict must still fail
        rows = [_row('fp64', e=-100.0), _row('cpu-fp64', e=-100.0),
                _row('fp32-screen', e=-100.0 + 5.5e-6)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        acc = by['fp32-screen']['accuracy']
        self.assertEqual(acc['verdict'], 'fail')
        self.assertIn('noise floor', acc['impl_agreement_note'])
        # other lanes do not get the automatic note
        self.assertIsNone(by['fp64']['accuracy']['impl_agreement_note'])

    def test_auto_lane_deviation_gets_named_tail_stall_note(self):
        # measured live on the issue #5 methoxy UKS sentinel: the auto
        # lane's fp64 tail can plateau ~4e-8 Ha above the fp64-lane
        # solution; that residual is the lane's own trajectory effect, so
        # R8 needs a named note while the tier-(1) verdict stays a pass
        rows = [_row('fp64', e=-100.0), _row('cpu-fp64', e=-100.0),
                _row('auto', e=-100.0 + 4.0e-8)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        acc = by['auto']['accuracy']
        self.assertEqual(acc['verdict'], 'pass')   # 4e-8 <= 1e-6 gate
        self.assertIn('tail', acc['impl_agreement_note'])
        self.assertIn('not a matched-settings implementation difference',
                      acc['impl_agreement_note'])

    def test_auto_lane_within_impl_level_gets_no_note(self):
        rows = [_row('fp64', e=-100.0), _row('cpu-fp64', e=-100.0),
                _row('auto', e=-100.0 + 1e-13)]
        runner.derive(rows)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertIsNone(by['auto']['accuracy']['impl_agreement_note'])


class UpsertTests(unittest.TestCase):
    def test_rerun_replaces_only_its_own_cell(self):
        rows = [_row('fp64', e=-100.0), _row('auto', e=-99.0)]
        runner.upsert(rows, _row('auto', e=-100.0 + 1e-13))
        self.assertEqual(len(rows), 2)
        by = {r['row_key']['lane']: r for r in rows}
        self.assertAlmostEqual(by['auto']['result']['e_tot'],
                               -100.0 + 1e-13)


class HitsKeyTests(unittest.TestCase):
    def test_call_site_mapping(self):
        self.assertEqual(runner._hits_key('gpu4pyscf.df.df_jk', 'get_jk'),
                         'df_jk')
        self.assertEqual(runner._hits_key('gpu4pyscf.dft.numint',
                                          '_eval_rho2'), 'eval_rho')
        self.assertEqual(runner._hits_key('gpu4pyscf.dft.numint',
                                          '_nr_rks_task'), 'xc_grid')
        self.assertEqual(runner._hits_key('gpu4pyscf.df.grad.rhf',
                                          '_jk_energy_per_atom'), 'grad_ip1')
        self.assertEqual(runner._hits_key('gpu4pyscf.scf.hf', '_kernel'),
                         'other')


class SwitchInfoTests(unittest.TestCase):
    def test_switch_detected_with_trigger(self):
        probe = runner.PrecisionProbe()
        # the mode flips AFTER the cycle whose |dE| < 1e-4 was computed
        # (callback runs before the switch check in scf/hf.py), so the
        # trigger reads dE between the last two fp32 cycles: 1e-6 < 1e-4
        probe.cycles = [(0, -100.5, 'fp32'), (1, -100.05, 'fp32'),
                        (2, -100.050001, 'fp32'), (3, -100.0500001, 'fp64')]
        info = probe.switch_info('auto')
        self.assertTrue(info['switched'])
        self.assertEqual(info['cycle'], 4)   # 1-based: fp64 tail began in
        # cycle 4 (0-based index 3, first fp64 mode after fp32 cycles 0-2)
        self.assertEqual(info['trigger'], 'dE')

    def test_switch_detected_chaos_guard(self):
        probe = runner.PrecisionProbe()
        # |dE| stays ~1e-2 Eh late in the fp32 phase: the chaos guard
        # forced the switch, not the convergence criterion
        probe.cycles = [(0, -100.5, 'fp32'), (1, -100.05, 'fp32'),
                        (2, -100.06, 'fp32'), (3, -100.05, 'fp64')]
        info = probe.switch_info('auto')
        self.assertTrue(info['switched'])
        self.assertEqual(info['trigger'], 'chaos-guard')

    def test_no_switch(self):
        probe = runner.PrecisionProbe()
        probe.cycles = [(0, -100.5, 'fp64'), (1, -100.1, 'fp64')]
        info = probe.switch_info('auto')
        self.assertFalse(info['switched'])
        self.assertEqual(info['trigger'], 'none')

    def test_non_auto_lane_never_switches(self):
        probe = runner.PrecisionProbe()
        probe.cycles = [(0, -100.5, 'fp32'), (1, -100.0, 'fp64')]
        self.assertEqual(probe.switch_info('fp64')['trigger'], 'none')


class CrossProgramTests(unittest.TestCase):
    def _rows(self):
        return [_row('fp64', e=-100.0), _row('auto', e=-100.0)]

    def test_provisional_entry_attached_as_unmatched(self):
        import tempfile, os
        curation = {'cases': [{
            'case': 'testX', 'classification': 'untriaged',
            'methods': ['B3LYP'],
            'g16_energy_hartree': -99.0,
            'g16_method_signature': '#P b3lyp/6-31g* test int=ultrafine 5d',
            'unmatched_fields': ['geometry-precision', 'xc-variant'],
            'attach_to_rows': [{'geom': 'methanol.xyz', 'xc': 'hf',
                                'basis': 'def2-svp'}]}]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'curation.json')
            with open(path, 'w') as f:
                json.dump(curation, f)
            rows = self._rows()
            runner.attach_cross_program(rows, path, lambda m: None)
        self.assertEqual(len(rows[0]['cross_program']), 1)
        entry = rows[0]['cross_program'][0]
        self.assertFalse(entry['matched'])
        self.assertIn('xc-variant', entry['unmatched_fields'])
        self.assertAlmostEqual(entry['dE'], -100.0 - (-99.0))
        # the tier-(3) number must not leak into the gate fields
        self.assertIsNone(rows[0]['accuracy']['dE_vs_gpu_fp64'] or None
                          if rows[0]['row_key']['lane'] == 'auto' else None)

    def test_rejected_entry_never_attached(self):
        import tempfile, os
        curation = {'cases': [{
            'case': 'testY', 'classification': 'rejected',
            'g16_energy_hartree': -99.0,
            'attach_to_rows': [{'geom': 'methanol.xyz', 'xc': 'hf',
                                'basis': 'def2-svp'}]}]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'curation.json')
            with open(path, 'w') as f:
                json.dump(curation, f)
            rows = self._rows()
            runner.attach_cross_program(rows, path, lambda m: None)
        self.assertEqual(rows[0]['cross_program'], [])

    def test_no_mapping_no_attachment(self):
        import tempfile, os
        curation = {'cases': [{
            'case': 'testZ', 'classification': 'untriaged',
            'g16_energy_hartree': -99.0}]}
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'curation.json')
            with open(path, 'w') as f:
                json.dump(curation, f)
            rows = self._rows()
            runner.attach_cross_program(rows, path, lambda m: None)
        self.assertEqual(rows[0]['cross_program'], [])


class ETermsTests(unittest.TestCase):
    def test_nlc_split_out_of_exc(self):
        terms = runner._e_terms({'exc': -0.5, 'e1': -185.0, 'coul': 55.0},
                                nlc=-0.1)
        self.assertAlmostEqual(terms['exc'], -0.4)
        self.assertAlmostEqual(terms['nlc'], -0.1)

    def test_missing_terms_stay_null(self):
        terms = runner._e_terms({}, None)
        self.assertIsNone(terms['dispersion'])
        self.assertIsNone(terms['gcp'])
        self.assertIsNone(terms['nlc'])


class FakeLibxc:
    '''libxc-shaped stub for the XC-definition contract (no pyscf needed).'''

    @staticmethod
    def parse_xc(xc):
        if xc == 'b3lyp':
            return ((0, 0, 0), ((402, 1),))
        if xc == 'b3lyp5':
            return ((0.2, 0.2, 0), ((1, 0.08), (106, 0.72), (131, 0.81),
                                     (7, 0.19)))
        raise KeyError(xc)

    @staticmethod
    def rsh_coeff(xc):
        if xc == 'b3lyp':
            return (0.0, 0.2, 0.0)
        if xc == 'b3lyp5':
            return (0.0, 0.2, 0.0)
        raise KeyError(xc)

    @staticmethod
    def hybrid_coeff(xc):
        if xc in ('b3lyp', 'b3lyp5'):
            return 0.2
        raise KeyError(xc)

    @staticmethod
    def nlc_coeff(xc):
        if xc in ('b3lyp', 'b3lyp5'):
            return ()
        raise KeyError(xc)


class XcDefinitionTests(unittest.TestCase):
    def test_b3lyp_records_libxc402_vwn3_variant(self):
        d = runner._xc_definition_from(FakeLibxc, 'b3lyp')
        self.assertEqual(d['xc'], 'b3lyp')
        self.assertEqual(d['libxc_decomposition'], [{'id': 402,
                                                     'factor': 1.0}])
        self.assertEqual(d['hyb_coeff'], 0.2)
        self.assertEqual(d['rsh'], {'omega': 0.0, 'alpha': 0.2, 'beta': 0.0})
        # B3LYP has no VV10: the field must say None, not be silently absent
        self.assertIsNone(d['nlc_params'])

    def test_b3lyp5_decomposition_differs_from_b3lyp(self):
        # the two variants must be distinguishable from the recorded block
        d3 = runner._xc_definition_from(FakeLibxc, 'b3lyp')
        d5 = runner._xc_definition_from(FakeLibxc, 'b3lyp5')
        self.assertNotEqual(d3['libxc_decomposition'],
                            d5['libxc_decomposition'])
        ids5 = {c['id'] for c in d5['libxc_decomposition']}
        self.assertIn(7, ids5)      # VWN5 correlation id present in b3lyp5

    def test_unknown_alias_yields_null_fields_not_a_crash(self):
        d = runner._xc_definition_from(FakeLibxc, 'not_a_functional')
        self.assertEqual(d['xc'], 'not_a_functional')
        self.assertIsNone(d['libxc_decomposition'])
        self.assertIsNone(d['hyb_coeff'])
        self.assertIsNone(d['rsh'])
        self.assertIsNone(d['nlc_params'])

    def test_nlc_params_recorded_when_present(self):
        class NlcLibxc(FakeLibxc):
            @staticmethod
            def nlc_coeff(xc):
                return (((6.0, 0.01), 1),)
        d = runner._xc_definition_from(NlcLibxc, 'wb97m-v')
        self.assertEqual(d['nlc_params'], [[6.0, 0.01, 1.0]])


class UksSentinelMappingTests(unittest.TestCase):
    def test_methoxy_radical_maps_to_doublet(self):
        # the issue #5 UKS sentinel: neutral CH3O radical, spin = 2S = 1
        self.assertEqual(runner.CHARGE_SPIN['methoxy.xyz'], (0, 1))

    def test_r14_mapping_retained(self):
        self.assertEqual(runner.CHARGE_SPIN['r14.xyz'], (1, 1))

    def test_unknown_geometry_defaults_to_neutral_singlet(self):
        self.assertNotIn('benzene.xyz', runner.CHARGE_SPIN)


class LaneConvTolIsolationTests(unittest.TestCase):
    '''The fp32-screen conv_tol floor must stay inside that lane.

    Live catch from the issue #5 B3LYP smoke: flooring a shared
    args.conv_tol for the whole invocation made the auto lane converge at
    its switch cycle (zero-length fp64 tail -> the returned energy was the
    fp32 one, 2.5e-6 Ha off) and stopped the fp64 lane 1.5e-8 Ha short of
    its 1e-10-converged energy.'''

    def test_fp32_screen_lane_is_floored(self):
        self.assertEqual(runner._lane_conv_tol('fp32-screen', 1e-10),
                         3e-5)

    def test_floor_never_tightens_a_looser_user_tolerance(self):
        self.assertEqual(runner._lane_conv_tol('fp32-screen', 1e-3), 1e-3)

    def test_other_lanes_keep_the_user_tolerance(self):
        for lane in ('fp64', 'auto', 'df-vs-conv'):
            self.assertEqual(runner._lane_conv_tol(lane, 1e-10), 1e-10)


class GridStabilityAssemblyTests(unittest.TestCase):
    '''Issue #6 meta-GGA grid check: the pure assembler must record the
    grid error explicitly, classify by trend (never by an invented
    threshold) and stay diagnostic -- nothing here feeds a verdict.'''

    @staticmethod
    def _entry(level, e, converged=True, cycles=10):
        return {'level': level, 'ngrids': 1000 * level, 'e_tot': e,
                'converged': converged, 'scf_cycles': cycles, 't_s': 1.0}

    def test_deltas_and_production_anchor(self):
        block = runner.assemble_grid_stability(
            [self._entry(3, -100.0), self._entry(4, -100.000001),
             self._entry(5, -100.0000011)], 3)
        self.assertEqual(block['production_level'], 3)
        by_level = {e['level']: e for e in block['entries']}
        self.assertAlmostEqual(by_level[3]['dE_vs_production'], 0.0)
        self.assertIsNone(by_level[3]['dE_vs_prev_level'])   # coarsest level
        self.assertAlmostEqual(by_level[4]['dE_vs_prev_level'], -1e-6)
        self.assertAlmostEqual(by_level[4]['dE_vs_production'], -1e-6)
        self.assertAlmostEqual(by_level[5]['dE_vs_prev_level'], -1e-7)
        # production level 3 error vs the finest scanned level (5)
        self.assertAlmostEqual(by_level[5]['dE_vs_production'], -1.1e-6)
        self.assertEqual(block['classification'], 'converging')

    def test_production_level_always_included_by_caller(self):
        # run_grid_stability unions the requested levels with
        # args.grids_level; the assembler itself only requires the anchor
        block = runner.assemble_grid_stability(
            [self._entry(3, -100.0), self._entry(4, -100.000001)], 4)
        by_level = {e['level']: e for e in block['entries']}
        self.assertAlmostEqual(by_level[3]['dE_vs_production'], 1e-6)
        self.assertAlmostEqual(by_level[4]['dE_vs_production'], 0.0)

    def test_shrinking_deltas_classify_converging(self):
        block = runner.assemble_grid_stability(
            [self._entry(3, -100.0), self._entry(4, -100.00001),
             self._entry(5, -100.000012)], 3)
        self.assertEqual(block['classification'], 'converging')

    def test_flat_or_growing_deltas_classify_not_converging(self):
        # last refinement moved the energy at least as much as the first
        block = runner.assemble_grid_stability(
            [self._entry(3, -100.0), self._entry(4, -100.00001),
             self._entry(5, -100.00003)], 3)
        self.assertEqual(block['classification'], 'not-converging')

    def test_single_level_scan(self):
        block = runner.assemble_grid_stability([self._entry(3, -100.0)], 3)
        self.assertEqual(block['classification'], 'single-level')
        self.assertIsNone(block['entries'][0]['dE_vs_prev_level'])

    def test_unsorted_entries_are_sorted_by_level(self):
        block = runner.assemble_grid_stability(
            [self._entry(5, -100.0000011), self._entry(3, -100.0),
             self._entry(4, -100.000001)], 3)
        self.assertEqual([e['level'] for e in block['entries']], [3, 4, 5])

    def test_note_states_the_recorded_grid_error(self):
        block = runner.assemble_grid_stability(
            [self._entry(3, -100.0), self._entry(4, -100.000001)], 3)
        self.assertIn('1.000e-06', block['note'])
        self.assertIn('never an acceptance gate', block['note'])
        self.assertIn('classification', block['note'])


if __name__ == '__main__':
    unittest.main()

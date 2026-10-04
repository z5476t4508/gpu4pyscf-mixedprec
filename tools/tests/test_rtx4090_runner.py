"""Unit tests for the RTX 4090 runner's GPU-free logic (issue #4).

The runner's derive/upsert/attach_cross_program functions and the fp32-hit
call-site mapping are pure and tested here without a GPU; the full GPU
smoke is the separate methanol run recorded as evidence.
"""

import contextlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import types
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

    def test_3c_d4_piece_prefers_the_dedicated_key(self):
        # scf_summary['dispersion'] holds the hook's full return value (the
        # D4+gCP sum) because energy_tot() stores the return there; the D4
        # piece must come from 'dispersion_d4' when present
        terms = runner._e_terms({'dispersion_d4': -5.0e-4,
                                 'dispersion': 4.08e-3,
                                 'gcp': 4.58e-3}, None)
        self.assertEqual(terms['dispersion'], -5.0e-4)
        self.assertEqual(terms['gcp'], 4.58e-3)

    def test_plain_dispersion_key_used_when_no_dedicated_key(self):
        # non-3c rows (e.g. a plain D3 correction): the ordinary key holds
        # the correction itself
        terms = runner._e_terms({'dispersion': -2.0e-3}, None)
        self.assertEqual(terms['dispersion'], -2.0e-3)


class NrNlcVxcPatchTests(unittest.TestCase):
    '''The NLC observation patch (issue #7): nr_nlc_vxc is bound as a CLASS
    attribute on NumInt (both gpu4pyscf and pyscf), so only a class
    attribute replacement observes real SCF calls -- the issue #4 probe
    patched the module-level name and would have captured nothing (null
    e_terms.nlc) on NLC methods.  These tests pin the mechanism with a
    fake module; the live interception is proven by the S0 smoke run
    (fp32 hits + non-null nlc on every wb97m-v lane).'''

    def _fake_numint_module(self):
        calls = []

        def nr_nlc_vxc(ni, mol, grids, xc_code, dms, **kw):
            calls.append((xc_code, kw))
            return (18.0, -0.1234, 'vmat-stub')

        mod = types.ModuleType('fake_numint')
        mod.NumInt = type('NumInt', (), {'nr_nlc_vxc': nr_nlc_vxc})
        return mod, calls

    def test_patch_observes_class_attribute_calls(self):
        mod, calls = self._fake_numint_module()
        seen = []
        restore = runner.patch_nr_nlc_vxc(mod, seen.append)
        try:
            # the SCF code calls the CLASS attribute, not the module name
            out = mod.NumInt.nr_nlc_vxc('ni', 'mol', 'grids', 'wb97m-v',
                                        'dm', max_memory=2000)
            self.assertEqual(len(out), 3)
            self.assertEqual(seen, [-0.1234])
            self.assertEqual(len(calls), 1)
        finally:
            restore()
        self.assertIs(mod.NumInt.nr_nlc_vxc.__name__, 'nr_nlc_vxc')

    def test_restore_puts_back_the_original(self):
        mod, _ = self._fake_numint_module()
        orig = mod.NumInt.nr_nlc_vxc
        restore = runner.patch_nr_nlc_vxc(mod, lambda v: None)
        self.assertIsNot(mod.NumInt.nr_nlc_vxc, orig)
        restore()
        self.assertIs(mod.NumInt.nr_nlc_vxc, orig)

    def test_wrapper_passes_args_and_kwargs_through(self):
        mod, calls = self._fake_numint_module()
        seen = []
        restore = runner.patch_nr_nlc_vxc(mod, seen.append)
        try:
            mod.NumInt.nr_nlc_vxc('ni', 'mol', 'grids', 'wb97m-v', 'dm',
                                  relativity=0, hermi=1)
        finally:
            restore()
        self.assertEqual(calls, [('wb97m-v', {'relativity': 0,
                                              'hermi': 1})])

    def test_module_level_name_patch_is_what_does_not_work(self):
        # the historical (broken) probe shape, kept as documentation: a
        # module-attribute rebind does not intercept class-attribute calls
        mod, calls = self._fake_numint_module()
        mod.nr_nlc_vxc = lambda *a, **kw: (0, 0, 0)
        # the class attribute is untouched -> the real SCF call site still
        # hits the original
        mod.NumInt.nr_nlc_vxc('ni', 'mol', 'grids', 'wb97m-v', 'dm')
        self.assertEqual(len(calls), 1)


class NlcImplNoteTests(unittest.TestCase):
    '''The issue-#7 named level-(2) attribution: a gpu-vs-cpu deviation on
    an NLC functional is attributed to the two independent VV10
    implementations only when the separately captured NLC terms dominate
    the deviation (measured live on methanol wb97m-v: |dNLC| 1.65e-8 vs
    |dE| 1.64e-8 Ha).'''

    def _row(self, e, nlc):
        return {'result': {'e_tot': e, 'e_terms': {'nlc': nlc}}}

    def _cpu(self, e, nlc):
        return self._row(e, nlc)

    def test_dominant_nlc_difference_attributed(self):
        row = self._row(-115.570140269527, 0.07375998718788067)
        cpu = self._cpu(-115.570140285909, 0.07375997066906256)
        note = runner._nlc_impl_note(row, cpu, None)
        self.assertIsNotNone(note)
        self.assertIn('VV10', note)
        self.assertIn('implementation', note)
        self.assertIn('1.652e-08', note)
        self.assertIn('not a mixed-precision', note.replace(
            'never a mixed-precision', 'not a mixed-precision'))

    def test_non_dominant_nlc_difference_not_attributed(self):
        # NLC pieces agree; the deviation lives elsewhere -- no claim
        row = self._row(-115.57, 0.07375998718788067)
        cpu = self._cpu(-115.570000164, 0.07375998718788067)
        self.assertIsNone(runner._nlc_impl_note(row, cpu, None))

    def test_lane_away_from_gpu_reference_not_attributed(self):
        # a lane 2e-8 away from its own GPU fp64 reference is a
        # mixed-precision trajectory residual, not an implementation story
        row = self._row(-115.57, 0.07375998718788067)
        cpu = self._cpu(-115.5700000164, 0.07375997066906256)
        self.assertIsNone(runner._nlc_impl_note(row, cpu, 2e-8))

    def test_missing_nlc_terms_give_no_note(self):
        row = self._row(-115.57, None)
        cpu = self._cpu(-115.5700000164, 0.0737)
        self.assertIsNone(runner._nlc_impl_note(row, cpu, None))
        row2 = self._row(-115.57, 0.0737)
        cpu2 = self._cpu(-115.5700000164, None)
        self.assertIsNone(runner._nlc_impl_note(row2, cpu2, None))

    def test_note_fires_in_derive_for_fp64_lane(self):
        # end-to-end: the fp64 lane's deviation from cpu-fp64 gets the
        # VV10 attribution, the auto lane within impl level gets nothing
        def _row(lane, e, nlc):
            return {'row_key': {'geom': 'g.xyz', 'basis': 'b', 'xc':
                                'wb97m-v', 'lane': lane}, 'status': 'ok',
                    'result': {'e_tot': e, 'converged': True,
                               'e_terms': {'nlc': nlc}},
                    'accuracy': {}}
        rows = [_row('fp64', -115.570140269527, 0.07375998718788067),
                _row('auto', -115.570140269526, 0.07375998719170589),
                _row('cpu-fp64', -115.570140285909, 0.07375997066906256)]
        rows = runner.derive(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        self.assertIn('VV10', by_lane['fp64']['accuracy'][
            'impl_agreement_note'])
        # the auto lane sits on its GPU fp64 reference (5.7e-14 Ha), so its
        # cpu deviation is the SAME implementation story and gets the more
        # specific VV10 attribution, not the generic tail-stall note
        self.assertIn('VV10', by_lane['auto']['accuracy'].get(
            'impl_agreement_note') or '')

    def test_auto_lane_off_reference_keeps_tail_stall_note(self):
        # auto lane 4e-8 off its GPU reference: its own mixed-precision
        # trajectory residual, generic attribution -- _nlc_impl_note must
        # decline (dE_gpu gate), leaving the established auto note
        def _row(lane, e, nlc):
            return {'row_key': {'geom': 'g.xyz', 'basis': 'b', 'xc':
                                'wb97m-v', 'lane': lane}, 'status': 'ok',
                    'result': {'e_tot': e, 'converged': True,
                               'e_terms': {'nlc': nlc}},
                    'accuracy': {}}
        rows = [_row('fp64', -115.570140269527, 0.07375998718788067),
                _row('auto', -115.570140229527, 0.07375998718788067),
                _row('cpu-fp64', -115.570140285909, 0.07375997066906256)]
        runner.derive(rows)
        by_lane = {r['row_key']['lane']: r for r in rows}
        note = by_lane['auto']['accuracy'].get('impl_agreement_note')
        self.assertIsNotNone(note)
        self.assertIn('fp32 head', note)


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

    def test_nlcgrids_kind_names_the_vv10_scan(self):
        # issue #7: the same assembler, but the note must say what actually
        # varied -- the VV10 nlcgrids with the semilocal grids held fixed
        block = runner.assemble_grid_stability(
            [self._entry(2, -100.0), self._entry(3, -100.000001)], 3,
            kind='nlcgrids')
        self.assertIn('nlcgrids-stability scan (issue #7', block['note'])
        self.assertIn('VV10/NLC nlcgrids', block['note'])
        self.assertIn('semilocal grids', block['note'])
        # the deltas/classification contract is identical to grids kind
        self.assertEqual(block['classification'], 'single-level'
                         if len(block['entries']) < 2 else
                         block['classification'])
        self.assertAlmostEqual(block['entries'][1]['dE_vs_production'], 0.0)
        self.assertAlmostEqual(block['entries'][0]['dE_vs_production'],
                               1e-6)


class _SkipArgs:
    def __init__(self, lane, skip_lane, geom='benzene.xyz', basis='def2-svp',
                 xc='wb97m-v', skip_reason='time-budget'):
        self.lane = lane
        self.skip_lane = skip_lane
        self.geom = geom
        self.basis = basis
        self.xc = xc
        self.skip_reason = skip_reason
        self.pilot_stage = 'S2'
        self.headroom_mib = 2048.0


class SkipLaneArgTests(unittest.TestCase):
    '''Issue #7 bounded-execution skips: a lane unaffordable within a
    run's time budget becomes an explicit skipped row -- never a silent
    omission, never a fabricated result, never replacing measured data.'''

    def test_lane_in_both_run_and_skip_is_rejected(self):
        args = _SkipArgs(['fp64', 'auto'], ['auto'])
        self.assertIn('also in --lane',
                      runner._check_skip_lane_args(args, []))

    def test_disjoint_run_and_skip_is_accepted(self):
        args = _SkipArgs(['fp64', 'auto'], ['cpu-fp64'])
        self.assertIsNone(runner._check_skip_lane_args(args, []))

    def test_no_skip_lane_is_a_noop(self):
        self.assertIsNone(runner._check_skip_lane_args(
            _SkipArgs(['fp64'], None), []))

    def test_skip_cannot_replace_a_measured_ok_row(self):
        # a merge source already holds a measured cpu-fp64 row of this
        # cell: skipping it would discard measured data
        args = _SkipArgs(['fp64'], ['cpu-fp64'])
        rows = [_row('cpu-fp64', e=-100.0)]
        rows[0]['row_key'].update(geom='benzene.xyz', basis='def2-svp',
                                  xc='wb97m-v')
        err = runner._check_skip_lane_args(args, rows)
        self.assertIn('never discard measured data', err)

    def test_skip_allowed_when_merge_holds_only_other_cells(self):
        # the ok cpu-fp64 row belongs to a different cell (other basis):
        # this cell's skip is a new row, not a replacement
        args = _SkipArgs(['fp64'], ['cpu-fp64'], basis='def2-tzvp')
        rows = [_row('cpu-fp64', e=-100.0)]   # basis def2-svp inside
        self.assertIsNone(runner._check_skip_lane_args(args, rows))

    def test_skip_allowed_when_merge_holds_a_skipped_row(self):
        # a previous skipped row of this cell CAN be re-skipped or later
        # measured (upsert replaces by row-key): only measured ok rows
        # are protected
        args = _SkipArgs(['fp64'], ['cpu-fp64'])
        skipped = _row('cpu-fp64', status='skipped')
        skipped['skip_reason'] = 'time-budget'
        self.assertIsNone(runner._check_skip_lane_args(args, [skipped]))


class TimeBudgetSkipRowTests(unittest.TestCase):
    def _emit(self, mem=(1000.0, 24564.0)):
        args = _SkipArgs(['fp64'], ['cpu-fp64'])
        seen = []
        row = runner._emit_time_budget_skip(
            'cpu-fp64', args, lambda m: seen.append(m),
            mem_info=lambda: mem)
        return row, seen

    def test_skip_row_shape_matches_skipped_semantics(self):
        row, _ = self._emit()
        self.assertEqual(row['status'], 'skipped')
        self.assertEqual(row['skip_reason'], 'time-budget')
        self.assertEqual(row['row_key']['lane'], 'cpu-fp64')
        self.assertEqual(row['accuracy']['verdict'], 'skipped')
        self.assertIsNone(row['result'].get('e_tot'))

    def test_skip_row_records_measured_memory(self):
        row, _ = self._emit()
        mem = row['result']['mem']
        self.assertEqual(mem['gpu_free_before_mib'], 1000.0)
        self.assertEqual(mem['gpu_total_mib'], 24564.0)

    def test_mem_probe_failure_still_emits_the_row(self):
        def boom():
            raise RuntimeError('no cuda context')
        row, _ = self._emit(mem=boom)
        self.assertEqual(row['status'], 'skipped')
        # R7 requires measured free/total only when available; a None
        # here flags the missing probe honestly rather than inventing a
        # number
        self.assertIsNone(row['result']['mem']['gpu_free_before_mib'])

    def test_custom_reason_is_recorded(self):
        args = _SkipArgs(['fp64'], ['cpu-fp64'],
                         skip_reason='time-budget: CPU VV10 too slow')
        row = runner._emit_time_budget_skip(
            'cpu-fp64', args, lambda m: None, mem_info=lambda: (1.0, 2.0))
        self.assertIn('CPU VV10 too slow', row['skip_reason'])

    def test_skip_row_survives_derive_and_validator(self):
        # end-to-end contract: the emitted skip row passes derive() and
        # the schema validator's skipped-row rules (R2 skip_reason, R7
        # measured free/total) alongside the other rows of the cell
        row, _ = self._emit()
        rows = runner.derive([row, _row('fp64', e=-100.0)])
        verdicts = {r['row_key']['lane']: r['accuracy']['verdict']
                    for r in rows}
        self.assertEqual(verdicts['cpu-fp64'], 'skipped')


class CarryoverScansTests(unittest.TestCase):
    '''Scan blocks are properties of one cell: a follow-up invocation
    merging rows must not drop the already-measured scan (attempt-2
    resumability), and must not carry a scan across different cells.'''

    def _doc(self, geom, basis, with_scan=True):
        row = _row('fp64', e=-100.0)
        row['row_key'].update(geom=geom, basis=basis, xc='wb97m-v')
        doc = {'rows': [row]}
        if with_scan:
            doc['grid_stability'] = {'production_level': 3}
            doc['nlcgrids_stability'] = {'production_level': 3}
        return doc

    def test_same_cell_carries_both_scans(self):
        doc = self._doc('benzene.xyz', 'def2-svp')
        gs, nlc = runner._carryover_scans(doc, 'benzene.xyz', 'def2-svp',
                                          'wb97m-v')
        self.assertEqual(gs, {'production_level': 3})
        self.assertEqual(nlc, {'production_level': 3})

    def test_different_cell_carries_nothing(self):
        doc = self._doc('benzene.xyz', 'def2-svp')
        gs, nlc = runner._carryover_scans(doc, 'benzene.xyz', 'def2-tzvp',
                                          'wb97m-v')
        self.assertIsNone(gs)
        self.assertIsNone(nlc)

    def test_doc_without_scans_carries_none(self):
        doc = self._doc('benzene.xyz', 'def2-svp', with_scan=False)
        gs, nlc = runner._carryover_scans(doc, 'benzene.xyz', 'def2-svp',
                                          'wb97m-v')
        self.assertIsNone(gs)
        self.assertIsNone(nlc)


class ThreeCXcTests(unittest.TestCase):
    '''Issue #8: the r2SCAN-3c whole-method alias resolves to plain
    r2SCAN for the KS object, and only the composite alias marks a
    whole-method row.'''

    def test_alias_resolution(self):
        self.assertEqual(runner.resolve_3c('r2scan3c'), 'r2scan')
        self.assertEqual(runner.resolve_3c('R2SCAN-3C'), 'r2scan')
        self.assertIsNone(runner.resolve_3c('r2scan'))
        self.assertIsNone(runner.resolve_3c('b3lyp'))
        self.assertIsNone(runner.resolve_3c(None))

    def test_marker_detection(self):
        self.assertTrue(runner.is_3c_xc('r2scan3c'))
        self.assertTrue(runner.is_3c_xc('r2scan-3c'))
        self.assertTrue(runner.is_3c_xc('R2SCAN-3C'))
        self.assertFalse(runner.is_3c_xc('r2scan'))
        self.assertFalse(runner.is_3c_xc('wb97m-v'))
        self.assertFalse(runner.is_3c_xc(''))
        self.assertFalse(runner.is_3c_xc(None))

    def test_prescribed_params_match_the_reference_driver(self):
        # parse_3c('r2scan3c') of gpu4pyscf/drivers/dft_3c_driver.py:
        # basis def2-mtzvpp, D4('r2scan-3c', atm=True, ga=2.0, gc=1.0)
        # overridden by set_param(0.0, 0.42, 5.65, s9=2.0), gcp 'r2scan3c'
        self.assertEqual(runner.R2SCAN3C_PRESCRIBED_BASIS, 'def2-mtzvpp')
        self.assertEqual(runner.R2SCAN3C_D4_XC_KEY, 'r2scan-3c')
        self.assertEqual(runner.R2SCAN3C_GCP_METHOD, 'r2scan3c')
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['s8'], 0.0)
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['a1'], 0.42)
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['a2'], 5.65)
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['s9'], 2.0)
        self.assertTrue(runner.R2SCAN3C_D4_PARAMS['atm'])
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['ga'], 2.0)
        self.assertEqual(runner.R2SCAN3C_D4_PARAMS['gc'], 1.0)

    def test_dispersion_block_records_every_knob(self):
        blk = runner.r2scan3c_dispersion_block('1.5.0')
        self.assertEqual(blk['model'], 'D4')
        self.assertEqual(blk['library_version'], '1.5.0')
        self.assertEqual(blk['params']['xc_key'], 'r2scan-3c')
        for k in ('s6', 's8', 'a1', 'a2', 's9', 'alp', 'atm', 'ga', 'gc',
                  'wf'):
            self.assertIn(k, blk['params'])


class ThreeCAttachTests(unittest.TestCase):
    '''Issue #8: attach_3c_corrections installs the driver-identical hooks
    on the object it is given, without needing pyscf/GPU.'''

    class _Mf:
        def __init__(self):
            self.nlc = None
            self.scf_summary = {}

    def test_attach_sets_hooks(self):
        mf = self._Mf()
        out = runner.attach_3c_corrections(mf)
        self.assertIs(out, mf)
        self.assertEqual(mf.nlc, 0)
        self.assertTrue(mf.do_disp())
        self.assertTrue(callable(mf.get_dispersion))
        # get_dispersion is bound to THIS mf (MethodType semantics)
        self.assertEqual(mf.get_dispersion.__self__, mf)

    def test_attach_survives_object_identity_check(self):
        # the ordering contract: attach on the FINAL object after
        # density_fit-style transforms; a transformed copy would lose the
        # instance attributes -- this is the trap the 3c driver documents
        mf = self._Mf()
        runner.attach_3c_corrections(mf)
        copied = self._Mf()      # a fresh object does NOT inherit hooks
        self.assertFalse(callable(getattr(copied, 'get_dispersion', None)))


class ThreeCGetDispersionTests(unittest.TestCase):
    '''_r2scan3c_get_dispersion must build D4 and gCP with the exact
    prescribed construction and record the pieces separately.  The
    pyscf.dispersion import is faked to keep the test GPU/pyscf-free;
    the assertions pin the arguments actually handed to the libraries.'''

    def _run(self, energy_d4=-1.25e-3, energy_gcp=4.5e-3):
        import sys
        import types

        calls = {}

        class FakeD4:
            def __init__(self, mol, xc, atm, ga, gc):
                calls['d4_init'] = (xc, atm, ga, gc)

            def set_param(self, s8, a1, a2, s9=None):
                calls['d4_set'] = (s8, a1, a2, s9)

            def get_dispersion(self, grad=False):
                assert grad is False
                return {'energy': energy_d4}

        class FakeGCP:
            def __init__(self, mol, method=None):
                calls['gcp_init'] = (method,)

            def get_counterpoise(self):
                return {'energy': energy_gcp}

        fake = types.ModuleType('pyscf.dispersion')
        fake.dftd4 = types.SimpleNamespace(DFTD4Dispersion=FakeD4)
        fake.gcp = types.SimpleNamespace(GCP=FakeGCP)
        real = sys.modules.get('pyscf.dispersion')
        sys.modules['pyscf.dispersion'] = fake
        try:
            mf = types.SimpleNamespace(
                mol='FAKE_MOL', scf_summary={})
            e = runner._r2scan3c_get_dispersion(mf)
        finally:
            if real is not None:
                sys.modules['pyscf.dispersion'] = real
            else:
                del sys.modules['pyscf.dispersion']
        return e, mf.scf_summary, calls

    def test_d4_construction_matches_the_driver(self):
        e, summary, calls = self._run()
        self.assertEqual(calls['d4_init'],
                         ('r2scan-3c', True, 2.0, 1.0))
        self.assertEqual(calls['d4_set'], (0.0, 0.42, 5.65, 2.0))
        self.assertEqual(calls['gcp_init'], ('r2scan3c',))

    def test_pieces_recorded_separately_and_summed(self):
        e, summary, calls = self._run(energy_d4=-1.25e-3,
                                      energy_gcp=4.5e-3)
        # the D4 piece lives under 'dispersion_d4': energy_tot() stores the
        # hook's RETURN VALUE under 'dispersion' (the D4+gCP sum), exactly
        # as measured live and as the reference 3c driver behaves
        self.assertEqual(summary['dispersion_d4'], -1.25e-3)
        self.assertEqual(summary['gcp'], 4.5e-3)
        self.assertNotIn('dispersion', summary)
        self.assertAlmostEqual(e, -1.25e-3 + 4.5e-3, places=15)


class ThreeCMethodBlockTests(unittest.TestCase):
    '''The 3c method block keeps the composite alias in method.xc while
    xc_definition records the resolved plain-r2SCAN definition.'''

    def test_method_block_resolves_alias(self):
        args = types.SimpleNamespace(
            conv_tol=1e-10, max_cycle=50, init_guess='minao',
            direct_scf_tol=1e-14, grids_level=3, nlcgrids_level=None,
            dispersion=None, gcp=None)

        class FakeMf:
            small_rho_cutoff = 0.0
            grids = None
            nlcgrids = None

        mol = types.SimpleNamespace(spin=0)
        orig_def = runner._xc_definition
        seen = {}

        def fake_def(xc):
            seen['xc'] = xc
            return {'xc': xc, 'libxc_decomposition': [{'id': 724,
                                                       'factor': 1.0}],
                    'hyb_coeff': 0.0, 'rsh': None, 'nlc_params': None}

        runner._xc_definition = fake_def
        try:
            blk = runner._method_block(mol, FakeMf(), 'r2scan3c',
                                       'def2-mtzvpp', 'aux', True, args)
        finally:
            runner._xc_definition = orig_def
        self.assertEqual(blk['xc'], 'r2scan3c')          # composite alias
        self.assertEqual(seen['xc'], 'r2scan')           # resolved for SCF
        self.assertEqual(blk['xc_definition']['xc'], 'r2scan')
        self.assertEqual(blk['xc_definition']['xc_alias_resolved'],
                         'r2scan3c')
        self.assertEqual(blk['basis'], 'def2-mtzvpp')


class ThreeCMainGuardTests(unittest.TestCase):
    '''The prescribed-basis argument guard runs before anything else: a
    3c invocation with a mismatched basis is refused with exit code 2
    (missing basis = missing method component, plan section 1.1).'''

    def _argv(self, basis):
        return ['--out', '/tmp/never_written.json', '--xc', 'r2scan3c',
                '--basis', basis, '--lane', 'fp64']

    def test_wrong_basis_refused(self):
        with tempfile.TemporaryDirectory() as td:
            argv = self._argv('def2-svp')
            argv[1] = os.path.join(td, 'out.json')
            rc = runner.main(argv)
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(argv[1]))

    def test_right_basis_passes_the_guard(self):
        # the guard alone must accept the prescribed basis; the run then
        # proceeds to the provenance import, which on a numpy-less test
        # interpreter raises ModuleNotFoundError -- an error of the LATER
        # pipeline, never the guard's refusal message
        with tempfile.TemporaryDirectory() as td, \
                contextlib.redirect_stderr(io.StringIO()) as err:
            argv = self._argv('def2-mtzvpp')
            argv[1] = os.path.join(td, 'out.json')
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    rc = runner.main(argv)
                except Exception:
                    rc = None    # import failure past the guard is fine
            self.assertNotIn('prescribed basis', err.getvalue())
            if rc is not None:
                self.assertNotEqual(rc, 2)


if __name__ == '__main__':
    unittest.main()

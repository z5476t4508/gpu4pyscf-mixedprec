'''Issue-12 rework regressions (Codex review 2026-10-06, P1): all-repeat
scientific aggregation.

The defect under test: gradient_runner.run_*_grad_cell kept only the LAST
repeated sample's energy/convergence/gradient -- an earlier repeat that was
unconverged or inaccurate silently disappeared from the reported row, so the
gate could pass on the strength of the final sample alone.

The fixed contract (CPU-only tests, no GPU/pyscf needed):

  * ``result.repeats`` carries EVERY repeat's scientific outputs (energy,
    convergence, cycles, gradient, gradient mode, per-repeat grad_ip1 hits,
    timings sample);
  * the aggregate that lands in result.gradient/e_tot/converged is the WORST
    repeat (max |dg| vs the cell's fp64 reference; the review's requirement:
    "use worst error across all repeats"), and the accuracy verdict requires
    EVERY repeat converged on every lane -- a nonconverged FP64 reference
    repeat cannot validate candidates;
  * mixed good/bad orders fail identically (order must not matter);
  * all-good repeats still pass;
  * a failed/missing fp64 reference leaves no gate pass;
  * malformed repeat records are rejected, never silently truncated;
  * historical documents WITHOUT result.repeats are explicitly marked
    insufficient for new acceptance (never silently upgraded).

Red-first: run this module against the pre-fix gradient_runner and the
aggregate tests fail; against the fixed runner they all pass.
'''

import importlib.util
import json
import pathlib
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gr = _load('gradient_runner')
sv = _load('schema_validator')

# --- fixtures --------------------------------------------------------------

G_REF = [[0.01, -0.02, 0.03], [-0.01, 0.02, -0.03]]
G_OK = [[0.01 + 1e-6, -0.02, 0.03], [-0.01, 0.02, -0.03]]      # inside gate
G_BAD = [[0.01 + 1e-3, -0.02, 0.03], [-0.01, 0.02, -0.03]]     # 33x the gate

TIMINGS = {'warmup_s': 1.0, 'samples_s': [1.0, 1.1, 1.2], 'median_s': 1.1,
           'min_s': 1.0, 'max_s': 1.2, 'end_to_end_s': 1.1, 't_cderi_s': 0.5,
           't_scf_s': 0.6, 't_grad_s': 0.4, 't_xc_grid_s': None,
           't_jk_s': None, 't_nlc_s': None, 't_disp_s': None,
           'sync': 'device-sync before/after each phase'}


def _repeat(idx, grad, converged=True, e=-114.5, grad_mode='fp64',
            grad_ip1=0):
    return {'index': idx, 'e_tot': e, 'converged': converged,
            'scf_cycles': 11, 'gradient': grad,
            'gradient_norm': gr._gradient_norm(grad),
            'gradient_mode': grad_mode, 'grad_ip1_hits': grad_ip1,
            'sample_s': 1.0 + idx * 0.1, 't_scf_s': 0.6, 't_grad_s': 0.4,
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'}}


def _row(lane, repeats, grad=None, converged=None, e=None, grad_mode=None,
         status='ok'):
    '''Build a row the way the FIXED runner builds it: repeats retained,
    top-level aggregate fields describing one repeat (the runner picks the
    worst; tests pass the selection explicitly).'''
    if grad is None and repeats:
        grad = repeats[0]['gradient']
    if converged is None and repeats:
        converged = all(r.get('converged') for r in repeats)
    if e is None and repeats:
        e = repeats[0]['e_tot']
    if grad_mode is None and repeats:
        grad_mode = repeats[0].get('gradient_mode', 'fp64')
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
            'gradient_mode': grad_mode,
            'repeats': [dict(r) for r in repeats],
            'e_terms': {'e1': -185.0, 'coul': 55.0, 'exc': -14.0,
                        'nlc': None, 'dispersion': None, 'gcp': None},
            'timings': dict(TIMINGS),
            'mem': {'gpu_free_before_mib': 24000.0,
                    'gpu_total_mib': 24564.0, 'gpu_peak_alloc_mib': 500.0,
                    'host_rss_peak_mib': 1200.0,
                    'headroom_budget_mib': 2048.0, 'within_budget': True},
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                'grad_ip1': sum(
                                    r.get('grad_ip1_hits', 0)
                                    for r in repeats)},
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


def _gpu_row(lane, grad, grad_mode, grad_ip1, repeats):
    row = _row(lane, repeats, grad=grad, grad_mode=grad_mode)
    row['result']['fp32_entry_hits']['grad_ip1'] = grad_ip1
    return row


def _cell(ref_repeats=None, auto_repeats=None):
    '''A five-lane cell; repeats default to 3 good samples each.'''
    def reps(good):
        if good is None:
            return [_repeat(i, G_REF if i == 0 else G_OK)
                    for i in range(3)]
        return good
    ref_reps = reps(ref_repeats)
    auto_reps = reps(auto_repeats)
    rows = [
        _gpu_row('fp64', ref_reps[0]['gradient'], 'fp64', 0, ref_reps),
        _gpu_row('auto', auto_reps[0]['gradient'], 'fp32', 3, auto_reps),
        _gpu_row('fp32-grad', [_repeat(0, G_OK), _repeat(1, G_OK),
                               _repeat(2, G_OK)][0]['gradient'],
                 'fp32', 3,
                 [_repeat(i, G_OK, grad_mode='fp32', grad_ip1=1)
                  for i in range(3)]),
        _gpu_row('auto-fp64-grad', G_OK, 'fp64', 0,
                 [_repeat(i, G_OK) for i in range(3)]),
        _row('cpu-fp64', [_repeat(i, G_REF) for i in range(3)]),
    ]
    return rows


# --- worst-repeat selection -------------------------------------------------

class WorstRepeatTests(unittest.TestCase):
    def test_worst_repeat_selected_first_bad(self):
        '''The aggregate gradient must be the WORST repeat: repeat 0 fails
        the gate, repeats 1-2 pass -- the row reports repeat 0's gradient
        (order must not save the row).'''
        reps = [_repeat(0, G_BAD, grad_mode='fp32', grad_ip1=1),
                _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)]
        worst = gr.select_worst_repeat(reps, G_REF)
        self.assertIs(worst, reps[0])

    def test_worst_repeat_selected_last_bad(self):
        reps = [_repeat(0, G_OK, grad_mode='fp32', grad_ip1=1),
                _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                _repeat(2, G_BAD, grad_mode='fp32', grad_ip1=1)]
        worst = gr.select_worst_repeat(reps, G_REF)
        self.assertIs(worst, reps[2])

    def test_unconverged_repeat_is_worst(self):
        '''An unconverged repeat is always the worst, even if its gradient
        happens to sit close to the reference.'''
        reps = [_repeat(0, G_OK, converged=False, grad_mode='fp32',
                        grad_ip1=1),
                _repeat(1, G_BAD, grad_mode='fp32', grad_ip1=1)]
        worst = gr.select_worst_repeat(reps, G_REF)
        self.assertIs(worst, reps[0])

    def test_reference_lane_worst_by_nonconvergence(self):
        '''The fp64 reference lane has no reference to differ from; its
        worst repeat is any unconverged one (ties broken by index).'''
        reps = [_repeat(0, G_REF), _repeat(1, G_REF, converged=False),
                _repeat(2, G_REF)]
        worst = gr.select_worst_repeat(reps, None)
        self.assertIs(worst, reps[1])

    def test_empty_or_missing_repeats_rejected(self):
        with self.assertRaises(ValueError):
            gr.select_worst_repeat([], G_REF)
        with self.assertRaises(ValueError):
            gr.select_worst_repeat(None, G_REF)


# --- derive_grad: all-repeat gating -----------------------------------------

class AllRepeatGateTests(unittest.TestCase):
    def test_first_repeat_unconverged_last_good_fails(self):
        '''THE P1 regression: repeat 0 unconverged, final repeat good and
        converged.  The aggregate MUST be verdict fail.'''
        auto_reps = [_repeat(0, G_OK, converged=False, grad_mode='fp32',
                             grad_ip1=1),
                     _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                     _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)]
        rows = gr.derive_grad(_cell(auto_repeats=auto_reps))
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        self.assertEqual(auto['accuracy']['verdict'], 'fail')
        self.assertFalse(auto['result']['converged'])

    def test_first_repeat_inaccurate_last_good_fails(self):
        '''Same, with an inaccurate (over-gate) first repeat: the stored
        gradient must be the bad repeat's, and the verdict fail.'''
        auto_reps = [_repeat(0, G_BAD, grad_mode='fp32', grad_ip1=1),
                     _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                     _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)]
        rows = gr.derive_grad(_cell(auto_repeats=auto_reps))
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        self.assertEqual(auto['accuracy']['verdict'], 'fail')
        self.assertAlmostEqual(auto['accuracy']['max_abs_dg_vs_gpu_fp64'],
                               1e-3, delta=1e-8)
        # the aggregate gradient IS the worst repeat's gradient
        self.assertAlmostEqual(auto['result']['gradient'][0][0],
                               G_BAD[0][0])

    def test_bad_order_independent(self):
        '''[bad, ok, ok] and [ok, ok, bad] produce identical aggregates.'''
        orders = [[_repeat(0, G_BAD, grad_mode='fp32', grad_ip1=1),
                   _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                   _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)],
                  [_repeat(0, G_OK, grad_mode='fp32', grad_ip1=1),
                   _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1),
                   _repeat(2, G_BAD, grad_mode='fp32', grad_ip1=1)],
                  [_repeat(0, G_OK, grad_mode='fp32', grad_ip1=1),
                   _repeat(1, G_BAD, grad_mode='fp32', grad_ip1=1),
                   _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)]]
        results = []
        for reps in orders:
            rows = gr.derive_grad(_cell(auto_repeats=reps))
            auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
            results.append((auto['accuracy']['verdict'],
                            round(auto['accuracy']
                                  ['max_abs_dg_vs_gpu_fp64'], 12)))
        self.assertEqual(len(set(results)), 1)
        self.assertEqual(results[0][0], 'fail')

    def test_all_good_repeats_pass(self):
        rows = gr.derive_grad(_cell())
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        self.assertEqual(auto['accuracy']['verdict'], 'pass')
        self.assertEqual(len(auto['result']['repeats']), 3)
        # every repeat's gradient retained, not just the last
        self.assertAlmostEqual(auto['result']['repeats'][0]['gradient'][0][0],
                               G_REF[0][0])

    def test_nonconverged_reference_repeat_invalidates_candidates(self):
        '''A nonconverged FP64 reference repeat cannot validate candidates:
        candidate lanes go no_reference/fail, never pass.'''
        ref_reps = [_repeat(0, G_REF, converged=False),
                    _repeat(1, G_REF), _repeat(2, G_REF)]
        rows = gr.derive_grad(_cell(ref_repeats=ref_reps))
        by_lane = {r['row_key']['lane']: r for r in rows}
        self.assertNotEqual(by_lane['auto']['accuracy']['verdict'], 'pass')
        self.assertNotEqual(by_lane['fp32-grad']['accuracy']['verdict'],
                            'pass')
        self.assertIn(by_lane['auto']['accuracy']['verdict'],
                      ('fail', 'no_reference'))
        self.assertIn(by_lane['fp64']['accuracy']['verdict'],
                      ('fail', 'no_reference'))

    def test_missing_reference_stays_no_reference(self):
        rows = _cell()
        rows = [r for r in rows if r['row_key']['lane'] != 'fp64']
        rows = gr.derive_grad(rows)
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        self.assertEqual(auto['accuracy']['verdict'], 'no_reference')

    def test_malformed_repeat_record_rejected(self):
        '''A repeat missing its gradient or convergence flag is malformed:
        derive_grad must fail loudly, never silently skip the repeat.'''
        reps = [_repeat(0, G_OK, grad_mode='fp32', grad_ip1=1),
                _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1)]
        bad = _repeat(2, G_OK, grad_mode='fp32', grad_ip1=1)
        del bad['converged']
        reps.append(bad)
        with self.assertRaises(ValueError):
            gr.derive_grad(_cell(auto_repeats=reps))
        reps2 = [_repeat(0, G_OK, grad_mode='fp32', grad_ip1=1)]
        bad2 = _repeat(1, G_OK, grad_mode='fp32', grad_ip1=1)
        bad2['gradient'] = 'not-a-matrix'
        reps2.append(bad2)
        with self.assertRaises(ValueError):
            gr.derive_grad(_cell(auto_repeats=reps2))

    def test_legacy_row_without_repeats_flagged_not_upgraded(self):
        '''A historical row with no result.repeats keeps its stored verdict
        but is explicitly marked repeat_evidence='legacy-insufficient':
        never silently treated as all-repeat evidence.'''
        rows = _cell()
        for r in rows:
            r['result'].pop('repeats', None)
            r['result']['repeat_evidence'] = 'legacy-insufficient'
        rows = gr.derive_grad(rows)
        for r in rows:
            self.assertEqual(r['result']['repeat_evidence'],
                             'legacy-insufficient')


# --- validator: all-repeat evidence contract --------------------------------

def _doc(rows):
    return {'schema_version': sv.GRADIENT_SCHEMA_VERSION,
            'meta': {
                'gpu_name': 'NVIDIA GeForce RTX 4090',
                'gpu_total_mem_mib': 24564, 'driver_version': 'x',
                'cuda_version': '12.8', 'cpu_model': 'x',
                'cpu_threads': 24, 'ram_gb': 64.0, 'python': '3.12.3',
                'cupy': '13.4.1', 'pyscf': '2.14.0',
                'gpu4pyscf_source_commit': '0' * 40,
                'gpu4pyscf_source_path': '/x', 'libxc': '7.0.0',
                'mem_headroom_mib': 2048,
                'timestamp_utc': '2026-10-06T00:00:00Z', 'run_id': 't'},
            'rows': rows}


class ValidatorAllRepeatTests(unittest.TestCase):
    def test_document_with_repeats_validates(self):
        rows = gr.derive_grad(_cell())
        issues = sv.validate_gradient_document(_doc(rows), evidence_root=None)
        self.assertEqual(issues, [], msg=json.dumps(issues, indent=1))

    def test_pass_with_missing_repeats_rejected(self):
        '''A passing row without result.repeats (and without the explicit
        legacy marker) violates G6: new passes need all-repeat evidence.'''
        rows = gr.derive_grad(_cell())
        for r in rows:
            if r['row_key']['lane'] in ('auto', 'fp32-grad',
                                        'auto-fp64-grad'):
                r['result'].pop('repeats')
        issues = sv.validate_gradient_document(_doc(rows),
                                               evidence_root=None)
        tags = {(i['rule'], i['tag']) for i in issues}
        self.assertIn(('G6', 'repeats-missing'), tags)
        self.assertNotIn(('G6', 'repeats-legacy-pass'), tags)

    def test_legacy_marked_row_cannot_pass(self):
        '''repeat_evidence='legacy-insufficient' rows may exist in the
        document (historical preservation) but a 'pass' verdict on one is
        rejected -- historical artifacts cannot be upgraded into passes.'''
        rows = gr.derive_grad(_cell())
        for r in rows:
            r['result'].pop('repeats', None)
            r['result']['repeat_evidence'] = 'legacy-insufficient'
        issues = sv.validate_gradient_document(_doc(rows),
                                               evidence_root=None)
        tags = {(i['rule'], i['tag']) for i in issues}
        self.assertIn(('G6', 'repeats-legacy-pass'), tags)

    def test_repeat_count_must_match_samples(self):
        '''len(result.repeats) must equal len(timings.samples_s) -- a
        truncated repeat list is malformed evidence, not an aggregate.'''
        rows = gr.derive_grad(_cell())
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        auto['result']['repeats'] = auto['result']['repeats'][:1]
        issues = sv.validate_gradient_document(_doc(rows),
                                               evidence_root=None)
        tags = {(i['rule'], i['tag']) for i in issues}
        self.assertIn(('G6', 'repeats-count'), tags)

    def test_all_repeats_converged_required_for_pass(self):
        '''A passing row with a nonconverged repeat in result.repeats is
        rejected (the aggregate claims converged; the evidence disagrees).'''
        rows = gr.derive_grad(_cell())
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        auto['result']['repeats'][0]['converged'] = False
        issues = sv.validate_gradient_document(_doc(rows),
                                               evidence_root=None)
        tags = {(i['rule'], i['tag']) for i in issues}
        self.assertIn(('G6', 'repeat-not-converged'), tags)

    def test_worst_repeat_gradient_must_be_aggregate(self):
        '''The stored top-level gradient must equal one of the repeats
        (the worst), and a stored verdict pass is rejected when ANY repeat's
        gradient exceeds the gate vs the reference's stored gradient.'''
        rows = gr.derive_grad(_cell())
        auto = next(r for r in rows if r['row_key']['lane'] == 'auto')
        # fabricate: top-level gradient inside the gate, one repeat's
        # gradient far outside -- the old P1 camouflage
        auto['result']['repeats'][1]['gradient'] = \
            [[0.01 + 1e-3, -0.02, 0.03], [-0.01, 0.02, -0.03]]
        auto['result']['repeats'][1]['gradient_norm'] = 0.0316
        issues = sv.validate_gradient_document(_doc(rows),
                                               evidence_root=None)
        tags = {(i['rule'], i['tag']) for i in issues}
        self.assertIn(('G6', 'repeat-worse-than-aggregate'), tags)


# --- energy runner audit -----------------------------------------------------

class EnergyRunnerAuditTests(unittest.TestCase):
    '''The energy runner had the same overwrite pattern: only the final
    sample's e_tot/converged/cycles survived.  Its fixed aggregate must
    likewise keep every repeat and gate on the worst.'''

    def test_energy_worst_repeat_selection(self):
        er = _load('rtx4090_runner')
        reps = [{'index': 0, 'e_tot': -114.5, 'converged': True,
                 'scf_cycles': 11, 'sample_s': 1.0},
                {'index': 1, 'e_tot': -114.5, 'converged': False,
                 'scf_cycles': 50, 'sample_s': 1.1},
                {'index': 2, 'e_tot': -114.5, 'converged': True,
                 'scf_cycles': 11, 'sample_s': 1.2}]
        worst = er.select_worst_energy_repeat(reps)
        self.assertIs(worst, reps[1])

    def test_energy_worst_by_energy_spread(self):
        '''With every repeat converged, the worst is the one farthest from
        the repeat-median energy (the aggregate reports the outlier, not
        the median).'''
        er = _load('rtx4090_runner')
        reps = [{'index': 0, 'e_tot': -114.5, 'converged': True,
                 'scf_cycles': 11, 'sample_s': 1.0},
                {'index': 1, 'e_tot': -114.500005, 'converged': True,
                 'scf_cycles': 11, 'sample_s': 1.1},
                {'index': 2, 'e_tot': -114.5, 'converged': True,
                 'scf_cycles': 11, 'sample_s': 1.2}]
        worst = er.select_worst_energy_repeat(reps)
        self.assertIs(worst, reps[1])

    def test_energy_empty_repeats_rejected(self):
        er = _load('rtx4090_runner')
        with self.assertRaises(ValueError):
            er.select_worst_energy_repeat([])
        with self.assertRaises(ValueError):
            er.select_worst_energy_repeat(None)


if __name__ == '__main__':
    unittest.main()

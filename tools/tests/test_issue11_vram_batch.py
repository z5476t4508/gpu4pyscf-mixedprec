'''Unit tests for the issue #11 deliverables (pure cores, no GPU/pyscf).

Covers:
  * vram_mapper: predict_footprint, gate_decision (device/headroom and
    host-RAM branches), derive_boundaries (oom vs skipped separation,
    honesty fields), cell_key, ladder parsing;
  * chunk_audit: fix-site presence, formula walks stay memory-bounded,
    new-fixed-extent scanner behavior on synthetic sources;
  * batch_channel: resume semantics (ok rows never re-run, failed rows
    only with --retry-failed, running reset to pending), crash-safe
    upsert, failure recording through the driver with a fake executor.

These import the modules through the repo layout used by the runners.
'''

import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
RTX4090 = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                       'mixedprec', 'benchmark', 'rtx4090')
sys.path.insert(0, RTX4090)

import vram_mapper   # noqa: E402
import chunk_audit   # noqa: E402
import batch_channel  # noqa: E402


class TestPredictFootprint(unittest.TestCase):
    def test_triangle_pairs_and_bytes(self):
        p = vram_mapper.predict_footprint(1000, 2000)
        self.assertEqual(p['npairs'], 1000 * 1001 // 2)
        self.assertEqual(p['cderi_upper_bound_bytes'],
                         p['npairs'] * 2000 * 8)
        self.assertAlmostEqual(p['cderi_upper_bound_gib'],
                               p['cderi_upper_bound_bytes'] / (1 << 30))

    def test_tamoxifen_tzvp_matches_measured_scale(self):
        # cross-check against the numbers measured while planning the
        # ladder: 1042 AO / 2626 naux -> ~10.6 GiB upper bound
        p = vram_mapper.predict_footprint(1042, 2626)
        self.assertAlmostEqual(p['cderi_upper_bound_gib'], 10.6, delta=0.2)


class TestGateDecisionCurrentFree(unittest.TestCase):
    '''Admission contract (review 2026-10-06, P1): the gate admits on the
    MEASURED current free VRAM minus the reserve, never on device total
    capacity.  Total capacity is theoretical reporting only.'''

    def test_p1_regression_low_free_never_admits(self):
        # The exact review reproduction: an 8 GiB request with 1 GiB free
        # and 2 GiB reserve was ADMITTED by the old total-based gate
        # (gate_decision(8, 24564, 1024, 2048) -> (True, None)).
        run, reason = vram_mapper.gate_decision(
            8.0, gpu_total_mib=24564, gpu_free_mib=1024, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('nonpositive-admission-budget', reason)

    def test_ample_capacity_admits(self):
        # normal free-VRAM admission: 10 GiB request, ~23 GiB free, 2 GiB
        # reserve -> budget ~21.3 GiB, admits
        run, reason = vram_mapper.gate_decision(
            10.0, gpu_total_mib=24564, gpu_free_mib=23656, reserve_mib=2048)
        self.assertTrue(run)
        self.assertIsNone(reason)

    def test_low_free_skips_even_when_capacity_suffices(self):
        # 8 GiB request on a 24 GB device with 9 GiB free, 2 GiB reserve:
        # budget 7 GiB < request -> skip; the reason must name the
        # current-free basis and point at a retry, not at the device
        run, reason = vram_mapper.gate_decision(
            8.0, gpu_total_mib=24564, gpu_free_mib=9216, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('predicted-exceeds-free-minus-reserve', reason)
        self.assertIn('retry', reason)

    def test_reserve_larger_than_free_skips_any_request(self):
        # free 1 GiB < reserve 2 GiB: budget is negative, even a 100 MiB
        # request must not be admitted
        run, reason = vram_mapper.gate_decision(
            0.1, gpu_total_mib=24564, gpu_free_mib=1024, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('nonpositive-admission-budget', reason)

    def test_reserve_equal_to_free_is_nonpositive_budget(self):
        # boundary: free == reserve -> budget 0 -> skip safely
        run, reason = vram_mapper.gate_decision(
            0.1, gpu_total_mib=24564, gpu_free_mib=2048, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('nonpositive-admission-budget', reason)

    def test_boundary_equality_admits(self):
        # request exactly equal to the admission budget admits (<=)
        # 7 GiB = (9216 - 2048) MiB / 1024
        run, reason = vram_mapper.gate_decision(
            7.0, gpu_total_mib=24564, gpu_free_mib=9216, reserve_mib=2048)
        self.assertTrue(run)
        self.assertIsNone(reason)
        # and just above it skips
        run2, reason2 = vram_mapper.gate_decision(
            7.0 + 1.0 / 1024, gpu_total_mib=24564, gpu_free_mib=9216,
            reserve_mib=2048)
        self.assertFalse(run2)
        self.assertIn('predicted-exceeds-free-minus-reserve', reason2)

    def test_impossible_request_beyond_device_capacity(self):
        # 59 GiB cderi vs ~21.9 GiB empty-device budget (Aza tzvp case):
        # skipped as beyond capacity, distinct from a low-free skip
        run, reason = vram_mapper.gate_decision(
            59.0, gpu_total_mib=24564, gpu_free_mib=23656, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('beyond-device-capacity', reason)

    def test_capacity_skip_preferred_when_both_apply(self):
        # a request beyond empty-device capacity stays a capacity skip
        # even when current free is also low (capacity is the permanent
        # verdict; low free is transient)
        run, reason = vram_mapper.gate_decision(
            59.0, gpu_total_mib=24564, gpu_free_mib=4096, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('beyond-device-capacity', reason)

    def test_host_fallback_admits_with_positive_gpu_working_budget(self):
        # host-resident cderi: admission only needs a positive GPU working
        # budget (free - reserve) plus host RAM for tensor + margin
        run, reason = vram_mapper.gate_decision(
            10.6, gpu_total_mib=24564, gpu_free_mib=23656, reserve_mib=2048,
            host_mode=True, host_avail_gib=20.0)
        self.assertTrue(run)
        self.assertIsNone(reason)

    def test_host_fallback_requires_host_avail(self):
        run, reason = vram_mapper.gate_decision(
            10.6, 24564, 23656, 2048, host_mode=True, host_avail_gib=None)
        self.assertFalse(run)
        self.assertIn('host_avail_gib', reason)

    def test_host_fallback_respects_host_ram_bound(self):
        # 10.6 GiB tensor + 4 GiB margin vs 12 GiB host -> skip
        run, reason = vram_mapper.gate_decision(
            10.6, 24564, 23656, 2048, host_mode=True, host_avail_gib=12.0)
        self.assertFalse(run)
        self.assertIn('predicted-exceeds-host-ram', reason)

    def test_host_fallback_still_needs_positive_gpu_working_budget(self):
        # host mode is not a bypass of the free-memory check: free below
        # reserve means no GPU working budget at all, skip
        run, reason = vram_mapper.gate_decision(
            10.6, 24564, 1024, 2048, host_mode=True, host_avail_gib=20.0)
        self.assertFalse(run)
        self.assertIn('nonpositive-admission-budget', reason)


class TestDeviceCapacityNote(unittest.TestCase):
    def test_capacity_report_is_separate_from_admission(self):
        note = vram_mapper.device_capacity_note(59.0, 24564, 2048)
        self.assertFalse(note['fits_empty_device'])
        self.assertEqual(note['gpu_total_mib'], 24564.0)
        self.assertEqual(note['reserve_mib'], 2048.0)
        # ~22.0 GiB empty-device budget at 24 GiB - 2 GiB reserve
        self.assertAlmostEqual(note['empty_device_budget_gib'],
                               (24564 - 2048) / 1024, delta=0.01)
        self.assertIn('never this', note['note'])

    def test_capacity_note_fitting_request(self):
        note = vram_mapper.device_capacity_note(10.6, 24564, 2048)
        self.assertTrue(note['fits_empty_device'])

    def test_capacity_note_zero_budget(self):
        # reserve >= total: no positive capacity, nothing fits
        note = vram_mapper.device_capacity_note(0.0, 2048, 2048)
        self.assertFalse(note['fits_empty_device'])


class TestGateDecisionLegacyKwarg(unittest.TestCase):
    def test_headroom_kwarg_is_the_reserve(self):
        # back-compat alias: the old callers' headroom_mib keyword is the
        # reserve; same numbers must give the same decision
        run_kw, _ = vram_mapper.gate_decision(
            10.0, gpu_total_mib=24564, gpu_free_mib=23656,
            headroom_mib=4096)
        run_pos, _ = vram_mapper.gate_decision(
            10.0, gpu_total_mib=24564, gpu_free_mib=23656, reserve_mib=4096)
        self.assertEqual(run_kw, run_pos)
        self.assertTrue(run_kw)


def _cell(status, nao=100, key='c', **kw):
    base = {'cell_key': key, 'status': status, 'geom': 'g', 'basis': 'b',
            'xc': 'x', 'lane': 'fp64', 'grad': False,
            'force_host_cderi': False,
            'predicted': {'nao': nao, 'cderi_upper_bound_gib': 1.0}}
    base.update(kw)
    return base


class TestDeriveBoundaries(unittest.TestCase):
    def test_oom_and_skipped_recorded_separately(self):
        cells = [_cell('ok', 222, 'small'),
                 _cell('ok', 537, 'mid'),
                 _cell('oom', 1042, 'wall',
                       result={'error': {'type': 'MemoryError',
                                         'message': 'out of memory'}}),
                 _cell('skipped', 1845, 'beyond', skip_reason='budget')]
        b = vram_mapper.derive_boundaries(cells)
        self.assertEqual(b['energy_fp64']['largest_ok_cell']['nao'], 537)
        self.assertEqual(len(b['oom_observed']), 1)
        self.assertEqual(b['oom_observed'][0]['cell'], 'wall')
        self.assertEqual(len(b['skipped_boundary']), 1)
        self.assertEqual(b['skipped_boundary'][0]['reason'], 'budget')
        self.assertIn('relabelled', b['honesty_note'])
        self.assertIn('separately', b['honesty_note'])

    def test_budget_violation_surfaced_not_hidden(self):
        cells = [_cell('ok', 222, 'v', result={'mem': {
            'gpu_peak_alloc_mib': 23000.0, 'within_budget': False}})]
        b = vram_mapper.derive_boundaries(cells)
        self.assertEqual(b['headroom_budget_violations'],
                         [{'cell': 'v', 'peak_mib': 23000.0}])

    def test_gradient_boundary_tracks_grad_cells_only(self):
        cells = [_cell('ok', 537, 'scf-only'),
                 _cell('ok', 222, 'grad-cell', grad=True,
                       result={'grad': {'max_abs_component_eh_bohr': 1e-3}})]
        b = vram_mapper.derive_boundaries(cells)
        self.assertEqual(b['gradient_fp64']['largest_ok_cell']['cell'],
                         'grad-cell')

    def test_host_cderi_cells_excluded_from_energy_boundary(self):
        cells = [_cell('ok', 1042, 'hostcell', force_host_cderi=True)]
        b = vram_mapper.derive_boundaries(cells)
        self.assertIsNone(b['energy_fp64']['largest_ok_cell'])

    def test_cderi_residency_collected(self):
        cells = [_cell('ok', 222, 'g1',
                       result={'mem': {'cderi_residency': 'gpu'}}),
                 _cell('ok', 537, 'h1', force_host_cderi=True,
                       result={'mem': {'cderi_residency': 'host'}})]
        b = vram_mapper.derive_boundaries(cells)
        self.assertEqual(len(b['cderi_residency']['gpu']), 1)
        self.assertEqual(len(b['cderi_residency']['host']), 1)
        self.assertEqual(
            b['cderi_residency']['host'][0]['cderi_upper_bound_gib'], 1.0)

    def test_non_terminal_status_listed(self):
        b = vram_mapper.derive_boundaries([_cell('timeout', 999, 't')])
        self.assertEqual(b['non_terminal_status_cells'], ['t'])


class TestAdmissionGateAudit(unittest.TestCase):
    '''The accepted-versus-skipped audit: every cell that went through
    VRAM admission must carry its current-free-minus-reserve gate record;
    missing records surface, never assumed (review 2026-10-06).'''

    def _gated(self, status, key='c', skip_reason=None, **kw):
        cell = _cell(status, 100, key, skip_reason=skip_reason, **kw)
        cell['gate'] = {
            'admission_basis': 'current-free-minus-reserve',
            'admitted': status == 'ok',
            'skip_reason': skip_reason if status == 'skipped' else None,
            'gpu_free_before_mib': 20000.0, 'gpu_total_mib': 24564.0,
            'reserve_mib': 2048.0, 'admission_budget_mib': 17952.0,
            'accounting': {'resident_request_gib': 1.0},
            'device_capacity': {'fits_empty_device': True}}
        return cell

    def test_missing_gate_record_surfaced(self):
        b = vram_mapper.derive_boundaries([
            self._gated('ok', 'with-gate'), _cell('ok', 222, 'no-gate')])
        audit = b['admission_gate_audit']
        self.assertEqual(audit['cells_missing_gate_record'], ['no-gate'])
        self.assertEqual(audit['admission_basis'],
                         'current-free-minus-reserve')

    def test_ok_oom_and_vram_skipped_cells_need_gate(self):
        cells = [self._gated('ok', 'a'),
                 _cell('oom', 1042, 'wall',
                       result={'error': {'type': 'MemoryError',
                                         'message': 'out of memory'}}),
                 self._gated('skipped', 's',
                             skip_reason='nonpositive-admission-budget: x')]
        b = vram_mapper.derive_boundaries(cells)
        audit = b['admission_gate_audit']
        self.assertEqual(audit['cells_missing_gate_record'], ['wall'])

    def test_time_budget_skip_exempt_from_gate_audit(self):
        # a time-budget skip never reached the GPU gate (parent-side
        # decision): it must not be reported as a missing gate record
        cells = [self._gated('ok', 'a'),
                 _cell('skipped', 999, 'tb', skip_reason='time-budget')]
        b = vram_mapper.derive_boundaries(cells)
        audit = b['admission_gate_audit']
        self.assertEqual(audit['cells_missing_gate_record'], [])

    def test_skip_reason_histogram_families(self):
        cells = [self._gated('skipped', 'a',
                             skip_reason='beyond-device-capacity: x'),
                 self._gated('skipped', 'b',
                             skip_reason='beyond-device-capacity: y'),
                 self._gated('skipped', 'c',
                             skip_reason='time-budget')]
        b = vram_mapper.derive_boundaries(cells)
        hist = b['admission_gate_audit']['skipped_reason_histogram']
        self.assertEqual(hist, {'beyond-device-capacity': 2,
                                'time-budget': 1})

    def test_honesty_note_names_the_admission_basis(self):
        b = vram_mapper.derive_boundaries([self._gated('ok', 'a')])
        self.assertIn('current free VRAM minus reserve',
                      b['honesty_note'])
        self.assertIn('never device total capacity', b['honesty_note'])


class TestCellKeyAndLadder(unittest.TestCase):
    def test_cell_key_normalizes(self):
        self.assertEqual(
            vram_mapper.cell_key('a/b.xyz', 'def2-tzvp', 'r2scan'),
            'b_def2tzvp_r2scan')
        self.assertEqual(
            vram_mapper.cell_key('a/b.xyz', 'def2-svp', 'r2scan', True),
            'b_def2svp_r2scan_hostcderi')

    def test_custom_ladder_parsing(self):
        args = type('A', (), {'cells':
                              'g.xyz:def2-svp:b3lyp:grad,h.xyz:def2-tzvp:r2scan:host'})
        ladder = vram_mapper.ladder_of(args)
        self.assertEqual(ladder[0][3], True)   # grad flag
        self.assertEqual(ladder[0][4], False)  # host flag
        self.assertEqual(ladder[1][3], False)
        self.assertEqual(ladder[1][4], True)

    def test_default_ladder_shape(self):
        for (geom, basis, xc, grad, host, note) in vram_mapper.DEFAULT_LADDER:
            self.assertTrue(os.path.exists(
                os.path.join(os.path.dirname(RTX4090), 'geoms', geom))
                or os.path.exists(geom), f'missing geometry {geom}')
            self.assertTrue(note)


class TestChunkAudit(unittest.TestCase):
    def test_fix_sites_present_in_current_tree(self):
        report = chunk_audit.run_audit()
        self.assertTrue(report['fix_sites_all_memory_consulted'])
        sites = [s['site'] for s in report['fix_sites']]
        self.assertEqual(len(sites), 6)  # H1-H4 + 2 df/grad paths

    def test_formula_walk_bounded(self):
        w = chunk_audit.formula_walk()
        self.assertTrue(w['all_batches_positive'])
        self.assertTrue(w['all_within_fractions'])
        # the 3370-AO row is the historical incident size: H1 must be
        # memory-bound there or the fixed-32 branch is documented
        row = [r for r in w['rows'] if r['nao'] == 3370][0]
        self.assertTrue(row['H1_fixed32_active']
                        or row['H1_within_eighth_when_memory_bound'])

    def test_formula_walk_low_memory_still_bounded(self):
        # 4 GiB mock: every site must shrink, never exceed its fraction
        w = chunk_audit.formula_walk(mem_mib=4096.0)
        self.assertTrue(w['all_batches_positive'])
        self.assertTrue(w['all_within_fractions'])

    def test_scanner_flags_literal_extent(self):
        import re
        src = ('x = cp.zeros([4096, nao, nao])\n'
               'y = cupy.empty((nao, nao))\n'
               'z = cp.zeros([32, 3, nao*nao])\n')
        hits = []
        for m in chunk_audit.LITERAL_ALLOC.finditer(src):
            if int(m.group(3)) >= 100:
                hits.append(int(m.group(3)))
        self.assertEqual(hits, [4096])

    def test_scanner_clean_on_current_tree(self):
        report = chunk_audit.run_audit()
        self.assertEqual(
            report['verdict']['new_sites_require_triage'], 0,
            f"untriaged literal-extent allocations: "
            f"{report['new_fixed_extent_findings']}")

    def test_h1_walk_formula(self):
        # mirrors the live itemsize-corrected formula:
        # min(32, natm, max(1, (avail/8) // 3*nao^2))
        mem = 22000 * 2**20
        b = chunk_audit.walk_h1_batch(3370, 95, mem)
        # 27.6 GB tensor shrunk to ~2.7 GiB = 1/8 of 22 GiB
        self.assertEqual(b, 10)
        self.assertLessEqual(b * 3 * 3370 * 3370 * 8, mem * 0.125 + 1)
        b2 = chunk_audit.walk_h1_batch(300, 10, mem)
        self.assertEqual(b2, 10)  # natm-limited at small AO
        b3 = chunk_audit.walk_h1_batch(3000, 95, 4096 * 2**20)
        self.assertLessEqual(b3 * 3 * 3000 * 3000 * 8, 4096 * 2**20 * 0.125 + 1)


class FakeExec:
    '''Executor stand-in for the batch driver tests.'''

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def __call__(self, cell):
        self.calls.append(batch_channel.row_key(cell))
        return self.outcomes.pop(0)


CELL = {'geom': 'a.xyz', 'basis': 'def2-svp', 'xc': 'r2scan', 'lane': 'auto'}


class TestBatchChannelResume(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, 'run.json')
        self.cells = [dict(CELL), dict(CELL, xc='b3lyp'),
                      dict(CELL, xc='hf')]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    def _run(self, outcomes, retry=False, cells=None):
        ex = FakeExec(outcomes)
        doc = batch_channel.run_batch(cells or self.cells, self.out,
                                      ex, retry_failed=retry, log=lambda *a: None)
        return doc, ex

    def test_all_new_cells_run_and_upsert(self):
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok',
             'result': {'e_tot': -1.0}},
            {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok',
             'result': {'e_tot': -2.0}},
            {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok',
             'result': {'e_tot': -3.0}}])
        self.assertEqual(len(ex.calls), 3)
        self.assertEqual(doc['summary']['status_counts'], {'ok': 3})

    def test_ok_rows_never_rerun(self):
        self._run([{'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'},
                   {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
                   {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}])
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}])
        self.assertEqual(ex.calls, [])  # everything already terminal
        self.assertEqual(doc['summary']['ran'], 0)

    def test_failed_rows_rerun_only_with_retry(self):
        fail = {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'failed',
                'error': {'type': 'X', 'message': 'boom'}}
        oks = [{'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
               {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}]
        self._run([fail] + oks)
        # without --retry-failed the failed row is left untouched
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'}])
        self.assertEqual(ex.calls, [])
        # with retry only the failed row re-runs and can recover
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'}],
            retry=True)
        self.assertEqual(ex.calls, ['a.xyz|def2-svp|r2scan|auto'])
        self.assertEqual(doc['summary']['status_counts'], {'ok': 3})

    def test_running_rows_reset_to_pending_and_rerun(self):
        stuck = {'row_key': 'a.xyz|def2-svp|r2scan|auto',
                 'status': 'running'}
        with open(self.out, 'w') as f:
            json.dump({'schema_version': 'batch-channel-v1',
                       'meta': {}, 'rows': [stuck]}, f)
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}])
        self.assertEqual(ex.calls[0], 'a.xyz|def2-svp|r2scan|auto')
        row = [r for r in doc['rows']
               if r['row_key'] == 'a.xyz|def2-svp|r2scan|auto'][0]
        self.assertEqual(row['status'], 'ok')

    def test_corrupt_output_restarts(self):
        with open(self.out, 'w') as f:
            f.write('{not json')
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}])
        self.assertEqual(len(ex.calls), 3)

    def test_wrong_schema_restarts(self):
        with open(self.out, 'w') as f:
            json.dump({'schema_version': 'something-else', 'rows': []}, f)
        doc, ex = self._run([
            {'row_key': 'a.xyz|def2-svp|r2scan|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|b3lyp|auto', 'status': 'ok'},
            {'row_key': 'a.xyz|def2-svp|hf|auto', 'status': 'ok'}])
        self.assertEqual(len(ex.calls), 3)

    def test_exception_in_executor_recorded_not_raised(self):
        def boom(cell):
            raise RuntimeError('executor exploded')
        doc = batch_channel.run_batch(
            [dict(CELL), dict(CELL, xc='b3lyp')], self.out, boom,
            log=lambda *a: None)
        # the driver converts a crashing executor into failed rows and
        # keeps going; nothing is raised and nothing is lost
        self.assertEqual(doc['summary']['status_counts'],
                         {'failed': 2})
        rows = {r['row_key']: r for r in doc['rows']}
        self.assertEqual(
            rows['a.xyz|def2-svp|r2scan|auto']['error']['type'],
            'RuntimeError')

    def test_state_written_after_every_cell(self):
        # two cells; the executor for the second crashes the process via
        # SystemExit to prove cell 1 was already durable
        ex = FakeExec([{'row_key': 'a.xyz|def2-svp|r2scan|auto',
                        'status': 'ok', 'result': {'e_tot': -1.0}}])

        class Stop(SystemExit):
            pass

        def exec2(cell):
            if ex.calls:
                raise Stop()
            return ex(cell)
        try:
            batch_channel.run_batch([dict(CELL), dict(CELL, xc='b3lyp')],
                                    self.out, exec2, log=lambda *a: None)
        except Stop:
            pass
        with open(self.out) as f:
            on_disk = json.load(f)
        keys = {r['row_key']: r['status'] for r in on_disk['rows']}
        self.assertEqual(keys.get('a.xyz|def2-svp|r2scan|auto'), 'ok')


class TestBatchChannelAdmission(unittest.TestCase):
    '''The batch channel's admission contract: skipped (budget) rows are
    re-attempted on a later invocation, unlike failed rows; the gate
    record is the vram_mapper shape.  Pure -- the executor is faked.'''

    def setUp(self):
        import shutil
        self.dir = tempfile.mkdtemp()
        self.out = os.path.join(self.dir, 'run.json')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.dir, ignore_errors=True)

    GATED_SKIP = {
        'row_key': 'a.xyz|def2-svp|r2scan|auto', 'lane': 'auto',
        'status': 'skipped',
        'skip_reason': 'predicted-exceeds-free-minus-reserve: request '
                       '2.0 GiB > budget 1.0 GiB (free 3072 MiB - reserve '
                       '2048 MiB); the request fits an empty device '
                       '(capacity reporting), so retry when more memory '
                       'is free rather than relabelling',
        'gate': {'admission_basis': 'current-free-minus-reserve',
                 'admitted': False, 'reserve_mib': 2048.0}}

    def test_skipped_rows_are_reattempted_on_restart(self):
        # a prior invocation skipped a cell for free-memory reasons; the
        # next invocation (maybe more VRAM free now) must re-attempt it
        with open(self.out, 'w') as f:
            json.dump({'schema_version': 'batch-channel-v1', 'meta': {},
                       'rows': [dict(self.GATED_SKIP)]}, f)
        ex = FakeExec([{'row_key': 'a.xyz|def2-svp|r2scan|auto',
                        'status': 'ok', 'result': {'e_tot': -1.0}}])
        doc = batch_channel.run_batch([dict(CELL)], self.out, ex,
                                      log=lambda *a: None)
        self.assertEqual(ex.calls, ['a.xyz|def2-svp|r2scan|auto'])
        row = doc['rows'][0]
        self.assertEqual(row['status'], 'ok')
        self.assertNotIn('skip_reason', row)  # the skip did not persist

    def test_failed_rows_still_need_retry_flag(self):
        # failed (execution error) rows keep the old semantics: only
        # re-run with --retry-failed; a budget skip is NOT a failure
        failed = {'row_key': 'a.xyz|def2-svp|r2scan|auto',
                  'status': 'failed',
                  'error': {'type': 'X', 'message': 'boom'}}
        with open(self.out, 'w') as f:
            json.dump({'schema_version': 'batch-channel-v1', 'meta': {},
                       'rows': [failed]}, f)
        ex = FakeExec([])
        batch_channel.run_batch([dict(CELL)], self.out, ex,
                                log=lambda *a: None)
        self.assertEqual(ex.calls, [])
        batch_channel.run_batch([dict(CELL)], self.out, ex,
                                retry_failed=True, log=lambda *a: None)

    def test_summary_reports_skip_counts(self):
        # accepted-versus-skipped audit: skips are counted in the batch
        # summary, never silently dropped
        ex = FakeExec([dict(self.GATED_SKIP)])
        doc = batch_channel.run_batch([dict(CELL)], self.out, ex,
                                      log=lambda *a: None)
        self.assertEqual(doc['summary']['status_counts'],
                         {'skipped': 1})
        self.assertEqual(doc['summary']['ran'], 1)

    def test_gate_record_shape_matches_vram_mapper(self):
        # the batch gate record is built by the same contract as the
        # vram_mapper probe gate: current-free-minus-reserve basis, the
        # same required fields, and the same decision for the same
        # numbers
        pred = vram_mapper.predict_footprint(537, 2626)   # ~2.8 GiB
        run, gate = batch_channel.admission_gate_record(
            pred, free_mib=23656, total_mib=24564, reserve_mib=2048)
        self.assertTrue(run)
        self.assertEqual(gate['admission_basis'],
                         'current-free-minus-reserve')
        for field in ('admitted', 'skip_reason', 'gpu_free_before_mib',
                      'gpu_total_mib', 'reserve_mib',
                      'admission_budget_mib', 'accounting'):
            self.assertIn(field, gate)
        self.assertAlmostEqual(gate['admission_budget_mib'],
                               23656 - 2048, delta=0.1)
        # identical numbers must give the identical decision as the
        # vram_mapper gate itself
        run2, reason2 = vram_mapper.gate_decision(
            pred['cderi_upper_bound_gib'], 24564, 23656, 2048)
        self.assertEqual((run, gate['skip_reason']), (run2, reason2))

    def test_gate_record_low_free_skips(self):
        # the same P1 scenario through the batch entry point: 8 GiB
        # request, 1 GiB free, 2 GiB reserve -> skipped row material
        pred = vram_mapper.predict_footprint(1000, 8590)   # ~8 GiB
        run, gate = batch_channel.admission_gate_record(
            pred, free_mib=1024, total_mib=24564, reserve_mib=2048)
        self.assertFalse(run)
        self.assertIn('nonpositive-admission-budget', gate['skip_reason'])


if __name__ == '__main__':
    unittest.main()

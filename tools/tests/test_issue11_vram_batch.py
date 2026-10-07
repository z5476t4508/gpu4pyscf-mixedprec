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


class TestGateDecision(unittest.TestCase):
    def test_fits(self):
        run, reason = vram_mapper.gate_decision(
            10.0, gpu_total_mib=24083, gpu_free_mib=23656, headroom_mib=2048)
        self.assertTrue(run)
        self.assertIsNone(reason)

    def test_exceeds_device_budget_skipped(self):
        # 59 GiB cderi vs ~21.5 GiB device budget: Aza tzvp case
        run, reason = vram_mapper.gate_decision(
            59.0, gpu_total_mib=24083, gpu_free_mib=23656, headroom_mib=2048)
        self.assertFalse(run)
        self.assertIn('predicted-exceeds-device-headroom', reason)

    def test_headroom_makes_small_tensor_skip(self):
        # tight headroom flips an otherwise fitting cell to skipped
        run, _ = vram_mapper.gate_decision(
            20.0, gpu_total_mib=24083, gpu_free_mib=23656, headroom_mib=4096)
        self.assertFalse(run)
        run2, _ = vram_mapper.gate_decision(
            20.0, gpu_total_mib=24083, gpu_free_mib=23656, headroom_mib=2048)
        self.assertTrue(run2)

    def test_host_mode_requires_host_avail(self):
        run, reason = vram_mapper.gate_decision(
            10.0, 24083, 23656, 2048, host_mode=True, host_avail_gib=None)
        self.assertFalse(run)
        self.assertIn('host_avail_gib', reason)

    def test_host_mode_respects_host_ram(self):
        # 10.6 GiB tensor + 4 GiB margin vs 12 GiB host -> skip
        run, reason = vram_mapper.gate_decision(
            10.6, 24083, 23656, 2048, host_mode=True, host_avail_gib=12.0)
        self.assertFalse(run)
        self.assertIn('predicted-exceeds-host-ram', reason)
        run2, _ = vram_mapper.gate_decision(
            10.6, 24083, 23656, 2048, host_mode=True, host_avail_gib=20.0)
        self.assertTrue(run2)


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


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Tests for upstream_cell_audit.py (per-cell deployed-contract +
honesty audit of a four-arm upstream comparison batch).

stdlib only; no GPU, no cupy import (the LC-205i operational rule:
nothing here may open a CUDA context while the batch driver is live).
The reporter contract is exercised through the sha256-pinned
deployment copies in tools/reporting/ with the official certificate and
evidence root mocked to a temp directory, reusing the synthetic-record
recipe proven against the deployed load_cell in
test_upstream_comparison.py.
"""

import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'
_TESTS = REPO / 'tools' / 'tests'
_REPORTING = REPO / 'tools' / 'reporting'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_tests(name):
    spec = importlib.util.spec_from_file_location(
        name, _TESTS / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))
if str(_REPORTING) not in sys.path:
    sys.path.insert(0, str(_REPORTING))

audit = _load('upstream_cell_audit')
_tuc = _load_tests('test_upstream_comparison')
uc = _tuc.uc
us = _tuc.us
_synthetic_record = _tuc._synthetic_record


class CellAuditTests(unittest.TestCase):
    BATCH = 'audit-batch'
    SCHED_SHA = 'e' * 64
    COMMIT = 'c' * 40

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        cert = self.base / 'reference.json'
        cert.write_text(json.dumps({
            'package_root': str(self.base / 'official-1.8.1'),
            'version': '1.8.1'}))
        self.cert = cert
        self.cert_hash = hashlib.sha256(cert.read_bytes()).hexdigest()
        # patch BOTH the tool's own effects/report instances (the tool
        # rebinds report.effects to its effects, so one pair covers all)
        patch_t = mock.patch.object(audit.effects, 'TASKS', self.base)
        patch_r = mock.patch.object(audit.report, 'REFERENCE', cert)
        patch_t.start()
        patch_r.start()
        self.addCleanup(patch_t.stop)
        self.addCleanup(patch_r.stop)

    def _write_cell(self, tag, seconds, energies=None, mutate=None,
                    include=('upstream', 'optimized_fp64',
                             'optimized_auto', 'strict_fp64')):
        '''One four-role cell written to the temp evidence root; returns
        the index entry (or a partial entry when `include` is short).'''
        energies = energies or {}
        sources = {}
        for role in ('upstream', 'optimized_fp64', 'optimized_auto',
                     'strict_fp64'):
            if role not in include:
                continue
            samples = []
            for b in range(4):
                rec = _synthetic_record(
                    role, b, self.BATCH, self.SCHED_SHA,
                    e=energies.get(role, -76.0),
                    lane='auto' if role == 'optimized_auto' else 'fp64')
                for r in rec['row']['result']['repeats']:
                    r.update(sample_s=seconds[role],
                             t_scf_s=seconds[role] / 2,
                             t_grad_s=seconds[role] / 4)
                if role == 'optimized_auto':
                    rec['row']['result']['repeats'][0][
                        'gradient_mode'] = 'fp32'
                    for r in rec['row']['result']['repeats']:
                        r['grad_ip1_hits'] = 3
                rec['provenance']['gpu4pyscf_package_root'] = (
                    str(self.base / 'official-1.8.1' / 'gpu4pyscf')
                    if role == 'upstream'
                    else str(self.base / 'snap' / 'gpu4pyscf'))
                rec['provenance']['source_commit'] = (
                    None if role == 'upstream' else self.COMMIT)
                rec['_path'] = str(self.base / f'{tag}_{role}_b{b}.json')
                samples.append(rec)
            schedule = {'batch_id': self.BATCH,
                        'settings': dict(us.SETTINGS),
                        'cells': [{'index': 0, 'tag': tag,
                                   'blocks': [['O', 'B', 'C', 'R']]}]}
            doc = uc.merge_role(samples, role, self.BATCH, schedule,
                                self.SCHED_SHA)
            doc['meta']['reference_manifest_sha256'] = self.cert_hash
            if role == 'upstream':
                doc['meta']['official_verification'] = {'x': 1}
            if mutate:
                mutate(role, doc)
            out = self.base / f'{tag}_{role}.json'
            out.write_text(json.dumps(doc))
            sources[role] = {'path': str(out), 'sha256': hashlib.sha256(
                out.read_bytes()).hexdigest()}
        return {'cell': {'tag': tag, 'xc': 'b3lyp', 'basis': 'def2-svp'},
                'sources': sources}

    def _index_file(self, entries):
        index = self.base / 'index.json'
        index.write_text(json.dumps({'batches': [
            {'id': self.BATCH, 'cells': entries}]}))
        return index

    def test_clean_cell_passes_with_snapshot_pin(self):
        entry = self._write_cell(
            'okcell', {'upstream': 12., 'optimized_fp64': 4.,
                       'optimized_auto': 2., 'strict_fp64': 6.})
        rec = audit.audit_cell(entry, self.BATCH, expect_commit=self.COMMIT)
        self.assertEqual(rec['status'], 'pass')
        self.assertEqual(rec['fail_reasons'], [])
        self.assertTrue(rec['gates']['B_vs_R']['passed'])
        self.assertTrue(rec['gates']['C_vs_R']['passed'])
        self.assertEqual(rec['gates']['B_vs_R']['n'], 4)
        self.assertEqual(rec['implementation_agreement_O_vs_R']['de'], 0.0)
        self.assertEqual(rec['source_commit']['strict_fp64'], self.COMMIT)
        self.assertIsNone(rec['source_commit']['upstream'])

    def test_honest_gate_fail_reported_not_dropped(self):
        entry = self._write_cell(
            'gatefail', {'upstream': 12., 'optimized_fp64': 4.,
                         'optimized_auto': 2., 'strict_fp64': 6.},
            energies={'optimized_fp64': -76.0 + 2e-6,
                      'optimized_auto': -76.0 + 2e-6})
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertEqual(rec['status'], 'fail')
        self.assertEqual(rec['category'], 'accuracy_gate')
        self.assertTrue(any('B_vs_R accuracy gate FAIL' in r
                            for r in rec['fail_reasons']))
        self.assertGreater(rec['gates']['B_vs_R']['de'], 1e-6)
        self.assertFalse(rec['gates']['B_vs_R']['passed'])

    def test_wrong_official_root_is_contract_fail(self):
        def mutate(role, doc):
            if role == 'upstream':
                doc['meta']['gpu4pyscf_package_root'] = '/elsewhere'
        entry = self._write_cell(
            'badroot', {'upstream': 12., 'optimized_fp64': 4.,
                        'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertEqual(rec['status'], 'fail')
        self.assertEqual(rec['category'], 'deployed_contract')
        self.assertIn('deployed contract', rec['fail_reasons'][0])

    def test_strict_fp32_entry_is_contract_fail(self):
        def mutate(role, doc):
            if role == 'strict_fp64':
                doc['rows'][0]['result']['fp32_entry_hits'] = {
                    'df_jk': 1, 'grad_ip1': 0}
        entry = self._write_cell(
            'strictfp32', {'upstream': 12., 'optimized_fp64': 4.,
                           'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertEqual(rec['category'], 'deployed_contract')

    def test_auto_without_fp32_evidence_is_contract_fail(self):
        def mutate(role, doc):
            if role == 'optimized_auto':
                doc['rows'][0]['result']['fp32_entry_hits'] = {
                    'df_jk': 0, 'grad_ip1': 0}
        entry = self._write_cell(
            'nofp32', {'upstream': 12., 'optimized_fp64': 4.,
                       'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertEqual(rec['category'], 'deployed_contract')

    def test_unconverged_repeat_is_honesty_fail(self):
        def mutate(role, doc):
            if role == 'optimized_fp64':
                doc['rows'][0]['result']['repeats'][2][
                    'converged'] = False
        entry = self._write_cell(
            'unconv', {'upstream': 12., 'optimized_fp64': 4.,
                       'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertEqual(rec['status'], 'fail')
        self.assertEqual(rec['category'], 'contract_or_honesty')
        self.assertTrue(any('did not converge' in r
                            for r in rec['fail_reasons']))

    def test_official_probe_present_is_honesty_fail(self):
        def mutate(role, doc):
            if role == 'upstream':
                doc['rows'][0]['result']['fp32_entry_hits'] = {
                    'df_jk': 2}
        entry = self._write_cell(
            'probe', {'upstream': 12., 'optimized_fp64': 4.,
                      'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec = audit.audit_cell(entry, self.BATCH)
        self.assertTrue(any('probe hits present' in r
                            for r in rec['fail_reasons']))

    def test_commit_pin_and_fabricated_wheel_commit(self):
        entry = self._write_cell(
            'commitpin', {'upstream': 12., 'optimized_fp64': 4.,
                          'optimized_auto': 2., 'strict_fp64': 6.})
        rec = audit.audit_cell(entry, self.BATCH, expect_commit='d' * 40)
        self.assertEqual(rec['status'], 'fail')
        self.assertTrue(any('source_commit' in r
                            for r in rec['fail_reasons']))

        def mutate(role, doc):
            if role == 'upstream':
                doc['meta']['source_commit'] = 'fake' * 10
        entry2 = self._write_cell(
            'commitpin2', {'upstream': 12., 'optimized_fp64': 4.,
                           'optimized_auto': 2., 'strict_fp64': 6.},
            mutate=mutate)
        rec2 = audit.audit_cell(entry2, self.BATCH,
                               expect_commit=self.COMMIT)
        self.assertTrue(any('fabricated source_commit' in r
                            for r in rec2['fail_reasons']))

    def test_planned_and_partial_kept_in_denominator(self):
        good = self._write_cell(
            'good', {'upstream': 12., 'optimized_fp64': 4.,
                     'optimized_auto': 2., 'strict_fp64': 6.})
        partial = self._write_cell(
            'part', {'upstream': 12., 'optimized_fp64': 4.},
            include=('upstream', 'optimized_fp64'))
        planned = {'cell': {'tag': 'later', 'xc': 'b3lyp',
                           'basis': 'def2-svp'}, 'sources': {}}
        index = self._index_file([good, partial, planned])
        records = audit.audit_batch(index, self.BATCH)
        by_tag = {r['tag']: r for r in records}
        self.assertEqual(by_tag['good']['status'], 'pass')
        self.assertEqual(by_tag['part']['status'], 'partial')
        self.assertEqual(by_tag['part']['roles_present'],
                         ['optimized_fp64', 'upstream'])
        self.assertEqual(by_tag['later']['status'], 'planned')
        summary = audit._summarize(records)
        self.assertEqual(summary['denominator'], 3)
        self.assertEqual(summary['audited'], 1)
        self.assertEqual(summary['partial'], 1)
        self.assertEqual(summary['planned'], 1)

    def test_main_writes_reports_and_exit_codes(self):
        good = self._write_cell(
            'good', {'upstream': 12., 'optimized_fp64': 4.,
                     'optimized_auto': 2., 'strict_fp64': 6.})
        bad = self._write_cell(
            'bad', {'upstream': 12., 'optimized_fp64': 4.,
                    'optimized_auto': 2., 'strict_fp64': 6.},
            energies={'optimized_fp64': -76.0 + 5e-6})
        index = self._index_file([good, bad])
        out = self.base / 'audit.json'
        md = self.base / 'audit.md'
        rc = audit.main(['--index', str(index), '--batch-id', self.BATCH,
                         '--expect-snapshot-commit', self.COMMIT,
                         '--out', str(out), '--md', str(md)])
        self.assertEqual(rc, 3)  # the honest gate FAIL is a finding
        payload = json.loads(out.read_text())
        self.assertEqual(payload['summary']['fail'], 1)
        self.assertEqual(payload['summary']['pass'], 1)
        self.assertEqual(payload['expect_snapshot_commit'], self.COMMIT)
        text = md.read_text()
        self.assertIn('bad', text)
        self.assertIn('FAIL', text)

        rc_clean = audit.main(['--index', str(index), '--batch-id',
                               self.BATCH, '--cells', 'good'])
        self.assertEqual(rc_clean, 0)

    def test_unknown_batch_fails_closed(self):
        index = self._index_file([])
        with self.assertRaises(SystemExit):
            audit.audit_batch(index, 'no-such-batch')

    def test_agreement_rejects_unmatched_settings(self):
        a = {'system': {'natm': 1}, 'method': {'xc': 'b3lyp'},
             'result': {'repeats': [{'e_tot': 1.0,
                                      'gradient': [0.0]}]}}
        b = {'system': {'natm': 2}, 'method': {'xc': 'b3lyp'},
             'result': {'repeats': [{'e_tot': 1.0,
                                      'gradient': [0.0]}]}}
        with self.assertRaises(ValueError):
            audit._agreement(a, b)


if __name__ == '__main__':
    unittest.main()

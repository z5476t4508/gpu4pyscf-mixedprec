#!/usr/bin/env python3
"""Tests for upstream_layered_analysis.py (size-band and basis-layered
tabulation on top of the adjudication facts json).

stdlib only; no GPU, no cupy import (the LC-205i operational rule:
nothing here may open a CUDA context while the batch driver is live).
"""

import importlib.util
import json
import pathlib
import sys
import tempfile
import unittest

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ul = _load('upstream_layered_analysis')


def _cell(tag, xc, basis, complete=True, ob=2.0, ob_scf=2.0,
          de=1e-12, dg=1e-8, passed=True, cr=0.7):
    return {
        'tag': tag, 'xc': xc, 'basis': basis, 'complete': complete,
        'status': 'complete' if complete else 'planned',
        'gates': {'optimized_fp64': {'de': de, 'dg': dg,
                                     'passed': passed}},
        'comparisons': {
            'O_over_B': {'speed': ob, 'scf_speed': ob_scf},
            'B_over_C': {'speed': 1.0},
            'O_over_C': {'speed': ob * 1.0},
        },
        'C_over_R': cr,
    }


def _write_lane_doc(path, nao):
    doc = {'schema_version': 'rtx4090-gradient-v1',
           'rows': [{'system': {'nao': nao}}]}
    path.write_text(json.dumps(doc), encoding='utf-8')
    return path


def _build(tmp, cells, nao_by_tag, batch_id='b1', interim=True,
           total=None, complete=None):
    total = total if total is not None else len(cells)
    complete_n = (complete if complete is not None
                  else sum(1 for c in cells if c['complete']))
    facts = {'batch_id': batch_id, 'interim': interim,
             'cells_complete': complete_n, 'cells_total': total,
             'generated_at_utc': '2026-10-19T16:00:00Z',
             'per_cell': cells}
    facts_path = tmp / 'facts.json'
    facts_path.write_text(json.dumps(facts), encoding='utf-8')

    index_cells = []
    for tag, nao in nao_by_tag.items():
        lane = _write_lane_doc(tmp / f'{tag}.json', nao)
        index_cells.append({'cell': {'tag': tag},
                            'sources': {'strict_fp64': {
                                'path': str(lane), 'sha256': '0' * 64}}})
    index_path = tmp / 'index.json'
    index_path.write_text(json.dumps(
        {'batches': [{'id': batch_id, 'cells': index_cells}]}),
        encoding='utf-8')
    return facts_path, index_path


class LayeredAnalysisTests(unittest.TestCase):
    def test_layer_medians_and_interim_carried(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [
                _cell('a_b3lyp_svp', 'b3lyp', 'def2-svp', ob=2.0),
                _cell('b_b3lyp_svp', 'b3lyp', 'def2-svp', ob=3.0),
                _cell('c_b3lyp_tzvp', 'b3lyp', 'def2-tzvp', ob=4.0),
            ]
            facts_path, index_path = _build(
                tmp, cells, {'a_b3lyp_svp': 50, 'b_b3lyp_svp': 150,
                             'c_b3lyp_tzvp': 120},
                interim=True, total=5, complete=3)
            layered = ul.layer_facts(
                json.loads(facts_path.read_text()),
                ul._nao_by_tag(str(index_path), 'b1'))
            self.assertTrue(layered['interim'])
            self.assertEqual(layered['cells_complete'], 3)
            self.assertEqual(layered['cells_total'], 5)
            m = layered['basis_method_medians']['def2-svp|b3lyp']
            self.assertEqual(m['n'], 2)
            self.assertEqual(m['OB'], 2.5)  # median(2.0, 3.0)
            self.assertEqual(m['BC'], 1.0)
            self.assertEqual(m['OC'], 2.5)
            # C/R lives top-level on the adjudicate per-cell record
            self.assertEqual(m['CR'], 0.7)
            self.assertEqual(
                layered['basis_method_medians']['def2-tzvp|b3lyp']['OB'],
                4.0)
            # size band layering: nao 50 -> nao<100, nao 150/120 ->
            # 100<=nao<250
            self.assertEqual(
                layered['method_band_medians']['b3lyp|nao<100']['n'], 1)
            self.assertEqual(
                layered['method_band_medians']['b3lyp|100<=nao<250']['n'],
                2)

    def test_band_boundaries(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [_cell(f'c{i}', 'b3lyp', 'def2-svp')
                     for i in range(4)]
            nao = {'c0': 99, 'c1': 100, 'c2': 249, 'c3': 400}
            facts_path, index_path = _build(tmp, cells, nao)
            layered = ul.layer_facts(
                json.loads(facts_path.read_text()),
                ul._nao_by_tag(str(index_path), 'b1'))
            bands = layered['method_band_medians']
            self.assertEqual(bands['b3lyp|nao<100']['n'], 1)
            self.assertEqual(bands['b3lyp|100<=nao<250']['n'], 2)
            self.assertEqual(bands['b3lyp|nao>=400']['n'], 1)
            self.assertNotIn('b3lyp|250<=nao<400', bands)

    def test_gate_fail_listed_and_kept_in_layer(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [
                _cell('ok_b3lyp_svp', 'b3lyp', 'def2-svp',
                      ob=2.0, de=1e-12, passed=True),
                _cell('bad_b3lyp_svp', 'b3lyp', 'def2-svp',
                      ob=2.0, de=2e-6, dg=5e-6, passed=False),
            ]
            facts_path, index_path = _build(
                tmp, cells, {'ok_b3lyp_svp': 100, 'bad_b3lyp_svp': 110})
            layered = ul.layer_facts(
                json.loads(facts_path.read_text()),
                ul._nao_by_tag(str(index_path), 'b1'))
            # the FAIL row stays in its layer denominator...
            m = layered['basis_method_medians']['def2-svp|b3lyp']
            self.assertEqual(m['n'], 2)
            # ...is listed by name in the accuracy table...
            acc = layered['accuracy_by_basis']['def2-svp']
            self.assertEqual(acc['n'], 2)
            self.assertEqual(acc['max_de'], 2e-6)
            self.assertEqual(acc['gate_fail_tags'], ['bad_b3lyp_svp'])
            # ...and in the global gate-fail list
            self.assertEqual(layered['gate_fail_tags'], ['bad_b3lyp_svp'])

    def test_wb97mv_scatter_sorted_and_filtered(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [
                _cell('big_wb97mv_svp', 'wb97mv', 'def2-svp',
                      ob=7.0, ob_scf=7.5),
                _cell('small_wb97mv_svp', 'wb97mv', 'def2-svp',
                      ob=2.0, ob_scf=2.1),
                _cell('hyb_b3lyp_svp', 'b3lyp', 'def2-svp', ob=1.7),
            ]
            nao = {'big_wb97mv_svp': 400, 'small_wb97mv_svp': 30,
                   'hyb_b3lyp_svp': 200}
            facts_path, index_path = _build(tmp, cells, nao)
            layered = ul.layer_facts(
                json.loads(facts_path.read_text()),
                ul._nao_by_tag(str(index_path), 'b1'))
            sc = layered['wb97mv_size_scatter']
            self.assertEqual(len(sc), 2)
            self.assertEqual([r['tag'] for r in sc],
                             ['small_wb97mv_svp', 'big_wb97mv_svp'])
            self.assertEqual(sc[0]['OB_scf'], 2.1)

    def test_missing_nao_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [_cell('x_b3lyp_svp', 'b3lyp', 'def2-svp')]
            # index without the cell's strict_fp64 source -> no nao
            facts_path, index_path = _build(tmp, cells, {})
            with self.assertRaises(SystemExit):
                ul.layer_facts(
                    json.loads(facts_path.read_text()),
                    ul._nao_by_tag(str(index_path), 'b1'))

    def test_unknown_batch_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [_cell('x_b3lyp_svp', 'b3lyp', 'def2-svp')]
            facts_path, index_path = _build(
                tmp, cells, {'x_b3lyp_svp': 100}, batch_id='b1')
            with self.assertRaises(SystemExit):
                ul._nao_by_tag(str(index_path), 'no-such-batch')

    def test_main_writes_pair_and_renders_md(self):
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            cells = [
                _cell('ok_b3lyp_svp', 'b3lyp', 'def2-svp', ob=2.0),
                _cell('bad_b3lyp_svp', 'b3lyp', 'def2-svp',
                      ob=2.0, de=2e-6, passed=False),
                _cell('planned_m062x_svp', 'm062x', 'def2-svp',
                      complete=False),
            ]
            facts_path, index_path = _build(
                tmp, cells, {'ok_b3lyp_svp': 100, 'bad_b3lyp_svp': 110,
                             'planned_m062x_svp': 120},
                interim=True, total=3, complete=2)
            out = tmp / 'layered.json'
            md = tmp / 'layered.md'
            rc = ul.main(['--facts', str(facts_path),
                          '--index', str(index_path),
                          '--batch-id', 'b1',
                          '--out', str(out), '--md', str(md)])
            self.assertEqual(rc, 0)
            payload = json.loads(out.read_text())
            self.assertTrue(payload['interim'])
            self.assertEqual(payload['gate_fail_tags'],
                             ['bad_b3lyp_svp'])
            text = md.read_text()
            self.assertIn('INTERIM', text)
            self.assertIn('bad_b3lyp_svp', text)
            self.assertIn('| def2-svp | b3lyp | 2 |', text)
            self.assertIn('planned not yet run: 1/3', text)


if __name__ == '__main__':
    unittest.main()

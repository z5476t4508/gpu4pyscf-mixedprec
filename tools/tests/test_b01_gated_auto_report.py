'''B01 (coordination board 2026-10-27): the SP-32 report discrepancy.

11/11 SP-32 cells passed their scientific gates, but 7 legitimate SP-14
size-gated auto cells (below _AUTO_HEAD_MIN_AO the auto lane resolves fp64
from cycle 0, so fp32_entry_hits is all-zero BY CONSTRUCTION) are rendered
unknown by the deployed reporter rule three_way_report.load_cell "no actual
FP32 entry evidence for auto" -- verified live at 74974d0 against the real
sp32-fournrole-landedsource-20261027-a index (7/11 REJECTED at load).

The repo merge layer (upstream_comparison.validate_comparison_doc, SP-18)
already proves the gated lane from the document itself (measured nao below
the gate threshold + unswitched policy + fp64 gradient).  This regression
pins the SAME proof on the reporter deployment copy
(tools/reporting/three_way_report.py, the proposed deploy candidate):

  * a zero-hit auto doc WITH the full gated proof loads;
  * zero-hit auto docs WITHOUT the full proof stay rejected (above-gate
    nao, absent nao, unresolvable threshold, fired switch, fp32-declaring
    gradient -- each individually);
  * render classifies gated cells as an explicit size-gated fallback
    category (NOT unknown), keeps every displayed number on the
    FP32-executed subset unchanged (no widening), and keeps unknown for
    genuinely unsupported cells;
  * the real SP-32 index renders 7 gated + 0 unknown with the medians
    byte-identical to the pinned 74974d0 report values.

Red at 74974d0: the gated-load and render-classification tests fail (the
reporter copy has no gated-lane allowance).  Green after the repair.'''

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
_REPORTING = REPO / 'tools' / 'reporting'
for _p in (str(_DIR), str(_REPORTING), str(pathlib.Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import test_upstream_comparison as tuc  # noqa: E402 (sibling fixtures)
import upstream_comparison as uc        # noqa: E402


def _load_reporting(name):
    spec = importlib.util.spec_from_file_location(
        f'b01_{name}', _REPORTING / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL_INDEX = pathlib.Path(
    '/data/work/gpu4pyscf-agent/state/long-chain/upstream-effects-index.json')


class B01GatedLaneTests(unittest.TestCase):
    '''load_cell accepts exactly the proven gated lane, nothing wider.'''

    BATCH = 'b01-batch'
    SCHED_SHA = 'd' * 64

    def _docs(self, tmp, tag='gated_cell', auto_mutate=None,
              auto_hits=None):
        '''Four-role merged docs via the merge contract; auto_mutate(row)
        may rewrite the auto row before the doc is written.'''
        base = pathlib.Path(tmp)
        cert = base / 'reference.json'
        cert.write_text(json.dumps({
            'package_root': str(base / 'official-1.8.1'),
            'version': '1.8.1'}))
        cert_hash = hashlib.sha256(cert.read_bytes()).hexdigest()
        sources = {}
        for role in uc.ROLES:
            samples = []
            for b in range(4):
                hits = auto_hits if (role == 'optimized_auto'
                                     and auto_hits is not None) else None
                rec = tuc._synthetic_record(
                    role, b, self.BATCH, self.SCHED_SHA,
                    lane=('auto' if role == 'optimized_auto' else 'fp64'),
                    fp32_hits=hits)
                for r in rec['row']['result']['repeats']:
                    r.update(sample_s={'upstream': 12., 'optimized_fp64': 4.,
                                       'optimized_auto': 2.,
                                       'strict_fp64': 6.}[role],
                             t_scf_s=1.0, t_grad_s=0.5)
                if role == 'optimized_auto' and auto_hits is None:
                    # the fp32-evidence fixture of the compat tests
                    rec['row']['result']['repeats'][0][
                        'gradient_mode'] = 'fp32'
                    for r in rec['row']['result']['repeats']:
                        r['grad_ip1_hits'] = 3
                rec['provenance']['gpu4pyscf_package_root'] = (
                    str(base / 'official-1.8.1' / 'gpu4pyscf')
                    if role == 'upstream' else str(base / 'snap'
                                                   / 'gpu4pyscf'))
                rec['_path'] = str(base / f'{tag}_{role}_{b}.json')
                samples.append(rec)
            schedule = self._schedule(tag)
            doc = uc.merge_role(samples, role, self.BATCH, schedule,
                                self.SCHED_SHA)
            doc['meta']['reference_manifest_sha256'] = cert_hash
            if role == 'upstream':
                doc['meta']['official_verification'] = {'x': 1}
            if role == 'optimized_auto' and auto_mutate is not None:
                auto_mutate(doc['rows'][0])
            out = base / f'{tag}_{role}.json'
            out.write_text(json.dumps(doc))
            sources[role] = {'path': str(out),
                             'sha256': hashlib.sha256(
                                 out.read_bytes()).hexdigest()}
        return cert, sources

    def _schedule(self, tag):
        return {'batch_id': self.BATCH, 'settings': dict(tuc.us.SETTINGS),
                'cells': [{'index': 0, 'tag': tag, 'basis': 'def2-svp',
                           'xc': 'b3lyp', 'charge': 0, 'spin': 0,
                           'blocks': [['O', 'B', 'C', 'R']] * 4}]}

    def _load_cell(self, tmp, **doc_kw):
        effects = _load_reporting('effects_report')
        report = _load_reporting('three_way_report')
        report.effects = effects
        cert, sources = self._docs(tmp, **doc_kw)
        return report, effects, sources, cert

    def test_gated_lane_loads_with_documented_proof(self):
        # nao=157 < 250, unswitched, fp64 gradient, all-zero hits: the
        # SP-14 gated lane proven from the document itself.
        with tempfile.TemporaryDirectory() as tmp:
            report, effects, sources, cert = self._load_cell(
                tmp, auto_hits={'df_jk': 0, 'grad_ip1': 0})
            with mock.patch.object(report, 'REFERENCE', cert), \
                    mock.patch.object(effects, 'TASKS',
                                      pathlib.Path(tmp)), \
                    mock.patch.object(report, '_auto_head_min_ao',
                                     return_value=250):
                rows = report.load_cell({'sources': sources}, self.BATCH)
            self.assertEqual(set(rows), set(uc.ROLES))
            self.assertEqual(
                rows['optimized_auto']['system']['nao'], 157)

    def test_zero_hit_auto_without_full_proof_still_rejected(self):
        # each variant drops exactly one leg of the gated-lane proof
        def set_nao(value):
            return lambda row: row['system'].update(nao=value)

        variants = {
            'above-gate nao': set_nao(300),
            'absent nao': lambda row: row['system'].pop('nao'),
            'fired switch': lambda row: row['result']['precision_switch'
                                                     ].update(
                switched=True, cycle=3, trigger='dE'),
            'fp32-declaring gradient': lambda row: row['result'].update(
                gradient_mode='fp32'),
        }
        for label, mutate in variants.items():
            with self.subTest(variant=label):
                with tempfile.TemporaryDirectory() as tmp:
                    report, effects, sources, cert = self._load_cell(
                        tmp, auto_mutate=mutate,
                        auto_hits={'df_jk': 0, 'grad_ip1': 0})
                    with mock.patch.object(report, 'REFERENCE', cert), \
                            mock.patch.object(effects, 'TASKS',
                                              pathlib.Path(tmp)), \
                            mock.patch.object(
                                report, '_auto_head_min_ao',
                                return_value=250):
                        with self.assertRaises(ValueError) as ctx:
                            report.load_cell({'sources': sources},
                                             self.BATCH)
                    self.assertIn('no actual FP32 entry evidence',
                                  str(ctx.exception))

    def test_unresolvable_threshold_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report, effects, sources, cert = self._load_cell(
                tmp, auto_hits={'df_jk': 0, 'grad_ip1': 0})
            with mock.patch.object(report, 'REFERENCE', cert), \
                    mock.patch.object(effects, 'TASKS',
                                      pathlib.Path(tmp)), \
                    mock.patch.object(report, '_auto_head_min_ao',
                                     return_value=None):
                with self.assertRaises(ValueError) as ctx:
                    report.load_cell({'sources': sources}, self.BATCH)
            self.assertIn('no actual FP32 entry evidence',
                          str(ctx.exception))


class B01RenderClassificationTests(unittest.TestCase):
    '''render: gated cells are an explicit category, not unknown; every
    displayed number stays on the FP32-executed subset (no widening).'''

    BATCH = 'b01-render'
    SCHED_SHA = 'e' * 64

    def _index(self, tmp, cells):
        '''cells: list of dicts(tag=..., gated=False, unsupported=False).
        Builds merged docs + an index whose LAST batch holds them all.'''
        base = pathlib.Path(tmp)
        cert = base / 'reference.json'
        cert.write_text(json.dumps({
            'package_root': str(base / 'official-1.8.1'),
            'version': '1.8.1'}))
        cert_hash = hashlib.sha256(cert.read_bytes()).hexdigest()
        index_cells = []
        schedule = {'batch_id': self.BATCH,
                    'settings': dict(tuc.us.SETTINGS), 'cells': []}
        for i, spec in enumerate(cells):
            tag = spec['tag']
            schedule['cells'].append(
                {'index': i, 'tag': tag, 'basis': 'def2-svp', 'xc': 'b3lyp',
                 'charge': 0, 'spin': 0, 'blocks': [['O', 'B', 'C', 'R']] * 4})
            sources = {}
            for role in uc.ROLES:
                samples = []
                zero = spec.get('gated') or spec.get('unsupported')
                for b in range(4):
                    rec = tuc._synthetic_record(
                        role, b, self.BATCH, self.SCHED_SHA,
                        lane=('auto' if role == 'optimized_auto'
                              else 'fp64'),
                        fp32_hits=({'df_jk': 0, 'grad_ip1': 0}
                                   if (role == 'optimized_auto' and zero)
                                   else None))
                    seconds = {'upstream': 12., 'optimized_fp64': 4.,
                               'optimized_auto': 2.,
                               'strict_fp64': 6.}[role]
                    for r in rec['row']['result']['repeats']:
                        r.update(sample_s=seconds, t_scf_s=seconds / 2,
                                 t_grad_s=seconds / 4)
                    if role == 'optimized_auto':
                        if zero:
                            if spec.get('unsupported'):
                                rec['row']['system'].update(nao=300)
                        else:
                            rec['row']['result']['repeats'][0][
                                'gradient_mode'] = 'fp32'
                            for r in rec['row']['result']['repeats']:
                                r['grad_ip1_hits'] = 3
                    rec['provenance']['gpu4pyscf_package_root'] = (
                        str(base / 'official-1.8.1' / 'gpu4pyscf')
                        if role == 'upstream'
                        else str(base / 'snap' / 'gpu4pyscf'))
                    rec['_path'] = str(base / f'{tag}_{role}_{b}.json')
                    samples.append(rec)
                doc = uc.merge_role(samples, role, self.BATCH, schedule,
                                    self.SCHED_SHA)
                doc['meta']['reference_manifest_sha256'] = cert_hash
                if role == 'upstream':
                    doc['meta']['official_verification'] = {'x': 1}
                out = base / f'{tag}_{role}.json'
                out.write_text(json.dumps(doc))
                sources[role] = {
                    'path': str(out),
                    'sha256': hashlib.sha256(
                        out.read_bytes()).hexdigest()}
            index_cells.append(
                {'cell': {'index': i, 'tag': tag, 'xc': 'b3lyp',
                          'basis': 'def2-svp', 'charge': 0, 'spin': 0},
                 'sources': sources})
        index = base / 'index.json'
        index.write_text(json.dumps(
            {'batches': [{'id': self.BATCH, 'cells': index_cells}]}))
        return str(index), cert

    def _render(self, tmp, cells):
        effects = _load_reporting('effects_report')
        report = _load_reporting('three_way_report')
        report.effects = effects
        index, cert = self._index(tmp, cells)
        with mock.patch.object(report, 'REFERENCE', cert), \
                mock.patch.object(effects, 'TASKS', pathlib.Path(tmp)), \
                mock.patch.object(report, '_auto_head_min_ao',
                                  return_value=250):
            return '\n'.join(report.render(index))

    def test_gated_separate_from_unknown_and_numbers_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = self._render(tmp, [
                {'tag': 'a_fp32'},
                {'tag': 'b_gated', 'gated': True},
                {'tag': 'c_unsupported', 'unsupported': True}])
        # the unsupported cell stays unknown; the gated one is NOT unknown
        self.assertIn('缺失/不可比单元：1/3', text)
        self.assertNotIn('缺失/不可比单元：2/3', text)
        # gated cells get their own explicit category line
        gated_lines = [ln for ln in text.splitlines()
                       if ln.startswith('- 尺寸门控回退：')]
        self.assertEqual(len(gated_lines), 1)
        self.assertIn('1/3', gated_lines[0])
        self.assertIn('R 精度门通过 1/1', gated_lines[0])
        # the footer names the gated category without counting it missing
        self.assertIn('另有 1 个尺寸门控回退单元', text)
        # every comparison count/median stays on the FP32-executed subset
        for label in ('原版 → 优化基准', '优化基准 → auto', '原版 → auto'):
            row = [ln for ln in text.splitlines()
                   if ln.startswith(f'| {label} |')]
            self.assertEqual(len(row), 1)
            self.assertIn('| 1/3 |', row[0])
        # the fp32 cell's ratio (O/C = 12/2 = 6.0) is unchanged by the
        # gated cell's presence (no widening, no mixing)
        self.assertIn('| 原版 → auto | 1/3 | 6.000×', text)
        # accuracy aggregates keep the pre-repair denominators (fp32 only)
        self.assertIn('auto/R 精度：通过 1/3', text)

    def test_no_gated_line_without_gated_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            text = self._render(tmp, [{'tag': 'only_fp32'}])
        self.assertNotIn('尺寸门控回退', text)
        self.assertIn('缺失/不可比单元：0/1', text)
        self.assertIn('| 原版 → auto | 1/1 | 6.000×', text)


@unittest.skipIf(not REAL_INDEX.is_file(), 'real SP-32 index absent')
class B01RealIndexTests(unittest.TestCase):
    '''The live sp32 index: 7 gated + 0 unknown, medians byte-identical.'''

    def test_real_sp32_renders_gated_not_unknown(self):
        if str(REPO) not in sys.path:
            sys.path.insert(0, str(REPO))  # resolver: repo gpu4pyscf tree
        effects = _load_reporting('effects_report')
        report = _load_reporting('three_way_report')
        report.effects = effects
        text = '\n'.join(report.render(str(REAL_INDEX)))
        self.assertIn('缺失/不可比单元：0/11', text)
        gated = [ln for ln in text.splitlines()
                 if ln.startswith('- 尺寸门控回退：')]
        self.assertEqual(len(gated), 1)
        self.assertIn('7/11', gated[0])
        self.assertIn('R 精度门通过 7/7', gated[0])
        # the pinned 74974d0 report values, unchanged by the repair
        self.assertIn('| 原版 → 优化基准 | 4/11 | 1.712× | 1.346×', text)
        self.assertIn('| 优化基准 → auto | 4/11 | 1.067× | 1.096×', text)
        self.assertIn('| 原版 → auto | 4/11 | 1.860× | 1.493×', text)


if __name__ == '__main__':
    unittest.main()

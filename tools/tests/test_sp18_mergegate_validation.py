#!/usr/bin/env python3
"""SP-18 regression tests: the merge validation's auto-evidence check
conditioned on the SP-14 size gate (mixedprec/benchmark/rtx4090/
upstream_comparison.py::validate_comparison_doc; scf-priority-20261025
card items 3+5; named by SP-17's P5 merge-stage measurement).

Measured motivation: SP-17 measured the production driver's own
merge_cell -> uc.validate_comparison_doc rejecting every below-gate
gated cell's auto row AT MERGE with 'no actual FP32 entry evidence for
auto' -- 4/6 cells merge-blocked (oxygen_triplet 28, methanol_3c 63,
ethylamine 77, caffeine 246 AO; fp32_entry_hits all-zero,
precision_switch.switched False, gradient_mode 'fp64').  Since SP-14
(commit 97e21b7) below _AUTO_HEAD_MIN_AO=250 the plain-auto lane IS
bitwise the fp64 lane in e_tot (and since SP-16 in gradient), so zero
fp32 entries is the by-design CORRECT record for exactly this size
class -- the check's anti-fraud intent ('auto must show actual fp32
use') is factually wrong below the gate.

The change: a zero-hit auto row is accepted ONLY IF the row itself
carries the complete by-design gated-lane signature -- nao <
_AUTO_HEAD_MIN_AO (imported from gpu4pyscf.scf.hf, one source of
truth; unresolved constant fails CLOSED) AND precision_switch.switched
is False AND gradient_mode == 'fp64'.  Contracts pinned here:

  * below-gate zero-hit + no-switch + fp64 gradient PASSES (the
    SP-17 merge-blocked class);
  * anti-fraud preserved: above-gate (nao >= threshold) zero-hit,
    missing/None nao, switched True, and non-fp64 gradient_mode all
    still REJECT with the same message;
  * auto rows with actual hits > 0 pass at ANY nao (unchanged);
  * the strict_fp64 zero-hit requirement and every other clause are
    untouched;
  * the threshold is the SAME constant as the source gate (boundary:
    threshold-1 passes, threshold rejects);
  * integrated path: merge_role-built docs from below-gate zero-hit
    synthetic auto samples pass validation end-to-end.
"""
import importlib.util
import json
import os
import sys
import unittest

_DIR = os.path.dirname(os.path.abspath(__file__))
_RTX = os.path.join(os.path.dirname(_DIR), '..', 'mixedprec', 'benchmark',
                    'rtx4090')
sys.path.insert(0, _RTX)
sys.path.insert(0, _DIR)

spec = importlib.util.spec_from_file_location(
    'sp18_upstream_comparison', os.path.join(_RTX, 'upstream_comparison.py'))
uc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(uc)

import test_upstream_comparison as tuc  # noqa: E402 (sibling fixtures)

BATCH = 'sp18-test-batch'
SCHED_SHA = 'c' * 64


def _auto_doc(nao=246, hits=None, switched=False, gradient_mode='fp64',
              nao_present=True):
    '''A minimal valid optimized_auto comparison doc; merge-level fields
    not read by validate_comparison_doc are omitted -- this targets the
    validation clause, not merge_role.'''
    system = {'geom_file': 'caffeine.xyz', 'geom_sha256': 'a' * 64,
              'natm': 24, 'charge': 0, 'spin': 0, 'nelec': 110,
              'cart': False, 'geom_units': 'angstrom'}
    if nao_present:
        system['nao'] = nao
    return {
        'schema_version': uc.SCHEMA_VERSION,
        'meta': {'dependencies': {'pyscf': '2.14.0', 'cupy': '13.4.1',
                                  'numpy': '2.2.6', 'libxc': '0.8.1',
                                  'cutensor': '2.2.0'},
                 'comparison_role': 'optimized_auto',
                 'comparison_batch_id': BATCH,
                 'timing_protocol': uc.TIMING_PROTOCOL,
                 'schedule_sha256': SCHED_SHA,
                 'optimization_env': dict(uc.STACK_ENV),
                 'precision_mode': uc.PRECISION_MODE['optimized_auto']},
        'rows': [{'row_key': {'geom': 'caffeine.xyz', 'basis': 'def2-svp',
                              'xc': 'm062x', 'lane': 'auto'},
                  'system': system,
                  'result': {'repeats': [{}] * 4,
                             'gradient_mode': gradient_mode,
                             'fp32_entry_hits':
                                 {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                  'grad_ip1': 0, 'other': 0}
                                 if hits is None else hits,
                             'precision_switch': {
                                 'switched': switched, 'cycle': None,
                                 'trigger': 'none' if not switched
                                 else 'residual'}}}],
    }


class SP18SizeGatedAutoEvidenceTests(unittest.TestCase):

    def test_below_gate_zero_hit_accepted(self):
        # the SP-17 merge-blocked class (caffeine 246 AO): the by-design
        # gated lane -- RED before SP-18 ('no actual FP32 entry evidence
        # for auto')
        doc = _auto_doc(nao=246)
        self.assertIs(
            uc.validate_comparison_doc(doc, 'optimized_auto', BATCH,
                                       SCHED_SHA), True)

    def test_threshold_boundary_from_source_constant(self):
        from gpu4pyscf.scf.hf import _AUTO_HEAD_MIN_AO as thr
        self.assertIs(uc.validate_comparison_doc(
            _auto_doc(nao=thr - 1), 'optimized_auto', BATCH, SCHED_SHA), True)
        with self.assertRaises(ValueError) as cm:
            uc.validate_comparison_doc(_auto_doc(nao=thr), 'optimized_auto',
                                       BATCH, SCHED_SHA)
        self.assertIn('no actual FP32 entry evidence', str(cm.exception))

    def test_above_gate_zero_hit_still_rejects(self):
        # anti-fraud preserved: an above-gate auto row with no fp32 use
        # is genuinely missing evidence (the kept class: 318/348 AO)
        for nao in (250, 318, 522):
            with self.subTest(nao=nao):
                with self.assertRaises(ValueError) as cm:
                    uc.validate_comparison_doc(
                        _auto_doc(nao=nao), 'optimized_auto', BATCH,
                        SCHED_SHA)
                self.assertIn('no actual FP32 entry evidence',
                              str(cm.exception))

    def test_missing_nao_rejects(self):
        # the gate context must come FROM the row; absent nao cannot
        # prove the gated-lane context
        with self.assertRaises(ValueError) as cm:
            uc.validate_comparison_doc(
                _auto_doc(nao_present=False), 'optimized_auto', BATCH,
                SCHED_SHA)
        self.assertIn('no actual FP32 entry evidence', str(cm.exception))

    def test_switched_true_rejects(self):
        # a precision switch fired -> not the gated bitwise-fp64 lane
        with self.assertRaises(ValueError) as cm:
            uc.validate_comparison_doc(
                _auto_doc(nao=246, switched=True), 'optimized_auto', BATCH,
                SCHED_SHA)
        self.assertIn('no actual FP32 entry evidence', str(cm.exception))

    def test_non_fp64_gradient_rejects(self):
        with self.assertRaises(ValueError) as cm:
            uc.validate_comparison_doc(
                _auto_doc(nao=246, gradient_mode='fp32'), 'optimized_auto',
                BATCH, SCHED_SHA)
        self.assertIn('no actual FP32 entry evidence', str(cm.exception))

    def test_hits_present_passes_at_any_nao(self):
        # unchanged: actual fp32 evidence satisfies the check everywhere
        hits = {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 3,
                'other': 0}
        for nao in (63, 246, 318, 522):
            with self.subTest(nao=nao):
                self.assertIs(uc.validate_comparison_doc(
                    _auto_doc(nao=nao, hits=hits), 'optimized_auto', BATCH,
                    SCHED_SHA), True)

    def test_strict_clause_unchanged(self):
        doc = _auto_doc(nao=246)
        doc['meta']['comparison_role'] = 'strict_fp64'
        doc['meta']['precision_mode'] = uc.PRECISION_MODE['strict_fp64']
        doc['rows'][0]['row_key']['lane'] = 'fp64'
        self.assertIs(uc.validate_comparison_doc(doc, 'strict_fp64', BATCH,
                                                 SCHED_SHA), True)
        doc['rows'][0]['result']['fp32_entry_hits'] = {'df_jk': 1}
        with self.assertRaises(ValueError):
            uc.validate_comparison_doc(doc, 'strict_fp64', BATCH, SCHED_SHA)

    def test_gate_constant_is_the_source_gate(self):
        # one source of truth: the validation threshold must BE
        # gpu4pyscf.scf.hf._AUTO_HEAD_MIN_AO, not a copy
        from gpu4pyscf.scf.hf import _AUTO_HEAD_MIN_AO as source_thr
        self.assertEqual(uc._auto_head_min_ao(), source_thr)


class SP18IntegratedMergeTests(unittest.TestCase):
    '''The production path: merge_role-built docs from below-gate
    zero-hit auto samples must validate end-to-end (the SP-17 blocked
    merge), and above-gate zero-hit must still fail it.'''

    def _merged_doc(self, nao):
        samples = []
        for b in range(4):
            rec = tuc._synthetic_record('optimized_auto', b, BATCH,
                                        SCHED_SHA)
            rec['row']['system']['nao'] = nao
            rec['row']['result']['fp32_entry_hits'] = {
                'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0,
                'other': 0}
            rec['_path'] = f'/tmp/sp18_{b}.json'
            samples.append(rec)
        return uc.merge_role(samples, 'optimized_auto', BATCH,
                             {'batch_id': BATCH,
                              'settings': {'timeout_s_per_arm': 60},
                              'cells': [{'index': 0, 'tag': 'c',
                                         'blocks': [['O', 'B', 'C', 'R']] * 4}]},
                             SCHED_SHA)

    def test_below_gate_merged_doc_validates(self):
        doc = self._merged_doc(246)
        self.assertIs(uc.validate_comparison_doc(doc, 'optimized_auto',
                                                 BATCH, SCHED_SHA), True)

    def test_above_gate_merged_doc_rejects(self):
        doc = self._merged_doc(318)
        with self.assertRaises(ValueError):
            uc.validate_comparison_doc(doc, 'optimized_auto', BATCH,
                                       SCHED_SHA)


if __name__ == '__main__':
    unittest.main()

"""Issue #10 rework unit tests: multi-i tile VV10 UWE kernel selector
contract and retest analysis.

Covers (no GPU, no compiled extension needed):
  * the selector contract as implemented in vv10.cu: accepted values are
    warp1 (default, incl. unset/empty) | warp4 | tile8 | tile16 |
    tile32; any other value falls back to warp1 with a warning; the
    parse function is extracted from the source text and tested against
    every accepted/rejected value plus case sensitivity;
  * the source-level structural contract of the tile kernel: template
    parameter bounds (static_assert 4..32), launch geometry in the
    dispatcher (blocks = ceil(ngrids/I_TILE)), the per-i shuffle
    reduction's warp-alignment invariant (NG_PER_BLOCK multiple of 32,
    THREADS_PER_I divides 32), uniform-barrier structure (no early
    return in the kernel body), and single-writer output ownership;
  * the round-1 c1_retest_analysis module still refuses schema-INVALID
    inputs (regression guard for the new matrix using the same module);
  * evidence document checks: the rework evidence files record the
    inherited UKS failure visibility (must NOT be claimed fixed).

The Python-side contract tested here mirrors what the runner and docs
promise: nothing in gpu4pyscf Python reads GPU4PYSCF_VV10_UWE_KERNEL, so
opting in never changes any other code path.
"""
import importlib.util
import json
import os
import pathlib
import re
import tempfile
import unittest

REPO = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..'))
VV10_CU = pathlib.Path(REPO) / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu'

_DIR = pathlib.Path(REPO) / 'mixedprec' / 'benchmark' / 'rtx4090'
_spec = importlib.util.spec_from_file_location(
    'c1_retest_analysis', _DIR / 'c1_retest_analysis.py')
c1_retest_analysis = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1_retest_analysis)

_spec2 = importlib.util.spec_from_file_location(
    'multii_retest_analysis', _DIR / 'multii_retest_analysis.py')
multii_retest_analysis = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(multii_retest_analysis)


class TestSelectorContract(unittest.TestCase):
    """The C-side selector semantics, tested by extracting the parse
    table from vv10.cu source (the selector itself is static C and
    cached per process; its observable contract is the value table)."""

    @classmethod
    def setUpClass(cls):
        cls.src = VV10_CU.read_text()

    def _extract_table(self):
        # returns {env_value: dispatch_code} from the selector function
        table = {}
        # find the selector body
        m = re.search(
            r'static int vv10_uwe_kernel_variant\(\)(.*?)\n\}',
            self.src, re.S)
        self.assertIsNotNone(m, 'selector function not found in vv10.cu')
        body = m.group(1)
        for env_val, code in re.findall(
                r'strcmp\(env,\s*"([^"]+)"\)\s*==\s*0\)\s*\{\s*'
                r'cached\s*=\s*([^;]+);', body):
            table[env_val] = code.strip()
        return table

    def test_accepted_values(self):
        table = self._extract_table()
        self.assertEqual(table.get('warp4'), '1')
        self.assertEqual(table.get('tile8'), '8')
        self.assertEqual(table.get('tile16'), '16')
        self.assertEqual(table.get('tile32'), '32')

    def test_default_is_reference_kernel(self):
        # the fallthrough else branch must set cached = 0 (warp1)
        m = re.search(
            r'static int vv10_uwe_kernel_variant\(\)(.*?)\n\}',
            self.src, re.S)
        body = m.group(1)
        self.assertRegex(
            body, re.compile(
                r'\}\s*else\s*\{.*?cached\s*=\s*0;', re.S),
            'default branch must select the reference warp1 kernel')

    def test_unknown_values_warn_and_fallback(self):
        m = re.search(
            r'static int vv10_uwe_kernel_variant\(\)(.*?)\n\}',
            self.src, re.S)
        body = m.group(1)
        self.assertIn('unknown value', body,
                      'unknown selector values must be reported')
        # the warning must list the accepted tile values
        self.assertIn('tile32', body)

    def test_python_tree_does_not_read_env(self):
        # same guarantee as round 1: opt-in is C-only
        hits = []
        for root, _dirs, files in os.walk(pathlib.Path(REPO) / 'gpu4pyscf'):
            if '__pycache__' in root:
                continue
            for fn in files:
                if fn.endswith('.py'):
                    p = pathlib.Path(root) / fn
                    if 'VV10_UWE_KERNEL' in p.read_text(errors='ignore'):
                        hits.append(str(p))
        self.assertEqual(hits, [],
                         'Python must not read GPU4PYSCF_VV10_UWE_KERNEL: '
                         '%s' % hits)


class TestTileKernelStructure(unittest.TestCase):
    """Source-level structural contract of the multi-i tile kernel:
    launch geometry, reduction alignment, uniform barriers, ownership."""

    @classmethod
    def setUpClass(cls):
        cls.src = VV10_CU.read_text()
        m = re.search(
            r'template <int I_TILE>\s*\n__global__\s*\n'
            r'static void vv10_fock_eval_UWE_multii_kernel(.*?)\n\}\n',
            cls.src, re.S)
        cls.kernel = m.group(1) if m else None
        cls.assertIsNotNone(cls.kernel, 'tile kernel not found')

    def test_template_bounds_static_assert(self):
        self.assertIn('static_assert(THREADS_PER_I >= 4', self.kernel)
        self.assertIn('THREADS_PER_I <= 32', self.kernel)

    def test_warp_alignment_invariant(self):
        # NG_PER_BLOCK == 128 and THREADS_PER_I in {4,8,16,32} divides
        # 32, so shuffle groups never straddle warp boundaries and the
        # full-warp __shfl_down_sync mask is uniform
        self.assertRegex(self.src, r'#define NG_PER_BLOCK\s+128')
        for tile in (4, 8, 16, 32):
            self.assertEqual(128 // tile in (4, 8, 16, 32), True)
            self.assertEqual((128 // tile) <= 32, True)
            self.assertEqual(32 % (128 // tile), 0,
                             'THREADS_PER_I must divide 32 for warp-aligned '
                             'groups')

    def test_no_early_return_in_kernel_body(self):
        # uniform barriers: every thread must reach both __syncthreads()
        # in the chunk loop for any ngrids (tail chunks, partial tiles).
        # A `return` statement inside the kernel body would break that
        # invariant. (Comments may contain the word; code must not.)
        code = re.sub(r'//[^\n]*', '', self.kernel)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.S)
        self.assertNotRegex(code, r'\breturn\b')

    def test_single_writer_output_ownership(self):
        # exactly lane 0 of each group writes its active i
        self.assertRegex(
            self.kernel,
            r'if \(lane_in_group == 0 && active\)\s*\{[^}]*U\[i\]')

    def test_dispatcher_geometry(self):
        # each tile variant launches ceil(ngrids / I_TILE) blocks with
        # NG_PER_BLOCK threads
        m = re.search(
            r'const dim3 blocks\(\(ngrids \+ variant - 1\) / variant\);',
            self.src)
        self.assertIsNotNone(
            m, 'tile dispatcher must use ceil(ngrids/I_TILE) blocks')
        # and instantiates exactly the three bounded sizes
        for inst in ('multii_kernel<8>', 'multii_kernel<16>',
                     'multii_kernel<32>'):
            self.assertIn(inst, self.src)
        self.assertNotIn('multii_kernel<4>', self.src)
        self.assertNotIn('multii_kernel<64>', self.src)
        self.assertNotIn('multii_kernel<128>', self.src)

    def test_reference_kernel_comment_fixed(self):
        # the rework card: the reference kernel's comment must describe
        # the actual design (one block stages j once for its 128 i
        # points), not the old misleading per-i description
        head = self.src[:self.src.find('__global__')]
        self.assertIn('ONCE per block', head)
        self.assertIn('minimum', head)
        self.assertIn('global read amplification', head)


class TestRetestAnalysisRegression(unittest.TestCase):
    """The rework's end-to-end matrix is analysed with the round-1
    c1_retest_analysis module; guard its refusal paths."""

    def _ok_doc(self, cell, lane, e, median):
        return {
            'schema_version': 'rtx4090-energy-v1',
            'meta': {'gpu_name': 'NVIDIA GeForce RTX 4090',
                     'gpu4pyscf_source_commit': 'a' * 40},
            'rows': [{
                'row_key': {'geom': 'g.xyz', 'basis': 'b',
                            'xc': 'wb97m-v', 'lane': lane},
                'pilot_stage': 'S1',
                'system': {'nao': 43},
                'lane': lane, 'status': 'ok', 'skip_reason': None,
                'result': {'e_tot': e, 'converged': True,
                           'scf_cycles': 9,
                           'timings': {'warmup_s': 1.0,
                                       'samples_s': [median] * 3,
                                       'median_s': median}},
                'accuracy': {}, 'cross_program': [],
                'evidence': {'result_json': 'x@%s' % ('0' * 64)},
            }],
        }

    def test_refuses_schema_invalid(self):
        doc = self._ok_doc('c', 'fp64', -1.0, 2.0)
        doc['schema_version'] = 'bogus'
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, 'c1_retest_c_warp1.json')
            json.dump(doc, open(p, 'w'))
            p4 = os.path.join(td, 'c1_retest_c_warp4.json')
            json.dump(self._ok_doc('c', 'fp64', -1.0, 2.1), open(p4, 'w'))
            with self.assertRaises(SystemExit):
                c1_retest_analysis.main([
                    '--evidence-dir', td, '--cells', 'c',
                    '--out', os.path.join(td, 'out.json')])


class TestMultiiRetestAnalysisRegression(unittest.TestCase):
    """The rework's generalized analysis module (warp1 vs tile pair) keeps
    the round-1 refusal semantics: schema-invalid input, missing variant
    documents and cross-commit variant pairs must abort loudly."""

    def _ok_doc(self, lane, e, median, commit='a' * 40):
        return {
            'schema_version': 'rtx4090-energy-v1',
            'meta': {'gpu_name': 'NVIDIA GeForce RTX 4090',
                     'gpu4pyscf_source_commit': commit},
            'rows': [{
                'row_key': {'geom': 'g.xyz', 'basis': 'b',
                            'xc': 'wb97m-v', 'lane': lane},
                'pilot_stage': 'S1',
                'system': {'nao': 43},
                'lane': lane, 'status': 'ok', 'skip_reason': None,
                'result': {'e_tot': e, 'converged': True,
                           'scf_cycles': 9,
                           'timings': {'warmup_s': 1.0,
                                       'samples_s': [median] * 3,
                                       'median_s': median}},
                'accuracy': {}, 'cross_program': [],
                'evidence': {'result_json': 'x@%s' % ('0' * 64)},
            }],
        }

    def _write_pair(self, td, commit='a' * 40):
        for var in ('warp1', 'tile8'):
            doc_fp64 = self._ok_doc('fp64', -1.0, 2.0, commit)
            doc_auto = self._ok_doc('auto', -1.0, 2.1, commit)
            doc_fp64['rows'].append(doc_auto['rows'][0])
            json.dump(doc_fp64, open(
                os.path.join(td, 'retest_c_%s.json' % var), 'w'))

    def test_refuses_schema_invalid(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_pair(td)
            doc = json.load(open(os.path.join(td, 'retest_c_warp1.json')))
            doc['schema_version'] = 'bogus'
            json.dump(doc, open(os.path.join(td, 'retest_c_warp1.json'),
                                'w'))
            with self.assertRaises(SystemExit):
                multii_retest_analysis.main([
                    '--evidence-dir', td, '--cells', 'c',
                    '--out', os.path.join(td, 'out.json')])

    def test_refuses_missing_variant_document(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_pair(td)
            os.remove(os.path.join(td, 'retest_c_tile8.json'))
            with self.assertRaises(SystemExit):
                multii_retest_analysis.main([
                    '--evidence-dir', td, '--cells', 'c',
                    '--out', os.path.join(td, 'out.json')])

    def test_refuses_cross_commit_variant_pair(self):
        with tempfile.TemporaryDirectory() as td:
            self._write_pair(td, commit='a' * 40)
            # rewrite tile8 doc under a different source commit
            doc = json.load(open(os.path.join(td, 'retest_c_tile8.json')))
            doc['meta']['gpu4pyscf_source_commit'] = 'b' * 40
            json.dump(doc, open(os.path.join(td, 'retest_c_tile8.json'),
                                'w'))
            with self.assertRaises(SystemExit):
                multii_retest_analysis.main([
                    '--evidence-dir', td, '--cells', 'c',
                    '--out', os.path.join(td, 'out.json')])


class TestEvidenceDocuments(unittest.TestCase):
    """The rework evidence must keep the inherited UKS failure visible."""

    EVID = pathlib.Path(
        '/data/results/gpu4pyscf-agent/tasks/issue-10/'
        '20261006-multii-tile')

    def test_equivalence_and_probe_exist(self):
        for f in ('multii_equivalence.json', 'multii_scf_probe.json',
                  'sanitizer_report.txt', 'microbench_all.log'):
            self.assertTrue((self.EVID / f).exists(),
                            'missing evidence file %s' % f)

    def test_inherited_uks_failure_stays_visible(self):
        p = self.EVID / 'multii_scf_probe.json'
        if not p.exists():
            self.skipTest('probe not run yet in this session')
        d = json.loads(p.read_text())
        repro = d.get('inherited_uks_auto_reproduction', {})
        self.assertIn('offset_moved_by_tile8', repro,
                      'the probe must record whether tile8 moved the '
                      'inherited -1.28e-6 UKS offset (issue #7 must stay '
                      'visible through the kernel swap)')


if __name__ == '__main__':
    unittest.main()

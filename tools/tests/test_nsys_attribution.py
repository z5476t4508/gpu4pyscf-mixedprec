"""Unit tests for the issue-#9 nsys kernel-bucket attribution.

Builds synthetic nsys sqlite exports (the schema subset nsys_attribution.py
reads) and checks the two pitfalls that produced 99.6-100% bogus
[unattributed] time on live data before the fix:

  * range names in the direct ``text`` column vs ``StringIds`` (textId);
  * nested NVTX ranges -- a kernel whose inner sibling range already ended
    must still be charged to its enclosing outer range (the VV10 case).

Also checks the warmup/instrumented separation and innermost-wins ordering.
"""
import importlib.util
import json
import pathlib
import sqlite3
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'
_spec = importlib.util.spec_from_file_location(
    'nsys_attribution', _DIR / 'nsys_attribution.py')
na = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(na)


def _make_db(path, ranges, kernels):
    '''ranges: (start, end, name, in_text_col); kernels: (start, end, name)'''
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.execute('CREATE TABLE NVTX_EVENTS (start, end, eventType, text, '
                'textId)')
    cur.execute('CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value)')
    cur.execute('CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start, end, '
                'demangledName, shortName)')
    # a few well-known string ids; range/kernel names resolve through them
    # when text is NULL
    sid = 100
    for s, e, name, in_text in ranges:
        if in_text:
            cur.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                        (s, e, 59, name, None))
        else:
            cur.execute('INSERT INTO StringIds VALUES (?,?)', (sid, name))
            cur.execute('INSERT INTO NVTX_EVENTS VALUES (?,?,?,?,?)',
                        (s, e, 59, None, sid))
            sid += 1
    for s, e, name in kernels:
        # real exports store a StringIds integer in demangledName; the name
        # text lives in the StringIds table (shortName only in old versions)
        cur.execute('INSERT INTO StringIds VALUES (?,?)', (sid, name))
        cur.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES '
                    '(?,?,?,?)', (s, e, sid, None))
        sid += 1
    con.commit()
    con.close()


class NsysAttributionTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = pathlib.Path(self.tmp.name) / 'nsys_test.sqlite'

    def tearDown(self):
        self.tmp.cleanup()

    def test_range_names_in_text_column_are_used(self):
        # live data: 945/1149 ranges carried their name in `text`; an inner
        # join on StringIds dropped them all -> everything [unattributed]
        _make_db(self.db,
                 ranges=[(0, 1_000_000_000, 'scf.xc.grid', True)],
                 kernels=[(100_000_000, 200_000_000, 'some_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(cell['unattributed_kernel_s'], 0.0)
        self.assertIn('scf.xc.grid', cell['buckets'])
        self.assertEqual(
            cell['buckets']['scf.xc.grid']['t_s'], 0.1)

    def test_range_names_in_stringids_are_used(self):
        _make_db(self.db,
                 ranges=[(0, 1_000_000_000, 'scf.nlc', False)],
                 kernels=[(100_000_000, 200_000_000, 'vv10_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(cell['unattributed_kernel_s'], 0.0)
        self.assertEqual(cell['buckets']['scf.nlc']['t_s'], 0.1)

    def test_nested_ranges_sibling_ended_does_not_orphan(self):
        # the live VV10 bug: kernel sits in an OUTER range, but an inner
        # sibling range that started earlier already ENDED before the kernel
        # -- the walk must skip past siblings, not break on them
        _make_db(self.db,
                 ranges=[(0, 1_000_000_000, 'scf.nlc', True),
                         (10_000_000, 50_000_000, 'scf.xc.grid_rho', True)],
                 kernels=[(100_000_000, 200_000_000, 'vv10_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(cell['unattributed_kernel_s'], 0.0)
        self.assertEqual(cell['buckets']['scf.nlc']['t_s'], 0.1)
        self.assertNotIn('scf.xc.grid_rho', cell['buckets'])

    def test_innermost_enclosing_range_wins(self):
        _make_db(self.db,
                 ranges=[(0, 1_000_000_000, 'scf.xc.grid', True),
                         (100_000_000, 300_000_000, 'scf.xc.grid_rho', True)],
                 kernels=[(150_000_000, 250_000_000, 'rho_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(cell['buckets']['scf.xc.grid_rho']['t_s'], 0.1)
        self.assertNotIn('scf.xc.grid', cell['buckets'])

    def test_kernel_outside_every_range_is_unattributed(self):
        _make_db(self.db,
                 ranges=[(0, 50_000_000, 'scf.init', True)],
                 kernels=[(100_000_000, 200_000_000, 'init_load_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(cell['unattributed_kernel_s'], 0.1)
        self.assertIn('[unattributed]', cell['buckets'])

    def test_warmup_sc_separated_from_instrumented(self):
        # two scf-kernel ranges: the warmup (first; the real profiler's
        # warmup has NO StageTimer, so no function buckets inside it) and
        # the instrumented sample (second, with function buckets); a kernel
        # in the first is warmup, one in the second keeps its bucket
        _make_db(self.db,
                 ranges=[(0, 400_000_000, 'scf-kernel', True),
                         (500_000_000, 900_000_000, 'scf-kernel', True),
                         (600_000_000, 700_000_000, 'scf.xc.grid', True)],
                 kernels=[(110_000_000, 150_000_000, 'warmup_gemm'),
                          (610_000_000, 650_000_000, 'instr_gemm'),
                          (750_000_000, 760_000_000, 'gap_kernel')])
        cell = na.attribute_sqlite(self.db)
        self.assertEqual(
            cell['buckets']['warmup/scf-kernel']['t_s'], 0.04)
        self.assertEqual(cell['buckets']['scf.xc.grid']['t_s'], 0.04)
        self.assertEqual(cell['buckets']['scf-kernel/gaps']['t_s'], 0.01)
        # instrumented = function buckets + gaps, warmup excluded
        self.assertAlmostEqual(cell['instrumented_kernel_time_s'], 0.05)
        self.assertAlmostEqual(cell['kernel_time_total_s'], 0.09)

    def test_missing_kernel_table_raises(self):
        con = sqlite3.connect(self.db)
        con.execute('CREATE TABLE NVTX_EVENTS (start, end, eventType, '
                    'text, textId)')
        con.commit()
        con.close()
        with self.assertRaises(sqlite3.OperationalError):
            na.attribute_sqlite(self.db)

    def test_live_wb97mv_export_attributes_vv10(self):
        '''The real nsys export of the wb97m-v cell (GPU-run, preserved
        evidence): the VV10 kernel must carry ~all of the NLC bucket and the
        unattributed share must be 0 (this is the regression test for the
        98.1% bogus unattributed time found live).'''
        path = pathlib.Path(
            '/data/results/gpu4pyscf-agent/tasks/issue-9/'
            'nsys_wb97mv_svp.sqlite')
        if not path.exists():
            self.skipTest('nsys sqlite export not present in this '
                          'environment')
        cell = na.attribute_sqlite(path)
        self.assertEqual(cell['unattributed_kernel_s'], 0.0)
        nlc = cell['buckets'].get('scf.nlc', {})
        top = dict(nlc.get('top_kernels', []))
        vv10 = [v for k, v in top.items() if 'vv10' in k]
        self.assertTrue(vv10, 'vv10 kernel missing from the NLC bucket')
        self.assertGreater(vv10[0]['t_s'], 5.0)

    def test_cli_writes_valid_json(self):
        _make_db(self.db,
                 ranges=[(0, 1_000_000_000, 'scf.xc.grid', True)],
                 kernels=[(100_000_000, 200_000_000, 'some_kernel')])
        # place the synthetic db under the tag the CLI looks for
        outdir = pathlib.Path(self.tmp.name)
        tagged = outdir / 'nsys_b3lyp_svp.sqlite'
        self.db.rename(tagged)
        rc = na.main(['--outdir', str(outdir), '--tags', 'b3lyp_svp'])
        self.assertEqual(rc, 0)
        doc = json.loads(
            (outdir / 'nsys_kernel_attribution.json').read_text())
        self.assertEqual(doc['schema_version'], 'nsys-kernel-attribution-v1')
        self.assertEqual(len(doc['cells']), 1)
        self.assertEqual(doc['cells'][0]['tag'], 'b3lyp_svp')


if __name__ == '__main__':
    unittest.main()

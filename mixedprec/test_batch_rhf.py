"""Tests for the batch DF-RHF driver.

The parsing / sharding / storage tests run without a GPU.  The final class
needs a GPU and is skipped when gpu4pyscf or cupy cannot be imported.

    python mixedprec/test_batch_rhf.py            # everything available
    python mixedprec/test_batch_rhf.py TestParseXYZ
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from batch_rhf import (  # noqa: E402
    COMPLETED, FAILED, PENDING, RUNNING, BatchError, ShardStore, fingerprint,
    iter_xyz_files, main, parse_comment_metadata, parse_xyz, read_result,
    shard_tasks)

try:
    import cupy  # noqa: F401
    import gpu4pyscf  # noqa: F401
    HAVE_GPU = True
except Exception:
    HAVE_GPU = False

WATER = '''3
water
O   0.000000  0.000000   0.117400
H  -0.757000  0.000000  -0.469600
H   0.757000  0.000000  -0.469600
'''

CONFIG = {
    'basis': 'def2-svp', 'auxbasis': 'def2-svp-jkfit', 'mode': 'fp32',
    'conv_tol': 3e-5, 'max_cycle': 50, 'default_charge': 0, 'default_spin': 0,
    'shard_index': 0, 'num_shards': 1, 'cderi_precision': 'fp64',
}


def write(path, text):
    with open(path, 'w') as f:
        f.write(text)
    return path


class TempDirCase(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)


class TestParseXYZ(TempDirCase):

    def test_basic(self):
        path = write(os.path.join(self.dir, 'w.xyz'), WATER)
        spec = parse_xyz(path)
        self.assertEqual(spec['natm'], 3)
        self.assertEqual(spec['charge'], 0)
        self.assertEqual(spec['spin'], 0)
        self.assertEqual(len(spec['atom'].splitlines()), 3)
        self.assertTrue(spec['atom'].startswith('O '))

    def test_trailing_blank_lines_ignored(self):
        path = write(os.path.join(self.dir, 'blank.xyz'), WATER + '\n\n  \n')
        self.assertEqual(parse_xyz(path)['natm'], 3)

    def test_comment_charge_and_spin(self):
        text = WATER.replace('water', 'charge=-1 spin=2')
        path = write(os.path.join(self.dir, 'ion.xyz'), text)
        spec = parse_xyz(path)
        self.assertEqual(spec['charge'], -1)
        self.assertEqual(spec['spin'], 2)

    def test_comment_defaults_apply(self):
        path = write(os.path.join(self.dir, 'w.xyz'), WATER)
        spec = parse_xyz(path, default_charge=1, default_spin=1)
        self.assertEqual(spec['charge'], 1)
        self.assertEqual(spec['spin'], 1)

    def test_comment_overrides_default(self):
        text = WATER.replace('water', 'title charge=2')
        path = write(os.path.join(self.dir, 'w.xyz'), text)
        spec = parse_xyz(path, default_charge=0, default_spin=3)
        self.assertEqual(spec['charge'], 2)
        self.assertEqual(spec['spin'], 3)

    def test_plain_title_is_not_metadata(self):
        self.assertEqual(parse_comment_metadata('Tamoxifen'), {})
        self.assertEqual(parse_comment_metadata('Structure: R14 Stoich: C12'), {})

    def test_bad_metadata_value(self):
        with self.assertRaises(BatchError):
            parse_comment_metadata('charge=abc')

    def test_atom_count_mismatch(self):
        path = write(os.path.join(self.dir, 'short.xyz'), '5\ntitle\nH 0 0 0\n')
        with self.assertRaises(BatchError) as ctx:
            parse_xyz(path)
        self.assertIn('declares 5 atoms', str(ctx.exception))

    def test_non_numeric_count(self):
        path = write(os.path.join(self.dir, 'bad.xyz'), 'x\ntitle\nH 0 0 0\n')
        with self.assertRaises(BatchError):
            parse_xyz(path)

    def test_non_numeric_coordinates(self):
        path = write(os.path.join(self.dir, 'bad.xyz'), '1\ntitle\nH 0 0 abc\n')
        with self.assertRaises(BatchError):
            parse_xyz(path)

    def test_short_coordinate_line(self):
        path = write(os.path.join(self.dir, 'bad.xyz'), '1\ntitle\nH 0 0\n')
        with self.assertRaises(BatchError):
            parse_xyz(path)

    def test_empty_file(self):
        path = write(os.path.join(self.dir, 'empty.xyz'), '')
        with self.assertRaises(BatchError):
            parse_xyz(path)

    def test_coordinates_survive_roundtrip(self):
        path = write(os.path.join(self.dir, 'w.xyz'), WATER)
        first = parse_xyz(path)['atom'].splitlines()[1].split()
        self.assertEqual(first[0], 'H')
        self.assertAlmostEqual(float(first[1]), -0.757)
        self.assertAlmostEqual(float(first[3]), -0.4696)


class TestDiscoveryAndSharding(TempDirCase):

    def test_sorted_and_recursive(self):
        os.makedirs(os.path.join(self.dir, 'sub'))
        write(os.path.join(self.dir, 'b.xyz'), WATER)
        write(os.path.join(self.dir, 'a.xyz'), WATER)
        write(os.path.join(self.dir, 'sub', 'c.xyz'), WATER)
        write(os.path.join(self.dir, 'note.txt'), 'ignored')

        ids = [mol_id for mol_id, _ in iter_xyz_files(self.dir)]
        self.assertEqual(ids, ['a.xyz', 'b.xyz', os.path.join('sub', 'c.xyz')])

    def test_empty_directory(self):
        with self.assertRaises(BatchError):
            iter_xyz_files(self.dir)

    def test_missing_directory(self):
        with self.assertRaises(BatchError):
            iter_xyz_files(os.path.join(self.dir, 'nope'))

    def test_shards_partition_exactly(self):
        tasks = [(f'{i}.xyz', f'/tmp/{i}.xyz') for i in range(10)]
        for num_shards in (1, 2, 3, 4, 7, 10, 13):
            chunks = [shard_tasks(tasks, i, num_shards) for i in range(num_shards)]
            flat = [t for chunk in chunks for t in chunk]
            self.assertEqual(flat, tasks)
            sizes = [len(c) for c in chunks]
            self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_invalid_shard_arguments(self):
        tasks = [('a.xyz', '/tmp/a.xyz')]
        with self.assertRaises(BatchError):
            shard_tasks(tasks, 0, 0)
        with self.assertRaises(BatchError):
            shard_tasks(tasks, 2, 2)
        with self.assertRaises(BatchError):
            shard_tasks(tasks, -1, 2)

    def test_fingerprint_is_order_sensitive(self):
        self.assertEqual(fingerprint(['a', 'b']), fingerprint(['a', 'b']))
        self.assertNotEqual(fingerprint(['a', 'b']), fingerprint(['b', 'a']))
        self.assertNotEqual(fingerprint(['a', 'b']), fingerprint(['ab']))


class TestShardStore(TempDirCase):

    def setUp(self):
        super().setUp()
        self.path = os.path.join(self.dir, 'results.h5')
        self.ids = ['a.xyz', 'b.xyz', 'c.xyz']

    def store(self, ids=None, config=None):
        return ShardStore(self.path, ids or self.ids, config or CONFIG)

    def test_new_file_starts_pending(self):
        with self.store() as s:
            self.assertTrue((s.status() == PENDING).all())

    def test_commit_success_roundtrip(self):
        dm = np.arange(9, dtype=np.float32).reshape(3, 3)
        with self.store() as s:
            s.commit_success(1, -76.02, dm, natm=3, converged=True,
                             cycles=7, wall_time=1.5)
            self.assertEqual(s.status()[1], COMPLETED)

        out = read_result(self.path, 'b.xyz')
        self.assertEqual(out['status'], 'completed')
        self.assertAlmostEqual(out['energy'], -76.02)
        self.assertEqual(out['dm'].dtype, np.float32)
        np.testing.assert_array_equal(out['dm'], dm)
        self.assertEqual(out['cycles'], 7)
        self.assertTrue(out['converged'])

    def test_ragged_density_matrices(self):
        small = np.full((2, 2), 1.0, dtype=np.float32)
        large = np.full((5, 5), 2.0, dtype=np.float32)
        with self.store() as s:
            s.commit_success(0, -1.0, small, 2, True, 3, 0.1)
            s.commit_success(2, -2.0, large, 5, True, 4, 0.2)

        np.testing.assert_array_equal(read_result(self.path, 'a.xyz')['dm'], small)
        np.testing.assert_array_equal(read_result(self.path, 'c.xyz')['dm'], large)

    def test_unfinished_entries_report_pending(self):
        with self.store():
            pass
        self.assertEqual(read_result(self.path, 'a.xyz')['status'], 'pending')

    def test_commit_failure(self):
        with self.store() as s:
            s.commit_failure(0, 'RuntimeError', 'SCF exploded', 0.5)
            self.assertEqual(s.status()[0], FAILED)

        out = read_result(self.path, 'a.xyz')
        self.assertEqual(out['status'], 'failed')
        self.assertEqual(out['error_type'], 'RuntimeError')
        self.assertIn('exploded', out['error_message'])

    def test_running_is_reset_to_pending_on_reopen(self):
        with self.store() as s:
            s.mark_running(2)
            self.assertEqual(s.status()[2], RUNNING)

        with self.store() as s:
            self.assertEqual(s.status()[2], PENDING)

    def test_completed_survives_reopen(self):
        dm = np.eye(2, dtype=np.float32)
        with self.store() as s:
            s.commit_success(0, -1.0, dm, 2, True, 3, 0.1)
        with self.store() as s:
            self.assertEqual(s.status()[0], COMPLETED)
            self.assertEqual(s.status()[1], PENDING)

    def test_changed_input_list_is_rejected(self):
        with self.store():
            pass
        with self.assertRaises(BatchError) as ctx:
            with self.store(ids=['a.xyz', 'b.xyz', 'd.xyz']):
                pass
        self.assertIn('input molecule list changed', str(ctx.exception))

    def test_changed_config_is_rejected(self):
        with self.store():
            pass
        with self.assertRaises(BatchError) as ctx:
            with self.store(config=dict(CONFIG, basis='6-31g')):
                pass
        self.assertIn('config mismatch', str(ctx.exception))

    def test_changed_conv_tol_is_rejected(self):
        with self.store():
            pass
        with self.assertRaises(BatchError):
            with self.store(config=dict(CONFIG, conv_tol=1e-8)):
                pass

    def test_same_config_reopens(self):
        with self.store():
            pass
        with self.store(config=dict(CONFIG)) as s:
            self.assertEqual(len(s.status()), 3)


class TestCLI(TempDirCase):

    def setUp(self):
        super().setUp()
        self.inp = os.path.join(self.dir, 'in')
        os.makedirs(self.inp)
        for name in ('a', 'b', 'c', 'd'):
            write(os.path.join(self.inp, f'{name}.xyz'), WATER)

    def test_fp32_cderi_rejected_outside_the_fp32_lane(self):
        rc = main(['--input-dir', self.inp,
                   '--output-dir', os.path.join(self.dir, 'out'),
                   '--mode', 'auto', '--cderi-precision', 'fp32'])
        self.assertEqual(rc, 2)

    def test_dry_run_lists_shard_only(self):
        rc = main(['--input-dir', self.inp,
                   '--output-dir', os.path.join(self.dir, 'out'),
                   '--shard-index', '1', '--num-shards', '2', '--dry-run'])
        self.assertEqual(rc, 0)
        self.assertFalse(os.path.exists(os.path.join(self.dir, 'out')))


@unittest.skipUnless(HAVE_GPU, 'gpu4pyscf/cupy not available')
class TestGPUBatch(unittest.TestCase):
    '''End-to-end GPU checks against a plain fp64 DF-RHF reference.'''

    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.mkdtemp()
        cls.inp = os.path.join(cls.dir, 'in')
        os.makedirs(cls.inp)
        write(os.path.join(cls.inp, 'water.xyz'), WATER)

        import pyscf
        from gpu4pyscf import scf
        mol = pyscf.M(atom=parse_xyz(os.path.join(cls.inp, 'water.xyz'))['atom'],
                      basis='def2-svp', verbose=0)
        mf = scf.RHF(mol).density_fit(auxbasis='def2-svp-jkfit')
        mf.conv_tol = 1e-10
        cls.e_ref = mf.kernel()
        cls.nao = mol.nao

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_fp32_lane_matches_reference(self):
        out = os.path.join(self.dir, 'fp32')
        rc = main(['--input-dir', self.inp, '--output-dir', out])
        self.assertEqual(rc, 0)

        result = read_result(os.path.join(out, 'results-00000-of-00001.h5'), 'water.xyz')
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['converged'])
        self.assertAlmostEqual(result['energy'], self.e_ref, delta=1e-3)
        self.assertEqual(result['dm'].shape, (self.nao, self.nao))
        self.assertEqual(result['dm'].dtype, np.float32)
        self.assertEqual(result['natm'], 3)

    def test_auto_lane_recovers_full_accuracy(self):
        out = os.path.join(self.dir, 'auto')
        rc = main(['--input-dir', self.inp, '--output-dir', out,
                   '--mode', 'auto', '--conv-tol', '1e-10'])
        self.assertEqual(rc, 0)

        result = read_result(os.path.join(out, 'results-00000-of-00001.h5'), 'water.xyz')
        self.assertAlmostEqual(result['energy'], self.e_ref, delta=1e-9)

    def test_resume_skips_completed_work(self):
        out = os.path.join(self.dir, 'resume')
        main(['--input-dir', self.inp, '--output-dir', out])
        h5 = os.path.join(out, 'results-00000-of-00001.h5')
        first = read_result(h5, 'water.xyz')

        main(['--input-dir', self.inp, '--output-dir', out])
        second = read_result(h5, 'water.xyz')
        self.assertEqual(first['energy'], second['energy'])
        self.assertEqual(first['wall_time'], second['wall_time'])

    def test_precision_mode_is_restored_after_a_run(self):
        from gpu4pyscf.lib import precision
        main(['--input-dir', self.inp, '--output-dir', os.path.join(self.dir, 'restore')])
        self.assertEqual(precision.get_precision(), 'fp64')


@unittest.skipUnless(HAVE_GPU, 'gpu4pyscf/cupy not available')
class TestGPUFailureHandling(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)
        self.inp = os.path.join(self.dir, 'in')
        os.makedirs(self.inp)
        write(os.path.join(self.inp, 'good.xyz'), WATER)
        write(os.path.join(self.inp, 'broken.xyz'), '2\ntitle\nH 0 0 0\n')

    def test_failed_molecule_does_not_stop_the_batch(self):
        out = os.path.join(self.dir, 'out')
        rc = main(['--input-dir', self.inp, '--output-dir', out])
        self.assertEqual(rc, 1)

        h5 = os.path.join(out, 'results-00000-of-00001.h5')
        broken = read_result(h5, 'broken.xyz')
        self.assertEqual(broken['status'], 'failed')
        self.assertEqual(broken['error_type'], 'BatchError')
        self.assertEqual(read_result(h5, 'good.xyz')['status'], 'completed')


if __name__ == '__main__':
    unittest.main()

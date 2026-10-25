'''Unit tests for the four-arm upstream comparison (control card
2026-10-18): the official-package isolation guard (fail-closed), provenance
verification, the balanced-blocks Latin-square schedule, the merge /
comparison-document contract, the arm environment discipline, the batch
index helpers, and a full-stack compatibility proof of the merged documents
against the operator's three_way_report.load_cell (via the deployment
copies in tools/reporting/, whose sha256 is checked against the deployed
/opt/qc-agent files).  No GPU / no pyscf import needed -- stdlib only.'''

import hashlib
import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import types
import unittest
from unittest import mock

REPO = pathlib.Path(__file__).parents[2]
_DIR = REPO / 'mixedprec' / 'benchmark' / 'rtx4090'
_REPORTING = REPO / 'tools' / 'reporting'

_DEPLOYED_SHA = {
    'three_way_report.py':
        'ebd47471591943c1107077b3ca5b5b7dadfa18e173018b3f029abe09323131ea',
    'effects_report.py':
        'ffc9c82527d517bb8e9ceac0d1d9782842ebf39702858de904543a7fa96d5837',
}


def _load(name):
    spec = importlib.util.spec_from_file_location(name, _DIR / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


uc = _load('upstream_comparison')
us = _load('upstream_schedule')
rb = _load('run_upstream_batch')
adj = _load('upstream_adjudicate')


def _load_reporting(name):
    # the deployment copies import each other by module name
    if str(_REPORTING) not in sys.path:
        sys.path.insert(0, str(_REPORTING))
    spec = importlib.util.spec_from_file_location(
        f'tools_reporting_{name}', _REPORTING / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# official-package isolation guard
# ---------------------------------------------------------------------------

class _FakeTree:
    '''A temp "official root" and a temp "repo root", each with its own
    gpu4pyscf package; used to prove the guard resolves only the official
    one and fails closed on names missing there.'''

    def __init__(self, root):
        self.root = root
        pkg = root / 'gpu4pyscf'
        pkg.mkdir()
        (pkg / '__init__.py').write_text('MARK = "official"\n')
        (pkg / 'leaf.py').write_text('VALUE = 1\n')
        sub = pkg / 'sub'
        sub.mkdir()
        (sub / '__init__.py').write_text('')
        (sub / 'mod.py').write_text('VALUE = 2\n')

    def add_repo_only_module(self):
        (self.root / 'gpu4pyscf' / 'repoonly.py').write_text(
            'VALUE = 3  # must never load in the official arm\n')


class IsolationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.official = pathlib.Path(self._tmp.name) / 'official-1.8.1'
        self.official.mkdir()
        self.tree = _FakeTree(self.official)
        self.repo_root = pathlib.Path(self._tmp.name) / 'repo'
        self.repo_root.mkdir()
        self._repo_tree = _FakeTree(self.repo_root)
        # purge any gpu4pyscf module state around each test
        self._saved_modules = {k: v for k, v in sys.modules.items()
                               if k == 'gpu4pyscf'
                               or k.startswith('gpu4pyscf.')}
        for k in list(self._saved_modules):
            sys.modules.pop(k, None)

    def tearDown(self):
        for k in list(sys.modules):
            if k == 'gpu4pyscf' or k.startswith('gpu4pyscf.'):
                sys.modules.pop(k, None)
        sys.modules.update(self._saved_modules)

    def _install(self):
        finder = uc.OfficialOnlyFinder(str(self.official))
        sys.meta_path.insert(0, finder)
        self.addCleanup(sys.meta_path.remove, finder)
        sys.path.insert(0, str(self.repo_root))
        self.addCleanup(sys.path.remove, str(self.repo_root))

    def test_guard_resolves_only_official(self):
        self._install()
        import gpu4pyscf
        self.assertIn(str(self.official), gpu4pyscf.__file__)
        self.assertEqual(gpu4pyscf.MARK, 'official')
        from gpu4pyscf.leaf import VALUE
        self.assertEqual(VALUE, 1)
        from gpu4pyscf.sub import mod
        self.assertEqual(mod.VALUE, 2)
        self.assertTrue(all(
            os.path.abspath(m.__file__).startswith(str(self.official))
            for n, m in sys.modules.items()
            if (n == 'gpu4pyscf' or n.startswith('gpu4pyscf.'))
            and hasattr(m, '__file__')))

    def test_guard_fails_closed_on_repo_only_module(self):
        self._repo_tree.add_repo_only_module()
        self._install()
        with self.assertRaises(ImportError):
            import gpu4pyscf.repoonly  # noqa: F401

    def test_install_isolation_drops_repo_paths_and_leaked_env(self):
        with mock.patch.dict(os.environ, {'GPU4PYSCF_XC_GRID_BLOCKSIZE':
                                          '16384'}):
            with self.assertRaises(RuntimeError):
                uc.install_official_isolation(str(self.official))
        env = {k: v for k, v in os.environ.items()
               if k.startswith('GPU4PYSCF_')}
        if env:
            self.skipTest('test host leaks GPU4PYSCF_* vars')
        with mock.patch.object(uc, 'REPO', str(self.repo_root)):
            sys.path.insert(0, str(self.repo_root))
            try:
                info = uc.install_official_isolation(str(self.official))
                self.assertEqual(info['repo_paths_dropped'],
                                 [str(self.repo_root)])
                self.assertNotIn(str(self.repo_root), sys.path)
            finally:
                if str(self.repo_root) in sys.path:
                    sys.path.remove(str(self.repo_root))

    def test_verify_provenance_accepts_official_and_rejects_drift(self):
        mod = types.ModuleType('gpu4pyscf.leaf')
        mod.__file__ = str(self.official / 'gpu4pyscf' / 'leaf.py')
        sys.modules['gpu4pyscf.leaf'] = mod
        manifest = {'checked_files': 1, 'files': {
            'gpu4pyscf/leaf.py': hashlib.sha256(
                (self.official / 'gpu4pyscf' / 'leaf.py')
                .read_bytes()).hexdigest()}}
        out = uc.verify_official_provenance(str(self.official), manifest)
        self.assertIn('gpu4pyscf/leaf.py', out['loaded_module_files'])
        # drifted content -> hard failure
        (self.official / 'gpu4pyscf' / 'leaf.py').write_text('tampered')
        with self.assertRaises(RuntimeError):
            uc.verify_official_provenance(str(self.official), manifest)
        # a module loaded from outside the official root -> hard failure
        (self.official / 'gpu4pyscf' / 'leaf.py').write_text('VALUE = 1\n')
        rogue = types.ModuleType('gpu4pyscf.rogue')
        rogue.__file__ = str(self.repo_root / 'gpu4pyscf' / 'leaf.py')
        sys.modules['gpu4pyscf.rogue'] = rogue
        try:
            with self.assertRaises(RuntimeError):
                uc.verify_official_provenance(str(self.official), manifest)
        finally:
            sys.modules.pop('gpu4pyscf.rogue', None)


# ---------------------------------------------------------------------------
# balanced-blocks schedule
# ---------------------------------------------------------------------------

class ScheduleTests(unittest.TestCase):
    def test_latin_squares_balanced_and_deterministic(self):
        specs = us.stage1_mediumchon_cells()
        self.assertEqual(len(specs), 48)
        cells = us.build_cells(specs)
        self.assertEqual(cells, us.build_cells(specs))
        for cell in cells:
            blocks = cell['blocks']
            self.assertEqual(len(blocks), 4)
            for row in blocks:
                self.assertEqual(sorted(row), sorted(['O', 'B', 'C', 'R']))
            for pos in range(4):
                self.assertEqual(
                    sorted(b[pos] for b in blocks),
                    sorted(['O', 'B', 'C', 'R']))

    def test_stage1_matches_frozen_mediumchon(self):
        specs = us.stage1_mediumchon_cells()
        self.assertEqual(
            [os.path.basename(s['geom']) for s in specs[:4]],
            ['aspirin.xyz'] * 4)
        for spec in specs:
            self.assertEqual(spec['charge'], 0)
            self.assertEqual(spec['spin'], 0)
        bases = {(s['xc'], s['basis']) for s in specs}
        self.assertEqual(bases, {('b3lyp', 'def2-svp'),
                                  ('m062x', 'def2-svp'),
                                  ('wb97mv', 'def2-svp'),
                                  ('r2scan3c', 'def2-mtzvpp')})

    def test_verify_schedule_rejects_broken(self):
        specs = us.stage1_mediumchon_cells()[:3]
        cells = us.build_cells(specs)
        good = {'schema_version': us.SCHEMA, 'seed': us.SEED,
                'batch_id': 'b', 'source_commit': 'c',
                'source_root': '/x', 'settings': us.SETTINGS,
                'cells': cells}
        us.verify_schedule(good)
        broken = json.loads(json.dumps(good))
        broken['cells'][0]['blocks'][0][0] = 'B'  # duplicate in a row
        with self.assertRaises(ValueError):
            us.verify_schedule(broken)
        broken = json.loads(json.dumps(good))
        broken['cells'][0]['blocks'] = broken['cells'][0]['blocks'][:3]
        with self.assertRaises(ValueError):
            us.verify_schedule(broken)
        broken = json.loads(json.dumps(good))
        broken['seed'] = 1
        with self.assertRaises(ValueError):
            us.verify_schedule(broken)
        broken = json.loads(json.dumps(good))
        broken['cells'][1]['tag'] = broken['cells'][0]['tag']
        with self.assertRaises(ValueError):
            us.verify_schedule(broken)


# ---------------------------------------------------------------------------
# arm environment discipline + role env
# ---------------------------------------------------------------------------

class EnvDisciplineTests(unittest.TestCase):
    def test_optimization_env_of(self):
        self.assertEqual(uc.optimization_env_of('upstream'), {})
        self.assertEqual(uc.optimization_env_of('strict_fp64'), {})
        self.assertEqual(uc.optimization_env_of('optimized_fp64'),
                         uc.optimization_env_of('optimized_auto'))
        self.assertIn('GPU4PYSCF_SCF_VMAT_F32',
                      uc.optimization_env_of('optimized_fp64'))
        self.assertEqual(
            uc.optimization_env_of('optimized_fp64')['GPU4PYSCF_SCF_VMAT_F32'],
            'headtail')

    def test_check_env_discipline(self):
        stack = uc.STACK_ENV
        with mock.patch.dict(os.environ, stack, clear=False):
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
            os.environ.update(stack)
            out = uc.check_env_discipline('optimized_fp64')
            self.assertEqual(out['seen'], stack)
            with self.assertRaises(RuntimeError):
                uc.check_env_discipline('strict_fp64')
        clean = {k: v for k, v in os.environ.items()
                 if not k.startswith('GPU4PYSCF_')}
        with mock.patch.dict(os.environ, clean, clear=True):
            uc.check_env_discipline('strict_fp64')
            uc.check_env_discipline('upstream')
            with self.assertRaises(RuntimeError):
                uc.check_env_discipline('optimized_auto')

    def test_driver_arm_env_scrubs_and_sets(self):
        base = {'PATH': '/bin', 'PYTHONPATH': '/somewhere',
                'GPU4PYSCF_DF_IP1_F32': '1', 'HOME': '/home/x'}
        with mock.patch.dict(os.environ, base, clear=True):
            env = rb.arm_env('strict_fp64')
            self.assertNotIn('PYTHONPATH', env)
            self.assertNotIn('GPU4PYSCF_DF_IP1_F32', env)
            env = rb.arm_env('optimized_auto')
            self.assertEqual(env['GPU4PYSCF_DF_IP1_F32'], '1')
            self.assertEqual(env['GPU4PYSCF_SCF_VMAT_F32'], 'headtail')
            env = rb.arm_env('upstream')
            self.assertFalse(
                [k for k in env if k.startswith('GPU4PYSCF_')])

    def test_dependency_versions_keys(self):
        deps = uc.dependency_versions()
        self.assertEqual(set(deps), {'pyscf', 'cupy', 'numpy', 'libxc',
                                    'cutensor'})


# ---------------------------------------------------------------------------
# merge + comparison document contract
# ---------------------------------------------------------------------------

def _synthetic_repeat(idx, e, block, sample_s=10.0, scf=5.0, grad=2.5,
                      converged=True):
    return {'index': idx, 'e_tot': e, 'converged': converged,
            'scf_cycles': 12,
            'gradient': [[0.01, 0.02, 0.03], [0.0, -0.01, 0.005]],
            'gradient_norm': 0.037,
            'gradient_mode': 'fp64', 'grad_ip1_hits': 0,
            'e_terms': None,
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'stability_verification': None, 'sample_s': sample_s,
            't_mol_s': 0.5, 't_cderi_s': 1.0, 't_scf_s': scf,
            't_grad_s': grad, 'comparison_block': block}


def _synthetic_record(role, block, batch_id, schedule_sha256, e=-76.0,
                      lane=None, fp32_hits=None, status='ok'):
    lane = lane or uc.ROLE_TO_LANE[role]
    fp32_hits = fp32_hits if fp32_hits is not None else (
        {} if role == 'upstream' else
        {'df_jk': 7, 'grad_ip1': 3} if role != 'strict_fp64' else
        {'df_jk': 0, 'grad_ip1': 0})
    row = {
        'row_key': {'geom': 'aspirin.xyz', 'basis': 'def2-svp',
                    'xc': 'b3lyp', 'lane': lane},
        'pilot_stage': 'S2',
        'system': {'geom_file': 'aspirin.xyz',
                   'geom_sha256': 'a' * 64, 'natm': 13, 'nao': 157,
                   'charge': 0, 'spin': 0, 'nelec': 66, 'cart': False,
                   'geom_units': 'angstrom'},
        'method': {'xc': 'b3lyp', 'basis': 'def2-svp',
                   'auxbasis': 'def2-universal-jkfit', 'with_df': True,
                   'grids': {'level': 3, 'atom_grid': None},
                   'nlcgrids': {'level': 3, 'atom_grid': None},
                   'conv_tol': 1e-10, 'max_cycle': 50},
        'status': status, 'skip_reason': None, 'gate': None,
        'result': {'e_tot': e, 'converged': True, 'scf_cycles': 12,
                   'gradient': [[0.01, 0.02, 0.03], [0.0, -0.01, 0.005]],
                   'gradient_norm': 0.037, 'gradient_mode': 'fp64',
                   'repeats': [_synthetic_repeat(0, e, block)],
                   'repeat_evidence': 'all-repeats', 'e_terms': None,
                   'timings': {'warmup_s': 9.0, 'samples_s': [10.0],
                               'median_s': 10.0, 'min_s': 10.0,
                               'max_s': 10.0, 'end_to_end_s': 10.0,
                               't_cderi_s': 1.0, 't_scf_s': 5.0,
                               't_grad_s': 2.5, 't_xc_grid_s': None,
                               't_jk_s': None, 't_nlc_s': None,
                               't_disp_s': None, 'sync': 'x'},
                   'mem': {'within_budget': True},
                   'fp32_entry_hits': fp32_hits,
                   'precision_switch': {'switched': False, 'cycle': None,
                                        'trigger': 'none'},
                   'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    return {
        'schema_version': uc.SAMPLE_SCHEMA, 'batch_id': batch_id,
        'schedule_sha256': schedule_sha256, 'cell_index': 0, 'role': role,
        'block': block, 'position': block % 4, 'lane': lane,
        'optimization_env': uc.optimization_env_of(role), 'row': row,
        'startup': {'t_import_and_setup_s': 8.0, 'warmup_s': 9.0},
        'provenance': {
            'gpu4pyscf_package_root':
                '/home/qczcode/gpu4pyscf-official-1.8.1/gpu4pyscf'
                if role == 'upstream' else '/snapshot/gpu4pyscf',
            'source_commit': None if role == 'upstream' else 'c' * 40,
            'source_root': None if role == 'upstream' else '/snapshot',
            'reference_manifest_sha256':
                '41348bc5660e33f35d19301524d72d0f8b5822e46c99e916681886d241251827'
                if role == 'upstream' else None},
        'command': 'python ...', 'recorded_at_utc': '2026-10-18T00:00:00Z',
        'meta_base': {'gpu_name': 'RTX 4090', 'driver_version': 'd',
                      'cuda_version': '12.8', 'cpu_model': 'c',
                      'python': '3.12', 'cupy': '13.4.1',
                      'pyscf': '2.14.0',
                      'dependencies': {'pyscf': '2.14.0', 'cupy': '13.4.1',
                                       'numpy': '2.2.6', 'libxc': '0.8.1',
                                       'cutensor': '2.2.0'}},
    }


class MergeContractTests(unittest.TestCase):
    BATCH = 'test-batch'
    SCHED_SHA = 'b' * 64

    def _samples(self, role, **kw):
        return [_synthetic_record(role, b, self.BATCH, self.SCHED_SHA, **kw)
                for b in range(4)]

    def _schedule(self):
        return {'batch_id': self.BATCH, 'settings': dict(us.SETTINGS),
                'cells': [{'index': 0, 'tag': 'aspirin_b3lyp_def2-svp',
                           'blocks': [['O', 'B', 'C', 'R']] * 4}]}

    def test_merge_and_validate_all_roles(self):
        for role in uc.ROLES:
            samples = self._samples(role)
            for s in samples:
                s['_path'] = f"/tmp/{role}_{s['block']}.json"
            doc = uc.merge_role(samples, role, self.BATCH, self._schedule(),
                                self.SCHED_SHA)
            uc.validate_comparison_doc(doc, role, self.BATCH, self.SCHED_SHA)
            self.assertEqual(len(doc['rows'][0]['result']['repeats']), 4)
            self.assertEqual(doc['meta']['timing_protocol'],
                             'balanced-blocks-v1')
            self.assertEqual(doc['meta']['schedule_sha256'], self.SCHED_SHA)
            self.assertEqual(doc['meta']['dependencies'],
                            {'pyscf': '2.14.0', 'cupy': '13.4.1',
                             'numpy': '2.2.6', 'libxc': '0.8.1',
                             'cutensor': '2.2.0'})
            self.assertEqual(
                doc['meta']['optimization_env'],
                uc.optimization_env_of(role))
            self.assertEqual(
                doc['meta']['precision_mode'],
                uc.PRECISION_MODE[role])

    def test_merge_rejects_incomplete_blocks(self):
        with self.assertRaises(ValueError):
            uc.merge_role(self._samples('upstream')[:3], 'upstream',
                          self.BATCH, self._schedule(), self.SCHED_SHA)

    def test_merge_rejects_dependency_drift(self):
        samples = self._samples('upstream')
        samples[2]['meta_base']['dependencies']['cupy'] = '14.2.0'
        for s in samples:
            s['_path'] = 'x'
        with self.assertRaises(ValueError):
            uc.merge_role(samples, 'upstream', self.BATCH, self._schedule(),
                          self.SCHED_SHA)

    def test_validator_rejects_broken_docs(self):
        def build(role, mutate):
            samples = self._samples(role)
            for s in samples:
                s['_path'] = 'x'
            doc = uc.merge_role(samples, role, self.BATCH,
                                self._schedule(), self.SCHED_SHA)
            mutate(doc)
            return doc, role

        cases = [
            ('strict fp32 hit', lambda d: d['rows'][0]['result']
                .__setitem__('fp32_entry_hits', {'grad_ip1': 1}),
             'strict_fp64'),
            ('strict gradient_mode', lambda d: d['rows'][0]['result']
                .__setitem__('gradient_mode', 'fp32'), 'strict_fp64'),
            ('auto without fp32', lambda d: d['rows'][0]['result']
                .__setitem__('fp32_entry_hits', {}), 'optimized_auto'),
            ('three repeats', lambda d: d['rows'][0]['result']
                ['repeats'].pop(), 'upstream'),
            ('wrong lane', lambda d: d['rows'][0]['row_key']
                .__setitem__('lane', 'auto'), 'strict_fp64'),
            ('wrong role meta', lambda d: d['meta']
                .__setitem__('comparison_role', 'other'), 'upstream'),
            ('upstream root not pinned', lambda d: d['meta']
                .__setitem__('gpu4pyscf_package_root', '/other/gpu4pyscf'),
             'upstream'),
        ]
        for name, mutate, role in cases:
            with self.subTest(name):
                doc, role = build(role, mutate)
                with self.assertRaises(ValueError):
                    uc.validate_comparison_doc(doc, role, self.BATCH,
                                               self.SCHED_SHA)


# ---------------------------------------------------------------------------
# index helpers
# ---------------------------------------------------------------------------

class IndexTests(unittest.TestCase):
    @staticmethod
    def _sched(n, batch='b1'):
        return {'batch_id': batch, 'cells': [
            {'index': i, 'tag': f'c{i}', 'geom': 'g.xyz',
             'basis': 'def2-svp', 'xc': 'b3lyp', 'charge': 0, 'spin': 0}
            for i in range(n)]}

    def test_prelisted_denominator_never_drops(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.json')
            schedule = self._sched(5)
            index, batch, added = rb.load_or_create_index(path, 'b1',
                                                          schedule)
            self.assertEqual(added, [])
            self.assertEqual(len(batch['cells']), 5)
            self.assertTrue(all(e['sources'] == {} for e in batch['cells']))
            rb.update_index(path, index, batch, schedule['cells'][2],
                            {'strict_fp64': {'path': 'p', 'sha256': 's'}})
            with open(path, encoding='utf-8') as f:
                reloaded = json.load(f)
            entry = reloaded['batches'][0]['cells'][2]
            self.assertEqual(entry['sources']['strict_fp64']['sha256'], 's')
            # re-running load keeps the filled entry and the denominator
            index2, batch2, _ = rb.load_or_create_index(path, 'b1', schedule)
            self.assertEqual(len(batch2['cells']), 5)
            self.assertEqual(batch2['cells'][2]['sources']
                             ['strict_fp64']['sha256'], 's')
            with self.assertRaises(RuntimeError):
                rb.update_index(path, index2, batch2,
                                {'index': 99, 'tag': 'x'}, {})

    def test_staged_fill_prelists_new_planned_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.json')
            index, batch, added = rb.load_or_create_index(
                path, 'b1', self._sched(5))
            self.assertEqual(added, [])
            rb.update_index(path, index, batch, self._sched(5)['cells'][2],
                            {'strict_fp64': {'path': 'p', 'sha256': 's'}})
            # the staged fill: the SAME batch id, schedule grown to 8 cells
            index2, batch2, added2 = rb.load_or_create_index(
                path, 'b1', self._sched(8))
            self.assertEqual(added2, ['c5', 'c6', 'c7'])
            self.assertEqual(len(batch2['cells']), 8)
            for i in (5, 6, 7):
                self.assertEqual(batch2['cells'][i]['sources'], {})
            self.assertEqual(batch2['cells'][2]['sources']
                             ['strict_fp64']['sha256'], 's')
            with open(path, encoding='utf-8') as f:
                on_disk = json.load(f)
            self.assertEqual(len(on_disk['batches'][0]['cells']), 8)
            self.assertTrue(all(on_disk['batches'][0]['cells'][i]
                                ['sources'] == {} for i in (5, 6, 7)))
            # a newly pre-listed cell can now be filled without surgery
            rb.update_index(path, index2, batch2, self._sched(8)['cells'][5],
                            {'upstream': {'path': 'p2', 'sha256': 's2'}})
            with open(path, encoding='utf-8') as f:
                on_disk = json.load(f)
            self.assertEqual(on_disk['batches'][0]['cells'][5]['sources']
                             ['upstream']['sha256'], 's2')

    def test_staged_fill_refuses_identity_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'index.json')
            base = self._sched(3)
            index, batch, _ = rb.load_or_create_index(path, 'b1', base)
            rb.update_index(path, index, batch, base['cells'][0],
                            {'strict_fp64': {'path': 'p', 'sha256': 's'}})
            mixed = json.loads(json.dumps(base))
            mixed['cells'][1]['tag'] = 'not-c1'
            with self.assertRaises(RuntimeError):
                rb.load_or_create_index(path, 'b1', mixed)

    def test_refuse_if_other_driver_scans_proc_cmdlines(self):
        with mock.patch('os.listdir', return_value=['1']), \
                mock.patch('builtins.open', mock.mock_open(
                    read_data=b'python\0run_upstream_batch.py\0--schedule'
                              b'\0x\0')):
            with self.assertRaises(RuntimeError):
                rb.refuse_if_other_driver()
        with mock.patch('os.listdir', return_value=['1']), \
                mock.patch('builtins.open', mock.mock_open(
                    read_data=b'python\0unrelated.py\0')):
            rb.refuse_if_other_driver()  # no other driver -> no raise


# ---------------------------------------------------------------------------
# full-stack compatibility with the operator's reporter (deployment copies)
# ---------------------------------------------------------------------------

class ReporterCompatibilityTests(unittest.TestCase):
    BATCH = 'test-batch'
    SCHED_SHA = 'a' * 64

    def test_reporting_copies_match_deployed(self):
        import subprocess
        for name, want in _DEPLOYED_SHA.items():
            deployed = pathlib.Path('/opt/qc-agent') / name
            if not deployed.is_file():
                self.skipTest(f'/opt/qc-agent/{name} absent')
            got = subprocess.run(['sha256sum', str(deployed)],
                                 capture_output=True, text=True).stdout
            self.assertEqual(got.split()[0], want,
                             f'{name} drifted from the copied deployment')

    def test_merged_docs_pass_operator_load_cell(self):
        effects = _load_reporting('effects_report')
        report = _load_reporting('three_way_report')
        # the copy imports the module by name; bind ours
        report.effects = effects
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            cert = base / 'reference.json'
            cert.write_text(json.dumps({
                'package_root': str(base / 'official-1.8.1'),
                'version': '1.8.1'}))
            cert_hash = hashlib.sha256(cert.read_bytes()).hexdigest()
            sources = {}
            for role, seconds in [('upstream', 12.), ('optimized_fp64', 4.),
                                  ('optimized_auto', 2.),
                                  ('strict_fp64', 6.)]:
                samples = []
                for b in range(4):
                    rec = _synthetic_record(
                        role, b, self.BATCH, self.SCHED_SHA,
                        lane='auto' if role == 'optimized_auto' else 'fp64')
                    for r in rec['row']['result']['repeats']:
                        r.update(sample_s=seconds, t_scf_s=seconds / 2,
                                 t_grad_s=seconds / 4)
                    if role == 'optimized_auto':
                        rec['row']['result']['repeats'][0][
                            'gradient_mode'] = 'fp32'
                        for r in rec['row']['result']['repeats']:
                            r['grad_ip1_hits'] = 3
                    rec['provenance']['gpu4pyscf_package_root'] = (
                        str(base / 'official-1.8.1' / 'gpu4pyscf')
                        if role == 'upstream' else str(base / 'snap'
                                                       / 'gpu4pyscf'))
                    rec['_path'] = str(base / f'{role}_{b}.json')
                    samples.append(rec)
                doc = uc.merge_role(samples, role, self.BATCH,
                                    {'batch_id': self.BATCH,
                                     'settings': dict(us.SETTINGS),
                                     'cells': [{'index': 0, 'tag': 'c',
                                                'blocks': [['O', 'B', 'C',
                                                            'R']]}]},
                                    self.SCHED_SHA)
                doc['meta']['reference_manifest_sha256'] = cert_hash
                if role == 'upstream':
                    doc['meta']['official_verification'] = {'x': 1}
                out = base / f'cell_{role}.json'
                out.write_text(json.dumps(doc))
                sources[role] = {'path': str(out),
                                 'sha256': hashlib.sha256(
                                     out.read_bytes()).hexdigest()}
            with mock.patch.object(report, 'REFERENCE', cert), \
                    mock.patch.object(effects, 'TASKS', base):
                rows = report.load_cell({'sources': sources}, self.BATCH)
                ratios = {label: effects.measure(rows[cand], rows[ref])
                          for label, ref, cand in report.COMPARISONS}
            self.assertEqual(set(rows), set(uc.ROLES))
            # the three ratios with the expected direction: O/B=3.0,
            # B/C=2.0, O/C=6.0 for the 12/4/2-second fixtures
            for label, speed in (('原版 → 优化基准', 3.0),
                                ('优化基准 → auto', 2.0),
                                ('原版 → auto', 6.0)):
                self.assertEqual(ratios[label]['n'], 4)
                self.assertAlmostEqual(ratios[label]['speed'], speed)
            self.assertTrue(all(r['passed'] for r in ratios.values()))


class AdjudicationFactsTests(unittest.TestCase):
    '''upstream_adjudicate: reporter-exact facts (load_cell/measure), P1-P6
    aggregates, Latin-square position effects, fail-closed incompleteness.'''

    BATCH = 'adj-batch'
    SCHED_SHA = 'c' * 64
    BLOCKS = [['C', 'B', 'R', 'O'], ['R', 'O', 'C', 'B'],
              ['B', 'R', 'O', 'C'], ['O', 'C', 'B', 'R']]

    def _build(self, tmp, cells, empty_last=False):
        # cells: list of (tag, xc, {role: seconds}); build schedule + docs
        # + index with the same recipe the reporter test proves load_cell
        # accepts (official cert mocked per test below)
        base = pathlib.Path(tmp)
        cert = base / 'reference.json'
        cert.write_text(json.dumps({
            'package_root': str(base / 'official-1.8.1'),
            'version': '1.8.1'}))
        cert_hash = hashlib.sha256(cert.read_bytes()).hexdigest()
        schedule = {'batch_id': self.BATCH, 'seed': 20261018,
                    'settings': dict(us.SETTINGS), 'cells': []}
        index_cells = []
        for i, (tag, xc, seconds) in enumerate(cells):
            schedule['cells'].append(
                {'index': i, 'tag': tag, 'basis': 'def2-svp',
                 'xc': xc, 'charge': 0, 'spin': 0,
                 'blocks': [list(b) for b in self.BLOCKS]})
            sources = {}
            for role in uc.ROLES:
                samples = []
                for b in range(4):
                    rec = _synthetic_record(
                        role, b, self.BATCH, self.SCHED_SHA,
                        lane='auto' if role == 'optimized_auto'
                        else 'fp64')
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
                        str(base / 'official-1.8.1' / 'gpu4pyscf')
                        if role == 'upstream'
                        else str(base / 'snap' / 'gpu4pyscf'))
                    rec['_path'] = str(
                        base / f'{tag}_{role}_b{b}.json')
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
                    'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}
            index_cells.append(
                {'cell': {'index': i, 'tag': tag, 'xc': xc,
                          'basis': 'def2-svp', 'charge': 0, 'spin': 0},
                 'sources': ({}
                             if (empty_last and i == len(cells) - 1)
                             else sources)})
        index = base / 'index.json'
        index.write_text(json.dumps(
            {'batches': [{'id': self.BATCH, 'cells': index_cells}]}))
        sched_path = base / 'schedule.json'
        sched_path.write_text(json.dumps(schedule))
        return str(index), str(sched_path), cert

    def _facts(self, tmp, cells, **gather_kw):
        index, schedule, cert = self._build(tmp, cells)
        with mock.patch.object(adj.report, 'REFERENCE', cert), \
                mock.patch.object(adj.effects, 'TASKS',
                                  pathlib.Path(tmp)):
            return adj.gather(index, self.BATCH, schedule), cert

    def test_facts_and_aggregates(self):
        # O=12s, B=4s, C=2s, R=6s -> O/B=3.0, B/C=2.0, O/C=6.0, C/R=1/3
        sec = {'upstream': 12., 'optimized_fp64': 4.,
               'optimized_auto': 2., 'strict_fp64': 6.}
        with tempfile.TemporaryDirectory() as tmp:
            facts, _ = self._facts(
                tmp, [('mol_b3lyp', 'b3lyp', sec),
                      ('mol_wb97mv', 'wb97mv', sec)])
            self.assertEqual(facts['cells_total'], 2)
            self.assertEqual(facts['cells_complete'], 2)
            agg = facts['aggregates']
            self.assertEqual(agg['n_complete'], 2)
            for med in (agg['P2_O_over_B']['medians'],
                        agg['P2_O_over_B']['passed_only_medians']):
                self.assertAlmostEqual(med['b3lyp'], 3.0)
                self.assertAlmostEqual(med['wb97mv'], 3.0)
            self.assertAlmostEqual(
                agg['P3_B_over_C']['medians']['b3lyp'], 2.0)
            self.assertAlmostEqual(
                agg['P3_B_over_C']['C_over_R_medians']['b3lyp'], 2. / 6.)
            self.assertAlmostEqual(
                agg['P4_O_over_C']['medians']['wb97mv'], 6.0)
            # band membership is mechanical and honest: b3lyp 3.0 is
            # OUTSIDE its predicted 1.2-2.0 band; wb97mv 3.0 is INSIDE
            # its >= 2.5 band; B/C 2.0 is OUTSIDE the 0.85-1.25 parity
            self.assertEqual(
                agg['P2_O_over_B']['band_counts']['b3lyp'],
                {'n': 1, 'inside_predicted': 0, 'min': 3.0, 'max': 3.0})
            self.assertEqual(
                agg['P2_O_over_B']['band_counts']['wb97mv']
                ['inside_predicted'], 1)
            self.assertEqual(
                agg['P3_B_over_C']['band_counts']['b3lyp']
                ['inside_predicted'], 0)
            self.assertEqual(
                agg['P4_O_over_C']['band_counts']['wb97mv']
                ['inside_predicted'], 1)  # 6.0 >= 3.0 (NLC class)
            self.assertEqual(
                agg['P4_O_over_C']['band_counts']['b3lyp']
                ['inside_predicted'], 0)  # 6.0 > 2.5 (non-NLC class)
            # P1: identical synthetic e/gradients across roles
            self.assertEqual(agg['P1_O_vs_R']['max_dE'], 0.0)
            self.assertEqual(agg['P1_O_vs_R']['max_dg'], 0.0)
            self.assertEqual(
                agg['P1_O_vs_R']['dE_classes']['<=1e-9'], 2)
            # P5: equal samples per role -> spreads 1.0, position
            # effects 1.0 at every Latin-square position
            for role in uc.ROLES:
                self.assertAlmostEqual(
                    agg['P5_protocol']['median_role_spread'][role], 1.0)
                for pos, value in (agg['P5_protocol']
                                   ['position_effects'][role].items()):
                    self.assertAlmostEqual(value, 1.0)
            # P6 honesty counts
            p6 = agg['P6_honesty']
            self.assertEqual(p6['strict_fp32_zero_cells'], 2)
            self.assertEqual(p6['auto_fp32_nonzero_cells'], 2)
            self.assertEqual(p6['o_fp32_empty_cells'], 2)
            self.assertEqual(p6['all_role_gates_passed_cells'], 2)
            self.assertEqual(p6['denominator'], 2)
            self.assertEqual(p6['failures_kept'], [])

    def test_incomplete_fails_closed_and_partial_optin(self):
        sec = {'upstream': 12., 'optimized_fp64': 4.,
               'optimized_auto': 2., 'strict_fp64': 6.}
        with tempfile.TemporaryDirectory() as tmp:
            index, schedule, cert = self._build(
                tmp, [('mol_b3lyp', 'b3lyp', sec),
                      ('mol_wb97mv', 'wb97mv', sec)], empty_last=True)
            out = os.path.join(tmp, 'facts.json')
            md = os.path.join(tmp, 'facts.md')
            with mock.patch.object(adj.report, 'REFERENCE', cert), \
                    mock.patch.object(adj.effects, 'TASKS',
                                      pathlib.Path(tmp)):
                with self.assertRaises(SystemExit):
                    adj.main(['--index', index, '--batch-id', self.BATCH,
                              '--schedule', schedule, '--out', out])
                self.assertFalse(os.path.exists(out))
                rc = adj.main(['--index', index, '--batch-id', self.BATCH,
                               '--schedule', schedule, '--out', out,
                               '--md', md, '--allow-partial'])
                self.assertEqual(rc, 0)
            facts = json.loads(open(out).read())
            self.assertTrue(facts['interim'])
            self.assertEqual(facts['cells_complete'], 1)
            # a not-yet-run cell is PLANNED, not a load failure
            self.assertEqual(facts['cells_planned_not_run'],
                             ['mol_wb97mv'])
            self.assertEqual(facts['cells_load_failed'], [])
            md_text = open(md).read()
            self.assertIn('INTERIM', md_text)
            self.assertIn('planned, not yet run', md_text)

    def test_broken_latin_square_fails_closed(self):
        sec = {'upstream': 12., 'optimized_fp64': 4.,
               'optimized_auto': 2., 'strict_fp64': 6.}
        with tempfile.TemporaryDirectory() as tmp:
            index, schedule, cert = self._build(
                tmp, [('mol_b3lyp', 'b3lyp', sec)])
            sched = json.loads(open(schedule).read())
            sched['cells'][0]['blocks'] = [['O', 'B', 'C', 'R']] * 4
            with open(schedule, 'w') as f:
                json.dump(sched, f)
            with mock.patch.object(adj.report, 'REFERENCE', cert), \
                    mock.patch.object(adj.effects, 'TASKS',
                                      pathlib.Path(tmp)):
                with self.assertRaises(ValueError):
                    adj.gather(index, self.BATCH, schedule)

    def test_index_must_prelist_schedule_denominator(self):
        sec = {'upstream': 12., 'optimized_fp64': 4.,
               'optimized_auto': 2., 'strict_fp64': 6.}
        with tempfile.TemporaryDirectory() as tmp:
            index, schedule, cert = self._build(
                tmp, [('mol_b3lyp', 'b3lyp', sec)])
            idx = json.loads(open(index).read())
            idx['batches'][0]['cells'] = idx['batches'][0]['cells'][:0]
            with open(index, 'w') as f:
                json.dump(idx, f)
            with mock.patch.object(adj.report, 'REFERENCE', cert), \
                    mock.patch.object(adj.effects, 'TASKS',
                                      pathlib.Path(tmp)):
                with self.assertRaises(SystemExit):
                    adj.gather(index, self.BATCH, schedule)


if __name__ == '__main__':
    unittest.main()

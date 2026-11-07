"""Unit tests for the issue-#9 hotspot attribution document validator.

The GPU profiler itself needs a CUDA device; these tests exercise only the
stdlib validate_attribution() contract (same discipline as the
rtx4090-energy-v1 validator tests): a compliant synthetic document passes,
and each rule violation is individually rejected.
"""
import copy
import importlib.util
import json
import pathlib
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'
_spec = importlib.util.spec_from_file_location(
    'hotspot_profiler', _DIR / 'hotspot_profiler.py')
hp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hp)


def _lane(kernel_s=1.0, grid_s=0.5, unattr_frac=0.05, lane='fp64',
          hits=None, e_diff=0.0, warmup=True):
    return {
        'lane': lane,
        'warmup_s': 0.8,
        'energy': {
            'e_tot': [-1.0],
            'e_tot_baseline': [-1.0 - e_diff],
            'converged': True,
            'cycles': [9],
            'precision_switch': None,
            'fp32_entry_hits': hits,
        },
        'phases': {
            'mol_s': 0.01, 'cderi_s': 0.1, 'kernel_s': kernel_s,
            'n_samples_instr': 1, 'n_samples_baseline': 1,
            'baseline_wall_s': 1.1, 'instrumented_wall_s': 1.2,
            'sync_serialisation_overhead': 1.09,
        },
        'functions': {
            'scf.xc.grid': {'t_s': grid_s, 'calls': 10},
            'scf.jk.both': {'t_s': 0.1, 'calls': 10},
            'scf.eig': {'t_s': 0.05, 'calls': 9},
            'scf.diis_fock': {'t_s': 0.02, 'calls': 18},
            'scf.rdm_occ': {'t_s': 0.02, 'calls': 18},
            'scf.grad_norm': {'t_s': 0.01, 'calls': 9},
            'scf.energy': {'t_s': 0.01, 'calls': 10},
            'scf.init': {'t_s': 0.1, 'calls': 3},
            'scf.orth': {'t_s': 0.01, 'calls': 1},
            'scf.chkfile': {'t_s': 0.01, 'calls': 9},
            **({'scf.nlc': {'t_s': 0.1, 'calls': 10},
                'scf.jk.k': {'t_s': 0.05, 'calls': 10}} if lane == 'fp64'
               else {}),
            **({'disp.d4': {'t_s': 0.05, 'calls': 1},
                'disp.gcp': {'t_s': 0.05, 'calls': 1}}
               if lane == 'fp64' else {}),
        },
        'unattributed': {
            't_s': round(kernel_s * unattr_frac, 6),
            'frac_of_kernel': unattr_frac,
            'within_10pct': unattr_frac <= 0.10,
        },
        'grid_internal': {
            'total_s': grid_s,
            'ao_eval_s': grid_s * 0.2, 'rho_s': grid_s * 0.3,
            'libxc_s': grid_s * 0.1, 'other_s': grid_s * 0.4,
            'note': 'x',
        },
        'notes': [],
    }


def _doc(cells=None):
    if cells is None:
        cells = [{
            'row_key': {'geom': 'benzene.xyz', 'basis': 'def2-svp',
                        'xc': 'b3lyp'},
            'pilot_stage': 'S1', 'nao': 114, 'spin': 0,
            'lanes': {
                'fp64': _lane(lane='fp64'),
                'fp32': _lane(lane='fp32',
                              hits={'df_jk': 10, 'eval_rho': 100,
                                    'xc_grid': 10, 'grad_ip1': 0,
                                    'other': 0}),
            },
            'notes': [],
        }]
    return {
        'schema_version': 'hotspot-attribution-v1',
        'meta': {
            'gpu_name': 'NVIDIA GeForce RTX 4090',
            'gpu4pyscf_source_commit': 'a' * 40,
            'gpu4pyscf_source_path': '/data/work/gpu4pyscf-mixedprec',
            'timing_discipline': 'device-synchronised wrappers',
            'unattributed_max_frac': 0.10,
        },
        'cells': cells,
    }


class ValidateAttributionTests(unittest.TestCase):

    def check(self, doc, rule):
        ok, issues = hp.validate_attribution(doc)
        self.assertFalse(ok)
        rules = {i['rule'] for i in issues}
        self.assertIn(rule, rules, f'expected {rule}, got {issues}')

    def test_compliant_document_passes(self):
        ok, issues = hp.validate_attribution(_doc())
        self.assertTrue(ok, issues)

    def test_h1_wrong_schema_version(self):
        doc = _doc()
        doc['schema_version'] = 'something-else'
        self.check(doc, 'H1')

    def test_h1_missing_meta_provenance(self):
        doc = _doc()
        del doc['meta']['gpu4pyscf_source_commit']
        self.check(doc, 'H1')

    def test_h1_max_frac_ceiling(self):
        doc = _doc()
        doc['meta']['unattributed_max_frac'] = 0.5
        self.check(doc, 'H1')

    def test_h2_lane_without_phases(self):
        doc = _doc()
        del doc['cells'][0]['lanes']['fp64']['phases']['kernel_s']
        self.check(doc, 'H2')

    def test_h2_missing_warmup(self):
        doc = _doc()
        del doc['cells'][0]['lanes']['fp64']['warmup_s']
        self.check(doc, 'H2')

    def test_h3_unattributed_exceeds_10pct(self):
        doc = _doc()
        lane = doc['cells'][0]['lanes']['fp64']
        lane['unattributed'] = {'t_s': 0.5, 'frac_of_kernel': 0.5,
                                'within_10pct': False}
        self.check(doc, 'H3')

    def test_h3_flag_lies_about_fraction(self):
        doc = _doc()
        lane = doc['cells'][0]['lanes']['fp64']
        lane['unattributed']['within_10pct'] = False  # frac still 0.05
        self.check(doc, 'H3')

    def test_h4_grid_internals_do_not_add_up(self):
        doc = _doc()
        lane = doc['cells'][0]['lanes']['fp64']
        lane['grid_internal']['other_s'] = 0.0
        # parts now 0.6 * grid vs total grid -> 40% off
        self.check(doc, 'H4')

    def test_h5_fp32_lane_without_hits(self):
        doc = _doc()
        doc['cells'][0]['lanes']['fp32']['energy']['fp32_entry_hits'] = {
            'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0,
            'other': 0}
        self.check(doc, 'H5')

    def test_h5_fp32_lane_without_fp64_reference(self):
        doc = _doc()
        del doc['cells'][0]['lanes']['fp64']
        self.check(doc, 'H5')

    def test_h6_3c_cell_missing_dispersion(self):
        doc = _doc()
        cell = doc['cells'][0]
        cell['row_key'] = {'geom': 'benzene.xyz', 'basis': 'def2-mtzvpp',
                           'xc': 'r2scan3c'}
        for lane in cell['lanes'].values():
            for k in ('disp.d4', 'disp.gcp', 'disp.d4+gcp'):
                lane['functions'].pop(k, None)
        self.check(doc, 'H6')

    def test_h6_nlc_cell_missing_nlc(self):
        doc = _doc()
        doc['cells'][0]['row_key']['xc'] = 'wb97m-v'
        for lane in doc['cells'][0]['lanes'].values():
            lane['functions'].pop('scf.nlc', None)
        self.check(doc, 'H6')

    def test_h7_instrumentation_altered_energy(self):
        doc = _doc()
        doc['cells'][0]['lanes']['fp64']['energy']['e_tot_baseline'] = [-1.001]
        self.check(doc, 'H7')

    def test_h7_nonfinite_energy(self):
        doc = _doc()
        doc['cells'][0]['lanes']['fp64']['energy']['e_tot'] = [float('inf')]
        self.check(doc, 'H7')

    def test_live_smoke_document_validates(self):
        '''The real profiler output (methanol/B3LYP smoke, GPU-run) validates
        through the same code path used for acceptance.'''
        path = pathlib.Path(
            '/data/results/gpu4pyscf-agent/tasks/issue-9/'
            'hotspot_smoke.json')
        if not path.exists():
            self.skipTest('smoke document not generated in this environment')
        with open(path, encoding='utf-8') as f:
            doc = json.load(f)
        ok, issues = hp.validate_attribution(doc)
        self.assertTrue(ok, issues)

    def test_validator_is_pure_stdlib_importable(self):
        # module-level import must not require CUDA (lazy cupy)
        self.assertTrue(callable(hp.validate_attribution))


if __name__ == '__main__':
    unittest.main()

"""Adversarial unit tests for the G16 curation gate (issue #4).

The bundled G16 tests are candidates, NOT usable references by default.
These tests attack the classifier with exactly the trap shapes the plan
warns about (plan section 3.6.2): MD routes hiding a single SCF, stability
opt jobs with multiple SCF records, multi-job Link1 inputs, error-only
terminations, oversized AO, and an index that disagrees with the raw log.
The machine must stay conservative: the best outcome is 'untriaged'
(provisional), and nothing may look reference_matched without the manual
audit.
"""

import importlib.util
import json
import os
import pathlib
import tempfile
import unittest

_DIR = pathlib.Path(__file__).parents[2] / 'mixedprec' / 'benchmark' / 'rtx4090'

_spec = importlib.util.spec_from_file_location(
    'g16_curation', _DIR / 'g16_curation.py')
g16 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g16)


def _log(routes, scf=('-115.7348716828283',), normal=True, error=False,
         charge_mult=(0, 1), nbasis=94, extra=''):
    '''Build a minimal synthetic G16 log.'''
    body = [' Entering Gaussian System']
    for r in routes:
        body.append(f' {r}')
    body.append(' -------------------')
    body.append(f' Charge = {charge_mult[0]} Multiplicity = {charge_mult[1]}')
    body.append(f' NBasis = {nbasis}')
    for label, e in zip(['RHF'] * len(scf), scf):
        body.append(f' SCF Done:  E({label}) = {e}     A.U. after 10 cycles')
    if normal:
        body.append(' Normal termination of Gaussian 16.')
    if error:
        body.append(' Error termination via Lnk1e.')
    body.append(extra)
    return '\n'.join(body) + '\n'


class RouteSemanticsTests(unittest.TestCase):
    def _cls(self, routes, **kw):
        with tempfile.NamedTemporaryFile('w', suffix='.log',
                                         delete=False) as f:
            f.write(_log(routes, **kw))
            path = f.name
        try:
            data = g16.parse_g16_log(path)
            return g16.classify_case(routes, data, ao_cap=120) + (data,)
        finally:
            os.unlink(path)

    def test_md_route_excluded_even_with_single_scf(self):
        # the historical trap: one SCF Done + normal termination inside an
        # MD route is NOT a single-point reference
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g* md=test'])
        self.assertEqual(cls, 'deferred')
        self.assertTrue(any('molecular dynamics' in r for r in reasons))

    def test_opt_job_deferred(self):
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g* opt'])
        self.assertEqual(cls, 'deferred')
        self.assertTrue(any('optimization' in r for r in reasons))

    def test_stability_job_with_two_scf_deferred(self):
        cls, reasons, _ = self._cls(
            ['#P b3lyp/6-31g* stable=(opt,qconly)'],
            scf=('-1417.30966897', '-1417.31143633'))
        self.assertEqual(cls, 'deferred')

    def test_single_point_b3lyp_is_provisional_only(self):
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g* test int=ultrafine 5d'])
        self.assertEqual(cls, 'untriaged')
        self.assertTrue(any('manual' in r.lower() for r in reasons))

    def test_m06_2x_token_matched(self):
        cls, _r, data = self._cls(['#P m06-2x/def2-svp'])
        self.assertEqual(cls, 'untriaged')
        self.assertEqual(data['scf_done'][0]['energy'], -115.7348716828283)

    def test_non_target_method_rejected(self):
        cls, reasons, _ = self._cls(['#P mp2/6-31g*'])
        self.assertEqual(cls, 'rejected')
        self.assertTrue(any('no target method' in r for r in reasons))

    def test_error_only_termination_rejected(self):
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g*'], normal=False,
                                    error=True)
        self.assertEqual(cls, 'rejected')

    def test_multi_scf_deferred(self):
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g*'],
                                    scf=('-1.0', '-1.1'))
        self.assertEqual(cls, 'deferred')
        self.assertTrue(any('SCF results' in r for r in reasons))

    def test_ao_cap_deferred(self):
        cls, reasons, _ = self._cls(['#P b3lyp/6-31g*'], nbasis=400)
        self.assertEqual(cls, 'deferred')
        self.assertTrue(any('AO cap' in r for r in reasons))

    def test_unreadable_log_deferred(self):
        cls = g16.classify_case(['#P b3lyp/6-31g*'], None, ao_cap=120)[0]
        self.assertEqual(cls, 'deferred')

    def test_missing_log_deferred(self):
        with tempfile.TemporaryDirectory() as td:
            missing = os.path.join(td, 'nope.log')
            self.assertIsNone(g16.parse_g16_log(missing))


class ReadPrecomputedTests(unittest.TestCase):
    def _case(self, **kw):
        with tempfile.NamedTemporaryFile('w', suffix='.log',
                                         delete=False) as f:
            f.write(_log(['#P b3lyp/6-31g*'], **kw))
            return {'case': 'test', 'log': f.name}

    def test_expectations_ok(self):
        case = self._case()
        try:
            data = g16.read_precomputed(
                case, {'normal_termination': True, 'n_scf': 1})
            self.assertEqual(len(data['scf_done']), 1)
        finally:
            os.unlink(case['log'])

    def test_expectation_violation_raises(self):
        case = self._case(scf=('-1.0', '-1.1'))
        try:
            with self.assertRaises(ValueError):
                g16.read_precomputed(case, {'n_scf': 1})
        finally:
            os.unlink(case['log'])

    def test_missing_log_raises(self):
        with self.assertRaises(ValueError):
            g16.read_precomputed({'case': 'x', 'log': '/nope.log'})


class IndexVerificationTests(unittest.TestCase):
    def test_log_wins_over_index(self):
        # adversarial: the index claims 1 normal termination and one SCF
        # energy, but the raw log says otherwise
        with tempfile.NamedTemporaryFile('w', suffix='.log',
                                         delete=False) as f:
            f.write(_log(['#P b3lyp/6-31g*'], scf=('-1.5', '-1.6')))
            path = f.name
        try:
            entry = {'normal_terminations': 1,
                     'has_error_termination': False,
                     'scf_results': [{'energy_hartree': -1.5}],
                     'methods': ['B3LYP'], 'nbasis': 94}
            problems = g16.verify_entry_against_log(entry,
                                                    g16.parse_g16_log(path))
            self.assertTrue(any('scf_results count' in p
                                for p in problems))
            # fix the count but keep the wrong energy
            entry['scf_results'] = [{'energy_hartree': -1.5},
                                    {'energy_hartree': -9.9}]
            problems = g16.verify_entry_against_log(entry,
                                                    g16.parse_g16_log(path))
            self.assertTrue(any('energy' in p for p in problems))
        finally:
            os.unlink(path)

    def test_consistent_index_reports_nothing(self):
        with tempfile.NamedTemporaryFile('w', suffix='.log',
                                         delete=False) as f:
            f.write(_log(['#P b3lyp/6-31g*']))
            path = f.name
        try:
            entry = {'normal_terminations': 1,
                     'has_error_termination': False,
                     'scf_results': [{'energy_hartree':
                                      -115.7348716828283}],
                     'methods': ['B3LYP'], 'nbasis': 94}
            self.assertEqual(
                g16.verify_entry_against_log(entry,
                                             g16.parse_g16_log(path)), [])
        finally:
            os.unlink(path)

    def test_method_mismatch_detected(self):
        with tempfile.NamedTemporaryFile('w', suffix='.log',
                                         delete=False) as f:
            f.write(_log(['#P b3lyp/6-31g*']))
            path = f.name
        try:
            entry = {'methods': ['M06-2X']}
            problems = g16.verify_entry_against_log(entry,
                                                    g16.parse_g16_log(path))
            self.assertTrue(any('methods' in p for p in problems))
        finally:
            os.unlink(path)


class CurateTests(unittest.TestCase):
    def test_full_pass_classifications(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, 'good.log')
            with open(good, 'w') as f:
                f.write(_log(['#P b3lyp/6-31g* test int=ultrafine 5d']))
            md = os.path.join(td, 'md.log')
            with open(md, 'w') as f:
                f.write(_log(['#P b3lyp/6-31g* md']))
            with open(good, 'rb') as f:
                good_sha = g16.sha256_file(good)
            entries = [
                {'case': 't1', 'input': '/x/t1.com', 'routes':
                 ['#P b3lyp/6-31g*'], 'methods': ['B3LYP'], 'log': good,
                 'log_sha256': good_sha, 'normal_terminations': 1,
                 'has_error_termination': False,
                 'scf_results': [{'label': 'RB3LYP',
                                  'energy_hartree': -115.7348716828283}],
                 'nbasis': 94},
                {'case': 't2', 'input': '/x/t2.com', 'routes':
                 ['#P b3lyp/6-31g* md'], 'methods': ['B3LYP'], 'log': md},
                {'case': 't3', 'input': '/x/t3.com', 'routes':
                 ['#P mp2/6-31g*'], 'methods': [], 'log': None},
            ]
            doc = g16.curate(entries, ao_cap=120)
            by_case = {c['case']: c for c in doc['cases']}
            self.assertEqual(by_case['t1']['classification'], 'untriaged')
            self.assertEqual(by_case['t2']['classification'], 'deferred')
            self.assertEqual(by_case['t3']['classification'], 'rejected')
            self.assertEqual(doc['counts']['by_classification'],
                             {'untriaged': 1, 'deferred': 1, 'rejected': 1})
            self.assertEqual(doc['counts']['provisional_candidates'], 1)
            self.assertTrue(by_case['t1']['log_sha256_verified'])
            self.assertEqual(by_case['t1']['machine_admission'],
                             'provisional')
            self.assertEqual(by_case['t1']['manual_audit'], 'pending')

    def test_wrong_index_hash_flagged(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, 'good.log')
            with open(good, 'w') as f:
                f.write(_log(['#P b3lyp/6-31g*']))
            entries = [{'case': 't1', 'routes': ['#P b3lyp/6-31g*'],
                        'methods': ['B3LYP'], 'log': good,
                        'log_sha256': '0' * 64}]
            doc = g16.curate(entries)
            self.assertFalse(doc['cases'][0]['log_sha256_verified'])

    def test_machine_never_says_reference_matched(self):
        with tempfile.TemporaryDirectory() as td:
            good = os.path.join(td, 'good.log')
            with open(good, 'w') as f:
                f.write(_log(['#P b3lyp/6-31g*']))
            entries = [{'case': 't1', 'routes': ['#P b3lyp/6-31g*'],
                        'methods': ['B3LYP'], 'log': good}]
            doc = g16.curate(entries)
            cls = doc['cases'][0]['classification']
            self.assertIn(cls, ('untriaged', 'deferred'))
            self.assertNotEqual(cls, 'reference_matched')

    def test_g16_energy_only_on_untriaged(self):
        # a deferred case (MD) must not even carry a reference energy that
        # something downstream might attach
        with tempfile.TemporaryDirectory() as td:
            md = os.path.join(td, 'md.log')
            with open(md, 'w') as f:
                f.write(_log(['#P b3lyp/6-31g* md']))
            entries = [{'case': 't1', 'routes': ['#P b3lyp/6-31g* md'],
                        'methods': ['B3LYP'], 'log': md}]
            doc = g16.curate(entries)
            self.assertIsNone(doc['cases'][0].get('g16_energy_hartree'))


class MethodsInRoutesTests(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(
            g16.methods_in_routes(['#P B3LYP/6-31G*']), ['B3LYP'])
        self.assertEqual(
            g16.methods_in_routes(['#P M06-2X/def2-svp']), ['M06-2X'])
        self.assertEqual(
            g16.methods_in_routes(['#P wb97m-v/def2-tzvp']),
            ['omegaB97M-V'])
        self.assertEqual(g16.methods_in_routes(['#P hf/sto-3g']), [])
        # r2SCAN-3c is a whole method; plain r2SCAN must NOT match it
        self.assertEqual(g16.methods_in_routes(['#P r2scan/def2-svp']), [])
        self.assertEqual(
            g16.methods_in_routes(['#P r2scan-3c']), ['r2SCAN-3c'])


if __name__ == '__main__':
    unittest.main()

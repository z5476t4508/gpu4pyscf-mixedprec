'''CPU-only stdlib tests for fd_contract_guard (SNT2-GRIDRESP-RECONCILE-01).

The guard enforces the GRIDRESP-CONTRACT-01 consequence rule (effects-index
entry 265): every future FD-vs-analytic comparison must carry either an
analytic gradient with grid_response=True or an explicit term budget that
covers the measured contract-class scale (worst 1.396e-4 Eh/Bohr on
uracil/r2SCAN-3c def2-mtzvpp grids-3). No pyscf/cupy import needed: the
gradient-object wiring is exercised through a stub mf/gradient pair.
'''
import importlib.util
import json
import os
import unittest

GUARD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
        __file__)))), 'mixedprec', 'benchmark', 'rtx4090',
    'fd_contract_guard.py')


def _load():
    spec = importlib.util.spec_from_file_location('fd_contract_guard',
                                                  GUARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RequireFdContractTests(unittest.TestCase):
    def setUp(self):
        self.g = _load()

    def test_grid_response_true_branch(self):
        r = self.g.require_fd_contract(analytic_grid_response=True,
                                       source='snt3-select')
        self.assertEqual(r['schema'], 'fd-contract-guard-v1')
        self.assertEqual(r['contract'], 'grid_response_true')
        self.assertIs(r['analytic_grid_response'], True)
        self.assertIsNone(r['term_budget_eh_bohr'])
        self.assertEqual(r['source'], 'snt3-select')
        json.dumps(r)  # manifest-embeddable

    def test_fail_closed_when_neither_branch(self):
        with self.assertRaises(ValueError) as cm:
            self.g.require_fd_contract(source='x')
        self.assertIn('GRIDRESP-CONTRACT-01', str(cm.exception))

    def test_grid_response_false_is_never_valid(self):
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(analytic_grid_response=False,
                                       source='x')
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(analytic_grid_response='true',
                                       source='x')

    def test_true_plus_budget_rejected(self):
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(analytic_grid_response=True,
                                       term_budget_eh_bohr=3e-4, source='x')

    def test_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(analytic_grid_response=True, source='')

    def test_term_budget_valid_at_class_scale(self):
        r = self.g.require_fd_contract(
            term_budget_eh_bohr=self.g.MEASURED_CLASS_SCALE_EH_BOHR,
            source='snt3-compare')
        self.assertEqual(r['contract'], 'term_budget')
        self.assertIs(r['analytic_grid_response'], False)
        self.assertEqual(r['term_budget_eh_bohr'],
                         self.g.MEASURED_CLASS_SCALE_EH_BOHR)

    def test_term_budget_below_class_scale_rejected(self):
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(term_budget_eh_bohr=1e-6, source='x')

    def test_term_budget_type_and_sign(self):
        for bad in (0, -3e-4, 0.0, '3e-4', True, None):
            with self.assertRaises(ValueError):
                self.g.require_fd_contract(term_budget_eh_bohr=bad,
                                           source='x')

    def test_measured_cell_scale_pinned(self):
        r = self.g.require_fd_contract(
            term_budget_eh_bohr=2.4e-5, source='snt2-reconcile',
            measured_class_scale_eh_bohr=2.37e-5)
        self.assertEqual(r['measured_class_scale_eh_bohr'], 2.37e-5)
        self.assertEqual(r['term_budget_eh_bohr'], 2.4e-5)

    def test_measured_cell_scale_validation(self):
        for bad in (0, -1.0, 'x', True, 5e-4):
            with self.assertRaises(ValueError):
                self.g.require_fd_contract(
                    term_budget_eh_bohr=1e-4, source='x',
                    measured_class_scale_eh_bohr=bad)
        # a "measured" scale larger than the class scale is a contradiction:
        # pass it directly as the budget instead
        with self.assertRaises(ValueError):
            self.g.require_fd_contract(
                term_budget_eh_bohr=1e-3, source='x',
                measured_class_scale_eh_bohr=5e-4)

    def test_nonfinite_budgets_rejected(self):
        # external-guard-review-20261104 finding 2: float('inf') used to be
        # accepted as a term budget (vacuously covering the contract class)
        for bad in (float('inf'), float('-inf'), float('nan')):
            with self.assertRaises(ValueError) as cm:
                self.g.require_fd_contract(term_budget_eh_bohr=bad,
                                           source='x')
            self.assertIn('finite', str(cm.exception))

    def test_nonfinite_measured_scale_rejected(self):
        # inf used to fail only via the misleading "exceeds the class scale,
        # pass it as the budget instead" message; nan via "must be positive"
        for bad in (float('inf'), float('-inf'), float('nan')):
            with self.assertRaises(ValueError) as cm:
                self.g.require_fd_contract(
                    term_budget_eh_bohr=1e-3, source='x',
                    measured_class_scale_eh_bohr=bad)
            self.assertIn('finite', str(cm.exception))


class _StubGrad:
    def __init__(self):
        self.grid_response = False


class _StubMf:
    def nuc_grad_method(self):
        return _StubGrad()


class _RuntimeTrueGrad:
    grid_response = True  # config override / pre-configured factory


class _StubMfReturningTrueGrad:
    def nuc_grad_method(self):
        return _RuntimeTrueGrad()


class _DropsAssignmentGrad:
    """Accepts the grid_response assignment but silently ignores it."""
    def __init__(self):
        self._grid_response = False

    @property
    def grid_response(self):
        return self._grid_response

    @grid_response.setter
    def grid_response(self, value):
        pass


class _StubMfReturningDropGrad:
    def nuc_grad_method(self):
        return _DropsAssignmentGrad()


class GradientWithContractTests(unittest.TestCase):
    def setUp(self):
        self.g = _load()

    def test_true_wiring_sets_flag(self):
        gobj, rec = self.g.gradient_with_contract(
            _StubMf(), grid_response=True, source='drv')
        self.assertIs(gobj.grid_response, True)
        self.assertEqual(rec['contract'], 'grid_response_true')
        self.assertIs(rec['runtime_grid_response'], True)
        json.dumps(rec)  # manifest-embeddable

    def test_budget_branch_keeps_default_flag(self):
        gobj, rec = self.g.gradient_with_contract(
            _StubMf(), term_budget_eh_bohr=self.g
            .MEASURED_CLASS_SCALE_EH_BOHR, source='drv')
        self.assertIs(gobj.grid_response, False)
        self.assertEqual(rec['contract'], 'term_budget')
        self.assertIs(rec['runtime_grid_response'], False)
        json.dumps(rec)

    def test_fail_closed_passthrough(self):
        with self.assertRaises(ValueError):
            self.g.gradient_with_contract(_StubMf(), source='drv')
        with self.assertRaises(ValueError):
            self.g.gradient_with_contract(_StubMf(), grid_response=False,
                                          source='drv')

    def test_explicit_false_with_budget_rejected(self):
        # external-guard-review-20261104 finding 1, review spelling:
        # grid_response=False + budget used to be silently reinterpreted as
        # the term-budget branch, returning a runtime-True object together
        # with a record claiming analytic_grid_response=False
        with self.assertRaises(ValueError) as cm:
            self.g.gradient_with_contract(
                _StubMfReturningTrueGrad(), grid_response=False,
                term_budget_eh_bohr=1.5e-4, source='drv')
        self.assertIn('grid_response', str(cm.exception))
        with self.assertRaises(ValueError):
            self.g.gradient_with_contract(
                _StubMf(), grid_response=False,
                term_budget_eh_bohr=self.g.MEASURED_CLASS_SCALE_EH_BOHR,
                source='drv')

    def test_budget_branch_rejects_runtime_true_object(self):
        # config override or pre-configured gradient factory handing back
        # grid_response=True under a term-budget declaration is false
        # provenance: the record would claim the term is omitted while the
        # object actually computes it
        with self.assertRaises(ValueError) as cm:
            self.g.gradient_with_contract(
                _StubMfReturningTrueGrad(),
                term_budget_eh_bohr=self.g.MEASURED_CLASS_SCALE_EH_BOHR,
                source='drv')
        self.assertIn('contradicts the declared term-budget contract',
                      str(cm.exception))

    def test_true_branch_rejects_object_dropping_assignment(self):
        # an object that accepts but silently ignores grid_response=True
        # cannot be certified as including the grid response
        with self.assertRaises(ValueError) as cm:
            self.g.gradient_with_contract(
                _StubMfReturningDropGrad(), grid_response=True, source='drv')
        self.assertIn('did not end up with grid_response=True',
                      str(cm.exception))

    def test_true_branch_on_runtime_true_object_agrees(self):
        gobj, rec = self.g.gradient_with_contract(
            _StubMfReturningTrueGrad(), grid_response=True, source='drv')
        self.assertIs(gobj.grid_response, True)
        self.assertIs(rec['analytic_grid_response'], True)
        self.assertIs(rec['runtime_grid_response'], True)

    def test_grid_response_invalid_values_rejected(self):
        for bad in (1, 0, 'true', [True], float('nan')):
            with self.assertRaises(ValueError) as cm:
                self.g.gradient_with_contract(
                    _StubMf(), grid_response=bad,
                    term_budget_eh_bohr=self.g.MEASURED_CLASS_SCALE_EH_BOHR,
                    source='drv')
            self.assertIn('accepts True', str(cm.exception))


if __name__ == '__main__':
    unittest.main()

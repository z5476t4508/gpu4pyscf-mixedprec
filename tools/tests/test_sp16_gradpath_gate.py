#!/usr/bin/env python3
"""SP-16 regression tests: the size-gated gradient precision
(gpu4pyscf/grad/rhf.py::_resolve_precision; scf-priority-20261025 card
item 3; named by SP-15's P3 attribution + SP-10's measured falsification).

Measured motivation: below _AUTO_HEAD_MIN_AO the plain-auto SCF is
bitwise the fp64 lane (SP-14), but the gradient still resolved 'auto' ->
'fp32' unconditionally (the geomeTRIC 3e-4 rationale), leaving the LC-88
dtype-shim fp32 entries in the gradient path: measured t_grad C/B
0.99-1.15 (largest where the hit count is largest: caffeine 40 hits,
1.153) and dg-vs-B 5.4e-8..8.6e-7, while SP-10 measured the fp32
grad casts NOT cheaper at 42-204 AO (NLC piece 0.86-1.04 vs a <=0.50
bar; GEMM-bound-shape-only).  The change: in the plain-'auto' branch of
_resolve_precision only, nao below the SAME _AUTO_HEAD_MIN_AO constant
resolves 'fp64' -- the gated auto lane becomes bitwise the fp64 lane in
e_tot AND gradient.  auto+verify/auto+strict keep fp32 gradients (their
contracts select solutions, not kernel precision); explicit overrides
and every nao >= 250 lane are untouched.

Contracts pinned here:
  * source: the gate lives inside _resolve_precision's plain-'auto'
    branch, imports the constant from gpu4pyscf.scf.hf (one source of
    truth), and touches no other branch;
  * semantics (CPU construction, no SCF run): below-gate plain auto
    resolves 'fp64'; threshold-lowered above-gate auto still resolves
    'fp32'; auto+verify/auto+strict still 'fp32' at every size; explicit
    grad precision_mode overrides win over the gate;
  * bitwise (GPU, subprocess-per-lane): a small-cell RKS auto lane's
    e_tot AND full gradient are EXACTLY the explicit-fp64 lane's
    (subprocess pairs avoid the within-process plan-cache reduction
    drift, the LC-96 5.7e-14 class).
"""
import os
import re
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

GRAD_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'rhf.py')
GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'benzene.xyz')
PY = sys.executable


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\n    def |\nclass |\Z)', src, re.S)
    return m.group(0) if m else None


def _clean_gpu_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith('GPU4PYSCF_')}


class TestSourceContractGradGate(unittest.TestCase):
    def test_gate_in_resolve_precision_auto_branch(self):
        src = _func_src(GRAD_SRC, '_resolve_precision')
        self.assertIsNotNone(src)
        self.assertIn(
            "if mode == 'auto':\n            mode = 'fp32'", src)
        # the SP-16 gate: same constant as the SCF-side gate, plain-auto
        # branch only
        self.assertIn('from gpu4pyscf.scf.hf import _AUTO_HEAD_MIN_AO',
                      src)
        self.assertIn('self.mol.nao < _AUTO_HEAD_MIN_AO', src)

    def test_verify_strict_branches_untouched(self):
        src = _func_src(GRAD_SRC, '_resolve_precision')
        self.assertIn("elif mode == 'auto+verify':", src)
        self.assertIn("elif mode == 'auto+strict':", src)
        # the gate text must NOT appear inside those branches: count
        # occurrences of the gate predicate -- exactly one (the plain
        # auto branch)
        self.assertEqual(src.count('self.mol.nao < _AUTO_HEAD_MIN_AO'), 1)


class TestGradGateSemantics(unittest.TestCase):
    '''_resolve_precision lane table (CPU construction, no SCF run).
    benzene/def2-svp = 114 AO < 250: the gated side.'''

    def _resolve(self, base_mode, grad_mode=None, threshold=None):
        import gpu4pyscf.scf.hf as hf
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        saved = getattr(hf, '_AUTO_HEAD_MIN_AO', None)
        try:
            if threshold is not None:
                if saved is None:
                    self.fail('gate constant absent (pristine source)')
                hf._AUTO_HEAD_MIN_AO = threshold
            mol = gto.M(atom=GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='b3lyp')
            if base_mode is not None:
                mf.precision_mode = base_mode
            g = mf.Gradients()
            if grad_mode is not None:
                g.precision_mode = grad_mode
            return g._resolve_precision()
        finally:
            if threshold is not None and saved is not None:
                hf._AUTO_HEAD_MIN_AO = saved

    def test_below_gate_auto_resolves_fp64(self):
        self.assertEqual(self._resolve('auto'), 'fp64',
                         'below the gate the plain-auto gradient must '
                         'ride the exact fp64 lane')

    def test_threshold_lowered_auto_still_fp32(self):
        # nao 114 with the threshold at 60 = above-gate: today's behavior
        self.assertEqual(self._resolve('auto', threshold=60), 'fp32')

    def test_verify_strict_keep_fp32_below_gate(self):
        self.assertEqual(self._resolve('auto+verify'), 'fp32')
        self.assertEqual(self._resolve('auto+strict'), 'fp32')

    def test_explicit_overrides_win(self):
        self.assertEqual(self._resolve('auto', grad_mode='fp32'), 'fp32')
        self.assertEqual(self._resolve('auto', grad_mode='fp64'), 'fp64')

    def test_default_none_stays_none(self):
        self.assertIsNone(self._resolve(None))


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestGradGateBitwise(unittest.TestCase):
    '''e_tot bitwise AND gradient envelope equality auto-vs-fp64 on a
    gated cell; subprocess-per-lane (the LC-96 plan-cache class).

    Measured during SP-16 red-green: the gradient is NOT bitwise
    reproducible across processes even fp64-vs-fp64 (max |dg| 1.78e-14
    on benzene/svp/b3lyp, 27/36 elements, while e_tot IS bitwise) --
    a cross-process reduction-order floor, so the honest gradient
    contract is ENVELOPE equality: the auto-vs-fp64 delta must sit in
    the same 1e-14 class as the in-test fp64-vs-fp64 control pair
    (<= 5x the control's max delta, and absolutely <= 1e-12; the
    pre-change dtype-shim signature was 1e-7-class, four orders
    above this band).'''

    CHILD = r'''
import sys
sys.path.insert(0, %(repo)r)
from pyscf import gto
from gpu4pyscf.dft import RKS
mode = sys.argv[1]
mol = gto.M(atom=%(geom)r, basis='def2-svp', verbose=0)
mf = RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = mode
mf.kernel()
g = mf.Gradients().kernel()
import numpy as np
print('E_TOT %%.17g' %% mf.e_tot, 'CONV', mf.converged)
print('GRAD ' + ' '.join('%%.17g' %% x for x in np.asarray(g).ravel()))
'''

    def _lane(self, script, mode):
        r = subprocess.run(
            [PY, '-c', script, mode],
            capture_output=True, text=True, timeout=600,
            env=_clean_gpu_env())
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        lines = r.stdout.strip().splitlines()
        e = lines[-2].split()[1]
        g = [float(x) for x in lines[-1].split()[1:]]
        return lines[-2], e, g

    def test_small_rks_auto_grad_equals_fp64_envelope(self):
        script = self.CHILD % {'repo': REPO, 'geom': GEOM}
        _, e_auto, g_auto = self._lane(script, 'auto')
        h_fp64, e_fp64, g_fp64 = self._lane(script, 'fp64')
        _, e_fp64b, g_fp64b = self._lane(script, 'fp64')  # control pair
        self.assertIn('CONV True', h_fp64)
        self.assertEqual(e_auto, e_fp64,
                         'e_tot must stay bitwise equal (SP-14 contract)')
        self.assertEqual(e_fp64, e_fp64b,
                         'fp64 e_tot cross-process bitwise (MR-12 class)')
        ctrl = max(abs(a - b) for a, b in zip(g_fp64, g_fp64b))
        d = max(abs(a - b) for a, b in zip(g_auto, g_fp64))
        self.assertLessEqual(d, max(5 * ctrl, 1e-14),
                             f'auto-vs-fp64 gradient delta {d:.3e} must '
                             f'sit in the fp64-vs-fp64 envelope {ctrl:.3e}'
                             f' (5x band); the dtype-shim signature is '
                             f'1e-7-class')
        self.assertLessEqual(d, 1e-12)


if __name__ == '__main__':
    unittest.main()

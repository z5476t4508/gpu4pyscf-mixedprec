#!/usr/bin/env python3
"""Regression tests for the LC-22 gradient-side XC grid-block wiring
(gpu4pyscf/grad/rks.py + gpu4pyscf/grad/uks.py, GPU4PYSCF_XC_GRID_BLOCKSIZE).

The change adds:
  * `loop_blksize` (derived via numint._effective_scf_blksize /
    _scf_grid_blksize — the LC-11 SCF-side knob) to the gradient XC
    loops: _get_exc_task LDA/GGA/MGGA loops (grad/rks.py and
    grad/uks.py) and get_nlc_exc's two VV10 loops (grad/rks.py, shared
    by UKS);
  * all loop-scratch buffers (aow_buf, rho_buf) sized to loop_blksize
    instead of the module MIN_BLK_SIZE.

Defaults are untouched: with the env var unset the effective block size
is MIN_BLK_SIZE and every buffer keeps its old size.  These tests are
CPU-only source-contract tests plus GPU numerical tests on a small cell
(default == old behavior; opt-in 16384 gradient matches default at the
same-orbital floor).
"""
import os
import re
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

RKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'rks.py')
UKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'grad', 'uks.py')


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestGradRksWiring(unittest.TestCase):
    def test_get_exc_task_derives_loop_blksize(self):
        src = _func_src(RKS_SRC, '_get_exc_task')
        self.assertIsNotNone(src)
        self.assertIn('loop_blksize = numint._effective_scf_blksize(',
                      src)
        self.assertIn('numint._scf_grid_blksize()', src)

    def test_get_exc_task_all_three_loops_wired(self):
        src = _func_src(RKS_SRC, '_get_exc_task')
        self.assertEqual(src.count('loop_blksize=loop_blksize'), 3)
        self.assertEqual(src.count('grid_range=(grid_start, grid_end)'), 3)

    def test_get_exc_task_buffers_scale(self):
        src = _func_src(RKS_SRC, '_get_exc_task')
        for pattern in (
                r'aow_buf = cupy\.empty\(loop_blksize \* max\(nao, 1\*nocc\)\)',
                r'aow_buf = cupy\.empty\(loop_blksize \* max\(nao \* 3, '
                r'2\*nocc, 4\)\)',
                r'aow_buf = cupy\.empty\(loop_blksize \* max\(nao \* 3, '
                r'2\*nocc, 5\)\)'):
            self.assertRegex(src, pattern)

    def test_get_nlc_exc_both_loops_wired(self):
        src = _func_src(RKS_SRC, 'get_nlc_exc')
        self.assertIsNotNone(src)
        self.assertEqual(src.count('loop_blksize=loop_blksize'), 2)
        self.assertIn('loop_blksize = numint._effective_scf_blksize(', src)


class TestGradUksWiring(unittest.TestCase):
    def test_get_exc_task_derives_loop_blksize(self):
        src = _func_src(UKS_SRC, '_get_exc_task')
        self.assertIsNotNone(src)
        self.assertIn('loop_blksize = numint._effective_scf_blksize(', src)

    def test_get_exc_task_all_three_loops_wired(self):
        src = _func_src(UKS_SRC, '_get_exc_task')
        self.assertEqual(src.count('loop_blksize=loop_blksize'), 3)

    def test_buffers_scale(self):
        src = _func_src(UKS_SRC, '_get_exc_task')
        self.assertIn('rho_buf = cupy.empty(2*ncomp*loop_blksize)', src)
        self.assertEqual(
            src.count('aow_buf = cupy.empty(loop_blksize * max(nao * 3, '
                      '2*nocc))'), 2)


class TestOtherGradLoopsUntouched(unittest.TestCase):
    '''The Hessian/response-style loops (get_exc_full_response,
    _get_denlc pieces at module bottom) keep their strict_grid_order
    default blocks.'''

    def test_response_loops_not_wired(self):
        src = open(RKS_SRC).read()
        m = re.search(r'def get_exc_full_response\(.*?(?=\ndef |\Z)',
                      src, re.S)
        self.assertIsNotNone(m)
        self.assertNotIn('loop_blksize', m.group(0))


class TestGpuNumericalEquivalence(unittest.TestCase):
    '''Small-cell numerical check: default and opt-in 16384 gradients
    match on identical orbitals; default path still runs.'''

    def test_gradient_default_vs_16384(self):
        import cupy
        import pyscf
        from gpu4pyscf import dft as gpu_dft
        import gpu4pyscf.dft.numint as gnumint

        mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                      verbose=0)
        mf = gpu_dft.RKS(mol, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 2
        mf.verbose = 0
        mf.kernel()
        self.assertTrue(mf.converged)

        stream = cupy.cuda.get_current_stream()

        def grad():
            stream.synchronize()
            g = mf.nuc_grad_method().kernel()
            stream.synchronize()
            return cupy.asnumpy(g)

        gnumint._XC_GRID_BLKSIZE = None
        os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
        mf.grids._non0ao_idx = None
        g_def = grad()

        gnumint._XC_GRID_BLKSIZE = None
        os.environ['GPU4PYSCF_XC_GRID_BLOCKSIZE'] = '16384'
        try:
            mf.grids._non0ao_idx = None
            g_16k = grad()
        finally:
            os.environ.pop('GPU4PYSCF_XC_GRID_BLOCKSIZE', None)
            gnumint._XC_GRID_BLKSIZE = None
        import numpy as np
        self.assertLess(float(np.abs(g_16k - g_def).max()), 1e-9)


if __name__ == '__main__':
    unittest.main()

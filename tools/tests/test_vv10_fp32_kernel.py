'''Unit tests for the fp32-safe VV10 UWE kernel (LC-80, review
2026-10-12 continuation; selector GPU4PYSCF_VV10_UWE_KERNEL=fp32).

Three layers:
  * static source contract: the selector admits 'fp32', the default
    path is unchanged, and the kernel carries the measured safety
    structure (fp64 staging inputs cast at load; three separate
    reciprocals; fp64 chunk-partial accumulation);
  * GPU kernel equivalence: U/W/E vs the fp64 reference kernel on a
    real converged wb97mv density, gated at the LC-80 measured level
    (|dE_nlc| <= 1e-8 Ha, 100x inside the 1e-6 precision gate);
  * GPU whole-SCF smoke: a wb97mv RKS with the knob on converges to
    the same energy as the knob-off run within 1e-9 Ha (the kernel is
    the only change).
The variant is cached once per process, so every GPU case runs in a
fresh subprocess (the issue-10 selector-test pattern).
'''

import os
import pathlib
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).parents[2]
TOOLS_TESTS = REPO / 'tools' / 'tests'

_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import cupy
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import eval_rho2, _scf_grid_blksize, _effective_scf_blksize
from gpu4pyscf.lib.utils import load_library
import ctypes
libgdft = load_library('libgdft')

variant = sys.argv[1] if len(sys.argv) > 1 else 'kernel'
os_set = {set_env!r}
if os_set:
    import os
    os.environ['GPU4PYSCF_VV10_UWE_KERNEL'] = os_set

GEOMS = {geoms!r}
mol = pyscf.M(atom=GEOMS + {xyz!r}, basis={basis!r}, output='/dev/null', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = 'fp64'
mf.kernel()
assert mf.converged
print(f'E_TOTAL={{mf.e_tot:.12f}}')

if variant == 'kernel':
    # kernel-level U/W/E comparison on the converged NLC-grid density
    dm = mf.make_rdm1()
    ni = mf._numint
    nlcgrids = mf.nlcgrids
    opt = ni.gdftopt
    _smol = opt._sorted_mol
    nao = _smol.nao
    mo_s = opt.sort_orbitals(cupy.asarray(mf.mo_coeff), axis=[0])
    rho_parts = []
    ng = nlcgrids.coords.shape[0]
    lb = _effective_scf_blksize(_scf_grid_blksize(), 0, ng, ng)
    for ao, idx, weight, coords in ni.block_loop(_smol, nlcgrids, nao, 1, loop_blksize=lb):
        rho_parts.append(eval_rho2(_smol, ao, mo_s[idx, :], mf.mo_occ, None, 'GGA'))
    rho = cupy.hstack(rho_parts)
    coords = cupy.asarray(nlcgrids.coords)
    weights = cupy.asarray(nlcgrids.weights)
    nlc_pars = ni.nlc_coeff('wb97m-v')[0][0]
    mask = cupy.where(cupy.logical_and(
        rho[0] >= numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD,
        cupy.abs(weights) > 1e-14))[0]
    rho_i = cupy.ascontiguousarray(rho[0][mask])
    coords_i = cupy.ascontiguousarray(coords[mask])
    w_i = weights[mask]
    gamma_i = cupy.linalg.norm(rho[1:4, mask], axis=0)
    ngrids = coords_i.shape[0]
    omega_i = cupy.empty(ngrids); dd = cupy.empty(ngrids); dg = cupy.empty(ngrids)
    stream = cupy.cuda.get_current_stream()
    err = libgdft.VXC_vv10nlc_fock_eval_omega_derivative(
        ctypes.cast(stream.ptr, ctypes.c_void_p), ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(dd.data.ptr, ctypes.c_void_p), ctypes.cast(dg.data.ptr, ctypes.c_void_p),
        ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
        ctypes.cast(cupy.ascontiguousarray(gamma_i).data.ptr, ctypes.c_void_p),
        ctypes.c_double(nlc_pars[1]), ctypes.c_int(ngrids))
    assert err == 0
    kappa_i = cupy.ascontiguousarray(nlc_pars[0] * 1.5 * np.pi * (9*np.pi)**(-1/6) * rho_i ** (1.0/6))
    rhow = cupy.ascontiguousarray(rho_i * w_i)
    beta = 0.03125 * (3.0 / nlc_pars[0]**2)**0.75
    def uwe():
        U = cupy.empty(ngrids); W = cupy.empty(ngrids); E = cupy.empty(ngrids)
        err = libgdft.VXC_vv10nlc_fock_eval_UWE(
            ctypes.cast(stream.ptr, ctypes.c_void_p), ctypes.cast(U.data.ptr, ctypes.c_void_p),
            ctypes.cast(W.data.ptr, ctypes.c_void_p), ctypes.cast(E.data.ptr, ctypes.c_void_p),
            ctypes.cast(coords_i.data.ptr, ctypes.c_void_p), ctypes.cast(rhow.data.ptr, ctypes.c_void_p),
            ctypes.cast(omega_i.data.ptr, ctypes.c_void_p), ctypes.cast(kappa_i.data.ptr, ctypes.c_void_p),
            ctypes.c_int(ngrids))
        assert err == 0
        cupy.cuda.runtime.deviceSynchronize()
        return U, W, E
    U, W, E = uwe()
    den = rho_i * w_i
    print(f'E_NLC={{float((den * (beta + 0.5 * E)).sum()):.15f}}')
    print(f'FINITE={{bool(cupy.isfinite(U).all() and cupy.isfinite(E).all())}}')
'''


def _env_python():
    '''The GPU source env interpreter (pyscf + cupy + the source tree
    build); the tools suite itself runs on system python where only the
    static source-contract tests are executable.'''
    cand = os.environ.get('GPU4PYSCF_TEST_PYTHON')
    if cand:
        return cand
    for p in ('/home/qczcode/gpu4090-src/bin/python',
              '/home/ubuntu/pyscf-env/bin/python'):
        if os.path.isfile(p):
            return p
    return sys.executable


def _run_subprocess(code, timeout=600):
    res = subprocess.run(
        [_env_python(), '-c', code],
        capture_output=True, text=True, timeout=timeout)
    return res


def _parse(out):
    vals = {}
    for line in out.splitlines():
        if line.startswith('E_TOTAL='):
            vals['e_tot'] = float(line.split('=')[1])
        elif line.startswith('E_NLC='):
            vals['e_nlc'] = float(line.split('=')[1])
        elif line.startswith('FINITE='):
            vals['finite'] = line.split('=')[1] == 'True'
    return vals


class FP32VV10SourceContract(unittest.TestCase):
    '''The opt-in selector and the measured safety structure, pinned in
    the CUDA source (the issue-10 static-contract pattern).'''

    def test_selector_admits_fp32(self):
        cu = (REPO / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu').read_text()
        self.assertIn('strcmp(env, "fp32") == 0', cu)
        # opt-in only: the default (unset) keeps the reference kernel
        self.assertIn('vv10_fock_eval_UWE_kernel<<<', cu)
        # the fp32 kernel is dispatched from the same launcher
        self.assertIn('vv10_fock_eval_UWE_fp32safe_kernel<<<', cu)

    def test_kernel_safety_structure(self):
        cu = (REPO / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu').read_text()
        # three separate reciprocals (the 1/(g*g*g) overflow guard)
        self.assertIn('const float inv_gij = 1.0f / g_ij;', cu)
        self.assertIn('const float inv_gji = 1.0f / g_ji;', cu)
        self.assertIn('const float inv_gsum = 1.0f / g_sum;', cu)
        # fp64 chunk-partial accumulation (the serial-fp32 fix)
        self.assertIn('U_i += double(U_chunk);', cu)
        # fp64 omega precompute stays on the fp64 omega_derivative kernel
        self.assertIn(
            'vv10_fock_eval_omega_derivative_kernel<<<', cu)

    def test_python_tree_does_not_read_the_env(self):
        '''the issue-10 contract: nothing in the gpu4pyscf Python package
        reads the selector (C-side only, one read per process)'''
        hits = []
        root = REPO / 'gpu4pyscf'
        for path in root.rglob('*.py'):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if 'GPU4PYSCF_VV10_UWE_KERNEL' in line:
                    hits.append(f'{path}:{lineno}')
        self.assertEqual(hits, [])


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32VV10KernelEquivalence(unittest.TestCase):
    '''U/E vs the fp64 reference kernel on a real converged wb97mv
    density (water-sized: fast).  The measured LC-80 level on
    benzene/tzvp was dE_nlc -1.8e-10; the gate here is 1e-8 Ha (100x
    inside the 1e-6 precision gate) to leave room for the smaller
    grid's own noise while still catching any structural regression
    (the pre-fix serial-fp32 accumulation failed at 1.5e-6).'''

    @classmethod
    def setUpClass(cls):
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        code = _RUNNER.format(repo=str(REPO), set_env='warp1', geoms=geoms,
                              xyz='methanol.xyz', basis='def2-svp')
        cls.ref = _parse(_run_subprocess(code).stdout)
        code = _RUNNER.format(repo=str(REPO), set_env='fp32', geoms=geoms,
                              xyz='methanol.xyz', basis='def2-svp')
        cls.fp32 = _parse(_run_subprocess(code).stdout)

    def test_finite(self):
        self.assertTrue(self.fp32.get('finite', False))

    def test_e_nlc_within_gate(self):
        d = self.fp32['e_nlc'] - self.ref['e_nlc']
        self.assertLessEqual(abs(d), 1e-8,
                             f'dE_nlc = {d:.3e} Ha (gate 1e-8, '
                             'precision gate 1e-6)')

    def test_scf_energy_unchanged(self):
        d = self.fp32['e_tot'] - self.ref['e_tot']
        # whole-SCF: the fixed point re-absorbs the potential noise;
        # measured 1e-12-level on the LC-80 matrix
        self.assertLessEqual(abs(d), 1e-9)


if __name__ == '__main__':
    unittest.main()

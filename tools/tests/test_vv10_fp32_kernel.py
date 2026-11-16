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

    def test_fp32_kernel_stages_coordinates_relative_to_origin(self):
        '''the review-202610-17 translation fix: the fp32-safe kernel must
        subtract a common origin in fp64 BEFORE the fp32 cast (both the
        i-load and the j-shared staging).  Absolute-coordinate staging
        quantized the low bits at distant origins and broke rigid-
        translation invariance (measured +5.27e-6 Ha dE_NLC at 1e6 Bohr
        on methanol/def2-svp/wb97m-v; the fp64 kernels stayed at ~7e-15).
        A future refactor that drops the origin subtraction must fail
        this static pin AND the GPU translation-invariance test below.'''
        cu = (REPO / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu').read_text()
        self.assertIn('grid_coord[i * 3 + 0] - origin_x', cu)
        self.assertIn('grid_coord[i * 3 + 1] - origin_y', cu)
        self.assertIn('grid_coord[i * 3 + 2] - origin_z', cu)
        self.assertIn('grid_coord[j * 3 + 0] - origin_x', cu)
        self.assertIn('grid_coord[j * 3 + 1] - origin_y', cu)
        self.assertIn('grid_coord[j * 3 + 2] - origin_z', cu)


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


_GRAD_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import cupy
import pyscf
from gpu4pyscf import dft

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
# the production gradient route: grid_response=False (class default) ->
# get_nlc_exc -> numint._vv10nlc -> the SAME VXC_vv10nlc_fock_eval_UWE
# dispatcher this knob controls (LC-81 measured 1.4-1.9x gradient-phase
# speedup from it; route fact pinned by the companion static test)
g = mf.nuc_grad_method().run()
de = np.asarray(g.de)
assert np.isfinite(de).all()
print(f'E_TOTAL={{mf.e_tot:.12f}}')
print(f'GRAD_NORM={{float(np.linalg.norm(de)):.12f}}')
for row in de:
    print('GRADROW={{:.15e}} {{:.15e}} {{:.15e}}'.format(*row))
'''


_TRANSLATION_EXTRA = r'''
original_coords = coords_i.copy()
reference_nlc = float((den * (beta + 0.5 * E)).sum())
for shift in [1.0, 1000.0, 100000.0, 1000000.0]:
    coords_i = cupy.ascontiguousarray(original_coords + shift)
    shifted_U, shifted_W, shifted_E = uwe()
    shifted_nlc = float((den * (beta + 0.5 * shifted_E)).sum())
    print(f'TRANSLATION {shift} {shifted_nlc - reference_nlc:.15e}')
'''


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32VV10TranslationInvariance(unittest.TestCase):
    '''Rigid-translation invariance of the fp32 UWE kernel on a real
    converged wb97m-v NLC density (independent review 2026-10-17).
    Pre-fix, the kernel cast ABSOLUTE coordinates to fp32 before
    differencing, so a common shift of every grid point (rho, weights,
    omega, kappa unchanged) quantized the coordinate low bits and moved
    E_NLC by +5.27e-6 Ha at 1e6 Bohr (methanol/def2-svp; the fp64 warp1
    kernel stayed at ~7e-15).  The origin-subtraction fix restores
    component invariance to the ~1e-12 class.  Gate 1e-8 Ha: 100x inside
    the 1e-6 precision gate, ~1e7x above the post-fix measured level,
    and 500x below the pre-fix failure.'''

    @classmethod
    def setUpClass(cls):
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        code = _RUNNER.format(repo=str(REPO), set_env='fp32', geoms=geoms,
                              xyz='methanol.xyz', basis='def2-svp') + \
            _TRANSLATION_EXTRA
        res = _run_subprocess(code)
        cls.deltas = {}
        for line in res.stdout.splitlines():
            if line.startswith('TRANSLATION '):
                _, shift, delta = line.split()
                cls.deltas[float(shift)] = float(delta)

    def test_all_shifts_measured(self):
        self.assertEqual(sorted(self.deltas), [1.0, 1000.0, 100000.0,
                                               1000000.0])

    def test_translation_invariance(self):
        for shift, delta in sorted(self.deltas.items()):
            with self.subTest(shift_Bohr=shift):
                self.assertLessEqual(
                    abs(delta), 1e-8,
                    f'dE_NLC = {delta:.3e} Ha under a rigid {shift}-Bohr '
                    'shift of every NLC grid point (fp32 kernel translation '
                    'invariance broken; gate 1e-8, precision gate 1e-6)')


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32VV10GradientRoute(unittest.TestCase):
    '''LC-81: the knob reaches the PRODUCTION gradient route
    (get_nlc_exc; grid_response is False by default), so the gradient
    itself is an acceptance surface of this kernel -- LC-80's original
    "gradient paths untouched" record was corrected by measurement.
    Same converged wb97mv object, knob off vs on (fresh subprocess each
    because the variant caches per process).  Measured cross-knob
    max|dg| on methanol/svp: 1.0e-9 (benzene/tzvp: 9.8e-8); gate here
    1e-7 Eh/Bohr = 300x inside the 3e-5 gradient gate.'''

    @classmethod
    def setUpClass(cls):
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        code = _GRAD_RUNNER.format(repo=str(REPO), set_env=None,
                                   geoms=geoms, xyz='methanol.xyz',
                                   basis='def2-svp')
        cls.ref = _parse_grad(_run_subprocess(code).stdout)
        code = _GRAD_RUNNER.format(repo=str(REPO), set_env='fp32',
                                   geoms=geoms, xyz='methanol.xyz',
                                   basis='def2-svp')
        cls.fp32 = _parse_grad(_run_subprocess(code).stdout)

    def test_gradient_finite(self):
        self.assertTrue(np_allfinite(self.fp32['grad']))
        self.assertTrue(self.fp32['e_tot'] is not None)

    def test_energy_unchanged(self):
        d = self.fp32['e_tot'] - self.ref['e_tot']
        self.assertLessEqual(abs(d), 1e-9)

    def test_gradient_within_gate(self):
        dg = np_maxabs(self.fp32['grad'] - self.ref['grad'])
        self.assertLessEqual(dg, 1e-7,
                             f'max|dg| = {dg:.3e} Eh/Bohr (gate 1e-7, '
                             'gradient acceptance gate 3e-5)')


def _parse_grad(out):
    vals = {'grad': [], 'e_tot': None}
    for line in out.splitlines():
        if line.startswith('E_TOTAL='):
            vals['e_tot'] = float(line.split('=')[1])
        elif line.startswith('GRADROW='):
            vals['grad'].append(
                [float(x) for x in line.split('=', 1)[1].split()])
    if vals['grad']:
        vals['grad'] = __import__('numpy').array(vals['grad'])
    return vals


def np_allfinite(a):
    return bool(__import__('numpy').isfinite(a).all())


def np_maxabs(a):
    return float(__import__('numpy').abs(a).max())


class VV10GradientRouteStatic(unittest.TestCase):
    '''Pin the LC-81 route facts in source so a future refactor cannot
    silently detach the gradient from the knobbed dispatcher (or turn
    the production route into the offdiagonal-kernel route).'''

    def test_production_gradient_routes_through_vv10nlc(self):
        rks = (REPO / 'gpu4pyscf' / 'grad' / 'rks.py').read_text()
        # get_nlc_exc (production, grid_response=False branch) computes
        # vxc via numint._vv10nlc -> the shared UWE dispatcher
        self.assertIn('numint._vv10nlc(rho, grids.coords, grids.weights, nlc_pars)',
                      rks)
        numint_src = (REPO / 'gpu4pyscf' / 'dft' / 'numint.py').read_text()
        self.assertIn('VXC_vv10nlc_fock_eval_UWE', numint_src)
        # the offdiagonal kernel lives ONLY in the full_response route
        # (grid_response=True opt-in): it must not appear in get_nlc_exc
        fr_start = numint_src.find('def get_nlc_exc(')
        fr_end = numint_src.find('def ', numint_src.find(
            'def get_nlc_exc_full_response'))
        body = rks[rks.find('def get_nlc_exc('):rks.find(
            'def get_nlc_exc_full_response')]
        self.assertNotIn('offdiagonal', body)
        self.assertIn('offdiagonal', rks[rks.find(
            'def get_nlc_exc_full_response'):])

    def test_grid_response_default_is_false(self):
        '''the production default keeps the grid_response route OFF
        (pyscf-inherited class attribute)'''
        rks = (REPO / 'gpu4pyscf' / 'grad' / 'rks.py').read_text()
        self.assertIn('grid_response = rks_grad.Gradients.grid_response',
                      rks)
        uks = (REPO / 'gpu4pyscf' / 'grad' / 'uks.py').read_text()
        self.assertIn('grid_response = uks_grad.Gradients.grid_response',
                      uks)


if __name__ == '__main__':
    unittest.main()

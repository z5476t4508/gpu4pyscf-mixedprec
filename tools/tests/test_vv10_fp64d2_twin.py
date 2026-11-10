'''Focused regression for the MP-461-PRECISION-MARGIN-01 opt-in VV10 UWE
fp64d2 twin (selector value fp64d2, code 48).

fp32d2 (the fp32safe template at I_TILE=128, NDIV=2) carries the
3->2-division algebraic identity AND the fp32 arithmetic discipline
together, so pairing it against the plain reference warp1 kernel mixes an
implementation change with the precision change (the 2026-11-09 user
source-fact on entry 300).  The fp64d2 kernel is the reference warp1 body
with ONLY the pair expression swapped to the 2-division identity in
double, making fp64d2<->fp32d2 a pure arithmetic-precision switch at
fixed algebra and fixed warp1 execution structure.

Layers (mirroring test_vv10_fp32tile_kernel.py, scoped to the twin):
  * static source contract: the selector admits fp64d2=48, the default
    and every pre-existing value keep their frozen dispatch, the
    reference kernel and the fp32safe template bodies are byte-unchanged,
    the launcher fails closed on unknown codes with 48 in the expected
    list, and the strict-fp64-reference override still wins;
  * GPU kernel equivalence on a real converged methanol/wb97m-v density:
    fp64d2 vs the reference warp1 kernel (override-switched in one
    process), |dE_nlc| <= 1e-13 Ha, max|dU|,|dW| <= 1e-11,
    max|dE_arr| <= 1e-12, all finite, and run-to-run bitwise determinism
    (no atomics in the warp1 structure);
  * GPU boundary: synthetic small grids (1/3/127/128/129/300/1025
    points) vs the reference kernel, relative band + finiteness;
  * GPU whole-SCF + production-gradient route in a fresh subprocess
    (env selection caches per process): e_tot within 1e-11 Ha of knob-off
    (absorbs the j3c atomicAdd ULP process-noise class ~1e-12), max|dg|
    <= 1e-8 Eh/Bohr, both converged, no unknown-selector warning on
    stderr.

Translation invariance is intentionally NOT gated here: fp64d2 keeps the
reference kernel's absolute-coordinate handling (the origin-relative
staging is an fp32-cast safety property, not part of the fp64 twin).
'''

import os
import pathlib
import re
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).parents[2]
VV10_CU = REPO / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu'

TWIN_SELECTOR = 'fp64d2'
TWIN_CODE = 48


def _env_python():
    cand = os.environ.get('GPU4PYSCF_TEST_PYTHON')
    if cand:
        return cand
    for p in ('/home/qczcode/gpu4090-src/bin/python',
              '/home/ubuntu/pyscf-env/bin/python'):
        if os.path.isfile(p):
            return p
    return sys.executable


def _run_subprocess(code, timeout=900):
    return subprocess.run([_env_python(), '-c', code],
                          capture_output=True, text=True, timeout=timeout)


# byte-pinned reference-kernel pair block (must stay unchanged)
_REFERENCE_PAIR_BLOCK = '''            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;'''

# byte-pinned fp32safe-template NDIV=2 pair block (the fp32d2 arithmetic)
_FP32D2_PAIR_BLOCK = '''                const float inv_q = 1.0f / (g_ij * g_ji);
                const float inv_gsum = 1.0f / g_sum;
                const float inv_gij = g_ji * inv_q;
                Phi_ij = -(inv_q * inv_gsum);
                E_ij = rho_weight_j * Phi_ij;
                U_ij = E_ij * (inv_gij + inv_gsum);'''

# the twin's pair block: the same expression in double, warp1 layout
_TWIN_PAIR_BLOCK = '''            const double inv_q = 1.0 / (g_ij * g_ji);
            const double inv_gsum = 1.0 / g_sum;
            const double inv_gij = g_ji * inv_q;
            const double Phi_ij = -(inv_q * inv_gsum); // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (inv_gij + inv_gsum);'''


class FP64D2SourceContract(unittest.TestCase):
    '''The opt-in selector row, the unchanged frozen paths, the twin body,
    and the fail-closed launcher, pinned in the CUDA source.'''

    @classmethod
    def setUpClass(cls):
        cls.src = VV10_CU.read_text()

    def _selector_body(self):
        m = re.search(
            r'static int vv10_uwe_kernel_variant\(\)(.*?)\n\}',
            self.src, re.S)
        self.assertIsNotNone(m, 'selector function not found')
        return m.group(1)

    def _table(self):
        body = self._selector_body()
        return dict(re.findall(
            r'strcmp\(env,\s*"([^"]+)"\)\s*==\s*0\)\s*\{\s*'
            r'cached\s*=\s*([^;]+);', body))

    def test_twin_selector_admitted_with_pinned_code(self):
        self.assertEqual(self._table().get(TWIN_SELECTOR), str(TWIN_CODE))

    def test_preexisting_values_unchanged(self):
        expect = {'warp4': '1', 'fp32': '2', 'tile8': '8', 'tile16': '16',
                  'tile32': '32', 'fp32tile8': '40', 'fp32tile16': '41',
                  'fp32tile32': '42', 'fp32d2': '43', 'fp32tile8d2': '44',
                  'fp32tile16d2': '45', 'fp32tile32d2': '46',
                  'fp32t128': '47'}
        for name, code in expect.items():
            self.assertEqual(self._table().get(name), code, name)

    def test_default_is_reference_kernel(self):
        self.assertRegex(
            self._selector_body(),
            re.compile(r'\}\s*else\s*\{.*?cached\s*=\s*0;', re.S))

    def test_twin_kernel_body_pinned(self):
        self.assertIn(_TWIN_PAIR_BLOCK, self.src,
                      'the fp64d2 twin must carry the 2-division identity '
                      'in double with the warp1 pair-block layout')
        m = re.search(
            r'static void vv10_fock_eval_UWE_fp64d2_kernel\(.*?\n\}',
            self.src, re.S)
        self.assertIsNotNone(m, 'twin kernel not found')
        body = m.group(0)
        # warp1 execution structure: one thread per i, serial chunk walk,
        # double3 shared staging, the reference write scaling
        for needle in ('const int i = blockIdx.x * blockDim.x + threadIdx.x;',
                       'for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {',
                       '__shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];',
                       'U[i] = -U_i * 1.5;',
                       'W[i] = -W_i * 1.5;',
                       'E[i] = E_i * 1.5;'):
            self.assertIn(needle, body, needle)

    def test_reference_kernel_body_byte_unchanged(self):
        self.assertIn(_REFERENCE_PAIR_BLOCK, self.src)

    def test_fp32d2_arithmetic_byte_unchanged(self):
        self.assertIn(_FP32D2_PAIR_BLOCK, self.src)

    def test_launcher_fails_closed_and_lists_48(self):
        m = re.search(
            r'vv10 UWE kernel launcher: unknown variant code %d\s*"'
            r'\s*"\(expected ([^)]*)\)', self.src)
        self.assertIsNotNone(m, 'fail-closed launcher message not found')
        self.assertIn(str(TWIN_CODE), m.group(1).split('|'))

    def test_strict_fp64_override_still_wins(self):
        # the override check precedes the env cache; strict-fp64-reference
        # contexts force code 0 so fp64d2 cannot leak into reference SCFs
        body = self._selector_body()
        ov = self.src.find('if (g_uwe_kernel_override >= 0)')
        cache = self.src.find('static int cached = -1;')
        self.assertGreater(ov, 0)
        self.assertGreater(cache, ov, 'override must be honored first')
        self.assertIn('return g_uwe_kernel_override;', body)

    def test_python_tree_does_not_read_the_env(self):
        hits = []
        root = REPO / 'gpu4pyscf'
        for path in root.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(errors='ignore').splitlines(), 1):
                if 'GPU4PYSCF_VV10_UWE_KERNEL' in line:
                    hits.append(f'{path}:{lineno}')
        self.assertEqual(hits, [])


# ---------------------------------------------------------------------------
# GPU layers: one process captures a real converged NLC density and
# switches variants through VV10_uwe_kernel_set_override (not cached);
# whole-SCF/gradient runs in a fresh subprocess.
# ---------------------------------------------------------------------------
_CAPTURE_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.utils import load_library
from gpu4pyscf.lib.cupy_helper import batched_vec_norm2
import ctypes
libgdft = load_library('libgdft')
libgdft.VV10_uwe_kernel_set_override.restype = ctypes.c_int
libgdft.VV10_uwe_kernel_set_override.argtypes = [ctypes.c_int]

mol = pyscf.M(atom={geoms!r} + {xyz!r}, basis={basis!r},
              output='/dev/null', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = 'fp64'
mf.kernel()
assert mf.converged

captured = {{}}
def recorder(rho_drho, coords, weights, nlc_pars):
    if 'rho' not in captured:
        captured['rho'] = cp.asarray(rho_drho).copy()
        captured['coords'] = cp.asarray(coords).copy()
        captured['weights'] = cp.asarray(weights).copy()
        captured['nlc_pars'] = tuple(float(x) for x in nlc_pars)
    return orig(rho_drho, coords, weights, nlc_pars)
orig = numint._vv10nlc
numint._vv10nlc = recorder
try:
    veff = mf.get_veff(mol, mf.make_rdm1())
finally:
    numint._vv10nlc = orig
del veff

THRESH = numint.NLC_REMOVE_ZERO_RHO_GRID_THRESHOLD
nlc_pars = captured['nlc_pars']
rho = captured['rho']; coords = captured['coords']
weights = captured['weights']
m = cp.where(cp.logical_and(rho[0] >= THRESH,
                            cp.abs(weights) > 1e-14))[0]
rho_i = cp.ascontiguousarray(rho[0][m])
coords_i = cp.ascontiguousarray(coords[m])
w_i = weights[m]
gamma_i = batched_vec_norm2(rho[1:4, m].T)
ngr = int(coords_i.shape[0])
omega_i = cp.empty(ngr); dd = cp.empty(ngr); dg = cp.empty(ngr)
stream = cp.cuda.get_current_stream()
err = libgdft.VXC_vv10nlc_fock_eval_omega_derivative(
    ctypes.cast(stream.ptr, ctypes.c_void_p),
    ctypes.cast(omega_i.data.ptr, ctypes.c_void_p),
    ctypes.cast(dd.data.ptr, ctypes.c_void_p),
    ctypes.cast(dg.data.ptr, ctypes.c_void_p),
    ctypes.cast(rho_i.data.ptr, ctypes.c_void_p),
    ctypes.cast(cp.ascontiguousarray(gamma_i).data.ptr, ctypes.c_void_p),
    ctypes.c_double(nlc_pars[1]), ctypes.c_int(ngr))
assert err == 0
kap_pref = nlc_pars[0] * 1.5 * np.pi * (9 * np.pi) ** (-1.0 / 6.0)
beta = 0.03125 * (3.0 / nlc_pars[0] ** 2) ** 0.75
kappa_i = cp.ascontiguousarray(kap_pref * rho_i ** (1.0 / 6.0))
rhow = cp.ascontiguousarray(rho_i * w_i)
den = rho_i * w_i

def uwe(cd, rw, om, ka, n):
    U = cp.empty(n); W = cp.empty(n); E = cp.empty(n)
    rc = libgdft.VXC_vv10nlc_fock_eval_UWE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U.data.ptr, ctypes.c_void_p),
        ctypes.cast(W.data.ptr, ctypes.c_void_p),
        ctypes.cast(E.data.ptr, ctypes.c_void_p),
        ctypes.cast(cd.data.ptr, ctypes.c_void_p),
        ctypes.cast(rw.data.ptr, ctypes.c_void_p),
        ctypes.cast(om.data.ptr, ctypes.c_void_p),
        ctypes.cast(ka.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n))
    assert rc == 0, 'UWE rc=%d' % rc
    cp.cuda.runtime.deviceSynchronize()
    return U, W, E

results = {{}}
for variant in {variants!r}:
    libgdft.VV10_uwe_kernel_set_override(variant)
    U, W, E = uwe(coords_i, rhow, omega_i, kappa_i, ngr)
    results[variant] = dict(
        U=cp.asnumpy(U), W=cp.asnumpy(W), E=cp.asnumpy(E),
        e_nlc=float((den * (beta + 0.5 * E)).sum()),
        finite=bool(cp.isfinite(U).all() and cp.isfinite(W).all()
                    and cp.isfinite(E).all()))
# run-to-run determinism of the twin (no atomics in the warp1 structure)
libgdft.VV10_uwe_kernel_set_override(48)
U2, W2, E2 = uwe(coords_i, rhow, omega_i, kappa_i, ngr)
det48 = dict(bitwiseU=bool((cp.asnumpy(U2) == results[48]['U']).all()),
             bitwiseW=bool((cp.asnumpy(W2) == results[48]['W']).all()),
             bitwiseE=bool((cp.asnumpy(E2) == results[48]['E']).all()))
libgdft.VV10_uwe_kernel_set_override(-1)
np.save({cache_path!r}, dict(results=results, det48=det48),
        allow_pickle=True)
print('CAPTURED ngrids=%d det48=%s' % (ngr, det48))
'''

_BOUNDARY_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import cupy as cp
from gpu4pyscf.lib.utils import load_library
import ctypes
libgdft = load_library('libgdft')
libgdft.VV10_uwe_kernel_set_override.restype = ctypes.c_int
libgdft.VV10_uwe_kernel_set_override.argtypes = [ctypes.c_int]
rng = np.random.default_rng(20261109)
stream = cp.cuda.get_current_stream()

def uwe(cd, rw, om, ka, n):
    U = cp.empty(n); W = cp.empty(n); E = cp.empty(n)
    rc = libgdft.VXC_vv10nlc_fock_eval_UWE(
        ctypes.cast(stream.ptr, ctypes.c_void_p),
        ctypes.cast(U.data.ptr, ctypes.c_void_p),
        ctypes.cast(W.data.ptr, ctypes.c_void_p),
        ctypes.cast(E.data.ptr, ctypes.c_void_p),
        ctypes.cast(cd.data.ptr, ctypes.c_void_p),
        ctypes.cast(rw.data.ptr, ctypes.c_void_p),
        ctypes.cast(om.data.ptr, ctypes.c_void_p),
        ctypes.cast(ka.data.ptr, ctypes.c_void_p),
        ctypes.c_int(n))
    assert rc == 0, 'UWE rc=%d' % rc
    cp.cuda.runtime.deviceSynchronize()
    return U, W, E

for ngr in {sizes!r}:
    coords = cp.ascontiguousarray(cp.asarray(rng.uniform(-6, 6, (ngr, 3))))
    omega = cp.ascontiguousarray(cp.asarray(0.005 + 0.02 * rng.random(ngr)))
    kappa = cp.ascontiguousarray(cp.asarray(0.5 + 9.0 * rng.random(ngr)))
    rhow = cp.ascontiguousarray(cp.asarray(1e-4 * rng.random(ngr)))
    libgdft.VV10_uwe_kernel_set_override(0)
    U0, W0, E0 = uwe(coords, rhow, omega, kappa, ngr)
    libgdft.VV10_uwe_kernel_set_override(48)
    U, W, E = uwe(coords, rhow, omega, kappa, ngr)
    libgdft.VV10_uwe_kernel_set_override(-1)
    # relative deviation on adversarial random inputs (order-noise class)
    for name, a, b in (('U', U, U0), ('W', W, W0), ('E', E, E0)):
        d = float(cp.abs(a - b).max().get())
        scale = float(cp.abs(b).max().get())
        assert d <= 1e-9 * max(1.0, scale), \
            'ngr=%d %s max|d|=%g scale=%g' % (ngr, name, d, scale)
    assert bool(cp.isfinite(U).all() and cp.isfinite(W).all()
                and cp.isfinite(E).all()), 'ngr=%d nonfinite' % ngr
    print('BOUNDARY_OK %d' % ngr)
'''

_SCF_GRAD_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import os
import numpy as np
import pyscf
from gpu4pyscf import dft
if {selector!r}:
    os.environ['GPU4PYSCF_VV10_UWE_KERNEL'] = {selector!r}
mol = pyscf.M(atom={geoms!r} + {xyz!r}, basis={basis!r},
              output='/dev/null', verbose=0)
mf = dft.RKS(mol, xc='wb97m-v').density_fit()
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = 'fp64'
mf.kernel()
assert mf.converged
g = mf.nuc_grad_method().run()
de = np.asarray(g.de)
assert np.isfinite(de).all()
print('E_TOTAL=%.12f' % mf.e_tot)
print('CYCLES=%d' % getattr(mf, 'cycles', -1))
for row in de:
    print('GRADROW=%.15e %.15e %.15e' % tuple(row))
'''


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP64D2KernelEquivalence(unittest.TestCase):
    '''The twin vs the reference warp1 kernel on a real converged
    methanol/wb97m-v/def2-svp NLC density (override-switched in one
    process; the reference kernel IS the comparison anchor).  Gates:
    dE_nlc <= 1e-13 Ha, max|dU|,|dW| <= 1e-11, max|dE_arr| <= 1e-12, all
    finite, bitwise run-to-run determinism (ulp-level identity noise in
    fp64; no far-pair class exists in double).'''

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix='vv64d2_equiv_')
        cls.cache = os.path.join(cls.tmp, 'results.npy')
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        code = _CAPTURE_RUNNER.format(
            repo=str(REPO), geoms=geoms, xyz='methanol.xyz',
            basis='def2-svp', variants=[0, 48],
            cache_path=cls.cache)
        res = _run_subprocess(code)
        if 'CAPTURED' not in res.stdout:
            raise AssertionError(f'capture failed: {res.stdout[-800:]} '
                                 f'{res.stderr[-800:]}')
        cls.ngrids = int(re.search(r'CAPTURED ngrids=(\d+)',
                                   res.stdout).group(1))
        import numpy as np
        blob = np.load(cls.cache, allow_pickle=True).item()
        cls.results = blob['results']
        cls.det48 = blob['det48']

    def test_capture_sane(self):
        self.assertGreater(self.ngrids, 1000)
        for v in (0, 48):
            self.assertTrue(self.results[v]['finite'], f'variant {v}')

    def test_twin_vs_reference_bands(self):
        ref, tw = self.results[0], self.results[48]
        import numpy as np
        dE = abs(tw['e_nlc'] - ref['e_nlc'])
        self.assertLessEqual(dE, 1e-13, f'dE_nlc={dE:.3e}')
        for key, band in (('U', 1e-11), ('W', 1e-11), ('E', 1e-12)):
            d = float(np.abs(tw[key] - ref[key]).max())
            self.assertLessEqual(d, band, f'max|d{key}|={d:.3e}')

    def test_twin_deterministic(self):
        self.assertTrue(self.det48['bitwiseU'])
        self.assertTrue(self.det48['bitwiseW'])
        self.assertTrue(self.det48['bitwiseE'])

    def test_twin_not_bitwise_reference(self):
        # the identity changes the rounding order: the twin must NOT be a
        # bitwise copy of the reference kernel (that would mean the
        # selector silently fell back to warp1)
        import numpy as np
        self.assertFalse(
            bool((self.results[48]['E'] == self.results[0]['E']).all()))


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP64D2Boundary(unittest.TestCase):
    def test_boundary_sizes(self):
        code = _BOUNDARY_RUNNER.format(
            repo=str(REPO), sizes=[1, 3, 127, 128, 129, 300, 1025])
        res = _run_subprocess(code, timeout=300)
        n_ok = len(re.findall(r'^BOUNDARY_OK \d+$', res.stdout, re.M))
        self.assertEqual(n_ok, 7,
                         f'boundary failures: {res.stdout[-500:]} '
                         f'{res.stderr[-500:]}')


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP64D2WholeSCFGradient(unittest.TestCase):
    '''Fresh subprocess per selector value (the env selection caches per
    process): fp64d2 vs knob-off on methanol/wb97m-v/def2-svp.'''

    @classmethod
    def setUpClass(cls):
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        runs = {}
        for tag, selector in (('off', None), ('twin', TWIN_SELECTOR)):
            code = _SCF_GRAD_RUNNER.format(
                repo=str(REPO), geoms=geoms, xyz='methanol.xyz',
                basis='def2-svp', selector=selector)
            res = _run_subprocess(code, timeout=900)
            if 'E_TOTAL=' not in res.stdout:
                raise AssertionError(f'{tag} failed: {res.stdout[-500:]} '
                                     f'{res.stderr[-500:]}')
            runs[tag] = dict(
                e=float(re.search(r'E_TOTAL=([-\d.]+)',
                                  res.stdout).group(1)),
                cycles=int(re.search(r'CYCLES=(\d+)',
                                     res.stdout).group(1)),
                grad=[[float(x) for x in
                       line.split('=')[1].split()]
                      for line in res.stdout.splitlines()
                      if line.startswith('GRADROW=')],
                stderr=res.stderr)
        cls.runs = runs

    def test_no_unknown_selector_warning(self):
        self.assertNotIn('unknown value', self.runs['twin']['stderr'])

    def test_energy_within_band(self):
        d = abs(self.runs['twin']['e'] - self.runs['off']['e'])
        self.assertLessEqual(d, 1e-11, f'|dE|={d:.3e}')

    def test_gradient_within_gate(self):
        import numpy as np
        g0 = np.asarray(self.runs['off']['grad'])
        g1 = np.asarray(self.runs['twin']['grad'])
        self.assertEqual(g0.shape, g1.shape)
        d = float(np.abs(g1 - g0).max())
        self.assertLessEqual(d, 1e-8, f'max|dg|={d:.3e}')

    def test_cycles_reported(self):
        # report-only: ulp-level identity noise must not destabilize the
        # convergence path; a persistent cycle change would be a finding
        for tag in ('off', 'twin'):
            self.assertGreater(self.runs[tag]['cycles'], 0)


if __name__ == '__main__':
    unittest.main()

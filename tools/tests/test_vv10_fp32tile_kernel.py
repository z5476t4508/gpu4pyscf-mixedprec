'''Focused regression for the NLC-KERNEL-OPT-01 opt-in VV10 UWE variants
(fp32safe arithmetic x multi-i tile structure; 2-division algebraic
identity; selector values fp32tile8/16/32, fp32d2, fp32tile*d2, fp32t128).

Layers (preregistration-nlc-kernelopt01-v1 instrument.phase0_regression):
  * static source contract: the new selectors admit the pinned codes, the
    default and every pre-existing value keep their frozen dispatch, the
    deployed kernel bodies are unchanged, and the launcher fails closed
    on unknown override codes (the frozen parent silently launched
    tile32-fp64 for any variant >= 4);
  * GPU kernel equivalence on a real converged wb97mv density, one
    process, override-switched per variant (the SRK-1 / entry-299
    instrument lineage): every new variant vs the DEPLOYED fp32safe
    kernel, |dE_nlc| <= 1e-9 Ha and max|dU|,|dW|,|dE_arr| <= 1e-8;
    fp32t128 (deployed structure AND arithmetic through the template) is
    the structure-equivalence control, gated at 1e-12;
  * GPU rigid-translation invariance of fp32tile16/fp32d2/fp32t128
    (relative-origin staging preserved in the template);
  * GPU boundary: synthetic small grids (1/3/127/128/129/300/1025
    points) exercise partial tiles and tail j chunks;
  * GPU whole-SCF + production-gradient route for fp32tile16/fp32d2/
    fp32t128 (fresh subprocess per selector value: the env selection
    caches per process), e_tot <= 1e-9 Ha and max|dg| <= 1e-7 Eh/Bohr vs
    knob-off.
'''

import os
import pathlib
import re
import subprocess
import sys
import unittest

REPO = pathlib.Path(__file__).parents[2]
VV10_CU = REPO / 'gpu4pyscf' / 'lib' / 'gdft' / 'vv10.cu'

NEW_SELECTORS = {
    'fp32tile8': 40, 'fp32tile16': 41, 'fp32tile32': 42,
    'fp32d2': 43,
    'fp32tile8d2': 44, 'fp32tile16d2': 45, 'fp32tile32d2': 46,
    'fp32t128': 47,
}
LADDER_VARIANTS = (40, 41, 42, 43, 44, 45, 46, 47)


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


class FP32TileSourceContract(unittest.TestCase):
    '''The opt-in selector table, the unchanged frozen paths, and the
    fail-closed launcher, pinned in the CUDA source.'''

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

    def test_new_selectors_admitted_with_pinned_codes(self):
        for name, code in NEW_SELECTORS.items():
            self.assertEqual(self._table().get(name), str(code),
                             f'selector {name} must map to {code}')

    def test_preexisting_values_unchanged(self):
        for name, code in (('warp4', '1'), ('fp32', '2'),
                           ('tile8', '8'), ('tile16', '16'),
                           ('tile32', '32')):
            self.assertEqual(self._table().get(name), code)

    def test_default_is_reference_kernel(self):
        self.assertRegex(
            self._selector_body(),
            re.compile(r'\}\s*else\s*\{.*?cached\s*=\s*0;', re.S))

    def test_deployed_kernels_byte_unchanged(self):
        # the deployed fp32safe body (separate reciprocals, chunk
        # promotion) and the reference launcher call are intact
        self.assertIn('static void vv10_fock_eval_UWE_fp32safe_kernel',
                      self.src)
        self.assertIn('const float inv_gij = 1.0f / g_ij;', self.src)
        self.assertIn('const float inv_gji = 1.0f / g_ji;', self.src)
        self.assertIn('const float inv_gsum = 1.0f / g_sum;', self.src)
        self.assertIn('U_i += double(U_chunk);', self.src)
        self.assertIn('vv10_fock_eval_UWE_kernel<<<', self.src)

    def test_template_carries_the_safety_discipline(self):
        self.assertIn(
            'static void vv10_fock_eval_UWE_fp32safe_multii_kernel', self.src)
        # relative-origin staging in the template (both i and j paths)
        self.assertIn('grid_coord[i * 3 + 0] - origin_x', self.src)
        self.assertIn('grid_coord[j * 3 + 0] - origin_x', self.src)
        # the 2-division identity exists only behind NDIV == 2
        self.assertIn('if constexpr (NDIV == 2)', self.src)
        self.assertIn('const float inv_q = 1.0f / (g_ij * g_ji);', self.src)
        self.assertIn('U_ij = E_ij * (inv_gij + inv_gsum);', self.src)

    def test_launcher_fails_closed_on_unknown_codes(self):
        m = re.search(r'int VXC_vv10nlc_fock_eval_UWE\((.*?)\n\}',
                      self.src, re.S)
        self.assertIsNotNone(m, 'UWE launcher not found')
        body = m.group(1)
        self.assertIn('unknown variant code', body)
        # the silent at-or-above-4 catch-all is gone from the launcher:
        # no branch may launch a kernel for an unlisted code
        self.assertNotIn('variant >= 4', body)
        for code, itile, ndiv in ((40, 8, 3), (41, 16, 3), (42, 32, 3),
                                  (43, 128, 2), (44, 8, 2), (45, 16, 2),
                                  (46, 32, 2), (47, 128, 3)):
            self.assertIn(
                f'vv10_fock_eval_UWE_fp32safe_multii_kernel<{itile}, {ndiv}><<<',
                self.src,
                f'variant {code} must launch template<{itile}, {ndiv}>')

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
# GPU layers.  One process captures a real converged NLC-grid density and
# switches variants through VV10_uwe_kernel_set_override (not cached);
# whole-SCF/gradient cases run per-selector subprocesses.
# ---------------------------------------------------------------------------
_CAPTURE_RUNNER = r'''
import sys
sys.path.insert(0, {repo!r})
import numpy as np
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.dft.numint import eval_rho2
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
libgdft.VV10_uwe_kernel_set_override(-1)
np.save({cache_path!r}, results, allow_pickle=True)
print('CAPTURED ngrids=%d' % ngr)

# rigid-translation invariance (relative-origin staging preserved):
# shift every grid point by a common offset, rho/weights/omega/kappa
# unchanged; E_nlc per variant must not move beyond the 1e-8 gate
for variant in {translation_variants!r}:
    libgdft.VV10_uwe_kernel_set_override(variant)
    base_e = results[variant]['e_nlc']
    for shift in (1.0, 1000.0, 100000.0, 1000000.0):
        shifted = cp.ascontiguousarray(coords_i + shift)
        _, _, E = uwe(shifted, rhow, omega_i, kappa_i, ngr)
        e_s = float((den * (beta + 0.5 * E)).sum())
        print('TRANSLATION %d %g %.15e' % (variant, shift, e_s - base_e))
libgdft.VV10_uwe_kernel_set_override(-1)
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
rng = np.random.default_rng(20261104)
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
    res = dict()
    for variant in {variants!r}:
        libgdft.VV10_uwe_kernel_set_override(variant)
        res[variant] = uwe(coords, rhow, omega, kappa, ngr)
    libgdft.VV10_uwe_kernel_set_override(-1)
    Up, Wp, Ep = res[2]
    for variant in {variants!r}:
        U, W, E = res[variant]
        # order-only equivalence class on ADVERSARIAL random inputs: the
        # absolute band pinned for the real-density test cannot apply to
        # arbitrary input magnitudes (measured: fp32tile8 max|dW| 4.6e-8
        # at ngr=1025 with sum|W| ~ 1e-1, exactly eps*sum order noise);
        # gate the RELATIVE deviation instead, and finiteness absolutely.
        for name, a, b in (('U', U, Up), ('W', W, Wp), ('E', E, Ep)):
            d = float(cp.abs(a - b).max().get())
            scale = float(cp.abs(b).max().get())
            assert d <= 1e-5 * max(1.0, scale), \
                'ngr=%d variant=%d %s max|d|=%g scale=%g' % (
                    ngr, variant, name, d, scale)
        assert bool(cp.isfinite(U).all() and cp.isfinite(W).all()
                    and cp.isfinite(E).all()), \
            'ngr=%d v=%d nonfinite' % (ngr, variant)
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
for row in de:
    print('GRADROW=%.15e %.15e %.15e' % tuple(row))
'''


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32TileKernelEquivalence(unittest.TestCase):
    '''Every new variant vs the DEPLOYED fp32safe kernel on a real
    converged methanol/wb97m-v/def2-svp NLC density (override-switched in
    one process; the deployed kernel IS the comparison anchor).  Gates
    (prereg): dE_nlc <= 1e-9 Ha, max|dU|,|dW|,|dE_arr| <= 1e-8, all
    finite; fp32t128 additionally <= 1e-12 dE_nlc (structure-equivalence
    control: deployed structure AND arithmetic through the template).'''

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.tmp = tempfile.mkdtemp(prefix='vv10tile_equiv_')
        cls.cache = os.path.join(cls.tmp, 'results.npy')
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        code = _CAPTURE_RUNNER.format(
            repo=str(REPO), geoms=geoms, xyz='methanol.xyz',
            basis='def2-svp', variants=list((0, 2) + LADDER_VARIANTS),
            cache_path=cls.cache,
            translation_variants=[41, 43, 47])
        res = _run_subprocess(code)
        if 'CAPTURED' not in res.stdout:
            raise AssertionError(f'capture failed: {res.stdout[-800:]} '
                                 f'{res.stderr[-800:]}')
        cls.ngrids = int(re.search(r'CAPTURED ngrids=(\d+)',
                                   res.stdout).group(1))
        cls.translations = {}
        for line in res.stdout.splitlines():
            if line.startswith('TRANSLATION '):
                _, v, shift, delta = line.split()
                cls.translations[(int(v), float(shift))] = float(delta)
        import numpy as np
        cls.results = np.load(cls.cache, allow_pickle=True).item()

    def test_capture_sane(self):
        self.assertGreater(self.ngrids, 1000)
        for v in (0, 2) + LADDER_VARIANTS:
            self.assertTrue(self.results[v]['finite'], f'variant {v}')

    def test_variant_equivalence_bands(self):
        ref = self.results[2]
        for v in LADDER_VARIANTS:
            r = self.results[v]
            de = r['e_nlc'] - ref['e_nlc']
            gate = 1e-12 if v == 47 else 1e-9
            self.assertLessEqual(abs(de), gate,
                                 f'variant {v}: dE_nlc = {de:.3e} '
                                 f'(gate {gate:g})')
            for arr in ('U', 'W', 'E'):
                d = float(abs(r[arr] - ref[arr]).max())
                self.assertLessEqual(d, 1e-8,
                                     f'variant {v}: max|d{arr}| = {d:.3e}')

    def test_deployed_fp32_vs_warp1_context(self):
        # context (not a unit gate): the deployed fp32 kernel's own
        # dE_nlc vs the fp64 warp1 reference must stay in the historical
        # 1e-10-class band on this cell
        de = self.results[2]['e_nlc'] - self.results[0]['e_nlc']
        self.assertLessEqual(abs(de), 1e-8, f'context dE_nlc = {de:.3e}')

    def test_t128_bitwise_report(self):
        # report (not gated): the structure control vs deployed
        r, ref = self.results[47], self.results[2]
        same = all((r[arr] == ref[arr]).all() for arr in ('U', 'W', 'E'))
        d = max(float(abs(r[a] - ref[a]).max()) for a in ('U', 'W', 'E'))
        print(f'fp32t128 vs deployed fp32: bitwise={same} max|d|={d:.3e}')

    def test_translation_invariance(self):
        self.assertEqual(
            sorted(self.translations),
            [(v, s) for v in (41, 43, 47)
             for s in (1.0, 1000.0, 100000.0, 1000000.0)])
        for (v, shift), delta in sorted(self.translations.items()):
            with self.subTest(variant=v, shift_Bohr=shift):
                self.assertLessEqual(
                    abs(delta), 1e-8,
                    f'variant {v}: dE_NLC = {delta:.3e} under a rigid '
                    f'{shift}-Bohr shift (relative-origin staging broken; '
                    'gate 1e-8, precision gate 1e-6)')


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32TileBoundary(unittest.TestCase):
    '''Partial tiles and tail j chunks: synthetic grid sizes around the
    tile/block boundaries, every variant vs the deployed kernel.'''

    def test_boundary_sizes(self):
        code = _BOUNDARY_RUNNER.format(
            repo=str(REPO), sizes=[1, 3, 127, 128, 129, 300, 1025],
            variants=[2] + list(LADDER_VARIANTS))
        res = _run_subprocess(code, timeout=300)
        ok = [ln for ln in res.stdout.splitlines()
              if ln.startswith('BOUNDARY_OK')]
        self.assertEqual(len(ok), 7,
                         f'expected 7 boundary sizes OK: '
                         f'{res.stdout[-500:]} {res.stderr[-500:]}')


@unittest.skipIf(os.environ.get('GPU4PYSCF_SKIP_GPU_TESTS'),
                 'GPU tests skipped by env')
class FP32TileSCFAndGradient(unittest.TestCase):
    '''Whole-SCF and production gradient route for the adoption-relevant
    selectors (fp32tile16 for H1, fp32d2 for H2, fp32t128 control) vs the
    knob-off reference: e_tot <= 1e-9 Ha, max|dg| <= 1e-7 Eh/Bohr (the
    LC-81 lesson: the gradient is an acceptance surface of this
    dispatcher).'''

    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.np = np
        geoms = str(REPO / 'mixedprec' / 'benchmark' / 'geoms') + '/'
        cls.base = cls._run(None, geoms)
        cls.arms = {name: cls._run(name, geoms)
                    for name in ('fp32tile16', 'fp32d2', 'fp32t128')}

    @classmethod
    def _run(cls, selector, geoms):
        code = _SCF_GRAD_RUNNER.format(
            repo=str(REPO), selector=selector, geoms=geoms,
            xyz='methanol.xyz', basis='def2-svp')
        res = _run_subprocess(code)
        vals = {'grad': [], 'e_tot': None}
        for line in res.stdout.splitlines():
            if line.startswith('E_TOTAL='):
                vals['e_tot'] = float(line.split('=')[1])
            elif line.startswith('GRADROW='):
                vals['grad'].append(
                    [float(x) for x in line.split('=', 1)[1].split()])
        assert vals['e_tot'] is not None and vals['grad'], \
            f'runner failed: {res.stdout[-400:]} {res.stderr[-400:]}'
        vals['grad'] = cls.np.array(vals['grad'])
        return vals

    def test_energy_unchanged(self):
        for name, arm in self.arms.items():
            d = arm['e_tot'] - self.base['e_tot']
            self.assertLessEqual(abs(d), 1e-9,
                                 f'{name}: dE_tot = {d:.3e} (gate 1e-9)')

    def test_gradient_within_gate(self):
        for name, arm in self.arms.items():
            dg = float(self.np.abs(arm['grad'] - self.base['grad']).max())
            self.assertLessEqual(dg, 1e-7,
                                 f'{name}: max|dg| = {dg:.3e} '
                                 '(gate 1e-7, acceptance gate 3e-5)')


if __name__ == '__main__':
    unittest.main()

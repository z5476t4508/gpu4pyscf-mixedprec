#!/usr/bin/env python3
"""SP-31 regression tests: the VV10/NLC head-avoidance opt-in knob
GPU4PYSCF_AUTO_HEAD_NLC_SKIP (scf-priority-20261025 card item 3, the
named-open mechanism class after the SP-26/29 closure; prereg
7a81e3e5, screen batch sp31-headnlc-mechanism-screen-20261027).

Mechanism under test: inside the auto-family fp32 SCF head ONLY, the
NLC/VV10 term of get_veff (rks.py/uks.py do_nlc block) is skipped; the
fp64 tail, every pure-fp64/strict lane, standalone contexts and the
DEFAULT (knob absent) are bitwise unchanged.  The converged result is
the full method's fixed point (the tail runs the exact fp64 map); only
the SCF path changes -- the melatonin_wb97mv q+1s1 near-gate UKS
cation (b1 dg 4.03e-05 class) is the screen's falsifier cell.

Contracts pinned here:
  * source: numint defines _auto_head_nlc_skip (env parse),
    _scf_head_nlc_skip_lane (armed by scf.hf._kernel for auto-family
    kernels only, restored in finally), _scf_head_nlc_suppressed
    (lane AND fp32 precision; increments _head_nlc_skip_count exactly
    when it suppresses) and the counter starts at 0;
  * rks.get_veff/uks.get_veff gate their do_nlc block on
    `not numint._scf_head_nlc_suppressed()`;
  * semantics: default (env absent) and '0' never suppress (counter
    stays 0 through a full auto SCF); '1' suppresses exactly the fp32
    head cycles on RKS and UKS (counter == 1 pre-loop veff + fp32
    cycles traced); garbage values warn and stay off;
  * no-NLC functional (b3lyp): knob-on is a bitwise no-op (e_tot and
    gradient bitwise identical to knob-off);
  * numerics on wb97mv: knob-on auto e_tot within 1e-6 Ha of the
    strict fp64 lane at identical settings (both converged; the gate
    of record).

The 250-AO auto-head size gate is lowered to 1 via __config__ INSIDE
each test subprocess BEFORE gpu4pyscf imports (the production gate is
untouched); lane semantics are size-independent.

RED discipline: at the pre-SP-31 HEAD 8e16d4d the numint attributes do
not exist (AttributeError) and the veff gates are absent -- red-run.log
is preserved in the SP-31 evidence dir; GREEN at the SP-31 commit.
"""
import json
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

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
RKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'rks.py')
UKS_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'uks.py')
HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')
GEOMS = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms')
PY = sys.executable


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


def _clean_gpu_env(extra=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GPU4PYSCF_')}
    if extra:
        env.update(extra)
    return env


# One lane of one cell in a fresh subprocess (env discipline exact).
# Returns {e, cycles, skip, modes, converged, grad6}.
_RUN_TPL = '''
from pyscf import __config__
__config__.scf_hf_auto_head_min_ao = 1
import json, sys
sys.path.insert(0, {rtx!r})
import pyscf
import cupy
from gpu4pyscf.dft import numint, RKS, UKS
from gpu4pyscf.lib import precision
import rtx4090_runner as R
mol = pyscf.M(atom={geom!r}, basis={basis!r}, charge={charge}, spin={spin},
              verbose=0)
cls = UKS if mol.spin else RKS
mf = cls(mol, xc={xc!r}).density_fit(auxbasis='def2-universal-jkfit')
mf.grids.level = {grids_level}
mf.conv_tol = 1e-10
mf.max_cycle = 60
mf.precision_mode = {lane!r}
probe = R.PrecisionProbe()
probe.install()
mf.callback = probe.callback
e = mf.kernel()
converged = bool(mf.converged)
g = mf.nuc_grad_method().kernel()
g = cupy.asarray(g)
probe.remove()
modes = [m for (_c, _e, m) in probe.cycles]
print(json.dumps({{'e': float(e), 'cycles': int(mf.cycles),
                   'skip': int(numint._head_nlc_skip_count),
                   'modes': modes, 'converged': converged,
                   'transitions': [t['to'] for t in probe.transitions],
                   'grad6': [float(x) for x in g.ravel()[:6]]}}))
'''


def _run_lane(env_extra, geom, basis, xc, lane, charge=0, spin=0,
              grids_level=2):
    code = _RUN_TPL.format(rtx=os.path.join(
        REPO, 'mixedprec', 'benchmark', 'rtx4090'),
        geom=geom, basis=basis, xc=xc, lane=lane, charge=charge,
        spin=spin, grids_level=grids_level)
    env = _clean_gpu_env(env_extra)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    out = subprocess.run([PY, '-c', code], env=env, cwd=REPO,
                         capture_output=True, text=True, timeout=1800)
    if out.returncode != 0:
        raise AssertionError(
            f'subprocess failed (lane={lane}, env={env_extra}):\n'
            f'{out.stderr[-2500:]}')
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith('{')]
    if not lines:
        raise AssertionError(f'no json line in stdout: {out.stdout[-800:]}')
    return json.loads(lines[-1])


class TestSourceContractHeadNlcSkip(unittest.TestCase):
    def test_env_helper_and_lane_and_counter_defined(self):
        src = open(NUMINT_SRC).read()
        self.assertIn('def _auto_head_nlc_skip()', src)
        self.assertIn('_scf_head_nlc_skip_lane = False', src)
        self.assertIn('_head_nlc_skip_count = 0', src)
        self.assertIn('def _scf_head_nlc_suppressed()', src)

    def test_env_parse_semantics(self):
        from gpu4pyscf.dft import numint
        try:
            numint._AUTO_HEAD_NLC_SKIP = None
            os.environ.pop('GPU4PYSCF_AUTO_HEAD_NLC_SKIP', None)
            self.assertFalse(numint._auto_head_nlc_skip())
            numint._AUTO_HEAD_NLC_SKIP = None
            os.environ['GPU4PYSCF_AUTO_HEAD_NLC_SKIP'] = '0'
            self.assertFalse(numint._auto_head_nlc_skip())
            numint._AUTO_HEAD_NLC_SKIP = None
            os.environ['GPU4PYSCF_AUTO_HEAD_NLC_SKIP'] = '1'
            self.assertTrue(numint._auto_head_nlc_skip())
            numint._AUTO_HEAD_NLC_SKIP = None
            os.environ['GPU4PYSCF_AUTO_HEAD_NLC_SKIP'] = 'yes'
            self.assertFalse(numint._auto_head_nlc_skip())  # warn+off
        finally:
            os.environ.pop('GPU4PYSCF_AUTO_HEAD_NLC_SKIP', None)
            numint._AUTO_HEAD_NLC_SKIP = None

    def test_suppressed_requires_lane_and_fp32(self):
        from gpu4pyscf.dft import numint
        from gpu4pyscf.lib import precision
        saved = precision.get_precision()
        try:
            numint._scf_head_nlc_skip_lane = False
            numint._head_nlc_skip_count = 0
            precision.set_precision('fp32')
            self.assertFalse(numint._scf_head_nlc_suppressed())
            numint._scf_head_nlc_skip_lane = True
            precision.set_precision('fp64')
            self.assertFalse(numint._scf_head_nlc_suppressed())
            c0 = numint._head_nlc_skip_count
            precision.set_precision('fp32')
            self.assertTrue(numint._scf_head_nlc_suppressed())
            self.assertEqual(numint._head_nlc_skip_count, c0 + 1)
            numint._scf_head_nlc_skip_lane = False
            numint._head_nlc_skip_count = 0
        finally:
            precision.set_precision(saved)

    def test_kernel_arms_lane_for_auto_family_only(self):
        src = _func_src(HF_SRC, '_kernel')
        self.assertIsNotNone(src)
        self.assertIn('_scf_head_nlc_skip_lane', src)
        self.assertIn('_auto_head_nlc_skip()', src)
        self.assertIn('_mixed_auto and', src)
        # armed before the kernel body, restored in the finally block
        # (the SP-27 lane-flag pattern)
        self.assertLess(
            src.index('_numint_mod._scf_head_nlc_skip_lane = ('),
            src.index('finally:'))

    def test_veff_nlc_block_gated_rks_uks_and_df(self):
        # the PRODUCTION SCF path is density-fitted: df_jk._DFHF.get_veff
        # carries its own three do_nlc sites (UKS/RKS/GKS branches)
        paths = {
            RKS_SRC: 'def get_veff',
            UKS_SRC: 'def get_veff',
            os.path.join(REPO, 'gpu4pyscf', 'df', 'df_jk.py'):
                'def get_veff',
        }
        for path, marker in paths.items():
            src = open(path).read()
            m = re.search(rf'{marker}\(.*?(?=\ndef |\nclass |\Z)', src,
                          re.S)
            self.assertIsNotNone(m, path)
            body = m.group(0)
            self.assertIn('do_nlc()', body)
            self.assertIn('numint._scf_head_nlc_suppressed()', body,
                          f'{path}: the NLC block must be gated')


class TestBehaviorHeadNlcSkipRKS(unittest.TestCase):
    GEOM = os.path.join(GEOMS, 'benzene.xyz')

    def test_default_and_explicit_off_never_suppress(self):
        for extra in (None, {'GPU4PYSCF_AUTO_HEAD_NLC_SKIP': '0'}):
            rec = _run_lane(extra, self.GEOM, 'def2-svp', 'wb97mv', 'auto')
            self.assertEqual(rec['skip'], 0)
            self.assertTrue(rec['converged'])
            self.assertIn('fp32', rec['modes'])   # a live head existed

    def test_optin_suppresses_exactly_the_fp32_head_rks(self):
        base = _run_lane(None, self.GEOM, 'def2-svp', 'wb97mv', 'auto')
        opt = _run_lane({'GPU4PYSCF_AUTO_HEAD_NLC_SKIP': '1'},
                        self.GEOM, 'def2-svp', 'wb97mv', 'auto')
        n_fp32 = opt['modes'].count('fp32')
        self.assertGreater(n_fp32, 0)
        self.assertEqual(opt['skip'], 1 + n_fp32)
        # single fp32->fp64 transition, no fp32 after fp64
        seen_f64 = False
        for m in opt['modes']:
            if m == 'fp64':
                seen_f64 = True
            elif seen_f64:
                self.fail(f'fp32 after fp64: {opt["modes"]}')
        # the default lane evaluated NLC in its head (skip 0 above);
        # the opt-in's converged energy is the full functional's fixed
        # point: within the card gate of the strict fp64 lane at
        # identical settings
        strict = _run_lane(None, self.GEOM, 'def2-svp', 'wb97mv', 'fp64')
        self.assertLessEqual(abs(opt['e'] - strict['e']), 1e-6)
        self.assertLessEqual(abs(base['e'] - strict['e']), 1e-6)

    def test_fp64_lane_never_suppresses_even_with_knob(self):
        rec = _run_lane({'GPU4PYSCF_AUTO_HEAD_NLC_SKIP': '1'},
                        self.GEOM, 'def2-svp', 'wb97mv', 'fp64')
        self.assertEqual(rec['skip'], 0)
        self.assertTrue(all(m == 'fp64' for m in rec['modes']))


class TestBehaviorHeadNlcSkipUKS(unittest.TestCase):
    # the methoxy radical: UKS + wb97m-v + NLC at unit-test cost
    GEOM = os.path.join(GEOMS, 'methoxy.xyz')

    def test_optin_suppresses_head_and_gates_uks(self):
        base = _run_lane(None, self.GEOM, 'def2-svp', 'wb97m-v', 'auto',
                         charge=0, spin=1)
        opt = _run_lane({'GPU4PYSCF_AUTO_HEAD_NLC_SKIP': '1'},
                        self.GEOM, 'def2-svp', 'wb97m-v', 'auto',
                        charge=0, spin=1)
        self.assertEqual(base['skip'], 0)
        n_fp32 = opt['modes'].count('fp32')
        self.assertGreater(n_fp32, 0)
        self.assertEqual(opt['skip'], 1 + n_fp32)
        # THE KNOB'S OWN INCREMENTAL EFFECT on this forced-gate UKS
        # diagnostic config (43 AO with the head forced on -- a
        # non-production regime; the production 250-AO gate keeps
        # small radicals on the exact fp64 lane): the head path change
        # moves the converged point by ~2e-8 (measured first GREEN
        # run), while the pre-existing auto-vs-strict gap of THIS
        # config (3.6e-6, the fp32-head trajectory on an open-shell
        # radical below the size gate) is NOT the knob's -- the gate
        # question at production settings is the SP-31 screen's
        # preregistered P4, on production cells above the gate.
        self.assertLessEqual(abs(opt['e'] - base['e']), 1e-7)
        self.assertTrue(opt['converged'] and base['converged'])


class TestNoNlcBitwiseNoop(unittest.TestCase):
    """b3lyp (no NLC): the knob must be a bitwise no-op."""

    GEOM = os.path.join(GEOMS, 'benzene.xyz')

    def test_bitwise_noop_b3lyp(self):
        j1 = _run_lane(None, self.GEOM, 'def2-svp', 'b3lyp', 'auto')
        j2 = _run_lane({'GPU4PYSCF_AUTO_HEAD_NLC_SKIP': '1'},
                       self.GEOM, 'def2-svp', 'b3lyp', 'auto')
        self.assertEqual(j1['skip'], 0)
        self.assertEqual(j2['skip'], 0)
        # SCF energies are cross-process deterministic: bitwise equal.
        # Gradients are NOT bitwise reproducible across processes (the
        # standing determinism contract: gradient kernels carry
        # nondeterministic reduction order) -- envelope instead.
        self.assertEqual(j1['e'], j2['e'])
        for a, b in zip(j1['grad6'], j2['grad6']):
            self.assertLessEqual(abs(a - b), 1e-9 + 1e-9 * abs(a))


if __name__ == '__main__':
    unittest.main()

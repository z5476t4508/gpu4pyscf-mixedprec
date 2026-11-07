#!/usr/bin/env python3
"""SP-27 regression tests: the auto-head rho-fp64 cast exemption --
REJECTED as the default by the SP-27 paired measurement, retained as
GPU4PYSCF_AUTO_RHO_F64=1 opt-in (gpu4pyscf/dft/numint.py::_eval_rho2 +
gpu4pyscf/scf/hf.py::_kernel lane flag; scf-priority-20261025 card
item 3, the profiling-selected minimal source change, prereg 931605f5).

The full SP-27 story (2026-10-26, source commit 01eeb48, 48/48 arms):
the per-call fp64 rho contraction is FASTER everywhere (head eval-rho
wall -10..-33%, confirming SP-26 P5 per-call), but the paired
whole-SCF 4-block ratio REGRESSED the winner cells 2-3.5% -- the more
precise rho head switches one cycle early on lidocaine (6 vs 7 head
cycles), trading a cheap fp32 head cycle for an expensive fp64 tail
cycle, and vitamin-924's head-cycle wall regresses +1.5% despite the
component gain; only benzene (the switch-immune loss class) gained
(t_scf ratio 0.988).  Verdict REJECT per the preregistered rule ->
the default flipped back to the pre-SP-27 fp32-cast head bitwise; the
knob stays as the documented opt-in.

Contracts pinned here (post-revert conditioning):
  * source: the lane flag + env helper exist in numint; _eval_rho2's
    cast branch is gated on `not _scf_rho_f64_lane`; hf._kernel arms
    the flag for auto-family kernels only and restores it in finally;
  * semantics (spy on the _eval_rho2-internal _contract_rho calls):
    the DEFAULT auto head keeps the fp32 rho cast (float32 records in
    the fp32 phase -- the reverted pre-SP-27 behavior); the opt-in env
    '1' records float64 on RKS and UKS; the explicit fp32 lane and the
    pure-fp64 lanes (incl the LC-115 headtail knob lane) are unchanged;
  * numerics: auto-vs-fp64 e_tot within 1e-9 Ha (regression pin, both
    converge), and |E_auto_default - E_auto_optin| <= 1e-9 (the
    LC-105/LC-115 knob-shift class is ~1e-13..7e-11; measured SP-27
    shifts 1.1e-12..1.7e-11).

RED discipline (the adoption-direction test at the pre-SP-27 HEAD
4aa5230 and its GREEN at 01eeb48 are preserved evidence:
2026-10-26-sp27-autohead-rho-f64/red-run.log + green-run.log; this
file was reconditioned to the reverted default afterwards, the SP-22
companion-conditioning precedent).
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

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')
GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'benzene.xyz')
UKS_GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methoxy.xyz')
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


class TestSourceContractRhoF64(unittest.TestCase):
    def test_lane_flag_and_env_helper_defined(self):
        src = open(NUMINT_SRC).read()
        self.assertIn('_scf_rho_f64_lane = False', src)
        self.assertIn('def _auto_rho_f64()', src)
        self.assertIn('GPU4PYSCF_AUTO_RHO_F64', src)

    def test_eval_rho2_cast_branch_gated_on_lane_flag(self):
        src = _func_src(NUMINT_SRC, '_eval_rho2')
        self.assertIsNotNone(src)
        self.assertIn(
            "if precision.get_precision() == 'fp32' and not "
            "_scf_rho_f64_lane:", src)

    def test_kernel_arms_flag_for_auto_family_and_restores(self):
        src = _func_src(HF_SRC, '_kernel')
        self.assertIsNotNone(src)
        self.assertIn(
            "_numint_mod._scf_rho_f64_lane = (\n"
            "        _mixed_auto and _numint_mod._auto_rho_f64())", src)
        self.assertIn('_numint_mod._scf_rho_f64_lane = _rho_f64_saved', src)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestHeadRhoSemantics(unittest.TestCase):
    '''One subprocess per variant; the child spies on the contraction
    dtype inside _eval_rho2 (the _contract_rho calls made while the
    _eval_rho2 wrapper frame is active) and prints the (precision mode,
    dtype) pairs.  benzene def2-svp (114 AO) with the SP-14 gate
    threshold overridden to 60 so the auto head engages, mirroring
    test_sp14_sizegate_autohead's pattern.'''

    CHILD = r'''
import sys
sys.path.insert(0, %(repo)r)
from pyscf import gto
from gpu4pyscf.dft import RKS, UKS
from gpu4pyscf.lib import precision
import gpu4pyscf.dft.numint as ni
import gpu4pyscf.scf.hf as hf

mode, cls_name, threshold = sys.argv[1:4]
threshold = int(threshold)
saved_gate = hf._AUTO_HEAD_MIN_AO
hf._AUTO_HEAD_MIN_AO = threshold

recs = []
in_rho2 = [False]
orig_er2 = ni._eval_rho2
orig_cr = ni._contract_rho

def er2(ao, cpos, xctype, with_lapl=False, buf=None, rho=None):
    in_rho2[0] = True
    try:
        return orig_er2(ao, cpos, xctype, with_lapl, buf=buf, rho=rho)
    finally:
        in_rho2[0] = False

def cr(bra, ket, rho=None):
    if in_rho2[0]:
        recs.append((precision.get_precision(), str(bra.dtype)))
    return orig_cr(bra, ket, rho=rho)

ni._eval_rho2 = er2
ni._contract_rho = cr

mol = gto.M(atom=%(geom)r, basis='def2-svp', verbose=0%(spin)s)
cls = {'RKS': RKS, 'UKS': UKS}[cls_name]
mf = cls(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.grids.level = 3
if mode != 'none':
    mf.precision_mode = mode
mf.kernel()
hf._AUTO_HEAD_MIN_AO = saved_gate
pairs = sorted(set(recs))
print('RHORECS', ' '.join(f'{m}:{d}' for m, d in pairs),
      'CONV', mf.converged, 'CYC', mf.cycles,
      'E %%.17g' %% mf.e_tot)
'''

    def _run(self, geom, cls_name, spin, mode, threshold, extra_env=None,
             timeout=900):
        script = self.CHILD % {'repo': REPO, 'geom': geom, 'spin': spin}
        r = subprocess.run(
            [PY, '-c', script, mode, cls_name, str(threshold)],
            capture_output=True, text=True, timeout=timeout,
            env=_clean_gpu_env(extra_env))
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        line = r.stdout.strip().splitlines()[-1]
        self.assertTrue(line.startswith('RHORECS'), line)
        toks = line.split()
        pairs = dict(t.split(':', 1) for t in toks[1:toks.index('CONV')])
        return pairs, line

    def test_default_auto_head_keeps_f32_rho_rks(self):
        # the reverted default (the SP-27 REJECT): the auto head's rho
        # contraction keeps the pre-SP-27 fp32 casts
        pairs, line = self._run(GEOM, 'RKS', '', 'auto', 60)
        self.assertIn('fp32', pairs, f'head must engage: {line}')
        self.assertEqual(
            pairs, {'fp32': 'float32', 'fp64': 'float64'},
            f'the default auto head must keep the fp32 rho cast '
            f'(the SP-27 REJECT revert): {line}')

    def test_default_auto_head_keeps_f32_rho_uks(self):
        # methoxy def2-svp is ~44 AO: threshold 30 keeps it above the
        # SP-14 gate so the head engages
        pairs, line = self._run(UKS_GEOM, 'UKS', ', spin=1', 'auto', 30)
        self.assertIn('fp32', pairs, f'head must engage: {line}')
        self.assertEqual(
            pairs, {'fp32': 'float32', 'fp64': 'float64'},
            f'the default UKS auto head must keep the fp32 rho cast: '
            f'{line}')

    def test_optin_head_rho_contraction_is_f64_rks(self):
        pairs, line = self._run(GEOM, 'RKS', '', 'auto', 60,
                                extra_env={'GPU4PYSCF_AUTO_RHO_F64': '1'})
        self.assertIn('fp32', pairs, f'head must engage: {line}')
        self.assertEqual(
            pairs, {'fp32': 'float64', 'fp64': 'float64'},
            f'the opt-in auto head rho contraction must run float64: '
            f'{line}')

    def test_optin_head_rho_contraction_is_f64_uks(self):
        pairs, line = self._run(UKS_GEOM, 'UKS', ', spin=1', 'auto', 30,
                                extra_env={'GPU4PYSCF_AUTO_RHO_F64': '1'})
        self.assertIn('fp32', pairs, f'head must engage: {line}')
        self.assertEqual(
            pairs, {'fp32': 'float64', 'fp64': 'float64'},
            f'the opt-in UKS auto head rho contraction must run '
            f'float64: {line}')

    def test_explicit_off_env_keeps_f32_head(self):
        pairs, line = self._run(GEOM, 'RKS', '', 'auto', 60,
                                extra_env={'GPU4PYSCF_AUTO_RHO_F64': '0'})
        self.assertEqual(
            pairs, {'fp32': 'float32', 'fp64': 'float64'},
            f"GPU4PYSCF_AUTO_RHO_F64=0 is the default (explicit off): "
            f'{line}')

    def test_explicit_fp32_lane_unchanged(self):
        pairs, line = self._run(GEOM, 'RKS', '', 'fp32', 60)
        self.assertEqual(
            pairs, {'fp32': 'float32'},
            f'explicit fp32 mode must keep the rho cast: {line}')

    def test_pure_fp64_lane_with_headtail_knob_unchanged(self):
        # the B lane (fp64 + LC-115 vmat-f32 headtail): rho is fp64
        # throughout, before and after SP-27
        pairs, line = self._run(
            GEOM, 'RKS', '', 'fp64', 60,
            extra_env={'GPU4PYSCF_SCF_VMAT_F32': 'headtail'})
        self.assertEqual(
            pairs, {'fp64': 'float64'},
            f'the pure-fp64 lane rho must stay float64: {line}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestNumericsEquivalence(unittest.TestCase):
    '''Subprocess e_tot pins on benzene def2-svp (threshold override 60:
    the head engages so the change is exercised).'''

    def _e_tot(self, mode, extra_env=None):
        script = TestHeadRhoSemantics.CHILD % {
            'repo': REPO, 'geom': GEOM, 'spin': ''}
        r = subprocess.run(
            [PY, '-c', script, mode, 'RKS', '60'],
            capture_output=True, text=True, timeout=900,
            env=_clean_gpu_env(extra_env))
        self.assertEqual(r.returncode, 0, r.stderr[-3000:])
        toks = r.stdout.strip().splitlines()[-1].split()
        assert toks[toks.index('CONV') + 1] == 'True', toks
        return float(toks[-1])

    def test_auto_within_1e9_of_fp64_lane(self):
        e_auto = self._e_tot('auto')
        e_fp64 = self._e_tot('fp64')
        self.assertLessEqual(abs(e_auto - e_fp64), 1e-9,
                             f'auto {e_auto!r} vs fp64 {e_fp64!r}')

    def test_change_shift_within_1e9(self):
        e_def = self._e_tot('auto')
        e_opt = self._e_tot('auto',
                            {'GPU4PYSCF_AUTO_RHO_F64': '1'})
        self.assertLessEqual(
            abs(e_def - e_opt), 1e-9,
            f'the head-policy shift must stay in the LC-105 knob-shift '
            f'class: default {e_def!r} opt-in {e_opt!r}')


if __name__ == '__main__':
    unittest.main()

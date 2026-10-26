#!/usr/bin/env python3
"""SP-14 regression tests: the size-gated auto head
(gpu4pyscf/scf/hf.py::_kernel, scf-priority-20261025 card item 3 via the
SP-12 NEW-CLASS branch and SP-13's preregistered decision).

Measured motivation (SP-12 full-envelope aggregate + SP-13 attribution,
preregistration 8b5899d9): below ~250 AO the auto policy's fp32 head is a
net LOSS -- per-cycle SCF excess C/B t_scf median 1.111 below 100 AO and
1.068 at 100-250 vs 0.978-1.008 at 250+; the excess is flat 1.10-1.13
across 0-150 AO (launch-bound signature), carried by the HEAD cycles
(head-wall ratios 1.24-2.31x on every probed loss cell) with the CDERI
share exactly 0; melatonin-318's head ratio 0.741 is the size-sign flip.
The change: a module constant _AUTO_HEAD_MIN_AO (default 250); when the
auto family runs on mol.nao below it, the lane degenerates to the EXACT
fp64 lane (_mixed_auto False: fp64 from cycle 0, no switch phase to
leave, vmat-knob lane eligibility identical to an explicit fp64 request)
so the SCF trajectory is bitwise the fp64 lane's.  nao >= 250 cells are
untouched (the head is a measured win from ~270 AO up).

Contracts pinned here:
  * source: the constant exists with the __config__ override pattern;
    the gate lives in _kernel, flips only _mixed_auto, and the fp32
    branch is `_mode == 'fp32' or _mixed_auto` (explicit fp32 mode can
    never be gated);
  * semantics: small-cell auto records ZERO fp32 get_veff entries (the
    switch check never finds an fp32 phase); default-None and explicit
    fp64 lanes are invisible to the gate; explicit fp32 mode unchanged;
    lowering the threshold below a cell's nao re-engages today's head;
  * bitwise: small-cell RKS and UKS auto e_tot equal the explicit-fp64
    lane's e_tot EXACTLY (subprocess-per-lane, so no within-process
    plan-cache reduction drift -- the LC-96 5.7e-14 class).
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

HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')
GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'benzene.xyz')
UKS_GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methoxy.xyz')
PY = sys.executable


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


def _clean_gpu_env():
    return {k: v for k, v in os.environ.items()
            if not k.startswith('GPU4PYSCF_')}


class TestSourceContractSizeGate(unittest.TestCase):
    def test_constant_defined(self):
        src = open(HF_SRC).read()
        self.assertIn(
            "_AUTO_HEAD_MIN_AO = getattr(__config__, "
            "'scf_hf_auto_head_min_ao', 250)", src)

    def test_gate_in_kernel_flips_only_mixed_auto(self):
        src = _func_src(HF_SRC, '_kernel')
        self.assertIsNotNone(src)
        # the gate keys on the PLAIN auto mode only: the verify/strict
        # variants keep their head at every size (the registered change
        # and the verifier's displacement scenarios stay untouched)
        self.assertIn("if _mode == 'auto' and mf.mol.nao < "
                      "_AUTO_HEAD_MIN_AO:", src)
        self.assertIn('_mixed_auto = False', src)
        # the auto family may no longer appear in the fp32 branch: the
        # gated lane rides the fp64 lane, and explicit fp32 can never
        # be gated
        self.assertIn("if _mode == 'fp32' or _mixed_auto:", src)

    def test_switch_block_and_tol_untouched(self):
        # the LC-115 mirror contract (the _kernel_body switch/tail flip
        # reads _PRECISION_SWITCH_TOL) is pinned by
        # test_scf_vmat_f32_headtail; here pin that the constant's value
        # and the kernel_body switch predicate are unchanged by SP-14
        src = open(HF_SRC).read()
        self.assertIn(
            "_PRECISION_SWITCH_TOL = getattr(__config__, "
            "'scf_hf_precision_switch_tol', 1e-4)", src)
        body = _func_src(HF_SRC, '_kernel_body')
        self.assertIn(
            'if _mixed_auto and precision.get_precision() == \'fp32\':',
            body)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestSizeGateSemantics(unittest.TestCase):
    '''get_veff-entry precision records per lane (benzene def2-svp = 114
    AO < 250: the gated side).'''

    def _records(self, mode, threshold=None):
        from pyscf import gto
        from gpu4pyscf.dft import RKS
        from gpu4pyscf.lib import precision
        import gpu4pyscf.scf.hf as hf
        if threshold is not None:
            if not hasattr(hf, '_AUTO_HEAD_MIN_AO'):
                self.fail('gate constant absent (pristine source)')
            saved = hf._AUTO_HEAD_MIN_AO
            hf._AUTO_HEAD_MIN_AO = threshold
        records = []
        mol = gto.M(atom=GEOM, basis='def2-svp', verbose=0)
        mf = RKS(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 3
        if mode is not None:
            mf.precision_mode = mode
        orig = mf.get_veff

        def wrapped(m, dm, *a, **kw):
            records.append(precision.get_precision())
            return orig(m, dm, *a, **kw)
        mf.get_veff = wrapped
        try:
            mf.kernel()
        finally:
            if threshold is not None:
                hf._AUTO_HEAD_MIN_AO = saved
        return records, bool(mf.converged), mf.cycles

    def test_small_cell_auto_zero_fp32_hits(self):
        recs, conv, cyc = self._records('auto')
        self.assertTrue(conv)
        self.assertEqual(set(recs), {'fp64'},
                         f'the gated auto lane must never enter fp32: '
                         f'{sorted(set(recs))}')

    def test_small_cell_default_lane_invisible(self):
        recs, conv, _ = self._records(None)
        self.assertTrue(conv)
        self.assertEqual(set(recs), {'fp64'})

    def test_small_cell_explicit_fp64_invisible(self):
        recs, conv, _ = self._records('fp64')
        self.assertTrue(conv)
        self.assertEqual(set(recs), {'fp64'})

    def test_explicit_fp32_lane_unchanged(self):
        # pure-fp32 cannot reach conv_tol 1e-10 (fp32 noise floor,
        # accuracy ~1e-4 by design); the control is the RECORD set only
        recs, conv, _ = self._records('fp32')
        self.assertEqual(set(recs), {'fp32'},
                         'explicit fp32 mode must not be size-gated')

    def test_threshold_lowered_head_engages(self):
        # benzene (114 AO) with the gate threshold at 60 behaves exactly
        # as today: fp32 head, switch to fp64 for the tail
        recs, conv, _ = self._records('auto', threshold=60)
        self.assertTrue(conv)
        self.assertEqual(set(recs), {'fp32', 'fp64'},
                         f'head must engage when nao >= threshold: '
                         f'{sorted(set(recs))}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestSizeGateBitwise(unittest.TestCase):
    '''e_tot bitwise equality auto-vs-fp64 on gated cells.  One
    subprocess per lane: within-process first-vs-second SCF runs differ
    by the plan-cache reduction class (LC-96, measured 5.7e-14), which
    would mask the comparison the contract is about.'''

    CHILD = r'''
import sys
sys.path.insert(0, %(repo)r)
from pyscf import gto
from gpu4pyscf.dft import RKS, UKS
mode = sys.argv[1]
mol = gto.M(atom=%(geom)r, basis='def2-svp', verbose=0%(spin)s)
cls = {'RKS': RKS, 'UKS': UKS}[%(cls)r]
mf = cls(mol, xc='b3lyp').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.grids.level = 3
mf.precision_mode = mode
mf.kernel()
print('E_TOT %%.17g' %% mf.e_tot, 'CONV', mf.converged, 'CYC', mf.cycles)
'''

    def _script(self, geom, cls, spin):
        return self.CHILD % {'repo': REPO, 'geom': geom, 'cls': cls,
                             'spin': spin}

    def _run(self, script):
        outs = []
        for mode in ('auto', 'fp64'):
            r = subprocess.run(
                [PY, '-c', script, mode],
                capture_output=True, text=True, timeout=600,
                env=_clean_gpu_env())
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            outs.append(r.stdout.strip().splitlines()[-1])
        return outs

    def test_small_rks_auto_equals_fp64_bitwise(self):
        script = self._script(GEOM, 'RKS', '')
        auto, fp64 = self._run(script)
        self.assertIn('CONV True', auto)
        self.assertIn('CONV True', fp64)
        self.assertEqual(auto.split()[1], fp64.split()[1],
                         f'gated auto must be bitwise the fp64 lane:\n'
                         f'auto={auto}\nfp64 ={fp64}')

    def test_small_uks_auto_equals_fp64_bitwise(self):
        script = self._script(UKS_GEOM, 'UKS', ', spin=1')
        auto, fp64 = self._run(script)
        self.assertIn('CONV True', auto)
        self.assertIn('CONV True', fp64)
        self.assertEqual(auto.split()[1], fp64.split()[1],
                         f'gated UKS auto must be bitwise the fp64 lane:\n'
                         f'auto={auto}\nfp64 ={fp64}')


if __name__ == '__main__':
    unittest.main()

#!/usr/bin/env python3
"""Regression tests for the LC-115 head/tail value of the LC-105
GPU4PYSCF_SCF_VMAT_F32 knob (gpu4pyscf/dft/numint.py parser +
scf/hf.py::_kernel_body tail flip).

LC-108 measured the full knob's size-safety boundary: at 687 AO the fp32
vmat contractions in the convergence tail displace the SCF fixed point by
+7.9e-9 into a ~1e-10 knife-edge limit cycle and the production lane fails
3/3 at the 50-cycle cap.  LC-114 probed the head/tail discipline (the auto
policy's own fp32-head/exact-fp64-tail rule, single-trigger |dE| <
_PRECISION_SWITCH_TOL) with zero source change and measured: 687 AO
restored (3/3 converged, 14 cycles == knob-off, +1.09e-11, t_scf 1.510x),
420 AO 1.428x/+6.8e-13, 222 AO 1.284x/+3.6e-11.  LC-115 lands it as the
opt-in value GPU4PYSCF_SCF_VMAT_F32=headtail.

Contracts:
  * parser: '1' -> True (full knob, unchanged), 'headtail' -> 'headtail'
    (truthy for the hf lane gate), garbage -> warn + False, unset -> False;
  * the tail flip lives in _kernel_body AFTER e_diff and BEFORE the
    convergence check, is gated on the lane flag (auto family and strict
    references never enter), and flips the flag to False (single trigger
    latch);
  * _kernel_body derives _vmat_f32_headtail from the PARSER value, not the
    lane flag, so the env knob never flips anything on non-pure-fp64 lanes;
  * GPU: fp64 lane dtype signature {float32, float64} (flip engages),
    '1' stays {float32} (no flip), auto lane e_tot bitwise unchanged vs
    env-unset (gate protection extended to the new value), UKS flip
    engages, whole-SCF e_tot bound vs knob-off.
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

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
HF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'hf.py')

GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'benzene.xyz')
UKS_GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methoxy.xyz')


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestKnobParserHeadtail(unittest.TestCase):
    def _parse(self, env_value):
        from gpu4pyscf.dft import numint
        if env_value is None:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
        else:
            os.environ['GPU4PYSCF_SCF_VMAT_F32'] = env_value
        numint._SCF_VMAT_F32 = None
        try:
            return numint._scf_vmat_f32()
        finally:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
            numint._SCF_VMAT_F32 = None
            numint._scf_vmat_f32_lane = False

    def test_headtail_value_accepted_and_truthy(self):
        v = self._parse('headtail')
        self.assertEqual(v, 'headtail')
        self.assertTrue(bool(v))   # the hf lane gate relies on truthiness

    def test_full_value_unchanged(self):
        self.assertIs(self._parse('1'), True)

    def test_garbage_warns_and_off(self):
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('yes'), False)

    def test_default_off(self):
        self.assertIs(self._parse(None), False)


class TestSourceContractHeadtail(unittest.TestCase):
    def test_parser_accepts_headtail(self):
        src = open(NUMINT_SRC).read()
        self.assertIn("raw == 'headtail'", src)
        self.assertIn("_SCF_VMAT_F32 = 'headtail'", src)

    def test_flip_in_kernel_body_position_and_latch(self):
        src = _func_src(HF_SRC, '_kernel_body')
        self.assertIsNotNone(src)
        # the mode is derived from the PARSER value, not the lane flag:
        # the env knob alone must never flip a non-pure-fp64 lane
        self.assertIn("_vmat_f32_headtail = (_numint_mod._scf_vmat_f32() "
                      "== 'headtail')", src)
        # single-trigger latch: gated on the lane flag (False on auto
        # family / strict references -> no flip), flips to False
        self.assertIn('if (_vmat_f32_headtail and _numint_mod._scf_vmat_f32_lane',
                      src)
        self.assertIn('and e_diff < _PRECISION_SWITCH_TOL', src)
        self.assertIn('_numint_mod._scf_vmat_f32_lane = False', src)
        # position: after e_diff, before the convergence check
        i_ediff = src.index('e_diff = abs(e_tot-last_hf_e)')
        i_flip = src.index('_vmat_f32_headtail and _numint_mod._scf_vmat_f32_lane')
        i_conv = src.index('e_diff < conv_tol and norm_gorb < conv_tol_grad')
        self.assertLess(i_ediff, i_flip)
        self.assertLess(i_flip, i_conv)

    def test_no_diis_reset_riding_the_flip(self):
        # the DIIS-reset hypothesis is a falsified family (issue #9
        # candidate C2); the head/tail flip must not smuggle one in
        src = _func_src(HF_SRC, '_kernel_body')
        flip_block = src[src.index('_vmat_f32_headtail and'):src.index(
            'e_diff < conv_tol and norm_gorb')]
        self.assertNotIn('mf_diis', flip_block)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestHeadtailFlipSemantics(unittest.TestCase):
    '''The flip's dtype signature per lane and value.

    Region tracking (the LC-105 dtype-shim discipline): the XC vmat loop
    and the NLC loop both call _scale_ao, and nr_nlc_vxc is OUT of the
    knob's scope (fp64 on every lane), so the NLC region is excluded --
    only the outside-NLC region carries the flip signature.
    '''

    def _run_rks(self, env_value, mode):
        from pyscf import gto
        from gpu4pyscf.dft import RKS, numint
        if env_value is None:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
        else:
            os.environ['GPU4PYSCF_SCF_VMAT_F32'] = env_value
        numint._SCF_VMAT_F32 = None
        records = []
        _region = {'r': 'xc'}
        _orig_scale = numint._scale_ao

        def w(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig_scale(ao, wv, *a, **kw)
        numint._scale_ao = w
        try:
            mol = gto.M(atom=GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.precision_mode = mode
            ni = mf._numint
            _orig_nlc = ni.nr_nlc_vxc

            def nlc_w(*a, **kw):
                prev = _region['r']
                _region['r'] = 'nlc'
                try:
                    return _orig_nlc(*a, **kw)
                finally:
                    _region['r'] = prev
            ni.nr_nlc_vxc = nlc_w
            try:
                mf.kernel()
            finally:
                ni.nr_nlc_vxc = _orig_nlc
        finally:
            numint._scale_ao = _orig_scale
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
            numint._SCF_VMAT_F32 = None
            numint._scf_vmat_f32_lane = False
        xc_dts = {dt for r, dt in records if r == 'xc'}
        return xc_dts, float(mf.e_tot), bool(mf.converged), mf.cycles

    def test_fp64_lane_headtail_flips_mid_scf(self):
        dts, e, conv, cycles = self._run_rks('headtail', 'fp64')
        self.assertTrue(conv)
        self.assertEqual(dts, {'float32', 'float64'},
                         f'fp32 head + exact-fp64 tail expected: {dts}')

    def test_full_value_does_not_flip(self):
        dts, e, conv, cycles = self._run_rks('1', 'fp64')
        self.assertTrue(conv)
        self.assertEqual(dts, {'float32'},
                         f"'1' semantics unchanged (no tail flip): {dts}")

    def test_headtail_whole_scf_energy_bound(self):
        _, e_off, conv_off, _ = self._run_rks(None, 'fp64')
        _, e_ht, conv_ht, _ = self._run_rks('headtail', 'fp64')
        self.assertTrue(conv_off and conv_ht)
        de = abs(e_ht - e_off)
        self.assertLessEqual(de, 1e-10,
                             f'headtail converged shift (LC-114 measured '
                             f'3.6e-11 at 222 AO, svp smaller): {de}')

    def test_auto_lane_noop_with_headtail_env(self):
        # the auto lane's own fp64 tail must survive the headtail env
        # value: the lane gate keeps the flag off, so the XC signature
        # stays fp32-head + fp64-tail (a gate leak would collapse it to
        # all-fp32).  e_tot is bounded, not bitwise: within-process
        # first-vs-second SCF runs differ by the plan-cache reduction
        # class (measured 5.7e-14 with the env UNSET on both runs -- the
        # LC-96 family), far below any real leak's tail-perturbation
        # shift (~1e-10..1e-9).
        dts_off, e_off, conv, _ = self._run_rks(None, 'auto')
        dts_ht, e_ht, conv2, _ = self._run_rks('headtail', 'auto')
        self.assertTrue(conv and conv2)
        self.assertEqual(dts_off, {'float32', 'float64'})
        self.assertEqual(dts_ht, {'float32', 'float64'},
                         f'gate leak would collapse the auto tail to '
                         f'fp32: {dts_ht}')
        self.assertLessEqual(abs(e_ht - e_off), 1e-11,
                             f'auto e_tot must be untouched by the headtail '
                             f'env value: {e_ht - e_off}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestHeadtailUKS(unittest.TestCase):
    def test_uks_flip_engages(self):
        from pyscf import gto
        from gpu4pyscf.dft import UKS, numint
        os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
        numint._SCF_VMAT_F32 = None
        records = []
        _region = {'r': 'xc'}
        _orig_scale = numint._scale_ao

        def w(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig_scale(ao, wv, *a, **kw)
        numint._scale_ao = w
        try:
            mol = gto.M(atom=UKS_GEOM, basis='def2-svp', spin=1, verbose=0)
            mf = UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.precision_mode = 'fp64'
            ni = mf._numint
            _orig_nlc = ni.nr_nlc_vxc

            def nlc_w(*a, **kw):
                prev = _region['r']
                _region['r'] = 'nlc'
                try:
                    return _orig_nlc(*a, **kw)
                finally:
                    _region['r'] = prev
            ni.nr_nlc_vxc = nlc_w
            try:
                mf.kernel()
            finally:
                ni.nr_nlc_vxc = _orig_nlc
        finally:
            numint._scale_ao = _orig_scale
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
            numint._SCF_VMAT_F32 = None
            numint._scf_vmat_f32_lane = False
        xc_dts = {dt for r, dt in records if r == 'xc'}
        self.assertTrue(mf.converged)
        self.assertEqual(xc_dts, {'float32', 'float64'},
                         f'UKS head/tail signature: {xc_dts}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestHeadtailRearmPerKernel(unittest.TestCase):
    def test_second_kernel_flips_again(self):
        # single trigger per SCF run: _kernel re-arms the lane flag at each
        # kernel entry, so a second kernel() on a fresh object must show
        # the fp32 head again (not an all-fp64 run).  Region tracking as
        # above: the NLC loop's fp64 _scale_ao calls are excluded.
        from pyscf import gto
        from gpu4pyscf.dft import RKS, numint
        os.environ['GPU4PYSCF_SCF_VMAT_F32'] = 'headtail'
        numint._SCF_VMAT_F32 = None
        records = []
        _region = {'r': 'xc'}
        _orig_scale = numint._scale_ao

        def w(ao, wv, *a, **kw):
            records.append((_region['r'], str(ao.dtype)))
            return _orig_scale(ao, wv, *a, **kw)
        numint._scale_ao = w

        def _make():
            mol = gto.M(atom=GEOM, basis='def2-svp', verbose=0)
            mf = RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.precision_mode = 'fp64'
            ni = mf._numint
            _orig_nlc = ni.nr_nlc_vxc

            def nlc_w(*a, **kw):
                prev = _region['r']
                _region['r'] = 'nlc'
                try:
                    return _orig_nlc(*a, **kw)
                finally:
                    _region['r'] = prev
            ni.nr_nlc_vxc = nlc_w
            return mf, ni, _orig_nlc
        try:
            mf, ni, _orig_nlc = _make()
            try:
                mf.kernel()
                n_first = len(records)
                mf2, ni2, _orig_nlc2 = _make()
                mf2.kernel()
            finally:
                ni.nr_nlc_vxc = _orig_nlc
                if 'ni2' in dir():
                    ni2.nr_nlc_vxc = _orig_nlc2
        finally:
            numint._scale_ao = _orig_scale
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
            numint._SCF_VMAT_F32 = None
            numint._scf_vmat_f32_lane = False
        self.assertTrue(mf.converged and mf2.converged)
        second_xc = {dt for r, dt in records[n_first:] if r == 'xc'}
        self.assertIn('float32', second_xc,
                      f'second kernel must re-arm the fp32 head: {second_xc}')
        self.assertIn('float64', second_xc,
                      f'second kernel must flip for its tail: {second_xc}')
        # and the flag is restored after each kernel (finally block)
        self.assertFalse(numint._scf_vmat_f32_lane)


if __name__ == '__main__':
    unittest.main()

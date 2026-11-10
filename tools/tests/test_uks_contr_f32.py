#!/usr/bin/env python3
"""Regression tests for the UKS-XCGRAD-CONTRF32-01 opt-in UKS twin of
the grid-contraction precision discipline (GPU4PYSCF_GRAD_GRID_F32,
gpu4pyscf/grad/uks.py::_get_exc_task).

Provenance: LC-89 (2026-10-13) measured the UKS main-XC fp32 cast
SLOWER on all four configs at nao <= 118 (methoxy svp/tzvp; -4..-14%
t_grad) and closed the surface with the scope-honesty clause that
reopening requires a new in-scope cell, not extrapolation.  Entry-400
(2026-11-06) decomposed the UKS xc_grid family interior on the
573/819-AO wb97m-v doublet cells: the deriv=2 contraction pool
(_gga_grad_sum_ + _tau_grad_dot_) is 0.9304/0.9454 of the family with
spin-doubled call counts 48/24 and 70/35 -- GEMM-bound per LC-87
(>= ~222 AO).  This card ports the RKS LC-88/LC-101 discipline: the
cast is contraction-only (cast ao_contr once per block, dtype-sized
contraction buffers via _gbuf(None)), rho keeps its fp64 AO and buffer
(bitwise-identical density), the dm_mask einsum and the exc1
accumulator stay float64.  Default OFF.  The strict fp64-reference
context forces the knob inert.  The UKS NLC sub-loop is covered by the
later UKS-NLC-CONTRF32-01 knob twin (fp32-mode stays SP-10-frozen;
test_nlc_grad_fp32.py pins it).

Contract tests (CPU, source-level):
  * the gate reads the SHARED home parser rks._grad_grid_f32 (no
    second env reader) and forces inert on the strict fp64-reference
    lane;
  * all three xctype branches cast exactly one contraction AO per
    block while eval_rho2 keeps the fp64 ao_mask (rho discipline);
  * the exc1 accumulator and dm_mask einsum stay float64;
  * the engagement counter increments through the home-module helper
    only (counter-fork lesson);
  * the UKS NLC path (rks.get_nlc_exc nset == 1 gate) is untouched.

Numerical tests (GPU):
  * dtype shim: knob ON under a fp64 resolved gradient -> float32 AO
    reaches _gga_grad_sum_/_tau_grad_dot_/_d1_dot_ from the main-XC
    loop while the NLC region stays float64; knob OFF -> float64
    everywhere (the default path is byte-equivalent to pre-change);
  * strict fp64-reference context: knob ON stays inert (float64
    contractions, no engagement bump);
  * same-orbital armed-vs-verbatim envelope on small UKS sentinels
    (O2 triplet wb97m-v MGGA+NLC; O2 triplet b3lyp plain GGA):
    max|dg| <= 5e-6 Eh/Bohr, e_tot anchor untouched by the knob;
  * engagement counter: +1 per armed gradient call, +0 when off.
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


class TestSourceContract(unittest.TestCase):
    def test_gate_reads_shared_parser_and_strict_ref_inert(self):
        task = _func_src(UKS_SRC, '_get_exc_task')
        self.assertIsNotNone(task)
        self.assertIn(
            'contr_f32 = rks_grad._grad_grid_f32() and '
            'not _strict_fp64_reference_active()', task)
        # the buffer discipline: dtype-sized allocations only when armed
        self.assertIn('_gbuf = (lambda b: None) if contr_f32 else (lambda b: b)',
                      task)

    def test_no_second_env_reader(self):
        # one env parser per knob (the home parser lives in grad.rks);
        # grad.uks must not grow its own GPU4PYSCF_GRAD_GRID_F32 read
        uks_src = open(UKS_SRC).read()
        self.assertNotIn("os.environ.get('GPU4PYSCF_GRAD_GRID_F32'", uks_src)

    def test_one_cast_per_branch_rho_keeps_fp64(self):
        task = _func_src(UKS_SRC, '_get_exc_task')
        self.assertEqual(task.count('ao_contr = ao_mask.astype(cupy.float32)'), 3,
                         'one contraction-AO cast per xctype branch')
        # rho keeps the fp64 AO in every branch (bitwise density)
        for branch in ("xctype == 'LDA'", "xctype == 'GGA'", "xctype == 'MGGA'"):
            self.assertIn(branch, task)
        self.assertNotIn('eval_rho2(_sorted_mol, ao_contr', task)
        # contraction calls consume the cast twin only
        self.assertNotIn('_gga_grad_sum_(ao_mask', task)
        self.assertNotIn('_tau_grad_dot_(ao_mask', task)
        self.assertNotIn('_d1_dot_(ao_mask', task)

    def test_accumulator_and_dm_einsum_stay_fp64(self):
        task = _func_src(UKS_SRC, '_get_exc_task')
        self.assertIn('exc1 = cupy.zeros((nao, 3))', task)
        self.assertEqual(
            task.count("exc1[idx] += cupy.einsum('nij,ij->ni', vtmp, dm_mask).T"), 6,
            'the fp64 dm_mask einsum accumulation in all three branches x 2 spins')

    def test_engagement_counter_home_module(self):
        src = open(UKS_SRC).read()
        self.assertIn('_CONTR_F32_GRADS = 0', src)
        self.assertIn('def _note_contr_f32_engaged():', src)
        task = _func_src(UKS_SRC, '_get_exc_task')
        self.assertIn('if contr_f32:\n            _note_contr_f32_engaged()', task)
        # no bare module-local fork of the increment inside the task
        self.assertNotIn('_CONTR_F32_GRADS += 1', task)

    def test_nlc_uks_path_knob_twin(self):
        # UKS-NLC-CONTRF32-01 (a later unit): the OPT-IN knob now
        # covers the UKS NLC sub-loop too (entry-401 evidence); the
        # fp32-mode cast stays SP-10-frozen nset==1-only
        nlc = _func_src(RKS_SRC, 'get_nlc_exc')
        self.assertIn('if nset == 1:\n        contr_f32 = fp32_grid or _knob',
                      nlc)
        self.assertIn(
            'contr_f32 = _knob and not _strict_fp64_reference_active()',
            nlc)
        self.assertEqual(nlc.count('ao_mask.astype(cupy.float32)'), 1)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestKnobDtypeShim(unittest.TestCase):
    '''dtype shim: which AO dtype reaches the contraction helpers from
    the UKS main-XC and NLC loops, per knob state on the fp64 lane.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def _run(self, knob_on):
        import pyscf
        from gpu4pyscf.dft import UKS
        from gpu4pyscf.grad import rks as rks_grad_mod
        from gpu4pyscf.grad import uks as uks_grad_mod

        records = []
        _orig_gga = rks_grad_mod._gga_grad_sum_
        _orig_tau = rks_grad_mod._tau_grad_dot_
        _orig_d1 = rks_grad_mod._d1_dot_
        _orig_get_exc = uks_grad_mod.get_exc
        _orig_nlc = rks_grad_mod.get_nlc_exc
        _region = {'r': 'outside'}

        def rec(name, fn):
            def w(ao, *a, **kw):
                records.append((name, _region['r'], str(ao.dtype)))
                return fn(ao, *a, **kw)
            return w

        def region_wrap(region, orig):
            def w(*a, **kw):
                prev = _region['r']
                _region['r'] = region
                try:
                    return orig(*a, **kw)
                finally:
                    _region['r'] = prev
            return w

        rks_grad_mod._gga_grad_sum_ = rec('gga', _orig_gga)
        rks_grad_mod._tau_grad_dot_ = rec('tau', _orig_tau)
        rks_grad_mod._d1_dot_ = rec('d1', _orig_d1)
        uks_grad_mod.get_exc = region_wrap('xc', _orig_get_exc)
        rks_grad_mod.get_nlc_exc = region_wrap('nlc', _orig_nlc)
        rks_grad_mod._GRAD_GRID_F32 = knob_on
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', charge=-1,
                          spin=1, verbose=0)   # methoxy radical doublet
            mf = UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method()
            g.precision_mode = 'fp64'
            g.kernel()
        finally:
            rks_grad_mod._gga_grad_sum_ = _orig_gga
            rks_grad_mod._tau_grad_dot_ = _orig_tau
            rks_grad_mod._d1_dot_ = _orig_d1
            uks_grad_mod.get_exc = _orig_get_exc
            rks_grad_mod.get_nlc_exc = _orig_nlc
            rks_grad_mod._GRAD_GRID_F32 = None
        return records

    def test_knob_on_casts_main_xc_only(self):
        records = self._run(True)
        xc = {r for r in records if r[1] == 'xc'}
        nlc = {r for r in records if r[1] == 'nlc'}
        self.assertTrue(xc, 'main-XC contraction records required')
        self.assertTrue({d for _, _, d in xc} == {'float32'},
                        f'main-XC contractions must see float32 AO, got {xc}')
        # UKS-NLC-CONTRF32-01 (a later unit) extends the knob to the
        # UKS NLC sub-loop; the fp32-mode pin (SP-10) lives in
        # test_nlc_grad_fp32.py
        if nlc:
            self.assertTrue({d for _, _, d in nlc} == {'float32'},
                            f'UKS NLC casts under the knob twin, got {nlc}')

    def test_knob_off_all_fp64(self):
        records = self._run(False)
        dtypes = {d for _, _, d in records}
        self.assertEqual(dtypes, {'float64'},
                         f'the default UKS path stays float64, got {dtypes}')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestStrictRefInert(unittest.TestCase):
    '''The knob must stay inert under the strict fp64-reference
    context (the reference sibling reproduces the plain DEFAULT lane).'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def test_strict_ref_forces_inert(self):
        import pyscf
        from gpu4pyscf.dft import UKS, numint
        from gpu4pyscf.grad import rks as rks_grad_mod
        from gpu4pyscf.grad import uks as uks_grad_mod

        records = []
        _orig_gga = rks_grad_mod._gga_grad_sum_
        rks_grad_mod._gga_grad_sum_ = (
            lambda ao, *a, **kw: (records.append(str(ao.dtype)),
                                  _orig_gga(ao, *a, **kw))[1])
        rks_grad_mod._GRAD_GRID_F32 = True   # knob ARMED
        before = uks_grad_mod._CONTR_F32_GRADS
        try:
            mol = pyscf.M(atom=self.GEOM, basis='def2-svp', charge=-1,
                          spin=1, verbose=0)
            mf = UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.kernel()
            g = mf.nuc_grad_method()
            g.precision_mode = 'fp64'
            with numint.strict_fp64_reference_ctx():
                g.kernel()
        finally:
            rks_grad_mod._gga_grad_sum_ = _orig_gga
            rks_grad_mod._GRAD_GRID_F32 = None
        after = uks_grad_mod._CONTR_F32_GRADS
        self.assertTrue(records)
        self.assertEqual(set(records), {'float64'},
                         f'strict-ref UKS contractions stay float64, got {set(records)}')
        self.assertEqual(after, before,
                         'no engagement bump under the strict fp64-reference context')


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestArmedVsVerbatim(unittest.TestCase):
    '''Same-orbital armed-vs-verbatim envelope on small UKS sentinels:
    one converged fp64 SCF, gradient twice (knob on / off).  The knob
    touches only the gradient contraction, so the e_tot anchor must be
    untouched and max|dg| stays far under the 3e-5 Eh/Bohr gate (LC-89
    measured 4e-8..2.7e-7 at the methoxy cells; bound 5e-6).'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def _envelope(self, xc):
        import numpy as np
        import pyscf
        from gpu4pyscf.dft import UKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        mol = pyscf.M(atom=self.GEOM, basis='def2-svp', charge=-1,
                      spin=1, verbose=0)
        mf = UKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 3
        mf.kernel()
        e_ref = mf.e_tot
        g = mf.nuc_grad_method()
        g.precision_mode = 'fp64'

        rks_grad_mod._GRAD_GRID_F32 = False
        try:
            dg_off = g.kernel()
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None
        rks_grad_mod._GRAD_GRID_F32 = True
        try:
            dg_on = g.kernel()
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None

        max_dg = float(np.abs(np.asarray(dg_on) - np.asarray(dg_off)).max())
        return max_dg, float(e_ref), float(mf.e_tot)

    def test_mgga_sentinel_wb97mv(self):
        max_dg, e_before, e_after = self._envelope('wb97m-v')
        self.assertLessEqual(max_dg, 5e-6)
        self.assertEqual(e_before, e_after,
                         'the knob must not perturb the SCF anchor')

    def test_gga_sentinel_b3lyp(self):
        max_dg, e_before, e_after = self._envelope('b3lyp')
        self.assertLessEqual(max_dg, 5e-6)
        self.assertEqual(e_before, e_after)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestEngagementCounter(unittest.TestCase):
    '''The home-module engagement counter bumps once per armed UKS
    gradient call and never when the knob is off.'''

    GEOM = os.path.join(REPO, 'mixedprec', 'benchmark', 'geoms', 'methanol.xyz')

    def _grad(self, knob_on):
        import pyscf
        from gpu4pyscf.dft import UKS
        from gpu4pyscf.grad import rks as rks_grad_mod

        mol = pyscf.M(atom=self.GEOM, basis='def2-svp', charge=-1,
                      spin=1, verbose=0)
        mf = UKS(mol, xc='wb97m-v').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.grids.level = 3
        mf.kernel()
        g = mf.nuc_grad_method()
        g.precision_mode = 'fp64'
        rks_grad_mod._GRAD_GRID_F32 = knob_on
        try:
            g.kernel()
        finally:
            rks_grad_mod._GRAD_GRID_F32 = None

    def test_counter_bumps_only_when_armed(self):
        from gpu4pyscf.grad import uks as uks_grad_mod
        self._grad(False)
        off = uks_grad_mod._CONTR_F32_GRADS
        self._grad(True)
        on = uks_grad_mod._CONTR_F32_GRADS
        self.assertEqual(on - off, 1,
                         'exactly one engagement bump per armed gradient call')


if __name__ == '__main__':
    unittest.main()

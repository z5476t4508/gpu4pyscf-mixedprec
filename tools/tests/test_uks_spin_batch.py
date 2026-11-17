#!/usr/bin/env python3
"""Focused regressions for the UKS-SPINBATCH-IMPL-01 opt-in across-spin
merge of the _nr_uks_task vmat-loop GEMMs (gpu4pyscf/dft/numint.py, knob
GPU4PYSCF_UKS_SPIN_BATCH).

Measured basis (entries 349/350, 2026-11-05): the per-spin GEMM families
of the UKS vmat loop are 37.5-68.0% of t_scf at the 4 exposed UKS cells;
entry 350 benched the across-spin merge -- two sequential per-spin GEMMs
as ONE wider GEMM (shared operand read, spin extent doubled, per-spin
elementwise scaling unchanged) -- production PASS for A-GGA raw dot
(1.0562-1.0565), A-MGGA contract (1.0550-1.0553) and C-tau train
(1.0565-1.0570); family B (rho cpos dot) benched NEGATIVE at production
and is NOT merged.  This implementation produces the TRANSPOSED
per-block contribution (rows-split output keeps add_sparse slices
contiguous); nr_uks symmetrizes the accumulated vmat (v + v.T) at the
end, so the result is mathematically exact with only GEMM-reduction ULP
differences -- numerically equivalent under the project error gates,
NOT bitwise.

Contract tests:
  * default OFF: knob unset -> legacy path runs (stats counters frozen);
  * armed: veff-level max|diff| within fp64 bands at GGA and MGGA,
    including the contr_f32 lane (GPU4PYSCF_SCF_VMAT_F32=1);
  * parser: non-'0'/'false'/'False' arms; those strings disarm;
  * inert while _strict_fp64_reference_active;
  * engagement counters: merged_blocks > 0 when armed, both frozen when
    unset;
  * end-to-end: tiny UKS SCF (GGA and MGGA), knob off vs on,
    |dE| <= 1e-10 Ha with cycle counts reported (equal expected; a drift
    is reported, not asserted away).
"""
import os
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy as np
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

from gpu4pyscf.dft import numint  # noqa: E402


def _reset_knob():
    numint._UKS_SPIN_BATCH = None


class _KnobEnv:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self._saved = os.environ.pop('GPU4PYSCF_UKS_SPIN_BATCH', None)
        _reset_knob()
        if self.value is not None:
            os.environ['GPU4PYSCF_UKS_SPIN_BATCH'] = self.value
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ['GPU4PYSCF_UKS_SPIN_BATCH'] = self._saved
        else:
            os.environ.pop('GPU4PYSCF_UKS_SPIN_BATCH', None)
        _reset_knob()


class _VmatF32Env:
    def __enter__(self):
        self._saved = os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
        numint._SCF_VMAT_F32 = None
        os.environ['GPU4PYSCF_SCF_VMAT_F32'] = '1'
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ['GPU4PYSCF_SCF_VMAT_F32'] = self._saved
        else:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_F32', None)
        numint._SCF_VMAT_F32 = None


def _stats_snapshot():
    return dict(numint._UKS_SPIN_BATCH_STATS)


class TestKnobParser(unittest.TestCase):
    def test_default_off(self):
        with _KnobEnv(None):
            self.assertFalse(numint._uks_spin_batch_enabled())

    def test_armed_by_one(self):
        with _KnobEnv('1'):
            self.assertTrue(numint._uks_spin_batch_enabled())

    def test_armed_by_any_nonfalse_string(self):
        with _KnobEnv('on'):
            self.assertTrue(numint._uks_spin_batch_enabled())

    def test_disarmed_by_zero_and_false(self):
        for off in ('0', 'false', 'False'):
            with _KnobEnv(off):
                self.assertFalse(numint._uks_spin_batch_enabled())

    def test_inert_under_strict_reference(self):
        saved = numint._strict_fp64_reference_active
        numint._strict_fp64_reference_active = True
        try:
            with _KnobEnv('1'):
                self.assertFalse(numint._uks_spin_batch_enabled())
        finally:
            numint._strict_fp64_reference_active = saved


def _uks_mf(xc):
    from pyscf import gto
    from gpu4pyscf.dft import UKS
    mol = gto.M(atom='O 0 0 0; O 0 0 1.21', spin=2, charge=0,
                basis='sto-3g', verbose=0)
    mf = UKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
    mf.conv_tol = 1e-12
    mf.grids.level = 2
    return mol, mf


def _veff_pair(xc, vmat_f32=False):
    '''Converged-dm veff with the knob off and on (same dm, same grids
    object fresh-built per call so no cross-call caches are shared).'''
    outs = {}
    for knob in ('0', '1'):
        mol, mf = _uks_mf(xc)
        with _KnobEnv(knob):
            st0 = _stats_snapshot()
            mf.kernel()
            dm = mf.make_rdm1()
            if vmat_f32:
                with _VmatF32Env():
                    veff = mf.get_veff(mol, dm)
            else:
                veff = mf.get_veff(mol, dm)
            outs[knob] = (veff, _stats_snapshot(), st0)
    return outs


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestVeffEquivalence(unittest.TestCase):
    def _check(self, xc, vmat_f32=False):
        outs = _veff_pair(xc, vmat_f32=vmat_f32)
        v0, after0, before0 = outs['0']
        v1, after1, before1 = outs['1']
        # engagement discipline on per-arm DELTAS (the module-global
        # counters accumulate across tests in one pytest process)
        d0 = {k: after0[k] - before0[k] for k in after0}
        d1 = {k: after1[k] - before1[k] for k in after1}
        self.assertEqual(d0['merged_blocks'], 0)
        self.assertEqual(d0['legacy_blocks'], 0)
        self.assertEqual(d0['tasks'], 0)
        self.assertGreater(d1['merged_blocks'], 0)
        self.assertGreater(d1['tasks'], 0)
        for s in (0, 1):
            self.assertLessEqual(
                float(cp.abs(v1[s] - v0[s]).max()),
                1e-11 * max(float(cp.abs(v0[s]).max()), 1.0),
                f'xc={xc} vmat_f32={vmat_f32} spin={s}')
            # post-nr_uks the vmat is symmetrized; merged output must
            # remain symmetric
            self.assertLessEqual(
                float(cp.abs(v1[s] - v1[s].T).max()),
                1e-11 * max(float(cp.abs(v1[s]).max()), 1.0))

    def test_gga_fp64(self):
        self._check('b3lyp')

    def test_mgga_fp64(self):
        self._check('m062x')

    def test_mgga_vmat_f32_lane(self):
        # the contr_f32 lane (vtmp=None scale path) must merge correctly
        self._check('m062x', vmat_f32=True)

    def test_default_off_frozen_counters(self):
        mol, mf = _uks_mf('b3lyp')
        with _KnobEnv(None):
            st0 = _stats_snapshot()
            mf.kernel()
            self.assertEqual(_stats_snapshot(), st0)


@unittest.skipUnless(HAS_GPU, 'needs GPU')
class TestTinyUKSSCF(unittest.TestCase):
    def _energy_pair(self, xc):
        results = {}
        for knob in ('0', '1'):
            with _KnobEnv(knob):
                mol, mf = _uks_mf(xc)
                e = mf.kernel()
                results[knob] = (e, getattr(mf, 'cycles', None))
        return results

    def test_gga_energy_invariance(self):
        results = self._energy_pair('b3lyp')
        e0, c0 = results['0']
        e1, c1 = results['1']
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        if c0 != c1:
            print(f'UKS-SPINBATCH-01 GGA cycle drift reported: '
                  f'{c0} -> {c1} (not asserted away)')

    def test_mgga_energy_invariance(self):
        results = self._energy_pair('m062x')
        e0, c0 = results['0']
        e1, c1 = results['1']
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        if c0 != c1:
            print(f'UKS-SPINBATCH-01 MGGA cycle drift reported: '
                  f'{c0} -> {c1} (not asserted away)')


if __name__ == '__main__':
    unittest.main()

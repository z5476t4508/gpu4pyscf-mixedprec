# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''D1-GRAD-JONLY-AUXFP64-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_DF_GRAD_JONLY_AUXFP64 knob (DEFAULT OFF) keeps the
single J-only gradient site (_j_energy_per_atom, df/grad/rhf.py -- the
UKS k_factor==0 route delegates there, so the RKS and UKS lanes share
it verbatim) on the DEPLOYED fp64 sum_ejk_int3c2e_ip1 kernel with its
fp64 dm/auxvec operands, even when GPU4PYSCF_DF_IP1_F32 (the D1-ON
lane) would have selected the f32 twin there.  M5-DRIFT-ATTRIB-01
(entry 645) measured that the f32 twin's aux accumulation decorrelates
from the fp64 2-center response and breaks the elementwise
aux-response cancellation behind the ~1e-6 Eh/Bohr k-repeat gradient
drift (the fp64 R lane passes via correlated cancellation, total
~1e-12); GRADAUX-F32-CANCEL-01 (entry 646) measured that operand-level
harmonization does NOT restore it.  This knob scopes the FP32/FP64
division per site instead: the armed path IS the deployed fp64 path
verbatim (kernel, operands, everything), so the armed gradient
reproduces the ip1-f32-off lane at the same-lane run-to-run scale.

Value-invariant class on the armed identity: no new arithmetic delta is
introduced (the armed lane runs deployed fp64 code); the band below is
the atomicAdd run-to-run reproducibility of the SAME lane, NOT a
bitwise claim.  The knob never fires on the global fp32 ('auto') lane
and stays inert under the strict fp64-reference context.  The hybrid
(J+K) route never calls the site and is untouched by construction.
'''
import os
import unittest
import warnings

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.df.grad import rhf as grad_dfrhf
from gpu4pyscf.dft import numint


def setUpModule():
    global mol_sph, mol_uks
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)
    mol_uks = pyscf.M(atom=mol_sph.atom, basis='def2-svp', charge=1, spin=1,
                      output='/dev/null', verbose=0)

def tearDownModule():
    global mol_sph, mol_uks
    mol_sph.stdout.close()
    mol_uks.stdout.close()
    del mol_sph, mol_uks

_ULP_BAND = 3e-12   # run-to-run reproducibility band of the SAME lane
                    # (the ip1/2c kernels accumulate with atomicAdd; two
                    # runs of the identical lane are not bitwise --
                    # measured marginal draws up to ~1.3e-12 at water
                    # scale, so the tripwire carries headroom above the
                    # observed flaky-draw class; still 4+ orders under
                    # the 3e-5 project gate)
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob touches (its own
    cache/counter/stats, the ip1-f32 prototype cache, the global
    precision mode, both env variables, AND the e646 auxresp
    counter/stats -- the j-only site updates those unconditionally on
    every pass, so this suite must restore them or it pollutes the
    adjacent GRADAUX suite's stats assertions when both run in one
    process).'''

    def setUp(self):
        self.s_knob = grad_dfrhf._JONLY_AUXFP64
        self.s_grads = grad_dfrhf._JONLY_AUXFP64_GRADS
        self.s_stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.s_aux_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.s_ip1f32 = grad_dfrhf._IP1_F32
        self.s_prec = numint.precision.get_precision()
        for var in ('GPU4PYSCF_DF_GRAD_JONLY_AUXFP64',
                    'GPU4PYSCF_DF_IP1_F32'):
            had = var in os.environ
            setattr(self, '_had_' + var, had)
            setattr(self, '_env_' + var, os.environ.get(var))

    def tearDown(self):
        grad_dfrhf._JONLY_AUXFP64 = self.s_knob
        grad_dfrhf._JONLY_AUXFP64_GRADS = self.s_grads
        grad_dfrhf._JONLY_AUXFP64_STATS.clear()
        grad_dfrhf._JONLY_AUXFP64_STATS.update(self.s_stats)
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_aux_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_aux_stats)
        grad_dfrhf._IP1_F32 = self.s_ip1f32
        numint.precision.set_precision(self.s_prec)
        for var in ('GPU4PYSCF_DF_GRAD_JONLY_AUXFP64',
                    'GPU4PYSCF_DF_IP1_F32'):
            if getattr(self, '_had_' + var):
                os.environ[var] = getattr(self, '_env_' + var)
            elif var in os.environ:
                del os.environ[var]

    def _set(self, knob=None, ip1f32=None):
        '''reset both caches to a known state and set the env vars.'''
        if knob is not None:
            grad_dfrhf._JONLY_AUXFP64 = None
            os.environ['GPU4PYSCF_DF_GRAD_JONLY_AUXFP64'] = knob
        if ip1f32 is not None:
            grad_dfrhf._IP1_F32 = None
            if ip1f32 == '':
                os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
            else:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = ip1f32


class _Ip1Recorder:
    '''wrap both sum_ejk_int3c2e_ip1 kernel entries on the shared
    libvhf_rys module (the same interception the GPU driver uses;
    df/grad/rhf.py resolves the attributes at call time), recording
    which entry each call hit.  A full molecular gradient drives TWO
    entries per call: the nuclear response (_grad_nuc_without_ecp,
    fp64 unless GPU4PYSCF_DF_GRAD_NUC_IP1_F32 is armed -- off in every
    test here) and the J-only / J+K DF site whose f32/f64 selection is
    the object under test; a direct jk_energy_per_atom(k_factor=0)
    call drives only the J-only entry.'''

    def __init__(self):
        self.calls = []
        self._lib = grad_dfrhf.libvhf_rys
        self._orig_f64 = self._lib.sum_ejk_int3c2e_ip1
        self._orig_f32 = self._lib.sum_ejk_int3c2e_ip1_f32

    def __enter__(self):
        calls = self.calls
        lib = self._lib

        def rec_f64(ejk, ejk_aux, dm, auxvec, *a, **k):
            calls.append('f64')
            return self._orig_f64(ejk, ejk_aux, dm, auxvec, *a, **k)

        def rec_f32(ejk, ejk_aux, dm, auxvec, *a, **k):
            calls.append('f32')
            return self._orig_f32(ejk, ejk_aux, dm, auxvec, *a, **k)

        lib.sum_ejk_int3c2e_ip1 = rec_f64
        lib.sum_ejk_int3c2e_ip1_f32 = rec_f32
        return calls

    def __exit__(self, *exc):
        self._lib.sum_ejk_int3c2e_ip1 = self._orig_f64
        self._lib.sum_ejk_int3c2e_ip1_f32 = self._orig_f32
        return False


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            grad_dfrhf._JONLY_AUXFP64 = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_JONLY_AUXFP64', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_JONLY_AUXFP64'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = grad_dfrhf._jonly_auxfp64()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='r2scan'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_knob_off_keeps_f32_twin(self):
        # knob OFF + ip1-f32 prototype ON: the site selects the f32 twin
        # (the D1-ON lane verbatim: one f32 j-only entry + one f64
        # nuclear entry per gradient), the armed counter never moves,
        # and the lane is reproducible at run-to-run scale
        self._set(knob='0', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS, c0)
        self.assertEqual(sorted(calls), ['f32', 'f32', 'f64', 'f64'])
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), False)
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_flips_to_deployed_fp64(self):
        # knob ON + ip1-f32 prototype ON: the j-only site runs the
        # deployed fp64 kernel (bitwise the ip1-f32-off lane), the
        # counter arms once per gradient, and the entry mix is exactly
        # the R-lane mix (two f64 entries per gradient)
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS - c0, 1)
        self.assertEqual(sorted(calls), ['f64', 'f64'])
        stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['natm'], mol_sph.natm)
        # the armed path IS the deployed fp64 path: reproduce it with the
        # prototype OFF (no knob) at the same-lane run-to-run scale
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _ULP_BAND)
        self.assertEqual(g_on.dtype, np.float64)

    def test_armed_uks_delegated_site(self):
        # UKS k_factor==0 delegates to the SAME _j_energy_per_atom site:
        # the knob arms once per gradient on the delegated route
        self._set(knob='1', ip1f32='1')
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS - c0, 1)
        self.assertEqual(sorted(calls), ['f64', 'f64'])
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), True)
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _ULP_BAND)
        self.assertLessEqual(abs(g_on - g_off).max(), _GRAD_GATE)

    def test_hybrid_route_untouched(self):
        # hybrid functional -> _jk_energy_per_atom: the J+K site keeps
        # the f32 twin under the prototype (its own selection line is
        # untouched -- the per-aux-batch launches of that route), the
        # j-only counter never moves, and the nuclear entry stays the
        # one f64 call
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph, xc='b3lyp')
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            g_hyb = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS, c0)
        self.assertGreaterEqual(calls.count('f32'), 1)
        self.assertEqual(calls.count('f64'), 1)
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), None)
        self._set(knob='0', ip1f32='1')
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_hyb - g_ref).max(), _ULP_BAND)

    def test_strict_reference_inert(self):
        # strict fp64-reference context: the knob is forced inert (the
        # site keeps the f32 twin under the prototype), no counter
        # movement, and the result reproduces the prototype lane
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS, c0)
        self.assertEqual(sorted(calls), ['f32', 'f64'])
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), False)
        self._set(knob='0', ip1f32='1')
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_ref).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_global_fp32_policy_inert(self):
        # the global fp32 ('auto' policy) lane: the knob never fires
        # (the policy's site selection is not the prototype's to flip),
        # the counter never moves
        numint.precision.set_precision('fp32')
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            g_fp32 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS, c0)
        self.assertIn('f32', calls)
        stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertIs(stats.get('site_fp32'), True)


class TestJonlyMechanics(_KnobState):
    '''the site called DIRECTLY (one jk_energy_per_atom(k_factor=0) call,
    no SCF): the dispatch, the operand dtype and the counter/stat
    geometry.'''

    def setUp(self):
        super().setUp()
        global mol_sph
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        mf = dft.RKS(mol_sph, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        self.mf = mf
        self.g = mf.nuc_grad_method()
        dm = mf.make_rdm1()
        import cupy as cp
        self.dm = cp.asarray(dm) if not isinstance(dm, cp.ndarray) else dm

    def test_direct_call_dispatch(self):
        # k_factor=0 hits the j-only site: knob ON + prototype ON flips
        # the kernel to the fp64 entry; knob OFF + prototype ON keeps
        # the f32 twin entry
        self._set(knob='1', ip1f32='1')
        c0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        with _Ip1Recorder() as calls:
            de_on = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                              k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS - c0, 1)
        self.assertEqual(calls, ['f64'])
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), True)
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_STATS['natm'],
                         mol_sph.natm)
        self._set(knob='0', ip1f32='1')
        with _Ip1Recorder() as calls:
            de_f32 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                               k_factor=0, hermi=1)
        self.assertEqual(calls, ['f32'])
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), False)
        # the armed J derivative IS the deployed fp64 evaluation of the
        # same inputs (same-lane run-to-run scale; the f32 lane carries
        # the ip1 f32 rounding of the whole J response)
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        de_fp64 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                            k_factor=0, hermi=1)
        self.assertLessEqual(abs(de_on - de_fp64).max(), _ULP_BAND)
        self.assertLess(abs(de_f32 - de_fp64).max(), _GRAD_GATE)

    def test_armed_is_verbatim_fp64_lane(self):
        # the armed call and the plain fp64 call run the IDENTICAL code
        # path (the deployed fp64 kernel entry): their kernel records
        # match exactly and the outputs agree at the same-lane
        # run-to-run scale; the plain-fp64 lane records armed=False
        # (the stats gap the first smoke run exposed on the R lane --
        # the site must record its realized state on every pass)
        self._set(knob='1', ip1f32='1')
        with _Ip1Recorder() as c_on:
            de_on = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                              k_factor=0, hermi=1)
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), True)
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        with _Ip1Recorder() as c_off:
            de_off = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                               k_factor=0, hermi=1)
        self.assertEqual(c_on, c_off)
        self.assertEqual(de_on.dtype, de_off.dtype)
        self.assertLessEqual(abs(de_on - de_off).max(), _ULP_BAND)
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), False)
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_STATS['natm'],
                         mol_sph.natm)


if __name__ == '__main__':
    print('Full test suite for D1-GRAD-JONLY-AUXFP64-01')
    print('Copyright 2021-2026 The PySCF Developers. All Rights Reserved.')
    unittest.main()

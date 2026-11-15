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
'''GRADAUX-F32-CANCEL-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_DF_GRAD_AUXRESP_F32 knob (DEFAULT OFF) routes the
j-only auxbasis-response 2-center evaluation of the molecular gradient
through the same float32 quantization the f32 ip1 J kernel applies to
its operands -- the dm_aux build of _j_energy_per_atom (df/grad/rhf.py,
the single J-only site: the UKS k_factor==0 route delegates there, so
the RKS and UKS lanes share it verbatim).  dm_aux is rebuilt as the
float32 outer product of the float32-cast auxvec (the same one-time
cast sum_ejk_int3c2e_ip1_f32 consumes on its armed lane) and
reassembled to float64 for the deployed e_int2c2e_ip1 kernel, whose
arithmetic stays float64 verbatim.  M5-DRIFT-ATTRIB-01 measured that
on the D1-ON f32 ip1 lane the fp64 2-center response decorrelates from
the f32-quantized ip1 aux accumulation and breaks the elementwise
aux-response cancellation behind the ~1e-6 Eh/Bohr k-repeat gradient
drift; this knob is the harmonization candidate.  The strict
fp64-reference context forces the knob inert.  The hybrid (J+K) route
never calls the site and is untouched by construction.

Value-perturbing class: the only arithmetic difference is the float32
rounding of each dm_aux entry (one rounding of the c_P*c_Q product of
float32-quantized coefficients).  Differences are expected at the
f32-rounding scale of ONE gradient sub-term; the band below is a
tripwire well under the 3e-5 project gradient gate, NOT a bitwise
claim.
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

_ULP_BAND = 1e-12   # run-to-run reproducibility band of the SAME lane
                    # (the ip1/2c kernels accumulate with atomicAdd; two
                    # runs of the identical lane are not bitwise)
_F32_BAND = 1e-6    # f32 rounding of the ONE 2c operand (each dm_aux
                    # entry carries one float32 rounding; measured
                    # 1.42e-07 worst assembled-gradient delta at water
                    # scale, 3.05e-08 on the direct J-only call) -> a
                    # tripwire with headroom over the measured scale and
                    # 30x under the 3e-5 project gate
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob touches.'''

    def setUp(self):
        self.s_f32 = grad_dfrhf._AUXRESP_F32
        self.s_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self._had_env = 'GPU4PYSCF_DF_GRAD_AUXRESP_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_DF_GRAD_AUXRESP_F32')

    def tearDown(self):
        grad_dfrhf._AUXRESP_F32 = self.s_f32
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_stats)
        if self._had_env:
            os.environ['GPU4PYSCF_DF_GRAD_AUXRESP_F32'] = self._env
        elif 'GPU4PYSCF_DF_GRAD_AUXRESP_F32' in os.environ:
            del os.environ['GPU4PYSCF_DF_GRAD_AUXRESP_F32']


class _AuxrespCapture:
    '''wrap the module-global int2c2e_ip1_per_atom name the site calls
    (the same interception the GPU driver uses; df/grad/rhf.py resolves
    the attribute at call time), counting calls and capturing the dm
    operand the site hands the 2-center kernel.'''

    def __init__(self):
        self.calls = []
        self._orig = grad_dfrhf.int2c2e_ip1_per_atom

    def __enter__(self):
        def rec(*a, **k):
            import cupy as cp
            dm = a[1]
            dm = cp.asnumpy(cp.asarray(dm)) if not isinstance(
                dm, np.ndarray) else np.asarray(dm)
            self.calls.append({'dm': dm.copy(), 'naux': dm.shape[0]})
            return self._orig(*a, **k)
        grad_dfrhf.int2c2e_ip1_per_atom = rec
        return self.calls

    def __exit__(self, *exc):
        grad_dfrhf.int2c2e_ip1_per_atom = self._orig
        return False


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            grad_dfrhf._AUXRESP_F32 = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_AUXRESP_F32', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_AUXRESP_F32'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = grad_dfrhf._auxresp_f32()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='r2scan'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_knob_off_bitwise_stability(self):
        # knob OFF: the deployed fp64 2-center operand runs identically
        # twice and never moves the armed counter; the off-lane dm is the
        # verbatim fp64 outer product (not f32-representable in general)
        grad_dfrhf._AUXRESP_F32 = False
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        with _AuxrespCapture() as calls:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, c0)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), False)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_STATS['naux'],
                         calls[0]['naux'])
        self.assertEqual(len(calls), 2)
        off = calls[0]['dm']
        self.assertFalse(bool(np.array_equal(
            off, off.astype(np.float32).astype(np.float64))))
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_rks_jonly_envelope(self):
        # knob ON (pure functional -> the j-only route): the site arms
        # once per gradient; envelope vs the deployed fp64 operand
        grad_dfrhf._AUXRESP_F32 = True
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        with _AuxrespCapture() as calls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - c0, 1)
        self.assertEqual(len(calls), 1)
        stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['naux'], calls[0]['naux'])
        on = calls[0]['dm']
        # every armed entry is an exact float32 value (the f32 outer
        # product reassembled to fp64); the whole-matrix scale differs
        # from the deployed fp64 operand only at the f32-rounding scale
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        grad_dfrhf._AUXRESP_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLess(err, _F32_BAND)   # f32 operand rounding, never bitwise
        self.assertLessEqual(err, _GRAD_GATE)
        self.assertEqual(g_on.dtype, np.float64)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), False)

    def test_armed_uks_delegated_site(self):
        # UKS k_factor==0 delegates to the SAME _j_energy_per_atom site:
        # the knob arms once per gradient; envelope vs the deployed call
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        grad_dfrhf._AUXRESP_F32 = True
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - c0, 1)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), True)
        grad_dfrhf._AUXRESP_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLess(err, _F32_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_hybrid_route_untouched(self):
        # hybrid functional -> _jk_energy_per_atom: the site is never
        # called, the counter never moves, the gradient is the deployed
        # lane verbatim
        mf = self._converged(mol_sph, xc='b3lyp')
        grad_dfrhf._AUXRESP_F32 = True
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        with _AuxrespCapture() as calls:
            g_hyb = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, c0)
        self.assertEqual(len(calls), 1)   # the hybrid auxresp site only
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), None)
        grad_dfrhf._AUXRESP_F32 = False
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_hyb - g_ref).max(), _ULP_BAND)

    def test_strict_reference_inert(self):
        # strict fp64-reference: the knob is forced inert, no counter
        # movement, the deployed fp64 operand runs, and the result
        # reproduces the plain deployed lane at run-to-run scale
        grad_dfrhf._AUXRESP_F32 = True
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        with _AuxrespCapture() as calls:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, c0)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), False)
        off = calls[0]['dm']
        self.assertFalse(bool(np.array_equal(
            off, off.astype(np.float32).astype(np.float64))))
        grad_dfrhf._AUXRESP_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_off).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)


class TestAuxrespMechanics(_KnobState):
    '''the site called DIRECTLY (one _j_energy_per_atom call, no SCF):
    the j-only dispatch, the operand quantization and the unit-scale
    envelope.'''

    def setUp(self):
        super().setUp()
        global mol_sph
        grad_dfrhf._AUXRESP_F32 = False
        mf = dft.RKS(mol_sph, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        self.mf = mf
        self.g = mf.nuc_grad_method()
        dm = mf.make_rdm1()
        import cupy as cp
        self.dm = cp.asarray(dm) if not isinstance(dm, cp.ndarray) else dm

    def test_direct_call_dispatch_and_quantization(self):
        # g.jk_energy_per_atom(k_factor=0) hits the j-only site: knob ON
        # quantizes the ONE dm operand the 2c kernel receives; knob OFF
        # hands the deployed fp64 outer product through verbatim
        c0 = grad_dfrhf._AUXRESP_F32_GRADS
        with _AuxrespCapture() as calls:
            de_off = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                               k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, c0)
        self.assertEqual(len(calls), 1)
        dm_off = calls[0]['dm']
        grad_dfrhf._AUXRESP_F32 = True
        with _AuxrespCapture() as calls:
            de_on = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                              k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, c0 + 1)
        self.assertEqual(len(calls), 1)
        dm_on = calls[0]['dm']
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), True)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_STATS['naux'],
                         dm_on.shape[0])
        self.assertEqual(dm_on.shape, dm_off.shape)
        # the quantization is EXACTLY the f32 outer product of the f32
        # auxvec reassembled to fp64: every entry an f32 value, and the
        # pairwise entry deltas sit at the f32-rounding scale of the
        # deployed products
        self.assertTrue(bool(np.array_equal(
            dm_on, dm_on.astype(np.float32).astype(np.float64))))
        rel = (abs(dm_on - dm_off).max()
               / max(1e-30, abs(dm_off).max()))
        self.assertLess(rel, 1e-6)
        self.assertGreater(rel, 0)
        # one 2c operand rounding: envelope of the assembled J derivative
        scale = max(1., float(abs(de_off).max()))
        self.assertLess(abs(de_on - de_off).max() / scale, _F32_BAND)

    def test_off_lane_dm_is_verbatim_outer_product(self):
        # knob OFF: the dm the 2c kernel receives is the deployed fp64
        # auxvec outer product (not all-f32-representable).  The operand
        # itself is NOT bitwise-stable between identical in-process
        # calls (measured 9.5e-14 max at water scale -- the
        # auxvec-pipeline fp64 reassociation noise floor of the deployed
        # 2c route); the armed f32 quantization collapses that floor
        # below the float32 ULP of every significant entry.
        with _AuxrespCapture() as c1:
            self.g.jk_energy_per_atom(self.dm, j_factor=1, k_factor=0,
                                      hermi=1)
        with _AuxrespCapture() as c2:
            self.g.jk_energy_per_atom(self.dm, j_factor=1, k_factor=0,
                                      hermi=1)
        off1, off2 = c1[0]['dm'], c2[0]['dm']
        self.assertFalse(bool(np.array_equal(
            off1, off1.astype(np.float32).astype(np.float64))))
        self.assertLessEqual(abs(off1 - off2).max(), _ULP_BAND)


if __name__ == '__main__':
    unittest.main()

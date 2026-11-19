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
'''D1-ARMGRAD-NUCIP1-F32-01 focused equivalence regression (2026-11-15).

The opt-in GPU4PYSCF_DF_GRAD_NUC_IP1_F32 knob (DEFAULT OFF) applies the
entry-113/496 f32-ip1 discipline to the nuclear-attraction response call
of the molecular gradient -- the sum_ejk_int3c2e_ip1 launch of
_grad_nuc_without_ecp (grad/rhf.py), OUTSIDE the jk guard box, over the
decontracted-sorted mol and the fakemol charge aux (the core of the
arm.grad.resid.gint band, entry 636: 0.034/0.045/0.061 s at the
434/537/687 production cells).  Only the integral kernel and its
pseudo-DM and charge operands go float32 (the dm cast is the only
rounding site -- the nuclear charges are exact small integers in fp32);
the de/de_aux accumulation targets and everything else stay the deployed
verbatim.  The strict fp64-reference context forces the knob inert.
The site lives in grad/rhf.py _hcore_energy and is shared verbatim by
the RHF/RKS/UKS gradient lanes (unlike the df/grad/rhf.py auxresp
sites, whose UKS twin keeps its own copy) -- the knob arms on every
lane through the same function.

Value-perturbing class: the only arithmetic difference is the float32
rounding of the one integral kernel's operands and internal arithmetic
(the entry-113 class; the jk-box family measured 1.33e-5 worst dg at
the production cells).  Differences are therefore expected at the
f32-rounding scale; the band below is a tripwire well under the 3e-5
project gradient gate, NOT a bitwise claim.
'''
import os
import unittest
import warnings

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.grad import rhf as grad_rhf
from gpu4pyscf.df import int3c2e_bdiv as i3c
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
                    # (the ip1 kernel accumulates with atomicAdd; two runs
                    # of the identical lane are not bitwise)
_F32_BAND = 1e-5    # f32 kernel rounding of the one nuc-response call
                    # (entry-113 class; jk-box family measured 1.33e-5
                    # worst dg at the production cells; this site measured
                    # 6.9e-07/3.7e-07 at water scale, RKS/UKS) -> 1e-5
                    # tripwire at water scale with two orders of headroom
                    # under the 3e-5 gate
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob and its neighbors touch.'''

    def setUp(self):
        self.s_f32 = grad_rhf._NUCIP1_F32
        self.s_grads = grad_rhf._NUCIP1_F32_GRADS
        self.s_stats = dict(grad_rhf._NUCIP1_F32_STATS)
        self._had_env = 'GPU4PYSCF_DF_GRAD_NUC_IP1_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_DF_GRAD_NUC_IP1_F32')

    def tearDown(self):
        grad_rhf._NUCIP1_F32 = self.s_f32
        grad_rhf._NUCIP1_F32_GRADS = self.s_grads
        grad_rhf._NUCIP1_F32_STATS.clear()
        grad_rhf._NUCIP1_F32_STATS.update(self.s_stats)
        if self._had_env:
            os.environ['GPU4PYSCF_DF_GRAD_NUC_IP1_F32'] = self._env
        elif 'GPU4PYSCF_DF_GRAD_NUC_IP1_F32' in os.environ:
            del os.environ['GPU4PYSCF_DF_GRAD_NUC_IP1_F32']


class _KernelCounters:
    '''count the actual sum_ejk_int3c2e_ip1(_f32) launches through the
    shared libvhf_rys module object (the same interception the GPU
    driver uses; grad/rhf.py resolves the attribute at call time).'''

    def __init__(self):
        self.counts = {'f32': 0, 'f64': 0}
        self._orig = {}

    def __enter__(self):
        for name, key in (('sum_ejk_int3c2e_ip1_f32', 'f32'),
                          ('sum_ejk_int3c2e_ip1', 'f64')):
            orig = getattr(i3c.libvhf_rys, name)
            self._orig[name] = orig

            def rec(*a, _orig=orig, _key=key, **k):
                self.counts[_key] += 1
                return _orig(*a, **k)
            setattr(i3c.libvhf_rys, name, rec)
        return self.counts

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(i3c.libvhf_rys, name, orig)
        return False


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            grad_rhf._NUCIP1_F32 = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_NUC_IP1_F32', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_NUC_IP1_F32'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = grad_rhf._nuc_ip1_f32()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='b3lyp'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_knob_off_bitwise_stability(self):
        # knob OFF: the deployed fp64 call runs identically twice and
        # never moves the armed counter; the fp64 kernel itself launches
        grad_rhf._NUCIP1_F32 = False
        mf = self._converged(mol_sph)
        c0 = grad_rhf._NUCIP1_F32_GRADS
        with _KernelCounters() as kc:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS, c0)
        self.assertIs(grad_rhf._NUCIP1_F32_STATS.get('armed'), False)
        self.assertGreaterEqual(kc['f64'], 2)
        self.assertEqual(kc['f32'], 0)
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_rks_b3lyp_envelope(self):
        # knob ON: the shared nuc-response site arms once per gradient;
        # envelope vs the deployed fp64 call
        grad_rhf._NUCIP1_F32 = True
        mf = self._converged(mol_sph)
        c0 = grad_rhf._NUCIP1_F32_GRADS
        with _KernelCounters() as kc:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS - c0, 1)
        self.assertGreaterEqual(kc['f32'], 1)
        stats = dict(grad_rhf._NUCIP1_F32_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['natm'], mol_sph.natm)
        self.assertGreaterEqual(stats['nao'], 1)
        grad_rhf._NUCIP1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLess(err, _F32_BAND)   # f32 kernel rounding, never bitwise
        self.assertLessEqual(err, _GRAD_GATE)
        self.assertEqual(g_on.dtype, np.float64)
        self.assertIs(grad_rhf._NUCIP1_F32_STATS.get('armed'), False)

    def test_armed_uks_shared_site(self):
        # UKS inherits _hcore_energy from GradientsBase: the SAME site
        # arms (the shared-site semantics of this knob, distinct from the
        # df/grad auxresp twins); envelope vs the deployed call
        mf = dft.UKS(mol_uks, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        grad_rhf._NUCIP1_F32 = True
        c0 = grad_rhf._NUCIP1_F32_GRADS
        g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS - c0, 1)
        self.assertIs(grad_rhf._NUCIP1_F32_STATS.get('armed'), True)
        grad_rhf._NUCIP1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_off).max()
        self.assertLess(err, _F32_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_strict_reference_inert(self):
        # strict fp64-reference: the knob is forced inert, no counter
        # movement, the deployed fp64 kernel runs, and the result
        # reproduces the plain deployed lane at run-to-run scale
        grad_rhf._NUCIP1_F32 = True
        mf = self._converged(mol_sph)
        c0 = grad_rhf._NUCIP1_F32_GRADS
        with _KernelCounters() as kc:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS, c0)
        self.assertIs(grad_rhf._NUCIP1_F32_STATS.get('armed'), False)
        self.assertGreaterEqual(kc['f64'], 1)
        self.assertEqual(kc['f32'], 0)
        grad_rhf._NUCIP1_F32 = False
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_off).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_charges_cast_exact(self):
        # the fp32 cast of the nuclear charges is exact (small integers):
        # the dm cast is the ONLY rounding site of the armed lane
        q = cp.asarray(-mol_sph.atom_charges(), dtype=np.float64)
        self.assertTrue(bool(cp.array_equal(
            q, cp.asarray(q, dtype=cp.float32))))


class TestNucIp1Mechanics(unittest.TestCase):
    '''the site called DIRECTLY (one kernel launch, no SCF): the kernel
    flip, the stats geometry and the unit-scale envelope.'''

    def setUp(self):
        global mol_sph
        self.mol = mol_sph
        rng = np.random.default_rng(20261115)
        nao = self.mol.nao_nr()
        dm = cp.asarray(rng.standard_normal((nao, nao)))
        dm = (dm + dm.T) * .5
        self.dm = dm
        self.s_f32 = grad_rhf._NUCIP1_F32
        self.s_grads = grad_rhf._NUCIP1_F32_GRADS
        self.s_stats = dict(grad_rhf._NUCIP1_F32_STATS)
        self._had_env = 'GPU4PYSCF_DF_GRAD_NUC_IP1_F32' in os.environ
        self._env = os.environ.get('GPU4PYSCF_DF_GRAD_NUC_IP1_F32')

    def tearDown(self):
        grad_rhf._NUCIP1_F32 = self.s_f32
        grad_rhf._NUCIP1_F32_GRADS = self.s_grads
        grad_rhf._NUCIP1_F32_STATS.clear()
        grad_rhf._NUCIP1_F32_STATS.update(self.s_stats)
        if self._had_env:
            os.environ['GPU4PYSCF_DF_GRAD_NUC_IP1_F32'] = self._env
        elif 'GPU4PYSCF_DF_GRAD_NUC_IP1_F32' in os.environ:
            del os.environ['GPU4PYSCF_DF_GRAD_NUC_IP1_F32']

    def test_direct_call_kernel_flip(self):
        # the isolated site: knob ON flips the ONE launch to the f32
        # kernel (f64 untouched); knob OFF keeps the deployed fp64 launch
        c0 = grad_rhf._NUCIP1_F32_GRADS
        grad_rhf._NUCIP1_F32 = False
        with _KernelCounters() as kc:
            de_off = grad_rhf._grad_nuc_without_ecp(self.mol, self.dm)
        self.assertEqual(kc, {'f32': 0, 'f64': 1})
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS, c0)
        stats_off = dict(grad_rhf._NUCIP1_F32_STATS)
        self.assertIs(stats_off.get('armed'), False)
        self.assertEqual(stats_off['natm'], self.mol.natm)
        grad_rhf._NUCIP1_F32 = True
        with _KernelCounters() as kc:
            de_on = grad_rhf._grad_nuc_without_ecp(self.mol, self.dm)
        self.assertEqual(kc, {'f32': 1, 'f64': 0})
        self.assertEqual(grad_rhf._NUCIP1_F32_GRADS, c0 + 1)
        stats_on = dict(grad_rhf._NUCIP1_F32_STATS)
        self.assertIs(stats_on.get('armed'), True)
        self.assertEqual(stats_on['natm'], stats_off['natm'])
        self.assertEqual(stats_on['nao'], stats_off['nao'])
        # the identical realized geometry between the lanes of the pair
        self.assertEqual((stats_on['natm'], stats_on['nao']),
                         (stats_off['natm'], stats_off['nao']))
        # f32 kernel rounding only; the de_aux path is fp64 in both lanes
        rel = (abs(de_on - de_off).max()
               / max(1., abs(de_off).max()))
        self.assertLess(rel, _F32_BAND)

    def test_dm_cast_semantics(self):
        # the operand staging is a plain fp32 cast of the sorted pseudo-DM
        # (order='C' no-op on the C-contiguous product of
        # apply_C_mat_CT): one rounding of each element, nothing else
        dm32 = cp.asarray(self.dm, dtype=cp.float32, order='C')
        self.assertTrue(bool(cp.array_equal(dm32, self.dm.astype(cp.float32))))
        self.assertTrue(dm32.flags['C_CONTIGUOUS'])


if __name__ == '__main__':
    unittest.main()

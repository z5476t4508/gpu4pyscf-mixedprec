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
'''GRADAUX-HARMDET-COMPOSE-01 focused composition regression (2026-11-16).

The composed harmonized-deterministic lane of the e645/e646/e649/e651
chain: BOTH knobs armed together at the single J-only gradient site
(_j_energy_per_atom, df/grad/rhf.py -- the UKS k_factor==0 route
delegates there).  Entry 651 (GRADAUX-DET-REDUCTION-01) closed the
determinism-alone hypothesis with the four-lane discriminator: the aux
sub-chain k-repeat drift is the upstream auxvec-pipeline wobble
CANCELLED in the assembled aux response only when both sides evaluate
the SAME argument (R and e649: both sides fp64 un-quantized; BASE and
e651: the 3c side f32-cast, argument mismatch ~4e-7; e646: both sides
cast but the 2c kernel's own atomicAdd reassociation re-breaks it).
The one untested combination the story predicts RESTORES: both sides
quantized at the SAME f32 cast (GPU4PYSCF_DF_GRAD_AUXRESP_F32, the
e646 operand knob) AND both kernels deterministic
(GPU4PYSCF_DF_GRAD_AUXDET, the e651 det entries) -- the composed CAND
of the D1-ON lane with no fp64 retreat.

This suite verifies the composition arming at the site: the entry mix
stays the e651 det mix (1x sum_ejk_int3c2e_ip1_f32_det + 1x fp64
nuclear + 1x e_int2c2e_ip1_det, atomic entries absent) while the
2-center operand carries the e646 quantization semantics (the f32
outer product of the f32-cast auxvec reassembled to fp64: every entry
an exact float32 value), both engagement counters fire once per
gradient, and both det entries still reproduce their accumulation
bitwise on pinned inputs.  Inertness classes of the neighbors hold:
the strict fp64-reference context forces BOTH knobs inert (the operand
stays the verbatim fp64 outer product); the global fp32 ('auto')
policy keeps auxdet inert (its arming requires the opt-in prototype
site lane) while the e646 knob still fires -- its documented semantics
is independence of the ip1-f32 selection; the e649 jonly flip reverts
the site to fp64 and leaves auxdet inert (site_fp32 False) with auxresp
still firing on the fp64 site operand.
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
                    # (both aux sub-chain kernels are deterministic at
                    # pinned inputs, but the auxvec pipeline upstream of
                    # the site still wobbles ~1e-13 and the hcore/nuclear
                    # components carry their own ~5e-14 class; the band is
                    # the measured same-lane draw class at water scale)
_COMPOSE_BAND = 1e-6  # composed-CAND-vs-BASE band: the e646 f32 rounding
                      # of the ONE 2c operand (measured 1.42e-07 worst
                      # assembled-gradient delta at water scale) plus the
                      # e651 reassociation (~1e-9) -> a tripwire with
                      # headroom, 30x under the 3e-5 project gate
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)

_ENV_VARS = ('GPU4PYSCF_DF_GRAD_AUXDET',
             'GPU4PYSCF_DF_GRAD_AUXRESP_F32',
             'GPU4PYSCF_DF_GRAD_JONLY_AUXFP64',
             'GPU4PYSCF_DF_IP1_F32')


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the composed lane touches (the
    auxdet cache/counter/stats, the auxresp cache/counter/stats, the
    jonly cache/counter/stats, the ip1-f32 prototype cache, the global
    precision mode) and all four env variables -- the j-only site updates
    the auxdet/jonly/auxresp stats unconditionally on every pass, so
    without the full restore this suite pollutes the adjacent
    JONLY/GRADAUX/AUXDET suites' stats assertions when several suites run
    in one process.'''

    def setUp(self):
        self.s_det_knob = grad_dfrhf._AUXDET
        self.s_det_grads = grad_dfrhf._AUXDET_GRADS
        self.s_det_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.s_aux_knob = grad_dfrhf._AUXRESP_F32
        self.s_aux_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.s_jl_knob = grad_dfrhf._JONLY_AUXFP64
        self.s_jl_grads = grad_dfrhf._JONLY_AUXFP64_GRADS
        self.s_jl_stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.s_ip1f32 = grad_dfrhf._IP1_F32
        self.s_prec = numint.precision.get_precision()
        for var in _ENV_VARS:
            had = var in os.environ
            setattr(self, '_had_' + var, had)
            setattr(self, '_env_' + var, os.environ.get(var))

    def tearDown(self):
        grad_dfrhf._AUXDET = self.s_det_knob
        grad_dfrhf._AUXDET_GRADS = self.s_det_grads
        grad_dfrhf._AUXDET_STATS.clear()
        grad_dfrhf._AUXDET_STATS.update(self.s_det_stats)
        grad_dfrhf._AUXRESP_F32 = self.s_aux_knob
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_aux_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_aux_stats)
        grad_dfrhf._JONLY_AUXFP64 = self.s_jl_knob
        grad_dfrhf._JONLY_AUXFP64_GRADS = self.s_jl_grads
        grad_dfrhf._JONLY_AUXFP64_STATS.clear()
        grad_dfrhf._JONLY_AUXFP64_STATS.update(self.s_jl_stats)
        grad_dfrhf._IP1_F32 = self.s_ip1f32
        numint.precision.set_precision(self.s_prec)
        for var in _ENV_VARS:
            if getattr(self, '_had_' + var):
                os.environ[var] = getattr(self, '_env_' + var)
            elif var in os.environ:
                del os.environ[var]

    def _set(self, det=None, auxresp=None, jonly=None, ip1f32=None):
        '''reset the caches to a known state and set the env vars.'''
        if det is not None:
            grad_dfrhf._AUXDET = None
            if det == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_AUXDET', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_AUXDET'] = det
        if auxresp is not None:
            grad_dfrhf._AUXRESP_F32 = None
            if auxresp == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_AUXRESP_F32', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_AUXRESP_F32'] = auxresp
        if jonly is not None:
            grad_dfrhf._JONLY_AUXFP64 = None
            if jonly == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_JONLY_AUXFP64', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_JONLY_AUXFP64'] = jonly
        if ip1f32 is not None:
            grad_dfrhf._IP1_F32 = None
            if ip1f32 == '':
                os.environ.pop('GPU4PYSCF_DF_IP1_F32', None)
            else:
                os.environ['GPU4PYSCF_DF_IP1_F32'] = ip1f32


class _Ip1Recorder:
    '''wrap the three sum_ejk_int3c2e_ip1 kernel entries on the shared
    libvhf_rys module, recording which entry each call hit.  On every
    DET call the wrap ALSO re-invokes the original entry twice with the
    SAME captured ctypes operands into fresh zero targets: with pinned
    inputs the deterministic lane must reproduce its accumulation
    bitwise (the unit-level determinism claim; the site's own call
    result is untouched by the re-invocations).'''

    def __init__(self):
        self.calls = []
        self.det_pinned = None   # [(ejk, ejk_aux), (ejk, ejk_aux)]
        self._lib = grad_dfrhf.libvhf_rys
        self._orig = {}
        for name in ('sum_ejk_int3c2e_ip1', 'sum_ejk_int3c2e_ip1_f32',
                     'sum_ejk_int3c2e_ip1_f32_det'):
            self._orig[name] = getattr(self._lib, name)

    def _enter(self):
        calls = self.calls
        for name, tag in (('sum_ejk_int3c2e_ip1', 'f64'),
                          ('sum_ejk_int3c2e_ip1_f32', 'f32'),
                          ('sum_ejk_int3c2e_ip1_f32_det', 'f32det')):
            orig = self._orig[name]

            def rec(*a, _orig=orig, _tag=tag, **k):
                calls.append(_tag)
                if _tag == 'f32det':
                    self._pinned_double_call(a)
                return _orig(*a, **k)

            setattr(self._lib, name, rec)
        return calls

    def _pinned_double_call(self, a):
        # re-invoke the det entry twice into fresh zero targets, reusing
        # every captured ctypes operand (pointers to the site's live
        # dm/auxvec buffers, the envs/offset objects and the partial
        # staging buffer geometry)
        import cupy as cp
        import ctypes
        natm = int(a[18].value)
        nbx = int(a[7].value)
        nby = int(a[8].value)
        targets = []
        for _ in range(2):
            ejk = cp.zeros((natm, 3))
            ejk_aux = cp.zeros((natm, 3))
            partial = cp.zeros((nbx*nby, 2*natm*3))
            b = list(a)
            b[0] = ctypes.cast(ejk.data.ptr, ctypes.c_void_p)
            b[1] = ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p)
            b[19] = ctypes.cast(partial.data.ptr, ctypes.c_void_p)
            self._orig['sum_ejk_int3c2e_ip1_f32_det'](*b)
            targets.append((ejk, ejk_aux))
        self.det_pinned = targets

    def __enter__(self):
        return self._enter()

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self._lib, name, orig)
        return False


class _C2cRecorder:
    '''wrap libpbc.e_int2c2e_ip1[_det] (resolved at call time inside
    pbc/df/int2c2e.py).  On every DET call the wrap re-invokes the det
    entry twice AND the atomic entry once with the SAME captured ctypes
    operands into fresh zero targets: the two det re-invocations must
    agree bitwise (pinned-input determinism) and the atomic
    re-invocation must agree at reassociation scale (same addends,
    arrival-order sum).  In the composed lane the captured dm operand is
    the e646-quantized twin, so the pinned determinism claim covers the
    quantized argument the det kernel actually consumes.'''

    def __init__(self):
        self.calls = []
        self.det_pinned = None   # (det_out_1, det_out_2, atomic_out)
        self._natm = None
        from gpu4pyscf.pbc.df import int2c2e as _i2c
        self._mod = _i2c
        self._orig = {}
        for name in ('e_int2c2e_ip1', 'e_int2c2e_ip1_det'):
            self._orig[name] = getattr(_i2c.libpbc, name)

    def __enter__(self):
        calls = self.calls
        orig = self._orig
        import cupy as cp
        import ctypes

        def rec_atomic(*a, **k):
            calls.append('2c')
            return orig['e_int2c2e_ip1'](*a, **k)

        def rec_det(*a, **k):
            calls.append('2cdet')
            natm = self._natm
            nbatches = int(a[7].value)
            det_outs = []
            for _ in range(2):
                out = cp.zeros((natm, 3))
                partial = cp.zeros((nbatches, natm*3))
                stage_v = cp.empty((nbatches, 256, 6))
                stage_atm = cp.empty((nbatches, 256, 2), dtype=cp.int32)
                b = list(a)
                b[0] = ctypes.cast(out.data.ptr, ctypes.c_void_p)
                b[11] = ctypes.cast(partial.data.ptr, ctypes.c_void_p)
                b[12] = ctypes.cast(stage_v.data.ptr, ctypes.c_void_p)
                b[13] = ctypes.cast(stage_atm.data.ptr, ctypes.c_void_p)
                orig['e_int2c2e_ip1_det'](*b)
                det_outs.append(out)
            out_atomic = cp.zeros((natm, 3))
            b = list(a[:11])
            b[0] = ctypes.cast(out_atomic.data.ptr, ctypes.c_void_p)
            orig['e_int2c2e_ip1'](*b)
            self.det_pinned = (det_outs[0], det_outs[1], out_atomic)
            return orig['e_int2c2e_ip1_det'](*a, **k)

        self._mod.libpbc.e_int2c2e_ip1 = rec_atomic
        self._mod.libpbc.e_int2c2e_ip1_det = rec_det
        return calls

    def set_shapes(self, natm):
        self._natm = natm

    def __exit__(self, *exc):
        for name, o in self._orig.items():
            setattr(self._mod.libpbc, name, o)
        return False


class _AuxrespCapture:
    '''wrap the module-global int2c2e_ip1_per_atom name the site calls,
    capturing the dm operand the site hands the 2-center kernel (the
    quantization-semantics probe: the composed lane's operand must be
    the f32 outer product reassembled to fp64, every entry an exact
    float32 value).'''

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


class TestComposedArming(_KnobState):

    def _converged(self, mol, xc='r2scan'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf

    def test_composed_armed_entry_mix(self):
        # BOTH knobs ON + ip1-f32 prototype ON (the composed CAND of the
        # D1-ON lane): the entry mix stays the e651 det mix (f32det 3c +
        # det 2c, atomic entries absent) while the 2-center operand
        # carries the e646 quantization semantics; BOTH engagement
        # counters fire once per gradient; both det entries reproduce
        # their accumulation bitwise on pinned inputs (the 2c pinned pair
        # on the quantized argument the det kernel consumes)
        self._set(det='1', auxresp='1', ip1f32='1')
        mf = self._converged(mol_sph)
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        rec1 = _Ip1Recorder()
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        cap = _AuxrespCapture()
        with rec1 as calls, rec2 as c2c, cap as auxcalls:
            g_on1 = mf.nuc_grad_method().kernel()
            g_on2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - d0, 2)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - a0, 2)
        self.assertEqual(sorted(calls), ['f32det', 'f32det', 'f64', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 2)
        self.assertEqual(c2c.count('2c'), 0)
        self.assertEqual(len(auxcalls), 2)
        det_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(det_stats.get('armed'), True)
        self.assertEqual(det_stats['natm'], mol_sph.natm)
        self.assertGreater(det_stats['nblocks'], 0)
        aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.assertIs(aux_stats.get('armed'), True)
        self.assertEqual(aux_stats['naux'], auxcalls[0]['naux'])
        # the composed operand: every entry an exact float32 value (the
        # f32 outer product of the f32-cast auxvec reassembled to fp64)
        on = auxcalls[0]['dm']
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        # pinned-input bitwise determinism of the 3c f32 det entry
        import cupy as cp
        (e_a, ea_a), (e_b, ea_b) = rec1.det_pinned
        self.assertTrue(bool(cp.array_equal(e_a, e_b)))
        self.assertTrue(bool(cp.array_equal(ea_a, ea_b)))
        # ... and of the 2c det entry ON THE QUANTIZED ARGUMENT, with
        # the atomic entry on the same pinned operands agreeing at
        # reassociation scale
        d1, d2, d_atomic = rec2.det_pinned
        self.assertTrue(bool(cp.array_equal(d1, d2)))
        self.assertLessEqual(float(cp.abs(d1 - d_atomic).max()), 1e-12)
        # same-lane repeat stays inside the full-gradient band
        self.assertLessEqual(abs(g_on1 - g_on2).max(), _ULP_BAND)
        # the composed lane reproduces the BASE lane (both knobs off) at
        # the f32-operand + reassociation scale
        self._set(det='0', auxresp='', ip1f32='1')
        g_base = mf.nuc_grad_method().kernel()
        err = abs(g_on1 - g_base).max()
        self.assertLess(err, _COMPOSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)
        self.assertEqual(g_on1.dtype, np.float64)

    def test_composed_uks_delegated_site(self):
        # UKS k_factor==0 delegates to the SAME _j_energy_per_atom site:
        # the composed lane arms once per gradient on the delegated route
        # with the same entry mix and the quantized operand
        self._set(det='1', auxresp='1', ip1f32='1')
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_uks.natm)
        cap = _AuxrespCapture()
        with _Ip1Recorder() as calls, rec2 as c2c, cap as auxcalls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - d0, 1)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - a0, 1)
        self.assertEqual(sorted(calls), ['f32det', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 1)
        self.assertEqual(len(auxcalls), 1)
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), True)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), True)
        on = auxcalls[0]['dm']
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        self._set(det='0', auxresp='', ip1f32='1')
        g_base = mf.nuc_grad_method().kernel()
        err = abs(g_on - g_base).max()
        self.assertLess(err, _COMPOSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)

    def test_composed_strict_reference_inert(self):
        # strict fp64-reference context: BOTH knobs forced inert (the
        # site keeps the f32 twin under the prototype with the atomic 2c
        # entry and the VERBATIM fp64 outer-product operand, not
        # f32-representable), neither counter moves
        self._set(det='1', auxresp='1', ip1f32='1')
        mf = self._converged(mol_sph)
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        cap = _AuxrespCapture()
        with _Ip1Recorder() as calls, rec2 as c2c, cap as auxcalls:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, d0)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS, a0)
        self.assertEqual(sorted(calls), ['f32', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertGreaterEqual(c2c.count('2c'), 1)
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), False)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), False)
        off = auxcalls[0]['dm']
        self.assertFalse(bool(np.array_equal(
            off, off.astype(np.float32).astype(np.float64))))
        self._set(det='0', auxresp='', ip1f32='1')
        g_base = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_base).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_composed_global_fp32_policy_auxdet_inert(self):
        # the global fp32 ('auto' policy) lane: auxdet never fires (its
        # arming requires the opt-in ip1-f32 prototype site lane, not the
        # policy's own selection), while the e646 knob still fires -- its
        # documented semantics is independence of the ip1-f32 selection;
        # the counters record exactly that split
        numint.precision.set_precision('fp32')
        self._set(det='1', auxresp='1', ip1f32='1')
        mf = self._converged(mol_sph)
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        cap = _AuxrespCapture()
        with _Ip1Recorder() as calls, rec2 as c2c, cap as auxcalls:
            g_fp32 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, d0)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - a0, 1)
        self.assertEqual(sorted(calls), ['f32', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertEqual(c2c.count('2c'), 1)
        det_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(det_stats.get('armed'), False)
        self.assertIs(det_stats.get('site_fp32'), True)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), True)
        on = auxcalls[0]['dm']
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        self.assertEqual(g_fp32.dtype, np.float64)

    def test_composed_jonly_flip_layering(self):
        # layering with the e649 knob: when jonly-auxfp64 reverts the
        # site to the deployed fp64 kernel, auxdet stays inert (the
        # realized site_fp32 is False) while the independent auxresp knob
        # still fires on the fp64 site operand (the e646 documented
        # semantics); the two repair knobs never compose into the det
        # lane
        self._set(det='1', auxresp='1', jonly='1', ip1f32='1')
        mf = self._converged(mol_sph)
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        j0 = grad_dfrhf._JONLY_AUXFP64_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        cap = _AuxrespCapture()
        with _Ip1Recorder() as calls, rec2 as c2c, cap as auxcalls:
            g_jl = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, d0)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - a0, 1)
        self.assertEqual(grad_dfrhf._JONLY_AUXFP64_GRADS - j0, 1)
        self.assertEqual(sorted(calls), ['f64', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertGreaterEqual(c2c.count('2c'), 1)
        det_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(det_stats.get('armed'), False)
        self.assertIs(det_stats.get('site_fp32'), False)
        self.assertIs(grad_dfrhf._JONLY_AUXFP64_STATS.get('armed'), True)
        self.assertIs(grad_dfrhf._AUXRESP_F32_STATS.get('armed'), True)
        on = auxcalls[0]['dm']
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        # vs the plain jonly lane (auxresp off): only the 2c operand
        # quantization differs
        self._set(det='0', auxresp='', ip1f32='1')
        g_jl_base = mf.nuc_grad_method().kernel()
        err = abs(g_jl - g_jl_base).max()
        self.assertLess(err, _COMPOSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)


class TestComposedMechanics(_KnobState):
    '''the site called DIRECTLY (one jk_energy_per_atom(k_factor=0) call,
    no SCF): the composed dispatch, the pinned-input bitwise determinism
    of both det entries on the quantized argument, and the counter/stat
    geometry.'''

    def setUp(self):
        super().setUp()
        global mol_sph
        self._set(det='0', auxresp='', jonly='', ip1f32='')
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

    def test_direct_call_composed_dispatch(self):
        # k_factor=0 hits the j-only site: BOTH knobs ON + prototype ON
        # routes the 3c kernel to the det entry AND the 2c response to
        # the det entry with the quantized operand; the BASE lane (both
        # knobs off) keeps the f32 twin + atomic 2c
        self._set(det='1', auxresp='1', ip1f32='1')
        d0 = grad_dfrhf._AUXDET_GRADS
        a0 = grad_dfrhf._AUXRESP_F32_GRADS
        rec1 = _Ip1Recorder()
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        cap = _AuxrespCapture()
        with rec1 as calls, rec2 as c2c, cap as auxcalls:
            de_on = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                              k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - d0, 1)
        self.assertEqual(grad_dfrhf._AUXRESP_F32_GRADS - a0, 1)
        self.assertEqual(calls, ['f32det'])
        self.assertEqual(c2c, ['2cdet'])
        self.assertEqual(len(auxcalls), 1)
        det_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(det_stats.get('armed'), True)
        self.assertEqual(det_stats['natm'], mol_sph.natm)
        self.assertGreater(det_stats['naux'], 0)
        self.assertGreater(det_stats['nblocks'], 0)
        aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.assertIs(aux_stats.get('armed'), True)
        self.assertEqual(aux_stats['naux'], auxcalls[0]['naux'])
        on = auxcalls[0]['dm']
        self.assertTrue(bool(np.array_equal(
            on, on.astype(np.float32).astype(np.float64))))
        # pinned-input bitwise determinism of both det entries driven
        # through the composed direct call (the 2c pinned pair on the
        # quantized argument the det kernel consumes)
        import cupy as cp
        (e_a, ea_a), (e_b, ea_b) = rec1.det_pinned
        self.assertTrue(bool(cp.array_equal(e_a, e_b)))
        self.assertTrue(bool(cp.array_equal(ea_a, ea_b)))
        d1, d2, d_atomic = rec2.det_pinned
        self.assertTrue(bool(cp.array_equal(d1, d2)))
        self.assertLessEqual(float(cp.abs(d1 - d_atomic).max()), 1e-12)
        # the composed lane vs the BASE lane: f32 operand + reassociation
        self._set(det='0', auxresp='', ip1f32='1')
        with _Ip1Recorder() as calls3:
            de_base = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                                k_factor=0, hermi=1)
        self.assertEqual(calls3, ['f32'])
        err = float(abs(de_on - de_base).max())
        self.assertLess(err, _COMPOSE_BAND)
        self.assertLessEqual(err, _GRAD_GATE)
        # ... and both stay inside the project gate of the plain fp64 lane
        self._set(det='0', auxresp='', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        de_fp64 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                            k_factor=0, hermi=1)
        self.assertLess(abs(de_base - de_fp64).max(), _GRAD_GATE)
        self.assertLess(abs(de_on - de_fp64).max(), _GRAD_GATE)
        self.assertEqual(de_on.dtype, np.float64)


if __name__ == '__main__':
    print('Full test suite for GRADAUX-HARMDET-COMPOSE-01')
    print('Copyright 2021-2026 The PySCF Developers. All Rights Reserved.')
    unittest.main()

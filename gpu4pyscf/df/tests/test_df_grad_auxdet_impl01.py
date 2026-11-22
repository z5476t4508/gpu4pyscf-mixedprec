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
'''GRADAUX-DET-REDUCTION-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_DF_GRAD_AUXDET knob (DEFAULT OFF) replaces, at the
single J-only gradient site (_j_energy_per_atom, df/grad/rhf.py -- the
UKS k_factor==0 route delegates there, so the RKS and UKS lanes share it
verbatim), the atomicAdd reassociation of BOTH aux sub-chain kernels --
the f32 ip1 twin's ejk/ejk_aux accumulation (sum_ejk_int3c2e_ip1_f32_det)
and the 2-center auxbasis response (e_int2c2e_ip1_det) -- with the
same-value fixed-order segmented reduction: per-block private partial
rows folded by a single-writer fixed-order reduce kernel.  M5-DRIFT-
ATTRIB-01 (entry 645) measured the two carriers' differently-reassociated
accumulations as the mechanism behind the ~1e-6 Eh/Bohr k-repeat gradient
drift at the drifted cell class (the fp64 R lane passes via correlated
cancellation, total ~1e-12); GRADAUX-F32-CANCEL-01 (entry 646) measured
that operand harmonization does NOT restore it; D1-GRAD-JONLY-AUXFP64-01
(entry 649) restores it by retreating the site to fp64 at a measured
+4.1% e2e price.  This knob keeps the f32 twin and makes the accumulation
deterministic instead.

Value-invariant class on the armed identity: the multiset of addends is
exactly the atomicAdd multiset (same staged values at the same program
points), so the armed-vs-knob-off band below is pure reassociation plus
the auxvec-pipeline wobble between separate calls, NOT a precision claim
-- the f32 rounding of the twin lane is unchanged.  The kernel-level
bitwise determinism with PINNED inputs is asserted directly (the
recorders re-invoke each det entry on the same captured ctypes operands
into fresh zero targets twice; the auxvec pipeline upstream of the site
still wobbles ~1e-13 between separate site calls, so site-level repeats
carry the same-lane band, not a bitwise claim).  The knob arms only on
the opt-in GPU4PYSCF_DF_IP1_F32 site lane (after the e649 jonly flip
decision -- if that knob reverts the site to fp64 this one stays inert),
never on the global fp32 ('auto') lane, and stays inert under the strict
fp64-reference context.  The hybrid (J+K) route never calls the site and
is untouched by construction.
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
                    # (the armed site is deterministic at the kernel level,
                    # but the auxvec pipeline upstream of the kernels still
                    # wobbles ~1e-13 and the hcore/nuclear components of the
                    # full gradient carry their own ~5e-14 class; the band is
                    # the measured same-lane draw class at water scale)
_REASSOC_BAND = 1e-9  # armed-vs-knob-off band: pure reassociation of the
                      # same addends plus the auxvec wobble between calls
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob touches (its own
    cache/counter/stats, the ip1-f32 prototype cache, the e649 jonly
    cache/counter/stats, the e646 auxresp counter/stats -- the j-only
    site updates those unconditionally on every pass, so this suite must
    restore them or it pollutes the adjacent JONLY/GRADAUX suites' stats
    assertions when several suites run in one process), the global
    precision mode and all three env variables.'''

    def setUp(self):
        self.s_knob = grad_dfrhf._AUXDET
        self.s_grads = grad_dfrhf._AUXDET_GRADS
        self.s_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.s_jl_knob = grad_dfrhf._JONLY_AUXFP64
        self.s_jl_grads = grad_dfrhf._JONLY_AUXFP64_GRADS
        self.s_jl_stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.s_aux_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.s_ip1f32 = grad_dfrhf._IP1_F32
        self.s_prec = numint.precision.get_precision()
        for var in ('GPU4PYSCF_DF_GRAD_AUXDET',
                    'GPU4PYSCF_DF_GRAD_JONLY_AUXFP64',
                    'GPU4PYSCF_DF_IP1_F32'):
            had = var in os.environ
            setattr(self, '_had_' + var, had)
            setattr(self, '_env_' + var, os.environ.get(var))

    def tearDown(self):
        grad_dfrhf._AUXDET = self.s_knob
        grad_dfrhf._AUXDET_GRADS = self.s_grads
        grad_dfrhf._AUXDET_STATS.clear()
        grad_dfrhf._AUXDET_STATS.update(self.s_stats)
        grad_dfrhf._JONLY_AUXFP64 = self.s_jl_knob
        grad_dfrhf._JONLY_AUXFP64_GRADS = self.s_jl_grads
        grad_dfrhf._JONLY_AUXFP64_STATS.clear()
        grad_dfrhf._JONLY_AUXFP64_STATS.update(self.s_jl_stats)
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_aux_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_aux_stats)
        grad_dfrhf._IP1_F32 = self.s_ip1f32
        numint.precision.set_precision(self.s_prec)
        for var in ('GPU4PYSCF_DF_GRAD_AUXDET',
                    'GPU4PYSCF_DF_GRAD_JONLY_AUXFP64',
                    'GPU4PYSCF_DF_IP1_F32'):
            if getattr(self, '_had_' + var):
                os.environ[var] = getattr(self, '_env_' + var)
            elif var in os.environ:
                del os.environ[var]

    def _set(self, knob=None, jonly=None, ip1f32=None):
        '''reset the caches to a known state and set the env vars.'''
        if knob is not None:
            grad_dfrhf._AUXDET = None
            if knob == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_AUXDET', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_AUXDET'] = knob
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
    libvhf_rys module (the same interception the GPU driver uses;
    df/grad/rhf.py resolves the attributes at call time), recording
    which entry each call hit.  A full molecular gradient drives the
    nuclear response entry (fp64) plus the J-only / J+K DF site; the
    2-center per-atom entry is hit once per auxbasis-response site.

    On every DET call the wrap ALSO re-invokes the original entry twice
    with the SAME captured ctypes operands into fresh zero targets:
    with pinned inputs the deterministic lane must reproduce its
    accumulation bitwise (the unit-level determinism claim; the site's
    own call result is untouched by the re-invocations).'''

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
        lib = self._lib
        for name, tag in (('sum_ejk_int3c2e_ip1', 'f64'),
                          ('sum_ejk_int3c2e_ip1_f32', 'f32'),
                          ('sum_ejk_int3c2e_ip1_f32_det', 'f32det')):
            orig = self._orig[name]

            def rec(*a, _orig=orig, _tag=tag, **k):
                calls.append(_tag)
                if _tag == 'f32det':
                    self._pinned_double_call(a)
                return _orig(*a, **k)

            setattr(lib, name, rec)
        return calls

    def _pinned_double_call(self, a):
        # a: (ejk, ejk_aux, dm, auxvec, n_dm, envs, shm, nbx, nby, shl,
        #     basij, ksh, gout, ao_pair, aux_off, nao, npairs, naux,
        #     natm, partial) -- re-invoke twice into fresh zero targets,
        #     reusing every captured ctypes operand (pointers to the
        #     site's live dm/auxvec buffers and the envs/offset objects)
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
    arrival-order sum).'''

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


class TestKnobParse(_KnobState):

    def test_parse_mapping(self):
        for raw, want in (('1', True), ('', False), ('0', False),
                          ('yes', False), ('2', False)):
            grad_dfrhf._AUXDET = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_AUXDET', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_AUXDET'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = grad_dfrhf._auxdet()
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
        # (the D1-ON lane verbatim: one f32 j-only entry + one f64 nuclear
        # entry per gradient; atomic 2c entry), the armed counter never
        # moves, and the lane is reproducible at run-to-run scale
        self._set(knob='0', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXDET_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with _Ip1Recorder() as calls, rec2 as c2c:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, c0)
        self.assertEqual(sorted(calls), ['f32', 'f32', 'f64', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertGreaterEqual(c2c.count('2c'), 2)
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), False)
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self.assertEqual(g1.dtype, np.float64)

    def test_armed_runs_det_lane(self):
        # knob ON + ip1-f32 prototype ON: the j-only site routes BOTH aux
        # sub-chain kernels through the deterministic entries (f32det +
        # 2cdet once per gradient; the nuclear entry stays the one f64
        # call), the counter arms once per gradient, and BOTH det entries
        # reproduce their accumulation targets BITWISE on pinned inputs
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXDET_GRADS
        rec1 = _Ip1Recorder()
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with rec1 as calls, rec2 as c2c:
            g_on1 = mf.nuc_grad_method().kernel()
            g_on2 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - c0, 2)
        self.assertEqual(sorted(calls), ['f32det', 'f32det', 'f64', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 2)
        self.assertEqual(c2c.count('2c'), 0)
        stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['natm'], mol_sph.natm)
        self.assertGreater(stats['nblocks'], 0)
        # pinned-input bitwise determinism of the 3c f32 det entry: the
        # two re-invocations on the same captured operands must agree
        # bitwise on BOTH accumulation targets
        import cupy as cp
        (e_a, ea_a), (e_b, ea_b) = rec1.det_pinned
        self.assertTrue(bool(cp.array_equal(e_a, e_b)))
        self.assertTrue(bool(cp.array_equal(ea_a, ea_b)))
        # ... and of the 2c det entry, with the atomic entry on the same
        # pinned operands agreeing at reassociation scale
        d1, d2, d_atomic = rec2.det_pinned
        self.assertTrue(bool(cp.array_equal(d1, d2)))
        self.assertLessEqual(float(cp.abs(d1 - d_atomic).max()), 1e-12)
        # same-lane repeat stays inside the full-gradient band (the site's
        # own aux sub-chain is deterministic; hcore/auxvec wobble remains)
        self.assertLessEqual(abs(g_on1 - g_on2).max(), _ULP_BAND)
        # the armed lane reproduces the knob-off f32-twin lane at the
        # reassociation scale (same addends, fixed order)
        self._set(knob='0', ip1f32='1')
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on1 - g_off).max(), _REASSOC_BAND)
        self.assertEqual(g_on1.dtype, np.float64)

    def test_armed_uks_delegated_site(self):
        # UKS k_factor==0 delegates to the SAME _j_energy_per_atom site:
        # the knob arms once per gradient on the delegated route and both
        # det entries fire
        self._set(knob='1', ip1f32='1')
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        c0 = grad_dfrhf._AUXDET_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_uks.natm)
        with _Ip1Recorder() as calls, rec2 as c2c:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - c0, 1)
        self.assertEqual(sorted(calls), ['f32det', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 1)
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), True)
        self._set(knob='0', ip1f32='1')
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _REASSOC_BAND)
        self.assertLessEqual(abs(g_on - g_off).max(), _GRAD_GATE)

    def test_hybrid_route_untouched(self):
        # hybrid functional -> _jk_energy_per_atom: the J+K site keeps
        # the f32 twin under the prototype (its own per-batch selection
        # line is untouched), the j-only counter never moves, the
        # nuclear entry stays the one f64 call and the 2c entry stays
        # the atomic one.  The stats are compared against a snapshot
        # taken inside this test (the hybrid route never passes through
        # the j-only site): the dict may legitimately carry a stale
        # armed=False from an earlier suite's j-only gradients when
        # several suites share one process.
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph, xc='b3lyp')
        c0 = grad_dfrhf._AUXDET_GRADS
        stats0 = dict(grad_dfrhf._AUXDET_STATS)
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with _Ip1Recorder() as calls, rec2 as c2c:
            g_hyb = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, c0)
        self.assertGreaterEqual(calls.count('f32'), 1)
        self.assertEqual(calls.count('f64'), 1)
        self.assertEqual(calls.count('f32det'), 0)
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertEqual(grad_dfrhf._AUXDET_STATS, stats0)
        self._set(knob='0', ip1f32='1')
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_hyb - g_ref).max(), _ULP_BAND)

    def test_strict_reference_inert(self):
        # strict fp64-reference context: the knob is forced inert (the
        # site keeps the f32 twin under the prototype; the atomic 2c
        # entry), no counter movement, and the result reproduces the
        # prototype lane
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXDET_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with _Ip1Recorder() as calls, rec2 as c2c:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, c0)
        self.assertEqual(sorted(calls), ['f32', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), False)
        self._set(knob='0', ip1f32='1')
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_ref).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_global_fp32_policy_inert(self):
        # the global fp32 ('auto' policy) lane: the knob never fires (the
        # policy's site selection is not the prototype's to reroute), the
        # counter never moves
        numint.precision.set_precision('fp32')
        self._set(knob='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXDET_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with _Ip1Recorder() as calls, rec2:
            _g_fp32 = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, c0)
        self.assertIn('f32', calls)
        self.assertNotIn('f32det', calls)
        stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertIs(stats.get('site_fp32'), True)

    def test_jonly_flip_makes_auxdet_inert(self):
        # layering with the e649 knob: when jonly-auxfp64 reverts the
        # site to the deployed fp64 kernel, the auxdet arming sees the
        # realized site_fp32=False and stays inert (the jonly lane is a
        # different repair; the two knobs never compose)
        self._set(knob='1', jonly='1', ip1f32='1')
        mf = self._converged(mol_sph)
        c0 = grad_dfrhf._AUXDET_GRADS
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with _Ip1Recorder() as calls, rec2 as c2c:
            _g_jl = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._AUXDET_GRADS, c0)
        self.assertEqual(sorted(calls), ['f64', 'f64'])
        self.assertEqual(c2c.count('2cdet'), 0)
        stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertIs(stats.get('site_fp32'), False)


class TestAuxdetMechanics(_KnobState):
    '''the site called DIRECTLY (one jk_energy_per_atom(k_factor=0) call,
    no SCF): the dispatch, the pinned-input bitwise determinism of both
    det entries, and the counter/stat geometry.'''

    def setUp(self):
        super().setUp()
        global mol_sph
        self._set(knob='0', jonly='', ip1f32='')
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
        # k_factor=0 hits the j-only site: knob ON + prototype ON routes
        # the 3c kernel to the det entry; knob OFF + prototype ON keeps
        # the f32 twin entry
        self._set(knob='1', ip1f32='1')
        c0 = grad_dfrhf._AUXDET_GRADS
        with _Ip1Recorder() as calls:
            de_on = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                              k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._AUXDET_GRADS - c0, 1)
        self.assertEqual(calls, ['f32det'])
        stats = dict(grad_dfrhf._AUXDET_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['natm'], mol_sph.natm)
        self.assertGreater(stats['naux'], 0)
        self.assertGreater(stats['nblocks'], 0)
        self._set(knob='0', ip1f32='1')
        with _Ip1Recorder() as calls:
            de_f32 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                               k_factor=0, hermi=1)
        self.assertEqual(calls, ['f32'])
        self.assertIs(grad_dfrhf._AUXDET_STATS.get('armed'), False)
        # same addends, fixed order: the armed J derivative agrees with
        # the f32-twin lane at the reassociation scale
        self.assertLessEqual(abs(de_on - de_f32).max(), _REASSOC_BAND)
        # ... and both stay inside the project gate of the plain fp64 lane
        self._set(knob='0', ip1f32='')
        grad_dfrhf._IP1_F32 = False
        de_fp64 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                            k_factor=0, hermi=1)
        self.assertLess(abs(de_f32 - de_fp64).max(), _GRAD_GATE)
        self.assertLess(abs(de_on - de_fp64).max(), _GRAD_GATE)

    def test_armed_site_repeat_band(self):
        # two DIRECT armed calls at the pinned dm: the aux sub-chain
        # kernels are deterministic (their inputs are re-derived from the
        # same dm), but the auxvec contract pipeline upstream of the site
        # still wobbles ~1e-13 between separate calls, so the site output
        # carries the same-lane band, not a bitwise claim (the pinned-input
        # bitwise claim lives in test_armed_runs_det_lane / the mechanics
        # tests below)
        self._set(knob='1', ip1f32='1')
        de1 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                        k_factor=0, hermi=1)
        de2 = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                        k_factor=0, hermi=1)
        self.assertLessEqual(abs(de1 - de2).max(), _ULP_BAND)

    def test_det_entries_pinned_inputs(self):
        # unit-level determinism of BOTH det entries with PINNED inputs,
        # driven through a direct armed site call: the recorders
        # re-invoke each det entry twice on the same captured ctypes
        # operands into fresh zero targets (and the 2c atomic entry once)
        self._set(knob='1', ip1f32='1')
        import cupy as cp
        rec1 = _Ip1Recorder()
        rec2 = _C2cRecorder()
        rec2.set_shapes(mol_sph.natm)
        with rec1 as calls, rec2 as c2c:
            de = self.g.jk_energy_per_atom(self.dm, j_factor=1,
                                           k_factor=0, hermi=1)
        self.assertEqual(calls, ['f32det'])
        self.assertEqual(c2c, ['2cdet'])
        (e_a, ea_a), (e_b, ea_b) = rec1.det_pinned
        self.assertTrue(bool(cp.array_equal(e_a, e_b)))
        self.assertTrue(bool(cp.array_equal(ea_a, ea_b)))
        # the 3c f32 det re-invocation is finite and shaped
        self.assertEqual(e_a.shape, (mol_sph.natm, 3))
        self.assertTrue(bool(cp.isfinite(e_a).all()))
        d1, d2, d_atomic = rec2.det_pinned
        self.assertTrue(bool(cp.array_equal(d1, d2)))
        self.assertLessEqual(float(cp.abs(d1 - d_atomic).max()), 1e-12)
        self.assertEqual(de.dtype, np.float64)


if __name__ == '__main__':
    print('Full test suite for GRADAUX-DET-REDUCTION-01')
    print('Copyright 2021-2026 The PySCF Developers. All Rights Reserved.')
    unittest.main()

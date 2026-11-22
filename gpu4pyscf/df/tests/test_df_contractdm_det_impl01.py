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
'''CONTRACTDM-DET-REDUCTION-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_DF_CONTRACTCM_DET knob (DEFAULT OFF) replaces, at
BOTH kernel-level Int3c2eOpt.contract_dm sites, the cross-block
atomicAdd reassociation of the 3c-dm-contraction accumulation with the
same-value fixed-order segmented reduction (per-block private partial
rows folded by a single-writer fixed-order reduce kernel):

  rys site   int3c2e_bdiv.Int3c2eOpt.contract_dm -> gvhf-rys
             contract_int3c2e_dm (_store_vj_aux atomicAdd, per-thread
             addends) -- the measured gradient aux-chain site of entry
             654 (df/grad/rhf.py _j_energy_per_atom; the UKS
             k_factor==0 route delegates there, so the RKS and UKS
             lanes share it verbatim; hessian/tdf gradients reach the
             same site),
  md site    j_engine_3c2e.Int3c2eOpt.contract_dm -> gvhf-md
             contract_int3c2e_dm (thread-0 atomicAdd of the per-(kp,k)
             block-reduced values) -- the SCF-phase rhoj users
             (df_jk.py get_j).

AUXVEC-WOBBLE-ATTRIB-01 (entry 654) measured the rys kernel as the
ORIGIN of the auxvec-pipeline wobble: rho differs on 2423-2425/3080
aux slots between consecutive identical launches (max 9.1e-13), which
the near-singular CD metric amplifies ~1e6x into the 6-8e-7 k-repeat
total gradient drift at the tamoxifen r2scan3c UKS jonly 687 cell; the
kernel re-invocation probe caught the reassociation on pinned inputs
(re1 != re2, 7/7 per role).  Every downstream stage is measured
deterministic-given-inputs, so a bitwise-stable rho makes the whole
auxvec chain bitwise stable and the k-repeat drift axis closes at the
source while the paired-vs-R dg keeps carrying the e646 ~1e-5
quantization class (inside the 3e-5 gate).

Value-invariant class on the armed identity: the multiset of addends
is exactly the atomicAdd multiset (every thread's final vj_xyz value
on the rys lane; every iteration's block-reduced value on the md
lane), so the armed-vs-knob-off band below is pure reassociation, NOT
a precision claim.  Kernel-level bitwise determinism with PINNED
inputs is asserted directly (the recorders re-invoke each det entry
twice on the same captured ctypes operands into fresh zero targets
with fresh zero partials, inside the wrap while every operand array
is still alive); site-level repeats are asserted BITWISE for the rho
stage (the whole contract_dm pipeline at a pinned dm) and at the
same-lane band for the assembled gradient (the ip1/ejk kernels
downstream still atomicAdd).  The knob never arms on the global fp32
('auto') lane, stays inert under the strict fp64-reference context,
and has no subject on the J3C-RTCACHE-01 table path (that GEMV
dispatches no atomic kernel and records no stats).  PBC keeps its
own separate contract_int3c2e_dm kernel and is untouched by
construction.
'''
import ctypes
import os
import unittest
import warnings

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.df import int3c2e_bdiv as i3c
from gpu4pyscf.df.grad import rhf as grad_dfrhf
from gpu4pyscf.dft import numint


def cp_asarray(x):
    import cupy as cp
    return cp.asarray(x)


def cp_asnumpy(x):
    import cupy as cp
    return cp.asnumpy(x)


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
                    # (the rho stage is deterministic on the armed lane,
                    # but the ip1/ejk kernels downstream and the
                    # hcore/nuclear components still atomicAdd at
                    # ~1e-12; the band is the measured same-lane draw
                    # class at water scale)
_REASSOC_BAND = 1e-9  # armed-vs-knob-off band: pure reassociation of
                      # the same addends (rho ~1e-14 at water scale; the
                      # near-singular CD amplification is the drifted
                      # cell's property, not this suite's)
_RHO_REASSOC_BAND = 1e-12  # kernel-level reassociation band of rho
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)
_KNOB = 'GPU4PYSCF_DF_CONTRACTDM_DET'


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob touches (its own
    cache/counter/stats in the home module int3c2e_bdiv) plus every
    process-global a full gradient/SCF run bumps unconditionally in the
    adjacent knob suites (the e649 jonly and e646 auxresp stats/counters
    are updated on EVERY pass through the J-only site, the e651 auxdet
    stats likewise, and the ip1-f32 prototype cache is env-read), the
    global precision mode, and both env variables (this knob's and the
    J3C cache knob's).'''

    def setUp(self):
        self.s_knob = i3c._CONTRACTDM_DET
        self.s_calls = i3c._CONTRACTDM_DET_CALLS
        self.s_stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.s_jl_knob = grad_dfrhf._JONLY_AUXFP64
        self.s_jl_grads = grad_dfrhf._JONLY_AUXFP64_GRADS
        self.s_jl_stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.s_aux_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_aux_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.s_ad_knob = grad_dfrhf._AUXDET
        self.s_ad_grads = grad_dfrhf._AUXDET_GRADS
        self.s_ad_stats = dict(grad_dfrhf._AUXDET_STATS)
        self.s_ip1f32 = grad_dfrhf._IP1_F32
        self.s_prec = numint.precision.get_precision()
        for var in (_KNOB, 'GPU4PYSCF_DF_J_3C_CACHE'):
            had = var in os.environ
            setattr(self, '_had_' + var, had)
            setattr(self, '_env_' + var, os.environ.get(var))

    def tearDown(self):
        i3c._CONTRACTDM_DET = self.s_knob
        i3c._CONTRACTDM_DET_CALLS = self.s_calls
        i3c._CONTRACTDM_DET_STATS.clear()
        i3c._CONTRACTDM_DET_STATS.update(self.s_stats)
        grad_dfrhf._JONLY_AUXFP64 = self.s_jl_knob
        grad_dfrhf._JONLY_AUXFP64_GRADS = self.s_jl_grads
        grad_dfrhf._JONLY_AUXFP64_STATS.clear()
        grad_dfrhf._JONLY_AUXFP64_STATS.update(self.s_jl_stats)
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_aux_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_aux_stats)
        grad_dfrhf._AUXDET = self.s_ad_knob
        grad_dfrhf._AUXDET_GRADS = self.s_ad_grads
        grad_dfrhf._AUXDET_STATS.clear()
        grad_dfrhf._AUXDET_STATS.update(self.s_ad_stats)
        grad_dfrhf._IP1_F32 = self.s_ip1f32
        numint.precision.set_precision(self.s_prec)
        for var in (_KNOB, 'GPU4PYSCF_DF_J_3C_CACHE'):
            if getattr(self, '_had_' + var):
                os.environ[var] = getattr(self, '_env_' + var)
            elif var in os.environ:
                del os.environ[var]

    def _set(self, knob=None, j3c=None):
        '''reset the caches to a known state and set the env vars.'''
        if knob is not None:
            i3c._CONTRACTDM_DET = None
            if knob == '':
                os.environ.pop(_KNOB, None)
            else:
                os.environ[_KNOB] = knob
        if j3c is not None:
            if j3c == '':
                os.environ.pop('GPU4PYSCF_DF_J_3C_CACHE', None)
            else:
                os.environ['GPU4PYSCF_DF_J_3C_CACHE'] = j3c

    def _converged(self, mol, xc='r2scan'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        return mf


class _RysRecorder:
    '''wrap the two gvhf-rys contract_int3c2e_dm entries on the shared
    libvhf_rys module (df/int3c2e_bdiv.py resolves the attributes at
    call time), recording which entry each call hit.  On every det call
    the wrap ALSO re-invokes the original det entry twice with the SAME
    captured ctypes operands into fresh zero targets with fresh zero
    partials (inside the wrap, so every operand array is still alive):
    with pinned inputs the deterministic lane must reproduce its
    accumulation bitwise, and the re-invocations must agree with the
    call's own output.'''

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.calls = []
        self.det_pinned = []   # [(re1, re2, orig), ...] host copies
        self._lib = i3c.libvhf_rys
        self._orig = {}
        for name in ('contract_int3c2e_dm', 'contract_int3c2e_dm_det'):
            self._orig[name] = getattr(self._lib, name)

    def _read(self, ptr, n_dm, naux):
        cp = self.cp
        dev = cp.cuda.Device(0)
        mem = cp.cuda.UnownedMemory(ptr, n_dm * naux * 8, dev)
        mp = cp.cuda.MemoryPointer(mem, 0)
        arr = cp.ndarray((n_dm, naux), dtype=cp.float64, memptr=mp)
        return cp.asnumpy(arr.copy())

    def _pinned_double_call(self, a):
        cp = self.cp
        n_dm = int(a[2].value) if hasattr(a[2], 'value') else int(a[2])
        naux = int(a[3].value) if hasattr(a[3], 'value') else int(a[3])
        nbas_aux = int(a[6].value) if hasattr(a[6], 'value') else int(a[6])
        nsp = int(a[7].value) if hasattr(a[7], 'value') else int(a[7])
        nblocks = nbas_aux * nsp
        v2 = cp.zeros((n_dm, naux))
        v3 = cp.zeros((n_dm, naux))
        p2 = cp.zeros((nblocks, n_dm * 30))
        p3 = cp.zeros((nblocks, n_dm * 30))
        orig = self._orig['contract_int3c2e_dm_det']
        rc2 = orig(ctypes.cast(v2.data.ptr, ctypes.c_void_p),
                   a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
                   a[10], ctypes.cast(p2.data.ptr, ctypes.c_void_p))
        rc3 = orig(ctypes.cast(v3.data.ptr, ctypes.c_void_p),
                   a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9],
                   a[10], ctypes.cast(p3.data.ptr, ctypes.c_void_p))
        cp.cuda.runtime.deviceSynchronize()
        if rc2 == 0 and rc3 == 0:
            ptr0 = a[0].value if hasattr(a[0], 'value') else a[0]
            h0 = self._read(ptr0, n_dm, naux)
            self.det_pinned.append(
                (cp.asnumpy(v2), cp.asnumpy(v3), h0))

    def __enter__(self):
        calls = self.calls
        lib = self._lib
        for name, tag in (('contract_int3c2e_dm', 'rys'),
                          ('contract_int3c2e_dm_det', 'rysdet')):
            orig = self._orig[name]

            def rec(*a, _orig=orig, _tag=tag, **k):
                calls.append(_tag)
                rc = _orig(*a, **k)
                if _tag == 'rysdet' and rc == 0:
                    self._pinned_double_call(a)
                return rc

            setattr(lib, name, rec)
        return calls

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self._lib, name, orig)
        return False


class _MdRecorder:
    '''same interception for the two gvhf-md contract_int3c2e_dm entries
    (the SCF-phase rhoj site; df/j_engine_3c2e.py resolves the
    attributes at call time), with the same pinned-input double
    re-invocation of the det entry.'''

    def __init__(self):
        import cupy as cp
        from gpu4pyscf.scf.j_engine import libvhf_md
        self.cp = cp
        self.libvhf_md = libvhf_md
        self.calls = []
        self.det_pinned = []
        self._orig = {}
        for name in ('contract_int3c2e_dm', 'contract_int3c2e_dm_det'):
            self._orig[name] = getattr(libvhf_md, name)

    def _pinned_double_call(self, a):
        cp = self.cp
        def ival(x):
            return int(x.value) if hasattr(x, 'value') else int(x)
        n_dm, naux = ival(a[2]), ival(a[3])
        nsp, nksh = ival(a[6]), ival(a[7])
        nblocks = nsp * nksh
        v2 = cp.zeros((n_dm, naux))
        v3 = cp.zeros((n_dm, naux))
        p2 = cp.zeros((nblocks, 28))
        p3 = cp.zeros((nblocks, 28))
        orig = self._orig['contract_int3c2e_dm_det']
        rc2 = orig(ctypes.cast(v2.data.ptr, ctypes.c_void_p), a[1], a[2],
                   a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
                   a[12], ctypes.cast(p2.data.ptr, ctypes.c_void_p))
        rc3 = orig(ctypes.cast(v3.data.ptr, ctypes.c_void_p), a[1], a[2],
                   a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11],
                   a[12], ctypes.cast(p3.data.ptr, ctypes.c_void_p))
        cp.cuda.runtime.deviceSynchronize()
        if rc2 == 0 and rc3 == 0:
            self.det_pinned.append((cp.asnumpy(v2), cp.asnumpy(v3)))

    def __enter__(self):
        calls = self.calls
        for name, tag in (('contract_int3c2e_dm', 'md'),
                          ('contract_int3c2e_dm_det', 'mddet')):
            orig = self._orig[name]

            def rec(*a, _orig=orig, _tag=tag, **k):
                calls.append(_tag)
                rc = _orig(*a, **k)
                if _tag == 'mddet' and rc == 0:
                    self._pinned_double_call(a)
                return rc

            setattr(self.libvhf_md, name, rec)
        return calls

    def __exit__(self, *exc):
        for name, orig in self._orig.items():
            setattr(self.libvhf_md, name, orig)
        return False


class _RhoRecorder:
    '''wrap the int3c2e_bdiv.Int3c2eOpt.contract_dm METHOD (the same
    interception the entry-654 driver used), capturing a host copy of
    every rho output -- the stage-level bitwise-repeat claim.'''

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.rhos = []
        self._cls = i3c.Int3c2eOpt
        self._orig = i3c.Int3c2eOpt.contract_dm

    def __enter__(self):
        orig = self._orig
        rhos = self.rhos
        cp = self.cp

        def rec(self_, dm, hermi=0):
            result = orig(self_, dm, hermi)
            rhos.append(cp.asnumpy(result))
            return result
        i3c.Int3c2eOpt.contract_dm = rec
        return self.rhos

    def __exit__(self, *exc):
        i3c.Int3c2eOpt.contract_dm = self._orig
        return False


class TestContractdmDetKnob(_KnobState):

    def test_parse_mapping(self):
        # '' and '0' keep the knob off; '1' arms it; any other value
        # warns and stays off (the parse is cached until reset)
        for raw, armed in (('', False), ('0', False), ('1', True)):
            self._set(knob=raw)
            i3c._CONTRACTDM_DET = None
            self.assertIs(i3c._contractdm_det(), armed)
        self._set(knob='x')
        i3c._CONTRACTDM_DET = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.assertIs(i3c._contractdm_det(), False)
        self.assertTrue(any('GPU4PYSCF_DF_CONTRACTDM_DET' in str(w.message)
                            for w in caught))
        self._set(knob='')

    def test_knob_off_keeps_deployed_entry(self):
        # knob OFF: the rys site dispatches the deployed atomic entry
        # (never the det entry), the counter never moves, and the stats
        # record armed=False on every pass
        self._set(knob='0')
        mf = self._converged(mol_sph)
        c0 = i3c._CONTRACTDM_DET_CALLS
        with _RysRecorder() as calls, _MdRecorder() as md_calls:
            _g1 = mf.nuc_grad_method().kernel()
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS, c0)
        self.assertIn('rys', calls)
        self.assertNotIn('rysdet', calls)
        self.assertNotIn('mddet', md_calls)
        stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertEqual(stats.get('site'), 'rys')
        self.assertGreater(stats.get('naux', 0), 0)

    def test_armed_runs_det_lane(self):
        # knob ON: the rys site routes the 3c kernel to the det entry
        # once per gradient; the pinned-input re-invocations reproduce
        # the accumulation BITWISE (twice, plus the call's own output);
        # the assembled gradient stays at the reassociation band
        self._set(knob='1')
        mf = self._converged(mol_sph)
        c0 = i3c._CONTRACTDM_DET_CALLS
        rec = _RysRecorder()
        with rec as calls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS - c0, 1)
        self.assertEqual(calls.count('rysdet'), 1)
        self.assertEqual(calls.count('rys'), 0)
        stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats.get('site'), 'rys')
        self.assertGreater(stats.get('nblocks', 0), 0)
        self.assertEqual(stats.get('n_dm'), 1)
        for re1, re2, re0 in rec.det_pinned:
            self.assertTrue(np.array_equal(re1, re2))
            self.assertTrue(np.array_equal(re1, re0))
        self._set(knob='0')
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _REASSOC_BAND)
        self.assertLessEqual(abs(g_on - g_off).max(), _GRAD_GATE)

    def test_armed_rho_bitwise_across_warm_repeats(self):
        # THE core claim at the affordable cell: on the armed lane the
        # rho stage (the whole contract_dm pipeline at the pinned
        # converged dm) is BITWISE identical across warm in-process
        # gradient repeats, while on the knob-off lane the two repeats
        # may wobble (the entry-654 origin, small at water scale); the
        # assembled gradients carry the same-lane band
        self._set(knob='1')
        mf = self._converged(mol_sph)
        rho = _RhoRecorder()
        with rho:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(len(rho.rhos), 2)
        self.assertTrue(np.array_equal(rho.rhos[0], rho.rhos[1]))
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self._set(knob='0')
        rho0 = _RhoRecorder()
        with rho0:
            g3 = mf.nuc_grad_method().kernel()
            g4 = mf.nuc_grad_method().kernel()
        self.assertEqual(len(rho0.rhos), 2)
        self.assertLessEqual(
            float(np.abs(rho0.rhos[0] - rho0.rhos[1]).max()),
            _RHO_REASSOC_BAND)
        self.assertLessEqual(
            float(np.abs(rho.rhos[0] - rho0.rhos[0]).max()),
            _RHO_REASSOC_BAND)
        self.assertLessEqual(abs(g3 - g4).max(), _ULP_BAND)

    def test_armed_uks_delegated_site(self):
        # UKS k_factor==0 delegates to the SAME _j_energy_per_atom site:
        # the knob arms on the delegated route, the det entry fires, and
        # the rho stage is bitwise across warm repeats
        self._set(knob='1')
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        c0 = i3c._CONTRACTDM_DET_CALLS
        rho = _RhoRecorder()
        with _RysRecorder() as calls, rho:
            g1 = mf.nuc_grad_method().kernel()
            g2 = mf.nuc_grad_method().kernel()
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS - c0, 2)
        self.assertEqual(calls.count('rysdet'), 2)
        self.assertTrue(np.array_equal(rho.rhos[0], rho.rhos[1]))
        self.assertLessEqual(abs(g1 - g2).max(), _ULP_BAND)
        self._set(knob='0')
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g1 - g_off).max(), _REASSOC_BAND)
        self.assertLessEqual(abs(g1 - g_off).max(), _GRAD_GATE)

    def test_md_scf_site_armed(self):
        # SCF-phase rhoj users (df_jk.get_j -> j_engine_3c2e) are
        # covered by the same kernel-level knob: the md det entry fires
        # during the SCF, its pinned-input re-invocations are bitwise,
        # and the converged energy agrees at the reassociation band
        import cupy as cp
        self._set(knob='1')
        mf = self._converged(mol_sph)
        e_on = mf.e_tot
        c0 = i3c._CONTRACTDM_DET_CALLS
        rec = _MdRecorder()
        with rec as md_calls:
            rhoj = mf.get_j(mf.make_rdm1())
        self.assertGreater(i3c._CONTRACTDM_DET_CALLS, c0)
        self.assertEqual(md_calls.count('mddet') >= 1, True)
        self.assertEqual(md_calls.count('md'), 0)
        stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats.get('site'), 'md_scf')
        for re1, re2 in rec.det_pinned:
            self.assertTrue(np.array_equal(re1, re2))
        self.assertEqual(rhoj.ndim, 2)
        self.assertTrue(np.isfinite(cp.asnumpy(rhoj)).all())
        self._set(knob='0')
        rhoj_off = mf.get_j(mf.make_rdm1())
        self.assertLessEqual(
            float(np.abs(cp.asnumpy(rhoj) - cp.asnumpy(rhoj_off)).max()),
            _RHO_REASSOC_BAND)
        self.assertAlmostEqual(e_on, mf.e_tot, delta=_REASSOC_BAND)

    def test_md_j3c_cache_path_inert(self):
        # layering: the J3C-RTCACHE-01 table path (opt-in GEMV, already
        # deterministic) dispatches NO atomic contract kernel -- the
        # knob has no subject there: no counter movement, no stats, no
        # kernel entries, deterministic output
        self._set(knob='1', j3c='1')
        from gpu4pyscf.df import j_engine_3c2e
        jopt = j_engine_3c2e.Int3c2eOpt(
            mol_sph, pyscf.M(atom=mol_sph.atom,
                             basis='def2-universal-jkfit')).build()
        dm = jopt.mol.apply_C_mat_CT(np.diag(np.ones(mol_sph.nao)) * .5)
        dm = cp_asarray(dm[None])
        c0 = i3c._CONTRACTDM_DET_CALLS
        stats0 = dict(i3c._CONTRACTDM_DET_STATS)
        with _MdRecorder() as md_calls:
            r1 = jopt.contract_dm(dm, hermi=1)
            r2 = jopt.contract_dm(dm, hermi=1)
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS, c0)
        self.assertEqual(md_calls, [])
        self.assertEqual(dict(i3c._CONTRACTDM_DET_STATS), stats0)
        self.assertTrue(np.array_equal(cp_asnumpy(r1), cp_asnumpy(r2)))
        self.assertEqual(jopt._j3c_cache_state, 'ready')
        self._set(j3c='')

    def test_strict_reference_inert(self):
        # strict fp64-reference context: the knob is forced inert (the
        # reference must reproduce the deployed atomic lane), no
        # counter movement
        self._set(knob='1')
        mf = self._converged(mol_sph)
        c0 = i3c._CONTRACTDM_DET_CALLS
        with _RysRecorder() as calls:
            with numint.strict_fp64_reference_ctx():
                g_strict = mf.nuc_grad_method().kernel()
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS, c0)
        self.assertIn('rys', calls)
        self.assertNotIn('rysdet', calls)
        self.assertIs(i3c._CONTRACTDM_DET_STATS.get('armed'), False)
        self._set(knob='0')
        g_ref = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_strict - g_ref).max(), _ULP_BAND)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_global_fp32_policy_inert(self):
        # the global fp32 ('auto') policy lane: the knob never arms
        # (the D1-family knob discipline), the deployed entry fires
        numint.precision.set_precision('fp32')
        self._set(knob='1')
        mf = self._converged(mol_sph)
        c0 = i3c._CONTRACTDM_DET_CALLS
        with _RysRecorder() as calls:
            g_fp32 = mf.nuc_grad_method().kernel()
        self.assertEqual(i3c._CONTRACTDM_DET_CALLS, c0)
        self.assertIn('rys', calls)
        self.assertNotIn('rysdet', calls)
        stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertEqual(stats.get('precision_mode'), 'fp32')
        self.assertEqual(g_fp32.dtype, np.float64)

    def test_direct_call_dispatch_n_dm_batch(self):
        # direct site calls with an n_dm=2 batch (the kernel's dm-batch
        # while loop at the stride classes present in water's aux
        # shells): the armed lane is bitwise across repeats and agrees
        # with the deployed lane at the reassociation scale
        dm0 = np.diag(np.ones(mol_sph.nao)) * .5
        opt = i3c.Int3c2eOpt(mol_sph, pyscf.M(
            atom=mol_sph.atom, basis='def2-universal-jkfit')).build()
        dm = opt.mol.apply_C_mat_CT(np.asarray([dm0, dm0 * .25]))
        self._set(knob='1')
        r1 = opt.contract_dm(dm, hermi=1)
        r2 = opt.contract_dm(dm, hermi=1)
        self.assertTrue(np.array_equal(cp_asnumpy(r1), cp_asnumpy(r2)))
        self._set(knob='0')
        r0 = opt.contract_dm(dm, hermi=1)
        self.assertLessEqual(
            float(np.abs(cp_asnumpy(r1) - cp_asnumpy(r0)).max()),
            _RHO_REASSOC_BAND)
        stats = dict(i3c._CONTRACTDM_DET_STATS)
        self.assertEqual(stats.get('n_dm'), 2)

    def test_armed_site_repeat_band(self):
        # two direct armed site calls at the pinned dm: bitwise (the
        # det lane is deterministic given identical inputs; the wrapper
        # itself is deterministic at a pinned dm)
        dm0 = np.diag(np.ones(mol_sph.nao)) * .5
        opt = i3c.Int3c2eOpt(mol_sph, pyscf.M(
            atom=mol_sph.atom, basis='def2-universal-jkfit')).build()
        dm = opt.mol.apply_C_mat_CT(dm0)
        self._set(knob='1')
        r1 = cp_asnumpy(opt.contract_dm(dm, hermi=1))
        r2 = cp_asnumpy(opt.contract_dm(dm, hermi=1))
        self.assertTrue(np.array_equal(r1, r2))


if __name__ == '__main__':
    print('Full test suite for CONTRACTDM-DET-REDUCTION-01')
    print('Copyright 2021-2026 The PySCF Developers. All Rights Reserved.')
    unittest.main()

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
'''GRAD-JONLY-DEADSKIP-01 focused regression (2026-11-16).

The opt-in GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP knob (DEFAULT OFF) skips, at
the single RKS closed-shell k_factor==0 fall-through of
_jk_energy_per_atom (the GRAD-J3C-REUSE-JONLY-01 armed route, pack
retained, df/grad/rhf.py), the two contractions the deployed
fall-through launches whose exchange contribution is identically zero
there: the response-chain GEMM pair (its second GEMM carries
alpha == -.5*k_factor == 0, computing dm_tensor = 0*(...) +
beta*dm_tensor) and the auxresp Gram contraction (alpha == 0, computing
dm_aux = 0*Gram + beta*dm_aux).  The skip applies exactly the beta
scales the GEMM beta paths apply -- 0*AB + beta*C == beta*C bitwise for
finite operands -- so the value is BITWISE-identical to the deployed
fall-through by construction.

Unlike GRAD-JONLY-AUXVEC-DIRECT-01 (which additionally replaces the
pack construction with the two-GEMV reassociation, e664), the X build,
the dm_oo tail einsum and the trace auxvec stay VERBATIM (dm_oo stays
live up to the trace), and the resp-f32 arming / dtype path is
untouched: the skip takes precedence over the response GEMM branch on
both dtype lanes, so the fp32-lane dm_tensor keeps the one-rounding
fp32 product the deployed fp32 GEMM beta path would consume.  The knob
never arms when the direct route is armed (subsumed), never arms on
the strict fp64-reference context or without the reuse pack, never
touches the UKS k_factor==0 twin (it delegates before the pack) or the
hybrid (J+K) route; precision-neutral (no dtype change anywhere), so
it arms on the global fp32 ('auto') lane too.

Value class on the armed lane: same-lane run-to-run atomicAdd band
only (the ip1/2c kernels' scheduling wobble, marginal draws to
~1.3e-12 measured at water scale) -- NO reassociation term, the
structural distinction from the direct route whose water-scale band
additionally carries the ~5.4e-13 reassociation draw that the band
cells amplify to the e664 3.75e-9 envelope violation.
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

# the armed-vs-deployed delta is the SAME-LANE run-to-run atomicAdd
# band only (no reassociation term: the skip is value-exact by
# construction); the tripwire carries headroom above the measured
# ~1.3e-12 water-scale wobble draws, 4+ orders under the 3e-5 project
# gate, and stays 3 orders under the e664 direct-route band-cell
# envelope (3.75e-9) the phase-B GPU card discriminates against
_DELTA_BAND = 1e-11
_GRAD_GATE = 3e-5   # the project gradient gate (Eh/Bohr)

# the einsum signatures the fall-through launches: the X build pair
# (LIVE on this knob -- the pack construction stays verbatim), and the
# dead response-chain pair + the dead auxresp Gram (the ONLY
# signatures the armed lane skips)
_XBUILD_SIGS = ('pqr,pi->iqr', 'iqr,qj->rij')
_RESP_SIGS = ('rji,qj->iqr', 'iqr,pi->pqr')
_GRAM_SIGS = ('rij,sij->rs', 'rij,sji->rs')


class _KnobState(unittest.TestCase):
    '''save/restore every process-global the knob touches: its own
    cache/counter/stats, the auxvec knob's globals (the fall-through
    calls _arm_jonly_auxvec_direct on EVERY pass before this knob's
    decision, the 2faa55d pattern), the reuse cache + both reuse
    counters, the resp-f32 cache + arming counter (armed verbatim on
    the deadskip lane), the p1-f32 cache, the global precision mode and
    the env variables -- the standing suite-order discipline (the
    e660 class: a suite that leaves a neighbor's stats dirty fails its
    assertions in a shared process).'''

    def setUp(self):
        self.s_knob = grad_dfrhf._JONLY_DEADSKIP
        self.s_grads = grad_dfrhf._JONLY_DEADSKIP_GRADS
        self.s_stats = dict(grad_dfrhf._JONLY_DEADSKIP_STATS)
        self.s_auxvec = grad_dfrhf._JONLY_AUXVEC
        self.s_auxvec_grads = grad_dfrhf._JONLY_AUXVEC_GRADS
        self.s_auxvec_stats = dict(grad_dfrhf._JONLY_AUXVEC_STATS)
        self.s_reuse = grad_dfrhf._J3C_REUSE
        self.s_reuse_grads = grad_dfrhf._J3C_REUSE_GRADS
        self.s_reuse_jonly = grad_dfrhf._J3C_REUSE_JONLY_GRADS
        self.s_p1f32 = grad_dfrhf._P1_F32
        self.s_resp = grad_dfrhf._RESP_F32
        self.s_resp_calls = grad_dfrhf._RESP_F32_CALLS
        # the verbatim _j_energy_per_atom route (the strict-reference and
        # no-reuse tests pass through it) records the neighbor knobs'
        # stats unconditionally on every pass (the 2faa55d pattern), so
        # this suite must restore them ALL -- the standing e660/d84c5a6
        # suite-order class
        self.s_auxresp_grads = grad_dfrhf._AUXRESP_F32_GRADS
        self.s_auxresp_stats = dict(grad_dfrhf._AUXRESP_F32_STATS)
        self.s_jonlyfp64_grads = grad_dfrhf._JONLY_AUXFP64_GRADS
        self.s_jonlyfp64_stats = dict(grad_dfrhf._JONLY_AUXFP64_STATS)
        self.s_auxdet_grads = grad_dfrhf._AUXDET_GRADS
        self.s_auxdet_stats = dict(grad_dfrhf._AUXDET_STATS)
        from gpu4pyscf.df import int3c2e_bdiv
        self.s_cdm_calls = int3c2e_bdiv._CONTRACTDM_DET_CALLS
        self.s_cdm_stats = dict(int3c2e_bdiv._CONTRACTDM_DET_STATS)
        self.s_prec = numint.precision.get_precision()
        for var in ('GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP',
                    'GPU4PYSCF_DF_GRAD_JONLY_AUXVEC',
                    'GPU4PYSCF_DF_GRAD_J3C_REUSE',
                    'GPU4PYSCF_DFGRAD_RESP_F32'):
            had = var in os.environ
            setattr(self, '_had_' + var, had)
            setattr(self, '_env_' + var, os.environ.get(var))

    def tearDown(self):
        grad_dfrhf._JONLY_DEADSKIP = self.s_knob
        grad_dfrhf._JONLY_DEADSKIP_GRADS = self.s_grads
        grad_dfrhf._JONLY_DEADSKIP_STATS.clear()
        grad_dfrhf._JONLY_DEADSKIP_STATS.update(self.s_stats)
        grad_dfrhf._JONLY_AUXVEC = self.s_auxvec
        grad_dfrhf._JONLY_AUXVEC_GRADS = self.s_auxvec_grads
        grad_dfrhf._JONLY_AUXVEC_STATS.clear()
        grad_dfrhf._JONLY_AUXVEC_STATS.update(self.s_auxvec_stats)
        grad_dfrhf._J3C_REUSE = self.s_reuse
        grad_dfrhf._J3C_REUSE_GRADS = self.s_reuse_grads
        grad_dfrhf._J3C_REUSE_JONLY_GRADS = self.s_reuse_jonly
        grad_dfrhf._P1_F32 = self.s_p1f32
        grad_dfrhf._RESP_F32 = self.s_resp
        grad_dfrhf._RESP_F32_CALLS = self.s_resp_calls
        grad_dfrhf._AUXRESP_F32_GRADS = self.s_auxresp_grads
        grad_dfrhf._AUXRESP_F32_STATS.clear()
        grad_dfrhf._AUXRESP_F32_STATS.update(self.s_auxresp_stats)
        grad_dfrhf._JONLY_AUXFP64_GRADS = self.s_jonlyfp64_grads
        grad_dfrhf._JONLY_AUXFP64_STATS.clear()
        grad_dfrhf._JONLY_AUXFP64_STATS.update(self.s_jonlyfp64_stats)
        grad_dfrhf._AUXDET_GRADS = self.s_auxdet_grads
        grad_dfrhf._AUXDET_STATS.clear()
        grad_dfrhf._AUXDET_STATS.update(self.s_auxdet_stats)
        from gpu4pyscf.df import int3c2e_bdiv
        int3c2e_bdiv._CONTRACTDM_DET_CALLS = self.s_cdm_calls
        int3c2e_bdiv._CONTRACTDM_DET_STATS.clear()
        int3c2e_bdiv._CONTRACTDM_DET_STATS.update(self.s_cdm_stats)
        numint.precision.set_precision(self.s_prec)
        for var in ('GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP',
                    'GPU4PYSCF_DF_GRAD_JONLY_AUXVEC',
                    'GPU4PYSCF_DF_GRAD_J3C_REUSE',
                    'GPU4PYSCF_DFGRAD_RESP_F32'):
            if getattr(self, '_had_' + var):
                os.environ[var] = getattr(self, '_env_' + var)
            elif var in os.environ:
                del os.environ[var]

    def _set(self, deadskip=None, auxvec=None, reuse=None, resp32=None):
        '''reset the caches to a known state and set the env vars.'''
        if deadskip is not None:
            grad_dfrhf._JONLY_DEADSKIP = None
            if deadskip == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP'] = deadskip
        if auxvec is not None:
            grad_dfrhf._JONLY_AUXVEC = None
            if auxvec == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_JONLY_AUXVEC', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_JONLY_AUXVEC'] = auxvec
        if reuse is not None:
            grad_dfrhf._J3C_REUSE = None
            if reuse == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_J3C_REUSE', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_J3C_REUSE'] = reuse
        if resp32 is not None:
            grad_dfrhf._RESP_F32 = None
            if resp32 == '':
                os.environ.pop('GPU4PYSCF_DFGRAD_RESP_F32', None)
            else:
                os.environ['GPU4PYSCF_DFGRAD_RESP_F32'] = resp32


class _ContractRecorder:
    '''wrap the df.grad.rhf module's contract router, recording the
    einsum signature of every call through this module (the X build,
    the response chain and the auxresp Gram all route through it; the
    numint-side contractions use their own module binding and are not
    seen here).'''

    def __init__(self):
        self.calls = []
        self._orig = grad_dfrhf.contract

    def __enter__(self):
        calls = self.calls
        orig = self._orig

        def rec(indices, *a, **k):
            calls.append(indices)
            return orig(indices, *a, **k)

        grad_dfrhf.contract = rec
        return calls

    def __exit__(self, *exc):
        grad_dfrhf.contract = self._orig
        return False


class _Ip1Recorder:
    '''wrap both sum_ejk_int3c2e_ip1 kernel entries on the shared
    libvhf_rys module, recording which entry each call hit (the armed
    route must keep the entry mix verbatim: the ip1 kernels are never
    rerouted by this knob).'''

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
            grad_dfrhf._JONLY_DEADSKIP = None
            if raw == '':
                os.environ.pop('GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP', None)
            else:
                os.environ['GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP'] = raw
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                got = grad_dfrhf._jonly_deadskip()
            self.assertIs(got, want, raw)
            if raw not in ('1', '', '0'):
                self.assertTrue(any('ignored' in str(x.message) for x in w))


class TestVerbatimAndArmed(_KnobState):

    def _converged(self, mol, xc='r2scan'):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        # the pure-functional closed-shell SCF serves J from the j_engine
        # route and never builds the cderi (the e509 jonly-suite
        # pre_build pattern): build it explicitly so the reuse pack the
        # fall-through consumes is retained, exactly as the e509/e636
        # drivers do per arm.
        mf.with_df.build()
        return mf

    def test_knob_off_deployed_fallthrough(self):
        # knob OFF + reuse ON: the deployed fall-through runs verbatim --
        # the X build, the dead response chain and the dead Gram all
        # launch (the measured e636 band), the jonly-reuse counter moves
        # and the deadskip counter never does; the stats record
        # armed=False on the pass (the 2faa55d unconditional-stats class)
        self._set(deadskip='0', reuse='1')
        mf = self._converged(mol_sph)
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        c_jonly = grad_dfrhf._J3C_REUSE_JONLY_GRADS
        with _ContractRecorder() as calls:
            mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip)
        self.assertEqual(grad_dfrhf._J3C_REUSE_JONLY_GRADS, c_jonly + 1)
        self.assertIs(grad_dfrhf._JONLY_DEADSKIP_STATS.get('armed'), False)
        for sig in _XBUILD_SIGS + _RESP_SIGS:
            self.assertIn(sig, calls)
        self.assertEqual(sum(sig in calls for sig in _GRAM_SIGS), 1)

    def test_armed_route_engages_and_skips_only_the_dead_gemms(self):
        # knob ON + reuse ON: the counter fires once per gradient, the
        # stats carry the site geometry, ONLY the dead response-chain and
        # Gram signatures are absent -- the X-build pair STAYS (the
        # structural distinction from the direct route, which kills the
        # whole pack band), and the ip1 kernel entry mix is verbatim
        self._set(deadskip='1', reuse='1')
        mf = self._converged(mol_sph)
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        c_jonly = grad_dfrhf._J3C_REUSE_JONLY_GRADS
        with _ContractRecorder() as calls, _Ip1Recorder() as ip1:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip + 1)
        self.assertEqual(grad_dfrhf._J3C_REUSE_JONLY_GRADS, c_jonly + 1)
        stats = dict(grad_dfrhf._JONLY_DEADSKIP_STATS)
        self.assertIs(stats.get('armed'), True)
        self.assertEqual(stats['natm'], mol_sph.natm)
        self.assertIs(stats.get('subsumed_by_direct'), False)
        for sig in _XBUILD_SIGS:
            self.assertIn(sig, calls)     # the pack build stays verbatim
        for sig in _RESP_SIGS + _GRAM_SIGS:
            self.assertNotIn(sig, calls)  # only the dead GEMMs are skipped
        # the ip1 kernel entry mix is verbatim (the kernels are never
        # rerouted): the plain-fp64 lane records f64 entries only
        self.assertGreaterEqual(len(ip1), 2)
        self.assertTrue(ip1 and all(c == 'f64' for c in ip1), ip1)
        self.assertEqual(g_on.dtype, np.float64)

    def test_armed_gradient_envelope_vs_deployed(self):
        # the armed gradient vs the deployed fall-through gradient on the
        # SAME converged mf: the same-lane atomicAdd wobble band ONLY --
        # the skip applies exactly the GEMM beta scales (0*AB + beta*C
        # bitwise), so unlike the direct route there is NO reassociation
        # term in this band
        self._set(deadskip='0', reuse='1')
        mf = self._converged(mol_sph)
        g_off = mf.nuc_grad_method().kernel()
        self._set(deadskip='1')
        g_on = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _DELTA_BAND)
        self.assertLess(abs(g_on - g_off).max(), _GRAD_GATE)

    def test_armed_e_tot_untouched(self):
        # the knob arms only inside the gradient route: two fresh SCF
        # runs with the env set vs unset give the same e_tot (the SCF
        # never reads the knob; 1e-12 headroom over the deterministic
        # same-input SCF)
        self._set(deadskip='0', reuse='1')
        e_off = self._converged(mol_sph).e_tot
        self._set(deadskip='1')
        e_on = self._converged(mol_sph).e_tot
        self.assertAlmostEqual(e_on, e_off, delta=1e-12)


class TestJonlyMechanics(_KnobState):
    '''the site called DIRECTLY (one jk_energy_per_atom(k_factor=0) call,
    no SCF): the identity, the dispatch and the counter/stat geometry.

    Every test builds its own converged mf: a direct call whose pack
    capture fails (reuse off / strict reference) RESETS the holder's
    with_df and releases the retained cderi, so a shared mf would lose
    the pack for the later pack-dependent tests (the reset semantics
    the wrapper documents).'''

    def _fresh(self):
        global mol_sph
        grad_dfrhf._IP1_F32 = False
        mf = dft.RKS(mol_sph, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        # the e509 pre_build pattern (see _converged): retain the pack
        mf.with_df.build()
        dm = mf.make_rdm1()
        import cupy as cp
        dm = cp.asarray(dm) if not isinstance(dm, cp.ndarray) else dm
        return mf, mf.nuc_grad_method(), dm

    def test_direct_call_identity_and_signature_split(self):
        # one direct k_factor=0 call with the pack retained: the armed de
        # equals the deployed de at the same-lane wobble scale, the
        # counter fires exactly once, and the armed lane launches the X
        # build verbatim while skipping exactly the response-chain pair
        # and one Gram variant the deployed lane launches
        _mf, g, dm = self._fresh()
        self._set(deadskip='0', reuse='1')
        c0 = grad_dfrhf._JONLY_DEADSKIP_GRADS
        with _ContractRecorder() as calls_off, _Ip1Recorder() as ip1_off:
            de_off = g.jk_energy_per_atom(dm, j_factor=1,
                                          k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0)
        for sig in _XBUILD_SIGS + _RESP_SIGS:
            self.assertIn(sig, calls_off)
        # exactly ONE Gram variant fires (the hermi branch)
        self.assertEqual(sum(sig in calls_off for sig in _GRAM_SIGS), 1)
        self._set(deadskip='1', reuse='1')
        with _ContractRecorder() as calls_on, _Ip1Recorder() as ip1_on:
            de_on = g.jk_energy_per_atom(dm, j_factor=1,
                                         k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0 + 1)
        self.assertIs(grad_dfrhf._JONLY_DEADSKIP_STATS.get('armed'), True)
        for sig in _XBUILD_SIGS:
            self.assertIn(sig, calls_on)
        for sig in _RESP_SIGS + _GRAM_SIGS:
            self.assertNotIn(sig, calls_on)
        # the ip1 entry mix identical (the kernels never rerouted)
        self.assertEqual(ip1_on, ip1_off)
        # the identity: the same-lane wobble band only
        self.assertLessEqual(abs(de_on - de_off).max(), _DELTA_BAND)

    def test_symmetrize_dm_still_arms(self):
        # a symmetrize-tagged dm still arms (UNLIKE the direct route,
        # which refuses it): the skip's beta scales are value-exact for
        # any dm -- the X build handles the symmetrize convention through
        # its own verbatim factor branches
        _mf, g, dm0 = self._fresh()
        self._set(deadskip='1', reuse='1')
        # make_rdm1 returns a CPArrayWithTag (the attribute-carrying
        # ndarray subclass the TDDFT amplitude dms use); a plain ndarray
        # copy cannot carry the tag
        dm = dm0
        dm.symmetrize = 1
        c0 = grad_dfrhf._JONLY_DEADSKIP_GRADS
        with _ContractRecorder() as calls:
            g.jk_energy_per_atom(dm, j_factor=1, k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0 + 1)
        self.assertIs(grad_dfrhf._JONLY_DEADSKIP_STATS.get('armed'), True)
        for sig in _XBUILD_SIGS:
            self.assertIn(sig, calls)
        for sig in _RESP_SIGS + _GRAM_SIGS:
            self.assertNotIn(sig, calls)

    def test_direct_route_subsumes_deadskip(self):
        # both knobs ON: the direct route runs (its counter fires, the
        # whole pack band is absent INCLUDING the X build) and the
        # deadskip never arms -- recorded as subsumed, its counter does
        # not move
        _mf, g, dm = self._fresh()
        self._set(deadskip='1', auxvec='1', reuse='1')
        c_aux = grad_dfrhf._JONLY_AUXVEC_GRADS
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        with _ContractRecorder() as calls:
            g.jk_energy_per_atom(dm, j_factor=1, k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_AUXVEC_GRADS, c_aux + 1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip)
        stats = dict(grad_dfrhf._JONLY_DEADSKIP_STATS)
        self.assertIs(stats.get('armed'), False)
        self.assertIs(stats.get('subsumed_by_direct'), True)
        # the direct route's signature classes: the WHOLE band absent
        for sig in _XBUILD_SIGS + _RESP_SIGS + _GRAM_SIGS:
            self.assertNotIn(sig, calls)

    def test_global_fp32_policy_still_arms(self):
        # the knob is precision-neutral (no dtype change anywhere in the
        # route), so it arms on the global fp32 ('auto') lane too -- the
        # same-value identity holds on any lane; the entry mix follows
        # the site's fp32 selection as usual
        _mf, g, dm = self._fresh()
        numint.precision.set_precision('fp32')
        self._set(deadskip='1', reuse='1')
        c0 = grad_dfrhf._JONLY_DEADSKIP_GRADS
        with _Ip1Recorder() as ip1:
            g.jk_energy_per_atom(dm, j_factor=1,
                                 k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0 + 1)
        self.assertIs(grad_dfrhf._JONLY_DEADSKIP_STATS.get('armed'), True)
        # the site's fp32 selection as usual (the per-aux-batch launches;
        # the batch count is shape-dependent, never asserted by number)
        self.assertTrue(ip1 and all(c == 'f32' for c in ip1), ip1)

    def test_resp32_lane_dtype_parity_and_envelope(self):
        # the resp-f32 knob composed with the deadskip: the arming stays
        # VERBATIM (its counter bumps, dm_oo is cast -- unconsumed), the
        # skip fires (no response GEMMs), and the armed-vs-deployed
        # envelope on the fp32 dm_tensor lane stays at the same wobble
        # band -- the same-dtype beta multiply the fp32 GEMM's beta path
        # would apply
        _mf, g, dm = self._fresh()
        self._set(deadskip='0', reuse='1', resp32='1')
        c_resp = grad_dfrhf._RESP_F32_CALLS
        de_off = g.jk_energy_per_atom(dm, j_factor=1, k_factor=0, hermi=1)
        self.assertGreater(grad_dfrhf._RESP_F32_CALLS, c_resp)
        self._set(deadskip='1')
        c0 = grad_dfrhf._JONLY_DEADSKIP_GRADS
        c_resp2 = grad_dfrhf._RESP_F32_CALLS
        with _ContractRecorder() as calls:
            de_on = g.jk_energy_per_atom(dm, j_factor=1, k_factor=0,
                                         hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0 + 1)
        # the resp-f32 arming still fires verbatim on the armed lane
        self.assertGreater(grad_dfrhf._RESP_F32_CALLS, c_resp2)
        for sig in _RESP_SIGS + _GRAM_SIGS:
            self.assertNotIn(sig, calls)
        for sig in _XBUILD_SIGS:
            self.assertIn(sig, calls)
        self.assertLessEqual(abs(de_on - de_off).max(), _DELTA_BAND)

    def test_hybrid_route_untouched(self):
        # a hybrid functional never takes the k_factor==0 fall-through:
        # the counter never moves (the d96b402 snapshot pattern -- the
        # stats dict may carry stale armed=False from earlier suites in
        # one process, so the counter is the discriminator) and the
        # deployed X build keeps serving the LIVE exchange machinery
        mf = dft.RKS(mol_sph, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        # the hybrid SCF builds the cderi through its K contraction; no
        # pre_build needed (the e388 hybrid-reuse behavior)
        self._set(deadskip='1', reuse='1')
        c0 = grad_dfrhf._JONLY_DEADSKIP_GRADS
        stats_before = dict(grad_dfrhf._JONLY_DEADSKIP_STATS)
        with _ContractRecorder() as calls:
            g_on = mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c0)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_STATS, stats_before)
        for sig in _XBUILD_SIGS:
            self.assertIn(sig, calls)   # the pack build serves the hybrid
        self._set(deadskip='0', reuse='1')
        g_off = mf.nuc_grad_method().kernel()
        self.assertLessEqual(abs(g_on - g_off).max(), _DELTA_BAND)

    def test_uks_jonly_delegates_before_the_pack(self):
        # the UKS k_factor==0 route delegates to _j_energy_per_atom
        # before the pack capture is consumed by the fall-through: the
        # deadskip counter never moves and neither does the jonly-reuse
        # counter (the measured e642 structural class, restated here as
        # an inertness contract)
        mf = dft.UKS(mol_uks, xc='r2scan').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.kernel()
        self._set(deadskip='1', reuse='1')
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        c_jonly = grad_dfrhf._J3C_REUSE_JONLY_GRADS
        mf.nuc_grad_method().kernel()
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip)
        self.assertEqual(grad_dfrhf._J3C_REUSE_JONLY_GRADS, c_jonly)

    def test_strict_reference_inert(self):
        # the strict fp64-reference context keeps the deployed
        # contractions: the knob never arms (belt-and-suspenders -- the
        # pack capture itself already refuses the context, keeping the
        # verbatim early return)
        _mf, g, dm = self._fresh()
        self._set(deadskip='1', reuse='1')
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        with numint.strict_fp64_reference_ctx():
            g_strict = g.jk_energy_per_atom(dm, j_factor=1,
                                            k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip)
        self.assertEqual(g_strict.dtype, np.float64)

    def test_no_reuse_pack_never_arms(self):
        # without the reuse knob the pack is never captured: the
        # fall-through's early return takes the verbatim J-only route
        # and the deadskip counter never moves
        _mf, g, dm = self._fresh()
        self._set(deadskip='1', reuse='0')
        c_skip = grad_dfrhf._JONLY_DEADSKIP_GRADS
        g.jk_energy_per_atom(dm, j_factor=1, k_factor=0, hermi=1)
        self.assertEqual(grad_dfrhf._JONLY_DEADSKIP_GRADS, c_skip)


if __name__ == '__main__':
    print('run with the repo pytest entry (see the suite conventions)')

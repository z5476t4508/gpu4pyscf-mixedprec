# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

"""Regression tests for the CONV-UKS-NEWTON-RESPF32-01 opt-in
vind-scope f32 response discipline (GPU4PYSCF_SOSCF_RESP_F32,
gpu4pyscf/scf/soscf.py).

Provenance: entry 416 (CONV-UKS-NEWTON-INTERIOR-01, 2026-11-07)
decomposed the newton interior at the 573-AO ibuprofen_anion wb97m-v
doublet into 29 vind response applications (81.4% of t_newton, flat
~3.22 s/call) -- the only remaining newton mass after entries 417
(tol-floor truncation NEGATIVE: collapses the grind 18 -> 10 but
stalls the endgame) and 420 (block precond NEGATIVE: vind trajectory
exactly invariant) closed the call-count axis.  Entry 421's source
audit: the deployed vmat/rho/jk f32 lane flags are armed ONLY inside
the plain-SCF kernel wrapper (hf.py kernel/finally), so the newton
vind scope runs pure fp64 even in arms carrying
GPU4PYSCF_SCF_VMAT_F32=headtail for their plain cycles.  This unit's
Phase A reachability audit sharpened it: the vind response evaluates
the XC kernel via nr_rks_fxc/nr_uks_fxc (NO f32 cast hooks) plus the
DF J/K builds (get_jk/get_k), so the OPERATIVE existing cast site on
the vind path is the df_jk K-transform fp32 branch
(_scf_jk_f32_lane, SCFJK-KBUILD-IMPL-01); the numint vmat/rho cast
sites live in the veff tasks _nr_rks_task/_nr_uks_task, which vind
never calls (structurally inert in the armed scope).  The knob arms
the deployed lane-flag triplet around each vind invocation of
gen_g_hop_uhf's h_op ONLY.

Contract tests (source-level, CPU):
  * exactly one env reader for the knob (home module soscf);
  * accepted values '1'/'on'; unset/0/off verbatim; malformed
    warn + ignore;
  * the arming site guards the verbatim
    ``vind = mf.gen_response(mo_coeff, mo_occ, hermi=1, with_nlc=False)``
    line inside gen_g_hop_uhf ONLY -- _wrap_vind_resp_f32 is
    referenced nowhere else, so the keyframe get_veff / acceptance
    path (the direct mf._scf.get_veff call in _rotate_orb_cc) is
    structurally outside the armed scope;
  * the wrapper flips the three deployed lane flags on unconditionally
    (no env consultation) and restores the triplet exactly in finally;
  * strict fp64-reference lanes never arm (flag on the newton wrapper
    OR the wrapped base mf both suppress);
  * tol UNCHANGED: the verbatim ah_conv_tol line still guards
    _rotate_orb_cc.

Numerical tests (GPU, small UKS sentinel, the entry-416 lane class:
DF-UKS newton, deterministic per the 416 bitwise C-arm pins):
  * default OFF: run-to-run bitwise e_tot, zero engagement counter;
  * armed: same basin (|dE| <= 1e-8 Ha), engagement counter ==
    vind call count, converged under cap (also with NO source env
    knob -- the knob itself selects the discipline);
  * malformed value: verbatim path (zero counter, bitwise);
  * strict fp64-reference flag (wrapper or base): never arms
    (zero counter, bitwise verbatim);
  * dtype-shim evidence: during the armed newton every get_jk entry
    INSIDE a vind invocation sees the jk lane flag armed, every
    get_jk/get_veff entry OUTSIDE vind sees it disarmed (the
    keyframe/acceptance path stays fp64), and zero numint f32 twin
    servings fire inside vind (the vmat/rho sites are structurally
    unreachable from the response -- the scope-honesty finding).
"""
import os
import re
import sys
import unittest
import warnings

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy)')

SOSCF_SRC = os.path.join(REPO, 'gpu4pyscf', 'scf', 'soscf.py')
KNOB = 'GPU4PYSCF_SOSCF_RESP_F32'
VMAT_KNOB = 'GPU4PYSCF_SCF_VMAT_F32'


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(SOSCF_SRC, '_soscf_resp_f32')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(SOSCF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # accepted values + warn-and-ignore
        self.assertIn("raw in ('1', 'on')", parser)
        self.assertIn("warnings.warn", parser)

    def test_counter_helper_home_module(self):
        helper = _func_src(SOSCF_SRC, '_note_resp_f32_engaged')
        self.assertIsNotNone(helper)
        self.assertIn('global _RESP_F32_CALLS', helper)

    def test_arming_site_guards_verbatim_line(self):
        gen = _func_src(SOSCF_SRC, 'gen_g_hop_uhf')
        self.assertIsNotNone(gen)
        self.assertIn(
            'vind = mf.gen_response(mo_coeff, mo_occ, hermi=1, '
            'with_nlc=False)', gen)
        self.assertIn('if _resp_f32_armed(mf):', gen)
        self.assertIn('vind = _wrap_vind_resp_f32(vind)', gen)
        # the wrapper is armed ONLY inside gen_g_hop_uhf (h_op scope);
        # a reference anywhere else would arm a foreign path (the def
        # line itself is excluded by the vind = prefix)
        src = open(SOSCF_SRC).read()
        gen_start = src.index('def gen_g_hop_uhf')
        gen_end = src.index('\ndef ', gen_start + 1)
        refs = [m.start() for m in re.finditer(
            r'(?<!def )vind = _wrap_vind_resp_f32\(vind\)', src)]
        self.assertTrue(refs, 'arming site missing')
        for r in refs:
            self.assertTrue(gen_start <= r < gen_end,
                            'wrapper armed outside gen_g_hop_uhf')

    def test_keyframe_veff_outside_arming_scope(self):
        rot = _func_src(SOSCF_SRC, '_rotate_orb_cc')
        self.assertIsNotNone(rot)
        # the acceptance path is the DIRECT base get_veff call
        self.assertIn(
            'vhf0 = mf._scf.get_veff(mf._scf.mol, dm, dm_last=dm0, '
            'vhf_last=vhf0)', rot)
        # and it never passes through the armed wrapper
        self.assertNotIn('_wrap_vind_resp_f32', rot)
        self.assertNotIn('_resp_f32_armed', rot)
        # the wrapper exists as a separate def (not inlined anywhere)
        self.assertIsNotNone(_func_src(SOSCF_SRC,
                                       '_wrap_vind_resp_f32'))
        # tol UNCHANGED: the verbatim tol line still guards the solver
        self.assertIn('ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)',
                      rot)

    def test_wrapper_triplet_contract(self):
        wrap = _func_src(SOSCF_SRC, '_wrap_vind_resp_f32')
        self.assertIsNotNone(wrap)
        # flips exactly the three deployed lane flags, unconditionally
        # (the knob itself selects the discipline; no env consultation)
        self.assertIn('_scf_vmat_f32_lane = True', wrap)
        self.assertIn('_scf_rho_f32_lane = True', wrap)
        self.assertIn('_scf_jk_f32_lane = True', wrap)
        for env_reader in ('_scf_vmat_f32()', '_scf_rho_f32()',
                           '_scf_jk_f32()', '_scf_jk_f32_arm_after()'):
            self.assertNotIn(env_reader, wrap,
                             'the wrapper must not consult the env '
                             'knobs: %s' % env_reader)
        # restores exactly in finally
        self.assertIn('finally:', wrap)
        self.assertIn('_scf_jk_f32_lane) = saved', wrap)
        # counts every armed invocation (real engagements)
        self.assertIn('_note_resp_f32_engaged()', wrap)

    def test_strict_ref_inert_on_both_objects(self):
        armed = _func_src(SOSCF_SRC, '_resp_f32_armed')
        self.assertIsNotNone(armed)
        self.assertIn("for obj in (mf, getattr(mf, '_scf', None)):", armed)
        self.assertIn("getattr(obj, '_strict_fp64_reference', False)", armed)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB), soscf._SOSCF_RESP_F32)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.soscf._SOSCF_RESP_F32 = cache

    def _parse(self, raw):
        self.soscf._SOSCF_RESP_F32 = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.soscf._soscf_resp_f32()

    def test_values(self):
        self.assertFalse(self._parse(None))
        self.assertFalse(self._parse(''))
        self.assertFalse(self._parse('0'))
        self.assertFalse(self._parse('off'))
        self.assertTrue(self._parse('1'))
        self.assertTrue(self._parse('on'))
        for bad in ('block', 'headtail', 'true', '2', 'nan'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                val = self._parse(bad)
            self.assertFalse(val, 'value %r must be ignored' % bad)
            self.assertTrue(any(KNOB in str(x.message) for x in w),
                            'value %r must warn' % bad)

    def test_armed_helper_strict_ref(self):
        class Mf:
            pass
        self._parse('1')
        mf = Mf()
        self.assertTrue(self.soscf._resp_f32_armed(mf))
        mf._strict_fp64_reference = True
        self.assertFalse(self.soscf._resp_f32_armed(mf))
        mf2 = Mf()
        mf2._scf = Mf()
        mf2._scf._strict_fp64_reference = True
        self.assertFalse(self.soscf._resp_f32_armed(mf2))
        self._parse(None)
        self.assertFalse(self.soscf._resp_f32_armed(Mf()))


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestWrapperUnit(unittest.TestCase):
    '''Direct wrapper behavior on a fake vind (no SCF): flag flipping,
    engagement counting, exception-safety.'''

    def setUp(self):
        from gpu4pyscf.scf import soscf
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        self.soscf, self.numint, self.df_jk = soscf, numint, df_jk
        self._saved_flags = (numint._scf_vmat_f32_lane,
                             numint._scf_rho_f32_lane,
                             df_jk._scf_jk_f32_lane)
        self._saved_calls = soscf._RESP_F32_CALLS

    def tearDown(self):
        (self.numint._scf_vmat_f32_lane,
         self.numint._scf_rho_f32_lane,
         self.df_jk._scf_jk_f32_lane) = self._saved_flags
        self.soscf._RESP_F32_CALLS = self._saved_calls

    def test_flag_flip_and_counter(self):
        seen = []
        vind = lambda dm1: seen.append(
            (self.numint._scf_vmat_f32_lane,
             self.numint._scf_rho_f32_lane,
             self.df_jk._scf_jk_f32_lane)) or dm1
        wrapped = self.soscf._wrap_vind_resp_f32(vind)
        out = wrapped('X')
        self.assertEqual(out, 'X')
        self.assertEqual(seen, [(True, True, True)],
                         'all three lane flags must flip during vind')
        self.assertEqual(
            (self.numint._scf_vmat_f32_lane,
             self.numint._scf_rho_f32_lane,
             self.df_jk._scf_jk_f32_lane), (False, False, False),
            'lane flags must be restored after vind')
        self.assertEqual(self.soscf._RESP_F32_CALLS
                         - self._saved_calls, 1)

    def test_exception_restores_flags(self):
        def vind(dm1):
            raise RuntimeError('boom')
        wrapped = self.soscf._wrap_vind_resp_f32(vind)
        with self.assertRaises(RuntimeError):
            wrapped('X')
        self.assertEqual(
            (self.numint._scf_vmat_f32_lane,
             self.numint._scf_rho_f32_lane,
             self.df_jk._scf_jk_f32_lane), (False, False, False),
            'finally must restore the flags on exception')
        self.assertEqual(self.soscf._RESP_F32_CALLS
                         - self._saved_calls, 1,
                         'the engagement counted (the call ran armed '
                         'before raising)')


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS b3lyp newton sentinel (the entry-416 lane class).
    Both numerical roles carry GPU4PYSCF_SCF_VMAT_F32='1' so the plain
    head runs the deployed discipline identically (the T-arm stack
    carries SCF_VMAT_F32=headtail; the C-vs-T difference is the unit
    knob ONLY), and the newton interior is where the knob differs.'''

    def _run(self, raw, vmat_raw='1', strict_ref=None):
        """Head = 2 CDIIS cycles, then public newton() under the unit
        knob value `raw` (None = unset).  Returns (e_tot_repr,
        counter_delta, vind_calls, converged)."""
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        from gpu4pyscf.scf import soscf
        for k, v in ((KNOB, raw), (VMAT_KNOB, vmat_raw)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        soscf._SOSCF_RESP_F32 = None      # force re-read
        numint._SCF_VMAT_F32 = None
        counter0 = soscf._RESP_F32_CALLS
        mol = gto.M(verbose=0, output='/dev/null',
                    atom=[['O', (0., 0., 0.)],
                          ['O', (0., 0., 1.2078)]],
                    basis='def2-svp', charge=0, spin=2)
        try:
            mf = dft.UKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.max_cycle = 2
            mf.conv_check = False
            mf.kernel()
            mfn = mf.newton()
            mfn.max_cycle = 20
            mfn.conv_tol_grad = 1e-5
            if strict_ref == 'wrapper':
                mfn._strict_fp64_reference = True
            elif strict_ref == 'base':
                mf._strict_fp64_reference = True
            vind_calls = [0]
            orig_gen_response = mfn.gen_response

            def gen_response_wrap(*a, **kw):
                vind = orig_gen_response(*a, **kw)

                def vind_wrap(dm1):
                    vind_calls[0] += 1
                    return vind(dm1)
                return vind_wrap
            mfn.gen_response = gen_response_wrap
            try:
                e = mfn.kernel()
            finally:
                del mfn.gen_response
            return (repr(float(e)),
                    soscf._RESP_F32_CALLS - counter0,
                    vind_calls[0],
                    bool(getattr(mfn, 'converged', False)))
        finally:
            mol.stdout.close()

    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB),
                       os.environ.get(VMAT_KNOB),
                       soscf._SOSCF_RESP_F32,
                       soscf._RESP_F32_CALLS)

    def tearDown(self):
        from gpu4pyscf.dft import numint
        raw, vraw, cache, calls = self._saved
        for k, v in ((KNOB, raw), (VMAT_KNOB, vraw)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.soscf._SOSCF_RESP_F32 = cache
        self.soscf._RESP_F32_CALLS = calls
        numint._SCF_VMAT_F32 = None

    def test_default_off_bitwise_and_armed(self):
        # (1) default OFF (source env set, unit knob unset): bitwise
        # run-to-run, zero engagements
        e1, d1, v1, c1 = self._run(None)
        self.assertTrue(c1, 'knob-off newton must converge')
        self.assertEqual(d1, 0, 'knob OFF must not engage')
        e2, d2, v2, c2 = self._run(None)
        self.assertTrue(c2)
        self.assertEqual(d2, 0)
        self.assertEqual(e1, e2, 'knob-off lane must be bitwise '
                         'run-to-run (416 determinism class)')
        # (2) armed: same basin within 1e-8 Ha, counter == vind calls
        e3, d3, v3, c3 = self._run('1')
        self.assertTrue(c3, 'armed newton must converge under cap')
        self.assertEqual(d3, v3,
                         'every armed vind invocation must count')
        self.assertGreaterEqual(d3, 1, 'armed knob must engage')
        self.assertLessEqual(abs(float(e3) - float(e1)), 1e-8,
                             'armed newton must stay in the same basin')
        # (3) malformed value: verbatim path, bitwise with (1)
        e4, d4, v4, c4 = self._run('not-a-flag')
        self.assertTrue(c4)
        self.assertEqual(d4, 0, 'malformed value must be ignored')
        self.assertEqual(e4, e1, 'malformed value must be bitwise '
                         'verbatim')
        # (4) armed + strict fp64-reference flag (either object):
        # never arms, bitwise verbatim
        e5, d5, v5, c5 = self._run('1', strict_ref='wrapper')
        self.assertTrue(c5)
        self.assertEqual(d5, 0,
                         'strict fp64-reference lanes never arm')
        self.assertEqual(e5, e1, 'strict-ref armed run must be '
                         'bitwise verbatim')
        e6, d6, v6, c6 = self._run('1', strict_ref='base')
        self.assertTrue(c6)
        self.assertEqual(d6, 0,
                         'base-mf strict flag must suppress too')
        self.assertEqual(e6, e1)

    def test_armed_without_source_env(self):
        # the knob itself selects the discipline (no env
        # consultation): armed with NO source env knob still engages
        # every vind call and stays in the same basin
        e0, d0, v0, c0 = self._run(None, vmat_raw=None)
        self.assertTrue(c0)
        self.assertEqual(d0, 0)
        e7, d7, v7, c7 = self._run('1', vmat_raw=None)
        self.assertTrue(c7, 'armed newton must converge under cap')
        self.assertEqual(d7, v7,
                         'every armed vind invocation must count')
        self.assertGreaterEqual(d7, 1)
        self.assertLessEqual(abs(float(e7) - float(e0)), 1e-8,
                             'armed newton must stay in the same basin')


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestDtypeShim(unittest.TestCase):
    '''dtype/attribution-shim evidence on the armed sentinel: during
    the newton, every get_jk entry INSIDE a vind invocation sees the
    jk lane flag armed (the df_jk K-transform cast site engages -- the
    operative existing cast site on the vind path), every get_jk /
    get_veff entry OUTSIDE vind sees it disarmed (the keyframe /
    acceptance path stays fp64), and ZERO numint f32 twin servings
    fire inside vind (the vmat/rho cast sites are structurally
    unreachable from the response -- the scope-honesty finding).'''

    def test_casts_engage_in_vind_scope_only(self):
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        from gpu4pyscf.scf import soscf
        os.environ[KNOB] = '1'
        soscf._SOSCF_RESP_F32 = None
        jk_entries = []     # (inside_vind, jk_lane_at_entry)
        veff_entries = []   # (inside_vind, jk_lane_at_entry)
        ao_servings = []    # (inside_vind,) for numint._ao_f32
        inside = [0]
        try:
            mol = gto.M(verbose=0, output='/dev/null',
                        atom=[['O', (0., 0., 0.)],
                              ['O', (0., 0., 1.2078)]],
                        basis='def2-svp', charge=0, spin=2)
            mf = dft.UKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.max_cycle = 2
            mf.conv_check = False
            mf.kernel()      # head: shims NOT yet installed
            mfn = mf.newton()
            mfn.max_cycle = 20
            mfn.conv_tol_grad = 1e-5

            base = mfn._scf
            # vind's `mf.get_jk` binds to the NEWTON WRAPPER (the
            # second-order class inherits the DF UKS class, so the
            # method resolves on the wrapper, mf=mfn inside
            # _gen_uhf_response); the keyframe get_veff path binds to
            # the BASE -- shim BOTH objects for full attribution
            orig_get_jk_base = base.get_jk
            orig_get_jk_mfn = mfn.get_jk

            def jk_shim(mol_, dms, hermi=0, *a, **kw):
                jk_entries.append(
                    (inside[0] > 0,
                     bool(df_jk._scf_jk_f32_lane)))
                return orig_get_jk_base(mol_, dms, hermi, *a, **kw)

            def jk_shim_mfn(mol_, dms, hermi=0, *a, **kw):
                jk_entries.append(
                    (inside[0] > 0,
                     bool(df_jk._scf_jk_f32_lane)))
                return orig_get_jk_mfn(mol_, dms, hermi, *a, **kw)
            base.get_jk = jk_shim
            mfn.get_jk = jk_shim_mfn

            orig_get_veff = base.get_veff

            def veff_shim(*a, **kw):
                veff_entries.append(
                    (inside[0] > 0,
                     bool(df_jk._scf_jk_f32_lane)))
                return orig_get_veff(*a, **kw)
            base.get_veff = veff_shim

            orig_gen_response = mfn.gen_response

            def gen_response_wrap(*a, **kw):
                vind = orig_gen_response(*a, **kw)

                def vind_wrap(dm1):
                    inside[0] += 1
                    try:
                        return vind(dm1)
                    finally:
                        inside[0] -= 1
                return vind_wrap
            mfn.gen_response = gen_response_wrap

            orig_serve = numint._ao_f32

            def serve_shim(ao, grids, opt, loop_blksize, block_id,
                           comp):
                ao_servings.append(inside[0] > 0)
                return orig_serve(ao, grids, opt, loop_blksize,
                                  block_id, comp)
            numint._ao_f32 = serve_shim
            try:
                counter0 = soscf._RESP_F32_CALLS
                head_calls0 = df_jk._scf_jk_f32_head_calls
                mfn.kernel()
                counter_delta = soscf._RESP_F32_CALLS - counter0
                head_calls_delta = (df_jk._scf_jk_f32_head_calls
                                    - head_calls0)
            finally:
                base.get_jk = orig_get_jk_base
                mfn.get_jk = orig_get_jk_mfn
                del base.get_veff
                del mfn.gen_response
                numint._ao_f32 = orig_serve
            mol.stdout.close()

            in_vind_jk = [e for e in jk_entries if e[0]]
            out_vind_jk = [e for e in jk_entries if not e[0]]
            self.assertGreater(
                len(in_vind_jk), 0,
                'get_jk must run inside the armed vind scope (the '
                'hybrid response J/K build -- dtype-shim evidence)')
            self.assertTrue(all(e[1] for e in in_vind_jk),
                            'get_jk inside vind must see the jk lane '
                            'armed (K-transform cast site engaged)')
            self.assertGreater(head_calls_delta, 0,
                               'the df_jk lane-armed counter must '
                               'record the engagements')
            self.assertTrue(out_vind_jk,
                            'expected keyframe-path get_jk calls '
                            'outside vind')
            self.assertTrue(all(not e[1] for e in out_vind_jk),
                            'keyframe-path get_jk must never see an '
                            'armed lane')
            out_vind_veff = [v for v in veff_entries if not v[0]]
            self.assertTrue(out_vind_veff,
                            'expected keyframe get_veff calls outside '
                            'vind (the acceptance path)')
            self.assertTrue(all(not v[1] for v in out_vind_veff),
                            'keyframe get_veff must never see an '
                            'armed lane')
            self.assertEqual(
                [s for s in ao_servings if s], [],
                'the numint f32 twin serving is structurally '
                'unreachable from the vind response (scope honesty); '
                'saw %d inside vind'
                % len([s for s in ao_servings if s]))
            self.assertGreater(counter_delta, 0)
        finally:
            os.environ.pop(KNOB, None)
            os.environ.pop(VMAT_KNOB, None)
            soscf._SOSCF_RESP_F32 = None


if __name__ == '__main__':
    print('Regression tests for the SOSCF vind-scope f32 response knob')
    unittest.main()

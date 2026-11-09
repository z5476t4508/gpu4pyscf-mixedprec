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

"""Regression tests for the CONV-UKS-NEWTON-KEYVEFF-F32-01 opt-in
keyframe-scope f32 discipline (GPU4PYSCF_SOSCF_KEYVEFF_F32,
gpu4pyscf/scf/soscf.py).

Provenance: entry 416 (CONV-UKS-NEWTON-INTERIOR-01, 2026-11-07)
decomposed the fp64 newton at the 573-AO ibuprofen_anion wb97m-v
doublet into vind mass (81.4%, closed by entries 417/420/422/426/427)
plus the KEYFRAME get_veff builds: veff walls [4.35, 4.34, 8.70,
2.18] s per macro = 19.57 s of the 114.69 s t_newton, keyframe
correlated.  The RESPF32 wrapper (entry 422, ece32b4) armed the
deployed vmat/rho/jk f32 lane triplet around vind ONLY and kept the
keyframe builds "structurally OUTSIDE the wrapper ... the exact fp64
map" BY DESIGN; the lane flags are otherwise set solely inside the
plain-SCF kernel wrapper (hf.py kernel/finally, entry 421 audit), so
the keyframe veff runs pure fp64 in every entry-414..435 arm even
under GPU4PYSCF_SCF_VMAT_F32=headtail -- the largest remaining fp64
block of the composed interior (~37% of its t_newton).  This knob
arms the SAME deployed triplet around exactly the two recurring
keyframe sites: the in-micro keyframe refresh in _rotate_orb_cc and
kernel()'s post-rotation macro acceptance get_veff.  Unlike the vind
scope, the numint vmat/rho cast sites ARE operative here (the
keyframe veff runs the _nr_uks_task/_nr_rks_task veff tasks).

Contract tests (source-level, CPU):
  * exactly one env reader for the knob (home module soscf);
  * accepted values '1'/'on'; unset/0/off verbatim; malformed
    warn + ignore;
  * the ONE arming site guards the verbatim in-micro keyframe
    refresh line behind an else-branch; _keyveff_get_veff is
    referenced nowhere else -- kernel()'s macro acceptance get_veff
    (the returned-e_tot map) and the one-time init-guess veff builds
    at kernel entry stay verbatim fp64 (Phase A: two-site arming
    displaced the sentinel's returned e_tot by 6.2e-7, above the
    1e-8 same-basin gate);
  * the helper flips the three deployed lane flags on unconditionally
    (no env consultation) and restores the triplet exactly in
    finally, counting each armed invocation;
  * strict fp64-reference lanes never arm (flag on the newton wrapper
    OR the wrapped base mf both suppress);
  * the RESPF32 vind contract is untouched (absence-contract scope
    guard, green at the parent by design).

Numerical tests (GPU, small UKS sentinel, the entry-416 lane class):
  * default OFF: run-to-run bitwise e_tot, zero engagement counter;
  * armed: same basin (|dE| <= 1e-8 Ha), converged under cap,
    engagement counter == the in-micro keyframe identity
    (sum of the yielded kfcount over macros; the macro acceptance
    build never counts);
  * malformed value: verbatim path (zero counter, bitwise);
  * strict fp64-reference flag (wrapper or base): never arms
    (zero counter, bitwise verbatim);
  * dtype-shim evidence: during the armed newton every keyframe
    get_veff entry sees the lane triplet armed at entry, the
    init-guess veff builds see it disarmed, get_jk inside a keyframe
    veff sees the jk lane armed (both the df_jk K-transform branch
    and the numint f32 twin servings are operative in this scope),
    and get_jk inside vind stays disarmed with RESPF32 absent (this
    knob never touches the response path).
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
KNOB = 'GPU4PYSCF_SOSCF_KEYVEFF_F32'
VMAT_KNOB = 'GPU4PYSCF_SCF_VMAT_F32'


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(SOSCF_SRC, '_soscf_keyveff_f32')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(SOSCF_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # accepted values + warn-and-ignore
        self.assertIn("raw in ('1', 'on')", parser)
        self.assertIn("warnings.warn", parser)

    def test_counter_helper_home_module(self):
        helper = _func_src(SOSCF_SRC, '_note_keyveff_f32_engaged')
        self.assertIsNotNone(helper)
        self.assertIn('global _KEYVEFF_F32_CALLS', helper)

    def test_arming_sites_guard_verbatim_lines(self):
        src = open(SOSCF_SRC).read()
        # the ONE guarded site: the in-micro keyframe refresh inside
        # _rotate_orb_cc; the macro acceptance inside kernel() stays
        # verbatim (the returned-e_tot map)
        rot = _func_src(SOSCF_SRC, '_rotate_orb_cc')
        self.assertIsNotNone(rot)
        self.assertIn('if _keyveff_f32_armed(mf):', rot)
        self.assertIn(
            'vhf0 = mf._scf.get_veff(mf._scf.mol, dm, dm_last=dm0, '
            'vhf_last=vhf0)', rot,
            'the verbatim in-micro keyframe line must survive behind '
            'the else-branch')
        self.assertIn('vhf0 = _keyveff_get_veff(mf, dm, dm0, vhf0)', rot)
        kern = _func_src(SOSCF_SRC, 'kernel')
        self.assertIsNotNone(kern)
        # the macro acceptance get_veff keeps its verbatim line with
        # NO guard anywhere in kernel()
        self.assertIn(
            'vhf = mf._scf.get_veff(mol, dm, dm_last=dm_last, '
            'vhf_last=vhf)', kern,
            'the verbatim macro-acceptance line must survive '
            'UNGUARDED (the returned-e_tot map stays exact fp64)')
        self.assertNotIn('_keyveff_f32_armed', kern)
        self.assertNotIn('_keyveff_get_veff', kern)
        # _keyveff_get_veff is armed at EXACTLY one site (the def
        # line excluded); a reference anywhere else would arm a
        # foreign path
        refs = [m.start() for m in re.finditer(
            r'(?<!def )_keyveff_get_veff\(mf', src)]
        self.assertEqual(len(refs), 1,
                         'expected exactly one arming site, found %d'
                         % len(refs))
        # the one-time init-guess veff builds at kernel entry stay
        # verbatim (never pass through the helper)
        init_site = kern.index('vhf = mf._scf.get_veff(mol, dm, '
                               'dm_last=dm_last,')
        init_end = kern.index('e_tot = mf._scf.energy_tot(dm, h1e, vhf)')
        init_block = kern[:init_end]
        self.assertEqual(init_block.count('mf._scf.get_veff(mol, dm)'),
                         2, 'the two init-guess veff builds')
        self.assertNotIn('_keyveff_get_veff',
                         init_block[:init_site],
                         'init-guess builds must stay verbatim fp64')

    def test_helper_triplet_contract(self):
        helper = _func_src(SOSCF_SRC, '_keyveff_get_veff')
        self.assertIsNotNone(helper)
        # flips exactly the three deployed lane flags, unconditionally
        # (the knob itself selects the discipline; no env consultation)
        self.assertIn('_scf_vmat_f32_lane = True', helper)
        self.assertIn('_scf_rho_f32_lane = True', helper)
        self.assertIn('_scf_jk_f32_lane = True', helper)
        for env_reader in ('_scf_vmat_f32()', '_scf_rho_f32()',
                           '_scf_jk_f32()', '_scf_jk_f32_arm_after()'):
            self.assertNotIn(env_reader, helper,
                             'the helper must not consult the env '
                             'knobs: %s' % env_reader)
        # restores exactly in finally
        self.assertIn('finally:', helper)
        self.assertIn('_scf_jk_f32_lane) = saved', helper)
        # counts every armed invocation (real engagements)
        self.assertIn('_note_keyveff_f32_engaged()', helper)

    def test_strict_ref_inert_on_both_objects(self):
        armed = _func_src(SOSCF_SRC, '_keyveff_f32_armed')
        self.assertIsNotNone(armed)
        self.assertIn("for obj in (mf, getattr(mf, '_scf', None)):", armed)
        self.assertIn("getattr(obj, '_strict_fp64_reference', False)", armed)

    def test_resp32_contract_untouched(self):
        # absence-contract scope guard (green at the parent by
        # design): the RESPF32 vind wrapper keeps its own scope
        src = open(SOSCF_SRC).read()
        gen = _func_src(SOSCF_SRC, 'gen_g_hop_uhf')
        self.assertIn('if _resp_f32_armed(mf):', gen)
        self.assertIn('vind = _wrap_vind_resp_f32(vind)', gen)
        # the keyveff helper never fires from the vind path
        self.assertNotIn('_keyveff_get_veff', gen)
        # tol UNCHANGED: the verbatim tol line still guards the solver
        rot = _func_src(SOSCF_SRC, '_rotate_orb_cc')
        self.assertIn('ah_conv_tol = min(norm_gorb**2, mf.ah_conv_tol)',
                      rot)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB), soscf._SOSCF_KEYVEFF_F32)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.soscf._SOSCF_KEYVEFF_F32 = cache

    def _parse(self, raw):
        self.soscf._SOSCF_KEYVEFF_F32 = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.soscf._soscf_keyveff_f32()

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
        self.assertTrue(self.soscf._keyveff_f32_armed(mf))
        mf._strict_fp64_reference = True
        self.assertFalse(self.soscf._keyveff_f32_armed(mf))
        mf2 = Mf()
        mf2._scf = Mf()
        mf2._scf._strict_fp64_reference = True
        self.assertFalse(self.soscf._keyveff_f32_armed(mf2))
        self._parse(None)
        self.assertFalse(self.soscf._keyveff_f32_armed(Mf()))


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestHelperUnit(unittest.TestCase):
    '''Direct helper behavior on a fake mf (no SCF): flag flipping,
    engagement counting, exception-safety, argument passthrough.'''

    def setUp(self):
        from gpu4pyscf.scf import soscf
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        self.soscf, self.numint, self.df_jk = soscf, numint, df_jk
        self._saved_flags = (numint._scf_vmat_f32_lane,
                             numint._scf_rho_f32_lane,
                             df_jk._scf_jk_f32_lane)
        self._saved_calls = soscf._KEYVEFF_F32_CALLS

    def tearDown(self):
        (self.numint._scf_vmat_f32_lane,
         self.numint._scf_rho_f32_lane,
         self.df_jk._scf_jk_f32_lane) = self._saved_flags
        self.soscf._KEYVEFF_F32_CALLS = self._saved_calls

    def _fake_mf(self, seen, raise_=False):
        class Scf:
            mol = 'MOL'
            def get_veff(self, mol, dm, dm_last=None, vhf_last=None):
                seen.append(
                    (mol,
                     (self_vmat(), self_rho(), self_jk())))
                if raise_:
                    raise RuntimeError('boom')
                return 'VHF'
        scf = Scf()
        class Mf:
            _scf = scf
        # late-bind the lane getters inside the closure
        def self_vmat():
            return self.numint._scf_vmat_f32_lane
        def self_rho():
            return self.numint._scf_rho_f32_lane
        def self_jk():
            return self.df_jk._scf_jk_f32_lane
        scf._lanes = (self_vmat, self_rho, self_jk)
        return Mf()

    def test_flag_flip_passthrough_and_counter(self):
        seen = []
        mf = self._fake_mf(seen)
        out = self.soscf._keyveff_get_veff(mf, 'DM', 'DML', 'VHFL')
        self.assertEqual(out, 'VHF')
        self.assertEqual(seen, [('MOL', (True, True, True))],
                         'the keyframe veff build must see all three '
                         'lane flags armed (incl. its mol/dm passthrough)')
        self.assertEqual(
            (self.numint._scf_vmat_f32_lane,
             self.numint._scf_rho_f32_lane,
             self.df_jk._scf_jk_f32_lane), (False, False, False),
            'lane flags must be restored after the keyframe build')
        self.assertEqual(self.soscf._KEYVEFF_F32_CALLS
                         - self._saved_calls, 1)

    def test_exception_restores_flags(self):
        seen = []
        mf = self._fake_mf(seen, raise_=True)
        with self.assertRaises(RuntimeError):
            self.soscf._keyveff_get_veff(mf, 'DM', None, None)
        self.assertEqual(
            (self.numint._scf_vmat_f32_lane,
             self.numint._scf_rho_f32_lane,
             self.df_jk._scf_jk_f32_lane), (False, False, False),
            'finally must restore the flags on exception')
        self.assertEqual(self.soscf._KEYVEFF_F32_CALLS
                         - self._saved_calls, 1,
                         'the engagement counted (the call ran armed '
                         'before raising)')


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS b3lyp newton sentinel (the entry-416 lane class).
    Both numerical roles carry GPU4PYSCF_SCF_VMAT_F32='1' so the plain
    head runs the deployed discipline identically; the unit knob arms
    ONLY inside the newton keyframe builds.'''

    def _run(self, raw, vmat_raw='1', strict_ref=None):
        """Head = 2 CDIIS cycles, then public newton() under the unit
        knob value `raw` (None = unset).  Returns (e_tot_repr,
        counter_delta, keyframe_identity, converged) where
        keyframe_identity = sum(kfcount) over macros from the macro
        recorder MINUS the macro count (the recorder stores yielded
        kfcount + 1; the armed class is the in-micro refreshes only,
        so the expected engagement count excludes the +1)."""
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        from gpu4pyscf.scf import soscf
        for k, v in ((KNOB, raw), (VMAT_KNOB, vmat_raw)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        soscf._SOSCF_KEYVEFF_F32 = None      # force re-read
        numint._SCF_VMAT_F32 = None
        counter0 = soscf._KEYVEFF_F32_CALLS
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
            macros = []

            def cb(loc):
                i = int(loc['imacro'])
                if any(m['imacro'] == i for m in macros):
                    return
                macros.append({'imacro': i,
                               'kfcount': int(loc['kfcount']) + 1})
            mfn.callback = cb
            e = mfn.kernel()
            identity = (sum(m['kfcount'] for m in macros)
                        - len(macros))
            return (repr(float(e)),
                    soscf._KEYVEFF_F32_CALLS - counter0,
                    identity,
                    bool(getattr(mfn, 'converged', False)))
        finally:
            mol.stdout.close()

    def setUp(self):
        from gpu4pyscf.scf import soscf
        self.soscf = soscf
        self._saved = (os.environ.get(KNOB),
                       os.environ.get(VMAT_KNOB),
                       soscf._SOSCF_KEYVEFF_F32,
                       soscf._KEYVEFF_F32_CALLS)

    def tearDown(self):
        from gpu4pyscf.dft import numint
        raw, vraw, cache, calls = self._saved
        for k, v in ((KNOB, raw), (VMAT_KNOB, vraw)):
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.soscf._SOSCF_KEYVEFF_F32 = cache
        self.soscf._KEYVEFF_F32_CALLS = calls
        numint._SCF_VMAT_F32 = None

    def test_default_off_bitwise_and_armed(self):
        # (1) default OFF (source env set, unit knob unset): bitwise
        # run-to-run, zero engagements
        e1, d1, i1, c1 = self._run(None)
        self.assertTrue(c1, 'knob-off newton must converge')
        self.assertEqual(d1, 0, 'knob OFF must not engage')
        e2, d2, i2, c2 = self._run(None)
        self.assertTrue(c2)
        self.assertEqual(d2, 0)
        self.assertEqual(e1, e2, 'knob-off lane must be bitwise '
                         'run-to-run (416 determinism class)')
        # (2) armed: same basin within 1e-8 Ha, counter == the
        # keyframe identity (the in-micro refreshes only; the macro
        # acceptance build stays verbatim and never counts)
        e3, d3, i3, c3 = self._run('1')
        self.assertTrue(c3, 'armed newton must converge under cap')
        self.assertGreaterEqual(d3, 1, 'armed knob must engage')
        self.assertEqual(d3, i3,
                         'every armed keyframe veff build must count '
                         '(counter %d != identity %d)' % (d3, i3))
        self.assertLessEqual(abs(float(e3) - float(e1)), 1e-8,
                             'armed newton must stay in the same basin')
        # (3) malformed value: verbatim path, bitwise with (1)
        e4, d4, i4, c4 = self._run('not-a-flag')
        self.assertTrue(c4)
        self.assertEqual(d4, 0, 'malformed value must be ignored')
        self.assertEqual(e4, e1, 'malformed value must be bitwise '
                         'verbatim')
        # (4) armed + strict fp64-reference flag (either object):
        # never arms, bitwise verbatim
        e5, d5, i5, c5 = self._run('1', strict_ref='wrapper')
        self.assertTrue(c5)
        self.assertEqual(d5, 0,
                         'strict fp64-reference lanes never arm')
        self.assertEqual(e5, e1, 'strict-ref armed run must be '
                         'bitwise verbatim')
        e6, d6, i6, c6 = self._run('1', strict_ref='base')
        self.assertTrue(c6)
        self.assertEqual(d6, 0,
                         'base-mf strict flag must suppress too')
        self.assertEqual(e6, e1)

    def test_armed_without_source_env(self):
        # the knob itself selects the discipline (no env
        # consultation): armed with NO source env knob still engages
        # every keyframe build and stays in the same basin
        e0, d0, i0, c0 = self._run(None, vmat_raw=None)
        self.assertTrue(c0)
        self.assertEqual(d0, 0)
        e7, d7, i7, c7 = self._run('1', vmat_raw=None)
        self.assertTrue(c7, 'armed newton must converge under cap')
        self.assertEqual(d7, i7,
                         'every armed keyframe veff build must count')
        self.assertGreaterEqual(d7, 1)
        self.assertLessEqual(abs(float(e7) - float(e0)), 1e-8,
                             'armed newton must stay in the same basin')


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestDtypeShim(unittest.TestCase):
    '''dtype/attribution-shim evidence on the armed sentinel: during
    the newton, every get_veff entry under the armed helper sees the
    lane triplet armed at entry (the keyframe builds), the init-guess
    veff builds see it disarmed, every get_jk entry INSIDE a keyframe
    veff sees the jk lane armed (the df_jk K-transform cast site
    engages), and numint f32 twin servings fire inside the keyframe
    scope (the vmat/rho cast sites ARE operative here -- the scope
    difference vs the vind wrapper); with RESPF32 absent, get_jk
    inside vind NEVER sees an armed lane (this knob never touches the
    response path).'''

    def test_casts_engage_in_keyframe_scope_only(self):
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        from gpu4pyscf.df import df_jk
        from gpu4pyscf.scf import soscf
        os.environ[KNOB] = '1'
        os.environ.pop('GPU4PYSCF_SOSCF_RESP_F32', None)
        soscf._SOSCF_KEYVEFF_F32 = None
        soscf._SOSCF_RESP_F32 = None
        veff_entries = []   # (vmat_lane, jk_lane, dm_last_given)
        jk_entries = []     # (inside_keyframe_veff, jk_lane)
        ao_servings = []    # (inside_keyframe_veff,)
        inside_kf = [0]
        inside_vind = [0]
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
            orig_get_jk_base = base.get_jk
            orig_get_jk_mfn = mfn.get_jk

            def jk_shim(mol_, dms, hermi=0, *a, **kw):
                jk_entries.append(
                    (inside_kf[0] > 0,
                     bool(df_jk._scf_jk_f32_lane)))
                return orig_get_jk_base(mol_, dms, hermi, *a, **kw)

            def jk_shim_mfn(mol_, dms, hermi=0, *a, **kw):
                jk_entries.append(
                    (inside_kf[0] > 0,
                     bool(df_jk._scf_jk_f32_lane)))
                return orig_get_jk_mfn(mol_, dms, hermi, *a, **kw)
            base.get_jk = jk_shim
            mfn.get_jk = jk_shim_mfn

            orig_get_veff = base.get_veff

            def veff_shim(mol_, dm, dm_last=None, vhf_last=None, *a,
                          **kw):
                armed = bool(numint._scf_vmat_f32_lane)
                veff_entries.append(
                    (armed,
                     bool(df_jk._scf_jk_f32_lane),
                     dm_last is not None))
                # mark ONLY armed (helper-invoked) keyframe builds;
                # the disarmed init/acceptance builds are a different
                # class and must not contaminate it
                if armed:
                    inside_kf[0] += 1
                try:
                    return orig_get_veff(mol_, dm, *a,
                                         dm_last=dm_last,
                                         vhf_last=vhf_last, **kw)
                finally:
                    if armed:
                        inside_kf[0] -= 1
            base.get_veff = veff_shim

            orig_gen_response = mfn.gen_response

            def gen_response_wrap(*a, **kw):
                vind = orig_gen_response(*a, **kw)

                def vind_wrap(dm1):
                    inside_vind[0] += 1
                    try:
                        return vind(dm1)
                    finally:
                        inside_vind[0] -= 1
                return vind_wrap
            mfn.gen_response = gen_response_wrap

            orig_serve = numint._ao_f32

            def serve_shim(ao, grids, opt, loop_blksize, block_id,
                           comp):
                ao_servings.append(inside_kf[0] > 0)
                return orig_serve(ao, grids, opt, loop_blksize,
                                  block_id, comp)
            numint._ao_f32 = serve_shim
            try:
                counter0 = soscf._KEYVEFF_F32_CALLS
                head_calls0 = df_jk._scf_jk_f32_head_calls
                mfn.kernel()
                counter_delta = (soscf._KEYVEFF_F32_CALLS
                                 - counter0)
                head_calls_delta = (df_jk._scf_jk_f32_head_calls
                                    - head_calls0)
            finally:
                base.get_jk = orig_get_jk_base
                mfn.get_jk = orig_get_jk_mfn
                del base.get_veff
                del mfn.gen_response
                numint._ao_f32 = orig_serve
            mol.stdout.close()

            armed_veff = [v for v in veff_entries if v[0] and v[1]]
            init_veff = [v for v in veff_entries if not (v[0] or v[1])]
            self.assertGreater(
                len(armed_veff), 0,
                'keyframe get_veff entries must see the lane triplet '
                'armed (the helper arms before the call)')
            self.assertTrue(all(v[2] for v in armed_veff),
                            'armed entries are keyframe builds '
                            '(dm_last given)')
            self.assertGreater(len(init_veff), 0,
                               'expected init-guess veff builds '
                              '(disarmed, verbatim fp64)')
            self.assertEqual(
                len([v for v in veff_entries
                     if v[0] != v[1]]), 0,
                'vmat and jk lane flags must arm together (the '
                'triplet)')
            self.assertEqual(counter_delta, len(armed_veff),
                             'the engagement counter must equal the '
                             'armed keyframe veff entry count')
            in_kf_jk = [e for e in jk_entries if e[0]]
            self.assertGreater(
                len(in_kf_jk), 0,
                'get_jk must run inside the armed keyframe veff (the '
                'DF J/K builds -- dtype-shim evidence)')
            self.assertTrue(all(e[1] for e in in_kf_jk),
                            'get_jk inside a keyframe veff must see '
                            'the jk lane armed (K-transform cast site '
                            'engaged)')
            self.assertGreater(head_calls_delta, 0,
                               'the df_jk lane-armed counter must '
                               'record the engagements')
            # with RESPF32 absent, get_jk on the vind path never arms
            self.assertEqual(
                [e for e in jk_entries if not e[0] and e[1]], [],
                'this knob must never arm a non-keyframe get_jk path')
            self.assertGreater(
                len([s for s in ao_servings if s]), 0,
                'numint f32 twin servings must fire inside the '
                'keyframe veff (the vmat/rho cast sites are operative '
                'in this scope, unlike vind)')
            self.assertGreater(counter_delta, 0)
        finally:
            os.environ.pop(KNOB, None)
            os.environ.pop(VMAT_KNOB, None)
            soscf._SOSCF_KEYVEFF_F32 = None


if __name__ == '__main__':
    print('Regression tests for the SOSCF keyframe-scope f32 knob')
    unittest.main()

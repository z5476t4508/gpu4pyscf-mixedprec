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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.
'''MEM-RELEASE-01 focused regression (2026-11-18).

The opt-in GPU4PYSCF_STEADY_MEM_RELEASE knob (DEFAULT OFF) paces one
full gc collection at each DF.build boundary -- the steady-restart
protocol's per-step allocation point.  The e726 MEMCYCLE-TRACE-01 walk
identified the sawtooth mechanism: the per-step graph (cderi <-
DF._cderi <- mf["with_df"] <- DFRKS <- Gradients.base <- bound-method
closures) is unreachable-but-cyclic garbage that survives the steady
driver's name drop + free_all_blocks; the phase-A identification probe
(2026-11-18-mem-release01/phase-a) pinned the cycle-closing
assignments to exactly TWO protocol-side MethodType self-edges
(mf.get_dispersion / g.get_dispersion, both arms, the r2SCAN-3c
D4/gCP contract hooks), with the 15-knob union composition adding NO
library-side cycle of its own.

Asserted here, before any GPU measurement: the parse mapping; the
value-neutrality of the paced collection (fixed-dm get_veff and the
live cderi bitwise identical across a fired collect); the in-repo
reproduction of the cyclic retention (weakrefs to the dropped step
graph stay ALIVE with gc disabled, knob off); the mechanism (with the
knob on, the PRIOR step's weakrefs DIE at the next step's DF.build,
the current step's stay alive until the following boundary, fires
counting fail-closed); strict-reference inertness (the real
numint.strict_fp64_reference_ctx); the no-leak re-run; and
fresh-process subprocess arms (e_tot agreement at the documented
cross-process band, engagement CAND-side only).
'''
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings
import weakref
from types import MethodType

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.df import df as dfmod


def setUpModule():
    global mol_sph
    mol_sph = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000''',
                      basis='def2-svp', output='/dev/null', verbose=0)


def tearDownModule():
    global mol_sph
    mol_sph.stdout.close()
    del mol_sph


def _edge_never_called(*args, **kwargs):
    '''The pinned cycle-closing self-edge CLASS (a bound method stored
    on its own __self__, the e726/probe identification).  Attached
    post-kernel/pre-drop only: the deployed arithmetic never calls it,
    so the values below are the pure deployed-lane values verbatim.'''
    raise AssertionError('the identification self-edge stub must never '
                         'be called; it exists only as the pinned '
                         'MethodType self-edge class')


class _KnobState:
    '''Saves/restores every process-global this knob touches (the
    standing suite-order class): the cached env parse, the
    fail-closed fire counter, the per-build stats list (identity
    preserved via slice assignment), the env var, the strict-reference
    flag, and the gc enabled state the retention tests flip.'''
    def __enter__(self):
        self._parsed = dfmod._STEADY_MEM_RELEASE
        self._fires = dfmod._STEADY_MEM_RELEASE_FIRES
        self._stats = list(dfmod._STEADY_MEM_RELEASE_STATS)
        self._had = 'GPU4PYSCF_STEADY_MEM_RELEASE' in os.environ
        self._val = os.environ.get('GPU4PYSCF_STEADY_MEM_RELEASE')
        self._strict = numint._strict_fp64_reference_active
        self._gc_enabled = gc.isenabled()
        return self

    def __exit__(self, *exc):
        dfmod._STEADY_MEM_RELEASE = self._parsed
        dfmod._STEADY_MEM_RELEASE_FIRES = self._fires
        dfmod._STEADY_MEM_RELEASE_STATS[:] = self._stats
        if self._had:
            os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = self._val
        else:
            os.environ.pop('GPU4PYSCF_STEADY_MEM_RELEASE', None)
        numint._strict_fp64_reference_active = self._strict
        if self._gc_enabled and not gc.isenabled():
            gc.enable()
        elif not self._gc_enabled and gc.isenabled():
            gc.disable()
        return False


def _arm(on):
    dfmod._STEADY_MEM_RELEASE = None
    if on:
        os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = '1'
    else:
        os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = '0'


def _run_step(dm0=None):
    '''One steady-protocol step at the small cell: fresh DF-SCF (with
    explicit dm0 when supplied -- the MD-restart contract), analytic
    gradient, the two pinned protocol-class self-edges attached
    post-kernel/pre-drop, weakrefs captured, names dropped,
    free_all_blocks -- the run_point structure verbatim.'''
    mf = dft.RKS(mol_sph, xc='b3lyp')
    mf = mf.density_fit()
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    mf.precision_mode = 'fp64'
    mf.with_df.build()
    e = mf.kernel(dm0=dm0)
    converged = bool(mf.converged)
    cycles = int(getattr(mf, 'cycles', 0) or 0)
    g = mf.nuc_grad_method()
    g.precision_mode = 'fp64'
    de = cp.asnumpy(g.kernel())
    mf.get_dispersion = MethodType(_edge_never_called, mf)
    g.get_dispersion = MethodType(_edge_never_called, g)
    wr = {'mf': weakref.ref(mf), 'g': weakref.ref(g),
          'with_df': weakref.ref(mf.with_df)}
    try:
        wr['cderi'] = weakref.ref(mf.with_df._cderi[0])
    except Exception:
        wr['cderi'] = None
    dm_out = cp.asarray(mf.make_rdm1())
    mf = g = None
    cp.get_default_memory_pool().free_all_blocks()
    return {'e_tot': float(e), 'converged': converged, 'cycles': cycles,
            'grad': de, 'dm': dm_out, 'wr': wr}


def _wr_alive(rec):
    return {k: ((v() is not None) if v is not None else None)
            for k, v in rec['wr'].items()}


_SUBPROC_STEP = '''
import json, sys
import cupy as cp
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df import df as dfmod

mol = pyscf.M(atom='O 0.0 0.0 0.1174; H -0.757 0.0 -0.4696; '
                   'H 0.757 0.0 -0.4696',
              basis='def2-svp', output='/dev/null', verbose=0)

def step(dm0=None):
    mf = dft.RKS(mol, xc='b3lyp').density_fit()
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    mf.with_df.build()
    e = mf.kernel(dm0=dm0)
    conv = bool(mf.converged)
    dm = cp.asarray(mf.make_rdm1())
    return float(e), conv, dm

e1, c1, dm = step()
e2, c2, _ = step(dm0=dm)
json.dump({'e_tot': [e1, e2], 'converged': [c1, c2],
           'fires': int(dfmod._STEADY_MEM_RELEASE_FIRES),
           'stats_len': len(dfmod._STEADY_MEM_RELEASE_STATS),
           'stats_fired': int(sum(1 for r in dfmod._STEADY_MEM_RELEASE_STATS
                                  if r.get('fired')))},
          open(sys.argv[1], 'w'))
'''


class TestSteadyMemRelease(unittest.TestCase):

    def test_knob_parse_mapping(self):
        with _KnobState():
            for raw, expect in ((None, False), ('', False), ('0', False),
                                ('false', False), ('False', False),
                                ('1', True), ('true', True),
                                ('True', True)):
                with self.subTest(raw=raw):
                    if raw is None:
                        os.environ.pop(
                            'GPU4PYSCF_STEADY_MEM_RELEASE', None)
                    else:
                        os.environ[
                            'GPU4PYSCF_STEADY_MEM_RELEASE'] = raw
                    dfmod._STEADY_MEM_RELEASE = None
                    self.assertIs(dfmod._steady_mem_release(), expect)
            with self.subTest(raw='bogus'):
                os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = 'bogus'
                dfmod._STEADY_MEM_RELEASE = None
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    value = dfmod._steady_mem_release()
                self.assertIs(value, False)
                self.assertEqual(len(w), 1)
                self.assertIn('GPU4PYSCF_STEADY_MEM_RELEASE',
                              str(w[0].message))

    def test_collect_value_neutrality_bitwise(self):
        # a paced collection (and a plain manual one) leave every LIVE
        # value bitwise identical: fixed-dm get_veff and the cderi
        # tensor itself.  The knob never touches arithmetic.
        with _KnobState():
            _arm(False)
            mf = dft.RKS(mol_sph, xc='b3lyp')
            mf = mf.density_fit()
            mf.grids.level = 1
            mf.conv_tol = 1e-10
            mf.precision_mode = 'fp64'
            mf.with_df.build()
            self.assertTrue(mf.kernel())
            self.assertTrue(mf.converged)
            dm = mf.make_rdm1()
            v1 = mf.get_veff(mol_sph, dm)
            cderi_before = cp.asnumpy(mf.with_df._cderi[0]).copy()
            fires_before = dfmod._STEADY_MEM_RELEASE_FIRES
            _arm(True)
            dfmod._steady_mem_release_pace(True)  # a real fired collect
            gc.collect()                          # plus a manual one
            v2 = mf.get_veff(mol_sph, dm)
            cderi_after = cp.asnumpy(mf.with_df._cderi[0])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_before + 1)
            self.assertTrue(bool(cp.array_equal(v1, v2)))
            self.assertTrue(np.array_equal(cderi_before, cderi_after))

    def test_knob_off_retention_and_counters(self):
        # the in-repo reproduction of the e726 mechanism: with gc
        # disabled and the knob off, the dropped per-step graph stays
        # ALIVE (cyclic garbage; the two pinned self-edges close it),
        # the counters stay exactly 0, and every build records an
        # armed=False stats entry.
        with _KnobState():
            _arm(False)
            gc.disable()
            step1 = _run_step()
            self.assertTrue(step1['converged'])
            alive1 = _wr_alive(step1)
            self.assertTrue(alive1['mf'] and alive1['g']
                            and alive1['with_df'])
            step2 = _run_step(dm0=step1['dm'])
            self.assertTrue(step2['converged'])
            alive1b = _wr_alive(step1)
            self.assertTrue(alive1b['mf'] and alive1b['g']
                            and alive1b['with_df'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES, 0)
            self.assertTrue(all(not r['armed'] and not r['fired']
                                for r in dfmod._STEADY_MEM_RELEASE_STATS))
            self.assertGreaterEqual(len(
                dfmod._STEADY_MEM_RELEASE_STATS), 2)
            # same-geometry fresh-cderi rebuild repeat band (measured
            # 2.8e-7 max|dg|, 2.77e-12 e on the knob-OFF arm -- pure
            # cross-build drift, no knob effect)
            self.assertLessEqual(abs(step2['e_tot'] - step1['e_tot']),
                                 1e-9)
            self.assertLessEqual(np.abs(
                step2['grad'] - step1['grad']).max(), 1e-6)

    def test_knob_on_releases_prior_graph(self):
        # the mechanism: arming the knob, the PRIOR step's weakrefs DIE
        # at the next step's DF.build (the paced collect), the current
        # step's graph stays alive until the FOLLOWING boundary, and
        # the fire counter bumps exactly once per armed build.
        with _KnobState():
            _arm(False)
            gc.disable()
            step1 = _run_step()
            self.assertTrue(_wr_alive(step1)['mf'])
            fires1 = dfmod._STEADY_MEM_RELEASE_FIRES
            _arm(True)
            step2 = _run_step(dm0=step1['dm'])
            dead1 = {k: (v() is None) for k, v in step1['wr'].items()
                     if v is not None}
            self.assertTrue(dead1['mf'] and dead1['g']
                            and dead1['with_df'])
            if step1['wr'].get('cderi') is not None:
                self.assertTrue(dead1['cderi'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires1 + 1)
            last = dfmod._STEADY_MEM_RELEASE_STATS[-1]
            self.assertTrue(last['armed'] and last['fired'])
            self.assertGreaterEqual(last['collected_objects'], 10)
            alive2 = _wr_alive(step2)
            self.assertTrue(alive2['mf'] and alive2['g']
                            and alive2['with_df'])
            step3 = _run_step(dm0=step2['dm'])
            dead2 = {k: (v() is None) for k, v in step2['wr'].items()
                     if v is not None}
            self.assertTrue(dead2['mf'] and dead2['g']
                            and dead2['with_df'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires1 + 2)
            # values: same computation, deployed arithmetic verbatim
            for st in (step2, step3):
                self.assertTrue(st['converged'])
                self.assertLessEqual(abs(st['e_tot'] - step1['e_tot']),
                                     1e-9)
                self.assertLessEqual(np.abs(
                    st['grad'] - step1['grad']).max(), 1e-6)

    def test_strict_reference_inertness(self):
        # the strict fp64-reference context (the REAL
        # numint.strict_fp64_reference_ctx) never feels the mechanism:
        # armed is False, fires 0, stats record armed False.
        with _KnobState():
            _arm(True)
            with numint.strict_fp64_reference_ctx():
                rec = _run_step()
            self.assertTrue(rec['converged'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES, 0)
            self.assertTrue(all(not r['armed'] for r
                                in dfmod._STEADY_MEM_RELEASE_STATS))
            # and outside the ctx the same knob fires again (no leak of
            # the refusal): one more step, one more build, fires 1
            rec2 = _run_step(dm0=rec['dm'])
            self.assertTrue(rec2['converged'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES, 1)

    def test_no_leak_rerun(self):
        # after knob-on usage, re-arming OFF stops the pacing: the
        # counter freezes, new builds record armed False, and the
        # values stay at the same-lane repeat band.
        with _KnobState():
            _arm(True)
            step1 = _run_step()
            self.assertTrue(step1['converged'])
            fires_after_on = dfmod._STEADY_MEM_RELEASE_FIRES
            self.assertGreaterEqual(fires_after_on, 1)
            _arm(False)
            step2 = _run_step(dm0=step1['dm'])
            self.assertTrue(step2['converged'])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_after_on)
            self.assertFalse(dfmod._STEADY_MEM_RELEASE_STATS[-1][
                'armed'])
            # same-geometry fresh-cderi rebuild repeat band (measured
            # 2.8e-7 max|dg|, 2.77e-12 e on the knob-OFF arm -- pure
            # cross-build drift, no knob effect)
            self.assertLessEqual(abs(step2['e_tot'] - step1['e_tot']),
                                 1e-9)
            self.assertLessEqual(np.abs(
                step2['grad'] - step1['grad']).max(), 1e-6)

    def test_subprocess_arms(self):
        # fresh process per knob state; e_tot agreement at the
        # documented cross-process band; engagement CAND-side only.
        outs = {}
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, '_mem_release_subproc.py')
            with open(script, 'w') as f:
                f.write(_SUBPROC_STEP)
            for on in (False, True):
                out = os.path.join(tmp, 'subproc-%s.json' % on)
                env = dict(os.environ)
                env['PYTHONPATH'] = os.pathsep.join(
                    [os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', '..', '..'))]
                    + ([env['PYTHONPATH']] if env.get('PYTHONPATH')
                       else []))
                if on:
                    env['GPU4PYSCF_STEADY_MEM_RELEASE'] = '1'
                else:
                    env.pop('GPU4PYSCF_STEADY_MEM_RELEASE', None)
                cp_out = subprocess.run(
                    [sys.executable, script, out], env=env,
                    capture_output=True, text=True, timeout=600)
                self.assertEqual(cp_out.returncode, 0,
                                 cp_out.stderr[-2000:])
                with open(out) as f:
                    outs[on] = json.load(f)
        for on in (False, True):
            self.assertTrue(all(outs[on]['converged']))
        self.assertEqual(outs[False]['fires'], 0)
        self.assertEqual(outs[False]['stats_fired'], 0)
        self.assertGreaterEqual(outs[True]['fires'], 2)
        self.assertEqual(outs[True]['stats_fired'],
                         outs[True]['fires'])
        self.assertEqual(outs[True]['stats_len'], 2)
        for i in (0, 1):
            self.assertLessEqual(
                abs(outs[True]['e_tot'][i] - outs[False]['e_tot'][i]),
                1e-10)


if __name__ == '__main__':
    print('full unittest suite')
    unittest.main()

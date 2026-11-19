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
'''MEM-POOLTRIM-UKS-01 focused regression (2026-11-19).

The e735 DEPTHCAP-UKS-01 depth-50 raws attribute the open-shell
trajectory accumulation to the CUPY pool's RETAINED FREE BLOCKS:
pool_used flat (10-22 MiB) while pool_total climbs +36.0-MiB stairs
and drops 144 MiB at the natural gen-2 releases (step 25: pool_total
194.4 -> 50.4) -- dead objects' memory cached by the pool between
release events, a class the landed MEM-RELEASE-01 gc pacing cannot
see (nothing for a gc.collect() to free; the e735 diagnostic itself
conflated gc.collect(2) with the kit's pool.free_all_blocks() in
released_mib).

Asserted here, before any GPU measurement: the extended parse mapping
('1' keeps the landed gc-only semantics VERBATIM; the new 'trim'
value arms the collection AND one pool free_all_blocks at the same
DF.build boundary; bogus values warn+ignore); the value-neutrality of
a fired trim (free_all_blocks releases only FREE blocks -- fixed-dm
get_veff and the live cderi bitwise identical); the trim mechanism
itself (pool_total drops when retained free blocks exist, live
pool_used untouched); the landed gc-only mode never trimming (the
e727 contract preserved); strict-reference inertness through the real
numint.strict_fp64_reference_ctx; the no-leak re-run; and
fresh-process subprocess arms per mode with engagement CAND-side
only.
'''
import gc
import json
import os
import subprocess
import sys
import tempfile
import unittest
import warnings

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.df import df as dfmod

mol_sph = None


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


def _arm(mode):
    '''mode in ('off', 'gc', 'trim'): sets the env + resets the
    cached parse, the deployed arming discipline.'''
    dfmod._STEADY_MEM_RELEASE = None
    env = {'off': '0', 'gc': '1', 'trim': 'trim'}[mode]
    os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = env


class _TrimState:
    '''Saves/restores every process-global the trim extension touches
    (the standing suite-order class): the cached env parse, both
    fail-closed counters (fires + trims), the per-build stats list
    (identity preserved via slice assignment), the env var, the
    strict-reference flag, and the gc enabled state.'''
    def __enter__(self):
        self._parsed = dfmod._STEADY_MEM_RELEASE
        self._fires = dfmod._STEADY_MEM_RELEASE_FIRES
        self._trims = dfmod._STEADY_MEM_RELEASE_TRIMS
        self._stats = list(dfmod._STEADY_MEM_RELEASE_STATS)
        self._had = 'GPU4PYSCF_STEADY_MEM_RELEASE' in os.environ
        self._val = os.environ.get('GPU4PYSCF_STEADY_MEM_RELEASE')
        self._strict = numint._strict_fp64_reference_active
        self._gc_enabled = gc.isenabled()
        return self

    def __exit__(self, *exc):
        dfmod._STEADY_MEM_RELEASE = self._parsed
        dfmod._STEADY_MEM_RELEASE_FIRES = self._fires
        dfmod._STEADY_MEM_RELEASE_TRIMS = self._trims
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
pool = cp.get_default_memory_pool()
json.dump({'e_tot': [e1, e2], 'converged': [c1, c2],
           'fires': int(dfmod._STEADY_MEM_RELEASE_FIRES),
           'trims': int(dfmod._STEADY_MEM_RELEASE_TRIMS),
           'stats_len': len(dfmod._STEADY_MEM_RELEASE_STATS),
           'stats_fired': int(sum(1 for r in dfmod._STEADY_MEM_RELEASE_STATS
                                  if r.get('fired'))),
           'stats_trimmed': int(sum(1 for r in dfmod._STEADY_MEM_RELEASE_STATS
                                    if r.get('trimmed'))),
           'pool_total_b': int(pool.total_bytes())},
          open(sys.argv[1], 'w'))
'''


class TestSteadyPoolTrim(unittest.TestCase):

    def test_knob_parse_mapping(self):
        with _TrimState():
            for raw, armed, trims in (
                    (None, False, False), ('', False, False),
                    ('0', False, False), ('false', False, False),
                    ('False', False, False),
                    ('1', True, False), ('true', True, False),
                    ('True', True, False),
                    ('trim', True, True)):
                with self.subTest(raw=raw):
                    if raw is None:
                        os.environ.pop(
                            'GPU4PYSCF_STEADY_MEM_RELEASE', None)
                    else:
                        os.environ[
                            'GPU4PYSCF_STEADY_MEM_RELEASE'] = raw
                    dfmod._STEADY_MEM_RELEASE = None
                    self.assertIs(dfmod._steady_mem_release(), armed)
                    self.assertIs(dfmod._steady_mem_release_trims(),
                                  trims)
            with self.subTest(raw='bogus'):
                os.environ['GPU4PYSCF_STEADY_MEM_RELEASE'] = 'bogus'
                dfmod._STEADY_MEM_RELEASE = None
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    value = dfmod._steady_mem_release()
                    trims = dfmod._steady_mem_release_trims()
                self.assertIs(value, False)
                self.assertIs(trims, False)
                self.assertEqual(len(w), 1)
                self.assertIn('GPU4PYSCF_STEADY_MEM_RELEASE',
                              str(w[0].message))

    def test_trim_value_neutrality_bitwise(self):
        # free_all_blocks releases only FREE blocks: every LIVE value
        # is bitwise identical across a fired trim (the paced collect
        # plus the pool trim, the deployed 'trim' mode pace).  The
        # knob never touches arithmetic.
        with _TrimState():
            _arm('off')
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
            live = cp.asarray(cp.arange(64, dtype=cp.float64))
            fires_before = dfmod._STEADY_MEM_RELEASE_FIRES
            trims_before = dfmod._STEADY_MEM_RELEASE_TRIMS
            _arm('trim')
            dfmod._steady_mem_release_pace(True)  # a real fired trim
            v2 = mf.get_veff(mol_sph, dm)
            cderi_after = cp.asnumpy(mf.with_df._cderi[0])
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_before + 1)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS,
                             trims_before + 1)
            self.assertTrue(bool(cp.array_equal(v1, v2)))
            self.assertTrue(np.array_equal(cderi_before, cderi_after))
            self.assertTrue(bool(cp.array_equal(
                live, cp.arange(64, dtype=cp.float64))))
            last = dfmod._STEADY_MEM_RELEASE_STATS[-1]
            self.assertTrue(last['armed'] and last['fired']
                            and last['trimmed'])
            self.assertEqual(last['mode'], 'trim')
            self.assertLessEqual(last['pool_total_after_b'],
                                 last['pool_total_before_b'])

    def test_trim_releases_retained_free_blocks(self):
        # the mechanism: dead arrays leave their blocks retained by
        # the pool (free blocks, pool_total at the high-water mark); a
        # paced trim releases exactly the retained free blocks
        # (pool_total drops) while every LIVE array survives verbatim.
        with _TrimState():
            _arm('off')
            pool = cp.get_default_memory_pool()
            keep = [cp.ones(4096, dtype=cp.float64) for _ in range(8)]
            drop = [cp.ones(4096, dtype=cp.float64) for _ in range(64)]
            del drop
            # the 64 dropped arrays' blocks are now FREE and retained
            # by the pool: total sits above the live level
            total_after_drop = pool.total_bytes()
            self.assertGreater(total_after_drop, pool.used_bytes())
            fires_before = dfmod._STEADY_MEM_RELEASE_FIRES
            trims_before = dfmod._STEADY_MEM_RELEASE_TRIMS
            _arm('trim')
            dfmod._steady_mem_release_pace(True)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_before + 1)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS,
                             trims_before + 1)
            # the trim released the retained free blocks
            self.assertLess(pool.total_bytes(), total_after_drop)
            # live blocks are NEVER released: the kept arrays stay
            # allocated and bitwise verbatim
            self.assertGreaterEqual(pool.used_bytes(), 8 * 4096 * 8)
            self.assertTrue(all(bool(cp.array_equal(
                k, cp.ones(4096, dtype=cp.float64))) for k in keep))
            del keep

    def test_gc_mode_never_trims(self):
        # the landed e727 semantics VERBATIM: '1' fires the paced
        # collect and NEVER the trim; the stats mode is 'gc'.
        with _TrimState():
            _arm('gc')
            trims_before = dfmod._STEADY_MEM_RELEASE_TRIMS
            fires_before = dfmod._STEADY_MEM_RELEASE_FIRES
            dfmod._steady_mem_release_pace(True)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_before + 1)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS,
                             trims_before)
            last = dfmod._STEADY_MEM_RELEASE_STATS[-1]
            self.assertTrue(last['armed'] and last['fired'])
            self.assertFalse(last['trimmed'])
            self.assertEqual(last['mode'], 'gc')
            # knob-off passes record the off mode and never fire
            _arm('off')
            dfmod._steady_mem_release_pace(True)
            last = dfmod._STEADY_MEM_RELEASE_STATS[-1]
            self.assertFalse(last['armed'] or last['fired']
                             or last['trimmed'])
            self.assertEqual(last['mode'], 'off')

    def test_strict_reference_inertness(self):
        # the strict fp64-reference context (the REAL
        # numint.strict_fp64_reference_ctx) never feels either
        # mechanism: armed is False, fires 0, trims 0.
        with _TrimState():
            _arm('trim')
            with numint.strict_fp64_reference_ctx():
                dfmod._steady_mem_release_pace(True)
                dfmod._steady_mem_release_pace(True)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES, 0)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS, 0)
            self.assertTrue(all(not r['armed'] for r
                                in dfmod._STEADY_MEM_RELEASE_STATS))
            # and outside the ctx the same knob fires again (no leak
            # of the refusal)
            dfmod._steady_mem_release_pace(True)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES, 1)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS, 1)

    def test_no_leak_rerun(self):
        # after trim-mode usage, re-arming OFF stops both mechanisms:
        # the counters freeze, new builds record armed False, and the
        # values stay at the same-lane repeat band.
        with _TrimState():
            _arm('trim')
            mf = dft.RKS(mol_sph, xc='b3lyp')
            mf = mf.density_fit()
            mf.grids.level = 1
            mf.conv_tol = 1e-10
            mf.with_df.build()
            e1 = float(mf.kernel())
            self.assertTrue(mf.converged)
            self.assertGreaterEqual(
                dfmod._STEADY_MEM_RELEASE_FIRES, 1)
            self.assertGreaterEqual(
                dfmod._STEADY_MEM_RELEASE_TRIMS, 1)
            fires_after_on = dfmod._STEADY_MEM_RELEASE_FIRES
            trims_after_on = dfmod._STEADY_MEM_RELEASE_TRIMS
            _arm('off')
            mf2 = dft.RKS(mol_sph, xc='b3lyp')
            mf2 = mf2.density_fit()
            mf2.grids.level = 1
            mf2.conv_tol = 1e-10
            mf2.with_df.build()
            e2 = float(mf2.kernel())
            self.assertTrue(mf2.converged)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_FIRES,
                             fires_after_on)
            self.assertEqual(dfmod._STEADY_MEM_RELEASE_TRIMS,
                             trims_after_on)
            self.assertFalse(dfmod._STEADY_MEM_RELEASE_STATS[-1][
                'armed'])
            self.assertLessEqual(abs(e2 - e1), 1e-10)

    def test_subprocess_arms(self):
        # fresh process per mode (off / gc / trim); e_tot agreement
        # at the documented cross-process band; engagement CAND-side
        # only: trim-mode trims == fires == stats_len, gc-mode trims
        # 0 with fires > 0, off-mode both 0.
        outs = {}
        with tempfile.TemporaryDirectory() as tmp:
            script = os.path.join(tmp, '_pool_trim_subproc.py')
            with open(script, 'w') as f:
                f.write(_SUBPROC_STEP)
            for mode in ('off', 'gc', 'trim'):
                out = os.path.join(tmp, 'subproc-%s.json' % mode)
                env = dict(os.environ)
                env['PYTHONPATH'] = os.pathsep.join(
                    [os.path.abspath(os.path.join(
                        os.path.dirname(__file__), '..', '..', '..'))]
                    + ([env['PYTHONPATH']] if env.get('PYTHONPATH')
                       else []))
                env['GPU4PYSCF_STEADY_MEM_RELEASE'] = {
                    'off': '0', 'gc': '1', 'trim': 'trim'}[mode]
                cp_out = subprocess.run(
                    [sys.executable, script, out], env=env,
                    capture_output=True, text=True, timeout=600)
                self.assertEqual(cp_out.returncode, 0,
                                 cp_out.stderr[-2000:])
                with open(out) as f:
                    outs[mode] = json.load(f)
        for mode in ('off', 'gc', 'trim'):
            self.assertTrue(all(outs[mode]['converged']))
        self.assertEqual(outs['off']['fires'], 0)
        self.assertEqual(outs['off']['trims'], 0)
        self.assertEqual(outs['off']['stats_fired'], 0)
        self.assertEqual(outs['off']['stats_trimmed'], 0)
        self.assertGreaterEqual(outs['gc']['fires'], 2)
        self.assertEqual(outs['gc']['trims'], 0)
        self.assertEqual(outs['gc']['stats_trimmed'], 0)
        self.assertEqual(outs['gc']['stats_fired'],
                         outs['gc']['fires'])
        self.assertGreaterEqual(outs['trim']['fires'], 2)
        self.assertEqual(outs['trim']['trims'],
                         outs['trim']['fires'])
        self.assertEqual(outs['trim']['stats_trimmed'],
                         outs['trim']['trims'])
        self.assertEqual(outs['trim']['stats_len'], 2)
        for i in (0, 1):
            self.assertLessEqual(
                abs(outs['trim']['e_tot'][i] - outs['off']['e_tot'][i]),
                1e-10)
            self.assertLessEqual(
                abs(outs['gc']['e_tot'][i] - outs['off']['e_tot'][i]),
                1e-10)


if __name__ == '__main__':
    print('full unittest suite')
    unittest.main()

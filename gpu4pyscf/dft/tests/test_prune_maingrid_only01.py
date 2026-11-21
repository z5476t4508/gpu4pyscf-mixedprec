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
'''SCF-PRUNE-MAINGRID-ONLY-01 focused Phase-A regression (2026-11-08).

The opt-in per-grid prune knob small_rho_cutoff_nlc (KohnShamDFT class
attribute, DEFAULT None = inherit small_rho_cutoff = verbatim upstream
both-prune semantics) extends the existing first-call density-threshold
prune wiring (rks.py initialize_grids) with per-grid cutoff semantics:
0 keeps the VV10 nlcgrids FULL (the entry-476 trajectory-coupling
question: which prune side carries the union hazard), >0 prunes the
nlcgrids at that independent threshold.  The main-grids branch
(rks.py:45-47 pre-change) is byte-identical; UKS inherits via the
existing uks.py rks.initialize_grids delegation; no tolerance, gate or
other default is touched.  rks_lowmem (a separate engine class, off the
benchmark lanes) is a DISCLOSED non-carrier of the knob (absence
contract below).

These tests precede any GPU measurement (preregistration
prunemain01-v1, frozen 2026-11-08T14:17:58Z): class defaults and
absence contracts, knob-off inertness (zero prune events anywhere),
None-inherit both-prune verbatim call/threshold sequence, main-only /
nlc-only / independent-threshold semantics with built-size controls,
UKS reachability, negative-value guarding by the existing >1e-20
conditions, and disarmed-vs-armed same-basin energy sanity.  The prune
instrument is a gen_grid.Grids.prune_by_density_ class wrap recording
(grid-object identity, threshold, pre/post sizes) -- evidence-state
only, restored in finally; it adds no computational state (the
understanding mirrors the entry-475 rhoprune01b driver instrument).
'''
import inspect
import unittest
import pyscf
from gpu4pyscf.dft import rks, uks, gen_grid

PRUNE_EVENTS = []

def _install_prune_wrap():
    orig = gen_grid.Grids.prune_by_density_
    def wrapped(self, rho, threshold=0):
        pre = self.coords.shape[0]
        ret = orig(self, rho, threshold)
        PRUNE_EVENTS.append({'grid': self, 'threshold': threshold,
                             'pre': pre, 'post': self.coords.shape[0]})
        return ret
    gen_grid.Grids.prune_by_density_ = wrapped
    def restore():
        gen_grid.Grids.prune_by_density_ = orig
    return restore

def _tag_events(mf):
    out = []
    for ev in PRUNE_EVENTS:
        g = ev['grid']
        if g is mf.grids:
            tag = 'grids'
        elif g is mf.nlcgrids:
            tag = 'nlcgrids'
        else:
            tag = 'other'
        out.append((tag, ev['threshold'], ev['pre'], ev['post']))
    return out

def _run_rks(cutoff=0.0, nlc=None):
    mol = pyscf.M(atom='O 0 0 0.1174; H -0.757 0 -0.4696; H 0.757 0 -0.4696',
                  basis='def2-svp', output='/dev/null', verbose=0)
    mf = rks.RKS(mol)
    mf.xc = 'wb97x-v'
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 50
    mf.small_rho_cutoff = cutoff
    if nlc is not None:
        mf.small_rho_cutoff_nlc = nlc
    PRUNE_EVENTS.clear()
    restore = _install_prune_wrap()
    try:
        e = mf.kernel()
    finally:
        restore()
    events = list(_tag_events(mf))
    return mf, float(e), events

def _run_uks(cutoff=0.0, nlc=None):
    mol = pyscf.M(atom='O 0 0 0.1174; H -0.757 0 -0.4696',
                  basis='def2-svp', spin=1, output='/dev/null', verbose=0)
    mf = uks.UKS(mol)
    mf.xc = 'wb97x-v'
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-10
    mf.max_cycle = 50
    mf.small_rho_cutoff = cutoff
    if nlc is not None:
        mf.small_rho_cutoff_nlc = nlc
    PRUNE_EVENTS.clear()
    restore = _install_prune_wrap()
    try:
        e = mf.kernel()
    finally:
        restore()
    events = list(_tag_events(mf))
    return mf, float(e), events

class TestScopeContracts(unittest.TestCase):
    '''Absence contracts: GREEN at parent 4c603d0 by design (the
    implementation changes none of these facts); they pin the knob's
    scope so later edits cannot silently widen it.'''

    def test_01_class_defaults_and_absence_contracts(self):
        # tree default prunes NOTHING anywhere
        self.assertEqual(rks.KohnShamDFT.small_rho_cutoff, 0)
        # absent at parent, present-but-None after Phase A
        self.assertIsNone(
            getattr(rks.KohnShamDFT, 'small_rho_cutoff_nlc', None))
        # a pure mf-attribute knob: no env parsing enters rks.py
        src = inspect.getsource(rks)
        self.assertNotIn('os.environ', src)
        self.assertNotIn('getenv', src)
        # rks_lowmem is a DISCLOSED non-carrier (separate engine class,
        # off the benchmark lanes): its prune calls pass no threshold
        lowmem_src = inspect.getsource(
            __import__('gpu4pyscf.dft.rks_lowmem', fromlist=['x'])
            .initialize_grids)
        self.assertIn('rks.prune_small_rho_grids_(', lowmem_src)
        self.assertNotIn('threshold=', lowmem_src)

class TestPerGridSemantics(unittest.TestCase):

    def test_02_none_inherits_both_prune_verbatim(self):
        # small_rho_cutoff_nlc left at None: the upstream/475-measured
        # both-prune class, same call sequence and thresholds
        mf, e, ev = _run_rks(cutoff=1e-7)
        self.assertTrue(mf.converged)
        self.assertEqual([t[0] for t in ev], ['grids', 'nlcgrids'])
        self.assertEqual([t[1] for t in ev], [1e-7, 1e-7])
        self.assertLess(ev[0][3], ev[0][2])
        self.assertLess(ev[1][3], ev[1][2])
        self.assertEqual(mf.grids.coords.shape[0], ev[0][3])
        self.assertEqual(mf.nlcgrids.coords.shape[0], ev[1][3])

    def test_03_maingrid_only_semantics(self):
        # THE U2 semantics: cutoff=1e-7 + nlc=0 prunes ONLY the main
        # grids; the nlcgrids stay FULL (built size from the disarmed
        # control run in the same process/settings)
        mf0, e0, ev0 = _run_rks()
        built_main = mf0.grids.coords.shape[0]
        built_nlc = mf0.nlcgrids.coords.shape[0]
        self.assertEqual(ev0, [])
        mf, e, ev = _run_rks(cutoff=1e-7, nlc=0)
        self.assertTrue(mf.converged)
        self.assertEqual(ev, [('grids', 1e-7, built_main,
                               mf.grids.coords.shape[0])])
        self.assertLess(mf.grids.coords.shape[0], built_main)
        # nlcgrids untouched: no event AND full built size
        self.assertEqual(mf.nlcgrids.coords.shape[0], built_nlc)

    def test_04_independent_nlc_threshold(self):
        # nlc=1e-4 prunes the nlcgrids at 1e-4 while the main grids keep
        # the 1e-7 cutoff (threshold visible in the instrument)
        mf, e, ev = _run_rks(cutoff=1e-7, nlc=1e-4)
        self.assertTrue(mf.converged)
        self.assertEqual([t[0] for t in ev], ['grids', 'nlcgrids'])
        self.assertEqual([t[1] for t in ev], [1e-7, 1e-4])
        self.assertLess(ev[0][3], ev[0][2])
        self.assertLess(ev[1][3], ev[1][2])

    def test_05_nlc_only_semantics(self):
        # cutoff=0 + nlc=1e-7: the main grids stay FULL, nlcgrids prune
        mf0, e0, ev0 = _run_rks()
        built_main = mf0.grids.coords.shape[0]
        mf, e, ev = _run_rks(cutoff=0.0, nlc=1e-7)
        self.assertTrue(mf.converged)
        self.assertEqual([t[0] for t in ev], ['nlcgrids'])
        self.assertEqual(ev[0][1], 1e-7)
        self.assertLess(ev[0][3], ev[0][2])
        self.assertEqual(mf.grids.coords.shape[0], built_main)

    def test_06_uks_maingrid_only(self):
        # UKS reachability via the uks.py rks.initialize_grids
        # delegation (the 573/819 anion-cell class path)
        mf0, e0, ev0 = _run_uks()
        built_main = mf0.grids.coords.shape[0]
        built_nlc = mf0.nlcgrids.coords.shape[0]
        self.assertEqual(ev0, [])
        mf, e, ev = _run_uks(cutoff=1e-7, nlc=0)
        self.assertTrue(mf.converged)
        self.assertEqual([t[0] for t in ev], ['grids'])
        self.assertEqual(ev[0][1], 1e-7)
        self.assertLess(ev[0][3], ev[0][2])
        self.assertEqual(mf.nlcgrids.coords.shape[0], built_nlc)

    def test_07_negative_nlc_value_guarded(self):
        # negative values never prune the nlcgrids (the existing
        # >1e-20 conditions guard; no crash, no warning machinery --
        # upstream small_rho_cutoff simplicity is mirrored)
        mf, e, ev = _run_rks(cutoff=1e-7, nlc=-1.0)
        self.assertTrue(mf.converged)
        self.assertEqual([t[0] for t in ev], ['grids'])
        self.assertLess(ev[0][3], ev[0][2])

    def test_08_armed_energy_same_basin(self):
        # tiny-cell same-basin sanity ONLY (formal 1e-6 Ha / 3e-5
        # Eh/Bohr gates live in the preregistered GPU phase; these are
        # not them)
        mf0, e0, _ = _run_rks()
        mf1, e1, _ = _run_rks(cutoff=1e-7, nlc=0)
        self.assertTrue(mf0.converged and mf1.converged)
        self.assertLessEqual(abs(e1 - e0), 1e-8)

    def test_09_disarmed_run_to_run_stable(self):
        # knob-off lanes are untouched by construction (both branch
        # conditions false; nothing prunes; the only new lines are the
        # nlc_cutoff read + None-inherit, control-flow no-ops when
        # small_rho_cutoff <= 1e-20).  Plain-engine e_tot is NOT
        # bitwise run-to-run (the j3c/atomicAdd ULP class), so the
        # honest contract is a tight envelope + structural equality.
        mf1, e1, ev1 = _run_rks()
        mf2, e2, ev2 = _run_rks()
        self.assertTrue(mf1.converged and mf2.converged)
        self.assertLessEqual(abs(e1 - e2), 1e-12)
        self.assertEqual(ev1, [])
        self.assertEqual(ev2, [])
        self.assertEqual(mf1.grids.coords.shape, mf2.grids.coords.shape)
        self.assertEqual(mf1.nlcgrids.coords.shape,
                         mf2.nlcgrids.coords.shape)

if __name__ == '__main__':
    unittest.main()

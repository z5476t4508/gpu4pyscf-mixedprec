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
import unittest

import numpy as np
import pyscf

from gpu4pyscf import scf
from gpu4pyscf.geomopt import conv_schedule


def setUpModule():
    global mol
    mol = pyscf.M(atom='''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000''',
                  basis='sto-3g', output='/dev/null', verbose=1)


def tearDownModule():
    global mol
    mol.stdout.close()
    del mol


class KnownValues(unittest.TestCase):

    def test_no_gradient_yet_keeps_caller_tol(self):
        '''the first geometry has no force to scale against'''
        sched = conv_schedule.SCFConvSchedule()
        self.assertEqual(sched(None, 1e-9), 1e-9)

    def test_ceiling_and_floor_bound_the_result(self):
        sched = conv_schedule.SCFConvSchedule(ceiling=1e-5, floor=1e-8)
        self.assertEqual(sched(1.0, 1e-9), 1e-5)      # huge force -> ceiling
        self.assertEqual(sched(1e-9, 1e-9), 1e-8)     # tiny force -> floor
        # an explicit floor overrides the caller's conv_tol
        self.assertEqual(sched(None, 1e-12), 1e-8)

    def test_caller_conv_tol_is_the_default_floor(self):
        '''without an explicit floor the schedule never asks for less than
        what the caller had already set'''
        sched = conv_schedule.SCFConvSchedule()
        self.assertEqual(sched(1e-8, 1e-7), 1e-7)

    def test_floor_wins_over_a_lower_ceiling(self):
        '''a conv_tol tighter than the ceiling must not be loosened'''
        sched = conv_schedule.SCFConvSchedule(ceiling=1e-5)
        self.assertEqual(sched(1.0, 1e-4), 1e-4)

    def test_tolerance_tracks_the_force(self):
        sched = conv_schedule.SCFConvSchedule()
        tols = [sched(g, 1e-12) for g in (1e-2, 1e-3, 1e-4)]
        self.assertTrue(tols[0] > tols[1] > tols[2])
        # the calibrated bound must hold at the point geomeTRIC converges
        gmax = 4.5e-4
        dg = conv_schedule.GRAD_ERROR_COEFF * sched(gmax, 1e-12) ** .5
        self.assertLess(dg, 0.1 * gmax)

    def test_make_accepts_the_documented_specs(self):
        self.assertIsNone(conv_schedule.make(None))
        self.assertIsNone(conv_schedule.make(False))
        for spec in ('auto', True):
            self.assertIsInstance(conv_schedule.make(spec),
                                  conv_schedule.SCFConvSchedule)
        made = conv_schedule.make({'ceiling': 1e-4, 'rel_error': 0.05})
        self.assertEqual(made.ceiling, 1e-4)
        self.assertEqual(made.rel_error, 0.05)
        self.assertEqual(conv_schedule.make(lambda gmax, floor: 7)(None, 1), 7)
        self.assertRaises(ValueError, conv_schedule.make, 'tight')

    def test_rejects_nonsense_parameters(self):
        self.assertRaises(ValueError, conv_schedule.SCFConvSchedule, rel_error=0)
        self.assertRaises(ValueError, conv_schedule.SCFConvSchedule, rel_error=2)
        self.assertRaises(ValueError, conv_schedule.SCFConvSchedule, ceiling=0)
        self.assertRaises(ValueError, conv_schedule.SCFConvSchedule,
                          ceiling=1e-8, floor=1e-5)

    def test_scanner_loosens_then_restores_conv_tol(self):
        '''the schedule must act on the second geometry and leave the
        mean-field object's own conv_tol untouched afterwards'''
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        mf.conv_tol_schedule = 'auto'
        g_scanner = mf.nuc_grad_method().as_scanner()

        seen = []
        base = g_scanner.base
        base_cls = type(base)
        orig_call = base_cls.__call__

        def spy(self, *args, **kwargs):
            seen.append(self.conv_tol)
            return orig_call(self, *args, **kwargs)

        base_cls.__call__ = spy
        try:
            g_scanner(mol)
            coords = mol.atom_coords() + 1e-3
            g_scanner(mol.set_geom_(coords, unit='Bohr', inplace=False))
        finally:
            base_cls.__call__ = orig_call

        self.assertEqual(seen[0], 1e-10)          # first geometry: unchanged
        self.assertGreater(seen[1], 1e-10)        # second: loosened
        self.assertEqual(mf.conv_tol, 1e-10)      # restored on the way out

    def test_scanner_ignores_the_schedule_when_unset(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-10
        g_scanner = mf.nuc_grad_method().as_scanner()
        g_scanner(mol)
        g_scanner(mol)
        self.assertEqual(g_scanner.base.conv_tol, 1e-10)
        self.assertIsNone(g_scanner._last_gmax)

    def test_gradient_object_overrides_the_mean_field_setting(self):
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol_schedule = 'auto'
        g = mf.nuc_grad_method()
        g.conv_tol_schedule = False
        self.assertIsNone(g.as_scanner()._resolve_conv_tol_schedule())

    def test_loose_schedule_keeps_the_gradient_usable(self):
        '''a scheduled gradient must stay far inside geomeTRIC's 3e-4 Eh/Bohr
        convergence threshold'''
        mf = scf.RHF(mol).density_fit()
        mf.conv_tol = 1e-12
        ref = mf.nuc_grad_method().as_scanner()(mol)[1]

        mf2 = scf.RHF(mol).density_fit()
        mf2.conv_tol = 1e-12
        mf2.conv_tol_schedule = {'ceiling': 1e-5}
        scanner = mf2.nuc_grad_method().as_scanner()
        scanner(mol)                      # first geometry sets the baseline
        scanner._last_gmax = 1e-1         # pretend we are far from a minimum
        de = scanner(mol)[1]
        self.assertLess(np.abs(de - ref).max(), 3e-4)


if __name__ == '__main__':
    print('Full tests for the adaptive SCF convergence schedule')
    unittest.main()

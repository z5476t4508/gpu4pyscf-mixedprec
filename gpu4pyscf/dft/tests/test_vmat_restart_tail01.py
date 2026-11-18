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
'''RESTART-TAIL-01 focused regression (2026-11-18).

The opt-in GPU4PYSCF_SCF_VMAT_RESTART_TAIL companion (DEFAULT OFF) for
GPU4PYSCF_SCF_VMAT_F32=headtail keeps the vmat f32 head lane DISARMED
when the SCF kernel restarts from an explicit dm0 (kernel(dm0=...)):
the restart runs the deployed exact-fp64 tail map from cycle 1.
Measured carrier (STEADYCYCLE-ATTRIB-01, e714, CARRIER_VMAT): the
fp32 vmat head tips the marginal restart cycle one late on the
16384-blocksize path; the rho head is innocent (NORHO reproduces the
coupling) and cold kernels (G0, dm0 is None) are cycle-exact.

Asserted here, before any GPU measurement: the parse mapping; the
arming gate on BOTH dm0 classes (dm0 supplied -> head disarmed with
the knob on, armed verbatim with it off; dm0 None -> armed verbatim
either way); the rho lane UNTOUCHED at restart (the attribution's
rho-innocent discipline); strict-reference inertness; the e_tot
envelope of the disarm (the knob-on restart is strictly more fp64);
convergence; the no-leak re-run; and a fresh-process subprocess arm
on the env-var path.
'''
import os
import subprocess
import sys
import unittest

import numpy as np
import pyscf
import cupy as cp

from gpu4pyscf import dft
from gpu4pyscf.dft import numint


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


class _KnobState:
    '''Saves/restores every process-global this policy touches (the
    standing suite-order class: the env caches, both lane flags, the
    rho-head block counter and the castscale stats recorded on every
    contr_f32 task pass).'''
    def __enter__(self):
        self._rt = numint._SCF_VMAT_RESTART_TAIL
        self._rt_had = 'GPU4PYSCF_SCF_VMAT_RESTART_TAIL' in os.environ
        self._rt_val = os.environ.get('GPU4PYSCF_SCF_VMAT_RESTART_TAIL')
        self._vm = numint._SCF_VMAT_F32
        self._vm_had = 'GPU4PYSCF_SCF_VMAT_F32' in os.environ
        self._vm_val = os.environ.get('GPU4PYSCF_SCF_VMAT_F32')
        self._rho = numint._SCF_RHO_F32
        self._rho_had = 'GPU4PYSCF_SCF_RHO_F32' in os.environ
        self._rho_val = os.environ.get('GPU4PYSCF_SCF_RHO_F32')
        self._lane_v = numint._scf_vmat_f32_lane
        self._lane_r = numint._scf_rho_f32_lane
        self._blocks = numint._rho_f32_head_blocks
        self._stats = dict(numint._VMAT_CASTSCALE_STATS)
        return self

    def __exit__(self, *exc):
        numint._SCF_VMAT_RESTART_TAIL = self._rt
        numint._SCF_VMAT_F32 = self._vm
        numint._SCF_RHO_F32 = self._rho
        for had, key, val in ((self._rt_had,
                               'GPU4PYSCF_SCF_VMAT_RESTART_TAIL',
                               self._rt_val),
                              (self._vm_had,
                               'GPU4PYSCF_SCF_VMAT_F32', self._vm_val),
                              (self._rho_had,
                               'GPU4PYSCF_SCF_RHO_F32', self._rho_val)):
            if had:
                os.environ[key] = val
            else:
                os.environ.pop(key, None)
        numint._scf_vmat_f32_lane = self._lane_v
        numint._scf_rho_f32_lane = self._lane_r
        numint._rho_f32_head_blocks = self._blocks
        numint._VMAT_CASTSCALE_STATS.clear()
        numint._VMAT_CASTSCALE_STATS.update(self._stats)
        return False


def _arm(head_knob, restart_tail):
    numint._SCF_VMAT_F32 = None
    numint._SCF_RHO_F32 = None
    numint._SCF_VMAT_RESTART_TAIL = None
    os.environ['GPU4PYSCF_SCF_VMAT_F32'] = \
        'headtail' if head_knob else '0'
    os.environ['GPU4PYSCF_SCF_RHO_F32'] = 'headtail'
    os.environ['GPU4PYSCF_SCF_VMAT_RESTART_TAIL'] = \
        '1' if restart_tail else '0'


def _fresh_mf():
    mf = dft.RKS(mol_sph, xc='b3lyp')
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    mf.max_cycle = 30
    mf.precision_mode = 'fp64'
    return mf


def _run(dm0=None):
    '''One whole SCF under the currently armed env; returns
    (e_tot, converged, vmat_head_fired, rho_head_blocks_delta).'''
    numint._VMAT_CASTSCALE_STATS.clear()
    stats0 = dict(numint._VMAT_CASTSCALE_STATS)
    blocks0 = numint._rho_f32_head_blocks
    mf = _fresh_mf()
    e = mf.kernel(dm0=dm0)
    stats1 = dict(numint._VMAT_CASTSCALE_STATS)
    return (float(e), bool(mf.converged),
            stats1 != stats0,          # a contr_f32 task pass ran
            numint._rho_f32_head_blocks - blocks0)


def _converged_dm():
    mf = _fresh_mf()
    mf.kernel()
    return cp.asarray(mf.make_rdm1())


_SUBPROC = r'''
import sys
sys.path.insert(0, {root!r})
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import dft
mol = pyscf.M(atom="""
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000""",
              basis="def2-svp", output="/dev/null", verbose=0)
mf = dft.RKS(mol, xc="b3lyp")
mf.grids.level = 1
mf.conv_tol = 1e-10
mf.max_cycle = 30
mf.precision_mode = "fp64"
mf.kernel()
dm = cp.asarray(mf.make_rdm1())
mf2 = dft.RKS(mol, xc="b3lyp")
mf2.grids.level = 1
mf2.conv_tol = 1e-10
mf2.max_cycle = 30
mf2.precision_mode = "fp64"
e = mf2.kernel(dm0=dm)
print("RESULT", repr(float(e)), mf2.converged,
      int(getattr(mf2, "cycles", 0) or 0))
'''


class RestartTailParse(unittest.TestCase):
    def setUp(self):
        self._saved = numint._SCF_VMAT_RESTART_TAIL
        self._had = 'GPU4PYSCF_SCF_VMAT_RESTART_TAIL' in os.environ
        self._val = os.environ.get('GPU4PYSCF_SCF_VMAT_RESTART_TAIL')

    def tearDown(self):
        numint._SCF_VMAT_RESTART_TAIL = self._saved
        if self._had:
            os.environ['GPU4PYSCF_SCF_VMAT_RESTART_TAIL'] = self._val
        else:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_RESTART_TAIL', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_VMAT_RESTART_TAIL', None)
        else:
            os.environ['GPU4PYSCF_SCF_VMAT_RESTART_TAIL'] = raw
        numint._SCF_VMAT_RESTART_TAIL = None
        return numint._scf_vmat_restart_tail()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('false'), False)
        self.assertIs(self._parse('False'), False)
        self.assertIs(self._parse('1'), True)
        self.assertIs(self._parse('true'), True)
        self.assertIs(self._parse('True'), True)
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('headtail'), False)


class RestartTailGating(unittest.TestCase):
    '''The arming gate on both dm0 classes, the rho-lane innocence,
    the envelope, convergence, and the no-leak re-run.'''

    def test_dm0_supplied_knob_off_head_armed(self):
        # deployed behavior verbatim: the head arms at a restart
        with _KnobState():
            _arm(True, False)
            dm = _converged_dm()
            e, conv, fired, rho = _run(dm0=dm)
            self.assertTrue(conv)
            self.assertTrue(fired, 'vmat f32 head must arm at restart '
                                   'with the companion off')
            self.assertGreater(rho, 0)
            ref = e
        # keep for the envelope test
        self.__class__._ref_restart_e = ref

    def test_dm0_supplied_knob_on_head_disarmed(self):
        with _KnobState():
            _arm(True, True)
            dm = _converged_dm()
            e, conv, fired, rho = _run(dm0=dm)
            self.assertTrue(conv)
            self.assertFalse(fired, 'vmat f32 head must stay disarmed '
                                    'at a restart with the companion on')
            self.assertGreater(rho, 0, 'the rho head lane is untouched '
                                       '(the e714 rho-innocent '
                                       'discipline)')

    def test_dm0_none_knob_on_head_armed_verbatim(self):
        # cold kernels are cycle-exact in the attribution; the policy
        # must never touch them
        with _KnobState():
            _arm(True, True)
            e, conv, fired, rho = _run()
            self.assertTrue(conv)
            self.assertTrue(fired, 'cold kernel (dm0 None): the head '
                                   'arms verbatim with the knob on')
            self.assertGreater(rho, 0)

    def test_envelope_and_cycles(self):
        # the knob-on restart is strictly more fp64: its e_tot sits
        # within the fp32 head's whole-SCF envelope of the knob-off
        # restart (LC-105 measured <= 6.8e-11 Ha at the band cells;
        # this small cell asserts 1e-9, far inside the 1e-6 gate)
        with _KnobState():
            _arm(True, False)
            dm = _converged_dm()
            e_off, conv_off, fired_off, _ = _run(dm0=dm)
            self.assertTrue(conv_off)
            _arm(True, True)
            e_on, conv_on, fired_on, _ = _run(dm0=dm)
            self.assertTrue(conv_on)
            self.assertFalse(fired_on)
            self.assertLessEqual(abs(e_on - e_off), 1e-9)

    def test_strict_reference_inertness(self):
        with _KnobState():
            _arm(True, True)
            dm = _converged_dm()
            numint._VMAT_CASTSCALE_STATS.clear()
            mf = _fresh_mf()
            mf._strict_fp64_reference = True
            mf.kernel(dm0=dm)
            self.assertEqual(numint._VMAT_CASTSCALE_STATS, {},
                             'strict reference must never arm the '
                             'vmat f32 lane, companion or not')

    def test_no_leak_rerun(self):
        # after a knob-on restart disarms the head, a knob-off restart
        # re-arms it (the lane flag restored by _kernel's finally)
        with _KnobState():
            _arm(True, True)
            dm = _converged_dm()
            _, _, fired_on, _ = _run(dm0=dm)
            self.assertFalse(fired_on)
            self.assertFalse(numint._scf_vmat_f32_lane)
            _arm(True, False)
            _, conv, fired_off, _ = _run(dm0=dm)
            self.assertTrue(conv)
            self.assertTrue(fired_off, 'the lane must re-arm after the '
                                       'companion is turned off')

    def test_subprocess_fresh_process(self):
        # the env-var path in a brand-new process (parse cache cold):
        # knob off vs on at a restart, e_tot agreement within the
        # cross-process ULP-plus-envelope band, both converged
        root = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..'))
        code = _SUBPROC.format(root=root)
        outs = {}
        for knob in ('0', '1'):
            env = dict(os.environ)
            env.update({'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
                        'GPU4PYSCF_SCF_RHO_F32': 'headtail',
                        'GPU4PYSCF_SCF_VMAT_RESTART_TAIL': knob})
            for k in list(env):
                if k.startswith('GPU4PYSCF_') and k not in (
                        'GPU4PYSCF_SCF_VMAT_F32',
                        'GPU4PYSCF_SCF_RHO_F32',
                        'GPU4PYSCF_SCF_VMAT_RESTART_TAIL'):
                    env.pop(k)
            r = subprocess.run([sys.executable, '-c', code], env=env,
                               capture_output=True, text=True,
                               timeout=600)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            line = [l for l in r.stdout.splitlines()
                    if l.startswith('RESULT')][-1]
            _, e, conv, cycles = line.split()
            outs[knob] = (float(e), conv == 'True', int(cycles))
        self.assertTrue(outs['0'][1] and outs['1'][1])
        self.assertLessEqual(abs(outs['0'][0] - outs['1'][0]), 1e-9)


if __name__ == '__main__':
    unittest.main()

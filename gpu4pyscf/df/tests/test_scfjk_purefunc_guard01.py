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
'''K32-PUREFUNC-GUARD-01 focused regression (2026-11-18).

The measured-evidence population gate for the opt-in
GPU4PYSCF_SCF_JK_F32 lane (the knob stays DEFAULT OFF): at PURE
functionals the SCF never calls get_k (rks.get_veff's own is_hybrid_xc
branch -- get_jk serves the J path only), and the e704 UNION-K32
falsification measured the lane arming there as +3 deterministic
cycles 14->17 with paired e2e median 0.8402 (a 16% regression; the
e705 readjud re-classed the cell the don't-arm row, population scoping
load-bearing).  The gate refuses the arming exactly there -- the lane
stays False and the SCF runs the deployed exact-fp64 map; hybrids (the
e704/e705 STACK_WIN population) and non-DFT SCFs arm verbatim.
Fail-closed engagement evidence: one _SCF_JK_F32_GUARD_STATS entry per
kernel-entry pass with the knob parse truthy, armed or refused;
_SCF_JK_F32_GUARD_REFUSALS bumps only on actual pure-functional
refusals (knob-off kernels record nothing).

Value assertions use the documented bands: the refusal path is the
knob-off code path verbatim, but sequential in-process SCFs carry the
documented cross-call wobble (test_xc_ao_carryover01) and fresh
processes the 1-2 ULP cross-process class -- the honest envelope is
1e-10 Ha with equal cycle counts; the path identity itself is proven
fail-closed by the lane/counter evidence, never by the value alone.
'''
import os
import subprocess
import sys
import tempfile
import unittest
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000'''

_ARM = '''
import os, sys
sys.path.insert(0, {repo!r})
os.environ['GPU4PYSCF_SCF_JK_F32'] = '{knob}'
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
mol = pyscf.M(atom="""{water}""", basis='def2-svp',
              output='/dev/null', verbose=0)
mf = dft.RKS(mol).density_fit()
mf.xc = 'r2scan'
mf.conv_tol = 1e-10
mf.grids.level = 2
mf.kernel()
line = ('E2EJSON ' + float(mf.e_tot).hex()
        + ' ' + str(bool(mf.converged))
        + ' ' + str(getattr(mf, 'cycles', 0))
        + ' ' + str(df_jk._SCF_JK_F32_GUARD_REFUSALS)
        + ' ' + str(len(df_jk._SCF_JK_F32_GUARD_STATS))
        + ' ' + str(df_jk._scf_jk_f32_head_calls))
if df_jk._SCF_JK_F32_GUARD_STATS:
    s = df_jk._SCF_JK_F32_GUARD_STATS[0]
    line += (' ' + s['population']
             + ' ' + str(bool(s['armed']))
             + ' ' + str(bool(s['refused_pure'])))
print(line)
'''

def setUpModule():
    global mol
    mol = pyscf.M(atom=WATER, basis='def2-svp',
                  output='/dev/null', verbose=0)

def tearDownModule():
    global mol
    mol.stdout.close()
    del mol

def _run_rks(xc, knob='headtail', arm_after=None):
    '''One small DF-RKS on the shared water cell under the given jk
    knob state.  Returns (mf, stats_entries, head_calls_delta,
    refusals_delta).'''
    n_stats = len(df_jk._SCF_JK_F32_GUARD_STATS)
    n_ref = df_jk._SCF_JK_F32_GUARD_REFUSALS
    n_head = df_jk._scf_jk_f32_head_calls
    if knob is None:
        os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)
    else:
        os.environ['GPU4PYSCF_SCF_JK_F32'] = knob
    if arm_after is None:
        os.environ.pop('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', None)
    else:
        os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = str(arm_after)
    df_jk._SCF_JK_F32 = None
    df_jk._SCF_JK_F32_ARM_AFTER = None
    try:
        mf = dft.RKS(mol).density_fit()
        mf.xc = xc
        mf.conv_tol = 1e-10
        mf.grids.level = 2
        mf.kernel()
    finally:
        os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)
        os.environ.pop('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', None)
        df_jk._SCF_JK_F32 = None
        df_jk._SCF_JK_F32_ARM_AFTER = None
    entries = df_jk._SCF_JK_F32_GUARD_STATS[n_stats:]
    return (mf, entries, df_jk._scf_jk_f32_head_calls - n_head,
            df_jk._SCF_JK_F32_GUARD_REFUSALS - n_ref)

class GuardState:
    '''Save/restore every process-global the guard touches (the
    standing suite-order class): the new counter + stats list
    (identity-preserving slice assignment), the lane flag, the cached
    knob parses, the engagement counter, and the env vars.'''
    def __init__(self):
        self.refusals = df_jk._SCF_JK_F32_GUARD_REFUSALS
        self.stats = df_jk._SCF_JK_F32_GUARD_STATS
        self.lane = df_jk._scf_jk_f32_lane
        self.head_calls = df_jk._scf_jk_f32_head_calls
        self.parse = df_jk._SCF_JK_F32
        self.parse_after = df_jk._SCF_JK_F32_ARM_AFTER
        self.env = os.environ.get('GPU4PYSCF_SCF_JK_F32')
        self.env_after = os.environ.get('GPU4PYSCF_SCF_JK_F32_ARM_AFTER')
        self.had_env = 'GPU4PYSCF_SCF_JK_F32' in os.environ
        self.had_env_after = 'GPU4PYSCF_SCF_JK_F32_ARM_AFTER' in os.environ

    def restore(self):
        df_jk._SCF_JK_F32_GUARD_REFUSALS = self.refusals
        self.stats[:] = self.stats[:0]  # identity-preserving clear
        df_jk._scf_jk_f32_lane = self.lane
        df_jk._scf_jk_f32_head_calls = self.head_calls
        df_jk._SCF_JK_F32 = self.parse
        df_jk._SCF_JK_F32_ARM_AFTER = self.parse_after
        if self.had_env:
            os.environ['GPU4PYSCF_SCF_JK_F32'] = self.env
        else:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)
        if self.had_env_after:
            os.environ['GPU4PYSCF_SCF_JK_F32_ARM_AFTER'] = self.env_after
        else:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', None)

class PopulationGateUnit(unittest.TestCase):
    '''The gate classifier itself, no GPU work.'''

    def setUp(self):
        self.state = GuardState()

    def tearDown(self):
        self.state.restore()

    def test_gate_unit(self):
        mf_hyb = dft.RKS(mol)
        mf_hyb.xc = 'b3lyp'
        self.assertTrue(df_jk._scf_jk_f32_population_ok(mf_hyb))
        mf_rsh = dft.RKS(mol)
        mf_rsh.xc = 'wb97m-v'
        self.assertTrue(df_jk._scf_jk_f32_population_ok(mf_rsh))
        mf_pure = dft.RKS(mol)
        mf_pure.xc = 'r2scan'
        self.assertFalse(df_jk._scf_jk_f32_population_ok(mf_pure))
        # non-DFT SCF: no .xc attribute, J+K both exist -- armable
        self.assertTrue(df_jk._scf_jk_f32_population_ok(scf.RHF(mol)))
        # unclassifiable xc (no libxc on the numint): fail-open on
        # classification only, never on value -- the deployed arming
        class _NoNumint:
            pass
        mf_stub = _NoNumint()
        mf_stub.xc = 'b3lyp'
        self.assertTrue(df_jk._scf_jk_f32_population_ok(mf_stub))

class PureFunctionalRefusal(unittest.TestCase):
    '''The refusal at the e704 falsification population: the lane never
    arms; the value sits at the documented wobble envelope with equal
    cycle counts (the path identity is the fail-closed counter
    evidence, not the value alone).'''

    def setUp(self):
        self.state = GuardState()

    def tearDown(self):
        self.state.restore()

    def test_refusal_envelope_knob_off(self):
        mf_off, entries_off, head_off, ref_off = _run_rks(
            'r2scan', knob=None)
        self.assertTrue(mf_off.converged)
        self.assertEqual(entries_off, [])  # knob-off kernels record nothing
        self.assertEqual(ref_off, 0)
        mf_on, entries_on, head_on, ref_on = _run_rks(
            'r2scan', knob='headtail')
        self.assertTrue(mf_on.converged)
        # the refusal: one stats entry, population pure, never armed
        self.assertEqual(len(entries_on), 1)
        self.assertEqual(entries_on[0]['population'], 'pure')
        self.assertFalse(entries_on[0]['armed'])
        self.assertTrue(entries_on[0]['refused_pure'])
        self.assertEqual(entries_on[0]['arm_after'], 0)
        self.assertEqual(ref_on, 1)
        self.assertEqual(head_on, 0)  # the get_jk fp32 branch never fired
        # the refused path IS the knob-off path: the honest in-process
        # envelope (sequential-SCF cross-call wobble) + equal cycles
        self.assertLessEqual(abs(mf_on.e_tot - mf_off.e_tot), 1e-10)
        self.assertEqual(getattr(mf_on, 'cycles', None),
                         getattr(mf_off, 'cycles', None))

    def test_refusal_pbe_value(self):
        # the same refusal at a second pure functional (pbe), knob '1'
        mf_off, _, head_off, _ = _run_rks('pbe', knob=None)
        mf_on, entries_on, head_on, ref_on = _run_rks('pbe', knob='1')
        self.assertEqual(entries_on[0]['population'], 'pure')
        self.assertTrue(entries_on[0]['refused_pure'])
        self.assertEqual(ref_on, 1)
        self.assertEqual(head_on, 0)
        self.assertEqual(head_off, 0)
        self.assertTrue(mf_on.converged and mf_off.converged)
        self.assertLessEqual(abs(mf_on.e_tot - mf_off.e_tot), 1e-10)

    def test_delayed_window_never_opens(self):
        # ARM_AFTER at a pure functional: the latch never opens (the
        # _kernel_body gate mirrors the entry refusal), fp64 throughout
        mf_off, _, _, _ = _run_rks('r2scan', knob=None)
        mf_on, entries_on, head_on, _ = _run_rks(
            'r2scan', knob='headtail', arm_after=1)
        self.assertTrue(mf_on.converged)
        self.assertEqual(entries_on[0]['population'], 'pure')
        self.assertTrue(entries_on[0]['refused_pure'])
        self.assertEqual(entries_on[0]['arm_after'], 1)
        self.assertFalse(entries_on[0]['armed'])
        self.assertEqual(head_on, 0)
        self.assertLessEqual(abs(mf_on.e_tot - mf_off.e_tot), 1e-10)
        self.assertEqual(getattr(mf_on, 'cycles', None),
                         getattr(mf_off, 'cycles', None))

class HybridArmedVerbatim(unittest.TestCase):
    '''The e704/e705 STACK_WIN population arms exactly as before the
    gate: stats population hybrid, armed, the engagement counter bumps.'''

    def setUp(self):
        self.state = GuardState()

    def tearDown(self):
        self.state.restore()

    def test_hybrid_arms(self):
        mf, entries, head_delta, ref_delta = _run_rks(
            'b3lyp', knob='headtail')
        self.assertTrue(mf.converged)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['population'], 'hybrid')
        self.assertTrue(entries[0]['armed'])
        self.assertFalse(entries[0]['refused_pure'])
        self.assertEqual(ref_delta, 0)
        self.assertGreater(head_delta, 0)  # the lane really engaged

class StrictAndLeak(unittest.TestCase):
    '''Strict-reference inertness of the recorded population truth and
    the no-leak knob-off re-run.'''

    def setUp(self):
        self.state = GuardState()

    def tearDown(self):
        self.state.restore()

    def test_strict_reference_inertness(self):
        # a hybrid cell with the strict-fp64-reference marker (the
        # deployed gate mechanism: the mf attribute the jk lane's own
        # conjunct reads -- the numint strict_fp64_reference_ctx gates
        # only the numint-internal paths by design, unchanged here):
        # armed False (the strict conjunct), the population truth still
        # recorded, and the lane never set
        n_stats = len(df_jk._SCF_JK_F32_GUARD_STATS)
        os.environ['GPU4PYSCF_SCF_JK_F32'] = 'headtail'
        df_jk._SCF_JK_F32 = None
        try:
            mf = dft.RKS(mol).density_fit()
            mf.xc = 'b3lyp'
            mf.conv_tol = 1e-10
            mf.grids.level = 2
            mf._strict_fp64_reference = True
            mf.kernel()
        finally:
            os.environ.pop('GPU4PYSCF_SCF_JK_F32', None)
            df_jk._SCF_JK_F32 = None
        self.assertTrue(mf.converged)
        entry = df_jk._SCF_JK_F32_GUARD_STATS[n_stats:][0]
        self.assertEqual(entry['population'], 'hybrid')
        self.assertFalse(entry['armed'])
        self.assertFalse(entry['refused_pure'])

    def test_no_leak_rerun(self):
        _run_rks('r2scan', knob='headtail')  # one refusal recorded
        n_stats = len(df_jk._SCF_JK_F32_GUARD_STATS)
        n_ref = df_jk._SCF_JK_F32_GUARD_REFUSALS
        mf, entries, head_delta, ref_delta = _run_rks(
            'r2scan', knob=None)
        self.assertTrue(mf.converged)
        self.assertEqual(entries, [])
        self.assertEqual(ref_delta, 0)
        self.assertEqual(head_delta, 0)
        self.assertEqual(len(df_jk._SCF_JK_F32_GUARD_STATS), n_stats)
        self.assertEqual(df_jk._SCF_JK_F32_GUARD_REFUSALS, n_ref)
        self.assertFalse(df_jk._scf_jk_f32_lane)

class SubprocessArms(unittest.TestCase):
    '''Whole-SCF arms, one fresh subprocess per knob setting at the
    pure-functional cell (the standing float-hex transport; the refusal
    path is the knob-off code path, so the honest band is the
    documented 1-2 ULP cross-process class = 1e-10 Ha with equal
    cycles); the fail-closed refusal evidence is on-arm only.'''

    def _run_arm(self, knob):
        repo = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(df_jk.__file__))))
        script = _ARM.format(knob=knob, water=WATER, repo=repo)
        with tempfile.NamedTemporaryFile(
                'w', suffix='.py', delete=False) as f:
            f.write(script)
            path = f.name
        try:
            out = subprocess.check_output(
                [sys.executable, path], timeout=1200, cwd=repo)
        finally:
            os.unlink(path)
        line = [l for l in out.decode().splitlines()
                if l.startswith('E2EJSON')][-1]
        parts = line.split()
        return {'e': float.fromhex(parts[1]), 'converged': parts[2] == 'True',
                'cycles': int(parts[3]), 'refusals': int(parts[4]),
                'n_stats': int(parts[5]), 'head_calls': int(parts[6]),
                'population': parts[7] if len(parts) > 7 else None,
                'armed': parts[8] == 'True' if len(parts) > 8 else None,
                'refused': parts[9] == 'True' if len(parts) > 9 else None}

    def test_subprocess_pure_refusal(self):
        off = self._run_arm('')
        on = self._run_arm('headtail')
        self.assertTrue(off['converged'] and on['converged'])
        # the refusal path is the knob-off path: the honest
        # cross-process envelope + equal cycles
        self.assertLessEqual(abs(on['e'] - off['e']), 1e-10)
        self.assertEqual(on['cycles'], off['cycles'])
        self.assertEqual(on['refusals'], 1)
        self.assertEqual(on['n_stats'], 1)
        self.assertEqual(on['population'], 'pure')
        self.assertFalse(on['armed'])
        self.assertTrue(on['refused'])
        self.assertEqual(on['head_calls'], 0)
        self.assertEqual(off['refusals'], 0)
        self.assertEqual(off['n_stats'], 0)
        self.assertEqual(off['head_calls'], 0)

if __name__ == '__main__':
    print('full test suite would be launched with pytest')

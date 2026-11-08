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
'''SCFJK-KACC-SYRK-IMPL-01 focused equivalence regression (2026-11-06).

The opt-in GPU4PYSCF_SCF_JK_KACC_SYRK knob (DEFAULT OFF) replaces the
per-block dense k_accum contraction of df_jk.get_jk
contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk) with a cuBLAS
dsyrk tril beta-accumulation plus one mirror fill on the fp64
single-DM unsymmetrized lane (BENCH-02 basis, entry 377: component
1.3351x/1.5300x at 819/1042, numerics 6.1e-15/5.5e-15 reassociation,
engine control neutral).  These tests precede any GPU measurement:
parse mapping, knob-off bitwise determinism, knob-on banded numerics
with exact mirror symmetry, engagement-counter semantics, strict
fp64-reference inertness, global-fp32-mode inertness, multi-DM
fallback, the UHF two-spin lane, and whole-SCF subprocess arms.
'''
import json
import os
import subprocess
import sys
import unittest
import numpy as np
import pyscf
import cupy as cp
from gpu4pyscf import scf
from gpu4pyscf import dft
from gpu4pyscf.df import df_jk
from gpu4pyscf.dft import numint
from gpu4pyscf.lib import precision


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

def _bitwise_equal(a, b):
    a, b = cp.asarray(a), cp.asarray(b)
    return (a.dtype == b.dtype and a.shape == b.shape
            and a.tobytes() == b.tobytes())

def _df_hf_dm(mol):
    '''Converged DF-HF; returns (dfobj, mo-tagged dm) so direct get_jk
    arms reproduce the per-cycle call shape (mo-marked dm,
    dm_factor_mode 0).'''
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.with_df, cp.asarray(mf.make_rdm1())

def _df_uhf_dm(mol):
    mf = scf.UHF(mol).density_fit()
    mf.conv_tol = 1e-10
    mf.kernel()
    return mf.with_df, cp.asarray(mf.make_rdm1())


class ScfJkKaccSyrkParse(unittest.TestCase):
    def setUp(self):
        self._saved = df_jk._SCF_JK_KACC_SYRK
        self._had_env = 'GPU4PYSCF_SCF_JK_KACC_SYRK' in os.environ
        self._env = os.environ.get('GPU4PYSCF_SCF_JK_KACC_SYRK')

    def tearDown(self):
        df_jk._SCF_JK_KACC_SYRK = self._saved
        if self._had_env:
            os.environ['GPU4PYSCF_SCF_JK_KACC_SYRK'] = self._env
        else:
            os.environ.pop('GPU4PYSCF_SCF_JK_KACC_SYRK', None)

    def _parse(self, raw):
        if raw is None:
            os.environ.pop('GPU4PYSCF_SCF_JK_KACC_SYRK', None)
        else:
            os.environ['GPU4PYSCF_SCF_JK_KACC_SYRK'] = raw
        df_jk._SCF_JK_KACC_SYRK = None
        return df_jk._scf_jk_kacc_syrk()

    def test_parse_mapping(self):
        self.assertIs(self._parse(None), False)
        self.assertIs(self._parse(''), False)
        self.assertIs(self._parse('0'), False)
        self.assertIs(self._parse('1'), True)
        with self.assertWarns(UserWarning):
            self.assertIs(self._parse('bogus'), False)


class ScfJkKaccSyrkDirect(unittest.TestCase):
    '''Direct df_jk.get_jk arms at fixed dm: knob-off determinism,
    knob-on numerics bands and mirror symmetry, counter semantics,
    inertness classes.'''

    def setUp(self):
        self._saved_knob = df_jk._SCF_JK_KACC_SYRK
        self._saved_calls = df_jk._SCF_JK_KACC_SYRK_CALLS
        self.dfobj, self.dm = _df_hf_dm(mol_sph)
        df_jk._SCF_JK_KACC_SYRK_CALLS = 0

    def tearDown(self):
        df_jk._SCF_JK_KACC_SYRK = self._saved_knob
        df_jk._SCF_JK_KACC_SYRK_CALLS = self._saved_calls

    def _knob(self, on):
        os.environ['GPU4PYSCF_SCF_JK_KACC_SYRK'] = '1' if on else '0'
        df_jk._SCF_JK_KACC_SYRK = None

    def test_off_bitwise_deterministic(self):
        self._knob(False)
        vj1, vk1 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        vj2, vk2 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        self.assertTrue(_bitwise_equal(vj1, vj2))
        self.assertTrue(_bitwise_equal(vk1, vk2))
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, 0)

    def test_on_banded_numerics_and_counter(self):
        self._knob(False)
        vj0, vk0 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        n_off = df_jk._SCF_JK_KACC_SYRK_CALLS
        self._knob(True)
        vj1, vk1 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        # J side untouched: bitwise identical
        self.assertTrue(_bitwise_equal(vj0, vj1))
        # K side: fp64 reassociation band only
        dv = cp.abs(cp.asarray(vk1) - cp.asarray(vk0)).max()
        scale = cp.abs(cp.asarray(vk0)).max()
        self.assertLess(float(dv / scale), 1e-10)
        # the mirrored triangle makes the on-knob K exactly symmetric
        k1 = cp.asarray(vk1)
        self.assertTrue(_bitwise_equal(k1, k1.T))
        # one armed engagement per get_jk call
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS - n_off, 1)
        # determinism of the armed path itself
        vj2, vk2 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        self.assertTrue(_bitwise_equal(vk1, vk2))
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS - n_off, 2)

    def test_strict_reference_inert(self):
        self._knob(True)
        with numint.strict_fp64_reference_ctx():
            vj1, vk1 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
            self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, 0)
        self._knob(False)
        vj0, vk0 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
        self.assertTrue(_bitwise_equal(vj0, vj1))
        self.assertTrue(_bitwise_equal(vk0, vk1))

    def test_fp32_mode_inert(self):
        # the global fp32 precision mode keeps the deployed fp32
        # branch; the syrk knob never engages on fp32 operands
        self._knob(True)
        saved = precision.get_precision()
        try:
            precision.set_precision('fp32')
            _, vk1 = self.dfobj.get_jk(self.dm, hermi=1, with_k=True)
            self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, 0)
        finally:
            precision.set_precision(saved)

    def test_multi_dm_fallback(self):
        self._knob(True)
        dm2 = cp.asarray(np.stack([self.dm.get(), self.dm.get()]))
        vj2, vk2 = self.dfobj.get_jk(dm2, hermi=1, with_k=True)
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, 0)
        self._knob(False)
        vj0, vk0 = self.dfobj.get_jk(dm2, hermi=1, with_k=True)
        self.assertTrue(_bitwise_equal(vj0, vj2))
        self.assertTrue(_bitwise_equal(vk0, vk2))

    def test_uhf_dm_falls_back_dense(self):
        # A plain UHF dm (2,nao,nao) reaches df_jk as n_dm == 2 (each
        # spin dm is an independent density; the nspin==2 lane is the
        # 4-d tagged TDDFT/hessian form).  The preregistered Phase-A
        # gate is n_dm == 1 (the RKS lane the BENCH-02 cells measure),
        # so UHF keeps the deployed dense contract: knob-on output
        # bitwise-equal knob-off, counter exactly 0.  A per-dm syrk
        # extension is a separate, separately-benched decision.
        dfu, dmu = _df_uhf_dm(mol_sph)
        self._knob(False)
        vj0, vk0 = dfu.get_jk(dmu, hermi=1, with_k=True)
        self._knob(True)
        vj1, vk1 = dfu.get_jk(dmu, hermi=1, with_k=True)
        self.assertEqual(df_jk._SCF_JK_KACC_SYRK_CALLS, 0)
        self.assertTrue(_bitwise_equal(vj0, vj1))
        self.assertTrue(_bitwise_equal(vk0, vk1))


class ScfJkKaccSyrkE2E(unittest.TestCase):
    '''Whole-SCF subprocess arms: knob-on energies inside the
    reassociation band with equal cycle counts (fresh processes, so
    the module cache cannot leak between arms).'''

    def _scf(self, extra_env):
        env = dict(os.environ)
        env.pop('GPU4PYSCF_SCF_JK_KACC_SYRK', None)
        env.update(extra_env)
        code = (
            'import json, pyscf\n'
            'from gpu4pyscf import scf\n'
            'mol = pyscf.M(atom="""\n'
            'O 0. 0. 0.1174\n'
            'H 0. -0.757 -0.4696\n'
            'H 0. 0.757 -0.4696""", basis="def2-svp",\n'
            '           output="/dev/null", verbose=0)\n'
            'mf = scf.RHF(mol).density_fit()\n'
            'mf.conv_tol = 1e-10\n'
            'data = {}\n'
            'mf.callback = lambda env: data.__setitem__(\n'
            '    "cycles", data.get("cycles", 0) + 1)\n'
            'e = mf.kernel()\n'
            'data["e_tot"] = float(e)\n'
            'data["converged"] = bool(mf.converged)\n'
            'print("JSON:" + json.dumps(data))\n')
        r = subprocess.run([sys.executable, '-c', code], env=env,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        for line in r.stdout.splitlines():
            if line.startswith('JSON:'):
                return json.loads(line[5:])
        raise AssertionError('no JSON line in stdout: ' + r.stdout[-500:])

    def test_e2e_band_and_cycles(self):
        off = self._scf({'GPU4PYSCF_SCF_JK_KACC_SYRK': '0'})
        on = self._scf({'GPU4PYSCF_SCF_JK_KACC_SYRK': '1'})
        self.assertTrue(off['converged'] and on['converged'])
        self.assertEqual(off['cycles'], on['cycles'])
        self.assertLess(abs(on['e_tot'] - off['e_tot']), 1e-10)


if __name__ == '__main__':
    unittest.main()

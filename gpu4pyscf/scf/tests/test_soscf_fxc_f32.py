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

"""Regression tests for the CONV-UKS-NEWTON-FXCF32-01 opt-in fxc
two-site float32 cast discipline (GPU4PYSCF_SOSCF_FXC_F32,
gpu4pyscf/dft/numint.py).

Provenance: entry 422 (RESPF32 NEGATIVE_CLOSE PARTIAL_CUT) measured the
newton vind residual at the 573-AO ibuprofen_anion wb97m-v doublet:
after the deployed df_jk K-transform cast cut its reachable share
(3.213 -> 2.333 s/call, e_dev +5.68e-12, counts invariant), the residual
2.33 s/call x 29 calls = 67.6 s of the 114.5 s t_newton is the nr_uks_fxc
grid contraction with NO deployed cast sites.  Entry 424's family
instrument self-perturbed (+0.27 s/call CUDA-event overhead, INVALID-
UNFUNDED, never adjudicated) but its non-gating diagnostic splits the
overhead-corrected residual A rho1-build ~0.8 / C vmat-accumulation ~1.0
with B negligible.  Staged by RERANK-29 (entry 425): the honest design
for a near-split is a TWO-SITE cast.  The knob casts, inside
_nr_uks_fxc_task ONLY: family A (the slow-dms-branch eval_rho
contractions: fp32 masked dm x fp32 per-block AO stack, fp64 rho1
accumulation -- _contract_rho's f32 kernel writes the float64 rho) and
family C (the MGGA/GGA/LDA vmat-accumulation chains: fp32
_tau_dot/_scale_ao/contract with an explicit cast-up before add_sparse,
whose raw-pointer kernel has no dtype dispatch).  fxc_w and the .5
scalings stay exact fp64.  The occ_coeff branch (eval_rho4) and the RKS
twin are off the vind path and untouched; the veff tasks never see the
knob.  Strict fp64-reference contexts never arm (the numint module-flag
guard, the _tau_dot_batched idiom).

Contract tests (source-level, CPU):
  * exactly one env reader for the knob (home module numint);
  * '1' arms; unset/0 verbatim; malformed warn + ignore;
  * the counter counts task entries under the knob only, at task entry
    (outside the block loop);
  * the dms-branch armed path drops the caller buffer (sized for f64
    views) and casts the masked dm at the eval_rho boundary; the
    default path keeps the verbatim buf=buf calls;
  * the accumulation armed path casts up explicitly before every
    add_sparse; the default path keeps the verbatim out=vtmp calls;
  * the occ_coeff branch and the RKS twin reference no cast site;
  * strict fp64-reference guard precedes the env cache read.

Numerical tests (GPU, small UKS sentinels, the entry-416 lane class):
  * MGGA (wb97m-v, the funded branch) and GGA (b3lyp) sentinels;
  * default OFF: run-to-run bitwise e_tot, zero engagement counter;
  * armed: same basin (|dE| <= 1e-8 Ha), engagement counter == vind
    call count, converged under cap;
  * malformed value: verbatim path (zero counter, bitwise);
  * strict fp64-reference context: never arms (zero counter, bitwise
    verbatim).

Dtype-shim evidence (GPU): inside the armed newton every eval_rho call
from the fxc task sees float32 dm operands and every add_sparse scatter
sees a float64 operand (the explicit cast-up); with the knob off all
dtypes stay float64.
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

NUMINT_SRC = os.path.join(REPO, 'gpu4pyscf', 'dft', 'numint.py')
KNOB = 'GPU4PYSCF_SOSCF_FXC_F32'


def _func_src(path, name):
    src = open(path).read()
    m = re.search(rf'def {name}\(.*?(?=\ndef |\Z)', src, re.S)
    return m.group(0) if m else None


class TestSourceContract(unittest.TestCase):
    def test_home_parser_contract(self):
        parser = _func_src(NUMINT_SRC, '_fxc_f32_enabled')
        self.assertIsNotNone(parser)
        self.assertIn("os.environ.get('%s'" % KNOB, parser)
        # exactly one env reader per knob (home module)
        src = open(NUMINT_SRC).read()
        self.assertEqual(src.count("os.environ.get('%s'" % KNOB), 1)
        # accepted values + warn-and-ignore
        self.assertIn("raw == '1'", parser)
        self.assertIn("raw in ('', '0')", parser)
        self.assertIn('warnings.warn', parser)
        # the strict fp64-reference guard PRECEDES the env cache read
        # (the _tau_dot_batched idiom)
        self.assertLess(parser.index('_strict_fp64_reference_active'),
                        parser.index('global _FXC_F32'))

    def test_counter_at_task_entry(self):
        task = _func_src(NUMINT_SRC, '_nr_uks_fxc_task')
        self.assertIsNotNone(task)
        self.assertIn('global _fxc_f32_tasks', task)
        self.assertIn('fxc_f32 = _fxc_f32_enabled()', task)
        self.assertIn('_fxc_f32_tasks += 1', task)
        # the counter bump sits BEFORE the block loop (task-entry class,
        # not per-block)
        self.assertLess(task.index('_fxc_f32_tasks += 1'),
                        task.index('ni.block_loop'))

    def test_dms_branch_cast_site(self):
        task = _func_src(NUMINT_SRC, '_nr_uks_fxc_task')
        # family A armed: f32 dm at the eval_rho boundary, buffer dropped
        self.assertIn('dma_mask.astype(cupy.float32)', task)
        self.assertIn('dmb_mask.astype(cupy.float32)', task)
        self.assertIn('eval_rho(_sorted_mol, ao_c,', task)
        # the default path keeps the verbatim buf=buf calls
        self.assertIn('xctype=xctype, hermi=hermi, buf=buf)', task)
        # the per-block AO twin is made exactly once per block
        self.assertEqual(task.count('ao.astype(cupy.float32)'), 1)

    def test_accumulation_cast_up_contract(self):
        task = _func_src(NUMINT_SRC, '_nr_uks_fxc_task')
        # every armed add_sparse scatter casts up explicitly (the
        # raw-pointer kernel has no dtype dispatch)
        armed_adds = [m.start() for m in re.finditer(
            r'add_sparse\([^\n]*?\.astype\(cupy\.float64\)', task)]
        self.assertEqual(len(armed_adds), 6,
                         '6 armed scatters (LDA/GGA/MGGA x alpha/beta)')
        # the armed tau train and main GEMM run on the f32 stack
        self.assertIn('_tau_dot(ao_c, ao_c,', task)
        self.assertIn("'ig, jg->ij', ao_c[0], aow_a", task)
        # the default path keeps the verbatim out=vtmp accumulation
        self.assertIn('beta=1, out=vtmp)', task)

    def test_occ_coeff_branch_and_rks_twin_untouched(self):
        task = _func_src(NUMINT_SRC, '_nr_uks_fxc_task')
        # the occ_coeff branch (eval_rho4) is off the vind path and
        # carries no cast site; anchor on the eval_rho4 body inside the
        # block loop (the early `if occ_coeff is not None:` unpacking
        # site precedes the shared block-top ao_c definition, which
        # family C's accumulation legitimately uses in both branches)
        m = re.search(r'rho1a = eval_rho4\(.*?else: # slow version',
                      task, re.S)
        self.assertIsNotNone(m)
        self.assertNotIn('ao_c', m.group(0))
        self.assertNotIn('float32', m.group(0))
        # the RKS twin never references the knob scope
        rks = _func_src(NUMINT_SRC, '_nr_rks_fxc_task')
        self.assertIsNotNone(rks)
        self.assertNotIn('fxc_f32', rks)
        self.assertNotIn('ao_c', rks)
        # the veff tasks are not touched by this knob either
        for veff in ('_nr_rks_task', '_nr_uks_task'):
            src = _func_src(NUMINT_SRC, veff)
            self.assertNotIn('_fxc_f32_enabled', src)

    def test_knob_module_state(self):
        src = open(NUMINT_SRC).read()
        self.assertIn('_FXC_F32 = None', src)
        self.assertIn('_fxc_f32_tasks = 0', src)


class TestParserValues(unittest.TestCase):
    def setUp(self):
        from gpu4pyscf.dft import numint
        self.numint = numint
        self._saved = (os.environ.get(KNOB), numint._FXC_F32)

    def tearDown(self):
        raw, cache = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.numint._FXC_F32 = cache

    def _parse(self, raw):
        self.numint._FXC_F32 = None
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        return self.numint._fxc_f32_enabled()

    def test_values(self):
        self.assertFalse(self._parse(None))
        self.assertFalse(self._parse(''))
        self.assertFalse(self._parse('0'))
        self.assertTrue(self._parse('1'))
        for bad in ('on', 'true', '2', 'nan', 'headtail'):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                val = self._parse(bad)
            self.assertFalse(val, 'value %r must be ignored' % bad)
            self.assertTrue(any(KNOB in str(x.message) for x in w),
                            'value %r must warn' % bad)

    def test_strict_ref_ctx_forces_off(self):
        self._parse('1')
        with self.numint.strict_fp64_reference_ctx():
            self.assertFalse(self.numint._fxc_f32_enabled(),
                             'strict fp64-reference contexts never arm')
        self.assertTrue(self.numint._fxc_f32_enabled())
        self._parse(None)


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestNumerical(unittest.TestCase):
    '''Small DF-UKS sentinels (the entry-416 lane class): MGGA wb97m-v
    (the funded tau branch) and GGA b3lyp.  Head = 2 CDIIS cycles, then
    the public newton() under the unit knob value.'''

    def _run(self, raw, xc='wb97m-v', strict_ref_ctx=False):
        """Returns (e_tot_repr, counter_delta, vind_calls, converged)."""
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        numint._FXC_F32 = None       # force re-read
        counter0 = numint._fxc_f32_tasks
        mol = gto.M(verbose=0, output='/dev/null',
                    atom=[['O', (0., 0., 0.)],
                          ['O', (0., 0., 1.2078)]],
                    basis='def2-svp', charge=0, spin=2)
        try:
            mf = dft.UKS(mol, xc=xc).density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.max_cycle = 2
            mf.conv_check = False
            mf.kernel()
            mfn = mf.newton()
            mfn.max_cycle = 20
            mfn.conv_tol_grad = 1e-5
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
                if strict_ref_ctx:
                    with numint.strict_fp64_reference_ctx():
                        e = mfn.kernel()
                else:
                    e = mfn.kernel()
            finally:
                del mfn.gen_response
            return (repr(float(e)),
                    numint._fxc_f32_tasks - counter0,
                    vind_calls[0],
                    bool(getattr(mfn, 'converged', False)))
        finally:
            mol.stdout.close()

    def setUp(self):
        from gpu4pyscf.dft import numint
        self.numint = numint
        self._saved = (os.environ.get(KNOB), numint._FXC_F32,
                       numint._fxc_f32_tasks)

    def tearDown(self):
        raw, cache, calls = self._saved
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        self.numint._FXC_F32 = cache
        self.numint._fxc_f32_tasks = calls

    def test_mgga_default_off_bitwise_and_armed(self):
        # (1) default OFF: bitwise run-to-run, zero engagements
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
        self.assertGreaterEqual(d3, 1, 'armed knob must engage')
        self.assertEqual(d3, v3,
                         'engagement counter must equal the vind call '
                         'count (one fxc task entry per vind call)')
        self.assertLessEqual(abs(float(e3) - float(e1)), 1e-8,
                             'armed newton must stay in the same basin')
        # (3) malformed value: verbatim path, bitwise with (1)
        e4, d4, v4, c4 = self._run('not-a-flag')
        self.assertTrue(c4)
        self.assertEqual(d4, 0, 'malformed value must be ignored')
        self.assertEqual(e4, e1, 'malformed value must be bitwise '
                         'verbatim')
        # (4) armed inside the strict fp64-reference context: never
        # arms, bitwise verbatim
        e5, d5, v5, c5 = self._run('1', strict_ref_ctx=True)
        self.assertTrue(c5)
        self.assertEqual(d5, 0,
                         'strict fp64-reference contexts never arm')
        self.assertEqual(e5, e1, 'strict-ref armed run must be '
                         'bitwise verbatim')

    def test_gga_armed_same_basin(self):
        # the GGA accumulation branch cast (a second branch of the
        # two-site design) with the same contract
        e0, d0, v0, c0 = self._run(None, xc='b3lyp')
        self.assertTrue(c0)
        self.assertEqual(d0, 0)
        e1, d1, v1, c1 = self._run('1', xc='b3lyp')
        self.assertTrue(c1, 'armed GGA newton must converge under cap')
        self.assertEqual(d1, v1, 'engagement counter == vind calls')
        self.assertGreaterEqual(d1, 1)
        self.assertLessEqual(abs(float(e1) - float(e0)), 1e-8,
                             'armed GGA newton must stay in the basin')


@unittest.skipIf(not HAS_GPU, 'needs GPU')
class TestDtypeShim(unittest.TestCase):
    '''dtype-shim evidence on the armed MGGA sentinel: inside the armed
    newton every eval_rho reaching the fxc task sees a float32 dm
    operand (family A engaged) and every add_sparse scatter sees a
    float64 operand (the explicit cast-up before the raw-pointer
    kernel); with the knob off every dtype stays float64.'''

    def _armed_newton(self, raw):
        from pyscf import gto
        from gpu4pyscf import dft
        from gpu4pyscf.dft import numint
        if raw is None:
            os.environ.pop(KNOB, None)
        else:
            os.environ[KNOB] = raw
        numint._FXC_F32 = None
        eval_rho_entries = []    # (dm dtype, ao dtype)
        add_entries = []         # operand dtype
        orig_eval_rho = numint.eval_rho
        orig_add = numint.add_sparse
        inside = [0]

        def eval_rho_shim(mol, ao, dm, *a, **kw):
            eval_rho_entries.append(
                (str(getattr(dm, 'dtype', None)),
                 str(getattr(ao[0] if isinstance(ao, (list, tuple))
                             else ao, 'dtype', None))))
            return orig_eval_rho(mol, ao, dm, *a, **kw)

        def add_shim(a, b, idx):
            add_entries.append(str(b.dtype))
            return orig_add(a, b, idx)

        mol = gto.M(verbose=0, output='/dev/null',
                    atom=[['O', (0., 0., 0.)],
                          ['O', (0., 0., 1.2078)]],
                    basis='def2-svp', charge=0, spin=2)
        try:
            mf = dft.UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.grids.level = 3
            mf.max_cycle = 2
            mf.conv_check = False
            mf.kernel()      # head: shims NOT yet installed
            mfn = mf.newton()
            mfn.max_cycle = 20
            mfn.conv_tol_grad = 1e-5
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
            numint.eval_rho = eval_rho_shim
            numint.add_sparse = add_shim
            try:
                counter0 = numint._fxc_f32_tasks
                mfn.kernel()
                counter_delta = numint._fxc_f32_tasks - counter0
            finally:
                numint.eval_rho = orig_eval_rho
                numint.add_sparse = orig_add
                del mfn.gen_response
            mol.stdout.close()
            return eval_rho_entries, add_entries, counter_delta
        finally:
            mol.stdout.close()

    def test_dtypes_in_armed_scope(self):
        evals, adds, delta = self._armed_newton('1')
        self.assertGreater(delta, 0, 'armed run must engage')
        self.assertTrue(evals, 'eval_rho must fire inside the fxc task')
        self.assertTrue(all(e[0] == 'float32' for e in evals),
                        'family A: the fxc-task dm operand must be '
                        'float32 in the armed run; saw %s'
                        % sorted(set(e[0] for e in evals)))
        self.assertTrue(all(e[1] == 'float32' for e in evals),
                        'family A: the fxc-task ao operand must be '
                        'float32 in the armed run; saw %s'
                        % sorted(set(e[1] for e in evals)))
        self.assertTrue(adds, 'add_sparse must fire in the fxc task')
        self.assertTrue(all(a == 'float64' for a in adds),
                        'the cast-up before add_sparse must hold; saw '
                        '%s' % sorted(set(adds)))
        # knob off: every dtype stays float64
        evals0, adds0, delta0 = self._armed_newton(None)
        self.assertEqual(delta0, 0)
        self.assertTrue(evals0)
        self.assertTrue(all(e[0] == 'float64' and e[1] == 'float64'
                            for e in evals0))
        self.assertTrue(all(a == 'float64' for a in adds0))


if __name__ == '__main__':
    print('Regression tests for the SOSCF fxc two-site f32 knob')
    unittest.main()

#!/usr/bin/env python
# Focused contracts for DF-GRAD-TASKGRAN-967-01: the (c) upstream #967
# uniform splits_by_blocksize re-split + multi-group launch count,
# armed at GPU4PYSCF_DF_TASKGRAN level 2 (level 1 = the entry-519 #970
# mechanisms (a)/(b), unchanged).  Scoped port per the card: the
# df/grad/rhf.py ip1 sum-kernel loop (site R) and the
# _int3c2e_ip1_evaluator fill machinery (site B2, self-consistent:
# evaluate_j3c carries its own packed group count).  The
# Int3c2eOpt.int3c2e_evaluator aux packing (upstream's bdiv:216 hunk) is
# a DISCLOSED NON-PORT: it changes aux_offsets semantics for every
# aux_batch_size caller (uhf/tdrhf/hessian/mp/pbc) whose consumer loops
# this card does not port, and it has zero measured reach at the scoped
# D1-base cells (hybrids run the j3c-reuse phase 1, bypassing the
# evaluator's j3c stream).  grad/uhf.py was likewise a disclosed
# non-carrier of this card; the TASKGRAN-UKS-PORT-01 successor armed it
# (armed-state contract below).
import os
import unittest
import warnings
import numpy as np
import pyscf

from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.df.grad import rhf as df_rhf_grad

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''

HE17 = '\n'.join(
    'He    %10.6f    %10.6f    %10.6f' % (2.5*i, 2.5*(i % 4), 2.5*(i // 4))
    for i in range(17))

_BDIV_SRC = os.path.join(os.path.dirname(bdiv.__file__), 'int3c2e_bdiv.py')
_RHF_SRC = os.path.join(os.path.dirname(df_rhf_grad.__file__), 'rhf.py')
_UHF_SRC = os.path.join(os.path.dirname(df_rhf_grad.__file__), 'uhf.py')


def _reset_knob_cache():
    bdiv._TASKGRAN = None
    df_rhf_grad._IP1_FUSED = None


def _resplit_counter():
    return getattr(bdiv, '_TASKGRAN_RESPLITS', 0)


def _paircap_counter():
    return getattr(bdiv, '_TASKGRAN_PAIRCAP_SETS', 0)


def _aux16_counter():
    return getattr(bdiv, '_TASKGRAN_AUX16_SPLITS', 0)


def _helpers_present():
    return (callable(getattr(bdiv, '_taskgran', None))
            and callable(getattr(bdiv, '_taskgran_resplit_active', None))
            and callable(getattr(bdiv, '_taskgran_resplit_splits', None))
            and callable(getattr(bdiv, '_taskgran_pair_batch', None))
            and callable(getattr(bdiv, '_taskgran_aux16_offsets', None)))


def _run_rks(atom, basis, xc, env_knob, grids_level=1, fused=None):
    '''One fresh DF SCF + analytic gradient with the knob env set BEFORE
    the knob cache resolves.  Returns (e_tot, grad, converged).'''
    os.environ['GPU4PYSCF_DF_TASKGRAN'] = env_knob
    if fused is not None:
        os.environ['GPU4PYSCF_DF_IP1_FUSED'] = fused
    else:
        os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
    _reset_knob_cache()
    mol = pyscf.M(atom=atom, basis=basis, output='/dev/null', verbose=0)
    from gpu4pyscf.dft import rks
    mf = rks.RKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.grids.level = grids_level
    mf.max_cycle = 100
    mf.conv_tol = 1e-11
    e_tot = mf.kernel()
    grad = np.asarray(mf.nuc_grad_method().kernel())
    return e_tot, grad, mf.converged


class TestTaskgran967Knob(unittest.TestCase):
    '''CPU-only contracts: parse levels, helper values, engagement
    counter, VRAM bound, strict-reference inertness, source structure.'''

    def test_helpers_and_counters_exist(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        self.assertIsInstance(_resplit_counter(), int)

    def test_knob_parse_levels_default_off_malformed(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        for raw, level in (('', 0), ('0', 0), ('1', 1), ('2', 2)):
            _reset_knob_cache()
            if raw:
                os.environ['GPU4PYSCF_DF_TASKGRAN'] = raw
            else:
                os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
            self.assertEqual(bdiv._taskgran(), level, raw)
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = 'yes'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.assertEqual(bdiv._taskgran(), 0)
        self.assertTrue(any('ignored' in str(x.message) for x in w))
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_resplit_values_and_engagement_counter(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        aux_offsets = np.array([0, 10, 20, 35, 37])
        # verbatim levels: the one-group-per-launch unit list
        for raw in ('', '0', '1'):
            _reset_knob_cache()
            if raw:
                os.environ['GPU4PYSCF_DF_TASKGRAN'] = raw
            else:
                os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
            n0 = _resplit_counter()
            self.assertEqual(bdiv._taskgran_resplit_splits(aux_offsets, 25),
                             [0, 1, 2, 3, 4])
            self.assertEqual(_resplit_counter(), n0)
        # armed level 2: the upstream #967 uniform packing
        from gpu4pyscf.lib.utils import splits_by_blocksize
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '2'
        n0 = _resplit_counter()
        splits = bdiv._taskgran_resplit_splits(aux_offsets, 25)
        self.assertEqual(splits, splits_by_blocksize(aux_offsets, 25))
        # 4 verbatim units -> 2 packed units: a real geometry change
        self.assertEqual(len(splits) - 1, 2)
        self.assertEqual(_resplit_counter(), n0 + 1)
        # already-uniform geometry: no counter, identical packing
        uni = np.array([0, 25, 50])
        n1 = _resplit_counter()
        self.assertEqual(bdiv._taskgran_resplit_splits(uni, 25),
                         splits_by_blocksize(uni, 25))
        self.assertEqual(len(bdiv._taskgran_resplit_splits(uni, 25)) - 1, 2)
        self.assertEqual(_resplit_counter(), n1)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_resplit_vram_bound_structural(self):
        '''Every armed packed unit spans at most batch_size functions when
        every input group does (the post-_split_l_ctr_pattern invariant at
        both armed sites): the verbatim one-group-per-launch buffer
        capacity bound cannot grow -- VRAM admission is preserved.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        _reset_knob_cache()
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '2'
        rng = np.random.default_rng(20261110)
        for trial in range(8):
            batch = int(rng.integers(4, 40))
            widths = rng.integers(1, batch + 1, size=int(rng.integers(2, 9)))
            cum = np.append(0, np.cumsum(widths))
            splits = bdiv._taskgran_resplit_splits(cum, batch)
            self.assertEqual(splits[0], 0)
            self.assertEqual(splits[-1], len(cum) - 1)
            for a, b_ in zip(splits[:-1], splits[1:]):
                self.assertGreater(b_, a)
                self.assertLessEqual(cum[b_] - cum[a], batch)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_resplit_strict_reference_inert(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        from gpu4pyscf.dft import numint
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '2'
        _reset_knob_cache()
        aux_offsets = np.array([0, 10, 20, 35, 37])
        n0 = _resplit_counter()
        with numint.strict_fp64_reference_ctx():
            self.assertFalse(bdiv._taskgran_resplit_active())
            self.assertEqual(bdiv._taskgran_resplit_splits(aux_offsets, 25),
                             [0, 1, 2, 3, 4])
        self.assertEqual(_resplit_counter(), n0)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_source_contracts(self):
        bdiv_src = open(_BDIV_SRC).read()
        rhf_src = open(_RHF_SRC).read()
        uhf_src = open(_UHF_SRC).read()
        # site B2 armed: the ip1 fill evaluator packs its aux batches and
        # passes the packed group count to its kernel
        self.assertIn('_taskgran_resplit_splits(aux_loc[ksh_offsets_cpu],'
                      ' batch_size)', bdiv_src)
        self.assertIn('ctypes.c_int(aux_split1 - aux_split0)', bdiv_src)
        # site B1 (Int3c2eOpt.int3c2e_evaluator) stays verbatim: disclosed
        # non-port, exactly one remaining one-group-per-launch site
        self.assertEqual(
            bdiv_src.count('aux_splits = range(len(ksh_offsets_cpu))'), 1)
        # site R armed: the evaluator-cap + blksize cap + the packed loop
        self.assertIn('_taskgran_resplit_active()', rhf_src)
        self.assertIn('batch_size = min(batch_size,'
                      ' int((aux_offsets[1:]-aux_offsets[:-1]).max()))',
                      rhf_src)
        self.assertIn('_taskgran_resplit_splits(aux_offsets, batch_size)',
                      rhf_src)
        self.assertIn('zip(aux_splits[:-1], aux_splits[1:])', rhf_src)
        self.assertIn('ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused'
                      ' else kend-kbatch)', rhf_src)
        # grad/uhf.py was this card's disclosed non-carrier of (c) and
        # is ARMED by the TASKGRAN-UKS-PORT-01 successor (the rhf twin
        # expressions, no local re-definitions)
        self.assertIn('_taskgran_resplit_splits(aux_offsets, batch_size)',
                      uhf_src)
        self.assertIn('ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused'
                      ' else kend-kbatch)', uhf_src)
        self.assertNotIn('def _taskgran', uhf_src)
        # the knob and its level semantics are documented in both home
        # modules
        for src in (bdiv_src, rhf_src):
            self.assertIn('GPU4PYSCF_DF_TASKGRAN', src)
            self.assertIn('#967', src)


class TestTaskgran967Equivalence(unittest.TestCase):
    '''GPU probe cells: armed equivalence bands, engagement counters,
    fused interplay, off-lane determinism, strict-reference and
    memory-constraint inertness classes.'''

    def test_armed_hybrid_grad_equivalence(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        e0, g0, c0 = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, ax0, rs0 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        e1, g1, c1 = _run_rks(WATER, 'def2-svp', 'b3lyp', '2')
        self.assertTrue(c1)
        pc1, ax1, rs1 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        d_e = abs(e1 - e0)
        d_g = np.max(np.abs(g1 - g0))
        print('TASKGRAN-967 b3lyp water: dE=%.3e dG=%.3e paircap+%d'
              ' aux16+%d resplit+%d'
              % (d_e, d_g, pc1 - pc0, ax1 - ax0, rs1 - rs0))
        # the hybrid ip1 loop is the (c) carrier: water aux groups pack
        # into fewer launches -> the resplit counter must move
        self.assertGreaterEqual(rs1 - rs0, 1)
        # level 2 keeps (a)/(b) armed
        self.assertGreaterEqual(pc1 - pc0, 1)
        self.assertEqual(ax1 - ax0, 0)
        self.assertLessEqual(d_e, 1e-10)
        self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_armed_pure_grad_resplit_noncarrier(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        e0, g0, c0 = _run_rks(HE17, 'def2-svp', 'pbe', '0')
        self.assertTrue(c0)
        pc0, ax0, rs0 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        e1, g1, c1 = _run_rks(HE17, 'def2-svp', 'pbe', '2')
        self.assertTrue(c1)
        pc1, ax1, rs1 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        d_e = abs(e1 - e0)
        d_g = np.max(np.abs(g1 - g0))
        print('TASKGRAN-967 pbe He17: dE=%.3e dG=%.3e paircap+%d aux16+%d'
              ' resplit+%d' % (d_e, d_g, pc1 - pc0, ax1 - ax0, rs1 - rs0))
        # the pure route returns through _j_energy_per_atom (full-width
        # kernel, no ip1 loop) and ipnuc's 17 charges stay within one
        # 32-function batch: (c) is a disclosed non-carrier here, (b)
        # must still engage at level 2
        self.assertEqual(rs1 - rs0, 0)
        self.assertGreaterEqual(ax1 - ax0, 1)
        self.assertGreaterEqual(pc1 - pc0, 1)
        self.assertLessEqual(d_e, 1e-10)
        self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_fused_skips_resplit(self):
        '''The LC-42 fused launch already covers all aux groups in one
        launch; the armed packing must be skipped (zero engagement) and
        the fused result stays in the armed-equivalence band.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        e0, g0, c0 = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        rs0 = _resplit_counter()
        e1, g1, c1 = _run_rks(WATER, 'def2-svp', 'b3lyp', '2', fused='1')
        self.assertTrue(c1)
        rs1 = _resplit_counter()
        print('TASKGRAN-967 fused water: dE=%.3e dG=%.3e resplit+%d'
              % (abs(e1 - e0), np.max(np.abs(g1 - g0)), rs1 - rs0))
        self.assertEqual(rs1 - rs0, 0)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        _reset_knob_cache()

    def test_offlane_determinism_zero_counters(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        e0, g0, c0 = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        rs0 = _resplit_counter()
        e1, g1, c1 = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c1)
        self.assertEqual(_resplit_counter(), rs0)
        # the ip1/j3c atomicAdd ULP run-to-run scatter class: banded,
        # not bitwise (band 1e-11 at water scale)
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_armed_grad_strict_reference_inert(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        from gpu4pyscf.dft import numint
        e0, g0, c0 = _run_rks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '2'
        _reset_knob_cache()
        rs0 = _resplit_counter()
        with numint.strict_fp64_reference_ctx():
            mol = pyscf.M(atom=WATER, basis='def2-svp', output='/dev/null',
                          verbose=0)
            from gpu4pyscf.dft import rks
            mf = rks.RKS(mol)
            mf.xc = 'b3lyp'
            mf = mf.density_fit()
            mf.grids.level = 1
            mf.max_cycle = 100
            mf.conv_tol = 1e-11
            e1 = mf.kernel()
            g1 = np.asarray(mf.nuc_grad_method().kernel())
        self.assertTrue(mf.converged)
        self.assertEqual(_resplit_counter(), rs0)
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_constrained_memory_stability(self):
        '''The #967 motivation class: memory-derived batch sizes must not
        change the gradient; armed packing stays within the same
        admission bound at two constrained capacities (VRAM admission
        preserved).'''
        self.assertTrue(_helpers_present(), 'TASKGRAN-967 helpers absent')
        results = {}
        for knob in ('0', '2'):
            for mem in (300000, 150000):
                os.environ['GPU4PYSCF_DF_TASKGRAN'] = knob
                _reset_knob_cache()
                with pyscf.lib.temporary_env(
                        bdiv, get_avail_mem=(lambda *a, _m=mem, **k: _m)):
                    with pyscf.lib.temporary_env(
                            df_rhf_grad,
                            get_avail_mem=(lambda *a, _m=mem, **k: _m)):
                        e, g, c = _run_rks(WATER, 'def2-svp', 'b3lyp', knob)
                self.assertTrue(c)
                results[(knob, mem)] = (e, g)
        for mem in (300000, 150000):
            d_e = abs(results[('2', mem)][0] - results[('0', mem)][0])
            d_g = np.max(np.abs(results[('2', mem)][1]
                                - results[('0', mem)][1]))
            print('TASKGRAN-967 constrained %d B: dE=%.3e dG=%.3e'
                  % (mem, d_e, d_g))
            self.assertLessEqual(d_e, 1e-10)
            self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()


if __name__ == '__main__':
    print('Full test suite: run via pytest from the repo root')
    unittest.main()

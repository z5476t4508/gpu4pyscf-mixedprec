#!/usr/bin/env python
# Focused contracts for TASKGRAN-UKS-PORT-01: porting the deployed
# GPU4PYSCF_DF_TASKGRAN mechanisms to the uhf-local ip1 sum-kernel loop
# (df/grad/uhf.py _jk_energy_per_atom) -- (a) the #970 pair cap at the
# aggregate_shl_pairs site (upstream #970 arms this site; the RKS twin
# is df/grad/rhf.py:577) and (c) the #967 uniform splits_by_blocksize
# re-split + multi-group launch count (the df/grad/rhf.py:644 twin; a
# fork extension -- the upstream #967 uhf-loop status was not
# re-verifiable on the network-blocked upstream-watch axis).
#
# SUPERSESSION note: the entry-519/entry-521 cards disclosed grad/uhf.py
# as a non-carrier and pinned the raw state with absence contracts
# (test_df_grad_taskgran01.py / test_df_grad_taskgran967_01.py).  This
# card supersedes exactly that scope; those contracts are updated in the
# same commit to pin the ARMED state instead.
#
# RED-at-parent classes (23cf854): every source-contract and engagement
# assertion below fails honestly (the uhf loop is raw at the parent, so
# no armed run can engage).  The value-equivalence/determinism
# comparisons are green-at-parent BY DESIGN (at the parent the knob
# cannot change the UKS loop, so armed == off trivially); they are the
# regression net for the port, disclosed per docstring.
import os
import unittest
import warnings
import numpy as np
import pyscf

from gpu4pyscf.df import int3c2e_bdiv as bdiv
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.df.grad import uhf as df_uhf_grad

WATER = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000    -0.0000000000    -0.4696000000
'''

HE17 = '\n'.join(
    'He    %10.6f    %10.6f    %10.6f' % (2.5*i, 2.5*(i % 4), 2.5*(i // 4))
    for i in range(17))

_UHF_SRC = os.path.join(os.path.dirname(df_uhf_grad.__file__), 'uhf.py')


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
    return (callable(getattr(bdiv, '_taskgran_pair_batch', None))
            and callable(getattr(bdiv, '_taskgran_resplit_splits', None)))


def _run_uks(atom, basis, xc, env_knob, charge=1, spin=1, grids_level=1,
             fused=None):
    '''One fresh DF UKS SCF + analytic gradient with the knob env set
    BEFORE the knob cache resolves.  Returns (e_tot, grad, converged).'''
    os.environ['GPU4PYSCF_DF_TASKGRAN'] = env_knob
    if fused is not None:
        os.environ['GPU4PYSCF_DF_IP1_FUSED'] = fused
    else:
        os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
    _reset_knob_cache()
    mol = pyscf.M(atom=atom, basis=basis, charge=charge, spin=spin,
                  output='/dev/null', verbose=0)
    from gpu4pyscf.dft import uks
    mf = uks.UKS(mol)
    mf.xc = xc
    mf = mf.density_fit()
    mf.grids.level = grids_level
    mf.max_cycle = 100
    mf.conv_tol = 1e-11
    e_tot = mf.kernel()
    grad = np.asarray(mf.nuc_grad_method().kernel())
    return e_tot, grad, mf.converged


class TestTaskgranUksPortContracts(unittest.TestCase):
    '''CPU-only source contracts: the uhf-local loop is armed through
    the home-module helpers (no-fork), the fall-through to the rhf
    J-only route is untouched, and the VRAM admission expression is
    unchanged.'''

    def test_source_contracts(self):
        src = open(_UHF_SRC).read()
        # (a) the #970 pair cap at the uhf aggregate_shl_pairs site
        self.assertIn('_taskgran_pair_batch(nsp_per_block[0], 4)', src)
        self.assertNotIn('bas_ij_cache, nsp_per_block[0]*4)', src)
        # (c) the re-split packing and the packed loop
        self.assertIn('aux_splits = _taskgran_resplit_splits('
                      'aux_offsets, batch_size)', src)
        self.assertIn('zip(aux_splits[:-1], aux_splits[1:])', src)
        self.assertIn('aux_offsets[kend] - aux_offsets[kbatch]', src)
        # the multi-group launch count reaches the kernel
        self.assertIn('ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused'
                      ' else kend-kbatch)', src)
        # no-fork: the helpers are imported from the home module and
        # never re-defined here
        self.assertIn('_taskgran_pair_batch,', src)
        self.assertIn('_taskgran_resplit_splits,', src)
        self.assertNotIn('def _taskgran', src)
        # the knob and this card are documented at the site
        self.assertIn('GPU4PYSCF_DF_TASKGRAN', src)
        self.assertIn('TASKGRAN-UKS-PORT-01', src)
        # the pure-functional fall-through to the armed rhf J-only
        # route is untouched (UKS pure functionals never reach the
        # uhf-local loop)
        self.assertIn('from gpu4pyscf.df.grad.rhf import '
                      '_j_energy_per_atom', src)
        # VRAM admission: the per-batch buffer expression is unchanged
        # (packed units span whole groups <= batch_size functions, the
        # shared-helper invariant)
        self.assertIn('buf = cp.empty((nao_pair*batch_size)', src)

    def test_fallthrough_route_unchanged(self):
        '''k_factor==0 UKS gradients route through rhf
        _j_energy_per_atom (armed (b) aux16 there); the uhf loop is
        hybrid-only.'''
        src = open(_UHF_SRC).read()
        idx_return = src.index('return _j_energy_per_atom(')
        idx_loop = src.index('_taskgran_pair_batch(')
        self.assertLess(idx_return, idx_loop)


class TestTaskgranUksPortEquivalence(unittest.TestCase):
    '''GPU probe cells: armed UKS engagement + equivalence bands, the
    pure-functional fall-through, fused interplay, off-lane determinism
    and strict-reference inertness.'''

    def test_armed_hybrid_uks_grad_equivalence(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, ax0, rs0 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        e1, g1, c1 = _run_uks(WATER, 'def2-svp', 'b3lyp', '2')
        self.assertTrue(c1)
        pc1, ax1, rs1 = (_paircap_counter(), _aux16_counter(),
                         _resplit_counter())
        d_e = abs(e1 - e0)
        d_g = np.max(np.abs(g1 - g0))
        print('TASKGRAN-UKS-PORT b3lyp water cation: dE=%.3e dG=%.3e '
              'paircap+%d aux16+%d resplit+%d'
              % (d_e, d_g, pc1 - pc0, ax1 - ax0, rs1 - rs0))
        # the uhf-local hybrid loop is the ported carrier: both (a) and
        # (c) must engage at the water-cation probe
        self.assertGreaterEqual(pc1 - pc0, 1)
        self.assertGreaterEqual(rs1 - rs0, 1)
        # hybrids never touch the J-only (b) route
        self.assertEqual(ax1 - ax0, 0)
        self.assertLessEqual(d_e, 1e-10)
        self.assertLessEqual(d_g, 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_level1_pairs_only(self):
        '''Level 1 arms (a) only: the resplit counter must not move.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, rs0 = _paircap_counter(), _resplit_counter()
        e1, g1, c1 = _run_uks(WATER, 'def2-svp', 'b3lyp', '1')
        self.assertTrue(c1)
        pc1, rs1 = _paircap_counter(), _resplit_counter()
        print('TASKGRAN-UKS-PORT level1: dE=%.3e dG=%.3e paircap+%d '
              'resplit+%d' % (abs(e1 - e0), np.max(np.abs(g1 - g0)),
                              pc1 - pc0, rs1 - rs0))
        self.assertGreaterEqual(pc1 - pc0, 1)
        self.assertEqual(rs1 - rs0, 0)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_pure_functional_fallthrough(self):
        '''UKS pure functionals early-return to the rhf J-only route:
        resplit stays 0 from the uhf site (structural), the armed
        classes are the rhf-route ones (aux16), and values stay in the
        band.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0 = _run_uks(HE17, 'def2-svp', 'pbe', '0')
        self.assertTrue(c0)
        ax0, rs0 = _aux16_counter(), _resplit_counter()
        e1, g1, c1 = _run_uks(HE17, 'def2-svp', 'pbe', '2')
        self.assertTrue(c1)
        ax1, rs1 = _aux16_counter(), _resplit_counter()
        print('TASKGRAN-UKS-PORT pbe He17 cation: dE=%.3e dG=%.3e '
              'aux16+%d resplit+%d'
              % (abs(e1 - e0), np.max(np.abs(g1 - g0)), ax1 - ax0,
                 rs1 - rs0))
        # the pure route never reaches the ported loop: no resplit
        self.assertEqual(rs1 - rs0, 0)
        # the rhf J-only (b) mechanism carries the armed delta
        self.assertGreaterEqual(ax1 - ax0, 1)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_fused_skips_resplit(self):
        '''The LC-42 fused launch covers all aux groups in one launch;
        the armed packing must be skipped (zero engagement) also on the
        UKS path.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        rs0 = _resplit_counter()
        e1, g1, c1 = _run_uks(WATER, 'def2-svp', 'b3lyp', '2', fused='1')
        self.assertTrue(c1)
        rs1 = _resplit_counter()
        print('TASKGRAN-UKS-PORT fused water cation: dE=%.3e dG=%.3e '
              'resplit+%d' % (abs(e1 - e0), np.max(np.abs(g1 - g0)),
                              rs1 - rs0))
        self.assertEqual(rs1 - rs0, 0)
        self.assertLessEqual(abs(e1 - e0), 1e-10)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-9)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        os.environ.pop('GPU4PYSCF_DF_IP1_FUSED', None)
        _reset_knob_cache()

    def test_offlane_determinism_zero_counters(self):
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        e0, g0, c0 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        pc0, rs0 = _paircap_counter(), _resplit_counter()
        e1, g1, c1 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c1)
        self.assertEqual(_paircap_counter(), pc0)
        self.assertEqual(_resplit_counter(), rs0)
        # the ip1/j3c atomicAdd ULP run-to-run scatter class: banded,
        # not bitwise (band 1e-11 at water scale)
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()

    def test_armed_uks_strict_reference_inert(self):
        '''The shared helpers refuse per call in strict-fp64-reference
        contexts, so reference anchors stay verbatim on the UKS path
        too.'''
        self.assertTrue(_helpers_present(), 'TASKGRAN helpers absent')
        from gpu4pyscf.dft import numint
        e0, g0, c0 = _run_uks(WATER, 'def2-svp', 'b3lyp', '0')
        self.assertTrue(c0)
        os.environ['GPU4PYSCF_DF_TASKGRAN'] = '2'
        _reset_knob_cache()
        pc0, rs0 = _paircap_counter(), _resplit_counter()
        with numint.strict_fp64_reference_ctx():
            mol = pyscf.M(atom=WATER, basis='def2-svp', charge=1, spin=1,
                          output='/dev/null', verbose=0)
            from gpu4pyscf.dft import uks
            mf = uks.UKS(mol)
            mf.xc = 'b3lyp'
            mf = mf.density_fit()
            mf.grids.level = 1
            mf.max_cycle = 100
            mf.conv_tol = 1e-11
            e1 = mf.kernel()
            g1 = np.asarray(mf.nuc_grad_method().kernel())
        self.assertTrue(mf.converged)
        self.assertEqual(_paircap_counter(), pc0)
        self.assertEqual(_resplit_counter(), rs0)
        self.assertLessEqual(abs(e1 - e0), 1e-11)
        self.assertLessEqual(np.max(np.abs(g1 - g0)), 1e-11)
        os.environ.pop('GPU4PYSCF_DF_TASKGRAN', None)
        _reset_knob_cache()


if __name__ == '__main__':
    print('run via pytest from the repo root '
          '(the gpu4090-src namespace-package trap)')
    raise SystemExit(1)

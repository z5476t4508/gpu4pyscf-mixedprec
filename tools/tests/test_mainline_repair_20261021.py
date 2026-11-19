#!/usr/bin/env python3
"""Mainline-repair (2026-10-21) red->green regression tests for the three
frozen-batch precision failures (epinephrine/B3LYP, methoxy/B3LYP,
methoxy/wB97M-V) and their shared root causes.

Root cause 1 -- GPU4PYSCF_XC_GRID_BLOCKSIZE changed physics: a loop block
larger than MIN_BLK_SIZE evaluated the UNION of the covered default blocks'
AO-screen shell sets, silently keeping tail contributions the default
partition drops.  Measured on epinephrine/b3lyp/def2-svp (A4 single-knob
isolation): converged e_tot shifted -1.932e-06 Ha (the energy-gate
failure); fixed-dm excsum -1.9e-06 Ha, max|dvmat| 1.1e-04, dnelec +7.9e-6;
an unscreened (AO_THRESHOLD=1e-300) cross-check showed both partitions
bitwise-identical, proving the screen was the sole mechanism.  The fix
(gpu4pyscf/dft/numint.py::_block_loop + gen_grid.py::get_partition_fix)
zeroes the AO rows each covered fine block's own screen dropped, inside
that fine block's point range: block size changes only GEMM batching and
reduction order (floor ~5e-15), never the screened physics.

Root cause 2 -- the strict fp64-reference sibling inherited process-global
opt-in knobs: armed GPU4PYSCF_XC_GRID_BLOCKSIZE / GPU4PYSCF_VV10_UWE_KERNEL
(=fp32, a C-side process-lifetime kernel selection) leaked into the
stability_verify reference trajectory, so it was not the default-lane
solution the strict-match contract compares against.  The fix
(gpu4pyscf/dft/numint.py::strict_fp64_reference_ctx, honored by
scf/stability_verify.py and by a new exported VV10_uwe_kernel_set_override
in libgdft) runs the sibling under the default partition and the reference
warp1 VV10 kernel, restoring everything on exit.

These tests are GPU tests (a CUDA context is created); they need the
gpu4pyscf source env.  The epinephrine SCF pair and the two methoxy SCFs
run in ~40 s total on the RTX 4090.
"""
import os
import subprocess
import sys
import unittest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)

try:
    import numpy  # noqa: F401
    import cupy as cp  # noqa: F401
    import pyscf  # noqa: F401
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

if not HAS_GPU:
    raise unittest.SkipTest('needs the gpu4pyscf source env (numpy/cupy/pyscf)')

EPINEPHRINE_XYZ = """
O 1.7394 -2.1169 -1.0894
O -2.2941 1.0781 -1.7979
O -3.6584 0.5842 0.5722
N 2.6359 1.0243 0.7030
C 1.6787 -1.1447 -0.0373
C 0.2684 -0.6866 0.1208
C 2.6376 0.0193 -0.3576
C -0.3658 -0.0099 -0.9212
C -0.4164 -0.9343 1.3105
C -1.6849 0.4191 -0.7732
C -1.7353 -0.5053 1.4585
C -2.3696 0.1713 0.4166
C 3.5645 2.1013 0.3950
H 2.0210 -1.6511 0.8741
H 2.3808 0.4742 -1.3225
H 3.6478 -0.3931 -0.4831
H 0.1501 0.1801 -1.8589
H 0.0640 -1.4598 2.1315
H 2.9013 0.5888 1.5858
H -2.2571 -0.7050 2.3907
H 2.6646 -2.4067 -1.1652
H 3.2862 2.6124 -0.5325
H 4.5925 1.7346 0.3078
H 3.5401 2.8441 1.1985
H -3.2008 1.2997 -1.5231
H -3.9690 0.3259 1.4570
"""

METHOXY_XYZ = """
O 0.000000 0.000000 0.000000
C 0.000000 0.000000 1.368000
H 0.000000 1.027670 1.731311
H -0.889988 -0.513835 1.731311
H 0.889988 -0.513835 1.731311
"""

# the optimized-stack knobs that armed the failures (upstream_comparison
# STACK_ENV minus the DF/grad gradient-side knobs these SCF-level tests
# cannot exercise)
STACK_ENV = {
    'GPU4PYSCF_XC_GRID_BLOCKSIZE': '16384',
    'GPU4PYSCF_VV10_UWE_KERNEL': 'fp32',
    'GPU4PYSCF_SCF_VMAT_F32': 'headtail',
}


def _scrub_gpu_env():
    for k in list(os.environ):
        if k.startswith('GPU4PYSCF_'):
            os.environ.pop(k)


class _EnvRestore:
    def __enter__(self):
        self.saved = {k: v for k, v in os.environ.items()
                      if k.startswith('GPU4PYSCF_')}
        _scrub_gpu_env()
        return self

    def __exit__(self, *exc):
        import gpu4pyscf.dft.numint as gnumint
        _scrub_gpu_env()
        os.environ.update(self.saved)
        gnumint._XC_GRID_BLKSIZE = None
        gnumint._strict_fp64_reference_active = False
        return False


def _clean_child(script):
    '''Run script in a fresh subprocess with every GPU4PYSCF_ env scrubbed
    (the plain default lane) and return its stdout-parsed JSON.'''
    import json
    env = {k: v for k, v in os.environ.items()
           if not k.startswith('GPU4PYSCF_')}
    env['PYTHONPATH'] = REPO
    proc = subprocess.run([sys.executable, '-c', script], env=env,
                          capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        raise AssertionError(
            f'clean-lane subprocess failed:\n{proc.stderr[-2000:]}')
    return json.loads(proc.stdout)


class TestBlocksizePhysicsInvariance(unittest.TestCase):
    '''Red->green for root cause 1 (epinephrine/B3LYP energy failure).'''

    def _mf(self, mol, xc='b3lyp'):
        from gpu4pyscf import dft as gpu_dft
        mf = gpu_dft.RKS(mol, xc=xc).density_fit(
            auxbasis='def2-universal-jkfit')
        mf.conv_tol = 1e-10
        mf.max_cycle = 50
        mf.direct_scf_tol = 1e-14
        mf.init_guess = 'minao'
        mf.grids.level = 3
        return mf

    def test_fixed_dm_rks_invariance(self):
        '''Single nr_rks call at the same dm: block size must not change
        nelec/excsum/vmat beyond the reduction-order floor.  Pre-fix
        (epinephrine): dexcsum -1.9e-06, dnelec +7.9e-6, max|dvmat| 1.1e-04.'''
        import gpu4pyscf.dft.numint as gnumint
        mol = pyscf.M(atom=EPINEPHRINE_XYZ, basis='def2-svp',
                      charge=0, spin=0, verbose=0)
        with _EnvRestore():
            mf = self._mf(mol)
            mf.grids.build(mol)
            dm = cp.asarray(mf.init_guess_by_minao())
            ni = mf._numint
            gnumint._XC_GRID_BLKSIZE = None
            nA, eA, vA = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
            gnumint._XC_GRID_BLKSIZE = 16384
            nB, eB, vB = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
        self.assertLessEqual(abs(float(nB) - float(nA)), 1e-11)
        self.assertLessEqual(abs(float(eB) - float(eA)), 1e-11)
        self.assertLessEqual(float(cp.abs(vA - vB).max()), 1e-9)

    def test_converged_scf_invariance(self):
        '''Whole converged SCF: 16384 vs the default partition.  Pre-fix:
        dE = -1.932e-06 Ha (the energy-gate failure).'''
        import gpu4pyscf.dft.numint as gnumint
        mol = pyscf.M(atom=EPINEPHRINE_XYZ, basis='def2-svp',
                      charge=0, spin=0, verbose=0)
        with _EnvRestore():
            gnumint._XC_GRID_BLKSIZE = None
            eA = self._mf(mol).kernel()
            gnumint._XC_GRID_BLKSIZE = 16384
            eB = self._mf(mol).kernel()
        self.assertLessEqual(abs(float(eB) - float(eA)), 1e-10)

    def test_partition_fix_dies_with_screen_cache(self):
        '''The derived partition-fix cache must not outlive the screening
        cache it was derived from: an external `grids._non0ao_idx = None`
        reset after a screening-relevant change (AO_THRESHOLD) must rebuild
        it.  Pre-cache-fix (MR1 development): a stale fix under a 1e-300
        screen perturbed excsum by 7.6 Ha.'''
        import gpu4pyscf.dft.numint as gnumint
        mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                      verbose=0)
        with _EnvRestore():
            mf = self._mf(mol, xc='b3lyp')
            mf.grids.build(mol)
            dm = cp.asarray(mf.init_guess_by_minao())
            ni = mf._numint
            saved_thr = gnumint.AO_THRESHOLD
            try:
                # build the fix under the default screen
                gnumint._XC_GRID_BLKSIZE = 16384
                nA, eA, vA = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
                # change the screen and reset the screening cache the
                # documented way; the fix must rebuild (not reuse)
                gnumint.AO_THRESHOLD = 1e-300
                mf.grids._non0ao_idx = None
                nB, eB, vB = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
                # same screen at the default block size for comparison
                gnumint._XC_GRID_BLKSIZE = None
                nC, eC, vC = ni.nr_rks(mol, mf.grids, 'b3lyp', dm)
            finally:
                gnumint.AO_THRESHOLD = saved_thr
                gnumint._XC_GRID_BLKSIZE = None
                mf.grids._non0ao_idx = None
        self.assertLessEqual(abs(float(eB) - float(eC)), 1e-11)
        self.assertLessEqual(abs(float(nB) - float(nC)), 1e-11)
        self.assertLessEqual(float(cp.abs(vB - vC).max()), 1e-10)


class TestStrictFp64ReferenceCtx(unittest.TestCase):
    '''Red->green for root cause 2 (reference-sibling knob isolation).'''

    def test_ctx_forces_default_partition_and_restores(self):
        import gpu4pyscf.dft.numint as gnumint
        with _EnvRestore():
            gnumint._XC_GRID_BLKSIZE = 16384
            self.assertEqual(gnumint._scf_grid_blksize(), 16384)
            with gnumint.strict_fp64_reference_ctx():
                self.assertEqual(gnumint._scf_grid_blksize(),
                                 gnumint.MIN_BLK_SIZE)
            self.assertEqual(gnumint._scf_grid_blksize(), 16384)

    def test_ctx_uwe_reference_kernel_roundtrip(self):
        '''The C-side override selects the reference warp1 kernel inside
        the ctx and restores the env-cached selection (fp32) on exit.'''
        os.environ['GPU4PYSCF_VV10_UWE_KERNEL'] = 'fp32'
        try:
            from gpu4pyscf.dft.numint import libgdft
            import gpu4pyscf.dft.numint as gnumint
            libgdft.VV10_uwe_kernel_set_override.restype = None
            mol = pyscf.M(atom='H 0 0 0; F 0 0 0.95', basis='def2-svp',
                          verbose=0)
            from gpu4pyscf import dft as gpu_dft
            mf = gpu_dft.RKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.grids.level = 2
            mf.nlcgrids.level = 2
            with gnumint.strict_fp64_reference_ctx():
                # an NLC call inside the ctx runs the warp1 kernel (no
                # error, override honored); the real numerical assertion is
                # the next test
                dm = cp.asarray(mf.init_guess_by_minao())
                mf._numint.nr_nlc_vxc(mol, mf.nlcgrids, 'wb97m-v', dm)
        finally:
            os.environ.pop('GPU4PYSCF_VV10_UWE_KERNEL', None)

    def test_wb97mv_reference_sibling_matches_clean_lane(self):
        '''Integration: with the STACK knobs armed in-process, the
        auto+strict route's adopted solution must equal the plain default
        lane's converged solution computed in a clean subprocess (methoxy
        UKS wb97m-v, frozen-batch failure 3's cell).'''
        import gpu4pyscf.dft.numint as gnumint
        child = (
            "import json, pyscf\n"
            "from gpu4pyscf import dft as gpu_dft\n"
            "mol = pyscf.M(atom='''" + METHOXY_XYZ.strip() + "''',\n"
            "              basis='def2-svp', charge=0, spin=1, verbose=0)\n"
            "mf = gpu_dft.UKS(mol, xc='wb97m-v').density_fit(\n"
            "    auxbasis='def2-universal-jkfit')\n"
            "mf.conv_tol = 1e-10\n"
            "mf.max_cycle = 50\n"
            "mf.direct_scf_tol = 1e-14\n"
            "mf.init_guess = 'minao'\n"
            "mf.grids.level = 3\n"
            "mf.nlcgrids.level = 3\n"
            "e = mf.kernel()\n"
            "dm = mf.make_rdm1()\n"
            "print(json.dumps({'e_tot': float(e),\n"
            "                  'converged': bool(mf.converged),\n"
            "                  'dm_max': float(dm.get().max())}))\n")
        ref = _clean_child(child)
        self.assertTrue(ref['converged'])

        mol = pyscf.M(atom=METHOXY_XYZ, basis='def2-svp',
                      charge=0, spin=1, verbose=0)
        with _EnvRestore():
            for k, v in STACK_ENV.items():
                os.environ[k] = v
            gnumint._XC_GRID_BLKSIZE = None
            from gpu4pyscf import dft as gpu_dft
            mf = gpu_dft.UKS(mol, xc='wb97m-v').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.max_cycle = 50
            mf.direct_scf_tol = 1e-14
            mf.init_guess = 'minao'
            mf.grids.level = 3
            mf.nlcgrids.level = 3
            mf.precision_mode = 'auto+strict'
            e = mf.kernel()
            sv = mf.scf_summary.get('stability_verification', {})
        self.assertTrue(mf.converged)
        self.assertIs(sv.get('strict_match_verified'), True,
                      f'strict match incomplete: {sv}')
        # 1e-11 also guards the UWE isolation: pre-fix the armed
        # GPU4PYSCF_VV10_UWE_KERNEL=fp32 leaked into the sibling and
        # displaced its solution by 4.2e-11 (measured in the 2026-10-21
        # operator pilot)
        self.assertLessEqual(abs(float(e) - ref['e_tot']), 1e-11,
                             f'adopted e_tot {e!r} vs clean lane '
                             f"{ref['e_tot']!r}; record {sv}")

    def test_uks_soft_mode_strict_match_b3lyp(self):
        '''methoxy UKS b3lyp (frozen-batch failure 2's cell): the auto
        policy's soft-mode branch must certify against and adopt the
        default-lane solution under the armed stack.'''
        import gpu4pyscf.dft.numint as gnumint
        child = (
            "import json, pyscf\n"
            "from gpu4pyscf import dft as gpu_dft\n"
            "mol = pyscf.M(atom='''" + METHOXY_XYZ.strip() + "''',\n"
            "              basis='def2-svp', charge=0, spin=1, verbose=0)\n"
            "mf = gpu_dft.UKS(mol, xc='b3lyp').density_fit(\n"
            "    auxbasis='def2-universal-jkfit')\n"
            "mf.conv_tol = 1e-10\n"
            "mf.max_cycle = 50\n"
            "mf.direct_scf_tol = 1e-14\n"
            "mf.init_guess = 'minao'\n"
            "mf.grids.level = 3\n"
            "e = mf.kernel()\n"
            "print(json.dumps({'e_tot': float(e),\n"
            "                  'converged': bool(mf.converged)}))\n")
        ref = _clean_child(child)
        self.assertTrue(ref['converged'])

        mol = pyscf.M(atom=METHOXY_XYZ, basis='def2-svp',
                      charge=0, spin=1, verbose=0)
        with _EnvRestore():
            for k, v in STACK_ENV.items():
                os.environ[k] = v
            gnumint._XC_GRID_BLKSIZE = None
            from gpu4pyscf import dft as gpu_dft
            mf = gpu_dft.UKS(mol, xc='b3lyp').density_fit(
                auxbasis='def2-universal-jkfit')
            mf.conv_tol = 1e-10
            mf.max_cycle = 50
            mf.direct_scf_tol = 1e-14
            mf.init_guess = 'minao'
            mf.grids.level = 3
            mf.precision_mode = 'auto+strict'
            e = mf.kernel()
            sv = mf.scf_summary.get('stability_verification', {})
        self.assertTrue(mf.converged)
        self.assertIs(sv.get('strict_match_verified'), True,
                      f'strict match incomplete: {sv}')
        self.assertLessEqual(abs(float(e) - ref['e_tot']), 1e-10,
                             f'adopted e_tot {e!r} vs clean lane '
                             f"{ref['e_tot']!r}; record {sv}")


if __name__ == '__main__':
    unittest.main()

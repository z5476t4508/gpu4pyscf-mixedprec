"""Step 8G: does ejk_int3c2e_ip2_f32 compute the right Hessian, and how fast?

The port measured 34.88s -> 18.78s on the JK half, i.e. the ip2 kernel appears
to have gone from 17.31s to ~1.3s. That is ~13x, far above the 2.7x its ported
sibling ejk_int3c2e_ip1_f32 achieved, and a speedup that large on a kernel
believed to be bandwidth-bound is a reason to suspect the kernel is not doing
the work, not a reason to celebrate. A kernel that returns early, writes
nothing, or is never launched also looks very fast.

So this checks the output, not the clock, in three ways of increasing strength:

  1. the raw JK Hessian block from `_jk_energy_per_atom` itself, float32 lane vs
     float64 -- this isolates the kernel from everything downstream, and a
     kernel doing nothing would show a ~100% relative difference;
  2. the translational sum rule: sum_A d2E/dR_A dR_B = 0 holds exactly for this
     contribution, so it tests the kernel against theory rather than against
     another run of the same code;
  3. the frequencies, which is what a user consumes.

The fp64 reference is computed by forcing precision_mode='fp64', which selects
the float64 kernel through the same dispatch.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8g_ip2_f32_verify.py [--xc b3lyp] [--xyz ...]
"""
import argparse
import os
import sys
import time

import cupy as cp
import numpy as np
import pyscf

sys.path.insert(0, os.path.dirname(__file__))
import vibanalysis  # noqa: E402

from gpu4pyscf import dft, scf                       # noqa: E402
from gpu4pyscf.df.hessian import rhf as df_rhf_hess   # noqa: E402
from gpu4pyscf.lib import precision                   # noqa: E402
from gpu4pyscf.scf.jk import libvhf_rys               # noqa: E402

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')

counts = {'fp64': 0, 'fp32': 0}


def count_launches():
    '''confirm which kernel actually ran -- "fast" is meaningless if the f32
    entry point was never called, or if it was called and returned an error'''
    for name, key in (('ejk_int3c2e_ip2', 'fp64'),
                      ('ejk_int3c2e_ip2_f32', 'fp32')):
        try:
            fn = getattr(libvhf_rys, name)
        except AttributeError:
            continue

        def wrapper(*a, _fn=fn, _key=key, **kw):
            counts[_key] += 1
            return _fn(*a, **kw)
        setattr(libvhf_rys, name, wrapper)


def build(mol, xc):
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit()
    else:
        mf = dft.RKS(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    return mf


def jk_block(mf, mode):
    '''the JK half alone, timed, in the given lane'''
    hessobj = mf.Hessian()
    hessobj.precision_mode = mode
    dm0 = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    if mf.mol.spin != 0:
        raise NotImplementedError
    if mf.with_df.intopt is None:
        mf.with_df.build(build_cderi=False)
    intopt = mf.with_df.intopt
    if hasattr(mf, 'xc'):
        _, alpha, hyb = mf._numint.rsh_and_hybrid_coeff(mf.xc, spin=0)
        args = (intopt, dm0, 1., hyb)
    else:
        args = (intopt, dm0)
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    with hessobj.jk_precision():
        ejk = df_rhf_hess._jk_energy_per_atom(*args)
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0
    return cp.asnumpy(ejk), dt


def full(mol, mf, mode):
    hessobj = mf.Hessian()
    hessobj.precision_mode = mode
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    h = hessobj.kernel()
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0
    nu = vibanalysis.frequencies(mol, h)
    zpe, s = vibanalysis.thermochemistry(mol, h)
    return nu, zpe, s, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '020_Vitamin_C.xyz'))
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--basis', default='def2-svp')
    args = p.parse_args()

    count_launches()
    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mol.nao} AO', flush=True)

    mf = build(mol, args.xc)

    # --- 1. the JK block alone -------------------------------------------
    e64, t64 = jk_block(mf, 'fp64')
    n64 = dict(counts)
    e32, t32 = jk_block(mf, 'auto')
    print(f'\n  JK block   fp64 {t64:6.2f}s -> fp32 {t32:6.2f}s  '
          f'({t64/t32:.2f}x)', flush=True)
    print(f'  kernel launches: float64 entry {n64["fp64"]}, '
          f'float32 entry {counts["fp32"]}', flush=True)
    if counts['fp32'] == 0:
        print('  *** the float32 kernel was never called: the speedup is not '
              'from this port ***', flush=True)

    scale = np.abs(e64).max()
    d = np.abs(e32 - e64)
    print(f'  |ejk|max {scale:.6e}   max|d| {d.max():.3e}   '
          f'rel {d.max()/scale:.3e}   rms.rel '
          f'{np.sqrt((( e32-e64)**2).mean())/scale:.3e}', flush=True)

    # --- 2. translational invariance --------------------------------------
    # sum over one atom index must vanish for this contribution
    for tag, e in (('fp64', e64), ('fp32', e32)):
        r = np.abs(e.sum(axis=0)).max() / scale
        print(f'  sum rule ({tag}): max|sum_A d2E/dR_A dR_B| / |ejk|max '
              f'= {r:.3e}', flush=True)

    # --- 3. frequencies ---------------------------------------------------
    nu64, zpe64, s64, T64 = full(mol, mf, 'fp64')
    nu32, zpe32, s32, T32 = full(mol, mf, 'auto')
    print(f'\n  full Hessian  fp64 {T64:6.2f}s -> lane {T32:6.2f}s  '
          f'({T64/T32:.2f}x)', flush=True)
    print(f'    max|dnu| {np.abs(nu32-nu64).max():9.4f}  '
          f'rms {np.sqrt(((nu32-nu64)**2).mean()):9.4f}  '
          f'dZPE {abs(zpe32-zpe64)*627.5095:9.2e} kcal/mol  '
          f'dSvib {abs(s32-s64)*627.5095*1000:9.2e} cal/mol/K', flush=True)
    print('  reference points: pre-port B3LYP lane 0.0058, sentinel detection '
          '0.0064,\n  known-bad change 0.0374 cm^-1', flush=True)


if __name__ == '__main__':
    main()

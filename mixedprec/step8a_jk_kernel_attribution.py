"""Step 8A: attribute the JK half of partial_hess_elec to individual CUDA kernels.

step6j split `_jk_energy_per_atom` by call signature, which answered "which
contraction" but not "which kernel". Before porting `ejk_int3c2e_ip2` to float32
(~550 lines of CUDA) we need its share of the phase: B3LYP's JK half is 34.9s of
84.4s, but if ip2 owns only a fifth of that, the whole port buys ~2%.

Method: wrap the ctypes entry points on the loaded library with
device-synchronised timers. ctypes CDLL caches attribute lookups in the instance
dict, so `setattr(libvhf_rys, name, wrapper)` is picked up by every later
`libvhf_rys.<name>` / `getattr(libvhf_rys, name)`. Both call sites do their
lookup at call time (`kern_ip2 = libvhf_rys.ejk_int3c2e_ip2`) or at evaluator
construction time (`_int3c2e_ip1_evaluator`), i.e. after this patch is applied.

Synchronisation matters: the wrapper's own `return` says nothing about when the
kernel finished, so each call is bracketed by deviceSynchronize(). That adds the
tail of any overlapped work to whichever kernel is timed, which is the honest
direction here (it cannot make a kernel look smaller than it is).

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8a_jk_kernel_attribution.py [--xc b3lyp] [--xyz ...]
"""
import argparse
import collections
import os
import time

import cupy as cp
import numpy as np
import pyscf

from gpu4pyscf import dft, scf
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.lib import precision
from gpu4pyscf.scf.jk import libvhf_rys

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')

# the kernels reachable from _jk_energy_per_atom / _j_energy_per_atom
KERNELS = ['ejk_int3c2e_ip2', 'ejk_int3c2e_ip2_f32', 'fill_int3c2e_ip1', 'fill_int3c2e_ipaux',
           'int2c2e_ip1', 'int2c2e_ip1ip2', 'int2c2e_ip1ip1']

t_kern = collections.Counter()
n_kern = collections.Counter()


def patch_kernels():
    for name in KERNELS:
        try:
            fn = getattr(libvhf_rys, name)
        except AttributeError:
            continue

        def wrapper(*a, _fn=fn, _name=name, **kw):
            cp.cuda.runtime.deviceSynchronize()
            t = time.perf_counter()
            r = _fn(*a, **kw)
            cp.cuda.runtime.deviceSynchronize()
            t_kern[_name] += time.perf_counter() - t
            n_kern[_name] += 1
            return r
        setattr(libvhf_rys, name, wrapper)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '057_Tamoxifen.xyz'))
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--mode', default='auto')
    args = p.parse_args()

    patch_kernels()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    if args.xc == 'hf':
        mf = scf.RHF(mol).density_fit()
    else:
        mf = dft.RKS(mol, xc=args.xc).density_fit()
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()

    hessobj = mf.Hessian()
    hessobj.precision_mode = args.mode

    orig = df_rhf_hess._jk_energy_per_atom
    t_jk = [0.0, 0]

    def timed(*a, **kw):
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        r = orig(*a, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t_jk[0] += time.perf_counter() - t
        t_jk[1] += 1
        return r
    df_rhf_hess._jk_energy_per_atom = timed

    with (precision.fp32() if args.mode in ('auto', 'fp32')
          else precision.fp64()):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        h = hessobj.partial_hess_elec()
        cp.cuda.runtime.deviceSynchronize()
        t_partial = time.perf_counter() - t0
    del h

    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mf.mol.nao} AO  '
          f'mode={args.mode}', flush=True)
    print(f'  partial_hess_elec      {t_partial:8.2f}s', flush=True)
    print(f'  _jk_energy_per_atom    {t_jk[0]:8.2f}s  '
          f'({100*t_jk[0]/t_partial:5.1f}% of partial, {t_jk[1]} calls)',
          flush=True)
    acc = 0.0
    for name, t in t_kern.most_common():
        acc += t
        print(f'      {name:22s} {t:8.2f}s  '
              f'({100*t/t_jk[0]:5.1f}% of JK, {n_kern[name]:5d} calls)',
              flush=True)
    print(f'      {"[non-kernel]":22s} {t_jk[0]-acc:8.2f}s  '
          f'({100*(t_jk[0]-acc)/t_jk[0]:5.1f}% of JK)  '
          f'GEMMs, metric solve, takes, host copies', flush=True)
    ip2 = t_kern.get('ejk_int3c2e_ip2', 0.0)
    if ip2:
        # 2.7x, not the card's 64:1 fp32:fp64 ratio: the fully-ported
        # ejk_int3c2e_ip1_f32 measured 3.181s -> 1.18s, so these Rys kernels
        # are bound by memory and shared-memory traffic, not fp64 arithmetic.
        # Use the measured sibling as the ceiling rather than the hardware one.
        gain = ip2 * (1 - 1/2.7)
        print(f'  ip2 fp32 port ceiling (2.7x, from ip1_f32): partial '
              f'{t_partial:.2f}s -> {t_partial-gain:.2f}s '
              f'({t_partial/(t_partial-gain):.3f}x)', flush=True)


if __name__ == '__main__':
    main()

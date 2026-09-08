"""Step 8B: split the 12.39s of `_jk_energy_per_atom` that is not a CUDA kernel.

step8a attributed the JK half of B3LYP's partial_hess_elec:

    ejk_int3c2e_ip2      17.51s   50.3%
    fill_int3c2e_ipaux    3.34s    9.6%
    fill_int3c2e_ip1      1.58s    4.5%
    [non-kernel]         12.39s   35.6%

The non-kernel third is bigger than a residual should be, and the two things it
is known to contain are of very different value: contractions are real work that
a precision port could touch, while `dm_oo_full = empty_mapped(...)` +
`dm_oo.get(out=...)` is a device->host->device round trip taken only to save
memory, and on a 32 GB card it may not be needed at all.

Method: the phase pulls its helpers into the module namespace at import
(`from ...cupy_helper import contract, ndarray, empty_mapped, ...`), so patching
`df_rhf_hess.<name>` is enough -- the call sites resolve through the module dict
every time. Each wrapper brackets the call with deviceSynchronize(), same
caveat as step8a: overlapped tails land on whoever is timed, which can only make
a region look bigger than it is.

`contract` is also called from make_h1/solve_mo1, so the counters are gated on
being inside `_jk_energy_per_atom`.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8b_jk_nonkernel_split.py [--xc b3lyp] [--xyz ...]
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

KERNELS = ['ejk_int3c2e_ip2', 'ejk_int3c2e_ip2_f32', 'fill_int3c2e_ip1', 'fill_int3c2e_ipaux',
           'int2c2e_ip1', 'int2c2e_ip1ip2', 'int2c2e_ip1ip1']
# module-level helpers worth separating; `empty_mapped` allocates pinned host
# memory, so it is timed for its own cost, and `.get()` shows up under 'D2H'
HELPERS = ['contract', 'empty_mapped', 'transpose_sum', 'fill_symmetric',
           'int2c2e', 'int2c2e_ip1']

t_reg = collections.Counter()
n_reg = collections.Counter()
inside = [False]


def _timed(fn, label, gate=True):
    def wrapper(*a, **kw):
        if gate and not inside[0]:
            return fn(*a, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        r = fn(*a, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t_reg[label] += time.perf_counter() - t
        n_reg[label] += 1
        return r
    return wrapper


def patch():
    for name in KERNELS:
        try:
            fn = getattr(libvhf_rys, name)
        except AttributeError:
            continue
        # kernels are launched only from this phase in this script's run,
        # but gate them anyway so make_h1 cannot leak in
        setattr(libvhf_rys, name, _timed(fn, 'kern:' + name))
    for name in HELPERS:
        fn = getattr(df_rhf_hess, name, None)
        if fn is None:
            continue
        setattr(df_rhf_hess, name, _timed(fn, 'py:' + name))
    # the device->host round trip of dm_oo: cupy.ndarray.get cannot be patched
    # (C type), so wrap the one caller pattern by patching ndarray creation of
    # the mapped array instead -- .get is timed via a subclass-free shim below
    orig_get = cp.ndarray.get

    def timed_get(self, *a, **kw):
        if not inside[0]:
            return orig_get(self, *a, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        r = orig_get(self, *a, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t_reg['py:D2H .get'] += time.perf_counter() - t
        n_reg['py:D2H .get'] += 1
        return r
    try:
        cp.ndarray.get = timed_get
    except TypeError:
        print('  (cp.ndarray.get is not patchable; D2H folded into residual)',
              flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '057_Tamoxifen.xyz'))
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--mode', default='auto')
    args = p.parse_args()

    patch()

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
        inside[0] = True
        try:
            r = orig(*a, **kw)
        finally:
            inside[0] = False
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
          f'({100*t_jk[0]/t_partial:5.1f}% of partial)', flush=True)
    acc = 0.0
    for label, t in t_reg.most_common():
        acc += t
        print(f'      {label:22s} {t:8.2f}s  '
              f'({100*t/t_jk[0]:5.1f}% of JK, {n_reg[label]:6d} calls)',
              flush=True)
    print(f'      {"[residual]":22s} {t_jk[0]-acc:8.2f}s  '
          f'({100*(t_jk[0]-acc)/t_jk[0]:5.1f}% of JK)', flush=True)


if __name__ == '__main__':
    main()

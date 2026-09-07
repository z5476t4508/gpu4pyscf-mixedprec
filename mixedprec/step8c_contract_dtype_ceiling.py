"""Step 8C: is the JK half's 11.25s of `contract` fp64-arithmetic bound?

step8b showed the JK half of B3LYP's partial_hess_elec is not the kernel monolith
step8a suggested:

    ejk_int3c2e_ip2   17.31s   49.6%   Rys CUDA kernel
    contract          11.25s   32.3%   855 cuTENSOR/cuBLAS calls
    fill_* kernels     4.92s   14.1%
    rest               1.4 s    4.0%

Those two large entries have very different ceilings and this script measures
the difference rather than assuming it:

  - the Rys kernels are memory- and shared-memory-bound. The fully ported
    sibling `ejk_int3c2e_ip1_f32` went 3.181s -> 1.18s, i.e. **2.7x**, nowhere
    near this card's 64:1 fp32:fp64 ratio.
  - a GEMM large enough to be compute-bound has no such excuse: on a GB202 the
    fp64 pipes are 1/64 of the fp32 ones, so the same contraction in float32
    could be far cheaper than 2.7x -- or, if these 855 calls are small and
    latency-bound, no cheaper at all.

Which of the two it is decides where the next day of work goes, so it is
measured here on the actual shapes: every `contract` call inside the phase is
recorded (spec + operand shapes + dtype), aggregated by spec, and the top specs
are then replayed in float64 and float32 on identically-shaped random operands.

The replay reports both the per-call time and the implied saving for the whole
phase. It does *not* say the port is accurate -- only what it would buy.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8c_contract_dtype_ceiling.py [--xc b3lyp] [--top 8]
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
from gpu4pyscf.lib.cupy_helper import contract

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')

t_spec = collections.Counter()
n_spec = collections.Counter()
shapes = {}
inside = [False]


def patch():
    orig = df_rhf_hess.contract

    def wrapper(spec, a, b, *rest, **kw):
        if not inside[0]:
            return orig(spec, a, b, *rest, **kw)
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        r = orig(spec, a, b, *rest, **kw)
        cp.cuda.runtime.deviceSynchronize()
        dt = time.perf_counter() - t
        key = (spec, tuple(a.shape), tuple(b.shape),
               np.dtype(a.dtype).name, np.dtype(b.dtype).name,
               kw.get('out') is not None)
        t_spec[key] += dt
        n_spec[key] += 1
        shapes[key] = (a.shape, b.shape)
        return r
    df_rhf_hess.contract = wrapper


def replay(key, ncall, t_measured):
    '''time the same contraction on random operands in fp64 and fp32'''
    spec, sa, sb, da, db, has_out = key
    reps = 3 if ncall < 20 else 1
    out = {}
    pool = cp.get_default_memory_pool()
    for dt in (np.float64, np.float32):
        a = b = r = None
        try:
            # allocate at the target dtype directly: an fp64 scratch plus an
            # astype() copy is what ran the card out of memory
            a = cp.random.random(sa, dtype=np.float32).astype(dt, copy=False)
            b = cp.random.random(sb, dtype=np.float32).astype(dt, copy=False)
            r = contract(spec, a, b)      # warm up / plan
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                r = contract(spec, a, b)
            cp.cuda.runtime.deviceSynchronize()
            out[dt] = (time.perf_counter() - t0) / reps
        except Exception as e:            # noqa: BLE001 - report, do not die
            return None, None, repr(e)[:60]
        finally:
            a = b = r = None
            pool.free_all_blocks()
    return out[np.float64], out[np.float32], None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '057_Tamoxifen.xyz'))
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--mode', default='auto')
    p.add_argument('--top', type=int, default=8)
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

    orig_jk = df_rhf_hess._jk_energy_per_atom
    t_jk = [0.0]

    def timed(*a, **kw):
        cp.cuda.runtime.deviceSynchronize()
        t = time.perf_counter()
        inside[0] = True
        try:
            r = orig_jk(*a, **kw)
        finally:
            inside[0] = False
        cp.cuda.runtime.deviceSynchronize()
        t_jk[0] += time.perf_counter() - t
        return r
    df_rhf_hess._jk_energy_per_atom = timed

    with (precision.fp32() if args.mode in ('auto', 'fp32')
          else precision.fp64()):
        h = hessobj.partial_hess_elec()
    del h
    hessobj = mf = None
    cp.get_default_memory_pool().free_all_blocks()

    total = sum(t_spec.values())
    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mol.nao} AO  '
          f'mode={args.mode}', flush=True)
    print(f'  _jk_energy_per_atom {t_jk[0]:.2f}s, of which contract '
          f'{total:.2f}s in {sum(n_spec.values())} calls\n', flush=True)
    print(f'  {"spec":24s} {"shapes":34s} {"calls":>6s} {"total":>8s} '
          f'{"fp64/c":>8s} {"fp32/c":>8s} {"ratio":>6s}', flush=True)

    saved = 0.0
    covered = 0.0
    for key, t in t_spec.most_common(args.top):
        spec, sa, sb, da, db, has_out = key
        t64, t32, err = replay(key, n_spec[key], t)
        covered += t
        if err:
            print(f'  {spec:24s} {str(sa)+"x"+str(sb):34s} '
                  f'{n_spec[key]:6d} {t:7.2f}s  replay failed: {err}',
                  flush=True)
            continue
        ratio = t64 / t32 if t32 > 0 else float('nan')
        # scale the measured phase time by the replayed ratio: the replay
        # excludes whatever per-call Python/plan overhead the real call carries,
        # so attribute only the arithmetic part
        saved += t * (1 - 1/ratio) if ratio > 1 else 0.0
        print(f'  {spec:24s} {str(sa)+"x"+str(sb):34s} '
              f'{n_spec[key]:6d} {t:7.2f}s {t64*1e3:7.2f}ms {t32*1e3:7.2f}ms '
              f'{ratio:5.2f}x', flush=True)

    print(f'\n  top {args.top} specs cover {covered:.2f}s of {total:.2f}s '
          f'({100*covered/total:.0f}%)', flush=True)
    print(f'  fp32 contract would save ~{saved:.2f}s of the {t_jk[0]:.2f}s '
          f'JK half ({100*saved/t_jk[0]:.1f}%)', flush=True)
    print('  compare: a full ejk_int3c2e_ip2 fp32 port (~550 lines CUDA) saves '
          f'~{17.31*(1-1/2.7):.2f}s at the ip1_f32-measured 2.7x', flush=True)


if __name__ == '__main__':
    main()

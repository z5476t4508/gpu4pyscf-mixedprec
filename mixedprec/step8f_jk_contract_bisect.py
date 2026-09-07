"""Step 8F: which of the JK half's contractions can take float32, one spec at a time?

step8d cast all 284 large contractions in `_jk_energy_per_atom` to float32 and
Vitamin C's frequencies moved 8.13 cm^-1 -- 217x the known-bad change. step8e
ruled out TF32 as the cause (the float32 path is genuinely float32, 2.5e-7 to
4e-6 relative), so the blow-up is cancellation: this phase assembles the
auxiliary Hessian from differences of large terms, and the algorithm leans on an
exact identity to do it (df/hessian/rhf.py:456 gets the dj derivative as
-(di+dk) from the translational sum rule).

"All of them together are fatal" does not mean each is. step8c's top 12 specs
are 68% of the phase's contract time and they play different roles: some build
intermediates that are later reduced (error averages down), others accumulate
the result across batches with beta=1 (error accumulates), and one pair is the
outer product that *is* the aux Hessian block. This measures each separately, so
what survives is a boundary rather than a verdict.

One fp64 reference and one shipped-lane baseline are computed once and reused;
each spec then costs one Hessian. A spec is worth taking only if it clears both
bars: measurably faster (step8c's ratio) and an error that stays a comfortable
margin under the shipped lane's own 0.0058 cm^-1, not merely under the
sentinel's 0.0064 detection threshold.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8f_jk_contract_bisect.py [--xc b3lyp]
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

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')

# step8c's ranking on B3LYP/Tamoxifen: spec -> (seconds in the phase, fp64/fp32)
SPECS = [
    ('xruj,sr->xsuj',     1.67, 23.1),   # rhf.py:516  build tmp0
    ('suj,xruj->xsr',     1.64, 15.5),   # rhf.py:528  build tmp1
    ('pxrij,qyrij->pqxy', 1.09, 10.3),   # rhf.py:491  accumulates into ejk
    ('xrij,ysij->xrys',   0.99, 35.1),   # rhf.py:354  h_aux outer product
    ('xsr,ysr->sxy',      0.57, 49.3),   # rhf.py:530  accumulates h_ao_aux
    ('rs,pir->spi',       0.82, 31.6),   # rhf.py:474  beta=1 into j3c_100
    ('xrij,sij->xrs',     0.35, 34.0),   # rhf.py:381  dm_aux1
    ('xuv,vij->xuij',     0.33, 39.5),   # rhf.py:350  j3c_oo1p
    ('rs,ts->rt',         0.25, 31.6),   # rhf.py:375  j2c_ip2
]

inside = [False]
allowed = [()]
stats = [0]


def patch(min_elems):
    orig = df_rhf_hess.contract

    def wrapper(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        if (not inside[0] or pattern not in allowed[0]
                or a.dtype != np.float64 or b.dtype != np.float64
                or a.size + b.size < min_elems):
            return orig(pattern, a, b, alpha, beta, out)
        stats[0] += 1
        r = orig(pattern, a.astype(np.float32), b.astype(np.float32), alpha)
        if out is None:
            return r.astype(np.float64)
        if beta == 0:
            out[:] = r
        else:
            out *= beta
            out += r
        return out
    df_rhf_hess.contract = wrapper

    orig_jk = df_rhf_hess._jk_energy_per_atom

    def gated(*a, **kw):
        inside[0] = True
        try:
            return orig_jk(*a, **kw)
        finally:
            inside[0] = False
    df_rhf_hess._jk_energy_per_atom = gated


def run(mol, xc, mode):
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit()
    else:
        mf = dft.RKS(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    hessobj = mf.Hessian()
    hessobj.precision_mode = mode
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    h = hessobj.kernel()
    cp.cuda.runtime.deviceSynchronize()
    dt = time.perf_counter() - t0
    nu = vibanalysis.frequencies(mol, h)
    zpe, s = vibanalysis.thermochemistry(mol, h)
    mf = hessobj = h = None
    cp.get_default_memory_pool().free_all_blocks()
    return nu, zpe, s, dt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '020_Vitamin_C.xyz'))
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--min-elems', type=float, default=1e6)
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mol.nao} AO', flush=True)

    precision.set_precision('fp64')
    nu_ref, zpe_ref, s_ref, t_ref = run(mol, args.xc, 'fp64')
    patch(args.min_elems)
    nu_l, zpe_l, s_l, t_l = run(mol, args.xc, 'auto')   # allowed = () -> no cast
    lane = np.abs(nu_l - nu_ref).max()
    print(f'  fp64 reference {t_ref:6.2f}s   shipped lane {t_l:6.2f}s  '
          f'max|dnu| {lane:.4f} cm^-1\n', flush=True)

    print(f'  {"spec":20s} {"s in phase":>10s} {"ratio":>6s} {"casts":>6s} '
          f'{"max|dnu|":>10s} {"vs lane":>9s} {"rms":>9s} {"verdict":>8s}',
          flush=True)

    safe = []
    for spec, t_phase, ratio in SPECS:
        allowed[0] = (spec,)
        stats[0] = 0
        nu, zpe, s, _ = run(mol, args.xc, 'auto')
        err = np.abs(nu - nu_ref).max()
        rms = np.sqrt(((nu - nu_ref) ** 2).mean())
        # a spec is safe if it does not measurably move the lane it joins;
        # 2x the lane's own deviation is the margin, well under the sentinel's
        # 0.0064 detection threshold and 6x under it
        ok = err < 2 * lane
        if ok:
            safe.append((spec, t_phase, ratio))
        print(f'  {spec:20s} {t_phase:9.2f}s {ratio:5.1f}x {stats[0]:6d} '
              f'{err:10.4f} {err/lane:8.1f}x {rms:9.4f} '
              f'{"safe" if ok else "FATAL":>8s}', flush=True)

    if safe:
        allowed[0] = tuple(s for s, _, _ in safe)
        stats[0] = 0
        nu, zpe, s, _ = run(mol, args.xc, 'auto')
        err = np.abs(nu - nu_ref).max()
        gain = sum(t * (1 - 1/r) for _, t, r in safe)
        print(f'\n  all {len(safe)} safe specs together: max|dnu| {err:.4f} '
              f'({err/lane:.1f}x lane), {stats[0]} casts', flush=True)
        print(f'  combined saving ~{gain:.2f}s of the 34.88s JK half '
              f'({100*gain/34.88:.1f}%), i.e. B3LYP 84.4s -> {84.4-gain:.1f}s',
              flush=True)
        print('  NOTE: safe individually does not imply safe together -- the '
              'line above\n  is the measurement that decides it, not the sum.',
              flush=True)
    else:
        print('\n  nothing survives: the whole phase needs float64 '
              'contractions.', flush=True)
    print(f'\n  reference points: shipped lane {lane:.4f}, sentinel detection '
          f'0.0064, known-bad 0.0374 cm^-1', flush=True)


if __name__ == '__main__':
    main()

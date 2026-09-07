"""Step 8D: can the JK half's contractions run in float32 at all?

step8c measured the *ceiling*: every large `contract` in `_jk_energy_per_atom`
runs 10-49x faster in float32, worth ~7.3s of the 34.9s phase (B3LYP/Tamoxifen).
That is the same order as a full `ejk_int3c2e_ip2` fp32 port (~10.9s) for a small
fraction of the work -- but only if the accuracy holds, and nothing so far has
measured that. These contractions are not integral evaluation; they *are* the
accumulation that produces the auxiliary Hessian blocks, over reduction lengths
of 2626-3080 (aux) and 10^4 (occ pairs), so fp32 accumulation is the risk.

This measures it before any code is changed, which is the step I skipped on the
`_get_vxc_deriv2` port and paid a day for.

Method: monkeypatch `contract` inside the phase to cast both operands to float32,
contract, and stage the result back into the float64 `out` (cuTENSOR will not
accumulate float32 operands into a float64 output with beta=1, so beta is
applied on the host side in float64). Casting is gated on operand size, since a
small latency-bound contraction gains no time and should not spend accuracy for
nothing.

The reported timing is NOT the timing of a real port: this probe *adds* two
casts per call. Only the frequency deviation is meaningful here.

Passing means the deviation stays comfortably under the sentinel's physical
budget, not merely under its detection threshold -- the two were conflated
earlier. The known-bad reference point is 0.0374 cm^-1.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8d_jk_contract_accuracy.py [--xc b3lyp] [--min-elems 1e6]
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

from gpu4pyscf import dft, scf                      # noqa: E402
from gpu4pyscf.df.hessian import rhf as df_rhf_hess  # noqa: E402
from gpu4pyscf.lib import precision                  # noqa: E402

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')

inside = [False]
stats = {'cast': 0, 'kept': 0}


def patch(min_elems):
    orig = df_rhf_hess.contract

    def wrapper(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        if (not inside[0] or a.dtype != np.float64 or b.dtype != np.float64
                or a.size + b.size < min_elems):
            stats['kept'] += inside[0]
            return orig(pattern, a, b, alpha, beta, out)
        stats['cast'] += 1
        r = orig(pattern, a.astype(np.float32), b.astype(np.float32), alpha)
        if out is None:
            return r.astype(np.float64)
        if beta == 0:
            out[:] = r
        else:
            # beta applied in float64: cuTENSOR cannot accumulate float32
            # operands into a float64 output
            out *= beta
            out += r
        return out
    df_rhf_hess.contract = wrapper

    orig_jk = df_rhf_hess._jk_energy_per_atom

    def timed(*a, **kw):
        inside[0] = True
        try:
            return orig_jk(*a, **kw)
        finally:
            inside[0] = False
    df_rhf_hess._jk_energy_per_atom = timed


def build(mol, xc):
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit()
    else:
        mf = dft.RKS(mol, xc=xc).density_fit()
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    return mf


def run(mol, xc, mode):
    mf = build(mol, xc)
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
    p.add_argument('--min-elems', type=float, default=1e6,
                   help='cast only contractions with at least this many '
                        'input elements; smaller ones are latency-bound and '
                        'gain no time')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mol.nao} AO  '
          f'min_elems={args.min_elems:.0e}', flush=True)

    precision.set_precision('fp64')
    nu_ref, zpe_ref, s_ref, t_ref = run(mol, args.xc, 'fp64')
    print(f'  fp64 reference        {t_ref:7.2f}s', flush=True)

    # the shipped lane, unmodified: the baseline this change must be judged
    # against, not the fp64 reference
    nu_lane, zpe_l, s_l, t_lane = run(mol, args.xc, 'auto')
    print(f'  shipped lane          {t_lane:7.2f}s  '
          f'max|dnu|={np.abs(nu_lane-nu_ref).max():.4f} cm^-1', flush=True)

    patch(args.min_elems)
    nu_new, zpe_n, s_n, t_new = run(mol, args.xc, 'auto')
    print(f'  + fp32 JK contract    {t_new:7.2f}s  '
          f'({stats["cast"]} cast, {stats["kept"]} kept fp64)', flush=True)
    print('    (timing meaningless here: the probe adds two casts per call)',
          flush=True)

    def report(tag, nu, zpe, s):
        print(f'    {tag:20s} max|dnu| {np.abs(nu-nu_ref).max():9.4f}  '
              f'rms {np.sqrt(((nu-nu_ref)**2).mean()):9.4f}  '
              f'dZPE {abs(zpe-zpe_ref)*627.5095:9.2e} kcal/mol  '
              f'dSvib {abs(s-s_ref)*627.5095*1000:9.2e} cal/mol/K',
              flush=True)

    print(flush=True)
    report('shipped lane', nu_lane, zpe_l, s_l)
    report('+ fp32 contract', nu_new, zpe_n, s_n)
    print(f'\n  vs shipped lane: max|dnu| moves '
          f'{np.abs(nu_lane-nu_ref).max():.4f} -> '
          f'{np.abs(nu_new-nu_ref).max():.4f} cm^-1', flush=True)
    print('  reference points: sentinel detection threshold 0.0064, '
          'known-bad change 0.0374 cm^-1', flush=True)


if __name__ == '__main__':
    main()

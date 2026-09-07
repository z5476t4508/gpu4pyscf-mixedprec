"""Step 6F: can the CPHF's response density take float32?

eval_rho4 is 33.8s of the 51.3s a r2SCAN/def2-SVP CPHF now spends, and it is
the last float64 piece there.  It was left alone on purpose: rho1 is a
*response* density, and the argument that licenses the float32 cast in
_eval_rho2 is a measured ground-state cancellation ratio
(sum|term|/|result| ~1.6) which says nothing about this one.

This captures real (ao, mo0, mo1) from a Hessian CPHF and measures, on that
data:

  - the cancellation ratio of the orbital sum, per grid point
  - the relative error float32 actually produces
  - what the contraction costs in each precision

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6f_response_rho_fp32.py [--xc r2scan]
"""
import argparse
import time

import cupy
import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.lib.cupy_helper import contract

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='r2scan')
    p.add_argument('--blocks', type=int, default=3,
                   help='how many grid blocks to capture')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    print(f'{mol.natm} atoms, {mol.nao} AOs, {args.xc}/{args.basis}', flush=True)

    captured = []
    orig = numint.eval_rho4

    def capture(mol_, ao, mo0, mo1, **kw):
        if len(captured) < args.blocks:
            captured.append((ao.copy(), mo0.copy(), mo1.copy(), kw.get('xctype')))
        return orig(mol_, ao, mo0, mo1, **kw)

    numint.eval_rho4 = capture
    try:
        # a Hessian run reaches the CPHF; stop as soon as we have the blocks
        mf.Hessian().kernel()
    finally:
        numint.eval_rho4 = orig

    print(f'captured {len(captured)} blocks\n', flush=True)

    for k, (ao, mo0, mo1, xctype) in enumerate(captured):
        nd = ao.shape[0] if ao.ndim == 3 else 1
        na, nao_sub, nocc = mo1.shape
        ng = ao.shape[-1]
        print(f'block {k}: xctype={xctype} ao{tuple(ao.shape)} '
              f'mo1{tuple(mo1.shape)}  (nd={nd}, nocc={nocc}, ngrids={ng})')

        # the orbital sum that builds the LDA component of the response density
        c0 = contract('nig,io->nog', ao, mo0)
        take = min(na, 8)
        ratios, errors = [], []
        for i in range(take):
            c_1 = contract('nig,io->nog', ao, mo1[i])
            term = c0[0] * c_1[0]                 # (nocc, ngrids)
            result = term.sum(axis=0)
            absum = cupy.abs(term).sum(axis=0)
            nz = cupy.abs(result) > cupy.abs(result).max() * 1e-6
            if nz.sum() > 0:
                ratios.append(float((absum[nz] / cupy.abs(result[nz])).max()))

            c_1_32 = contract('nig,io->nog', ao.astype(cupy.float32),
                              mo1[i].astype(cupy.float32))
            r32 = (c0[0].astype(cupy.float32) * c_1_32[0]).sum(axis=0)
            denom = cupy.abs(result).max()
            errors.append(float(cupy.abs(r32.astype(cupy.float64) - result).max()
                                / denom))

        print(f'  cancellation sum|term|/|result|:  median '
              f'{np.median(ratios):.1f}   max {max(ratios):.1f}')
        print(f'  float32 relative error:           median '
              f'{np.median(errors):.2e}   max {max(errors):.2e}')

        # cost of the dominant contraction in each precision
        ao32 = ao.astype(cupy.float32)
        mo1_32 = mo1.astype(cupy.float32)
        for label, a, m in (('fp64', ao, mo1), ('fp32', ao32, mo1_32)):
            cupy.cuda.runtime.deviceSynchronize()
            t = time.time()
            for i in range(take):
                contract('nig,io->nog', a, m[i])
            cupy.cuda.runtime.deviceSynchronize()
            dt = (time.time() - t) / take
            print(f'  contract nig,io->nog {label}: {dt * 1e3:7.3f} ms/rhs')
        print(flush=True)

    print('For reference: _eval_rho2 casts the ground-state density on a '
          'measured ratio of ~1.6, and the fp32 CPHF Fock side already in '
          'place moves the frequencies 0.000 cm^-1.')


if __name__ == '__main__':
    main()

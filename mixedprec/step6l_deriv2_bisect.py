"""Step 6L: bisect which float32 buffer in _get_vxc_deriv2_task costs accuracy.

The first port of that function ran r2SCAN's _get_exc_deriv2 in 16.7s instead of
32.5s (Tamoxifen total 74.1s -> 58.2s, 5.15x vs fp64) but moved the frequencies
by 5.579 cm^-1 rms 0.475 -- 55x over the 0.1 cm^-1 bound. Fast and wrong.

Note what did NOT catch it: the water-molecule unit tests, all 8 of which pass,
including the one that bounds frequencies at 0.1 cm^-1. Earlier in this work I
argued that Hessian *elements* on water are a misleading proxy for frequencies
on a real molecule. The same limit applies one level up: water *frequencies* are
not a sufficient proxy for Tamoxifen frequencies either. Any variant that looks
good here still has to clear the real molecule.

Suspects, in order of prior probability:
  1. ipip / vmat_dm accumulated in float32. ipip is (3,3,nao,nao) summed with
     beta=1 over every grid block and carries the result -- exactly the layer
     that commit 0913d2c's rule says must stay float64 ("an accumulator's
     precision only matters where its consumer preserves it"; here it does).
  2. ao1 in float32, upcast for the pinned eval_rho2. Flagged as an untested
     deviation from step6i when it was written: the upcast keeps the sum over
     nao in float64 but cannot undo ao1's own error.
  3. Something else, in which case this script says so rather than confirming
     a story I already like.

The float64 reference is cached to disk, so each variant costs one Hessian
(~60s) instead of one Hessian plus a 300s reference.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6l_deriv2_bisect.py --label "ipip fp64"
"""
import argparse
import os
import time

import numpy as np
import pyscf

from gpu4pyscf import dft

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
CACHE = '/tmp/step6l_ref_{xc}_{basis}.npz'
HARTREE2WAVENUMBER = 219474.6313632


def frequencies(mol, hess):
    mass = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -.5
    n = 3 * mol.natm
    h = hess.transpose(0, 2, 1, 3).reshape(n, n) * mass[:, None] * mass[None, :]
    ev = np.linalg.eigvalsh(h) / 1822.888486209
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='r2scan')
    p.add_argument('--label', default='variant')
    p.add_argument('--refresh-ref', action='store_true')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()

    cache = CACHE.format(xc=args.xc, basis=args.basis)
    if os.path.exists(cache) and not args.refresh_ref:
        d = np.load(cache)
        ref, t64 = d['ref'], float(d['t64'])
        print(f'{args.xc}/{args.basis}: float64 {t64:.1f}s (cached)', flush=True)
    else:
        t = time.time()
        ref = mf.Hessian().kernel()
        t64 = time.time() - t
        ref = np.asarray(getattr(ref, 'get', lambda: ref)())
        np.savez(cache, ref=ref, t64=t64)
        print(f'{args.xc}/{args.basis}: float64 {t64:.1f}s (computed, cached)',
              flush=True)

    mf.precision_mode = 'auto'
    try:
        t = time.time()
        hess = mf.Hessian().kernel()
        dt = time.time() - t
    finally:
        mf.precision_mode = None
    hess = np.asarray(getattr(hess, 'get', lambda: hess)())

    vib = slice(6, None)
    nu = frequencies(mol, hess)[vib]
    nu_ref = frequencies(mol, ref)[vib]
    print(f'  {args.label:24s} {dt:6.1f}s {t64 / dt:5.2f}x  '
          f'max|dH|={np.abs(hess - ref).max():.2e}  '
          f'max|dnu|={np.abs(nu - nu_ref).max():.3f}  '
          f'rms={np.sqrt(((nu - nu_ref) ** 2).mean()):.3f} cm^-1', flush=True)


if __name__ == '__main__':
    main()

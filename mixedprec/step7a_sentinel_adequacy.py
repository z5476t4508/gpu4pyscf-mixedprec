"""Step 7A: which molecule is cheap enough to run yet still detects a regression?

The float32 _get_vxc_deriv2_task port passed all 8 water unit tests -- including
the one that bounds frequencies at 0.1 cm^-1 -- while moving Tamoxifen by
5.579 cm^-1. Water cannot detect that class of error for a structural reason:
frequencies are a square root of the Hessian, so an error propagates as

    dw ~ dH / (2 * mu * w)

divided by the mode frequency. Water's lowest vibration is ~1600 cm^-1, so dH is
divided by a large number; a molecule with 20-100 cm^-1 torsions divides by
almost nothing. Water is also 24 basis functions on a small grid, so a float32
reduction barely accumulates. It is the least sensitive case on all three axes
at once.

So a sentinel molecule has to be *demonstrated* adequate, not assumed. The
oracle used here is a known-bad change with a known magnitude: bracketing
_get_exc_deriv2 in float32 without porting it, which moves Tamoxifen by
0.181 cm^-1 (r2SCAN) / 0.118 (B3LYP) and water by 0.001. That is the mildest
regression measured so far, so a molecule that catches it is sensitive enough
for the coarser ones too.

A candidate is adequate if max|dnu| under the oracle clears the 0.1 cm^-1 bound
the unit test uses. The cheapest adequate candidate is the one worth adding to
the test suite; the report also prints the lowest frequency and nao so the
result can be read against the theory rather than just accepted.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step7a_sentinel_adequacy.py [--xc r2scan] [--mols water,vitc]
"""
import argparse
import time

import cupy
import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.hessian import rks as rks_hess
from gpu4pyscf.lib import precision
from mixedprec.vibanalysis import frequencies

TESTS = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/'

WATER = '''O  0.0000000000  -0.0000000000   0.1174000000
H -0.7570000000  -0.0000000000  -0.4696000000
H  0.7570000000   0.0000000000  -0.4696000000'''

# Real geometries already in the repo, so no hand-built coordinates (a bug
# source of their own). Each is here for a different axis of the theory.
CANDIDATES = {
    # control: known NOT to detect the port. Included so a run that shows it
    # detecting something means the harness is wrong, not that water is fine.
    'water': dict(atom=WATER, basis='def2-svp',
                  why='control -- lowest mode ~1600 cm^-1, 24 AO'),
    'vitc': dict(atom=TESTS + '020_Vitamin_C.xyz', basis='def2-svp',
                 why='20 atoms, OH/COH rotors -> soft modes, cheap'),
    'r14': dict(atom=TESTS + 'r14.xyz', basis='def2-svp', charge=1,
                why='39 atoms with Ru: density dynamic range + a cation'),
    'tamoxifen': dict(atom=TESTS + '057_Tamoxifen.xyz', basis='def2-svp',
                      why='reference point -- known 0.181 cm^-1 under the oracle'),
}


def run(mf, oracle):
    '''one Hessian; `oracle` brackets _get_exc_deriv2 in float32 unported'''
    orig = rks_hess._get_exc_deriv2

    def bracketed(*a, **kw):
        with precision.fp32():
            return orig(*a, **kw)

    if oracle:
        rks_hess._get_exc_deriv2 = bracketed
    try:
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        h = mf.Hessian().kernel()
        cupy.cuda.runtime.deviceSynchronize()
        return np.asarray(getattr(h, 'get', lambda: h)()), time.time() - t0
    finally:
        rks_hess._get_exc_deriv2 = orig


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xc', default='r2scan')
    p.add_argument('--basis', default=None, help='override each candidate')
    p.add_argument('--mols', default='water,vitc')
    p.add_argument('--bound', type=float, default=0.1)
    args = p.parse_args()

    for name in args.mols.split(','):
        spec = dict(CANDIDATES[name])
        why = spec.pop('why')
        if args.basis:
            spec['basis'] = args.basis
        mol = pyscf.M(verbose=0, **spec)

        mf = dft.RKS(mol, xc=args.xc).density_fit()
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()

        ref, t64 = run(mf, oracle=False)
        hess, t32 = run(mf, oracle=True)

        nu_ref = frequencies(mol, ref)
        nu = frequencies(mol, hess)
        dnu = np.abs(nu - nu_ref).max()
        soft = np.abs(nu_ref).min()
        verdict = 'DETECTS' if dnu > args.bound else 'blind'
        print(f'{name:10s} {mol.natm:3d} atoms {mol.nao:4d} AO  '
              f'min|nu|={soft:7.1f} n_imag={int((nu_ref < 0).sum()):3d}  '
              f'{t64:6.1f}s/{t32:6.1f}s  '
              f'max|dnu|={dnu:7.3f}  rms={np.sqrt(((nu-nu_ref)**2).mean()):6.3f}  '
              f'-> {verdict}', flush=True)
        print(f'           ({why})', flush=True)


if __name__ == '__main__':
    main()

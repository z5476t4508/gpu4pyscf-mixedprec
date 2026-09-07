"""Step 6G: does widening the float32 bracket to make_h1 cost frequencies?

The 0.131 / 0.181 cm^-1 measured earlier came from a bracket covering all three
Hessian phases at once, so it never identified which phase was responsible.
make_h1 is now two thirds of a r2SCAN Hessian (133.5s of 179s, almost all of it
_d1_dot_ inside _get_vxc_deriv1_task), and reaching it means widening the
bracket -- which drags numint._eval_rho2 in, since that function casts itself.

This widens to make_h1 alone with nothing inside it ported, separating the cost
of the widening from the gain of the port that would follow:

  0.000 cm^-1  -> the earlier damage was partial_hess_elec's; widening is free
  ~0.1 cm^-1   -> _eval_rho2 in make_h1 is the culprit and needs handling first

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6g_widen_bracket.py [--xc r2scan,PBE]
"""
import argparse
import time

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.hessian.rhf import HessianBase
from gpu4pyscf.lib import precision

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
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
    p.add_argument('--xc', default='r2scan,PBE')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    orig_make_h1 = HessianBase.make_h1

    for xc in args.xc.split(','):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()

        t = time.time()
        ref = mf.Hessian().kernel()
        t64 = time.time() - t

        # the shipped lane: the bracket covers solve_mo1 only
        mf.precision_mode = 'auto'
        t = time.time()
        h_now = mf.Hessian().kernel()
        t_now = time.time() - t

        # widened: make_h1 runs in float32 too, but nothing in it is ported
        def wide(self, *a, **kw):
            with precision.fp32():
                return orig_make_h1(self, *a, **kw)

        HessianBase.make_h1 = wide
        try:
            t = time.time()
            h_wide = mf.Hessian().kernel()
            t_wide = time.time() - t
        finally:
            HessianBase.make_h1 = orig_make_h1
            mf.precision_mode = None

        nu_ref = frequencies(mol, ref)
        vib = slice(6, None)
        print(f'{xc}/{args.basis}: float64 reference {t64:.1f}s', flush=True)
        for label, hess, dt in (('solve_mo1 only', h_now, t_now),
                                ('+ make_h1', h_wide, t_wide)):
            nu = frequencies(mol, hess)
            print(f'  {label:16s} {dt:6.1f}s  {t64 / dt:5.2f}x  '
                  f'max|dH|={np.abs(hess - ref).max():.2e}  '
                  f'max|dnu|={np.abs(nu[vib] - nu_ref[vib]).max():.3f}  '
                  f'rms={np.sqrt(((nu[vib] - nu_ref[vib]) ** 2).mean()):.3f} cm^-1',
                  flush=True)


if __name__ == '__main__':
    main()

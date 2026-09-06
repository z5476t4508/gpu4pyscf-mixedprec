"""Step 6B: how wide should the Hessian's float32 bracket be?

Bracketing all of hess_elec in float32 wins 3.38x on DF-RHF but measured
1.01x on PBE while moving its frequencies 0.131 cm^-1 -- for a pure functional
the CPHF has no exact-exchange K, so the float32 J/K branch never fires and
the only thing the global flag reaches is the XC grid: cost without benefit.

This compares three bracket widths on the same molecule:

    fp64       no float32 anywhere (reference)
    solve_mo1  float32 only around the CPHF solve
    hess_elec  float32 around the whole electronic Hessian (what ships)

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6b_hessian_bracket.py [--xc PBE] [--basis def2-svp]
"""
import argparse
import time

import numpy as np
import pyscf

from gpu4pyscf import scf
from gpu4pyscf.lib import precision
from gpu4pyscf.hessian.rhf import HessianBase

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
    p.add_argument('--xc', default=None)
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    if args.xc:
        from gpu4pyscf import dft
        mf = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
    else:
        mf = scf.RHF(mol).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    print(f'{mol.natm} atoms, {mol.nao} AOs, {args.xc or "RHF"}/{args.basis}',
          flush=True)

    orig_solve_mo1 = HessianBase.solve_mo1

    def run(label, bracket):
        mf.precision_mode = 'auto' if bracket == 'hess_elec' else None
        if bracket == 'solve_mo1':
            def wrapped(self, *a, **kw):
                with precision.fp32():
                    return orig_solve_mo1(self, *a, **kw)
            HessianBase.solve_mo1 = wrapped
        try:
            t0 = time.time()
            hess = mf.Hessian().kernel()
            return label, time.time() - t0, hess
        finally:
            HessianBase.solve_mo1 = orig_solve_mo1
            mf.precision_mode = None

    _, t_ref, ref = run('fp64', 'none')
    nu_ref = frequencies(mol, ref)
    vib = slice(6, None)
    print(f'{"fp64":10s} {t_ref:7.1f}s   (reference)', flush=True)

    for bracket in ('solve_mo1', 'hess_elec'):
        _, t, hess = run(bracket, bracket)
        nu = frequencies(mol, hess)
        print(f'{bracket:10s} {t:7.1f}s  {t_ref / t:5.2f}x  '
              f'max|dH|={np.abs(hess - ref).max():.2e}  '
              f'max|dnu|={np.abs(nu[vib] - nu_ref[vib]).max():.3f} cm^-1  '
              f'rms={np.sqrt(((nu[vib] - nu_ref[vib]) ** 2).mean()):.3f}',
              flush=True)


if __name__ == '__main__':
    main()

"""r2SCAN-3c Hessian: how much does the float32 CPHF reach a composite method?

r2SCAN-3c is r2scan/def2-mTZVPP + D4 + gCP -- a *pure* meta-GGA, so its CPHF
has no exact-exchange K and takes the J-only branch of the DF J/K build.  That
is the same split that once made pure functionals miss the float32 gradient
kernel entirely, so it needs measuring rather than assuming.

Setup follows gpu4pyscf/drivers/dft_3c_driver.py so the numbers correspond to
what that driver produces.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6c_r2scan3c.py [--xyz ...] [--atom-grid 99,590]
"""
import argparse
import time
from types import MethodType

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_fun

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
HARTREE2WAVENUMBER = 219474.6313632


def frequencies(mol, hess):
    mass = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -.5
    n = 3 * mol.natm
    h = hess.transpose(0, 2, 1, 3).reshape(n, n) * mass[:, None] * mass[None, :]
    ev = np.linalg.eigvalsh(h) / 1822.888486209
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER


def build(xyz, xc_name, auxbasis, atom_grid):
    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc_name)
    mol = pyscf.M(atom=xyz, basis=basis, ecp=ecp, verbose=0)
    mf = dft.KS(mol, xc=pyscf_xc).density_fit(auxbasis=auxbasis)
    mf.grids.atom_grid = atom_grid
    # the 3c corrections are patched onto the instance, as the driver does
    mf.nlc = nlc
    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
    mf.do_disp = lambda: True
    mf.conv_tol = 1e-10
    mf.verbose = 0
    return mol, mf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--xc', default='r2scan3c')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--atom-grid', default='99,590')
    args = p.parse_args()

    atom_grid = tuple(int(x) for x in args.atom_grid.split(','))
    mol, mf = build(args.xyz, args.xc, args.auxbasis, atom_grid)
    print(f'{args.xc}: {mol.natm} atoms, {mol.nao} AOs, grid {atom_grid}',
          flush=True)

    t0 = time.time()
    mf.kernel()
    print(f'SCF {time.time() - t0:.1f}s ({mf.cycles} cycles)', flush=True)

    timings = {}
    cls = type(mf.Hessian())

    def timed(name):
        orig = getattr(cls, name)

        def wrapper(self, *a, **kw):
            t = time.time()
            r = orig(self, *a, **kw)
            timings.setdefault(name, []).append(time.time() - t)
            return r
        setattr(cls, name, wrapper)
        return orig

    originals = {n: timed(n) for n in
                 ('partial_hess_elec', 'make_h1', 'solve_mo1')}
    try:
        results = {}
        for mode in ('fp64', 'auto'):
            mf.precision_mode = None if mode == 'fp64' else 'auto'
            for v in timings.values():
                del v[:]
            t0 = time.time()
            results[mode] = mf.Hessian().kernel()
            t = time.time() - t0
            phases = '  '.join(f'{n}={sum(v):.1f}s'
                               for n, v in sorted(timings.items()))
            print(f'{mode:5s} total {t:7.1f}s   {phases}', flush=True)
            results[mode + '_t'] = t
    finally:
        for n, orig in originals.items():
            setattr(cls, n, orig)
        mf.precision_mode = None

    ref, h32 = results['fp64'], results['auto']
    nu_ref, nu = frequencies(mol, ref), frequencies(mol, h32)
    vib = slice(6, None)
    print(f'\nspeedup {results["fp64_t"] / results["auto_t"]:.2f}x')
    print(f'max |dH|          {np.abs(h32 - ref).max():.3e} Eh/Bohr^2')
    print(f'max |dnu| (vib)   {np.abs(nu[vib] - nu_ref[vib]).max():.3f} cm^-1')
    print(f'rms |dnu| (vib)   '
          f'{np.sqrt(((nu[vib] - nu_ref[vib]) ** 2).mean()):.3f} cm^-1')
    print(f'lowest vib mode   {nu_ref[6]:.2f} -> {nu[6]:.2f} cm^-1')
    print(f'highest vib mode  {nu_ref[-1]:.2f} -> {nu[-1]:.2f} cm^-1')


if __name__ == '__main__':
    main()

"""Step 7B: optimize the sentinel geometry, so the acceptance metrics mean something.

Everything measured so far ran on geometries that are not stationary points at
the level of theory being measured: Vitamin C has 11 imaginary modes at
r2SCAN/def2-SVP, Tamoxifen 9. Three consequences, all of which cost accuracy of
interpretation rather than accuracy of the code:

  - translations and rotations are not exact zero modes, so projecting them out
    is only approximate and the "rotations" carry gradient contamination (that
    is what put a spurious 15.24 cm^-1 mode into the Tamoxifen comparison);
  - ZPE and S_vib have to drop the imaginary modes, which makes their absolute
    values meaningless -- only the difference between two lanes on the same
    geometry is usable;
  - the soft-mode axis was never actually sampled. Vitamin C's softest real
    mode is 57.9 cm^-1 at the saddle; at a true minimum a molecule with OH and
    COH rotors should have genuinely soft torsions, which is the regime where
    dw ~ dH/(2*mu*w) amplifies most.

This optimizes the geometry in strict float64 (an optimization run under the
mixed lane would bake the lane's own error into the reference geometry, which
would then be compared against itself) and writes it out for the sentinel test
to use.

Two things a plain `optimize()` call does not give:

  - geomeTRIC's default thresholds (gmax 4.5e-4) are set for finding a
    structure, not for locating a stationary point well enough to differentiate
    twice. Tightened here by ~50x.
  - a quasi-Newton optimizer converges to a *stationary point*, which may be a
    saddle. Vitamin C's repo geometry is one: optimizing it with the defaults
    reached gmax 1.86e-4 with 7 imaginary modes, the lowest -275 cm^-1, which
    is a real saddle rather than a convergence artifact (ascorbic acid has
    several OH rotors). So this follows the imaginary modes downhill and
    re-optimizes until none are left, reporting honestly if it runs out of
    budget instead of writing out a geometry that is still a saddle.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step7b_optimize_sentinel.py [--xc r2scan] [--basis def2-svp]
"""
import argparse
import os
import time

import numpy as np
import pyscf
from pyscf.geomopt.geometric_solver import optimize
from pyscf.hessian import thermo

from gpu4pyscf import dft
from gpu4pyscf.lib import precision

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')
OUT = os.path.join(os.path.dirname(__file__), 'geometries')
BOHR = 0.52917721092

CONV = dict(convergence_energy=1e-8, convergence_grms=1e-5,
            convergence_gmax=2e-5, convergence_drms=4e-5,
            convergence_dmax=6e-5)


def analyse(mol, hess):
    hess = np.asarray(getattr(hess, 'get', lambda: hess)())
    r = thermo.harmonic_analysis(mol, hess, imaginary_freq=False)
    nu = np.asarray(r['freq_wavenumber']).real
    return nu, np.asarray(r['norm_mode'])


def displaced(mol, nu, modes, scale=0.25):
    '''push downhill along every imaginary mode at once, scaled so the largest
    atomic displacement is `scale` Angstrom'''
    d = modes[nu < 0].sum(axis=0)
    n = np.abs(d).max()
    if n < 1e-12:
        return mol
    new = mol.copy()
    new.set_geom_(mol.atom_coords() * BOHR + d * (scale / n), unit='Angstrom')
    return new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '020_Vitamin_C.xyz'))
    p.add_argument('--name', default='vitamin_c')
    p.add_argument('--xc', default='r2scan')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--maxsteps', type=int, default=300)
    p.add_argument('--cycles', type=int, default=8,
                   help='saddle-escape attempts before giving up')
    args = p.parse_args()

    os.makedirs(OUT, exist_ok=True)
    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)

    # strict float64: the reference geometry must not carry the lane's error
    precision.set_precision('fp64')

    def build_mf(m):
        mf = dft.RKS(m, xc=args.xc).density_fit()
        mf.conv_tol = 1e-11
        mf.verbose = 0
        return mf

    t_start = time.time()
    best = None
    for cycle in range(args.cycles):
        mol = optimize(build_mf(mol), maxsteps=args.maxsteps, **CONV)
        mf = build_mf(mol)
        mf.kernel()
        grad = mf.Gradients().kernel()
        gmax = np.abs(np.asarray(getattr(grad, 'get', lambda: grad)())).max()
        nu, modes = analyse(mol, mf.Hessian().kernel())
        n_imag = int((nu < 0).sum())
        print(f'  cycle {cycle}: gmax={gmax:.2e}  imaginary={n_imag:3d}  '
              f'min|nu|={np.abs(nu).min():7.2f}  lowest={nu.min():8.2f} cm^-1  '
              f'({time.time()-t_start:.0f}s)', flush=True)
        # keep the best *optimized* geometry seen, never the displaced one:
        # the loop below overwrites `mol` with a displacement, so writing `mol`
        # after the loop exhausts would emit an unoptimized structure.
        if best is None or n_imag < best[1]:
            best = (mol, n_imag, gmax, nu)
        if n_imag == 0:
            break
        mol = displaced(mol, nu, modes)
    else:
        print(f'  gave up after {args.cycles} cycles; writing the best '
              f'geometry found ({best[1]} imaginary)', flush=True)

    mol, n_imag, gmax, nu = best

    tag = f'{args.name}_{args.xc}_{args.basis}'.replace(',', '')
    path = os.path.join(OUT, tag + '.xyz')
    coords = mol.atom_coords() * BOHR
    with open(path, 'w') as f:
        f.write(f'{mol.natm}\n')
        f.write(f'{args.name} optimized at {args.xc}/{args.basis} (fp64), '
                f'gmax={gmax:.2e} Eh/Bohr, {n_imag} imaginary modes\n')
        for i in range(mol.natm):
            f.write(f'{mol.atom_symbol(i):2s} {coords[i,0]:18.12f} '
                    f'{coords[i,1]:18.12f} {coords[i,2]:18.12f}\n')

    print(f'{args.name} {args.xc}/{args.basis}: {time.time()-t_start:.0f}s total',
          flush=True)
    print(f'  gmax      = {gmax:.2e} Eh/Bohr', flush=True)
    print(f'  imaginary = {n_imag}', flush=True)
    print(f'  min|nu|   = {np.abs(nu).min():.2f} cm^-1', flush=True)
    print(f'  lowest 6  = {np.round(np.sort(nu)[:6], 2)}', flush=True)
    print(f'  written  -> {path}', flush=True)
    if n_imag:
        print('  NOT A MINIMUM: the sentinel bounds stay difference-only, and '
              'the soft-mode axis is still unsampled.', flush=True)


if __name__ == '__main__':
    main()

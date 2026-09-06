"""Compare two optimized geometries by re-evaluating both in strict float64.

The only meaningful test of a faster optimization lane is whether it lands in
the same place.  Coordinates alone do not answer that -- near a minimum the
surface is flat, so two geometries can differ by more than the optimizer's
displacement threshold and still be the same minimum.  Re-evaluating both with
conv_tol=1e-11 in float64 does answer it.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/compare_geoms.py a_geom.npy b_geom.npy [--basis def2-svp]
"""
import argparse

import numpy as np
import pyscf

from gpu4pyscf import scf

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'


def energy(mol, coords, auxbasis):
    mol = mol.set_geom_(coords, unit='Bohr', inplace=False)
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-11
    mf.verbose = 0
    return mf.kernel()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('geoms', nargs=2)
    p.add_argument('--labels', nargs=2, default=['a', 'b'])
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    coords = [np.load(f) for f in args.geoms]

    energies = []
    for label, c in zip(args.labels, coords):
        e = energy(mol, c, args.auxbasis)
        energies.append(e)
        print(f'{label:8s} {e:.12f}')

    d = abs(energies[0] - energies[1])
    print(f'\nenergy difference   {d:.3e} Eh = {d * 627.5095:.3e} kcal/mol')
    print(f'max |dx|            {np.abs(coords[0] - coords[1]).max():.3e} Bohr')
    print('(geomeTRIC converges on 1e-6 Eh and 1.8e-3 Angstrom of displacement)')


if __name__ == '__main__':
    main()

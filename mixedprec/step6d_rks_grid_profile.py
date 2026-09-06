"""Step 6D: which of the RKS Hessian's grid loops actually cost anything?

hessian/rks.py runs about ten grid loops of its own, none of which cast the AO
values to float32.  Porting all ten is a large job; porting the one or two that
dominate may not be.  This times them individually so the decision rests on
measurement.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6d_rks_grid_profile.py [--xc r2scan] [--basis def2-svp]
"""
import argparse
import time

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.hessian import rks as rks_hess
from gpu4pyscf.hessian import rhf as rhf_hess

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'

# module-level functions worth timing, in the order they run
TARGETS = [
    (rks_hess, '_get_exc_deriv2'),      # XC second derivatives (partial_hess_elec)
    (rks_hess, '_get_vxc_diag'),
    (rks_hess, '_get_vxc_deriv2'),
    (rks_hess, '_get_vxc_deriv1'),      # make_h1
    (rks_hess, '_get_enlc_deriv2'),
    (rhf_hess, '_partial_hess_ejk'),    # the DF J/K second derivatives
    (rhf_hess, '_get_jk_ip1'),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='r2scan')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    mf = dft.RKS(mol, xc=args.xc).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.kernel()
    print(f'{mol.natm} atoms, {mol.nao} AOs, {args.xc}/{args.basis}', flush=True)

    timings = {}
    originals = []

    def wrap(module, name):
        orig = getattr(module, name, None)
        if orig is None:
            return

        def wrapper(*a, **kw):
            t = time.time()
            r = orig(*a, **kw)
            timings[name] = timings.get(name, 0.0) + time.time() - t
            return r
        setattr(module, name, wrapper)
        originals.append((module, name, orig))

    for module, name in TARGETS:
        wrap(module, name)

    # the class methods route through the module functions above, except these
    cls = type(mf.Hessian())
    phase = {}
    phase_orig = {}
    for name in ('partial_hess_elec', 'make_h1', 'solve_mo1'):
        orig = getattr(cls, name)
        phase_orig[name] = orig

        def make(n, o):
            def wrapper(self, *a, **kw):
                t = time.time()
                r = o(self, *a, **kw)
                phase[n] = phase.get(n, 0.0) + time.time() - t
                return r
            return wrapper
        setattr(cls, name, make(name, orig))

    try:
        t0 = time.time()
        mf.Hessian().kernel()
        total = time.time() - t0
    finally:
        for module, name, orig in originals:
            setattr(module, name, orig)
        for name, orig in phase_orig.items():
            setattr(cls, name, orig)

    print(f'\ntotal {total:.1f}s')
    print('  phases:')
    for name in ('partial_hess_elec', 'make_h1', 'solve_mo1'):
        t = phase.get(name, 0.0)
        print(f'    {name:22s} {t:7.1f}s  {t / total * 100:5.1f}%')
    print('  inner functions (grid loops unless noted):')
    for name, t in sorted(timings.items(), key=lambda kv: -kv[1]):
        print(f'    {name:22s} {t:7.1f}s  {t / total * 100:5.1f}%')
    accounted = sum(timings.values())
    print(f'    {"(unattributed)":22s} {total - accounted:7.1f}s  '
          f'{(total - accounted) / total * 100:5.1f}%')


if __name__ == '__main__':
    main()

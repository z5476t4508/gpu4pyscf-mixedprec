"""Step 5A: map conv_tol -> gradient error, at a fixed geometry.

The adaptive-conv_tol schedule needs one number it does not have yet: how much
nuclear gradient error a loosely converged SCF actually costs.  geomeTRIC
converges on gmax < 4.5e-4 Eh/Bohr and |dE| < 1e-6 Eh, so the schedule may
loosen conv_tol only as far as those two stay safe.

For each conv_tol this runs a full SCF from the same starting guess and
compares energy and gradient against a strict fp64 reference (conv_tol=1e-12).
Both lanes are measured: 'auto' (fp32 start, fp64 tail) and 'fp64'.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step5a_convtol_error.py [--basis def2-svp]
"""
import argparse
import json
import time

import numpy as np
import pyscf

from gpu4pyscf import scf
from gpu4pyscf.lib import precision

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
TOLS = [1e-4, 3e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]


def run(mol, auxbasis, conv_tol, mode, count_cycles=None):
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = conv_tol
    mf.precision_mode = mode
    mf.verbose = 0
    n_fp64 = [0]
    if count_cycles is not None:
        def cb(envs):
            if precision.get_precision() == 'fp64':
                n_fp64[0] += 1
        mf.callback = cb
    t0 = time.time()
    e = mf.kernel()
    t_scf = time.time() - t0
    g = mf.nuc_grad_method()
    t0 = time.time()
    de = g.kernel()
    t_grad = time.time() - t0
    return dict(e=float(e), de=np.asarray(de), t_scf=t_scf, t_grad=t_grad,
                cycles=int(mf.cycles), n_fp64=n_fp64[0], converged=bool(mf.converged))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--geom', default=None,
                   help='optional .npy of Bohr coordinates (e.g. a near-minimum geometry)')
    p.add_argument('--out', default='/tmp/step5a_convtol_error.json')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    if args.geom:
        mol.set_geom_(np.load(args.geom), unit='Bohr')
    print(f'{mol.natm} atoms, {mol.nao} AOs, basis={args.basis}', flush=True)

    print('reference: fp64, conv_tol=1e-12, fp64 gradient', flush=True)
    ref = run(mol, args.auxbasis, 1e-12, 'fp64')
    e_ref, g_ref = ref['e'], ref['de']
    print(f"  E={e_ref:.12f}  gmax={np.abs(g_ref).max():.4e}  "
          f"{ref['cycles']} cycles  scf={ref['t_scf']:.2f}s grad={ref['t_grad']:.2f}s\n",
          flush=True)

    rows = []
    for mode in ('auto', 'fp64'):
        for tol in TOLS:
            r = run(mol, args.auxbasis, tol, mode, count_cycles=True)
            dg = np.abs(r['de'] - g_ref).max()
            de_err = abs(r['e'] - e_ref)
            row = dict(mode=mode, conv_tol=tol, e_err=de_err, g_err=float(dg),
                       cycles=r['cycles'], n_fp64=r['n_fp64'],
                       t_scf=r['t_scf'], t_grad=r['t_grad'],
                       converged=r['converged'])
            rows.append(row)
            print(f"{mode:5s} conv_tol={tol:.0e}  dE={de_err:9.2e}  dg_max={dg:9.2e}  "
                  f"{r['cycles']:2d} cyc ({r['n_fp64']} fp64)  scf={r['t_scf']:5.2f}s"
                  f"{'' if r['converged'] else '  NOT CONVERGED'}", flush=True)
        print(flush=True)

    print('geomeTRIC thresholds for reference: gmax 4.5e-4, grms 3.0e-4, dE 1e-6')
    with open(args.out, 'w') as f:
        json.dump(dict(args=vars(args), nao=int(mol.nao), e_ref=e_ref,
                       gmax_ref=float(np.abs(g_ref).max()), rows=rows), f, indent=1)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

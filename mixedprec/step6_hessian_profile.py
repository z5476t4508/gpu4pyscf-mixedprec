"""Step 6: where does a DF-RHF Hessian spend its time?

The Hessian has only picked up float32 through helpers it shares with the
gradient (measured 1.15x overall), and there is no precision policy in
gpu4pyscf/hessian at all.  Before porting anything, split the run into the
three phases the code already delimits:

  partial_hess_elec  second-derivative integrals (int2e_ipip1 etc.)
  make_h1            first-derivative Fock matrices in the MO basis
  solve_mo1          the CPHF/CPSCF solve -- iterative, one J/K build per
                     iteration over 3*natm right-hand sides

Only a phase that is both large and float64 is worth porting.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6_hessian_profile.py [--basis def2-svp] [--mode fp64]
"""
import argparse
import json
import time

import numpy as np
import pyscf

from gpu4pyscf import scf

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--mode', default='fp64', help="mean-field precision_mode")
    p.add_argument('--out', default='/tmp/step6_hessian.json')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    print(f'{mol.natm} atoms, {mol.nao} AOs, basis={args.basis}, '
          f'mode={args.mode}', flush=True)

    mf = scf.RHF(mol).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = 1e-10
    mf.precision_mode = args.mode
    mf.verbose = 0
    t0 = time.time()
    mf.kernel()
    t_scf = time.time() - t0
    print(f'SCF {t_scf:.2f}s ({mf.cycles} cycles)', flush=True)

    h = mf.Hessian()
    h.verbose = 0
    cls = type(h)

    timings = {}
    n_cphf = [0]

    def timed(name):
        orig = getattr(cls, name)

        def wrapper(self, *a, **kw):
            t = time.time()
            r = orig(self, *a, **kw)
            timings[name] = timings.get(name, 0.0) + time.time() - t
            return r
        setattr(cls, name, wrapper)
        return orig

    originals = {n: timed(n) for n in
                 ('partial_hess_elec', 'make_h1', 'solve_mo1')}

    # count CPHF iterations: gen_vind returns the callable the solver applies
    orig_gen_vind = cls.gen_vind

    def counting_gen_vind(self, *a, **kw):
        fx = orig_gen_vind(self, *a, **kw)

        def counted(*fa, **fkw):
            n_cphf[0] += 1
            return fx(*fa, **fkw)
        return counted
    cls.gen_vind = counting_gen_vind

    try:
        t0 = time.time()
        hess = h.kernel()
        t_hess = time.time() - t0
    finally:
        for n, orig in originals.items():
            setattr(cls, n, orig)
        cls.gen_vind = orig_gen_vind

    accounted = sum(timings.values())
    print(f'\nHessian total {t_hess:.2f}s')
    for name in ('partial_hess_elec', 'make_h1', 'solve_mo1'):
        t = timings.get(name, 0.0)
        print(f'  {name:20s} {t:7.2f}s  {t / t_hess * 100:5.1f}%')
    print(f'  {"(rest)":20s} {t_hess - accounted:7.2f}s  '
          f'{(t_hess - accounted) / t_hess * 100:5.1f}%')
    print(f'  CPHF applications of fx: {n_cphf[0]}')

    freq = np.linalg.eigvalsh(
        hess.transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm))
    print(f'  hessian eigenvalue range: {freq.min():.6f} .. {freq.max():.6f}')

    with open(args.out, 'w') as f:
        json.dump(dict(args=vars(args), nao=int(mol.nao), t_scf=t_scf,
                       t_hess=t_hess, phases=timings, n_cphf=n_cphf[0],
                       eig=[float(freq.min()), float(freq.max())]), f, indent=1)
    print(f'wrote {args.out}')
    np.save(args.out.replace('.json', '_hess.npy'), hess)


if __name__ == '__main__':
    main()

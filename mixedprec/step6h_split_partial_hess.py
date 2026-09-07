"""Step 6H: split partial_hess_elec in two and bracket each half separately.

partial_hess_elec is the last big phase of a r2SCAN Hessian (~47s of 74.1s) and
was left outside the float32 lane because widening the bracket over it cost
0.131-0.181 cm^-1. But that bracket covered the whole phase at once, so it never
said which half was responsible. The DF phase has two:

  _jk_energy_per_atom + _hcore_energy   the JK/hcore second-derivative integrals
  _get_exc_deriv2                       the XC second derivatives on the grid

Reading the code predicts the answer: nothing under _jk_energy_per_atom consults
precision.get_precision(), so bracketing it should be a no-op in both time and
accuracy, and the whole 0.131-0.181 cm^-1 should reappear under _get_exc_deriv2
alone. This measures that rather than trusting it -- a no-op that turns out to
cost frequencies would mean a self-casting function I have not accounted for.

It also times each half under fp64, which says how much is even on the table:
a phase that is 5s of 47s is not worth a kernel port however safe it is.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6h_split_partial_hess.py [--xc r2scan,B3LYP]
"""
import argparse
import time

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.df.hessian import rhf as df_rhf_hess
from gpu4pyscf.hessian import rks as rks_hess
from gpu4pyscf.lib import precision

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
HARTREE2WAVENUMBER = 219474.6313632


def frequencies(mol, hess):
    mass = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -.5
    n = 3 * mol.natm
    h = hess.transpose(0, 2, 1, 3).reshape(n, n) * mass[:, None] * mass[None, :]
    ev = np.linalg.eigvalsh(h) / 1822.888486209
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER


class Timer:
    '''wall time and call count for one function, summed over a Hessian'''

    def __init__(self):
        self.t = 0.
        self.n = 0

    def wrap(self, fn, fp32=False):
        import cupy
        def inner(*a, **kw):
            cupy.cuda.runtime.deviceSynchronize()
            t0 = time.time()
            try:
                if fp32:
                    with precision.fp32():
                        return fn(*a, **kw)
                return fn(*a, **kw)
            finally:
                cupy.cuda.runtime.deviceSynchronize()
                self.t += time.time() - t0
                self.n += 1
        return inner


def run(mf, jk_fp32, xc_fp32):
    '''one Hessian with each half optionally bracketed; returns hess and times'''
    orig_jk = df_rhf_hess._jk_energy_per_atom
    orig_xc = rks_hess._get_exc_deriv2
    t_jk, t_xc = Timer(), Timer()
    df_rhf_hess._jk_energy_per_atom = t_jk.wrap(orig_jk, jk_fp32)
    rks_hess._get_exc_deriv2 = t_xc.wrap(orig_xc, xc_fp32)
    try:
        t0 = time.time()
        hess = mf.Hessian().kernel()
        return hess, time.time() - t0, t_jk, t_xc
    finally:
        df_rhf_hess._jk_energy_per_atom = orig_jk
        rks_hess._get_exc_deriv2 = orig_xc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='r2scan,B3LYP')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)

    for xc in args.xc.split(','):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()

        ref, t64, jk64, xc64 = run(mf, False, False)
        nu_ref = frequencies(mol, ref)[6:]
        print(f'{xc}/{args.basis}: float64 {t64:.1f}s '
              f'(_jk_energy_per_atom {jk64.t:.1f}s x{jk64.n}, '
              f'_get_exc_deriv2 {xc64.t:.1f}s x{xc64.n})', flush=True)

        # the shipped lane, then each half of partial_hess_elec added to it
        mf.precision_mode = 'auto'
        try:
            for label, jk32, xc32 in (('lane as shipped', False, False),
                                      ('+ JK/hcore half', True, False),
                                      ('+ XC half', False, True),
                                      ('+ both', True, True)):
                hess, dt, tj, tx = run(mf, jk32, xc32)
                nu = frequencies(mol, hess)[6:]
                print(f'  {label:17s} {dt:6.1f}s {t64 / dt:5.2f}x  '
                      f'jk={tj.t:5.1f}s xc={tx.t:5.1f}s  '
                      f'max|dH|={np.abs(hess - ref).max():.2e}  '
                      f'max|dnu|={np.abs(nu - nu_ref).max():.3f}  '
                      f'rms={np.sqrt(((nu - nu_ref) ** 2).mean()):.3f} cm^-1',
                      flush=True)
        finally:
            mf.precision_mode = None


if __name__ == '__main__':
    main()

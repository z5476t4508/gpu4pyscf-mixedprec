"""Step 6I: is _get_exc_deriv2's 0.181 cm^-1 entirely the rho cast?

Step 6H established that bracketing _get_exc_deriv2 in float32 costs 0.181 cm^-1
and buys 0.2s of 32.6s. Reading the code says why: nothing inside that phase is
ported, so the only float32 that happens is numint._eval_rho2 casting itself --
and the rho it produces is fed to eval_xc_eff at order 2, where fxc amplifies a
density error far more than the order-1 vxc in make_h1 does (which measured
0.001 cm^-1 under the same naive widening).

If that account is right, the damage and the (negligible) gain have the same
single source and can be separated: force _eval_rho2 back to float64 inside the
bracket and 0.181 should collapse to ~0.001 while the time returns to ~32.6s.

  ~0.001 cm^-1  -> the damage source is removable; porting _get_vxc_deriv2_task
                   the way _get_vxc_deriv1_task was ported (heavy buffers in
                   float32, rho kept float64 for libxc) is worth doing, and
                   32.6s is 44% of the current 74.1s r2SCAN Hessian.
  still ~0.18    -> the attribution is wrong, something else in the phase casts,
                   and 32.6s should be abandoned rather than ported blind.

The second outcome is the useful one to be able to see: it would mean the same
mistake as the numint._nr_rks_fxc_task port, caught before writing the code
instead of after.

Note: _eval_rho2 is forced via a process-global context, and _get_vxc_deriv2_task
may run on a worker thread. With one GPU the phases are sequential, so this is
sound here; it would need rethinking on a multi-device box.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6i_rho_cast_attribution.py [--xc r2scan]
"""
import argparse
import time

import numpy as np
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.dft import numint
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


def run(mf, xc_fp32, rho_fp64):
    '''one Hessian; optionally bracket _get_exc_deriv2 and pin _eval_rho2'''
    import cupy
    orig_xc = rks_hess._get_exc_deriv2
    orig_rho2 = numint._eval_rho2

    def pinned_rho2(*a, **kw):
        with precision.fp64():
            return orig_rho2(*a, **kw)

    def wrapped_xc(*a, **kw):
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        try:
            if rho_fp64:
                numint._eval_rho2 = pinned_rho2
            if xc_fp32:
                with precision.fp32():
                    return orig_xc(*a, **kw)
            return orig_xc(*a, **kw)
        finally:
            numint._eval_rho2 = orig_rho2
            cupy.cuda.runtime.deviceSynchronize()
            wrapped_xc.t += time.time() - t0

    wrapped_xc.t = 0.
    rks_hess._get_exc_deriv2 = wrapped_xc
    try:
        t0 = time.time()
        hess = mf.Hessian().kernel()
        return hess, time.time() - t0, wrapped_xc.t
    finally:
        rks_hess._get_exc_deriv2 = orig_xc
        numint._eval_rho2 = orig_rho2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--xc', default='r2scan')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)

    for xc in args.xc.split(','):
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.kernel()

        ref, t64, xc64 = run(mf, False, False)
        nu_ref = frequencies(mol, ref)[6:]
        print(f'{xc}/{args.basis}: float64 {t64:.1f}s (_get_exc_deriv2 {xc64:.1f}s)',
              flush=True)

        mf.precision_mode = 'auto'
        try:
            for label, xc32, rho64 in (
                    ('lane as shipped', False, False),
                    ('+ XC half (rho fp32)', True, False),
                    ('+ XC half, rho pinned fp64', True, True)):
                hess, dt, txc = run(mf, xc32, rho64)
                nu = frequencies(mol, hess)[6:]
                print(f'  {label:27s} {dt:6.1f}s  xc={txc:5.1f}s  '
                      f'max|dH|={np.abs(hess - ref).max():.2e}  '
                      f'max|dnu|={np.abs(nu - nu_ref).max():.3f}  '
                      f'rms={np.sqrt(((nu - nu_ref) ** 2).mean()):.3f} cm^-1',
                      flush=True)
        finally:
            mf.precision_mode = None


if __name__ == '__main__':
    main()

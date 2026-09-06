"""Step 6E: what actually costs time inside the CPHF's XC response?

Two predictions about where a DFT Hessian spends its time have now been wrong,
both made by subtracting one timer from another rather than measuring the thing
itself.  This times the individual pieces directly:

  block_loop         AO evaluation on the grid
  eval_rho4          density side of the response kernel (nset right-hand sides)
  _scale_ao          Fock side
  _contract_rho1_fxc weight contraction
  _d1_dot_           the accumulations in hessian/rks.py's own loops
  eval_rho2          density side of _get_vxc_deriv1

Counts are reported alongside times, because a function that is called a
million times cheaply needs a different fix from one called twice expensively.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step6e_inner_profile.py [--xc r2scan] [--phase solve_mo1]
"""
import argparse
import time

import cupy
import pyscf

from gpu4pyscf import dft
from gpu4pyscf.dft import numint
from gpu4pyscf.grad import rks as rks_grad

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'

TARGETS = [
    (numint, 'eval_rho4'),
    (numint, 'eval_rho2'),
    (numint, '_scale_ao'),
    (numint, '_contract_rho1_fxc'),
    (numint, '_tau_dot'),
    (rks_grad, '_d1_dot_'),
    (rks_grad, '_gga_grad_sum_'),
    (rks_grad, '_tau_grad_dot_'),
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

    stats = {}
    originals = []
    current_phase = ['(outside)']

    def wrap(module, name):
        orig = getattr(module, name, None)
        if orig is None:
            print(f'  (missing: {name})')
            return

        def wrapper(*a, **kw):
            cupy.cuda.runtime.deviceSynchronize()
            t = time.time()
            r = orig(*a, **kw)
            cupy.cuda.runtime.deviceSynchronize()
            # attribute to the phase that is running: a bare per-function total
            # cannot tell make_h1's calls from the CPHF's, and guessing at that
            # split is what sent the last two attempts after the wrong target
            s = stats.setdefault((current_phase[0], name), [0.0, 0])
            s[0] += time.time() - t
            s[1] += 1
            return r
        setattr(module, name, wrapper)
        originals.append((module, name, orig))

    for module, name in TARGETS:
        wrap(module, name)

    # phase boundaries so the inner numbers can be attributed
    cls = type(mf.Hessian())
    phase = {}
    phase_orig = {}
    for name in ('partial_hess_elec', 'make_h1', 'solve_mo1'):
        orig = getattr(cls, name)
        phase_orig[name] = orig

        def make(n, o):
            def wrapper(self, *a, **kw):
                cupy.cuda.runtime.deviceSynchronize()
                t = time.time()
                outer, current_phase[0] = current_phase[0], n
                try:
                    r = o(self, *a, **kw)
                finally:
                    current_phase[0] = outer
                cupy.cuda.runtime.deviceSynchronize()
                phase[n] = phase.get(n, 0.0) + time.time() - t
                return r
            return wrapper
        setattr(cls, name, make(name, orig))

    def run(label, mode):
        mf.precision_mode = mode
        stats.clear()
        phase.clear()
        cupy.cuda.runtime.deviceSynchronize()
        t0 = time.time()
        mf.Hessian().kernel()
        total = time.time() - t0
        print(f'\n=== {label} ===   total {total:.1f}s   (timers synchronise '
              f'the device, so this is slower than an untimed run)')
        for name in ('partial_hess_elec', 'make_h1', 'solve_mo1'):
            t = phase.get(name, 0.0)
            print(f'  {name:22s} {t:7.1f}s  {t / total * 100:5.1f}%')
            inner = [(k[1], v) for k, v in stats.items() if k[0] == name]
            for fn, (ft, n) in sorted(inner, key=lambda kv: -kv[1][0]):
                if ft < 0.5:
                    continue
                print(f'      {fn:20s} {ft:7.1f}s  {n:8d} calls  '
                      f'{ft / n * 1e3:7.3f} ms/call')
        return {k: v[0] for k, v in stats.items()}

    try:
        base = run('fp64', None)
        fast = run('auto (fp32)', 'auto')
    finally:
        for module, name, orig in originals:
            setattr(module, name, orig)
        for name, orig in phase_orig.items():
            setattr(cls, name, orig)
        mf.precision_mode = None

    print('\n=== what float32 actually changed (per phase x function) ===')
    keys = sorted(set(base) | set(fast), key=lambda k: -(base.get(k, 0)))
    for k in keys:
        b, f = base.get(k, 0.0), fast.get(k, 0.0)
        if max(b, f) < 0.5:
            continue
        ratio = f'{b / f:5.2f}x' if f > 0 else '   n/a'
        print(f'  {k[0]:20s} {k[1]:20s} {b:7.1f}s -> {f:7.1f}s  {ratio}')


if __name__ == '__main__':
    main()

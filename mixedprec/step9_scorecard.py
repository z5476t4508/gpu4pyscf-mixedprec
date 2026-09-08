"""Step 9: the scorecard -- what the mixed-precision lane is actually worth.

Every number reported so far has been a *phase* speedup measured against this
same code in float64: gradient 6.80x, Hessian 4.05x, SCF 1.38x. Two things are
wrong with reporting only those.

  1. A user does not run a phase, they run a calculation. The phase numbers do
     not combine the way they look like they should -- the gradient call is
     6.80x but an end-to-end geometry optimization is 1.55-1.88x, because the
     per-step CDERI rebuild and the float64 SCF tail are untouched by any of
     this.
  2. "x times faster than our own float64" is a self-comparison. It says
     nothing about whether the float64 baseline was any good, and it cannot
     catch an error that both lanes share. Direwolf's VALIDATION.md is the
     counter-example worth copying: every claim there is a number against a
     *named external code*.

So this script reports each quantity three ways -- against CPU PySCF (a
different implementation, hence a real oracle for both speed and accuracy),
against this code in float64 (what the precision work actually bought), and in
absolute wall-clock -- and it reports the accuracy against CPU alongside, so a
speedup can never be quoted without the error it cost.

CPU thread count is recorded in the output: "GPU vs CPU" is meaningless without
it, and the honest comparison uses every core the machine has.

The CPU Hessian is O(hours) at 537 AO and is gated behind --cpu-hessian.
Without it the Hessian row still reports GPU fp64 vs GPU auto.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step9_scorecard.py [--xc hf] [--basis def2-svp] \
        [--xyz ...] [--skip-hessian] [--cpu-hessian]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

TESTS = os.path.join(os.path.dirname(__file__), '..', 'gpu4pyscf', 'tests')


def timed(fn):
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    r = fn()
    cp.cuda.runtime.deviceSynchronize()
    return r, time.perf_counter() - t0


def cpu_run(mol, xc, do_grad, do_hess, auxbasis):
    from pyscf import scf as cpu_scf, dft as cpu_dft
    out = {}
    if xc == 'hf':
        mf = cpu_scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = cpu_dft.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    t0 = time.perf_counter()
    e = mf.kernel()
    out['e'] = e
    out['t_scf'] = time.perf_counter() - t0
    if do_grad:
        t0 = time.perf_counter()
        out['g'] = mf.Gradients().kernel()
        out['t_grad'] = time.perf_counter() - t0
    if do_hess:
        t0 = time.perf_counter()
        out['h'] = mf.Hessian().kernel()
        out['t_hess'] = time.perf_counter() - t0
    return out


def gpu_run(mol, xc, mode, do_grad, do_hess, auxbasis):
    import cupy as cp
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    out = {}
    if xc == 'hf':
        mf = gpu_scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = gpu_dft.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = mode
    (e,), out['t_scf'] = timed(lambda: (mf.kernel(),))
    out['e'] = float(e)
    if do_grad:
        g, out['t_grad'] = timed(lambda: mf.Gradients().kernel())
        out['g'] = cp.asnumpy(g) if hasattr(g, 'get') else np.asarray(g)
    if do_hess:
        h, out['t_hess'] = timed(lambda: mf.Hessian().kernel())
        out['h'] = cp.asnumpy(h) if hasattr(h, 'get') else np.asarray(h)
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return out


def row(name, t_cpu, t64, t32, err_cpu, err_64, unit):
    def sp(a, b):
        return f'{a/b:6.2f}x' if (a and b) else '     --'
    c = f'{t_cpu:8.2f}s' if t_cpu else '      --'
    print(f'  {name:10s} {c} {t64:8.2f}s {t32:8.2f}s   '
          f'{sp(t_cpu, t32):>8s} {sp(t_cpu, t64):>8s} {sp(t64, t32):>8s}   '
          f'{err_cpu:>11s} {err_64:>11s}  {unit}', flush=True)


def fmt(x):
    return '--' if x is None else f'{x:.2e}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=os.path.join(TESTS, '057_Tamoxifen.xyz'))
    p.add_argument('--xc', default='hf')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--skip-cpu', action='store_true')
    p.add_argument('--skip-hessian', action='store_true')
    p.add_argument('--cpu-hessian', action='store_true',
                   help='CPU Hessian is O(hours) at 537 AO; off by default')
    p.add_argument('--auxbasis', default='def2-universal-jkfit',
                   help="DF auxiliary basis, pinned on BOTH sides. The defaults "
                        "differ -- PySCF picks def2-universal-jfit for a pure "
                        "functional (no exchange to fit) while gpu4pyscf always "
                        "takes jkfit -- so leaving it unset compares two "
                        "different levels of theory and reports the gap as a "
                        "code discrepancy (measured: 7.11e-04 Eh on "
                        "r2SCAN/Tamoxifen, which is not an error at all).")
    args = p.parse_args()

    import pyscf
    from pyscf import lib as pyscf_lib
    nthreads = pyscf_lib.num_threads()
    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    do_hess = not args.skip_hessian

    print(f'{args.xc}/{args.basis}  {mol.natm} atoms, {mol.nao} AO', flush=True)
    print(f'CPU: PySCF on {nthreads} threads   GPU: gpu4pyscf', flush=True)
    print(f'DF aux basis pinned on both sides: {args.auxbasis}', flush=True)

    cpu = {}
    if not args.skip_cpu:
        print('\nrunning CPU reference ...', flush=True)
        cpu = cpu_run(mol, args.xc, True, do_hess and args.cpu_hessian,
                      args.auxbasis)
    print('running GPU float64 ...', flush=True)
    g64 = gpu_run(mol, args.xc, 'fp64', True, do_hess, args.auxbasis)
    print('running GPU auto (shipped lane) ...', flush=True)
    g32 = gpu_run(mol, args.xc, 'auto', True, do_hess, args.auxbasis)

    print(f'\n  {"":10s} {"CPU":>9s} {"GPU64":>9s} {"GPUauto":>9s}   '
          f'{"auto/CPU":>8s} {"f64/CPU":>8s} {"auto/f64":>8s}   '
          f'{"err vs CPU":>11s} {"err vs f64":>11s}', flush=True)

    e_cpu = cpu.get('e')
    row('energy', cpu.get('t_scf'), g64['t_scf'], g32['t_scf'],
        fmt(abs(g32['e'] - e_cpu) if e_cpu is not None else None),
        fmt(abs(g32['e'] - g64['e'])), 'Eh')

    if 'g' in g64:
        gc = cpu.get('g')
        row('gradient', cpu.get('t_grad'), g64['t_grad'], g32['t_grad'],
            fmt(np.abs(g32['g'] - gc).max() if gc is not None else None),
            fmt(np.abs(g32['g'] - g64['g']).max()), 'Eh/Bohr')

    if 'h' in g64:
        hc = cpu.get('h')
        if hc is not None:
            hc = np.asarray(hc)
        row('hessian', cpu.get('t_hess'), g64['t_hess'], g32['t_hess'],
            fmt(np.abs(g32['h'] - hc).max() if hc is not None else None),
            fmt(np.abs(g32['h'] - g64['h']).max()), 'Eh/Bohr^2')

        # frequencies are what a user of the Hessian actually reads
        import vibanalysis
        nu64 = vibanalysis.frequencies(mol, g64['h'])
        nu32 = vibanalysis.frequencies(mol, g32['h'])
        line = (f'\n  frequencies: max|dnu| auto vs GPU-f64 '
                f'{np.abs(nu32-nu64).max():.4f} cm^-1')
        if hc is not None:
            nuc = vibanalysis.frequencies(mol, hc)
            line += f',  vs CPU {np.abs(nu32-nuc).max():.4f} cm^-1'
        print(line, flush=True)

    tot64 = g64['t_scf'] + g64.get('t_grad', 0)
    tot32 = g32['t_scf'] + g32.get('t_grad', 0)
    totc = cpu.get('t_scf', 0) + cpu.get('t_grad', 0)
    print(f'\n  energy+force together: GPU64 {tot64:.2f}s -> auto {tot32:.2f}s '
          f'({tot64/tot32:.2f}x)', flush=True)
    if totc:
        print(f'  {"":25s}CPU {totc:.2f}s -> auto {tot32:.2f}s '
              f'({totc/tot32:.2f}x)', flush=True)
    print('\n  NB: these are single-point phases. An end-to-end geometry '
          'optimization\n  is gated by the per-step CDERI rebuild and the '
          'float64 SCF tail -- see\n  the "端到端几何优化" section of '
          'STATUS.md (1.55-1.88x, size dependent).', flush=True)


if __name__ == '__main__':
    main()

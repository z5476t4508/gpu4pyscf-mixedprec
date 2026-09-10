#!/usr/bin/env python3
"""External validation: gpu4pyscf energy vs g16 on the same geometry+level.

g16's tests/methanols0.fchk is a converged RB3LYP/6-31+G(d,p) methanol,
E = -115.7348716828283 Eh. geoms/methanol.xyz is extracted from the same file
(extract_methanol.py), so a gpu4pyscf run at the SAME level of theory on that
geometry compares like with like.

This is deliberately NOT part of run_benchmark.py's matrix: the benchmark runs
DF + def2-svp for timing, and a DF/def2-svp energy against this g16 number
would fold the DF error and the basis mismatch into what looks like a code
error. Validation here runs conventional (no density fitting), b3lyp,
6-31+G(d,p).

CPU PySCF at the same level is printed alongside: it is the third leg of the
triangle (gpu4pyscf vs PySCF vs g16), so a discrepancy can be attributed.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/benchmark/validate_methanol.py [--mode auto] [--skip-cpu]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# source tree first: the venv wheel lacks the mixed-precision code, so
# precision_mode would be a silently-ignored attribute (measured 2026-09-10)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
GEOMS = os.path.join(HERE, 'geoms')

E_G16 = -115.7348716828283   # "Total Energy", methanols0.fchk
LEVEL = 'b3lyp'
BASIS = '6-31+g(d,p)'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='auto', help='precision_mode (gpu4pyscf)')
    p.add_argument('--fp64', action='store_true',
                   help='also run the GPU lane in fp64 (default: auto only, '
                        'plus CPU PySCF as the fp64 oracle)')
    p.add_argument('--skip-cpu', action='store_true')
    args = p.parse_args()

    import pyscf
    mol = pyscf.M(atom=os.path.join(GEOMS, 'methanol.xyz'), basis=BASIS,
                  verbose=0)
    # Gaussian's Pople basis sets default to Cartesian d (6d); PySCF's
    # default spherical 5d costs 1.66e-3 Eh and is NOT a code error.
    # With cart=True the CPU PySCF energy matches g16 to -2.19e-07 Eh.
    mol.cart = True
    print(f'{LEVEL}/{BASIS} (cart, 6d), {mol.natm} atoms, {mol.nao} AO')
    print(f'g16 reference: {E_G16:.10f} Eh (methanols0.fchk, conventional '
          f'integrals, no DF)', flush=True)

    if not args.skip_cpu:
        from pyscf import dft as cpu_dft
        mf = cpu_dft.RKS(mol, xc=LEVEL)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        e_cpu = mf.kernel()
        print(f'\nCPU PySCF   : {e_cpu:.10f} Eh   dE vs g16 = '
              f'{e_cpu - E_G16:+.3e} Eh', flush=True)

    from gpu4pyscf import dft as gpu_dft
    modes = ['fp64', args.mode] if args.fp64 else [args.mode]
    for mode in modes:
        import cupy as cp
        mf = gpu_dft.RKS(mol, xc=LEVEL)
        mf.conv_tol = 1e-10
        mf.verbose = 0
        mf.precision_mode = mode
        e = mf.kernel()
        print(f'GPU {mode:5s}: {float(e):.10f} Eh   dE vs g16 = '
              f'{float(e) - E_G16:+.3e} Eh', flush=True)
        mf = None
        cp.get_default_memory_pool().free_all_blocks()


if __name__ == '__main__':
    main()

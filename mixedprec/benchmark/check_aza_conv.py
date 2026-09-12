"""Experiment: is Aza-tzvp auto's SCF non-convergence just a cycle budget issue?

fp64 converged in <=50 cycles; auto (fp32 start) did not. Probe max_cycle
100/150 with fresh SCF each time, compare energy against the fp64 reference.
"""
import json
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

d = json.load(open('/home/tong/soft/gpu4pyscf/mixedprec/benchmark/results/benchmark.json'))
e64 = {(r['xc'], r['lane']): r.get('e') for r in d['rows']
       if r['geom'].startswith('095') and r['basis'] == 'def2-tzvp'}
print('fp64 references:', {k: f'{v:.8f}' for k, v in e64.items() if v and k[1] == 'fp64'})

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/095_Azadirachtin.xyz',
              basis='def2-tzvp', verbose=0)
from gpu4pyscf import scf, dft  # noqa: E402

AUX = 'def2-universal-jkfit'

for xc in ('hf', 'b3lyp'):
    for max_cycle in (100, 150):
        if xc == 'hf':
            mf = scf.RHF(mol).density_fit(auxbasis=AUX)
        else:
            mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=AUX)
        mf.conv_tol = 1e-10
        mf.max_cycle = max_cycle
        mf.verbose = 0
        mf.precision_mode = 'auto'
        t0 = time.perf_counter()
        e = mf.kernel()
        dt = time.perf_counter() - t0
        ref = e64.get((xc, 'fp64'))
        de = f' dE_vs_fp64={float(e) - ref:+.2e}' if ref else ''
        print(f'{xc:7s} max_cycle={max_cycle}: converged={mf.converged} '
              f'E={float(e):.8f}{de} ({dt:.1f}s)')
        mf = None
        cp.get_default_memory_pool().free_all_blocks()

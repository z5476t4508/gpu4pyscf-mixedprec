"""Probe 3: test whether r2scan runs poison the subsequent HF auto lane.

The benchmark runner interleaves hf and r2scan per molecule; Vitamin C (the
second molecule) already lost the auto speedup. Reproduce the runner's exact
per-molecule sequence on methanol then Tamoxifen.
"""
import os
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')

import cupy as cp
import pyscf

from step9_scorecard import gpu_run
from gpu4pyscf.lib import precision

G = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms'
AUX = 'def2-universal-jkfit'

def lane(mol, xc, mode, tag):
    out = gpu_run(mol, xc, mode, True, False, AUX)
    print(f'{tag}: scf {out["t_scf"]:7.3f}s grad {out["t_grad"]:7.3f}s  '
          f'[global={precision.get_precision()}]')

m1 = pyscf.M(atom=f'{G}/methanol.xyz', basis='def2-svp', verbose=0)
lane(m1, 'hf', 'fp64', 'meoh  hf  fp64 ')
lane(m1, 'hf', 'auto', 'meoh  hf  auto ')
lane(m1, 'r2scan', 'fp64', 'meoh  r2s fp64 ')
lane(m1, 'r2scan', 'auto', 'meoh  r2s auto ')
cp.get_default_memory_pool().free_all_blocks()

m3 = pyscf.M(atom=f'{G}/020_Vitamin_C.xyz', basis='def2-svp', verbose=0)
lane(m3, 'hf', 'fp64', 'vitc  hf  fp64 ')
lane(m3, 'hf', 'auto', 'vitc  hf  auto ')
lane(m3, 'r2scan', 'fp64', 'vitc  r2s fp64 ')
lane(m3, 'r2scan', 'auto', 'vitc  r2s auto ')
cp.get_default_memory_pool().free_all_blocks()

m2 = pyscf.M(atom=f'{G}/057_Tamoxifen.xyz', basis='def2-svp', verbose=0)
lane(m2, 'hf', 'fp64', 'tamo  hf  fp64 ')
lane(m2, 'hf', 'auto', 'tamo  hf  auto ')

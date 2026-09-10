"""Probe: does running the CPU reference in the same process (as step9 does)
change the GPU auto lane's timing? Replicates step9_scorecard.main's flow.
"""
import os
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')

import cupy as cp
import numpy as np
import pyscf

from step9_scorecard import cpu_run, gpu_run, timed

XYZ = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/057_Tamoxifen.xyz'
AUX = 'def2-universal-jkfit'

mol = pyscf.M(atom=XYZ, basis='def2-svp', verbose=0)

print('--- with cpu_run first (step9 order) ---')
cpu = cpu_run(mol, 'hf', True, False, AUX)
print(f"cpu scf {cpu['t_scf']:.2f}s grad {cpu['t_grad']:.2f}s")

g64 = gpu_run(mol, 'hf', 'fp64', True, False, AUX)
g32 = gpu_run(mol, 'hf', 'auto', True, False, AUX)
print(f"fp64: scf {g64['t_scf']:.3f}s grad {g64['t_grad']:.3f}s")
print(f"auto: scf {g32['t_scf']:.3f}s grad {g32['t_grad']:.3f}s")

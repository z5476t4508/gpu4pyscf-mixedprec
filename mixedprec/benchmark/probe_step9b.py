"""Probe 2: bisect why repeated gpu_run calls lose the auto-lane speedup.

Sequence: fp64, auto, auto-again, auto-on-fresh-mol. If the SECOND auto run
slows to fp64 speed, the loss is tied to repeating runs on the same object;
if it stays fast, the runner's slow lane must come from something else.
"""
import os
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')

import cupy as cp
import pyscf

from step9_scorecard import gpu_run

XYZ = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/057_Tamoxifen.xyz'
AUX = 'def2-universal-jkfit'

mol = pyscf.M(atom=XYZ, basis='def2-svp', verbose=0)

out = gpu_run(mol, 'hf', 'fp64', True, False, AUX)
print(f"fp64 (1st): scf {out['t_scf']:.3f}s grad {out['t_grad']:.3f}s")

out = gpu_run(mol, 'hf', 'auto', True, False, AUX)
print(f"auto (1st): scf {out['t_scf']:.3f}s grad {out['t_grad']:.3f}s")

out = gpu_run(mol, 'hf', 'auto', True, False, AUX)
print(f"auto (2nd): scf {out['t_scf']:.3f}s grad {out['t_grad']:.3f}s")

cp.get_default_memory_pool().free_all_blocks()
mol2 = pyscf.M(atom=XYZ, basis='def2-svp', verbose=0)
out = gpu_run(mol2, 'hf', 'auto', True, False, AUX)
print(f"auto (fresh mol): scf {out['t_scf']:.3f}s grad {out['t_grad']:.3f}s")

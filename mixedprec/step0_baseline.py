"""Step 0: FP64 baseline benchmark for DF-RHF + gradient on RTX 5090.

Times the two pieces that matter for the mixed-precision plan:
  - SCF (all iterations, FP64)
  - gradient (single call, FP64)
Run from outside the repo directory so the installed wheel is used.
"""
import time
import numpy as np
import pyscf
from pyscf import scf as cpu_scf
from gpu4pyscf import scf

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/020_Vitamin_C.xyz'
BASIS = 'def2-tzvp'

mol = pyscf.M(atom=XYZ, basis=BASIS, verbose=0)
nao = mol.nao_nr()
print(f'Vitamin C: {mol.natm} atoms, {nao} AOs, basis={BASIS}')

mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10

# --- SCF timing ---
mf.scf_summary = {}  # silence
t0 = time.time()
e_gpu = mf.kernel()
t_scf = time.time() - t0
n_iter = mf.cycles if hasattr(mf, 'cycles') else '?'

# --- gradient timing ---
t0 = time.time()
g_gpu = mf.Gradients().kernel()
t_grad = time.time() - t0

# --- CPU reference (energy only, single point) ---
mf_cpu = cpu_scf.RHF(mol).density_fit()
mf_cpu.conv_tol = 1e-10
e_cpu = mf_cpu.kernel()
g_cpu = mf_cpu.Gradients().kernel()

print(f'\n--- FP64 baseline (5090) ---')
print(f'SCF    : {t_scf:8.2f}s   ({n_iter} iterations)')
print(f'gradient: {t_grad:7.2f}s')
print(f'gradient share of (SCF+grad): {t_grad/(t_scf+t_grad)*100:.1f}%')
print(f'\nenergy GPU-CPU diff : {abs(e_gpu-e_cpu):.2e}')
print(f'grad   GPU-CPU max  : {np.abs(g_gpu-g_cpu).max():.2e}')

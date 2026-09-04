"""Step 1 prep: profile DF-RHF SCF and gradient internals on a 3-amino-acid-sized
system (Tamoxifen, 56 atoms, def2-TZVP ~ 1200 AOs).

Writes pstats breakdowns to mixedprec/profile_scf.txt and profile_grad.txt
plus a summary to mixedprec/profile_summary.txt
"""
import cProfile
import io
import pstats
import time

import numpy as np
import pyscf
from gpu4pyscf import scf

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec'

# warm up cupy kernels on a tiny system first
warm = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp', verbose=0)
scf.RHF(warm).density_fit().kernel()

mol = pyscf.M(atom=XYZ, basis='def2-tzvp', verbose=0)
print(f'Tamoxifen: {mol.natm} atoms, {mol.nao_nr()} AOs', flush=True)

mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10

# --- profile SCF ---
pr = cProfile.Profile()
t0 = time.time()
pr.enable()
e = mf.kernel()
pr.disable()
t_scf = time.time() - t0
n_iter = mf.cycles

# --- profile gradient ---
g_obj = mf.Gradients()
pr2 = cProfile.Profile()
t0 = time.time()
pr2.enable()
g = g_obj.kernel()
pr2.disable()
t_grad = time.time() - t0

def dump(prof, path, wall):
    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s)
    ps.sort_stats('cumulative').print_stats(35)
    with open(path, 'w') as f:
        f.write(f'wall time: {wall:.2f}s\n\n')
        f.write(s.getvalue())

dump(pr, f'{OUT}/profile_scf.txt', t_scf)
dump(pr2, f'{OUT}/profile_grad.txt', t_grad)

with open(f'{OUT}/profile_summary.txt', 'w') as f:
    f.write(f'Tamoxifen def2-TZVP DF-RHF: {mol.nao_nr()} AOs\n')
    f.write(f'SCF     : {t_scf:.2f}s ({n_iter} iterations)\n')
    f.write(f'gradient: {t_grad:.2f}s\n')
    f.write(f'gradient share: {t_grad/(t_scf+t_grad)*100:.1f}%\n')
    f.write(f'energy: {e:.10f}\n')
print(f'SCF {t_scf:.2f}s ({n_iter} it), grad {t_grad:.2f}s, '
      f'grad share {t_grad/(t_scf+t_grad)*100:.1f}%', flush=True)
print('DONE', flush=True)

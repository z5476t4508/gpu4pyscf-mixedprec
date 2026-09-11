"""Is the 0.5 cm^-1 cross-code Hessian gap a grid-convergence issue?

methanol r2SCAN: GPU fp64 freq vs CPU PySCF freq at grid levels 3/4/5.
If the gap shrinks with level, the two implementations' default grids
differ in convergence, and the benchmark must pin the grid explicitly.
"""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec/benchmark')

import numpy as np
import pyscf

import vibanalysis

MOL = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/methanol.xyz'

# GPU fp64 reference from the smoke run (benchmark.json): recompute here
from gpu4pyscf import dft  # noqa: E402

mol = pyscf.M(atom=MOL, basis='def2-svp', verbose=0)
mf = dft.RKS(mol, xc='r2scan').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.verbose = 0
mf.precision_mode = 'fp64'
mf.kernel()
h_gpu = np.asarray(mf.Hessian().kernel())
nu_gpu = vibanalysis.frequencies(mol, h_gpu)
print(f'GPU fp64 grid level: {mf.grids.level}')

for level in (3, 4, 5):
    mol2 = pyscf.M(atom=MOL, basis='def2-svp', verbose=0)
    from pyscf import dft as cpu_dft
    mfc = cpu_dft.RKS(mol2, xc='r2scan').density_fit(
        auxbasis='def2-universal-jkfit')
    mfc.conv_tol = 1e-10
    mfc.verbose = 0
    mfc.grids.level = level
    mfc.kernel()
    h_cpu = np.asarray(mfc.Hessian().kernel())
    nu_cpu = vibanalysis.frequencies(mol2, h_cpu)
    print(f'CPU grid level {level}: max|dnu| vs GPU = '
          f'{np.abs(nu_cpu - nu_gpu).max():.3e} cm-1 '
          f'(ngrid={mfc.grids.coords.shape[0]})')

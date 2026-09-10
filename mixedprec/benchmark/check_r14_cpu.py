"""Control: does CPU PySCF converge r14? And check the r2scan lane too."""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import pyscf

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
AUX = 'def2-universal-jkfit'

from pyscf import scf as cpu_scf, dft as cpu_dft  # noqa: E402

mf = cpu_scf.UHF(mol).density_fit(auxbasis=AUX)
mf.conv_tol = 1e-10
mf.max_cycle = 100
mf.verbose = 0
e = mf.kernel()
print(f'CPU UHF   : converged={mf.converged} E={e:.8f}')

mf = cpu_dft.UKS(mol, xc='r2scan').density_fit(auxbasis=AUX)
mf.conv_tol = 1e-10
mf.max_cycle = 100
mf.verbose = 0
e = mf.kernel()
print(f'CPU r2scan: converged={mf.converged} E={e:.8f}')

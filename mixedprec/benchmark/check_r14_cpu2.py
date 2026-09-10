"""Confirm the huckel-init converged UHF solution cross-code (CPU PySCF)."""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import pyscf

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
from pyscf import scf as cpu_scf  # noqa: E402

mf = cpu_scf.UHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.max_cycle = 100
mf.init_guess = 'huckel'
mf.verbose = 0
e = mf.kernel()
print(f'CPU UHF huckel: converged={mf.converged} E={e:.8f}')
print(f'GPU UHF huckel:               E=-2049.86009516 (from remedy run)')

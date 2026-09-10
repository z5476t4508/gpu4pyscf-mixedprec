"""Decide: is the auto lane's lower r14 solution real, or fp32 noise?

Restart fp64 from the auto lane's converged orbitals. If fp64 converges to
the same energy, the auto solution is a genuine SCF solution; if it escapes
to the fp64-native solution, the auto result was a phantom.
"""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
from gpu4pyscf import scf  # noqa: E402

AUX = 'def2-universal-jkfit'

def fresh(mode):
    mf = scf.UHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.verbose = 0
    mf.init_guess = 'huckel'
    mf.precision_mode = mode
    return mf

mf_a = fresh('auto')
e_a = mf_a.kernel()
print(f'auto     : converged={mf_a.converged} E={float(e_a):.8f}')

mf_f = fresh('fp64')
mf_f.precision_mode = 'fp64'
dm0 = mf_a.make_rdm1(mf_a.mo_coeff, mf_a.mo_occ)
e_f = mf_f.kernel(dm0=dm0)
print(f'fp64@auto: converged={mf_f.converged} E={float(e_f):.8f}')

mf_f2 = fresh('fp64')
e_f2 = mf_f2.kernel()
print(f'fp64     : converged={mf_f2.converged} E={float(e_f2):.8f}')

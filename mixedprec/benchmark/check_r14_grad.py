"""Honest UHF fp32 gradient error: same SCF state, gradient A/B only."""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pyscf  # noqa: E402

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
from gpu4pyscf import scf  # noqa: E402

AUX = 'def2-universal-jkfit'

mf = scf.UHF(mol).density_fit(auxbasis=AUX)
mf.conv_tol = 1e-10
mf.max_cycle = 100
mf.verbose = 0
mf.init_guess = 'huckel'
mf.precision_mode = 'fp64'
e = mf.kernel()
print(f'SCF fp64: converged={mf.converged} E={float(e):.8f}')

grads = {}
for mode in ('fp64', 'fp32'):
    g = mf.Gradients()
    g.precision_mode = mode
    de = g.kernel()
    grads[mode] = np.asarray(de)
    print(f'grad {mode}: max|de| = {np.abs(de).max():.3e}')
print(f'|dg| fp32 vs fp64, same state: {np.abs(grads["fp32"]-grads["fp64"]).max():.3e}')

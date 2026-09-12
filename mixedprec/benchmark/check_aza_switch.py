"""Diagnose: does the auto lane ever switch to fp64 on Aza-tzvp?

Watch the per-cycle |dE| of the auto lane vs the 1e-4 switch threshold.
If dE plateaus above 1e-4 (fp32 noise floor at this size), the lane never
switches and grinds in fp32 forever -- explaining the non-convergence.
"""
import sys

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import pyscf

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/095_Azadirachtin.xyz',
              basis='def2-tzvp', verbose=0)
from gpu4pyscf import scf  # noqa: E402

mf = scf.RHF(mol).density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.max_cycle = 40
mf.verbose = 4
mf.precision_mode = 'auto'
mf.kernel()

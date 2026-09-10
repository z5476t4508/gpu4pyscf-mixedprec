"""Try to actually converge r14: more cycles, then SOSCF (Newton) fallback."""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
from gpu4pyscf import scf  # noqa: E402

AUX = 'def2-universal-jkfit'

def fresh():
    mf = scf.UHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    return mf

# attempt 1: more cycles
mf = fresh()
mf.max_cycle = 150
t0 = time.perf_counter()
e = mf.kernel()
print(f'max_cycle=150: converged={mf.converged} E={float(e):.8f} '
      f'({time.perf_counter()-t0:.1f}s)')
mf = None
cp.get_default_memory_pool().free_all_blocks()

# attempt 2: Newton SOSCF from a fresh run's orbitals
mf = fresh()
e = mf.kernel()
print(f'baseline (50cyc): converged={mf.converged} E={float(e):.8f}')
if not mf.converged:
    mf2 = mf.newton()
    mf2.conv_tol = 1e-10
    mf2.verbose = 0
    try:
        t0 = time.perf_counter()
        e2 = mf2.kernel(mf.make_rdm1(mf.mo_coeff, mf.mo_occ))
        print(f'newton: converged={mf2.converged} E={float(e2):.8f} '
              f'({time.perf_counter()-t0:.1f}s)')
    except Exception as exc:
        print(f'newton FAILED: {exc!r}')

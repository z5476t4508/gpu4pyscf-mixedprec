"""GPU-side convergence remedies for r14: level shift, damping, huckel guess."""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/r14.xyz',
              basis='def2-svp', charge=1, spin=1, verbose=0)
from gpu4pyscf import scf  # noqa: E402

AUX = 'def2-universal-jkfit'

def attempt(tag, **attrs):
    mf = scf.UHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.max_cycle = 100
    mf.verbose = 0
    for k, v in attrs.items():
        setattr(mf, k, v)
    t0 = time.perf_counter()
    try:
        e = mf.kernel()
        print(f'{tag}: converged={mf.converged} E={float(e):.8f} '
              f'({time.perf_counter()-t0:.1f}s)')
    except Exception as exc:
        print(f'{tag}: FAILED {exc!r}')
    mf = None
    cp.get_default_memory_pool().free_all_blocks()

attempt('level_shift=0.25', level_shift=0.25)
attempt('level_shift=0.5 ', level_shift=0.5)
attempt('diis_space=12   ', diis_space=12)
attempt('init_guess=huck ', init_guess='huckel')
attempt('diis+damp 0.2   ', diis_space=12, diis_damping=0.2)

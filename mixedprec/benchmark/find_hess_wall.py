"""Wall-finder: at ~3300 AO (Azadirachtin/def2-qzvp), which allocation dies?

Prerequisite for the out-of-core design: identify the unbounded tensor in
the r2SCAN Hessian path. Captures the full cupy OOM message (requested
bytes) and the traceback's gpu4pyscf frames.
"""
import sys
import traceback

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/095_Azadirachtin.xyz',
              basis='def2-qzvp', verbose=0)
print(f'nao={mol.nao} natm={mol.natm}', flush=True)
from gpu4pyscf import dft  # noqa: E402

mf = dft.RKS(mol, xc='r2scan').density_fit(
    auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.verbose = 4
mf.precision_mode = 'fp64'
mf.kernel()
print('SCF done; Hessian next', flush=True)
try:
    h = mf.Hessian().kernel()
    print(f'HESS OK {h.shape} -- wall is above {mol.nao} AO')
except Exception as exc:
    print(f'OOM: {exc}', flush=True)
    tb = traceback.format_exc()
    keep = [ln.strip()[:130] for ln in tb.splitlines()
            if 'gpu4pyscf' in ln or 'memory.pyx' in ln or 'OOM' in ln
            or 'allocat' in ln.lower()]
    print('\n'.join(keep[-15:]), flush=True)

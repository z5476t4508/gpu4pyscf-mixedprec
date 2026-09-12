"""Experiment 1a: does a float32 CDERI push the r2SCAN Hessian through the
32 GB wall, and what does it cost in accuracy?

The wall: r2SCAN Hessian OOMs at 934 AO (svp) / 1042 AO (tzvp); the CDERI
tensor (~24 GB fp64 @ 934 AO) is the bulk. set_cderi_precision('fp32')
halves it (eigendecomposition metric, screening-grade).

Measure on Azadirachtin svp r2scan:
  1. fp64 CDERI (reference: OOM expected -- confirm)
  2. fp32 CDERI, fp64 Hessian lane: fits? time? frequencies vs fp64-CDERI
     fp64 lane (need a reference Hessian: compute on the smaller Vitamin C
     first for the accuracy ratio; for Aza itself the reference is the
     JSON's auto row is HESS-OOM... so the only Aza reference is CPU-free.
     Compare GPU fp32-CDERI frequencies against... nothing local. So also
     run Vitamin C svp r2scan where fp64-CDERI works: accuracy there
     transfers as the per-CDERI-precision cost.)
Plan:
  A. Vitamin C svp r2scan: Hessian with fp64 CDERI (ref) vs fp32 CDERI ->
     rms/max dnu (the CDERI-precision cost).
  B. Azadirachtin svp r2scan: fp32 CDERI Hessian -> fits or OOM; time.
"""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec/benchmark')

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pyscf  # noqa: E402

import vibanalysis  # noqa: E402
from gpu4pyscf.lib import precision  # noqa: E402

GE = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms'
AUX = 'def2-universal-jkfit'


def hess_of(geom_file, cderi_prec):
    mol = pyscf.M(atom=f'{GE}/{geom_file}', basis='def2-svp', verbose=0)
    mf = pyscf_scanscf(mol)
    precision.set_cderi_precision(cderi_prec)
    t0 = time.perf_counter()
    h = mf.Hessian().kernel()
    dt = time.perf_counter() - t0
    precision.set_cderi_precision('fp64')
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return np.asarray(h), dt


def pyscf_scanscf(mol):
    from gpu4pyscf import dft
    mf = dft.RKS(mol, xc='r2scan').density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = 'fp64'
    mf.kernel()
    return mf


print('--- A. Vitamin C svp r2scan: CDERI precision cost ---')
mol = pyscf.M(atom=f'{GE}/020_Vitamin_C.xyz', basis='def2-svp', verbose=0)
mf = pyscf_scanscf(mol)
h64, t64 = hess_of('020_Vitamin_C.xyz', 'fp64')
nu64 = vibanalysis.frequencies(mol, h64)
h32, t32 = hess_of('020_Vitamin_C.xyz', 'fp32')
nu32 = vibanalysis.frequencies(mol, h32)
dn = nu32 - nu64
print(f'fp64 CDERI: hess {t64:.1f}s | fp32 CDERI: hess {t32:.1f}s')
print(f'dnu: max {np.abs(dn).max():.3e} rms {np.sqrt((dn**2).mean()):.3e} cm-1')
z64, s64 = vibanalysis.thermochemistry(mol, h64)
z32, s32 = vibanalysis.thermochemistry(mol, h32)
print(f'dZPE {z32 - z64:+.3e} Eh  dS_vib {s32 - s64:+.3e} Eh/K')

print('--- B. Azadirachtin svp r2scan: does fp32 CDERI fit? ---')
try:
    h, dt = hess_of('095_Azadirachtin.xyz', 'fp32')
    print(f'fp32 CDERI Hessian at 934 AO: {dt:.1f}s -- FITS')
except Exception as exc:
    print(f'fp32 CDERI Hessian at 934 AO FAILED: {exc!r}')

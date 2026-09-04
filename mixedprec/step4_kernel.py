"""Step 4 benchmark: mixed-precision HF with kernel-level fp32.

Runs three lanes on Tamoxifen (def2-SVP and def2-TZVP):
  FP64   - unmodified code path
  auto   - SCF with precision_mode='auto' (fp32 contractions until |dE|<1e-4)
           + fp32 gradient kernel (sum_ejk_int3c2e_ip1_f32)
  fp32   - pure fp32 contractions (screening lane, conv 1e-5)

Run:  cd /tmp && PYTHONPATH=/home/tong/soft/gpu4pyscf \
      /home/tong/soft/gpu4pyscf/.venv/bin/python <this script>
"""
import sys
import time
import numpy as np
import pyscf
from gpu4pyscf import scf
from gpu4pyscf.lib import precision
import gpu4pyscf.grad.rhf as grhf_mod

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec/step4_result.txt'
REPORT = []

def log(msg):
    REPORT.append(msg)
    with open(OUT, 'w') as f:
        f.write('\n'.join(REPORT) + '\n')
    print(msg, flush=True)

def warmup():
    w = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp', verbose=0)
    scf.RHF(w).density_fit().kernel()

def run_single(basis, mode, conv):
    mol = pyscf.M(atom=XYZ, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = conv
    if mode:
        mf.precision_mode = mode
    t0 = time.time()
    e = mf.kernel()
    t_scf = time.time() - t0
    return t_scf, e, mf.cycles

def run_grad(basis, conv=1e-10):
    mol = pyscf.M(atom=XYZ, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = conv
    mf.kernel()
    g = mf.nuc_grad_method()
    t0 = time.time()
    g64 = g.kernel()
    t64 = time.time() - t0
    with precision.fp32():
        t0 = time.time()
        g32 = g.kernel()
        t32 = time.time() - t0
    return t64, t32, g64, g32

def run_opt(basis, mixed, maxsteps=10):
    mol = pyscf.M(atom=XYZ, basis=basis, verbose=0)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = 1e-7
    orig_kernel = grhf_mod.Gradients.kernel
    if mixed:
        mf.precision_mode = 'auto'
        def fp32_kernel(self, *a, **kw):
            with precision.fp32():
                return orig_kernel(self, *a, **kw)
        grhf_mod.Gradients.kernel = fp32_kernel
    from pyscf.geomopt.geometric_solver import optimize
    try:
        t0 = time.time()
        optimize(mf, maxsteps=maxsteps)
        t = time.time() - t0
    finally:
        grhf_mod.Gradients.kernel = orig_kernel
    return t, mf.e_tot

warmup()
for basis in ('def2-svp', 'def2-tzvp'):
    log(f'=== Tamoxifen {basis} single point ===')
    t64, e64, c64 = run_single(basis, None, 1e-10)
    log(f'FP64 : {t64:6.2f}s  {e64:.10f}  {c64} cycles')
    t_au, e_au, c_au = run_single(basis, 'auto', 1e-10)
    log(f'auto : {t_au:6.2f}s  {e_au:.10f}  {c_au} cycles  '
        f'err={abs(e_au-e64):.2e}  speedup {t64/t_au:.2f}x')
    t32, e32, c32 = run_single(basis, 'fp32', 1e-5)
    log(f'fp32 : {t32:6.2f}s  {e32:.8f}  {c32} cycles  '
        f'abs-err={abs(e32-e64):.1e}  speedup {t64/t32:.2f}x')

    log(f'=== Tamoxifen {basis} gradient ===')
    t64g, t32g, g64, g32 = run_grad(basis)
    rel = np.linalg.norm(g32-g64)/np.linalg.norm(g64)
    log(f'grad FP64 {t64g:.2f}s -> fp32 {t32g:.2f}s  speedup {t64g/t32g:.2f}x  '
        f'max|diff| {np.abs(g32-g64).max():.1e} rel-norm {rel:.1e}')

log('=== Tamoxifen def2-svp geometry optimization (10 steps) ===')
t_o64, e_o64 = run_opt('def2-svp', mixed=False)
log(f'FP64 opt : {t_o64:6.1f}s')
t_om, e_om = run_opt('def2-svp', mixed=True)
log(f'mixed opt: {t_om:6.1f}s  speedup {t_o64/t_om:.2f}x  '
    f'final E diff {abs(e_om-e_o64):.1e}')
log('DONE')

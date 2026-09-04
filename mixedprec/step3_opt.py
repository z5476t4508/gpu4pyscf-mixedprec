"""Step 3: geometry-optimization benchmark (FP64 vs mixed precision).

Setup (per optimization step, Tamoxifen def2-TZVP):
  - SCF warm-started from previous step's DM (SCF_Scanner)
  - with_df.reset() per step -> CDERI rebuilt every geometry (~4.2s)
  - gradient: df.grad.rhf.Gradients.kernel (~12.3s, always FP64 here)

Mixed policy:
  - every SCF run starts in FP32 (mode reset via mf.kernel wrapper — the
    global mode never resets itself, without this every step after the first
    runs pure FP64)
  - switches to FP64 once |dE| < 1e-4, runs the tail in FP64 (conv 1e-7)
  - gradients always FP64

Fixes vs the overnight attempt:
  - explicit `from pyscf.geomopt.geometric_solver import optimize`
  - e_delta derived from envs['e_tot'] - envs['last_hf_e']
  - gradient timing hook patches gpu4pyscf.df.grad.rhf.Gradients (what
    density_fit() actually uses), not gpu4pyscf.grad.rhf.Gradients
"""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')
import numpy as np
import pyscf
from pyscf.geomopt.geometric_solver import optimize
from gpu4pyscf import scf
import fp32_jk

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec'
MAXSTEPS = 10
SWITCH_TOL = 1e-4
REPORT = []

def log(msg):
    REPORT.append(msg)
    with open(f'{OUT}/step3_result.txt', 'w') as f:
        f.write('\n'.join(REPORT) + '\n')
    print(msg, flush=True)

import gpu4pyscf.df.grad.rhf as df_grad_mod
_orig_grad_kernel = df_grad_mod.Gradients.kernel
_orig_scf_kernel = scf.hf.RHF.kernel

def run_opt(tag, mf, mixed):
    grad_times, scf_times = [], []
    step_t0 = [None]

    def timed_grad(self, *a, **kw):
        prev = fp32_jk.PRECISION_MODE
        fp32_jk.PRECISION_MODE = 'fp64'   # gradient always FP64
        t0 = time.time()
        r = _orig_grad_kernel(self, *a, **kw)
        grad_times.append(time.time() - t0)
        fp32_jk.PRECISION_MODE = prev
        return r

    def timed_scf(self, *a, **kw):
        if mixed:
            fp32_jk.PRECISION_MODE = 'fp32'  # every SCF run starts fast
        t0 = time.time()
        r = _orig_scf_kernel(self, *a, **kw)
        scf_times.append((time.time() - t0, self.cycles))
        return r

    def cb(envs):
        if not mixed:
            return
        e_tot = envs.get('e_tot')
        last_hf_e = envs.get('last_hf_e')
        if e_tot is None or last_hf_e is None:
            return
        if fp32_jk.PRECISION_MODE == 'fp32' and abs(e_tot - last_hf_e) < SWITCH_TOL:
            fp32_jk.PRECISION_MODE = 'fp64'

    df_grad_mod.Gradients.kernel = timed_grad
    scf.hf.RHF.kernel = timed_scf
    mf.callback = cb
    try:
        t0 = time.time()
        mol_opt = optimize(mf, maxsteps=MAXSTEPS)
        t = time.time() - t0
    finally:
        df_grad_mod.Gradients.kernel = _orig_grad_kernel
        scf.hf.RHF.kernel = _orig_scf_kernel
        fp32_jk.PRECISION_MODE = 'fp64'

    e_final = mf.e_tot
    scf_total = sum(x for x, _ in scf_times)
    grad_total = sum(grad_times)
    log(f'{tag}: {t:.1f}s total, {len(grad_times)} grad calls')
    log(f'  SCF: {scf_total:.1f}s ({scf_total/t*100:.0f}%) '
        f'per-step {[f"{x:.1f}/{c}c" for x, c in scf_times]}')
    log(f'  grad: {grad_total:.1f}s ({grad_total/t*100:.0f}%) '
        f'avg {np.mean(grad_times):.2f}s')
    log(f'  E_final = {e_final:.8f}')
    return t, e_final

# ---------- Run A: FP64 ----------
mol = pyscf.M(atom=XYZ, basis='def2-tzvp', verbose=0)
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-7
t_fp64, e_fp64 = run_opt('FP64', mf, mixed=False)

# ---------- Run B: mixed ----------
fp32_jk.install()
mol2 = pyscf.M(atom=XYZ, basis='def2-tzvp', verbose=0)
mf2 = scf.RHF(mol2).density_fit()
mf2.conv_tol = 1e-7
t_mixed, e_mixed = run_opt('mixed', mf2, mixed=True)

log(f'energy diff mixed vs FP64: {abs(e_mixed - e_fp64):.2e}')
log(f'speedup: {t_fp64/t_mixed:.2f}x')
log('DONE')

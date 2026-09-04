"""Step 2: mixed-precision SCF prototype benchmark.

Run A: FP64 baseline (fresh DF object, CDERI built FP64)
Run B: FP32 GEMMs until |e_tot - last_hf_e| < SWITCH_TOL, then FP64 tail.
Run C: same as B but conv_tol=1e-7 (geometry-optimization setting) — warm
       steps in an optimization only need ~1e-6-1e-7, which trims the FP64 tail.

All compared against CPU pyscf DF-RHF energy.

History of fixes vs the first overnight run (1.03x bogus):
  - e_delta never existed in callback locals(); now derived from
    e_tot - last_hf_e. Single trigger (2-consecutive rule cost extra cycles
    because the FP32 noise floor makes dE bounce back up right after switch).
"""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')
import pyscf
from pyscf import scf as cpu_scf
from gpu4pyscf import scf
import fp32_jk

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec'
SWITCH_TOL = 1e-4

mol = pyscf.M(atom=XYZ, basis='def2-tzvp', verbose=0)

def make_cb():
    def cb(envs):
        e_tot = envs.get('e_tot')
        last_hf_e = envs.get('last_hf_e')
        if e_tot is None or last_hf_e is None:
            return
        if fp32_jk.PRECISION_MODE == 'fp32' and abs(e_tot - last_hf_e) < SWITCH_TOL:
            fp32_jk.PRECISION_MODE = 'fp64'
    return cb

def run_mixed(conv_tol):
    fp32_jk.PRECISION_MODE = 'fp32'
    stats = {'fp32': 0, 'fp64': 0}
    def cb(envs):
        stats[fp32_jk.PRECISION_MODE] += 1
        make_cb()(envs)
    mf = scf.RHF(mol).density_fit()
    mf.conv_tol = conv_tol
    mf.callback = cb
    t0 = time.time()
    e = mf.kernel()
    t = time.time() - t0
    fp32_jk.PRECISION_MODE = 'fp64'
    return e, t, mf.cycles, stats

# ---------- Run A: FP64 baseline ----------
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10
t0 = time.time()
e_fp64 = mf.kernel()
t_fp64 = time.time() - t0
n_fp64 = mf.cycles

# ---------- Run B: mixed, tight conv ----------
fp32_jk.install()
e_mixed, t_mixed, n_mixed, stats = run_mixed(1e-10)

# ---------- Run C: mixed, geom-opt conv (1e-7) ----------
e_mixed7, t_mixed7, n_mixed7, stats7 = run_mixed(1e-7)

# ---------- FP64 reference at 1e-7 ----------
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-7
t0 = time.time()
e_fp64_7 = mf.kernel()
t_fp64_7 = time.time() - t0

# ---------- CPU reference ----------
t0 = time.time()
mf_cpu = cpu_scf.RHF(mol).density_fit()
mf_cpu.conv_tol = 1e-10
e_cpu = mf_cpu.kernel()
t_cpu = time.time() - t0

lines = [
    f'Tamoxifen def2-TZVP DF-RHF ({mol.nao_nr()} AOs), switch tol {SWITCH_TOL:g}',
    f'CPU reference : {e_cpu:.10f}  ({t_cpu:.1f}s, {mf_cpu.cycles} iters)',
    '',
    f'FP64   conv1e-10: {e_fp64:.10f}  ({t_fp64:.2f}s, {n_fp64} iters)  err={abs(e_fp64-e_cpu):.2e}',
    f'mixed  conv1e-10: {e_mixed:.10f}  ({t_mixed:.2f}s, {n_mixed} iters: '
    f'{stats["fp32"]} fp32 + {stats["fp64"]} fp64)  err={abs(e_mixed-e_cpu):.2e}',
    f'  -> speedup {t_fp64/t_mixed:.2f}x',
    '',
    f'FP64   conv1e-7 : {e_fp64_7:.10f}  ({t_fp64_7:.2f}s)',
    f'mixed  conv1e-7 : {e_mixed7:.10f}  ({t_mixed7:.2f}s, {n_mixed7} iters: '
    f'{stats7["fp32"]} fp32 + {stats7["fp64"]} fp64)  err={abs(e_mixed7-e_cpu):.2e}',
    f'  -> speedup {t_fp64_7/t_mixed7:.2f}x',
]
txt = '\n'.join(lines)
print(txt, flush=True)
with open(f'{OUT}/step2_result.txt', 'w') as f:
    f.write(txt + '\n')
print('DONE')

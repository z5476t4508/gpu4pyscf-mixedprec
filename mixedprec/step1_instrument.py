"""Step 1b: instrumented decomposition of DF-RHF gradient time.

Wraps (with stream syncs):
  - contract            -> cutensor GEMM contractions (FP32-able)
  - sum_ejk_int3c2e_ip1 -> derivative 3c2e integral kernel (FP64 CUDA)
  - int3c2e_evaluator   -> 3c2e integral evaluation kernel (FP64 CUDA)
"""
import sys
import time
from collections import defaultdict

import cupy as cp
import pyscf
from gpu4pyscf import scf

TIMER = defaultdict(float)

def _timed(fn, key):
    def wrapper(*a, **kw):
        cp.cuda.get_current_stream().synchronize()
        t0 = time.perf_counter()
        r = fn(*a, **kw)
        cp.cuda.get_current_stream().synchronize()
        TIMER[key] += time.perf_counter() - t0
        return r
    return wrapper

# patch contract in the modules that use it
import gpu4pyscf.df.grad.rhf as gmod
import gpu4pyscf.df.df_jk as jkmod
_orig_contract = gmod.contract
gmod.contract = _timed(_orig_contract, 'contract')
jkmod.contract = _timed(jkmod.contract, 'contract')

# patch derivative integral kernel
import gpu4pyscf.df.int3c2e_bdiv as bdiv
bdiv.libvhf_rys.sum_ejk_int3c2e_ip1 = _timed(
    bdiv.libvhf_rys.sum_ejk_int3c2e_ip1, 'sum_ejk_int3c2e_ip1')

# patch 3c2e evaluator construction
_orig_ev = bdiv.Int3c2eOpt.int3c2e_evaluator
def _patched_ev(self, *a, **kw):
    result = _orig_ev(self, *a, **kw)
    if callable(result):
        return _timed(result, 'eval_j3c')
    return (_timed(result[0], 'eval_j3c'),) + tuple(result[1:])
bdiv.Int3c2eOpt.int3c2e_evaluator = _patched_ev

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec'

warm = pyscf.M(atom='H 0 0 0; H 0 0 0.74', basis='def2-svp', verbose=0)
scf.RHF(warm).density_fit().kernel()

mol = pyscf.M(atom=XYZ, basis='def2-tzvp', verbose=0)
mf = scf.RHF(mol).density_fit()
mf.conv_tol = 1e-10

t0 = time.time()
mf.kernel()
t_scf = time.time() - t0
scf_timers = dict(TIMER)

TIMER.clear()
t0 = time.time()
g = mf.Gradients().kernel()
t_grad = time.time() - t0
grad_timers = dict(TIMER)

def report(tag, wall, timers):
    known = sum(timers.values())
    lines = [f'\n=== {tag}: wall {wall:.2f}s (sync overhead inflates wall) ===']
    for k, v in sorted(timers.items(), key=lambda x: -x[1]):
        lines.append(f'  {k:24s} {v:7.2f}s  ({v/known*100:5.1f}% of instrumented)')
    lines.append(f'  {"other (launch/mem/small)":24s} {wall-known:7.2f}s')
    return '\n'.join(lines)

txt = report('SCF', t_scf, scf_timers) + report('GRADIENT', t_grad, grad_timers)
print(txt)
with open(f'{OUT}/instrument_result.txt', 'w') as f:
    f.write(txt + '\n')
print('DONE')

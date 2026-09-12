"""Measure the allocation overhead of _contract_rho1_fxc's temp materialization.

Current impl materializes (rho1*fxc) at (1, nv2, nv2, ngrids) -- 100x the
output size -- then sums. Candidate: batched matvec into a preallocated out.
If the per-call saving, times natm x nblocks calls, is <1% of a Hessian,
the candidate closes without touching the kernel.
"""
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
from gpu4pyscf.dft.numint import _contract_rho1_fxc  # noqa: E402

nvar, blk = 5, 32768           # r2SCAN meta-GGA, typical block
fxc = cp.random.rand(2 * nvar, 2 * nvar, blk)   # (nv2, nv2, ngrids)
rho1 = cp.random.rand(2, nvar, blk)             # (nspin, nvar, ngrids) slice
N = 300

ref = _contract_rho1_fxc(rho1, fxc)

cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(N):
    wv = _contract_rho1_fxc(rho1, fxc)
cp.cuda.runtime.deviceSynchronize()
t_cur = (time.perf_counter() - t0) / N

# candidate: out[g,u] = sum_v fxc[v,u,g] * rho1[v,g]
nv2 = 2 * nvar
out = cp.empty((blk, nv2))
cp.cuda.runtime.deviceSynchronize()
t0 = time.perf_counter()
for _ in range(N):
    cp.matmul(fxc.transpose(2, 0, 1),
              rho1.reshape(nv2, blk).T[:, :, None], out=out[:, :, None])
cp.cuda.runtime.deviceSynchronize()
t_mat = (time.perf_counter() - t0) / N

wv = cp.asarray(out).reshape(blk, nv2)[:, 0].reshape(-1)  # spot-check below
ok = cp.allclose(cp.matmul(fxc.transpose(2, 0, 1),
                           rho1.reshape(nv2, blk).T[:, :, None])[:, :, 0].T,
                 ref.reshape(2, nvar, blk).reshape(nv2, blk), atol=1e-10)
print(f'numerically identical: {bool(ok)}')
print(f'current (alloc temp): {t_cur * 1e6:7.1f} us/call')
print(f'batched matvec + out : {t_mat * 1e6:7.1f} us/call')
# Hessian loop: nset(=natm) x nblocks calls; Tamo svp: 57 atoms, ~270k grids
for natm, nblocks, hess_s in ((57, 9, 300), (95, 9, 1687)):
    save = (t_cur - t_mat) * 1e-6 * natm * nblocks
    print(f'projected saving on {natm}-atom Hessian ({hess_s}s): '
          f'{save:.2f}s = {100 * save / hess_s:.2f}%')

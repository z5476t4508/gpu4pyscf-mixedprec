"""Confirm the Azadirachtin r2scan Hessian OOM in a FRESH process.

The failure happened right after the hf fp64 Hessian (2532s of huge tensors),
so fragmentation is a suspect. This runs the single lane clean, sampling
peak GPU memory.
"""
import subprocess
import sys
import threading
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')

peak = [0]

# simple sampler loop
def sampler2(stop):
    while not stop.is_set():
        try:
            out = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used',
                 '--format=csv,noheader,nounits'], text=True).strip()
            peak[0] = max(peak[0], int(out.splitlines()[0]))
        except Exception:
            pass
        time.sleep(1)

import cupy as cp
import pyscf

mol = pyscf.M(atom='/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/095_Azadirachtin.xyz',
              basis='def2-svp', verbose=0)
from gpu4pyscf import dft  # noqa: E402

stop = threading.Event()
t = threading.Thread(target=sampler2, args=(stop,), daemon=True)
t.start()

mf = dft.RKS(mol, xc='r2scan').density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-10
mf.verbose = 0
mf.precision_mode = 'fp64'
t0 = time.perf_counter()
try:
    e = mf.kernel()
    t_scf = time.perf_counter() - t0
    g = mf.Gradients().kernel()
    t_grad = time.perf_counter() - t0 - t_scf
    t0 = time.perf_counter()
    h = mf.Hessian().kernel()
    t_hess = time.perf_counter() - t0
    print(f'CLEAN PROCESS: scf {t_scf:.2f}s grad {t_grad:.2f}s '
          f'hess {t_hess:.2f}s  E={float(e):.8f}  peak mem {peak[0]} MiB')
except Exception as exc:
    print(f'CLEAN PROCESS: FAILED {exc!r}  peak mem {peak[0]} MiB')
finally:
    stop.set()

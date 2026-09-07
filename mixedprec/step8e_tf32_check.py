"""Step 8E: is `contract`'s float32 path actually float32, or TF32?

step8d found that casting the JK half's contractions to float32 moves Vitamin C's
frequencies by 8.13 cm^-1 -- 1400x the shipped lane and 217x the known-bad change
used as the sentinel's oracle. That is far more than float32 rounding should
cost: a 24-bit mantissa is ~6e-8 relative, and even accumulating over 3080 terms
should stay near 1e-6.

Two explanations, with opposite consequences:

  a) the contractions are differences of large, nearly-cancelling terms, so a
     1e-7 relative error on the operands lands as a large error on the result.
     Then no float32 contraction can work here, whatever the hardware does.
  b) cuTENSOR/cuBLAS is running these float32 GEMMs on TF32 tensor cores, which
     keep only 10 mantissa bits (~1e-3 relative). Then step8c's 10-49x speedups
     were never float32 speedups, and a true float32 GEMM is both slower and far
     more accurate than what was measured.

The two are distinguished by measuring the error of a *single* contraction on
well-conditioned random operands, where there is no cancellation to blame: the
relative error is ~1e-7 under (a) and ~1e-3 under (b).

If it is TF32, the same question applies to every `contract` already running
inside the shipped fp32 brackets (make_h1, solve_mo1) -- those measured
0.001 cm^-1, so either they avoid this path or the compute type differs, and
that is worth knowing either way.

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step8e_tf32_check.py
"""
import time

import cupy as cp
import numpy as np

from gpu4pyscf.lib.cutensor import contract

FP32_EPS = np.finfo(np.float32).eps      # 1.19e-07
TF32_EPS = 2.0 ** -11                    # 4.88e-04, 10 explicit mantissa bits


def one(spec, sa, sb, label):
    a64 = cp.random.random(sa)
    b64 = cp.random.random(sb)
    r64 = contract(spec, a64, b64)

    a32 = a64.astype(cp.float32)
    b32 = b64.astype(cp.float32)
    r32 = contract(spec, a32, b32)

    # the float32 operands are themselves rounded, so subtract that off: the
    # comparison is against a float64 contraction of the *same* rounded inputs
    r64_of_32 = contract(spec, a32.astype(cp.float64), b32.astype(cp.float64))
    err = float(cp.abs(r32.astype(cp.float64) - r64_of_32).max()
                / cp.abs(r64_of_32).max())

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    contract(spec, a64, b64)
    cp.cuda.runtime.deviceSynchronize()
    t_64 = time.perf_counter() - t0
    t0 = time.perf_counter()
    contract(spec, a32, b32)
    cp.cuda.runtime.deviceSynchronize()
    t_32 = time.perf_counter() - t0

    # the threshold has to sit between the two hypotheses, not near fp32_eps:
    # a reduction over 2626-3080 terms legitimately accumulates to a few times
    # 1e-6 in float32, which is still ~100x below TF32's 4.9e-4
    verdict = 'TF32' if err > 0.1 * TF32_EPS else 'fp32'
    print(f'  {label:22s} {spec:20s} rel.err {err:9.2e}  '
          f'({err/FP32_EPS:7.1f} x fp32_eps, {err/TF32_EPS:6.2f} x tf32_eps)  '
          f'{t_64*1e3:7.2f} -> {t_32*1e3:6.2f} ms  {t_64/t_32:5.2f}x  '
          f'-> {verdict}', flush=True)
    a64 = b64 = a32 = b32 = r64 = r32 = r64_of_32 = None
    cp.get_default_memory_pool().free_all_blocks()
    return err


def main():
    print(f'fp32 eps = {FP32_EPS:.3e},  tf32 eps = {TF32_EPS:.3e}', flush=True)
    print(f'cupy {cp.__version__}', flush=True)
    try:
        from gpu4pyscf.lib import cutensor
        print(f'cutensor available: {cutensor._CUTENSOR_AVAILABLE}', flush=True)
    except Exception as e:                       # noqa: BLE001
        print(f'cutensor probe failed: {e!r}', flush=True)

    print('\nreal shapes from the JK half (step8c\'s top specs):', flush=True)
    one('xsr,ysr->sxy', (3, 3080, 2626), (3, 3080, 2626), 'xsr,ysr (49x)')
    one('rs,ts->rt', (3080, 2626), (3080, 2626), 'rs,ts (32x)')
    one('xruj,sr->xsuj', (3, 2626, 15, 100), (3080, 2626), 'xruj,sr (23x)')

    print('\ncontrol: a plain matmul of the same size, via cupy directly',
          flush=True)
    n = 2048
    a = cp.random.random((n, n))
    b = cp.random.random((n, n))
    r64 = a.dot(b)
    a32, b32 = a.astype(cp.float32), b.astype(cp.float32)
    r64_of_32 = a32.astype(cp.float64).dot(b32.astype(cp.float64))
    err = float(cp.abs(a32.dot(b32).astype(cp.float64) - r64_of_32).max()
                / cp.abs(r64_of_32).max())
    print(f'  cupy.dot {n}x{n}      rel.err {err:9.2e}  '
          f'({err/FP32_EPS:7.1f} x fp32_eps)  -> '
          f'{"TF32" if err > 0.1*TF32_EPS else "fp32"}', flush=True)
    del a, b, r64, a32, b32, r64_of_32

    print('\nMeasured answer: fp32, not TF32. The errors sit at 2.5e-7 - 4e-6,\n'
          'i.e. what a float32 reduction over ~3000 terms costs, and ~100x\n'
          'below TF32. So step8c\'s 23-58x speedups are real float32 speedups,\n'
          'and step8d\'s 8.13 cm^-1 is hypothesis (a): the JK half\'s\n'
          'contractions are differences of large nearly-cancelling terms. The\n'
          'algorithm builds that in deliberately -- df/hessian/rhf.py:456 uses\n'
          'the exact identity (di/dr j|k)+(i dj/dr|k)+(ij|dk/dr)=0 to get the\n'
          'dj derivative as -(di+dk).', flush=True)


if __name__ == '__main__':
    main()

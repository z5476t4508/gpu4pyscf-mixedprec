# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

GPU4PySCF is a GPU-accelerated plugin for PySCF. Python-level code uses CuPy arrays (not NumPy) as the default array container; performance-critical kernels are C++/CUDA in `gpu4pyscf/lib/`, built with CMake and exposed to Python through `gpu4pyscf/lib/__init__.py` (via `cupy.RawModule`/ctypes wrappers). libXC is a separate wheel (`gpu4pyscf-libxc-cuda11x/12x/13x`), not built here.

## Build

```sh
cmake -S gpu4pyscf/lib -B build/temp.gpu4pyscf
cmake --build build/temp.gpu4pyscf -j 4
export PYTHONPATH="$(pwd):$PYTHONPATH"   # run from source without pip install
```

`build.sh` does the same with `CUDA_ARCHITECTURES="70-real;80"` and sets `CUPY_ACCELERATORS=cub,cutensor`. Requires nvcc; minimum compute capability is sm_70 (do not use features requiring sm_80+). Code must compile with CUDA 11 and 12 toolchains.

## Tests

Tests live in `tests/` subdirectories of each module (e.g. `gpu4pyscf/scf/tests/`). Default CI run:

```sh
pytest -m 'not slow and not benchmark and not special'
```

Single file / single test:

```sh
pytest gpu4pyscf/scf/tests/test_uhf.py
pytest gpu4pyscf/scf/tests/test_uhf.py::test_something
```

Benchmark suite is in `gpu4pyscf/tests/test_benchmark_*.py` (uses pytest-benchmark; run with `-m benchmark` or `--benchmark-disable` to run as plain tests). Tests require a GPU — they import `cupy` at module level.

## Lint

```sh
ruff check --config .ruff.toml gpu4pyscf
flake8 --config .flake8 gpu4pyscf
```

CI runs `ruff==0.15` with `--unsafe-fixes` plus a NumPy-usage check (`ruff check --select NPY --ignore NPY002 gpu4pyscf`). Line length is 120 (ruff) / 160 (flake8).

## Architecture

- **Module layout mirrors PySCF**: `scf/`, `dft/`, `df/`, `grad/`, `hessian/`, `tdscf/`, `mp/`, `cc/`, `solvent/`, `pbc/`, `geomopt/`, etc. GPU classes generally mirror the corresponding PySCF class names and signatures (e.g. `gpu4pyscf.dft.rks.RKS` vs `pyscf.dft.RKS`) but do not necessarily inherit from them.
- **`gpu4pyscf/__init__.py`** imports submodules and installs a custom CuPy memory-pool allocator. **`_patch_pyscf.py`** monkey-patches PySCF classes with `to_gpu()` methods so CPU objects convert to GPU ones. GPU objects should provide `to_cpu()` for the reverse direction.
- **`gpu4pyscf/lib/`**: C++/CUDA sources. `gint` (electron integral kernels, incl. derivative `_ip` variants), `gvhf`, `gdft`, `cupy_helper` (CuPy utility kernels), `cutensor.py`/`cublas.py`/`cusolver.py` (library wrappers; cutensor falls back to cupy with a UserWarning — that warning signals a broken cutensor/cupy installation). Python callers import these through `gpu4pyscf.lib.xxx` and must handle the compiled artifacts being absent until built.
- **Version pinning matters**: cupy and cutensor versions are strongly coupled (see `requirements.txt`, e.g. cupy 13.4.1 + cutensor 2.2.0). Mixing versions causes runtime failures.

## Conventions (from CONTRIBUTING.md)

- Performance is the primary goal; API differences from PySCF's CPU code are acceptable.
- Functions must accept mixed CuPy/NumPy inputs (PySCF CPU code may produce either) — convert as needed. Outputs only need to be consumable by GPU4PySCF, not by PySCF.
- If inheriting from PySCF classes, mute unsupported methods by overriding with a `NotImplementedError`-raising stub or assigning `None`/`NotImplemented`.
- Unit tests should verify results against the corresponding PySCF CPU values and cover handling of CPU-produced (NumPy) data.

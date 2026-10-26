# Upstream-contribution bundle: two root-caused GPU defect fixes applicable to gpu4pyscf 1.8.1

Prepared 2026-10-24/25 by the RTX 4090 mainline-repair workstream (evidence dir
`2026-10-25-mr14-upstream-contribution`, preregistration
`preregistration-mr14.json` registered 2026-10-24T22:40:43Z before any
measurement in this unit).  All facts below were measured on this server;
nothing has been sent to upstream.  The operator decides whether/where to
contribute.

## What this bundle contains

| file | content |
|---|---|
| `0001-screen-index-race-barrier.patch` | one unconditional `__syncthreads()` at the `_screen_index` tile-loop top in `gpu4pyscf/lib/gdft/nr_eval_gto.cu` (10 lines incl. comment) |
| `0002-duplicate-pair-canonicalization.as-committed.patch` | `_canonicalize_duplicate_pair_columns` + one `DF.build()` call in `gpu4pyscf/df/df.py` (pure Python, 53 lines) |
| `0000-build-host-fortran-optional.patch` | OPTIONAL build-machinery-only patch: makes Fortran optional (SMD solvent target skipped) so hosts without gfortran can build; not a defect fix |

Both defect patches apply cleanly (`git apply --check`) to the pristine
upstream 1.8.1 source tree (git `0a4cf33`).

## Defect 1: `_screen_index` tile-cache race (patch 0001)

Mechanism (characterized at evidence `2026-10-23-mr9-anomaly-mechanism/` and
`2026-10-23-mr10-screen-race-fix/`): the kernel caches grid-coordinate tiles
in shared memory with a single barrier AFTER the tile load; a thread that
takes the `is_large` early-`continue` starts overwriting
`grid*_cache[thread_id]` while slower threads still consume tile k.  The race
is intermittent (scheduling-dependent), its STABLE output mode silently drops
truth-keep AO screen cells, and the corrupted fine screen propagates into
every DFT grid evaluation.

Measured impact on official gpu4pyscf 1.8.1 (wheel, unmodified):
- epinephrine/B3LYP/def2-SVP (247 AO), grids level 3, DF: total energy
  **1.932e-6 Ha off a pyscf 2.14.0 CPU-fp64 arbiter** with identical settings
  (CPU sides with the post-fix value at 1.61e-11; preregistered arbiter
  `cpu-arbiter-epinephrine-result.json`, batch `mr13-regression-20261023-b1`).
  The magnitude alone exceeds a 1e-6 Ha accuracy target on this cell.
- Cross-run/cross-process e_tot variation classes of order 1e-14..1e-12 Ha
  traced to the same kernel population (MR-9/MR-10 evidence).

## Defect 2: reversed duplicate AO-pair CDERI columns (patch 0002)

Mechanism (characterized at evidence
`2026-10-23-mr12-gpu-process-nondeterminism/`): `pair_and_diag_indices`
enumerates the full AO rectangle inside diagonal shell-pair blocks, so the
CDERI stores BOTH (i,j) and (j,i) columns for cross AO pairs of one shell.
The two columns hold the same integrals but reach them through different
contraction orders, differing at the last bits (<=3.34e-13 measured).
`fill_symmetric` writes both symmetric cells from every column, so the twin
writes race per element and the unpacked K tensor varies per launch —
a per-process e_tot nondeterminism envelope ~4.5e-13 Ha.  The official 1.8.1
package shows the identical structure (583 duplicate classes on
dopamine/def2-tzvp, 374 bitwise-unequal).  The J path is correct throughout
(diagonal-shell-block halving design).

The fix canonicalizes duplicate columns bitwise at build time (0.018 s at
73827 pairs), after which e_tot is measured bitwise-identical across fresh
processes (RHF 4/4, B3LYP 3/3, wB97M-V RSH+VV10 3/3).

## Verification in this unit (red -> green on a PRISTINE 0a4cf33 build)

Fresh CUDA 12.8 build of the pristine tree (+ optional patch 0000 only, to
configure on a host without Fortran), then measured on the RTX 4090:

- RED control (pristine, unpatched): epinephrine lane lands at the racy-stable
  value ~1.9e-6 Ha off CPU truth (the intermittent race may occasionally land
  clean; census evidence in MR-9/MR-10).
- GREEN (pristine + 0001): the lane moves to the CPU-truth value.
- GREEN (pristine + 0002): dopamine/def2-TZVP duplicate classes all
  bitwise-equal (pre-patch: hundreds unequal).

| measurement | result |
|---|---|
| RED: pristine epinephrine, unpatched | e = -630.7161527938961 — **1.932e-6 Ha off CPU truth** (9.1e-13 from the official-wheel value; 13 cycles, converged) |
| GREEN: pristine + 0001 only | e = -630.7161547258822 / -630.7161547258825 (2 runs) — **1.59e-11 / 1.61e-11 off CPU truth**; run 2 and the both-patches run are **bitwise-identical** to the post-fix production value |
| RED: pristine dopamine/def2-TZVP dup census | 583 duplicate classes, **374 bitwise-unequal**, max 3.344e-13 — exactly the official-wheel census |
| GREEN: pristine + 0002 | 583 classes, **0 bitwise-unequal**, max 0.0 |

Full adjudication: `final-adjudication.json` (all preregistered clauses HIT,
decision rule MET).  Gradient check: patched-tree gradient norms within
2.6e-9 absolute of the CPU arbiter (0.0606689 vs 0.0606692).

## Where the full evidence lives (private)

`/data/results/gpu4pyscf-agent/tasks/issue-9/2026-10-23-mr9-anomaly-mechanism/`,
`.../2026-10-23-mr10-screen-race-fix/`, `.../2026-10-23-mr12-gpu-process-nondeterminism/`,
`.../2026-10-23-mr13-regression-b1/` (CPU arbiters),
`.../2026-10-25-mr14-upstream-contribution/` (this unit).  Repo doc:
`docs/mainline-repair-20261021.md` sections MR-9 through MR-14.

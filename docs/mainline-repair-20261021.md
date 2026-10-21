# GPU correctness mainline recovery — 2026-10-21

Status: root-cause fix MR1 landed, formally retested (10/10 pass at the original six-knob configuration, batch mr1-retest-20261021-a), and the 30-cell cohort regression backfill is COMPLETE with all four preregistered prediction clauses holding. Broader regression coverage continues in staged batches. No independent acceptance, merge, default change, or gate relaxation.

## Three targeted failures

- Epinephrine/B3LYP/def2-SVP (charge 0, spin 0): the existing knob-isolation experiment attributed the energy shift to the opt-in XC grid block size 16384. This is not a radical failure.
- Methoxy/B3LYP/def2-SVP (charge 0, spin 1): a soft orbital direction permits a very small energy difference with a material density/force difference; energy convergence alone did not certify force accuracy.
- Methoxy/wB97M-V/def2-SVP (charge 0, spin 1): different SCF solutions produced similar energies but different forces. A lower-energy solution must not silently replace the identical-start FP64 reference contract.

The CPU-only ladder-extension study was reversibly parked with its checkpoint preserved. Mainline work returns to targeted fixes, GPU checks, adjacent controls, and full regression rather than further landscape scans.

## GPU safety pilot

The pilot used the frozen comparison source `21f61f9b1dd906b73ae9319d4f0803697d908fdd`; its production `gpu4pyscf/` source was checked identical to `5810867`. Each case has a separate strict-FP64 run and a candidate run, each with an independent warmup and three formal repeats. Every candidate repeat was compared against every reference repeat.

Candidate: the existing `auto+strict` reference-matching policy plus the measured opt-in stack **without** `GPU4PYSCF_XC_GRID_BLOCKSIZE=16384`. The other five opt-ins remain: VV10 fp32, fused DF ip1, DF ip1 fp32, gradient-grid fp32, and SCF VMAT headtail. This is an explicit safety workaround, not a claim that the block-size root cause is fully fixed.

| Case | Max energy error / Ha | Max gradient-component error / Eh/Bohr | Strict-FP64 median / s | Candidate median / s |
|---|---:|---:|---:|---:|
| Epinephrine / B3LYP | 1.3643e-12 | 5.0109e-6 | 3.697172 | 5.828105 |
| Methoxy / B3LYP | 0 | 1.5753e-6 | 1.421113 | 2.889218 |
| Methoxy / wB97M-V | 4.2349e-11 | 1.7902e-6 | 4.598944 | 7.454938 |

All formal samples converged and all candidate records have `strict_match_verified=true`. Both methoxy cases adopted the FP64 reference in all three repeats; epinephrine retained the certified main result. These adoptions and verification costs are included in candidate timings. The candidate is **1.58–2.03 times slower than this source-FP64 baseline**, not a new speedup result. GPU-to-GPU agreement is not a global physical-stability certificate.

Unchanged gates: energy <= 1e-6 Ha, maximum gradient-component difference <= 3e-5 Eh/Bohr, all repeats and references converged, full method/correction consistency, no unsafe allocations/OOM. Original failing rows in `uc-20261018-a` remain immutable; this pilot is separate evidence and does not turn that frozen batch green.

## Next implementation and regression

Add red-to-green GPU-backed regressions and a general, explicit opt-in guarded policy; do not hard-code molecular names or change global defaults. Fix or safely isolate block-size dependence, preserve the actual initial-density contract, and certify gradients after reference adoption. Then test the additional optimized-FP64 failure methoxy/M06-2X/def2-TZVP, neighboring RKS/UKS cases and all four methods; measure real fallback costs before claiming throughput improvements.

## Root-cause fix MR1 (commit 8b6a5c2, branch fix/mainline-repair-20261021)

Two mechanisms were identified, fixed, and regression-tested; gates and defaults are unchanged.

**Root cause 1 — block size changed physics through the AO screen.** A loop block larger than `MIN_BLK_SIZE` (4096) evaluated the UNION of the covered default blocks' screen shell sets on every point of the coarse block, silently keeping tail contributions the default partition drops. Reproduced standalone on epinephrine/B3LYP/def2-SVP at fixed dm: `excsum` shifted -1.9e-6 Ha, max|dvmat| 1.1e-4, `nelec` +7.9e-6; with the screen disabled (`AO_THRESHOLD=1e-300`) both partitions were bitwise identical, proving the screen was the sole mechanism (the block-union arithmetic is exact). The fix (`gpu4pyscf/dft/numint.py::_block_loop` + `gen_grid.py::get_partition_fix`) zeroes the AO rows each covered fine block's own screen dropped, inside that fine block's point range, so the block size changes only GEMM batching and reduction order (floor ~4e-15), never the screened physics. Post-fix the epinephrine SCF pair is bit-identical at 4096 vs 16384. Honest finding, recorded without a default change: on epinephrine the DEFAULT 4096 screen itself sits +1.9e-6 Ha above an unscreened integral; both lanes and every gate use the default partition, so this is an accuracy budget question for a separate decision, not a gate failure.

**Root cause 2 — strict fp64-reference siblings inherited process-global knobs.** The reference trajectory inside `auto+verify`/`auto+strict` ran with the armed block-size knob and the C-side process-lifetime fp32 VV10 UWE kernel selection, so it was not the default-lane solution the strict-match contract compares against (pilot-measured 4.2e-11 sibling displacement on methoxy/wB97M-V). The fix (`numint.strict_fp64_reference_ctx`, a new `VV10_uwe_kernel_set_override` export in libgdft, wrapped around both sibling trajectories in `scf/stability_verify.py`) forces the default partition and the reference warp1 UWE kernel during reference runs and restores everything on exit.

Tests: `tools/tests/test_mainline_repair_20261021.py` (7 GPU tests, red pre-fix at exactly the failing magnitudes, green post-fix), plus the full tools suite 806/806 on the fixed source.

## MR1 formal retest (batch mr1-retest-20261021-a, preregistered before measurements)

Ten cells at the ORIGINAL failing configuration — the full six-knob optimized stack INCLUDING `GPU4PYSCF_XC_GRID_BLOCKSIZE=16384` — against the fixed source (commit 8b6a5c2): each cell a strict-FP64 reference (clean env) and an `auto-strict` candidate, three formal samples each with independent warmup, every candidate repeat compared against every reference repeat (3x3, independently recomputed; runner rc=0 never counted as a pass).

| Case | worst \|dE\| / Ha | worst max\|dg\| / Eh/Bohr | strict match | FP64 median / s | Candidate median / s |
|---|---:|---:|---|---:|---:|
| epinephrine / B3LYP (failure 1) | 9.1e-13 | 5.07e-6 | 3/3 | 3.85 | 5.42 |
| methoxy / B3LYP (failure 2) | 0 | 1.58e-6 | 3/3 | 1.16 | 2.32 |
| methoxy / wB97M-V (failure 3) | 8.5e-14 | 1.95e-6 | 3/3 | 4.40 | 5.69 |
| methoxy / M06-2X / def2-TZVP (extra) | 9.6e-12 | 1.60e-6 | 3/3 | 1.87 | 4.19 |
| epinephrine / M06-2X | 2.0e-11 | 6.62e-6 | 3/3 | 5.35 | 8.09 |
| epinephrine / wB97M-V | 1.0e-10 | 5.79e-6 | 3/3 | 52.92 | 57.33 |
| dopamine / B3LYP | 8.4e-12 | 6.71e-6 | 3/3 | 2.52 | 4.23 |
| ethoxy / B3LYP | 4.2e-12 | 4.39e-6 | 3/3 | 1.34 | 2.72 |
| oxygen triplet / B3LYP | 5.7e-14 | 3.21e-6 | 3/3 | 0.71 | 1.18 |
| methoxy / r2SCAN-3c | 2.7e-12 | 1.20e-6 | 3/3 | 1.63 | 3.90 |

10/10 pass under the unchanged gates (|dE| <= 1e-6 Ha, max|dg| <= 3e-5 Eh/Bohr, all repeats and references converged, `strict_match_verified` on every candidate repeat). The block-size knob is now physics-neutral at these cells, so the pilot's knob-removal workaround is no longer the only safe configuration — but the six-knob stack is still NOT a new default (per-knob benefit claims remain those of the frozen comparison batch). The candidate lane's verification overhead (reference SCF per sample) is 1.3x-2.2x the plain FP64 lane on these cells and is included in the candidate timings; no mixed-precision speedup is claimed from this batch. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## Publication repair

The manifest generator and protected publisher file-count caps are raised together from 100 to **256**, allowing the current >100-file cumulative patch. Tests accept 106 and 256 files, reject 257, and retain the 1 MB per-file / 5 MB total-content limits. Path/hidden-file restrictions, credential scanning, fixed-repository transport, credential isolation, draft-only publication and manual merge/acceptance restrictions are unchanged.

## Cohort regression backfill (batch mr1-backfill-20261021-a, COMPLETE 2026-10-21T14:54:59Z)

Thirty frozen `uc-20261018-a` cells — the failure/control molecule strata (methoxy, ethoxy, dopamine, epinephrine, oxygen_triplet; identical settings, Latin squares and cell indices; schedule sha256 `b22caccc`, preregistration registered 10:36:01Z before the first measurement) — re-run at the fixed source `44cb789` under the four-arm balanced-blocks protocol (O/B/C/R, four formal samples per role). The driver ran 10:38Z-14:55Z; every role-row comparison below is the worst over the 4x4 repeat cross, recomputed from the raw sample records by the strict-tag adjudicator (rc=0 never counted as a pass; charge/spin variant files never cross-compared).

All four preregistered prediction clauses hold on 30/30 cells:

- **R (clean fp64)**: reproduces the frozen R lane (worst U-vs-R dE 9.1e-13, gradient 2.5e-8 class everywhere).
- **B (six-knob fp64, blocksize on)**: every stack-carrying epinephrine energy deviation dropped to the reduction floor at the ORIGINAL six-knob stack — the MR1 target:

  | Cell | frozen B dE / Ha | fixed-source B dE / Ha |
  |---|---:|---:|
  | epinephrine/B3LYP/def2-SVP | 1.932e-06 | 4.5e-13 |
  | epinephrine/M06-2X/def2-SVP | 7.164e-07 | 1.1e-12 |
  | epinephrine/wB97M-V/def2-SVP | 2.491e-07 | 8.4e-11 |

  The other 27 cells were already at the floor and stay there (largest B dE 1.1e-10). B energy passes <= 1e-6 on all 30 cells.
- **C (six-knob auto)**: epinephrine C energies also dropped to the floor (1.1e-12 / 2.0e-11 / 1.0e-10). The methoxy soft-mode/branch gradient deviations persist BY DESIGN in the plain auto lane (no reference adoption): methoxy/B3LYP B 3.6e-4 / C 1.3e-3, methoxy/M06-2X/TZVP B 1.2e-4, methoxy/wB97M-V B 7.8e-5 / C 3.0e-2 Eh/Bohr — recorded honestly as the four known soft-mode rows (energies at floor: 3.9e-9 / 1.4e-8 / 1.7e-13 Ha); the certified safe route for those cells is the `auto-strict` policy measured in mr1-retest (10/10 with `strict_match_verified`). 27/30 cells pass all three roles.
- **O (official 1.8.1)**: unchanged implementation-agreement rows (worst O-vs-R dE 9.1e-13).

The intermittent partition-fix fail-closed guard fired twice in ~170 blocksize-carrying arms during the batch (retry-clean both times; attempts preserved); a dedicated bounded GPU repro (2 exact failing arm commands x 40 fresh processes) was staged to measure the firing rate. Its first two launch attempts failed on launcher defects, both fixed and preserved as erratum evidence (a CPU-only tool pattern matching the user-parked LC-215 process at pre-flight; a missing six-knob `STACK_ENV` that the arm's own env-discipline gate rejected in ~0 s with zero GPU time). The measured protocol itself was never altered.

MAINLINE-REPAIR-7 regression-coverage batch 1 is staged (not launched): the 22 highest frozen gate-margin remainder cells (of 183 = 213 - 30), schedule sha256 `549bdea3`, preregistered 13:38:20Z, calibrated ~4.2 h.

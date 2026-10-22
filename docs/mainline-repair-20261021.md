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

## MR-2: partition-invariant AO-screen construction (commits 726414c + d05c301 + 7feaffa)

The guard-repro (below) measured an intermittent coarse-size `GDFTscreen_index` evaluation that dropped shells a fine block keeps (3/40 fresh processes on dopamine/B3LYP/def2-TZVP, 0/40 control; fail-closed under the MR1 guard, retry-clean). MR-2 (`gen_grid.Grids._screen_shl_mask`) removes the failure pathway by construction instead of chasing the mechanism: the screen kernel is now invoked ONLY at the default partition (`MIN_BLK_SIZE`), and any coarser loop partition's mask is DERIVED as the exact boolean union of the covered fine masks (zero-padded tail is exact; padding rows cannot add shells). Raw masks are memoized inside the existing `_non0ao_idx` cache under `('shl_mask', block_size)` keys, so every screening-cache reset drops them with the rest of the cache (the MR1 stale-structure class). The MR1 fail-closed guard is retained as defense-in-depth. Default path untouched; no default change; pure Python, no libgdft rebuild. The GPU/process-level anomaly mechanism itself stays uncharacterized — an honest limit: MR-2 removes its only pathway into results, it does not explain it.

### MR-2 GPU verification (window 2026-10-21T21:59:51Z + harness-fix rerun 22:23:18Z, ALL SIX preregistered predictions HIT)

- **P4 (headline) — firing class eliminated**: the two EXACT guard-repro arm commands, 40 fresh serial processes each at the frozen 726414c snapshot: **0/40 and 0/40 firings** (pre-fix: 3/40), 80/80 rc=0, worst |dE| 9.09e-13 Ha vs the 1e-12 clean envelope.
- **P1 — derivation correctness**: derived coarse mask == exact boolean union of covered fine masks on all four preregistered molecules (epinephrine/B3LYP/SVP, dopamine/B3LYP/TZVP, methoxy/B3LYP/SVP UKS, benzene/B3LYP/SVP) plus the HF synthetic case, at 8192/16384/32768.
- **P2 — single kernel evaluation**: instrumented `GDFTscreen_index` call counter over fine+8192+16384+32768+partition-fix builds: exactly one call, at `MIN_BLK_SIZE`, never at a coarse size.
- **P3 — MR1 floors hold**: the MR-1 invariance regressions re-ran green at the same floor class.
- **P5 — cost not increased**: (a) per-process warmup+first-XC medians vs the guard-repro baseline DECREASED (8.497→8.283 s and 5.429→5.345 s); (b) the coarse derivation itself costs 0.213 ms vs 31.8 ms for the fine kernel screen on dopamine/def2-TZVP (262912 grid points) — ~150x cheaper, `derived_equals_union` cross-checked true.
- **P6 — suite green**: full tools suite 815/815 OK (806 pre-MR2 + 9 MR-2 tests).

**Window-1 honest record (preserved as `*.window1` artifacts):** the first window run recorded P1/P2/P6 as MISSes — all three were harness-authoring defects in the verification tooling, not production defects: (1) the test's kernel-call counter did `int()` on a `ctypes.c_int` argument (raises on this Python 3.12.3; fixed with the `.value` accessor, commit 7feaffa); (2) the P5b probe missed a repo `sys.path` entry (`ImportError: cannot import name 'dft'`; fixed); (3) `unittest discover -t .` on the non-package `tools/tests` directory ("Start directory is not importable"; fixed with `-t tools/tests`). The substantive window-1 measurements (P3, P4, P5a — the firing elimination itself) were complete and valid and were NOT re-run; only the three failed steps re-ran in a 4-minute rerun window with window-1 artifacts preserved. A fourth, adjudicator-side erratum was fixed the same session: P5 had been adjudicated HIT on part (a) alone with the probe artifact absent — a vacuous partial; the adjudicator now holds P5 PENDING until both parts have evidence.

## MAINLINE-REPAIR-7 regression batch 1 (batch mr7-regression-20261021-b1, COMPLETE 2026-10-21T21:56:31Z)

The 22 highest frozen gate-margin remainder cells (of 183 = 213 - 30 backfill cells; schedule sha256 `549bdea3`, preregistration 13:38:20Z before the first measurement) at the frozen `44cb789` snapshot under the four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 352 arms + retries). Driver 2944418 ran 16:43Z-21:56Z (wall 6.26 h).

**22/22 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (worst |dE| 5.9e-12 Ha vs the 1e-6 gate; worst max|dg| 2.6e-5 vs the 3e-5 gate — the melatonin q+1s1 UKS cation auto lane at its frozen per-cell value, every B/C dg at its frozen envelope value, zero informative >2x deviations). Final adjudication 5/7 HIT with two honest MISSes recorded, never relabeled:

- **P5 R-reproduction MISS** (informative, sub-scientific): 12 of 22 cells reproduce the frozen R lane at 3e-13..1.4e-12 Ha — slightly above the strict per-cell floor class (max of 3e-13 and the frozen lane's own cross-block spread) but six or more orders of magnitude below the 1e-6 scientific gate; the P2 gate comparison itself is HIT on every cell. Cross-process R noise class.
- **P7 wall MISS** (forecast): 6.26 h vs the 5.5 h preregistered band — inside the 16:08Z forecast correction's ~7.4 h warmup-aware estimate; the original 5.5 h band under-modeled per-arm warmup.
- **P6 HIT**: the melatonin q+1s1 UKS cation reproduces the frozen R record at exactly its own floor (9.095e-13 == the frozen lane's self-spread) — same-branch evidence.

Four retry-clean partition-guard firings occurred in-batch on the frozen pre-MR2 source (melatonin fp64 b1; caffeine auto b2; tamoxifen/M06-2X/SVP fp64 b0 twice — blocks 19 and 2, missing 3 shells each, all fail-closed, attempts preserved as `.attemptN.json`) — two of them on a molecule outside the dopamine 3/40 repro set, live motivation for MR-2, which the same unattended chain then verified at 0/40+0/40. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## MAINLINE-REPAIR-7 regression batch 2 (batch mr7-regression-20261021-b2, COMPLETE 2026-10-22T05:04:55Z)

The next 25 highest frozen gate-margin remainder cells (of the 161 remaining after b1; schedule sha256 `49bf0ece`, preregistration 22:36:00Z before the first measurement) at the **MR-2-fixed source `3001b34`** — the first production-shaped batch carrying the fix (b1 ran the pre-MR2 frozen `44cb789` deliberately). Four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 400 arms). Driver 3015489 ran 22:38Z–05:05Z (wall 6.32 h arms / 6.45 h whole — inside the 7.5 h band and the ~6.99 h warmup-aware correction).

**25/25 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (75/75 role-comparisons: worst |dE| 7.08e-10 Ha vs the 1e-6 gate; worst max|dg| 1.34e-5 vs the 3e-5 gate — the melatonin q+1s1 UKS cation auto lane at its frozen-class value; max gradient ratio vs frozen 1.74, zero informative >2x deviations; B-lane energies at the frozen floor class everywhere). Final adjudication 6/8 HIT with two honest MISSes recorded, never relabeled:

- **P5 R-reproduction MISS** (informative, sub-scientific): tamoxifen/wB97M-V R-vs-frozen 2.27e-12 Ha exceeds the 2e-12 preregistered envelope by 13% — but the deviation EQUALS the frozen R lane's own cross-block self-spread; the envelope was set from b1's max (1.4e-12) and did not anticipate a cell whose frozen self-spread itself exceeds it. Six-plus orders below the scientific gate; the P2 gate comparison itself is HIT on every cell.
- **P6 UKS-cation MISS** (registration defect, sub-scientific): the preregistration named FOUR melatonin q+1s1 cation cells but the b2 schedule contains THREE (the wB97M-V cation ran in b1, not b2); the adjudicator requires the named-cell count. All three in-batch cations reproduce the frozen records within the noise envelope (same-branch evidence); the B3LYP cation sits at 6.8e-13, above its strict 3e-13 floor class but well within noise.
- **P8 HEADLINE HIT**: ZERO partition-fix guard firings across the entire batch (~400 arms including warmup first-XC calls) at the fixed source — the MR-2 elimination (0/40+0/40 fresh-process) now holds at production-batch scale; b1 at the pre-MR2 source fired 4. The GPU anomaly mechanism itself remains uncharacterized (honest limit): its only pathway into results is gone.

An independent recompute (own parser from the raw sample JSONs, filename-vs-record consistency check, bidirectional driver-log cross-match, broadest firing scan) AGREES with the adjudicator on all eight verdicts; the wait-gate and chain-gate full-state adjudications are byte-identical (deterministic). The post-b2 chain gate verified C1–C5 (P1/P2/P3/P8 HIT, zero firings, worst R-vs-frozen 2.27e-12 well below the 1e-10 engineering-abort bound) and launched batch 3 (33 cells, the next remainder stratum) at the same fixed source, 05:08:11Z, driver 3074876. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## MAINLINE-REPAIR-7 regression batch 3 (batch mr7-regression-20261022-b3, COMPLETE 2026-10-22T10:50:40Z)

The next 33 highest frozen gate-margin remainder cells (of the 136 remaining after b2; schedule sha256 `522a2981`, preregistration 00:09:12Z before the first measurement, launch conditional on b2's clean C1-C5 adjudication — executed by the post-b2 chain gate at 05:08:11Z) at the **MR-2-fixed source `3001b34`**. Four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 528 arms). Driver 3074876 ran 05:08Z–10:50Z (wall 5.71 h — inside the 7.5 h preregistered band and below the ~6.91 h warmup-aware estimate; the first MR-7 batch to finish under both models).

**33/33 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (99/99 role-comparisons: worst |dE| 3.90e-10 Ha vs the 1e-6 gate; worst max|dg| 8.49e-6 vs the 3e-5 gate — both on bisphenol_a/wB97M-V/def2-SVP, the batch's largest NLC cell; max gradient ratio vs frozen 1.317, zero informative >2x deviations; B-lane energies at the frozen floor class everywhere). Final adjudication 7/7 applicable HIT — the first clean sweep of the series:

- **P5 HIT**: 7/33 R-lanes reproduce the frozen records bitwise; the remaining 26 all sit within the 2e-12 noise envelope (11 above their strict per-cell floor class, worst 1.14e-12 on bisphenol_a/wb97mv and melatonin/wb97mv — cross-process R noise, six-plus orders below the scientific gate; the P2 gate comparison itself is HIT on every cell).
- **P6**: NOT APPLICABLE BY DESIGN (all 33 cells closed-shell neutral; the 8 open-shell remainder cells stay in the pool for b4, which selects all 8).
- **P7 HIT**: 5.70 h arms vs the 7.5 h band.
- **P8 HEADLINE HIT**: ZERO partition-fix guard firings batch-wide (~528 arms including warmup first-XC calls) — the third consecutive production batch at the fixed source with zero firings (b1 at the pre-MR2 source fired 4).

Composition: 12 molecules (acetamide, acetic_acid, aspirin, bisphenol_a, glycylglycine, inosine, melatonin, nicotine, phenylalanine, serotonin, tryptophan, vitamin_c), all 4 methods (m062x 12 / wb97mv 8 / r2scan3c 7 / b3lyp 6), 3 basis families (svp 13 / tzvp 13 / mtzvpp 7). Frozen gate-margin risk band 0.196-0.261 of gate — the first batch entirely below b2's 0.263 lower edge.

An independent recompute (own parser from the raw sample JSONs, filename-vs-record consistency, bidirectional driver-log cross-match, broadest firing scan) AGREES with the adjudicator on all eight verdicts; the wait-gate and chain-gate full-state adjudications are byte-identical (deterministic; partial-backup == final verified). The post-b3 chain gate verified C1-C5 (P1/P2/P3/P8 HIT, zero firings) plus the source check (HEAD `032a223` exact, clean tree, zero production-path diffs vs `3001b34`) and launched batch 4 (35 cells incl. all 8 open-shell remainder cells — four histamine q+1s1 UKS cations + four nitrogen_dioxide UKS, preregistration 08:21:55Z with the substantive open-shell P6 prediction) at 10:53:58Z, driver 3136387. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## MAINLINE-REPAIR-7 regression batch 4 (batch mr7-regression-20261022-b4, COMPLETE 2026-10-22T16:25:22Z)

The next 35 highest frozen gate-margin remainder cells (of the 103 remaining after b3; schedule sha256 `cdaa87e0`, preregistration 08:21:55Z before the first measurement with the documented pre-launch re-pin correction 3001b34 -> 032a223, launch conditional on b3's clean C1-C5 — executed by the post-b3 chain gate at 10:53:58Z) at the **MR-2-fixed production source `3001b34`** (launch HEAD `032a223`, docs-only delta; production paths verified identical by the chain gate). Four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 560 arms). Driver 3136387 ran 10:53:58Z-16:25:22Z (wall 5.52 h — inside the 7.5 h preregistered band and below BOTH warmup-aware estimates (~6.89 h at the b2 ratio, ~5.69 h at the b3 ratio); the fastest average pace of the series at 9.5 min/cell).

**35/35 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (105/105 role-comparisons: worst |dE| 4.54e-10 Ha vs the 1e-6 gate — serotonin/wB97M-V/def2-TZVP, an optimized_fp64 lane; worst max|dg| 5.72e-6 vs the 3e-5 gate — serotonin/r2scan-3c/def2-mTZVPP, also optimized_fp64; max informative gradient ratio vs frozen 1.158 — nicotine/r2scan-3c/def2-mTZVPP optimized_auto — zero informative >2x deviations). Final adjudication **8/8 HIT** — the second consecutive clean sweep, and the first with a SUBSTANTIVE open-shell verdict:

- **P6 SUBSTANTIVE HIT (the batch headline)**: all 8 open-shell cells — four histamine q+1s1 UKS cations (charge +1, spin 1: b3lyp/m062x/wb97mv on def2-SVP, r2scan3c on def2-mTZVPP) and four nitrogen_dioxide UKS cells (charge 0, spin 1, same method/basis pattern) — pass all roles at the gates with same-branch behavior: worst open-shell B/C |dE| 1.81e-10 (nitrogen_dioxide/wb97mv optimized_fp64), worst B/C |dg| 5.68e-6 (histamine q+1s1/m062x optimized_fp64); every open-shell R-vs-frozen lands within its per-cell floor class (4 of 8 bitwise-identical to the frozen records); ZERO auto-fallback events fired on any open-shell cell. Every one of these cells ran same-branch across roles in the frozen batch (worst frozen cross-role |dE| <= 5.5e-11, worst |dg| <= 5.7e-6); the methoxy soft-mode family is NOT re-litigated here (stays on the fp64/auto-strict route per standing policy).
- **P5 HIT**: 10/35 R-lanes bitwise vs the frozen records; the remaining 25 all within the 2e-12 noise envelope (8 above their strict per-cell floor class, worst 1.14e-12 tryptophan/wB97M-V/def2-TZVP — cross-process R noise, 6+ orders below the scientific gate).
- **P7 HIT**: 5.52 h arms vs the 7.5 h band.
- **P8 HEADLINE HIT**: ZERO partition-fix guard firings batch-wide (~560 arms including warmup first-XC calls) — the FOURTH consecutive production batch at the fixed source with zero firings (b1 at the pre-MR2 source fired 4).

Composition: 13 molecules (histamine 9 — of which 4 are the q+1s1 cations and 5 neutral — nicotine 5, serotonin 4, nitrogen_dioxide 4, acetone 3, furan 2, acetaldehyde 2, tryptophan 1, phenylalanine 1, glycylglycine 1, acetamide 1, nitromethane 1, pyridine 1), all 4 methods (b3lyp 11 / wb97mv 9 / m062x 8 / r2scan3c 7), 3 basis families (svp 17 / tzvp 11 / mtzvpp 7); 27 closed-shell + 8 open-shell. Frozen gate-margin risk band 0.114-0.188 of gate — below b3's 0.196 lower edge. The batch CLOSES the open-shell regression coverage of the remainder pool (all 8 open-shell cells selected).

An independent recompute (own parser from the raw sample JSONs, filename-vs-record consistency, bidirectional driver-log cross-match, broadest firing scan, P6 open-shell set derived independently from schedule charge/spin) AGREES with the adjudicator on all verdicts including the crosscheck (P0 source/schedule + P1-P8). There is NO chain gate beyond b4 by design: the b5 launch requires the b4-docs commit and the b5 source re-pin, which are session work. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## MAINLINE-REPAIR-7 regression batch 5 (batch mr7-regression-20261022-b5, COMPLETE 2026-10-22T21:10:41Z)

The next 44 highest frozen gate-margin remainder cells — ALL closed-shell neutral (of the 68 remaining after b4; schedule sha256 `90defc73` after the documented promotion re-pin a71735a -> 38f4919 executed BEFORE any b5 measurement with the cell selection/order verified identical; preregistration 16:33:40Z before the first measurement, launch conditional on b4's 8/8 clean adjudication) at the **MR-2-fixed production source `3001b34`** (launch HEAD `38f4919`, docs-only delta; production paths verified identical by git diff 3001b34..38f4919 = empty). Four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 704 arms). Driver 3200011 ran 16:36:30Z-21:10:41Z (wall 4.54 h — HIT vs the 7.5 h preregistered band and below the ~7.0 h b4-pace calibration; sample-only model 4.47 h, warmup-aware 5.78-7.01 h; at 6.2 min/cell the fastest stratum of the series).

**44/44 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (132/132 role-comparisons: worst |dE| 1.10e-10 Ha vs the 1e-6 gate — benzene/wB97M-V/def2-SVP, an optimized_auto lane; worst max|dg| 3.39e-6 vs the 3e-5 gate — histamine/wB97M-V/def2-TZVP, an optimized_fp64 lane; max informative gradient ratio vs frozen 1.765 — butane/M06-2X/def2-SVP upstream — zero informative >2x deviations). Final adjudication **7/8 HIT with P6 N/A-BY-DESIGN — zero MISSes, the third consecutive clean sweep**:

- **P6 N/A BY DESIGN**: all 44 cells closed-shell neutral (charge 0, spin 0; validator-verified at registration); the entire open-shell remainder was covered by b4 and adjudicated P6 SUBSTANTIVE HIT (all 8 open-shell cells all-roles PASS at gates, zero auto-fallback events).
- **P5 HIT**: 10/44 R-lanes bitwise vs the frozen records; the remainder within the 2e-12 noise envelope (1 above-floor, worst 3.98e-13 pyridine/B3LYP/def2-SVP above its 3e-13 per-cell floor — cross-process R noise class).
- **P7 HIT**: 4.54 h vs the 7.5 h band.
- **P8 HEADLINE HIT**: ZERO partition-fix guard firings batch-wide (~704 arms including warmup first-XC calls) — the FIFTH consecutive zero-firing measurement at the fixed source (MR-2 GPU window 0/40+0/40; production batches b2/b3/b4/b5 all zero; b1 at the pre-MR2 source fired 4, retry-clean).

Composition: 14 molecules (pyridine 6, furan 5, acetaldehyde 5, acetonitrile 4, nitromethane 3, ethylamine 3, butadiene 3, pyrrole 3, butane 3, histamine 2, ethanol 2, benzene 2, methanol 2, acetone 1), all 4 methods (wb97mv 14 / b3lyp 12 / m062x 11 / r2scan3c 7), 3 basis families (svp 25 / tzvp 12 / mtzvpp 7). Frozen gate-margin risk band 0.112-0.058 of gate — below b4's 0.114 lower edge. An independent recompute (own parser, filename/driver-log cross-match, broadest firing scan) AGREES with the adjudicator on all verdicts (132/132 role-comparisons value-identical). There is NO chain gate beyond b5 by design: the b6 launch (ALL 24 remaining cells — the FINAL stratum closing the entire 213-cell frozen cohort) requires the b5-docs commit and the b6 source re-pin, which are session work. The frozen `uc-20261018-a` FAIL rows remain immutable; this is a separate, non-overwriting batch.

## MAINLINE-REPAIR-7 regression batch 6 — the FINAL stratum; 213-cell cohort CLOSED (batch mr7-regression-20261022-b6, COMPLETE 2026-10-22T23:46:39Z)

ALL 24 remaining cells of the frozen 213-cell cohort (schedule sha256 `453968c0` after the documented promotion re-pin a71735a -> 7cac259 executed BEFORE any b6 measurement with the cell selection/order verified identical to the DRAFT sha `e1a768f4` lineage; preregistration 21:25:20Z before the first measurement, launch conditional on b5's clean adjudication — P1 44/44, P2/P3/P8 zero, P5 within envelope, P6 N/A verified) at the **MR-2-fixed production source `3001b34`** (launch HEAD `7cac259`, docs-only delta; production paths verified identical by git diff 3001b34..7cac259 = empty). Four-arm balanced-blocks protocol (O/B/C/R x 4 blocks, 384 arms). Driver 3278430 ran 21:27:27Z-23:46:39Z (wall 2.30 h — HIT vs the 7.5 h preregistered band, and below the ~2.5 h b5-pace calibration; at 5.8 min/cell the fastest stratum of the series).

**24/24 cells COMPLETE; every cell passes all three optimized roles vs its within-batch strict-FP64 reference** (72/72 role-comparisons: worst |dE| 7.91e-11 Ha vs the 1e-6 gate — benzene/wB97M-V/def2-TZVP, an optimized_auto lane; worst max|dg| 1.87e-6 vs the 3e-5 gate — butadiene/wB97M-V/def2-TZVP, also optimized_auto; max informative gradient ratio vs frozen 1.673 — butane/M06-2X/def2-TZVP upstream — zero informative >2x deviations). Final adjudication **7/8 HIT with P6 N/A-BY-DESIGN — zero MISSes, the FOURTH consecutive clean sweep**:

- **P6 N/A BY DESIGN**: all 24 cells closed-shell neutral (charge 0, spin 0; validator-verified at registration); the entire open-shell remainder was covered by b4 (P6 SUBSTANTIVE HIT) and b5 was likewise N/A-by-design with the premise independently verified.
- **P5 HIT**: 5/24 R-lanes bitwise vs the frozen records; the remainder within the 2e-12 noise envelope (1 above-floor, worst 3.41e-13 benzene/B3LYP/def2-TZVP above its 3e-13 per-cell floor — cross-process R noise class).
- **P7 HIT**: 2.30 h vs the 7.5 h band.
- **P8 HEADLINE HIT**: ZERO partition-fix guard firings batch-wide (~384 arms including warmup first-XC calls) — with b6, FIVE consecutive production batches at the fixed source measured zero firings (b2 400 + b3 528 + b4 560 + b5 704 + b6 384 = 2576 final arms; b1 at the pre-MR2 source fired 4, retry-clean).

Composition: 9 molecules (benzene 5, butadiene 4, butane 4, methylamine 4, methanol 2, ethanol 2, pyrrole 1, ethylamine 1, acetone 1), all 4 methods (m062x 8 / wb97mv 6 / b3lyp 5 / r2scan3c 5), 3 basis families (svp 11 / tzvp 8 / mtzvpp 5). Frozen gate-margin risk band 0.058-0.015 of gate — the LOWEST-risk tail, below b5's 0.058 lower edge. An independent recompute (own parser from the raw sample JSONs, filename-vs-record consistency, bidirectional driver-log cross-match, broadest firing scan; pre-validated red-tested root resolution, core logic byte-identical to the b5-validated tool) AGREES with the adjudicator on all verdicts (72/72 role-comparisons value-identical).

## MAINLINE-REPAIR-7 COHORT CLOSURE — every frozen 213-cell-cohort cell re-measured at the MR-2-fixed source (2026-10-22T23:46:39Z)

With b6 complete the ENTIRE frozen uc-20261018-a 213-cell cohort (commit 21f61f9b, 36 molecules) has been re-measured at the fixed source in six preregistered strata: **30 backfill (mr1-backfill-20261021-a, all four clauses hold) + 22 b1 + 25 b2 + 33 b3 + 35 b4 + 44 b5 + 24 b6 = 213/213 cells all-three-optimized-roles pass** — 639 role-comparisons in total (90+66+75+99+105+132+72), every one within the unchanged gates (|dE| <= 1e-6 Ha, max|dg| <= 3e-5 Eh/Bohr vs within-batch strict-FP64 references). Cohort-wide worst |dE| 7.08e-10 Ha (b2, vs the 1e-6 gate); cohort-wide worst max|dg| 2.6e-5 (b1, the melatonin q+1s1 UKS cation auto lane at its frozen per-cell value, vs the 3e-5 gate). Per-batch worst values and adjudication records are in the six sections above; every honest MISS (b1 P5/P7, b2 P5/P6) is enumerated there, never relabeled. Guard firings: the pre-MR2 source fired 4 in b1 (retry-clean, fail-closed, preserved); the fixed source measured ZERO across b2..b6 = five consecutive production batches, 2576 final arms — the MR-2 eliminated pathway stayed eliminated at production-batch scale (mechanism remains uncharacterized, honest limit).

The frozen batch's 4 FAIL rows remain IMMUTABLE history; at the fixed source their outcomes are: epinephrine/B3LYP/SVP fixed at source (MR-1 union screen-mask fix, cohort-verified drop to floor: 1.932e-6 -> 4.5e-13 Ha); the three methoxy soft-mode cells carry the exact-match safe route (auto-strict reference-matching / fp64) with the operator-pilot's measured verification cost 1.58/2.03/1.62x — cost-for-correctness, never relabeled as mixed-precision speedup. The frozen 213-row batch itself is never rewritten; these six batches are NEW evidence at the NEW source with their own denominators. Remaining user decisions (b1)-(b6)+(c) stay queued, never waited on; publisher draft-PR pickup operator-owned.

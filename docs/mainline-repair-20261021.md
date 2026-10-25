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

## MAINLINE-REPAIR-8: verification-cost reduction — the strict-fp64 verify sibling reuses the main run's CDERI (commit b144083, measurement 2026-10-23T01:33Z)

The mainline card's open tail item (再压低验证成本). Profile-first decomposition of the operator pilot's auto-strict lanes located the verification overhead: the strict-fp64 sibling SCF (the same-start reference — the contract's structural floor, irreducible per invocation) plus a **duplicate 3c2e Cholesky CDERI build** in every sibling (`_build_fp64_sibling` constructed a fresh DF object, rebuilding the tensor the main run had already built; pilot-measured 0.066–0.186 s per invocation on the three precision cells; 2.611 s + a 0.999 s range-separated sub-build at 537 AO). MR-8 removes the duplicate: when the main run's CDERI is a built fp64 tensor, the sibling receives the main run's DF object verbatim (`density_fit`'s documented `with_df=` sharing API). Refusal preconditions fall back to the original fresh-build path: a process armed for fp32 CDERI (`precision.set_cderi_precision`, screening-grade only), any non-fp64 stored tensor (including range-separated sub-objects), or an unbuilt state. No SCF-path env knob affects the build (the DF_IP1 knobs are gradient-side; grid-block/VV10 are neutralized by `strict_fp64_reference_ctx`), and the build is deterministic for identical mol/auxbasis, so the shared tensor is bitwise what the sibling would have built — regressed directly (`test_deterministic_cderi_build_bitwise`).

**Verification of the change (13 new red→green tests, `gpu4pyscf/scf/tests/test_stability_verify_cderi_reuse.py`)**: precondition logic (fp32 tensor / process flag / unbuilt refusals); object-identity sharing; and the soundness core — shared-vs-fresh parity, **bitwise on b3lyp/closed-shell**, noise-floor ≤1e-12 on the wb97mv UKS split cell **where two DISABLED-sharing control runs differ by 3e-14 themselves** (measured same-day: cross-run GPU nondeterminism on the range-separated path, not a sharing effect). Existing suites green: stability_verify 45+13, mainline-repair 7, gradient runner/validator 44, tools 815 all OK.

**Paired before/after measurement** (preregistered sha `5a77d482` BEFORE measurement: before = ff129b8 git worktree, after = b144083; cells epinephrine/B3LYP/def2-SVP 247 AO + methoxy/B3LYP + methoxy/wB97M-V + tamoxifen/wB97M-V 537 AO; lanes strict-fp64 + auto-strict under the pilot's exact 5-knob env; 1 warmup + 2–3 formal samples each, fresh subprocess per sample, DF.build instrumented, peak memory polled; driver 08:36–09:33Z, 60/60 samples rc=0; wait-gate auto-ran the registered adjudicator). **P1 mechanism HIT everywhere**: DF.build counts 2→1 on b3lyp cells and 4→2 on wb97mv cells (RSH sub-builds counted), reuse counters confirm, per-invocation build-time savings **0.099 / 0.136 / 0.231 / 2.572 s** (methoxy-b3lyp / methoxy-wb97mv / epinephrine / tamoxifen); tamoxifen t_scf median 247.3→245.6 s. **Every in-run strict-match contract clause HIT on all 60 samples**: all converged, `strict_match_verified=true` on every auto-strict repeat, decisions identical to the pilot (epinephrine same/main; both methoxy cells distinct/reference), dE bands reproduced (-4.026e-08; +1.2826e-06). Two preregistered band clauses MISS for demonstrated environmental reasons, never relabeled: (P2) the cross-session reference envelope on methoxy/wb97mv measured 4.22e-11 vs the 1e-11 band — **the BEFORE state (no sharing) deviates from the pilot by the identical -4.22e-11** (before-vs-after agree to ~3e-14, the same-session floor), i.e. session-level GPU variation, 4.5 orders below the 1e-6 gate; (P4) small-cell fp64 drift 10.7–11.8% vs the ±10% band on a code path bitwise-identical between states (the 200 s-scale tamoxifen fp64 ratio is 0.998) — a fixed additive after-block effect on short samples. Peak device memory drops are positive but small (2–149 MiB; cupy pool recycling absorbs the duplicate allocation) — informative MISS vs the GB-scale prediction, characterized. Per the preregistered decision rule (P1 AND P2 both HIT) MR-8 is therefore **not formally declared effective by this adjudication alone**; the committed change stands on its independent test-suite evidence (determinism premise + parity) plus the P1 mechanism measurement and the in-run contract clauses.

**Structural floor (documented decision input for (b))**: per-invocation strict verification costs at least one same-start fp64 SCF — the certified auto-strict lane can never beat the plain fp64 lane at equal settings (it is the fp64 lane plus the auto trajectory plus comparison). The safe-route menu for soft-mode cells is therefore: plain fp64 = 1.0x, certified-by-construction; or auto-strict ≈ 1.4–2x when the fast path is attempted with per-invocation certification. MR-8 lowers the fixed-cost component (duplicate CDERI) of that 1.4–2x; the sibling SCF itself is the floor. Evidence: `2026-10-23-mr8-verification-cost/` (73-line hash manifest), effects-index `mr8-verification-cost-20261023` (146).

## MAINLINE-REPAIR-9: the MR-2 anomaly mechanism characterized — transient fine-launch `_screen_index` output corruption (measurement 2026-10-23T02:26–02:40Z)

The checkpoint's last open technical item ("MR-2 GPU anomaly mechanism uncharacterized — pathway removed, not explained"). The intermittent partition-fix guard firing that MR-1 made fail-closed and MR-2 eliminated by construction (mr1-guard-repro-a: 3/40 fresh processes on dopamine/B3LYP/def2-TZVP optimized_fp64 b0, distinct blocks, retry-clean; kernel predicate per-(shell,point) deterministic — "mathematically unreachable for identical inputs") was re-measured on an **instrumented pre-MR2 worktree at 44cb789** (the firing-era source; the b1 snapshot's exact `.so` binaries; a 14-line hook patch in `gen_grid.py` plus a capture module): every screen build is shadowed by an **immediate same-input replay launch** (same stream, private output buffer), with per-build input bit-compares (fine-vs-coarse coords/env/bas/atm), processed-vs-raw-mask recomputation, on-firing in-process re-evaluation, and a 6-build ring dumped per firing event; plus 8 controlled in-process cache-reset rebuilds per process after the first clean partition-fix build (one pass, one-shot latch; loop-exit cache hygiene so a firing's wrong pair can never persist into the production path — both design rules were learned from the fault-inject smoke validation and are documented there).

**Measurement** (preregistration sha `351f1655` registered 02:23:27Z BEFORE the launch, schedule pinned `b22caccc` after the schedule-validator flagged the missing pin; N=60 serialized fresh processes of the EXACT arm command, GPU-freedom pre-checked per run; driver 3380606 wall 14.1 min; adjudicator red-tested pre-launch, auto-run by the wait-gate): **1/60 natural first-build firings** (rate 1.7%, Wilson95 [0.3%, 8.9%] — consistent with the 7.5% prior), **8/480 loop-rebuild firings** (1.7%/iteration), and fine-launch replay-diff rate **19/660 builds (2.9%)** vs coarse **8/660 (1.2%)**. Adjudication: **P0/P1/P2/P3/P5/P6 HIT, P4 honest MISS** (the fine-vs-coarse replay-diff ratio measured 2.4x, below the preregistered ≥3x band; direction holds, magnitude does not — the smoke-informed prior overstated it; never relabeled) — **decision rule SUCCEEDED**: P0+P2 hold and P3 confirms one consistent mechanism family across ALL 7 captured firing events. Production integrity: all 59 non-firing runs rc=0 with e_tot within **4.547e-13** of the batch reference (the exact cross-process envelope) — the instrumentation never perturbed the physics. An independent recompute (separate logic) AGREES on every value.

**Mechanism (characterized; upgrades the checkpoint's "uncharacterized, honest limit")**: **transient single-launch output corruption of the fine (4096) AO-screen kernel, spurious-keep direction**. Per event: the fine mask contains keep bits ABSENT from its own immediate same-input replay (5/7 events `fine_confirmed`; 2/7 `both_launch_keep` — the race recurs on the same cell across consecutive launches, the systematic component); substituting the replayed fine mask eliminates the union violation; the coarse screen is replay-stable (7/7); fine-vs-coarse inputs are bitwise identical (7/7 — H1a input divergence refuted); the Python post-processing is faithful to the raw masks (7/7 — H3 refuted); a fresh in-process re-evaluation is clean (1/1 natural — H4 refuted). The guard message "coarse screen is missing shells a covered fine block keeps" describes the corrupted FINE screen from the other side. Static root cause (documented limitation: source analysis + behavioral signature, not a device-level trace): `nr_eval_gto.cu` `_screen_index` has one `__syncthreads()` per tile iteration (between the shared-tile LOAD and the CONSUMER) and **no barrier between the consumer of tile k and the load of tile k+1** — a thread that finished consuming (or took the `is_large` early-`continue`) overwrites `gridx_cache[thread_id]` while a slower thread still reads tile k, feeding wrong coordinates into the per-(shell,point) predicate. This explains the intermittency (scheduling), the spurious-keep direction (a nearer wrong point sets the predicate true), the single-launch transience (a re-launch reschedules), the same-cell recurrence subclass, and the original 3/40.

**Production relevance (recorded; nothing changed here)**: post-MR2 HEAD removed the coarse kernel call but the DEFAULT fine (4096) screen launch remains — the race is a live run-to-run nondeterminism source at screened-tail magnitude (2.9% of launches corrupted by 1–few shells in one block; the drop-direction subset is silent). It is a candidate concrete contributor to the measured cross-run energy envelope (~3e-14, MR-8) and cross-session drift (~4.22e-11, the MR-8 P2 environmental MISS) — MR-9 supplies a mechanism class producing exactly that order of tail-level variation; not claimed as the sole cause. **Named follow-up (NOT executed in-flight)**: a preregistered kernel fix at HEAD (add the missing barrier / restructure the early-`continue` so all threads reach one `__syncthreads()` between consume-k and load-(k+1)), libgdft rebuild, red→green determinism regression, and a measured verification that the replay-diff class drops to 0 — its own preregistration and review; gates and defaults unchanged.

Errata trail (both pre-acceptance, artifacts preserved): the adjudicator's first run mis-selected the natural firing's guard pair (the ring tail after the re-eval, not `ring[:-2]`) — fixed, first output preserved (`redtest/adjudication-first-run-erratum.json`), hand-verified before the fix; the driver's first PID capture pgrep-matched the wrapper shell (the known self-match class) — premature wait-gate killed, corrected before any adjudication ran. Evidence: `2026-10-23-mr9-anomaly-mechanism/` (302-line hash manifest incl. the 60 canaries + 7 capture pairs + smoke validation with its 13-event bycatch), effects-index `mr9-anomaly-mechanism-20261023` (147).

## MAINLINE-REPAIR-10: the screen-kernel race fix — one barrier in `_screen_index` (implementation + verification 2026-10-23T02:58–03:20Z)

The MR-9-named follow-up, executed as its own preregistered work unit (`ff2760f4` registered 02:58:30Z before any post-fix measurement; prior evidence disclosed: MR-9 + the pre-fix baseline collected 02:55Z — dopamine **148/150 launches equal, 2 corrupted**, the reproduced race). **Implementation**: one unconditional `__syncthreads()` at the tile-loop top of `nr_eval_gto.cu` `_screen_index` — the consumer of shared tile k is now barrier-ordered before the load of tile k+1; every path through an iteration passes it, so the `is_large` early-`continue` stays deadlock-free; the existing post-load barrier is kept. libgdft rebuilt via the existing CMake build dir (4 jobs; new `.so` `2c063a1b`, untracked — publisher manifests are source-only). New regression test `tools/tests/test_mr10_screen_kernel_determinism_20261023.py` + the pre-fix baseline JSON.

**Verification vs the preregistered clauses**: **P1 HIT (primary)** — post-fix, 450/450 fresh production-path launches (dopamine/benzene/water × 150) bitwise-equal, plus a 37-process census all in-process deterministic; red evidence = the baseline's 2/150 on the pre-fix binary. **P2 MISS as preregistered, fully characterized** — 36/37 processes reproduce the pre-fix consensus mask; one process produced a layout-variant, and the cause is NOT the fix but a **pre-existing upstream class: `_GDFTOpt` per-process layout nondeterminism** (two fresh processes differ in 132/240 sorted-mol env slots + the bas PTR offsets; in ~1/37 processes the per-shell primitive summation order changes enough to flip a razor-margin cell). The fix's semantics are proven independently: the fixed mask equals the **exact CPU predicate replication bitwise (0/10725 cells)** on the same inputs — and that same truth IS the pre-fix consensus. The consensus-comparison premise was falsified by the layout class; the test was corrected to the in-process CPU ground truth with the design erratum in its docstring and the first MISS output preserved. **P3 HIT** — 118 existing tests green on the rebuilt binary (MR-2 9, mainline-repair 7, stability_verify 45, cderi-reuse 13, gradient runner 19 + validator 25). **P4 HIT** — full dopamine/b3lyp/def2-tzvp optimized_fp64 b0 arm at HEAD + fixed binary: rc=0, 10.8 s, converged, **e_tot bitwise-equal to the batch reference (|dE| = 0.0)**. **P5 HIT** (budget/contract).

**Verdict (honest)**: the preregistered decision rule (P1 AND P2) is not met by the letter; the fix stands on red→green determinism + CPU-truth semantics + suites + bitwise end-to-end continuity — the MR-9-measured race class (19/660 = 2.9% of launches corrupted) is eliminated at the source. **Named follow-up MR-11 (not executed)**: the `_GDFTOpt` per-process layout nondeterminism class (candidate: PYTHONHASHSEED-dependent dict/set iteration in `from_mol`; tail-level impact bounded by the measured 4.5e-13 cross-process envelope). Evidence: `2026-10-23-mr10-screen-race-fix/` (12-line hash manifest), effects-index `mr10-screen-race-fix-20261023` (148).

## MAINLINE-REPAIR-11: deterministic sorted-mol env layout + the mask-anchor discovery (implementation + verification 2026-10-23T03:50–04:35Z)

The MR-10-named follow-up, executed as its own preregistered work unit (`f374bbcf` registered 03:50:00Z BEFORE the implementation; all characterization measurements were completed pre-registration and cited as priors). **Root cause (pinned, pure-pyscf, CPU-only probe)**: pyscf 2.14.0 `Mole.build` collects element symbols into a SET (`uniq_atoms = {a[0] for a in self._atom}`, mole.py:2559) whose iteration order is PYTHONHASHSEED-dependent; `_parse_default_basis` → `format_basis` → `make_env` (`for symb, basis_add in basis.items(): _env.append(env0)`, mole.py:1069) packs the per-element basis data blocks into `Mole._env` in that order. gpu4pyscf inherits it verbatim (`basis_seg_contraction` copies `mol._env`; `_GDFTOpt.build` sorts on PTR-free columns and keeps the PTRs). Measured: **22 distinct sorted-env layouts in a 60-process census** (dopamine/def2-tzvp), a pure VALUE permutation (`np.sort(env)` invariant; MR-10's 132/240 differing slots are relocations), bas-noptr/atm/grids constant.

**Implementation**: `_canonical_env_layout(bas, env)` in `gpu4pyscf/gto/mole.py` + one call at the end of `basis_seg_contraction` — re-packs the basis blocks in first-occurrence order over bas rows, remaps PTR columns, zeroes dead slots inside the old basis region, fails closed (RuntimeError) on overlapping blocks; value-preserving by construction (all blocks read before any new position is written). Covers every consumer (`_GDFTOpt`, `_GDFTOpt2C`, int3c2e, int3c1e, int4c2e, `group_basis`). New regression test `tools/tests/test_mr11_env_layout_determinism_20261023.py` (5 tests).

**Verification vs the preregistered clauses**: **P1 HIT** (red preserved: 4 distinct dopamine layouts across seeds 0/1/2/random; green: identical env/bas/atm hashes across all seeds, all molecules); **P2 HIT** (per-shell blocks bitwise preserved + synthetic permutation-invariance/dead-slot/dedupe/overlap-refusal cases); **P4 HIT** (suites green on the fixed tree: MR-2 9, mainline-repair 7, stability_verify 45 + cderi-reuse 13, gradient runner 19 + validator, MR-10 2, MR-11 5, int2c2e cache 12, int3c2e 13, gto script rc=0); **P5 HIT** (post-fix dopamine arm ×3: rc=0, converged, e_tot within the pre-fix process value set, max |dE| 4.85e-13 ≤ the 5e-13 band). **P3 MISS by the letter — anchor erratum, fully characterized before any green claim**: the clause demanded mask equality with the MR-10 baseline consensus shas; the first post-fix run FAILED on dopamine/benzene (preserved), and the diagnosis is the **mask-anchor discovery**: the MR-10 baseline (collected on the PRE-MR10-FIX racy libgdft) recorded as "consensus" the racy kernel's STABLE, systematically-corrupted output — the racy kernel rebuilt from source 51e1b71 gives benzene `90e56ca9` 30/30 launches and dopamine `aa8525d9` 25/30 (plus 5 launches of a second corrupted variant), each **dropping exactly one truth-keep cell** vs the exact CPU predicate (benzene 1/3240, dopamine 1/10725; zero extra-keep cells). So the MR-10 baseline's "2/150 corrupted" were actually the CLEAN minority, and the first MR-10 semantics-test MISS was a CORRECT mask failing against the corrupted anchor (its preserved transcript's "- want / + got" annotation had the assertEqual argument order swapped). The mask is layout-INVARIANT on the fixed kernel (both benzene layouts and multiple dopamine layouts give the CPU-truth mask), so the MR-10 P2 "layout-variant mask" mechanism is falsified; the corrected invariant — GPU == in-process CPU predicate — holds everywhere measured, and the test now asserts it (erratum documented in the docstring). The preregistration itself had also misquoted the dopamine anchor (wrote the census majority where the JSON records `aa8525d9`) — recorded in OUTCOME, not silently corrected. Verdict: P3 misses by the letter → the fix stands on P1+P2+P4+P5 + the characterized erratum.

**Production relevance (recorded)**: sorted-mol kernel inputs are now bitwise-stable across processes (cross-process output censuses are no longer confounded by the layout class — the MR-10 P2 consensus premise becomes valid going forward). The mask cleanliness dates to the MR-10 barrier fix: every arm run before 6fde65c carried a one-dropped-truth-keep-cell screened-tail systematic (e_tot effect measured bitwise-zero on the dopamine cell by the MR-10 P4 arm; batch verdicts self-consistent as all arms shared the racy kernel — no historical verdict changes). **Two honest non-claims**: (i) the cross-process e_tot envelope (~4.5e-13) is NOT layout-caused (seed-controlled arms: the SAME seed produced 3 distinct e_tot values) and PERSISTS post-fix — a named follow-up class (per-process GPU nondeterminism, layout/seed-independent; candidate: memory-address-dependent library kernel paths; untested); (ii) the single unexplained 1/37 MR-10 census process stays unexplained. Evidence: `2026-10-23-mr11-env-layout-determinism/` (46-line hash manifest), effects-index `mr11-env-layout-determinism-20261023` (149).

## MR-12: the per-process/per-call GPU e_tot nondeterminism class — root-caused (reversed duplicate AO-pair CDERI columns racing in fill_symmetric) and fixed (2026-10-23)

The MR-11-named follow-up class ("cross-process e_tot envelope ~4.5e-13, layout/seed-independent"), executed as two preregistered work units (characterization probes registered 2026-10-23T05:06:07Z; fix registered 2026-10-23T05:32:31Z, both BEFORE their measurements).

**Root cause (measured, cell-level)**: `pair_and_diag_indices` (int3c2e_bdiv.py) enumerates the full AO rectangle inside diagonal shell-pair blocks, so the CDERI stores BOTH (i,j) and (j,i) columns for cross AO pairs of one shell — 583 unordered duplicate classes (1166 of 73827 columns) on dopamine/def2-tzvp, all same-shell. The two columns hold the same integrals but reach them through different cart2sph/solve contraction orders, differing at the last bits (≤2.2e-13 locally; official gpu4pyscf 1.8.1 shows the identical structure — 583 classes, 374 bitwise-unequal, max 3.34e-13 — so the class is upstream-inherited; d_t_kernel SASS byte-identical between our build and the official .so). `fill_symmetric` writes both symmetric cells (i,j) and (j,i) from every column, so bitwise-unequal duplicates race per element: 10/10 launches → 10 distinct outputs and race-winner signatures; output asymmetric in 408–854 cells; **100.0% of deviating cells contained in duplicate-pair cells** (3 independent runs); same-buffer refills vary (launch-level); the hash/D2H path is clean; block semantics correct (the "missing aux0" static-analysis misgiving was wrong — rows [p0:p1) are read correctly). This is the prework's "pure copy varies" contradiction resolved: the kernel is a pure copy; the INPUT has 1166 columns whose pairwise values disagree, and the kernel faithfully copies both into the same cells.

**J path exonerated with the mechanism understood**: production J matches pyscf CPU DF to 4.04e-12 relative. The duplicate pairs sit inside the diagonal shell blocks whose `dm_sparse` entries are halved, so both orientations sum to the correct tril weight — the design is coherent for J; the defect class is confined to the unpack (K) path. (A first reconstruction that double-counted was a probe bug — halving only exact i==j instead of the whole diagonal shell block; the verbatim-instrument replication settled it. VJ's bitwise stability across calls was never evidence of a race here — the auxvec/scatter path is order-fixed.)

compute-sanitizer: memcheck 0 errors on both fill kernel paths. racecheck flags an unrelated int2e_qcond_kernel shared-memory hazard in the BUILD machinery (side observation, out of MR-12 scope; the build is measured bitwise-reproducible — no observed value impact).

**Implementation**: `_canonicalize_duplicate_pair_columns(cderi, pair_addresses, nao)` in gpu4pyscf/df/df.py + one call in `DF.build()` after `_cholesky_eri` (covers the RSH sub-objects' own builds): for every unordered duplicate pair class (any multiplicity/orientation), copy the first-occurrence column bitwise onto the other columns, on every device array (cupy or host numpy). The stored tensor then satisfies the intended bitwise (ij|P) == (ji|P) symmetry and the racing duplicate writes become bitwise-identical (harmless). No defaults, no gates, no kernel changes. New regression test `tools/tests/test_mr12_duplicate_pair_canonicalization_20261023.py` (6 tests: deterministic duplicate-canonicalization invariant; fill == numpy expansion AND bitwise symmetric; 8-launch fill determinism; production get_k bitwise stability; J-vs-pyscf-CPU guard for the halving design; non-vacuity guard).

**Verification vs the preregistered clauses**: **R1 HIT** (pre-fix RED: 34/56 duplicate classes bitwise-unequal on water/def2-tzvp, deterministic; fill≠expansion FAIL; J guard passes pre-fix as predicted; transcripts preserved; honest note: the two probabilistic per-call tests pass pre-fix on water — small stable race set — the per-call flips were measured at dopamine scale, so those two are green-side guards); **G1 HIT** (6/6 green); **G2 HIT** (J vs CPU unchanged, 4.04e-12); **G3 HIT** — the headline: dopamine/def2-tzvp fp64 DF e_tot **bitwise-identical across fresh processes** — RHF 4/4 (`-0x1.00da5145705d7p+9`), b3lyp 3/3 (`-0x1.026fada8e066bp+9`), wb97m-v (RSH + VV10) 3/3 (`-0x1.0255f4962b7fbp+9`) — vs the pre-fix 3-value 4.548e-13 envelope; fill 10 launches → 1 distinct (pre-fix 10/10 distinct); production get_k 5 calls → 1 distinct (pre-fix 15/15 distinct); **G4 HIT** (canonicalization 0.018 s at 73827 pairs, idempotent, no measurable VRAM); **S HIT** (suites green: MR-2 9, mainline-repair 7, stability_verify 45, cderi-reuse 13, gradient runner 19 + validator 25, MR-10 2, MR-11 5, int2c2e cache 12, int3c2e 13, MR-12 6, gto rc=0).

**Honest MISS (characterization preregistration, recorded not relabeled)**: P3(b) predicted the deviation class to be READ-side (both twins deviating with the same wrong value). The measurement falsified it — 99.9% of deviating cells are single-sided twins — and the true mechanism (a store race between duplicate columns, each writing BOTH twins its own value) was not among the preregistered classes. The twin discriminator worked as designed; its interpretation inverted. The sanitizer clause therefore ran as a completeness check (clean) rather than the R1-mandated gate.

**Production relevance (recorded)**: with MR-10 (screen race), MR-11 (env layout) and MR-12 (duplicate-pair race) landed, the dopamine-class fp64 DF/DFT stack is measured bitwise-reproducible across processes at the e_tot level — cross-run/cross-process comparison noise floors (the ~3e-14 cross-run / ~4.5e-13 cross-process / 4.22e-11 cross-session RSH envelopes that gated earlier preregistrations' bands) collapse to zero on the measured cells; future preregistrations can tighten bands accordingly (cross-session reference drift remains a separately-documented environmental class). The residual nondeterminism surface is now: (i) the 1/37 unexplained MR-10 census process (unexplained), (ii) the int2e_qcond_kernel build-machinery smem hazard (sanitizer observation, no observed value impact), (iii) any lane not covered by the measured cells. Upstream contribution candidate: the duplicate-column structure and the canonicalization fix apply to official gpu4pyscf 1.8.1 identically. Evidence: `2026-10-23-mr12-gpu-process-nondeterminism/` (53-file hash manifest; the dir-level manifest was regenerated at MR-12 exit, superseding the prework-era manifest — documented in the effects-index note), effects-index `mr12-duplicate-pair-race-fix-20261023` (151).

## MR-13: new-source (f52735a) production-continuity regression batch b1 (2026-10-23)

Mainline card queue item 4 — the first production batch at the current frozen source `f52735a` (launch HEAD == the production source commit; = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11 86f63b2 + MR-12 f52735a). The 213-cell cohort closed at 3001b34 BEFORE the four landed production-path changes; each was verified in its own work unit, never together at batch scale.

**Selection** (curated, risk-descending, 24 cells): the card-named molecules and failure/soft-mode cells (epinephrine_b3lyp_def2-svp the MR-1 headline; the three methoxy soft-mode cells; dopamine b3lyp_tzvp + wb97mv_svp; ethoxy_m062x_svp; oxygen_triplet wb97mv + b3lyp), the highest-risk 3001b34-class stratum (melatonin_wb97mv_def2-svp_q+1s1 UKS cation, frozen ratio 0.854; lidocaine ×4; caffeine; aspirin), coverage cells (epinephrine_r2scan3c, benzene/butadiene wb97mv_tzvp, methanol, methylamine, butane_r2scan3c, tamoxifen ×2 537 AO). All four methods; svp/tzvp/mtzvpp; spin 0/1/2; charge 0/+1. Protocol: balanced-blocks-v1 (4 roles × 4 blocks = 384 arms; O=official-1.8.1, R=strict_fp64 knobs-off, B=optimized_fp64 6-knob, C=optimized_auto 6-knob — the plain auto lane, not auto-strict); frozen snapshot cp -a 7295 files incl. the rebuilt libgdft; GPU-idle-gated serial arms, fresh subprocess per arm. Preregistration registered 06:28:34Z before any measurement (two documented PRE-measurement corrections: P1 auto-lane wording; the fixed_source_commit sha tail), validator PASS; launch gate verified HEAD==f52735a/clean/GPU-idle; driver 3425278 launched 06:36:01Z, wait-gate armed 06:36:54Z (8h cap).

**Final verdicts (adjudicate_final + independent recompute AGREE, crosscheck HIT)**: **P1 HIT** 24/24 cells, 384 arms rc=0, all converged, zero retries. **P3 HIT** (B floor ≤1e-9), **P4 HIT** (B/C within 2x frozen). **P6 HIT — the decisive new clause: 24/24 cells have R-lane e_tot pairwise BITWISE-identical across the four fresh-process blocks** (band was ≥19/24; pre-fix class = the 3-value ~4.5e-13 envelope, b5 10/44 / b6 5/24 bitwise) — the MR-12 mechanism is total at batch scale; no exceptions, no follow-up opened. Informative: R gradients still vary across blocks at 1e-15..7.6e-13 (gradient-path residual ULP class, 8+ orders below the gate). **P7 HIT** — all three methoxy soft-mode cells persist as honest FAILs at their frozen values (dg 3.0e-02 / 1.34e-03 / 1.16e-04). **P9 HIT** wall 5.95h ≤ 6.5h. **P10 HIT** zero partition-guard firings, the sixth consecutive zero-firing prediction, now with the kernel race itself fixed in-source. **Decision rule: the production-continuity claim (P1∧P2∧P7∧P10) is NOT met — P2 MISS; the MR-12 batch-scale claim IS met.**

**P2 MISS (two honest gate-failing cells, both diagnosed to mechanism, both kept in the denominator)**: (1) **melatonin_wb97mv_def2-svp_q+1s1** optimized_auto-vs-R max|dg| = 4.028e-05 > 3e-5 (frozen 2.561e-05; dE 2.06e-10 ≪ gate). Mechanism: the auto lane's dE-triggered precision switch moved cycle 9→12 at the new source (fp32 head 3 cycles longer; df_jk fp32 hits 36→48), landing the fp64 tail 2.96e-10 away; uniform over all 16 role-R pairs; the lane is bitwise-deterministic across processes; the strict lane is bitwise-continuous with b1 (upstream agreement 4.5e-13 floor). The mixed-precision soft-direction sensitivity of the near-gate UKS cation class. Named follow-up MR-13-FU1 launched same day: the card's safe candidate (5-knob SAFE_STACK + auto-strict reference matching) spot-check on melatonin + epinephrine + methoxy_wb97mv; decisive quantity = stability_verify_dm_tol 1e-3 (methoxy precedent dm 0.0111 → adoption; melatonin's displacement ~2 orders smaller — the same-solution branch is a live coverage-gap possibility). (2) **epinephrine_b3lyp_def2-svp** upstream-vs-R |dE| = 1.932e-06 > 1e-6 (frozen 9.1e-13; dg 4.10e-06) — see the arbiter below: the deviant lane is the official package, not ours.

**P5/P8 MISS, attributed via two preregistered pyscf-CPU-fp64 arbiters (CPU-only, run during the batch)**: **lidocaine_b3lyp_def2-tzvp** and **epinephrine_b3lyp_def2-svp** are the only cells whose strict fp64 lane moved vs the 3001b34-class frozen lane (−8.19e-08 and −1.932e-06; all our lanes coherent; optimized_fp64 follows bitwise; 13 SCF cycles and gnorm unchanged; all O/B/C-vs-R gates still pass on lidocaine, epinephrine fails only via the U role). The unchanged official 1.8.1 package stays at the old values. Arbiters (preregistered 10:27:59Z / 11:36:36Z, H1/H2/H3 with 1e-9 margin): pyscf 2.14.0 CPU fp64 lands **1.31e-10 / 1.61e-11 from our new values and 8.18e-08 / 1.932e-06 from the old/upstream ones — H1 confirmed on both, H2 (source regression) falsified**. The mechanism (MR-11's direct measurement: the racy `_screen_index` kernel's stable output drops truth-keep screen cells; MR-10 restored them): the b-series-era agreement between our lanes and the official package was SHARED grid-incompleteness; the official 1.8.1 wheel still carries the racy kernel, so its own epinephrine/b3lyp/def2-svp total energy is ~1.9e-06 Ha off pyscf CPU truth — **an upstream-inherited defect whose magnitude alone exceeds the project's 1e-6 Ha gate on that cell** (dopamine_b3lyp_tzvp U = 0.0 bitwise — its dropped cell has negligible e_tot impact; benzene_wb97mv at floor). With the kernel fixed, the 6-knob C-role on epinephrine sits at 4.5e-13 vs strict: the MR-1 A4 "XC_GRID_BLOCKSIZE alone" deviation is mechanistically the same racy-kernel class (partition change → different screen cells → different truth-keep drops) — recorded as an annotation on the immutable history (the MR-1 fix's same-start contract was and remains correct operationally), never a rewrite. P8's ratio-exceedance list adds lidocaine_r2scan3c (6.82e-13, floor-class).

**Honest limits**: curated 24-cell coverage, not the full cohort; regression continuity, not new speedup evidence; the auto-strict production safe route is not exercised by this protocol (MR-8's own 60-sample measurement + 13 suite tests cover it; FU1 extends it to the melatonin class); P5 mixes source change and cross-session drift; no acceptance/default claims; gates unchanged. Evidence: `2026-10-23-mr13-regression-b1/` (29-file hash manifest incl. both CPU-arbiter preregistrations+results; 384 raw samples), effects-index `mr13-newsource-regression-b1-20261023` (152; full-index sweep 935 OK / 0 missing / exactly the 3 previously-documented mismatches).

## MR-13-FU1: the safe-candidate (5-knob + auto-strict) spot-check at f52735a — melatonin PASSES (2026-10-23)

The MR-13 Class-A named follow-up, executed the same day. Preregistration registered 2026-10-23T12:36:04Z (DRAFT authored CPU-only during the MR-13 batch; launch conditions + GPU-idle verified) BEFORE the first measurement; the melatonin timeout retry registered 13:12:49Z BEFORE the retry launch. Protocol: the operator-pilot protocol verbatim — per cell `gradient_runner` fp64 lane then auto-strict lane (`--merge-from`, 3 samples, SAFE_STACK 5-knob env WITHOUT XC_GRID_BLOCKSIZE) at the MR-13 frozen snapshot f52735a; driver 3477109 + one bounded retry driver.

**Verdicts (all preregistered P1–P6)**: **P1 HIT** — 6/6 final runs rc=0, all converged, every auto-strict repeat strict_match_verified=True; the FIRST melatonin auto-strict attempt hit the pilot-verbatim 900 s timeout rc=124 (structural: the fp64 lane alone took 804 s, auto-strict needs ~960 s; killed mid-sample-3, runlog preserved, no partial JSON) — ONE bounded 1800 s retry with the justification registered before launch, protocol otherwise identical, completed rc=0; both attempts preserved, never a silent rerun. **P2 (PRIMARY) HIT — the melatonin gate PASSES via the SAME-SOLUTION branch**: dm_max_abs_diff = 1.545e-05, 65x below stability_verify_dm_tol 1e-3 → main kept, verified on all 3 repeats (identical; deterministic); the kept 5-knob auto density's gradient vs the fp64 reference: worst max|dg| = **1.391e-05 ≤ 3e-5** (46% of gate; dE 1.087e-09); the precision switch fired at cycle 9. **P3 HIT** — epinephrine dm 1.16e-06 main-kept (dg 5.71e-06, within 3x of the pilot 5.011e-06); methoxy_wb97mv dm 3.70e-01 ADOPTION fired, selected=reference (dE 0.0 bitwise; dg 1.95e-06; within 3x of the pilot 1.790e-06). **P4 HIT** (worst |dE| 1.09e-09). **P5 measured**: t_scf median ratios 1.23x melatonin (233 s vs 189 s — BELOW the pilot end-to-end 1.58–2.03x band; the MR-8 CDERI sharing amortizes at 318-AO UKS RSH) / 2.68x epinephrine / 2.22x methoxy. **P6 HIT**: the FU1 melatonin fp64 reference is BITWISE −763.8562377313342 == the MR-13 R lane (MR-12 determinism across yet another fresh process).

**Scientific finding (the MR-13 Class-A record)**: the 6-knob melatonin failure (4.028e-05; switch 9→12) is NOT reproduced by the 5-knob safe candidate (switch 9; dg 1.391e-05). At the FIXED screen kernel the XC_GRID_BLOCKSIZE=16384 knob remains fp32-head trajectory-sensitive (partition change → different summation orders → different dE-trigger switch point → different landing on this soft UKS cation); its historical A4 "100% energy deviation" was the racy-kernel amplification of exactly this sensitivity. The safe candidate's removal of the knob is now MEASURED to reduce near-gate UKS trajectory sensitivity: the card's Class-1 temporary safety policy is validated at f52735a, and the production guidance for the melatonin class is the 5-knob auto-strict route (1.23x verification cost at this size, certified on every repeat). MR-13's P2 MISS cell stands unchanged in its own denominator (different lane — the 6-knob benchmark protocol C-role); nothing relabeled. **Honest limits**: 3 cells; the dm-tolerance structural boundary stands (a below-tol displacement with an above-gate gradient would be kept unverified-matching — not realized at f52735a on this stack); the 6-knob trajectory sensitivity is characterized on one cell; no speedup/acceptance/default claims; gates unchanged. Evidence: `2026-10-23-mr13-followup-autstrict/` (23-file hash manifest incl. both melatonin attempts and the retry record), effects-index `mr13-fu1-autstrict-spotcheck-20261023` (153; sweep 956 OK / 0 missing / the 3 documented mismatches).


## MR-13-b2: new-source regression batch b2 — the second risk stratum (2026-10-23)

**Scope**: mainline card queue item 4 continuation — the second production
batch at the current frozen source (launch HEAD `2f2ad84`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b2 adds only the b20106e/2f2ad84 docs commits).
b1 (same day) covered the named/highest-risk stratum; b2 extends to the NEXT
risk stratum: the **top-24 of the 189-cell remaining pool by frozen B/C-vs-R
gate ratio (0.504 -> 0.306), mechanically selected** (cells-selection
generator + sha-recorded selection; no coverage re-optimization). ALL 24
cells expected-pass (no soft-mode cells remain in the pool); one UKS cation
(melatonin_m062x q+1s1, ratio 0.448 — the second Class-A data point, half
the frozen margin of b1's melatonin wb97mv 0.854).

**Selection composition** (from cells-selection-mr13-b2.json): lidocaine x3,
caffeine x6, tryptophan x3, tamoxifen x2, vitamin_c x3, melatonin x3 (two
m062x incl. the q+1s1 cation + one b3lyp), inosine x2, aspirin,
phenylalanine — all four methods; bases 15 svp (incl. the cation) / 5 tzvp /
3 mtzvpp; RKS closed-shell (the one UKS cation excepted); risk-descending
order; frozen references extracted from the raw samples of the 3001b34-class
source batches (b1-adjudicator role_errors logic).

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot cp -a (7295+
files incl. the rebuilt libgdft .so); GPU-idle-gated serial arms; fresh
subprocess per arm.

**Preregistration** (registered 2026-10-23T13:40:19Z before any measurement;
sha-recorded schedule b9efc602…): THREE documented PRE-measurement corrections,
all fail-closed by the launcher before any measurement ran — (1) the
fixed_source_commit sha tail (transcription caught by the launch-gate
rev-parse check), (2) b1-cell wording for the validator's named-cell check,
(3) schedule schema_version/seed/settings alignment (incl. headroom+timeout)
to the runner contract, caught by fail-closed launcher attempts. Bands and
decision rules unchanged by all three. Honest limit stated at registration:
the b1 CPU-arbiter protocol is referenced, not re-registered per cell (a b2
U-role mover triggers a fresh arbiter preregistration before ITS measurement).

**Predictions (P1-P10)**: as registered — completion; the 24 expected-pass
cells within gates, with sharpened b1-informed predictions: (i) any U-role
gate excess attributes via the fresh pyscf-CPU-fp64 arbiter to the official
wheel's racy-kernel grid-incompleteness (H1: OUR lane correct; b1 instances:
lidocaine 8.2e-08, epinephrine 1.93e-06, both CPU-verified); (ii) the
melatonin_m062x q+1s1 C-role stays WITHIN gate at frozen ratio 0.448 (a
crossing = honest MISS with the b1 auto-switch-cycle mechanism note); B floor
<=1e-9; B/C dg within 2x frozen; R continuity vs the 3001b34-class lane
(2e-12 envelope, movers get the CPU arbiter); **P6 (decisive): R-lane e_tot
bitwise-identical across the four in-batch blocks in >= 19/24 cells and
within envelope in all** (b1 measured 24/24; pre-fix class = 3-value
~4.5e-13 envelope); P7 N/A-by-design (no soft-mode cells in the pool);
O-lane continuity (5e-12 / 2e-7 / 3x, violations -> CPU arbiter);
wall <= 7.5 h (model 3.95 h; b1-calibrated warmup-aware 1.70x -> 6.72 h
predicted); zero guard firings (seventh consecutive prediction).

**Launch gate verified**: HEAD == 2f2ad84 exact; clean tree; GPU idle; no
other driver. Driver 3484964 launched 13:46:40Z; wait-gate 3485108 armed
13:47Z (8 h cap) auto-runs both adjudicators on exit.

**Early signal (13:57Z, cell 1/24)**: CONFIRMED FINAL — cell 1
lidocaine_m062x_def2-tzvp (highest frozen ratio 0.504) completed 14:00:30Z
all-roles PASS with a bitwise 4/4 R lane at e_tot=-731.4111034147093;
both the frozen 3001b34-class lanes AND the unchanged official 1.8.1
upstream package sit at -731.411103298507x (displacement -1.162e-07) ->
the batch's first U-role mover, arbitrated H1 (below).

**Results** (final adjudication 2026-10-23T19:29:30Z, auto-run by the
wait-gate; independent recompute AGREES on every clause, including the
PENDING->N/A P7 vacuous-verdict guard): **8/10 HIT — production-continuity
claim MET for this stratum, MR-12 batch-scale claim extension MET.**

- **P2 no-gate-regression HIT (decisive)**: all 24 expected-pass cells
  O/B/C vs R within gates — worst U dE 1.162e-07 (lidocaine_m062x,
  attributed below), worst non-mover O dE 5.91e-12 (vitamin_c_m062x,
  the frozen-spread class), worst B/C dE 4.12e-10 (inosine_wb97mv auto);
  worst any-role dg 1.49e-05 (caffeine_r2scan3c B) vs 3e-5; **the
  melatonin_m062x q+1s1 C-role PASSES** (dg 9.6e-06, dE 3.5e-11 — the
  frozen-margin 0.448 prediction confirmed; contrast b1's melatonin
  wb97mv q+1s1 C-role 4.03e-05 MISS at margin 0.854, consistent with the
  margin story).
- **P6 R-bitwise HIT (decisive)**: **24/24 cells bitwise-identical R
  blocks** (band >= 19/24) — zero exceptions; gradients informative
  ULP-residual. The MR-12 mechanism holds at batch scale for the second
  consecutive batch.
- P1 completion HIT: 24/24 cells, 384 final arm-runs rc=0, all converged,
  zero retries, admission gates respected. (Tooling erratum: the verdict
  field n_pass_cells_evaluated='21' is a b1-era hardcoded label — the P2
  loop evaluated all 24 cells; b2 has zero soft-mode exclusions.)
- P3 B-floor HIT: worst B dE 4.11e-10 (inosine_wb97mv; band 1e-9).
- P4 gradient envelope HIT: all B/C dg ratios within 2x frozen (max
  1.09x, melatonin_b3lyp auto); the >2x set is empty — the lidocaine
  U-lane 25.8x is P8's mover class, not P4.
- P9 wall HIT: 5.68 h <= 6.5 h band (model 3.95 h; warmup-aware predicted
  6.72 h).
- P10 zero-guard-firings HIT: 0/384 arms, sixth consecutive zero-firing
  batch, now with the MR-10 kernel race fixed in-source.
- **P5 R-continuity honest MISS (three cells, all attributed before
  interpretation)**:
  - lidocaine_m062x_def2-tzvp 1.16e-07: CPU arbiter (preregistered
    14:02:50Z before measurement) **H1** — CPU lands 9.4e-11 from OUR new
    value vs 1.16e-07 from the old/upstream one; the frozen lane's value
    was grid-incomplete (racy screen kernel), our MR-10 fix restored the
    dropped truth-keep cell; upstream-inherited, not our regression.
  - lidocaine_wb97mv_def2-tzvp 7.90e-08 (first RSH+VV10 instance): v1
    total-energy arbiter **honest H3** (CPU 2.07e-08 from NEW vs 9.98e-08
    from OLD, outside the 1e-9 margin — the RSH+VV10 CPU-vs-GPU
    implementation floor dominates); the preregistered per-term escalation
    v2 ran (one bounded retry after an attempt-1 json numpy-bool crash,
    preserved): **H3 with term-level directional corroboration toward the
    NEW lane** — CPU total AND e1 AND folded-exc closer to NEW; the e1/coul
    differences (~1.2e-05) cancel in the total, leaving the folded-exc
    difference as the discriminating term, dominated by the CPU-VV10-vs-
    GPU-VV10 floor (~4e-7). Not margin-certified like the m062x/b3lyp
    instances; recorded as the honest attribution.
  - vitamin_c_m062x_def2-svp 5.91e-12: **H3_below_discrimination_scale as
    preregistered** (prereg 19:10:58Z) — the displacement sits INSIDE the
    cell's frozen R self-spread 6.14e-12 (the pre-MR-12 nondeterminism
    class; frozen blocks -...725, -...729, -...668, -...668; the new
    bitwise value -683.9176677883727 lies inside the old value set); CPU
    distances 1.27-1.91e-11 from all seven candidates — the ~6e-12 span
    rides on the ~1e-10 CPU-vs-GPU m062x implementation floor. Not a
    grid-incompleteness mover; P5/P8 flags stay as measured.
- **P8 O-continuity honest MISS (two cells)**: the same two lidocaine
  movers (worst O-vs-R dE 1.162e-07, worst dg 7.16e-07) — attributed to
  the official package's own upstream-inherited classes via the same
  arbiters; on vitamin_c the official U lane itself splits 2+2 across
  blocks (-...731 / -...668, its own unfixed duplicate-pair class).
- P7 N/A-by-design (no soft-mode cells in the b2 pool; b1 P7 HIT holds
  the persistence evidence).

**Decision rule**: production_continuity_claim=TRUE (P1+P2+P10 gate) +
mr12_batch_scale_claim=TRUE (P6 gate) for this stratum — both claims
carried by this batch's own measurements, never by the arbiters (which
only attribute mechanism).

**Honest limits**: 24 of the 189-cell remaining pool (not the full
remainder; a regression confined to an unselected cell would not be caught);
mechanical top-ratio selection (coverage not re-optimized); regression
continuity, not new speedup evidence; no cross-batch timing claims; P5
mixes source change and cross-session drift (tamoxifen b2 precedent); the
O lane runs the unchanged official package with its own upstream-inherited
duplicate-pair + racy-screen classes; no acceptance/default claims; gates
unchanged.

Evidence: `2026-10-23-mr13-regression-b2/` (hash manifest at exit);
effects-index `mr13-newsource-regression-b2-20261023` (154).

## MR-13-b3: new-source regression batch b3 — the third risk stratum (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the third production
batch at the current frozen source (launch HEAD `faf8171`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b1/b2 measured it at batch scale, both P2+P6
clean). b3 takes the NEXT stratum: **pool ranks 25-48 of the 165-cell
post-b2 remainder by frozen B/C-vs-R gate ratio (0.305 -> 0.249),
mechanically selected** (cells-selection generator + sha-recorded
selection `2af05341…`; schedule `44386061…`; no coverage re-optimization).
ALL 24 cells expected-pass; TWO UKS cations (melatonin_b3lyp_def2-svp
q+1s1 ratio 0.294 and melatonin_r2scan3c_def2-mtzvpp q+1s1 ratio 0.276 —
both BELOW the b2 melatonin_m062x 0.448 pass and far below the b1
melatonin_wb97mv 0.854 crossing) + dopamine_b3lyp_def2-svp (0.251, the
card's named adjacent control molecule) in this stratum.

**Selection composition** (from cells-selection-mr13-b3.json; risk span
0.305 -> 0.249, risk-descending order): melatonin x5 (incl. the two q+1s1
cations + b3lyp/wb97mv tzvp neutrals), phenylalanine x3, serotonin x3,
inosine x2, bisphenol_a x2, acetamide x2, and one each of vitamin_c,
nicotine, tryptophan, glycylglycine, aspirin, dopamine, acetic_acid;
methods b3lyp x6 / wb97mv x8 / m062x x4 / r2scan3c x4; bases def2-svp x16 /
def2-tzvp x2 / def2-mtzvpp x4 (the two cations among the svp cells); RKS
closed-shell except the two UKS cations; frozen references extracted from
the raw samples of the 3001b34-class source batches (b1-adjudicator
role_errors logic; source_batch b2 for the rank-1 cell, otherwise the
b-series roots recorded per cell).

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot cp -a; GPU-idle-
gated serial arms; fresh subprocess per arm.

**Preregistration** (registered 2026-10-23T19:46:20Z before any measurement):
three documented PRE-measurement corrections, all validator/launcher-driven
before any measurement ran — (1) prior-batch cell mentions carry in-b1/in-b2
historical markers; (2) prior-batch full cell tags use the slash-shorthand
convention per the validator's no-exemption full-tag check; (3) schedule
schema_version/seed/settings aligned to the runner contract after the
fail-closed launcher refused the first attempt (the exact b2 correction-3
class; selection CONTENT and ORDER byte-identical, only the block-latin-square
seed differs). Bands and decision rules unchanged by all three.

**Results** (driver 3542133, 19:47:31Z-01:05:43Z, wall 5.32 h; 384/384 final
arms rc=0, zero retries, zero guard firings — seventh consecutive
zero-firing batch):

- **P1 HIT**: 24/24 cells complete 4 roles x 4 blocks; all converged; all
  admitted.
- **P2 HIT — production-continuity claim MET, the third consecutive clean
  stratum**: batch-wide worst |dE| U/O 1.14e-12, B 4.13e-10, C 4.18e-10
  (vs 1e-6); worst max|dg| U/O 2.77e-08, B 9.02e-06, C 8.90e-06 (vs 3e-5).
  **ZERO U-role movers** — the first MR-13 batch with no official-lane
  floor violation anywhere (b1: 2 arbiters; b2: 3 arbiters).
- **Both melatonin UKS cations PASSED** (the preregistered 0.294/0.276
  predictions held): melatonin_b3lyp q+1s1 C-role dg 8.68e-06 = 29% of
  gate (frozen 8.42e-06, dE 1.82e-12); melatonin_r2scan3c q+1s1 C-role
  dg 8.04e-06 = 27% of gate (frozen 8.29e-06, dE 4.27e-11); both
  bitwise-R. The auto-switch-cycle crossing class did not recur at these
  margins. dopamine_b3lyp (the card's named adjacent control): C dE
  1.05e-11, dg 6.53e-06 (frozen 6.79e-06), bitwise-R — PASS.
- **P6 HIT decisive — third consecutive batch**: R-lane e_tot
  bitwise-identical across the four fresh-process blocks in **24/24 cells**
  (band >= 19/24; b1 24/24, b2 24/24; pre-fix class = the 3-value ~4.5e-13
  envelope, b6 5/24). Gradients ULP-residual across blocks.
- **P3 HIT**: B-lane energies at the frozen floor (worst 4.13e-10 <= 1e-9).
- **P4 informative MISS (two entries, corrected reading)**:
  tryptophan_r2scan3c B dE 0 -> 2.27e-13 and phenylalanine_r2scan3c C dE
  0 -> 4.55e-13, both vs frozen bitwise-zero references (floored 2x
  denominator max(frozen, 1e-13) = 2e-13). ULP-class cross-process moves,
  3+ orders below the P3 floor, 6+ orders below the gate; the dg half is
  clean (max B/C ratio 1.03x — the tightest of all batches so far).
  Erratum: the adjudicator's energy branch covers only the B role (the
  phenylalanine C-role entry was found by the exit review applying the
  preregistered P4 letter; both entries recorded; the recompute's P4
  implements only the dg half — its P4 "HIT" is vacuous on the energy
  side; documented in tooling-errata-20261024T0115Z.md).
- **P5 HIT**: R-lane continuity vs the 3001b34-class lane — worst
  |dE| 9.09e-13 <= 2e-12 with five above-floor noise-class cells
  (3.41e-13..9.09e-13, all inside the envelope; melatonin_wb97mv_svp
  6.82e-13 > per-cell floor 4.55e-13 = its frozen self-spread class).
- **P7 N/A-by-design** (no soft-mode cells in the b3 pool; b1 P7 HIT
  holds the persistence evidence; the adjudicator's summary renders this
  "PENDING" — the known b2-era display label, na_by_design=True).
- **P8 informative MISS (two cells)**: inosine_r2scan3c and
  aspirin_r2scan3c O-lane dE 0 -> 4.55e-13 with dg moves at ULP level
  vs frozen bitwise-zero references (>3x floored ratio) — the official
  wheel's own duplicate-pair class on r2scan3c/mtzvpp cells. The
  batch-wide O bands are MET (worst dE 1.14e-12 <= 5e-12, worst dg
  2.77e-08 <= 2e-7); no arbiter required (no floor violation anywhere).
- **P9 HIT**: wall 5.32 h <= 7.5 h (model 3.77 h; predicted 6.41 h;
  measured/predicted 0.83x — b2 measured 0.85x; the warmup factor is
  converging below the 1.70 calibration).
- **P10 HIT**: zero partition-fix guard firings across 384+ arms.

**Pattern observation** (recorded, no follow-up launched): all four
informative P4/P8 entries are r2scan3c_def2-mtzvpp cells whose frozen
references were bitwise-zero; the ULP-level cross-process drift
concentrates in that method/basis combination. Magnitudes 2.27e-13..4.55e-13
are inside the b1-measured cross-process noise envelope class.

**Decision rule**: production_continuity_claim=TRUE (P1+P2+P10 gate) +
mr12_batch_scale_claim=TRUE (P6 gate) for this stratum.

**Honest limits**: 24 of the 141-cell post-b3 remainder (not the full
remainder; a regression confined to an unselected cell would not be
caught); mechanical top-ratio selection (coverage not re-optimized);
regression continuity, not new speedup evidence; no cross-batch timing
claims; P5 mixes source change and cross-session drift; the O lane runs
the unchanged official package with its own upstream-inherited
duplicate-pair + racy-screen classes; no acceptance/default claims; gates
unchanged. Effects-index sweep at this exit: 1048 OK / 0 missing /
exactly 4 documented mismatches (the 3 prior + a NEW bookkeeping erratum:
the b2 exit session filled its docs-section-DRAFT TODOs at 19:34:27Z,
after the effects-index-154 sweep had recorded the pre-fill hash — a
post-record mutation of a staging file; the final b2 section ships
hash-stable in the faf8171 docs commit, only the staging copy drifted).

Evidence: `2026-10-23-mr13-regression-b3/` (30-file hash manifest + 96
role samples; mid-batch recompute dry runs + 33/33 value-level crosscheck;
exit-review tooling errata documented); effects-index
`mr13-newsource-regression-b3-20261024` (155).

## MR-13-b4: new-source regression batch b4 — the fourth risk stratum, the first all-closed-shell stratum (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the fourth production
batch at the current frozen source (launch HEAD `4c14f76`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b1/b2/b3 measured it at batch scale, three
consecutive P2+P6 clean strata). b4 takes the NEXT stratum: **pool ranks
49-72 of the 141-cell post-b3 remainder by frozen B/C-vs-R gate ratio
(0.248 -> 0.203), mechanically selected** (generator re-pinned to 4c14f76
before regeneration; schedule sha `2785f31d…`; selection sha `66d4c158…`;
selection CONTENT and ORDER byte-identical to the DRAFT generated during
the b3 batch — only source_commit changed). ALL 24 cells expected-pass,
**ALL closed-shell ZERO UKS — the first all-closed-shell MR-13 stratum**
(b1 had the two UKS cations + melatonin_wb97mv 0.854; b2 one UKS cation
0.448; b3 two UKS cations 0.294/0.276; b4 none).

**Selection composition** (from cells-selection-mr13-b4.json; risk span
0.248 -> 0.203, risk-descending order): glycylglycine x4 (all four methods),
aspirin x4 (b3lyp/m062x-svp/m062x-tzvp/wb97mv), acetic_acid x3
(b3lyp/r2scan3c/wb97mv), bisphenol_a x2 (m062x/r2scan3c), epinephrine x2
(m062x/wb97mv — the named U-mover candidates, ratios 0.224/0.205, both
with frozen O-vs-R dE already at 9.09e-13/6.82e-13), melatonin x2
(m062x-tzvp/r2scan3c), tryptophan x2 (b3lyp/m062x), and one each of
acetamide, dopamine, nicotine, phenylalanine, vitamin_c; methods m062x x10 /
r2scan3c x6 / b3lyp x4 / wb97mv x4; bases def2-tzvp x10 / def2-svp x8 /
def2-mtzvpp x6; ALL charge0/spin0 RKS; frozen references extracted from the
raw samples of the 3001b34-class source batches (source_batch b3 for the
rank-25-48 overlap cells, otherwise backfill). The six r2scan3c cells
(frozen we 1.14e-13 .. 7.73e-12, O bitwise-zero) carry the b3-identified
P4 ULP-floor candidate class.

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot cp -a at
/data/tmp/gpu4pyscf-agent/execution/mr13-b4-snapshot-4c14f76; GPU-idle-gated
serial arms; fresh subprocess per arm; driver 3587252 launched
2026-10-24T01:29:01Z, batch pass finished 05:10:16Z (wall 3.68 h); wait-gate
auto-ran both adjudicators on driver exit (INCREMENTAL_DONE + FINAL_DONE
rc=0); exit monitor (sleep-240 ordering fix) ran the final recompute
(RECOMPUTE_DONE rc=0; crosscheck vs adjudicator AGREE on every clause).

**Preregistration** (registered 2026-10-24T01:26:36Z before any
measurement; sha `540983c4…`): NO corrections at registration (the schedule
was regenerated at the re-pinned SOURCE_COMMIT before this registration; the
two known validator classes from the b3 registration did not recur);
validate_preregistration_vs_schedule PASS (the memory rule: run the
validator before registering any batch preregistration).

**Pre-exit tooling errata** (corrected 2026-10-24T01:44-01:48Z, mid-batch,
BEFORE the wait-gate first ran either script; tooling-errata-20261024T0148Z.md):
the b4 adjudicator/recompute were adapted from the b2 tools and two b2-era
facts survived the narrative adaptation — (1) WALL_H/WALL_BAND_H = 6.5 (b2's
P9 band) vs the b4 preregistered 7.5 h band, corrected in both tools before
any exit adjudication ran (b3's adjudicator carried the same latent 6.5;
harmless there because b3's actual wall 5.32 h passed both bands — recorded
as a latent b3 tooling erratum, never a b3 verdict error); (2) stale b2
narrative (docstrings, "21 expected-pass cells / three soft-mode cells",
the P9 note) aligned to the b4 preregistration letter. The adjudicator's P4
energy branch covers only the B role — the known b3 exit-review class; the
preregistered P4 letter was applied to BOTH roles at review (same rule as
b3). Exit-review additions: the adjudicator's P10 note renders "sixth
consecutive" — a stale b2-era ordinal label; the preregistration letter says
EIGHTH (b1, b2, b3 and the five b-series batches before) and the verdict
n_firings=0 is unaffected (display-only). The evidence-manifest writer's
header formula was corrected at 05:49Z before anything depended on it (the
counted values were always correct; only the parenthetical formula text was
misleading) and the manifest re-written with the corrected final content.

**Results**: 24/24 cells complete, 384 arms rc=0, all converged, zero
retries, zero guard firings; independent recompute final: P0/P1/P2/P3/P6/
P9/P10 HIT, P7 N/A-by-design (the PENDING display label is the known
b2/b3-era rendering with na_by_design=True), P4/P5/P8 as below; crosscheck
vs adjudicator AGREE on every clause.
- **P1 HIT**: 24/24 cells, 4 roles x 4 balanced blocks, 384 final arm-runs,
  all rc=0/converged/admitted.
- **P2 HIT — production-continuity claim MET, fourth consecutive stratum**
  (P1 AND P2 AND P10): zero gate failures batch-wide; worst |dE| vs R:
  U 7.16e-07 (the arbitrated epinephrine_m062x mover, under 1e-6), B
  1.37e-10, C 1.44e-10 (both aspirin_wb97mv — tighter than b3's
  4.13e-10/4.18e-10); worst max|dg| U 7.51e-06 / B 9.01e-06 / C 8.46e-06
  vs 3e-5. The two named U-mover candidate cells BOTH fired the
  preregistered arbiter protocol (below); the auto-switch-cycle class did
  not apply (zero UKS cells).
- **P3 HIT**: B-lane floor 1.37e-10 <= 1e-9 (b1 3.5e-11, b2 4.11e-10,
  b3 4.13e-10).
- **P4 informative MISS** (envelope deviation, never a gate failure; the
  both-roles letter applied at review): B-role 2 entries (acetic_acid_
  r2scan3c 5.68e-14 -> 2.27e-13 = 4.0x; glycylglycine_r2scan3c 0 ->
  5.68e-13 vs bitwise-zero frozen refs) + C-role 1 entry found by the
  letter (bisphenol_a_r2scan3c 0 -> 1.59e-12); the O-role ULP-floor class
  extends over all six r2scan3c cells at 1.1e-13..5.7e-13 (all within the
  5e-12 batch band; the adjudicator's per-cell 3x list flags the two
  largest: melatonin_r2scan3c 4.55e-13, glycylglycine_r2scan3c 5.68e-13 —
  the official wheel's own duplicate-pair class, upstream-inherited);
  dg ratios (dg > 1e-8 noise floor): our lanes max 1.46x (epinephrine_
  wb97mv B), b3 was 1.03x.
- **P5 honest MISS** (attributed via the preregistered arbiters BEFORE
  interpretation, below): the two epinephrine strict-lane movers —
  epinephrine_m062x dE 7.16e-07 and epinephrine_wb97mv dE 2.49e-07, both
  > the 2e-12 envelope; seven further above-floor cells all INSIDE the
  envelope as the noise class (worst 1.14e-12 tryptophan_b3lyp; melatonin_
  m062x 6.82e-13, aspirin_m062x 6.82e-13, vitamin_c_m062x 9.09e-13,
  glycylglycine_wb97mv 6.82e-13, melatonin_r2scan3c 4.55e-13,
  glycylglycine_r2scan3c 5.68e-13).
- **P6 HIT DECISIVE — the FOURTH consecutive batch**: R-lane e_tot
  bitwise-identical across the four fresh-process blocks in 24/24 cells
  (band >= 19/24; pre-fix class = 3-value ~4.5e-13 envelope); gradients
  ULP-residual (worst R-block dg spread 7.22e-13 melatonin_r2scan3c,
  0/24 bitwise — same class as b2/b3).
- **P7 N/A-by-design** (zero soft-mode cells in the selection pool).
- **P8 honest MISS** (same two arbitrated cells): batch-wide worst O dE
  7.16e-07 (the epinephrine_m062x mover) / worst O dg 7.51e-06 vs the
  5e-12/2e-7 floors; per-cell 3x exceedances = the two r2scan3c ULP-floor
  cells + the two arbitrated epinephrine movers.
- **P9 HIT**: wall 3.68 h <= 7.5 h band; measured/predicted 0.72x (5.12 h
  predicted; b3 0.83x, b2 0.85x — the warmup factor keeps converging; the
  all-closed-shell svp-heavy composition ran the fastest stratum pace yet).
- **P10 HIT**: ZERO partition-fix guard firings batch-wide — the
  preregistered "EIGHTH consecutive" prediction held (b1, b2, b3 and the
  five b-series batches before; the adjudicator's P10 note ordinal "sixth"
  is the stale b2-era display label documented above).

**U-mover arbitration (both preregistered BEFORE any interpretation; the
predictions on file held)**: both named candidates fired, both the SAME
mechanism class — the b4 strict lane moved AWAY from the 3001b34-era value
that the frozen reference AND the official 1.8.1 wheel still share (the
pre-MR-10 shared grid-incompleteness; the wheel carries the racy kernel):
1. **epinephrine_m062x_def2-svp**: b4 R bitwise -630.4449191174929 vs
   frozen/upstream -630.4449184011164/-630.4449184011157 (displacement
   +7.16e-07; B 2.27e-13 / C 2.02e-11 from R hold). Arbiter preregistered
   03:34:00Z before measurement; pure-pyscf CPU fp64 (converged, 12
   cycles, 46.5 s) e_tot -630.4449191174863: **|CPU - NEW| = 6.59e-12 vs
   |CPU - OLD| = 7.16e-07 — H1 DECISIVE, CPU sides with our new value**
   (the fifth H1-class arbiter of the series; the b1 epinephrine/b3lyp
   variant of the same molecule was H1 at 1.61e-11).
2. **epinephrine_wb97mv_def2-svp** (RSH+VV10): b4 R bitwise
   -630.4748492695908 vs frozen/upstream -630.474849020397x (displacement
   -2.49e-07; B 9.25e-11 / C 9.80e-11 hold). Arbiter preregistered
   05:11:00Z before measurement; v1 pure-pyscf CPU fp64 (converged, 13
   cycles, 1672 s) e_tot -630.4748493020086: **|CPU - NEW| = 3.24e-08 vs
   |CPU - OLD| = 2.82e-07 — H3 by the letter (8.7x closer to NEW but
   outside the 1e-9 margin), the anticipated RSH+VV10 class**; the
   pre-declared per-term escalation v2 (preregistered 05:47:30Z before
   its computation; v1 CPU terms + the b4 b0 samples' own recorded GPU
   terms, no re-run): **3/3 terms closer to NEW** (e1 1.06e-05 vs
   3.35e-05; coul 1.05e-05 vs 3.29e-05; folded-exc 7.80e-08 vs 3.26e-07)
   **plus the total — H3 with term-level directional corroboration toward
   the NEW lane** (the b2 lidocaine_wb97mv attribution pattern: the
   CPU-vs-GPU wb97mv implementation floor dominates the residual; the
   e1/coul floors ~1e-5 cancel between comparisons). Both cells recorded
   as upstream-inherited, never our regression; P5/P8 stay honest MISSes
   in their own denominators with the attribution attached.

**Honest limits**: 24 of the 117-cell post-b4 remainder covered;
mechanical top-ratio selection; regression continuity, not speedup; no
acceptance or default claims; gates unchanged; the b1/b2/b3 honest events
stay in their own denominators. Evidence: `2026-10-24-mr13-regression-b4/`
(65-file hash manifest incl. both launch-race pid strays + 96 role-sample
files; 13 mid-batch incremental prechecks + 2 independent-recompute dry
runs + 2 value-level crosschecks all AGREE; both arbiters'
prereg/result/log/pid files preserved); effects
index `mr13-newsource-regression-b4-20261024` (156; sweep 1113 OK /
0 missing / exactly the 4 documented mismatches — the same set as the
index-155 sweep, zero new; the b4 staging files stay hash-stable per the
b2-erratum lesson: the batch-dir docs draft is never mutated in place).

## MR-13-b5: new-source regression batch b5 — the fifth risk stratum, UKS returns (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the fifth production
batch at the current frozen source (launch HEAD `80f7d23`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b1/b2/b3/b4 measured it at batch scale, four
consecutive P2+P6 clean strata). b5 takes the NEXT stratum: **pool ranks
73-96 of the 117-cell post-b4 remainder by frozen B/C-vs-R gate ratio
(0.198 -> 0.153), mechanically selected** (generator re-pinned to 80f7d23
at the b4 exit; schedule sha `4861d6b4…`; selection sha `77db2040…`;
selection CONTENT and ORDER byte-identical to the DRAFT generated during
the b4 batch — only source_commit and the b4-exclusion changed). ALL 24
cells expected-pass. **UKS content RETURNS** after b4's all-closed-shell
stratum: histamine_m062x q+1s1 (0.189, the b1-melatonin cation
switch-cycle class; b1 melatonin crossed dg at 0.854 of gate, the b2/b3
cation variants passed at 0.448/0.294/0.276) and the two nitrogen_dioxide
q0s1 open-shell NEUTRAL doublets (b3lyp 0.172, r2scan3c 0.166 — the first
q0s1 cells in the MR-13 series; the MR-7 b-series covered 8 open-shell
cells all-pass at 3001b34).

**Selection composition** (from cells-selection-mr13-b5.json; risk span
0.198 -> 0.153, risk-descending order): serotonin x4 (all four methods),
epinephrine x3 (m062x-tzvp 0.183 / wb97mv-tzvp 0.175 / b3lyp-tzvp 0.153 —
the named U-mover candidates, the b4 svp siblings of the same molecule
moved 7.16e-07 / 2.49e-07 and were arbitrated H1 / H3-with-corroboration),
dopamine x3, nicotine x3, glycylglycine x2, nitrogen_dioxide x2 (the q0s1
doublets), phenylalanine x2, and one each of acetic_acid, acetamide, furan,
histamine (the q+1s1 cation), tryptophan; methods m062x x8 / wb97mv x7 /
b3lyp x7 / r2scan3c x2; bases def2-tzvp x15 / def2-svp x7 / def2-mtzvpp x2;
21 closed-shell RKS + 3 UKS cells; frozen references extracted from the
raw samples of the 3001b34-class source batches (source_batch b4 x15,
backfill x6, b3 x3). The two r2scan3c cells (frozen we_bc 6.14e-12 /
2.61e-12, frozen O-vs-R bitwise-zero) carry the b3/b4-identified P4
ULP-floor candidate class.

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot SELF-CREATED
by the runner at /data/tmp/gpu4pyscf-agent/execution/mr13-b5-snapshot-80f7d23
(7295 files read-only + inventory + verified — the launch attempt-1 cp -a
snapshot was refused by the runner's fail-closed marker check, evidence
preserved; one bounded retry clean); GPU-idle-gated serial arms; fresh
subprocess per arm; driver 3641634 launched 2026-10-24T06:05:05Z, batch
pass finished 2026-10-24T10:52:13Z (wall 4.779 h); wait-gate auto-ran both
adjudicators on driver exit (INCREMENTAL_DONE 10:52:33Z + FINAL_DONE
10:52:34Z, both rc=0); exit monitor (sleep-240 ordering fix) ran the final
recompute (RECOMPUTE_DONE 10:57:29Z rc=0; crosscheck vs adjudicator AGREE —
24 cells compared, zero value disagreements).

**Preregistration** (registered 2026-10-24T06:02:00Z before any
measurement; sha `dc801a73…`): NO corrections at registration
(the schedule was regenerated at the re-pinned SOURCE_COMMIT before this
registration; validator PASS first try — the memory rule: run the
validator before registering any batch preregistration; the two b3-era
validator classes and the b4-era schema corrections did not recur).

**Tooling verification** (checked 2026-10-24T06:15Z mid-batch, BEFORE the
wait-gate first ran either script): the adjudicator/recompute were adapted
from the b4-CORRECTED versions at promotion and this session verified by
direct inspection WALL_H = 7.5 (the b5 preregistration band), the P10 note
= NINTH consecutive, n_pass_cells = 24, and zero stale b4 narrative — the
b2/b3/b4 pre-exit tooling-erratum class fixed one stage earlier this
batch. The known gap REMAINS by design: the adjudicator's P4 energy branch
covers only the B role — the preregistered P4 letter is applied to BOTH
roles at review (same rule as the b3/b4 exit reviews).

**Results**: FULL CLEAN SWEEP — the first MR-13 stratum with every clause
HIT (P7 N/A-by-design): 24/24 cells complete, ALL 24 all-three-roles PASS
under the unchanged gates; 384 final arms rc=0; zero exhausted retries;
zero guard firings; recompute P0/P1/P2/P3/P4/P5/P6/P8/P9/P10 HIT + P7
N/A; crosscheck_vs_adjudicator AGREE (24 cells, zero value disagreements);
decision rules both TRUE (production-continuity P1+P2+P10 = the FIFTH
consecutive stratum; MR-12 batch-scale P6 = the FIFTH consecutive batch).
- **P1 HIT**: 24/24 cells complete (n_scheduled 24), zero exhausted-retry
  arms (notes: 0 retry-clean attempt events).
- **P2 HIT — ZERO gate failures batch-wide, the first b-batch with NO
  movers of ANY kind**: worst B dE 4.538e-10 / worst C dE 4.575e-10 (both
  serotonin_wb97mv_def2-tzvp); worst dg B 5.976e-06 / C 5.641e-06 (both
  phenylalanine_b3lyp_def2-tzvp) vs the 3e-5 gate. The U-mover watch: ALL
  THREE named epinephrine tzvp candidates DID NOT FIRE — epinephrine_m062x
  U 2.3e-13 (frozen 4.5e-13), epinephrine_wb97mv U 9.1e-13 (frozen 6.8e-13,
  the batch's worst O-lane dE and still under the 5e-12 P8 floor),
  epinephrine_b3lyp U landed bitwise-ON the frozen value (1.1e-12 -> 0.0) —
  the b4 svp-sibling mover class (7.16e-07 / 2.49e-07) did NOT recur on any
  tzvp sibling. The UKS switch-cycle watch: histamine_m062x q+1s1 PASSED
  comfortably (C dE 1.3e-12, C dg 5.4e-06 = 0.18x of gate; auto switch
  fired at cycle 8 of 20 and converged = the benign b2/b3 cation class,
  NOT the b1-melatonin 0.854 crossing); nitrogen_dioxide_b3lyp q0s1
  PASSED clean (U 5.7e-14, C 5.7e-14, dg 5.0e-06; switch cycle 6 of 12
  benign) and nitrogen_dioxide_r2scan3c q0s1 PASSED clean (U 0.0 -> 1.7e-13
  = the preregistered ULP-floor class; C 1.4e-12; dg 4.4e-06; switch cycle
  7 of 11 benign) — the first q0s1 open-shell neutrals of the MR-13 series
  behave like the MR-7 b-series precedent.
- **P3 HIT**: B-lane floor worst dE 4.538e-10 (serotonin_wb97mv; band
  1e-9; sequence b1 3.5e-11, b2 4.11e-10, b3 4.13e-10, b4 1.37e-10,
  b5 4.54e-10).
- **P4 HIT with the both-roles letter applied at review**: the
  adjudicator's B branch found ZERO informative over-2x deviations; the
  letter (B AND C roles) worst C-role dg ratio = 1.140x
  (dopamine_m062x_def2-svp) — clean. The only over-2x entry in ANY lane is
  nitrogen_dioxide_r2scan3c upstream dg 1.22e-10 vs frozen 2.59e-13
  (121x): both values sit far below the 1e-8 cross-process noise floor the
  P4/P8 tests exclude from ratio testing — a near-zero-denominator
  artifact of the preregistered r2scan3c ULP-floor class, absolutely
  negligible (nine orders under the 3e-5 gate). Our-lane max ratio
  1.140x (b4 1.46x).
- **P5 HIT**: R continuity vs the 3001b34-class frozen R lane — worst dE
  6.821e-13 (tryptophan_wb97mv_def2-tzvp) within the 2e-12 envelope; two
  above-floor noise-class cells recorded (dopamine_wb97mv 4.55e-13 >
  floor 3.00e-13, glycylglycine_m062x 4.55e-13 > floor 3.41e-13 — both
  inside the envelope, the documented cross-session noise class); NO
  movers (b4's two arbitrated epinephrine svp movers have no b5 analog).
- **P6 HIT DECISIVE**: 24/24 cells R-block bitwise-identical across the
  four fresh-process blocks (prereg band >= 19/24) — the FIFTH
  consecutive batch (b1-b5); zero non-bitwise exceptions, zero envelope
  violations; the MR-12 mechanism holds at batch scale.
- **P7 N/A-by-design** (zero soft-mode cells in the selection pool; the
  adjudicator's PENDING display label is the known rendering with
  na_by_design=True; the soft-mode persistence evidence lives in b1 P7).
- **P8 HIT**: official-lane continuity floors MET with zero ratio
  exceedances — worst O dE 9.095e-13 (epinephrine_wb97mv_def2-tzvp) vs the
  5e-12 band; worst O dg 2.281e-08 (tryptophan_wb97mv_def2-tzvp) vs the
  2e-7 band. The preregistered arbiter protocol (runbook 2b) was NOT
  triggered: no cell's official lane exceeded its floor band.
- **P9 HIT**: wall 4.779 h <= the 7.5 h preregistration band = 0.80x of
  the 6.00 h conservative prediction (model 3.53 h x the 1.70 b1-calibrated
  warmup factor; factor sequence b2 0.85x, b3 0.83x, b4 0.72x, b5 0.80x —
  the conservative model stays an upper bound).
- **P10 HIT**: ZERO guard firings on all 384 final arms — the preregistered
  NINTH-consecutive zero-firing prediction held.

**No arbiters this batch**: no U-role mover fired (P5/P8 both HIT), so the
preregistered conditional pyscf-CPU-fp64 arbiter protocol was never
activated; the b1-b4 arbiter tally stands at seven (five H1 = our lane
correct, two H3-with-corroboration).

**Tooling erratum (exit-review, documented per the b3/b4 convention)**: the
final adjudicator's P9 note text hardcodes "model 3.01 h" — a stale b4-era
figure (b4: 3.01 h x 1.70 = 5.12 h predicted) that survived the promotion
edit; the b5 preregistration and the docstring both carry the correct
3.53 h. Verdict UNAFFECTED: the P9 clause tests the measured 4.779 h wall
against the 7.5 h band (both correct in the output); the model figure is
informational context only. The 06:15Z pre-exit inspection checklist
(WALL_H / P10 ordinal / n_pass_cells / stale narrative) did not include
the P9 note's model figure — checklist extended for any successor
adjudicator adaptation. Found at exit review 2026-10-24T10:5xZ, recorded
before docs; final-adjudication.json left as-written (immutable output),
this section is the erratum record.

**Honest limits**: 24 of the 93-cell post-b5 remainder; mechanical
top-ratio selection (coverage not re-optimized); regression continuity,
not new speedup evidence; no cross-batch timing claims; P5 mixes source
change and cross-session drift; the O lane runs the unchanged official
package with its own upstream-inherited classes; the q0s1 open-shell
class is new to the MR-13 series though covered by the MR-7 b-series; no
acceptance/default claims; gates unchanged.

## MR-13-b6: new-source regression batch b6 — the sixth risk stratum, the card-named open-shell controls (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the sixth production
batch at the current frozen source (launch HEAD `8879d0e`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b1-b5 measured it at batch scale — b5 the FULL
CLEAN SWEEP). b6 takes the NEXT stratum: **pool ranks 97-120 of the 93-cell
post-b5 remainder by frozen B/C-vs-R gate ratio (0.153 -> 0.109),
mechanically selected** (generator re-pinned to 8879d0e at the b5 exit;
schedule sha `8ebfeb60…`; selection sha `216129eb…`). ALL 24 cells
expected-pass. **The card-named adjacent controls arrive**: ethoxy_b3lyp
q0s1 0.149 (the methoxy-sibling radical the mainline card lists as an
adjacent control) and oxygen_triplet q0s2 0.130 (the first s2 cell of the
MR-13 series). 7 UKS cells total: THREE histamine q+1s1 cations (b3lyp
0.142, r2scan3c 0.125, wb97mv 0.114 — the b1-melatonin switch-cycle class;
b5's histamine_m062x q+1s1 passed at 0.18x of gate with a benign switch at
cycle 8/20), THREE q0s1 open-shell neutral doublets (nitrogen_dioxide_m062x
0.153 the stratum top ratio, nitrogen_dioxide_wb97mv 0.142, ethoxy_b3lyp),
oxygen_triplet q0s2.

**Selection composition** (from cells-selection-mr13-b6.json; risk span
0.153 -> 0.109, risk-descending order): histamine x9 (all four methods
incl. the three q+1s1 cations), acetone x3, acetaldehyde x2, furan x2,
nicotine x2, nitrogen_dioxide x2, pyridine, nitromethane, ethoxy, oxygen;
methods b3lyp x7 / wb97mv x7 / m062x x5 / r2scan3c x5; bases def2-svp x12 /
def2-tzvp x5 / def2-mtzvpp x4; 17 closed-shell RKS + 7 UKS cells; frozen
references extracted from the raw samples of the 3001b34-class source
batches. The FIVE r2scan3c/mtzvpp cells (nicotine 0.148, nitromethane
0.134, histamine 0.125, histamine q+1s1 0.125, acetone 0.117) carry the
b3/b4/b5-identified P4 ULP-floor candidate class.

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot SELF-CREATED
by the runner at /data/tmp/gpu4pyscf-agent/execution/mr13-b6-snapshot-8879d0e
(7295 files read-only + inventory + verified); GPU-idle-gated serial arms;
fresh subprocess per arm; driver 3693881 launched 2026-10-24T11:16:51Z,
batch pass finished 14:14:37Z (wall 2.96 h); wait-gate
auto-ran both adjudicators on driver exit (INCREMENTAL_DONE +
FINAL_DONE rc=0 both at 14:15:21Z); exit monitor (sleep-240 ordering fix)
ran the final recompute (RECOMPUTE_DONE rc=0 14:19:46Z; crosscheck vs
adjudicator AGREE, zero value disagreements).

**Preregistration** (registered 2026-10-24T11:14:24Z before any
measurement; sha `4cb5ff87…`): TWO validator-caught corrections at
authoring, documented in the file (the designed b2-P6 class: two tag
shorthands in prose fixed to plain prose; no prediction content changed);
validator PASS after both; the memory rule (validator BEFORE registration)
applied.

**Tooling verification** (pre-launch 2026-10-24T11:15Z, BEFORE the batch
started — one stage earlier than the b2/b3/b4 mid-batch inspections): the
adjudicator/recompute/launcher adapted from the b5 versions with the
EXTENDED checklist (the b5-exit lesson added the P9-note model figure).
The checklist CAUGHT five stale b5 classes pre-launch, two of them
runtime-fatal (bare `mr13-b5-schedule.json` load paths in
adjudicate_final.py:132 and independent_recompute_mr13.py:61,209) plus the
launcher's b5 evidence-root dirs, the split-line NINTH ordinal, and the
docstring model/predicted figures. After fixes: WALL_H 7.5, P10 TENTH, P9
note 'model 2.66 h' (the b5-exit erratum class fixed one stage earlier).
Red tests honest (rc=1 refusals on missing driver.log; incremental rc=0
0-cell). The known-by-design gap REMAINS: the adjudicator's P4 energy
branch covers only the B role — the preregistered P4 letter is applied to
BOTH roles at review (same rule as the b3/b4/b5 exit reviews).

**Launch-race erratum** (bounded, no damage): the first exit-monitor
instance was armed with an EMPTY pid — the arming command's `&&...&`
backgrounding placed the DPID assignment in a subshell, so `$DPID` was
unset at the monitor invocation; it logged a false DRIVER_EXITED
11:17:16Z and entered its sleep-240; killed ~11:17:4xZ BEFORE any
premature recompute ran (no RECOMPUTE_DONE line from the broken instance;
its two log lines stay in exit-monitor.log as the erratum record);
re-armed correctly 11:17:45Z pid=3693881. The wait-gate was armed
correctly from the start (11:17:14Z).

**Results**: FULL CLEAN SWEEP — the SECOND consecutive MR-13 stratum with
every clause HIT (P7 N/A-by-design): 24/24 cells complete, every cell
all-three-roles PASS, all 384 final arms rc=0, zero exhausted retries,
zero guard firings; independent recompute final AGREE on every clause
(CROSSCHECK_vs_adjudicator HIT, 24 cells zero value disagreements).
- **P1 HIT**: 24/24 completion, zero exhausted-retry arms (the corrected
  P1 wording: the C role is the plain auto lane, no strict-match
  certification exists in this protocol).
- **P2 HIT**: ZERO gate failures batch-wide — the second consecutive
  b-batch with no movers of any kind. Worst dE: U 6.821e-13
  (nicotine_m062x_tzvp), B 1.811e-10 (nitrogen_dioxide_wb97mv_svp), C
  1.360e-10 (same cell) vs the 1e-6 gate; worst dg: U 2.012e-08
  (acetone_wb97mv_tzvp), B 4.628e-06 / C 4.472e-06 (both
  nicotine_r2scan3c_mtzvpp) vs 3e-5. **All 7 UKS watch cells PASSED
  all-three-roles**: the three histamine q+1s1 cations (b3lyp U dE
  2.27e-13 dg 8.02e-10; r2scan3c 1.14e-13 / 3.22e-13; wb97mv 2.27e-13 /
  7.33e-09 — the b2/b3 benign cation class, no switch-cycle story), the
  card-named ethoxy_b3lyp q0s1 methoxy-sibling control (U dE 1.14e-13 dg
  1.05e-10 — the methoxy soft-direction class does NOT recur at ethoxy),
  both nitrogen_dioxide q0s1 doublets (m062x 5.68e-14 / 1.52e-10; wb97mv
  1.71e-13 / 1.42e-09 — the b5 doublet class holds), and
  oxygen_triplet_m062x q0s2 (U dE 0.0 BITWISE-ON the frozen value, dg
  2.66e-11 — the first s2 cell of the series, clean).
- **P3 HIT**: B-lane energy floor worst 1.811e-10 vs the 1e-9 band
  (sequence b1 3.5e-11, b2 4.11e-10, b3 4.13e-10, b4 1.37e-10, b5
  4.54e-10, b6 1.81e-10).
- **P4 HIT** with the both-roles letter applied at review (the
  adjudicator's B-only branch + the C role by the letter): worst dg ratio
  vs frozen B 1.082x (nicotine_r2scan3c_mtzvpp), C 1.130x
  (nitromethane_r2scan3c_mtzvpp); ZERO over-2x entries anywhere; all 48
  role-entries above the 1e-8 noise floor so the ratio test applied
  everywhere; the five r2scan3c ULP-floor candidates all landed within
  the envelope (tightest our-lane ratios of the series so far).
- **P5 HIT**: R continuity vs the 3001b34-class R lane — zero movers;
  two above-floor noise-class cells (acetone_b3lyp_svp 3.41e-13 vs floor
  3.00e-13; nicotine_m062x_tzvp 9.09e-13 vs floor 6.82e-13), both within
  the 2e-12 preregistered cross-process envelope.
- **P6 HIT DECISIVE**: 24/24 R-block bitwise (band >= 19/24 preregistered)
  — the SIXTH consecutive batch at 24/24, zero exceptions, zero envelope
  violations.
- **P7 N/A-by-design** (zero soft-mode cells in the selection pool; the
  adjudicator's PENDING display label is the known rendering with
  na_by_design=True; the soft-mode persistence evidence lives in b1 P7
  HIT).
- **P8 HIT**: official-lane continuity zero exceedances — worst U dE
  6.821e-13 (nicotine_m062x_tzvp; 7.3x under the 5e-12 floor), worst U dg
  2.012e-08 (acetone_wb97mv_tzvp; 9.9x under the 2e-7 floor), zero
  per-cell 3x ratio exceedances.
- **P9 HIT**: wall 2.96 h <= 7.5 h = 0.65x the 4.53 h conservative
  prediction (model 2.66 h x 1.70) — the FASTEST stratum pace of the
  series (factor sequence b2 0.85x, b3 0.83x, b4 0.72x, b5 0.80x, b6
  0.65x).
- **P10 HIT**: ZERO guard firings on all 384 final arms — the
  preregistered TENTH-consecutive zero-firing prediction held.

**No arbiters this batch**: no U-role mover fired (P5/P8 both HIT), so the
preregistered conditional pyscf-CPU-fp64 arbiter protocol was never
activated; the b1-b4 arbiter tally stands at seven (five H1 = our lane
correct, two H3-with-corroboration).

**Tooling errata (exit-review, documented per the b3/b4/b5 convention)**:
(1) the final adjudicator's P9 note carries a stale PREDICTED figure and
stale factor narrative — "b1-calibrated 1.70x warmup factor -> 5.12 h
predicted; b3 actual 5.32 h vs its 6.41 h predict = 0.83x, b2 0.85x" is
verbatim b4/b5-era text; the b6 preregistration carries the correct
"4.53 h predicted; measured/predicted sequence b2 0.85x, b3 0.83x, b4
0.72x, b5 0.80x", and the model figure in the same note (2.66 h) IS
correct — the pre-launch EXTENDED checklist verified the model figure
(the b5-exit erratum class) but not the predicted figure in the same
sentence. Verdict UNAFFECTED: the P9 clause tests the measured 2.96 h
wall against the 7.5 h band. final-adjudication.json left as-written
(immutable output); the successor checklist now verifies EVERY figure in
the P9 note. (2) the staged manifest writer's mid-batch FILES section
carried the b5-era 0613Z-1053Z monitoring list unadapted (those files
live in the b5 directory); caught at exit review BEFORE the first write
(the writer's fail-closed MISSING check would have refused), fixed in
place with the actual b6 files — the pre-launch checklist covered the
adjudicator/recompute/launcher but not the manifest writer's FILES list;
checklist extended for successors. Both found 2026-10-24T14:2xZ,
recorded before docs; no wrong artifact was ever emitted.

**Honest limits**: 24 of the 69-cell post-b6 remainder; mechanical
top-ratio selection (coverage not re-optimized); regression continuity,
not new speedup evidence; no cross-batch timing claims; P5 mixes source
change and cross-session drift; the O lane runs the unchanged official
package with its own upstream-inherited classes; the ethoxy q0s1 /
oxygen q0s2 cell classes are new to the MR-13 series though covered by
adjacent precedents (b5 q0s1 doublets, the MR-7 open-shell b-series); no
acceptance/default claims; gates unchanged.

Evidence: `2026-10-24-mr13-regression-b6/` (51-file hash manifest incl.
both sessions' prechecks + crosschecks + exit artifacts + the
erratum-corrected writer; 96 role-sample files; 384 raw arms); effects
index `mr13-newsource-regression-b6-20261024` (158; full-index sweep 1226
OK / 0 missing / exactly the 4 documented mismatches — the same set as
157, zero new; the batch-dir docs DRAFT never mutated in place, the
filled section ships in this commit).

## MR-13-b7: new-source regression batch b7 — the seventh risk stratum, new-molecule coverage + the open-shell classes at new methods (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the seventh production
batch at the current frozen source (launch HEAD `21cd3dc`; production code
identical to `f52735a` = 3001b34 + MR-8 b144083 + MR-10 6fde65c + MR-11
86f63b2 + MR-12 f52735a; b1-b6 measured it at batch scale — b5/b6 the two
FULL CLEAN SWEEPS). b7 takes the NEXT stratum: **pool ranks 121-144 of the
69-cell post-b6 remainder by frozen B/C-vs-R gate ratio (0.106 -> 0.077),
mechanically selected** (generator re-pinned to 21cd3dc at the b6 exit;
schedule sha `17c0267e…`; selection sha `8f18b3db…`). ALL 24 cells
expected-pass. **FIVE new-to-series molecules arrive** (acetonitrile x4
across all four methods — r2scan3c 0.106 the stratum top, ethanol-b3lyp
0.101, butadiene-r2scan3c 0.091, pyrrole-wb97mv 0.081, ethylamine-wb97mv
0.077) and the open-shell classes at NEW methods: ethoxy q0s1 x3
(wb97mv-svp 0.091 the RSH+VV10 watch class, r2scan3c 0.084 the ULP-floor
class, wb97mv-tzvp 0.077) and oxygen_triplet q0s2 at r2scan3c 0.079 (the
SECOND s2 cell of the series).

**Selection composition** (from cells-selection-mr13-b7.json; risk span
0.106 -> 0.077, risk-descending order): acetonitrile x4 (all methods),
nitromethane x3, acetaldehyde x3 (tzvp), pyridine x3, furan x2, ethoxy x3
(q0s1), histamine x1 (m062x-tzvp), ethanol, butadiene, pyrrole,
ethylamine, oxygen_triplet; methods wb97mv x8 / r2scan3c x6 / b3lyp x5 /
m062x x5; bases def2-svp x13 / def2-tzvp x5 / def2-mtzvpp x6; 20
closed-shell RKS + 4 UKS cells; frozen references extracted from the raw
samples of the 3001b34-class source batches. The SIX r2scan3c/mtzvpp
cells (acetonitrile 0.106, butadiene 0.091, pyridine 0.085, ethoxy q0s1
0.084, furan 0.081, oxygen q0s2 0.079) carry the b3/b4/b5/b6-identified
P4 ULP-floor candidate class.

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs; O=official-1.8.1 upstream, R=strict_fp64 knobs-off, B=optimized_fp64
6-knob stack, C=optimized_auto 6-knob stack); frozen snapshot SELF-CREATED
by the runner at /data/tmp/gpu4pyscf-agent/execution/mr13-b7-snapshot-21cd3dc
(7295 files read-only + inventory + verified); GPU-idle-gated serial arms;
fresh subprocess per arm; driver 3731864 launched 2026-10-24T14:38:00Z,
batch pass finished 17:04:32Z (wall 2.44 h); wait-gate
auto-ran both adjudicators on driver exit (INCREMENTAL_DONE +
FINAL_DONE rc=0 both at 17:04:45Z); exit monitor (sleep-240 ordering fix)
ran the final recompute (RECOMPUTE_DONE rc=0 17:08:49Z; crosscheck vs
adjudicator AGREE, zero value disagreements).

**Preregistration** (registered 2026-10-24T14:37:24Z before any
measurement; validator PASS; schedule sha `17c0267e…` / selection sha
`8f18b3db…`): no pre-registration corrections needed this batch (the
validator's coverage note only); the memory rule (validator BEFORE
registration) applied.

**Tooling verification** (pre-launch 2026-10-24T14:3xZ, BEFORE the batch
started): the adjudicator/recompute/launcher/manifest-writer adapted from
the b6 versions with the EXTENDED-EXTENDED checklist — both b6-exit
erratum lessons applied one stage earlier: (a) the P9 note carries EVERY
figure correct (model 2.41 h, predicted 4.10 h, the full b2-b6 factor
narrative — verified at exit: the note reads exactly as registered); (b)
the manifest writer shipped with a FRESH FILES list authored for b7 with
an EMPTY monitoring section, extended at exit with the actual b7 files
(zero stale entries — the class stayed fixed). P10 note ELEVENTH; WALL_H
7.5; the exit-monitor DPID captured from driver.pid (the b6 launch-race
class — NO empty-pid instance this launch). Red tests honest (rc=1
refusals on missing driver.log/prereg; incremental rc=0 0-cell). **ZERO
exit-review tooling errata this batch** — the first b-batch with none.
The known-by-design gap REMAINS: the adjudicator's P4 energy branch
covers only the B role — the preregistered P4 letter is applied to BOTH
roles at review (same rule as the b3-b6 exit reviews).

**Results**: 24/24 cells complete, every cell all-three-roles PASS, all
384 final arms rc=0, zero exhausted retries, zero guard firings;
independent recompute final AGREE on every clause
(CROSSCHECK_vs_adjudicator HIT, 24 cells zero value disagreements).
- **P1 HIT**: 24/24 completion, zero exhausted-retry arms.
- **P2 HIT**: ZERO gate failures batch-wide — the third consecutive
  no-movers batch. Worst dE: U 3.41e-13 (furan_r2scan3c), B 7.01e-11
  (nitromethane_wb97mv), C 4.93e-11 (ethylamine_wb97mv) vs the 1e-6 gate;
  worst dg: U 9.81e-09 (furan_wb97mv), B 3.10e-06 (histamine_m062x_tzvp),
  C 3.31e-06 (nitromethane_m062x) vs 3e-5. **All 4 UKS cells PASSED
  all-three-roles**: ethoxy_wb97mv_svp q0s1 the RSH+VV10 class (U dE
  1.71e-13, dg 3.82e-09), ethoxy_r2scan3c q0s1 (U dE 0.0 BITWISE-ON the
  frozen value, dg 1.26e-13), ethoxy_wb97mv_tzvp q0s1 (U dE 5.68e-14, dg
  5.06e-09) — the methoxy-sibling class stays benign at ALL THREE new
  methods — and oxygen_triplet_r2scan3c q0s2 (U dE 0.0 BITWISE-ON the
  frozen value, dg 2.49e-14, cleaner than the b6 m062x instance).
- **P3 HIT**: B-lane energy floor worst 7.01e-11 vs the 1e-9 band (the
  lowest since b1; sequence b1 3.5e-11, b2 4.11e-10, b3 4.13e-10, b4
  1.37e-10, b5 4.54e-10, b6 1.81e-10, b7 7.01e-11).
- **P4 HIT** with the both-roles letter applied at review: worst dg ratio
  vs frozen B 1.075x (butadiene_r2scan3c), C 1.174x (pyridine_r2scan3c);
  ZERO over-2x entries anywhere; all 48 role-entries above the 1e-8 noise
  floor so the ratio test applied everywhere; the six r2scan3c ULP-floor
  candidates all landed within the envelope.
- **P5 HIT**: R continuity vs the 3001b34-class R lane — zero movers
  (third consecutive batch); two above-floor noise-class cells
  (furan_r2scan3c 3.41e-13 and pyrrole_wb97mv 3.41e-13, both vs floor
  3.00e-13), both within the 2e-12 preregistered cross-process envelope.
- **P6 HIT DECISIVE**: 24/24 R-block bitwise (band >= 19/24 preregistered)
  — the SEVENTH consecutive batch at 24/24, zero exceptions, zero
  envelope violations.
- **P7 N/A-by-design** (zero soft-mode cells in the selection pool; the
  adjudicator's PENDING display label is the known rendering with
  na_by_design=True).
- **P8 honest informative MISS** — exactly the preregistered r2scan3c
  ULP-floor class: ONE per-cell 3x exceedance, furan_r2scan3c O dE
  0.0->3.41e-13 / dg 1.81e-13->2.45e-13, against a BITWISE-ZERO frozen
  denominator (any nonzero new value exceeds 3x of zero). The batch-wide
  floors PASS decisively (worst U dE 3.41e-13 = 14.7x under the 5e-12
  floor; worst U dg 9.81e-09 = 20x under the 2e-7 floor); no mover of any
  kind; the class was preregistered as anticipated-informative ("the
  r2scan3c/mtzvpp O-lane entries may recur on the six b7 r2scan3c
  cells"). Recorded as the same class as the b3/b4 P4/P8 informative
  misses and the b5 noise-floor-excluded 121x entry — never a gate
  failure, never relabeled; the b1-b4 arbiter tally stands (b5/b6/b7 zero
  arbiters).
- **P9 HIT**: wall 2.44 h <= 7.5 h = 0.60x the 4.10 h conservative
  prediction (model 2.41 h x 1.70) — the fastest stratum pace of the
  series for the second consecutive batch (factor sequence b2 0.85x, b3
  0.83x, b4 0.72x, b5 0.80x, b6 0.65x, b7 0.60x).
- **P10 HIT**: ZERO guard firings on all 384 final arms — the
  preregistered ELEVENTH-consecutive zero-firing prediction held.

**Honest limits**: 24 of the 45-cell post-b7 remainder; mechanical
top-ratio selection (coverage not re-optimized); regression continuity,
not new speedup evidence; no cross-batch timing claims; P5 mixes source
change and cross-session drift; the O lane runs the unchanged official
package with its own upstream-inherited classes; the P8 informative MISS
is a near-zero-denominator artifact of the preregistered ULP-floor class,
nine orders under the gate; the five new-to-series molecules and the
ethoxy-at-new-methods / oxygen-at-r2scan3c classes extend coverage though
covered by adjacent precedents (b6 ethoxy-b3lyp + oxygen-m062x both
clean, the b5/b6 q0s1 doublets, the MR-7 open-shell b-series); no
acceptance/default claims; gates unchanged.

Evidence: `2026-10-24-mr13-regression-b7/` (46-file hash manifest incl.
the session's prechecks + dry runs + exit artifacts; 96 role-sample
files; 384 raw arms); effects-index
`mr13-newsource-regression-b7-20261024` (159; full-index sweep 1272 OK /
0 missing / exactly the 4 documented mismatches — the same set as 158,
zero new; the batch-dir docs DRAFT never mutated in place, the filled
section ships in this commit).

## MR-13-b8: new-source regression batch b8 — the eighth risk stratum, the methoxy molecule returns at a new method (2026-10-24)

**Scope**: mainline card queue item 4 continuation — the eighth production
batch at the current frozen source (launch HEAD `b85c045`; production code
identical to `f52735a`; b5/b6 the two FULL CLEAN SWEEPS, b7 all-HIT + the
one preregistered informative MISS). b8 takes the NEXT stratum: **pool
ranks 145-168 of the 45-cell post-b7 remainder by frozen B/C-vs-R gate
ratio (0.077 -> 0.059), mechanically selected** (schedule sha `a489280c…`;
selection sha `c0c4ed55…`). ALL 24 cells expected-pass. THREE q0s1 UKS
cells: ethoxy m062x-tzvp 0.073 + ethoxy b3lyp-tzvp 0.071 (the
methoxy-sibling card-named control, fourth/fifth series instances) and
**METHOXY m062x-svp 0.066 — the card-named methoxy molecule itself at a
NEW method** (the original soft-mode failures were the b3lyp/wb97mv
benchmark cells, NOT in this batch; the m062x variant is new to the
series, expected-pass at 0.066 of gate). SIX new-to-series molecule
instances (methanol, benzene x2, butane x2, methoxy) and THREE
r2scan3c/mtzvpp ULP-floor watch cells (acetaldehyde, ethanol, pyrrole).

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 384 final
arm-runs); frozen snapshot SELF-CREATED at
/data/tmp/gpu4pyscf-agent/execution/mr13-b8-snapshot-b85c045; driver
3770057 launched 2026-10-24T17:22:05Z, batch pass finished
2026-10-24T19:47:13Z (wall 2.42 h); wait-gate auto-ran both
adjudicators (INCREMENTAL_DONE + FINAL_DONE rc=0); exit monitor ran the
final recompute (RECOMPUTE_DONE rc=0; crosscheck
RECOMPUTE_DONE 19:51:40Z rc=0; crosscheck HIT, zero value disagreements across 24 cells).

**Preregistration** (registered 2026-10-24T17:22:02Z before any
measurement; validator PASS; schedule sha `a489280c…`): no corrections
needed (coverage note only).

**Tooling** (pre-launch): the stable EXTENDED-EXTENDED checklist (zero
errata in b7): P9 note model 2.39 / predicted 4.06 / full b2-b7 narrative;
manifest writer FRESH FILES; P10 TWELFTH; DPID from driver.pid. Red tests
honest. Generator authoring note: two bounded false starts (docstring span
extraction + a heredoc escaping class), both caught by ast.parse before
anything ran. The known-by-design gap REMAINS: P4 letter BOTH roles at
review.

**Results**: FULL CLEAN SWEEP — third of the series (b5/b6/b8): 24/24
cells pass every role and every clause; P1-P6 + P8-P10 all HIT, P7
N/A-by-design. Zero movers, zero arbiters batch-wide.
- **P1 **HIT**: 24/24 complete, 384/384 arm-runs rc=0, zero exhausted retries.**: completion.
- **P2 **HIT**: zero gate failures batch-wide (worst dE U 2.84e-13 pyridine_wb97mv / B 1.04e-10 acetone_wb97mv / C 1.09e-10 benzene_wb97mv; worst dg U 1.59e-08 pyridine_wb97mv / B 2.39e-06 methanol_b3lyp / C 2.24e-06 butadiene_b3lyp — all far under 1e-6 / 3e-5). The three q0s1 UKS watch: gate regression incl. the three q0s1 UKS watch
  cells (ethoxy m062x/b3lyp tzvp; METHOXY m062x svp the card-named
  molecule at a new method).
- **P3 **HIT**: B-lane energy floor 1.04e-10 acetone_wb97mv (sequence b1 3.5e-11, b2 4.11e-10, b3 4.13e-10, b4 1.37e-10, b5 4.54e-10, b6 1.81e-10, b7 7.01e-11, b8 1.04e-10).**: B-lane energy floor (b1 3.5e-11, b2 4.11e-10, b3
  4.13e-10, b4 1.37e-10, b5 4.54e-10, b6 1.81e-10, b7 7.01e-11).
- **P4 **HIT** with the both-roles letter: worst above-floor dg ratio vs frozen B 1.083x methanol_b3lyp / C 1.262x pyrrole_r2scan3c, zero over-2x entries anywhere (b7 B 1.075x / C 1.174x).**: frozen gradient envelope (both-roles letter; the
  three r2scan3c ULP-floor candidates; b7 B 1.075x / C 1.174x).
- **P5 **HIT**: zero movers, zero above-floor noise-class cells (cleaner than b7; fourth consecutive no-movers batch).**: R continuity (b5/b6/b7 zero movers three
  consecutive batches).
- **P6 **HIT DECISIVE**: 24/24 R-block bitwise, EIGHTH consecutive batch.**: R-block bitwise (band >= 19/24; b1-b7 seven
  consecutive 24/24).
- **P7 N/A-by-design** (zero soft-mode cells in the pool; PENDING display
  = known rendering).
- **P8 **HIT**: worst O dE 2.84e-13 = 17.6x under the 5e-12 floor, worst O dg 1.59e-08 = 12.6x under 2e-7, zero per-cell ratio exceedances — the three r2scan3c ULP-floor watch cells (acetaldehyde 2.27e-13 / ethanol 1.99e-13 / pyrrole) all landed on NONZERO frozen denominators, so the preregistered b7-furan bitwise-zero informative class did NOT fire this batch (floors 5e-12 / 2e-7;: official-lane continuity (floors 5e-12 / 2e-7; the
  b7 furan bitwise-zero-denominator informative class preregistered for
  the three r2scan3c cells).
- **P9 **HIT**: wall 2.42h <= 7.5h = 0.60x the 4.06h prediction (model 2.39h x 1.70; sequence b2 0.85x b3 0.83x b4 0.72x b5 0.80x b6 0.65x b7 0.60x b8 0.60x).**: wall (band 7.5 h; model 2.39 h; predicted 4.06 h;
  sequence b2 0.85x b3 0.83x b4 0.72x b5 0.80x b6 0.65x b7 0.60x).
- **P10 **HIT**: zero firings across 384 arms, TWELFTH consecutive prediction held.**: guard firings (the TWELFTH consecutive
  zero-firing prediction).

**Arbiters**: none — zero movers of any kind (the b5/b6/b7 precedent;
the b1-b4 tally stands seven: five H1 + two H3-with-corroboration).

**Honest limits**: 24 of the 21-cell post-b8 remainder (b9 would be the
FINAL stratum emptying the 189-cell pool); mechanical top-ratio selection;
regression continuity, not new speedup evidence; the methoxy m062x cell
is a new-method instance of the card-named molecule, expected-pass class,
NOT a retest of the soft-mode b3lyp/wb97mv failures (those stand in their
own denominators); no acceptance/default claims; gates unchanged.

Evidence: `2026-10-24-mr13-regression-b8/` (27-file hash manifest — the
FRESH-FILES monitoring section extended at exit with the actual b8
mid-batch crosscheck + zero arbiter files; 96 role-sample files; 384 raw
arms); effects-index `mr13-newsource-regression-b8-20261024` (160;
full-index sweep 1299 OK / 0 missing / exactly the 4 documented
mismatches — the same set as 159, zero new; the batch-dir docs DRAFT
never mutated in place, the filled section ships in this commit).

## MR-13-b9: new-source regression batch b9 — the FINAL stratum, the 189-cell pool EMPTIED (2026-10-24)

**Scope**: mainline card queue item 4 completion — the ninth and FINAL
production batch at the current frozen source (launch HEAD `21706ae`;
production code identical to `f52735a`; b5/b6/b8 the three FULL CLEAN
SWEEPS). b9 takes ALL 21 remaining cells of the 189-cell
3001b34-class pool after b2..b8 consumed ranks 1-168: **pool ranks
169-189 by frozen B/C-vs-R gate ratio (0.058 -> 0.015), mechanically
selected — the batch that EMPTIES the pool and closes the
production-continuity regression coverage** (schedule sha `964e35cb…`;
selection sha from cells-selection-mr13-b9.json). ALL 21 cells
expected-pass. ONE q0s1 UKS cell: **METHOXY r2scan3c-mtzvpp 0.040 — the
card-named molecule at its FOURTH series method (b1 wb97mv/b3lyp svp +
m062x tzvp; b8 m062x svp all-benign) AND an ULP-floor cell — the double
watch class**. FIVE r2scan3c/mtzvpp ULP-floor watch cells (methanol,
benzene, ethylamine, methoxy, methylamine). ZERO new-to-series molecules
(every molecule has b1-b8 series precedent — the ranked-pool tail).

**Protocol**: identical balanced-blocks-v1 (4 roles x 4 blocks, 336
final arm-runs); frozen snapshot SELF-CREATED at
/data/tmp/gpu4pyscf-agent/execution/mr13-b9-snapshot-21706ae; driver
PID 3807526 from driver.pid launched 2026-10-24T20:06:39Z, batch pass
finished 2026-10-24T22:11:18Z (wall 2.07 h); wait-gate auto-ran both
adjudicators (INCREMENTAL_DONE + FINAL_DONE rc=0 at 22:12:01Z); exit
monitor ran the final recompute (RECOMPUTE_DONE rc=0 at 22:15:58Z;
crosscheck 210/210 role-value comparisons AGREE, zero disagreements —
21 cells both sides, saved midbatch-crosscheck-2026102422165Z.json).

**Preregistration** (registered 2026-10-24T20:05:19Z before any
measurement; validator PASS; schedule sha `964e35cb…`): no corrections
needed at validation (coverage note only; the P1 auto-lane wording —
plain auto lane, no strict-match certification in this protocol — ships
natively in the b9 prereg as the corrected b-series wording).

**Tooling** (pre-launch): adapted from the b8 FULL-CLEAN-SWEEP set with
ast.parse + residual-hardcode sweeps; cell-count fixes N 24->21 and P6
band 19/24 -> 17/21 (the same >=79% preregistered fraction); P9 note
model 2.07 / predicted 3.52 / b2-b8 narrative; P10 THIRTEENTH; source
narrative 21706ae. Red tests honest (incremental rc=0 at 0 cells;
recompute/crosscheck/final fail closed pre-launch on the missing-file
class). Generator: single clean pass, zero authoring false starts. The
known-by-design gap REMAINS: P4 letter BOTH roles at review.

**Results**: FULL CLEAN SWEEP — the fourth of the series (b5/b6/b8/b9):
P1-P6+P8-P10 all HIT, P7 N/A-by-design; zero movers, zero arbiters.
- **P1 HIT**: completion (21/21 cells, 336 final arm-runs, zero
  exhausted-retry arms, zero retry events).
- **P2 HIT**: zero gate failures batch-wide — the FIFTH consecutive
  no-movers batch. The double-watch cell methoxy r2scan3c q0s1 PASSED
  clean: U dE 0.0 (bitwise-ON the frozen zero) dg 1.38e-12; B dE
  2.50e-12 dg 9.65e-07; C dE 2.61e-12 dg 1.16e-06. Batch-wide worst:
  dE U 3.98e-13 / B 4.07e-11 / C 4.24e-11; dg U 8.45e-09 (butane
  family U lane) / B 1.77e-06 / C 1.86e-06 vs the 3e-5 gate.
- **P3 HIT**: B-lane energy floor 4.07e-11 — the lowest tier of the
  series with b1/b7 (b1 3.5e-11, b2 4.11e-10, b3 4.13e-10, b4 1.37e-10,
  b5 4.54e-10, b6 1.81e-10, b7 7.01e-11, b8 1.04e-10, b9 4.07e-11).
- **P4 HIT** with the both-roles letter: worst dg ratio vs frozen C
  1.380x (pyrrole_b3lyp_svp) / B 1.139x (methanol_r2scan3c); ZERO
  over-2x entries anywhere; all role-entries above the 1e-8 noise
  floor (b8 B 1.083x / C 1.262x).
- **P5 HIT**: zero movers (FIFTH consecutive); ONE above-floor
  noise-class cell acetone_m062x_def2-svp 4.55e-13 vs floor 3.00e-13,
  within the 2e-12 envelope.
- **P6 HIT DECISIVE**: 21/21 R-block bitwise — the NINTH consecutive
  batch (band >= 17/21; zero exceptions, zero envelope violations).
- **P7 N/A-by-design** (zero soft-mode cells in the pool; PENDING
  display = known rendering; the soft-mode persistence evidence lives
  in b1 P7 HIT).
- **P8 HIT**: worst O dE 3.98e-13 = 12.6x under the 5e-12 floor; worst
  O dg 8.45e-09 = 23.7x under 2e-7; zero per-cell 3x exceedances — the
  five r2scan3c watch cells all landed on NONZERO frozen denominators
  so the b7-furan bitwise-zero informative class did NOT fire (same
  as b8).
- **P9 HIT**: wall 2.07 h <= 7.5 h band = 0.59x the 3.52 h prediction
  (model 2.07 x 1.70) — the FASTEST pace of the series (b2 0.85x,
  b3 0.83x, b4 0.72x, b5 0.80x, b6 0.65x, b7 0.60x, b8 0.60x, b9
  0.59x).
- **P10 HIT**: zero partition-fix guard firings across all 336 arms —
  the THIRTEENTH consecutive zero-firing prediction held.

Zero arbiters (no mover fired; the b1-b4 tally stands seven: five H1 +
two H3-with-corroboration).

**Evidence**: manifest 27 files all verified (the FRESH-FILES monitoring
section extended at exit with the actual b9 exit-time crosscheck, zero
arbiter files; 84 role-sample files; 336 raw final arms); effects-index
`mr13-newsource-regression-b9-20261024` (161; full-index sweep 1326 OK /
0 missing / exactly the 4 documented mismatches — the same set as
158/159/160, zero new; the batch-dir docs DRAFT never mutated in place,
the filled section ships in this commit).

**Honest limits**: the FINAL stratum — 21 of the 21-cell post-b8
remainder (the pool is EMPTY after this batch; no post-b9 remainder);
mechanical top-ratio selection (coverage not re-optimized); regression
continuity, not new speedup evidence; no cross-batch timing claims; P5
mixes source change and cross-session drift; the O lane runs the
unchanged official package with its own upstream-inherited classes; the
methoxy r2scan3c q0s1 cell is the card-named molecule at a fourth
method, expected-pass class, NOT a retest of the soft-mode b3lyp/wb97mv
failures (those stand in their own denominators); zero new-to-series
molecules; no acceptance/default claims; gates unchanged.

## MR-13 pool closure: the b1–b9 series complete, production-continuity regression coverage CLOSED (2026-10-24)

**What closed.** The nine new-source regression batches b1–b9 (commits
b20106e..4264bae) are complete. The 189-cell ranked pool over the
frozen 3001b34-class reference — constructed mechanically by frozen
B/C-vs-R gate ratio after b1's risk-ranked first stratum — is EMPTY:
b2–b8 consumed ranks 1-168 (24 cells each, 384 final arm-runs each),
b9 took all 21 remaining (ranks 169-189, 336 arms). Series totals
(measured from the nine final-adjudication.json files + evidence
manifests): **213 cells, 3,408 final arm-runs, every arm rc=0, total
wall 35.29 h** (per-batch walls 5.95 / 5.68 / 5.32 / 3.68 / 4.78 /
2.96 / 2.44 / 2.41 / 2.07 h — no cross-batch timing claims, the strata
differ; the wall sequence is operational record only).

**The production-continuity claim held in ALL NINE strata** (P1
completion AND P2 no-gate-regression AND P10 zero guard firings, every
batch). P6 R-block bitwise ran 100% in every batch (b1–b8 24/24, b9
21/21 — the MR-12 deterministic-e_tot mechanism at batch scale, nine
consecutive). P10 zero partition-fix firings across all ~3,408 arms
(THIRTEEN consecutive preregistered zero-firing predictions held).
Four FULL CLEAN SWEEPS (every applicable clause HIT): b5, b6, b8, b9.

**Honest events stay in their own denominators, never relabeled:**
- b1: P2 MISS two cells — melatonin q+1s1 UKS dg 4.03e-05 (near-gate)
  + epinephrine U-lane dE 1.93e-06 — BOTH arbitrated H1 (pure-pyscf
  CPU FP64 sides with OUR lane; the U-lane gap = the official 1.8.1
  wheel's shared `_screen_index` grid-incompleteness,
  upstream-inherited).
- b2: P2 HIT but worst U dE 1.16e-07 (attributed, under gate);
  P5/P8 MISS three mover cells, arbitrated — lidocaine_m062x H1,
  lidocaine_wb97mv H3-with-corroboration (RSH+VV10 class, directional
  toward NEW), vitamin_c_m062x H3-below-discrimination-scale.
- b3: P4/P8 informative MISS — the r2scan3c/mtzvpp ULP-floor class vs
  bitwise-zero frozen refs (near-zero-denominator artifacts, orders
  below every gate).
- b4: P5/P8 MISS on the two epinephrine U-movers — m062x H1 DECISIVE,
  wb97mv H3 + 3/3-term corroboration toward NEW; both
  upstream-inherited (our MR-10-fixed lanes moved off the 3001b34-era
  shared value that the frozen reference AND the official wheel still
  carry).
- b7: P8 honest informative MISS of the preregistered
  furan_r2scan3c bitwise-zero-denominator class.
- Arbiter tally for the series: seven (five H1 + two
  H3-with-corroboration). b5/b6/b8/b9: zero movers, zero arbiters
  (FIVE consecutive no-movers batches at the tail).
- The b7-furan bitwise-zero informative class did NOT fire again
  (b8/b9 watch cells all landed on nonzero frozen denominators).
- The soft-mode methoxy b3lyp/wb97mv benchmark FAIL rows and the
  frozen uc-20261018-a 213-row history remain immutable; the methoxy
  molecule itself PASSED clean at its other three methods + the b9
  r2scan3c fourth method (q0s1 U-lane bitwise-ON its frozen zero).

**Standing after closure.** The frozen-source regression program the
mainline card's queue item 4 defined is complete; new-source
production continuity is measured over the full 189-cell pool + the
b1 risk stratum, at unchanged gates (E<=1e-6 Ha, max|dg|<=3e-5
Eh/Bohr). Residual recorded classes (all upstream-inherited or
informative, none ours to relabel): the official wheel's racy screen
kernel + duplicate-pair splits; the r2scan3c/mtzvpp ULP-floor
denominators; the b2/b4 RSH+VV10 mover attributions. Effects-index
datasets 152–161 carry the per-batch sources (b1–b9 with the FU1
spot-check at 153; final full-index sweep
1326 OK / 0 missing / exactly the 4 documented mismatches). Next
mainline work picks from fresh evidence, not from this closed pool.

### FU1 verification-cost decomposition: the safe route is AT its structural floor (2026-10-24)

Bounded CPU-only analysis of the FU1 raw timings (the named residual
"reduce the safe-route verification cost"), from the three guarded-cell
`*_guarded.json` rows (f52735a, 3 repeats, medians):

| cell (lane t_scf / cycles / t_grad, s) | fp64 | auto-strict |
|---|---|---|
| epinephrine/B3LYP/SVP (247 AO) | 1.95 / 13 / 1.24 | 5.09 / 13 / 0.61 |
| methoxy/wB97M-V/SVP (UKS q0s1) | 3.69 / 11 / 0.58 | 8.21 / 11 / 0.51 |
| melatonin/wB97M-V/SVP (318 AO UKS cation) | 188.85 / 32 / 10.74 | 233.13 / 29 / 4.05 |

Reading: `t_cderi_s = 0.0` in BOTH lanes of every cell — the MR-8
CDERI sharing is already effective; the sibling costs no DF rebuild.
The auto-strict lane converges in the SAME cycle count as fp64
(13/13, 11/11; melatonin 29 vs 32), so the overhead is per-cycle
evaluation, not extra iterations. The overhead IS the same-start
strict-FP64 sibling SCF — exactly the certification contract's
minimum ("strict_match_verified" requires the sibling's own FP64
trajectory from the same dm0). At small cells that minimum is a large
fraction of a tiny total (2.6x/2.2x); at the 318-AO UKS RSH cell it
amortizes to 1.23x. The certified lane's gradient is CHEAPER than the
fp64 lane's (0.61 vs 1.24; 4.05 vs 10.74) — one certified gradient
replaces the uncached fp64 one.

**Conclusion (negative finding, evidence-backed): no further
verification-cost reduction is available at the current contract.**
The remaining reducible components were already taken (CDERI shared;
gradient computed once in the certified lane); the sibling SCF cannot
shrink without weakening same-start strict matching, which is a
contract change, not an optimization. The cost line is CLOSED at the
measured floor unless the user changes the certification contract.

### MR-14: the upstream-contribution bundle verified red→green on PRISTINE upstream 1.8.1 source (2026-10-24/25)

The MR-12-named "upstream contribution candidate", executed as its own
preregistered work unit (registered 2026-10-24T22:40:43Z before any
measurement; amendment 22:46Z disclosed the build-enabling patch 0000 =
commit 082ed72's Fortran-optional CMakeLists change, build machinery only).
The two defect fixes were extracted as MINIMAL patches against pristine
git 0a4cf33 (= the upstream 1.8.1 source our tree derives from): 0001 the
MR-10 `_screen_index` tile-loop barrier (10 lines, one `__syncthreads()`)
and 0002 the MR-12 duplicate-pair CDERI canonicalization (pure Python, 53
lines); both `git apply --check` clean on pristine. A fresh CUDA 12.8 build
of the pristine tree (+0000 only) produced all 11 libraries with zero
errors, then red→green was measured on the RTX 4090 (single GPU job, ~4 min
GPU-active, GPU idle-gated, LC-215 parked throughout).

**Red (pristine, unpatched)**: epinephrine/B3LYP/def2-SVP lands at
-630.7161527938961 — 1.932e-6 Ha off the preregistered pyscf-CPU-fp64
arbiter (9.1e-13 from the official-wheel value; 13 cycles, converged) —
the upstream defect reproduces from pristine source. The dopamine/def2-TZVP
CDERI duplicate census on pristine reads 583 classes / 374 bitwise-unequal /
max 3.344e-13 — EXACTLY the official-wheel census (MR-12), independently
confirming pristine-source == wheel behavior.

**Green (patches applied to the same pristine tree)**: with 0001 only,
epinephrine moves to 1.59e-11 / 1.61e-11 off CPU truth (two runs), and one
run plus the both-patches run are BITWISE-IDENTICAL to our production fixed
value (hex -0x1.3b5baaf543454p+9). With 0002, the duplicate census reads
0/583 bitwise-unequal (max 0.0). Patched-tree gradient norms agree with the
CPU arbiter to 2.6e-9 absolute.

**Adjudication: all preregistered clauses HIT (P1 red within the 1e-9 band
of the racy value; P2 within 1e-9 of CPU truth AND 5e-12 of the production
value at 13±2 cycles; P3 374→0 unequal classes; P4 smoke satisfied); the
decision rule (P1-observed AND P2 AND P3) is MET.** The upstream-contribution
claim holds as measured: the two minimal patches cure the two
upstream-inherited defect classes on pristine 1.8.1 source — the barrier
alone moves epinephrine from 1.932e-6 Ha off CPU truth to 1.6e-11, bitwise
onto our production value.

Honest notes: one barrier-only run sits 2 ULPs (2.27e-13) from the
production value (the pre-MR-12 per-launch ULP class; 0002 is its measured
eliminator); patch 0000 was present in EVERY run (red and green alike, so
it cannot differentiate them); no production defaults changed; our
production tree and the official read-only wheel untouched; nothing sent
anywhere — the bundle (`upstream-contribution/` in-repo: README + 3 patches)
is operator-disposition only. Evidence: `2026-10-25-mr14-upstream-contribution/`
(18-file hash manifest incl. preregistration, both patches, all six run
JSONs, build log, libgdft hashes, final adjudication); effects-index 162.

### MR-15: the 1/37 census residual bounded — post-fix large-N screen-mask census, CLOSED-AS-UNREPRODUCED (2026-10-24/25)

The last code-adjacent named open residual ("the single unexplained 1/37
MR-10 census process", MR-11 honest non-claim P6) was closed by its own
preregistered measurement-only work unit (sha
`3dfebb84...` registered 23:29:54Z BEFORE any measurement; zero source
changes; libgdft.so = the recorded MR-10-fixed build `2c063a1b...`; GPU
idle-gated; single job). Design: N fresh sequential dopamine/def2-tzvp/
level-3 processes (the event cell; 262912 points, 10725 mask cells) under
production conditions (natural random PYTHONHASHSEED, verified
hash_randomization=true in every record), each computing the production
fine-screen mask and compared to the pinned census majority `280062cbc884`
(== CPU truth per MR-10/11), with the in-process exact CPU predicate +
full npz dump reserved for any mismatch; input-layout hashes recorded per
process. V0 (2 processes, own-truth mode) passed 2/2 at 5 s/process and
selected N=300 by the preregistered wall rule.

**Result: 300/300 processes produced mask `280062cbc884` — zero variants,
zero errors, zero retries (wall 1371 s).** The variant path never fired, so
zero GPU-vs-own-CPU mismatches by construction. Layout hashes were
single-valued across all 302 dopamine processes (env `f37fa3d71e54`, bas
`8c6f88504621`, atm `9e0c903e2787`, coords/weights == the census60
constants) — the first large-N corroboration of the MR-11 layout fix
(census60 was pre-fix at N=60 with layouts varying). The optional benzene
control (preregistered wall condition met) passed 60/60 GPU == own
in-process CPU truth bitwise. All clauses P0–P4 HIT; decision rule:
**CLOSED-AS-UNREPRODUCED** — on the fixed tree (MR-10 barrier + MR-11
layout + MR-12 canonicalization) the class rate is bounded < 1.0% (95%
rule-of-three at 0/300), and the historical 1/37 (2.7%) rate is excluded
at p = (36/37)^300 = 2.7e-4.

Honest limits: the HISTORICAL 1/37 event itself stays *unexplained* — this
unit bounds its recurrence on the fixed tree, it does not retro-diagnose
the 2026-10-23 process (which ran pre-MR-11-layout-fix). The residual
nondeterminism surface after MR-15 is: (i) the cross-session RSH reference
drift class (environmental, separately documented), (ii) lanes not covered
by measured cells. No defaults changed; no historical row relabeled.
Evidence: `2026-10-25-mr15-mask-census/` (23-file hash manifest incl.
preregistration, probe/driver/chain, per-phase records.jsonl + driver
logs, final adjudication); effects-index 163 (full sweep 1367 OK /
0 missing / exactly the 4 documented mismatches, zero new).

### MR-16: lane-coverage extension stratum 1 — def2-qzvp, the anion class, and tzvp class completion (2026-10-25)

The MR-15-named residual class "lanes not covered by measured cells",
executed as the mainline card's queue item 4 (extend the cohort
regression with new non-overwriting batches): the first new-lane stratum
beyond the frozen 213-cell cohort. The mr13 b1–b9 series covered exactly
seven (xc,basis) lanes at classes (0,0)/(0,1)/(0,2)/(1,1); MR-16
measures the uncovered space: **def2-qzvp at b3lyp/m062x/wb97mv (never
measured anywhere in the program, incl. UKS at qzvp and the 522-AO
benzene cell), the anion q=−1 class (superoxide at three methods +
closed-shell methoxide), and the tzvp (0,2)/(1,1) completion.** 12 cells
(6 RKS + 6 UKS), balanced-blocks-v1, 4 roles × 4 blocks = 192 arms,
within-batch strict-FP64 comparator (no frozen references exist for new
lanes). Preregistration `6f623b45` registered 2026-10-25T00:24:37Z
BEFORE any measurement (validator PASS); launch gate HEAD 1920bd4
clean / GPU idle / LC-215 parked; snapshot self-built by the driver;
driver 00:26:03Z→01:45:18Z.

**Completion + determinism: P1 HIT and P4 HIT DECISIVELY.** 192/192
final arms rc=0, zero retries, zero partition-fix firings (the
fourteenth consecutive zero-firing prediction, P10 HIT), all SCF
converged, wall 1.31 h ≤ 6 h (P9; model 1.49 h, 1.70-factor predicted
2.54 h). **All 12 cells' R lanes are pairwise bitwise-identical across
their four fresh-process blocks** — the MR-12 cross-process determinism
extends to qzvp and anion lanes — AND on every one of the twelve cells
the official 1.8.1 wheel's fp64 lane agrees with our strict R lane at
bitwise class (worst O dE 1.4e-13; several exactly 0.0): the strongest
lane coverage yet of wheel-vs-production fp64 corroboration, on lanes
the wheel had also never been measured on. The independent recompute
(second implementation) agrees on every value; one adjudicator tooling
erratum (P9 wall regex missing re.M — PENDING at first run) was fixed
and re-run, documented here.

**P2 MISS (two honest gate failures, both preregistered-arbitrated,
both kept in the denominator):**
- `methoxy_b3lyp_def2-qzvp` q0s1 — the known methoxy soft-mode class on
  the new qzvp lane: C dE 1.62e-7 / C dg 2.38e-3 (B dg 2.17e-4), fp32
  head extends the trajectory 11→14 cycles. Preregistered pyscf-CPU-fp64
  arbiter (registered 01:48:52Z before the run): CPU lands 1.65e-12 from
  R (and 5.7e-14-class from the wheel) — **H1a confirmed: the fp64 truth
  is R/O; the auto deviation is the soft-mode trajectory**, ours-R-correct
  at CPU truth.
- `oxygen_triplet_r2scan3c_def2-mtzvpp_q-1s1` (the one cell not
  CPU-screenable pre-batch — composite) — a NEW class: the plain-auto
  lane lands **dE 2.23e-6 / dg 9.72e-5** away from a lane where strict
  R, optimized B (1.1e-12) and the official wheel (1.1e-13) all agree.
  Arbiter verdict **H3**: the CPU fp64 SCF itself lands 2.24e-6 from GPU
  fp64 R (15 vs 22 cycles; 4.46e-6 from C) — this superoxide r2SCAN-3c
  cell is intrinsically soft at the 2e-6 scale, so no truth
  discrimination at the 1e-9 margin is possible; the auto-vs-R gate
  failure stands as measured. Production guidance for this class is the
  certified safe routes (plain fp64 or auto-strict reference matching),
  NOT plain auto; no defaults changed.

**P3 MISS (informative floor-band class, science gates passed):** the
optimized 6-knob stack shifts the fp64 SCF trajectory on anion cells —
superoxide/b3lyp/svp B=C −150.173201386883 vs R=O
−150.1732014236570 (3.68e-8, 8→9 cycles) and superoxide/wb97mv/tzvp
8.79e-9 — both >2 orders under the 1e-6 gate; methoxy qzvp B 1.34e-9
(marginal). Arbiter C: CPU 1.19e-8 from R vs 2.49e-8 from B — H3 below
the discrimination scale, directionally corroborating R; arbiter D: H3
as preregistered (RSH CPU–GPU floor ~1e-8). Recorded as a new measured
class: **on q=−1 anions the optimized stack's own fp64 role can shift
landing by ~1e-8-class** — within every science gate, flagged for the
floor bands of future anion preregistrations.

Every other cell passes clean: worst remaining B dE 5.9e-12, C dE
2.3e-11 (non-anion qzvp cells at the familiar floor classes); worst
passing dg 4.3e-6. Decision rule (P1∧P2∧P4∧P10) **NOT_MET as
measured** — P2's two rows above; never relabeled; the safe routes for
both classes are already-contractual. Evidence:
`2026-10-25-mr16-lane-coverage/` (469-file hash manifest incl. both
preregistrations, schedule/selection/generator, all four adjudication
documents, the CPU-arbiter prereg/results/adjudication, driver +
wait-gate + exit-monitor logs); effects-index 164 (full sweep 1391 OK /
0 missing / exactly the 4 documented mismatches, zero new). Remaining
uncovered space after MR-16: qzvp at r2scan3c is out-of-spec (3c
prescribes mtzvpp); (0,2)/(1,1) at qzvp and larger-AO qzvp/anion cells
remain unmeasured — next strata pick from fresh evidence.

### FU2: superoxide/r2scan3c safe-route spot-check — auto+strict measured on a 3c cell for the FIRST time, guidance certified WITH cost (2026-10-25)

MR-16's one unmeasured follow-up class: the new-lane soft-anion auto
failure `oxygen_triplet_r2scan3c_def2-mtzvpp_q-1s1` carried production
guidance "certified safe routes fp64/auto-strict, NOT plain auto", but
auto-strict had never been measured on ANY r2SCAN-3c cell (the pilot
and FU1 covered b3lyp/wb97mv only), and neither safe route had an e2e
cost number on this class. FU2 closes both gaps in one bounded unit
(preregistration `cd24f903…` registered 2026-10-25T02:07:28Z BEFORE
measurement; live tree at 0f0a2c0 asserted clean at driver start; 4
lanes R/B/C/S × 4 Latin-square blocks = 16 fresh-subprocess arms
through the production `gradient_runner` path, each arm = own warmup +
1 formal sample; 16/16 rc=0, zero retries, zero gate skips, wall 2 m
56 s).

**All six preregistered clauses HIT; decision rule MET.**

- **P1 recurrence (denominator):** plain auto FAILS both gates exactly
  as MR-16 measured — dE 2.2254877194427536e-06 (bitwise-identical to
  the MR-16 value, reproduced across sessions and fresh processes),
  max|dg| 9.7207e-5; both inside the preregistered recurrence bands.
- **P2 fp64 safe route:** passes with dE 1.08e-12 and dg 3.14e-6 —
  both the MR-16 values reproduced (dE within the ×10 clause at 1.0×);
  25 cycles vs R's 22 (stack-env cycle shift, energies agree at the
  1e-12 floor).
- **P3 auto+strict (first 3c measurement) — HIT DECISIVE:** the
  matcher's own telemetry shows the mechanism working end-to-end:
  `distinct_solutions: true` with beta occ-overlap **0.561** and
  dm_max_abs_diff **0.098** (a genuinely distinct soft-anion branch,
  not fp32 noise), `strict_match_verified: true` 4/4,
  `selected: reference` — the lane ADOPTED the fp64 reference: S e_tot
  **bitwise == R** (dE exactly 0), max|dg| 2.72e-6 (fp32 gradient
  kernels on the CORRECT density pass the 3e-5 gate with an order of
  margin — the plain-auto failure is the wrong solution, not the fp32
  gradient kernels).
- **P4 cost (the card's 验证/回退真实总成本 for this class):** S/C
  sample-time ratio **2.32×** (B/C 1.11×) on this 48-AO cell; the
  mechanism decomposition: S t_scf 4.201 s = C 1.661 + R 2.382 + 0.158
  s matcher residual, at the SAME 22-cycle count — per-cycle overhead
  (the strict-fp64 sibling SCF), not extra iterations; warmups 7.77 vs
  5.07 s. Informative: the MAIN lane's own CDERI build costs ~0.065 s
  on every lane (MR-8 sharing removes the sibling rebuild, not the
  main build).
- **P0/P5 controls:** R e_tot bitwise across its 4 fresh processes
  (grad spread 6.0e-14, inside the preregistered ≤1e-9 clause); zero
  fallback events; all 16 samples converged; source clean at 0f0a2c0.
- **Erratum (fixed + rerun before reporting):** the first adjudication
  run implemented the P5 R-stability clause stricter than the
  preregistered text (bitwise-exact gradients instead of "bitwise E;
  grad spread ≤1e-9"); the prereg text is authoritative, the clause
  was reimplemented as registered and the whole adjudication rerun.

Independent recompute (separate code path over the raw records):
**38/38 checks AGREE, zero value disagreements.** Evidence:
`2026-10-25-fu2-superoxide-safeway-cost/` (44-file hash manifest);
effects-index 165 (full sweep 1435 OK / 0 missing / exactly the 4
documented mismatches, zero new). Standing guidance for the
superoxide/r2scan3c class is unchanged in content but now
cost-certified: plain fp64 (1.11× auto) or auto-strict (2.32× auto,
reference-adopting with fail-closed semantics); plain auto remains a
recorded FAIL. No defaults changed; no historical row relabeled.

### FU3: methoxy/qzvp soft-mode safe-stack spot-check — auto+strict passes under BOTH stacks (sibling LC-105-protected), plain fp64-under-stack fails under both (2026-10-25)

The second MR-16 P2 class, `methoxy_b3lyp_def2-qzvp` (soft-mode UKS
doublet, 204 AO): MR-16 had measured C (auto, 6-knob) failing the
gradient gate at dg 2.379e-3 AND B (fp64, 6-knob) failing at dg
2.168e-4, with the CPU arbiter confirming R == CPU truth. Three routes
had never been measured on this class: auto-strict under the 6-knob
stack, auto-strict under the 5-knob (BLOCKSIZE-off) stack, and plain
fp64 under the 5-knob stack. FU3 measures all of them plus fresh
R/C6 references (preregistration `71ff03fc…` registered
2026-10-25T02:27:17Z BEFORE measurement; 6 lanes × 4 blocks = 24
fresh-subprocess arms, 24/24 rc=0, zero retries, wall 4 m 08 s).

**Adjudicated exactly as preregistered — P0/P1/P2/P7 HIT, P3/P4/P6
MISS, P5 HIT; decision rule NOT_MET as preregistered, never
relabeled. The measured gate outcomes carry the guidance:**

- **P1/P2 recurrences (bitwise):** C6 reproduces MR-16 exactly (dE
  1.615e-7 passes, dg 2.379e-3 FAILS; e_tot bitwise-identical) and B6
  likewise (dE 1.341e-9 passes, dg 2.168e-4 FAILS). Denominator
  established.
- **P3 band MISS via the preregistered broadening clause, in the
  direction of safety:** S6 (6-knob auto-strict) was predicted to
  adopt the stack-env fp64 sibling and fail dg in [8e-5, 6e-4].
  Measured: the sibling's LC-105 `_strict_fp64_reference` mark forces
  bitwise-fp64 vmat even under the `GPU4PYSCF_SCF_VMAT_F32` knob, so
  the adopted solution lands **bitwise on R** (dE 0.0) and
  **dg 8.53e-7 — gates PASS**. The preregistered mechanism (sibling ≈
  B6 trajectory) is falsified by measurement: the stack env cannot
  propagate into the certified solution.
- **P4 MISS as measured (decisive negative):** B5 (5-knob plain fp64)
  fails the gradient gate at **dg 6.21e-5 (2.07× over)** with dE
  1.10e-10 (the 1e-11 predicted scale was also over-tight). Combined
  with B6: **plain fp64 under EITHER stack does not pass on this
  soft-mode cell** — the stack's env-level knobs (BLOCKSIZE, VMAT
  head/tail) shift the plain lane's landing; only the marked strict
  sibling (or the no-stack R lane) reaches the certified solution.
- **P5 HIT:** S5 (5-knob auto-strict) passes — dE 0.0 bitwise == R,
  dg 6.12e-7, strict_match_verified 4/4.
- **P6 informative band MISS:** R/C6 sample ratio 1.166 sits under the
  [1.2, 4.0] band (R's fp64 gradient phase is the slow part; the SCF
  difference is small at this size); S5/C6 2.199 is in band. Sibling
  mechanism holds: S6 t_scf 3.18 s = C6 1.46 + B6 1.13 + 0.59
  matcher; S5 3.66 = 2.86 + 0.80.
- **Cost (this class):** S5/C6 2.20×, S6/C6 1.90×, R/C6 1.17×,
  B5/C6 1.02×.

**Operative measured guidance for the qzvp soft-mode class:** safe
routes = auto-strict under EITHER stack (both bitwise-adopt R with
fp32 gradient kernels at dg 6–9e-7) or the strict R lane itself
(1.17× auto); plain fp64 under the optimized stacks is measured
FAILING under both (B6 2.17e-4, B5 6.21e-5) and must not be
presented as the safe route on this class. This refines FU1's
near-gate UKS guidance: the protective mechanism is the strict
sibling, not the 5-vs-6-knob stack choice. Independent recompute
53/53 AGREE (one badly-designed pseudo-check in the first recompute
run was replaced with value cross-checks and rerun — documented, no
value ever disagreed). Evidence:
`2026-10-25-fu3-methoxy-qzvp-safeway-cost/` (60-file hash manifest);
effects-index 166 (full sweep 1495 OK / 0 missing / exactly the 4
documented mismatches, zero new). No defaults changed; no historical
row relabeled.

## MR-17: lane-coverage stratum 2 — the qzvp charge/spin lattice COMPLETED, the 405–492-AO qzvp window measured, the methoxy soft-mode class shown molecule-specific (2026-10-25)

Batch `mr17-lanecoverage-20261025-a` at HEAD 8ed0595 (production code
identical to the f52735a lineage; only docs changed since the MR-16
launch commit 1920bd4): 12 def2-qzvp cells closing the stratum-1 gaps —
qzvp (0,2) at b3lyp + m062x (O2-triplet), the qzvp (1,1) cation
(ethylamine+ wb97mv, 381 AO — the largest charged cell measured to
date), qzvp (0,1) at m062x + wb97mv (methoxy doublets), the
anion-x-qzvp crossing (superoxide wb97mv q-1s1, closed-shell methoxide
b3lyp q-1s0), the larger-qzvp neutral window 405–492 AO (pyridine,
furan, pyrrole), and two chemically NEW qzvp doublet families (NO2 at
wb97mv; ethoxy at b3lyp — the mainline card's named adjacent control).
Preregistration sha c0b0982f registered 2026-10-25T03:03:16Z BEFORE
any measurement (schedule-vs-preregistration validator PASS after one
"closed-shell neutrals"→"closed-shell RKS cells" reword of a substring
false-positive; CPU convergence screens 9/9 open-shell/charged cells
all converged, recorded in `cpu-screens-mr17.json`). Within-batch
strict-FP64 R comparator; frozen uc-20261018-a, b-series and MR-16
rows immutable.

Driver 03:03:55Z–04:32:17Z: snapshot self-built (7299 files verified),
12 cells × 4 roles × 4 balanced blocks = **192/192 arms rc=0, zero
retries, zero partition-fix firings (15th consecutive batch), wall
1.47 h ≤ 6 h** (predicted 2.88 h; the b1 warmup factor stays
conservative). All 12 cells all-roles converged.

**P4 HIT (decisive): all 12 cells R-bitwise across fresh processes**
(cross-block e_tot spread exactly 0.0 on every cell) — the MR-12
determinism class now measured on every stratum-2 lane — and the
official 1.8.1 wheel corroborates R on all 12 (worst O dE 6.1e-10 on
the anion cell, worst O dg 7.6e-9; O dE ≤ 8.5e-14 on the methoxy
doublets).

**P2 primary gate: MISS — three honest rows, ALL on the methoxy qzvp
soft-mode family, each arbitrated:**

| cell | role | worst dE vs R | worst dg vs R | arbiter |
|---|---|---|---|---|
| methoxy_wb97mv_def2-qzvp | B | 1.69e-8 | 9.50e-4 | H3 directional-R |
| methoxy_wb97mv_def2-qzvp | C | 1.83e-6 | 1.39e-2 | H3 directional-R |
| methoxy_m062x_def2-qzvp | B | 3.22e-9 | 5.55e-5 | H1a CONFIRMED |

- **methoxy_wb97mv (B+C miss, the predicted soft-radical-at-RSH
  class):** CPU fp64 itself lands 2.8e-8 below GPU R (|CPU−R| 2.81e-8,
  |CPU−B| 4.50e-8, |CPU−C| 1.86e-6) — the MR-16-superoxide H3 shape:
  no 1e-9 truth discrimination between CPU/R/B on this intrinsically
  soft cell, C the decisive outlier. Energy-degenerate branches with
  divergent forces; both rows stay in the denominator.
- **methoxy_m062x (the FIRST measured INVERTED class): B-only miss
  with C clean.** C matches R to 4.2e-12 / dg 1.2e-6 while B descends
  to a marginally-lower (3.2e-9) branch with dg 5.5e-5. Arbiter H1a
  CONFIRMED: |CPU−R| = 3.3e-12 ≤ 1e-9 — CPU fp64 sides with the R/C/O
  triple agreement; the B lane alone reached a CPU-discordant soft
  branch. R-correct, ours-B-trajectory.

**9/12 cells pass ALL gates clean**, including every new-lane class
this batch existed to cover: the qzvp cation ethylamine+ (dg 9.97e-7),
the anion-x-qzvp crossing (superoxide wb97mv dg 5.04e-7 — **the
r2scan3c soft-anion auto class does NOT generalize to this lane**;
methoxide closed-shell dg 8.97e-7), the 405–492-AO neutrals
(pyridine 2.15e-6, furan 1.52e-6, pyrrole 1.11e-6), and the chemically
new qzvp doublets **NO2 (B dg 2.81e-6) and ethoxy (B dE 0.0 bitwise
== R, dg 9.2e-7)**. P3 floor MISS = the same two methoxy soft cells
plus the preregistered anion trajectory-shift informative class
(2.89e-8, two orders under the science gate). P6 HIT (8/8 UKS cells
converged all roles, same R branch per cell).

**Class conclusion (measured):** the methoxy qzvp soft-mode class is
molecule-specific and method-general — it fires at b3lyp (stratum 1),
m062x and wb97mv (here), always as near-degenerate energies with
divergent forces — while the chemically independent qzvp doublets NO2
and ethoxy pass everything: it is NOT a general qzvp-doublet, UKS, or
method defect. Production guidance unchanged: certified safe routes
(strict R / auto-strict same-start matching) on the methoxy-class
cells, per FU2/FU3.

Decision rule **NOT_MET as measured** (P1∧P2∧P4∧P10 required; P2
MISS), never relabeled. Independent recompute (separate
implementation, both-ways log cross-match) AGREE — zero value
disagreements. Evidence: `2026-10-25-mr17-lane-coverage/` (474-file
hash manifest incl. the preregistered CPU arbiters); effects-index
167 (full sweep 1969 OK / 0 missing / exactly the 4 documented
mismatches, zero new). No defaults changed; no historical row
relabeled.

## FU4: both new MR-17 methoxy soft cells safe-route-certified WITH cost — the family guidance table closes (2026-10-25)

Preregistration sha 18babd9c registered 2026-10-25T04:51:38Z BEFORE
measurement; driver 04:51:43Z–04:59:11Z at HEAD 1712ce2 (production
code identical to the f52735a lineage): 2 cells
(methoxy_m062x_def2-qzvp, methoxy_wb97mv_def2-qzvp — the two MR-17 P2
failures) x R/B6/C6/S6 x 4 balanced blocks = **32 fresh-subprocess
arms through the production gradient_runner path, 32/32 rc=0, zero
retries, wall 7m28s**. B6/C6 run under `upstream_comparison.STACK_ENV`
exactly as the MR-17 batch roles did; R under a zero-GPU4PYSCF env;
S6 = auto-strict same-start strict matching. **ALL clauses P0–P7 HIT,
decision rule MET**; independent recompute 62/62 AGREE, zero value
disagreements.

- **m062x (the FIRST INVERTED class):** B6 reproduces MR-17 **bitwise**
  (dE 3.217e-9 / dg 5.547e-5, dg gate FAIL — the B-only soft branch);
  C6 stays clean (4.235e-12 / 1.212e-6); **S6 passes both gates**
  (dE 4.235e-12 / dg 1.212e-6) with strict_match_verified 4/4 — the
  matcher reports **same-solution, selected=main** (the FU1 melatonin
  shape): on this cell plain auto is itself measured-clean and the
  strict matcher certifies it rather than substituting.
- **wb97mv:** B6 (dE 1.691e-8 / dg 9.498e-4) and C6 (dE 1.829e-6 /
  dg 1.392e-2) reproduce MR-17 **bitwise** and fail their gates; **S6
  adopts the reference BITWISE** (dE 0.0, dg 5.931e-7;
  distinct_solutions true, selected=reference — exactly the FU2/FU3
  mechanism).
- **R bitwise both cells** across fresh processes (grad spread
  <= 1.6e-14). Zero fallbacks, zero gate skips.
- **Cost:** S6/C6 1.79x (m062x) / 2.64x (wb97mv); R/C6 0.91x / 2.08x;
  B6/C6 0.66x / 0.98x. Cycle counts identical within every cell-lane.

**Operative guidance table for the methoxy qzvp soft-mode family is
now complete** — every measured failure in the family (b3lyp stratum 1,
m062x, wb97mv; the methoxy b3lyp/wb97mv svp/tzvp history) has a
cost-certified safe route: **auto-strict** (all three methods) or the
**strict R lane**; on m062x plain auto is also measured-clean; plain
fp64-under-stack fails on b3lyp+wb97mv and is the inverted-class B-only
failure on m062x. Gates/defaults unchanged; every failure above stays
in its denominator. Evidence:
`2026-10-25-fu4-methoxy-new-soft-safeway-cost/` (78-file hash
manifest); effects-index 168 (full sweep 2047 OK / 0 missing / exactly
the 4 documented mismatches, zero new). One driver tooling erratum:
the progress filename inherited the fu3- prefix from the template and
was renamed post-hoc (zero data impact, documented).
## MR-18: lane-coverage stratum 3 — anion-x-qzvp COMPLETED, the 924-AO capacity screen ADMITTED, qzvp ceiling 522→924 AO (2026-10-25)

Preregistration sha 8d304ef3 registered 2026-10-25T05:39:29Z BEFORE
measurement (validator PASS after one closed-shell-neutral substring
false-positive reword, the MR-17 documented class, zero semantic change;
CPU screens 7/7 converged). Driver 05:45:20Z–07:52:56Z at HEAD f2d5900
(7299-file frozen snapshot self-built by the driver): 12 def2-qzvp
cells x O/B/C/R x 4 balanced blocks = **192 fresh-subprocess arms,
192/192 rc=0, zero retries, zero partition-fix firings (16th
consecutive), wall 2.13 h** (0.41x the 5.16 h prediction — the fastest
stratum yet). Composition: **anion-x-qzvp completion** (superoxide O2−
b3lyp/m062x q-1s1 + methoxide CH3O− m062x/wb97mv q-1s0), the
**>522-AO qzvp capacity screen** (vitamin_c 924 AO x 3 methods, the
only feasible cohort molecule above benzene 522; inosine/bisphenol
~1440 AO are the infeasible class), benzene-522 method completion
(m062x/wb97mv), and doublet-family breadth (NO2 b3lyp, ethoxy
m062x/wb97mv q0s1). 7 RKS + 5 UKS; charges −1 x4, 0 x8; nao 114–924.
The checkpoint-named r2scan3c-qzvp lane is STRUCTURALLY REFUSED by the
production prescribed-basis guard (r2SCAN-3c must use def2-mtzvpp) —
documented in `generate_mr18_schedule.py`, never silently relabeled.
No frozen references exist on these lanes: comparator = within-batch
strict-R.

**Verdicts: 9/10 clauses HIT, decision rule MET** (P1 ∧ P2 ∧ P4 ∧ P10).
Independent recompute AGREEs on every value, zero disagreements.

- **P2 primary gate HIT — zero gate failures batch-wide**, including
  the two preregistered RISK classes that did NOT recur: (a) the
  superoxide qzvp soft-anion auto class stays inside the gates (b3lyp
  C dE 4.18e-8 / dg 6.06e-7; m062x C 8.98e-7 / 5.22e-6 — the largest
  anion dE in the series, 10% under the 1e-6 gate, honest record); (b)
  `ethoxy_wb97mv_def2-qzvp` — the RSH+VV10 methoxy-sibling radical
  crossing qzvp NLC — is CLEAN (B 1.34e-11 / 1.12e-6, C 2.79e-12 /
  1.59e-6): the methoxy soft-mode class remains molecule-specific.
- **P3 B-floor honest MISS on the two preregistered anion
  trajectory-shift rows** (superoxide b3lyp 4.27e-8, superoxide m062x
  8.80e-7; dg floors 3.5e-7 / 5.7e-6 all inside the science gates) —
  the same informative class as strata 1–2 (3.68e-8 / 2.89e-8), never
  relabeled.
- **P4 R-determinism HIT: 12/12 cells R-bitwise across fresh
  processes** — the MR-12 class now extends to anion doublets and the
  924-AO lanes (zero exceptions, zero envelope cells).
- **Capacity screen: vitamin_c 924 AO ADMITTED at all three methods**
  (peaks: O 14 622 MiB / B 10 422 / C 10 486 / R 10 350 against the
  21 608 MiB admission budget; 14 cycles everywhere, auto-b3lyp 16).
  B/C gates pass with margin (dg 7.4–9.8e-6 = 25–33% of the 3e-5
  gate); O corroborates R on the 924-AO lanes (O dE <= 1.2e-10, worst
  O dg 1.6e-8). **The measured qzvp ceiling moves 522 → 924 AO.**
- **Methoxide q-1s0 at m062x/wb97mv: the FIRST closed-shell NLC anion
  at qzvp, bitwise-clean** (m062x B/C dE 8.5e-14; wb97mv B 6.8e-13 /
  3.4e-7, C 2.4e-12 / 1.14e-6). The b3lyp methoxide soft-mode failure
  (MR-16, frozen) does NOT generalize across methods on the anion
  side. These are DIFFERENT cells from MR-17's failing neutral
  methoxy radicals (`_q-1s0` anion vs no-suffix q0s1 radical — the
  shared name prefix is the only overlap).
- **P5 O-role first measurements on all 12 lanes (informative):**
  worst O dE 9.26e-9 (superoxide m062x, the upstream racy-kernel
  lane), worst O dg 5.65e-8; O within 2e-13 dE of R on 7 cells.
- **Cost (median e2e ratios):** O/B 1.66–3.43; B/C 0.87–1.03;
  O/C 1.61–3.37. **C(auto) is faster than B on 9/12 cells**; the three
  above-parity cells are methoxy_wb97mv_q-1s0 (1.023, 321 AO) and
  vitamin_c m062x/wb97mv (1.027 / 1.015, 924 AO) — the NLC/large-AO
  family, which is precisely the SCF/verification cost-profiling
  target of the 2026-10-25 priority card. Strict-route cost on the
  same lanes: R/B 1.31–3.30 (R-scf/B-scf 1.13–2.78, largest on the
  VV10 cells).
- **Counting correction applied per the 2026-10-25 card:** failure
  rows are counted per comparison row, not per combination — MR-17's
  three B/C failure rows involve two combinations; this batch has
  zero failure rows.

Gates/defaults unchanged; every MISS stays in its denominator; no
historical row rewritten. Evidence: `2026-10-25-mr18-lane-coverage/`
(hash manifest); effects-index 169. One exit-time tooling erratum: the
manifest writer's pre-authored FILES list carried a `cpu-screens-mr18.log`
entry that was never created (the CPU screens wrote JSON only) — caught
at exit BEFORE the first manifest write, fixed in place, zero data
impact. This stratum CLOSES the lane-coverage series per the
2026-10-25 priority card: no MR-19; the mainline pivots to SCF +
strict-verification cost profiling and source optimization.


## SP-1: the SCF + strict-verification cost profile — the 2026-10-25 priority mainline opens (2026-10-25)

Per the scf-priority-20261025 card (acknowledged 07:56:18Z, sha a4aa2d47):
the lane-coverage series CLOSED at MR-18; the mainline pivots to SCF cost
profiling and source optimization. Two steps, both preregistered/measured:

**Step 1 (CPU, existing raw timings, 40 cells x 6 batches).** C(auto)
already beats B(optimized-FP64) on 9/12 MR-18 cells; the 3 above-parity
cells are the NLC family (methoxy_wb97mv q-1s0 1.023, vitamin_c
m062x/wb97mv 1.027/1.015); historical B/C medians by method: b3lyp 0.938,
m062x 0.949, r2scan3c 0.988, wb97mv 0.980 — and small cells lose (oxygen
svp 1.101, oxygen 3c 1.094, pyrrole 1.064). Auto cycle deltas are +1..+2
typical (+8 only the methoxy_m062x soft class). Strict cost: R/B 1.43-2.67
by method with R-scf/B-scf 1.21-2.55 (worst wb97mv); O/B 1.81-2.76. The
FU2-FU4 S-lane decomposition: matcher+residual only 0.16-0.50 s — the
sibling strict SCF IS the strict-route cost.

**Step 2 (SP-1 GPU profile, prereg 2ddb2b7b 08:27:21Z before measurement;
6 cells x R/B/C x 3 samples + warmup, production gradient_runner path,
observation-only phase wraps; 18/18 rc=0, 15.1 min wall).** Decision rule
MET, profile VALID (all e2e anchors reproduce: vitamin_c R 71.79 vs 71.3
frozen, B 25.38/25.5, C 25.33/25.1; superoxide cycles 25/21 exact; ethoxy
14/14/15 == frozen MR-18). Wrapper coverage 71.7-98.5% of t_scf.

- **L1 — THE strict cost is fp64 VV10:** the strict-R lane's nr_nlc_vxc
  runs 4.1-5.0x slower than the stack's fp32-UWE VV10 (benzene qzvp
  7.410 vs 1.484 s; ethoxy 6.082 vs 1.427; methoxy 2.432 vs 0.592;
  vitamin_c 31.659 vs 6.614) = 51-64% of R t_scf. R-scf/B-scf 2.18-2.80
  on wb97mv is this wall plus RSH jk. The only route to a cheaper strict
  anchor is a provably-equivalent faster fp64 VV10 (algorithmic).
- **L2 — why C loses to B on the NLC family:** B and C share the whole
  stack (LC-115 fp32 vmat-head contractions, fp32-UWE VV10,
  GRAD_GRID_F32); C's ONLY difference is the auto fp32 head
  (eval_rho/xc_grid fp32 GEMMs, numint.py:283/414/774). Measured: the
  head HELPS plain b3lyp (-5% t_scf) and r2scan3c (-12%) but HURTS every
  qzvp VV10 cell (t_xc +5..+11%, t_nlc +6..15%) — exactly the 3 MR-18
  above-parity cells. Minimal-change candidate: shape/method-aware
  fp32-grid gating in the auto head (never molecule-name picking),
  equivalence-tested, paired before/after.
- **L3 — cycles are not the cost:** C cycles equal or better than B
  everywhere measured (worst +1); the +8 methoxy_m062x class did not
  appear on its wb97mv sibling.
- **L4 — strict non-VV10 overhead is the fp64 gradient** (benzene R
  t_grad 2.66 vs B 0.51 = 5.2x): structural for the strict anchor.

One adjudication erratum: the P3 ethoxy prediction band (~9 cycles) was
authored without checking the frozen table — measured 14/14/15 MATCHES
MR-18 exactly; 5/6 cells inside ±1. Two zero-GPU-measurement launch
failures (wrong interpreter; missing ns_args attributes) fixed before any
data, documented. Gates/defaults unchanged; measurement-only unit.
Evidence: `2026-10-25-scf-verification-cost-profiling/`; effects-index
next dataset.

### SP-2 attempt-1: the L2 minimal-change hypothesis FALSIFIED and REVERTED (2026-10-25)

The first source-change candidate from the SP-1 profile — gate the auto
policy's fp32-head SCF-grid contractions behind GPU4PYSCF_AUTO_GRID_F32
(default on = bitwise-unchanged, four numint sites) — was implemented and
measured on benzene/wb97m-v/qzvp (3 samples, production path): **C′ e2e
6.80 s vs C 5.68 and B 5.45 — a +20% REGRESSION (t_xc 2.909 vs 1.745)**
(evidence: `sp2-attempt1-C2-benzene-wb97mv-REVERTED-KNOB.json`). The
mechanism reading was wrong: the auto-head fp32 grids are NET-POSITIVE
even on the VV10 class — with them off, the head runs the fp64 GEMM plus
a double pass in _nr_*_task (fp64 rho pass, then the LC-115 lane flag
still forces a per-block fp32 astype for the vmat contraction). The
C-vs-B +5..11% t_xc gap is therefore NOT the numint fp32 branches; the
residual candidates are the per-block astype-vs-GEMM balance at
tall-skinny qzvp shapes, the head's df_jk fp32 entries, and mode-switch
bookkeeping — any second attempt must first measure the astype/GEMM
crossover by shape. **The source change was REVERTED (tree back at the
eb037f3 state); the knob never enters production; the negative result
stands as evidence.** The ranked mainline lever remains SP-1 L1: the
strict lane's fp64 VV10 at 4.1–5.0x (51–64% of strict t_scf) — an
algorithmically-equivalent faster fp64 VV10 with a bitwise/1e-12
equivalence proof is the next candidate unit.

### SP-3: the L1 lever FALSIFIED with precision — the fp64 VV10 kernel is at the sm_89 arithmetic floor (2026-10-25)

The ranked L1 candidate (algorithmically-equivalent faster fp64 VV10,
preregistration `24844c15` registered 09:14:12Z) was tested kernel-level
on production-constructed NLC-grid inputs (mol+mf via the production
`_gpu_mol_mf`, dm0 = minao init guess, inputs captured through one real
`mf.get_veff` call; bench body validated BITWISE against
`numint._vv10nlc` at variant 0, max Δexc = Δvxc = 0.0 on all cells).
**Both preregistered routes are dead by measurement; no source change
was emitted.**

* **Tiles give no parallelism win at production sizes**: tile8/16/32 vs
  the warp1 reference = 1.15–1.22x at 57,856 points (methoxy, occupancy
  limited) but **1.00–1.04x at 143,616 (benzene) and 247,808
  (vitamin_c)** — the fp64 pipes are saturated. Equivalence of the tile
  family confirmed: |Δe_NLC| ≤ 2.8e-17 / 6.1e-16 / 2.4e-15 (≥ 41x under
  the 1e-13 band; U max-rel 2.4–6.4e-14 grows ~√ngrids as expected for
  summation-order rounding).
* **warp1 is at the arithmetic floor**: after calibrating the true fp64
  FMA rate (645 G-FMA/s), warp1 measures **20.6 FMA-slots/pair at BOTH
  benzene and vitamin_c** — i.e. the ~20 FMA-class ops of the minimal
  pair expression plus a ~5-slot hardware divide. A standalone probe
  (`div_probe`) measured IEEE fp64 divide ≈ **5.0 FMA-slots** and an
  fp32-seeded 3-step Newton reciprocal ≈ **13.4** — the software
  reciprocal is SLOWER than the divide it would replace, and there are
  no other ops to remove. The op-reduction route died pre-implementation.
* **Corollaries**: the strict-lane fp64 VV10 cost is hardware-structural
  (the 1/64 fp64 rate times a minimal expression); the fp32safe kernel
  runs 8–16x faster kernel-level (≈3x above its own floor, issue/LDS
  bound) which dilutes to the known 4–5x at the `t_nlc` level; and R
  e2e at large AO is dominated by the fp64 gradient anyway (vitamin_c R
  e2e 289.5 s vs SCF phases 52.8 s), consistent with SP-1 L4. Only a
  genuinely different algorithm (pair screening / multipole with a
  rigorous ≤1e-12 bound proof) could move the strict VV10 — a separate
  large unit, explicitly NOT the minimal-change class the card calls
  for.

Adjudication: P0 HIT, P2 HIT-band/threshold-miss (both preregistered
routes falsified), P3 partial (Δe_NLC decisive HIT; the 1e-14 rel
sub-band was an authoring error — did not scale with accumulation
length), P1 MISS-high (kernel fp32 ratio 8–16x vs the [3,6] band
transcribed from e2e ratios — the fixed-part dilution is now measured),
P4 HIT (the floor reading is the deliverable), P5 HIT. Three tooling
errata recorded (launch-verify variant-list/bytes-crash truncated JSON;
an override=2 leak contaminating only the prod-capture diagnostic on
2 cells — inputs and all recorded timings unaffected; the probe's
slots normalization corrected ×2.13). Evidence:
`2026-10-25-sp3-vv10-fp64-kernel/` (7-file hash manifest). **The
measurable C-vs-B surface per the card is now exactly the SP-2 residual
list; next unit SP-4 = the astype-vs-GEMM shape-crossover
micro-measurement that the SP-2 disposition requires before any second
auto-head source change.** Gates/defaults unchanged; measurement-only;
no historical row relabeled.

### SP-4: the auto-head residual localized — DIFFUSE, no minimal-change target; the item-3 program closes (2026-10-25)

The SP-2 disposition required an astype-vs-GEMM shape-crossover
measurement before any second auto-head source change. Preregistration
`03505410` (09:33:54Z); Part A = one-cycle production `get_veff` A/B
(arm-C-head under `precision.fp32()` vs arm-B-head fp64, both with the
full STACK_ENV and `_scf_vmat_f32_lane=True` replicating the LC-115
head), Part B = standalone block-level density-side cost table.
**Verdict: DIFFUSE — the preregistered no-further-source-change branch
fires.**

* The head penalty is real but small and localizes INSIDE `nr_rks`:
  t_xc fp32/fp64 = **1.082 (benzene), 1.018 (vitamin_c), 1.426
  (methoxy)** — while the density side is equal-or-faster (t_rho
  0.989/0.985/**0.529**) and the head's df_jk fp32 entries are a **1.6x
  WIN** (t_jk 0.612/0.611), eliminating that SP-2 residual candidate as
  a loss source. The residual excess is per-block overhead class
  (AO-eval/alloc/cast bookkeeping), diffuse, and the absolute stakes are
  ~0.1–1% of e2e on the affected cells (the whole density side is ~20
  ms/cycle at benzene).
* Shape table: at the production blocksize (G=16384) the fp32
  density-side block beats fp64 at every shape (ratios
  0.765/0.284/0.829/0.287 for nao 114/216/522/924) — the cast alone is
  up to 61% of the fp32 block cost at (522, 16384) — while at G=128
  fp32 LOSES (1.29–1.63): a real crossover exists only BELOW production
  block sizes. The auto fp32 head is NET-POSITIVE at the mechanism
  level on every production cell; SP-2 attempt-1's e2e conclusion is
  now explained piece-by-piece.

Clauses: P1 PARTIAL (methoxy 1.426 out of [0.90, 1.15]; the two large
cells in band), P2 PARTIAL (vitamin_c t_nlc 0.916 = arm-order variance
— nr_nlc_vxc has no precision branch; recorded as a noise caveat), P3
MISS (the universal ≤ 0.5 prediction does not hold — small-G losses and
a 0.829 at (522, 16384)), P4 HIT (the localization is the deliverable),
P5 HIT. Two interpretation errata documented (t_rho nests inside
t_xc/t_nlc; the vitc t_nlc variance class).

**The card item-3 minimal-change program is now CLOSED with three
measured falsifications/decompositions: SP-2 (direct L2 gating, +20%
e2e regression, reverted), SP-3 (the L1 equivalent-kernel class, dead
at the sm_89 arithmetic floor), SP-4 (the L2 residual list, diffuse
with no removable ≥50% piece). The C(auto)-vs-B gap on the 3 qzvp VV10
cells (+0.3–5.6% e2e) stands as measured, with a complete mechanism
decomposition.** The mainline continues with the three-comparison
continuous reporting (card item 5, existing reporter) and the only
remaining honest L1 route — an algorithmic strict-VV10 study
(pair-screening/multipole with a rigorous ≤1e-12 bound proof) — stays a
separately-scoped feasibility analysis, not a minimal change, and is
not started without fresh card-visible justification (SP-3 also showed
R e2e at large AO is gradient-dominated, bounding any VV10-only win).
Evidence: `2026-10-25-sp4-autohead-shape-crossover/` (4-file hash
manifest). Gates/defaults unchanged; measurement-only; no historical
row relabeled.

### SP-5: three-comparison reporting integrated — all 19 four-role batches registered; the daily report now renders MR-18 (2026-10-25)

Card item 5 (`三种对照持续分报`) + item 6 registration duty. A
CPU-only reporting/integration unit over already-preregistered,
already-adjudicated measurements — no new measurement, hence no
preregistration surface; no row relabeled.

* **Registration gap closed**: the 19 completed four-role O/B/C/R
  batches measured since 2026-10-21 (mr1-backfill; mr7-b1…b6; mr13
  b1–b9; mr16/17/18 — 462 cells, 1,848 role documents) existed only as
  per-batch `upstream-index-*.json` files; the main
  `upstream-effects-index.json` — the ONLY file the deployed operator
  daily report reads — still ended at `uc-20261018-a`. All 462 cells
  were first validated through the DEPLOYED reporter's
  `three_way_report.load_cell` (provenance gate + per-file sha256):
  **462/462 PASS, zero failures**; the batches were then appended
  chronologically (first `meta.timestamp_utc`) after the original
  batch, atomic replace, pre-integration backup kept. Every batch
  keeps its own id — each ran at a distinct frozen snapshot
  (44cb789 → 3001b34 → 38f4919 → 7cac259 → f52735a → 2f2ad84 →
  faf8171 → 4c14f76 → 80f7d23 → 8879d0e → 21cd3dc → b85c045 →
  21706ae → 1920bd4 → 8ed0595 → f2d5900), per the card's
  source-change rule.
* **Latest batch (what the daily report now renders)** —
  `mr18-lanecoverage-20261025-a` (snapshot f2d5900, 2026-10-25), all
  gate-passing fixed subsets:

  | 对照 | pass/planned | e2e 中位数 | SCF 中位数 | 梯度中位数 | 固定通过子集总耗时比 |
  |---|---:|---:|---:|---:|---:|
  | O→B 原版→优化基准 | 12/12 | 1.918× | 1.626× | 3.774× | 2.380× (177.52s→74.60s) |
  | B→C 优化基准→auto | 12/12 | 0.952× | 0.934× | 0.992× | 0.985× (74.60s→75.70s) |
  | O→C 原版→auto | 12/12 | 1.849× | 1.479× | 3.753× | 2.345× (177.52s→75.70s) |

  Precision anchors: B/R and C/R both 12/12 (max |ΔE| 8.80e-07 /
  8.98e-07 Ha — the preregistered superoxide anion trajectory-shift
  class, under gate; max |Δg| 8.41e-06 / 9.77e-06); O/R
  implementation consistency 12/12 (max |ΔE| 9.26e-09); missing 0/12.
  **B→C < 1 on this stratum is the honest open item-3 status**: the 12
  qzvp-heavy cells carry the 3 known above-parity NLC cells; auto is
  not yet faster than the optimized fp64 baseline at unchanged gates,
  exactly as SP-1…SP-4 decomposed.
* **20-batch trend (the 持续分报 view, `sp5-trend-table.json`)**: O→B
  e2e medians **1.68–2.01×** with fixed-subset totals 2.38–4.63× across
  EVERY source snapshot — the optimized-baseline gain over official
  1.8.1 is stable across four days of source evolution; B→C sits in the
  **0.91–1.04× parity band** batch-wide (auto-favored cells exist in
  specific strata but never carry a batch median); O→C totals
  2.23–4.61×. The wheel-vs-source-build caveat stands: O→B is a whole
  product comparison, not a single-kernel attribution.
* **Item-6 coverage**: MR-13 b1–b9 / 16 / 17 / 18 datasets already pin
  schedule + final-adjudication + independent-recompute (+ per-cell
  four-role docs for 17/18); the per-batch index files pin ALL four-role
  cells transitively and are now hash-anchored in the effects-index
  (dataset `sp5-threecomparison-integration-20261025`, 26 sources).
  Newly pinned: `mr7-b1-schedule.json`. Honest absences recorded (never
  fabricated): MR-7 b1 and MR-1-backfill predate the
  independent-recompute convention.
* **Reporter deployment-copy maintenance (red→green)**: the operator
  updated the deployed `/opt/qc-agent/effects_report.py` at 07:39Z
  (adds exactly the item-6 guards — schedule-only and mechanism/audit
  datasets now render an explanatory note instead of empty legacy
  performance tables). The drift test was observed RED on the stale
  copy; `tools/reporting/` copies + the `_DEPLOYED_SHA` pin +
  PROVENANCE.md were refreshed together; **25 passed GREEN** after.
  `three_way_report.py` unchanged since 2026-10-18.

Evidence: `2026-10-25-sp5-threecomparison-integration/` (6-file hash
manifest: README, integrated-index snapshot + pre-integration backup,
20-batch trend table, deployed-renderer latest-batch output).
Effects-index dataset 173; full sweep **2184 OK / 0 missing / 5
mismatches = exactly the 4 documented + the 1 documented-explainable
sp1_arm.py — zero new**. Gates/defaults unchanged; no historical row
relabeled; the living index is pinned only via an immutable snapshot
copy (future batch appends must not break dataset hashes).

### SP-6: the strict safe route (auto-strict) per-phase cost decomposition — sibling SCF is the structural floor; the strict route cannot beat B; SP-7 candidate named (2026-10-25)

Card item 2's decomposition list names 严格 sibling SCF、匹配与 fallback —
the one split SP-1 (R/B/C only) never measured. SP-6 (prereg
`b946154f` registered 10:38:09Z) runs the SAME six SP-1 representative
cells on the `auto-strict` safe route under the full STACK_ENV through
the production `gradient_runner` path, warmup + 3 samples per arm,
observation-only wraps on the SP-1 phase timers PLUS the
`stability_verify` internals (sibling build, sibling `kernel`,
matcher `_orbital_grad_norm`/`_occ_overlap_min`/`_max_abs_dm_diff`,
`verify_reference_match` total; MR-8 reuse counters). Driver wall
9.4 min, 6/6 arms rc=0, `strict_match_verified` 18/18, GPU serial.
SP-1 anchors pinned (sp1-profile sha `a1741d10`).

* **The sibling IS the R lane**: `t_sib_scf` = 0.917–0.987× the R-anchor
  `t_scf` on all six cells, sibling cycle counts EXACTLY the R-anchor
  cycles (7/9/14/12/22/14); CDERI reuse active on every verify
  (18/18 shared=1/built=0), sibling fixed cost ≤ 0.0007 s. The
  additive model holds almost exactly: S_e2e − C_anchor ≈ verify_total
  within ±0.4 s everywhere (e.g. vitamin_c 56.04 vs 56.41 s).
* **Structural floor (P5 HIT)**: the sibling SCF carries **89.9–96.5%
  of the verification overhead** on every cell. S/B **time** ratios
  1.833 / 3.332 / 3.100 / 2.721 / 2.032 / 3.206 — the strict safe
  route including ALL verification cost is measured SLOWER than the
  optimized FP64 baseline on every class. **Card item 3's second half
  (量出 strict 安全路线含全部验证成本能否更快) is answered: no on this
  hardware/source** — the cost is the contractually-required
  same-start fp64-lane trajectory itself (card item 4 immutable), at
  the SP-3-measured sm_89 arithmetic floor on the VV10 cells.
* **Matcher diagnostics (P3 PARTIAL)**: the two `_orbital_grad_norm`
  full-`get_veff` re-evaluations + occ-SVD + dm-diff carry
  4.2–10.6% of S e2e; within that, the gnorm re-evals are ≥ 94% (the
  decision itself uses the dm diff alone — 0.00026 s max). Honest
  MISS-high on benzene_b3lyp (10.6% vs its [2,8] band; 0.44 s on a
  4.13 s cell).
* **Fail-closed contract exercised twice (P6 PARTIAL)**: methoxy
  (predicted, the FU4 soft class) AND superoxide_r2scan3c (unpredicted
  distinct: dm 0.098, the known anion trajectory-shift class, auto
  dE −2.2e-6 vs R) both adopt the reference **bitwise** (d=0.0);
  the four same-solution cells keep main with S e_tot **bitwise ==
  the C anchor** (d=0.0 exactly, 4/4).
* **SP-7 candidate NAMED per the preregistered decision rule** (gnorm
  diagnostics ≥ 5% of S e2e on 2 cells): same-solution fast-path skip
  of the audit-only matcher diagnostics — record-only fields, never
  decision inputs; they remain computed on the distinct path where
  `displaced_auto_point` needs them. Sibling, decision path, gates and
  defaults untouched; red→green equivalence + ≥ 4 independent
  process-block paired before/after required before any adoption.
  Even fully removed, S stays 1.8–3.3× above B — this is a
  verification-cost reduction, not a route-to-B fix.

Clauses: P0/P1/P2/P4/P5 HIT, P3/P6 PARTIAL (bands/deviations recorded
above, never relabeled). Four tooling errata recorded in the
adjudication (system-python first launch = zero-measurement rc=1 x6;
driver.pid written to repo cwd then relocated; driver decomposition
key stored B/S under an S/B name; warmup verify records not
persisted — 18/18 sample verifies are the persisted evidence).

Evidence: `2026-10-25-sp6-strict-route-cost-decomposition/` (20-file
hash manifest: preregistration, adjudication, profile, driver +
log, 6 arm JSONs + stdouts). Effects-index dataset 174
`sp6-strictroute-cost-20261025` (5 sources); full sweep **2189 OK /
0 missing / 5 mismatches = exactly the documented set, zero new**.
Gates/defaults unchanged; measurement-only; no historical row
relabeled; LC-215 untouched.

### SP-7 ADOPTED: auto-strict matcher fast path — the audit-only gnorm re-evaluations skip the same-solution branch (2026-10-25)

The candidate SP-6's preregistered decision rule named (gnorm
diagnostics ≥ 5% of S e2e on 2 cells). Prereg `b4b3d285` registered
11:17:36Z; before-blocks at the clean `fdb08ce` tree 11:18–11:25Z;
red→green; source change `46db317`; after-blocks at the clean `46db317`
tree 11:40–11:49Z; GPU wall 15.5 min total.

* **Change (opt-in auto-strict route only)**: in
  `verify_reference_match` the two `_orbital_grad_norm` re-evaluations
  (each a full `get_veff` on a converged density; 94–99.7% of matcher
  cost) now run ONLY on the distinct path, where
  `displaced_auto_point` needs the displaced point's residual. The
  same-solution record carries `orbital_grad_norm_main/ref = None` +
  `matcher_diagnostics = 'skipped_same_solution_fast_path …'` —
  explicit, never silently absent; occ-overlap SVD (9–12 ms) and the
  dm diff still run on every path; the nonfinite-dm fail-closed branch
  marks the same skip. `verify_solution_stability` (auto+verify)
  untouched; no knob; gates/defaults/sibling untouched; library
  default `precision_mode=None` users never enter this code.
* **Red→green**: `test_strict_same_solution_skips_audit_gnorm` FAILED
  at the pristine `fdb08ce` source (via git-stash:
  `gnorm.call_count 2 != 0`) → GREEN with the change;
  `test_strict_distinct_keeps_audit_gnorm` green both sides (the
  distinct-path invariant guard). Full stability + CDERI-reuse suites
  60 passed + 9 subtests; `tools/tests` OK under the env python.
* **Paired ≥4-block measurement (3 cells, production path, STACK_ENV)**:
  e_tot BITWISE identical before/after on every cell and dm diff
  identical to the last bit; `strict_match_verified` True 60/60
  samples. S e2e **−11.9%** on benzene_b3lyp (4.241→3.737 s) and
  **−5.2%** on benzene_wb97mv (18.154→17.218 s); matcher share of S
  e2e 10.2%/6.2% → **0.5%/0.05%**; ZERO gnorm invocations on the
  same-solution blocks (timer-key absence). Methoxy distinct control
  unchanged: gnorm kept (0.396–0.414 s), e2e +1.3% (noise), e_tot
  bitwise == the SP-6 R anchor.
* **Context (unchanged verdict)**: the S route remains 1.8–3.3× above
  B — the SP-6 structural floor is the sibling SCF. SP-7 shaves the
  only measured reducible piece: an honest verification-cost reduction
  of the safe route, not a route-to-B fix.

Evidence: `2026-10-25-sp7-matcher-fastpath/` (50-file hash manifest:
preregistration, adjudication, both drivers + logs, 9 before + 12
after arm JSONs + stdouts). Effects-index dataset 175
`sp7-matcher-fastpath-20261025` (7 sources); full sweep **2196 OK /
0 missing / 5 mismatches = exactly the documented set, zero new**.
Gates/defaults unchanged; no historical row relabeled; LC-215
untouched. Two tooling notes in the adjudication (sp6_arm's fixed
verify-record key list predates the new fields — evidence via unit
tests + timer-key absence; both detached drivers' pid-file writes
landed in the repo cwd, removed immediately).

### SP-8: the C-vs-B cycle surface measured and CLOSED — the excess is an intrinsic soft-mode endgame stall, invariant to switch policy (2026-10-25)

The last unattributed piece of the C(auto)-vs-B(optimized-fp64-SCF)
e2e residual (item 3's open goal): SP-4 had dispositioned the diffuse
per-cycle excess (+3.4% median); the concentrated cycle excess
(+1..+8 cycles on 14/36 recent cells, worst methoxy_m062x qzvp +8 =
1.424× e2e) had never been decomposed. Preregistration `f5894eb9`
registered 12:20:53Z before all GPU measurement (Part A — CPU-only
re-analysis of frozen mr16/17/18+fu2/3/4 — declared complete in the
prereg itself; evidence `sp8-partA-cycle-decomposition.json`).

**Part A (existing evidence, 36 auto rows)**: the auto switch fires at
cycle 5–9 (median 6.5), **always** by the ΔE trigger (zero
chaos-guard firings anywhere in the recent batches); the fp64 tail is
3–15 cycles (median 4); the head always ends before B finishes. The
cycle excess is entirely **tail length**.

**Part B (per-cycle GPU traces, production path, observation-only
`PrecisionProbe.callback` patch; 5 cells × B/C, warmup + 1 sample)**:
P1 controls reproduce the frozen cycle counts EXACTLY (benzene_m062x
8/9, NO2_wb97mv 11/13, methoxy_m062x 12/20, benzene_wb97mv 9/9,
superoxide-3c 25/21). The anatomy, on methoxy_m062x: C's head
trajectory is IDENTICAL to B's through cycle 6 (|g| matches to 3
digits — the fp32 head costs nothing in convergence); after the
switch C's |g| **freezes at 5.2e-5 for cycles 10–13** (per-cycle
factor 1.0 while |ddm| keeps shrinking), then DIIS thrashes (positive
ΔE, |ddm| bouncing at 1e-3) cycles 14–18, and crosses 1e-5 only at
c19 — 8 cycles in band total. B transits the same band geometrically
(~3.5×/cycle, 4 cycles). **B itself stalls and thrashes at 2.5e-5 on
superoxide-3c** (25 cycles, |g| frozen c8–13): the stall is intrinsic
soft-mode DIIS behavior at the frozen conv criteria
(`conv_tol` 1e-10, |g| < 1e-5), amplified by the fp32-head map
inconsistency in C. P2 PARTIAL (the C-side stagnation clause HIT
decisively; the authored B-side "≤3 cycles" comparator measured 4 =
band-authoring error, SP-1-P3 precedent; the 8-vs-4 factor-1.0-freeze
vs geometric-transit contrast stands).

**The switch point is not a lever (P3, strengthened to invariance)**:
tolerance sweep 3e-5 / 3e-4 / 1e-3 on the worst cell and an NLC
control — switch cycle moves (8/7/6) but **total cycles IDENTICAL at
every tolerance on both cells** (20/20/20/20 and 9/9/9/9); no head
stall at 3e-5 (the fp32 noise floor does not block the trigger);
e_tot stable to 1.4e-11 (P4 HIT « 5e-10 band). The convergence takes
what it takes from wherever the switch lands.

**The falsified DIIS-intervention family, retested on new trace
evidence — negative decisively**: the stall on methoxy breaks exactly
when the last fp32-era entry leaves the 8-slot DIIS space (c14),
which warranted one bounded post-prereg exploratory probe (labeled as
such; confined to arm processes via the callback; repo untouched):
purging the fp32-era `_bookkeep` entries at the switch costs **+4..+6
cycles on ALL 5 cells** (methoxy 20→26, NO2 13→17, both benzenes
9→15, superoxide 21→26) with e_tot unchanged (≤4e-11). The stale
fp32-era DIIS entries are actively useful — consistent with LC-114
and the C2 reset falsification, now measured on the qzvp/3c classes
C2 never tested. The stall is NOT subspace pollution.

**P5 (directional)**: C head cycles are NOT cheaper than B head
cycles (median ratios 0.90–1.29 across the 5 cells) — B already
harvests every safe fp32 piece via STACK_ENV (fp32-UWE VV10, DF-IP1
fp32, fp32-vmat head/tail, fp32 grad grid), so the auto head's extra
fp32 (grid contractions) buys ~0/cycle at these shapes and pays the
endgame stall — the mechanism-level counterpart of SP-4's diffuse
finding.

**DECISION (the preregistered close branch)**: the cycle surface
CLOSES. P2 stagnation confirmed + no tolerance variant reduces cycles
+ the DIIS-at-switch family falsified twice (C2 svp/tzvp reset; SP-8
qzvp/3c purge). The C-vs-B residual is now **fully attributed with
every minimal-change lever measured falsified**: SP-2 (fp32-grid
gating: +20% e2e regression, reverted), SP-4 (per-cycle: diffuse),
SP-8 (cycles: policy-invariant soft-mode endgame stall). No source
change; gates/defaults untouched; C stays honestly at the
measured parity band (9/12 mr18 cells faster, aggregate ≈ +1.5%,
excess concentrated on soft/anion classes); no historical row
relabeled. The only remaining route to a faster C remains the
separately-scoped algorithmic VV10 program, not started without
fresh card-visible justification.

Evidence: `2026-10-25-sp8-cycle-decomposition/` (54-file hash
manifest: prereg, Part A table, both drivers + logs, 21 arm JSONs +
stdouts, adjudication). Effects-index dataset 176
`sp8-cycle-decomposition-20261025` (5 sources); full sweep **2201 OK /
0 missing / 5 mismatches = exactly the documented set, zero new**.
GPU: two serial detached drivers (3.2 + 1.1 min), idle at exit.

### SP-9 — the SP-8-named remaining route measured and closed: VV10 distance-screening is INFEASIBLE at production shapes (F-INFEASIBLE, CPU-only, 2026-10-25)

**Unit**: the SP-8 close named "the only remaining route to a faster C =
the separately-scoped algorithmic strict-VV10 program (screening/multipole
+ rigorous <=1e-12 proof), not started without fresh card-visible
justification". SP-9 measured the **screening half** of that route for
feasibility BEFORE any program start — a bounded, preregistered
(prereg `b8183731`, 13:00:50Z, before measurement), CPU-only unit on four
cells: benzene/wb97mv/qzvp (522 AO, compact production), vitamin_c/wb97mv/qzvp
(924 AO, largest admitted), methoxy/wb97mv/qzvp q-1s0 (the soft-mode strict
class), tamoxifen/wb97mv/svp (537 AO, the most EXTENDED in-history molecule —
the best case screening could ever fire).

**Method**: numpy mirror of the exact production kernel expressions
(`lib/gdft/vv10.cu` omega-derivative + UWE pair kernels; Phi = -1/(g_ij g_ji
(g_ij+g_ji)), g = omega r^2 + kappa) on the production grid pipeline (full
level-3 nlcgrids, NO density pruning — both stacks default small_rho_cutoff
= 0; kernel mask rho >= 1e-10, |w| > 1e-14), SAD mirror density
(preregistered stand-in; verdict keys on >=2x-band margins), 1024 uniformly
sampled i-points per cell × all masked j in chunks; kept-fraction K(Rc)
with bootstrap-over-i CIs, conservative absolute tail-truncation estimates,
per-i deterministic bounds, and the mass-weighted decay profile.

**Validation (P0)**: internal chunked-vs-dense transcription check exact
(0.0); external validation vs pyscf CPU's independent VV10 implementation
to **1.9e-16 relative** on matched point sets. Two prereg authoring
corrections recorded with zero measurement impact: the wb97m-v NLC constant
is the library value (6.0, 0.01), not the prereg-authored 0.0093; and the
first external comparison (1.17e-5 rel) was precisely attributed to the
**CPU-vs-GPU zero-rho threshold gap (pyscf 1e-8 vs gpu4pyscf 1e-10)** — a
documented upstream implementation-difference micro-finding consistent with
the historical CPU-vs-GPU VV10 ~4e-7 Ha floor class.

**Result — every clause HIT the INFEASIBLE band, decisively**:
K(Rc@1e-12 Ha) = 0.9989 (benzene), 0.9996 (vitc), 0.9951 (methoxy),
0.99988 (tamoxifen); at the loose 1e-9 gate-scale bar K >= 0.9709
everywhere. The safe cutoff is 1.4-1.5x the molecular+grid extent
(15-36 Bohr vs extents 12.5-25.4) and keeps >99.5% of pairs — screening
ceiling <= 1.05x at the gate-scale bar, <= 1.005x at the sub-noise bar,
on every in-scope class including the extended best case. Decay exponents
-4.76..-5.83: at Rc = extent the remaining tail is 4.6e-8..3.5e-4 Ha —
real dispersion physics to the grid boundary, not quadrature noise.

**DECISION (the preregistered F-INFEASIBLE branch)**: distance screening
of the VV10 UWE kernel is infeasible at production shapes; no SP-10 GPU
prototype. Together with SP-2 (+20% reverted), SP-3 (arithmetic floor),
SP-4 (diffuse residual) and SP-8 (policy-invariant cycle surface), the
card's C(auto)-faster-than-B target is **measured-unreachable at unchanged
gates/defaults on sm_89 within every route class ever named**; C stands at
the honest parity band (9/12 mr18 cells faster than B, aggregate ~+1.5%).
Scope note: the multipole/FMM far-field family is a different algorithm
class, NOT falsified here, but remains a large research program (the
g-prefactors are two-endpoint density-dependent, not a pure 1/r^6 kernel)
whose target lanes SP-6 already answered (strict safe route incl all
verification = 1.8-3.3x B, NO) — standing unchanged: not started without
fresh card-visible justification.

Evidence: `2026-10-25-sp9-vv10-screening-feasibility/` (10-file hash
manifest: prereg, results, adjudication, all 6 run logs, script).
Effects-index dataset 177 `sp9-vv10-screening-feasibility-20261025`
(5 sources); full sweep **2206 OK / 0 missing / 5 mismatches = exactly
the documented set, zero new**. GPU: none (CPU-only process; idle
throughout). Gates/defaults unchanged; no source change; no historical
row relabeled.

## SP-10 (2026-10-25): the LC-101 UKS-NLC gradient cast regression — repaired, measured, REVERTED per the preregistered rule

The card item-2 gradient-slice source audit (this session) found a latent
regression: LC-101 (d2135b6, 2026-10-13) regrouped `get_nlc_exc`'s NLC
gradient cast condition from LC-88's un-gated `if fp32_grid: astype(fp32)`
to `(fp32_grid or _grad_grid_f32()) and nset == 1`, un-casting the UKS
path (nset==2) **even in fp32 mode** — contradicting the comment LC-101
itself added ("the UKS NLC path keeps the fp32-mode-only cast") and its
commit-message claim "auto lanes structural no-ops" (the dtype-shim proof
was RKS-only). Since 2026-10-13 every auto-lane UKS+VV10 gradient
(methoxy/ethoxy/NO2 q0s1 wb97mv doublet families) paid fp64 NLC grid
contractions with numerics drifted off the LC-88 discipline. The suite
never caught it: the LC-88 functional tests are RKS-only, and
`test_grad_grid_f32.py::test_nlc_loop_knob_gate` asserted the defective
expression string verbatim — the regression was locked into the tests.

SP-10 preregistered (sha `d98c5b95`, 13:54:52Z, before any measurement)
the repair + paired measurement: one-line regroup
`fp32_grid or (_grad_grid_f32() and nset == 1)` (knob stays RKS-only per
LC-89), red->green via a NEW functional UKS dtype shim
(`TestUksFp32ModeCast`), and >=4 fresh-process blocks per state on
4 production UKS+NLC cells (methoxy/ethoxy/NO2 q0s1 wb97mv svp +
methoxy qzvp, 43-204 AO; 20+20 arms, all rc=0; R-lane controls each
state). Red CONFIRMED at clean e8d5f64; fix at d6aceef; suites green
(LC files 24, upstream rks/uks grad 27, full tools/tests 831+31
subtests, zero failures).

**Measured outcome — numerics harmless, cost justification FALSIFIED.**
e_tot bitwise identical before/after on every cell/block; dg swap
3.3e-10..1.1e-9 (exactly the LC-88 same-orbital class); R lane unchanged
to the ~1e-14 atomic floor; the methoxy "gate" readings (0.0369 svp /
0.0121 qzvp) are the frozen issue-7 branch-multiplicity classes —
identical before/after to 6 digits (zero fix contribution; a precision
gate on them is an invalid comparator by documented class — clause-
authoring erratum recorded). But the NLC piece ratio after/before =
1.041 / 0.998 / 0.993 / 0.857 against the preregistered <=0.50x bar,
t_grad 0.971-1.018, e2e 0.985-1.012: **no shrink anywhere**. Mechanism:
the LC-87/LC-101 cast wins (4.6x piece / 1.16-2.47x t_grad) were
measured at 222-298 AO GEMM-bound shapes; at 42-204 AO the deriv=2 NLC
contractions are launch/memory-bound and fp32 buys nothing (LC-101's
own honest b3lyp 0.805x regime). Session-drift control: the R lane's
UNCHANGED code path drifted +10-22% on the same sub-second pieces
between states.

**DECISION (the preregistered REVERT branch)**: no measured payoff, no
behavior change in production source (SP-2 precedent). Reverted at
e50e40c: the LC-101 expression returns with an HONEST comment (UKS NLC
un-cast on every lane BY MEASUREMENT; knob RKS-only stands); the tests
now pin the measured-kept behavior — `test_uks_fp32_mode_nlc_ao_is_
float64_measured_kept` turns RED if anyone re-enables the cast without
fresh measured evidence, and `test_nlc_loop_knob_gate` bars the SP-10
grouping. Post-revert confirmation: 4 arms (one per cell) reproduce the
before-blocks e_tot BITWISE — production numerics exactly restored.
Net source effect vs e8d5f64: the corrected comment + the falsification
tests only.

**Scientific finding** (completes the LC-87/88/89/101 gradient-cast
story on its last untested class): the fp32-cast mechanism is
GEMM-bound-shape-only (>= ~222 AO); the UKS+NLC production doublet
families' NLC gradient piece (26-45% of auto t_grad) is launch/memory-
bound — future UKS-NLC gradient cost work must target launch overhead,
not arithmetic dtype. This is NOT a route to the card's C>B target
(B's UKS-NLC path is un-cast by the same measurement; both lanes sit in
the same regime).

Evidence: `2026-10-25-sp10-uksnlc-grad-cast-repair/` (14-file hash
manifest: prereg + sha, driver, analyzer, before/after jsonl + logs,
analysis, adjudication, red-green log, post-revert confirmation,
tools-tests log). Effects-index dataset 178
`sp10-uksnlc-grad-cast-repair-20261025` (13 sources); full sweep
**2219 OK / 0 missing / 5 mismatches = exactly the documented set, zero
new**. GPU: serial single-GPU drivers + suites only; idle at every
launch. Gates/defaults unchanged; no historical row relabeled. Tooling
errata: 4 pre-data driver relaunches (argparse positional; PYTHONPATH;
cupy sync API; cupy/numpy conversion) + pid-file-in-repo-cwd recurrence
+ stale-setsid-wrapper PID false alarm — all zero-measurement.

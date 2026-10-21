# GPU correctness mainline recovery — 2026-10-21

Status: preliminary safety pilot passed; production-policy integration and broader regression remain pending. No independent acceptance, merge, default change, or gate relaxation.

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

## Publication repair

The manifest generator and protected publisher file-count caps are raised together from 100 to **256**, allowing the current >100-file cumulative patch. Tests accept 106 and 256 files, reject 257, and retain the 1 MB per-file / 5 MB total-content limits. Path/hidden-file restrictions, credential scanning, fixed-repository transport, credential isolation, draft-only publication and manual merge/acceptance restrictions are unchanged.

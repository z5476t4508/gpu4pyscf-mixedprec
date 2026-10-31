# benchv1 first full evaluation — published record copies (2026-10-30/31)

Verbatim copies of the frozen benchmark-v1 evaluation records, published for
review. Originals live in the private evidence tree
`/data/results/gpu4pyscf-agent/tasks/issue-9/2026-10-30-benchv1-exec`
(stats/recompute/daily), `.../2026-10-30-benchv1-sentinels/{snt1,snt2}`
(sentinel first-view records) and `.../2026-10-30-benchv1-gpu-regression`
(F3 erratum); every file here is byte-identical to its evidence-tree
original (sha256 pinned in `state/long-chain/effects-index.json` datasets
#205 obcr / #206 s where applicable).

Execution provenance (both evaluations): source-pinned revision 8d144ed,
frozen snapshot `benchv1-snapshot-8d144ed`, RTX 4090 single-GPU serial
driver `run_upstream_batch.py`, 8 blocks per role per cell, 64 frozen
PubChem cohort cells (16 molecules x {b3lyp, m062x, wb97mv} def2-SVP +
r2scan3c def2-mTZVPP). O/B/C/R run 2026-10-30 (schedule sha256 1387f51b…),
S (auto-strict, paired B) run completed 2026-10-31T11:40Z (schedule sha256
f3be9950…). Statistics by `benchmark_v1_stats.py` (tool commit d898e96);
independent recomputation by `benchv1_independent_recompute.py`
(commit ada374a; raw-sample re-derivation, merge-fidelity and role-pin
re-verification, no GPU).

## Files

- `benchv1-obcr-stats.{md,json}` — 64-cell O/B/C/R first evaluation
  (planned 64 / gate-pass 64 / fail-missing 0; 35 actual-SCF-FP32,
  29 gated fallback). O_to_B geomean 2.3581x [2.3188, 2.3989],
  B_to_C 0.9989x [0.9844, 1.0150] (95% LB < 1: faster-with-confidence NOT
  met — net-neutral), O_to_C 2.3556x.
- `benchv1-s-stats.{md,json}` — 64-cell S (auto-strict) evaluation,
  64/64 certified, zero failures. B_to_S geomean 0.3600x (n=64):
  ratio < 1 means the certified auto-strict route COSTS MORE than plain
  optimized FP64 (honest direction; S cost includes its own
  verification/matching/fallback, never borrows C speedups).
- `benchv1-daily-comparisons{,-s}.json` — the two daily comparison
  snapshots written by the chain (obcr 2026-10-30T20:06Z, s
  2026-10-31T11:40Z).
- `recompute-obcr-final.json` / `recompute-s-final.json` — full-coverage
  independent recompute verdicts: mode-obcr rc=0 verdict OK citable,
  64/64, vs_official AGREE; mode-s rc=0 verdict OK citable full_coverage,
  64/64, vs_official AGREE, 128 role docs 0 merge mismatches, 0 pin
  defects. These two records are the pre-registered requirement before
  any figure above is cited.
- `snt1-compare.{md,json}` + `SNT1_FIRSTVIEW_BAND_RECORD_*` — SNT-1
  tier-2 CPU-FP64-vs-GPU-strict-R implementation-agreement sentinel,
  4 cells; band decision recorded AT FIRST VIEW per card (not a
  preregistered acceptance band); tier-2 evidence only, never the
  mixed-precision gate.
- `fd-results.{md,json}` + `SNT2_FIRSTVIEW_INTERPRETATION_RECORD_*` —
  SNT-2 finite-difference force diagnostics, 2 cells x 2 steps, worst
  |delta| 2.368e-5 Eh/Bohr, all within 3x nominal 1e-4 floor; diagnostic
  only, can neither pass nor fail the 3e-5 gate; interpretation rules
  preregistered 2026-10-30T06:56Z before any FD output existed.
- `F3_ERRATUM_20261030.md` — rotation-invariance original contract
  FAIL/unresolved (dE 5.4379e-6 Ha, back-rotated max dg 8.5989e-5 vs
  gates 1e-8 / 3e-5) and the separate narrower F3b FP64-vs-FP64
  diagnostic; preserved verbatim, nothing relabeled.

## Status discipline

All results are first-evaluation measurements under the frozen protocol;
review_pending; no acceptance, no default/gate change. F3 stays
FAIL/unresolved. Failures/fallbacks stay in denominators. Publication is
not acceptance; every verdict waits for manual review.

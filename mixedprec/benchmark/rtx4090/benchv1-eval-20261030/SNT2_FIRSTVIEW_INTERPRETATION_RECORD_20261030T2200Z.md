# SNT-2 first-view interpretation record (2026-10-30T22:00Z, session 20261030T210359087602Z)

Interpretation rules were preregistered 2026-10-30T06:56:53Z (before any FD output existed) and applied
VERBATIM at first view (fd-results.json first read 2026-10-30T21:58:45Z, immediately after the chain
compare stage wrote it). No band re-decision was made. Diagnostic only — never the 3e-5 Eh/Bohr
mixed-precision gradient gate; not acceptance.

| cell | step (Bohr) | max abs delta (Eh/Bohr) | threshold | verdict |
|---|---|---|---|---|
| cytosine_b3lyp_def2-svp (atoms #8 H_ring-N1, #12 H_amino) | 0.001 | 1.217e-05 | 3e-4 | WITHIN |
| cytosine_b3lyp_def2-svp | 0.002 | 1.223e-05 | 3e-4 | WITHIN |
| acetaminophen_wb97mv_def2-svp (atoms #19 H_phenolic, #2 N_amide) | 0.001 | 2.368e-05 | 3e-4 | WITHIN |
| acetaminophen_wb97mv_def2-svp | 0.002 | 2.368e-05 | 3e-4 | WITHIN |

Fail-closed status: 48/48 displaced runs rc=0 (compare stage is fail-closed on missing/non-ok rows);
both steps reported; no substitution.

Non-normative observations (interpretation notes, no new claims):

- measured noise floor 0.0 at every cell-step (all 3 repeats per run bitwise-identical e_tot), so repeat
  scatter sets no floor; the preregistered nominal 1e-4 cancellation floor remains the band basis.
- per-component deltas are nearly equal across the two steps instead of scaling ~h^2 or scattering: the
  FD-vs-analytic difference is dominated by a small systematic component common to both steps. Root
  cause is not claimed here.
- worst deltas (~2.4e-5) are below but of the order of the 3e-5 gate; per prereg semantics this
  diagnostic can neither pass nor fail that gate.

Pins: fd-results.json sha256 105de9090e6844de9d6b275b182f59e88bbf1cd4e7c55a9d33c8430fca1030b8;
preregistration sha256 0e64677a7687b56ea6889d2b59de52bb3c28f4f09a4d9bd675a66b5a9341668d;
harness sha256 dda1cd31... (matches prereg pin). Machine record:
SNT2_FIRSTVIEW_INTERPRETATION_RECORD_20261030T2200Z.json. review_pending=true.

Chain continuity: the S batch launched at 2026-10-30T21:56:15Z after HEAD gate PASS (d898e96); this
record did not interrupt or modify the chain.

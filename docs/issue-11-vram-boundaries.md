# Issue #11 (a): RTX 4090 VRAM capacity boundary mapping & batch throughput

**Status**: review_requested (provisional; predecessors #2–#10 review_requested,
inherited as provisional under QC_CONTINUE_UNREVIEWED=1). Attempt 1/3, 2026-10-04;
**reworked 2026-10-06** per Codex review P1 (safe current-free VRAM admission;
see "Admission contract rework" below).
**Scope**: part (a) only — safe capacity measurement, chunk-audit re-verification,
optional batch single-point channel. Part (b) (tiling/streaming/pinned-host/
overlap/out-of-core host-RAM/VRAM cooperation) is explicitly **blocked-on-research**
and NOT implemented here, per the plan's issue-#11 split.

## Deliverables

| Deliverable | Location | Evidence |
|---|---|---|
| Capacity mapper (fresh subprocess/cell, gate + oom/skip separation) | `mixedprec/benchmark/rtx4090/vram_mapper.py` | `vram_boundary_ladder.json` + runlog, sha256-anchored |
| Memory-aware chunk audit (H1–H4 re-verification + new-site scan) | `mixedprec/benchmark/rtx4090/chunk_audit.py` | `chunk_audit.json` |
| Batch single-point throughput channel (resumable, crash-safe) | `mixedprec/benchmark/rtx4090/batch_channel.py` | `batch_energy_matrix.json` + runlog |
| H1 itemsize fix (audit finding) | `gpu4pyscf/grad/rhf.py` `get_grad_hcore` | live regression check below |
| Admission contract (2026-10-06 P1 rework) | `vram_mapper.gate_decision` / `gate_record`, shared by all runners | `issue-11-rework-20261006/` (ladder, live probes, batch, runner smokes) |
| Unit tests (pure cores, no GPU needed) | `tools/tests/test_issue11_vram_batch.py` (51 tests) | both QC policies: 332 OK |

## Admission contract rework (2026-10-06, Codex review P1)

**Defect (P1)**: the original `gate_decision` admitted on DEVICE TOTAL
capacity: `gate_decision(8, 24564, 1024, 2048)` returned `(True, None)` —
an 8 GiB request admitted with 1 GiB free and 2 GiB reserve. Total
capacity is not a runtime admission budget (other processes or pool
carry-over can hold most of the device). Reproduced by the regression
test `test_p1_regression_low_free_never_admits` (red on the old gate,
green on the new one) and live (below).

**The contract now** (`vram_mapper.gate_decision`, single source shared by
the mapper probe, `batch_channel.admission_gate_record`, and
`rtx4090_runner.cderi_admission_gate` used by the energy and gradient
runners):

1. **Admission basis = measured current free VRAM − reserve**
   (`--headroom-mib`). Boundary: request == budget admits (≤).
2. **Nonpositive budget (free ≤ reserve) skips safely** — no request is
   admitted, whatever its size; the skip reason names the budget
   arithmetic, and no OOM is ever provoked to "discover" a boundary.
3. **Separate reasons**, never conflated:
   `nonpositive-admission-budget` → `predicted-exceeds-host-ram` (host
   mode) → `beyond-device-capacity` (theoretical: request > total −
   reserve on an empty device; permanent verdict) →
   `predicted-exceeds-free-minus-reserve` (transient: fits an empty
   device but not now; retry later, do not relabel).
4. **Explicit accounting, no invented numbers**: the gate record carries
   the resident request (conservative cderi fp64 upper bound,
   npairs × naux × 8 B), the work reserve, and — separately —
   `device_capacity` (theoretical empty-device statement, `fits_empty_
   device`), which is never the decision. The SCF/JK/grid working set is
   NOT predicted (any estimate would be invented); the reserve covers it
   and the actual peak is measured after the run
   (`mem.gpu_peak_alloc_mib` / `gate.measured_gpu_peak_alloc_mib`).
5. **Host-CDERI fallback cells**: no GPU residency request (the tensor is
   host-resident); admission still requires a positive GPU working budget
   (free − reserve, recorded as `gpu_working_budget_mib`) plus a host RAM
   bound: request + staging margin (4 GiB) ≤ host available
   (`/proc/meminfo` MemAvailable, recorded as `host_available_gib`).

**Integration**: every cell/row of the mapper probe, the batch channel,
and the energy/gradient GPU lanes now carries its gate record (accepted
rows too). The batch channel additionally treats a gate `skipped` row as
a transient state — re-attempted on the next invocation (a skip reflects
the device's free memory at that moment), unlike `failed` rows
(execution errors, `--retry-failed` only) and `ok` rows (never re-run).
A gate that cannot evaluate (e.g. AO counts unavailable) is recorded as
`gate-indeterminate` and the cell runs ungated with measured memory —
never a fabricated admission. `derive_boundaries` audits that every cell
that reached VRAM admission carries a `current-free-minus-reserve` gate
record (`admission_gate_audit.cells_missing_gate_record`; time-budget
parent-side skips are exempt) plus a skip-reason histogram.

**Measured evidence (2026-10-06, `/data/results/gpu4pyscf-agent/tasks/
issue-11-rework-20261006/`, all sha256-anchored)**:

- **Live P1 scenario** (`live_probe_low_free.json`): a 20.5 GiB holder
  process pinned the device; probing Tamoxifen/def2-svp (2.8 GiB request)
  measured free 2272 MiB, reserve 2048 MiB → budget 224 MiB → **skipped**
  with `predicted-exceeds-free-minus-reserve` and the note that it fits
  an empty device (capacity reporting) — the exact scenario the old gate
  admitted. No run was attempted (no peak key in the row).
- **Live nonpositive budget** (`live_probe_nonpositive_budget.json`):
  with free 120 MiB < reserve 2048 MiB, even the tiny benzene probe skips
  with `nonpositive-admission-budget` — no OOM attempt.
- **8-cell ladder rerun** (`vram_boundary_ladder.json`, new gate): 7 ok
  (peaks 655/973/2669/3311/8045/2371-host/10245 MiB) + the Aza/tzvp
  cell now skipped as `beyond-device-capacity` (59 GiB > 21.5 GiB
  empty-device budget) — zero unexpected OOM; `cells_missing_gate_
  record: []`. The host-CDERI cell records `gpu_working_budget_mib`
  21607.8 and `host_available_gib` 29.3.
- **Batch channel** (`batch_smoke.json`): 4 cells — 3 ok (gate records
  admitted=true, peaks 1411–1469 MiB) + Aza/tzvp gate-skipped; re-
  invocation re-attempted ONLY the skipped cell (0 re-runs of ok rows).
- **Energy/gradient runners** (`energy_gate_smoke.json`,
  `gradient_gate_smoke.json`): methanol fp64+auto lanes schema-VALID with
  `gate.admission_basis = current-free-minus-reserve` and measured peaks
  86–126 MiB recorded on the gate records.

## Measured boundary map (fp64 DF, def2-universal-jkfit, grids level 3)

Peak device memory vs AO, fresh process per cell, single sample, conv_tol 1e-9
(capacity probe — NOT an rtx4090-energy-v1 accuracy/timing document):

| Cell | AO | cderi (upper bound) | Residency | GPU peak MiB | converged |
|---|---|---|---|---|---|
| benzene/def2-tzvp r2SCAN | 222 | 0.1 GiB | gpu | 655 | yes |
| VitaminC/def2-svp wb97m-v (NLC) | 208 | 0.2 GiB | gpu | 979 | yes |
| Tamoxifen/def2-svp r2SCAN | 537 | 2.8 GiB | gpu | 2669 | yes |
| Tamoxifen/def2-svp B3LYP + **gradient** | 537 | 2.8 GiB | gpu | 3311 | yes |
| Tamoxifen/def2-tzvp r2SCAN | 1042 | 10.6 GiB | gpu | 8047 | yes |
| Azadirachtin/def2-svp r2SCAN | 934 | 15.1 GiB | gpu | 10247 | yes |
| Azadirachtin/def2-tzvp r2SCAN | 1845 | 59.0 GiB | — | **skipped** (beyond-device-capacity, 2026-10-06 gate; earlier label: predicted-exceeds-device-headroom) | — |

**Boundaries (measured)**:
- Zero unexpected OOM at every admitted level (acceptance criterion: no
  unexpected OOM within the energy matrix). The largest measured GPU-resident
  energy cell is 1042 AO (peak 8047 MiB, within the 2048 MiB headroom budget);
  the largest measured gradient cell is 537 AO B3LYP (peak 3311 MiB).
- The first ladder cell beyond the budget (Azadirachtin/tzvp, cderi upper
  bound 59 GiB > 21.5 GiB empty-device budget) is an explicit skip row with
  the prediction recorded — a budget decision, never presented as an OOM.
- 934-AO Azadirachtin/svp (peak 10247 MiB) passes with the full 2048 MiB
  headroom: the S2→S3 boundary on this hardware sits between 934 and 1845 AO
  for def2-svp-class footprints, consistent with the plan's 5090 prior
  (energy/gradient unconstrained) but now measured on the 4090.
- **CDERI residency boundary**: all cells up to 15.1 GiB cderi upper bound
  remained GPU-resident under the automatic 0.95-word-avail rule. Forcing the
  existing host fallback (`use_gpu_memory=False`) on Tamoxifen/tzvp moved the
  GPU peak from 8047 → 2371 MiB at host RSS 8.4 GiB, same energy to ~1e-12 Ha,
  SCF 30.4 s vs (GPU-resident, capacity probe) single-sample timing — a
  capacity diagnostic only: **no offload/streaming claim** (that is part (b),
  blocked-on-research).

## Chunk audit: one real bug found and fixed

The four historical fix sites (STATUS.md whack-a-mole) were re-verified on
the RTX 4090 tree; all still consult `get_avail_mem`. The audit found one
real defect:

- **H1 (`gpu4pyscf/grad/rhf.py:get_grad_hcore`)**: the 5090-era fix capped the
  work buffer with `int(mem_avail * .125) // (3 * nao1 * nao1)` — missing the
  fp64 itemsize, so the cap actually bounded the tensor to **100% of free
  VRAM**, not the documented 1/8. (H2/H3 divide by 8 bytes correctly.) At
  24 GB free the buggy cap allowed ~24 GB work tensors — precisely the class
  of allocation that OOM'd at 3370 AO on the 5090. Fixed by adding the `* 8`
  divisor (`chunk_audit.json` incident note records the finding). Live
  regression check: methanol/B3LYP gradient after the fix matches the
  pre-fix value to ~7e-12 relative (same physics, smaller cap: batch shrinks
  only when memory-bound, which S0–S2 sizes never trigger on a free GPU).
- The fifth historical incident (`_get_vh1` qzvp shape bug) is **not** a
  memory issue and stays out of scope (Hessian phase; l=4/qzvp only).
- New-fixed-extent scan of the DF energy/gradient path: **0 findings**
  (no literal-leading-dimension allocations ≥100 since the fixes).

## Batch throughput channel

In-process sequential executor with crash-safe per-cell persistence
(JSON upsert by row_key; `ok` rows never re-run; `failed` rows re-run only
with `--retry-failed`; `running` rows from a crashed process reset to
pending; **admission-gate `skipped` rows re-attempted each invocation** —
a skip is a transient device-free-memory state, not a cell property;
per-cell exceptions recorded, batch continues). Since the 2026-10-06
rework each cell passes the shared admission gate before building
anything (see above). Measured on the
admitted energy-matrix cohort: 10 cells (methanol/benzene/methoxy ×
r2scan/b3lyp/wb97m-v/r2scan3c, both fp64 and auto lanes, plus
Tamoxifen/svp), **10/10 ok, 808 cells/hour**, per-cell GPU peaks
1465–2039 MiB — batch mode carries a pool-carry-over footprint but stays
well inside budget. Re-invocation ran 0 cells (resume verified). The
2026-10-06 rework smoke additionally exercised the skip path (Aza/tzvp
gate-skipped, re-attempted on re-invocation, still skipped — see
`batch_smoke.json`).

## Honest limitations

- The gradient boundary is measured at 537 AO only; larger gradient cells
  were not run (time budget; issue #12 owns the full gradient matrix).
- The mapper is a capacity probe: single sample, conv_tol 1e-9, no warmup
  separation — it does not satisfy the rtx4090-energy-v1 timing/accuracy
  contract and makes no speedup or precision claims.
- The host-CDERI cell is a diagnostic of existing behavior (upstream
  automatic fallback), not new offload development; part (b) remains
  blocked-on-research with its stated launch conditions.
- All numbers above are RTX 4090 measured (schema `vram-boundary-v1`);
  nothing from the RTX 5090 history is cited as evidence.

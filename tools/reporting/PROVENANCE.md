ebd47471591943c1107077b3ca5b5b7dadfa18e173018b3f029abe09323131ea  /opt/qc-agent/three_way_report.py
ffc9c82527d517bb8e9ceac0d1d9782842ebf39702858de904543a7fa96d5837  /opt/qc-agent/effects_report.py
3f7999f5f626f7b03a985849ac34ad2239fdb26ff406b5c945b207080a52849c  tools/reporting/three_way_report.py (B01 proposed repair, superseded by the line below)
9dace4498b433bf677d991a253d4c5b05caee93fe0b3c78ea140831b98b864c1  tools/reporting/three_way_report.py (B01 proposed repair + R01 re-review failure-visibility fix, review_pending)
d494e39b0f482539de14a6e927e8ae705221684bed037996099b5dd26c3e68d1  tools/reporting/three_way_report.py (B01+R01 repair + benchv1 batch-separation repair 2482844, review_pending)
# Deployment copies of the operator's read-only /opt/qc-agent reporter
Copied 2026-10-18 by the server executor for version backup and local
testing only (control card upstream-comparison-20261018.md authorizes
read-only copies; the DEPLOYED /opt/qc-agent files are never modified by
this repository). Any drift between these copies and /opt/qc-agent is
detected by tools/tests/test_upstream_comparison.py, which compares the
sha256 of the deployed files against this provenance record at test time.
#
# Refresh 2026-10-25 (SP-5): the operator updated the deployed
# effects_report.py (mtime 2026-10-25T07:39:27Z); the added guards make
# schedule-only and mechanism/audit datasets render an explanatory note
# instead of empty legacy performance tables (scf-priority-20261025 card
# item 6). three_way_report.py is unchanged since 2026-10-18. Copy, test
# pin and this record updated together; the drift test was observed RED
# on the stale copy before the refresh and GREEN after.
#
# Intentional divergence 2026-10-27 (B01, coordination board 2026-10-27):
# the deployed three_way_report.load_cell rule "no actual FP32 entry
# evidence for auto" (deployed lines 59-60) renders the 7 legitimate
# SP-14 size-gated SP-32 cells as unknown (7/11 at batch
# sp32-fournrole-landedsource-20261027-a; all 11 cells passed their
# scientific gates). The repo copy of three_way_report.py is now the
# PROPOSED repair: an all-zero FP32 record is accepted ONLY as a gated
# lane proven from the document itself (measured nao below the resolved
# _AUTO_HEAD_MIN_AO threshold + unswitched policy + fp64 gradient -- the
# same contract as the SP-18 merge-layer validator
# upstream_comparison.validate_comparison_doc), and render() shows such
# cells as an explicit size-gated fallback category, outside the
# FP32-executed medians and outside the unknown count. Every displayed
# median/count is unchanged; nothing wider is accepted. Deployment
# requirement if the operator adopts it: the source tree containing
# gpu4pyscf/scf/hf.py must be on sys.path of the reporting runtime (the
# threshold is resolved by AST scan, never by import; unresolvable fails
# CLOSED -- rejection, not silent acceptance). The DEPLOYED files remain
# the two hashes above, untouched; the drift test pins them, not this
# copy. Status: review_pending (Codex review + operator deployment
# decision), never self-accepted. Regression:
# tools/tests/test_b01_gated_auto_report.py (red at 74974d0, green
# after); evidence
# /data/results/gpu4pyscf-agent/tasks/issue-9/2026-10-27-b01-gated-auto-reporting/.

#
# Intentional divergence 2026-10-31 (benchv1 chain-complete boundary,
# session 20261031T103225818961Z): after the post-batch chain appended
# benchv1-20261030-obcr and benchv1-20261030-obcr-S to the upstream
# comparison index, the deployed render() batches[-1] rule presented the
# S batch (paired auto-strict, roles B/S only) as a fabricated all-missing
# O/B/C table (0/64 pass, 64/64 缺失) in the daily summary; the benchv1
# obcr batch's 8-block merged docs also fail the SP-32-era four-repeat
# provenance contract and would render the same misleading way. The repo
# copy now classifies every batch by the role keys its cells pin
# (_batch_kind: 'obcr' | 's' | None) BEFORE any doc load: S batches and
# protocol-incompatible newer batches render as explicit
# "另表/未在本表渲染" notes (never missing, never in the denominator),
# the table renders the NEWEST VERIFIABLE four-role batch (backward
# search with a load_cell probe on the newest cell), render(index,
# batch_id=...) selects deterministically for regression pinning, and
# incompatible/unknown batch_id selection fails closed with ValueError.
# No displayed median/count of a rendered batch changes (SP-32 pinned
# values byte-identical). The DEPLOYED /opt/qc-agent files remain the
# two hashes above, untouched; deployment stays operator-owned. Status:
# review_pending. Regression: tools/tests/test_b01_gated_auto_report.py
# (B01BatchSeparationTests + the batch_id-pinned real-index test; red at
# 09b3dbe, green at 2482844).

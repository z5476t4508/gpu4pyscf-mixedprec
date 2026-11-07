ebd47471591943c1107077b3ca5b5b7dadfa18e173018b3f029abe09323131ea  /opt/qc-agent/three_way_report.py
ffc9c82527d517bb8e9ceac0d1d9782842ebf39702858de904543a7fa96d5837  /opt/qc-agent/effects_report.py
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

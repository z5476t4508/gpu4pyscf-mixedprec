# SNT-1 first-view band record (2026-10-30T21:00Z, session 20261030T170617560740Z)

Bands were preregistered 2026-10-30T06:56:53Z (before any SNT-1 output existed) and applied VERBATIM at first view.
All 4 cells IN-BAND. Tier-2 CPU-vs-GPU implementation agreement only — never the 1e-6/3e-5 matching-R gate; not acceptance.

| cell | class | band dE | abs dE | band dg | max abs dg | verdict |
|---|---|---|---|---|---|---|
| uracil_b3lyp_def2-svp | hybrid-gga | 1e-8 | 5.150e-11 | 1e-5 | 2.361e-06 | IN-BAND |
| propranolol_m062x_def2-svp | hybrid-mgga | 1e-8 | 2.547e-11 | 1e-5 | 9.079e-07 | IN-BAND |
| adenine_wb97mv_def2-svp | vv10-nlc | 1e-6 | 1.824e-08 (<4e-7 floor: no follow-up) | 1e-5 | 7.651e-07 | IN-BAND |
| benzoic_acid_r2scan3c_def2-mtzvpp | d4-gcp | 1e-8 | 2.387e-12 | 1e-5 | 1.387e-06 | IN-BAND |

Exposure caveat retained (GPU R rows on disk at band-derivation, unread then): if Codex rules it disqualifying,
outputs are exploratory diagnostics and independent acceptance stays pending. review_pending=true.

Machine record: SNT1_FIRSTVIEW_BAND_RECORD_20261030T2100Z.json (sha256-pinned compare sources).

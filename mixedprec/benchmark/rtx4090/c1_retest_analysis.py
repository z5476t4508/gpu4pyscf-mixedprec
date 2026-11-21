#!/usr/bin/env python3
"""Issue #10 C1 retest analysis: warp4 vs warp1 (reference) VV10 UWE kernel
across the admitted omegaB97M-V matrix.

Inputs are the per-cell, per-variant benchmark documents written by
rtx4090_runner.py (`c1_retest_<cell>_<variant>.json`). For each cell and
lane (fp64, auto) the analysis derives, directly from the schema rows:

  * precision: dE(warp4 - warp1) per lane, plus dE(auto - fp64) WITHIN each
    variant (the issue-#10 acceptance quantity: the auto lane's accuracy
    against its own-variant fp64 reference must stay <= 1e-6 Ha);
  * speed: median end-to-end lane time ratio warp1/warp4 (medians only are
    compared; raw sample arrays are carried into the output for audit);
  * convergence: cycles and converged flags for every lane.

Rules of the plan that this analysis enforces/records:
  * no cross-setting timing comparison -- both variants share identical
    settings by construction (same runner defaults, same source commit);
  * the precision gate compares same-implementation lanes only (warp4 auto
    vs warp4 fp64), never warp4 vs warp1 as a gate -- the warp4-vs-warp1
    difference is reduction-order noise reported as a diagnostic;
  * fp32-screen rows are absent by design (issue #9: non-convergent), so
    any unexpected lane makes the analysis fail loudly rather than
    silently skip.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schema_validator import validate_file  # noqa: E402

PRECISION_GATE_HA = 1e-6
EXPECTED_LANES = ('fp64', 'auto')


def _row_by_lane(doc, lane):
    for row in doc['rows']:
        if row['row_key']['lane'] == lane:
            return row
    return None


def load_variant(path, evidence_root):
    issues = validate_file(path, evidence_root=evidence_root)
    if issues:
        raise SystemExit(
            f'{path}: schema-INVALID, refusing to analyse:\n'
            + '\n'.join(f'  {i["rule"]}/{i["tag"]}: {i["message"]}'
                        for i in issues))
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def analyse_cell(cell, paths, evidence_root):
    docs = {v: load_variant(p, evidence_root)
            for v, p in paths.items() if os.path.exists(p)}
    if set(docs) != {'warp1', 'warp4'}:
        raise SystemExit(f'{cell}: expected warp1 and warp4 documents, '
                         f'found {sorted(docs)}')
    entry = {'cell': cell,
             'source_commit': docs['warp1']['meta']['gpu4pyscf_source_commit']}
    if docs['warp4']['meta']['gpu4pyscf_source_commit'] != \
            entry['source_commit']:
        raise SystemExit(f'{cell}: warp1/warp4 documents come from '
                         'different source commits; cross-variant speed '
                         'comparison would violate the no-cross-setting rule')
    gate_ok = True
    for lane in EXPECTED_LANES:
        r1 = _row_by_lane(docs['warp1'], lane)
        r4 = _row_by_lane(docs['warp4'], lane)
        if r1 is None or r4 is None or r1['status'] != 'ok' or \
                r4['status'] != 'ok':
            raise SystemExit(f'{cell}/{lane}: missing or non-ok row '
                             f'(warp1={None if r1 is None else r1["status"]},'
                             f' warp4={None if r4 is None else r4["status"]})'
                             '; honest-failure rows must be handled, not '
                             'skipped by the analysis')
        t1, t4 = r1['result']['timings'], r4['result']['timings']
        e1, e4 = r1['result']['e_tot'], r4['result']['e_tot']
        entry[lane] = {
            'e_warp1': e1, 'e_warp4': e4,
            'dE_variant': e4 - e1,
            'cycles_warp1': r1['result']['scf_cycles'],
            'cycles_warp4': r4['result']['scf_cycles'],
            'converged_warp4': r4['result']['converged'],
            'median_s_warp1': t1['median_s'], 'median_s_warp4': t4['median_s'],
            'samples_s_warp1': t1['samples_s'],
            'samples_s_warp4': t4['samples_s'],
            'speedup_median_warp4': t1['median_s'] / t4['median_s'],
        }
    # the acceptance quantity per variant: auto vs its OWN fp64 lane
    for v in ('warp1', 'warp4'):
        dE_auto = docs[v]['rows'][0]  # placeholder replaced below
        dE_auto = None
        e_fp64 = _row_by_lane(docs[v], 'fp64')['result']['e_tot']
        e_auto = _row_by_lane(docs[v], 'auto')['result']['e_tot']
        dE_auto = e_auto - e_fp64
        entry[f'dE_auto_vs_fp64_{v}'] = dE_auto
        if abs(dE_auto) > PRECISION_GATE_HA:
            gate_ok = False
    entry['precision_gate_within_1e-6'] = gate_ok
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(
        description='Derive the C1 warp4-vs-warp1 retest table from '
                    'rtx4090-energy-v1 documents')
    p.add_argument('--evidence-dir', default=os.path.join(
        '/data/results/gpu4pyscf-agent/tasks', 'issue-10'))
    p.add_argument('--cells', nargs='+', default=[
        's0_methanol_svp', 's1_benzene_svp', 's1_methoxy_svp_uks',
        's2_benzene_tzvp', 's2_methoxy_tzvp_uks'])
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)

    cells = []
    for cell in args.cells:
        paths = {v: os.path.join(args.evidence_dir,
                                 f'c1_retest_{cell}_{v}.json')
                 for v in ('warp1', 'warp4')}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise SystemExit(f'{cell}: missing retest documents: {missing}')
        cells.append(analyse_cell(cell, paths, '/data/results'))

    doc = {
        'schema': 'issue10-c1-retest-v1',
        'purpose': 'C1 full-matrix retest: warp4 VV10 UWE kernel vs the '
                   'reference warp1 kernel on the admitted omegaB97M-V '
                   'cells; per-variant auto-lane precision gate and '
                   'cross-variant speed diagnostic',
        'precision_gate_ha': PRECISION_GATE_HA,
        'cells': cells,
        'all_cells_gate_ok': all(c['precision_gate_within_1e-6']
                                 for c in cells),
    }
    text = json.dumps(doc, indent=1)
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            f.write(text + '\n')
        print(f'wrote {args.out}')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

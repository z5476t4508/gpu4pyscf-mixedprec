#!/usr/bin/env python3
"""Issue #10 rework analysis: multi-i tile VV10 UWE kernel vs the reference
warp1 kernel across the admitted omegaB97M-V matrix.

Same contract as c1_retest_analysis.py (which is warp4-specific), with the
variant pair generalized so the rework matrix (warp1 vs tile8, the best
microbenchmark variant) is analysed with identical rules:

  * precision: dE(tile - warp1) per lane (reduction-order diagnostic only,
    never a gate), plus dE(auto - fp64) WITHIN each variant -- the issue-#10
    acceptance quantity, gated at 1e-6 Ha;
  * speed: median end-to-end lane time ratio warp1/tile (medians compared;
    raw sample arrays carried for audit);
  * convergence: cycles and converged flags for every lane.

Both variant documents must come from the same source commit (no
cross-setting timing comparison); missing or non-ok rows abort the
analysis loudly rather than silently skipping.
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


def analyse_cell(cell, paths, ref_variant, new_variant, evidence_root):
    docs = {v: load_variant(p, evidence_root)
            for v, p in paths.items() if os.path.exists(p)}
    if set(docs) != {ref_variant, new_variant}:
        raise SystemExit(f'{cell}: expected {ref_variant} and '
                         f'{new_variant} documents, found {sorted(docs)}')
    entry = {'cell': cell,
             'source_commit':
                 docs[ref_variant]['meta']['gpu4pyscf_source_commit']}
    if docs[new_variant]['meta']['gpu4pyscf_source_commit'] != \
            entry['source_commit']:
        raise SystemExit(f'{cell}: {ref_variant}/{new_variant} documents '
                         'come from different source commits; cross-variant '
                         'speed comparison would violate the '
                         'no-cross-setting rule')
    gate_ok = True
    for lane in EXPECTED_LANES:
        r_ref = _row_by_lane(docs[ref_variant], lane)
        r_new = _row_by_lane(docs[new_variant], lane)
        if r_ref is None or r_new is None or r_ref['status'] != 'ok' or \
                r_new['status'] != 'ok':
            raise SystemExit(f'{cell}/{lane}: missing or non-ok row '
                             f'({ref_variant}='
                             f'{None if r_ref is None else r_ref["status"]},'
                             f' {new_variant}='
                             f'{None if r_new is None else r_new["status"]})'
                             '; honest-failure rows must be handled, not '
                             'skipped by the analysis')
        t_ref, t_new = r_ref['result']['timings'], r_new['result']['timings']
        e_ref, e_new = r_ref['result']['e_tot'], r_new['result']['e_tot']
        entry[lane] = {
            f'e_{ref_variant}': e_ref, f'e_{new_variant}': e_new,
            'dE_variant': e_new - e_ref,
            f'cycles_{ref_variant}': r_ref['result']['scf_cycles'],
            f'cycles_{new_variant}': r_new['result']['scf_cycles'],
            f'converged_{new_variant}': r_new['result']['converged'],
            f'median_s_{ref_variant}': t_ref['median_s'],
            f'median_s_{new_variant}': t_new['median_s'],
            f'samples_s_{ref_variant}': t_ref['samples_s'],
            f'samples_s_{new_variant}': t_new['samples_s'],
            f'speedup_median_{new_variant}':
                t_ref['median_s'] / t_new['median_s'],
        }
    # the acceptance quantity per variant: auto vs its OWN fp64 lane
    for v in (ref_variant, new_variant):
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
        description='Derive the multi-i tile retest table from '
                    'rtx4090-energy-v1 documents')
    p.add_argument('--evidence-dir', default=os.path.join(
        '/data/results/gpu4pyscf-agent/tasks', 'issue-10',
        '20261006-multii-tile'))
    p.add_argument('--cells', nargs='+', default=[
        's0_methanol_svp', 's1_benzene_svp', 's1_methoxy_svp_uks',
        's2_benzene_tzvp', 's2_methoxy_tzvp_uks'])
    p.add_argument('--variants', nargs=2, default=['warp1', 'tile8'],
                   metavar=('REFERENCE', 'NEW'),
                   help='variant pair to compare (document filename '
                        'pattern retest_<cell>_<variant>.json)')
    p.add_argument('--prefix', default='retest',
                   help='document filename prefix')
    p.add_argument('--out', default=None)
    args = p.parse_args(argv)

    ref_variant, new_variant = args.variants
    cells = []
    for cell in args.cells:
        paths = {v: os.path.join(
                    args.evidence_dir,
                    f'{args.prefix}_{cell}_{v}.json')
                 for v in (ref_variant, new_variant)}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise SystemExit(f'{cell}: missing retest documents: {missing}')
        cells.append(analyse_cell(cell, paths, ref_variant, new_variant,
                                  '/data/results'))

    doc = {
        'schema': 'issue10-multii-retest-v1',
        'purpose': 'multi-i tile VV10 UWE kernel retest: %s vs the '
                   'reference %s kernel on the admitted omegaB97M-V cells; '
                   'per-variant auto-lane precision gate and cross-variant '
                   'speed diagnostic' % (new_variant, ref_variant),
        'variants': [ref_variant, new_variant],
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

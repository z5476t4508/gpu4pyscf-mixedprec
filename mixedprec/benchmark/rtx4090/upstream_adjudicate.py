#!/usr/bin/env python3
'''P1-P6 adjudication FACTS for a four-arm upstream comparison batch.

Computes every number the stage adjudication needs from the LIVE batch
index, using the operator's DEPLOYED reporter code (the sha256-pinned
deployment copies under tools/reporting/: three_way_report.load_cell +
effects_report.measure), so the adjudication facts are NUMBER-IDENTICAL
to the daily report -- this tool deliberately defines no ratio or gate of
its own.  It emits per-cell numbers, method medians, and mechanical
band-membership counts against the preregistered prediction ranges; the
HIT/MISS adjudication itself is written by the session from these facts
(no verdict is auto-claimed here).

Read-only: never writes to the index, the batch docs, or the schedule.
Fail-closed: every cell in the schedule's denominator must be pre-listed
in the index; by default every cell must be complete (use --allow-partial
for explicitly-labeled interim facts).

Example (stage-1, after its 48 cells complete):
  python upstream_adjudicate.py \
      --index <upstream-effects-index.json> --batch-id uc-20261018-a \
      --schedule <formal-schedule-stage1.json> \
      --prereg <preregistration.json> \
      --out <evidence>/stage1-adjudication-facts.json \
      --md <evidence>/stage1-adjudication-facts.md
'''

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REPORTING = os.path.join(REPO, 'tools', 'reporting')

LETTER_TO_ROLE = {'O': 'upstream', 'B': 'optimized_fp64',
                  'C': 'optimized_auto', 'R': 'strict_fp64'}
COMPARISONS = (('O_over_B', 'upstream', 'optimized_fp64'),
               ('B_over_C', 'optimized_fp64', 'optimized_auto'),
               ('O_over_C', 'upstream', 'optimized_auto'))
NLC_CLASS = {'wb97mv': 'nlc-vv10'}
# verbatim ranges from preregistration.json predictions
PREDICTION_BANDS = {
    'P2_O_over_B': {'wb97mv': (2.5, None), 'b3lyp': (1.2, 2.0),
                    'm062x': (1.2, 2.0), 'r2scan3c': (1.2, 1.8)},
    'P3_B_over_C': (0.85, 1.25),
    'P4_O_over_C_non_nlc': (1.2, 2.5),
    'P4_O_over_C_nlc': (3.0, None),
}


def _load_reporting(name):
    if REPORTING not in sys.path:
        sys.path.insert(0, REPORTING)
    spec = importlib.util.spec_from_file_location(
        f'adjudicate_{name}', os.path.join(REPORTING, f'{name}.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


effects = _load_reporting('effects_report')
report = _load_reporting('three_way_report')
# three_way_report imported its own effects_report copy through sys.path;
# rebind both names to ONE module object so tests can mock a single target
report.effects = effects


def utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _median_time(row, key):
    values = [effects.finite(s[key]) for s in row['result']['repeats']]
    if min(values) <= 0:
        raise ValueError('nonpositive timing')
    return statistics.median(values)


def _position_map(blocks, role):
    '''{block index -> arm position of `role` in that block's order},
    from the schedule's per-cell Latin squares; verifies the square.'''
    letter = next(k for k, v in LETTER_TO_ROLE.items() if v == role)
    out = {}
    for b, order in enumerate(blocks):
        if sorted(order) != sorted(LETTER_TO_ROLE):
            raise ValueError(f'block {b} is not a four-arm permutation')
        out[b] = order.index(letter)
    if sorted(out.values()) != [0, 1, 2, 3]:
        raise ValueError('Latin square violated: positions not a '
                         'permutation across blocks')
    return out


def _per_method(complete, values_of, methods):
    out = {}
    for m in methods:
        vals = [values_of(c) for c in complete if c['xc'] == m]
        out[m] = statistics.median(vals) if vals else None
    return out


def _band_counts(complete, values_of, bands, methods):
    counts = {}
    for m in methods:
        lo, hi = bands[m]
        vals = [values_of(c) for c in complete if c['xc'] == m]
        if not vals:
            continue
        counts[m] = {
            'n': len(vals),
            'inside_predicted': sum(
                1 for v in vals if (lo is None or v >= lo)
                and (hi is None or v <= hi)),
            'min': min(vals), 'max': max(vals)}
    return counts


def gather(index_path, batch_id, schedule_path, prereg_path=None):
    '''All facts for the batch; raises SystemExit on contract breaks.'''
    with open(schedule_path, encoding='utf-8') as f:
        schedule = json.load(f)
    sched_sha = hashlib.sha256(
        open(schedule_path, 'rb').read()).hexdigest()
    tags = [c['tag'] for c in schedule['cells']]
    blocks_by_tag = {c['tag']: c.get('blocks', [])
                     for c in schedule['cells']}
    with open(index_path, encoding='utf-8') as f:
        idx = json.load(f)
    batch = next((b for b in idx.get('batches', [])
                  if b.get('id') == batch_id), None)
    if batch is None:
        raise SystemExit(f'batch {batch_id} not found in {index_path}')
    by_tag = {c['cell']['tag']: c for c in batch['cells']}
    missing = [t for t in tags if t not in by_tag]
    if missing:
        raise SystemExit('index does not pre-list the schedule '
                         f'denominator; missing tags: {missing[:5]}...')

    per_cell, failures = [], []
    planned, partial = [], []
    for tag in tags:
        cell = by_tag[tag]
        entry = {'tag': tag, 'xc': cell['cell']['xc'],
                 'basis': cell['cell']['basis'],
                 'complete': bool(cell.get('sources'))}
        sources = cell.get('sources') or {}
        if not sources:
            # planned but not yet run: NOT a failure, never dropped from
            # the denominator (prepared/queued/running/passed distinct)
            entry['status'] = 'planned'
            planned.append(tag)
            per_cell.append(entry)
            continue
        if set(sources) != set(report.ROLES):
            # some arms recorded but not all four: an honest gap
            entry['status'] = 'incomplete_partial'
            entry['roles_present'] = sorted(sources)
            partial.append(tag)
            per_cell.append(entry)
            continue
        try:
            rows = report.load_cell(cell, batch_id)
        except Exception as error:  # CellFailure / OSError / ValueError...
            failures.append({'tag': tag,
                             'reason': f'{type(error).__name__}: {error}'})
            entry['status'] = 'load_failed'
            per_cell.append(entry)
            continue
        gates = {role: effects.measure(rows[role], rows['strict_fp64'])
                 for role in report.ROLES}
        comparisons = {}
        for key, ref, cand in COMPARISONS:
            m = effects.measure(rows[cand], rows[ref])
            m['passed'] &= gates[ref]['passed'] and gates[cand]['passed']
            m['scf_speed'] = (_median_time(rows[ref], 't_scf_s')
                              / _median_time(rows[cand], 't_scf_s'))
            comparisons[key] = {k: m[k] for k in
                                ('de', 'dg', 'passed', 'speed', 'seconds',
                                 'ref_seconds', 'grad_speed', 'scf_speed')}
        role_block_times = {
            role: {r['comparison_block']: effects.finite(r['sample_s'])
                   for r in rows[role]['result']['repeats']}
            for role in report.ROLES}
        role_times = {role: list(t.values())
                      for role, t in role_block_times.items()}
        entry.update({
            'status': 'complete',
            'gates': {role: {'passed': gates[role]['passed'],
                            'de': gates[role]['de'],
                            'dg': gates[role]['dg']}
                      for role in report.ROLES},
            'comparisons': comparisons,
            'C_over_R': (statistics.median(role_times['optimized_auto'])
                         / statistics.median(role_times['strict_fp64'])),
            'role_block_times': role_block_times,
            'role_spread': {role: max(t) / min(t)
                            for role, t in role_times.items()},
            'o_fp32_entry_hits': rows['upstream']['result'].get(
                'fp32_entry_hits'),
            'strict_fp32_zero': not any(
                effects.finite(v) > 0 for v in
                rows['strict_fp64']['result']['fp32_entry_hits'].values()),
            'auto_fp32_nonzero': any(
                effects.finite(v) > 0 for v in
                rows['optimized_auto']['result']['fp32_entry_hits'].values()),
            'positions': {role: _position_map(blocks_by_tag[tag], role)
                          for role in report.ROLES},
        })
        per_cell.append(entry)

    facts = {
        'generated_at_utc': utc(),
        'batch_id': batch_id,
        'schedule_path': schedule_path,
        'schedule_sha256': sched_sha,
        'cells_total': len(tags),
        'cells_complete': sum(1 for c in per_cell
                              if c.get('status') == 'complete'),
        'cells_planned_not_run': planned,
        'cells_partial': partial,
        'cells_load_failed': failures,
        'per_cell': per_cell,
        'prediction_bands': PREDICTION_BANDS,
    }
    if prereg_path:
        with open(prereg_path, encoding='utf-8') as f:
            prereg = json.load(f)
        facts['preregistration'] = {
            'path': prereg_path,
            'sha256': hashlib.sha256(
                open(prereg_path, 'rb').read()).hexdigest(),
            'preregistration_sha256': prereg.get('preregistration_sha256'),
            'preregistered_at_utc': prereg.get('preregistered_at_utc'),
            'predictions': prereg.get('predictions'),
        }
    facts['aggregates'] = aggregate(per_cell, len(tags))
    return facts


def aggregate(per_cell, denominator):
    '''Method medians + mechanical band counts for P1-P6.'''
    complete = [c for c in per_cell if c.get('status') == 'complete']
    methods = sorted({c['xc'] for c in complete})
    agg = {'methods': methods, 'n_complete': len(complete)}

    # P1: O/R implementation agreement (never a precision error)
    or_de = [c['gates']['upstream']['de'] for c in complete]
    or_dg = [c['gates']['upstream']['dg'] for c in complete]
    agg['P1_O_vs_R'] = {
        'max_dE': max(or_de) if or_de else None,
        'max_dg': max(or_dg) if or_dg else None,
        'dE_classes': {'<=1e-9': sum(d <= 1e-9 for d in or_de),
                       '1e-9..1e-7': sum(1e-9 < d <= 1e-7 for d in or_de),
                       '1e-7..1e-6': sum(1e-7 < d <= 1e-6 for d in or_de),
                       '>1e-6 (gate fail)': sum(d > 1e-6 for d in or_de)},
        'dg_classes': {'<=1e-8': sum(d <= 1e-8 for d in or_dg),
                       '1e-8..1e-6': sum(1e-8 < d <= 1e-6 for d in or_dg),
                       '1e-6..3e-5': sum(1e-6 < d <= 3e-5 for d in or_dg),
                       '>3e-5 (gate fail)': sum(d > 3e-5 for d in or_dg)},
        'per_method_median_dE': _per_method(
            complete, lambda c: c['gates']['upstream']['de'], methods),
    }

    # P2: O/B per-method medians vs the predicted ranges
    ob = lambda c: c['comparisons']['O_over_B']['speed']  # noqa: E731
    ob_passed = (lambda c: c['comparisons']['O_over_B']['speed']
                 if c['comparisons']['O_over_B']['passed'] else None)
    agg['P2_O_over_B'] = {
        'medians': _per_method(complete, ob, methods),
        'passed_only_medians': {
            m: statistics.median(
                [v for v in (ob_passed(c) for c in complete
                             if c['xc'] == m) if v is not None])
            if any(c['xc'] == m and c['comparisons']['O_over_B']['passed']
                   for c in complete) else None for m in methods},
        'band_counts': _band_counts(
            complete, ob, PREDICTION_BANDS['P2_O_over_B'], methods),
    }
    # P3: B/C parity class + C/R medians
    bc = lambda c: c['comparisons']['B_over_C']['speed']  # noqa: E731
    agg['P3_B_over_C'] = {
        'medians': _per_method(complete, bc, methods),
        'C_over_R_medians': _per_method(
            complete, lambda c: c['C_over_R'], methods),
        'band_counts': _band_counts(
            complete, bc,
            {m: PREDICTION_BANDS['P3_B_over_C'] for m in methods}, methods),
    }
    # P4: O/C per NLC class
    oc = lambda c: c['comparisons']['O_over_C']['speed']  # noqa: E731
    agg['P4_O_over_C'] = {
        'medians': _per_method(complete, oc, methods),
        'band_counts': _band_counts(
            complete, oc,
            {m: (PREDICTION_BANDS['P4_O_over_C_nlc']
                 if NLC_CLASS.get(m) == 'nlc-vv10'
                 else PREDICTION_BANDS['P4_O_over_C_non_nlc'])
             for m in methods}, methods),
        'nlc_class': NLC_CLASS,
    }
    # P5: protocol validity -- within-role spread + Latin-square position
    # effects (median over cells of t(position)/cell-role-median)
    agg['P5_protocol'] = {
        'median_role_spread': {
            role: (statistics.median(
                [c['role_spread'][role] for c in complete])
                if complete else None) for role in report.ROLES},
        'position_effects': {},
        'position_effect_definition':
            'per role and arm position p (0-3 from the schedule Latin '
            'squares): median over cells of sample_s(position p) / '
            'median(sample_s of that role in that cell); ~1.0 across p '
            'means position effects are small relative to role effects',
    }
    for role in report.ROLES:
        per_pos = {p: [] for p in range(4)}
        for c in complete:
            times = c['role_block_times'][role]
            med = statistics.median(times.values())
            for block, t in times.items():
                per_pos[c['positions'][role][int(block)]].append(t / med)
        agg['P5_protocol']['position_effects'][role] = {
            p: (statistics.median(v) if v else None)
            for p, v in per_pos.items()}
    # P6: honesty gates
    agg['P6_honesty'] = {
        'strict_fp32_zero_cells': sum(
            c.get('strict_fp32_zero', False) for c in complete),
        'auto_fp32_nonzero_cells': sum(
            c.get('auto_fp32_nonzero', False) for c in complete),
        'o_fp32_empty_cells': sum(
            c.get('o_fp32_entry_hits') == {} for c in complete),
        'all_role_gates_passed_cells': sum(
            all(c['gates'][role]['passed'] for role in report.ROLES)
            for c in complete),
        'denominator': denominator,
        'failures_kept': [c['tag'] for c in per_cell
                          if c.get('status') == 'load_failed'],
    }
    return agg


def _fmt(value, spec='.3f'):
    return format(value, spec) if isinstance(value, (int, float)) else '—'


def render_md(facts, interim):
    agg = facts['aggregates']
    lines = [f"# {facts['batch_id']} adjudication facts "
             f"({facts['cells_complete']}/{facts['cells_total']} cells"
             + ('; INTERIM — not the final adjudication' if interim
                else '') + ')', '',
             f"generated {facts['generated_at_utc']}; schedule sha "
             f"{facts['schedule_sha256'][:16]}...; numbers computed with "
             'the deployed reporter load_cell/measure (daily-report '
             'parity); band counts are mechanical, HIT/MISS verdicts are '
             'written by the session, not by this tool.', '',
             '## per-method medians (complete cells)', '',
             '| method | O/B | B/C | O/C | C/R |', '|---|---:|---:|---:|---:|']
    for m in agg['methods']:
        lines.append(
            f"| {m} | {_fmt(agg['P2_O_over_B']['medians'].get(m))}x | "
            f"{_fmt(agg['P3_B_over_C']['medians'].get(m))}x | "
            f"{_fmt(agg['P4_O_over_C']['medians'].get(m))}x | "
            f"{_fmt(agg['P3_B_over_C']['C_over_R_medians'].get(m))}x |")
    p1 = agg['P1_O_vs_R']
    lines += ['', '## P1 O/R implementation agreement (not a precision error)',
              f"- max |dE| = {_fmt(p1['max_dE'], '.3e')} Ha; max|dg| = "
              f"{_fmt(p1['max_dg'], '.3e')} Eh/Bohr",
              f"- dE classes: {json.dumps(p1['dE_classes'])}",
              f"- dg classes: {json.dumps(p1['dg_classes'])}"]
    p5 = agg['P5_protocol']
    lines += ['', '## P5 protocol validity',
              f"- median within-role spread (max/min of the 4 formal "
              f"samples): {json.dumps(p5['median_role_spread'])}",
              f"- position effects: "
              f"{json.dumps(p5['position_effects'])}"]
    p6 = agg['P6_honesty']
    lines += ['', '## P6 honesty gates',
              f"- strict zero-FP32 cells: {p6['strict_fp32_zero_cells']}"
              f"/{agg['n_complete']}; auto nonzero-FP32: "
              f"{p6['auto_fp32_nonzero_cells']}/{agg['n_complete']}; "
              f"O fp32_entry_hits empty (honest absence): "
              f"{p6['o_fp32_empty_cells']}/{agg['n_complete']}",
              f"- all-role-gates-passed cells: "
              f"{p6['all_role_gates_passed_cells']}/{agg['n_complete']}; "
              f"denominator {p6['denominator']} never dropped"]
    if facts['cells_load_failed'] or facts['cells_partial']:
        lines += ['', '## load failures and partial cells '
                      '(kept in the denominator)', '']
        lines += [f"- {f['tag']}: LOAD FAILED — {f['reason']}"
                  for f in facts['cells_load_failed']]
        lines += [f"- {t}: PARTIAL — some arms recorded, others missing"
                  for t in facts['cells_partial']]
    if facts['cells_planned_not_run']:
        lines += ['', f"## planned, not yet run: "
                      f"{len(facts['cells_planned_not_run'])}/"
                      f"{facts['cells_total']} cells "
                      '(kept in the denominator; not failures)', '']
    return '\n'.join(lines) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_adjudicate',
        description='P1-P6 adjudication FACTS for a four-arm upstream '
                    'comparison batch, computed with the deployed '
                    'reporter code (daily-report number parity).')
    p.add_argument('--index', required=True)
    p.add_argument('--batch-id', required=True)
    p.add_argument('--schedule', required=True)
    p.add_argument('--prereg', default=None)
    p.add_argument('--out', required=True)
    p.add_argument('--md', default=None)
    p.add_argument('--allow-partial', action='store_true',
                   help='emit explicitly-labeled interim facts when '
                        'cells are still incomplete (default: require '
                        'every scheduled cell complete)')
    args = p.parse_args(argv)

    facts = gather(args.index, args.batch_id, args.schedule, args.prereg)
    interim = facts['cells_complete'] < facts['cells_total']
    if interim and not args.allow_partial:
        raise SystemExit(
            f"batch incomplete: {facts['cells_complete']}/"
            f"{facts['cells_total']} cells complete; rerun with "
            '--allow-partial for explicitly-labeled interim facts')
    facts['interim'] = interim
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(facts, f, indent=1)
    if args.md:
        with open(args.md, 'w', encoding='utf-8') as f:
            f.write(render_md(facts, interim))
    print(f"wrote {args.out} ({facts['cells_complete']}/"
          f"{facts['cells_total']} cells"
          + ('; INTERIM' if interim else '') + ')')
    return 0


if __name__ == '__main__':
    sys.exit(main())

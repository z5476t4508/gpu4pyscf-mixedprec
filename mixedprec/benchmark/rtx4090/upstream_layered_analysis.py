#!/usr/bin/env python3
"""Size-band and basis-layered tabulation of a four-arm upstream
comparison batch, on top of the adjudication FACTS json produced by
upstream_adjudicate.py.

Why this exists: the post-batch adjudication runbook (LC-205u/t) asks
the session to report, alongside per-method medians, (a) the wb97mv
size-band structure, (b) the size+method-layered basis structure, and
(c) the accuracy structure by basis (the epinephrine basis-scope
fact).  Those are tabulations of already-measured numbers, not new
measurements; this tool does exactly that tabulation so the interim
and final sessions report identical structures.

This tool is READ-ONLY with respect to the batch: it consumes the
adjudicate facts json plus the upstream-effects-index (for per-cell
nao from the strict_fp64 lane document) and writes only its own
report pair.  It emits no verdicts -- HIT/MISS conclusions belong to
the session.  Descriptive nao bands below are NOT preregistered; they
are fixed round cut points chosen once so interim and final runs of
this tool are directly comparable.

stdlib only; no GPU, no cupy import (nothing here may open a CUDA
context while the batch driver is live).
"""

import argparse
import datetime
import json
import os
import statistics
import sys

# fixed descriptive cut points (NOT preregistered; round numbers so
# interim and final runs share one definition)
NAO_BANDS = ((0, 100, 'nao<100'),
             (100, 250, '100<=nao<250'),
             (250, 400, '250<=nao<400'),
             (400, float('inf'), 'nao>=400'))
METHODS = ('b3lyp', 'm062x', 'r2scan3c', 'wb97mv')


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        '%Y-%m-%dT%H:%M:%SZ')


def _band(nao):
    for lo, hi, label in NAO_BANDS:
        if lo <= nao < hi:
            return label
    raise ValueError(f'nao {nao} outside all bands')


def _median(values):
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


def _load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _nao_by_tag(index_path, batch_id):
    """nao per cell tag from the strict_fp64 lane document (the
    accuracy anchor; identical geometry/basis across the four arms)."""
    idx = _load_json(index_path)
    batch = None
    for b in idx.get('batches', []):
        if b.get('id') == batch_id:
            batch = b
            break
    if batch is None:
        raise SystemExit(f'batch {batch_id!r} not found in {index_path}')
    out = {}
    for cell in batch.get('cells', []):
        sources = cell.get('sources') or {}
        rec = sources.get('strict_fp64')
        tag = (cell.get('cell') or {}).get('tag')
        if rec is None or not tag:
            continue
        doc = _load_json(rec['path'])
        rows = doc.get('rows') or []
        if not rows:
            raise SystemExit(f'strict_fp64 doc for {tag} has no rows')
        nao = (rows[0].get('system') or {}).get('nao')
        if not isinstance(nao, (int, float)):
            raise SystemExit(f'no nao in strict_fp64 doc for {tag}')
        out[tag] = int(nao)
    return out


def _ratio(c, key):
    comp = c.get('comparisons') or {}
    entry = comp.get(key) or {}
    sp = entry.get('speed')
    if not isinstance(sp, (int, float)):
        # C_over_R is a top-level scalar on the adjudicate per-cell
        # record (a single measured ratio, not a speed/de/dg bundle)
        sp = c.get('C_over_R') if key == 'C_over_R' else None
    return float(sp) if isinstance(sp, (int, float)) else None


def _scf_ratio(c):
    entry = (c.get('comparisons') or {}).get('O_over_B') or {}
    sp = entry.get('scf_speed')
    return float(sp) if isinstance(sp, (int, float)) else None


def layer_facts(facts, nao_map):
    """All layered tables.  Denominator honesty: every complete cell
    appears in exactly one (basis, method) layer and one size band;
    gate-fail cells stay in their layers and are ALSO listed by name.
    Planned/incomplete cells are counted, never dropped silently."""
    per_cell = facts.get('per_cell') or []
    complete = [c for c in per_cell if c.get('complete')]
    nao_missing = [c['tag'] for c in complete if c['tag'] not in nao_map]
    if nao_missing:
        raise SystemExit('nao unavailable for complete cells: '
                         + ', '.join(nao_missing[:5])
                         + f' ({len(nao_missing)} total) -- fail closed')

    # Table A: per-basis x method medians of the four reported ratios
    table_a = {}
    for c in complete:
        key = (c['basis'], c['xc'])
        table_a.setdefault(key, []).append(c)

    basis_methods = {}
    for (basis, xc), cells in sorted(table_a.items()):
        basis_methods[f'{basis}|{xc}'] = {
            'n': len(cells),
            'OB': _median([_ratio(c, 'O_over_B') for c in cells]),
            'BC': _median([_ratio(c, 'B_over_C') for c in cells]),
            'OC': _median([_ratio(c, 'O_over_C') for c in cells]),
            'CR': _median([_ratio(c, 'C_over_R') for c in cells]),
        }

    # Table B: per-method x nao-band medians of O/B (total and scf)
    table_b = {}
    for c in complete:
        table_b.setdefault((c['xc'], _band(nao_map[c['tag']])), []).append(c)
    band_methods = {}
    for (xc, band), cells in sorted(table_b.items()):
        band_methods[f'{xc}|{band}'] = {
            'n': len(cells),
            'OB': _median([_ratio(c, 'O_over_B') for c in cells]),
            'OB_scf': _median([_scf_ratio(c) for c in cells]),
        }

    # Table C: wb97mv per-cell size scatter (the VV10/NLC wall band)
    scatter = []
    for c in complete:
        if c['xc'] != 'wb97mv':
            continue
        scatter.append({'tag': c['tag'], 'nao': nao_map[c['tag']],
                        'basis': c['basis'],
                        'OB': _ratio(c, 'O_over_B'),
                        'OB_scf': _scf_ratio(c)})
    scatter.sort(key=lambda r: (r['nao'], r['tag']))

    # Table D: accuracy structure by basis (B vs R) -- gate fails
    # listed by name, never dropped from their layer
    acc = {}
    gate_fails = []
    for c in complete:
        g = (c.get('gates') or {}).get('optimized_fp64') or {}
        acc.setdefault(c['basis'], []).append(
            {'tag': c['tag'], 'de': g.get('de'), 'dg': g.get('dg'),
             'passed': bool(g.get('passed'))})
        if g.get('passed') is False:
            gate_fails.append(c['tag'])
    accuracy_by_basis = {}
    for basis, rows in sorted(acc.items()):
        accuracy_by_basis[basis] = {
            'n': len(rows),
            'max_de': max(r['de'] for r in rows if r['de'] is not None),
            'max_dg': max(r['dg'] for r in rows if r['dg'] is not None),
            'gate_fail_tags': [r['tag'] for r in rows if not r['passed']],
        }

    return {
        'batch_id': facts.get('batch_id'),
        'interim': bool(facts.get('interim')),
        'cells_complete': facts.get('cells_complete'),
        'cells_total': facts.get('cells_total'),
        'facts_generated_at_utc': facts.get('generated_at_utc'),
        'nao_bands': [b[2] for b in NAO_BANDS],
        'nao_bands_note': 'descriptive fixed cut points, NOT preregistered',
        'basis_method_medians': basis_methods,
        'method_band_medians': band_methods,
        'wb97mv_size_scatter': scatter,
        'accuracy_by_basis': accuracy_by_basis,
        'gate_fail_tags': sorted(gate_fails),
    }


def render_md(layered):
    interim = layered['interim']
    head = ('INTERIM -- not the final tabulation' if interim
            else 'FINAL tabulation of the complete batch')
    lines = [
        f"# uc layer analysis -- batch {layered['batch_id']} "
        f"({layered['cells_complete']}/{layered['cells_total']} cells; {head})",
        '',
        f"facts generated {layered['facts_generated_at_utc']}; "
        "descriptive nao bands "
        f"({', '.join(layered['nao_bands'])}) are fixed cut points, "
        "NOT preregistered; this tool tabulates measured numbers only, "
        "session writes verdicts.",
        '',
        '## per-basis x method medians (complete cells; n kept)',
        '',
        '| basis | method | n | O/B | B/C | O/C | C/R |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for key, m in layered['basis_method_medians'].items():
        basis, xc = key.split('|')
        def fmt(v):
            return f'{v:.3f}x' if isinstance(v, float) else '--'
        lines.append(f"| {basis} | {xc} | {m['n']} | {fmt(m['OB'])} | "
                     f"{fmt(m['BC'])} | {fmt(m['OC'])} | {fmt(m['CR'])} |")
    lines += [
        '',
        '## O/B by method x nao band (complete cells)',
        '',
        '| method | nao band | n | O/B | O/B scf |',
        '|---|---|---:|---:|---:|',
    ]
    for key, m in layered['method_band_medians'].items():
        xc, band = key.split('|', 1)
        def fmt(v):
            return f'{v:.3f}x' if isinstance(v, float) else '--'
        lines.append(f"| {xc} | {band} | {m['n']} | {fmt(m['OB'])} | "
                     f"{fmt(m['OB_scf'])} |")
    lines += [
        '',
        '## wb97mv size scatter (per cell, sorted by nao)',
        '',
        '| nao | basis | O/B | O/B scf | tag |',
        '|---:|---|---:|---:|---|',
    ]
    for r in layered['wb97mv_size_scatter']:
        lines.append(f"| {r['nao']} | {r['basis']} | {r['OB']:.3f}x | "
                     f"{r['OB_scf']:.3f}x | {r['tag']} |")
    lines += [
        '',
        '## accuracy structure by basis (B vs R; gate fails listed, '
        'never dropped)',
        '',
        '| basis | n | max |dE| Ha | max |dg| Eh/Bohr | gate fails |',
        '|---|---:|---:|---:|---|',
    ]
    for basis, a in layered['accuracy_by_basis'].items():
        fails = ', '.join(a['gate_fail_tags']) if a['gate_fail_tags'] else '--'
        lines.append(f"| {basis} | {a['n']} | {a['max_de']:.3e} | "
                     f"{a['max_dg']:.3e} | {fails} |")
    planned = layered['cells_total'] - layered['cells_complete']
    lines += [
        '',
        f"gate-fail cells: {len(layered['gate_fail_tags'])} "
        f"({', '.join(layered['gate_fail_tags']) or 'none'}); "
        f"planned not yet run: {planned}/{layered['cells_total']} "
        "(kept in the denominator; not failures)",
        '',
    ]
    return '\n'.join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='upstream_layered_analysis',
        description='Size-band and basis-layered tabulation of a '
                    'four-arm upstream comparison batch from its '
                    'adjudication facts json (no verdicts).')
    p.add_argument('--facts', required=True,
                   help='adjudication facts json from upstream_adjudicate.py')
    p.add_argument('--index', required=True,
                   help='upstream-effects-index.json (for per-cell nao)')
    p.add_argument('--batch-id', required=True)
    p.add_argument('--out', required=True, help='JSON report path')
    p.add_argument('--md', default=None, help='Markdown report path')
    args = p.parse_args(argv)

    facts = _load_json(args.facts)
    nao_map = _nao_by_tag(args.index, args.batch_id)
    layered = layer_facts(facts, nao_map)
    layered['tool'] = 'upstream_layered_analysis'
    layered['generated_at_utc'] = utc()
    layered['facts_path'] = os.path.abspath(args.facts)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(layered, f, indent=1, sort_keys=True)
        f.write('\n')
    if args.md:
        with open(args.md, 'w', encoding='utf-8') as f:
            f.write(render_md(layered))
    print(f"wrote {args.out} ({layered['cells_complete']}/"
          f"{layered['cells_total']} cells"
          + ('; INTERIM' if layered['interim'] else '; FINAL') + ')')
    return 0


if __name__ == '__main__':
    sys.exit(main())

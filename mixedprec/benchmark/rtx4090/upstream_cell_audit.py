#!/usr/bin/env python3
'''Per-cell deployed-contract + honesty audit for a four-arm upstream
comparison batch (control card 2026-10-18).

For every COMPLETE cell of the batch index this tool re-runs, read-only:

  1. source integrity: every role document is loaded through the
     deployed reporter's load_source (sha256 vs the index entry,
     evidence-root containment, schema rtx4090-gradient-v1);
  2. the deployed comparison contract: three_way_report.load_cell --
     roles / batch id / timing_protocol / schedule sha256 / identical
     dependencies / O pinned to the official certificate and package
     root / B, C, R on one snapshot root distinct from O / precision
     modes / optimization envs / strict-R zero FP32 entry / auto
     FP32-entry evidence;
  3. honesty checks on top of the deployed contract: every formal
     repeat converged, mem.within_budget, no errors, no fallback
     events, the official lane's honest probe absence (empty
     fp32_entry_hits), strict-R precision_switch switched=False, the
     official lane carries no invented source_commit, and optionally
     the B/C/R source_commit pinned to the frozen batch snapshot;
  4. the accuracy gates via effects_report.measure for B-vs-R and
     C-vs-R (|dE| <= 1e-6 Ha, max|dg| <= 3e-5 Eh/Bohr, all repeats and
     references converged) -- an honest FAIL is reported as a finding,
     never dropped from the denominator and never reframed;
  5. O-vs-R implementation agreement (cross-repeat max |dE| / |dg|),
     reported SEPARATELY and never as a precision error.

The tool writes no verdict and changes nothing: gate-fail rows stay in
the report and in the denominator; scientific conclusions belong to the
session.  Read-only on the index, the documents and the schedule; it
writes only its own report pair.

Exit codes: 0 all audited cells clean; 3 findings (contract or gate
FAIL rows -- the audit itself ran to completion); 2 usage/structural.
'''

import argparse
import importlib.util
import json
import os
import statistics
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
REPORTING = os.path.join(REPO, 'tools', 'reporting')

ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')
SNAPSHOT_ROLES = ('optimized_fp64', 'optimized_auto', 'strict_fp64')


def _load_reporting(name):
    if REPORTING not in sys.path:
        sys.path.insert(0, REPORTING)
    spec = importlib.util.spec_from_file_location(
        f'cell_audit_{name}', os.path.join(REPORTING, f'{name}.py'))
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


def _median(row, key):
    values = [effects.finite(s[key]) for s in row['result']['repeats']]
    return statistics.median(values) if values else None


def _agreement(cand_row, ref_row):
    '''Cross-repeat max |dE| / max|dg|.  Implementation-agreement class
    (O vs R): reported separately, never a precision gate.'''
    if (cand_row['system'] != ref_row['system']
            or cand_row['method'] != ref_row['method']):
        raise ValueError('unmatched geometry or method settings')
    de = dg = 0.
    for a in cand_row['result']['repeats']:
        for b in ref_row['result']['repeats']:
            de = max(de, abs(effects.finite(a['e_tot'])
                            - effects.finite(b['e_tot'])))
            ga, gb = effects.gradient(a), effects.gradient(b)
            if len(ga) != len(gb):
                raise ValueError('unmatched gradient shape')
            dg = max(dg, max(abs(x - y) for x, y in zip(ga, gb)))
    return {'de': de, 'dg': dg}


def _honesty_problems(rows):
    problems = []
    for role, row in rows.items():
        res = row['result']
        if any(r.get('converged') is not True for r in res['repeats']):
            problems.append(f'{role}: a formal repeat did not converge')
        if res.get('mem', {}).get('within_budget') is not True:
            problems.append(f'{role}: mem not within budget')
        if res.get('errors'):
            problems.append(f'{role}: errors recorded')
        if res.get('fallback_events'):
            problems.append(f'{role}: fallback events recorded')
    if rows['upstream']['result'].get('fp32_entry_hits') != {}:
        problems.append('upstream: probe hits present (official has none)')
    if (rows['strict_fp64']['result'].get('precision_switch', {})
            .get('switched')) is not False:
        problems.append('strict_fp64: precision_switch not switched=False')
    return problems


def _commit_problems(docs, expect_commit):
    '''Snapshot pinning: B/C/R must carry the frozen batch source commit
    (when expected) and the official wheel must carry none (never a
    fabricated git SHA).'''
    problems = []
    for role in SNAPSHOT_ROLES:
        observed = docs[role]['meta'].get('source_commit')
        if expect_commit is not None and observed != expect_commit:
            problems.append(f'{role}: source_commit {observed!r} != frozen '
                            f'snapshot {expect_commit!r}')
    if docs['upstream']['meta'].get('source_commit') is not None:
        problems.append('upstream: fabricated source_commit on the wheel')
    return problems


def _gate_numbers(cand_row, ref_row):
    m = effects.measure(cand_row, ref_row)
    return {'de': m['de'], 'dg': m['dg'], 'passed': m['passed'],
            'n': m['n'], 'fallbacks': m['fallbacks']}


def audit_cell(entry, batch_id, expect_commit=None):
    '''One complete cell -> the audit record.  Never raises for
    scientific or provenance findings; they are returned honestly.
    Raises only on structural surprises that make the record unusable
    (e.g. a role source missing from a "complete" entry).'''
    tag = entry['cell']['tag']
    rec = {'tag': tag,
           'xc': entry['cell'].get('xc'),
           'basis': entry['cell'].get('basis')}
    sources = entry.get('sources') or {}
    missing = [r for r in ROLES if r not in sources]
    if missing:
        rec['status'] = 'error'
        rec['fail_reasons'] = [f'incomplete sources: {missing}']
        return rec
    docs = {role: effects.load_source(sources[role]) for role in ROLES}
    fail_reasons = []
    try:
        rows = report.load_cell(entry, batch_id)
    except Exception as exc:  # CellFailure / ValueError from the contract
        rec['status'] = 'fail'
        rec['fail_reasons'] = [f'deployed contract: {exc}']
        rec['category'] = 'deployed_contract'
        return rec
    honesty = _honesty_problems(rows)
    commits = _commit_problems(docs, expect_commit)
    if honesty or commits:
        fail_reasons.extend(honesty + commits)
    gates = {'B_vs_R': _gate_numbers(rows['optimized_fp64'],
                                     rows['strict_fp64']),
             'C_vs_R': _gate_numbers(rows['optimized_auto'],
                                     rows['strict_fp64'])}
    for pair in ('B_vs_R', 'C_vs_R'):
        if not gates[pair]['passed']:
            fail_reasons.append(
                f'{pair} accuracy gate FAIL: de={gates[pair]["de"]:.3e} '
                f'dg={gates[pair]["dg"]:.3e}')
    agreement = _agreement(rows['upstream'], rows['strict_fp64'])
    rec.update({
        'status': 'fail' if fail_reasons else 'pass',
        'fail_reasons': fail_reasons,
        'gates': gates,
        'implementation_agreement_O_vs_R': agreement,
        'cycles': {role: [r.get('scf_cycles')
                           for r in rows[role]['result']['repeats']]
                   for role in ROLES},
        'median_sample_s': {role: _median(rows[role], 'sample_s')
                            for role in ROLES},
        'source_commit': {role: docs[role]['meta'].get('source_commit')
                          for role in ROLES},
    })
    if fail_reasons and not honesty and not commits:
        rec['category'] = 'accuracy_gate'
    elif fail_reasons:
        rec['category'] = 'contract_or_honesty'
    return rec


def audit_batch(index_path, batch_id, only_tags=None, expect_commit=None):
    with open(index_path, encoding='utf-8') as f:
        idx = json.load(f)
    batch = next((b for b in idx.get('batches', [])
                  if b.get('id') == batch_id), None)
    if batch is None:
        raise SystemExit(f'batch {batch_id} not found in {index_path}')
    cells = batch.get('cells', [])
    out = []
    for entry in cells:
        tag = entry['cell']['tag']
        if only_tags and tag not in only_tags:
            continue
        nsrc = len(entry.get('sources') or {})
        if nsrc == 0:
            out.append({'tag': tag, 'status': 'planned'})
        elif nsrc < len(ROLES):
            out.append({'tag': tag, 'status': 'partial',
                        'roles_present': sorted(entry['sources'])})
        else:
            out.append(audit_cell(entry, batch_id, expect_commit))
    return out


def _summarize(records):
    audited = [r for r in records if r['status'] in ('pass', 'fail')]
    failed = [r for r in records if r['status'] == 'fail']
    worst_gate = None
    worst_agree = None
    for r in audited:
        for pair in ('B_vs_R', 'C_vs_R'):
            g = r.get('gates', {}).get(pair)
            if g and (worst_gate is None or g['de'] > worst_gate['de']):
                worst_gate = {'tag': r['tag'], 'pair': pair,
                              'de': g['de'], 'dg': g['dg'],
                              'passed': g['passed']}
        a = r.get('implementation_agreement_O_vs_R')
        if a and (worst_agree is None or a['de'] > worst_agree['de']):
            worst_agree = {'tag': r['tag'], 'de': a['de'], 'dg': a['dg']}
    return {'denominator': len(records),
            'planned': sum(r['status'] == 'planned' for r in records),
            'partial': sum(r['status'] == 'partial' for r in records),
            'audited': len(audited),
            'pass': sum(r['status'] == 'pass' for r in audited),
            'fail': len(failed),
            'fail_categories': {
                k: sum(r.get('category') == k for r in failed)
                for k in ('deployed_contract', 'contract_or_honesty',
                          'accuracy_gate')},
            'worst_accuracy_gate': worst_gate,
            'worst_O_vs_R_agreement': worst_agree}


def _markdown(summary, records, batch_id):
    lines = [
        f'# Upstream cell audit -- batch {batch_id}',
        '',
        f'Generated {utc()} (true UTC).  Read-only audit through the '
        'deployed reporter copies in tools/reporting/.',
        '',
        f'- denominator {summary["denominator"]} cells: '
        f'{summary["audited"]} audited / {summary["partial"]} partial / '
        f'{summary["planned"]} planned (never dropped)',
        f'- pass {summary["pass"]} / FAIL {summary["fail"]}'
        f' (categories: {summary["fail_categories"]})',
    ]
    wg, wa = summary['worst_accuracy_gate'], summary['worst_O_vs_R_agreement']
    if wg:
        lines.append(f'- worst accuracy gate: {wg["tag"]} {wg["pair"]} '
                     f'dE={wg["de"]:.3e} Ha dg={wg["dg"]:.3e} '
                     f'({"pass" if wg["passed"] else "FAIL"})')
    if wa:
        lines.append(f'- worst O/R implementation agreement: {wa["tag"]} '
                     f'dE={wa["de"]:.3e} Ha dg={wa["dg"]:.3e} '
                     '(agreement class, never a precision gate)')
    failed = [r for r in records if r['status'] == 'fail']
    if failed:
        lines += ['', '## Honest FAIL rows (kept in the denominator)', '']
        for r in failed:
            lines.append(f'- {r["tag"]}: ' + '; '.join(r['fail_reasons']))
    lines.append('')
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--index', required=True,
                    help='upstream-effects-index.json')
    ap.add_argument('--batch-id', required=True)
    ap.add_argument('--cells', default=None,
                    help='comma-separated tags to audit (default: all)')
    ap.add_argument('--expect-snapshot-commit', default=None,
                    help='the frozen batch source commit B/C/R must carry')
    ap.add_argument('--out', default=None, help='JSON report path')
    ap.add_argument('--md', default=None, help='Markdown report path')
    args = ap.parse_args(argv)

    only = set(args.cells.split(',')) if args.cells else None
    records = audit_batch(args.index, args.batch_id, only,
                          args.expect_snapshot_commit)
    summary = _summarize(records)
    payload = {'tool': 'upstream_cell_audit',
               'generated_at_utc': utc(),
               'index': os.path.abspath(args.index),
               'batch_id': args.batch_id,
               'expect_snapshot_commit': args.expect_snapshot_commit,
               'summary': summary, 'cells': records}
    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=1, sort_keys=True)
            f.write('\n')
    if args.md:
        with open(args.md, 'w', encoding='utf-8') as f:
            f.write(_markdown(summary, records, args.batch_id))
    findings = summary['fail']
    print(json.dumps(summary, indent=1, sort_keys=True))
    print(f'audit complete: {summary["audited"]} audited, '
          f'{summary["pass"]} pass, {findings} honest FAIL rows')
    return 3 if findings else 0


if __name__ == '__main__':
    sys.exit(main())

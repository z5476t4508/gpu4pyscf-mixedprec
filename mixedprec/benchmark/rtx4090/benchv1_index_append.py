#!/usr/bin/env python3
"""Benchmark-v1 post-batch index append + daily-comparison figures
(card benchmark-resume-20261030 item 8; SP-32 dual-index recipe adapted).

CPU-only, stdlib-only.  Runs AFTER the batch driver exits, for either
batch:

  --mode obcr (default)  the 64-case O/B/C/R first evaluation
  --mode s               the separate 64-case auto-strict S/B batch

  1. verifies the driver-maintained upstream-effects-index batch entry
     is COMPLETE (all planned cells, every role pin -- four roles for
     obcr, auto_strict + optimized_fp64 for s -- every pinned file
     still hashes to its recorded sha256) -- read-only, never rewrites
     what the driver wrote;
  2. verifies the stats document (benchmark_v1_stats.py --mode obcr or
     --mode s) belongs to the same planned denominator and extracts
     the daily figures (O->B / B->C / O->C comparisons for obcr; the
     single B_to_S certified-subset ratio for s);
  3. emits the machine-readable daily-comparisons JSON (figures +
     counts + source/date; obcr keeps the legacy SP32 policy medians
     clearly historical; s is its own schema and NEVER mixes into the
     O/B/C/R comparisons);
  4. appends ONE dataset entry per mode to effects-index.json pinning
     the schedule, driver log, stats, daily figures and every merged
     cell document.

Fail-closed invariants (each has a CPU test):

  * refuses while any batch driver process is alive (a real append is
    a post-batch act; --dry-run is the read-only pre-flight);
  * refuses unless the upstream batch entry has exactly the planned
    cell count with every role pin intact -- a missing or mutated cell
    means the batch is not complete, and no partial cohort is ever
    relabeled a completed evaluation;
  * refuses on stats schema mismatch or a planned-count mismatch with
    the schedule denominator;
  * idempotent: a second append of the same dataset id refuses;
  * every write is backup -> temp-file + validate + os.replace, and
    the re-read file is re-verified hash-by-hash after the append.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

UP_INDEX_DEFAULT = ('/data/work/gpu4pyscf-agent/state/long-chain/'
                    'upstream-effects-index.json')
DS_INDEX_DEFAULT = ('/data/work/gpu4pyscf-agent/state/long-chain/'
                    'effects-index.json')
DRIVER_MARKER = 'run_upstream_batch.py'
STATS_SCHEMA = 'benchmark-v1-stats-v1'
S_STATS_SCHEMA = 'benchmark-v1-s-stats-v1'
DAILY_SCHEMA = 'benchv1-daily-comparisons-v1'
S_DAILY_SCHEMA = 'benchv1-daily-comparisons-s-v1'
ROLES = ('upstream', 'optimized_fp64', 'optimized_auto', 'strict_fp64')
S_ROLES = ('auto_strict', 'optimized_fp64')
COMPARISON_LABELS = ('O_to_B', 'B_to_C', 'O_to_C')
MODE_DEFAULTS = {
    'obcr': {'schedule': 'schedule-v2.json',
             'batch_id': 'benchv1-20261030-obcr',
             'dataset_id': 'benchv1-20261030-obcr-eval',
             'daily_out': 'benchv1-daily-comparisons.json',
             'batch_dir': 'batch-obcr'},
    's': {'schedule': 'schedule-s.json',
          'batch_id': 'benchv1-20261030-obcr-S',
          'dataset_id': 'benchv1-20261030-s-eval',
          'daily_out': 'benchv1-daily-comparisons-s.json',
          'batch_dir': 'batch-s'},
}
# The SP32 11-cell full-policy medians the card names as LEGACY context;
# quoted for continuity only, never as new acceptance or as bench-v1 results.
LEGACY_POLICY_FIGURES = {'O_to_B': 1.785, 'B_to_C': 1.010, 'O_to_C': 1.819}


def fail(msg):
    raise SystemExit(f'FAIL-CLOSED: {msg}')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utcnow():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_write_json(path, obj):
    tmp = path + '.tmp-bv1append'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    json.load(open(tmp, encoding='utf-8'))          # validate before replace
    os.replace(tmp, path)


def guard_no_driver(proc_root='/proc'):
    """Refuse a real append while any batch driver process is alive."""
    me = os.getpid()
    for pid in sorted(os.listdir(proc_root)):
        if not pid.isdigit() or int(pid) == me:
            continue
        try:
            with open(os.path.join(proc_root, pid, 'cmdline'),
                      'rb') as f:
                cmd = f.read().decode('utf-8', 'replace').split('\0')
        except OSError:
            continue
        if any(DRIVER_MARKER in part for part in cmd):
            fail(f'batch driver alive (pid {pid}); the append is a '
                 'post-batch act -- rerun after the driver exits '
                 '(--dry-run is the read-only pre-flight)')


def verify_upstream_batch(up_index_path, batch_id, require_cells,
                          roles=ROLES):
    '''Read-only completeness + hash-pin verification of the driver's
    batch entry.  Returns a summary dict; refuses on any gap.'''
    if not os.path.isfile(up_index_path):
        fail(f'upstream index not found: {up_index_path}')
    with open(up_index_path, encoding='utf-8') as f:
        index = json.load(f)
    matches = [b for b in index.get('batches', [])
               if b.get('id') == batch_id]
    if not matches:
        fail(f'batch {batch_id!r} not present in the upstream index -- '
             'the driver creates it; nothing to append from')
    if len(matches) > 1:
        fail(f'batch {batch_id!r} appears {len(matches)} times in the '
             'upstream index')
    cells = matches[0].get('cells', [])
    if len(cells) != require_cells:
        fail(f'batch {batch_id!r} has {len(cells)} cells, planned '
             f'{require_cells} -- the batch is NOT complete; a partial '
             'cohort is never relabeled a completed evaluation')
    tags = []
    for c in cells:
        tag = c.get('cell', {}).get('tag')
        srcs = c.get('sources', {})
        if not srcs:
            fail(f'cell {tag!r} has no role pins -- not merged yet '
                 '(planned-denominator placeholder); the batch is NOT '
                 'complete')
        for role in roles:
            pin = srcs.get(role)
            if not pin or 'path' not in pin or 'sha256' not in pin:
                fail(f'cell {tag!r} lacks a {role} pin in the upstream '
                     'index')
            if not os.path.isfile(pin['path']):
                fail(f'cell {tag!r} {role} pinned file missing: '
                     f'{pin["path"]}')
            if sha256_file(pin['path']) != pin['sha256']:
                fail(f'cell {tag!r} {role} pin hash mismatch: '
                     f'{pin["path"]}')
        tags.append(tag)
    return {'cells': len(cells), 'role_pins': len(cells) * len(roles),
            'tags': tags}


def load_and_check_stats(stats_path, require_cells, mode='obcr'):
    '''Validate the stats document against the planned denominator and
    extract the daily figures.  Returns (stats, figures).'''
    want_schema = STATS_SCHEMA if mode == 'obcr' else S_STATS_SCHEMA
    stats_prog = f'benchmark_v1_stats.py --mode {mode}'
    if not os.path.isfile(stats_path):
        fail(f'stats document not found: {stats_path} -- run '
             f'{stats_prog} first (from the LIVE repo, not the frozen '
             'snapshot)')
    with open(stats_path, encoding='utf-8') as f:
        stats = json.load(f)
    if stats.get('schema') != want_schema:
        fail(f'stats schema {stats.get("schema")!r} != {want_schema!r} '
             f'-- wrong mode or forged document')
    planned = stats.get('counts', {}).get('planned')
    if planned != require_cells:
        fail(f'stats planned count {planned!r} != schedule denominator '
             f'{require_cells} -- the stats document belongs to a '
             'different cohort')
    if mode == 'obcr':
        figures = {}
        for label in COMPARISON_LABELS:
            e = stats.get('comparisons', {}).get(label) or {}
            figures[label] = {k: e.get(k) for k in
                              ('n', 'primary_geomean', 'median_speedup',
                               'total_time_ratio', 'ci95_bootstrap')}
    else:
        e = stats.get('b_to_s') or {}
        figures = {'B_to_S': {k: e.get(k) for k in
                              ('n', 'primary_geomean', 'median_ratio',
                               'total_time_ratio')}}
    return stats, figures


def build_daily(stats, figures, stats_tool_commit=None,
                generated_at=None):
    counts = stats['counts']
    daily = {
        'schema': DAILY_SCHEMA,
        'generated_at_utc': generated_at or utcnow(),
        'arms_source_commit': stats['protocol'].get('source_commit'),
        'schedule_sha256': stats['protocol'].get('schedule_sha256'),
        'stats_generated_at_utc': stats.get('generated_at_utc'),
        'stats_tool_commit': stats_tool_commit,
        'counts': counts,
        'full_cohort_available': stats.get('full_cohort_available'),
        'comparisons': figures,
        'legacy_reference': {
            'policy_figures': LEGACY_POLICY_FIGURES,
            'note': ('SP32 11-cell full-policy medians at 9bf27a3 '
                     '(historical context, NOT new acceptance and NOT '
                     'bench-v1 results)'),
        },
        'auto_strict': ('kept separate: the S route reports its own '
                        'verification/fallback costs in its own batch'),
    }
    return daily


def daily_line(daily):
    c = daily['counts']
    parts = [f"planned {c['planned']}, pass {c['gate_pass']}, "
             f"fail/missing {c['gate_fail_or_missing']} "
             f"(SCF-FP32 C {c['actual_scf_fp32_C']}, gated C "
             f"{c['gated_fallback_C']})"]
    for label in COMPARISON_LABELS:
        f = daily['comparisons'][label]
        if f.get('primary_geomean') is None:
            parts.append(f'{label}: undefined (no complete pairs)')
        else:
            ci = f.get('ci95_bootstrap')
            ci_s = (f' [{ci[0]:.4f}, {ci[1]:.4f}]'
                    if ci else ' [no CI]')
            parts.append(f"{label}: geomean {f['primary_geomean']:.4f}"
                         f" (n={f.get('n')}){ci_s}")
    parts.append('full-cohort headline '
                 + ('available' if daily['full_cohort_available']
                    else 'UNAVAILABLE'))
    return '; '.join(parts)


def build_label(stats, figures, up_summary):
    p = stats['protocol']
    c = stats['counts']
    seg = []
    for label in COMPARISON_LABELS:
        f = figures[label]
        seg.append(f'{f["primary_geomean"]:.4f}'
                   if f.get('primary_geomean') is not None else 'undef')
    label = (
        f'BENCH-V1 first 64-case O/B/C/R evaluation at arms source '
        f'{p.get("source_commit")} (schedule '
        f'{str(p.get("schedule_sha256"))[:8]}, '
        f'{p.get("blocks_per_role_per_case")} blocks/role/case, '
        f'bootstrap seed {p.get("bootstrap", {}).get("seed")}): planned '
        f'{c["planned"]}, gate-pass {c["gate_pass"]}, fail/missing '
        f'{c["gate_fail_or_missing"]}, SCF-FP32 C '
        f'{c["actual_scf_fp32_C"]}, gated-FP64 C '
        f'{c["gated_fallback_C"]}; geomean O->B {seg[0]} / B->C '
        f'{seg[1]} / O->C {seg[2]}; upstream-index verification '
        f'{up_summary["cells"]} cells / {up_summary["role_pins"]} role '
        f'pins all hash-verified; full-cohort headline '
        f'{"available" if stats.get("full_cohort_available") else "UNAVAILABLE"}; '
        'auto-strict S kept separate; legacy SP32 policy medians '
        '1.785/1.010/1.819 are historical, not new acceptance'
    )
    return label


def build_daily_s(stats, figures, stats_tool_commit=None,
                  generated_at=None):
    '''The S-route daily document: its OWN schema, never mixed into the
    O/B/C/R daily comparisons.  No performance claim is derived here
    and S never borrows C speedups.'''
    counts = {k: stats['counts'].get(k) for k in
              ('planned', 'certified_ok', 'failed_or_missing')}
    return {
        'schema': S_DAILY_SCHEMA,
        'generated_at_utc': generated_at or utcnow(),
        'arms_source_commit': stats['protocol'].get('source_commit'),
        'schedule_sha256': stats['protocol'].get('schedule_sha256'),
        'stats_generated_at_utc': stats.get('generated_at_utc'),
        'stats_tool_commit': stats_tool_commit,
        'counts': counts,
        'b_to_s': figures['B_to_S'],
        'note': ('S 成本含其自身验证/匹配/回退开销，绝不借用 C 加速；'
                 '比值<1 表示认证路线比纯优化 FP64 更贵（诚实方向）；'
                 '仅证书齐全的个案进入 B_to_S 子集，缺失/未认证个案'
                 '保持可见，绝不以子集替代全集'),
    }


def daily_line_s(daily):
    c = daily['counts']
    f = daily['b_to_s']
    if f.get('primary_geomean') is None:
        ratio = 'B_to_S: undefined (no certified pairs)'
    else:
        ratio = (f"B_to_S: geomean {f['primary_geomean']:.4f} "
                 f"(n={f.get('n')})")
    return (f"planned {c['planned']}, certified_ok "
            f"{c['certified_ok']}, fail/missing "
            f"{c['failed_or_missing']}; {ratio}")


def build_label_s(stats, figures, up_summary):
    p = stats['protocol']
    c = stats['counts']
    f = figures['B_to_S']
    gm = f.get('primary_geomean')
    seg = f'{gm:.4f}' if gm is not None else 'undef'
    md = f.get('median_ratio')
    md_s = f'{md:.4f}' if md is not None else 'undef'
    tt = f.get('total_time_ratio')
    tt_s = f'{tt:.4f}' if tt is not None else 'undef'
    return (
        f'BENCH-V1 S (auto-strict) 64-case evaluation at arms source '
        f'{p.get("source_commit")} (schedule '
        f'{str(p.get("schedule_sha256"))[:8]}, '
        f'{p.get("blocks_per_role_per_case")} blocks/role/case): '
        f'planned {c["planned"]}, certified_ok {c["certified_ok"]}, '
        f'fail/missing {c["failed_or_missing"]}; B_to_S geomean {seg} / '
        f'median {md_s} / total {tt_s} over the certified subset only '
        f'(n={f.get("n")}); upstream-index verification '
        f'{up_summary["cells"]} cells / {up_summary["role_pins"]} role '
        f'pins all hash-verified; S charges its own '
        'verification/matching/fallback costs and never borrows C '
        'speedups; ratio<1 means the certified route costs more'
    )


def collect_sources(evidence_root, schedule_path, stats_path, stats_md,
                    daily_path, batch_dir='batch-obcr'):
    '''Pin the batch control files + every merged cell document.'''
    paths = [schedule_path, stats_path, daily_path]
    for extra in (stats_md,
                  os.path.join(evidence_root, batch_dir, 'driver.log'),
                  os.path.join(evidence_root, f'{batch_dir}-launch.log'),
                  os.path.join(evidence_root, 'PHASE_RECORD.md')):
        if os.path.isfile(extra):
            paths.append(extra)
        else:
            print(f'note: optional source absent, not pinned: {extra}')
    cells_dir = os.path.join(evidence_root, batch_dir, 'cells')
    if os.path.isdir(cells_dir):
        paths += sorted(
            os.path.join(cells_dir, f)
            for f in os.listdir(cells_dir) if f.endswith('.json'))
    return [{'path': p, 'sha256': sha256_file(p)} for p in paths]


def main(argv=None, proc_root='/proc'):
    ap = argparse.ArgumentParser(
        description='post-batch effects-index append + bench-v1 daily '
                    'comparison figures (SP-32 recipe; fail-closed)')
    ap.add_argument('--mode', choices=('obcr', 's'), default='obcr',
                    help='obcr: the 64-case O/B/C/R batch; s: the '
                         'separate auto-strict S/B batch (own dataset, '
                         'own daily schema, never mixed)')
    ap.add_argument('--evidence-root', required=True)
    ap.add_argument('--schedule',
                    default=None,
                    help='default <evidence-root>/schedule-v2.json '
                         '(obcr) or schedule-s.json (s)')
    ap.add_argument('--stats-json', required=True)
    ap.add_argument('--stats-md', default=None,
                    help='default <stats-json> with .md extension')
    ap.add_argument('--daily-out', default=None,
                    help='default <evidence-root>/benchv1-daily-'
                         'comparisons.json (obcr) or '
                         'benchv1-daily-comparisons-s.json (s)')
    ap.add_argument('--up-index', default=UP_INDEX_DEFAULT)
    ap.add_argument('--ds-index', default=DS_INDEX_DEFAULT)
    ap.add_argument('--batch-id', default=None,
                    help='default benchv1-20261030-obcr (obcr) or '
                         'benchv1-20261030-obcr-S (s)')
    ap.add_argument('--dataset-id', default=None,
                    help='default benchv1-20261030-obcr-eval (obcr) or '
                         'benchv1-20261030-s-eval (s)')
    ap.add_argument('--require-cells', type=int, default=64)
    ap.add_argument('--stats-tool-commit', default=None,
                    help='commit of benchmark_v1_stats.py that produced '
                         'the stats (recorded beside the outputs)')
    ap.add_argument('--dry-run', action='store_true',
                    help='read-only pre-flight: verify everything, '
                         'write nothing')
    a = ap.parse_args(argv)

    md = MODE_DEFAULTS[a.mode]
    if a.schedule is None:
        a.schedule = os.path.join(a.evidence_root, md['schedule'])
    if a.batch_id is None:
        a.batch_id = md['batch_id']
    if a.dataset_id is None:
        a.dataset_id = md['dataset_id']
    if a.daily_out is None:
        a.daily_out = os.path.join(a.evidence_root, md['daily_out'])
    if a.stats_md is None:
        a.stats_md = os.path.splitext(a.stats_json)[0] + '.md'
    if not a.dry_run:
        guard_no_driver(proc_root)

    if not os.path.isfile(a.schedule):
        fail(f'schedule not found: {a.schedule}')
    roles = S_ROLES if a.mode == 's' else ROLES
    up = verify_upstream_batch(a.up_index, a.batch_id, a.require_cells,
                               roles=roles)
    stats, figures = load_and_check_stats(a.stats_json, a.require_cells,
                                          mode=a.mode)
    if a.mode == 's':
        daily = build_daily_s(stats, figures, a.stats_tool_commit)
        label_builder = build_label_s
        line = daily_line_s(daily)
    else:
        daily = build_daily(stats, figures, a.stats_tool_commit)
        label_builder = build_label
        line = daily_line(daily)

    if a.dry_run:
        print(f'[dry-run] upstream batch {a.batch_id}: {up["cells"]} '
              f'cells, {up["role_pins"]} role pins verified')
        print(f'[dry-run] {line}')
        print(f'[dry-run] would write {a.daily_out}')
        print(f'[dry-run] would append dataset {a.dataset_id} to '
              f'{a.ds_index} (sources counted after the daily write)')
        return 0

    # daily figures first, so the dataset entry can pin them
    atomic_write_json(a.daily_out, daily)
    v = json.load(open(a.daily_out, encoding='utf-8'))
    want_daily = S_DAILY_SCHEMA if a.mode == 's' else DAILY_SCHEMA
    assert v['schema'] == want_daily and v['counts'] == stats['counts']

    ds = json.load(open(a.ds_index, encoding='utf-8'))
    if any(x.get('id') == a.dataset_id for x in ds['datasets']):
        fail(f'dataset {a.dataset_id!r} already present in the '
             'effects-index -- the append is once-only')
    sources = collect_sources(a.evidence_root, a.schedule, a.stats_json,
                              a.stats_md, a.daily_out,
                              batch_dir=md['batch_dir'])
    entry = {'id': a.dataset_id,
             'label': label_builder(stats, figures, up),
             'recorded_at': utcnow(),
             'sources': sources}
    shutil.copy2(a.ds_index, a.ds_index + '.backup-pre-' + a.dataset_id)
    ds['datasets'].append(entry)
    atomic_write_json(a.ds_index, ds)

    v2 = json.load(open(a.ds_index, encoding='utf-8'))
    assert v2['datasets'][-1]['id'] == a.dataset_id
    for s in v2['datasets'][-1]['sources']:
        assert sha256_file(s['path']) == s['sha256'], \
            f'pin broken post-write: {s["path"]}'
    print(f'daily comparisons -> {a.daily_out}')
    print(line)
    print(f"effects-index: dataset #{len(v2['datasets'])} "
          f"({len(sources)} sources), every pin hash-verified")
    return 0


if __name__ == '__main__':
    sys.exit(main())

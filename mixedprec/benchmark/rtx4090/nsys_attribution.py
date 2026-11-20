#!/usr/bin/env python
'''Issue #9: attribute CUDA kernel durations to NVTX function buckets.

Companion of run_nsys_attribution.sh: reads one nsys sqlite export (produced
by `nsys export --type sqlite` from a profile of the hotspot profiler with
HOTSPOT_NVTX=1) and charges every CUDA kernel's duration to the innermost
enclosing NVTX range.  Extracted from the shell script into a module so the
two attribution pitfalls that bit live data are unit-tested:

  * NVTX range names live either in the direct ``text`` column (cupy
    RangePush) or in ``StringIds`` (``textId``) -- an inner join on StringIds
    silently drops most ranges and everything lands in [unattributed];
  * NVTX ranges NEST (scf-kernel > scf.xc.grid > scf.xc.grid_ao_eval).  When
    walking back from a kernel start, a range that already ended is merely a
    past SIBLING of an outer range -- breaking the walk there orphans every
    kernel whose inner sibling ended earlier (found live: the entire VV10
    kernel population, 98% of wb97m-v time).

The warmup SCF of the profiler process (no StageTimer ranges installed) is
bucketed separately from the instrumented sample so the function-bucket
composition is pure instrumented run.
'''
import bisect
import json
import sqlite3


def _load_ranges(cur):
    rows = cur.execute('''
        SELECT NVTX_EVENTS.start, NVTX_EVENTS.end,
               COALESCE(NVTX_EVENTS.text, StringIds.value)
        FROM NVTX_EVENTS
        LEFT JOIN StringIds ON NVTX_EVENTS.textId = StringIds.id
        WHERE NVTX_EVENTS.eventType = 59''').fetchall()
    return [(s, e, t) for s, e, t in rows
            if s is not None and e is not None and t]


def _load_kernels(cur):
    try:
        return cur.execute('''
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.demangledName = s.id''').fetchall()
    except sqlite3.OperationalError:
        return cur.execute('''
            SELECT k.start, k.end, s.value
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id''').fetchall()


def attribute_sqlite(path):
    '''One nsys sqlite export -> attribution cell dict (see the script for
    the bucket semantics).  Raises sqlite3.OperationalError on unreadable
    exports so the caller decides which tags are missing.'''
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        ranges = sorted(_load_ranges(cur))
        kernels = _load_kernels(cur)
    finally:
        con.close()

    starts = [r[0] for r in ranges]
    # the profiler process runs warmup THEN the instrumented sample; only the
    # instrumented sample pushes function buckets, so the LAST top-level
    # 'scf-kernel' range bounds it
    scf_ranges = [(s, e) for s, e, t in ranges if t == 'scf-kernel']
    instr_start = scf_ranges[-1][0] if scf_ranges else None

    attrib = {}
    unattrib_total = 0.0
    for ks, ke, kname in kernels:
        if ks is None or ke is None:
            continue
        # innermost enclosing range: candidates are visited in decreasing
        # start order, so the first enclosing range is the innermost; a
        # range that ended before the kernel is a past sibling -- skip, never
        # break
        i = bisect.bisect_right(starts, ks) - 1
        bucket = None
        while i >= 0:
            rs, re, rt = ranges[i]
            if rs <= ks and ke <= re:
                bucket = rt
                break
            i -= 1
        dur = (ke - ks) / 1e9
        if bucket is None:
            unattrib_total += dur
            bucket = '[unattributed]'
        elif bucket == 'scf-kernel':
            bucket = ('warmup/scf-kernel'
                      if instr_start is not None and ks < instr_start
                      else 'scf-kernel/gaps')
        entry = attrib.setdefault(bucket, {}).setdefault(
            kname, {'t_s': 0.0, 'calls': 0})
        entry['t_s'] += dur
        entry['calls'] += 1

    total = sum(v['t_s'] for b in attrib.values() for v in b.values())
    instr_total = sum(
        v['t_s'] for b, ks in attrib.items()
        if b not in ('warmup/scf-kernel', '[unattributed]')
        for v in ks.values())
    return {
        'nvtx_ranges': len(ranges),
        'cuda_kernels_ns': len(kernels),
        'kernel_time_total_s': round(total, 6),
        'instrumented_kernel_time_s': round(instr_total, 6),
        'unattributed_kernel_s': round(unattrib_total, 6),
        'buckets': {
            b: {'t_s': round(sum(v['t_s'] for v in ks.values()), 6),
                'top_kernels': sorted(
                    ks.items(), key=lambda kv: -kv[1]['t_s'])[:12]}
            for b, ks in attrib.items()},
    }


def main(argv=None):
    import argparse
    import os
    p = argparse.ArgumentParser(
        description='attribute CUDA kernel time to NVTX function buckets '
                    '(issue #9)')
    p.add_argument('--outdir',
                   default='/data/results/gpu4pyscf-agent/tasks/issue-9')
    p.add_argument('--tags', nargs='+',
                   default=['b3lyp_svp', 'm062x_svp', 'wb97mv_svp',
                            'r2scan3c_s1'])
    args = p.parse_args(argv)

    doc = []
    for tag in args.tags:
        path = os.path.join(args.outdir, f'nsys_{tag}.sqlite')
        if not os.path.exists(path):
            print(f'[nsys-attrib] MISSING {path}')
            continue
        cell = {'tag': tag}
        cell.update(attribute_sqlite(path))
        doc.append(cell)
    out = os.path.join(args.outdir, 'nsys_kernel_attribution.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'schema_version': 'nsys-kernel-attribution-v1',
                   'note': ('CUDA kernel durations charged to the innermost '
                            'enclosing NVTX function bucket pushed by the '
                            'issue-#9 hotspot profiler. The profiler process '
                            'runs a discarded warmup before the instrumented '
                            'sample; kernels of the warmup SCF sit in '
                            'warmup/scf-kernel, un-wrapped host-side sections '
                            'of the instrumented SCF (DIIS algebra, grid '
                            'setup) in scf-kernel/gaps, and everything else '
                            'keeps its function bucket. '
                            'instrumented_kernel_time_s = total minus warmup '
                            'minus [unattributed] (init).'),
                   'cells': doc}, f, indent=1)
    print(f'[nsys-attrib] wrote {out}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

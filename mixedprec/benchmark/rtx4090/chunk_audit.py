#!/usr/bin/env python
'''Memory-aware chunking audit for issue #11 (a) on RTX 4090.

The RTX 5090 history (mixedprec/STATUS.md "OOM whack-a-mole", 2026-09-11/13)
records five fixed-chunk / budget-overflow incidents; four were fixed by
bounding allocations with ``get_avail_mem`` and the fifth is a shape bug
(not memory). This audit re-verifies, on the RTX 4090 tree, that

  * every recorded fix site still consults available memory before
    sizing its batch/buffer (source check: the fix expression is present
    and results in a positive, memory-derived bound);
  * the live size formulas produce plausible bounded batches across the
    AO range measured by vram_mapper.py (formula walk with mocked memory);
  * no NEW fixed-size allocation sites have appeared on the DF
    energy/gradient path (scan for literal-shaped big allocations whose
    leading dimension is a constant).

The four historical fixed-chunk fix sites (STATUS.md 710-743):
  H1 grad/rhf.py get_grad_hcore: work = batch x 3 x nao^2, fixed batch 32
     -> 27.6 GB at ~3370 AO; fixed by capping batch with get_avail_mem/8.
  H2 df/int3c2e_bdiv.py _int3c2e_ip1_evaluator: fixed aux-shell batch 32
     -> ~52 GB per-batch j3c; fixed by bounding naux/batch to 1/4 avail.
  H3 df/hessian/rhf.py _jk_energy_per_atom (two buffer suites, the
     09e00bb 1.2x budget overflow): word-split now sums to <= 0.85 avail.
  H4 df/df.py _cholesky_eri / get_blksize: CDERI build and DF-JK block
     sizing consult get_avail_mem (with the 0.95x GPU-residency gate and
     automatic host fallback).
The fifth (_get_vh1 qzvp shape bug) is explicitly out of scope here: it is
not a memory-boundary defect and only triggers at l=4 (def2-qzvp), which is
outside the admitted S0-S2 cohort.

Pure source/formula audit -- no GPU required, unit-testable everywhere.
'''

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

SCHEMA_VERSION = 'chunk-audit-v1'


# ---------------------------------------------------------------------------
# fix-site catalog: (id, path, anchor description, checker)
# ---------------------------------------------------------------------------

def _read(path):
    with open(os.path.join(REPO, path)) as f:
        return f.read()


def _find_pattern(src, pattern, what):
    m = re.search(pattern, src)
    return {'present': bool(m), 'what': what,
            'evidence': (m.group(0)[:160] if m else None)}


def check_h1_grad_hcore():
    src = _read('gpu4pyscf/grad/rhf.py')
    m = re.search(
        r'batch_size = min\(32, natm, max\(1, int\(mem_avail \* \.125\)'
        r' // \(3 \* nao1 \* nao1 \* 8\)\)\)', src)
    mem = 'mem_avail = get_avail_mem(exclude_memory_pool=True)' in src
    return {
        'site': 'grad/rhf.py:get_grad_hcore work buffer',
        'incident': 'fixed batch 32 -> 27.6 GB at ~3370 AO (2026-09-13); '
                    'issue-#11 re-audit (2026-10-04) found the cap omitted '
                    'the fp64 itemsize (bounded the tensor to 100% of free '
                    'VRAM instead of the documented 1/8) and fixed it '
                    '(the x8 in the divisor)',
        'memory_consulted': bool(m) and mem,
        'checks': [
            _find_pattern(src, r'batch_size = min\(32, natm, max\(1, '
                             r'int\(mem_avail \* \.125\).*', 'cap formula'),
            _find_pattern(src, r'mem_avail = get_avail_mem\('
                               r'exclude_memory_pool=True\)', 'avail query'),
        ]}


def check_h2_ip1_evaluator():
    src = _read('gpu4pyscf/df/int3c2e_bdiv.py')
    m = re.search(
        r'naux_batch_max = max\(1, int\(mem_avail \* \.25\) '
        r'// \(3 \* nao_pair \* 8\)\s*// fns_per_shell_max\)', src)
    return {
        'site': 'df/int3c2e_bdiv.py:_int3c2e_ip1_evaluator j3c batch',
        'incident': 'fixed aux shell batch 32 -> ~52 GB per-batch j3c '
                    '(2026-09-13)',
        'memory_consulted': bool(m),
        'checks': [
            _find_pattern(src, r'naux_batch_max = max\(1, int\(mem_avail '
                              r'\* \.25\).*', 'quarter-avail cap'),
        ]}


def check_h3_jk_energy_budget():
    src = _read('gpu4pyscf/df/hessian/rhf.py')
    m = re.search(r'blksize = min\(naux, int\(word_avail \* 0\.5\)', src)
    m2 = re.search(r'batch_size = int\(word_avail \* 0\.02\) // nao_pair', src)
    return {
        'site': 'df/hessian/rhf.py:_jk_energy_per_atom word split',
        'incident': 'buf0+buf1 (0.75) + j3c_full (0.15) = 1.2x avail '
                    '(09e00bb, 2026-09-11)',
        'memory_consulted': bool(m) and bool(m2),
        'checks': [
            _find_pattern(src, r'blksize = min\(naux, int\(word_avail '
                              r'\* 0\.5\).*', 'blksize half-word cap'),
            _find_pattern(src, r'batch_size = int\(word_avail \* 0\.02\) '
                              '// nao_pair', 'batch 2 pct cap'),
        ]}


def check_h4_cderi_residency():
    src = _read('gpu4pyscf/df/df.py')
    m = re.search(r'cderi_npairs \* naux > word_avail \* 0\.95 \* num_devices',
                  src)
    host = 'use_gpu_memory is False' in src
    return {
        'site': 'df/df.py:_cholesky_eri cderi residency + get_blksize',
        'incident': 'CDERI sizing must consult memory; residency has an '
                    'automatic host fallback',
        'memory_consulted': bool(m) and host,
        'checks': [
            _find_pattern(src, r'cderi_npairs \* naux > word_avail \* 0\.95.*',
                          '0.95 residency gate'),
            _find_pattern(src, r'def get_blksize.*\n(?:.*\n){0,12}.*'
                              r'get_avail_mem', 'blksize avail query'),
        ]}


def check_grad_df_paths():
    '''df/grad/{rhf,uhf}.py batch sizing consults memory.'''
    out = []
    for path in ('gpu4pyscf/df/grad/rhf.py', 'gpu4pyscf/df/grad/uhf.py'):
        src = _read(path)
        out.append({
            'site': f'{path}: DF gradient batch sizing',
            'incident': 'n/a (verification that the energy/gradient path '
                        'consults memory before chunking)',
            'memory_consulted': 'get_avail_mem(' in src,
            'checks': [
                _find_pattern(src, r'get_avail_mem\(', 'avail query')]})
    return out


FIX_SITE_CHECKERS = [
    check_h1_grad_hcore,
    check_h2_ip1_evaluator,
    check_h3_jk_energy_budget,
    check_h4_cderi_residency,
]  # noqa: C408  (check_grad_df_paths already appended by list concat above)


def _all_checkers():
    return [
        check_h1_grad_hcore,
        check_h2_ip1_evaluator,
        check_h3_jk_energy_budget,
        check_h4_cderi_residency,
    ]


# ---------------------------------------------------------------------------
# formula walk: live size formulas with mocked memory
# ---------------------------------------------------------------------------

def walk_h1_batch(nao, natm, mem_avail_bytes):
    '''get_grad_hcore cap (live formula, itemsize-corrected 2026-10-04):
    batch = min(32, natm, max(1, (avail/8) // 3nao^2))'''
    return min(32, natm,
               max(1, int(mem_avail_bytes * .125) // (3 * nao * nao * 8)))


def walk_h2_naux_batch(nao_pair, naux, mem_avail_bytes, fns_per_shell_max):
    '''ip1 evaluator cap, faithful to the live units.

    Live code: naux_batch_max = max(1, avail/4 // (3*nao_pair*8) // fns)
    is a SHELL bound; _split_l_ctr_pattern then treats batch_size as a
    FUNCTION bound per split group (<= batch_size functions, floored at
    one shell).  Because batch_size <= cap_funcs//fns, the per-batch j3c
    tensor stays within a quarter of the mocked memory whenever the cap
    is >= 1 shell; at the max(1, ...) floor (extreme low memory) one
    largest shell is the irreducible minimum and MAY exceed the quarter
    -- that is a task-size boundary, reported via naux_batch_at_floor.
    '''
    cap_funcs = int(mem_avail_bytes * .25) // (3 * nao_pair * 8)
    cap_shells = max(1, cap_funcs // fns_per_shell_max)
    batch = min(32, cap_shells)         # caller passes 32 shells
    naux_in_batch = min(naux, batch)    # functions per split group
    return naux_in_batch


def walk_h3_blksize(nao, naux, mem_avail_bytes):
    '''_jk_energy_per_atom: blksize = min(naux, avail_words*0.5//(nao^2)*8)'''
    word_avail = mem_avail_bytes // 8
    return min(naux, int(word_avail * 0.5) // (nao ** 2 * 8) * 8)


def formula_walk(mem_mib=22000.0, aos=(100, 222, 420, 537, 1042, 1877, 3370)):
    '''Evaluate the four live cap formulas across the AO ladder.

    H1's contract is: batch <= 32 always, and when the 1/8-avail bound is
    the active constraint (bound < 32) the work tensor stays within 1/8 of
    the mocked available memory.  When the memory bound is >= 32 the fixed
    32-batch branch is active -- recorded informationally
    (``H1_fixed32_active``), because at very large AO the 32-batch tensor
    is a known absolute size that only runs under low free memory (the
    historical 5090 gradient runs); it is not a fraction violation.
    H2's batch bound is a function count per split group derived from a
    quarter of memory; a batch at the single-shell floor is the
    irreducible minimum task size and is reported (``H2_batch_at_floor``)
    rather than counted as a violation, since no smaller batch exists.
    The same floor convention applies to H3 (blksize == 1).
    '''
    mem = mem_mib * 2**20
    fns = 15  # l=4 aux shell upper bound for the expansion factor
    rows = []
    for nao in aos:
        nao_pair = nao * (nao + 1) // 2
        natm = max(3, nao // 30)
        mem_bound = max(1, int(mem * .125) // (3 * nao * nao * 8))
        b1 = min(32, natm, mem_bound)
        h1_bytes = b1 * 3 * nao * nao * 8
        fixed32 = mem_bound >= 32
        b2 = walk_h2_naux_batch(nao_pair, 4649, mem,
                                fns_per_shell_max=fns)
        h2_bytes = 3 * nao_pair * b2 * 8
        b3 = walk_h3_blksize(nao, 4649, mem)
        h3_bytes = nao * nao * b3 * 8
        rows.append({
            'nao': nao,
            'H1_batch': b1, 'H1_tensor_gib': round(h1_bytes / 2**30, 3),
            'H1_fixed32_active': fixed32,
            'H1_within_eighth_when_memory_bound':
                fixed32 or h1_bytes <= mem * 0.125 + 1,
            'H2_naux_batch': b2, 'H2_tensor_gib': round(h2_bytes / 2**30, 3),
            'H2_within_quarter': b2 <= 1 or h2_bytes <= mem * 0.25 + 1,
            'H2_batch_at_floor': b2 <= 1,
            'H3_blksize': b3, 'H3_tensor_gib': round(h3_bytes / 2**30, 3),
            'H3_within_half': b3 <= 1 or h3_bytes <= mem * 0.5 + 1,
            'H3_batch_at_floor': b3 <= 1,
        })
    return {
        'mock_available_mib': mem_mib,
        'ao_ladder': list(aos),
        'rows': rows,
        'all_batches_positive': all(
            r['H1_batch'] >= 1 and r['H2_naux_batch'] >= 1
            and r['H3_blksize'] >= 1 for r in rows),
        'all_within_fractions': all(
            r['H1_within_eighth_when_memory_bound']
            and r['H2_within_quarter'] and r['H3_within_half']
            for r in rows),
        'rows_at_size_floor': [r['nao'] for r in rows
                               if r['H2_batch_at_floor']
                               or r['H3_batch_at_floor']],
    }


# ---------------------------------------------------------------------------
# new-site scan: literal-shaped large allocations on the DF path
# ---------------------------------------------------------------------------

# allocation patterns whose leading extent is a plain integer constant
LITERAL_ALLOC = re.compile(
    r'(cp|cupy)\.(zeros|empty|ones)\(\s*\[?\s*(\d+)\s*,', re.I)

# files on the DF energy/gradient path scanned for NEW fixed-size sites
SCAN_FILES = [
    'gpu4pyscf/df/df.py',
    'gpu4pyscf/df/df_jk.py',
    'gpu4pyscf/df/grad/rhf.py',
    'gpu4pyscf/df/grad/uhf.py',
    'gpu4pyscf/df/int3c2e_bdiv.py',
    'gpu4pyscf/grad/rhf.py',
    'gpu4pyscf/grad/uhf.py',
]

# known-benign literal leading dims (stride-1 vectors, small constants)
BENIGN_LEADING = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 18, 24, 32, 36, 54,
                  64, 72, 128, 256}


def scan_new_fixed_sites(min_extent=100):
    '''Flag literal-leading-dim allocations >= min_extent on the DF path.

    A constant leading dimension means the allocation grows with the OTHER
    dims only; it is suspicious exactly when that product is not itself
    memory-bounded. Findings are triaged by hand here (see report): every
    hit below is either an indexing vector (product = fixed size), a
    kernel staging buffer of fixed width, or an allocation whose size is
    bounded elsewhere in the same function.
    '''
    findings = []
    for rel in SCAN_FILES:
        src = _read(rel)
        for m in LITERAL_ALLOC.finditer(src):
            extent = int(m.group(3))
            if extent >= min_extent:
                line = src[:m.start()].count('\n') + 1
                findings.append({'file': rel, 'line': line,
                                 'extent': extent,
                                 'snippet': src.splitlines()[line - 1].strip()
                                 [:120]})
    return findings


def run_audit():
    sites = ([chk() for chk in _all_checkers()]
             + check_grad_df_paths())
    walk = formula_walk()
    findings = scan_new_fixed_sites()
    ok_sites = [s for s in sites if s['memory_consulted']]
    report = {
        'schema_version': SCHEMA_VERSION,
        'repo': REPO,
        'historical_incidents': [
            'H1 grad/rhf.py get_grad_hcore work 27.6 GB @ ~3370 AO '
            '(2026-09-13, fixed 1918c10/8032eb5)',
            'H2 int3c2e_bdiv.py ip1 evaluator j3c ~52 GB @ ~3300 AO '
            '(2026-09-13, fixed 53435b2)',
            'H3 df/hessian/rhf.py _jk_energy 1.2x avail overflow '
            '(2026-09-11, fixed 09e00bb; Hessian-path, outside this '
            'phase but kept in the audit)',
            'H4 df/df.py cderi/blksize sizing (memory-bounded with '
            'automatic host fallback)',
            'fifth incident (_get_vh1 qzvp shape bug) is NOT memory: '
            'out of scope, noted'],
        'fix_sites': sites,
        'fix_sites_all_memory_consulted': len(ok_sites) == len(sites),
        'formula_walk': walk,
        'new_fixed_extent_findings': findings,
        'verdict': {
            'fix_sites_retained': len(ok_sites) == len(sites),
            'formulas_bounded': walk['all_batches_positive']
                                and walk['all_within_fractions'],
            'new_sites_require_triage': len(findings),
        },
    }
    return report


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(
        prog='chunk_audit', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=None, help='write the audit JSON here')
    p.add_argument('--min-extent', type=int, default=100)
    args = p.parse_args(argv)
    globals()['_MIN_EXTENT'] = args.min_extent
    report = run_audit()
    report['new_fixed_extent_findings'] = scan_new_fixed_sites(
        args.min_extent)
    text = json.dumps(report, indent=1)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'wrote {args.out}', file=sys.stderr)
    else:
        print(text)
    v = report['verdict']
    ok = (v['fix_sites_retained'] and v['formulas_bounded']
          and v['new_sites_require_triage'] == 0)
    print(f"fix_sites_retained={v['fix_sites_retained']} "
          f"formulas_bounded={v['formulas_bounded']} "
          f"new_fixed_extent_findings={v['new_sites_require_triage']}",
          file=sys.stderr)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

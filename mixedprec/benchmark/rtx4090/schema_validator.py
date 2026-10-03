#!/usr/bin/env python3
"""Schema validator for RTX 4090 benchmark results (``rtx4090-energy-v1``).

The contract is docs/consumer-gpu-plan.md section 4 (mirrored in
docs/rtx4090-benchmark-schema.md).  A benchmark JSON that fails this
validator is NOT acceptance evidence -- regardless of how good the numbers
look.  The validator is deliberately dependency-free (stdlib only) so it can
run anywhere and be unit-tested without a GPU.

Eight mandatory rules (plan section 4 "mandatory rules"), each reported with
its rule id:

  R1  timing discipline: >= 3 independent samples, separate warmup,
      median/min/max consistent, device-sync evidence on GPU lanes;
  R2  status semantics: non-ok rows carry skip_reason/errors, skipped/OOM
      rows are never silently missing or invented into passes, verdict
      'pass' requires convergence AND |dE_vs_gpu_fp64| <= 1e-6 Ha
      (the fp64 reference lane itself passes on convergence alone);
  R3  lane effectiveness: an ok 'auto'/'fp32-screen' row with all-zero
      fp32_entry_hits is 'lane-ineffective' (wheel shadowing / mode never
      set) and can never be a pass;
  R4  method completeness: r2SCAN-3c rows need dispersion + gcp terms and
      the def2-mTZVPP basis; omegaB97M-V rows need a non-null nlc term;
  R5  error-tier isolation: dE_df_approximation exists only on df-vs-conv
      rows; cross_program entries carry the matched flag and never feed
      the level-(1) precision gate;
  R6  evidence anchoring: every row's evidence files exist under the
      evidence root and their sha256 matches;
  R7  memory discipline: ok rows have a complete mem block
      (free/total/peak/host-RSS/headroom/within_budget); a breached
      headroom budget must carry an explanation;
  R8  implementation agreement: |dE_vs_cpu_fp64| > 1e-8 Ha without a named
      impl_agreement_note is 'impl-unexplained'.

Usage:
    python schema_validator.py BENCHMARK.json [--evidence-root /data/results]

Exit codes: 0 = valid, 1 = validation issues, 2 = usage/IO error.
"""

import argparse
import hashlib
import json
import os
import sys

SCHEMA_VERSION = 'rtx4090-energy-v1'

LANES = ('fp64', 'auto', 'fp32-screen', 'cpu-fp64', 'df-vs-conv',
         'xref-g16', 'xref-orca')
GPU_LANES = ('fp64', 'auto', 'fp32-screen')   # lanes that must show CUDA sync
STATUSES = ('ok', 'skipped', 'oom', 'error')
VERDICTS = ('pass', 'fail', 'no_reference', 'skipped')
PILOT_STAGES = ('S0', 'S1', 'S2', 'S3', 'S4')

PRECISION_GATE_HA = 1e-6      # level-(1) acceptance gate (plan section 1.2)
IMPL_NOTE_TOL_HA = 1e-8       # level-(2) deviation that demands a named note
MIN_SAMPLES = 3

META_KEYS = ('gpu_name', 'gpu_total_mem_mib', 'driver_version', 'cuda_version',
             'cpu_model', 'cpu_threads', 'ram_gb', 'python', 'cupy', 'pyscf',
             'gpu4pyscf_source_commit', 'gpu4pyscf_source_path', 'libxc',
             'mem_headroom_mib', 'timestamp_utc', 'run_id')
SYSTEM_KEYS = ('geom_file', 'geom_sha256', 'natm', 'nao', 'charge', 'spin',
               'nelec', 'cart', 'geom_units')
METHOD_KEYS = ('xc', 'basis', 'auxbasis', 'with_df', 'grids', 'nlcgrids',
               'dispersion', 'gcp', 'conv_tol', 'conv_tol_grad',
               'direct_scf_tol', 'init_guess', 'max_cycle', 'open_shell')
MEM_KEYS = ('gpu_free_before_mib', 'gpu_total_mib', 'gpu_peak_alloc_mib',
            'host_rss_peak_mib', 'headroom_budget_mib', 'within_budget')
E_TERMS_KEYS = ('e1', 'coul', 'exc', 'nlc', 'dispersion', 'gcp')
FP32_HITS_KEYS = ('df_jk', 'eval_rho', 'xc_grid', 'grad_ip1')

SKIP_REASONS = ('headroom-budget', 'oom-observed', 'unsupported',
                'deferred-candidate', 'manual-admission-pending', 'other')


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


class Issues:
    '''Accumulates validation problems with stable rule ids and tags.'''

    def __init__(self):
        self.items = []

    def add(self, rule, tag, message, row=None):
        self.items.append({'rule': rule, 'tag': tag, 'row': row,
                           'message': message})

    def __len__(self):
        return len(self.items)

    def dump(self):
        for it in self.items:
            where = f'row {it["row"]}: ' if it['row'] else ''
            print(f'[{it["rule"]}/{it["tag"]}] {where}{it["message"]}',
                  file=sys.stderr)


def _row_label(row):
    rk = row.get('row_key') or {}
    return '/'.join(str(rk.get(k)) for k in ('geom', 'basis', 'xc', 'lane'))


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _check_evidence_ref(issues, ref, field, row_label, evidence_root):
    '''Check one "path@sha256" evidence reference (rule R6).

    evidence_root=None means format-check-only (used by unit tests and
    pre-publication lint); existence and hash are verified only when an
    evidence root is given.'''
    if not isinstance(ref, str) or '@' not in ref:
        issues.add('R6', 'evidence-format',
                   f'{field} must be "path@sha256", got {ref!r}', row_label)
        return
    path, _, digest = ref.rpartition('@')
    if len(digest) != 64 or any(c not in '0123456789abcdef'
                                for c in digest.lower()):
        issues.add('R6', 'evidence-format',
                   f'{field} sha256 part is not 64 hex chars: {digest!r}',
                   row_label)
        return
    if evidence_root is None:
        return
    root = str(evidence_root)
    if not path.startswith(root.rstrip('/') + '/'):
        issues.add('R6', 'evidence-outside-root',
                   f'{field} {path} is not under the evidence root '
                   f'{root}', row_label)
        return
    if not _isfile(path):
        issues.add('R6', 'evidence-missing',
                   f'{field} {path} does not exist', row_label)
        return
    actual = _sha256_file(path)
    if actual != digest:
        issues.add('R6', 'evidence-hash-mismatch',
                   f'{field} {path}: recorded sha256 {digest[:12]} != '
                   f'actual {actual[:12]}', row_label)


def _isfile(path):
    return os.path.isfile(path)


# ---------------------------------------------------------------------------
# row-level rules
# ---------------------------------------------------------------------------

def _check_structure(issues, row, seen_keys):
    label = _row_label(row) or '<missing row_key>'
    rk = row.get('row_key')
    if not isinstance(rk, dict):
        issues.add('structure', 'row-key-missing',
                   'row has no row_key dict', None)
        return label
    for k in ('geom', 'basis', 'xc', 'lane'):
        if not isinstance(rk.get(k), str) or not rk.get(k):
            issues.add('structure', 'row-key-field',
                       f'row_key.{k} must be a non-empty string',
                       label)
    key = (rk.get('geom'), rk.get('basis'), rk.get('xc'), rk.get('lane'))
    if key in seen_keys:
        issues.add('structure', 'duplicate-row-key',
                   f'duplicate row_key {key}', label)
    seen_keys.add(key)
    if rk.get('lane') not in LANES:
        issues.add('structure', 'lane-enum',
                   f'lane {rk.get("lane")!r} not in {LANES}', label)
    if row.get('pilot_stage') not in PILOT_STAGES:
        issues.add('structure', 'pilot-stage',
                   f'pilot_stage {row.get("pilot_stage")!r} not in '
                   f'{PILOT_STAGES}', label)
    if row.get('status') not in STATUSES:
        issues.add('structure', 'status-enum',
                   f'status {row.get("status")!r} not in {STATUSES}', label)

    sysb = row.get('system')
    meth = row.get('method')
    # full system/method blocks are only mandatory for rows that actually
    # ran; skipped/oom/error stubs are governed by R2/R7 instead
    if row.get('status') == 'ok':
        if not isinstance(sysb, dict):
            issues.add('structure', 'system-block',
                       'system block missing', label)
        else:
            for k in SYSTEM_KEYS:
                if k not in sysb:
                    issues.add('structure', 'system-field',
                               f'system.{k} missing', label)
            for k in ('natm', 'nao', 'charge', 'spin', 'nelec'):
                if not is_int(sysb.get(k)):
                    issues.add('structure', 'system-field',
                               f'system.{k} must be an integer', label)
            if not isinstance(sysb.get('cart'), bool):
                issues.add('structure', 'system-field',
                           'system.cart must be a boolean', label)
            if sysb.get('geom_units') not in ('angstrom', 'bohr'):
                issues.add('structure', 'system-field',
                           'system.geom_units must be "angstrom" or "bohr"',
                           label)
        if not isinstance(meth, dict):
            issues.add('structure', 'method-block',
                       'method block missing', label)
        else:
            for k in METHOD_KEYS:
                if k not in meth:
                    issues.add('structure', 'method-field',
                               f'method.{k} missing', label)
            for k in ('grids', 'nlcgrids'):
                g = meth.get(k)
                if not isinstance(g, dict) or 'level' not in g or \
                        'atom_grid' not in g:
                    issues.add('structure', 'method-field',
                               f'method.{k} must be {{level, atom_grid}}',
                               label)
            if not is_num(meth.get('conv_tol')) or \
                    (meth.get('conv_tol') or 0) <= 0:
                issues.add('structure', 'method-field',
                           'method.conv_tol must be a positive number',
                           label)
            if not is_int(meth.get('max_cycle')) or \
                    (meth.get('max_cycle') or 0) < 1:
                issues.add('structure', 'method-field',
                           'method.max_cycle must be a positive integer',
                           label)

    acc = row.get('accuracy')
    if not isinstance(acc, dict):
        issues.add('structure', 'accuracy-block',
                   'accuracy block missing', label)
    elif acc.get('verdict') not in VERDICTS:
        issues.add('structure', 'verdict-enum',
                   f'accuracy.verdict {acc.get("verdict")!r} not in '
                   f'{VERDICTS}', label)

    cp = row.get('cross_program')
    if cp is not None:
        if not isinstance(cp, list):
            issues.add('structure', 'cross-program',
                       'cross_program must be a list', label)
        else:
            for e in cp:
                if not isinstance(e, dict) or not e.get('program') or \
                        not e.get('method_key') or \
                        not isinstance(e.get('matched'), bool):
                    issues.add('structure', 'cross-program',
                               'cross_program entries need program, '
                               'method_key and boolean matched', label)
                elif e.get('matched') and (not is_num(e.get('e_tot')) or
                                            not is_num(e.get('dE'))):
                    issues.add('structure', 'cross-program',
                               'matched cross_program entries need numeric '
                               'e_tot and dE', label)
    return label


def _check_timings(issues, row, label):
    '''Rule R1: warmup separated from >= 3 independent samples.'''
    res = row.get('result') or {}
    t = res.get('timings')
    if not isinstance(t, dict):
        issues.add('R1', 'timings-block',
                   'ok row has no timings block', label)
        return
    samples = t.get('samples_s')
    if not isinstance(samples, list) or len(samples) < MIN_SAMPLES:
        issues.add('R1', 'samples-count',
                   f'timings.samples_s needs >= {MIN_SAMPLES} entries, got '
                   f'{samples if not isinstance(samples, list) else len(samples)}',
                   label)
    elif not all(is_num(s) and s > 0 for s in samples):
        issues.add('R1', 'samples-values',
                   'timings.samples_s entries must be positive numbers',
                   label)
    else:
        srt = sorted(samples)
        n = len(srt)
        med = srt[n // 2] if n % 2 else (srt[n // 2 - 1] + srt[n // 2]) / 2
        for name, want in (('median_s', med), ('min_s', srt[0]),
                           ('max_s', srt[-1])):
            if not is_num(t.get(name)) or abs(t[name] - want) > 1e-9:
                issues.add('R1', 'sample-statistics',
                           f'timings.{name} {t.get(name)!r} inconsistent '
                           f'with samples (expected {want!r})', label)
    if not is_num(t.get('warmup_s')) or t['warmup_s'] < 0:
        issues.add('R1', 'warmup-missing',
                   'timings.warmup_s must be a number >= 0 (independent '
                   'warmup, never mixed into samples)', label)
    if not is_num(t.get('end_to_end_s')) or t['end_to_end_s'] < 0:
        issues.add('R1', 'end-to-end',
                   'timings.end_to_end_s must be >= 0 (includes CDERI '
                   'build, precision switch, FP64 tail, corrections)',
                   label)
    sync = t.get('sync')
    if not isinstance(sync, str) or not sync:
        issues.add('R1', 'sync-evidence',
                   'timings.sync must state the synchronization discipline',
                   label)
    elif row['row_key'].get('lane') in GPU_LANES and \
            'device-sync' not in sync.lower():
        issues.add('R1', 'sync-evidence',
                   f'GPU lane needs device-sync evidence in timings.sync, '
                   f'got {sync!r}', label)


def _check_status_semantics(issues, row, label):
    '''Rule R2: skipped/OOM honest, pass requires the gate.'''
    status = row.get('status')
    acc = row.get('accuracy') or {}
    verdict = acc.get('verdict')
    res = row.get('result') or {}
    skip_reason = row.get('skip_reason')
    errors = res.get('errors')

    if status == 'ok' and skip_reason:
        issues.add('R2', 'skip-reason-on-ok',
                   f'ok row carries skip_reason {skip_reason!r}', label)
    if status != 'ok' and not skip_reason and not errors:
        issues.add('R2', 'non-ok-unexplained',
                   f'status {status!r} needs skip_reason or errors', label)
    if status == 'skipped' and not skip_reason:
        issues.add('R2', 'skip-reason-missing',
                   'skipped rows must record why (never silently omitted)',
                   label)
    if status in ('oom', 'error') and not (
            isinstance(errors, dict) and errors.get('type') and
            errors.get('message')):
        issues.add('R2', 'errors-missing',
                   f'status {status!r} needs errors '
                   f'{{type, message}} (OOM is a hardware boundary record, '
                   'not a skip)', label)
    if status != 'ok' and verdict == 'pass':
        issues.add('R2', 'pass-on-non-ok',
                   f'status {status!r} rows can never be verdict pass',
                   label)
    if verdict not in VERDICTS and verdict is not None:
        pass  # structural check already reported the enum violation

    converged = res.get('converged')
    if status == 'ok':
        if converged is not True:
            if verdict == 'pass':
                issues.add('R2', 'pass-not-converged',
                           'verdict pass requires converged == true', label)
            elif verdict not in ('fail', 'no_reference'):
                issues.add('R2', 'verdict-on-unconverged',
                           'unconverged rows can only be verdict fail or '
                           'no_reference', label)
        if verdict == 'pass':
            dE = acc.get('dE_vs_gpu_fp64')
            lane = row['row_key'].get('lane')
            if dE is None and lane not in ('fp64', 'df-vs-conv'):
                issues.add('R2', 'pass-without-reference',
                           'verdict pass on a lane without dE_vs_gpu_fp64 '
                           'requires being the fp64 reference (or a '
                           'df-vs-conv row with its dE_df_approximation)',
                           label)
            if lane == 'df-vs-conv' and \
                    acc.get('dE_df_approximation') is None:
                issues.add('R2', 'pass-without-reference',
                           'a passing df-vs-conv row must report '
                           'dE_df_approximation', label)
            if is_num(dE) and abs(dE) > PRECISION_GATE_HA:
                issues.add('R2', 'gate-exceeded',
                           f'|dE_vs_gpu_fp64| = {abs(dE):.3e} Ha exceeds the '
                           f'{PRECISION_GATE_HA:.0e} Ha precision gate but '
                           'verdict is pass', label)
            if is_num(dE) and lane == 'fp64' and abs(dE) > 0:
                # the fp64 lane IS the reference; a nonzero self-delta means
                # two rows were mislabeled
                issues.add('R2', 'fp64-self-delta',
                           'fp64 reference row has nonzero dE_vs_gpu_fp64',
                           label)


def _check_lane_effectiveness(issues, row, label):
    '''Rule R3: mixed-precision lanes must actually hit fp32 branches.'''
    lane = row['row_key'].get('lane')
    if lane not in ('auto', 'fp32-screen') or row.get('status') != 'ok':
        return
    hits = (row.get('result') or {}).get('fp32_entry_hits') or {}
    if not isinstance(hits, dict) or not any(
            is_int(v) and v > 0 for v in hits.values()):
        issues.add('R3', 'lane-ineffective',
                   f'{lane} row has all-zero fp32_entry_hits: the '
                   'mixed-precision code path never executed (wheel '
                   'shadowing or precision_mode never set)', label)
        if (row.get('accuracy') or {}).get('verdict') == 'pass':
            issues.add('R3', 'lane-ineffective-pass',
                       'an ineffective lane can never be verdict pass',
                       label)


def _check_method_completeness(issues, row, label):
    '''Rule R4: whole-method rows carry every required term.'''
    if row.get('status') != 'ok':
        return
    xc = ((row.get('method') or {}).get('xc')
          or (row.get('row_key') or {}).get('xc') or '')
    xc_l = str(xc).lower()
    basis = str((row.get('method') or {}).get('basis')
                or (row.get('row_key') or {}).get('basis') or '').lower()
    e_terms = ((row.get('result') or {}).get('e_terms')) or {}
    if 'r2scan' in xc_l and '3c' in xc_l:
        if e_terms.get('dispersion') is None or e_terms.get('gcp') is None:
            issues.add('R4', 'r2scan3c-terms-missing',
                       'r2SCAN-3c rows need non-null e_terms.dispersion '
                       'AND e_terms.gcp (missing term = wrong method, not '
                       'a small error)', label)
        if basis != 'def2-mtzvpp':
            issues.add('R4', 'r2scan3c-basis',
                       f'r2SCAN-3c rows must use def2-mTZVPP, got '
                       f'{basis!r}', label)
    if 'b97m-v' in xc_l and e_terms.get('nlc') is None:
        issues.add('R4', 'nlc-missing',
                   'omegaB97M-V family rows need non-null e_terms.nlc '
                   '(VV10/NLC must be on)', label)


def _check_tier_isolation(issues, row, label):
    '''Rule R5: DF approximation and cross-program stay out of the gate.'''
    acc = row.get('accuracy') or {}
    lane = row['row_key'].get('lane')
    if acc.get('dE_df_approximation') is not None and lane != 'df-vs-conv':
        issues.add('R5', 'df-approx-misplaced',
                   'dE_df_approximation may only appear on df-vs-conv rows '
                   '(never mixed into the precision gate)', label)
    if lane == 'df-vs-conv' and row.get('status') == 'ok' and \
            acc.get('verdict') == 'pass' and \
            acc.get('dE_df_approximation') is None:
        issues.add('R5', 'df-approx-missing',
                   'a passing df-vs-conv row must report '
                   'dE_df_approximation', label)


def _check_memory(issues, row, label):
    '''Rule R7: the mem block, complete for ok rows.'''
    status = row.get('status')
    mem = (row.get('result') or {}).get('mem')
    if status == 'ok':
        if not isinstance(mem, dict):
            issues.add('R7', 'mem-block-missing',
                       'ok row has no mem block', label)
            return
        for k in MEM_KEYS:
            v = mem.get(k)
            if k == 'within_budget':
                if not isinstance(v, bool):
                    issues.add('R7', 'mem-field',
                               f'mem.within_budget must be a boolean', label)
            elif not is_num(v) or v is None:
                issues.add('R7', 'mem-field',
                           f'mem.{k} missing or not numeric (measured, not '
                           'estimated)', label)
        if mem.get('within_budget') is False and not (
                isinstance(mem.get('budget_explanation'), str) and
                mem['budget_explanation'].strip()):
            issues.add('R7', 'budget-breach-unexplained',
                       'within_budget == false requires a budget_explanation '
                       '(stop/escalate, never silently proceed)', label)
    elif status == 'skipped':
        if not isinstance(mem, dict) or not is_num(
                mem.get('gpu_free_before_mib')) or not is_num(
                mem.get('gpu_total_mib')):
            issues.add('R7', 'mem-block-skipped',
                       'skipped rows must at least record measured '
                       'gpu_free_before_mib and gpu_total_mib', label)


def _check_impl_agreement(issues, row, label):
    '''Rule R8: level-(2) deviations need a named implementation note.'''
    acc = row.get('accuracy') or {}
    dE = acc.get('dE_vs_cpu_fp64')
    if is_num(dE) and abs(dE) > IMPL_NOTE_TOL_HA:
        note = acc.get('impl_agreement_note')
        if not isinstance(note, str) or not note.strip():
            issues.add('R8', 'impl-unexplained',
                       f'|dE_vs_cpu_fp64| = {abs(dE):.3e} Ha exceeds the '
                       f'{IMPL_NOTE_TOL_HA:.0e} Ha matched-settings level '
                       'without a named impl_agreement_note (screening '
                       'defaults, libxc variant, ...); it must not be '
                       'attributed to mixed precision or silently passed',
                       label)


def _check_evidence(issues, row, evidence_root):
    label = _row_label(row)
    ev = row.get('evidence')
    if not isinstance(ev, dict):
        issues.add('R6', 'evidence-block',
                   'row has no evidence block', label)
        return
    for field in ('raw_log', 'result_json'):
        _check_evidence_ref(issues, ev.get(field), field, label,
                            evidence_root)
    cmd = ev.get('command')
    if not isinstance(cmd, str) or not cmd.strip():
        issues.add('R6', 'repro-command',
                   'evidence.command must be the full reproduction command',
                   label)


# ---------------------------------------------------------------------------
# document level
# ---------------------------------------------------------------------------

def validate_document(doc, evidence_root=None):
    '''Validate one benchmark document; returns a list of issue dicts.'''
    issues = Issues()
    if not isinstance(doc, dict):
        issues.add('structure', 'not-a-document',
                   'benchmark file is not a JSON object')
        return issues.items
    if doc.get('schema_version') != SCHEMA_VERSION:
        issues.add('structure', 'schema-version',
                   f'schema_version must be {SCHEMA_VERSION!r}, got '
                   f'{doc.get("schema_version")!r}')
    meta = doc.get('meta')
    if not isinstance(meta, dict):
        issues.add('structure', 'meta-block', 'meta block missing')
    else:
        for k in META_KEYS:
            if k not in meta:
                issues.add('structure', 'meta-field', f'meta.{k} missing')
        if not is_num(meta.get('mem_headroom_mib')) or \
                meta['mem_headroom_mib'] < 0:
            issues.add('structure', 'meta-field',
                       'meta.mem_headroom_mib must be a number >= 0')
    rows = doc.get('rows')
    if not isinstance(rows, list) or not rows:
        issues.add('structure', 'rows',
                   'rows must be a non-empty list')
        return issues.items

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            issues.add('structure', 'row-type', 'row is not an object')
            continue
        label = _check_structure(issues, row, seen)
        if row.get('status') == 'ok':
            _check_timings(issues, row, label)
        _check_status_semantics(issues, row, label)
        _check_lane_effectiveness(issues, row, label)
        _check_method_completeness(issues, row, label)
        _check_tier_isolation(issues, row, label)
        _check_memory(issues, row, label)
        _check_impl_agreement(issues, row, label)
        _check_evidence(issues, row, evidence_root)
    return issues.items


def validate_file(path, evidence_root=None):
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    return validate_document(doc, evidence_root=evidence_root)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='schema_validator',
        description='Validate an RTX 4090 benchmark JSON against '
                    'rtx4090-energy-v1 (docs/consumer-gpu-plan.md section 4)')
    p.add_argument('file', help='benchmark result JSON')
    p.add_argument('--evidence-root', default='/data/results',
                   help='root directory all evidence files must live under '
                        '(empty string = format check only)')
    p.add_argument('--json', action='store_true',
                   help='emit issues as JSON instead of text')
    args = p.parse_args(argv)
    try:
        doc_issues = validate_file(
            args.file,
            evidence_root=(args.evidence_root or None))
    except (OSError, ValueError) as exc:
        print(f'error: cannot read {args.file}: {exc}', file=sys.stderr)
        return 2
    if doc_issues:
        if args.json:
            print(json.dumps({'valid': False, 'issues': doc_issues},
                             indent=1))
        else:
            print(f'{args.file}: INVALID ({len(doc_issues)} issues)',
                  file=sys.stderr)
            for it in doc_issues:
                where = f'row {it["row"]}: ' if it['row'] else ''
                print(f'  [{it["rule"]}/{it["tag"]}] {where}'
                      f'{it["message"]}', file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps({'valid': True, 'issues': []}))
    else:
        print(f'{args.file}: VALID ({SCHEMA_VERSION})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

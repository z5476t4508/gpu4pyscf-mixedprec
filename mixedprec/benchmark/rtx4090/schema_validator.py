#!/usr/bin/env python3
"""Schema validator for RTX 4090 benchmark results (``rtx4090-energy-v1``).

The contract is docs/consumer-gpu-plan.md section 4 (mirrored in
docs/rtx4090-benchmark-schema.md).  A benchmark JSON that fails this
validator is NOT acceptance evidence -- regardless of how good the numbers
look.  The validator is deliberately dependency-free (stdlib only) so it can
run anywhere and be unit-tested without a GPU.

Eight mandatory rules (plan section 4 "mandatory rules"), each reported with
its rule id, plus the two optional-block consistency rules added by issues
#6/#7:

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
      impl_agreement_note is 'impl-unexplained';
  R9  grid stability (optional block, issue #6): when a document carries a
      grid_stability block it must be internally consistent -- production
      level present among the scanned entries, per-entry level/ngrids/
      e_tot/converged present, classification from the allowed set, and the
      deltas recomputed from the stored energies.  Diagnostic only: never an
      acceptance gate (the precision gate compares lanes at identical grids,
      where grid error cancels).  The same consistency checks apply to the
      issue-#7 nlcgrids_stability block (the VV10 nonlocal-correlation
      grid scan; identical shape, diagnostic only);
  R10 NLC term consistency (issue #7): on rows of an NLC functional (the
      VV10 family, e.g. wb97m-v/wb97x-v) every ok lane must carry a
      non-null e_terms.nlc (a dropped NLC term is the whole method wrong,
      not a small error), and the NLC pieces of the auto and cpu-fp64 lanes
      must agree with the fp64 reference lane within the 1e-6 Ha precision
      gate -- the NLC path is exactly the part that had never run under
      mixed precision before issue #7, so its agreement is checked
      explicitly rather than inferred from the total energy alone;
  R11 r2SCAN-3c correction consistency (issue #8): the composite method's
      D4 dispersion and gCP basis-set correction are deterministic
      geometry-only terms evaluated by the same pyscf-dispersion library on
      every lane, so every ok lane must carry both terms (already R4) and
      the pieces must agree across the auto/fp32-screen/cpu-fp64/df-vs-conv
      lanes with the fp64 reference -- near-bitwise is expected; a
      disagreement means the corrections were attached inconsistently (the
      3c to_gpu ordering trap), never a precision effect.

Usage:
    python schema_validator.py BENCHMARK.json [--evidence-root /data/results]

Exit codes: 0 = valid, 1 = validation issues, 2 = usage/IO error.
"""

import argparse
import hashlib
import json
import math
import os
import sys

SCHEMA_VERSION = 'rtx4090-energy-v1'
GRADIENT_SCHEMA_VERSION = 'rtx4090-gradient-v1'

LANES = ('fp64', 'auto', 'fp32-screen', 'cpu-fp64', 'df-vs-conv',
         'xref-g16', 'xref-orca')
GPU_LANES = ('fp64', 'auto', 'fp32-screen')   # lanes that must show CUDA sync
STATUSES = ('ok', 'skipped', 'oom', 'error')
VERDICTS = ('pass', 'fail', 'no_reference', 'skipped')
PILOT_STAGES = ('S0', 'S1', 'S2', 'S3', 'S4')

PRECISION_GATE_HA = 1e-6      # level-(1) acceptance gate (plan section 1.2)
IMPL_NOTE_TOL_HA = 1e-8       # level-(2) deviation that demands a named note
MIN_SAMPLES = 3

# ---------------------------------------------------------------------------
# rtx4090-gradient-v1 (issue #12): gradient lanes and gates
# ---------------------------------------------------------------------------

GRADIENT_LANES = (
    'fp64',            # GPU fp64 SCF + fp64 gradient kernels -- the reference
    'auto',            # production mixed lane: auto SCF (fp32 head + fp64
                       # tail) and the gradient resolved as fp32 kernels
                       # (grad/rhf.py _resolve_precision: 'auto' selects fp32
                       # for a gradient -- there is no fp64 tail to converge)
    'fp32-grad',       # kernel-isolation control: fp64 SCF + fp32 gradient
                       # kernels (g.precision_mode='fp32' override) -- the
                       # gradient-kernel error alone, no SCF-path difference
    'auto-fp64-grad',  # trajectory-isolation control: auto SCF + fp64
                       # gradient kernels (g.precision_mode='fp64' override)
                       # -- the SCF convergence-path effect alone
    'auto-verify',     # issue-7 diagnosed route: precision_mode='auto+verify'
                       # SCF (auto policy + one strict-fp64 reference
                       # trajectory from the same init guess; the lower
                       # converged solution kept) with fp32 gradient kernels
                       # -- the retest lane for the methoxy UKS failures
    'auto-strict',     # LC-76 strict fp64-reference matching route:
                       # precision_mode='auto+strict' SCF (auto policy +
                       # strict-fp64 reference trajectory; the fp64-lane
                       # solution adopted unconditionally on a distinct
                       # result or fail-closed with the exact reason) with
                       # fp32 gradient kernels -- the contract of matching
                       # the identical-settings FP64 trajectory
    'cpu-fp64',        # PySCF CPU sentinel (implementation agreement)
)
GRADIENT_GPU_LANES = ('fp64', 'auto', 'fp32-grad', 'auto-fp64-grad',
                      'auto-verify', 'auto-strict')
# the gradient acceptance gate: one order of magnitude below geomeTRIC's
# default gmax convergence threshold 3e-4 Eh/Bohr (plan issue #12: "低于优化
# 阈值一个量级"; historical RTX 5090 mixed-precision gradient error ~1e-5)
GRADIENT_GATE_HA_BOHR = 3e-5
# level-(2) gradient deviation that demands a named implementation note
GRAD_IMPL_NOTE_TOL = 1e-6

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

def _check_structure(issues, row, seen_keys, lanes=LANES):
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
    if rk.get('lane') not in lanes:
        issues.add('structure', 'lane-enum',
                   f'lane {rk.get("lane")!r} not in {lanes}', label)
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


def _check_timings(issues, row, label, gpu_lanes=GPU_LANES, grad=False):
    '''Rule R1: warmup separated from >= 3 independent samples.  With
    ``grad=True`` (rtx4090-gradient-v1) the block must additionally report
    t_grad_s -- the gradient phase timed separately from the SCF, both
    device-synced.'''
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
    if grad:
        if not is_num(t.get('t_grad_s')) or t['t_grad_s'] < 0:
            issues.add('R1', 'grad-phase-timing',
                       'timings.t_grad_s must be a number >= 0 (the '
                       'gradient phase timed separately from the SCF, '
                       'device-synced)', label)
        if not is_num(t.get('t_scf_s')) or t['t_scf_s'] < 0:
            issues.add('R1', 'scf-phase-timing',
                       'timings.t_scf_s must be a number >= 0 (SCF and '
                       'gradient phases reported separately)', label)
    sync = t.get('sync')
    if not isinstance(sync, str) or not sync:
        issues.add('R1', 'sync-evidence',
                   'timings.sync must state the synchronization discipline',
                   label)
    elif row['row_key'].get('lane') in gpu_lanes and \
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


def _check_status_semantics_grad(issues, row, label):
    '''Rule R2, gradient variant (rtx4090-gradient-v1): verdict 'pass'
    requires convergence AND max|dg| within the gradient gate, and the gate
    must be evaluated against the SAME implementation's fp64 reference --
    a row whose only reference is cross-implementation (cpu-fp64) can never
    gate-pass.  Runs after the shared R2 semantics; only adds checks.'''
    acc = row.get('accuracy') or {}
    verdict = acc.get('verdict')
    if verdict != 'pass' or row.get('status') != 'ok':
        return
    lane = (row.get('row_key') or {}).get('lane')
    maxdg = acc.get('max_abs_dg_vs_gpu_fp64')
    if lane != 'fp64':
        if maxdg is None:
            issues.add('R2', 'grad-pass-without-reference',
                       'gradient verdict pass requires '
                       'max_abs_dg_vs_gpu_fp64 (the lane is not the fp64 '
                       'reference)', label)
        elif maxdg > GRADIENT_GATE_HA_BOHR:
            issues.add('R2', 'grad-gate-exceeded',
                       f'max|dg| vs gpu-fp64 = {maxdg:.3e} Eh/Bohr exceeds '
                       f'the {GRADIENT_GATE_HA_BOHR:.0e} gradient gate but '
                       'verdict is pass', label)
    elif is_num(maxdg) and maxdg > 0:
        issues.add('R2', 'grad-fp64-self-delta',
                   'fp64 reference row has nonzero '
                   'max_abs_dg_vs_gpu_fp64', label)


def _check_grad_lane_effectiveness(issues, row, label):
    '''Rule R3, gradient variant.  Two hard invariants:

    * a lane whose gradient ran fp32 kernels (gradient_mode == 'fp32')
      must show a nonzero grad_ip1 hit count -- the recorded claim and
      the instrumentation must agree (the wheel-shadowing regression
      guard);
    * an auto/fp32-grad row whose grad_ip1 is zero NEVER executed the
      fp32 gradient kernel, so it can never be verdict pass -- on the
      UKS hybrid path (df/grad/uhf.py has no fp32 branch) such a row is
      the honest fp64 fallback and must stay fail with its note.

    gradient_mode records the kernel lane that ACTUALLY executed
    (measured from the hit counter), so an 'auto' UKS row carrying
    gradient_mode='fp64', grad_ip1=0 and verdict fail is the correct,
    honest documentation of the structural gap.'''
    if row.get('status') != 'ok':
        return
    lane = (row.get('row_key') or {}).get('lane')
    if lane not in GRADIENT_LANES:
        return
    grad_mode = ((row.get('result') or {}).get('gradient_mode'))
    hits = (row.get('result') or {}).get('fp32_entry_hits') or {}
    grad_ip1 = hits.get('grad_ip1') if isinstance(hits, dict) else None
    if grad_mode not in ('fp32', 'fp64'):
        issues.add('R3', 'grad-mode-unrecorded',
                   f'result.gradient_mode must be "fp32" or "fp64", got '
                   f'{grad_mode!r}', label)
        return
    if grad_mode == 'fp32' and (not isinstance(grad_ip1, int)
                                or grad_ip1 <= 0):
        # the row claims fp32 kernels but the counter saw none
        issues.add('R3', 'grad-lane-ineffective',
                   f'{lane} row records gradient_mode fp32 but '
                   f'fp32_entry_hits.grad_ip1 = {grad_ip1!r}: the claimed '
                   'kernel lane never executed (wheel shadowing / '
                   'precision_mode never set)', label)
        if (row.get('accuracy') or {}).get('verdict') == 'pass':
            issues.add('R3', 'grad-lane-ineffective-pass',
                       'an ineffective gradient lane can never be '
                       'verdict pass', label)
    # 'auto' is NOT in this set since SP-22: the plain-auto gradient
    # resolves fp64 BY DESIGN at every nao (SP-21 measured the fp32
    # casts never faster: 0.942-1.000 above gate; SP-16: 0.99-1.15
    # below), so a zero-grad-hit auto row is the by-design lane
    # signature, not a silent fp64 fallback; its mixed precision lives
    # in the SCF head (evidenced by _check_lane_effectiveness / the
    # head counters).  The lanes that still REQUEST fp32 gradient
    # kernels keep the strict never-pass rule.
    if lane in ('fp32-grad', 'auto-verify', 'auto-strict'):
        # these lanes REQUEST fp32 gradient kernels ('auto-verify' inherits
        # the same kernel resolution: the verification pass selects the
        # solution, not the kernel lane); zero hits = the request silently
        # fell back to fp64 (structural gap) or never executed -- either
        # way the mixed-precision gradient lane is not effective and can
        # never pass
        if not isinstance(grad_ip1, int) or grad_ip1 <= 0:
            if (row.get('accuracy') or {}).get('verdict') == 'pass':
                issues.add('R3', 'grad-lane-ineffective-pass',
                           f'{lane} row has fp32_entry_hits.grad_ip1 = '
                           f'{grad_ip1!r}: the fp32 gradient kernel never '
                           'executed (on the UKS hybrid path df/grad/uhf.py '
                           'has no fp32 branch); an ineffective '
                           'mixed-precision lane can never be verdict '
                           'pass', label)
            if grad_mode == 'fp32':
                # contradiction already flagged above
                pass
    elif lane in ('fp64', 'auto-fp64-grad'):
        # these lanes must NOT have run the fp32 gradient kernel
        if isinstance(grad_ip1, int) and grad_ip1 > 0:
            issues.add('R3', 'grad-lane-contaminated',
                       f'{lane} row has fp32_entry_hits.grad_ip1 = {grad_ip1}'
                       ' > 0: the fp32 gradient kernel executed although '
                       'this lane claims fp64 gradient kernels', label)


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
    # plan section 3.4: the method contract must pin the resolved XC
    # definition (B3LYP variant etc.); 'hf' has no XC part and is exempt
    if xc_l not in ('hf', ''):
        xcd = (row.get('method') or {}).get('xc_definition')
        if not isinstance(xcd, dict) or not xcd.get('libxc_decomposition'):
            issues.add('R4', 'xc-definition-missing',
                       'DFT rows need method.xc_definition with a non-empty '
                       'libxc_decomposition (the exact functional variant: '
                       "e.g. pyscf 'b3lyp' = libxc 402 = VWN3 form, not "
                       "'b3lyp5')", label)


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


NLC_XC_MARKERS = ('b97m-v', 'b97x-v', 'vv10')


def _is_nlc_xc(xc):
    xc_l = str(xc or '').lower()
    return any(m in xc_l for m in NLC_XC_MARKERS)


def _check_nlc_consistency(issues, rows, label_of):
    '''Rule R10 (issue #7): NLC-functional rows must carry the NLC term on
    every ok lane, and the NLC piece must agree across the auto/cpu-fp64
    lanes with the fp64 reference within the precision gate.

    This is the term whose absence is the whole method wrong (plan section
    1.1: omegaB97M-V without VV10 is not omegaB97M-V), and the NLC path had
    never executed under a mixed-precision lane before issue #7 -- so the
    NLC agreement is verified explicitly instead of being inferred from
    e_tot alone.  Rows are grouped by (geom, basis, xc) exactly like the
    tier-(1) gate; lanes without an ok fp64 reference in the group get no
    NLC comparison (their total-energy verdict already handles the missing
    reference case).'''
    by_group = {}
    for row in rows:
        if row.get('status') != 'ok':
            continue
        rk = row.get('row_key') or {}
        if not _is_nlc_xc(rk.get('xc')):
            continue
        nlc = ((row.get('result') or {}).get('e_terms') or {}).get('nlc')
        if nlc is None:
            issues.add('R10', 'nlc-term-missing',
                       'NLC-functional row has null e_terms.nlc -- a dropped '
                       'VV10 term is the whole method wrong, not a small '
                       'error (was the NLC probe installed and do_nlc '
                       'true?)', label_of(row))
        by_group.setdefault(
            (rk.get('geom'), rk.get('basis'), rk.get('xc')), {})[
                rk.get('lane')] = nlc
    for group, lanes in by_group.items():
        ref = lanes.get('fp64')
        if ref is None:
            continue
        for lane in ('auto', 'fp32-screen', 'cpu-fp64', 'df-vs-conv'):
            val = lanes.get(lane)
            if val is None:
                continue
            if abs(val - ref) > PRECISION_GATE_HA:
                issues.add('R10', 'nlc-inconsistent',
                           f'{lane} lane e_terms.nlc differs from the fp64 '
                           f'reference by {abs(val - ref):.3e} Ha (> '
                           f'{PRECISION_GATE_HA:.0e} Ha): the VV10 term '
                           'disagrees across lanes at identical settings',
                           '/'.join(str(k) for k in group) + f'/{lane}')


def _check_3c_consistency(issues, rows, label_of):
    '''Rule R11 (issue #8): r2SCAN-3c correction consistency.

    The D4 dispersion and gCP correction are deterministic geometry-only
    terms evaluated by the same pyscf-dispersion library on every lane, so
    their pieces must agree across lanes -- near-bitwise identical is
    expected, and the 1e-6 Ha precision gate is only the (generous)
    consistency bound.  A disagreement means the corrections were attached
    inconsistently (the to_gpu/density_fit ordering trap of the 3c driver),
    never a mixed-precision effect.  Rows are grouped by (geom, basis, xc)
    exactly like the tier-(1) gate; lanes without an ok fp64 reference in
    the group get no comparison (their total-energy verdict already handles
    the missing-reference case).  The per-row presence of both terms is
    rule R4's job (r2scan3c-terms-missing).'''
    by_group = {}
    for row in rows:
        if row.get('status') != 'ok':
            continue
        rk = row.get('row_key') or {}
        xc_l = str(rk.get('xc') or '').lower()
        if not ('r2scan' in xc_l and '3c' in xc_l):
            continue
        e_terms = (row.get('result') or {}).get('e_terms') or {}
        by_group.setdefault(
            (rk.get('geom'), rk.get('basis'), rk.get('xc')), {})[
                rk.get('lane')] = (e_terms.get('dispersion'),
                                   e_terms.get('gcp'))
    for group, lanes in by_group.items():
        ref = lanes.get('fp64')
        if ref is None:
            continue
        for lane in ('auto', 'fp32-screen', 'cpu-fp64', 'df-vs-conv'):
            val = lanes.get(lane)
            if val is None:
                continue
            for term, term_idx in (('dispersion', 0), ('gcp', 1)):
                got, want = val[term_idx], ref[term_idx]
                if got is None or want is None:
                    continue    # R4 already flags the missing term
                if abs(got - want) > PRECISION_GATE_HA:
                    issues.add('R11', 'correction-inconsistent',
                               f'{lane} lane e_terms.{term} {got!r} differs '
                               f'from the fp64 reference {want!r} by '
                               f'{abs(got - want):.3e} Ha (> '
                               f'{PRECISION_GATE_HA:.0e} Ha): D4/gCP are '
                               'deterministic geometry-only corrections '
                               'evaluated by the same pyscf-dispersion '
                               'library on every lane -- a disagreement means '
                               'the corrections were attached inconsistently '
                               '(3c to_gpu/density_fit ordering trap), never '
                               'a precision effect',
                               '/'.join(str(k) for k in group) + f'/{lane}')


GRID_STABILITY_CLASSIFICATIONS = ('converging', 'not-converging',
                                  'single-level')


# ---------------------------------------------------------------------------
# rtx4090-gradient-v1 (issue #12) rules
# ---------------------------------------------------------------------------

GRAD_ACC_KEYS = ('max_abs_dg_vs_gpu_fp64', 'rms_dg_vs_gpu_fp64',
                 'dE_vs_gpu_fp64', 'dE_vs_cpu_fp64', 'impl_agreement_note',
                 'verdict')

# all-repeat evidence contract (issue-12 rework, Codex review 2026-10-06
# P1): the repeat record keys a gradient row must retain for EVERY timed
# sample, and the marker values result.repeat_evidence may carry
GRAD_REPEAT_KEYS_V = ('index', 'e_tot', 'converged', 'scf_cycles',
                      'gradient', 'gradient_norm', 'gradient_mode',
                      'grad_ip1_hits', 'sample_s', 'precision_switch')
REPEAT_EVIDENCE_VALUES = ('all-repeats', 'legacy-insufficient')


def _check_grad_structure(issues, row, label):
    '''G1: every ok gradient row carries the gradient payload -- the full
    (natm, 3) gradient of the WORST repeat (the all-repeat aggregate,
    rule G6), its norm (the optimizer-facing quantity), the SCF
    convergence flag and cycles, and the recorded gradient kernel mode.'''
    if row.get('status') != 'ok':
        return
    res = row.get('result') or {}
    g = res.get('gradient')
    natm = (((row.get('system') or {}).get('natm')))
    if not isinstance(g, list) or not g or \
            not all(isinstance(r, list) and len(r) == 3 and
                    all(is_num(x) for x in r) for r in g):
        issues.add('G1', 'gradient-missing',
                   'ok gradient row needs result.gradient as a non-empty '
                   '(natm, 3) numeric matrix', label)
    elif is_int(natm) and len(g) != natm:
        issues.add('G1', 'gradient-shape',
                   f'result.gradient has {len(g)} rows but system.natm = '
                   f'{natm}', label)
    for k in ('converged', 'scf_cycles'):
        if k not in res:
            issues.add('G1', 'grad-scf-fields',
                       f'result.{k} missing (the gradient row still '
                       'records the SCF convergence evidence)', label)
    norm = res.get('gradient_norm')
    if not is_num(norm) or norm < 0:
        issues.add('G1', 'gradient-norm',
                   'result.gradient_norm must be a number >= 0 (the '
                   'optimizer-facing max-norm of the worst repeat)',
                   label)
    elif isinstance(g, list) and g:
        want = max(math.sqrt(sum(x * x for x in r)) for r in g)
        if abs(norm - want) > 1e-9 + 1e-6 * abs(want):
            issues.add('G1', 'gradient-norm-inconsistent',
                       f'result.gradient_norm {norm!r} != recomputed '
                       f'{want!r} from the stored gradient', label)


def _check_grad_accuracy_semantics(issues, row, label):
    '''G2: the accuracy block carries the gradient metrics and its verdict
    respects the reference tier.  dE_vs_gpu_fp64/dE_vs_cpu_fp64 remain the
    ENERGY deltas of the same SCF (a gradient row always records its SCF
    energy too), never the gate: the gradient gate is
    max_abs_dg_vs_gpu_fp64 vs the same-implementation fp64 reference.'''
    acc = row.get('accuracy') or {}
    if row.get('status') != 'ok':
        return
    for k in GRAD_ACC_KEYS:
        if k not in acc:
            issues.add('G2', 'grad-accuracy-field',
                       f'accuracy.{k} missing on an ok gradient row', label)
    lane = (row.get('row_key') or {}).get('lane')
    if lane == 'fp64':
        # the reference lane: its own max|dg| must be null or 0
        v = acc.get('max_abs_dg_vs_gpu_fp64')
        if is_num(v) and v > 0:
            issues.add('G2', 'grad-fp64-self-delta',
                       'the fp64 reference row must carry null/zero '
                       'max_abs_dg_vs_gpu_fp64', label)
    # a lane whose group has no ok gpu-fp64 reference row cannot carry a
    # same-implementation gradient delta at all (checked groupwise in
    # _check_grad_group_consistency)


def _check_grad_group_consistency(issues, rows, label_of):
    '''G3: groupwise gradient-delta consistency.  Rows are grouped by
    (geom, basis, xc).  Within a group that has an ok fp64 reference row:
    every other ok lane's max_abs_dg_vs_gpu_fp64 and rms_dg_vs_gpu_fp64
    must be recompute-consistent with the stored gradients (stored vs
    recomputed), and a lane missing the delta while its group has a
    reference is 'grad-no-reference'.  Without an ok fp64 reference, no
    lane in the group may carry a numeric delta (it could only have been
    computed against the wrong tier, e.g. cpu-fp64).'''
    by_group = {}
    for row in rows:
        if row.get('status') != 'ok':
            continue
        rk = row.get('row_key') or {}
        by_group.setdefault(
            (rk.get('geom'), rk.get('basis'), rk.get('xc')), {})[
                rk.get('lane')] = row
    for group, lanes in by_group.items():
        label_g = '/'.join(str(k) for k in group)
        ref = lanes.get('fp64')
        g_ref = ((ref or {}).get('result') or {}).get('gradient')
        for lane, row in lanes.items():
            if lane in ('fp64', 'cpu-fp64'):
                # the reference lane carries no self-delta; the cpu-fp64
                # sentinel is a tier-(2) row -- its gradient delta vs the
                # gpu lanes is cross-implementation diagnostics (G4's
                # business), never a same-implementation gate input
                continue
            acc = row.get('accuracy') or {}
            maxdg = acc.get('max_abs_dg_vs_gpu_fp64')
            if ref is None or not isinstance(g_ref, list):
                if is_num(maxdg):
                    issues.add('G3', 'grad-delta-without-reference',
                               f'{label_g}/{lane}: max_abs_dg_vs_gpu_fp64 '
                               'is numeric but the group has no ok fp64 '
                               'reference row (the delta can only be '
                               'cross-tier; the gate is same-implementation '
                               'only)', label_of(row))
                continue
            g_lane = ((row.get('result') or {}).get('gradient'))
            if not isinstance(g_lane, list) or len(g_lane) != len(g_ref):
                # shape problems already reported by G1
                continue
            if maxdg is None:
                issues.add('G3', 'grad-no-reference',
                           f'{label_g}/{lane}: group has an ok fp64 '
                           'reference but the lane carries no '
                           'max_abs_dg_vs_gpu_fp64', label_of(row))
                continue
            want_max = max(abs(g_lane[i][j] - g_ref[i][j])
                           for i in range(len(g_ref))
                           for j in range(3))
            if abs(maxdg - want_max) > 1e-9 + 1e-6 * abs(want_max):
                issues.add('G3', 'grad-delta-inconsistent',
                           f'{label_g}/{lane}: max_abs_dg_vs_gpu_fp64 '
                           f'{maxdg!r} != recomputed {want_max!r} from the '
                           'stored gradients', label_of(row))
            rms = acc.get('rms_dg_vs_gpu_fp64')
            if is_num(rms):
                n = len(g_ref) * 3
                want_rms = math.sqrt(sum(
                    (g_lane[i][j] - g_ref[i][j]) ** 2
                    for i in range(len(g_ref)) for j in range(3)) / n)
                if abs(rms - want_rms) > 1e-9 + 1e-6 * abs(want_rms):
                    issues.add('G3', 'grad-delta-inconsistent',
                               f'{label_g}/{lane}: rms_dg_vs_gpu_fp64 '
                               f'{rms!r} != recomputed {want_rms!r}',
                               label_of(row))


def _check_grad_impl_agreement(issues, row, label):
    '''G4: level-(2) gradient deviation vs the cpu-fp64 sentinel beyond
    GRAD_IMPL_NOTE_TOL needs a named impl_agreement_note -- exactly the
    energy rule R8's discipline, with the gate values that belong to the
    gradient tier (cross-implementation gradient differences of ~1e-6
    Eh/Bohr are the historical GPU-vs-CPU level; beyond that, name the
    difference, never blame mixed precision and never pass silently).'''
    acc = row.get('accuracy') or {}
    d = acc.get('max_abs_dg_vs_cpu_fp64')
    if is_num(d) and abs(d) > GRAD_IMPL_NOTE_TOL:
        note = acc.get('impl_agreement_note')
        if not isinstance(note, str) or not note.strip():
            issues.add('G4', 'grad-impl-unexplained',
                       f'max|dg| vs cpu-fp64 = {abs(d):.3e} Eh/Bohr exceeds '
                       f'{GRAD_IMPL_NOTE_TOL:.0e} without a named '
                       'impl_agreement_note', label)


def _check_grad_allrepeat(issues, row, label, doc_rows):
    '''G6: all-repeat scientific evidence (issue-12 rework, Codex review
    2026-10-06 P1 -- "only the final repeat's energy/convergence/gradient
    survived; earlier bad samples could disappear from the gate").

    An ok gradient row must carry result.repeats with EVERY repeat's
    scientific outputs (energy, convergence, cycles, gradient, kernel
    mode, per-repeat grad_ip1 hits, timings sample), one record per timed
    sample.  The verdict 'pass' additionally requires:

      * every repeat converged (repeat-not-converged);
      * no repeat's gradient is further from the cell's fp64 reference
        than the stored aggregate claims (repeat-worse-than-aggregate):
        the old defect could hide a bad early repeat behind a good last
        one -- this recomputation catches the camouflage;
      * the row is not explicitly legacy (repeats-legacy-pass): a
        historical row without all-repeat evidence may exist in the
        document but its 'pass' is explicitly insufficient for new
        acceptance.

    Rows whose group lacks an ok fp64 reference skip the
    repeat-vs-aggregate recomputation (there is nothing to compare
    against; the verdict semantics already forbid a gate pass there).'''
    if row.get('status') != 'ok':
        return
    res = row.get('result') or {}
    lane = (row.get('row_key') or {}).get('lane')
    verdict = (row.get('accuracy') or {}).get('verdict')
    marker = res.get('repeat_evidence')
    if marker is not None and marker not in REPEAT_EVIDENCE_VALUES:
        issues.add('G6', 'repeat-evidence-marker',
                   f'result.repeat_evidence must be one of '
                   f'{REPEAT_EVIDENCE_VALUES}, got {marker!r}', label)
    reps = res.get('repeats')
    if reps is None:
        if marker == 'legacy-insufficient':
            # historical artifact: retained verbatim, but a pass on it is
            # explicitly insufficient for new acceptance
            if verdict == 'pass':
                issues.add('G6', 'repeats-legacy-pass',
                           'verdict pass on a row marked '
                           "repeat_evidence='legacy-insufficient': "
                           'historical records without all-repeat evidence '
                           'are not sufficient for new acceptance', label)
            return
        if verdict == 'pass' and lane in GRADIENT_LANES:
            issues.add('G6', 'repeats-missing',
                       "an ok gradient row needs result.repeats (every "
                       "repeat's energy/convergence/gradient; the worst "
                       'repeat is the aggregate) or the explicit '
                       "repeat_evidence='legacy-insufficient' marker",
                       label)
        return
    if not isinstance(reps, list) or not reps:
        issues.add('G6', 'repeats-missing',
                   'result.repeats must be a non-empty list of per-repeat '
                   'records', label)
        return
    samples = ((res.get('timings') or {}).get('samples_s'))
    if isinstance(samples, list) and len(samples) != len(reps):
        issues.add('G6', 'repeats-count',
                   f'len(result.repeats) = {len(reps)} but '
                   f'timings.samples_s has {len(samples)} entries -- a '
                   'truncated repeat list is malformed evidence, not an '
                   'aggregate', label)
    for r in reps:
        if not isinstance(r, dict) or any(k not in r for k in
                                          GRAD_REPEAT_KEYS_V):
            missing = [k for k in GRAD_REPEAT_KEYS_V
                       if not isinstance(r, dict) or k not in r]
            issues.add('G6', 'repeat-record',
                       f'repeat record missing {missing}: per-repeat '
                       'scientific outputs must be retained for every '
                       'sample', label)
            return
        if not isinstance(r.get('converged'), bool):
            issues.add('G6', 'repeat-record',
                       f'repeat {r.get("index")!r} converged is not a '
                       'boolean', label)
            return
        g = r.get('gradient')
        if not isinstance(g, list) or not g or \
                not all(isinstance(x, (int, float)) for row_ in g
                        for x in row_):
            issues.add('G6', 'repeat-record',
                       f'repeat {r.get("index")!r} gradient is not a '
                       'numeric (natm, 3) matrix', label)
            return
        if verdict == 'pass' and not r['converged']:
            issues.add('G6', 'repeat-not-converged',
                       f'verdict pass but repeat {r["index"]} did not '
                       'converge: every repeat must converge for the row '
                       'to pass', label)
    if verdict != 'pass':
        return
    # the aggregate must be the worst repeat: no repeat's gradient may sit
    # further from the cell's fp64 reference than the stored aggregate
    g_agg = res.get('gradient')
    if not isinstance(g_agg, list) or not g_agg:
        return
    rk = row.get('row_key') or {}
    group = (rk.get('geom'), rk.get('basis'), rk.get('xc'))
    ref = None
    for other in doc_rows:
        if other is row or other.get('status') != 'ok':
            continue
        ork = other.get('row_key') or {}
        if (ork.get('geom'), ork.get('basis'), ork.get('xc')) != group:
            continue
        if ork.get('lane') == 'fp64':
            ref = other
            break
    if ref is None:
        return
    g_ref = ((ref.get('result') or {}).get('gradient'))
    if not isinstance(g_ref, list) or len(g_ref) != len(g_agg):
        return
    stored_maxdg = None
    for r in reps:
        g = r.get('gradient')
        if not isinstance(g, list) or len(g) != len(g_ref):
            return
        mx = max(abs(g[i][j] - g_ref[i][j])
                 for i in range(len(g_ref)) for j in range(3))
        if stored_maxdg is None or mx > stored_maxdg:
            stored_maxdg = mx
    acc_maxdg = (row.get('accuracy') or {}).get('max_abs_dg_vs_gpu_fp64')
    if lane != 'fp64' and stored_maxdg is not None and \
            is_num(acc_maxdg) and stored_maxdg > acc_maxdg + 1e-12:
        issues.add('G6', 'repeat-worse-than-aggregate',
                   f'a repeat exceeds the stored max|dg| ({stored_maxdg:.3e}'
                   f' > {acc_maxdg:.3e} Eh/Bohr): the aggregate must be the '
                   'WORST repeat, never the last sample', label)


def _check_grad_energy_gate_alignment(issues, rows, label_of):
    '''G5: gradient acceptance is provisional while the energy gate has a
    known failure in the same (geom, basis, xc) cell.  The runner records
    energy_gate_reference on gradient rows; when that reference reports a
    FAILED energy gate for this cell, an 'pass' verdict must carry an
    explicit provisional note naming the inherited failure (validation
    pending), never a silent pass.'''
    for row in rows:
        ref = ((row.get('result') or {}).get('energy_gate_reference') or {})
        if not isinstance(ref, dict) or not ref.get('gate_failed_cells'):
            continue
        if (row.get('accuracy') or {}).get('verdict') != 'pass':
            continue
        note = ((row.get('accuracy') or {}).get('impl_agreement_note')) or ''
        # the provisional marker may live on accuracy.verdict_note
        vnote = ((row.get('accuracy') or {}).get('verdict_note')) or ''
        if 'provisional' not in note.lower() and \
                'provisional' not in vnote.lower():
            issues.add('G5', 'grad-pass-with-failed-energy-gate',
                       'verdict pass on a cell whose energy gate has a '
                       'known failure (see result.energy_gate_reference) '
                       'must carry an explicit provisional/validation-pending '
                       'note', label_of(row))


def validate_gradient_document(doc, evidence_root=None):
    '''Validate one rtx4090-gradient-v1 document; returns issue dicts.'''
    issues = Issues()
    if not isinstance(doc, dict):
        issues.add('structure', 'not-a-document',
                   'benchmark file is not a JSON object')
        return issues.items
    if doc.get('schema_version') != GRADIENT_SCHEMA_VERSION:
        issues.add('structure', 'schema-version',
                   f'schema_version must be {GRADIENT_SCHEMA_VERSION!r}, '
                   f'got {doc.get("schema_version")!r}')
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
        label = _check_structure(issues, row, seen, lanes=GRADIENT_LANES)
        if row.get('status') == 'ok':
            _check_timings(issues, row, label,
                           gpu_lanes=GRADIENT_GPU_LANES, grad=True)
        _check_status_semantics(issues, row, label)
        _check_status_semantics_grad(issues, row, label)
        _check_grad_lane_effectiveness(issues, row, label)
        _check_method_completeness(issues, row, label)
        _check_tier_isolation(issues, row, label)
        _check_memory(issues, row, label)
        _check_impl_agreement(issues, row, label)
        _check_grad_structure(issues, row, label)
        _check_grad_accuracy_semantics(issues, row, label)
        _check_grad_impl_agreement(issues, row, label)
        _check_grad_allrepeat(issues, row, label, rows)
        _check_grad_strict_match(issues, row, label)
        _check_evidence(issues, row, evidence_root)
    _check_nlc_consistency(issues, rows, _row_label)
    _check_3c_consistency(issues, rows, _row_label)
    _check_grad_group_consistency(issues, rows, _row_label)
    _check_grad_energy_gate_alignment(issues, rows, _row_label)
    return issues.items


def _check_grad_strict_match(issues, row, label):
    '''G7 (LC-76): the 'auto-strict' lane claims the strict fp64-reference
    MATCHING contract -- every repeat must carry a complete certification
    (stability_verification.policy == 'auto+strict' with
    strict_match_verified True).  An incomplete certification (reference
    not converged / nonfinite, nonfinite dm comparison, skipped
    verification) can never be verdict pass: the lane then honestly
    reports the converged-but-uncertified auto solution as a fail, never
    a silent selection.'''
    if row.get('status') != 'ok':
        return
    lane = (row.get('row_key') or {}).get('lane')
    if lane != 'auto-strict':
        return
    reps = (row.get('result') or {}).get('repeats')
    if not isinstance(reps, list) or not reps:
        return
    verdict = (row.get('accuracy') or {}).get('verdict')
    for r in reps:
        sv = r.get('stability_verification')
        if not isinstance(sv, dict) or \
                sv.get('policy') != 'auto+strict':
            if verdict == 'pass':
                issues.add('G7', 'strict-match-record-missing',
                           f'repeat {r.get("index")!r} carries no '
                           "auto+strict stability_verification record: "
                           'the strict-match certification is missing '
                           '(the lane cannot claim a certified fp64-lane '
                           'match)', label)
            continue
        if sv.get('strict_match_verified') is not True:
            if verdict == 'pass':
                issues.add(
                    'G7', 'strict-match-incomplete-pass',
                    f'repeat {r.get("index")!r} strict_match_verified is '
                    f'{sv.get("strict_match_verified")!r} '
                    f'({sv.get("strict_match_incomplete_reason") or sv.get("skipped") or "reason unrecorded"}): '
                    'an incomplete strict-match certification can never '
                    'be verdict pass -- the row must fail closed with '
                    'the exact reason', label)


def _check_grid_stability(issues, doc):
    '''Rule R9 (optional blocks, issues #6/#7): internal consistency of the
    grid-stability / nlcgrids-stability scans.  Diagnostic only -- never an
    acceptance gate: the level-(1) precision gate compares lanes at
    identical grids, where the grid error cancels exactly, so a
    grid-stability delta can neither pass nor fail a row.'''
    for block_name in ('grid_stability', 'nlcgrids_stability'):
        _check_one_grid_block(issues, doc, block_name)


def _check_one_grid_block(issues, doc, block_name):
    gs = doc.get(block_name)
    if gs is None:
        return
    if not isinstance(gs, dict):
        issues.add('R9', 'grid-stability-type',
                   f'{block_name} must be an object when present')
        return
    prod = gs.get('production_level')
    if not is_int(prod):
        issues.add('R9', 'grid-stability-production',
                   f'{block_name}.production_level must be an integer')
    entries = gs.get('entries')
    if not isinstance(entries, list) or not entries:
        issues.add('R9', 'grid-stability-entries',
                   f'{block_name}.entries must be a non-empty list')
        return
    levels = []
    for en in entries:
        if not isinstance(en, dict):
            issues.add('R9', 'grid-stability-entry',
                       'grid_stability entries must be objects')
            continue
        lv = en.get('level')
        if is_int(lv):
            levels.append(lv)
        else:
            issues.add('R9', 'grid-stability-entry',
                       f'entry.level must be an integer, got {lv!r}')
        if not is_int(en.get('ngrids')) and en.get('ngrids') is not None:
            issues.add('R9', 'grid-stability-entry',
                       f'entry.ngrids must be an integer or null, got '
                       f'{en.get("ngrids")!r}')
        if not is_num(en.get('e_tot')) and en.get('error') is None:
            issues.add('R9', 'grid-stability-entry',
                       f'entry (level {lv!r}) needs e_tot or an error record')
        if not isinstance(en.get('converged'), bool):
            issues.add('R9', 'grid-stability-entry',
                       f'entry (level {lv!r}) needs a boolean converged')
    if is_int(prod) and prod not in levels:
        issues.add('R9', 'grid-stability-production',
                   f'production_level {prod} is not among the scanned '
                   f'levels {levels}')
    if gs.get('classification') not in GRID_STABILITY_CLASSIFICATIONS:
        issues.add('R9', 'grid-stability-classification',
                   f'grid_stability.classification must be one of '
                   f'{GRID_STABILITY_CLASSIFICATIONS}, got '
                   f'{gs.get("classification")!r}')
    note = gs.get('note')
    if not isinstance(note, str) or not note.strip():
        issues.add('R9', 'grid-stability-note',
                   f'{block_name}.note must state the recorded grid error '
                   '(explicitly recorded, never silently dropped)')
    # recompute the stored deltas from the stored energies (the block must
    # be self-consistent evidence, not a transcription)
    ok_entries = [en for en in entries if isinstance(en, dict)
                  and is_num(en.get('e_tot'))]
    by_level = {en['level']: en for en in ok_entries if is_int(en.get('level'))}
    srt = sorted(by_level.values(), key=lambda en: en['level'])
    for i, en in enumerate(srt):
        if 'dE_vs_prev_level' in en:
            want = None if i == 0 else en['e_tot'] - srt[i - 1]['e_tot']
            got = en['dE_vs_prev_level']
            if want is None or got is None:
                if want != got:
                    issues.add('R9', 'grid-stability-delta',
                               f'entry level {en["level"]} dE_vs_prev_level '
                               f'{got!r} inconsistent (expected {want!r})')
            elif abs(got - want) > 1e-12:
                issues.add('R9', 'grid-stability-delta',
                           f'entry level {en["level"]} dE_vs_prev_level '
                           f'{got!r} != recomputed {want!r}')
        if 'dE_vs_production' in en and is_int(prod) and prod in by_level:
            want = en['e_tot'] - by_level[prod]['e_tot']
            got = en['dE_vs_production']
            if got is None or abs(got - want) > 1e-12:
                issues.add('R9', 'grid-stability-delta',
                           f'entry level {en["level"]} dE_vs_production '
                           f'{got!r} != recomputed {want!r}')


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

def validate_document(doc, evidence_root=None, gradient=False):
    '''Validate one benchmark document; returns a list of issue dicts.
    ``gradient`` selects the rtx4090-gradient-v1 contract (issue #12);
    dispatch is normally automatic via validate_file/validate_any on the
    document's schema_version.'''
    if gradient:
        return validate_gradient_document(doc, evidence_root=evidence_root)
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
    _check_nlc_consistency(issues, rows, _row_label)
    _check_3c_consistency(issues, rows, _row_label)
    _check_grid_stability(issues, doc)
    return issues.items


def validate_file(path, evidence_root=None):
    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    return validate_any(doc, evidence_root=evidence_root)


def validate_any(doc, evidence_root=None):
    '''Dispatch on schema_version: rtx4090-energy-v1 documents run the
    energy contract, rtx4090-gradient-v1 documents (issue #12) run the
    gradient contract.'''
    if isinstance(doc, dict) and \
            doc.get('schema_version') == GRADIENT_SCHEMA_VERSION:
        return validate_gradient_document(doc, evidence_root=evidence_root)
    return validate_document(doc, evidence_root=evidence_root)


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='schema_validator',
        description='Validate an RTX 4090 benchmark JSON against '
                    'rtx4090-energy-v1 or rtx4090-gradient-v1 '
                    '(docs/consumer-gpu-plan.md section 4 / issue #12)')
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
        schema = SCHEMA_VERSION
        try:
            with open(args.file, encoding='utf-8') as f:
                schema = json.load(f).get('schema_version', SCHEMA_VERSION)
        except (OSError, ValueError):
            pass
        print(f'{args.file}: VALID ({schema})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

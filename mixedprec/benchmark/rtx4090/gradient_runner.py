#!/usr/bin/env python3
"""RTX 4090 gradient benchmark runner producing ``rtx4090-gradient-v1``
documents (issue #12 of docs/consumer-gpu-plan.md).

The gradient problem the plan names is that a gradient error conflates two
different sources: the SCF convergence-path difference (an auto-lane SCF
converges through an fp32 head to a slightly different density/mo set than
a pure-fp64 SCF) and the gradient-kernel precision difference (the fp32
sum_ejk_int3c2e_ip1 kernel itself).  This runner therefore runs FOUR GPU
lanes per cell -- a homotopy-style decomposition:

  fp64            fp64 SCF + fp64 gradient kernels        (reference)
  auto            auto SCF + fp32 gradient kernels        (production:
                  grad/rhf.py _resolve_precision resolves 'auto' -> fp32
                  for a gradient: no iteration, no fp64 tail to converge)
  fp32-grad       fp64 SCF + fp32 gradient kernels        (kernel error
                  alone: g.precision_mode='fp32' override)
  auto-fp64-grad  auto SCF + fp64 gradient kernels        (SCF-path error
                  alone: g.precision_mode='fp64' override)

so max|dg(auto)| decomposes approximately into the fp32-grad and
auto-fp64-grad deltas (plus a small interaction).  A cpu-fp64 sentinel row
(tier 2) completes each cell.

Everything else follows the energy runner's discipline (rtx4090_runner.py):
warmup separated from >= 3 independent samples (fresh mol + mf + gradient
object each), deviceSynchronize before/after every phase, row-key merge
across invocations, memory discipline with headroom budget, fp32
entry-hit instrumentation (the gradient rows must show nonzero grad_ip1
on the fp32-kernel lanes -- the wheel-shadowing regression guard), the
3c/NLC method contracts, evidence anchoring by sha256, and schema
validation at the end.

Gradient rows additionally carry result.gradient (the LAST sample's
(natm,3) matrix -- the quantity the derived columns recompute from),
result.gradient_norm, result.gradient_mode and the SCF evidence
(converged/cycles/e_terms/e_tot/precision_switch).  The gradient gate is
max|dg| vs the same-implementation gpu-fp64 reference <= 3e-5 Eh/Bohr
(one order below geomeTRIC's default gmax 3e-4).

Energy-gate alignment (G5): --energy-gate-doc accepts the issue #5-#8
energy documents; cells whose auto lane failed the 1e-6 Ha energy gate
get result.energy_gate_reference and their gradient verdicts carry an
explicit provisional/validation-pending note.  Under the 2026-10-04 user
policy gradients are developed experimentally despite unresolved energy
findings, but never silently.

Usage (methanol b3lyp smoke):
    PYTHONPATH=<repo> python gradient_runner.py \
        --out /data/results/.../b3lyp_methanol_grad.json \
        --lane fp64 auto fp32-grad auto-fp64-grad cpu-fp64 --samples 3
"""

import argparse
import copy
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from types import MethodType

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(HERE))))
sys.path.insert(0, HERE)

import schema_validator                                    # noqa: E402
import rtx4090_runner as energy_runner                     # noqa: E402

SCHEMA_VERSION = 'rtx4090-gradient-v1'
GRADIENT_GATE = schema_validator.GRADIENT_GATE_HA_BOHR
GRAD_IMPL_NOTE_TOL = schema_validator.GRAD_IMPL_NOTE_TOL

GRAD_LANES = ('fp64', 'auto', 'fp32-grad', 'auto-fp64-grad', 'cpu-fp64')
# gradient-kernel precision per lane: (scf precision_mode, grad
# precision_mode override); None = inherit the base object's mode
LANE_SPEC = {
    'fp64': ('fp64', 'fp64'),
    'auto': ('auto', None),          # _resolve_precision: auto -> fp32
    'fp32-grad': ('fp64', 'fp32'),
    'auto-fp64-grad': ('auto', 'fp64'),
    'cpu-fp64': (None, None),
}


def sha256_file(path):
    return energy_runner.sha256_file(path)


def _num(v):
    return energy_runner._num(v)


# ---------------------------------------------------------------------------
# one independent GPU run: SCF + gradient, fully synced
# ---------------------------------------------------------------------------

def _gpu_grad_point(geom, basis, xc, auxbasis, args, scf_mode, grad_mode,
                    probe=None):
    '''Fresh GPU mol + mf + gradient object; SCF then gradient, each phase
    device-synced.  Returns (e, converged, cycles, nlc, summary, grad,
    grad_mode_effective, grad_hits, phases) with phases {t_mol_s,
    t_cderi_s, t_scf_s, t_grad_s}.

    grad_mode_effective is MEASURED, not requested: the delta of the
    probe's grad_ip1 counter across the gradient phase says whether any
    fp32 gradient-kernel branch actually executed.  This distinction is
    load-bearing on the UKS hybrid path, where df/grad/uhf.py has no fp32
    branch -- an 'auto' gradient silently runs fp64 kernels there, and
    recording the requested mode would mislabel the row.'''
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    mol, mf = energy_runner._gpu_mol_mf(geom, basis, xc, True, auxbasis,
                                         args, scf_mode, probe)
    if scf_mode is not None:
        mf.precision_mode = scf_mode
    cp.cuda.runtime.deviceSynchronize()
    t_mol = time.perf_counter() - t0

    t1 = time.perf_counter()
    mf.with_df.build()
    cp.cuda.runtime.deviceSynchronize()
    t_cderi = time.perf_counter() - t1

    t1 = time.perf_counter()
    e = mf.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_scf = time.perf_counter() - t1
    nlc = probe.last_nlc_energy if probe is not None else None
    converged = bool(mf.converged)
    cycles = int(getattr(mf, 'cycles', 0) or 0)
    summary = dict(getattr(mf, 'scf_summary', {}) or {})

    g = mf.nuc_grad_method()
    if energy_runner.is_3c_xc(xc):
        # the 3c driver's gradient hook: D4+gCP gradients (pyscf-dispersion,
        # deterministic, geometry-only); must be attached AFTER every
        # transform, exactly like the energy-side attach_3c_corrections
        g.get_dispersion = MethodType(_r2scan3c_get_dispersion_grad, g)
        g.auxbasis_response = True
    if grad_mode is not None:
        g.precision_mode = grad_mode
    hits_before = probe.hits['grad_ip1'] if probe is not None else 0
    t1 = time.perf_counter()
    de = g.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_grad = time.perf_counter() - t1
    grad_hits = (probe.hits['grad_ip1'] - hits_before) if probe is not None \
        else 0
    de = cp.asnumpy(de)
    grad_mode_effective = 'fp32' if grad_hits > 0 else 'fp64'
    # observe which kernel lane actually executed (the probe counts
    # precision.get_precision() == 'fp32' entries attributed to grad
    # modules while g.kernel() ran)
    mol = mf = g = None
    cp.get_default_memory_pool().free_all_blocks()
    return (float(e), converged, cycles, nlc, summary, de,
            grad_mode_effective, grad_hits,
            {'t_mol_s': t_mol, 't_cderi_s': t_cderi, 't_scf_s': t_scf,
             't_grad_s': t_grad})


def _r2scan3c_get_dispersion_grad(mf_grad, disp=None, with_3body=None,
                                   verbose=None):
    '''g.get_dispersion for r2SCAN-3c gradients: D4 + gCP gradients, each
    recorded separately under the same dedicated scf_summary keys the
    energy-side hook uses (dispersion_d4 / gcp), mirroring
    gen_disp_grad_fun of the reference 3c driver (which sums the two).'''
    from pyscf.dispersion import dftd4, gcp
    import numpy as np
    mol = mf_grad.base.mol
    d4 = dftd4.DFTD4Dispersion(
        mol, xc=energy_runner.R2SCAN3C_D4_XC_KEY,
        atm=energy_runner.R2SCAN3C_D4_PARAMS['atm'],
        ga=energy_runner.R2SCAN3C_D4_PARAMS['ga'],
        gc=energy_runner.R2SCAN3C_D4_PARAMS['gc'])
    d4.set_param(energy_runner.R2SCAN3C_D4_PARAMS['s8'],
                 energy_runner.R2SCAN3C_D4_PARAMS['a1'],
                 energy_runner.R2SCAN3C_D4_PARAMS['a2'],
                 s9=energy_runner.R2SCAN3C_D4_PARAMS['s9'])
    res = d4.get_dispersion(grad=True)
    g_d4 = np.asarray(res['gradient'])
    gcp_model = gcp.GCP(mol, method=energy_runner.R2SCAN3C_GCP_METHOD)
    res = gcp_model.get_counterpoise(grad=True)
    g_gcp = np.asarray(res['gradient'])
    return g_d4 + g_gcp


def _cpu_grad_point(geom, basis, xc, auxbasis, args, nlc_capture=None):
    '''CPU PySCF sentinel: DF fp64 SCF + gradient, same independent-run
    discipline, no CUDA compute.'''
    from pyscf import scf as cpu_scf, dft as cpu_dft
    three_c = energy_runner.is_3c_xc(xc)
    xc_run = energy_runner.resolve_3c(xc) or xc
    t0 = time.perf_counter()
    mol = energy_runner.make_mol(geom, basis, args.charge, args.spin)
    if xc_run == 'hf':
        mf = (cpu_scf.UHF if mol.spin else cpu_scf.RHF)(mol)
    else:
        mf = (cpu_dft.UKS if mol.spin else cpu_dft.RKS)(mol, xc=xc_run)
    mf = mf.density_fit(auxbasis=auxbasis)
    energy_runner._apply_common(mf, args, True)
    if three_c:
        energy_runner.attach_3c_corrections(mf)
    t_mol = time.perf_counter() - t0
    t1 = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t1
    nlc = nlc_capture.last_nlc_energy if nlc_capture is not None else None
    converged = bool(mf.converged)
    cycles = int(getattr(mf, 'cycles', 0) or 0)
    summary = dict(getattr(mf, 'scf_summary', {}) or {})

    g = mf.nuc_grad_method()
    if three_c:
        # CPU-side 3c gradient hook, mirroring the GPU lane (the CPU
        # kernel calls base.do_disp() then self.get_dispersion() exactly
        # like the GPU one; without the hook the sentinel would silently
        # drop D4/gCP gradients)
        g.get_dispersion = MethodType(_r2scan3c_get_dispersion_grad, g)
    t1 = time.perf_counter()
    de = g.kernel()
    t_grad = time.perf_counter() - t1
    de = de if hasattr(de, 'tolist') else de
    return (float(e), converged, cycles, nlc, summary,
            __import__('numpy').asarray(de),
            'fp64',
            {'t_mol_s': t_mol, 't_cderi_s': 0.0, 't_scf_s': t_scf,
             't_grad_s': t_grad})


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def _grad_timing_block(warmup_s, samples, scf_times, grad_times, lane):
    samples = [round(s, 6) for s in samples]
    if lane == 'cpu-fp64':
        sync = ('cpu lane: no CUDA device; wall-clock samples, each an '
                'independent fresh mol + mf + gradient run')
    else:
        sync = ('device-sync before/after each phase; each sample is an '
                'independent fresh mol + mf + gradient run including CDERI '
                'build, SCF and the gradient evaluation')
    return {'warmup_s': round(warmup_s, 6), 'samples_s': samples,
            'median_s': round(statistics.median(samples), 6),
            'min_s': min(samples), 'max_s': max(samples),
            'end_to_end_s': round(statistics.median(samples), 6),
            't_cderi_s': 0.0,
            't_scf_s': round(statistics.median(scf_times), 6)
            if scf_times else None,
            't_grad_s': round(statistics.median(grad_times), 6)
            if grad_times else None,
            't_xc_grid_s': None, 't_jk_s': None, 't_nlc_s': None,
            't_disp_s': None, 'sync': sync}


def _grad_mem_block(sampler, free_mib, total_mib, headroom, peak_alloc,
                    within_budget, explanation=None):
    return energy_runner._mem_block(sampler, free_mib, total_mib, headroom,
                                    peak_alloc, within_budget, explanation)


def _gradient_norm(g):
    '''max atom-norm of the (natm,3) gradient -- the optimizer-facing
    quantity (geomeTRIC gmax works on per-atom norms).'''
    return float(max((x * x + y * y + z * z) ** 0.5 for x, y, z in g))


def run_gpu_grad_cell(geom, basis, xc, lane, args, log):
    '''One GPU gradient row (fp64 / auto / fp32-grad / auto-fp64-grad).'''
    import cupy as cp
    scf_mode, grad_mode = LANE_SPEC[lane]
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': lane}

    if args.min_free_mib is not None and \
            free_mib - args.headroom_mib < args.min_free_mib:
        log(f'{lane}: skipped (headroom-budget): free {free_mib:.0f} MiB - '
            f'headroom {args.headroom_mib:.0f} MiB < required '
            f'{args.min_free_mib:.0f} MiB')
        return energy_runner._skipped_row(row_key, args, 'headroom-budget',
                                           free_mib, total_mib)

    probe = energy_runner.PrecisionProbe()
    probe.install()
    sampler = energy_runner.MemorySampler()
    try:
        with sampler:
            try:
                log(f'{lane}: warmup ...')
                warm = _gpu_grad_point(geom, basis, xc, args.auxbasis,
                                       args, scf_mode, grad_mode)
                warmup_s = sum(warm[8].values())
            except Exception as exc:
                return _grad_failed_row(row_key, args, exc, sampler,
                                        free_mib, total_mib, lane, log)
            samples, scf_times, grad_times = [], [], []
            e = converged = cycles = nlc = None
            summary, grad, grad_mode_resolved = {}, None, None
            for i in range(args.samples):
                log(f'{lane}: sample {i + 1}/{args.samples} ...')
                try:
                    (e, converged, cycles, nlc, summary, grad,
                     grad_mode_resolved, _ghits, phases) = _gpu_grad_point(
                        geom, basis, xc, args.auxbasis, args, scf_mode,
                        grad_mode, probe=probe)
                except Exception as exc:
                    return _grad_failed_row(row_key, args, exc, sampler,
                                            free_mib, total_mib, lane, log)
                samples.append(sum(phases.values()))
                scf_times.append(phases['t_scf_s'])
                grad_times.append(phases['t_grad_s'])
    finally:
        probe.remove()

    peak_alloc = max(free_mib - (sampler.min_free_mib or free_mib), 0.0)
    within, expl = energy_runner._budget_check(peak_alloc, total_mib,
                                                args.headroom_mib)
    mol, mf = energy_runner._gpu_mol_mf(geom, basis, xc, True,
                                         args.auxbasis, args, None)
    row = {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'system': energy_runner._system_block(geom, mol),
        'method': energy_runner._method_block(mol, mf, xc, basis,
                                               args.auxbasis, True, args),
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': e, 'converged': converged, 'scf_cycles': cycles,
            'gradient': grad.tolist(), 'gradient_norm':
                _gradient_norm(grad),
            'gradient_mode': grad_mode_resolved,
            'e_terms': energy_runner._e_terms(summary, nlc),
            'timings': _grad_timing_block(warmup_s, samples, scf_times,
                                          grad_times, lane),
            'mem': _grad_mem_block(sampler, free_mib, total_mib,
                                   args.headroom_mib, peak_alloc, within,
                                   expl),
            'fp32_entry_hits': dict(probe.hits),
            'precision_switch': probe.switch_info(
                'auto' if lane in ('auto', 'auto-fp64-grad') else lane),
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    log(f'{lane}: ok E={e:.12f} conv={converged} cyc={cycles} '
        f'gnorm={row["result"]["gradient_norm"]:.3e} '
        f'gmode={grad_mode_resolved} peak={peak_alloc:.0f}MiB '
        f'grad_ip1={probe.hits["grad_ip1"]} '
        f'switch={row["result"]["precision_switch"]}')
    return row


def _grad_failed_row(row_key, args, exc, sampler, free_mib, total_mib,
                     lane, log):
    return energy_runner._failed_row(row_key, args, exc, sampler, free_mib,
                                      total_mib, lane, log)


def run_cpu_grad_cell(geom, basis, xc, args, log):
    '''cpu-fp64 sentinel gradient row: independent runs, no CUDA compute.'''
    import cupy as cp
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': 'cpu-fp64'}
    nlc_cap = energy_runner.CpuNlcCapture()
    nlc_cap.install()
    try:
        try:
            log('cpu-fp64: warmup ...')
            _w = _cpu_grad_point(geom, basis, xc, args.auxbasis, args,
                                 nlc_capture=nlc_cap)
            warmup_s = sum(_w[7].values())
            samples, scf_times, grad_times = [], [], []
            e = converged = cycles = nlc = None
            summary, grad = {}, None
            for i in range(args.samples):
                log(f'cpu-fp64: sample {i + 1}/{args.samples} ...')
                (e, converged, cycles, nlc, summary, grad, _gm,
                 phases) = _cpu_grad_point(geom, basis, xc, args.auxbasis,
                                           args, nlc_capture=nlc_cap)
                samples.append(sum(phases.values()))
                scf_times.append(phases['t_scf_s'])
                grad_times.append(phases['t_grad_s'])
        except Exception as exc:
            log(f'cpu-fp64: error: {exc!r}')
            return {'row_key': row_key, 'pilot_stage': args.pilot_stage,
                    'status': 'error', 'skip_reason': None, 'system': None,
                    'method': None,
                    'result': {'errors': {'type': type(exc).__name__,
                                          'message': repr(exc)[:2000]}},
                    'accuracy': {'dE_vs_gpu_fp64': None,
                                 'dE_vs_cpu_fp64': None,
                                 'impl_agreement_note': None,
                                 'verdict': 'fail'},
                    'cross_program': [], 'evidence': None}
    finally:
        nlc_cap.remove()

    mol = energy_runner.make_mol(geom, basis, args.charge, args.spin)
    from pyscf import scf as cpu_scf, dft as cpu_dft
    probe_xc = energy_runner.resolve_3c(xc) or xc
    if probe_xc == 'hf':
        mf = (cpu_scf.UHF if mol.spin else cpu_scf.RHF)(mol)
    else:
        mf = (cpu_dft.UKS if mol.spin else cpu_dft.RKS)(mol, xc=probe_xc)
    mf = mf.density_fit(auxbasis=args.auxbasis)
    energy_runner._apply_common(mf, args, True)
    row = {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'system': energy_runner._system_block(geom, mol),
        'method': energy_runner._method_block(mol, mf, xc, basis,
                                               args.auxbasis, True, args),
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': e, 'converged': converged, 'scf_cycles': cycles,
            'gradient': grad.tolist(), 'gradient_norm':
                _gradient_norm(grad),
            'gradient_mode': 'fp64',
            'e_terms': energy_runner._e_terms(summary, nlc),
            'timings': _grad_timing_block(warmup_s, samples, scf_times,
                                          grad_times, 'cpu-fp64'),
            'mem': _grad_mem_block(energy_runner.MemorySampler(), free_mib,
                                   total_mib, args.headroom_mib, 0.0, True),
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                'grad_ip1': 0, 'other': 0},
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    log(f'cpu-fp64: ok E={e:.12f} conv={converged} cyc={cycles} '
        f'gnorm={row["result"]["gradient_norm"]:.3e}')
    return row


# ---------------------------------------------------------------------------
# derived columns (pure; unit-testable without GPU)
# ---------------------------------------------------------------------------

def _grad_stats(g_lane, g_ref):
    import math
    n = len(g_ref) * 3
    mx = max(abs(g_lane[i][j] - g_ref[i][j])
             for i in range(len(g_ref)) for j in range(3))
    rms = math.sqrt(sum((g_lane[i][j] - g_ref[i][j]) ** 2
                        for i in range(len(g_ref))
                        for j in range(3)) / n)
    return mx, rms


def _maxdg(g_lane, g_ref):
    return _grad_stats(g_lane, g_ref)[0]


def derive_grad(rows, gate_failed_cells=None):
    '''Fill gradient accuracy blocks.

    Tier discipline (exactly the energy runner's): max|dg| and rms|dg| are
    computed ONLY against the same cell's ok gpu-fp64 row; the cpu-fp64
    row is the tier-(2) reference -- other lanes report max|dg| vs it as
    diagnostics (impl agreement), never as the gate.  Verdict pass
    requires converged AND max|dg| <= 3e-5 Eh/Bohr.  A cell listed in
    gate_failed_cells (the inherited issue #7 energy failure) marks every
    gradient verdict provisional: validation pending, never silently
    accepted.'''
    by_key = energy_runner.key_of
    for row in rows:
        acc = row.setdefault('accuracy', {})
        acc.setdefault('max_abs_dg_vs_gpu_fp64', None)
        acc.setdefault('rms_dg_vs_gpu_fp64', None)
        acc.setdefault('dE_vs_gpu_fp64', None)
        acc.setdefault('dE_vs_cpu_fp64', None)
        acc.setdefault('impl_agreement_note', None)
        if row.get('status') != 'ok':
            if acc.get('verdict') is None:
                acc['verdict'] = 'skipped' \
                    if row.get('status') == 'skipped' else 'fail'
            continue
        base = by_key(row)[:3]
        lane = row['row_key']['lane']
        e = row['result']['e_tot']
        converged = row['result']['converged']
        ref64 = next((r for r in rows if by_key(r)[:3] == base
                      and r['row_key']['lane'] == 'fp64'
                      and r.get('status') == 'ok'), None)
        refcpu = next((r for r in rows if by_key(r)[:3] == base
                       and r['row_key']['lane'] == 'cpu-fp64'
                       and r.get('status') == 'ok'), None)
        if lane == 'fp64':
            acc['dE_vs_gpu_fp64'] = None
        elif ref64 is not None:
            acc['dE_vs_gpu_fp64'] = e - ref64['result']['e_tot']
        if refcpu is not None and lane != 'cpu-fp64':
            acc['dE_vs_cpu_fp64'] = e - refcpu['result']['e_tot']
            # R8 energy attribution (same story as the issue #5-#8 energy
            # documents): a gpu-vs-cpu energy deviation beyond the matched
            # -settings level needs the named note, and for NLC functionals
            # the runner reuses the energy runner's measured VV10 two-
            # implementation attribution when the NLC piece dominates the
            # deviation (the note function itself refuses lanes that do
            # not sit on their own GPU fp64 reference)
            if abs(acc['dE_vs_cpu_fp64']) > \
                    schema_validator.IMPL_NOTE_TOL_HA and \
                    not acc.get('impl_agreement_note'):
                note = energy_runner._nlc_impl_note(
                    row, refcpu, acc.get('dE_vs_gpu_fp64'))
                if note is not None:
                    acc['impl_agreement_note'] = note
        # gradient deltas: same-implementation gate vs gpu-fp64
        if lane != 'fp64' and ref64 is not None:
            g_lane = row['result']['gradient']
            g_ref = ref64['result']['gradient']
            mx, rms = _grad_stats(g_lane, g_ref)
            acc['max_abs_dg_vs_gpu_fp64'] = mx
            acc['rms_dg_vs_gpu_fp64'] = rms
        # tier-(2) gradient diagnostics vs the cpu sentinel
        if lane != 'cpu-fp64' and refcpu is not None:
            g_lane = row['result']['gradient']
            g_cpu = refcpu['result']['gradient']
            acc['max_abs_dg_vs_cpu_fp64'] = _maxdg(g_lane, g_cpu)
        # verdicts
        if lane == 'cpu-fp64':
            acc['verdict'] = 'no_reference' if converged else 'fail'
            continue
        if not converged:
            acc['verdict'] = 'fail'
        elif ref64 is None:
            acc['verdict'] = 'no_reference'
        elif lane == 'fp64':
            acc['verdict'] = 'pass'
        elif acc['max_abs_dg_vs_gpu_fp64'] <= GRADIENT_GATE:
            acc['verdict'] = 'pass'
        else:
            acc['verdict'] = 'fail'
        # an auto/fp32-grad lane that REQUESTED fp32 gradient kernels but
        # measured zero grad_ip1 hits never ran the mixed-precision
        # gradient: the lane is ineffective (structural gap, e.g. the UKS
        # hybrid path df/grad/uhf.py has no fp32 branch) and must stay
        # fail with the gap named -- never a silent fp64 fallback
        # presented as a mixed-precision result
        if lane in ('auto', 'fp32-grad') and \
                row['result'].get('gradient_mode') != 'fp32':
            acc['verdict'] = 'fail'
            acc['verdict_note'] = (
                'mixed-precision gradient lane ineffective: fp32 gradient '
                'kernels were requested but the measured grad_ip1 hit '
                'count is zero -- the gradient silently ran fp64 kernels '
                '(structural gap; on the UKS hybrid path '
                'gpu4pyscf/df/grad/uhf.py has no fp32 branch, unlike the '
                'RKS df/grad/rhf.py path). The row is retained as an '
                'honest fp64-fallback measurement, never a mixed-precision '
                'pass')
        # energy-gate alignment (G5): an unresolved energy failure in this
        # cell makes any gradient verdict explicitly provisional
        cell_key = '/'.join(str(k) for k in base)
        if gate_failed_cells and cell_key in gate_failed_cells \
                and acc['verdict'] == 'pass':
            acc['verdict_note'] = (
                'provisional: the energy gate of this cell has an '
                'unresolved failure (inherited from issue #7: '
                f'{gate_failed_cells[cell_key]}); gradient validation '
                'pending the energy fix -- never accepted gradient '
                'accuracy')
    # G4 notes: cpu gradient deviations beyond the tolerance get the named
    # two-implementation note (the same attribution the energy runner
    # records for NLC; here the whole gradient pipeline differs)
    for row in rows:
        acc = row.get('accuracy') or {}
        d = acc.get('max_abs_dg_vs_cpu_fp64')
        if row.get('status') == 'ok' and d is not None \
                and abs(d) > GRAD_IMPL_NOTE_TOL:
            acc['impl_agreement_note'] = (
                'matched-settings implementation difference: the CPU pyscf '
                'and gpu4pyscf gradient pipelines are two independent '
                'implementations (different screening defaults, '
                'integral batching and summation order); measured '
                f'max|dg| vs cpu-fp64 = {abs(d):.3e} Eh/Bohr. Level-(2) '
                'diagnostic only, never a mixed-precision error: the '
                'level-(1) gradient gate compares lanes within the same '
                'implementation')
    return rows


def read_energy_gate_documents(paths, log):
    '''Read issue #5-#8 energy documents and collect the cells whose auto
    lane FAILED the 1e-6 Ha energy gate (honest fail rows only -- the
    2026-10-04 policy keeps them unresolved and visible here).  Returns
    {cell_key: worst dE note}.'''
    failed = {}
    for path in paths or []:
        try:
            with open(path, encoding='utf-8') as f:
                doc = json.load(f)
        except (OSError, ValueError) as exc:
            log(f'energy-gate: cannot read {path}: {exc!r} (ignored)')
            continue
        for row in doc.get('rows', []):
            rk = row.get('row_key') or {}
            if rk.get('lane') != 'auto' or row.get('status') != 'ok':
                continue
            acc = row.get('accuracy') or {}
            dE = acc.get('dE_vs_gpu_fp64')
            if dE is not None and abs(dE) > schema_validator.PRECISION_GATE_HA:
                cell_key = '/'.join([str(rk.get('geom')), str(rk.get('basis')),
                                     str(rk.get('xc'))])
                # the energy documents carry no numeric issue field; the
                # issue number lives in the evidence path
                # (/data/results/.../tasks/issue-N/...)
                import re as _re
                _m = _re.search(r'issue-(\d+)', path)
                _src = f'issue-{_m.group(1)}' if _m else 'issue-?'
                note = (f'{_src} auto lane '
                        f'dE_vs_gpu_fp64 = {dE:.3e} Ha (>{schema_validator.PRECISION_GATE_HA:.0e} gate)')
                failed[cell_key] = note
                log(f'energy-gate: FAILED cell {cell_key}: {note}')
    return failed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(
        prog='gradient_runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='RTX 4090 gradient benchmark runner producing '
                    'rtx4090-gradient-v1 documents (issue #12). One '
                    'geometry x basis x functional per invocation; rows '
                    'merge by row_key across invocations.',
        epilog='Lanes: fp64 (GPU reference), auto (production: auto SCF + '
               'fp32 gradient kernels, per grad/rhf.py _resolve_precision), '
               'fp32-grad (fp64 SCF + fp32 gradient kernels: the kernel '
               'error alone), auto-fp64-grad (auto SCF + fp64 gradient '
               'kernels: the SCF-path error alone), cpu-fp64 (PySCF '
               'sentinel). Together the middle three decompose the auto '
               'gradient error into its two sources (homotopy controls). '
               'The gradient gate is max|dg| vs gpu-fp64 <= '
               f'{GRADIENT_GATE:.0e} Eh/Bohr, one order below geomeTRIC\'s '
               'default gmax 3e-4.')
    p.add_argument('--out', required=True, help='output schema JSON path')
    p.add_argument('--geom', default='methanol.xyz')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--lane', nargs='*', default=[
        'fp64', 'auto', 'fp32-grad', 'auto-fp64-grad', 'cpu-fp64'],
        choices=list(GRAD_LANES))
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--conv-tol', type=float, default=1e-10)
    p.add_argument('--max-cycle', type=int, default=50)
    p.add_argument('--init-guess', default='minao')
    p.add_argument('--direct-scf-tol', type=float, default=1e-14)
    p.add_argument('--grids-level', type=int, default=3)
    p.add_argument('--nlcgrids-level', type=int, default=None)
    p.add_argument('--headroom-mib', type=float, default=2048.0)
    p.add_argument('--min-free-mib', type=float, default=None)
    p.add_argument('--pilot-stage', default='S0',
                   choices=['S0', 'S1', 'S2', 'S3', 'S4'])
    p.add_argument('--charge', type=int, default=None)
    p.add_argument('--spin', type=int, default=None)
    p.add_argument('--dispersion', default=None)
    p.add_argument('--gcp', default=None)
    p.add_argument('--merge-from', default=None)
    p.add_argument('--energy-gate-doc', nargs='*', default=None,
                   help='issue #5-#8 energy documents; cells whose auto '
                        'lane failed the 1e-6 Ha energy gate get an '
                        'explicit provisional gradient verdict (G5)')
    p.add_argument('--evidence-root', default='/data/results')
    p.add_argument('--no-validate', action='store_true')
    args = p.parse_args(argv)
    cmdline = ' '.join([sys.executable or 'python', os.path.abspath(__file__)]
                       + (sys.argv[1:] if argv is None else list(argv)))

    if args.samples < 3:
        print('error: --samples must be >= 3 (schema rule R1)',
              file=sys.stderr)
        return 2
    if energy_runner.is_3c_xc(args.xc) and \
            args.basis.lower() != energy_runner.R2SCAN3C_PRESCRIBED_BASIS:
        print(f'error: r2SCAN-3c rows must use the prescribed basis '
              f'{energy_runner.R2SCAN3C_PRESCRIBED_BASIS} (schema rule R4), '
              f'got {args.basis!r}', file=sys.stderr)
        return 2

    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(out_path))[0]
    raw_path = os.path.join(out_dir, stem + '.raw.json')
    log_path = os.path.join(out_dir, stem + '.runlog.log')

    lines = []

    def log(msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        lines.append(line)

    # source-tree provenance guard (wheel shadowing; issue #3 regression)
    import gpu4pyscf
    from gpu4pyscf.lib import precision  # noqa: F401
    if not gpu4pyscf.__file__.startswith(energy_runner.REPO):
        print(f'error: gpu4pyscf resolves to {gpu4pyscf.__file__}, not the '
              f'source tree {energy_runner.REPO}; refusing to run (wheel '
              'shadowing)', file=sys.stderr)
        return 2
    log(f'source provenance OK: {gpu4pyscf.__file__}')

    # r2SCAN-3c whole-method setup: identical to the energy runner
    # (alias resolution, prescribed-basis guard above, D4/gCP probe,
    # method.dispersion/gcp auto-fill)
    if energy_runner.is_3c_xc(args.xc):
        try:
            import importlib.metadata
            disp_version = importlib.metadata.version('pyscf-dispersion')
            args.dispersion = energy_runner.r2scan3c_dispersion_block(
                disp_version)
            args.gcp = energy_runner.R2SCAN3C_GCP_METHOD
            from types import SimpleNamespace
            probe_mol = energy_runner.make_mol(args.geom, args.basis,
                                                args.charge, args.spin)
            carrier = SimpleNamespace(mol=probe_mol, scf_summary={})
            energy_runner._r2scan3c_get_dispersion(carrier)
            log(f'r2SCAN-3c mode: D4+gCP probe OK '
                f'(D4={carrier.scf_summary["dispersion_d4"]:.6e} '
                f'gCP={carrier.scf_summary["gcp"]:.6e} Ha)')
        except Exception as exc:
            print(f'error: r2SCAN-3c D4/gCP construction failed on '
                  f'{args.geom}: {exc!r}; refusing to run a wrong method',
                  file=sys.stderr)
            return 2

    # NLC guard: identical to the energy runner
    libxc_xc = energy_runner.resolve_3c(args.xc) or args.xc
    from pyscf.dft import libxc as cpu_libxc
    try:
        is_nlc = bool(cpu_libxc.is_nlc(libxc_xc))
    except Exception as exc:
        print(f'error: cannot resolve XC alias {libxc_xc!r} ({exc!r}); '
              'refusing to run an unknown functional', file=sys.stderr)
        return 2
    if is_nlc:
        from gpu4pyscf.dft import libxc as gpu_libxc
        if not gpu_libxc.is_nlc(libxc_xc):
            print(f'error: {libxc_xc!r} is NLC on pyscf libxc but not on '
                  'the gpu4pyscf libxc wheel; refusing to run',
                  file=sys.stderr)
            return 2
        log(f'NLC guard OK: {libxc_xc!r} is_nlc on both libxc stacks')

    # energy-gate alignment (G5): read the inherited energy documents and
    # collect honestly-failed cells BEFORE any lane runs
    gate_failed = read_energy_gate_documents(args.energy_gate_doc, log)

    rows = []
    if args.merge_from and os.path.isfile(args.merge_from):
        with open(args.merge_from, encoding='utf-8') as f:
            previous = json.load(f)
        rows = [r for r in previous.get('rows', [])
                if r.get('status') in ('ok', 'skipped', 'oom', 'error')]
        log(f'merged {len(rows)} rows from {args.merge_from}')

    meta = energy_runner.build_meta(args)

    for lane in args.lane:
        if lane == 'cpu-fp64':
            row = run_cpu_grad_cell(args.geom, args.basis, args.xc, args,
                                    log)
        else:
            row = run_gpu_grad_cell(args.geom, args.basis, args.xc, lane,
                                    args, log)
        energy_runner.upsert(rows, row)

    derive_grad(rows, gate_failed_cells=gate_failed)
    # attach the energy-gate reference to gradient rows of failed cells
    for row in rows:
        cell_key = '/'.join(str(k) for k in energy_runner.key_of(row)[:3])
        if gate_failed and cell_key in gate_failed:
            row.setdefault('result', {})['energy_gate_reference'] = {
                'source_issues': 'issue-7 (inherited)',
                'gate_failed_cells': [cell_key],
                'note': gate_failed[cell_key]}

    # phase 1: immutable raw result + run log
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump({'schema_version': SCHEMA_VERSION, 'meta': meta,
                   'rows': rows}, f, indent=1)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    raw_sha, log_sha = sha256_file(raw_path), sha256_file(log_path)

    # phase 2: schema document anchoring the raw artifacts
    for row in rows:
        row['evidence'] = {
            'raw_log': f'{log_path}@{log_sha}',
            'result_json': f'{raw_path}@{raw_sha}',
            'command': cmdline}
    doc = {'schema_version': SCHEMA_VERSION, 'meta': meta, 'rows': rows}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)

    if args.no_validate:
        log(f'wrote {out_path} (validation skipped)')
        return 0
    issues = schema_validator.validate_gradient_document(
        doc, evidence_root=(args.evidence_root or None))
    if issues:
        log(f'VALIDATION FAILED: {len(issues)} issues')
        for it in issues:
            log(f'  [{it["rule"]}/{it["tag"]}] {it["row"] or ""} '
                f'{it["message"]}')
        return 1
    n_pass = sum(1 for r in rows
                 if (r.get('accuracy') or {}).get('verdict') == 'pass')
    log(f'wrote {out_path} -- VALID ({len(rows)} rows, {n_pass} pass, '
        f'raw {raw_sha[:12]}, log {log_sha[:12]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

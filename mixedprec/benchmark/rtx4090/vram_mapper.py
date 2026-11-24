#!/usr/bin/env python
'''RTX 4090 VRAM capacity boundary mapper (issue #11 part (a)).

Maps peak device memory vs AO count along the pilot ladder, separating

  * ``ok``            -- measured, peak recorded;
  * ``oom``           -- an allocation actually failed (observed boundary);
  * ``skipped``       -- NOT run on purpose: predicted footprint exceeds the
                          device (+headroom) budget, or the host RAM budget
                          for host-fallback cells, or the wall-clock budget.

The two boundary kinds are never conflated: an ``oom`` row is a measured
hardware boundary, a ``skipped`` row is a pre-run budget decision
(docs/consumer-gpu-plan.md section 1.2 item 5 and the issue #11 (a) scope).
Admission contract (2026-10-06 P1 rework): a cell is admitted only if its
predicted resident footprint (the conservative cderi upper bound) fits in
the MEASURED current free VRAM minus the reserve (``--headroom-mib``);
the device total capacity is never an admission budget -- it is reported
separately as theoretical capacity inside every gate record.  A
nonpositive budget (free <= reserve) skips safely whatever the request;
no OOM is ever provoked to discover a boundary, and no working-set
estimate is invented (the reserve covers the working set, the actual
peak is measured after the run).  Host-CDERI cells additionally carry a
documented GPU working budget (free - reserve, positive by the same
check) and a host RAM bound (tensor + staging margin <= host available).
Offload/streaming research (part (b)) is explicitly out of scope; the only
host-memory behaviour measured here is the *existing* CDERI residency
fallback that upstream ``DF`` already performs automatically (or that
``--force-host-cderi`` selects explicitly) -- measuring it is capacity
mapping, not offload development.

Probe discipline: one fresh subprocess per cell (clean CUDA context, no
pool carry-over between cells), deviceSynchronize around every timed
phase, MemorySampler polling device-wide free memory (pool-cached blocks
are not returned to the driver, so the free-memory watermark is the true
high-water mark), single sample per cell.  This is a capacity probe, NOT
an rtx4090-energy-v1 accuracy/timing document: conv_tol 1e-9, no warmup,
no 3-sample timing, no precision-gate claims.  Energies are recorded only
as sanity checks that the intended path (DF + method) actually ran.

Usage (GPU python):
    python vram_mapper.py --out boundary.json [--cells default] ...
    python vram_mapper.py --probe ...        # internal: one cell to stdout
    python vram_mapper.py --plan             # ladder + predictions, no GPU
'''

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rtx4090_runner import (  # noqa: E402  (repo layout, not an installed pkg)
    MemorySampler, make_mol, gpu_static_info, git_full_sha, GEOMS)

SCHEMA_VERSION = 'vram-boundary-v1'

# The default ladder: admitted-cohort anchors plus S3-preview cells whose
# admission is decided BY this mapping (plan section 3.5).  'grad' cells also
# run one analytic gradient after the SCF (fix-site live check for the
# issue-#11 chunk audit; gradient boundary data feeds issue #12).
DEFAULT_LADDER = [
    # (geom, basis, xc, grad, force_host_cderi, note)
    ('benzene.xyz', 'def2-tzvp', 'r2scan', False, False,
     'anchor: cross-check against issue-#5 measured peak 316 MiB'),
    ('020_Vitamin_C.xyz', 'def2-svp', 'wb97m-v', False, False,
     'NLC/VV10 path memory probe (nlcgrids + VV10 staging)'),
    ('057_Tamoxifen.xyz', 'def2-svp', 'r2scan', False, False,
     'S3 preview, pure functional'),
    ('057_Tamoxifen.xyz', 'def2-svp', 'b3lyp', True, False,
     'S3 preview, hybrid + analytic gradient (chunk-audit live cell)'),
    ('057_Tamoxifen.xyz', 'def2-tzvp', 'r2scan', False, False,
     'cderi ~10.6 GB upper bound: GPU-resident expectation'),
    ('057_Tamoxifen.xyz', 'def2-tzvp', 'r2scan', False, True,
     'existing CDERI host-residency fallback, forced (diagnostic only; '
     'not an offload claim)'),
    ('095_Azadirachtin.xyz', 'def2-svp', 'r2scan', False, False,
     'cderi ~15.1 GB upper bound: within 24 GB, beyond 2048 MiB headroom'),
    ('095_Azadirachtin.xyz', 'def2-tzvp', 'r2scan', False, False,
     'predicted ~65 GB: beyond empty-device capacity at this reserve; '
     'recorded skipped, never run'),
]

GIB = 1 << 30


# ---------------------------------------------------------------------------
# pure planning helpers (unit-testable without pyscf/cupy)
# ---------------------------------------------------------------------------

def predict_footprint(nao, naux):
    '''Upper-bound device footprint of the DF cderi tensor (fp64).

    npairs uses the full nao*(nao+1)/2 triangle; the stored cderi compresses
    general contractions, so this is an upper bound -- deliberate, gates
    must be conservative.
    '''
    npairs = nao * (nao + 1) // 2
    cderi_bytes = npairs * naux * 8
    return {'nao': int(nao), 'naux': int(naux), 'npairs': int(npairs),
            'cderi_upper_bound_bytes': int(cderi_bytes),
            'cderi_upper_bound_gib': cderi_bytes / GIB}


def device_capacity_note(request_gib, gpu_total_mib, reserve_mib):
    '''Theoretical device-capacity statement (NOT an admission decision).

    Reports whether ``request_gib`` could fit on an EMPTY device of this
    capacity at this reserve.  This is capacity mapping information only:
    runtime admission always uses the measured CURRENT free memory
    (gate_decision), because capacity is not availability -- other
    processes or earlier cells' memory pools may hold most of the device
    (P1 review 2026-10-06: the old gate admitted an 8 GiB request on a
    device with 24 GB total but only 1 GiB free).
    '''
    capacity_mib = gpu_total_mib - reserve_mib
    capacity_gib = capacity_mib * 2**20 / GIB
    return {
        'fits_empty_device': bool(capacity_mib > 0
                                  and request_gib <= capacity_gib),
        'request_gib': float(request_gib),
        'gpu_total_mib': float(gpu_total_mib),
        'reserve_mib': float(reserve_mib),
        'empty_device_budget_gib': round(capacity_gib, 3),
        'note': ('theoretical capacity only (empty device); runtime '
                 'admission uses current free minus reserve, never this'),
    }


def gate_decision(request_gib, gpu_total_mib, gpu_free_mib, reserve_mib=None,
                  host_mode=False, host_avail_gib=None,
                  host_margin_gib=4.0, headroom_mib=None):
    '''Pre-run ADMISSION gate on current-free VRAM minus reserve.

    The admission basis is the MEASURED current free memory at gate time
    (memGetInfo) minus the reserve (the configured headroom budget,
    ``--headroom-mib``) that must remain free.  The device TOTAL is never
    an admission budget; it only feeds the separate theoretical-capacity
    report (device_capacity_note) and the beyond-capacity skip reason.

    ``request_gib`` is the predicted GPU-resident footprint upper bound
    (predict_footprint's cderi bound: npairs x naux x 8 B -- conservative
    by construction, it uses the full pair triangle).  The SCF/JK/grid
    working set on top of the resident tensor is deliberately NOT
    predicted (any number would be invented); the reserve is what covers
    it, and the actual peak is measured after the run
    (mem.gpu_peak_alloc_mib).  No OOM is ever provoked to decide safety.

    Checks, in order, each with its own honest skip reason:
      1. nonpositive admission budget (free <= reserve): nothing can be
         admitted, whatever the request size -- skip, do not attempt;
      2. host_mode (existing CDERI host-residency fallback): the tensor
         lives in host RAM, so there is no GPU residency request; the GPU
         still needs a positive working budget (free - reserve) for the
         SCF/JK/grid workspace (guaranteed by check 1) -- the working
         budget is recorded by the caller and the actual workspace is
         measured after the run; host RAM must hold the tensor plus a
         staging margin (host_avail_gib, host_margin_gib);
      3. request beyond what an empty device could hold at this reserve:
         can never run GPU-resident on this device (theoretical, from
         total) -- a capacity-mapping skip, not a runtime-availability
         one;
      4. request exceeds current free minus reserve: not admitted NOW
         even though it fits an empty device; retry when more memory is
         free rather than relabelling.

    Boundary semantics: request == budget admits (<=).
    '''
    if reserve_mib is None:
        reserve_mib = headroom_mib     # legacy keyword alias
    if reserve_mib is None:
        raise ValueError('reserve_mib (or the legacy headroom_mib alias) '
                         'is required')
    budget_mib = gpu_free_mib - reserve_mib
    budget_gib = budget_mib * 2**20 / GIB
    if budget_mib <= 0:
        return False, ('nonpositive-admission-budget: free '
                       f'{gpu_free_mib:.0f} MiB - reserve {reserve_mib:.0f} '
                       f'MiB = {budget_mib:.0f} MiB <= 0; no request can be '
                       'admitted (budget decision, not an OOM attempt)')
    if host_mode:
        if host_avail_gib is None:
            return False, ('host-mode cell requires host_avail_gib '
                           '(host RAM bound)')
        if request_gib + host_margin_gib > host_avail_gib:
            return False, ('predicted-exceeds-host-ram: request '
                           f'{request_gib:.1f} GiB + staging margin '
                           f'{host_margin_gib:.1f} GiB > host available '
                           f'{host_avail_gib:.1f} GiB')
        # cderi is host-resident: no GPU residency request; the positive
        # GPU working budget (free - reserve) from check 1 is the
        # documented workspace budget, and the workspace itself is
        # measured (gpu_peak_alloc_mib), never predicted
        return True, None
    capacity_gib = (gpu_total_mib - reserve_mib) * 2**20 / GIB
    if request_gib > capacity_gib:
        return False, ('beyond-device-capacity: request '
                       f'{request_gib:.1f} GiB > empty-device budget '
                       f'{capacity_gib:.1f} GiB (total {gpu_total_mib:.0f} '
                       f'MiB - reserve {reserve_mib:.0f} MiB); can never '
                       'run GPU-resident on this device at this reserve')
    if request_gib > budget_gib:
        return False, ('predicted-exceeds-free-minus-reserve: request '
                       f'{request_gib:.1f} GiB > budget {budget_gib:.1f} '
                       f'GiB (free {gpu_free_mib:.0f} MiB - reserve '
                       f'{reserve_mib:.0f} MiB); the request fits an empty '
                       'device (capacity reporting), so retry when more '
                       'memory is free rather than relabelling')
    return True, None


def gate_record(request_gib, gpu_free_mib, gpu_total_mib, reserve_mib,
                host_mode=False, host_avail_gib=None, host_margin_gib=4.0):
    '''Evaluate the admission contract and build the full gate record.

    The single source of the admission contract shared by the mapper
    probe (probe_cell), the batch channel and the energy/gradient
    runners: current free minus reserve decides; the resident request,
    the reserve (work budget) and the separate theoretical device
    capacity are all recorded explicitly.  Returns (admitted, record).
    '''
    run, reason = gate_decision(
        request_gib, gpu_total_mib, gpu_free_mib, reserve_mib,
        host_mode=host_mode, host_avail_gib=host_avail_gib,
        host_margin_gib=host_margin_gib)
    record = {
        'admission_basis': 'current-free-minus-reserve',
        'admitted': bool(run), 'skip_reason': reason,
        'gpu_free_before_mib': round(gpu_free_mib, 1),
        'gpu_total_mib': round(gpu_total_mib, 1),
        'reserve_mib': float(reserve_mib),
        'admission_budget_mib': round(gpu_free_mib - reserve_mib, 1),
        'accounting': {
            'resident_request_gib': round(request_gib, 3),
            'resident_request_kind': (
                'cderi fp64 upper bound (npairs x naux x 8 B), '
                'host-resident in this mode' if host_mode else
                'cderi fp64 upper bound (npairs x naux x 8 B)'),
            'work_reserve_mib': float(reserve_mib),
            'work_note': ('the SCF/JK/grid working set is not predicted '
                          '(no invented estimate); the reserve covers it '
                          'and the actual peak is measured after the run'),
        },
        'device_capacity': device_capacity_note(
            request_gib, gpu_total_mib, reserve_mib),
    }
    if host_mode:
        record['gpu_working_budget_mib'] = round(
            gpu_free_mib - reserve_mib, 1)
        record['accounting']['host_staging_margin_gib'] = \
            float(host_margin_gib)
        if host_avail_gib is not None:
            record['host_available_gib'] = float(host_avail_gib)
    return run, record


def derive_boundaries(cells):
    '''Derive the boundary statements from measured cells (pure).

    Never turns an oom/skip into a pass; violations of the headroom budget
    on ok cells are surfaced, not hidden.
    '''
    ok = [c for c in cells if c.get('status') == 'ok']

    def key(c):
        return (c.get('predicted') or {}).get('nao', 0)
    oom = [c for c in cells if c.get('status') == 'oom']
    skipped = [c for c in cells if c.get('status') == 'skipped']
    other = [c for c in cells if c.get('status') not in
             ('ok', 'oom', 'skipped')]

    energy_ok = [c for c in ok if not c.get('force_host_cderi')]
    largest_ok = max(energy_ok, key=key) if energy_ok else None
    grad_ok = [c for c in ok if c.get('grad')
               and (c.get('result') or {}).get('grad')]
    largest_grad_ok = max(grad_ok, key=key) if grad_ok else None

    residency = {}
    for c in ok:
        r = (c.get('result') or {}).get('mem', {}).get('cderi_residency')
        if r:
            residency.setdefault(r, []).append(
                {'geom': c['geom'], 'basis': c['basis'],
                 'cderi_upper_bound_gib': round(
                     c['predicted']['cderi_upper_bound_gib'], 2)})

    budget_violations = [
        {'cell': c['cell_key'],
         'peak_mib': c['result']['mem']['gpu_peak_alloc_mib']}
        for c in ok
        if (c.get('result') or {}).get('mem', {}).get('within_budget') is False]

    # admission audit: every cell that went through VRAM admission
    # (ran, OOMed, or was skipped for a VRAM reason -- time-budget skips
    # and parent-fabricated timeout/parse-error cells never reached the
    # gate) must carry its gate record on the current-free-minus-reserve
    # basis; missing records are surfaced, never assumed
    def _vram_admitted(c):
        if c.get('status') in ('ok', 'oom'):
            return True
        return (c.get('status') == 'skipped'
                and c.get('skip_reason') != 'time-budget')

    gate_audit = {
        'admission_basis': 'current-free-minus-reserve',
        'cells_missing_gate_record': [
            c['cell_key'] for c in cells
            if _vram_admitted(c) and
            (c.get('gate') or {}).get('admission_basis') !=
            'current-free-minus-reserve'],
        'skipped_reason_histogram': {},
    }
    for c in skipped:
        r = c.get('skip_reason') or 'unknown'
        fam = r.split(':')[0]
        gate_audit['skipped_reason_histogram'][fam] = \
            gate_audit['skipped_reason_histogram'].get(fam, 0) + 1

    return {
        'energy_fp64': {
            'largest_ok_cell': ({'nao': largest_ok['predicted']['nao'],
                                 'cell': largest_ok['cell_key']}
                                if largest_ok else None),
            'ok_cells': [c['cell_key'] for c in sorted(energy_ok, key=key)],
        },
        'gradient_fp64': {
            'largest_ok_cell': ({'nao': largest_grad_ok['predicted']['nao'],
                                 'cell': largest_grad_ok['cell_key']}
                                if largest_grad_ok else None),
        },
        'oom_observed': [
            {'cell': c['cell_key'], 'nao': c['predicted']['nao'],
             'error': (c.get('result') or {}).get('error')}
            for c in oom],
        'skipped_boundary': [
            {'cell': c['cell_key'], 'nao': c['predicted']['nao'],
             'reason': c.get('skip_reason')}
            for c in skipped],
        'cderi_residency': residency,
        'headroom_budget_violations': budget_violations,
        'non_terminal_status_cells': [c['cell_key'] for c in other],
        'admission_gate_audit': gate_audit,
        'honesty_note': (
            'oom rows are measured hardware boundaries; skipped rows are '
            'pre-run budget decisions; the two are recorded separately and '
            'neither is relabelled. Cells with status error/timeout are '
            'listed under non_terminal_status_cells. Admission (since the '
            '2026-10-06 P1 rework) uses measured current free VRAM minus '
            'reserve, never device total capacity; device capacity is '
            'reported separately inside each gate record.'),
    }


def cell_key(geom, basis, xc, force_host=False):
    base = f'{os.path.splitext(os.path.basename(geom))[0]}_{basis.replace("-", "")}_{xc.replace("-", "")}'
    return base + ('_hostcderi' if force_host else '')


# ---------------------------------------------------------------------------
# GPU probe (runs in a fresh subprocess per cell)
# ---------------------------------------------------------------------------

def _host_avail_gib():
    with open('/proc/meminfo') as f:
        for line in f:
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024 / GIB
    return None


def probe_cell(args, log=print):
    '''One cell: gate, run one fp64 single point (+ optional gradient).

    Prints a JSON object to stdout (the parent collector parses it).
    '''
    import cupy as cp

    out = {
        'cell_key': cell_key(args.geom, args.basis, args.xc,
                             args.force_host_cderi),
        'geom': args.geom, 'basis': args.basis, 'xc': args.xc,
        'lane': 'fp64', 'grad': bool(args.run_grad),
        'force_host_cderi': bool(args.force_host_cderi),
        'conv_tol': args.conv_tol, 'note': args.note or '',
    }
    from pyscf import df as cpu_df
    mol = make_mol(args.geom, args.basis, args.charge, args.spin)
    auxmol = cpu_df.make_auxmol(mol, args.auxbasis)
    pred = predict_footprint(mol.nao, auxmol.nao)
    out['predicted'] = pred
    out['system'] = {'natm': mol.natm, 'nao': mol.nao, 'naux': auxmol.nao,
                     'charge': mol.charge, 'spin': mol.spin}

    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    request_gib = pred['cderi_upper_bound_gib']
    # the admission record every cell carries (accepted or skipped alike):
    # what was requested, on what basis it was decided, and the explicit
    # resident/work/staging accounting.  The theoretical device capacity
    # is reported INSIDE this record but is never the admission basis.
    run, gate = gate_record(
        request_gib, free_mib, total_mib, args.headroom_mib,
        host_mode=args.force_host_cderi,
        host_avail_gib=_host_avail_gib())
    out['gate'] = gate
    if not run:
        out['status'] = 'skipped'
        out['skip_reason'] = gate['skip_reason']
        out['mem'] = {'gpu_free_before_mib': round(free_mib, 1),
                      'gpu_total_mib': round(total_mib, 1),
                      'headroom_budget_mib': args.headroom_mib}
        log(json.dumps(out))
        return 0

    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    import resource

    cp.cuda.runtime.deviceSynchronize()
    t_e2e = time.perf_counter()
    result = {'timings': {}, 'mem': {
        'gpu_free_before_mib': round(free_mib, 1),
        'gpu_total_mib': round(total_mib, 1),
        'headroom_budget_mib': args.headroom_mib}}
    out['result'] = result
    try:
        if args.xc == 'hf':
            cls = gpu_scf.UHF if mol.spin else gpu_scf.RHF
            mf = cls(mol)
        else:
            cls = gpu_dft.UKS if mol.spin else gpu_dft.RKS
            mf = cls(mol, xc=args.xc)
        mf = mf.density_fit(auxbasis=args.auxbasis)
        mf.conv_tol = args.conv_tol
        mf.max_cycle = args.max_cycle
        mf.verbose = 0
        if args.force_host_cderi:
            mf.with_df.use_gpu_memory = False

        t0 = time.perf_counter()
        mf.with_df.build()
        cp.cuda.runtime.deviceSynchronize()
        result['timings']['t_cderi_s'] = round(time.perf_counter() - t0, 3)
        cderi0 = mf.with_df._cderi[0] if mf.with_df._cderi else None
        result['mem']['cderi_residency'] = (
            'gpu' if cderi0 is not None and type(cderi0).__module__.startswith(
                'cupy') else 'host')

        t0 = time.perf_counter()
        with MemorySampler() as samp_scf:
            e = mf.kernel()
            cp.cuda.runtime.deviceSynchronize()
        result['timings']['t_scf_s'] = round(time.perf_counter() - t0, 3)
        result['e_tot'] = float(e)
        result['converged'] = bool(mf.converged)
        result['scf_cycles'] = int(getattr(mf, 'cycles', 0) or 0)
        result['mem']['gpu_peak_alloc_mib'] = round(
            total_mib - samp_scf.min_free_mib, 1)
        result['mem']['scf_min_free_mib'] = round(samp_scf.min_free_mib, 1)

        if args.run_grad:
            t0 = time.perf_counter()
            with MemorySampler() as samp_grad:
                g = mf.nuc_grad_method().kernel()
                cp.cuda.runtime.deviceSynchronize()
            result['timings']['t_grad_s'] = round(time.perf_counter() - t0, 3)
            import numpy as np
            gnorm = float(np.abs(cp.asnumpy(g)).max()) if g is not None else None
            result['grad'] = {
                'max_abs_component_eh_bohr': gnorm,
                'gpu_peak_alloc_mib_during_grad': round(
                    total_mib - samp_grad.min_free_mib, 1)}
            out['grad'] = True

        result['timings']['end_to_end_s'] = round(
            time.perf_counter() - t_e2e, 3)
        result['timings']['sync'] = 'deviceSynchronize before/after each phase'
        result['mem']['host_rss_peak_mib'] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        result['mem']['within_budget'] = bool(
            result['mem']['gpu_peak_alloc_mib'] + args.headroom_mib <= total_mib)
        out['status'] = 'ok'
    except Exception as exc:  # OOM or genuine error: record honestly
        import traceback
        oom_like = ('out of memory' in str(exc).lower()
                    or isinstance(exc, MemoryError))
        out['status'] = 'oom' if oom_like else 'error'
        result['error'] = {'type': type(exc).__name__,
                           'message': str(exc)[:500],
                           'traceback_head': traceback.format_exc()[:800]}
        result['mem']['host_rss_peak_mib'] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    log(json.dumps(out))
    return 0


# ---------------------------------------------------------------------------
# ladder driver (parent: no cupy import, fresh subprocess per cell)
# ---------------------------------------------------------------------------

def run_ladder(args):
    cells = []
    started = time.time()
    for (geom, basis, xc, grad, host, note) in ladder_of(args):
        elapsed = time.time() - started
        if args.time_budget_s and elapsed > args.time_budget_s:
            nao, naux = _planned_nao_naux(geom, basis, args)
            cells.append({
                'cell_key': cell_key(geom, basis, xc, host), 'geom': geom,
                'basis': basis, 'xc': xc, 'lane': 'fp64', 'grad': grad,
                'force_host_cderi': host,
                'predicted': predict_footprint(nao, naux),
                'status': 'skipped', 'skip_reason': 'time-budget',
                'note': note})
            continue
        cmd = [sys.executable, os.path.abspath(__file__), '--probe',
               '--geom', geom, '--basis', basis, '--xc', xc,
               '--auxbasis', args.auxbasis, '--conv-tol', str(args.conv_tol),
               '--max-cycle', str(args.max_cycle),
               '--headroom-mib', str(args.headroom_mib)]
        if grad:
            cmd += ['--run-grad']
        if host:
            cmd += ['--force-host-cderi']
        if args.note:
            cmd += ['--note', args.note]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=args.cell_timeout_s,
                                  cwd=REPO)
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            cell = json.loads(lines[-1]) if lines else None
            if cell is None:
                raise RuntimeError('probe produced no JSON output; stderr='
                                   + proc.stderr[-500:])
            if proc.returncode != 0 and cell.get('status') == 'ok':
                cell['status'] = 'error'
                cell.setdefault('result', {})['error'] = {
                    'type': 'ProbeReturnCode',
                    'message': f'probe exit code {proc.returncode}'}
            cell['note'] = note
        except subprocess.TimeoutExpired:
            cell = {'cell_key': cell_key(geom, basis, xc, host),
                    'geom': geom, 'basis': basis, 'xc': xc, 'lane': 'fp64',
                    'grad': grad, 'force_host_cderi': host,
                    'status': 'timeout',
                    'skip_reason': f'cell-timeout-{args.cell_timeout_s}s',
                    'note': note}
        except json.JSONDecodeError as exc:
            cell = {'cell_key': cell_key(geom, basis, xc, host),
                    'geom': geom, 'basis': basis, 'xc': xc, 'lane': 'fp64',
                    'grad': grad, 'force_host_cderi': host,
                    'status': 'error',
                    'result': {'error': {'type': 'ProbeOutputParseError',
                                         'message': str(exc)}},
                    'note': note}
        cell['probe_wall_s'] = round(time.time() - t0, 1)
        cells.append(cell)
        print(f"[{time.strftime('%H:%M:%S')}] {cell['cell_key']}: "
              f"{cell['status']} peak="
              f"{(cell.get('result', {}).get('mem', {}) or {}).get('gpu_peak_alloc_mib', '-')} MiB "
              f"({cell['probe_wall_s']}s)", flush=True)

    meta = {'schema_note': 'capacity probe: conv_tol 1e-9, single sample '
                           'per cell, fresh subprocess per cell; NOT an '
                           'rtx4090-energy-v1 accuracy/timing document',
            'mem_headroom_mib': args.headroom_mib,
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%S+00:00',
                                           time.gmtime()),
            'run_id': str(uuid.uuid4()),
            'gpu4pyscf_source_commit': git_full_sha(REPO),
            'gpu4pyscf_source_path': REPO,
            'auxbasis': args.auxbasis}
    if not args.no_gpu_meta:
        meta.update(gpu_static_info())
    doc = {'schema_version': SCHEMA_VERSION, 'meta': meta, 'cells': cells,
           'boundaries': derive_boundaries(cells)}
    with open(args.out, 'w') as f:
        json.dump(doc, f, indent=1)
    print(f"wrote {args.out}")
    return doc


def ladder_of(args):
    if args.cells == 'default':
        return DEFAULT_LADDER
    # custom: geom:basis:xc[:grad][:host] comma list
    out = []
    for spec in args.cells.split(','):
        parts = spec.split(':')
        geom, basis, xc = parts[0], parts[1], parts[2]
        grad = 'grad' in parts[3:]
        host = 'host' in parts[3:]
        out.append((geom, basis, xc, grad, host, 'custom cell'))
    return out


def _planned_nao_naux(geom, basis, args):
    '''AO counts for skip rows; falls back to (0, 0) without pyscf.'''
    try:
        from pyscf import gto, df
        mol = gto.M(atom=geom if os.path.isabs(geom)
                    else os.path.join(GEOMS, geom), basis=basis, verbose=0)
        return mol.nao, df.make_auxmol(mol, args.auxbasis).nao
    except Exception:
        return 0, 0


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='vram_mapper', description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=None,
                   help='output boundary JSON (ladder mode)')
    p.add_argument('--cells', default='default',
                   help="'default' ladder or geom:basis:xc[:grad][:host] "
                        'comma list')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--conv-tol', type=float, default=1e-9)
    p.add_argument('--max-cycle', type=int, default=50)
    p.add_argument('--headroom-mib', type=float, default=2048.0)
    p.add_argument('--time-budget-s', type=float, default=1500.0,
                   help='overall wall budget; remaining cells -> skipped')
    p.add_argument('--cell-timeout-s', type=float, default=480.0)
    p.add_argument('--plan', action='store_true',
                   help='print the ladder with predictions, no GPU')
    p.add_argument('--no-gpu-meta', action='store_true')
    p.add_argument('--geom', default='benzene.xyz',
                   help='probe mode: geometry (name under geoms or absolute)')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--xc', default='hf')
    p.add_argument('--charge', type=int, default=None)
    p.add_argument('--spin', type=int, default=None)
    p.add_argument('--note', default=None)
    p.add_argument('--run-grad', action='store_true')
    p.add_argument('--force-host-cderi', action='store_true')
    p.add_argument('--probe', action='store_true',
                   help='internal: run one cell, print JSON, exit')
    args = p.parse_args(argv)

    if args.plan:
        rows = []
        for (geom, basis, xc, grad, host, note) in ladder_of(args):
            nao, naux = _planned_nao_naux(geom, basis, args)
            pred = predict_footprint(nao, naux)
            rows.append({'cell_key': cell_key(geom, basis, xc, host),
                         'grad': grad, 'force_host_cderi': host,
                         'note': note, **pred})
        print(json.dumps({'schema_version': SCHEMA_VERSION,
                          'cells': rows}, indent=1))
        return 0
    if args.probe:
        return probe_cell(args)
    if not args.out:
        p.error('--out is required in ladder mode')
    return 0 if run_ladder(args) else 1


if __name__ == '__main__':
    sys.exit(main())

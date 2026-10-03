#!/usr/bin/env python3
"""RTX 4090 benchmark runner producing ``rtx4090-energy-v1`` documents.

Issue-#4 deliverable of docs/consumer-gpu-plan.md.  It reuses the verified
skeleton of mixedprec/benchmark/run_benchmark.py (fresh-mol timing cells,
row-key upsert/merge across invocations, CPU sentinel lanes, derived error
columns) and adds everything the RTX 4090 schema demands:

  * warmup strictly separated from >= 3 independent samples (new mol + new
    mf each), CUDA deviceSynchronize() before/after every timed phase;
  * end-to-end per-sample wall time that includes mol.build, the CDERI
    build, the SCF (with the auto lane's fp32 head, precision switch and
    fp64 tail) and all corrections;
  * memory discipline: measured free/total before the cell, in-run peak
    device allocation (polled memGetInfo high-water), host RSS high-water,
    a configurable headroom budget, honest ``skipped`` rows when the budget
    cannot be guaranteed, and ``oom`` rows when the hardware boundary is
    actually hit;
  * fp32 entry-hit instrumentation (counting wrapper on
    gpu4pyscf.lib.precision.get_precision attributed to the calling branch)
    proving the mixed-precision path really executed -- the regression
    guard against the historical wheel-shadowing silent-fp64 incident;
  * precision-switch evidence (cycle + inferred trigger) observed from the
    SCF callback, without touching physics code;
  * e_terms breakdown (e1/coul/exc/nlc/dispersion/gcp) from scf_summary
    plus an instrumented nr_nlc_vxc capture for the separate NLC term;
  * df-vs-conv rows isolating the DF approximation error from the
    precision error, and cpu-fp64 sentinel rows for the level-(2)
    implementation-agreement tier;
  * evidence anchoring: the run writes an immutable raw JSON + run log,
    then a schema document whose rows reference both by sha256.

The validator (schema_validator.py, same directory) runs over the final
document; a run whose output fails validation reports every issue and
exits nonzero.

Usage (methanol smoke, see docs/rtx4090-benchmark-schema.md):
    PYTHONPATH=<repo> python rtx4090_runner.py \
        --out /data/results/gpu4pyscf-agent/tasks/issue-4/methanol_smoke.json \
        --lane fp64 auto cpu-fp64 df-vs-conv --samples 3
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
import uuid

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
GEOMS = os.path.join(os.path.dirname(HERE), 'geoms')
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)      # for schema_validator / g16_curation

import schema_validator  # noqa: E402  (stdlib only, safe to import anywhere)

SCHEMA_VERSION = 'rtx4090-energy-v1'
FP32_SCREEN_CONV_TOL = 3e-5   # measured DF J/K fp32 noise floor (precision.py)


def _lane_conv_tol(lane, conv_tol):
    '''The pure-fp32 screening lane cannot converge below the measured DF
    J/K fp32 noise floor, so its conv_tol is floored at 3e-5 -- for THAT
    lane only.  Flooring a shared args.conv_tol for the whole invocation
    (the issue #4 behavior) silently loosens every other lane's
    convergence: found live in issue #5, where a one-invocation
    [fp64 auto cpu-fp64 df-vs-conv fp32-screen] run converged the auto
    lane at the switch cycle itself, leaving a zero-length fp64 tail and
    returning the fp32 energy (2.5e-6 Ha off), while the fp64 lane
    stopped 1.5e-8 Ha short of its 1e-10-converged energy.  Only floor,
    never tighten below what the user asked.'''
    if lane == 'fp32-screen':
        return max(conv_tol, FP32_SCREEN_CONV_TOL)
    return conv_tol

# molecule -> (charge, spin); everything else is neutral singlet.  Kept in
# sync with run_benchmark.py (r14's xyz header is wrong about spin; see the
# historical notes there).  methoxy.xyz is the issue #5 UKS sentinel: the
# CH3O radical ground state is a doublet (spin = 2S = 1).
CHARGE_SPIN = {'r14.xyz': (1, 1), 'methoxy.xyz': (0, 1)}


def _hits_key(module, func):
    '''fp32 branch call site -> schema hit counter.'''
    if module == 'gpu4pyscf.df.df_jk':
        return 'df_jk'
    if module == 'gpu4pyscf.dft.numint':
        if func in ('_eval_rho2', 'eval_rho4'):
            return 'eval_rho'
        if func.startswith('_nr_'):
            return 'xc_grid'
        return 'other'
    if module.startswith(('gpu4pyscf.grad', 'gpu4pyscf.df.grad',
                          'gpu4pyscf.hessian', 'gpu4pyscf.df.hessian')):
        return 'grad_ip1'
    return 'other'


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _num(v):
    '''JSON-safe float (scf_summary values may arrive as 0-dim cupy/numpy).'''
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# environment probing (lazy: none of this runs for --help / unit tests)
# ---------------------------------------------------------------------------

def gpu_static_info():
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version',
             '--format=csv,noheader'], text=True).strip().splitlines()[0]
        name, total, driver = [p.strip() for p in out.split(',')]
        return {'gpu_name': name,
                'gpu_total_mem_mib': float(total.split()[0]),
                'driver_version': driver}
    except Exception:
        return {'gpu_name': None, 'gpu_total_mem_mib': None,
                'driver_version': None}


def cpu_ram_info():
    cpu_model, ram_gb = None, None
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_model = line.split(':', 1)[1].strip()
                    break
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                    break
    except OSError:
        pass
    return cpu_model, ram_gb


def git_full_sha(repo):
    try:
        return subprocess.check_output(
            ['git', '-C', repo, 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def cuda_version_str():
    try:
        import cupy as cp
        v = cp.cuda.runtime.runtimeGetVersion()
        return f'{v // 1000}.{(v % 1000) // 10}'
    except Exception:
        return None


def libxc_version():
    try:
        from gpu4pyscf.lib import libxc
        return getattr(libxc, '__version__', None)
    except Exception:
        try:
            from gpu4pyscf.dft.numint import libxc
            return getattr(libxc, '__version__', None)
        except Exception:
            return None


# ---------------------------------------------------------------------------
# memory sampling
# ---------------------------------------------------------------------------

class MemorySampler:
    '''Polls device free memory (memGetInfo) while a cell runs.

    memGetInfo reports device-wide free memory, so a drop in free memory
    reflects the process's true high-water allocation even when CuPy's pool
    caches freed blocks (cached blocks are NOT returned to the driver).
    host_rss is the process ru_maxrss high-water mark (may include earlier
    cells of the same process; noted in the schema doc).
    '''

    def __init__(self, interval=0.05):
        self.interval = interval
        self.min_free_mib = None
        self.max_pool_mib = None
        self.total_mib = None
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        import cupy as cp
        while not self._stop.is_set():
            try:
                free, total = cp.cuda.runtime.memGetInfo()
                used_pool = cp.get_default_memory_pool().used_bytes()
            except Exception:
                return
            self.total_mib = total / 2**20
            if self.min_free_mib is None or free / 2**20 < self.min_free_mib:
                self.min_free_mib = free / 2**20
            if self.max_pool_mib is None or \
                    used_pool / 2**20 > self.max_pool_mib:
                self.max_pool_mib = used_pool / 2**20
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def host_rss_peak_mib(self):
        # ru_maxrss is KiB on Linux and is the process-lifetime high-water
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# ---------------------------------------------------------------------------
# precision instrumentation (observation only; physics code untouched)
# ---------------------------------------------------------------------------

class PrecisionProbe:
    '''Counts fp32 branch entries by call site, records mode transitions,
    and logs per-cycle (cycle, E, mode) from the SCF callback to locate the
    auto-lane fp32->fp64 switch.'''

    def __init__(self):
        self.hits = {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0, 'grad_ip1': 0,
                     'other': 0}
        self.transitions = []
        self.cycles = []          # (cycle, e_tot, mode) per SCF iteration
        self.last_nlc_energy = None
        self._orig_get = None
        self._orig_set = None
        self._orig_nlc = None

    def install(self):
        import inspect
        from gpu4pyscf.lib import precision
        orig_get, orig_set = precision.get_precision, precision.set_precision

        def counting_get():
            val = orig_get()
            if val == 'fp32':
                # inspect.stack()[1] is a FrameInfo (not a frame) on all
                # supported pythons; the caller's module lives one .frame
                # level down (same attribution scheme as the issue-3 smoke)
                fi = inspect.stack()[1]
                key = _hits_key(fi.frame.f_globals.get('__name__', '?'),
                                fi.frame.f_code.co_name)
                self.hits[key] += 1
            return val

        def counting_set(mode):
            self.transitions.append({'to': mode,
                                     'at': time.perf_counter()})
            return orig_set(mode)

        precision.get_precision = counting_get
        precision.set_precision = counting_set
        self._orig_get, self._orig_set = orig_get, orig_set
        # NLC term capture: RKS folds enlc into exc (gpu4pyscf/dft/rks.py);
        # wrapping nr_nlc_vxc observes the separate term without changing
        # any computation.
        try:
            from gpu4pyscf.dft import numint as gnumint
            orig_nlc = gnumint.nr_nlc_vxc
            probe = self

            def nlc_probe(*a, **kw):
                out = orig_nlc(*a, **kw)
                if len(out) == 3:
                    probe.last_nlc_energy = _num(out[1])
                return out

            gnumint.nr_nlc_vxc = nlc_probe
            self._orig_nlc = (gnumint, orig_nlc)
        except Exception:
            pass

    def remove(self):
        from gpu4pyscf.lib import precision
        if self._orig_get is not None:
            precision.get_precision = self._orig_get
        if self._orig_set is not None:
            precision.set_precision = self._orig_set
        if self._orig_nlc is not None:
            module, orig = self._orig_nlc
            module.nr_nlc_vxc = orig

    def callback(self, locs):
        if self._orig_get is None:
            return
        try:
            self.cycles.append(
                (int(locs.get('cycle', -1)), _num(locs.get('e_tot')) or 0.0,
                 self._orig_get()))
        except Exception:
            pass

    def switch_info(self, lane):
        '''(switched, cycle, trigger) for the auto lane; none otherwise.'''
        if lane != 'auto':
            return {'switched': False, 'cycle': None, 'trigger': 'none'}
        for i, (cycle, _e, mode) in enumerate(self.cycles):
            if mode == 'fp64' and i > 0 and self.cycles[i - 1][2] == 'fp32':
                trigger = 'unknown'
                if i >= 2:
                    dE = abs(self.cycles[i - 1][1] - self.cycles[i - 2][1])
                    if dE < 1e-4:
                        trigger = 'dE'
                    elif dE > 1e-2:
                        trigger = 'chaos-guard'
                return {'switched': True, 'cycle': cycle + 1,
                        'trigger': trigger}
        return {'switched': False, 'cycle': None, 'trigger': 'none'}


# ---------------------------------------------------------------------------
# cells
# ---------------------------------------------------------------------------

def make_mol(geom, basis, charge=None, spin=None):
    import pyscf
    path = geom if os.path.isabs(geom) else os.path.join(GEOMS, geom)
    default_charge, default_spin = CHARGE_SPIN.get(os.path.basename(geom),
                                                   (0, 0))
    return pyscf.M(atom=path, basis=basis,
                   charge=default_charge if charge is None else charge,
                   spin=default_spin if spin is None else spin, verbose=0)


def _apply_common(mf, args, with_df):
    mf.conv_tol = args.conv_tol
    mf.max_cycle = args.max_cycle
    mf.verbose = 0
    mf.init_guess = args.init_guess
    if with_df and hasattr(mf.with_df, 'direct_scf_tol'):
        mf.with_df.direct_scf_tol = args.direct_scf_tol
    if hasattr(mf, 'grids'):
        mf.grids.level = args.grids_level
    return mf


def _gpu_mol_mf(geom, basis, xc, with_df, auxbasis, args, precision_mode,
                probe=None):
    '''Fresh GPU mol + mean-field object for one independent run.'''
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    mol = make_mol(geom, basis, args.charge, args.spin)
    if xc == 'hf':
        cls = gpu_scf.UHF if mol.spin else gpu_scf.RHF
        mf = cls(mol)
    else:
        cls = gpu_dft.UKS if mol.spin else gpu_dft.RKS
        mf = cls(mol, xc=xc)
    if with_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    _apply_common(mf, args, with_df)
    if precision_mode is not None:
        mf.precision_mode = precision_mode
    if probe is not None:
        mf.callback = probe.callback
    return mol, mf


def _gpu_single_point(geom, basis, xc, with_df, auxbasis, args,
                      precision_mode, probe=None):
    '''One complete independent run: mol.build + CDERI + SCF, fully synced.

    Returns (e, converged, cycles, nlc, scf_summary, phases) where phases
    carries the per-phase wall times (t_mol_s, t_cderi_s, t_scf_s).
    '''
    import cupy as cp
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    mol, mf = _gpu_mol_mf(geom, basis, xc, with_df, auxbasis, args,
                          precision_mode, probe)
    cp.cuda.runtime.deviceSynchronize()
    t_mol = time.perf_counter() - t0

    t_cderi = 0.0
    if with_df:
        t1 = time.perf_counter()
        mf.with_df.build()
        cp.cuda.runtime.deviceSynchronize()
        t_cderi = time.perf_counter() - t1

    t1 = time.perf_counter()
    e = mf.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_scf = time.perf_counter() - t1
    nlc = probe.last_nlc_energy if probe is not None else None
    out = (float(e), bool(mf.converged), int(getattr(mf, 'cycles', 0) or 0),
           nlc, dict(getattr(mf, 'scf_summary', {}) or {}),
           {'t_mol_s': t_mol, 't_cderi_s': t_cderi, 't_scf_s': t_scf})
    mol = mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return out


def _cpu_single_point(geom, basis, xc, auxbasis, args):
    '''CPU PySCF sentinel: same independent-run discipline, no CUDA.'''
    from pyscf import scf as cpu_scf, dft as cpu_dft
    t0 = time.perf_counter()
    mol = make_mol(geom, basis, args.charge, args.spin)
    if xc == 'hf':
        cls = cpu_scf.UHF if mol.spin else cpu_scf.RHF
        mf = cls(mol)
    else:
        cls = cpu_dft.UKS if mol.spin else cpu_dft.RKS
        mf = cls(mol, xc=xc)
    mf = mf.density_fit(auxbasis=auxbasis)
    _apply_common(mf, args, True)
    t_mol = time.perf_counter() - t0
    t1 = time.perf_counter()
    e = mf.kernel()
    t_scf = time.perf_counter() - t1
    return (float(e), bool(mf.converged),
            int(getattr(mf, 'cycles', 0) or 0),
            dict(getattr(mf, 'scf_summary', {}) or {}),
            {'t_mol_s': t_mol, 't_cderi_s': 0.0, 't_scf_s': t_scf})


def _e_terms(summary, nlc):
    '''e_terms block: corrections listed separately; the NLC piece is split
    out of exc (RKS folds enlc into exc).'''
    exc = _num(summary.get('exc'))
    if nlc is not None and exc is not None:
        exc = exc - nlc
    return {'e1': _num(summary.get('e1')), 'coul': _num(summary.get('coul')),
            'exc': exc, 'nlc': nlc,
            'dispersion': _num(summary.get('dispersion')),
            'gcp': _num(summary.get('gcp'))}


def _system_block(geom, mol):
    path = geom if os.path.isabs(geom) else os.path.join(GEOMS, geom)
    return {'geom_file': os.path.basename(path),
            'geom_sha256': sha256_file(path), 'natm': int(mol.natm),
            'nao': int(mol.nao), 'charge': int(mol.charge),
            'spin': int(mol.spin), 'nelec': int(mol.nelectron),
            'cart': bool(mol.cart), 'geom_units': 'angstrom'}


def _grids_block(mf):
    def blk(g):
        if g is None:
            return {'level': None, 'atom_grid': None}
        ag = getattr(g, 'atom_grid', None)
        if isinstance(ag, (tuple, list)):
            ag = list(ag)
        return {'level': getattr(g, 'level', None), 'atom_grid': ag}
    return blk(getattr(mf, 'grids', None)), blk(getattr(mf, 'nlcgrids', None))


def _xc_definition_from(libxc_mod, xc):
    '''Plan section 3.4 XC-definition contract, as a pure function of a
    pyscf.dft.libxc-compatible module (unit-testable without pyscf).

    Records what the alias actually resolves to -- the libxc functional
    decomposition, the HF-exchange coefficients and the range-separation
    parameters -- so a B3LYP row can never be confused with another
    variant: pyscf 'b3lyp' is libxc id 402 (VWN3/VWN_RPA form, the
    Gaussian-compatible one), which differs from 'b3lyp5' (VWN5) by
    ~7 mHa on H2.  Probes fail individually (unknown alias -> None fields)
    rather than dropping the whole block.'''
    def _decomp():
        try:
            _hyb, facs = libxc_mod.parse_xc(xc)
            out = [{'id': int(fid), 'factor': float(fac)}
                   for fid, fac in facs]
            return out or None
        except Exception:
            return None

    def _rsh():
        # rsh_coeff: (omega, alpha=LR_HF, beta=SR_HF-LR_HF)
        try:
            omega, alpha, beta = libxc_mod.rsh_coeff(xc)
            return {'omega': float(omega), 'alpha': float(alpha),
                    'beta': float(beta)}
        except Exception:
            return None

    def _hyb():
        try:
            return float(libxc_mod.hybrid_coeff(xc))
        except Exception:
            return None

    def _nlc():
        # nlc_coeff terms: ((b, C), factor) per VV10 component
        try:
            terms = libxc_mod.nlc_coeff(xc)
            out = [[float(params[0]), float(params[1]), float(fac)]
                   for params, fac in terms]
            return out or None
        except Exception:
            return None

    return {'xc': xc, 'libxc_decomposition': _decomp(),
            'hyb_coeff': _hyb(), 'rsh': _rsh(), 'nlc_params': _nlc()}


def _xc_definition(xc):
    '''Lazy pyscf.dft.libxc wrapper around _xc_definition_from.'''
    from pyscf.dft import libxc
    return _xc_definition_from(libxc, xc)


def _method_block(mol, mf, xc, basis, auxbasis, with_df, args):
    grids, nlcgrids = _grids_block(mf)
    return {'xc': xc, 'basis': basis,
            'xc_definition': _xc_definition(xc),
            'auxbasis': auxbasis if with_df else None,
            'auxbasis': auxbasis if with_df else None,
            'with_df': bool(with_df), 'grids': grids, 'nlcgrids': nlcgrids,
            'dispersion': args.dispersion, 'gcp': args.gcp,
            'conv_tol': args.conv_tol, 'conv_tol_grad': None,
            'direct_scf_tol': args.direct_scf_tol if with_df else None,
            'init_guess': args.init_guess, 'max_cycle': args.max_cycle,
            'open_shell': bool(mol.spin)}


def _timing_block(warmup_s, samples, scf_times, cderi_times, lane):
    '''Statistics are computed FROM the rounded sample list so the stored
    median/min/max are exactly reproducible from samples_s.'''
    samples = [round(s, 6) for s in samples]
    if lane == 'cpu-fp64':
        sync = ('cpu lane: no CUDA device; wall-clock samples, each an '
                'independent fresh mol + mf run')
    else:
        sync = ('device-sync before/after each phase; each sample is an '
                'independent fresh mol + mf run including CDERI build, SCF '
                'with precision switch and fp64 tail')
    return {'warmup_s': round(warmup_s, 6), 'samples_s': samples,
            'median_s': round(statistics.median(samples), 6),
            'min_s': min(samples), 'max_s': max(samples),
            'end_to_end_s': round(statistics.median(samples), 6),
            't_cderi_s': round(statistics.median(cderi_times), 6)
            if cderi_times else 0.0,
            't_scf_s': round(statistics.median(scf_times), 6)
            if scf_times else None,
            # kernel-level decomposition is issue #9's attribution table
            't_xc_grid_s': None, 't_jk_s': None, 't_nlc_s': None,
            't_disp_s': None, 'sync': sync}


def _mem_block(sampler, free_mib, total_mib, headroom, peak_alloc,
               within_budget, explanation=None):
    mem = {'gpu_free_before_mib': round(free_mib, 1) if free_mib else None,
           'gpu_total_mib': round(total_mib, 1) if total_mib else None,
           'gpu_peak_alloc_mib': round(peak_alloc, 1),
           'host_rss_peak_mib': round(sampler.host_rss_peak_mib(), 1),
           'headroom_budget_mib': float(headroom),
           'within_budget': within_budget}
    if explanation:
        mem['budget_explanation'] = explanation
    return mem


def _budget_check(peak_alloc, total_mib, headroom):
    if (peak_alloc + headroom) <= total_mib:
        return True, None
    return False, (f'measured peak allocation {peak_alloc:.0f} MiB + '
                   f'headroom {headroom:.0f} MiB exceeds device total '
                   f'{total_mib:.0f} MiB; cell recorded as measured -- '
                   'stop/escalate before scheduling larger settings')


def _skipped_row(row_key, args, reason, free_mib, total_mib):
    return {'row_key': row_key, 'pilot_stage': args.pilot_stage,
            'status': 'skipped', 'skip_reason': reason,
            'system': None, 'method': None,
            'result': {'mem': _mem_block(MemorySampler(), free_mib,
                                         total_mib, args.headroom_mib,
                                         0.0, True)},
            'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                         'impl_agreement_note': None, 'verdict': 'skipped'},
            'cross_program': [], 'evidence': None}


def run_gpu_cell(geom, basis, xc, lane, args, log):
    '''One row of a GPU lane (fp64 / auto / fp32-screen / df-vs-conv).'''
    import copy

    import cupy as cp
    with_df = lane != 'df-vs-conv'
    mode = {'fp64': 'fp64', 'auto': 'auto', 'fp32-screen': 'fp32',
            'df-vs-conv': 'fp64'}[lane]
    # lane-local convergence tolerance: the fp32-screen floor must not
    # leak into the other lanes' cells of the same invocation
    lane_tol = _lane_conv_tol(lane, args.conv_tol)
    if lane_tol != args.conv_tol:
        args = copy.copy(args)
        args.conv_tol = lane_tol

    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': lane}

    # pre-run headroom gate: if the budget cannot even be guaranteed,
    # record skipped with the measured free memory -- never a dangerous
    # allocation and never a silent omission
    if args.min_free_mib is not None and \
            free_mib - args.headroom_mib < args.min_free_mib:
        log(f'{lane}: skipped (headroom-budget): free {free_mib:.0f} MiB - '
            f'headroom {args.headroom_mib:.0f} MiB < required '
            f'{args.min_free_mib:.0f} MiB')
        return _skipped_row(row_key, args, 'headroom-budget', free_mib,
                            total_mib)

    probe = PrecisionProbe()
    probe.install()
    sampler = MemorySampler()
    try:
        with sampler:
            try:
                log(f'{lane}: warmup ...')
                warm = _gpu_single_point(geom, basis, xc, with_df,
                                         args.auxbasis, args, mode)
                warmup_s = sum(warm[5].values())
            except Exception as exc:
                return _failed_row(row_key, args, exc, sampler, free_mib,
                                   total_mib, lane, log)
            samples, scf_times, cderi_times = [], [], []
            e = converged = cycles = nlc = None
            summary = {}
            for i in range(args.samples):
                log(f'{lane}: sample {i + 1}/{args.samples} ...')
                try:
                    (e, converged, cycles, nlc, summary, phases) = \
                        _gpu_single_point(geom, basis, xc, with_df,
                                          args.auxbasis, args, mode,
                                          probe=probe)
                except Exception as exc:
                    return _failed_row(row_key, args, exc, sampler,
                                       free_mib, total_mib, lane, log)
                samples.append(sum(phases.values()))
                scf_times.append(phases['t_scf_s'])
                cderi_times.append(phases['t_cderi_s'])
    finally:
        probe.remove()

    peak_alloc = max(free_mib - (sampler.min_free_mib or free_mib), 0.0)
    within, expl = _budget_check(peak_alloc, total_mib, args.headroom_mib)
    mol, mf = _gpu_mol_mf(geom, basis, xc, with_df, args.auxbasis, args,
                          None)
    row = {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'system': _system_block(geom, mol),
        'method': _method_block(mol, mf, xc, basis, args.auxbasis, with_df,
                                args),
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': e, 'converged': converged, 'scf_cycles': cycles,
            'e_terms': _e_terms(summary, nlc),
            'timings': _timing_block(warmup_s, samples, scf_times,
                                     cderi_times, lane),
            'mem': _mem_block(sampler, free_mib, total_mib,
                              args.headroom_mib, peak_alloc, within, expl),
            'fp32_entry_hits': dict(probe.hits),
            'precision_switch': probe.switch_info(lane),
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    log(f'{lane}: ok E={e:.12f} converged={converged} cycles={cycles} '
        f'peak={peak_alloc:.0f}MiB hits={probe.hits} '
        f'switch={row["result"]["precision_switch"]}')
    return row


def _failed_row(row_key, args, exc, sampler, free_mib, total_mib, lane, log):
    import cupy as cp
    is_oom = 'OutOfMemory' in type(exc).__name__ or \
        'MemoryAllocation' in repr(exc)
    status = 'oom' if is_oom else 'error'
    peak = max(free_mib - (sampler.min_free_mib or free_mib), 0.0)
    log(f'{lane}: {status}: {exc!r}')
    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass
    return {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'status': status,
        'skip_reason': 'oom-observed' if is_oom else None,
        'system': None, 'method': None,
        'result': {'errors': {
            'type': 'OOM' if is_oom else type(exc).__name__,
            'message': repr(exc)[:2000]},
            'mem': _mem_block(sampler, free_mib, total_mib,
                              args.headroom_mib, peak, None)},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': 'fail'},
        'cross_program': [], 'evidence': None}


def run_cpu_cell(geom, basis, xc, args, log):
    '''cpu-fp64 sentinel row: independent runs, no CUDA compute.'''
    import cupy as cp
    free_b, total_b = cp.cuda.runtime.memGetInfo()
    free_mib, total_mib = free_b / 2**20, total_b / 2**20
    row_key = {'geom': os.path.basename(geom), 'basis': basis, 'xc': xc,
               'lane': 'cpu-fp64'}
    try:
        log('cpu-fp64: warmup ...')
        _w = _cpu_single_point(geom, basis, xc, args.auxbasis, args)
        warmup_s = sum(_w[4].values())
        samples, scf_times, cderi_times = [], [], []
        e = converged = cycles = None
        summary = {}
        for i in range(args.samples):
            log(f'cpu-fp64: sample {i + 1}/{args.samples} ...')
            (e, converged, cycles, summary, phases) = _cpu_single_point(
                geom, basis, xc, args.auxbasis, args)
            samples.append(sum(phases.values()))
            scf_times.append(phases['t_scf_s'])
            cderi_times.append(0.0)
    except Exception as exc:
        log(f'cpu-fp64: error: {exc!r}')
        return {'row_key': row_key, 'pilot_stage': args.pilot_stage,
                'status': 'error', 'skip_reason': None, 'system': None,
                'method': None,
                'result': {'errors': {'type': type(exc).__name__,
                                      'message': repr(exc)[:2000]}},
                'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                             'impl_agreement_note': None, 'verdict': 'fail'},
                'cross_program': [], 'evidence': None}

    mol = make_mol(geom, basis, args.charge, args.spin)
    from pyscf import scf as cpu_scf, dft as cpu_dft
    if xc == 'hf':
        mf = (cpu_scf.UHF if mol.spin else cpu_scf.RHF)(mol)
    else:
        mf = (cpu_dft.UKS if mol.spin else cpu_dft.RKS)(mol, xc=xc)
    mf = mf.density_fit(auxbasis=args.auxbasis)
    _apply_common(mf, args, True)
    row = {
        'row_key': row_key, 'pilot_stage': args.pilot_stage,
        'system': _system_block(geom, mol),
        'method': _method_block(mol, mf, xc, basis, args.auxbasis, True,
                                args),
        'status': 'ok', 'skip_reason': None,
        'result': {
            'e_tot': e, 'converged': converged, 'scf_cycles': cycles,
            'e_terms': _e_terms(summary, None),
            'timings': _timing_block(warmup_s, samples, scf_times,
                                     cderi_times, 'cpu-fp64'),
            # the cpu lane performs no GPU compute; the GPU fields record
            # the machine state, peak GPU allocation of this cell is 0
            'mem': _mem_block(MemorySampler(), free_mib, total_mib,
                              args.headroom_mib, 0.0, True),
            'fp32_entry_hits': {'df_jk': 0, 'eval_rho': 0, 'xc_grid': 0,
                                'grad_ip1': 0, 'other': 0},
            'precision_switch': {'switched': False, 'cycle': None,
                                 'trigger': 'none'},
            'fallback_events': [], 'errors': None},
        'accuracy': {'dE_vs_gpu_fp64': None, 'dE_vs_cpu_fp64': None,
                     'impl_agreement_note': None, 'verdict': None},
        'cross_program': [], 'evidence': None,
    }
    log(f'cpu-fp64: ok E={e:.12f} converged={converged} cycles={cycles}')
    return row


# ---------------------------------------------------------------------------
# row merge + derived columns (pure functions; unit-testable without GPU)
# ---------------------------------------------------------------------------

def key_of(row):
    rk = row['row_key']
    return (rk['geom'], rk['basis'], rk['xc'], rk['lane'])


def upsert(rows, row):
    '''Row-key merge: a rerun replaces its own cell, keeps every other row
    (the resume discipline the plan requires of runners).'''
    k = key_of(row)
    rows[:] = [r for r in rows if key_of(r) != k]
    rows.append(row)


def derive(rows):
    '''Fill accuracy blocks: tier (1) vs gpu-fp64, tier (2) vs cpu-fp64,
    and the DF-approximation delta on df-vs-conv rows.  Never mixes tiers:
    the DF delta lives only on df-vs-conv rows, cross-program numbers never
    enter these fields at all.'''
    by_key = {key_of(r): r for r in rows}
    for row in rows:
        base = key_of(row)[:3]
        lane = row['row_key']['lane']
        acc = row.setdefault('accuracy', {})
        acc.setdefault('dE_vs_gpu_fp64', None)
        acc.setdefault('dE_vs_cpu_fp64', None)
        acc.setdefault('impl_agreement_note', None)
        if row.get('status') != 'ok':
            if acc.get('verdict') is None:
                acc['verdict'] = 'skipped' \
                    if row.get('status') == 'skipped' else 'fail'
            continue
        if lane in ('xref-g16', 'xref-orca'):
            acc['verdict'] = 'no_reference'
            continue
        ref64 = next((r for r in rows if key_of(r)[:3] == base
                      and r['row_key']['lane'] == 'fp64'
                      and r.get('status') == 'ok'), None)
        refcpu = next((r for r in rows if key_of(r)[:3] == base
                       and r['row_key']['lane'] == 'cpu-fp64'
                       and r.get('status') == 'ok'), None)
        e = row['result']['e_tot']
        converged = row['result']['converged']
        if lane == 'df-vs-conv':
            # plan section 3.3: the DF approximation delta is reported ONLY
            # here and never enters the level-(1) precision gate
            acc['dE_df_approximation'] = None
            if ref64 is not None:
                acc['dE_df_approximation'] = e - ref64['result']['e_tot']
                acc['verdict'] = 'pass' if converged else 'fail'
            else:
                acc['verdict'] = 'no_reference' if converged else 'fail'
            continue
        if lane == 'fp64':
            acc['dE_vs_gpu_fp64'] = None    # this row IS the reference
        elif ref64 is not None:
            acc['dE_vs_gpu_fp64'] = e - ref64['result']['e_tot']
        if refcpu is not None and lane != 'cpu-fp64':
            acc['dE_vs_cpu_fp64'] = e - refcpu['result']['e_tot']
            if abs(acc['dE_vs_cpu_fp64']) > \
                    schema_validator.IMPL_NOTE_TOL_HA:
                # named attribution (R8): a lane's deviation from cpu-fp64
                # that exceeds the matched-settings implementation level
                # must be attributed to the lane's own precision policy,
                # never to an implementation difference -- the fp64 lane of
                # the same row group quantifies the implementation-agreement
                # level itself.  The tier-(1) verdict above still applies
                # the 1e-6 gate honestly.
                if lane == 'fp32-screen':
                    acc['impl_agreement_note'] = (
                        'fp32-screen lane: pure-fp32 contractions carry the '
                        'measured DF J/K noise floor (~4e-5 Eh, gpu4pyscf/'
                        'lib/precision.py); this |dE_vs_cpu_fp64| is the '
                        "lane's own precision loss at its floored "
                        'conv_tol, not a matched-settings implementation '
                        'difference (the fp64 lane quantifies the '
                        'implementation-agreement level separately)')
                elif lane == 'auto':
                    # measured live in the issue #5 methoxy UKS sentinel:
                    # the fp32 head can leave the SCF (DIIS history /
                    # density) on a trajectory whose fp64 tail plateaus
                    # ~4e-8 Ha above the fp64-lane solution with the
                    # gradient stalled at ~1.9e-6 (a strict conv_tol_grad
                    # run then fails to converge in 60 cycles: it is a
                    # stall, not slow convergence; same <S^2>, same
                    # solution).  Within the 1e-6 gate; a DIIS reset at
                    # the precision switch is an issue-#10 candidate.
                    acc['impl_agreement_note'] = (
                        'auto lane: the fp32 head leaves the SCF trajectory '
                        '(DIIS history/density) such that the fp64 tail can '
                        'plateau above the fp64-lane solution (measured '
                        '~4e-8 Ha on the methoxy UKS sentinel with |g| '
                        'stalled ~1.9e-6; strict conv_tol_grad=1e-7 then '
                        'fails to converge in 60 cycles, so it is a tail '
                        'stall, not slow convergence); this |dE_vs_cpu_fp64|'
                        ' is the lane\'s own mixed-precision trajectory '
                        'residual, not a matched-settings implementation '
                        'difference (the fp64 lane quantifies the '
                        'implementation-agreement level separately)')
        dE = acc.get('dE_vs_gpu_fp64')
        if not converged:
            acc['verdict'] = 'fail'
        elif dE is None and lane != 'fp64':
            acc['verdict'] = 'no_reference'
        elif dE is None or abs(dE) <= schema_validator.PRECISION_GATE_HA:
            acc['verdict'] = 'pass'
        else:
            acc['verdict'] = 'fail'
    return rows


def attach_cross_program(rows, curation_path, log):
    '''Optional G16 precomputed references -> cross_program blocks.

    Only entries carrying an explicit attach_to_rows mapping are attached,
    always with matched=False and named unmatched fields: machine admission
    is provisional until the manual audit, so no cross-program number may
    ever look gate-eligible (plan sections 3.4/3.6).  Reading precomputed
    outputs only -- this never runs Gaussian.
    '''
    with open(curation_path, encoding='utf-8') as f:
        curation = json.load(f)
    attached = 0
    for entry in curation.get('cases', []):
        if entry.get('classification') == 'rejected':
            continue
        mapping = entry.get('attach_to_rows') or []
        if not mapping or entry.get('g16_energy_hartree') is None:
            continue
        for row in rows:
            rk = row.get('row_key') or {}
            if not any(m.get('geom') == rk.get('geom') and
                       m.get('xc') == rk.get('xc') and
                       m.get('basis') == rk.get('basis') for m in mapping):
                continue
            e = (row.get('result') or {}).get('e_tot')
            row.setdefault('cross_program', []).append({
                'program': 'g16',
                'method_key': entry.get('g16_method_signature', ''),
                'matched': False,
                'unmatched_fields': entry.get('unmatched_fields',
                                              ['manual-audit-pending']),
                'e_tot': entry['g16_energy_hartree'],
                'dE': e - entry['g16_energy_hartree'] if e is not None
                else None})
            attached += 1
    log(f'cross_program: attached {attached} provisional (matched=false) '
        f'G16 reference entries from {curation_path}')
    return rows


# ---------------------------------------------------------------------------
# grid stability (issue #6: meta-GGA grid-sensitivity check)
# ---------------------------------------------------------------------------

def assemble_grid_stability(entries, production_level):
    '''Pure post-processing of the grid-stability scan (unit-testable).

    ``entries`` are raw single-point results sorted by level, each
    ``{level, ngrids, e_tot, converged, scf_cycles, t_s}``.  The assembled
    block records, per level, the energy differences against the production
    grid and against the previous (coarser) level, plus an honest
    trend-based classification:

      * 'converging'     -- the last refinement changed the energy less than
                            the first refinement did (deltas shrinking);
      * 'not-converging' -- the last refinement moved the energy at least as
                            much as the first (grid error still growing or
                            flat at the finest scanned level);
      * 'single-level'   -- only one level was scanned (nothing to compare).

    No absolute pass/fail threshold is invented here: the deltas themselves
    ARE the deliverable (plan issue #6: "确认精度对比所用网格下能量已网格
    收敛，或显式记录网格误差" -- the grid error is always recorded
    explicitly).  The block is a level-(2)-style diagnostic; the level-(1)
    precision gate compares lanes at identical grids, where the grid error
    cancels exactly and never enters the gate.
    '''
    entries = sorted(entries, key=lambda en: en['level'])
    prod_e = next((en['e_tot'] for en in entries
                   if en['level'] == production_level), None)
    out = []
    for i, en in enumerate(entries):
        item = {'level': en['level'], 'ngrids': en.get('ngrids'),
                'e_tot': en['e_tot'], 'converged': en['converged'],
                'scf_cycles': en.get('scf_cycles'),
                't_s': en.get('t_s')}
        item['dE_vs_production'] = (None if prod_e is None
                                    else en['e_tot'] - prod_e)
        item['dE_vs_prev_level'] = (None if i == 0
                                    else en['e_tot'] - entries[i - 1]['e_tot'])
        out.append(item)
    deltas = [abs(out[i]['e_tot'] - out[i - 1]['e_tot'])
              for i in range(1, len(out))]
    if not deltas:
        classification = 'single-level'
    elif deltas[-1] < deltas[0]:
        classification = 'converging'
    else:
        classification = 'not-converging'
    finest = out[-1]
    note = (
        f'grid-stability scan (issue #6, meta-GGA grid sensitivity): fp64 '
        f'single points at levels {[en["level"] for en in out]}; production '
        f'level {production_level} carries a grid error of '
        + (f'{abs(finest["e_tot"] - prod_e):.3e} Ha vs the finest scanned '
           f'level {finest["level"]}' if prod_e is not None else 'n/a') +
        f'; consecutive-level |dE| = {[f"{d:.3e}" for d in deltas]}; '
        f'classification {classification!r} is trend-based (deltas '
        'shrinking vs not), never an acceptance gate -- the level-(1) '
        'precision gate compares lanes at identical grids, where grid error '
        'cancels exactly. Single-shot energies for diagnosis only; t_s is '
        'informational, not a schema timing lane.')
    return {'production_level': int(production_level), 'entries': out,
            'classification': classification, 'note': note}


def run_grid_stability(geom, basis, xc, args, log, levels):
    '''Run the fp64 single points of a grid-stability scan (issue #6).

    Fresh mol + mf per level, DF with the pinned auxbasis, conv_tol and
    init_guess identical to the production lanes; only ``grids.level``
    varies.  The production level is always included in the scan (it is the
    anchor for dE_vs_production).'''
    import copy

    import cupy as cp
    levels = sorted({int(lv) for lv in levels} | {int(args.grids_level)})
    entries = []
    for lv in levels:
        a = copy.copy(args)
        a.grids_level = lv
        mol, mf = _gpu_mol_mf(geom, basis, xc, True, args.auxbasis, a, 'fp64')
        try:
            mf.grids.build()
            ngrids = int(mf.grids.coords.shape[0])
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            e = mf.kernel()
            cp.cuda.runtime.deviceSynchronize()
            entry = {'level': lv, 'ngrids': ngrids, 'e_tot': float(e),
                     'converged': bool(mf.converged),
                     'scf_cycles': int(getattr(mf, 'cycles', 0) or 0),
                     't_s': round(time.perf_counter() - t0, 3)}
        except Exception as exc:
            log(f'grid-stability: level {lv} failed: {exc!r}')
            entry = {'level': lv, 'ngrids': None, 'e_tot': None,
                     'converged': False, 'scf_cycles': None, 't_s': None,
                     'error': repr(exc)[:500]}
        entries.append(entry)
        log(f'grid-stability: level {lv} ngrids={entry["ngrids"]} '
            f'E={entry["e_tot"]} conv={entry["converged"]} '
            f'cyc={entry["scf_cycles"]}')
        mol = mf = None
        cp.get_default_memory_pool().free_all_blocks()
    return assemble_grid_stability(entries, args.grids_level)


# ---------------------------------------------------------------------------
# document assembly
# ---------------------------------------------------------------------------

def build_meta(args):
    import pyscf
    try:
        import cupy as cp
        cupy_version = cp.__version__
    except Exception:
        cupy_version = None
    cpu_model, ram_gb = cpu_ram_info()
    gpu = gpu_static_info()
    return {
        'gpu_name': gpu['gpu_name'],
        'gpu_total_mem_mib': gpu['gpu_total_mem_mib'],
        'driver_version': gpu['driver_version'],
        'cuda_version': cuda_version_str(), 'cpu_model': cpu_model,
        'cpu_threads': os.cpu_count(), 'ram_gb': ram_gb,
        'python': platform.python_version(), 'cupy': cupy_version,
        'pyscf': pyscf.__version__,
        'gpu4pyscf_source_commit': git_full_sha(REPO),
        'gpu4pyscf_source_path': REPO, 'libxc': libxc_version(),
        'mem_headroom_mib': float(args.headroom_mib),
        'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'run_id': str(uuid.uuid4()),
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='rtx4090_runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='RTX 4090 benchmark runner producing '
                    'rtx4090-energy-v1 documents (docs/consumer-gpu-plan.md '
                    'section 4). One geometry x basis x functional per '
                    'invocation; rows merge by row_key across invocations.',
        epilog='Lanes: fp64 (reference), auto (fp32 head + fp64 tail), '
               'fp32-screen (pure fp32; conv_tol floored at 3e-5 for this '
               'lane only, the measured DF J/K noise floor), cpu-fp64 '
               '(PySCF sentinel), df-vs-conv (conventional fp64; DF '
               'approximation delta). Evidence raw+log files are written '
               'next to --out and must live under --evidence-root.')
    p.add_argument('--out', required=True,
                   help='output schema JSON path')
    p.add_argument('--geom', default='methanol.xyz',
                   help='geometry file (name under mixedprec/benchmark/'
                        'geoms or absolute path)')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--xc', default='hf', help="functional ('hf' = DF-RHF)")
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--lane', nargs='+', default=['fp64', 'auto', 'cpu-fp64'],
                   choices=['fp64', 'auto', 'fp32-screen', 'cpu-fp64',
                            'df-vs-conv'])
    p.add_argument('--samples', type=int, default=3,
                   help='independent timed samples per lane (>= 3 for '
                        'schema compliance)')
    p.add_argument('--conv-tol', type=float, default=1e-10)
    p.add_argument('--max-cycle', type=int, default=50)
    p.add_argument('--init-guess', default='minao')
    p.add_argument('--direct-scf-tol', type=float, default=1e-14)
    p.add_argument('--grids-level', type=int, default=3)
    p.add_argument('--grid-stability', nargs='+', type=int, default=None,
                   metavar='LEVEL',
                   help='issue #6 meta-GGA grid check: fp64 single points at '
                        'these grid levels (the production --grids-level is '
                        'always included); recorded as a document-level '
                        'grid_stability block (diagnostic, never gated)')
    p.add_argument('--headroom-mib', type=float, default=2048.0,
                   help='safety headroom budget (MiB)')
    p.add_argument('--min-free-mib', type=float, default=None,
                   help='pre-run gate: skip the cell unless measured free '
                        '- headroom >= this (measured, not estimated)')
    p.add_argument('--pilot-stage', default='S0',
                   choices=['S0', 'S1', 'S2', 'S3', 'S4'])
    p.add_argument('--charge', type=int, default=None)
    p.add_argument('--spin', type=int, default=None)
    p.add_argument('--dispersion', default=None,
                   help='JSON object for method.dispersion (e.g. r2SCAN-3c '
                        'D4 params)')
    p.add_argument('--gcp', default=None)
    p.add_argument('--merge-from', default=None,
                   help='existing schema JSON whose rows are carried over '
                        '(row-key upsert) before this run adds new cells')
    p.add_argument('--xref-g16', default=None,
                   help='optional g16_curation.py output; attaches '
                        'matched=false cross_program entries')
    p.add_argument('--evidence-root', default='/data/results',
                   help='evidence files must live under this root')
    p.add_argument('--no-validate', action='store_true')
    args = p.parse_args(argv)
    cmdline = ' '.join([sys.executable or 'python', os.path.abspath(__file__)]
                       + (sys.argv[1:] if argv is None else list(argv)))

    if args.samples < 3:
        print('error: --samples must be >= 3 (schema rule R1)',
              file=sys.stderr)
        return 2
    if args.dispersion:
        try:
            args.dispersion = json.loads(args.dispersion)
        except ValueError:
            print('error: --dispersion must be a JSON object', file=sys.stderr)
            return 2
    # fp32-screen noise floor: DF J/K ~4e-5 (gpu4pyscf/lib/precision.py);
    # applied per-lane inside run_gpu_cell (_lane_conv_tol) so a screening
    # lane in the same invocation never loosens the other lanes' tolerance

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

    # source-tree provenance guard: the wheel lacks the precision module
    # and silently degrades to fp64 (issue #3 regression)
    import gpu4pyscf
    from gpu4pyscf.lib import precision  # noqa: F401
    if not gpu4pyscf.__file__.startswith(REPO):
        print(f'error: gpu4pyscf resolves to {gpu4pyscf.__file__}, not the '
              f'source tree {REPO}; refusing to run (wheel shadowing)',
              file=sys.stderr)
        return 2
    log(f'source provenance OK: {gpu4pyscf.__file__}')

    rows = []
    if args.merge_from and os.path.isfile(args.merge_from):
        with open(args.merge_from, encoding='utf-8') as f:
            previous = json.load(f)
        rows = [r for r in previous.get('rows', [])
                if r.get('status') in ('ok', 'skipped', 'oom', 'error')]
        log(f'merged {len(rows)} rows from {args.merge_from}')

    meta = build_meta(args)
    log(f'meta: gpu={meta["gpu_name"]} total={meta["gpu_total_mem_mib"]}MiB '
        f'commit={meta["gpu4pyscf_source_commit"][:12]}')

    for lane in args.lane:
        if lane == 'cpu-fp64':
            row = run_cpu_cell(args.geom, args.basis, args.xc, args, log)
        else:
            row = run_gpu_cell(args.geom, args.basis, args.xc, lane, args,
                               log)
        upsert(rows, row)

    if args.xref_g16:
        attach_cross_program(rows, args.xref_g16, log)

    derive(rows)

    grid_stability = None
    if args.grid_stability:
        log(f'grid-stability: scanning levels {args.grid_stability} '
            f'(production level {args.grids_level} included) ...')
        grid_stability = run_grid_stability(args.geom, args.basis, args.xc,
                                            args, log, args.grid_stability)

    # phase 1: immutable raw result + run log
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump({'schema_version': SCHEMA_VERSION, 'meta': meta,
                   'rows': rows,
                   'grid_stability': grid_stability}, f, indent=1)
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    raw_sha, log_sha = sha256_file(raw_path), sha256_file(log_path)

    # phase 2: schema document whose rows anchor the raw artifacts
    for row in rows:
        row['evidence'] = {
            'raw_log': f'{log_path}@{log_sha}',
            'result_json': f'{raw_path}@{raw_sha}',
            'command': cmdline}
    doc = {'schema_version': SCHEMA_VERSION, 'meta': meta, 'rows': rows}
    if grid_stability is not None:
        doc['grid_stability'] = grid_stability
        log(f'grid-stability: classification='
            f'{grid_stability["classification"]}; production level '
            f'{grid_stability["production_level"]} grid error vs finest '
            'scanned level recorded in the block')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)

    if args.no_validate:
        log(f'wrote {out_path} (validation skipped)')
        return 0
    issues = schema_validator.validate_document(
        doc, evidence_root=(args.evidence_root or None))
    if issues:
        log(f'VALIDATION FAILED: {len(issues)} issues')
        for it in issues:
            log(f'  [{it["rule"]}/{it["tag"]}] {it["row"] or ""} '
                f'{it["message"]}')
        return 1
    log(f'wrote {out_path} -- VALID ({len(rows)} rows, raw '
        f'{raw_sha[:12]}, log {log_sha[:12]})')
    return 0


if __name__ == '__main__':
    sys.exit(main())

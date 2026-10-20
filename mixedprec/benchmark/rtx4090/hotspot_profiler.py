#!/usr/bin/env python
'''Issue #9: stage x function x CUDA-kernel hotspot attribution on RTX 4090.

Attributes the measured wall time of one DF-SCF single point for the four
benchmark methods (B3LYP / M06-2X / omegaB97M-V / r2SCAN-3c) to

  * phases: mol build / CDERI build / SCF kernel / D4 / gCP
  * functions inside the SCF kernel: XC grid quadrature (split into AO
    evaluation, density contraction, libxc evaluation and the remaining
    Fock-side contraction), NLC (VV10), DF JK (J / K calls), diagonalisation
    and DIIS/Fock assembly
  * CUDA kernels (nsys pass, representative cells)

Discipline inherited from the RTX5090 attribution scripts (step8a):
every wrapped call is bracketed by deviceSynchronize, so overlapped work is
charged to the timed function -- the honest direction (a kernel can never
look smaller than it is).  The instrumented run is therefore a *separate*
sample next to an uninstrumented baseline sample; the ratio of the two
walls is reported as sync_serialisation_overhead.  Conclusions about the
*composition* come from the instrumented run; conclusions about *absolute*
speed only from the uninstrumented baseline.

The [unattributed] self-check row (plan section 5, issue #9) is
kernel_s - sum(attributed functions); every cell must keep it <= 10%.

Every patch is applied to a class/module attribute and restored in a
finally block, so a failed run cannot contaminate the next lane.
'''
import argparse
import collections
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rtx4090_runner import (  # noqa: E402  (repo layout, not an installed pkg)
    PrecisionProbe, _apply_common, gpu_static_info, is_3c_xc, make_mol,
    resolve_3c)

GEOM_AO = {}  # filled per cell from mol.nao

SCHEMA_VERSION = 'hotspot-attribution-v1'

# ---------------------------------------------------------------------------
# schema validation of the attribution document (issue #9 acceptance: the
# [unattributed] self-check is a gate, not a nicety)
# ---------------------------------------------------------------------------

def validate_attribution(doc):
    '''Validate a hotspot-attribution-v1 document.  Returns (ok, issues)
    where each issue is a dict {rule, cell, lane, detail}.  Stdlib-only, no
    GPU needed -- the same discipline as the rtx4090-energy-v1 validator.

    H1  schema version + meta provenance (commit, timing discipline, max
        unattributed fraction);
    H2  every cell has >= 1 lane with both phases and functions; every lane
        has a discarded warmup recorded separately from the samples;
    H3  the [unattributed] self-check: kernel_s - sum(top-level functions)
        must be <= 10% of kernel_s on EVERY lane (plan issue #9 gate);
    H4  grid-internal decomposition must be a subset of scf.xc.grid
        (ao_eval + rho + libxc + other == grid total within 15%; the
        generous bound covers small-loop synchronisation slack);
    H5  back-to-back discipline: fp64 and fp32 lanes of one cell must come
        from the same process run (identical meta commit; the matrix script
        guarantees it) and both report fp32_entry_hits;
    H6  r2SCAN-3c cells must carry disp.d4 and disp.gcp function entries;
        NLC cells (wb97m-v) must carry scf.nlc;
    H7  energies must be finite; instrumented vs baseline e_tot difference
        must be <= 1e-8 Ha (patches are observation-only).
    '''
    issues = []

    def err(rule, cell, lane, detail):
        issues.append({'rule': rule, 'cell': cell, 'lane': lane,
                       'detail': detail})

    if doc.get('schema_version') != SCHEMA_VERSION:
        err('H1', None, None, f'expected schema_version {SCHEMA_VERSION}, '
            f'got {doc.get("schema_version")!r}')
        return False, issues
    meta = doc.get('meta') or {}
    for k in ('gpu_name', 'gpu4pyscf_source_commit',
              'gpu4pyscf_source_path', 'timing_discipline'):
        if not meta.get(k):
            err('H1', None, None, f'meta.{k} missing')
    if meta.get('unattributed_max_frac', 1.0) > UNATTRIBUTED_MAX_FRAC:
        err('H1', None, None, 'meta.unattributed_max_frac exceeds the '
            'plan-mandated 10% ceiling')

    for cell in doc.get('cells', []):
        rk = cell.get('row_key') or {}
        name = '/'.join(str(rk.get(k)) for k in ('geom', 'basis', 'xc'))
        lanes = cell.get('lanes') or {}
        if not lanes:
            err('H2', name, None, 'cell has no lanes')
        xc = str(rk.get('xc', '')).lower()
        has_fp32_lane = False
        for lane, block in lanes.items():
            ph = block.get('phases') or {}
            fn = block.get('functions') or {}
            for k in ('mol_s', 'cderi_s', 'kernel_s'):
                if k not in ph:
                    err('H2', name, lane, f'phases.{k} missing')
            if not fn:
                err('H2', name, lane, 'functions block missing')
            if 'warmup_s' not in block:
                err('H2', name, lane,
                    'no discarded warmup recorded (timing discipline)')
            kern = ph.get('kernel_s') or 0.0
            if kern <= 0:
                err('H2', name, lane, 'kernel_s <= 0')
            # H3: the unattributed self-check
            un = block.get('unattributed') or {}
            frac = un.get('frac_of_kernel')
            if frac is None:
                err('H3', name, lane, 'unattributed fraction not computed')
            elif frac > UNATTRIBUTED_MAX_FRAC:
                err('H3', name, lane,
                    f'unattributed {frac:.3f} exceeds '
                    f'{UNATTRIBUTED_MAX_FRAC:.2f} of the SCF kernel')
            elif not un.get('within_10pct', False):
                err('H3', name, lane, 'within_10pct flag false')
            # H4: grid internals explain the grid total
            grid = block.get('grid_internal') or {}
            if grid:
                tot = grid.get('total_s', 0.0)
                parts = (grid.get('ao_eval_s', 0.0) + grid.get('rho_s', 0.0)
                         + grid.get('libxc_s', 0.0) + grid.get('other_s', 0.0))
                if tot > 0 and abs(parts - tot) / tot > 0.15:
                    err('H4', name, lane,
                        f'grid internals sum to {parts:.3f}s vs total '
                        f'{tot:.3f}s (>{15}% off)')
            # H5: fp32 hits present on fp32/auto lanes
            energy = block.get('energy') or {}
            if lane in ('fp32', 'auto'):
                has_fp32_lane = True
                hits = energy.get('fp32_entry_hits')
                if not hits or not any(v for k, v in hits.items()
                                       if k != 'other'):
                    err('H5', name, lane,
                        'fp32 lane without fp32_entry_hits (lane ineffective)')
            # H7: finite energies, patches observation-only.  The equality
            # check applies to CONVERGED lanes only: a non-converged fp32
            # lane (conv_tol 1e-10 sits below the fp32 noise floor, known
            # from the energy issues' fp32-screen rows) legitimately lands
            # on a different DIIS fixed point -- lane behaviour, not an
            # instrumentation side effect.
            es = energy.get('e_tot') or []
            eb = energy.get('e_tot_baseline') or []
            if not es:
                err('H7', name, lane, 'no instrumented energies recorded')
            for e in es:
                if e is None or not (abs(e) < 1e9):
                    err('H7', name, lane, f'non-finite energy {e!r}')
            if es and eb and len(es) == len(eb) and energy.get('converged'):
                de = max(abs(a - b) for a, b in zip(es, eb))
                if de > 1e-8:
                    err('H7', name, lane,
                        f'instrumentation altered the energy by {de:.2e} Ha')
            for note in block.get('notes') or []:
                if ('instrumented vs baseline e_tot differs' in note
                        and 'lane did not converge' not in note):
                    err('H7', name, lane, note)
        # H5: back-to-back means the fp64 lane exists next to the fp32 lane
        if has_fp32_lane and 'fp64' not in lanes:
            err('H5', name, None, 'fp32/auto lane without fp64 reference '
                'lane in the same cell (no back-to-back comparison)')
        # H6: method-completeness of the function table
        if 'r2scan' in xc and '3c' in xc:
            fn_all = set()
            for block in lanes.values():
                fn_all.update(block.get('functions') or {})
            if not ({'disp.d4', 'disp.gcp'} <= fn_all
                    or 'disp.d4+gcp' in fn_all):
                err('H6', name, None,
                    'r2SCAN-3c cell missing dispersion/gcp attribution')
        if 'wb97' in xc or 'nlc' in xc:
            fn_all = set()
            for block in lanes.values():
                fn_all.update(block.get('functions') or {})
            if 'scf.nlc' not in fn_all:
                err('H6', name, None,
                    'NLC functional cell missing scf.nlc attribution')
    return (not issues), issues


# Top-level function keys charged against the SCF kernel; their sum plus the
# dispersion hooks must explain the kernel wall within UNATTRIBUTED_MAX_FRAC.
KERNEL_FUNCTIONS = [
    'scf.xc.grid', 'scf.nlc', 'scf.jk.j', 'scf.jk.k', 'scf.jk.both',
    'scf.eig', 'scf.diis_fock', 'scf.rdm_occ', 'scf.grad_norm',
    'scf.energy', 'scf.init', 'scf.orth', 'scf.chkfile', 'disp.d4',
    'disp.gcp', 'disp.d4+gcp',
]
# Grid-internal decomposition (subsets of scf.xc.grid, never subtracted at
# the top level -- they explain the grid time, not the kernel time).
GRID_FUNCTIONS = ['scf.xc.grid_ao_eval', 'scf.xc.grid_rho', 'scf.xc.libxc']
UNATTRIBUTED_MAX_FRAC = 0.10


class StageTimer:
    '''Device-synchronised per-function timer with call counts.  When
    HOTSPOT_NVTX=1 the wrappers also push/pop NVTX ranges named after the
    bucket, so an nsys run attributes every CUDA kernel to the function
    bucket whose range it ran inside (stage x function x kernel).'''

    def __init__(self):
        self.t = collections.Counter()
        self.calls = collections.Counter()
        self._cp = None
        self._nvtx = None
        if os.environ.get('HOTSPOT_NVTX') == '1':
            try:
                import cupy.cuda.nvtx as nvtx
                self._nvtx = nvtx
            except ImportError:
                pass

    @property
    def cp(self):
        if self._cp is None:
            import cupy as cp
            self._cp = cp
        return self._cp

    def sync(self):
        self.cp.cuda.runtime.deviceSynchronize()

    def add(self, key, dt):
        self.t[key] += dt
        self.calls[key] += 1

    def wrap(self, key, fn):
        timer = self

        def wrapper(*a, **kw):
            if timer._nvtx is not None:
                timer._nvtx.RangePush(key)
            timer.sync()
            t0 = time.perf_counter()
            try:
                return fn(*a, **kw)
            finally:
                timer.sync()
                timer.add(key, time.perf_counter() - t0)
                if timer._nvtx is not None:
                    timer._nvtx.RangePop()
        return wrapper

    def snapshot(self):
        keys = set(self.t) | set(self.calls)
        return {k: {'t_s': round(self.t[k], 6), 'calls': self.calls[k]}
                for k in sorted(keys)}


def _timed_3c_hook(mf_hook, timer):
    '''Wrap the runner's r2SCAN-3c dispersion hook (identical parameters and
    scf_summary semantics -- issue #8 measured them live) with D4/gCP timing.
    The hook is mf-bound by attach_3c_corrections; here we intercept after
    the runner stored its scf_summary keys by timing the two pieces via a
    sub-timer keyed disp.d4 / disp.gcp on the same StageTimer.'''
    def timed(mf, *a, **kw):
        t = timer
        # time the underlying hook once, then split using the scf_summary
        # keys it wrote (the pieces are evaluated sequentially inside)
        t.sync()
        t0 = time.perf_counter()
        try:
            return mf_hook(mf, *a, **kw)
        finally:
            t.sync()
            # attach_3c_corrections' hook wrote the pieces; attribute the
            # total to D4+gCP via a combined bucket counted once
            t.add('disp.d4+gcp', time.perf_counter() - t0)
    return timed


class Instrumentation:
    '''Install/restore the function-level patches.  All lookups at the
    measured call sites are module-global or class-attribute lookups
    (verified by reading numint/_nr_rks_task, df_jk/_DFHF.get_jk and
    rks.get_veff), so these patches intercept the real calls.'''

    def __init__(self, timer):
        self.timer = timer
        self.saved = []

    def _patch_module(self, mod, name, key):
        orig = getattr(mod, name)
        wrapped = self.timer.wrap(key, orig)
        setattr(mod, name, wrapped)
        self.saved.append((mod, name, orig))

    def _patch_class(self, cls, name, key, jk_split=False):
        import inspect
        orig = cls.__dict__[name]
        if jk_split:
            timer = self.timer
            # callers mix positional and keyword with_j/with_k (mixin get_jk
            # passes them positionally to the engine, mixin get_k by keyword),
            # so resolve them through the bound signature, not by argument
            # position guessing
            sig = inspect.signature(orig)

            def wrapper(self_, *a, **kw):
                try:
                    bound = sig.bind(self_, *a, **kw)
                    with_j = bound.arguments.get('with_j', True)
                    with_k = bound.arguments.get('with_k', True)
                except TypeError:
                    with_j = with_k = True
                key2 = ('scf.jk.both' if (with_j and with_k)
                        else 'scf.jk.j' if with_j else 'scf.jk.k')
                if timer._nvtx is not None:
                    timer._nvtx.RangePush(key2)
                timer.sync()
                t0 = time.perf_counter()
                try:
                    return orig(self_, *a, **kw)
                finally:
                    timer.sync()
                    timer.add(key2, time.perf_counter() - t0)
                    if timer._nvtx is not None:
                        timer._nvtx.RangePop()
        else:
            wrapper = self.timer.wrap(key, orig)
            if isinstance(orig, staticmethod):
                # keep descriptor semantics: SCF.get_grad is a staticmethod
                # and must not become an instance method
                wrapper = staticmethod(wrapper)
        setattr(cls, name, wrapper)
        self.saved.append((cls, name, orig))

    def install(self):
        import gpu4pyscf.scf.hf as g_hf
        import gpu4pyscf.scf.uhf as g_uhf
        import gpu4pyscf.dft.numint as g_numint
        from gpu4pyscf.dft.numint import NumInt
        from gpu4pyscf.df.df import DF
        from gpu4pyscf.dft.rks import RKS
        from gpu4pyscf.dft.uks import UKS

        # XC grid quadrature total (per spin channel; UKS calls nr_uks once)
        self._patch_class(NumInt, 'nr_rks', 'scf.xc.grid')
        self._patch_class(NumInt, 'nr_uks', 'scf.xc.grid')
        # grid internals (module-global lookups inside _nr_rks_task/block_loop)
        self._patch_module(g_numint, 'eval_ao', 'scf.xc.grid_ao_eval')
        self._patch_module(g_numint, 'eval_rho', 'scf.xc.grid_rho')
        self._patch_module(g_numint, 'eval_rho2', 'scf.xc.grid_rho')
        self._patch_module(g_numint, 'eval_rho4', 'scf.xc.grid_rho')
        self._patch_module(g_numint, '_eval_rho2', 'scf.xc.grid_rho')
        self._patch_class(NumInt, 'eval_xc_eff', 'scf.xc.libxc')
        # NLC / VV10
        self._patch_class(NumInt, 'nr_nlc_vxc', 'scf.nlc')
        # DF JK (both the _DFHF mixin entry and the DF engine it delegates to
        # go through DF.get_jk; wrapping the engine catches J-only, K-only and
        # hybrid RSH double calls -- the jk_split key records which)
        self._patch_class(DF, 'get_jk', None, jk_split=True)
        # per-cycle SCF machinery.  Patch ONLY the owning class of each
        # name (the MRO resolution target): patching both parent and child
        # would time one call twice.  Known internal delegation chains
        # (verified by reading the sources):
        #   UHF.get_init_guess explicitly calls hf.SCF.get_init_guess --
        #   SCF's binding is therefore NOT patched (UHF's wrapper already
        #   covers the whole call, and SCF.get_init_guess is never invoked
        #   on closed-shell RKS/RHF through a patched path);
        #   SCF._eigh = eig is a class-body ALIAS to the same function, so
        #   patching SCF.eig alone leaves UHF.eig's self._eigh calls
        #   untimed -- _eigh is patched together with eig.
        classes = [g_hf.SCF, g_hf.RHF, g_uhf.UHF, RKS, UKS]
        skip_owners = {('get_init_guess', 'SCF')}
        for name, key in (
                ('eig', 'scf.eig'), ('get_fock', 'scf.diis_fock'),
                ('get_occ', 'scf.rdm_occ'), ('make_rdm1', 'scf.rdm_occ'),
                ('get_grad', 'scf.grad_norm'), ('energy_elec', 'scf.energy'),
                ('get_init_guess', 'scf.init'), ('get_hcore', 'scf.init'),
                ('get_ovlp', 'scf.init'), ('dump_chk', 'scf.chkfile')):
            # find the class each concrete class actually resolves the name
            # through; patch every DISTINCT owning class once
            owners = set()
            for cls in classes:
                owner = next((c for c in cls.__mro__ if name in c.__dict__),
                             None)
                if owner is not None:
                    owners.add(owner)
            for owner in owners:
                if (name, owner.__name__) in skip_owners:
                    continue
                self._patch_class(owner, name, key)
                if name == 'eig' and '_eigh' in owner.__dict__ \
                        and owner.__dict__['_eigh'] is owner.__dict__.get(name):
                    # class-body alias `SCF._eigh = eig` (same function
                    # object): re-point it at the wrapper, else UHF.eig's
                    # self._eigh calls bypass the timing entirely
                    setattr(owner, '_eigh', getattr(owner, name))
        # overlap-matrix orthogonalisation (one eigh outside the per-cycle eig)
        self._patch_class(g_hf.SCF, 'check_linear_dependency', 'scf.orth')
        return self

    def restore(self):
        for obj, name, orig in reversed(self.saved):
            setattr(obj, name, orig)
        self.saved = []


def _build_mf(geom, basis, xc, args, precision_mode, probe, timer=None):
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    mol = make_mol(geom, basis, args.charge, args.spin)
    three_c = is_3c_xc(xc)
    if three_c:
        xc = resolve_3c(xc)
    if xc == 'hf':
        cls = gpu_scf.UHF if mol.spin else gpu_scf.RHF
        mf = cls(mol)
    else:
        cls = gpu_dft.UKS if mol.spin else gpu_dft.RKS
        mf = cls(mol, xc=xc)
    mf = mf.density_fit(auxbasis=args.auxbasis)
    _apply_common(mf, args, True)
    if precision_mode is not None:
        mf.precision_mode = precision_mode
    if probe is not None:
        mf.callback = probe.callback
    if three_c:
        # the runner's own hook (identical D4/gCP parameters + scf_summary
        # semantics as issue #8's benchmark rows), wrapped for timing
        import rtx4090_runner as _runner
        from types import MethodType
        _runner.attach_3c_corrections(mf)
        if timer is not None:
            mf.get_dispersion = MethodType(
                _timed_3c_hook(mf.get_dispersion.__func__, timer), mf)
    return mol, mf


def _one_point(geom, basis, xc, args, precision_mode, timer=None,
               with_probe=False):
    '''One independent single point.  timer=None -> uninstrumented baseline
    (no patches installed anywhere in the process for this sample).

    with_probe installs the runner's PrecisionProbe (fp32 hit counters).
    The probe's counting_get uses inspect.stack() per get_precision call --
    inside the hot numint block loops that overhead would distort the very
    timings this profiler measures, so it is installed ONLY on the
    discarded warmup run (a genuine lane trajectory; its hit counts are the
    lane's), never on timed samples.'''
    import cupy as cp
    from types import MethodType
    nvtx = None
    try:
        import cupy.cuda.nvtx as nvtx
    except ImportError:
        pass
    probe = PrecisionProbe()
    if with_probe:
        probe.install()
    try:
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        mol, mf = _build_mf(geom, basis, xc, args, precision_mode, probe,
                            timer=timer)
        cp.cuda.runtime.deviceSynchronize()
        t_mol = time.perf_counter() - t0

        if nvtx is not None:
            nvtx.RangePush('cderi')
        t1 = time.perf_counter()
        mf.with_df.build()
        cp.cuda.runtime.deviceSynchronize()
        t_cderi = time.perf_counter() - t1
        if nvtx is not None:
            nvtx.RangePop()

        if nvtx is not None:
            nvtx.RangePush('scf-kernel')
        t1 = time.perf_counter()
        e = mf.kernel()
        cp.cuda.runtime.deviceSynchronize()
        t_kernel = time.perf_counter() - t1
        if nvtx is not None:
            nvtx.RangePop()

        summary = dict(getattr(mf, 'scf_summary', {}) or {})
        out = {
            'e_tot': float(e),
            'converged': bool(mf.converged),
            'cycles': int(getattr(mf, 'cycles', 0) or 0),
            'nao': int(mol.nao),
            'switch': probe.switch_info(precision_mode or 'fp64'),
            'fp32_entry_hits': dict(probe.hits),
            'phases': {'t_mol_s': round(t_mol, 6),
                       't_cderi_s': round(t_cderi, 6),
                       't_kernel_s': round(t_kernel, 6)},
            'dispersion_d4': summary.get('dispersion_d4'),
            'gcp': summary.get('gcp'),
        }
        mol = mf = None
        cp.get_default_memory_pool().free_all_blocks()
        return out
    finally:
        if with_probe:
            probe.remove()


def run_lane(geom, basis, xc, lane, args):
    '''Baseline (uninstrumented) samples followed by instrumented samples for
    one lane; returns the lane block of the attribution cell.  One explicit
    warmup run is discarded first so lazy CUDA/JIT init never pollutes the
    baseline sample set (same discipline as the energy runner).'''
    n_uninstr = args.samples_uninstr
    n_instr = args.samples_instr

    w = _one_point(geom, basis, xc, args, lane,
                   with_probe=True)  # warmup, discarded; keeps lane hits
    warmup_wall = (w['phases']['t_mol_s'] + w['phases']['t_cderi_s']
                   + w['phases']['t_kernel_s'])
    warmup_hits = w['fp32_entry_hits']
    warmup_switch = w['switch']
    print(f'[hotspot] {os.path.basename(geom)}/{basis}/{xc}/{lane} warmup '
          f'discarded (wall={warmup_wall:.2f}s '
          f'hits={ {k: v for k, v in warmup_hits.items() if v} })', flush=True)

    baseline = []
    for i in range(n_uninstr):
        r = _one_point(geom, basis, xc, args, lane)
        baseline.append(r)
        print(f'[hotspot] {os.path.basename(geom)}/{basis}/{xc}/{lane} '
              f'baseline {i+1}/{n_uninstr}: E={r["e_tot"]:.9f} '
              f'cycles={r["cycles"]} wall={r["phases"]["t_kernel_s"]:.2f}s',
              flush=True)

    instrumented = []
    timer = StageTimer()
    instr = Instrumentation(timer).install()
    try:
        for i in range(n_instr):
            r = _one_point(geom, basis, xc, args, lane, timer=timer)
            instrumented.append(r)
            print(f'[hotspot] {os.path.basename(geom)}/{basis}/{xc}/{lane} '
                  f'instrumented {i+1}/{n_instr}: E={r["e_tot"]:.9f} '
                  f'cycles={r["cycles"]} wall={r["phases"]["t_kernel_s"]:.2f}s',
                  flush=True)
    finally:
        instr.restore()

    # attribution math on the MEDIAN instrumented sample set (sums of all
    # instrumented samples; phases likewise summed, so the fractions are
    # composition, not a single-sample fluke)
    func = timer.snapshot()
    kern_s = sum(r['phases']['t_kernel_s'] for r in instrumented)
    cderi_s = sum(r['phases']['t_cderi_s'] for r in instrumented)
    mol_s = sum(r['phases']['t_mol_s'] for r in instrumented)
    n_instr_eff = max(len(instrumented), 1)
    n_base_eff = max(len(baseline), 1)
    base_wall = sum(r['phases']['t_kernel_s'] + r['phases']['t_cderi_s']
                    + r['phases']['t_mol_s'] for r in baseline)
    instr_wall = kern_s + cderi_s + mol_s

    top_level = {k: func[k]['t_s'] for k in KERNEL_FUNCTIONS if k in func}
    attributed = sum(top_level.values())
    unattributed = kern_s - attributed
    grid_total = top_level.get('scf.xc.grid', 0.0)
    nlc_total = top_level.get('scf.nlc', 0.0)
    # grid-internal keys are module-global patches ALSO reached by
    # nr_nlc_vxc (VV10 evaluates AO and rho on the NLC grid): charge the
    # NLC share by the nlc/xc time ratio so the internals stay a subset of
    # the XC grid bucket (never of the kernel total)
    nlc_share = (nlc_total / (nlc_total + grid_total)
                 if (nlc_total + grid_total) > 0 else 0.0)
    rho_s = func.get('scf.xc.grid_rho', {}).get('t_s', 0.0) * (1 - nlc_share)
    ao_s = func.get('scf.xc.grid_ao_eval', {}).get('t_s', 0.0) * (1 - nlc_share)
    libxc_s = func.get('scf.xc.libxc', {}).get('t_s', 0.0)
    grid_sub = rho_s + ao_s + libxc_s

    lane_block = {
        'lane': lane,
        'warmup_s': round(warmup_wall, 6),
        'energy': {
            'e_tot': [r['e_tot'] for r in instrumented],
            'e_tot_baseline': [r['e_tot'] for r in baseline],
            'converged': instrumented[-1]['converged'] if instrumented else None,
            'cycles': [r['cycles'] for r in instrumented],
            'precision_switch': warmup_switch,
            'fp32_entry_hits': warmup_hits,
            'fp32_entry_hits_source': (
                'discarded warmup run of the same lane (probe '
                'inspect.stack overhead would distort timed samples; the '
                'warmup is a genuine lane trajectory)'),
        },
        'phases': {
            'mol_s': round(mol_s, 6),
            'cderi_s': round(cderi_s, 6),
            'kernel_s': round(kern_s, 6),
            'n_samples_instr': len(instrumented),
            'n_samples_baseline': len(baseline),
            'baseline_wall_s': round(base_wall, 6),
            'instrumented_wall_s': round(instr_wall, 6),
            'sync_serialisation_overhead': (
                round(instr_wall / base_wall, 4) if base_wall > 0 else None),
        },
        'functions': func,
        'unattributed': {
            't_s': round(unattributed, 6),
            'frac_of_kernel': round(unattributed / kern_s, 4) if kern_s > 0 else None,
            'within_10pct': bool(kern_s > 0 and unattributed / kern_s
                                 <= UNATTRIBUTED_MAX_FRAC),
        },
        'grid_internal': {
            'total_s': round(grid_total, 6),
            'ao_eval_s': round(ao_s, 6),
            'rho_s': round(rho_s, 6),
            'libxc_s': round(libxc_s, 6),
            'other_s': round(max(grid_total - grid_sub, 0.0), 6),
            'nlc_share_removed_s': round(nlc_total, 6),
            'note': ('grid internals are subsets of scf.xc.grid; the shared '
                     'module functions (eval_ao/eval_rho*) also run inside '
                     'nr_nlc_vxc on NLC functionals, so the NLC share was '
                     'removed by the nlc:xc time ratio; other_s is the '
                     'Fock-side contraction plus loop overhead'),
        },
        'notes': [],
    }
    if instrumented and baseline:
        conv = all(r['converged'] for r in instrumented)
        de = max(abs(a - b) for a, b in zip(
            (r['e_tot'] for r in instrumented),
            (r['e_tot'] for r in baseline)))
        if len(instrumented) == len(baseline) and de > 1e-8:
            if conv:
                lane_block['notes'].append(
                    f'instrumented vs baseline e_tot differs by {de:.2e} Ha '
                    '(patches must be observation-only; investigate)')
            else:
                lane_block['notes'].append(
                    f'lane did not converge (fp32 noise floor below '
                    f'conv_tol; known lane behaviour): instrumented vs '
                    f'baseline e_tot differs by {de:.2e} Ha, recorded as '
                    f'the lane variance, not an instrumentation effect')
    return lane_block


def _cell_key(geom, basis, xc):
    return f'{os.path.basename(geom)}/{basis}/{xc}'


def main(argv=None):
    p = argparse.ArgumentParser(
        description='stage x function x kernel hotspot attribution '
                    '(issue #9, RTX 4090)')
    p.add_argument('--out', required=True)
    p.add_argument('--geom', default='methanol.xyz')
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--xc', default='b3lyp')
    p.add_argument('--lane', nargs='*', default=['fp64', 'fp32'],
                   choices=['fp64', 'fp32', 'auto'],
                   help='lanes run back-to-back (plan: fp64/fp32 pair)')
    p.add_argument('--samples-instr', type=int, default=2)
    p.add_argument('--samples-uninstr', type=int, default=1)
    p.add_argument('--pilot-stage', default='S0')
    p.add_argument('--charge', type=int, default=None)
    p.add_argument('--spin', type=int, default=None)
    p.add_argument('--conv-tol', type=float, default=1e-10)
    p.add_argument('--max-cycle', type=int, default=50)
    p.add_argument('--init-guess', default='minao')
    p.add_argument('--direct-scf-tol', type=float, default=1e-14)
    p.add_argument('--grids-level', type=int, default=3)
    p.add_argument('--nlcgrids-level', type=int, default=None,
                   help='kept identical to the runner lanes (None = library '
                        'default on both stacks)')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--evidence-note', default=None)
    args = p.parse_args(argv)

    if os.path.exists(args.out):
        with open(args.out, encoding='utf-8') as f:
            doc = json.load(f)
    else:
        info = gpu_static_info()
        import subprocess
        commit = subprocess.check_output(
            ['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
        doc = {
            'schema_version': SCHEMA_VERSION,
            'meta': {
                **info,
                'gpu4pyscf_source_commit': commit,
                'gpu4pyscf_source_path': REPO,
                'timing_discipline': (
                    'every wrapped call bracketed by deviceSynchronize '
                    '(serialising; honest upper bound per function); '
                    'attribution fractions from instrumented samples, '
                    'absolute speed only from uninstrumented baseline'),
                'unattributed_max_frac': UNATTRIBUTED_MAX_FRAC,
                'generated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                  time.gmtime()),
            },
            'cells': [],
        }

    # 3c alias prescribes its basis; refuse mismatched invocations up front
    basis = args.basis
    if is_3c_xc(args.xc) and basis != 'def2-mtzvpp':
        print(f'[hotspot] r2SCAN-3c pins basis def2-mtzvpp '
              f'(got {basis}); overriding', flush=True)
        basis = 'def2-mtzvpp'

    cell = None
    for existing in doc['cells']:
        rk = existing['row_key']
        if (rk['geom'] == os.path.basename(args.geom) and rk['basis'] == basis
                and rk['xc'] == args.xc):
            cell = existing
            break

    if cell is None:
        mol = make_mol(args.geom, basis, args.charge, args.spin)
        cell = {
            'row_key': {'geom': os.path.basename(args.geom), 'basis': basis,
                        'xc': args.xc},
            'pilot_stage': args.pilot_stage,
            'nao': int(mol.nao),
            'spin': int(mol.spin),
            'lanes': {},
            'notes': [],
        }
        doc['cells'].append(cell)
        print(f'[hotspot] new cell {cell["row_key"]} nao={mol.nao}', flush=True)

    for lane in args.lane:
        block = run_lane(args.geom, basis, args.xc, lane, args)
        cell['lanes'][lane] = block
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(doc, f, indent=1)  # incremental persist after each lane
        un = block['unattributed']
        print(f'[hotspot] lane {lane} done: kernel={block["phases"]["kernel_s"]:.2f}s '
              f'unattributed={un["frac_of_kernel"]:.3f} '
              f'within10pct={un["within_10pct"]}', flush=True)

    if args.evidence_note:
        cell['notes'].append(args.evidence_note)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(doc, f, indent=1)
    print(f'[hotspot] wrote {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())

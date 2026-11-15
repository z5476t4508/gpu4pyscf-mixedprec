#!/usr/bin/env python
'''LC-2 (long-chain work unit 2): gradient-path hotspot attribution on the
RTX 4090.

Issue #9 attributed the *SCF* path (stage x function x kernel) for the four
benchmark methods and explicitly deferred the gradient path ("梯度热点需单
独归属"; its doc §5).  Issue #12's gradient rows record only coarse
t_scf/t_grad phases.  This profiler closes that gap with the same discipline
as hotspot_profiler.py (issue #9):

  * every wrapped call is bracketed by deviceSynchronize (a bucket can never
    look smaller than it is; the instrumented run is a separate sample next
    to an uninstrumented baseline and the ratio is reported as
    sync_serialisation_overhead);
  * composition comes from the instrumented run, absolute speed only from
    the uninstrumented baseline;
  * patches are class/module-attribute patches restored in a finally block;
  * the [unattributed] self-check is a gate: t_grad_total minus the sum of
    the top-level buckets must stay <= 10% on every lane;
  * observation-only is verified: the instrumented gradient must equal the
    baseline gradient to <= 1e-8 Eh/Bohr (V5).

Attributed buckets (call sites verified by reading the sources):

  grad.hcore_e1    GradientsBase._hcore_energy (kinetic/nuc/overlap
                   derivative contractions; gpu4pyscf/grad/rhf.py:550)
  grad.e2_total    DF energy_ee (df.grad.{rhf,uhf,rks,uks}.Gradients)
    grad.xc_grid     module get_exc / get_exc_full_response
                     (gpu4pyscf.grad.{rks,uks}; called from energy_ee)
    grad.nlc_vv10    module _get_denlc (VV10/NLC gradient; wb97m-v)
    grad.df_jk_ip1   Gradients.jk_energy_per_atom on the four DF gradient
                     classes (CDERI ip1 fill + sum_ejk contraction +
                     auxbasis response; called TWICE for range-separated
                     hybrids -- both calls accumulate into the bucket)
    grad.e2_other    derived: e2_total - the three above (rdm build,
                     grids.build, tensor assembly)
  grad.nuc         GradientsBase.grad_nuc
  grad.disp_d4_gcp instance get_dispersion hook (r2SCAN-3c only;
                   D4+gCP gradient, pyscf-dispersion, deterministic)
  [unattributed]   t_grad_total - top-level buckets (make_rdm1/make_rdm1e,
                   symmetrize, python glue)

Lanes: fp64 (SCF fp64 + gradient fp64) and auto (SCF auto + gradient
resolved from the base policy; the actual kernel lane is MEASURED via the
PrecisionProbe grad_ip1 counter, never assumed from the request).

Each lane: one fresh mol+mf+SCF, one discarded warmup gradient, then N
uninstrumented baseline gradient evaluations and N instrumented evaluations
on the SAME converged mean-field object (the gradient is a deterministic
functional of the converged orbitals; repeating it measures gradient-phase
variance only, which is exactly what this profiler attributes).
'''
import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

from rtx4090_runner import (  # noqa: E402
    PrecisionProbe, _apply_common, gpu_static_info, is_3c_xc, make_mol,
    resolve_3c)
from hotspot_profiler import StageTimer  # noqa: E402  same timer discipline
import gradient_runner  # noqa: E402  3c gradient hook

SCHEMA_VERSION = 'grad-hotspot-attribution-v1'
UNATTRIBUTED_MAX_FRAC = 0.10
E2_OTHER_MAX_FRAC = 0.20      # of e2_total; covers grids.build + rdm build
# V5 tolerance: the instrumented-vs-baseline gradient difference must not
# exceed the lane's own uninstrumented run-to-run variance by a factor 3
# (floor 1e-8 Eh/Bohr).  A fixed 1e-8 would be WRONG for lanes whose
# gradient is genuinely not bit-reproducible: measured on benzene/def2-
# mtzvpp r2scan3c auto, identical unpatched repeats differ by up to 2e-7
# (fp32 ip1 kernel + D4/gCP nondeterministic GPU reductions), so the check
# must compare against that variance, not against zero.
EQUIV_VARIANCE_FACTOR = 3.0
EQUIV_ABS_FLOOR = 1e-8

TOP_LEVEL = ['grad.hcore_e1', 'grad.e2_total', 'grad.nuc', 'grad.disp_d4_gcp']
NESTED = ['grad.xc_grid', 'grad.nlc_vv10', 'grad.df_jk_ip1']

# representative cells: the four methods, closed-shell small/moderate plus
# the UKS sentinel (the LC-1 cell) and the 3c cell (D4+gCP)
CELLS = {
    'b3lyp_methanol_svp':   dict(geom='methanol.xyz', basis='def2-svp',
                                 xc='b3lyp'),
    'b3lyp_benzene_svp':    dict(geom='benzene.xyz', basis='def2-svp',
                                 xc='b3lyp'),
    'm062x_benzene_svp':    dict(geom='benzene.xyz', basis='def2-svp',
                                 xc='m062x'),
    'wb97mv_benzene_svp':   dict(geom='benzene.xyz', basis='def2-svp',
                                 xc='wb97m-v'),
    'wb97mv_methoxy_svp_uks': dict(geom='methoxy.xyz', basis='def2-svp',
                                   xc='wb97m-v'),
    'r2scan3c_benzene_s1':  dict(geom='benzene.xyz', basis='def2-mtzvpp',
                                 xc='r2scan3c'),
}


def _source_commit():
    try:
        return subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=REPO, capture_output=True,
            text=True, check=True).stdout.strip()
    except Exception:
        return None


class GradientPatcher:
    '''Install/restore the gradient-path patches.  Every lookup below was
    verified by reading the call sites: energy_ee resolves get_exc /
    _get_denlc through the *module* namespaces and jk_energy_per_atom
    through class dicts (the DF rks/uks classes hold their own aliases of
    the DF rhf/uhf functions -- each class-dict entry is wrapped around the
    ORIGINAL function object, so a call is counted exactly once and no
    patched entry calls another patched entry).'''

    def __init__(self, timer, grad_obj=None):
        self.timer = timer
        self.saved = []
        self.grad_obj = grad_obj   # for the 3c instance-attr dispersion hook

    def _patch_module(self, mod, name, key):
        orig = getattr(mod, name)
        setattr(mod, name, self.timer.wrap(key, orig))
        self.saved.append((mod, name, orig))

    def _patch_class(self, cls, name, key):
        orig = cls.__dict__[name]
        setattr(cls, name, self.timer.wrap(key, orig))
        self.saved.append((cls, name, orig))

    def install(self):
        from gpu4pyscf.grad import rhf as g_rhf, rks as g_rks, uks as g_uks
        from gpu4pyscf.df.grad import rhf as dg_rhf, uhf as dg_uhf
        from gpu4pyscf.df.grad import rks as dg_rks, uks as dg_uks

        # h1e + overlap-derivative contractions and the nuclear part:
        # GradientsBase owns both bindings for every gradient class
        self._patch_class(g_rhf.GradientsBase, '_hcore_energy',
                          'grad.hcore_e1')
        self._patch_class(g_rhf.GradientsBase, 'grad_nuc', 'grad.nuc')
        # two-electron total (container bucket)
        for cls in (dg_rhf.Gradients, dg_uhf.Gradients,
                    dg_rks.Gradients, dg_uks.Gradients):
            self._patch_class(cls, 'energy_ee', 'grad.e2_total')
        # DF ip1 JK contraction (two calls for range-separated hybrids)
        for cls in (dg_rhf.Gradients, dg_uhf.Gradients,
                    dg_rks.Gradients, dg_uks.Gradients):
            self._patch_class(cls, 'jk_energy_per_atom', 'grad.df_jk_ip1')
        # XC grid quadrature and VV10/NLC gradient pieces (module lookups
        # inside energy_ee; uks._get_denlc is an alias of the rks function
        # object -- both module names are patched so both call sites see the
        # wrapper)
        for mod in (g_rks, g_uks):
            self._patch_module(mod, 'get_exc', 'grad.xc_grid')
            self._patch_module(mod, 'get_exc_full_response', 'grad.xc_grid')
            self._patch_module(mod, '_get_denlc', 'grad.nlc_vv10')
        # 3c dispersion gradient (instance attribute set by the runner hook)
        if self.grad_obj is not None and 'get_dispersion' in vars(
                self.grad_obj):
            orig = self.grad_obj.get_dispersion
            self.grad_obj.get_dispersion = self.timer.wrap(
                'grad.disp_d4_gcp', orig)
            self.saved.append((self.grad_obj, 'get_dispersion', orig))
        return self

    def restore(self):
        for obj, name, orig in reversed(self.saved):
            if isinstance(obj, type):
                setattr(obj, name, orig)
            else:
                setattr(obj, name, orig)
        self.saved = []


def _build_mf(geom, basis, xc, args, precision_mode, probe):
    '''Fresh GPU mol + DF mean-field with the benchmark-matched knobs
    (mirrors gradient_runner._gpu_grad_point).'''
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    mol = make_mol(geom, basis)
    three_c = is_3c_xc(xc)
    xc_run = resolve_3c(xc) or xc
    if xc_run == 'hf':
        cls = gpu_scf.UHF if mol.spin else gpu_scf.RHF
        mf = cls(mol)
    else:
        cls = gpu_dft.UKS if mol.spin else gpu_dft.RKS
        mf = cls(mol, xc=xc_run)
    mf = mf.density_fit(auxbasis=args['auxbasis'])
    _apply_common(mf, _ArgsView(args), True)
    if precision_mode is not None:
        mf.precision_mode = precision_mode
    if probe is not None:
        mf.callback = probe.callback
    if three_c:
        from rtx4090_runner import attach_3c_corrections
        attach_3c_corrections(mf)
    return mol, mf


class _ArgsView:
    '''Minimal attribute view over the args dict for _apply_common.'''

    def __init__(self, d):
        self._d = d

    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)


def run_lane(cell_key, lane, args, samples=3):
    '''One lane: fresh SCF, warmup, N baseline + N instrumented gradient
    evaluations.  Returns the lane record.'''
    import cupy as cp
    from types import MethodType

    cell = CELLS[cell_key]
    scf_mode = 'fp64' if lane == 'fp64' else 'auto'
    probe = PrecisionProbe()
    probe.install()
    try:
        mol, mf = _build_mf(cell['geom'], cell['basis'], cell['xc'], args,
                            scf_mode, probe)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        e = mf.kernel()
        cp.cuda.runtime.deviceSynchronize()
        t_scf = time.perf_counter() - t0
        if not mf.converged:
            raise RuntimeError(f'{cell_key}/{lane}: SCF did not converge')

        g = mf.nuc_grad_method()
        if is_3c_xc(cell['xc']):
            g.get_dispersion = MethodType(
                gradient_runner._r2scan3c_get_dispersion_grad, g)
            g.auxbasis_response = True
        if lane == 'fp64':
            g.precision_mode = 'fp64'
        # lane 'auto': leave g.precision_mode None; the base policy resolves
        # it and the probe MEASURES which kernel lane actually executed

        # warmup (discarded; also builds the XC grids inside energy_ee)
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        de_warm = cp.asnumpy(g.kernel())
        cp.cuda.runtime.deviceSynchronize()

        # baseline samples (uninstrumented; absolute speed source).  Also
        # records the lane's own run-to-run gradient variance: identical
        # repeats are NOT bit-reproducible on every path (measured
        # r2scan3c auto: up to 2e-7 Eh/Bohr between unpatched repeats from
        # the fp32 ip1 kernel + D4/gCP GPU reductions), so the V5
        # observation-only check compares against this variance.
        baseline_s = []
        de_base = None
        base_variants = []
        for _ in range(samples):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            de = g.kernel()
            cp.cuda.runtime.deviceSynchronize()
            baseline_s.append(time.perf_counter() - t0)
            de_base = cp.asnumpy(de)
            base_variants.append(de_base.copy())
        base_var_max = 0.0
        for dv in base_variants[1:]:
            base_var_max = max(base_var_max,
                               float(abs(dv - base_variants[0]).max()))

        # instrumented samples (composition source)
        timer = StageTimer()
        patcher = GradientPatcher(timer, g).install()
        try:
            inst_s = []
            inst_samples = []
            hits_before = probe.hits['grad_ip1']
            for _ in range(samples):
                snap0 = timer.snapshot()
                cp.cuda.runtime.deviceSynchronize()
                t0 = time.perf_counter()
                de = g.kernel()
                cp.cuda.runtime.deviceSynchronize()
                inst_s.append(time.perf_counter() - t0)
                snap1 = timer.snapshot()
                per = {k: {'t_s': round(snap1[k]['t_s'] - snap0.get(k, {'t_s': 0.0})['t_s'], 6),
                           'calls': snap1[k]['calls'] - snap0.get(k, {'calls': 0})['calls']}
                       for k in snap1}
                inst_samples.append({
                    't_grad_total_s': round(inst_s[-1], 6),
                    'buckets': per,
                    'grad_max_abs_diff_vs_baseline': float(
                        abs(cp.asnumpy(de) - de_base).max()),
                })
            grad_hits = probe.hits['grad_ip1'] - hits_before
        finally:
            patcher.restore()

        # aggregate over samples: median per bucket
        def med_bucket(key):
            vals = [s['buckets'][key]['t_s'] for s in inst_samples
                    if key in s['buckets']]
            calls = [s['buckets'][key]['calls'] for s in inst_samples
                     if key in s['buckets']]
            if not vals:
                return None
            return {'t_s': round(statistics.median(vals), 6),
                    'calls': statistics.median(calls)}

        buckets = {}
        for k in set().union(*[set(s['buckets']) for s in inst_samples]):
            buckets[k] = med_bucket(k)

        t_total_med = statistics.median(inst_s)
        top_sum = sum(b['t_s'] for k, b in buckets.items()
                      if k in TOP_LEVEL and b)
        e2 = buckets.get('grad.e2_total')
        nested_sum = sum(b['t_s'] for k, b in buckets.items()
                         if k in NESTED and b)
        lane_rec = {
            'lane': lane,
            'scf': {'e_tot': float(e), 't_scf_s': round(t_scf, 6),
                    'cycles': int(getattr(mf, 'cycles', 0) or 0),
                    'converged': bool(mf.converged)},
            'warmup_s': None,  # folded into the discarded run above
            'baseline': {
                'samples_s': [round(s, 6) for s in baseline_s],
                'median_s': round(statistics.median(baseline_s), 6),
                'run_to_run_grad_var_max': base_var_max,
            },
            'instrumented': {
                'samples_s': [round(s, 6) for s in inst_s],
                'median_s': round(t_total_med, 6),
                'per_sample': inst_samples,
            },
            'buckets': buckets,
            'derived': {
                'unattributed_s': round(t_total_med - top_sum, 6),
                'unattributed_frac': round(
                    (t_total_med - top_sum) / t_total_med, 4),
                'e2_other_s': round(e2['t_s'] - nested_sum, 6) if e2 else None,
                'e2_other_frac_of_e2': round(
                    (e2['t_s'] - nested_sum) / e2['t_s'], 4) if e2 else None,
                'sync_serialisation_overhead': round(
                    t_total_med / statistics.median(baseline_s), 4),
            },
            'fp32_entry': {
                'grad_ip1_hits': int(grad_hits),
                'grad_mode_effective': 'fp32' if grad_hits > 0 else 'fp64',
                'note': ('measured via the precision-probe grad_ip1 counter '
                         'across the instrumented gradient evaluations'),
            },
            'equiv_max_abs_diff': max(
                s['grad_max_abs_diff_vs_baseline'] for s in inst_samples),
            'equiv_tolerance': max(EQUIV_ABS_FLOOR,
                                   EQUIV_VARIANCE_FACTOR * base_var_max),
            'equiv_observation_only': bool(
                max(s['grad_max_abs_diff_vs_baseline']
                    for s in inst_samples) <=
                max(EQUIV_ABS_FLOOR,
                    EQUIV_VARIANCE_FACTOR * base_var_max)),
        }
        return lane_rec
    finally:
        probe.remove()
        import gc
        mol = mf = g = None
        cp.get_default_memory_pool().free_all_blocks()
        gc.collect()


def validate_document(doc):
    '''grad-hotspot-attribution-v1 rules.  Returns (ok, issues).'''
    issues = []

    def err(rule, cell, lane, detail):
        issues.append({'rule': rule, 'cell': cell, 'lane': lane,
                       'detail': detail})

    if doc.get('schema_version') != SCHEMA_VERSION:
        err('V1', None, None, 'wrong schema_version')
    meta = doc.get('meta') or {}
    for k in ('gpu_name', 'gpu4pyscf_source_commit', 'timestamp_utc'):
        if not meta.get(k):
            err('V1', None, None, f'meta.{k} missing')
    for cell in doc.get('cells', []):
        ck = cell['cell']
        if len(cell.get('lanes', [])) < 1:
            err('V2', ck, None, 'no lanes')
        for lane in cell.get('lanes', []):
            lk = lane['lane']
            inst = lane.get('instrumented') or {}
            if len(inst.get('samples_s') or []) < 3:
                err('V2', ck, lk, 'fewer than 3 instrumented samples')
            if not lane.get('baseline', {}).get('samples_s'):
                err('V2', ck, lk, 'no baseline (uninstrumented) samples')
            # V3 unattributed gate
            d = lane.get('derived') or {}
            if d.get('unattributed_frac') is None or \
                    d['unattributed_frac'] > UNATTRIBUTED_MAX_FRAC:
                err('V3', ck, lk,
                    f'unattributed_frac={d.get("unattributed_frac")} '
                    f'> {UNATTRIBUTED_MAX_FRAC}')
            # V4 method-specific buckets
            xc = cell.get('xc')
            b = lane.get('buckets') or {}
            if xc == 'wb97m-v':
                if not b.get('grad.nlc_vv10') or \
                        b['grad.nlc_vv10']['t_s'] <= 0:
                    err('V4', ck, lk, 'wb97m-v lane missing grad.nlc_vv10')
            if xc == 'r2scan3c':
                if not b.get('grad.disp_d4_gcp') or \
                        b['grad.disp_d4_gcp']['t_s'] <= 0:
                    err('V4', ck, lk, 'r2scan3c lane missing '
                                      'grad.disp_d4_gcp')
            if not b.get('grad.df_jk_ip1') or \
                    b['grad.df_jk_ip1']['t_s'] <= 0:
                err('V4', ck, lk, 'missing grad.df_jk_ip1')
            if d.get('e2_other_frac_of_e2') is not None and \
                    d['e2_other_frac_of_e2'] > E2_OTHER_MAX_FRAC:
                err('V4', ck, lk,
                    f'e2_other_frac_of_e2={d["e2_other_frac_of_e2"]} '
                    f'> {E2_OTHER_MAX_FRAC}')
            # V5 observation-only: the instrumented gradient must not
            # deviate from the baseline beyond the lane's own uninstrumented
            # run-to-run variance (factor 3, floor 1e-8) -- a fixed 1e-8
            # would mislabel lanes whose gradient is genuinely not
            # bit-reproducible (measured r2scan3c auto variance ~2e-7)
            if not lane.get('equiv_observation_only'):
                err('V5', ck, lk,
                    f'instrumented-vs-baseline gradient diff '
                    f'{lane.get("equiv_max_abs_diff")} exceeds tolerance '
                    f'{lane.get("equiv_tolerance")} (lane run-to-run '
                    f'variance {(lane.get("baseline") or {}).get("run_to_run_grad_var_max")}')
            # V6 fp32 entry evidence
            fp = lane.get('fp32_entry') or {}
            if 'grad_ip1_hits' not in fp or 'grad_mode_effective' not in fp:
                err('V6', ck, lk, 'fp32 entry evidence missing')
            if lane['lane'] == 'auto' and fp.get('grad_ip1_hits', 0) == 0:
                err('V6', ck, lk,
                    'auto lane executed zero fp32 gradient-kernel entries '
                    '(measured); record honestly, do not assume')
            # V7 sync overhead reported
            if d.get('sync_serialisation_overhead') is None:
                err('V7', ck, lk, 'sync_serialisation_overhead missing')
    return (not issues), issues


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', required=True)
    p.add_argument('--cells', nargs='*', default=list(CELLS),
                   help='cell keys (default: all six)')
    p.add_argument('--lanes', nargs='*', default=['fp64', 'auto'])
    p.add_argument('--samples', type=int, default=3)
    args = p.parse_args(argv)

    argsd = {'auxbasis': 'def2-universal-jkfit', 'conv_tol': 1e-10,
             'max_cycle': 50, 'init_guess': 'minao',
             'direct_scf_tol': 1e-14, 'grids_level': 3,
             'nlcgrids_level': None, 'charge': None, 'spin': None}

    doc = {
        'schema_version': SCHEMA_VERSION,
        'meta': {
            **gpu_static_info(),
            'gpu4pyscf_source_commit': _source_commit(),
            'gpu4pyscf_source_path': REPO,
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                           time.gmtime()),
            'timing_discipline': (
                'per-call deviceSynchronize wrappers (composition source); '
                'uninstrumented baseline samples on the same converged '
                'mean-field are the absolute-speed source; ratio reported '
                'as sync_serialisation_overhead'),
            'scf_settings': argsd,
            'samples_per_lane': args.samples,
        },
        'cells': [],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for ck in args.cells:
        cell = CELLS[ck]
        entry = {'cell': ck, 'xc': cell['xc'], 'geom': cell['geom'],
                 'basis': cell['basis'], 'nao': None, 'lanes': []}
        for lane in args.lanes:
            t0 = time.time()
            rec = run_lane(ck, lane, argsd, samples=args.samples)
            entry['lanes'].append(rec)
            print(f'[{ck}/{lane}] t_grad base='
                  f'{rec["baseline"]["median_s"]:.3f}s inst='
                  f'{rec["instrumented"]["median_s"]:.3f}s unattr='
                  f'{rec["derived"]["unattributed_frac"]:.3f} '
                  f'({time.time()-t0:.0f}s)', flush=True)
        doc['cells'].append(entry)
        # incremental save (resume-safe; a killed run keeps finished lanes)
        with open(args.out, 'w') as f:
            json.dump(doc, f, indent=1)

    ok, issues = validate_document(doc)
    doc['validation'] = {'ok': ok, 'issues': issues,
                         'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                                     time.gmtime())}
    with open(args.out, 'w') as f:
        json.dump(doc, f, indent=1)
    print('VALID' if ok else 'INVALID')
    for i in issues:
        print(' issue:', i)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

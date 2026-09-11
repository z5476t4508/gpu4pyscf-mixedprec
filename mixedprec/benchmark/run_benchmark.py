#!/usr/bin/env python3
"""Rigorous benchmark for the mixed-precision lane (mixedprec/benchmark/).

Design (2026-09-10, user-approved):
  - 4 closed-shell systems (r14 excluded: multi-solution SCF, see STATUS.md),
    x {def2-svp, def2-tzvp} x {DF-RHF, DF-r2SCAN, DF-B3LYP} x {fp64, auto}.
  - Every cell carries accuracy, not just time: energy, gradient (max-norm)
    and Hessian -> projected frequencies (max|dnu|, rms dnu) + thermochemistry
    (dZPE, dS_vib), each against GPU-fp64 AND CPU PySCF where the reference
    exists. CPU Hessians are the long pole; cells without them say so
    explicitly instead of silently omitting the column.
  - Timing rigor: energy+gradient are repeated (--eg-repeat, default 3) and
    reported as the median (first repeat doubles as the CUDA-lazy-init
    warmup); the Hessian is single-shot (documented: several fp64 cells are
    10+ minutes and deterministic kernel-count bound).
  - Azadirachtin/tzvp cells that exceed 32 GB record the OOM as their result
    (a hardware boundary, not a skip).
  - Results land in results/benchmark.json, merged by row key across
    invocations so GPU and CPU phases (separate processes) combine.

Usage:
    .venv/bin/python mixedprec/benchmark/run_benchmark.py            # gpu + cpu E/G
        [--skip-gpu] [--skip-cpu] [--cpu-hessian] [--eg-repeat 3]
        [--geoms ...] [--bases ...] [--xc ...] [--mode fp64 auto]
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
# source tree first: the venv's wheel lacks the mixed-precision code and
# would silently degrade to fp64 (see memory gpu4pyscf-5090-env). NOTE: REPO
# must be the REPO ROOT -- dirname(HERE) is mixedprec/, and pointing sys.path
# there instead silently imported the wheel and cost the auto lane its
# speedup while looking perfectly healthy (measured 2026-09-10).
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(HERE))          # for step9_scorecard
sys.path.insert(0, HERE)                           # for vibanalysis
GEOMS = os.path.join(HERE, 'geoms')
RESULTS = os.path.join(HERE, 'results')
JSON_PATH = os.path.join(RESULTS, 'benchmark.json')

# g16 methanols0.fchk header: "Freq RB3LYP 6-31+G(d,p)"; "Total Energy" record
# of the same file. The geometry in geoms/methanol.xyz is extracted from the
# same fchk by extract_methanol.py, so the energy comparison is apples-to-apples.
G16_METHANOL = {
    'level': 'RB3LYP/6-31+G(d,p)',
    'e': -115.7348716828283,
}

# molecule -> (charge, spin); everything else is neutral singlet.
# r14 (excluded from the default matrix): the xyz header claims
# "charge=1 spin=0" but the stoichiometry C12H21O2N2PRu+ has 181 electrons
# (odd) -- spin=0 is impossible. Take the doublet; open-shell rows route to
# UHF/UKS. Kept for --geoms r14.xyz investigations only.
CHARGE_SPIN = {'r14.xyz': (1, 1)}

from step9_scorecard import cpu_run, gpu_run, timed  # noqa: E402
import vibanalysis  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=REPO, text=True).strip()
    except Exception:
        return 'unknown'


def gpu_name():
    try:
        return subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            text=True).strip().splitlines()[0]
    except Exception:
        return 'unknown'


def make_mol(name, basis):
    import pyscf
    charge, spin = CHARGE_SPIN.get(name, (0, 0))
    return pyscf.M(atom=os.path.join(GEOMS, name), basis=basis,
                   charge=charge, spin=spin, verbose=0)


def gmax(g):
    return float(np.abs(np.asarray(g)).max())


# --------------------------------------------------------------------------
# GPU lanes
# --------------------------------------------------------------------------

def gpu_eg(mol, xc, mode, auxbasis, repeat):
    """Energy+gradient, `repeat` times; returns median timings + last values.

    Repeat 1 doubles as the CUDA lazy-init warmup; the median discards it.
    """
    import cupy as cp
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    if xc == 'hf':
        mf = gpu_scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = gpu_dft.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = mode
    t_scf, t_grad = [], []
    for _ in range(repeat):
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        e = mf.kernel()
        cp.cuda.runtime.deviceSynchronize()
        t_scf.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        g = mf.Gradients().kernel()
        cp.cuda.runtime.deviceSynchronize()
        t_grad.append(time.perf_counter() - t0)
    out = {'e': float(e),
           't_scf': sorted(t_scf)[len(t_scf) // 2],
           't_grad': sorted(t_grad)[len(t_grad) // 2],
           't_scf_all': [round(v, 4) for v in t_scf],
           't_grad_all': [round(v, 4) for v in t_grad],
           'converged': bool(mf.converged),
           'g': np.asarray(cp.asnumpy(g) if hasattr(g, 'get') else g,
                           dtype=np.float64)}
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return out


def gpu_hess(mol, xc, mode, auxbasis):
    """Hessian, single shot (fp64 cells run to tens of minutes; see docstring)."""
    import cupy as cp
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    if xc == 'hf':
        mf = gpu_scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = gpu_dft.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = mode
    (e,), t_scf = timed(lambda: (mf.kernel(),))
    g, t_grad = timed(lambda: mf.Gradients().kernel())
    h, t_hess = timed(lambda: mf.Hessian().kernel())
    out = {'e': float(e), 'converged': bool(mf.converged),
           't_scf': t_scf, 't_grad': t_grad, 't_hess': t_hess,
           'g': np.asarray(cp.asnumpy(g) if hasattr(g, 'get') else g,
                           dtype=np.float64),
           'h': np.asarray(cp.asnumpy(h) if hasattr(h, 'get') else h,
                           dtype=np.float64)}
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return out


# --------------------------------------------------------------------------
# CPU reference
# --------------------------------------------------------------------------

def cpu_cell(mol, xc, auxbasis, do_hess):
    """CPU PySCF reference: energy+gradient (the oracle), Hessian if asked."""
    from pyscf import scf as cpu_scf, dft as cpu_dft
    if xc == 'hf':
        mf = cpu_scf.RHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = cpu_dft.RKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    out = {}
    t0 = time.perf_counter()
    e = mf.kernel()
    out['e'] = float(e)
    out['t_scf'] = time.perf_counter() - t0
    out['converged'] = bool(mf.converged)
    t0 = time.perf_counter()
    out['g'] = np.asarray(mf.Gradients().kernel(), dtype=np.float64)
    out['t_grad'] = time.perf_counter() - t0
    if do_hess:
        t0 = time.perf_counter()
        out['h'] = np.asarray(mf.Hessian().kernel(), dtype=np.float64)
        out['t_hess'] = time.perf_counter() - t0
    return out


def cpu_cell_os(mol, xc, auxbasis, do_hess):
    """Open-shell variant (UHF/UKS); used only via --geoms r14.xyz.

    init_guess='huckel' + max_cycle=100: the default minao guess traps the
    GPU lane's SCF in a wrong state DIIS never escapes (see STATUS.md).
    """
    from pyscf import scf as cpu_scf, dft as cpu_dft
    if xc == 'hf':
        mf = cpu_scf.UHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = cpu_dft.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.init_guess = 'huckel'
    mf.max_cycle = 100
    out = {}
    t0 = time.perf_counter()
    out['e'] = float(mf.kernel())
    out['t_scf'] = time.perf_counter() - t0
    out['converged'] = bool(mf.converged)
    t0 = time.perf_counter()
    out['g'] = np.asarray(mf.Gradients().kernel(), dtype=np.float64)
    out['t_grad'] = time.perf_counter() - t0
    if do_hess:
        t0 = time.perf_counter()
        out['h'] = np.asarray(mf.Hessian().kernel(), dtype=np.float64)
        out['t_hess'] = time.perf_counter() - t0
    return out


# --------------------------------------------------------------------------
# Persistence: merge rows by key, derive error columns from stored g/h
# --------------------------------------------------------------------------

def load_json():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            d = json.load(f)
        return d.get('meta', {}), d.get('rows', [])
    return {}, []


def dump_json(meta, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(JSON_PATH, 'w') as f:
        json.dump({'meta': meta, 'rows': rows}, f, indent=1, default=str)


def key_of(r):
    return (r['geom'], r['basis'], r['xc'], r['lane'])


def upsert(rows, rec):
    k = key_of(rec)
    rows[:] = [r for r in rows if key_of(r) != k]
    rows.append(rec)


def derive(rows):
    """Recompute every cross-lane error column from the stored arrays.

    Runs after every phase so metrics appear as soon as both inputs exist
    (e.g. auto-vs-CPU frequencies as soon as the CPU Hessian lands).
    """
    freq_cache = {}

    def freqs(rec):
        k = key_of(rec)
        if k not in freq_cache:
            mol = make_mol(rec['geom'], rec['basis'])
            freq_cache[k] = vibanalysis.frequencies(mol, np.asarray(rec['h']))
        return freq_cache[k]

    by_key = {key_of(r): r for r in rows}
    for r in rows:
        if 'error' in r:
            continue
        base = (r['geom'], r['basis'], r['xc'])
        r64, ra, rc = (by_key.get(base + (lane,)) for lane in
                       ('fp64', 'auto', 'cpu'))
        if ra is None:
            continue
        g_a = np.asarray(ra['g']) if 'g' in ra else None
        if r64 and 'g' in r64 and g_a is not None:
            ra['gerr_f64'] = float(np.abs(g_a - np.asarray(r64['g'])).max())
        if rc and 'g' in rc and g_a is not None:
            ra['gerr_cpu'] = float(np.abs(g_a - np.asarray(rc['g'])).max())
        if rc and 'e' in rc and 'e' in ra:
            ra['eerr_cpu'] = float(abs(ra['e'] - rc['e']))
        for other, tag in ((r64, 'f64'), (rc, 'cpu')):
            if other and 'h' in other and 'h' in ra:
                nu_a, nu_o = freqs(ra), freqs(other)
                dn = nu_a - nu_o
                zpe_a, s_a = vibanalysis.thermochemistry(
                    make_mol(r['geom'], r['basis']), ra['h'])
                zpe_o, s_o = vibanalysis.thermochemistry(
                    make_mol(r['geom'], r['basis']), other['h'])
                ra[f'dnu_max_{tag}'] = float(np.abs(dn).max())
                ra[f'dnu_rms_{tag}'] = float(np.sqrt((dn ** 2).mean()))
                ra[f'dzpe_{tag}'] = float(zpe_a - zpe_o)
                ra[f'dsvib_{tag}'] = float(s_a - s_o)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geoms', nargs='*', default=[
        'methanol.xyz', '020_Vitamin_C.xyz', '057_Tamoxifen.xyz',
        '095_Azadirachtin.xyz'])
    p.add_argument('--bases', nargs='*', default=['def2-svp', 'def2-tzvp'])
    p.add_argument('--xc', nargs='*', default=['hf', 'r2scan', 'b3lyp'])
    p.add_argument('--mode', nargs='*', default=['fp64', 'auto'])
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--eg-repeat', type=int, default=3)
    p.add_argument('--skip-gpu', action='store_true')
    p.add_argument('--skip-cpu', action='store_true')
    p.add_argument('--cpu-hessian', action='store_true',
                   help='CPU Hessian reference (the long pole: minutes for '
                        'methanol, hours for Tamoxifen; Azadirachtin is '
                        'deliberately out -- days per functional)')
    args = p.parse_args()

    import cupy as cp
    import pyscf
    from pyscf import lib as pyscf_lib

    meta, rows = load_json()
    meta.update({'git_commit': git_commit(), 'gpu': gpu_name(),
                 'cpu_threads': pyscf_lib.num_threads(),
                 'python': platform.python_version(),
                 'pyscf': pyscf.__version__,
                 'cupy': cp.__version__,
                 'basis_list': args.bases, 'xc_list': args.xc,
                 'auxbasis': args.auxbasis, 'conv_tol': 1e-10,
                 'eg_repeat': args.eg_repeat,
                 'timestamp': time.strftime('%F %T')})
    print(f'meta: {json.dumps(meta)}', flush=True)

    for basis in args.bases:
        for name in args.geoms:
            try:
                mol = make_mol(name, basis)
            except Exception as exc:
                print(f'{name:24s} {basis:10s} BUILD FAILED: {exc!r}',
                      flush=True)
                continue
            for xc in args.xc:
                tag = f'{name:24s} {basis:10s} {xc:7s}'
                # ---- CPU reference --------------------------------------
                if not args.skip_cpu:
                    runner = cpu_cell_os if mol.spin else cpu_cell
                    lane = 'cpu'
                    print(f'{tag} cpu ...', flush=True)
                    try:
                        out = runner(mol, xc, args.auxbasis,
                                     args.cpu_hessian)
                        rec = {'geom': name, 'basis': basis, 'xc': xc,
                               'lane': lane, 'natm': mol.natm,
                               'nao': mol.nao}
                        rec.update({k: out[k] for k in
                                    ('e', 't_scf', 't_grad', 't_hess',
                                     'converged') if k in out})
                        if 'g' in out:
                            rec['g'] = out['g'].tolist()
                        if 'h' in out:
                            rec['h'] = out['h'].tolist()
                        if name == 'methanol.xyz':
                            rec['e_g16'] = G16_METHANOL['e']
                            rec['g16_level'] = G16_METHANOL['level']
                        upsert(rows, rec)
                    except Exception as exc:
                        print(f'{tag} cpu FAILED: {exc!r}', flush=True)
                    dump_json(meta, rows)
                # ---- GPU lanes ------------------------------------------
                if args.skip_gpu:
                    continue
                for mode in args.mode:
                    print(f'{tag} {mode:5s} E/G x{args.eg_repeat} ...',
                          flush=True)
                    try:
                        out = gpu_eg(mol, xc, mode, args.auxbasis,
                                     args.eg_repeat)
                        rec = {'geom': name, 'basis': basis, 'xc': xc,
                               'lane': mode, 'natm': mol.natm,
                               'nao': mol.nao}
                        rec.update({k: out[k] for k in
                                    ('e', 't_scf', 't_grad', 't_scf_all',
                                     't_grad_all', 'converged')})
                        rec['g'] = out['g'].tolist()
                        if name == 'methanol.xyz':
                            rec['e_g16'] = G16_METHANOL['e']
                            rec['g16_level'] = G16_METHANOL['level']
                        upsert(rows, rec)
                    except Exception as exc:
                        upsert(rows, {'geom': name, 'basis': basis,
                                      'xc': xc, 'lane': mode,
                                      'error': repr(exc)})
                        print(f'{tag} {mode} E/G FAILED: {exc!r}', flush=True)
                    dump_json(meta, rows)
                for mode in args.mode:
                    print(f'{tag} {mode:5s} Hessian ...', flush=True)
                    try:
                        out = gpu_hess(mol, xc, mode, args.auxbasis)
                        prev = next((r for r in rows if key_of(r) ==
                                     (name, basis, xc, mode)), {})
                        rec = dict(prev)          # keep E/G medians + meta
                        rec.update({'geom': name, 'basis': basis,
                                    'xc': xc, 'lane': mode,
                                    'e': out['e'],
                                    'converged': out['converged'],
                                    'g': out['g'].tolist(),
                                    'h': out['h'].tolist(),
                                    't_hess': out['t_hess']})
                        upsert(rows, rec)
                    except Exception as exc:
                        print(f'{tag} {mode} Hessian FAILED: {exc!r}',
                              flush=True)
                        # keep the E/G part of the row; record the OOM
                        prev = next((r for r in rows if key_of(r) ==
                                     (name, basis, xc, mode)), None)
                        if prev is not None:
                            prev['hess_error'] = repr(exc)
                        else:
                            upsert(rows, {'geom': name, 'basis': basis,
                                          'xc': xc, 'lane': mode,
                                          'hess_error': repr(exc)})
                    dump_json(meta, rows)
                cp.get_default_memory_pool().free_all_blocks()
            mol = None
            cp.get_default_memory_pool().free_all_blocks()

    derive(rows)
    dump_json(meta, rows)

    # ---- table ------------------------------------------------------------
    hdr = (f'  {"geom":22s} {"basis":10s} {"xc":7s} {"lane":5s} '
           f'{"t_scf":>9s} {"t_grad":>9s} {"t_hess":>9s} '
           f'{"dnu/f64":>9s} {"dnu/cpu":>9s} {"|dg|/f64":>9s} {"|dg|/cpu":>9s}')
    print('\n' + hdr, flush=True)
    print('  ' + '-' * (len(hdr) - 2), flush=True)
    for r in sorted(rows, key=lambda r: (r['geom'], r['basis'], r['xc'],
                                         r['lane'])):
        def ts(k):
            return f'{r[k]:8.2f}s' if r.get(k) else '      --'
        def fe(k):
            v = r.get(k)
            return f'{v:9.2e}' if v is not None else '       --'
        notes = []
        if r.get('hess_error'):
            notes.append('HESS-OOM' if 'MemoryAllocation' in r['hess_error']
                         else 'HESS-FAIL')
        if r.get('converged') is False:
            notes.append('NOT-CONV')
        print(f'  {r["geom"]:22s} {r["basis"]:10s} {r["xc"]:7s} '
              f'{r["lane"]:5s} {ts("t_scf")} {ts("t_grad")} {ts("t_hess")} '
              f'{fe("dnu_max_f64")} {fe("dnu_max_cpu")} '
              f'{fe("gerr_f64")} {fe("gerr_cpu")} '
              f'{" ".join(notes)}', flush=True)
        if 'e_g16' in r:
            print(f'  {"":44s}energy vs g16 {r["g16_level"]}: '
                  f'dE = {r["e"] - r["e_g16"]:+.3e} Eh', flush=True)

    print(f'\nresults written to {JSON_PATH}', flush=True)


if __name__ == '__main__':
    main()

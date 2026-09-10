#!/usr/bin/env python3
"""Benchmark runner for mixedprec/benchmark/ -- the three-purpose benchmark set.

  1. external validation  (methanol row prints the energy against g16's
     methanols0.fchk, RB3LYP/6-31+G(d,p), E = -115.7348716828283 Eh);
  2. standard performance benchmark (fixed molecules x methods x modes,
     timed, results/benchmark.json);
  3. (future) gpu4pyscf vs Direwolf comparison, same matrix.

Reuses step9_scorecard's cpu_run/gpu_run (imported, not copied) so the numbers
here are directly comparable to the scorecard's -- same conv_tol (1e-10), same
DF aux basis pinned on both sides (def2-universal-jkfit, see step9_scorecard
for why that pinning matters).

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/benchmark/run_benchmark.py [--hessian] [--cpu] \
        [--geoms methanol.xyz ...] [--xc hf r2scan] [--mode fp64 auto]
"""
import argparse
import json
import os
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
sys.path.insert(0, HERE)
GEOMS = os.path.join(HERE, 'geoms')
RESULTS = os.path.join(HERE, 'results')

# g16 methanols0.fchk header: "Freq RB3LYP 6-31+G(d,p)"; "Total Energy" record
# of the same file. The geometry in geoms/methanol.xyz is extracted from the
# same fchk by extract_methanol.py, so the energy comparison is apples-to-apples.
G16_METHANOL = {
    'level': 'RB3LYP/6-31+G(d,p)',
    'e': -115.7348716828283,
}

# molecule -> (charge, spin); everything else is neutral singlet.
# r14: the xyz header claims "charge=1 spin=0" but the stoichiometry
# C12H21O2N2PRu+ has 181 electrons (odd) -- spin=0 is impossible. Take the
# doublet (Ru(III) d5 low spin is the natural assignment for +1 with neutral
# ligands); the header's spin=0 is a source-data inconsistency, noted in
# STATUS.md.
CHARGE_SPIN = {'r14.xyz': (1, 1)}

from step9_scorecard import cpu_run, gpu_run, timed  # noqa: E402


def gpu_run_os(mol, xc, mode, do_grad, do_hess, auxbasis):
    """gpu_run for open-shell systems: UHF (or UKS) instead of RHF/RKS."""
    import cupy as cp
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft
    out = {}
    if xc == 'hf':
        mf = gpu_scf.UHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = gpu_dft.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = mode
    (e,), out['t_scf'] = timed(lambda: (mf.kernel(),))
    out['e'] = float(e)
    if do_grad:
        g, out['t_grad'] = timed(lambda: mf.Gradients().kernel())
        out['g'] = cp.asnumpy(g) if hasattr(g, 'get') else np.asarray(g)
    if do_hess:
        h, out['t_hess'] = timed(lambda: mf.Hessian().kernel())
        out['h'] = cp.asnumpy(h) if hasattr(h, 'get') else np.asarray(h)
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return out


def cpu_run_os(mol, xc, do_grad, do_hess, auxbasis):
    """cpu_run for open-shell systems."""
    from pyscf import scf as cpu_scf, dft as cpu_dft
    out = {}
    if xc == 'hf':
        mf = cpu_scf.UHF(mol).density_fit(auxbasis=auxbasis)
    else:
        mf = cpu_dft.UKS(mol, xc=xc).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    t0 = time.perf_counter()
    out['e'] = mf.kernel()
    out['t_scf'] = time.perf_counter() - t0
    if do_grad:
        t0 = time.perf_counter()
        out['g'] = mf.Gradients().kernel()
        out['t_grad'] = time.perf_counter() - t0
    return out


def git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(HERE), text=True).strip()
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--geoms', nargs='*', default=[
        'methanol.xyz', '020_Vitamin_C.xyz', '057_Tamoxifen.xyz',
        '095_Azadirachtin.xyz', 'r14.xyz'])
    p.add_argument('--xc', nargs='*', default=['hf', 'r2scan'])
    p.add_argument('--mode', nargs='*', default=['fp64', 'auto'])
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit',
                   help='pinned on both sides (see step9_scorecard: the '
                        'defaults differ and the gap is level of theory, '
                        'not code error)')
    p.add_argument('--hessian', action='store_true',
                   help='also time the Hessian (expensive)')
    p.add_argument('--cpu', action='store_true',
                   help='also run the CPU PySCF reference (slow; run AFTER '
                        'the GPU passes, sequentially, not in parallel)')
    args = p.parse_args()

    import cupy as cp
    import pyscf
    from pyscf import lib as pyscf_lib

    os.makedirs(RESULTS, exist_ok=True)
    meta = {'git_commit': git_commit(), 'gpu': gpu_name(),
            'cpu_threads': pyscf_lib.num_threads(), 'basis': args.basis,
            'auxbasis': args.auxbasis,
            'timestamp': time.strftime('%F %T')}
    print(f'meta: {json.dumps(meta)}', flush=True)

    # grads[(geom, xc, mode)] = gradient array, kept in memory only for the
    # cross-lane error columns; gradients are NOT written to the json.
    grads = {}
    rows = []

    def dump(rs):
        # write incrementally so a crash partway through keeps what ran
        with open(os.path.join(RESULTS, 'benchmark.json'), 'w') as f:
            json.dump({'meta': meta, 'rows': rs}, f, indent=1, default=str)

    def run_one(name, mol, xc, mode):
        rec = {'geom': name, 'xc': xc, 'mode': mode,
               'natm': mol.natm, 'nao': mol.nao}
        runner = gpu_run_os if mol.spin else gpu_run
        try:
            out = runner(mol, xc, mode, True, args.hessian, args.auxbasis)
            rec.update({k: out[k] for k in
                        ('e', 't_scf', 't_grad', 't_hess') if k in out})
            if 'g' in out:
                grads[(name, xc, mode)] = out['g']
        except Exception as exc:            # e.g. r14 SCF failure: mark it
            import traceback                # in the table, never skip silently
            tb = traceback.format_exc().strip().splitlines()[-3:]
            rec['error'] = f'{exc!r} | ' + ' / '.join(tb)
        return rec

    def run_cpu(name, mol, xc):
        rec = {}
        runner = cpu_run_os if mol.spin else cpu_run
        try:
            out = runner(mol, xc, True, False, args.auxbasis)
            rec['e_cpu'] = out['e']
            rec['t_scf_cpu'] = out.get('t_scf')
            grads[(name, xc, 'cpu')] = out['g']
        except Exception as exc:
            rec['error_cpu'] = repr(exc)
        return rec

    for name in args.geoms:
        mol = make_mol(name, args.basis)
        for xc in args.xc:
            ref = run_cpu(name, mol, xc) if args.cpu else {}
            for mode in args.mode:
                print(f'{name:24s} {xc:7s} {mode:5s} ({mol.natm} atoms, '
                      f'{mol.nao} AO) ...', flush=True)
                rec = run_one(name, mol, xc, mode)
                rec.update(ref)
                if name == 'methanol.xyz' and 'e' in rec:
                    rec['e_g16'] = G16_METHANOL['e']
                    rec['g16_level'] = G16_METHANOL['level']
                rows.append(rec)
                dump(rows)                  # crash-safe: one row, one write
            cp.get_default_memory_pool().free_all_blocks()
        mol = None

    # ---- cross-lane errors, filled from the in-memory gradients ----------
    for r in rows:
        key = (r['geom'], r['xc'])
        if (key + ('auto',)) in grads and (key + ('fp64',)) in grads:
            r['gerr_f64'] = float(np.abs(
                grads[key + ('auto',)] - grads[key + ('fp64',)]).max())
        if (key + ('auto',)) in grads and (key + ('cpu',)) in grads:
            r['gerr_cpu'] = float(np.abs(
                grads[key + ('auto',)] - grads[key + ('cpu',)]).max())

    with open(os.path.join(RESULTS, 'benchmark.json'), 'w') as f:
        json.dump({'meta': meta, 'rows': rows}, f, indent=1, default=str)    # ---- table -----------------------------------------------------------
    hdr = (f'  {"geom":24s} {"xc":7s} {"mode":6s} {"t_scf":>9s} '
           f'{"t_grad":>9s} {"t_hess":>9s} {"|dg|/f64":>10s} {"|dg|/cpu":>10s}')
    print('\n' + hdr, flush=True)
    print('  ' + '-' * (len(hdr) - 2), flush=True)
    for r in rows:
        if 'error' in r:
            print(f'  {r["geom"]:24s} {r["xc"]:7s} {r["mode"]:6s} '
                  f'FAILED: {r["error"][:60]}', flush=True)
            continue
        def ts(k):
            return f'{r[k]:8.2f}s' if k in r else '      --'
        def fe(k):
            v = r.get(k)
            return f'{v:10.2e}' if v is not None else '        --'
        print(f'  {r["geom"]:24s} {r["xc"]:7s} {r["mode"]:6s} '
              f'{ts("t_scf")} {ts("t_grad")} {ts("t_hess")} '
              f'{fe("gerr_f64")} {fe("gerr_cpu")}', flush=True)
        if 'e_g16' in r:
            print(f'  {"":40s}energy vs g16 {r["g16_level"]} '
                  f'(geometry AND energy from methanols0.fchk): '
                  f'dE = {r["e"] - r["e_g16"]:+.3e} Eh', flush=True)

    print(f'\nresults written to {RESULTS}/benchmark.json', flush=True)


if __name__ == '__main__':
    main()

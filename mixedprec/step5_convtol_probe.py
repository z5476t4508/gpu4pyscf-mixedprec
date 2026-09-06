"""Step 5 probe: where does the SCF time go inside a geometry optimization?

The gradient is now 8% of an optimization step (see STATUS.md), so the SCF is
the only remaining target.  Before changing anything, measure what a full
optimization actually spends per geometry step:

  - how many SCF cycles, split into fp32 cycles and fp64-tail cycles
  - wall time of each cycle
  - |dE| and |g|_orb reached at the end of the SCF
  - the nuclear gmax that the optimizer sees for that geometry

The question this answers: early geometries are far from the minimum, yet the
SCF is converged to conv_tol=1e-9 at every one of them.  How many cycles is
that worth, and how many of them are the expensive fp64-tail ones?

Usage:
    PYTHONPATH=/home/tong/soft/gpu4pyscf .venv/bin/python \
        mixedprec/step5_convtol_probe.py [--basis def2-svp] [--maxsteps 60]
"""
import argparse
import json
import time

import numpy as np
import pyscf
from pyscf.geomopt.geometric_solver import optimize

from gpu4pyscf import scf
from gpu4pyscf.lib import precision

XYZ = '/home/tong/soft/gpu4pyscf/gpu4pyscf/tests/057_Tamoxifen.xyz'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--xyz', default=XYZ)
    p.add_argument('--basis', default='def2-svp')
    p.add_argument('--auxbasis', default='def2-universal-jkfit')
    p.add_argument('--maxsteps', type=int, default=60)
    p.add_argument('--conv-tol', type=float, default=1e-9)
    p.add_argument('--mode', default='auto')
    p.add_argument('--schedule', default=None,
                   help="'auto', or a JSON dict of SCFConvSchedule kwargs")
    p.add_argument('--out', default='/tmp/step5_probe.json')
    args = p.parse_args()

    mol = pyscf.M(atom=args.xyz, basis=args.basis, verbose=0)
    print(f'{mol.natm} atoms, {mol.nao} AOs, basis={args.basis}', flush=True)

    mf = scf.RHF(mol).density_fit(auxbasis=args.auxbasis)
    mf.conv_tol = args.conv_tol
    mf.precision_mode = args.mode
    mf.verbose = 0
    if args.schedule:
        spec = args.schedule
        if spec.startswith('{'):
            spec = json.loads(spec)
        mf.conv_tol_schedule = spec
        print(f'conv_tol_schedule = {spec!r}', flush=True)

    # per-SCF-cycle record, filled by the callback below
    cycles = []
    scf_runs = []

    def cb(envs):
        cycles.append(dict(
            cycle=envs['cycle'] + 1,
            e_tot=float(envs['e_tot']),
            dE=float(abs(envs['e_tot'] - envs['last_hf_e'])),
            gorb=float(envs['norm_gorb']),
            mode=precision.get_precision(),
            t=time.time(),
        ))

    mf.callback = cb

    # wrap the SCF kernel to bracket each run
    scf_cls = type(mf)
    orig_scf_kernel = scf_cls.kernel

    def timed_scf(self, *a, **kw):
        del cycles[:]
        t0 = time.time()
        e = orig_scf_kernel(self, *a, **kw)
        dt = time.time() - t0
        recs = list(cycles)
        # per-cycle wall time: differences of the callback timestamps, with the
        # first cycle measured from the start of the run (includes the CDERI
        # build and the initial guess, which is what we want to see)
        prev = t0
        for r in recs:
            r['dt'] = r['t'] - prev
            prev = r['t']
            r.pop('t')
        scf_runs.append(dict(total=dt, cycles=recs, conv_tol=float(self.conv_tol)))
        return e

    scf_cls.kernel = timed_scf

    g = mf.nuc_grad_method()
    grad_cls = type(g)
    orig_grad_kernel = grad_cls.kernel
    grad_times = []

    def timed_grad(self, *a, **kw):
        t0 = time.time()
        r = orig_grad_kernel(self, *a, **kw)
        grad_times.append(time.time() - t0)
        return r

    grad_cls.kernel = timed_grad

    steps = []

    def opt_cb(envs):
        grad = envs['gradients']
        run = scf_runs[-1]
        n_fp32 = sum(1 for c in run['cycles'] if c['mode'] == 'fp32')
        n_fp64 = sum(1 for c in run['cycles'] if c['mode'] == 'fp64')
        t_fp32 = sum(c['dt'] for c in run['cycles'] if c['mode'] == 'fp32')
        t_fp64 = sum(c['dt'] for c in run['cycles'] if c['mode'] == 'fp64')
        rec = dict(
            step=len(steps) + 1,
            energy=float(envs['energy']),
            gmax=float(np.abs(grad).max()),
            grms=float(np.sqrt((grad ** 2).mean())),
            scf_total=run['total'],
            n_cycles=len(run['cycles']),
            n_fp32=n_fp32, n_fp64=n_fp64,
            t_fp32=t_fp32, t_fp64=t_fp64,
            grad_time=grad_times[-1],
            conv_tol=run['conv_tol'],
            final_dE=run['cycles'][-1]['dE'] if run['cycles'] else None,
            cycles=run['cycles'],
        )
        steps.append(rec)
        print(f"step {rec['step']:3d}  E={rec['energy']:.9f}  gmax={rec['gmax']:.3e}  "
              f"tol={rec['conv_tol']:.1e}  "
              f"scf={rec['scf_total']:6.2f}s ({n_fp32}x fp32 {t_fp32:5.2f}s + "
              f"{n_fp64}x fp64 {t_fp64:5.2f}s)  grad={rec['grad_time']:5.2f}s",
              flush=True)

    t0 = time.time()
    try:
        mol_eq = optimize(mf, maxsteps=args.maxsteps, callback=opt_cb)
        converged = True
    except Exception as exc:                       # geomeTRIC raises on maxsteps
        print(f'optimizer stopped: {type(exc).__name__}: {exc}', flush=True)
        mol_eq = None
        converged = False
    wall = time.time() - t0

    total_scf = sum(s['scf_total'] for s in steps)
    total_grad = sum(s['grad_time'] for s in steps)
    total_fp64 = sum(s['t_fp64'] for s in steps)
    total_fp32 = sum(s['t_fp32'] for s in steps)
    print(f'\nsteps={len(steps)}  wall={wall:.1f}s  scf={total_scf:.1f}s '
          f'(fp32 {total_fp32:.1f}s / fp64 {total_fp64:.1f}s)  grad={total_grad:.1f}s')
    print(f'cycles: {sum(s["n_cycles"] for s in steps)} total, '
          f'{sum(s["n_fp32"] for s in steps)} fp32 + {sum(s["n_fp64"] for s in steps)} fp64')

    with open(args.out, 'w') as f:
        json.dump(dict(args=vars(args), wall=wall, converged=converged,
                       nao=int(mol.nao), steps=steps), f, indent=1)
    print(f'wrote {args.out}')
    if mol_eq is not None:
        np.save(args.out.replace('.json', '_geom.npy'), mol_eq.atom_coords())


if __name__ == '__main__':
    main()

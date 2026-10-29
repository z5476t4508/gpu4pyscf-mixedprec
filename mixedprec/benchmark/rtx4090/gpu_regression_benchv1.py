#!/usr/bin/env python3
"""BENCH-EXEC-01 focused GPU regression (finite scope card
2026-10-30-benchv1-gpu-regression/REGRESSION_SCOPE_CARD.md).

Three families, existing development cells only, one serial GPU pass:

  F1 cache repair: host-resident DF-JK (use_gpu_memory=False) on a
     closed-shell development cell; J/K must complete and the density
     matrix transformed by both host/device CDERI paths must agree
     tightly (the 74974d0 F1 repair class).
  F2 radical failures: methoxy/b3lyp UKS (spin 1) on lanes
     fp64+auto+auto-verify via the production gradient_runner CLI; the
     runner's own validator applies the 1e-6 / 3e-5 gates.
  F3 rigid transformation: aspirin auto lane on original vs
     translated+rotated geometry (fixed documented rotation); energy
     agreement and back-rotated gradient agreement.

Every family writes raw JSON + a verdict line; the script exits nonzero
if any family fails.  Failures are recorded, never retried away.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
MEDIUMCHON = ('/data/work/gpu4pyscf-agent/inventory/'
              'gpu4pyscf-mediumchon-pubchem-20261017-v1')
METHOXY = ('/data/datasets/gpu4pyscf-chemistry-cohort-20261013-v1/'
           'methoxy.xyz')
# fixed, documented rigid transform: R = Rz(.3) @ Ry(.7) @ Rx(1.1),
# COMPUTED from the angles (a hand-typed literal matrix was not
# orthonormal -- the first F3 run deformed the molecule by ~0.2-2.5%
# bond length and honestly FAILED with dE 7.5e-3 Ha; evidence retained,
# erratum recorded in the scope card)
def _rot(a, b, c):
    ca, sa, cb, sb, cc, sc = (math.cos(a), math.sin(a), math.cos(b),
                              math.sin(b), math.cos(c), math.sin(c))
    rz = ((ca, -sa, 0.0), (sa, ca, 0.0), (0.0, 0.0, 1.0))
    ry = ((cb, 0.0, sb), (0.0, 1.0, 0.0), (-sb, 0.0, cb))
    rx = ((1.0, 0.0, 0.0), (0.0, cc, -sc), (0.0, sc, cc))
    def mat(m, n):
        return tuple(tuple(sum(m[i][k] * n[k][j] for k in range(3))
                           for j in range(3)) for i in range(3))
    return mat(mat(rz, ry), rx)


ROT = _rot(0.3, 0.7, 1.1)
SHIFT = (1.5, -2.0, 0.7)                  # Angstrom


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def read_xyz(path):
    with open(path, encoding='utf-8') as f:
        lines = f.read().splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for ln in lines[2:2 + n]:
        parts = ln.split()
        atoms.append((parts[0], [float(x) for x in parts[1:4]]))
    return atoms


def write_xyz(path, atoms, comment):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'{len(atoms)}\n{comment}\n')
        for sym, (x, y, z) in atoms:
            f.write(f'{sym} {x:.8f} {y:.8f} {z:.8f}\n')


def family1_host_dfjk(out_dir, geom):
    '''F1: host-resident CDERI get_jk vs device-resident; the F1 repair
    (host arrays must not masquerade as device fp32 blocks) makes the
    host path take the original loop() with per-call casts.'''
    import numpy as np
    import cupy as cp
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(HERE))))
    from gpu4pyscf import scf as gpu_scf, dft as gpu_dft

    from pyscf import gto
    atoms = read_xyz(geom)
    mol = gto.M(atom=[(s, p) for s, p in atoms], basis='def2-svp',
                charge=0, spin=0, verbose=0)
    results = {}
    for label, host in (('device_cderi', False), ('host_cderi', True)):
        mf = gpu_dft.RKS(mol, xc='b3lyp').density_fit(
            auxbasis='def2-universal-jkfit')
        mf.grids.level = 3
        mf.conv_tol = 1e-10
        if host:
            mf.with_df.use_gpu_memory = False
        t0 = time.time()
        e = mf.kernel()
        cp.cuda.runtime.deviceSynchronize()
        results[label] = {'e_tot': float(e),
                          'converged': bool(mf.converged),
                          'wall_s': round(time.time() - t0, 3),
                          'cderi_is_host': bool(host)}
        del mf
        cp.get_default_memory_pool().free_all_blocks()
    de = abs(results['device_cderi']['e_tot'] -
             results['host_cderi']['e_tot'])
    ok = (results['device_cderi']['converged'] and
          results['host_cderi']['converged'] and de <= 1e-10)
    results['verdict'] = 'PASS' if ok else 'FAIL'
    results['max_dE'] = de
    with open(os.path.join(out_dir, 'f1_host_dfjk.json'), 'w') as f:
        json.dump(results, f, indent=1)
    log(f"F1 host-DFJK: dE={de:.3e} verdict={results['verdict']}")
    return ok


def run_gradient(out_name, out_dir, geom, basis, xc, lanes, spin=None,
                 samples=3, extra_env=None):
    out = os.path.join(out_dir, out_name)
    cmd = [PY, os.path.join(HERE, 'gradient_runner.py'), '--out', out,
           '--geom', geom, '--basis', basis, '--xc', xc,
           '--lane'] + lanes + ['--samples', str(samples)]
    if spin is not None:
        cmd += ['--spin', str(spin)]
    env = dict(os.environ)
    env.update({'PYTHONDONTWRITEBYTECODE': '1'})
    if extra_env:
        env.update(extra_env)
    with open(out + '.log', 'w') as lf:
        rc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                            env=env, timeout=2700).returncode
    with open(out, encoding='utf-8') as f:
        doc = json.load(f)
    return rc, doc


def family2_radical(out_dir):
    '''F2: methoxy UKS radical, the mainline-repair contract lanes.'''
    rc, doc = run_gradient('f2_methoxy_grad.json', out_dir, METHOXY,
                           'def2-svp', 'b3lyp', ['fp64', 'auto'],
                           spin=1)
    rows = {r['row_key']['lane']: r for r in doc['rows']}
    fp64 = rows['fp64']['result']
    auto = rows['auto']['result']
    de = abs(auto['e_tot'] - fp64['e_tot'])
    dg = max(abs(x - y) for ra, rb in zip(auto['gradient'],
                                          fp64['gradient'])
             for x, y in zip(ra, rb))
    ok = (rc == 0 and auto['converged'] and fp64['converged'] and
          de <= 1e-6 and dg <= 3e-5)
    verdict = {'rc': rc, 'max_dE': de, 'max_dg': dg,
               'auto_converged': auto['converged'],
               'fp64_converged': fp64['converged'],
               'verdict': 'PASS' if ok else 'FAIL'}
    with open(os.path.join(out_dir, 'f2_methoxy_verdict.json'),
              'w') as f:
        json.dump(verdict, f, indent=1)
    log(f"F2 methoxy radical: dE={de:.3e} dg={dg:.3e} "
        f"verdict={verdict['verdict']}")
    return ok


def family3_rigid(out_dir, geom):
    '''F3: aspirin auto lane, original vs translated+rotated.'''
    atoms = read_xyz(geom)
    moved = []
    for sym, (x, y, z) in atoms:
        rx = (ROT[0][0] * x + ROT[0][1] * y + ROT[0][2] * z + SHIFT[0])
        ry = (ROT[1][0] * x + ROT[1][1] * y + ROT[1][2] * z + SHIFT[1])
        rz = (ROT[2][0] * x + ROT[2][1] * y + ROT[2][2] * z + SHIFT[2])
        moved.append((sym, (rx, ry, rz)))
    moved_path = os.path.join(out_dir, 'f3_aspirin_rotated.xyz')
    write_xyz(moved_path, moved,
              'aspirin rigid transform Rz(.3)Ry(.7)Rx(1.1)+(1.5,-2,0.7)')

    rc0, doc0 = run_gradient('f3_aspirin_orig.json', out_dir, geom,
                             'def2-svp', 'b3lyp', ['auto'])
    rc1, doc1 = run_gradient('f3_aspirin_rot.json', out_dir, moved_path,
                             'def2-svp', 'b3lyp', ['auto'])
    g0 = doc0['rows'][0]['result']['gradient']
    g1 = doc1['rows'][0]['result']['gradient']
    e0 = doc0['rows'][0]['result']['e_tot']
    e1 = doc1['rows'][0]['result']['e_tot']
    de = abs(e1 - e0)
    # g'(x') = R g(x): pull the rotated-frame gradient back and compare
    dg = 0.0
    for ga, gb in zip(g0, g1):
        back = tuple(sum(ROT[k][i] * gb[k] for k in range(3))
                     for i in range(3))
        dg = max(dg, max(abs(back[i] - ga[i]) for i in range(3)))
    # per-atom norms are the rotation-invariant cross-check
    dn = max(abs(math.sqrt(sum(x * x for x in ga)) -
                 math.sqrt(sum(x * x for x in gb)))
             for ga, gb in zip(g0, g1))
    ok = (rc0 == 0 and rc1 == 0 and de <= 1e-8 and dg <= 3e-5)
    verdict = {'rc_orig': rc0, 'rc_rot': rc1, 'max_dE': de,
               'max_dg_backrotated': dg, 'max_d_atomnorm': dn,
               'transform': {'rotation': ROT, 'shift_angstrom': SHIFT},
               'verdict': 'PASS' if ok else 'FAIL'}
    with open(os.path.join(out_dir, 'f3_rigid_verdict.json'), 'w') as f:
        json.dump(verdict, f, indent=1)
    log(f"F3 rigid transform: dE={de:.3e} dg(back-rotated)={dg:.3e} "
        f"verdict={verdict['verdict']}")
    return ok


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--skip', nargs='*', default=[],
                   help='families to skip (f1 f2 f3)')
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    if 'f1' not in args.skip:
        results['f1'] = family1_host_dfjk(
            args.out_dir, os.path.join(MEDIUMCHON, 'aspirin.xyz'))
    if 'f2' not in args.skip:
        results['f2'] = family2_radical(args.out_dir)
    if 'f3' not in args.skip:
        results['f3'] = family3_rigid(
            args.out_dir, os.path.join(MEDIUMCHON, 'aspirin.xyz'))
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({'families': results,
                   'all_pass': all(results.values()),
                   'recorded_at_utc':
                       time.strftime('%Y-%m-%dT%H:%M:%SZ',
                                     time.gmtime())}, f, indent=1)
    log(f"regression summary: {results}")
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())

"""gpu4pyscf vs Direwolf on the shared benchmark set, same level of theory.

Level pinned on BOTH sides: DF-r2SCAN / def2-svp / aux = ccpvdzjkfit
(Direwolf's stock library lacks def2-universal-jkfit, so the aux migrates
to Direwolf's; ccpvdzjkfit covers C/H/O/N/P). Same geometry files.
Direwolf runs on all 24 CPU threads; gpu4pyscf lanes fp64 + auto on the GPU.

Output: results/direwolf/<cell>.out + a summary table on stdout.
"""
import json
import re
import subprocess
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
sys.path.insert(0, '/home/tong/soft/gpu4pyscf/mixedprec')

import cupy as cp  # noqa: E402
import numpy as np  # noqa: E402
import pyscf  # noqa: E402

GE = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms'
DW = '/home/tong/soft/Direwolf/Direwolf'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/results/direwolf'
AUX = 'ccpvdzjkfit'
BASIS_DW = 'def2svp'
BASIS_GP = 'def2-svp'

CELLS = [('methanol.xyz', 6), ('020_Vitamin_C.xyz', 20),
         ('057_Tamoxifen.xyz', 57), ('095_Azadirachtin.xyz', 95)]


def write_inp(path, name):
    lines = open(f'{GE}/{name}').read().splitlines()[2:]
    atoms = '\n'.join(
        f'{l.split()[0]:2s} {float(l.split()[1]):15.8f} '
        f'{float(l.split()[2]):15.8f} {float(l.split()[3]):15.8f}'
        for l in lines)
    with open(path, 'w') as f:
        f.write(f"&molecule\n imult      = 1\n icharge    = 0\n"
                f" functional = 'R2SCAN'\n baselabel  = '{BASIS_DW}'\n"
                f" J          = 'RI'\n K          = 'RI'\n"
                f" ri_aux_basis = '{AUX}'\n calc_force = .true.\n"
                f" n_threads  = 24\n&end\n\n&atoms\n{atoms}\n&end\n")


def run_direwolf(name):
    stem = name.replace('.xyz', '')
    inp = f'{OUT}/dw_{stem}.inp'
    write_inp(inp, name)
    t0 = time.perf_counter()
    r = subprocess.run([DW, f'dw_{stem}.inp'], cwd=OUT,
                       env={'OMP_NUM_THREADS': '24', 'PATH': '/usr/bin:/bin'},
                       capture_output=True, text=True, timeout=7200)
    wall = time.perf_counter() - t0
    out = open(f'{OUT}/dw_{stem}.out').read()
    e = float(re.search(r'Total Energy \(Hartree\)\s+=\s+(-?\d+\.\d+)',
                        out).group(1))
    fm = re.search(r'forces\(hartree/bohr\):\n((?:\s+-?\d+\.\d+.*\n)+)',
                   out)
    g = np.array([[float(x) for x in ln.split()] for ln in
                  fm.group(1).strip().splitlines()])
    scf = float(re.search(r'PROFILE\s+scf_total\s+([\d.]+)\s*s', out).group(1))
    force = float(re.search(r'PROFILE\s+force\s+([\d.]+)\s*s', out).group(1))
    return {'e': e, 'g': g, 't_scf': scf, 't_grad': force, 't_wall': wall}


def run_gpu(name, mode):
    import gpu4pyscf  # noqa: F401
    from gpu4pyscf import dft
    mol = pyscf.M(atom=f'{GE}/{name}', basis=BASIS_GP, verbose=0)
    mf = dft.RKS(mol, xc='r2scan').density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = mode
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    e = mf.kernel()
    cp.cuda.runtime.deviceSynchronize()
    t_scf = time.perf_counter() - t0
    t0 = time.perf_counter()
    g = np.asarray(mf.Gradients().kernel())
    cp.cuda.runtime.deviceSynchronize()
    t_grad = time.perf_counter() - t0
    conv = bool(mf.converged)
    mf = None
    cp.get_default_memory_pool().free_all_blocks()
    return {'e': float(e), 'g': g, 't_scf': t_scf, 't_grad': t_grad,
            'converged': conv}


rows = []
for name, _ in CELLS:
    print(f'=== {name} ===', flush=True)
    dw = run_direwolf(name)
    print(f'  Direwolf   : E={dw["e"]:.9f} t_scf={dw["t_scf"]:8.2f}s '
          f't_grad={dw["t_grad"]:8.2f}s', flush=True)
    rec = {'geom': name, 'dw_e': dw['e'], 'dw_t_scf': dw['t_scf'],
           'dw_t_grad': dw['t_grad'], 'dw_g': dw['g'].tolist()}
    for mode in ('fp64', 'auto'):
        gp = run_gpu(name, mode)
        de = gp['e'] - dw['e']
        dg = float(np.abs(gp['g'] - dw['g']).max())
        print(f'  gpu {mode:5s}: E={gp["e"]:.9f} t_scf={gp["t_scf"]:8.2f}s '
              f't_grad={gp["t_grad"]:8.2f}s | dE={de:+.3e} '
              f'max|dg|={dg:.3e} conv={gp["converged"]}', flush=True)
        rec[f'{mode}_e'] = gp['e']
        rec[f'{mode}_t_scf'] = gp['t_scf']
        rec[f'{mode}_t_grad'] = gp['t_grad']
        rec[f'{mode}_dE'] = de
        rec[f'{mode}_dg'] = dg
        rec[f'{mode}_conv'] = gp['converged']
    rows.append(rec)
    with open(f'{OUT}/comparison.json', 'w') as f:
        json.dump(rows, f, indent=1)

print('\n=== summary (times: Direwolf 24 CPU threads vs gpu4pyscf RTX5090) ===')
print(f'{"geom":22s} {"E_dE(gpu64-dw)":>15s} {"max|dg|":>10s} '
      f'{"scf: dw/f64":>12s} {"dw/auto":>9s} {"grad: dw/f64":>13s} {"dw/auto":>9s}')
for r in rows:
    print(f'{r["geom"]:22s} {r["fp64_dE"]:+15.3e} {r["fp64_dg"]:10.3e} '
          f'{r["dw_t_scf"]/r["fp64_t_scf"]:11.1f}x '
          f'{r["dw_t_scf"]/r["auto_t_scf"]:8.1f}x '
          f'{r["dw_t_grad"]/r["fp64_t_grad"]:12.1f}x '
          f'{r["dw_t_grad"]/r["auto_t_grad"]:8.1f}x')

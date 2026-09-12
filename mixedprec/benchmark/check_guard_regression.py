"""Acceptance (b): the divergence guard must not disturb healthy auto lanes.

Cases that converged BEFORE the guard, cold-start SCF each, timed:
compare against the benchmark.json reference times.
"""
import json
import sys
import time

sys.path.insert(0, '/home/tong/soft/gpu4pyscf')
import cupy as cp  # noqa: E402
import pyscf  # noqa: E402

d = json.load(open('/home/tong/soft/gpu4pyscf/mixedprec/benchmark/results/benchmark.json'))
ref = {(r['geom'][:3], r['basis'], r['xc'], r['lane']): r for r in d['rows']}

GE = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms'
FILES = {'met': 'methanol.xyz', '020': '020_Vitamin_C.xyz',
         '057': '057_Tamoxifen.xyz', '095': '095_Azadirachtin.xyz'}
from gpu4pyscf import scf, dft  # noqa: E402

AUX = 'def2-universal-jkfit'

for pre, basis, xc in (('057', 'def2-svp', 'hf'), ('057', 'def2-tzvp', 'hf'),
                       ('095', 'def2-svp', 'hf'), ('095', 'def2-tzvp', 'r2scan'),
                       ('met', 'def2-svp', 'b3lyp')):
    mol = pyscf.M(atom=f'{GE}/{FILES[pre]}', basis=basis, verbose=0)
    if xc == 'hf':
        mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    else:
        mf = dft.RKS(mol, xc=xc).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-10
    mf.verbose = 0
    mf.precision_mode = 'auto'
    t0 = time.perf_counter()
    e = mf.kernel()
    dt = time.perf_counter() - t0
    r = ref.get((pre, basis, xc, 'auto'), {})
    t_ref = r.get('t_scf_all', [None])[0]
    e_ref = next((x.get('e') for x in d['rows']
                  if x['geom'][:3] == pre and x['basis'] == basis
                  and x['xc'] == xc and x['lane'] == 'fp64'), None)
    de = f'{float(e) - e_ref:+.2e}' if e_ref else 'n/a'
    print(f'{pre} {basis} {xc:7s}: conv={mf.converged} '
          f't={dt:7.2f}s (ref cold {t_ref}) dE_vs_fp64={de}')
    mf = None
    cp.get_default_memory_pool().free_all_blocks()

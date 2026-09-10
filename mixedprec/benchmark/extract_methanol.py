#!/usr/bin/env python3
"""Extract the methanol geometry from g16's test-suite fchk into an xyz file.

g16/tests/methanols0.fchk is the source of truth (a converged methanol from
the Gaussian test suite); the coordinates are in Bohr.
"""
import sys

FCHK = '/home/tong/soft/g16/tests/methanols0.fchk'
OUT = '/home/tong/soft/gpu4pyscf/mixedprec/benchmark/geoms/methanol.xyz'

# Atomic number -> symbol for the elements methanol can contain
SYMBOLS = {1: 'H', 6: 'C', 7: 'N', 8: 'O'}

lines = open(FCHK).read().splitlines()
coords = []
natoms = None
for i, ln in enumerate(lines):
    if ln.startswith('Atomic numbers'):
        natoms = int(ln.split()[-1])
        zs, ln_rest = [], lines[i + 1:]
        # fchk wraps at 6 ints per line (2E12 format may vary); parse flexibly
        toks = ' '.join(lines[i + 1:i + 6]).split()
        zs = [int(t) for t in toks[:natoms]]
    if ln.startswith('Current cartesian coordinates'):
        nvals = int(ln.split()[-1])
        toks = []
        j = i + 1
        while len(toks) < nvals:
            toks.extend(lines[j].split())
            j += 1
        coords = [float(t) for t in toks[:nvals]]
        break

assert natoms == 6 and len(coords) == 18, (natoms, len(coords))
BOHR = 0.52917721092
xyz = [(SYMBOLS[zs[i]], *[c * BOHR for c in coords[3 * i:3 * i + 3]])
       for i in range(natoms)]
with open(OUT, 'w') as f:
    f.write('6\nMethanol (from g16 tests/methanols0.fchk, Bohr->Angstrom)\n')
    for s, x, y, z in xyz:
        f.write(f'{s:<2s} {x:16.10f} {y:16.10f} {z:16.10f}\n')
print(open(OUT).read())

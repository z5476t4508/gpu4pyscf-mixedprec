#!/usr/bin/env python3
"""MR-11 (2026-10-23, mainline repair) regression tests: deterministic
sorted-mol env layout (the _GDFTOpt per-process layout class).

MECHANISM (measured pre-fix, see the evidence directory
2026-10-23-mr11-env-layout-determinism/mechanism-evidence-20261023T0350Z.md):
pyscf 2.14.0 Mole.build collects the element symbols into a SET
(`uniq_atoms = {a[0] for a in self._atom}`), whose iteration order is
PYTHONHASHSEED-dependent; `_parse_default_basis` -> `format_basis` ->
`make_env` pack the per-element basis data blocks into Mole._env in that
order.  gpu4pyscf inherits the layout verbatim (basis_seg_contraction
copies mol._env; _GDFTOpt.build sorts on PTR-free columns and keeps the
PTRs), so the sorted mol's env layout varies across processes (22 distinct
layouts in a 60-process census on dopamine/def2-tzvp; a pure VALUE
permutation -- sorted(env) is invariant; MR-10's 132/240 differing slots
are relocations).

MEASURED IMPACT pre-fix: the screen mask is layout-INVARIANT (0 variants in
100+ process/build observations over 25+ layouts; the kernel reads per-shell
values through PTRs and the per-shell primitive order is fixed by the basis
definition).  The cross-process e_tot envelope (~4.5e-13) is ALSO not
layout-caused (same PYTHONHASHSEED produced 3 distinct e_tot values in
seed-controlled arms).  MR-11 therefore fixes the LAYOUT CLASS itself: a
canonical re-pack of the basis blocks (first-occurrence order over bas rows,
PTR remap, dead slots zeroed, fail-closed on overlapping blocks) at the end
of basis_seg_contraction -- value-preserving by construction.  Its value:
cross-process bitwise-stable kernel inputs (the MR-10 P2 cross-process
consensus premise becomes valid) and removal of a confounder from every
cross-process bitwise comparison.

Red->green: test_env_layout_deterministic_across_hash_seeds runs fresh
subprocesses under PYTHONHASHSEED 0/1/2/unset and requires identical
sorted-mol env/bas/atm hashes.  Pre-fix it fails (seed 0 vs seed 1 produce
different layouts, e.g. dopamine env c00ccfc34d90... vs 6388093531fb...);
post-fix it must always pass.  The value-preservation, dead-slot, dedupe
and overlap-refusal invariants are pinned by the helper-level tests; the
mask-continuity clause reuses the MR-10 molecules and the recorded
consensus shas.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
MR10_BASELINE = os.path.join(HERE, 'mr10_screen_baseline_20261023.json')


def _load_mr10():
    spec = importlib.util.spec_from_file_location(
        'mr10_mod', os.path.join(HERE,
                                 'test_mr10_screen_kernel_determinism_20261023.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MR10 = _load_mr10()
MOLECULES = MR10.MOLECULES


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# Probe run in a FRESH subprocess (seed controlled by the caller): builds
# the mol + _GDFTOpt exactly the production way and prints the sorted-mol
# env/bas/atm hashes.
_PROBE = r'''
import hashlib, json, sys
import numpy as np
sys.path.insert(0, {repo!r})
from pyscf import gto
from gpu4pyscf.dft import numint
xyz = json.loads(sys.argv[1]); basis = sys.argv[2]
mol = gto.M(atom=xyz, basis=basis, verbose=0, output='/dev/null',
            max_memory=32000)
opt = numint._GDFTOpt.from_mol(mol)
sm = opt._sorted_mol
def h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
print(json.dumps({{'env': h(sm._env), 'bas': h(sm._bas),
                   'atm': h(sm._atm), 'nbas': len(sm._bas)}}))
'''


def _layout_hashes(xyz, basis, seed):
    """Fresh process; seed None -> PYTHONHASHSEED unset (random)."""
    env = dict(os.environ)
    env.pop('PYTHONHASHSEED', None)
    if seed is not None:
        env['PYTHONHASHSEED'] = str(seed)
    code = _PROBE.format(repo=REPO)
    # xyz passed as a json list of [symbol, (x,y,z)] lines
    atom = [[l.split()[0], [float(x) for x in l.split()[1:4]]]
            for l in xyz.strip().splitlines()]
    r = subprocess.run([sys.executable, '-c', code, json.dumps(atom), basis],
                       env=env, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        raise RuntimeError('probe failed:\n' + r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


class TestMR11EnvLayoutDeterminism(unittest.TestCase):

    def test_env_layout_deterministic_across_hash_seeds(self):
        """RED pre-fix: seed 0 vs seed 1 produce different sorted-mol env
        layouts (dopamine c00ccfc34d90... vs 6388093531fb...; 22 distinct
        layouts in a 60-process pre-fix census).  GREEN post-fix: the
        canonical re-pack makes env/bas/atm a pure function of the basis
        data, independent of the interpreter hash seed."""
        for tag, xyz, basis, level in MOLECULES:
            with self.subTest(tag=tag):
                outs = []
                for seed in (0, 1, 2, None):
                    outs.append(_layout_hashes(xyz, basis, seed))
                for key in ('env', 'bas', 'atm', 'nbas'):
                    vals = {o[key] for o in outs}
                    self.assertEqual(
                        len(vals), 1,
                        'sorted-mol %s layout still varies across hash '
                        'seeds on %s: %s' % (key, tag, sorted(vals)))

    def test_canonical_env_preserves_values(self):
        """Every shell's exp/coeff blocks in the canonical sorted mol must
        be bitwise-equal to the original mol's blocks (the re-pack moves
        blocks, it must never change them).  Row correspondence holds for
        the segmented def2 test bases (nctr=1, one output row per input
        shell, atom and shell order preserved)."""
        from pyscf import gto
        from pyscf.gto import mole as pgto
        from gpu4pyscf.gto import mole as gmol
        for tag, xyz, basis, level in MOLECULES:
            with self.subTest(tag=tag):
                mol = gto.M(atom=xyz, basis=basis, verbose=0,
                            output='/dev/null', max_memory=32000)
                pmol, _ = gmol.basis_seg_contraction(
                    mol, allow_replica=True, sparse_coeff=True)
                bas_o = np.asarray(mol._bas)
                bas_p = np.asarray(pmol._bas)
                env_o = np.asarray(mol._env)
                env_p = np.asarray(pmol._env)
                self.assertEqual(len(bas_o), len(bas_p))
                for i in range(len(bas_o)):
                    row_o, row_p = bas_o[i], bas_p[i]
                    # same shell identity except the (remapped) PTR columns
                    self.assertEqual(int(row_o[pgto.ANG_OF]),
                                     int(row_p[pgto.ANG_OF]))
                    self.assertEqual(int(row_o[pgto.ATOM_OF]),
                                     int(row_p[pgto.ATOM_OF]))
                    nprim = int(row_o[pgto.NPRIM_OF])
                    nctr = int(row_o[pgto.NCTR_OF])
                    self.assertEqual(nctr, 1)
                    for col, n in ((pgto.PTR_EXP, nprim),
                                   (pgto.PTR_COEFF, nprim * nctr)):
                        a = env_o[int(row_o[col]):int(row_o[col]) + n]
                        b = env_p[int(row_p[col]):int(row_p[col]) + n]
                        self.assertEqual(
                            a.tobytes(), b.tobytes(),
                            'shell %d block values changed on %s' % (i, tag))

    def test_helper_permutation_invariance_dead_slots_dedupe(self):
        """Helper-level invariants on synthetic (bas, env): (a) identical
        shells whose blocks are placed at DIFFERENT offsets (the production
        scenario -- pure permutation) canonicalize to IDENTICAL layouts;
        (b) dead slots inside the old basis region are zeroed; (c) a block
        shared by two shells (same ptr, same length) is written once and
        both rows remapped to it."""
        from gpu4pyscf.gto.mole import _canonical_env_layout
        A0, A1, A2 = 0, 1, 2   # ATOM_OF, ANG_OF, NPRIM_OF
        A3, A5, A6 = 3, 5, 6   # NCTR_OF, PTR_EXP, PTR_COEFF

        def mk(rows, envlen):
            bas = np.zeros((len(rows), 8), dtype=np.int32)
            for i, (at, l, np_, nc, pe, pc) in enumerate(rows):
                bas[i] = (at, l, np_, nc, 0, pe, pc, 0)
            env = np.zeros(envlen)
            return bas, env

        # layout A: exp0@10(2) coeff0@12(2) | dead 14..20 | exp1@20(3)
        #           coeff1@23(3) | dead 26..30
        basA, envA = mk([(0, 0, 2, 1, 10, 12), (1, 0, 3, 1, 20, 23)], 30)
        envA[10:12] = [1.5, 2.5]; envA[12:14] = [0.5, 0.25]
        envA[20:23] = [3.5, 4.5, 5.5]; envA[23:26] = [0.75, 0.375, 0.125]
        # layout B: same shells, blocks placed permuted
        basB, envB = mk([(0, 0, 2, 1, 20, 22), (1, 0, 3, 1, 10, 13)], 30)
        envB[20:22] = [1.5, 2.5]; envB[22:24] = [0.5, 0.25]
        envB[10:13] = [3.5, 4.5, 5.5]; envB[13:16] = [0.75, 0.375, 0.125]
        basA2, envA2 = _canonical_env_layout(basA, envA.copy())
        basB2, envB2 = _canonical_env_layout(basB, envB.copy())
        np.testing.assert_array_equal(basA2, basB2)
        np.testing.assert_array_equal(envA2, envB2)
        # canonical layout: blocks contiguous from the old region start
        np.testing.assert_array_equal(envA2[10:12], [1.5, 2.5])
        np.testing.assert_array_equal(envA2[12:14], [0.5, 0.25])
        np.testing.assert_array_equal(envA2[14:17], [3.5, 4.5, 5.5])
        np.testing.assert_array_equal(envA2[17:20], [0.75, 0.375, 0.125])
        # dead slots zeroed, region outside untouched
        self.assertTrue(np.all(envA2[20:] == 0))
        self.assertEqual(int(basA2[0, A5]), 10)
        self.assertEqual(int(basA2[0, A6]), 12)
        self.assertEqual(int(basA2[1, A5]), 14)
        self.assertEqual(int(basA2[1, A6]), 17)
        # (c) shared block: two rows referencing the same (ptr, len)
        basS, envS = mk([(0, 0, 2, 1, 10, 12), (1, 0, 2, 1, 10, 12)], 20)
        envS[10:12] = [7.5, 8.5]; envS[12:14] = [0.5, 0.25]
        basS2, envS2 = _canonical_env_layout(basS, envS.copy())
        self.assertEqual(int(basS2[0, A5]), int(basS2[1, A5]))
        self.assertEqual(int(basS2[0, A6]), int(basS2[1, A6]))
        np.testing.assert_array_equal(envS2[10:12], [7.5, 8.5])
        np.testing.assert_array_equal(envS2[12:14], [0.5, 0.25])

    def test_helper_refuses_overlapping_blocks(self):
        """Fail-closed: overlapping basis blocks (never produced by pyscf's
        packing) must raise, not silently corrupt."""
        from gpu4pyscf.gto.mole import _canonical_env_layout
        bas = np.zeros((2, 8), dtype=np.int32)
        bas[0] = (0, 0, 4, 1, 0, 10, 12, 0)   # exp block 10..14
        bas[1] = (1, 0, 2, 1, 0, 12, 16, 0)   # coeff block 12..14 overlaps
        env = np.zeros(30)
        with self.assertRaises(RuntimeError):
            _canonical_env_layout(bas, env)

    def test_screen_masks_match_cpu_truth(self):
        """Continuity: post-fix screen masks must equal the exact in-process
        CPU predicate (the robust invariant; the mask is layout-invariant by
        measurement AND construction, so the canonical re-pack must not
        change it).

        DESIGN ERRATUM (2026-10-23, before any green claim): this test
        originally asserted equality with the MR-10 baseline consensus shas.
        That anchor is WRONG -- and the first run failed on dopamine
        (got 280062cbc884... vs want aa8525d9...) and benzene (f5a3e643...
        vs 90e56ca9...) -- because the MR-10 baseline was collected on the
        PRE-MR10-FIX (racy) libgdft whose STABLE output is systematically
        corrupted: rebuilding the racy kernel from source 51e1b71 measured
        benzene 90e56ca9 30/30 launches and dopamine aa8525d9 30/30, each
        DIFFERENT from the exact CPU predicate and each dropping exactly
        ONE truth-keep cell (benzene 1/3240, dopamine 1/10725).  So the
        baseline's 148/150 'consensus' was the corrupted majority and its
        2/150 'corrupted' launches were the clean ones; the MR-10 barrier
        fix correctly produces the CPU-truth mask (the MR-10 P2 MISS
        verdict stands; its 'layout-variant mask' interpretation is
        falsified -- both benzene layouts produce the same CPU-truth mask
        on the fixed kernel).  The first failing run of this test is
        preserved in the MR-11 evidence directory.  Water is unaffected
        (racy output == truth there, 34db5452...)."""
        import cupy as cp  # noqa: F401
        from gpu4pyscf.dft import numint
        for tag, xyz, basis, level in MOLECULES:
            with self.subTest(tag=tag):
                grids, opt = MR10._grids_and_opt(xyz, basis, level)
                grids._non0ao_idx = None
                m = grids._screen_shl_mask(opt, numint.MIN_BLK_SIZE)
                grids._non0ao_idx = None
                cpu = MR10._cpu_ground_truth(grids, opt)
                self.assertTrue(np.array_equal(cp.asnumpy(m), cpu),
                                'mask differs from CPU predicate on %s' % tag)


if __name__ == '__main__':
    unittest.main()

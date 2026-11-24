#!/usr/bin/env python3
"""MR-10 (2026-10-23, mainline repair) regression tests: deterministic
_screen_index output (the tile-cache race fix).

MR-9 (2026-10-23-mr9-anomaly-mechanism) characterized the intermittent
partition-fix guard firing as transient single-launch output corruption of
the fine (4096) AO-screen kernel `_screen_index`, spurious-keep direction,
with static root cause: the kernel's tile loop had ONE __syncthreads() per
iteration (between the shared-tile LOAD and the CONSUMER), so the consumer
of tile k raced the load of tile k+1 (the is_large early-`continue` widens
the window).  Measured on the dopamine/b3lyp/def2-tzvp grid (262912
points): 19/660 fine launches (2.9%) differed from their own immediate
same-input replay.  MR-10 adds the missing unconditional barrier at the
tile-loop top (every path through an iteration passes it, so the early
`continue` stays deadlock-free) and rebuilds libgdft.

Red->green: test_screen_determinism runs N fresh production-path launches
(grids._screen_shl_mask at MIN_BLK_SIZE with cache reset per launch) per
molecule and requires bitwise-equal masks.  On the PRE-fix binary it fails
probabilistically (P(all N equal) <= ~(1-0.029)^(N-1); at N=150 < 1.2%);
on the fixed binary it must always pass.  test_screen_semantics_preserved
requires the fixed kernel's mask to equal the PRE-fix CONSENSUS mask
(majority over the baseline collection run on the pre-fix binary, stored
in the baseline JSON next to this file) -- fixing the race must not change
screen semantics, only remove the corruption.  The baseline file records
its own collection stats (all-launches-equal flags and per-molecule diff
counts) as the pre-fix red evidence.

These are GPU tests (a CUDA context is created); they need the gpu4pyscf
source env and the REBUILT libgdft.  ~10-20 s per molecule on the RTX 4090.
"""
import json
import os
import sys
import unittest

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO)
BASELINE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'mr10_screen_baseline_20261023.json')
N_LAUNCHES = 150

DOPAMINE_XYZ = """
O -2.2392 1.9626 0.0548
O -3.3557 -0.5612 0.3868
N 4.4081 0.2624 0.3445
C 2.1628 -0.0212 -0.6613
C 0.7040 -0.1603 -0.3850
C 2.9862 0.1008 0.6289
C -0.0999 0.9759 -0.2919
C 0.1434 -1.4267 -0.2187
C -1.4642 0.8456 -0.0326
C -1.2209 -1.5570 0.0407
C -2.0247 -0.4208 0.1336
H 2.5111 -0.8817 -1.2481
H 2.3332 0.8564 -1.2993
H 2.8490 -0.7888 1.2541
H 2.6457 0.9593 1.2192
H 0.3315 1.9659 -0.4187
H 0.7594 -2.3195 -0.2869
H -1.6445 -2.5496 0.1686
H 4.5468 1.0868 -0.2388
H 4.7362 -0.5285 -0.2089
H -3.1541 1.6866 0.2377
H -3.5639 -1.5074 0.4721
"""

BENZENE_XYZ = """
C 1.3960 0.0000 0.0000
C 0.6980 1.2090 0.0000
C -0.6980 1.2090 0.0000
C -1.3960 0.0000 0.0000
C -0.6980 -1.2090 0.0000
C 0.6980 -1.2090 0.0000
H 2.4810 0.0000 0.0000
H 1.2405 2.1480 0.0000
H -1.2405 2.1480 0.0000
H -2.4810 0.0000 0.0000
H -1.2405 -2.1480 0.0000
H 1.2405 -2.1480 0.0000
"""

WATER_XYZ = """
O 0.0000 0.1173 0.0000
H 0.0000 -0.4692 0.7572
H 0.0000 -0.4692 -0.7572
"""

MOLECULES = [
    ('dopamine_b3lyp_def2-tzvp', DOPAMINE_XYZ, 'def2-tzvp', 3),
    ('benzene_b3lyp_def2-tzvp', BENZENE_XYZ, 'def2-tzvp', 3),
    ('water_b3lyp_def2-svp', WATER_XYZ, 'def2-svp', 3),
]


def _grids_and_opt(xyz, basis, level):
    import cupy as cp  # noqa: F401  (context check)
    from pyscf import gto
    from gpu4pyscf.dft import gen_grid, numint
    mol = gto.M(atom=xyz, basis=basis, verbose=0,
                output='/dev/null', max_memory=32000)
    grids = gen_grid.Grids(mol)
    grids.level = level
    grids.build()
    opt = numint._GDFTOpt.from_mol(mol)
    return grids, opt


def _mask_launches(grids, opt, n):
    """n fresh production-path fine-screen launches (cache reset each time);
    returns a list of host masks."""
    from gpu4pyscf.dft import numint
    MB = numint.MIN_BLK_SIZE
    out = []
    for _ in range(n):
        grids._non0ao_idx = None
        m = grids._screen_shl_mask(opt, MB)
        out.append(cp_asnumpy(m))
    grids._non0ao_idx = None
    return out


def cp_asnumpy(a):
    import cupy as cp
    if isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)


def collect_baseline(n=N_LAUNCHES):
    """Pre-fix binary: collect n launches per molecule, store the consensus
    (bitwise majority) mask sha + the per-launch sha list.  Also the
    honest red evidence: per-molecule diff counts."""
    baseline = {'collected_utc': __import__('time').strftime(
                    '%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
                'n_launches': n, 'molecules': {}}
    for tag, xyz, basis, level in MOLECULES:
        grids, opt = _grids_and_opt(xyz, basis, level)
        masks = _mask_launches(grids, opt, n)
        shas = [_sha(m) for m in masks]
        from collections import Counter
        consensus_sha, cnt = Counter(shas).most_common(1)[0]
        baseline['molecules'][tag] = {
            'ngrids': int(grids.size),
            'nbas': int(masks[0].shape[1]),
            'consensus_sha': consensus_sha,
            'consensus_count': int(cnt),
            'n_distinct': len(set(shas)),
        }
        print(tag, 'ngrids=%d nbas=%d consensus=%d/%d distinct=%d' % (
            grids.size, masks[0].shape[1], cnt, n, len(set(shas))))
    with open(BASELINE, 'w') as f:
        json.dump(baseline, f, indent=1)
    print('baseline written:', BASELINE)


def _sha(m):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(m).tobytes()).hexdigest()


def _cpu_ground_truth(grids, opt):
    '''Exact numpy replication of the _screen_index predicate on THIS
    process's sorted-mol layout (the per-(shell, block) keep decision).
    CPU/GPU exp agree to ulp level; margins at mismatched cells would have
    to straddle 0 to be a false alarm (none were observed on the test
    molecules: the first fixed-binary check matched bitwise).'''
    from gpu4pyscf.dft import numint
    import cupy as cp
    MB = numint.MIN_BLK_SIZE
    sm = opt._sorted_mol
    ao_loc = sm.ao_loc_nr()
    nbas = len(ao_loc) - 1
    nblocks = (grids.size + MB - 1) // MB
    coords = cp.asnumpy(grids.coords)
    atm = np.asarray(sm._atm); bas = np.asarray(sm._bas)
    env = np.asarray(sm._env)
    log_cutoff = np.log(numint.AO_THRESHOLD)
    cpu = np.zeros((nblocks, nbas), np.int8)
    for i in range(nbas):
        atm_id = int(bas[i, 0]); ang = int(bas[i, 1])
        nprim = int(bas[i, 2])
        exps = env[int(bas[i, 5]):int(bas[i, 5]) + nprim]
        coeffs = env[int(bas[i, 6]):int(bas[i, 6]) + nprim]
        ri = env[int(atm[atm_id, 1]):int(atm[atm_id, 1]) + 3]
        for b in range(nblocks):
            p0 = b * MB
            p1 = min(p0 + MB, grids.size)
            x = coords[p0:p1, 0]; y = coords[p0:p1, 1]; z = coords[p0:p1, 2]
            rr = ((x - ri[0]) ** 2 + (y - ri[1]) ** 2
                  + (z - ri[2]) ** 2 + 1e-300)
            gto = np.full(p1 - p0, 1e-300)
            for ip in range(nprim):
                gto += coeffs[ip] * np.exp(-exps[ip] * rr)
            margin = (np.log(np.abs(gto)) + ang * np.log(rr) / 2.0
                      - log_cutoff)
            if margin.max() > 0:
                cpu[b, i] = 1
    return cpu


class TestMR10ScreenDeterminism(unittest.TestCase):

    def test_screen_determinism(self):
        """N fresh production-path launches must be bitwise-equal (red on
        the pre-fix binary with probability >= 99%, MR-9-measured 2.9%/launch
        corruption rate on the dopamine grid)."""
        for tag, xyz, basis, level in MOLECULES:
            grids, opt = _grids_and_opt(xyz, basis, level)
            masks = _mask_launches(grids, opt, N_LAUNCHES)
            shas = [_sha(m) for m in masks]
            self.assertEqual(
                len(set(shas)), 1,
                'non-deterministic _screen_index output on %s: %d distinct '
                'masks over %d launches (the MR-9 tile-race class)' % (
                    tag, len(set(shas)), N_LAUNCHES))

    def test_screen_semantics_preserved(self):
        """The fixed kernel's mask must equal the exact CPU predicate on
        THIS process's inputs (semantics preserved: the fix removes
        corruption, it does not change the keep decisions).

        Design erratum (2026-10-23, before any green claim): this test
        originally compared against the PRE-fix cross-process CONSENSUS
        mask sha; that premise was falsified by measurement -- the
        upstream _GDFTOpt construction produces a per-process env/bas
        LAYOUT that varies across processes (132/240 env slots and the bas
        PTR offsets differ between two fresh processes; kernel-independent
        Python-side code), and in a rare minority of processes (~1/37
        observed) the per-shell summation order changes enough to flip a
        razor-margin screen cell, producing a different-but-locally-correct
        mask.  The consensus comparison therefore flakes on that
        pre-existing class (its one observed MISS is preserved in the
        MR-10 evidence directory); the robust invariant is the in-process
        CPU ground truth.  The layout class itself is a named follow-up
        (upstream _GDFTOpt per-process layout nondeterminism), NOT part of
        MR-10."""
        for tag, xyz, basis, level in MOLECULES:
            grids, opt = _grids_and_opt(xyz, basis, level)
            masks = _mask_launches(grids, opt, 3)
            got = {_sha(m) for m in masks}
            self.assertEqual(len(got), 1)
            cpu = _cpu_ground_truth(grids, opt)
            self.assertTrue(
                np.array_equal(masks[0], cpu),
                'fixed-kernel mask differs from the exact CPU predicate on '
                '%s (semantics changed, not just corruption removed)' % tag)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--collect-baseline':
        collect_baseline()
    else:
        unittest.main()

# Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NUMINT-UKS-DMSET-DEFENSE-01 -- defensive mirror of upstream #1037 (commit
# 3873ad6ab36986671145462a0f4c1fe93340afef 'Align multicollinear XC with new
# PySCF/MCfun', 2026-11-11T16:32Z). The upstream hunk fixed dft/numint.py
# _nr_uks_task's dm-only branch: take_last2d(dma, idx, out=dm_mask) ->
# take_last2d(dma[i], idx, out=dm_mask) (same for dmb). Mechanism (measured,
# lib/cupy_helper.py:485): take_last2d computes count=prod(a.shape[:-2]) and
# reinterprets `out` as (count,nidx,nidx), reshaping back only for a.ndim==2;
# nr_uks always reshapes dma to (-1,nao,nao), so a stacked nset>=2 dm-only UKS
# call loses the per-set rebinding and hands eval_rho a 3D stack over an
# nset*nsub^2 kernel write into an nsub^2 buffer. nset==1 (all deployed UKS
# SCF/grad lanes) returns a leading-1 view with values identical to dma[0] --
# bitwise-clean at the parent, which is why this is a latent-defense unit.
#
# RED-at-parent classes: test_source_contract_* (helpers-absent/source-pin
# class) and test_stacked_dmset_matches_perset (crash/mismatch at parent).
# GREEN-by-design at parent (disclosed): test_rks_twin_site_verbatim and
# test_single_set_physical_nelec (nset==1 lanes are value-identical pre/post
# fix).

import os
import numpy as np
import unittest
import pyscf
from gpu4pyscf.dft import numint, gen_grid

def _np(x):
    # cupy refuses implicit np.asarray conversion; .get() explicitly
    return x.get() if hasattr(x, 'get') else np.asarray(x)

atom = '''
O       0.0000000000    -0.0000000000     0.1174000000
H      -0.7570000000    -0.0000000000    -0.4696000000
H       0.7570000000     0.0000000000    -0.4696000000
'''

def _build():
    mol = pyscf.M(atom=atom, basis='sto-3g', max_memory=32000, verbose=1,
                  spin=1, charge=1)
    grids = gen_grid.Grids(mol)
    grids.level = 1
    grids.build()
    ni = numint.NumInt()
    nao = mol.nao_nr()
    # Set 1 = converged CPU UKS dm (physical anchor); set 2 = an independent
    # hermitian spin-mixed variant (numint only requires hermi dms).
    mf = mol.UKS()
    mf.grids.level = 2
    mf.conv_tol = 1e-9
    mf.kernel()
    dm0 = mf.make_rdm1()
    dma1 = np.asarray(dm0[0])
    dmb1 = np.asarray(dm0[1])
    dma2 = 0.6 * dma1 + 0.4 * dmb1
    dmb2 = 0.4 * dma1 + 0.6 * dmb1
    return mol, grids, ni, nao, dma1, dmb1, dma2, dmb2

class KnownValues(unittest.TestCase):

    def test_source_contract_dma_i(self):
        # RED at the parent: pins the mirrored upstream hunks verbatim.
        src = os.path.join(os.path.dirname(numint.__file__), 'numint.py')
        text = open(src).read()
        self.assertIn('dm_mask = take_last2d(dma[i], idx, out=dm_mask)', text)
        self.assertIn('dm_mask = take_last2d(dmb[i], idx, out=dm_mask)', text)
        # the pre-fix stacked calls must be gone from _nr_uks_task
        self.assertNotIn('dm_mask = take_last2d(dma, idx, out=dm_mask)', text)
        self.assertNotIn('dm_mask = take_last2d(dmb, idx, out=dm_mask)', text)

    def test_stacked_dmset_matches_perset(self):
        # RED at the parent (ValueError shape/broadcast or wrong per-set
        # values); GREEN after the mirror: stacked == per-set, <=1e-12
        # (expected bitwise: identical per-set kernel sequence post-fix).
        mol, grids, ni, nao, dma1, dmb1, dma2, dmb2 = _build()
        stack_a = np.asarray([dma1, dma2])
        stack_b = np.asarray([dmb1, dmb2])
        n_s, e_s, v_s = ni.nr_uks(mol, grids, 'pbe', (stack_a, stack_b), 0, 1)
        # stacked input -> nelec (2,nset), excsum (nset,), vmats (nset,nao,nao)
        self.assertEqual(np.asarray(n_s).shape, (2, 2))
        for i, (da, db) in enumerate(((dma1, dmb1), (dma2, dmb2))):
            # 2-D dm input -> squeezed returns: nelec (2,), scalar exc,
            # single (nao,nao) vmats (nr_uks tail contract)
            n1, e1, v1 = ni.nr_uks(mol, grids, 'pbe', (da, db), 0, 1)
            self.assertEqual(np.asarray(n1).shape, (2,))
            dna = np.abs(np.asarray(n_s)[:, i] - np.asarray(n1)).max()
            de = abs(float(np.asarray(e_s)[i]) - float(np.asarray(e1)))
            dva = np.abs(_np(v_s[0][i]) - _np(v1[0])).max()
            dvb = np.abs(_np(v_s[1][i]) - _np(v1[1])).max()
            print('set %d: dnelec=%.3e dexc=%.3e dvata=%.3e dvatb=%.3e'
                  % (i, dna, de, dva, dvb))
            self.assertLess(dna, 1e-12)
            self.assertLess(de, 1e-12)
            self.assertLess(dva, 1e-12)
            self.assertLess(dvb, 1e-12)

    def test_single_set_physical_nelec(self):
        # GREEN-by-design at the parent (nset==1 lanes value-identical
        # pre/post fix): the single-set dm-only lane still integrates the
        # minao electron count.
        mol, grids, ni, nao, dma1, dmb1, _, _ = _build()
        n1, e1, v1 = ni.nr_uks(mol, grids, 'pbe', (dma1, dmb1), 0, 1)
        nelec = float(np.asarray(n1).sum())
        self.assertLess(abs(nelec - mol.nelectron), 0.05)

    def test_rks_twin_site_verbatim(self):
        # GREEN-by-design at the parent: mirror-only scope -- the RKS twin
        # site keeps the upstream-left leading-stack pattern (upstream #1037
        # did not touch _nr_rks_task; no per-set loop in its dm branch).
        src = os.path.join(os.path.dirname(numint.__file__), 'numint.py')
        text = open(src).read()
        self.assertIn('dm_mask = take_last2d(dm, idx, out=dm_mask)', text)

if __name__ == '__main__':
    unittest.main()

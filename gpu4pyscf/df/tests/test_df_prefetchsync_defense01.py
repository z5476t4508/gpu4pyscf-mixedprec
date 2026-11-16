# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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
'''DF-PREFETCH-SYNC-DEFENSE-01 focused regression (2026-11-11).

Defensive mirror of upstream gpu4pyscf #1036 (fix #1031, commit
d72206c1, also cherry-picked to the 1.9.2 release branch as 0e7c3606)
in ``DF.loop``'s host-resident branch (gpu4pyscf/df/df.py): the FIRST
``buf_prefetch.set(cderi_sparse[:blksize], stream=io_stream)`` must be
preceded by ``compute_event.record(comput_stream)`` +
``io_stream.wait_event(compute_event)``.  Without it, ``buf_prefetch``
(``cp.empty_like`` pool memory) may reuse a pool block that pending
compute-stream work still accesses, and the non-blocking io_stream
H2D copy races it (upstream issue #1031: corrupted DF data,
timing-dependent).

Adjudication facts (UPSTREAM-RANGE-AUDIT-02, 2026-11-11, evidence
/data/results/gpu4pyscf-agent/tasks/issue-9/2026-11-11-upstreamrangeaudit02/):

* bug-at-fork YES: df.py:322 first prefetch had no cross-stream
  dependency at parent 8bce2d5; the steady-state loop at df.py:334
  already carries ``io_stream.wait_event(compute_event)`` (inherited
  from the common ancestor; upstream's fix touches only the first
  set, matching this shape).
* corpus reach NONE on the deployed VRAM-admitted corpus: the racing
  branch fires only when cderi is host-resident (loop's
  ``on_gpu = isinstance(cderi_sparse, cp.ndarray)`` else-branch);
  host-residency arises from ``use_gpu_memory=False`` or the
  constrained-VRAM automatic fallback (df.py:737-741) -- the
  operator-gated later-phase host-RAM/out-of-core axis (issue #11).
  The eager-plan warmup caller (df_jk.py:502-506) explicitly refuses
  host-resident cderi, and the fp32-cderi lane requires on_gpu
  (df.py:751).
* mirror = the upstream 2 statements + comment, verbatim; deployed
  on_gpu branch untouched.

Focused RED at parent 8bce2d5: 1 honest failure (the source contract
below).  The behavioral identity/determinism tests are
by-design-green at the parent (the race is timing-dependent and does
not fire on an idle small cell; they pin the mirror's value-identity
and guard against regressions), disclosed here per the house pattern.
'''
import os
import unittest
import numpy
import cupy as cp
from pyscf import gto
from gpu4pyscf.df.df import DF

DF_PY = os.path.join(os.path.dirname(__file__), '..', 'df.py')

def _loop_source():
    with open(DF_PY) as f:
        return f.read()

def _first_prefetch_window(src):
    '''(start, end) byte window of the host-resident branch: from the
    io_stream/io_event declarations through the FIRST buf_prefetch.set.'''
    decl = src.index('io_event = cp.cuda.Event()')
    # the call form only: the upstream #1031 comment itself contains
    # the literal 'buf_prefetch.set(' and must not end the window
    first_set = src.index('buf_prefetch.set(cderi_sparse', decl)
    return src[decl:first_set]

def _host_probe_dfobj(verbose=0):
    # water/def2-svp (the entry-518 rsomega probe cell): naux ~ 100+
    # so a naux//4 blksize exercises several prefetch blocks.
    mol = gto.M(atom='O 0 0 0; H 0 0 0.9572; H 0.9264 0 -0.24',
                basis='def2-svp', verbose=verbose)
    dfobj = DF(mol, 'def2-SVP-JKfit').build()
    return dfobj

def _collect_blocks(dfobj, blksize, unpack=False):
    blocks = []
    for _, blk in dfobj.loop(blksize=blksize, unpack=unpack):
        blocks.append(cp.asnumpy(blk).copy())
    return blocks

class TestPrefetchSyncDefense(unittest.TestCase):

    def test_source_contract_first_prefetch_sync(self):
        # RED at parent 8bce2d5: the upstream #1036 statements must sit
        # between the io_event declaration and the FIRST buf_prefetch.set.
        window = _first_prefetch_window(_loop_source())
        self.assertIn('compute_event.record(comput_stream)', window)
        self.assertIn('io_stream.wait_event(compute_event)', window)

    def test_source_contract_steady_state_sync(self):
        # By-design green at parent: the steady-state prefetch (the
        # in-loop buf_prefetch[:...]set) keeps its wait dependency
        # immediately before it.
        src = _loop_source()
        steady_set = src.index('buf_prefetch[:')
        preceding = src.rindex('io_stream.wait_event(compute_event)', 0, steady_set)
        self.assertLess(steady_set - preceding, 200)

    def test_source_contract_on_gpu_branch_untouched(self):
        # By-design green at parent: the deployed on_gpu branch (cderi
        # resident on the GPU) carries no io_stream machinery at all;
        # the mirror must not reach it.
        src = _loop_source()
        on_gpu_branch = src[src.index('if on_gpu:'):src.index('else:', src.index('if on_gpu:'))]
        self.assertNotIn('io_stream', on_gpu_branch)

    def test_source_contract_warmup_refuses_host_resident(self):
        # By-design green at parent: the df_jk eager-warmup loop() caller
        # self-refuses host-resident cderi (reach analysis pinned).
        df_jk_path = os.path.join(os.path.dirname(__file__), '..', 'df_jk.py')
        with open(df_jk_path) as f:
            src = f.read()
        warm = src[src.index('eager plan warmup skipped')-600:
                   src.index('eager plan warmup skipped')]
        self.assertIn('isinstance(cderi_sparse, cp.ndarray)', warm)

    def test_host_resident_lane_value_identity(self):
        # Mirror equivalence probe: host-resident loop() blocks are
        # bitwise-identical to the on_gpu branch's blocks (the mirror
        # adds ordering only, never data transformation). By-design
        # green at the parent (race is timing-dependent, does not fire
        # on this idle small cell); pins value-identity under the fix.
        dfobj = _host_probe_dfobj()
        dev = cp.cuda.Device().id
        gpu_cderi = dfobj._cderi[dev]
        self.assertIsInstance(gpu_cderi, cp.ndarray)
        naux = gpu_cderi.shape[0]
        blksize = max(1, naux // 4)  # >= 3 blocks: exercise the
        # steady-state prefetch too, not only the first block
        ref = _collect_blocks(dfobj, blksize=blksize)
        self.assertGreaterEqual(len(ref), 3)
        try:
            dfobj._cderi[dev] = cp.asnumpy(gpu_cderi)
            host1 = _collect_blocks(dfobj, blksize=blksize)
            host2 = _collect_blocks(dfobj, blksize=blksize)
        finally:
            dfobj._cderi[dev] = gpu_cderi  # restore the resident build
        self.assertEqual(len(ref), len(host1))
        for a, b in zip(ref, host1):
            self.assertTrue(numpy.array_equal(a, b))

    def test_host_resident_lane_repeat_determinism(self):
        # Repeat-iteration determinism of the host-resident lane: two
        # consecutive loop() sweeps must be bitwise identical (a firing
        # pool-reuse race corrupts blocks non-deterministically).
        # By-design green at the parent; disclosed.
        dfobj = _host_probe_dfobj()
        dev = cp.cuda.Device().id
        gpu_cderi = dfobj._cderi[dev]
        blksize = max(1, gpu_cderi.shape[0] // 4)
        try:
            dfobj._cderi[dev] = cp.asnumpy(gpu_cderi)
            host1 = _collect_blocks(dfobj, blksize=blksize)
            host2 = _collect_blocks(dfobj, blksize=blksize)
        finally:
            dfobj._cderi[dev] = gpu_cderi
        for a, b in zip(host1, host2):
            self.assertTrue(numpy.array_equal(a, b))

if __name__ == '__main__':
    unittest.main()

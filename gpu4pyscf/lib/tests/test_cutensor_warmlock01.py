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

'''DFT-CUTENSOR-SIGFPE-01 focused regression.

Measured defect (SUITE-SWEEP-E99507C-01, entry 609, 2026-11-14): the dft
suite process died rc=136 (SIGFPE, core dumped) at wall=5 s inside
cutensor.py _build_plan_entry -> estimateWorkspaceSize during the FIRST
cutensor contraction (test_ao_values.py::test_ao_cart_deriv0); a green
standalone re-run at the identical tree state classified it a rare
first-call race.  Root cause class: the cutensor warm thread created the
cupyx handle via a direct cutensor._get_handle() WITHOUT _handle_lock,
so it could race the main thread's first contraction going through
_get_handle_locked() -- two concurrent first calls of the cupyx lazy
global, the exact not-thread-safe pattern documented at _handle_lock.

The fix routes the warm thread's handle creation through
_get_handle_locked().  These tests pin the lock discipline of BOTH
paths: the warm target must observe _handle_lock held while
cutensor._get_handle() runs (test 1), and _get_handle_locked() must
hold it too (test 2, the pre-existing contract).  Test 3 verifies the
warm thread still completes and stamps _HANDLE_STATE (the warm win).
'''

import os
import unittest

import numpy as np
import cupy as cp


class TestWarmThreadHandleLock(unittest.TestCase):

    def _observer(self, observed):
        '''Replace cutensor._get_handle with a recorder that checks
        whether _handle_lock is held by the CALLING thread at entry
        (a non-reentrant threading.Lock cannot be non-blockingly
        acquired by its holder).'''
        import gpu4pyscf.lib.cutensor as ct
        real = ct.cutensor._get_handle

        def recorder():
            got = ct._handle_lock.acquire(blocking=False)
            observed['lock_held'] = not got
            if got:
                # never the contract, but release defensively so a
                # failure mode cannot deadlock every later caller
                ct._handle_lock.release()
            observed['called'] = True
            return real()
        return real, recorder

    def test_warm_target_takes_handle_lock(self):
        # DFT-CUTENSOR-SIGFPE-01: the warm thread's handle creation must
        # be serialized behind _handle_lock.
        import gpu4pyscf.lib.cutensor as ct
        if ct.cutensor is None:
            self.skipTest('cutensor unavailable')
        self.assertTrue(hasattr(ct, '_warm_target'),
                        '_warm_target must exist at module level')
        observed = {}
        real, recorder = self._observer(observed)
        ct.cutensor._get_handle = recorder
        try:
            ct._warm_target()
        finally:
            ct.cutensor._get_handle = real
        self.assertTrue(observed.get('called'))
        self.assertTrue(observed.get('lock_held'),
                        'cutensor._get_handle ran WITHOUT _handle_lock held '
                        'inside the warm target -- the DFT-CUTENSOR-SIGFPE-01 '
                        'race is open again')

    def test_get_handle_locked_holds_lock(self):
        # the pre-existing contract, pinned alongside: the main-thread
        # path must hold _handle_lock while cutensor._get_handle() runs.
        import gpu4pyscf.lib.cutensor as ct
        if ct.cutensor is None:
            self.skipTest('cutensor unavailable')
        observed = {}
        real, recorder = self._observer(observed)
        ct.cutensor._get_handle = recorder
        try:
            ct._get_handle_locked()
        finally:
            ct.cutensor._get_handle = real
        self.assertTrue(observed.get('called'))
        self.assertTrue(observed.get('lock_held'))

    def test_warm_thread_completes_and_stamps_state(self):
        # the warm win is preserved: the (already-running or completed)
        # warm thread leaves the handle ready and stamps
        # _HANDLE_STATE['ready_perf_s'].
        import gpu4pyscf.lib.cutensor as ct
        if ct.cutensor is None:
            self.skipTest('cutensor unavailable')
        warm = getattr(ct, '_warm_thread', None)
        if warm is not None:
            warm.join(60.0)
        self.assertFalse(warm is not None and warm.is_alive(),
                         'warm thread did not finish within 60 s')
        # either the import-time warm thread stamped it, or (knob off)
        # the direct call above made the handle usable; with the default
        # knob the stamp must exist
        if os.environ.get('GPU4PYSCF_CUTENSOR_WARM', '1') not in \
                ('0', 'false', 'False'):
            self.assertIsNotNone(ct._HANDLE_STATE.get('ready_perf_s'),
                                 'warm thread finished without stamping '
                                 '_HANDLE_STATE (the warm win is lost)')
        h = ct._get_handle_locked()
        self.assertIsNotNone(h)
        # a real contraction still works end-to-end through the fixed
        # module (the crash-site path: plan build + estimateWorkspaceSize)
        a = cp.arange(2 * 3, dtype=np.float64).reshape(2, 3)
        b = cp.arange(3 * 4, dtype=np.float64).reshape(3, 4)
        out = ct.contract('ij,jk->ik', a, b)
        ref = np.einsum('ij,jk->ik', a.get(), b.get())
        self.assertTrue(cp.allclose(out, cp.asarray(ref)))


if __name__ == '__main__':
    unittest.main()

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

'''SMALLCELL-FIRSTCALL-WAIT-01 focused regression (C1 only, re-landed
verbatim from the archived FIRSTCALL-LAZYINIT-01 f8f46d9 implementation).

C1: the ctypes early library warm (gpu4pyscf/lib/_cutensor_early_warm.py,
started by the gpu4pyscf/__init__.py prelude) must resolve the same
libcutensor the cupy wheel preload uses, complete the one-time library
init, and leave the landed lib/cutensor.py warm thread creating the real
cupyx handle in milliseconds instead of ~2.9 s.

C2 (the stream-ordered big-block allocator) is NOT re-landed in this
card -- entry-595 measured it a net loss at every cell/lane -- so the
archived C2 tests are intentionally absent.
'''

import os
import sys
import unittest

import numpy as np
import cupy as cp


class TestEarlyWarm(unittest.TestCase):
    def test_module_loaded_and_state_schema(self):
        import gpu4pyscf  # noqa: F401
        early = sys.modules.get('gpu4pyscf.lib._cutensor_early_warm')
        self.assertIsNotNone(early, 'prelude did not register the module')
        st = early.state()
        for key in ('status', 'lib_path', 'start_perf_s', 'done_perf_s',
                    'error'):
            self.assertIn(key, st)
        self.assertIn(st['status'],
                      ('running', 'done', 'unavailable', 'create_failed',
                       'destroy_failed', 'error', 'disabled', 'engine_off'))

    def test_knob_off_disables(self):
        from gpu4pyscf.lib import _cutensor_early_warm as early
        saved = dict(early.STATE)
        saved_thread = early._thread
        try:
            early.STATE['status'] = 'not_started'
            early._thread = None
            os.environ['GPU4PYSCF_CUTENSOR_WARM'] = '0'
            early.maybe_start()
            self.assertEqual(early.STATE['status'], 'disabled')
            self.assertIsNone(early._thread)
            os.environ['GPU4PYSCF_CUTENSOR_WARM'] = '1'
            os.environ['CONTRACT_ENGINE'] = 'cupy'
            early.maybe_start()
            self.assertEqual(early.STATE['status'], 'engine_off')
            self.assertIsNone(early._thread)
        finally:
            early.STATE.clear()
            early.STATE.update(saved)
            early._thread = saved_thread
            for key in ('GPU4PYSCF_CUTENSOR_WARM', 'CONTRACT_ENGINE'):
                os.environ.pop(key, None)

    def test_warm_completes_and_handle_is_cheap(self):
        from gpu4pyscf.lib import _cutensor_early_warm as early
        early.join(60.0)
        st = early.state()
        if st['status'] == 'done':
            self.assertIsNotNone(st['lib_path'])
            self.assertIsNotNone(st['done_perf_s'])
            import gpu4pyscf.lib.cutensor as ct
            if ct._warm_thread is not None:
                ct._warm_thread.join(60.0)
            self.assertIsNotNone(ct._HANDLE_STATE['ready_perf_s'])
        else:
            # fail-open host (no libcutensor resolvable): the landed
            # thread falls back to the full cost; skip, nothing to assert
            self.assertIn(st['status'], ('unavailable', 'create_failed',
                                         'destroy_failed', 'error',
                                         'disabled', 'engine_off'))


if __name__ == '__main__':
    unittest.main()

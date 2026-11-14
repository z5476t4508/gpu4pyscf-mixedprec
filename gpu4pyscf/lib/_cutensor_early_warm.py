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

'''FIRSTCALL-LAZYINIT-01 C1: earliest-possible cuTensor library warm.

Measured on this host (probe, 2026-11-14): the first ``cutensorCreate`` in
a process costs 2.88-3.14 s of host time, but a SECOND ``cutensorCreate``
in the same process costs 0.011 s -- the cost is almost entirely
LIBRARY-LEVEL one-time initialization (kernel-image module loads, symbol
resolution, internal machinery), not per-handle work.  The landed warm
thread in gpu4pyscf/lib/cutensor.py cannot start earlier than ~1.45 s into
``import gpu4pyscf`` because it needs the ~1.4 s cupy import chain, so the
handle (ready at ~4.1 s) lands after the first cutensor contraction of the
first SCF cycle at the production cells and the main thread blocks on the
handle lock (measured 0.5-1.35 s inside the FIRST get_veff).

This module is loaded by a stdlib-only file-path prelude at the very top
of gpu4pyscf/__init__.py, BEFORE the pyscf/cupy import chain, and starts a
ctypes daemon thread that runs one THROWAWAY cutensorCreate +
cutensorDestroy.  ctypes foreign calls release the GIL, so the ~2.9 s
library init truly overlaps the python import work instead of racing the
SCF.  The landed warm thread then joins this thread and creates the real
cupyx handle in ~11 ms, before the first contraction needs it.

The library is resolved to the SAME file the cupy wheel preload uses
(soname first, then the cutensor-cuXX wheel's cutensor/lib directory), so
the ctypes dlopen and cupy's preload share one library instance and this
warm applies to the handle cupy will create.  On hosts where the soname
resolves to a different copy than the wheel, the warm is simply wasted
background work and the landed thread falls back to the full cost --
never a correctness issue.

Fail-open by design: any resolution or creation failure leaves the landed
warm thread paying the full cost exactly as before.  Respects the landed
opt-out GPU4PYSCF_CUTENSOR_WARM=0 and skips when CONTRACT_ENGINE selects
a non-cuTensor contraction engine (no handle will ever be requested).

This module must stay stdlib-only at import time (ctypes/os/threading/
time; importlib.metadata is imported lazily inside _resolve_library) --
it is loaded before numpy/cupy exist.
'''

import ctypes
import os
import threading
import time

STATE = {
    'status': 'not_started',   # not_started|running|done|unavailable|
                               # create_failed|destroy_failed|error|
                               # disabled|engine_off
    'lib_path': None,
    'start_perf_s': None,
    'done_perf_s': None,
    'error': None,
}

_thread = None

_LIB_SONAMES = ('libcutensor.so.2', 'libcutensor.so.3', 'libcutensor.so')
_WHEEL_DIST_NAMES = ('cutensor-cu12', 'cutensor-cu13', 'cutensor-cu11',
                     'cutensor-cu10')


def _try_cdll(path):
    try:
        return ctypes.CDLL(path)
    except OSError:
        return None


def _resolve_library():
    '''Return (path, CDLL) for the libcutensor cupy will use, or
    (None, None).  Never raises.'''
    # (1) the process loader path (LD_LIBRARY_PATH-configured hosts)
    for soname in _LIB_SONAMES:
        lib = _try_cdll(soname)
        if lib is not None:
            return soname, lib
    # (2) the cutensor-cuXX wheel file -- the same file cupy's
    # _environment preload resolves and dlopens
    try:
        import importlib.metadata as md
        for dist_name in _WHEEL_DIST_NAMES:
            try:
                dist = md.distribution(dist_name)
            except Exception:
                continue
            for soname in _LIB_SONAMES:
                path = dist.locate_file('cutensor/lib/' + soname)
                if path is not None and os.path.exists(str(path)):
                    lib = _try_cdll(str(path))
                    if lib is not None:
                        return str(path), lib
    except Exception:
        pass
    # (3) the cutensor namespace-package fallback
    try:
        import glob
        import importlib.util
        spec = importlib.util.find_spec('cutensor')
        for base in (getattr(spec, 'submodule_search_locations', None) or ()):
            for path in sorted(
                    glob.glob(os.path.join(str(base), 'lib',
                                           'libcutensor.so.*')),
                    reverse=True):
                lib = _try_cdll(path)
                if lib is not None:
                    return path, lib
    except Exception:
        pass
    return None, None


def _warm():
    STATE['status'] = 'running'
    try:
        path, lib = _resolve_library()
        if lib is None:
            STATE['status'] = 'unavailable'
            return
        STATE['lib_path'] = path
        handle = ctypes.c_void_p()
        rc = lib.cutensorCreate(ctypes.byref(handle))
        if rc != 0:
            STATE['status'] = 'create_failed'
            STATE['error'] = 'cutensorCreate rc=%s' % (rc,)
            return
        rc = lib.cutensorDestroy(handle)
        if rc != 0:
            STATE['status'] = 'destroy_failed'
            STATE['error'] = 'cutensorDestroy rc=%s' % (rc,)
            return
        STATE['status'] = 'done'
    except Exception as exc:  # noqa: BLE001 - fail-open by design
        STATE['status'] = 'error'
        STATE['error'] = repr(exc)[:300]
    finally:
        STATE['done_perf_s'] = time.perf_counter()


def maybe_start():
    '''Start the early library warm thread unless the landed opt-out or a
    non-cuTensor contraction engine makes it pointless.  Called once from
    the gpu4pyscf/__init__.py prelude, before any heavy import.'''
    global _thread
    if os.environ.get('GPU4PYSCF_CUTENSOR_WARM', '1') in ('0', 'false',
                                                          'False'):
        STATE['status'] = 'disabled'
        return
    if os.environ.get('CONTRACT_ENGINE', ''):
        # lib/cutensor.py skips its own warm thread on the same condition:
        # a non-default contraction engine means no cuTensor handle is
        # ever requested, so the library init would be wasted work.
        STATE['status'] = 'engine_off'
        return
    STATE['start_perf_s'] = time.perf_counter()
    _thread = threading.Thread(
        target=_warm, name='gpu4pyscf-cutensor-early-warm', daemon=True)
    _thread.start()


def join(timeout=30.0):
    '''Join the early warm thread (best effort).  Called from the LC-60
    atexit ordering in gpu4pyscf/lib/cutensor.py, which registers AFTER
    cupy's own teardown handlers and therefore runs BEFORE them (atexit
    is LIFO).  This module deliberately registers no atexit handler of
    its own: it is loaded before cupy, so its handler would run after
    cupy's teardown -- exactly the frozen-mid-create SIGSEGV class the
    LC-60 join was added to prevent.'''
    if _thread is not None:
        _thread.join(timeout)


def state():
    '''A json-safe copy of the module state (evidence instrumentation).'''
    return dict(STATE)

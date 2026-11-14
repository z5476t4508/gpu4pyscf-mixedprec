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

import os
import sys
import threading
import time

import numpy as np
import cupy
from gpu4pyscf.lib import logger

try:
    import cupy_backends.cuda.libs.cutensor  # NOQA
    from cupyx import cutensor
    from cupy_backends.cuda.libs import cutensor as cutensor_backend
    ALGO_DEFAULT = cutensor_backend.ALGO_DEFAULT
    OP_IDENTITY = cutensor_backend.OP_IDENTITY
    JIT_MODE_NONE = cutensor_backend.JIT_MODE_NONE
    WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_MIN
    #WORKSPACE_RECOMMENDED = cutensor_backend.WORKSPACE_RECOMMENDED
    _tensor_descriptors = {}
except (ImportError, AttributeError):
    cutensor = None
    ALGO_DEFAULT = None
    OP_IDENTITY = None
    JIT_MODE_NONE = None
    WORKSPACE_RECOMMENDED = None

# cuTensorHandleCreate costs ~3 s of host-side time on the first call
# (measured RTX 4090, cuTensor 2.x; constant across processes).  With the
# lazy handle this lands inside the first SCF cycle (first _tau_dot for
# MGGA, first DF get_jk for hybrids), i.e. 44-73% of measured whole-SCF
# time at consumer sizes.  cupyx._get_handle()'s lazy global is not
# thread-safe (two racing first calls return different handles), so all
# gpu4pyscf access is serialized through this lock and one daemon thread
# (started at the end of this module, once the engine choice is known)
# warms the handle behind the ~1.8 s gpu4pyscf import chain.  Opt out
# with GPU4PYSCF_CUTENSOR_WARM=0.
_handle_lock = threading.Lock()

# FIRSTCALL-LAZYINIT-01 C1 evidence instrumentation: the perf_counter
# stamp of the moment the cupyx handle became usable in this process
# (set by the warm thread; None until then).
_HANDLE_STATE = {'ready_perf_s': None}

def _get_handle_locked():
    if cutensor is None:
        return None
    with _handle_lock:
        return cutensor._get_handle()

# Contraction-plan reuse cache (DFJK-PHASE-01 C1).  Every contraction()
# call otherwise rebuilds three tensor descriptors, an operation
# descriptor, a plan preference, a workspace-size estimate and finally
# the plan itself -- all pure host-side work that lands in the SCF's
# unattributed-host bucket (~530 contraction calls per SCF at the frozen
# b3lyp cells pace the jk-phase kernel stream; entry-323 measurement).
# The cache maps a full (pattern, dtypes, shapes, strides, ops, algo,
# ws_pref, device) key to an entry holding every cutensor object alive
# (descriptors, modes, operation descriptor, plan preference, plan) plus
# a reusable workspace buffer.  A cache hit executes the SAME plan object
# -- same kernel, same reduction order -- with alpha/beta still passed
# per call, so results are bitwise identical to the uncached path.
# Opt-in (GPU4PYSCF_CONTRACT_PLAN_CACHE=1), bounded, fail-closed to the
# per-call build past the cap or on any error in the cached path.
_plan_cache = {}
_plan_cache_lock = threading.Lock()
_PLAN_CACHE_MAX_ENTRIES = 4096

# WARMUP-PRELOAD-01 M1: opt-in eager first-use population of the plan
# cache at the caller's real first-call shapes (default off).  While
# enabled it implies the plan cache is active -- warmed entries are
# only ever served through the cached lookup path.
_warmup_stats = {'built': 0, 'reused': 0, 'failures': 0, 'host_ms': 0.0,
                 'patterns': {}}

def _reset_warmup_stats():
    '''Re-initialize the warmup stats to the canonical schema (a bare
    dict.clear() would drop the keys the callers read).'''
    _warmup_stats.update(built=0, reused=0, failures=0, host_ms=0.0)
    _warmup_stats['patterns'] = {}

def _eager_warmup_scope():
    '''WARMUP-PRELOAD-02: parse the knob value into a warm scope name.
    'jbuild' -> warm ONLY the j_build family ('np,Lp->nL' keys; the
    entry-332 per-family cold-cost table's single expensive family,
    629/653 ms vs 1-4 ms for the other three); any other enabled value
    -> the WARMUP-PRELOAD-01 four-family behavior; default off.'''
    value = os.environ.get('GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP', '')
    if value in ('', '0', 'false', 'False'):
        return 'off'
    if value == 'jbuild':
        return 'jbuild'
    return 'all'

def _eager_warmup_enabled():
    return _eager_warmup_scope() != 'off'

def _plan_cache_enabled():
    if _eager_warmup_enabled():
        return True
    return os.environ.get('GPU4PYSCF_CONTRACT_PLAN_CACHE', '') not in \
        ('', '0', 'false', 'False')

def _normalize_zero_strides(a):
    '''Return a view with placeholder strides for cutensor when the
    operand has 0-stride (broadcast) axes -- identical to the object
    _create_tensor_descriptor would otherwise build per call.
    '''
    if any(x == 0 for x in a.strides):
        strides = list(a.strides)
        if strides[0] == 0:
            strides[0] = a.nbytes
        for i, x in enumerate(strides[1:]):
            if x == 0:
                strides[i+1] = strides[i]
        return cupy.ndarray(a.shape, a.dtype, a.data, strides)
    return a

def _build_plan_entry(a, b, c, mode_a, mode_b, mode_c,
                      op_a, op_b, op_c, algo, jit_mode, compute_desc,
                      ws_pref):
    '''Build the descriptor/operator/plan stack for one contraction.
    Returns an entry tuple that keeps every cutensor object alive.
    '''
    desc_a = cutensor.create_tensor_descriptor(a)
    desc_b = cutensor.create_tensor_descriptor(b)
    desc_c = cutensor.create_tensor_descriptor(c)
    operator = cutensor.create_contraction(
        desc_a, mode_a, op_a, desc_b, mode_b, op_b, desc_c, mode_c, op_c,
        compute_desc)
    plan_pref = cutensor.create_plan_preference(algo=algo, jit_mode=jit_mode)
    ws_size = cutensor_backend.estimateWorkspaceSize(
        _get_handle_locked().ptr, operator.ptr, plan_pref.ptr, ws_pref)
    plan = cutensor.create_plan(operator, plan_pref, ws_limit=ws_size)
    return (desc_a, desc_b, desc_c, mode_a, mode_b, mode_c,
            operator, plan_pref, plan, ws_size)

def _auto_create_mode(array, mode):
    if not isinstance(mode, cutensor.Mode):
        mode = cutensor.create_mode(*mode)
    if array.ndim != mode.ndim:
        raise ValueError(
            'ndim mismatch: {} != {}'.format(array.ndim, mode.ndim))
    return mode

def _plan_cache_key(pattern, a, b, c, op_a, op_b, op_c, algo, jit_mode,
                    compute_desc, ws_pref):
    '''The single cache-key construction shared by contraction() and
    warm_contraction() -- warmed entries are inserted under exactly the
    keys the next real call computes (identical member order).'''
    return (pattern, a.device.id,
            a.dtype.char, a.shape, a.strides,
            b.dtype.char, b.shape, b.strides,
            c.dtype.char, c.shape, c.strides,
            op_a, op_b, op_c, algo, jit_mode, compute_desc, ws_pref)

def warm_contraction(pattern, a, b, out=None,
                     op_a=OP_IDENTITY, op_b=OP_IDENTITY,
                     op_c=OP_IDENTITY, algo=ALGO_DEFAULT,
                     jit_mode=JIT_MODE_NONE, compute_desc=0,
                     ws_pref=WORKSPACE_RECOMMENDED):
    '''WARMUP-PRELOAD-01 M1: build and cache the plan stack for one
    contraction at the caller's REAL operand shapes/strides/dtypes --
    the same _build_plan_entry host-side path the first cold call
    would take -- WITHOUT executing the contraction.  Operand arrays
    are only described (cutensor descriptors are host metadata built
    from shape/stride/dtype/device; data is never read), so warmup
    pays no device compute.  Mirrors contraction() exactly up to the
    cache insert; a later contraction() with the same operands hits
    the warmed entry and executes the same plan object the per-call
    build would have produced (bitwise, DFJK-PHASE-01 evidence).
    Returns True when a new entry was built.
    '''
    if a.size == 0 or b.size == 0:
        # contraction() takes the einsum fallback here; nothing to warm
        return False
    pattern = pattern.replace(' ', '')
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k: v for k, v in zip(key, val)}
    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    if len(mode_c) != len(set(mode_c)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    if out is None:
        c = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    else:
        c = out
    if a.size == 0 or b.size == 0 or c.size == 0:
        # contraction() raises on zero-sized c; no df_jk shape reaches
        # here -- treat as nothing to warm rather than fail the warmup
        return False

    a = _normalize_zero_strides(a)
    b = _normalize_zero_strides(b)
    c = _normalize_zero_strides(c)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)

    cache_key = _plan_cache_key(pattern, a, b, c, op_a, op_b, op_c,
                                algo, jit_mode, compute_desc, ws_pref)
    with _plan_cache_lock:
        if cache_key in _plan_cache:
            _warmup_stats['reused'] += 1
            return False
        entry = _build_plan_entry(
            a, b, c, mode_a, mode_b, mode_c,
            op_a, op_b, op_c, algo, jit_mode, compute_desc, ws_pref)
        if len(_plan_cache) >= _PLAN_CACHE_MAX_ENTRIES:
            # capped: leave the entry to the per-call build path
            return False
        _plan_cache[cache_key] = entry
    _warmup_stats['built'] += 1
    _warmup_stats['patterns'][pattern] = \
        _warmup_stats['patterns'].get(pattern, 0) + 1
    return True

def _contract_einsum(pattern, a, b, alpha, beta, out=None, einsum=cupy.einsum):
    if out is None:
        out = einsum(pattern, a, b)
        out *= alpha
    elif beta == 0.:
        out[:] = einsum(pattern, a, b)
        out *= alpha
    else:
        out *= beta
        tmp = einsum(pattern, a, b)
        tmp *= alpha
        out += tmp
    return out

def contraction(
    pattern, a, b, alpha, beta,
    out=None,
    op_a=OP_IDENTITY,
    op_b=OP_IDENTITY,
    op_c=OP_IDENTITY,
    algo=ALGO_DEFAULT,
    jit_mode=JIT_MODE_NONE,
    compute_desc=0,
    ws_pref=WORKSPACE_RECOMMENDED
):
    if a.size == 0 or b.size == 0:
        # cutensor does not support the 0-sized operands
        return _contract_einsum(pattern, a, b, alpha, beta, out)

    pattern = pattern.replace(" ", "")
    str_a, rest = pattern.split(',')
    str_b, str_c = rest.split('->')
    key = str_a + str_b
    val = list(a.shape) + list(b.shape)
    shape = {k:v for k, v in zip(key, val)}

    mode_a = list(str_a)
    mode_b = list(str_b)
    mode_c = list(str_c)
    if len(mode_c) != len(set(mode_c)):
        raise ValueError('Output subscripts string includes the same subscript multiple times.')

    dtype = np.result_type(a.dtype, b.dtype)
    a = cupy.asarray(a, dtype=dtype)
    b = cupy.asarray(b, dtype=dtype)
    if out is None:
        out = cupy.empty([shape[k] for k in str_c], order='C', dtype=dtype)
    c = out

    if a.size == 0 or b.size == 0 or c.size == 0:
        raise ValueError(f"cutensor contraction doesn't support zero-sized array (a.shape = {a.shape}, b.shape = {b.shape}, expected c.shape = {c.shape})")

    # normalize broadcast (0-stride) operands exactly as the per-call
    # _create_tensor_descriptor path did, so cached descriptors are built
    # from the same layouts on every call
    a = _normalize_zero_strides(a)
    b = _normalize_zero_strides(b)
    c = _normalize_zero_strides(c)

    mode_a = _auto_create_mode(a, mode_a)
    mode_b = _auto_create_mode(b, mode_b)
    mode_c = _auto_create_mode(c, mode_c)

    if _plan_cache_enabled():
        key = _plan_cache_key(pattern, a, b, c, op_a, op_b, op_c,
                              algo, jit_mode, compute_desc, ws_pref)
        try:
            with _plan_cache_lock:
                entry = _plan_cache.get(key)
                if entry is None:
                    entry = _build_plan_entry(
                        a, b, c, mode_a, mode_b, mode_c,
                        op_a, op_b, op_c, algo, jit_mode, compute_desc,
                        ws_pref)
                    if len(_plan_cache) < _PLAN_CACHE_MAX_ENTRIES:
                        _plan_cache[key] = entry
        except Exception:
            # fail-closed: fall through to the uncached per-call build
            entry = _build_plan_entry(
                a, b, c, mode_a, mode_b, mode_c,
                op_a, op_b, op_c, algo, jit_mode, compute_desc, ws_pref)
        plan, ws_size = entry[8], entry[9]
    else:
        entry = _build_plan_entry(
            a, b, c, mode_a, mode_b, mode_c,
            op_a, op_b, op_c, algo, jit_mode, compute_desc, ws_pref)
        plan, ws_size = entry[8], entry[9]
    # the workspace is scratch overwritten by each execute; per-call
    # allocation keeps device memory bounded and is a cheap pool hit
    # once sizes stabilize (ws_size is fixed per key)
    ws = cupy.empty(ws_size, dtype=np.int8)
    out = c

    alpha = np.asarray(alpha, dtype=dtype)
    beta = np.asarray(beta, dtype=dtype)

    handler = _get_handle_locked()
    cutensor_backend.contract(handler.ptr, plan.ptr,
                             alpha.ctypes.data, a.data.ptr, b.data.ptr,
                             beta.ctypes.data, c.data.ptr, out.data.ptr,
                             ws.data.ptr, ws_size)
    return out

contract_engine = None
if cutensor is None:
    contract_engine = 'cupy'  # default contraction engine
contract_engine = os.environ.get('CONTRACT_ENGINE', contract_engine)

# override the 'contract' function if einsum is customized or cutensor is not found
if contract_engine is not None:
    einsum = None
    if contract_engine == 'opt_einsum':
        import opt_einsum
        einsum = opt_einsum.contract
    elif contract_engine == 'cuquantum':
        from cuquantum import contract as einsum # type: ignore
    elif contract_engine == 'cupy':
        einsum = cupy.einsum
    else:
        raise RuntimeError('unknown tensor contraction engine.')

    import warnings
    warnings.warn(f'using {contract_engine} as the tensor contraction engine.')
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        try:
            return _contract_einsum(pattern, a, b, alpha, beta, out, einsum)
        except cupy.cuda.memory.OutOfMemoryError:
            print('Out of memory error caused by cupy.einsum. '
                  'It is recommended to install cutensor to resolve this.')
            raise
else:
    def contract(pattern, a, b, alpha=1.0, beta=0.0, out=None):
        '''
        a wrapper for general tensor contraction
        pattern has to be a standard einsum notation
        '''
        return contraction(pattern, a, b, alpha, beta, out=out)

    # cutensor is the active engine: warm its handle now (see comment at
    # _handle_lock).  Daemon thread so it never blocks a running program.
    # It MUST be joined at interpreter exit though: CPython does not join
    # daemon threads, so without this the thread can be frozen mid
    # cutensorCreate while cupy's own atexit cleanup (registered earlier,
    # hence running later in LIFO order) tears down the CUDA context --
    # measured as a rare teardown SIGSEGV/SIGABRT (~2-7% of short-lived
    # processes that import the library and exit; 0/60 with the warm
    # thread disabled, 0/50 with bare cupyx handle creation, LC-60).
    # The atexit join runs BEFORE cupy's cleanup (registered later than
    # cupy's own handlers), so the handle is fully created before any
    # teardown handler touches CUDA.
    if os.environ.get('GPU4PYSCF_CUTENSOR_WARM', '1') not in \
            ('0', 'false', 'False'):
        # FIRSTCALL-LAZYINIT-01 C1: the gpu4pyscf/__init__.py prelude
        # started a ctypes-only daemon thread that pays the ~2.9 s
        # LIBRARY-level one-time cutensorCreate cost (module loads, symbol
        # resolution -- measured per-process: first create 2.88-3.14 s,
        # second create 11 ms) concurrently with the python import chain.
        # Join it here before creating the cupyx handle, which then costs
        # ~11 ms instead of ~2.9 s and is ready before the first SCF
        # contraction instead of blocking it on _handle_lock (measured
        # 0.5-1.35 s main-thread wait inside the FIRST get_veff).  Missing
        # or failed early warm (lib unresolvable, knob off) simply falls
        # back to the full direct handle creation below.
        _early_warm = sys.modules.get('gpu4pyscf.lib._cutensor_early_warm')
        if _early_warm is None:
            try:
                import gpu4pyscf.lib._cutensor_early_warm as _early_warm
            except Exception:
                _early_warm = None
        _early_thread = getattr(_early_warm, '_thread', None) \
            if _early_warm is not None else None

        def _warm_target():
            if _early_thread is not None:
                _early_thread.join(60.0)
            cutensor._get_handle()
            _HANDLE_STATE['ready_perf_s'] = time.perf_counter()

        _warm_thread = threading.Thread(
            target=_warm_target, name='gpu4pyscf-cutensor-warm',
            daemon=True)
        _warm_thread.start()
        import atexit

        def _join_warm_threads(timeout=30.0):
            _warm_thread.join(timeout)
            if _early_thread is not None:
                _early_thread.join(timeout)

        atexit.register(_join_warm_threads)
    else:
        # even with the landed warm thread opted out, the early library
        # thread may still be running (its knob check is identical, so
        # normally it is not); join it at exit in the same LC-60-safe
        # ordering should it ever exist.
        _early_warm = sys.modules.get('gpu4pyscf.lib._cutensor_early_warm')
        _early_thread = getattr(_early_warm, '_thread', None) \
            if _early_warm is not None else None
        if _early_thread is not None and _early_thread.is_alive():
            import atexit
            atexit.register(_early_thread.join, 30.0)

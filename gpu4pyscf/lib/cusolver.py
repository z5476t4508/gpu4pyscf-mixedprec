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


import numpy as np
import cupy
import ctypes
import re
from ctypes.util import find_library
from cupy_backends.cuda.libs import cusolver
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device


def _load_libcusolver():
    '''ctypes handle to the SAME libcusolver cupy has mapped into this process.

    find_library('cusolver') returns a bare SONAME and leaves the final
    resolution to the loader; on hosts with several CUDA toolchains
    installed the loader may bind it to a DIFFERENT toolkit's libcusolver
    than the one cupy loaded.  eigh() below passes cupy's solver handle
    into this module's library, and a handle created by one library
    version is not valid for another -- measured on a CUDA 12.8 + 13.0
    host (cupy loads 12.8's libcusolver.so.11.7.3.90 while find_library
    resolves libcusolver.so.12 to 13.0's library): generalized
    sygvd/zhegvd segfaults outside narrow n windows (17<=n<=64 real,
    13<=n<=64 complex; see the LC-54 evidence in
    tasks/issue-9/2026-10-07-lc54-upstream-test-sweep-2).  Loading the
    library cupy already mapped makes the handle/library pairing correct
    by construction; cupy's own mapping is visible in /proc/self/maps and
    cusolver._getVersion() identifies which mapped library is cupy's.
    '''
    try:
        with open('/proc/self/maps') as f:
            mapped = sorted(set(
                re.findall(r'(/\S*libcusolver\.so\S*)', f.read())))
    except OSError:
        mapped = []
    want = None
    try:
        # version of the library cupy itself loaded, e.g. (11, 7, 3)
        want = cusolver._getVersion()
    except Exception:
        pass
    for path in mapped:
        try:
            lib = ctypes.CDLL(path)  # already mapped: no second load
        except OSError:
            continue
        if not hasattr(lib, 'cusolverDnDsygvd_bufferSize'):
            continue
        if want is None:
            return lib, path
        ver = ctypes.c_int(0)
        try:
            if lib.cusolverGetVersion(ctypes.byref(ver)) != 0:
                continue
        except AttributeError:
            # library too old to report a version: accept it only when
            # it is the sole candidate (the usual single-toolkit host)
            if len(mapped) == 1:
                return lib, path
            continue
        if ver.value == want[0]*1000 + want[1]*100 + want[2]:
            return lib, path
    # no mapped library identified (e.g. no /proc): historical lookup
    name = find_library('cusolver')
    if name is None:
        raise ImportError('libcusolver not found')
    return ctypes.CDLL(name), name


libcusolver, _libcusolver_path = _load_libcusolver()

# workspace size (lwork) provided by the cusolver*_bufferSize is an 32-bit
# integer. For arrays above this dimension, the workspace size would overflow.
MAX_EIGH_DIM = 23150

CUSOLVER_EIG_TYPE_1 = 1
CUSOLVER_EIG_TYPE_2 = 2
CUSOLVER_EIG_TYPE_3 = 3

CUSOLVER_EIG_MODE_NOVECTOR = 0
CUSOLVER_EIG_MODE_VECTOR = 1

libcusolver.cusolverDnDsygvd_bufferSize.restype = int
libcusolver.cusolverDnDsygvd.restype = int

_buffersize = {}

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnDsygvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnDsygvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

# https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-t-sygvd
libcusolver.cusolverDnZhegvd_bufferSize.argtypes = [
    ctypes.c_void_p, # handle
    ctypes.c_int,    # itype
    ctypes.c_int,    # jobz
    ctypes.c_int,    # uplo
    ctypes.c_int,    # n
    ctypes.c_void_p, # *A
    ctypes.c_int,    # lda
    ctypes.c_void_p, # *B
    ctypes.c_int,    # ldb
    ctypes.c_void_p, # *w
    ctypes.c_void_p  # *lwork
]

libcusolver.cusolverDnZhegvd.argtypes = [
    ctypes.c_void_p,  # handle
    ctypes.c_int,     # itype
    ctypes.c_int,     # jobz
    ctypes.c_int,     # uplo
    ctypes.c_int,     # n
    ctypes.c_void_p,  # *A
    ctypes.c_int,     # lda
    ctypes.c_void_p,  # *B
    ctypes.c_int,     # ldb
    ctypes.c_void_p,  # *w
    ctypes.c_void_p,  # *work
    ctypes.c_int,     # lwork
    ctypes.c_void_p   # *devInfo
]

def eigh(h, s, overwrite=False):
    '''
    solve generalized eigenvalue problem
    '''
    assert h.dtype == s.dtype
    assert h.dtype in (np.float64, np.complex128)
    n = h.shape[0]
    if h.dtype == np.complex128 and h.flags.c_contiguous:
        # zhegvd requires the matrices in F-order. For hermitian matrices,
        # .T.copy() is equivalent to .conj()
        A = h.conj()
        B = s.conj()
    elif overwrite:
        A = h
        B = s
    else:
        A = h.copy()
        B = s.copy()
    _handle = device.get_cusolver_handle()
    w = cupy.zeros(n)

    # TODO: reuse workspace
    if (h.dtype, n) in _buffersize:
        lwork = _buffersize[h.dtype, n]
    else:
        lwork = ctypes.c_int(0)
        if h.dtype == np.float64:
            fn = libcusolver.cusolverDnDsygvd_bufferSize
        else:
            fn = libcusolver.cusolverDnZhegvd_bufferSize
        status = fn(
            _handle,
            CUSOLVER_EIG_TYPE_1,
            CUSOLVER_EIG_MODE_VECTOR,
            cublas.CUBLAS_FILL_MODE_LOWER,
            n,
            A.data.ptr,
            n,
            B.data.ptr,
            n,
            w.data.ptr,
            ctypes.byref(lwork)
        )
        lwork = lwork.value
        _buffersize[h.dtype, n] = lwork

        if status != 0:
            raise LinAlgError("failed in buffer size")

    if h.dtype == np.float64:
        fn = libcusolver.cusolverDnDsygvd
    else:
        fn = libcusolver.cusolverDnZhegvd
    work = cupy.empty(lwork, dtype=h.dtype)
    devInfo = cupy.empty(1, dtype=np.int32)
    status = fn(
        _handle,
        CUSOLVER_EIG_TYPE_1,
        CUSOLVER_EIG_MODE_VECTOR,
        cublas.CUBLAS_FILL_MODE_LOWER,
        n,
        A.data.ptr,
        n,
        B.data.ptr,
        n,
        w.data.ptr,
        work.data.ptr,
        lwork,
        devInfo.data.ptr,
    )

    if status != 0:
        raise LinAlgError("failed in eigh kernel")
    return w, A.T

def cholesky(A):
    n = len(A)
    if A.flags.f_contiguous:
        A = A.T
    x = A.copy(order='C')
    handle = device.get_cusolver_handle()
    if A.dtype == np.float64:
        potrf = cusolver.dpotrf
        potrf_bufferSize = cusolver.dpotrf_bufferSize
    else:
        potrf = cusolver.zpotrf
        potrf_bufferSize = cusolver.zpotrf_bufferSize
    buffersize = potrf_bufferSize(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n)
    workspace = cupy.empty(buffersize, dtype=A.dtype)
    dev_info = cupy.empty(1, dtype=np.int32)
    potrf(handle, cublas.CUBLAS_FILL_MODE_UPPER, n, x.data.ptr, n,
        workspace.data.ptr, buffersize, dev_info.data.ptr)

    if dev_info[0] != 0:
        raise LinAlgError('failed to perform Cholesky Decomposition')
    cupy.linalg._util._tril(x,k=0)
    return x

class LinAlgError(RuntimeError):
    pass

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

import unittest
import numpy as np
import scipy.linalg
import cupy as cp
from gpu4pyscf.lib.cusolver import eigh, cholesky

def test_eigh_real():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n)
    a = a + a.T
    b = np.random.rand(n, n)
    b = b.dot(b.T)
    ref = scipy.linalg.eigh(a, b)
    e, c = eigh(cp.asarray(a), cp.asarray(b))
    assert abs(e.get() - ref[0]).max() < 1e-10
    ovlp = c.get().T.dot(b).dot(ref[1])
    assert abs(abs(ovlp) - np.eye(n)).max() < 1e-10

def test_eigh_cmplx():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    a = a + a.conj().T
    b = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    b = b.dot(b.conj().T)
    ref = scipy.linalg.eigh(a, b)
    e, c = eigh(cp.asarray(a), cp.asarray(b))
    assert abs(e.get() - ref[0]).max() < 1e-10
    ovlp = c.get().conj().T.dot(b).dot(ref[1])
    assert abs(abs(ovlp) - np.eye(n)).max() < 1e-10

def test_cholesky_real():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n)
    a = a.dot(a.T)
    ref = np.linalg.cholesky(a)
    x = cholesky(cp.asarray(a))
    assert abs(x.get() - ref).max() < 1e-12

def test_cholesky_cmplx():
    np.random.seed(6)
    n = 12
    a = np.random.rand(n, n) + np.random.rand(n, n) * 1j
    a = a.dot(a.conj().T)
    ref = np.linalg.cholesky(a)
    x = cholesky(cp.asarray(a))
    assert abs(x.get() - ref).max() < 1e-12

def test_loader_pairs_with_cupy_library():
    '''the ctypes library must be the SAME libcusolver cupy loaded (the
    handle passed into sygvd/zhegvd is cupy's).  On a multi-toolkit host
    (CUDA 12.8 + 13.0 both in ld.so.conf.d) the historical
    find_library() lookup let the loader bind libcusolver.so.12 to the
    OTHER toolkit's library while cupy's handle came from 12.8 -- the
    LC-54 cross-version handle mismatch that segfaults generalized
    eigh outside narrow n windows.  The version check is the invariant:
    the loaded library must report the version cupy's own binding
    reports (e.g. 11703 == (11, 7, 3)).'''
    import ctypes
    from gpu4pyscf.lib import cusolver as cs
    ver = ctypes.c_int(0)
    assert cs.libcusolver.cusolverGetVersion(ctypes.byref(ver)) == 0
    want = cs.cusolver._getVersion()
    assert ver.value == want[0]*1000 + want[1]*100 + want[2]

def test_eigh_outside_safe_windows():
    '''generalized eigh at sizes that SEGFAULTED with the mismatched
    library pair (LC-54 windows: dsygvd 17<=n<=64 only, zhegvd
    13<=n<=64 only -- n=70 and n=264 crashed both dtypes).  The
    historical test sizes (n=12) sit below every window, which is why
    the defect was invisible to this suite.  Correctness is checked
    against scipy on the same matrices.'''
    np.random.seed(6)
    for n in (70, 128, 264):
        a = np.random.rand(n, n)
        a = a + a.T
        b = np.random.rand(n, n)
        b = b.dot(b.T) + np.eye(n)*2.0
        ref = scipy.linalg.eigh(a, b)
        e, c = eigh(cp.asarray(a), cp.asarray(b))
        assert abs(e.get() - ref[0]).max() < 1e-9
        ac = np.random.rand(n, n) + np.random.rand(n, n) * 1j
        ac = ac + ac.conj().T
        bc = np.random.rand(n, n) + np.random.rand(n, n) * 1j
        bc = bc.dot(bc.conj().T) + np.eye(n)*2.0
        refc = scipy.linalg.eigh(ac, bc)
        ec, cc = eigh(cp.asarray(ac), cp.asarray(bc))
        assert abs(ec.get() - refc[0]).max() < 1e-9

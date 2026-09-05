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

'''
Global precision mode for mixed-precision computations.

gpu4pyscf defaults to float64 everywhere.  Setting the global mode to 'fp32'
switches the performance-critical tensor contractions (DF J/K build, CDERI
generation and gradient kernels) to float32 inputs while keeping accumulators
and returned results in float64.  The SCF driver (see scf.hf) offers an
automatic 'auto' policy that runs early iterations in fp32 and switches to
fp64 once the energy change drops below a threshold, recovering full float64
accuracy for the converged result.

Example::

    from gpu4pyscf.lib import precision
    with precision.fp32():
        mf = scf.RHF(mol).density_fit()
        mf.kernel()

    # or the automatic mixed-precision policy:
    mf = scf.RHF(mol).density_fit()
    mf.precision_mode = 'auto'
    mf.kernel()
'''

import contextlib

__all__ = ['set_precision', 'get_precision', 'fp32', 'fp64',
           'set_cderi_precision', 'get_cderi_precision',
           'PRECISION_MODE', 'CDERI_PRECISION']

PRECISION_MODE = 'fp64'    # 'fp64' (default) or 'fp32'

# Precision of the CDERI build. Kept separate from PRECISION_MODE: the 'auto'
# SCF policy starts in fp32 but has to finish in fp64, which needs a float64
# CDERI. Only the pure-fp32 screening lane can use a float32 one.
CDERI_PRECISION = 'fp64'

def set_precision(mode):
    '''Set the global precision mode ('fp64' or 'fp32').'''
    global PRECISION_MODE
    if mode not in ('fp64', 'fp32'):
        raise ValueError(f"precision mode must be 'fp64' or 'fp32', got {mode!r}")
    PRECISION_MODE = mode

def get_precision():
    '''Return the current global precision mode.'''
    return PRECISION_MODE

def set_cderi_precision(mode):
    '''Set the precision of the CDERI build ('fp64' or 'fp32').

    In 'fp32' the aux transformation (the dominant cost of the build) runs in
    float32 and the tensor is stored in float32.  This requires the metric to
    be decomposed by eigendecomposition rather than Cholesky: the Cholesky
    factor produces heavy cancellation in that contraction and loses ~200x
    more accuracy.  Screening-grade only; not accurate enough for the float64
    tail of the 'auto' SCF policy.
    '''
    global CDERI_PRECISION
    if mode not in ('fp64', 'fp32'):
        raise ValueError(f"cderi precision must be 'fp64' or 'fp32', got {mode!r}")
    CDERI_PRECISION = mode

def get_cderi_precision():
    '''Return the precision used for the CDERI build.'''
    return CDERI_PRECISION

@contextlib.contextmanager
def fp32():
    '''Context manager to run the enclosed block in float32 mode.

    Only the tensor contractions run in float32; persistent data (CDERI
    storage) keeps float64 precision.
    '''
    global PRECISION_MODE
    saved = PRECISION_MODE
    PRECISION_MODE = 'fp32'
    try:
        yield
    finally:
        PRECISION_MODE = saved

@contextlib.contextmanager
def fp64():
    '''Context manager to run the enclosed block in float64 mode.'''
    global PRECISION_MODE
    saved = PRECISION_MODE
    PRECISION_MODE = 'fp64'
    try:
        yield
    finally:
        PRECISION_MODE = saved

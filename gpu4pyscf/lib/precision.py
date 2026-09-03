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
           'PRECISION_MODE']

PRECISION_MODE = 'fp64'    # 'fp64' (default) or 'fp32'

def set_precision(mode):
    '''Set the global precision mode ('fp64' or 'fp32').'''
    global PRECISION_MODE
    if mode not in ('fp64', 'fp32'):
        raise ValueError(f"precision mode must be 'fp64' or 'fp32', got {mode!r}")
    PRECISION_MODE = mode

def get_precision():
    '''Return the current global precision mode.'''
    return PRECISION_MODE

@contextlib.contextmanager
def fp32():
    '''Context manager to run the enclosed block in float32 mode.'''
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

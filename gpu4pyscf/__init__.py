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

__version__ = '1.8.1'

# FIRSTCALL-LAZYINIT-01 C1: start the cuTensor LIBRARY-level one-time init
# (measured 2.88-3.14 s, of which only ~11 ms is per-handle) as early as
# the interpreter allows -- BEFORE the ~1.4 s pyscf/cupy import chain the
# landed lib/cutensor.py warm thread must wait for.  The early thread runs
# one throwaway cutensorCreate+destroy on a ctypes daemon thread (ctypes
# releases the GIL, so this truly overlaps the python import work) and the
# landed warm thread then joins it and creates the real cupyx handle in
# ~11 ms instead of ~2.9 s, removing the measured handle-lock wait from
# the FIRST get_veff call.  The module is loaded by file path so this
# prelude does NOT trigger the heavy gpu4pyscf.lib import chain, and is
# registered in sys.modules so lib/cutensor.py finds the same instance.
# Fail-open: any error here leaves the landed warm thread paying the full
# cost exactly as before.  Opt out with GPU4PYSCF_CUTENSOR_WARM=0.
try:
    import importlib.util as _ilu
    import os as _os
    import sys as _sys
    if _os.environ.get('GPU4PYSCF_CUTENSOR_WARM', '1') not in \
            ('0', 'false', 'False'):
        _early_spec = _ilu.spec_from_file_location(
            'gpu4pyscf.lib._cutensor_early_warm',
            _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                          'lib', '_cutensor_early_warm.py'))
        if _early_spec is not None and _early_spec.loader is not None:
            _early_mod = _ilu.module_from_spec(_early_spec)
            _sys.modules.setdefault('gpu4pyscf.lib._cutensor_early_warm',
                                    _early_mod)
            _early_spec.loader.exec_module(_early_mod)
            _early_mod.maybe_start()
except Exception:
    pass

from . import _patch_pyscf

from . import lib, grad, hessian, solvent, scf, dft, tdscf, nac

try:
    from .lib import precision  # noqa: F401
except ImportError:
    import warnings
    warnings.warn(
        'gpu4pyscf.lib.precision is missing: this is the release wheel or a '
        'stale source tree, and precision_mode will be a silently-ignored '
        'attribute (every calculation runs float64). Measured cost on a '
        'Tamoxifen auto-lane gradient: 2.70x speedup silently lost. If you '
        'meant to run the source tree, put the REPOSITORY ROOT (not a '
        'subdirectory) on sys.path or set PYTHONPATH.',
        stacklevel=2)

# Overwrite the cupy memory allocator. Make memory pool manage small-sized
# arrays only.
lib.cupy_helper.set_conditional_mempool_malloc()

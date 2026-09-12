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

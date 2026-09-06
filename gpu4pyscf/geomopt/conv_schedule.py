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
Adaptive SCF convergence threshold for geometry optimization.

A geometry optimizer only needs the gradient to be accurate relative to the
gradient itself: at a geometry whose largest force is 1e-1 Eh/Bohr, an error of
1e-3 changes the step it takes by 1%.  Converging the SCF to conv_tol=1e-9
there -- seven orders of magnitude below the energy change the next step will
produce -- buys nothing and costs iterations.  Those iterations are the
expensive ones: under the 'auto' precision policy the tail runs in float64,
measured at 5.8x the cost of a float32 cycle.

This module turns the largest force at the previous geometry into a
convergence threshold for the next one, so early geometries converge loosely
and the threshold tightens on its own as the optimization approaches the
minimum.

Usage -- set it on the mean-field object before optimizing; the gradient
scanner the optimizer builds picks it up::

    mf = scf.RHF(mol).density_fit()
    mf.precision_mode = 'auto'
    mf.conv_tol_schedule = 'auto'
    mol_eq = optimize(mf)

`conv_tol_schedule` accepts 'auto'/True for the default policy, a dict of
`SCFConvSchedule` keyword arguments, any callable ``f(gmax, floor) -> tol``,
or None/False to disable (the default).  It has no effect on a plain
``mf.kernel()``: only a gradient scanner consults it, because only a geometry
optimization supplies the force that drives it.

Calibration
-----------

The error a loose SCF leaves in the nuclear gradient was measured on
Tamoxifen (57 atoms, 537 AO, def2-SVP/def2-universal-jkfit) against a
conv_tol=1e-12 float64 reference::

    conv_tol   1e-4    1e-5    1e-6    1e-7    1e-8    1e-9
    dg_max     6.6e-4  1.2e-4  3.8e-5  6.7e-6  2.1e-6  1.0e-6

which is bounded by ``dg_max <= GRAD_ERROR_COEFF * sqrt(conv_tol)`` with
GRAD_ERROR_COEFF = 0.07.  Inverting that bound for a target error gives the
threshold below.

The same table for the 'auto' precision lane flattens out at dg_max ~ 8.5e-6
from conv_tol=1e-7 down: that is the float32 gradient's own noise floor, and
no conv_tol below it improves the gradient at all.
'''

__all__ = ['SCFConvSchedule', 'make']

# dg_max <= GRAD_ERROR_COEFF * sqrt(conv_tol); see the calibration note above.
GRAD_ERROR_COEFF = 0.07

# Loosest threshold the default policy will ask for.  1e-5 keeps a short
# float64 tail under the 'auto' precision policy (which switches lanes at
# |dE| < 1e-4), so the energy handed to the optimizer stays float64-grade.
# Raising it above that switch point converges in pure float32 instead, which
# is faster still but leaves ~2e-4 Eh of noise on the energy -- fine against
# the millihartree changes of an early step, not against geomeTRIC's 1e-6 Eh
# convergence criterion.
DEFAULT_CEILING = 1e-5

# Gradient error the schedule aims for, as a fraction of the largest force.
DEFAULT_REL_ERROR = 0.03


class SCFConvSchedule:
    '''Map the largest force at the previous geometry to an SCF conv_tol.

    Args:
        rel_error: target gradient error as a fraction of the largest force.
            The default 0.03 leaves the optimizer's step direction intact and,
            at the point where geomeTRIC declares convergence (gmax 4.5e-4),
            corresponds to an absolute error of ~1.4e-5 Eh/Bohr.
        ceiling: loosest conv_tol the schedule will return.
        floor: tightest conv_tol it will return.  None (the default) means the
            conv_tol the caller had already set, so the schedule never
            converges less tightly than was asked for.
    '''

    def __init__(self, rel_error=DEFAULT_REL_ERROR, ceiling=DEFAULT_CEILING,
                 floor=None):
        if not 0 < rel_error < 1:
            raise ValueError(f'rel_error must be in (0, 1), got {rel_error}')
        if ceiling <= 0:
            raise ValueError(f'ceiling must be positive, got {ceiling}')
        if floor is not None and not 0 < floor <= ceiling:
            raise ValueError(
                f'floor must be positive and <= ceiling, got {floor}')
        self.rel_error = rel_error
        self.ceiling = ceiling
        self.floor = floor

    def __call__(self, gmax, floor):
        '''conv_tol to use at the next geometry.

        `gmax` is the largest force at the previous geometry, or None at the
        first one -- with no force to scale against there is nothing to base a
        relaxation on, so the caller's own conv_tol is returned unchanged.
        '''
        if self.floor is not None:
            floor = self.floor
        if gmax is None:
            return floor
        tol = (self.rel_error * gmax / GRAD_ERROR_COEFF) ** 2
        return min(max(tol, floor), max(self.ceiling, floor))

    def __repr__(self):
        return (f'{type(self).__name__}(rel_error={self.rel_error}, '
                f'ceiling={self.ceiling}, floor={self.floor})')


def make(spec):
    '''Build a schedule from the value of a `conv_tol_schedule` attribute.

    Accepts None/False (no schedule), 'auto'/True (the default policy), a dict
    of SCFConvSchedule keyword arguments, or any callable f(gmax, floor)->tol.
    '''
    if spec is None or spec is False:
        return None
    if spec is True or spec == 'auto':
        return SCFConvSchedule()
    if isinstance(spec, dict):
        return SCFConvSchedule(**spec)
    if callable(spec):
        return spec
    raise ValueError(
        "conv_tol_schedule must be None, 'auto', a dict of SCFConvSchedule "
        f'arguments, or a callable f(gmax, floor)->tol, got {spec!r}')

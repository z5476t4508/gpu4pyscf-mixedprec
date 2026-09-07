"""Acceptance metrics for the mixed-precision Hessian benchmarks.

This is a thin wrapper over pyscf.hessian.thermo. It exists for the metric
choices, not the maths -- PySCF already does the harmonic analysis correctly,
and an earlier version of this file reimplemented it before I checked (the two
agree to 3.5e-6 cm^-1 on Tamoxifen; the hand-rolled one is gone).

What it is for:

1. Projection, not slicing. Every max|dnu| before this file took the
   vibrational modes to be eigvalsh(H_mass)[6:] -- valid only at a stationary
   geometry, and none of the repo's benchmark geometries is one at the level of
   theory being measured. Tamoxifen/def2-SVP under r2SCAN has nine imaginary
   modes, so sorting pushed the three most negative vibrations out of the
   compared set and pulled translations and rotations in. Since frequency error
   goes as dw ~ dH/(2*mu*w), a near-zero mode amplifies a Hessian error while
   carrying no physics:

       projected (correct)   min|nu| =  53.65 cm^-1
       naive [6:]            min|nu| =  15.24 cm^-1   <- a contaminated rotation

   That inflated the oracle figure on Tamoxifen from 0.142 to 0.181 (1.27x).

2. rms and thermochemistry over max. max|dnu| is taken over modes, so it grows
   with mode count and systematically penalises larger molecules. Measured
   under the same oracle: Vitamin C (54 modes) vs Tamoxifen (165 modes) differ
   4.8x on max (0.038 / 0.181) but only 1.8x on rms (0.009 / 0.016) -- most of
   the max gap is sampling, not a larger per-mode error. ZPE and S_vib are what
   a user actually consumes, and S_vib weights the soft modes that carry the
   error, so they are the metrics that transfer between molecules.
"""
import numpy as np
from pyscf.hessian import thermo


def frequencies(mol, hess):
    '''Vibrational frequencies in cm^-1, 3N-6 of them (3N-5 if linear),
    translations and rotations projected out. Imaginary modes come back as
    negative reals so they can be subtracted between two calculations.
    '''
    hess = np.asarray(getattr(hess, 'get', lambda: hess)())
    res = thermo.harmonic_analysis(mol, hess, imaginary_freq=False)
    return np.asarray(res['freq_wavenumber']).real


def thermochemistry(mol, hess, temperature=298.15):
    '''ZPE (Eh) and vibrational entropy (Eh/K) from a Hessian.

    Imaginary modes are dropped rather than folded in: they make both
    quantities undefined, and every benchmark geometry here has some. That
    stays consistent when comparing two calculations on the *same* geometry,
    but it does mean the absolute values are meaningless off a minimum -- only
    the difference between two lanes is.
    '''
    freq = frequencies(mol, hess)
    nu_au = freq[freq > 0] / thermo.nist.HARTREE2WAVENUMBER
    kt = thermo.nist.BOLTZMANN / thermo.nist.HARTREE2J * temperature
    x = nu_au / kt
    zpe = .5 * nu_au.sum()
    # S_vib/k = sum[ x/(e^x - 1) - ln(1 - e^-x) ]
    s_vib = (x / np.expm1(x) - np.log1p(-np.exp(-x))).sum() * kt / temperature
    return zpe, s_vib

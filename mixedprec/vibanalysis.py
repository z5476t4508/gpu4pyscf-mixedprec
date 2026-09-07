"""Shared vibrational analysis for the mixed-precision benchmarks.

Why this exists: the benchmarks up to step7a took the vibrational modes to be
`eigvalsh(H_mass)[6:]` -- the eigenvalues after the six smallest. That is only
right at a stationary geometry. The repo's Tamoxifen geometry has ELEVEN
imaginary modes at r2SCAN/def2-SVP:

    -500.8 -489.3 -477.8 -326.7 -295.3 -202.0 -176.9 -144.8 -122.9 -113.0
     -67.7   15.2   65.6 ...

so sorting puts genuine imaginary vibrations in the first six slots and pushes
the translations and rotations *into* the set being compared. Those have
frequencies near zero, and since an error propagates as dw ~ dH/(2*mu*w), a
near-zero mode amplifies any Hessian error without carrying any physics. Every
max|dnu| quoted before this file was therefore partly translation/rotation
noise.

(The conclusions drawn from those numbers still hold -- the rejected
_get_vxc_deriv2_task port raised max|dH| from 6.4e-7 to 2.75e-4, a 400x jump,
with rms 0.475 cm^-1 over 165 modes, neither of which depends on where the
first six modes land. But the headline "5.579 cm^-1" was not a clean physical
frequency error and should not have been quoted as one.)

`frequencies` projects the six translation/rotation vectors out of the
mass-weighted Hessian (the standard Eckart treatment) instead of assuming they
sort to the front, and returns only the 3N-6 vibrational frequencies. At a
non-stationary geometry the rotations are not exact zero modes, so the
projection is approximate there too -- which is a reason to run the acceptance
suite on optimized geometries, not a reason to skip the projection.
"""
import numpy as np

HARTREE2WAVENUMBER = 219474.6313632
AMU2AU = 1822.888486209


def _trans_rot_basis(coords, masses):
    '''orthonormal mass-weighted translation and rotation vectors, (6, 3N)'''
    natm = len(masses)
    sqrt_m = np.sqrt(masses)
    com = (coords * masses[:, None]).sum(axis=0) / masses.sum()
    r = coords - com

    vecs = np.zeros((6, natm, 3))
    for i in range(3):
        vecs[i, :, i] = sqrt_m                      # translations
    # rotations: sqrt(m) * (e_i x r)
    for i, (a, b) in enumerate(((1, 2), (2, 0), (0, 1))):
        vecs[3 + i, :, a] = sqrt_m * -r[:, b]
        vecs[3 + i, :, b] = sqrt_m * r[:, a]

    vecs = vecs.reshape(6, natm * 3)
    # Gram-Schmidt; a linear molecule leaves one rotation vector null, so drop
    # anything that is numerically dependent rather than assuming rank 6.
    basis = []
    for v in vecs:
        for b in basis:
            v = v - b * (b @ v)
        n = np.linalg.norm(v)
        if n > 1e-8:
            basis.append(v / n)
    return np.array(basis)


def frequencies(mol, hess, project=True):
    '''Vibrational frequencies in cm^-1, sign kept for imaginary modes.

    hess : (natm, natm, 3, 3) array (CuPy or NumPy)
    project : project out translations/rotations. Leave True; see module
        docstring for what taking eigvalsh(...)[6:] instead costs.
    '''
    hess = np.asarray(getattr(hess, 'get', lambda: hess)())
    natm = mol.natm
    n = 3 * natm
    masses = mol.atom_mass_list(isotope_avg=True)
    inv_sqrt_m = np.repeat(masses, 3) ** -.5
    h = hess.transpose(0, 2, 1, 3).reshape(n, n)
    h = h * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

    if not project:
        ev = np.linalg.eigvalsh(h)
    else:
        basis = _trans_rot_basis(mol.atom_coords(), masses)
        # Diagonalize inside the orthogonal complement of the trans/rot space,
        # rather than projecting and then dropping the smallest eigenvalues.
        # The latter repeats the bug this module exists to fix: with imaginary
        # modes present, the projected-out ~0 eigenvalues sort *after* the
        # negative ones, so a leading slice discards real vibrations instead.
        q, _ = np.linalg.qr(np.hstack([basis.T, np.eye(n)]))
        comp = q[:, len(basis):n]
        ev = np.linalg.eigvalsh(comp.T @ h @ comp)

    ev = ev / AMU2AU
    return np.sign(ev) * np.sqrt(np.abs(ev)) * HARTREE2WAVENUMBER

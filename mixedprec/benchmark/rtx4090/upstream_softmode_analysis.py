#!/usr/bin/env python3
"""LC-207 soft-mode curvature analysis (CPU fp64 pyscf, standalone).

Characterizes the methoxy/b3lyp/def2-svp cos +1.000 displaced-solution
family (batch uc-20261018-a FAIL rows B/C + the A5 R0_AUTO branch) at
the orbital-Hessian level, using the LC-206/LC-207 saved converged
densities as the family points and R (the batch strict-fp64 default
lane) as the reference:

  anchors   -- CPU energies at each family dm (direct evaluation AND
               seeded SCF) vs the frozen GPU lane anchor energies; CPU
               nuclear gradient of the R seeded state vs the batch R
               anchor gradient (implementation-agreement axis, NOT a
               precision gate)
  rotation  -- family displacement vectors kappa in R's canonical MO
               basis (pyscf (nvir,nocc) x1 layout: kappa =
               dDM_MO[virt,occ]), occ/virt block purity, pairwise
               collinearity (rotation space and AO space)
  spectrum  -- dense exact eigh of the internal-stability Hessian at
               R's seeded state (the pyscf newton_ah.gen_g_hop_uhf
               operator with the stability() x2 convention); softest
               modes, separation, Davidson cross-check
  align     -- cos(kappa_i, v_1) per family point + spectral weights
  spectrumC -- seeded SCF from dm_C, dense stability there, softest
               mode overlap with R's (AO-space density-perturbation
               cosine; the two MO bases differ, so the overlap is
               computed basis-independently)
  profile   -- E(t . v1) and E(s . kappa_C) scans with exactly
               antisymmetric +- pairs (odd terms cancel; LC-65
               discipline), symmetric curvature fits, and the family
               points' energy placement on the line

Everything here is mechanical CPU measurement (pyscf 2.14.0, the
independent implementation); verdicts live in the session docs.  b3lyp
carries no NLC response (pyscf's UHF internal-stability operator uses
with_nlc=False), so no custom NLC operator is needed -- the wb97mv
multi-basin case would need the LC-65 NLC-inclusive operator and is
out of scope for this unit.

numpy/pyscf/scipy are imported lazily inside the numeric paths so the
module stays importable stdlib-only for the unit tests (which cover
plan validation, grid/curvature arithmetic, and block bookkeeping).
"""

import argparse
import hashlib
import json
import os
import sys
import time

SAMPLE_SCHEMA = 'softmode-analysis-v1'

REQUIRED_PLAN_KEYS = (
    'unit', 'amendment_id', 'batch_id', 'evidence_dir', 'cell',
    'settings', 'family_dms', 'profile',
)
REQUIRED_CELL_KEYS = ('tag', 'index', 'geom', 'basis', 'xc', 'charge',
                      'spin')
REQUIRED_SETTINGS_KEYS = ('auxbasis', 'grids_level', 'conv_tol',
                          'direct_scf_tol', 'max_cycle', 'init_guess')
FAMILY_POINTS = ('R', 'R0_AUTO', 'C', 'B')

PROFILE_FRACS = (0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75, 1.0, -1.0,
                 1.25, -1.25)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


# ---------------------------------------------------------------------------
# plan handling (stdlib-only; unit-tested)
# ---------------------------------------------------------------------------

def load_plan(path):
    with open(path, encoding='utf-8') as f:
        plan = json.load(f)
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise RuntimeError(f'softmode plan missing required key {key!r}')
    for key in REQUIRED_CELL_KEYS:
        if key not in plan['cell']:
            raise RuntimeError(f'plan cell missing {key!r}')
    for key in REQUIRED_SETTINGS_KEYS:
        if key not in plan['settings']:
            raise RuntimeError(f'plan settings missing {key!r}')
    fam = plan['family_dms']
    for point in FAMILY_POINTS:
        if point not in fam:
            raise RuntimeError(f'plan family_dms missing {point!r}')
        entry = fam[point]
        for key in ('path', 'anchor_e', 'anchor_source'):
            if key not in entry:
                raise RuntimeError(
                    f'family point {point!r} missing {key!r}')
        if not isinstance(entry['anchor_e'], (int, float)):
            raise RuntimeError(
                f'family point {point!r} has a non-numeric anchor_e')
    if 'R' not in fam:
        raise RuntimeError('the reference point R is mandatory')
    for key in ('t_ref_source', 'fracs'):
        if key not in plan['profile']:
            raise RuntimeError(f'plan profile missing {key!r}')
    return plan


def verify_inputs(plan):
    '''File-level fail-closed checks at run time (paths must exist).'''
    missing = []
    if not os.path.isfile(plan['cell']['geom']):
        missing.append(('geom', plan['cell']['geom']))
    for point, entry in plan['family_dms'].items():
        if not os.path.isfile(entry['path']):
            missing.append((f'dm:{point}', entry['path']))
    if missing:
        raise RuntimeError(
            'softmode plan inputs missing: ' + '; '.join(
                f'{k}={p}' for k, p in missing))
    return True


# ---------------------------------------------------------------------------
# pure arithmetic helpers (stdlib-only; unit-tested)
# ---------------------------------------------------------------------------

def t_grid(t_ref, fracs=PROFILE_FRACS):
    '''Sorted unique scan amplitudes (fractions of the reference
    amplitude); the 0 point appears exactly once.'''
    ts = sorted({round(f * t_ref, 12) for f in fracs})
    if 0.0 not in ts:
        ts.insert(0, 0.0)
    return ts


def sym_curvature(e_minus, e_zero, e_plus, t):
    '''Symmetric (antisymmetric-rotation) curvature estimate
    (E(t)+E(-t)-2E(0))/t^2 -- odd terms cancel exactly.'''
    return (e_plus + e_minus - 2.0 * e_zero) / (t * t)


def odd_part(e_minus, e_plus):
    '''(E(+t)-E(-t))/2 -- zero for an exactly antisymmetric rotation
    pair at a stationary point; its size measures the residual
    gradient/precision floor along the direction.'''
    return (e_plus - e_minus) / 2.0


def curvature_series(profile):
    '''profile: dict t -> E.  Returns [(t, lambda_t)] for every t>0
    whose -t partner is present.'''
    out = []
    for t in sorted(profile):
        if t <= 0 or -t not in profile:
            continue
        lam = sym_curvature(profile[-t], profile[0.0], profile[t], t)
        out.append((t, lam))
    return out


def spectral_weight(cos_value):
    '''Fraction of a displacement captured by an eigenmode given the
    cosine with the (unit-normalized) eigenvector.'''
    return cos_value * cos_value


def ladder_rows(points):
    '''points: list of dicts with keys dE (vs R), t_proj (projected
    amplitude), label.  Returns rows sorted by |t_proj| with the
    dE/t^2 quadratic-consistency ratio per row (None when t is 0).'''
    rows = []
    for p in sorted(points, key=lambda q: abs(q.get('t_proj') or 0.0)):
        t = p.get('t_proj')
        d = p.get('dE')
        ratio = (d / (t * t)) if (t and d is not None) else None
        rows.append({'label': p['label'], 'dE': d, 't_proj': t,
                     'dE_over_t2': ratio})
    return rows


# ---------------------------------------------------------------------------
# numeric layer (numpy/pyscf/scipy imported lazily)
# ---------------------------------------------------------------------------

def dm_from_npz(path):
    '''Basin-probe npz round-trip: sorted s-keys -> tuple when more
    than one block (UKS), else the single array (RKS).'''
    import numpy as np
    z = np.load(path)
    keys = sorted(z.files)
    blocks = [np.asarray(z[k]) for k in keys]
    return tuple(blocks) if len(blocks) > 1 else blocks[0]


def build_mol_mf(cell, settings):
    from pyscf import gto, dft
    mol = gto.M(atom=cell['geom'], basis=cell['basis'],
                charge=cell['charge'], spin=cell['spin'],
                unit='Angstrom', verbose=0)
    mf = dft.UKS(mol).density_fit(auxbasis=settings['auxbasis'])
    mf.xc = cell['xc']
    mf.grids.level = settings['grids_level']
    mf.conv_tol = settings['conv_tol']
    mf.direct_scf_tol = settings['direct_scf_tol']
    mf.max_cycle = settings['max_cycle']
    mf.init_guess = settings['init_guess']
    return mol, mf


def seeded_scf(mf, dm, label, log):
    t0 = time.time()
    e = mf.kernel(dm0=dm)
    wall = time.time() - t0
    rec = {
        'label': label,
        'e_tot': float(e),
        'converged': bool(mf.converged),
        'cycles': int(getattr(mf, 'cycles', 0) or 0),
        'wall_s': wall,
    }
    log(f'seeded SCF {label}: E={e:.12f} conv={mf.converged} '
        f'cyc={rec["cycles"]} ({wall:.1f}s)')
    return rec


def energy_at(mf, dm):
    return float(mf.energy_tot(dm=dm))


def gradient_of_state(mf):
    g = mf.nuc_grad_method()
    return g.kernel()


def occ_virt(mo_occ_spin):
    import numpy as np
    occ = np.where(mo_occ_spin > 0)[0]
    virt = np.where(mo_occ_spin == 0)[0]
    return occ, virt


def mo_space_dm(dm_ao_spin, mo, s):
    import numpy as np
    return mo.T @ s @ dm_ao_spin @ s @ mo


def extract_rotation(dm_mo, occ, virt):
    '''kappa (pyscf x1 layout, shape (nvir,nocc)) = dDM_MO[virt,occ].
    Returns (kappa, purity_residual_fraction) where the residual is the
    Frobenius mass OUTSIDE the two off-diagonal occ-virt blocks
    (occ-occ/virt-virt blocks are gauge rotations, not family
    displacement; for a pure rotation they are O(kappa^2)).  The
    in-block mass is the joint norm sqrt(|A|^2+|A^T|^2) of the
    transpose-pair blocks, NOT |A|+|A^T| (which would exceed the total
    norm by sqrt(2) for a pure rotation).'''
    import numpy as np
    kappa = dm_mo[np.ix_(virt, occ)]
    total = float(np.linalg.norm(dm_mo))
    if total == 0.0:
        return kappa, 0.0
    cross_sq = float(np.sum(dm_mo[np.ix_(virt, occ)] ** 2)
                     + np.sum(dm_mo[np.ix_(occ, virt)] ** 2))
    return kappa, 1.0 - (cross_sq ** 0.5) / total


def flatten_x(x1a, x1b):
    import numpy as np
    return np.concatenate([np.asarray(x1a).ravel(),
                           np.asarray(x1b).ravel()])


def rotation_matrix(x1, occ, virt, nao):
    '''K (nao,nao) antisymmetric: K[virt,occ]=x1, K[occ,virt]=-x1^T,
    zeros within the occ/virt blocks (occ-occ rotations are gauge).'''
    import numpy as np
    K = np.zeros((nao, nao))
    K[np.ix_(virt, occ)] = x1
    K[np.ix_(occ, virt)] = -x1.T
    return K


def rotated_density(mo_spin, mo_occ_spin, x1, t):
    '''Density of the orbitals rotated by t*x1 (exact exponential, no
    small-t approximation).'''
    from scipy.linalg import expm
    occ, virt = occ_virt(mo_occ_spin)
    nao = mo_spin.shape[0]
    K = rotation_matrix(x1 * t, occ, virt, nao)
    U = expm(K)
    Cp = mo_spin @ U
    return Cp[:, occ] @ Cp[:, occ].T


def dm_perturbation_ao(mo_spin, mo_occ_spin, x1):
    '''AO-space first-order density perturbation of the rotation x1
    (pyscf gen_g_hop_uhf convention: d1 = orbv @ x1 @ orbo^T + h.c.).
    Basis-independent object for cross-basis mode overlaps.'''
    occ, virt = occ_virt(mo_occ_spin)
    orbv = mo_spin[:, virt]
    orbo = mo_spin[:, occ]
    d1 = orbv @ x1 @ orbo.T
    return d1 + d1.T


def ao_cosine(dm_a, dm_b):
    '''Spin-summed Frobenius cosine of two (per-spin tuple) AO-space
    density directions.'''
    import numpy as np
    num = 0.0
    na = 0.0
    nb = 0.0
    for a, b in zip(dm_a, dm_b):
        num += float(np.sum(a * b))
        na += float(np.sum(a * a))
        nb += float(np.sum(b * b))
    if na == 0.0 or nb == 0.0:
        return None
    return num / (na * nb) ** 0.5


def cosine(a, b):
    import numpy as np
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(a @ b / (na * nb))


def dense_stability(mf, log):
    '''Dense exact eigh of the internal-stability Hessian at the mf
    state (pyscf stability() convention: H = 2 * gen_g_hop_uhf hop).'''
    import numpy as np
    from pyscf.soscf import newton_ah
    from pyscf import lib as pyscf_lib

    g, hop, hdiag = newton_ah.gen_g_hop_uhf(
        mf, mf.mo_coeff, mf.mo_occ, with_symmetry=False)
    hdiag = hdiag * 2
    dim = int(g.size)

    t0 = time.time()
    H = np.empty((dim, dim))
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H[:, i] = hop(e).real * 2
    wall = time.time() - t0
    asym = float(np.abs(H - H.T).max())
    w, v = np.linalg.eigh(H)
    r1 = float(np.linalg.norm(H @ v[:, 0] - w[0] * v[:, 0]))
    hdiag_dev = float(np.abs(np.diag(H) - hdiag).max())

    # Davidson cross-check (pyscf's own solver on the same operator)
    def hop2(x):
        return hop(x).real * 2

    def precond(dx, e, x0):
        hd = hdiag - e
        hd[abs(hd) < 1e-8] = 1e-8
        return dx / hd

    x0 = np.zeros_like(g)
    x0[g != 0] = 1.0 / hdiag[g != 0]
    e_dav, _ = pyscf_lib.davidson(hop2, x0, precond, tol=1e-9, nroots=3,
                                  verbose=0)
    e_dav = np.sort(np.asarray(e_dav).ravel())[:3]

    log(f'dense stability: dim={dim} build {wall:.0f}s '
        f'asym={asym:.2e} softest-5={np.array2string(w[:5], precision=8)} '
        f'davidson-3={np.array2string(e_dav, precision=8)}')
    return {
        'dim': dim,
        'wall_s': wall,
        'operator_asymmetry': asym,
        'orbital_gradient_norm': float(np.linalg.norm(g)),
        'eigenvalues': w.tolist(),
        'eigenvectors': v,
        'softest_residual_norm': r1,
        'hdiag_vs_dense_dev': hdiag_dev,
        'davidson_softest3': e_dav.tolist(),
    }


def mode_overlaps(spec_r, spec_c, mf_r, mf_c):
    '''AO-space cosine of the softest eigenvectors of the two spectra
    (the MO bases differ; the density perturbation does not).'''
    import numpy as np
    x1a_r, x1b_r = split_x(spec_r['eigenvectors'][:, 0],
                            mf_r.mo_occ[0], mf_r.mo_occ[1])
    x1a_c, x1b_c = split_x(spec_c['eigenvectors'][:, 0],
                            mf_c.mo_occ[0], mf_c.mo_occ[1])
    d_r = (dm_perturbation_ao(mf_r.mo_coeff[0], mf_r.mo_occ[0], x1a_r),
           dm_perturbation_ao(mf_r.mo_coeff[1], mf_r.mo_occ[1], x1b_r))
    d_c = (dm_perturbation_ao(mf_c.mo_coeff[0], mf_c.mo_occ[0], x1a_c),
           dm_perturbation_ao(mf_c.mo_coeff[1], mf_c.mo_occ[1], x1b_c))
    return ao_cosine(d_r, d_c)


def split_x(x, mo_occ_a, mo_occ_b):
    '''Split the concatenated pyscf rotation vector into per-spin
    (nvir,nocc) blocks (gen_g_hop_uhf layout: [x_a, x_b], each
    nvir-major ravel of an (nvir,nocc) matrix).'''
    import numpy as np
    nva = int((mo_occ_a == 0).sum())
    noa = int((mo_occ_a > 0).sum())
    nvb = int((mo_occ_b == 0).sum())
    nob = int((mo_occ_b > 0).sum())
    x = np.asarray(x)
    return (x[:nva * noa].reshape(nva, noa),
            x[nva * noa:].reshape(nvb, nob))


def profile_scan(mf, mo, mo_occ, x1a, x1b, t_ref, fracs, log):
    '''E(t) along a rotation direction (exact exponential rotations);
    one fresh energy_tot per point on the shared built grids.'''
    out = {}
    for t in t_grid(t_ref, fracs):
        dma = rotated_density(mo[0], mo_occ[0], x1a, t)
        dmb = rotated_density(mo[1], mo_occ[1], x1b, t)
        e = energy_at(mf, (dma, dmb))
        out[t] = e
    e0 = out[0.0]
    for t in sorted(out):
        log(f'  profile t={t:+.4f}: dE={out[t] - e0:+.3e}')
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(plan, out_json, out_md, log):
    import numpy as np

    verify_inputs(plan)
    cell, settings = plan['cell'], plan['settings']
    mol, mf = build_mol_mf(cell, settings)
    log(f'cell {cell["tag"]}: nao={mol.nao_nr()} '
        f'nelec={mol.nelec} spin={cell["spin"]}')

    # family densities + anchor energies
    dms = {p: dm_from_npz(plan['family_dms'][p]['path'])
           for p in FAMILY_POINTS}

    def as_ukp_spin2(x):
        '''The basin-probe npz round-trip preserves make_rdm1's EXACT
        structure: GPU UKS dms arrive as ONE stacked (2,nao,nao) block
        (single s0 key), a 2-tuple stays a tuple.  Both index per-spin
        via [0]/[1]; accept exactly those two UKS forms.'''
        import numpy as np
        if isinstance(x, tuple) and len(x) == 2:
            return x
        if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[0] == 2:
            return x
        return None

    for p in FAMILY_POINTS:
        if as_ukp_spin2(dms[p]) is None:
            raise RuntimeError(
                f'family point {p} density is not a 2-spin UKS dm: '
                f'{type(dms[p])} {getattr(dms[p], "shape", None)}')
    inputs = {p: {
        'path': plan['family_dms'][p]['path'],
        'sha256': sha256_file(plan['family_dms'][p]['path']),
        'anchor_e': plan['family_dms'][p]['anchor_e'],
        'anchor_source': plan['family_dms'][p]['anchor_source'],
    } for p in FAMILY_POINTS}

    # cross-implementation anchors: direct energy evaluation at each dm
    energy_at_dm = {}
    for p in FAMILY_POINTS:
        e = energy_at(mf, dms[p])
        energy_at_dm[p] = {
            'e_cpu': e,
            'anchor_e': inputs[p]['anchor_e'],
            'diff': e - inputs[p]['anchor_e'],
        }
        log(f'anchor {p}: E_cpu(dm)={e:.12f} '
            f'dE_vs_gpu={e - inputs[p]["anchor_e"]:+.3e}')

    # R state: seeded SCF from dm_R (stationary point; canonical MOs)
    scf_R = seeded_scf(mf, dms['R'], 'R', log)
    g_R = gradient_of_state(mf)
    grad_anchor_path = plan['family_dms']['R'].get('anchor_gradient_path')
    grad_anchor = None
    if grad_anchor_path:
        with open(grad_anchor_path, encoding='utf-8') as f:
            doc = json.load(f)
        g_ref = np.asarray(
            doc['rows'][0]['result']['gradient'], dtype=float)
        grad_anchor = {
            'path': grad_anchor_path,
            'sha256': sha256_file(grad_anchor_path),
            'max_abs_diff': float(np.abs(g_R - g_ref).max()),
            'cpu_grad_norm': float(np.linalg.norm(g_R)),
        }
        log(f'gradient anchor R: max|dg| vs batch R doc = '
            f'{grad_anchor["max_abs_diff"]:.3e}')

    # P1b coverage: seeded SCF from EVERY family dm on a fresh object
    # (each is a stationary point; the seeds must stay in place)
    seeded_family = {'R': dict(scf_R)}
    for p in ('R0_AUTO', 'C', 'B'):
        if p == 'C':
            continue          # seeded below on its own object (spectrum)
        mf_p = build_mol_mf(cell, settings)[1]
        seeded_family[p] = seeded_scf(mf_p, dms[p], p, log)

    # rotation space: kappa of each displaced point in R's MO basis
    mo, mo_occ = mf.mo_coeff, mf.mo_occ
    s = mol.intor_symmetric('int1e_ovlp')
    dm_R = dms['R']
    kappas = {}
    rot_info = {}
    for p in ('R0_AUTO', 'C', 'B'):
        dd = [np.asarray(dms[p][k]) - np.asarray(dm_R[k])
              for k in range(2)]
        dm_mo = [mo_space_dm(dd[k], mo[k], s) for k in range(2)]
        occ_a, virt_a = occ_virt(mo_occ[0])
        occ_b, virt_b = occ_virt(mo_occ[1])
        ka, ra = extract_rotation(dm_mo[0], occ_a, virt_a)
        kb, rb = extract_rotation(dm_mo[1], occ_b, virt_b)
        kappas[p] = flatten_x(ka, kb)
        rot_info[p] = {
            'kappa_norm': float(np.linalg.norm(kappas[p])),
            'purity_residual_fraction': (ra + rb) / 2.0,
            'ao_dm_norm': float(sum(np.linalg.norm(x) for x in dd)),
            'anchor_dE': inputs[p]['anchor_e'] - inputs['R']['anchor_e'],
        }
        log(f'rotation {p}: |kappa|={rot_info[p]["kappa_norm"]:.4f} '
            f'residual={rot_info[p]["purity_residual_fraction"]:.3f} '
            f'dE_anchor={rot_info[p]["anchor_dE"]:+.3e}')

    # family collinearity (rotation space + AO space)
    def ao_dir(p):
        return tuple(np.asarray(dms[p][k]) - np.asarray(dm_R[k])
                     for k in range(2))

    family_cos = {}
    pairs = (('B', 'C'), ('B', 'R0_AUTO'), ('C', 'R0_AUTO'))
    for a, b in pairs:
        family_cos[f'{a}_{b}'] = {
            'rotation_space': cosine(kappas[a], kappas[b]),
            'ao_space': ao_cosine(ao_dir(a), ao_dir(b)),
        }

    # dense stability at R
    spec_R = dense_stability(mf, log)
    w = np.asarray(spec_R['eigenvalues'])
    v1 = spec_R['eigenvectors'][:, 0]
    spec_R_summary = {
        'dim': spec_R['dim'],
        'softest_10': w[:10].tolist(),
        'hardest': float(w[-1]),
        'separation_ratio_l2_over_abs_l1':
            float(abs(w[1]) / abs(w[0])) if w[0] != 0 else None,
        'operator_asymmetry': spec_R['operator_asymmetry'],
        'softest_residual_norm': spec_R['softest_residual_norm'],
        'hdiag_vs_dense_dev': spec_R['hdiag_vs_dense_dev'],
        'davidson_softest3': spec_R['davidson_softest3'],
        'orbital_gradient_norm': spec_R['orbital_gradient_norm'],
        'wall_s': spec_R['wall_s'],
    }

    # soft-mode alignment of the family
    alignment = {}
    for p in ('R0_AUTO', 'C', 'B'):
        c = cosine(kappas[p], v1)
        alignment[p] = {
            'cos_with_v1': c,
            'spectral_weight_v1': spectral_weight(c) if c is not None
            else None,
            't_proj': float(kappas[p] @ v1),
            'top5_weights': top5_weights(kappas[p],
                                         spec_R['eigenvectors']),
        }

    # C-point state + spectrum
    mf_c = build_mol_mf(cell, settings)[1]
    scf_C = seeded_scf(mf_c, dms['C'], 'C', log)
    seeded_family['C'] = dict(scf_C)
    spec_C = dense_stability(mf_c, log)
    w_c = np.asarray(spec_C['eigenvalues'])
    overlap = mode_overlaps(spec_R, spec_C, mf, mf_c)
    spec_C_summary = {
        'dim': spec_C['dim'],
        'softest_10': w_c[:10].tolist(),
        'separation_ratio_l2_over_abs_l1':
            float(abs(w_c[1]) / abs(w_c[0])) if w_c[0] != 0 else None,
        'softest_residual_norm': spec_C['softest_residual_norm'],
        'operator_asymmetry': spec_C['operator_asymmetry'],
        'wall_s': spec_C['wall_s'],
    }
    mode_overlap = {
        'ao_density_perturbation_cos': overlap,
        'lambda1_R': float(w[0]),
        'lambda1_C': float(w_c[0]),
    }

    # profile scans (exact rotations; antisymmetric +- pairs)
    fracs = plan['profile']['fracs']
    x1a_v1, x1b_v1 = split_x(v1, mo_occ[0], mo_occ[1])
    t_C = alignment['C']['t_proj']
    t_ref_v1 = abs(t_C) if t_C else rot_info['C']['kappa_norm']
    prof_v1 = profile_scan(mf, mo, mo_occ, x1a_v1, x1b_v1, t_ref_v1,
                           fracs, log)
    x1a_c, x1b_c = split_x(kappas['C'], mo_occ[0], mo_occ[1])
    kn_C = rot_info['C']['kappa_norm']
    prof_kc = profile_scan(mf, mo, mo_occ, x1a_c / kn_C, x1b_c / kn_C,
                           kn_C, fracs, log)

    e0_v1 = prof_v1[0.0]
    e0_kc = prof_kc[0.0]
    profile_out = {
        'v1_line': {
            't_ref': t_ref_v1,
            't_ref_source': plan['profile']['t_ref_source'],
            'points': {f'{t:.6g}': e for t, e in prof_v1.items()},
            'curvatures': curvature_series(prof_v1),
        },
        'kappa_C_line': {
            't_ref': kn_C,
            'points': {f'{t:.6g}': e for t, e in prof_kc.items()},
            'curvatures': curvature_series(prof_kc),
        },
    }

    # family placement on the v1 line
    placement = {}
    for p in ('R0_AUTO', 'C', 'B'):
        tp = alignment[p]['t_proj']
        e_line = interp_line(prof_v1, tp)
        placement[p] = {
            't_proj': tp,
            'E_line': e_line,
            'dE_line': (e_line - e0_v1) if e_line is not None else None,
            'dE_anchor': rot_info[p]['anchor_dE'],
        }

    ladder = ladder_rows([
        {'label': p, 'dE': rot_info[p]['anchor_dE'],
         't_proj': alignment[p]['t_proj']}
        for p in ('R0_AUTO', 'C', 'B')])

    report = {
        'schema_version': SAMPLE_SCHEMA,
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': utc_now(),
        'inputs': inputs,
        'cell': cell,
        'settings': settings,
        'energy_at_dm': energy_at_dm,
        'seeded_scf_family': seeded_family,
        'seeded_scf_R': scf_R,
        'seeded_scf_C': scf_C,
        'gradient_anchor_R': grad_anchor,
        'rotation_space': {
            'per_point': rot_info,
            'family_cos': family_cos,
            'alignment_with_v1': alignment,
        },
        'spectrum_R': spec_R_summary,
        'spectrum_C': spec_C_summary,
        'mode_overlap': mode_overlap,
        'profile': profile_out,
        'family_placement_v1': placement,
        'quadratic_ladder': ladder,
    }
    with open(out_json + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    os.replace(out_json + '.tmp', out_json)
    if out_md:
        with open(out_md + '.tmp', 'w', encoding='utf-8') as f:
            f.write(render_md(report))
        os.replace(out_md + '.tmp', out_md)
    return report


def top5_weights(kappa, eigenvectors):
    import numpy as np
    dots = np.asarray(eigenvectors.T @ kappa)
    nrm = np.linalg.norm(kappa)
    if nrm == 0:
        return []
    w = (dots / nrm) ** 2
    idx = np.argsort(w)[::-1][:5]
    return [{'mode': int(i), 'weight': float(w[i])} for i in idx]


def interp_line(profile, t):
    '''Energy on the scanned line at amplitude t: exact point when the
    grid carries it, else the linear interpolation of its neighbors
    (the grid is symmetric and dense around the family amplitudes).'''
    ts = sorted(profile)
    for x in ts:
        if abs(x - t) < 1e-9:
            return profile[x]
    for lo, hi in zip(ts, ts[1:]):
        if lo < t < hi:
            f = (t - lo) / (hi - lo)
            return profile[lo] + f * (profile[hi] - profile[lo])
    return None


def render_md(report):
    lines = [f"# Soft-mode analysis {report['amendment_id']}",
             f"generated {report['generated_at_utc']} (mechanical "
             "output; verdicts live in the session docs)", '']
    lines.append('## cross-implementation anchors')
    for p, r in report['energy_at_dm'].items():
        lines.append(f"- {p}: E_cpu(dm)={r['e_cpu']:.12f} "
                     f"dE_vs_gpu={r['diff']:+.3e}")
    ga = report.get('gradient_anchor_R')
    if ga:
        lines.append(f"- gradient anchor R: max|dg|={ga['max_abs_diff']:.3e}")
    lines.append('')
    lines.append('## family rotation vectors (R MO basis)')
    for p, r in report['rotation_space']['per_point'].items():
        lines.append(
            f"- {p}: |kappa|={r['kappa_norm']:.4f} "
            f"residual={r['purity_residual_fraction']:.3f} "
            f"dE_anchor={r['anchor_dE']:+.3e}")
    for pair, r in report['rotation_space']['family_cos'].items():
        lines.append(
            f"- cos {pair}: rotation={r['rotation_space']} "
            f"ao={r['ao_space']}")
    lines.append('')
    lines.append('## spectrum at R')
    s = report['spectrum_R']
    lines.append(f"- dim={s['dim']} softest-5="
                 f"{[round(x, 8) for x in s['softest_10'][:5]]}")
    lines.append(f"- separation |l2|/|l1|="
                 f"{s['separation_ratio_l2_over_abs_l1']}")
    lines.append(f"- asym={s['operator_asymmetry']:.2e} "
                 f"davidson-3={s['davidson_softest3']}")
    lines.append('')
    lines.append('## alignment with v1')
    for p, r in report['rotation_space']['alignment_with_v1'].items():
        lines.append(
            f"- {p}: cos={r['cos_with_v1']} w1={r['spectral_weight_v1']} "
            f"t_proj={r['t_proj']}")
    lines.append('')
    lines.append('## spectrum at C')
    s = report['spectrum_C']
    lines.append(f"- dim={s['dim']} softest-5="
                 f"{[round(x, 8) for x in s['softest_10'][:5]]}")
    lines.append(f"- mode overlap (AO cosine)="
                 f"{report['mode_overlap']['ao_density_perturbation_cos']}")
    lines.append('')
    lines.append('## profiles')
    for name, prof in report['profile'].items():
        lines.append(f"### {name} (t_ref={prof['t_ref']:.4f})")
        pts = prof['points']
        e0 = pts.get('0')
        for t in sorted(pts, key=float):
            lines.append(f"- t={t}: dE={pts[t] - e0:+.3e}")
        for t, lam in prof['curvatures']:
            lines.append(f"- curvature @t={t:.4f}: {lam:+.6e}")
    lines.append('')
    lines.append('## family placement on the v1 line')
    for p, r in report['family_placement_v1'].items():
        lines.append(f"- {p}: t_proj={r['t_proj']} "
                     f"dE_line={r['dE_line']} dE_anchor={r['dE_anchor']}")
    lines.append('')
    lines.append('## quadratic ladder (dE / t^2)')
    for row in report['quadratic_ladder']:
        lines.append(f"- {row['label']}: dE={row['dE']:+.3e} "
                     f"t={row['t_proj']} dE/t^2={row['dE_over_t2']}")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='LC-207 b3lyp soft-mode curvature analysis (CPU)')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--log', default=None)
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    log_path = args.log or os.path.join(
        os.path.dirname(os.path.abspath(args.out_json)),
        'softmode-analysis.log')
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lines = []

    def log(msg):
        line = f'[{utc_now()}] {msg}'
        print(line, flush=True)
        lines.append(line)

    log(f'plan {args.plan} sha256={sha256_file(args.plan)}')
    report = run(plan, args.out_json, args.out_md, log)
    log(f'wrote {args.out_json}')
    with open(log_path + '.tmp', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(log_path + '.tmp', log_path)
    del report
    return 0


if __name__ == '__main__':
    sys.exit(main())

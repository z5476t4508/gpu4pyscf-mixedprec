#!/usr/bin/env python3
"""LC-208 wb97mv multi-basin soft-mode analysis (CPU fp64 pyscf, NLC-inclusive).

Characterizes the methoxy/wb97m-v/def2-svp UKS multi-basin displaced-solution
family (batch uc-20261018-a FAIL rows B/C + the LC-206 R0_AUTO branch) at the
orbital-Hessian level with the NLC-INCLUSIVE internal-stability operator --
the LC-65 validated clone of gen_g_hop_uhf whose vind carries the VV10
response (pyscf.hessian.rks.get_vnlc_resp via gen_response(with_nlc=True)).
The LC-207 b3lyp tool used the stock newton_ah operator because b3lyp has no
NLC; wb97mv requires this operator, and LC-65 already measured its two
extremes on this very cell (fp64 branch = R: saddle, lambda_exact
-1.6692519e-5; auto branch = R0_AUTO: minimum, lambda_exact +1.2725266e-5,
both dense-exact 586-dim).  LC-208 re-derives those two within one
independent implementation pass and extends to the two remaining family
points the batch FAIL rows actually rest on (C = optimized_auto lane,
B = optimized_fp64 lane).

Family points (dms from the LC-206/LC-207/LC-208 basin-probe producers,
default-guess SCF trajectories; R = batch strict_fp64 reference):

  anchors   -- CPU energies at each family dm (direct evaluation AND
               seeded SCF on a fresh object per point) vs the frozen GPU
               lane anchor energies; CPU nuclear gradient of the R seeded
               state vs the batch R doc (implementation-agreement axis,
               NOT a precision gate)
  rotation  -- family displacement vectors kappa in R's canonical MO
               basis (pyscf (nvir,nocc) x1 layout), occ/virt block
               purity, pairwise collinearity (rotation space and AO space)
  spectra   -- dense exact eigh of the NLC-inclusive internal-stability
               Hessian at EVERY plan point (the stability() x2
               convention), resumable column-by-column with npz
               checkpoints; operator symmetry, hdiag cross-check, pyscf
               Davidson cross-check per point
  align     -- cos(kappa_i, v1) per displaced point + spectral weights +
               top-5 mode decomposition against R's full spectrum; the
               quadratic-model prediction dE_quad = kappa^T H_R kappa / 2
               vs the anchor dE (the multi-basin analogue of the LC-207
               single-valley quadratic ladder)
  profiles  -- E(t . v1), E(s . kappa_C), E(s . kappa_R0AUTO) scans with
               exactly antisymmetric +- pairs (odd terms cancel; LC-65
               discipline) and symmetric curvature fits

Everything here is mechanical CPU measurement (pyscf 2.14.0, the
independent implementation); verdicts live in the session docs.

numpy/pyscf/scipy are imported lazily inside the numeric paths so the
module stays importable stdlib-only for the unit tests (which cover plan
validation, checkpoint/segment bookkeeping, quadratic-prediction
arithmetic, and line-selection logic).
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import upstream_softmode_analysis as base     # sibling, stdlib-safe

SAMPLE_SCHEMA = 'softmode-nlc-v1'

REQUIRED_PLAN_KEYS = base.REQUIRED_PLAN_KEYS + (
    'spectra_points', 'spectra_ckpt_dir', 'profile_lines',
)
REQUIRED_SETTINGS_KEYS = base.REQUIRED_SETTINGS_KEYS + ('nlcgrids_level',)

SPECTRA_MUST_INCLUDE = 'R'
PROFILE_LINE_FORMS = ('v1',) + tuple(f'kappa_{p}' for p in
                                     base.FAMILY_POINTS if p != 'R')


def load_nlc_plan(path):
    '''base plan validation + the NLC-layer requirements (nlcgrids level,
    explicit frozen spectra points and profile lines).'''
    with open(path, encoding='utf-8') as f:
        plan = json.load(f)
    for key in base.REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise RuntimeError(f'softmode-nlc plan missing {key!r}')
    for key in REQUIRED_PLAN_KEYS:
        if key not in plan:
            raise RuntimeError(f'softmode-nlc plan missing {key!r}')
    for key in REQUIRED_SETTINGS_KEYS:
        if key not in plan['settings']:
            raise RuntimeError(f'plan settings missing {key!r}')
    pts = plan['spectra_points']
    if (not isinstance(pts, list) or not pts
            or len(set(pts)) != len(pts)):
        raise RuntimeError('spectra_points must be a non-empty unique list')
    for p in pts:
        if p not in base.FAMILY_POINTS:
            raise RuntimeError(f'spectra_points entry {p!r} is not a '
                               'family point')
    if SPECTRA_MUST_INCLUDE not in pts:
        raise RuntimeError(f'spectra_points must include '
                           f'{SPECTRA_MUST_INCLUDE!r}')
    lines = plan['profile_lines']
    if not isinstance(lines, list) or not lines:
        raise RuntimeError('profile_lines must be a non-empty list')
    for line in lines:
        if line not in PROFILE_LINE_FORMS:
            raise RuntimeError(
                f'profile line {line!r} is not one of {PROFILE_LINE_FORMS}')
    if 'v1' not in lines:
        raise RuntimeError('profile_lines must include the v1 line')
    return base.load_plan(path)


# ---------------------------------------------------------------------------
# checkpoint/segment bookkeeping (stdlib-only; unit-tested)
# ---------------------------------------------------------------------------

def columns_to_run(n_done, dim, segment=None):
    '''Column indices still to compute given n_done finished columns of a
    dim-column dense build; optional segment caps the count (bounded
    per-invocation work for ceiling survival).'''
    if n_done < 0 or dim < 0 or n_done > dim:
        raise RuntimeError(f'invalid build state n_done={n_done} dim={dim}')
    todo = list(range(n_done, dim))
    if segment is not None:
        if segment <= 0:
            raise RuntimeError('segment must be positive')
        todo = todo[:segment]
    return todo


def ckpt_due(i_finished, ckpt_every):
    '''True when a checkpoint should be written after i_finished columns
    (0-based count of finished columns; 0 never triggers).'''
    if ckpt_every <= 0:
        raise RuntimeError('ckpt_every must be positive')
    return i_finished > 0 and i_finished % ckpt_every == 0


# ---------------------------------------------------------------------------
# quadratic-model prediction (stdlib-only core; unit-tested)
# ---------------------------------------------------------------------------

def quad_prediction(evals, evec_rows, kappa):
    '''dE_quad = sum_i lambda_i * (kappa . v_i)^2 / 2 for a displacement
    kappa on a surface whose Hessian eigen-decomposition is (evals,
    evec_rows) with orthonormal rows v_i.  Pure python so the unit tests
    need no numpy.'''
    total = 0.0
    for lam, v in zip(evals, evec_rows):
        t = 0.0
        for k, x in zip(kappa, v):
            t += k * x
        total += lam * t * t
    return 0.5 * total


def ladder_rows_nlc(points):
    '''points: dicts with dE (vs R), t_proj, quad_pred, label.  Rows sorted
    by |t_proj| carrying dE/t^2 and the dE/quad ratio (None where the
    denominator vanishes).'''
    rows = []
    for p in sorted(points, key=lambda q: abs(q.get('t_proj') or 0.0)):
        t = p.get('t_proj')
        d = p.get('dE')
        q = p.get('quad_pred')
        ratio_t2 = (d / (t * t)) if (t and d is not None) else None
        ratio_q = (d / q) if (q and d is not None) else None
        rows.append({'label': p['label'], 'dE': d, 't_proj': t,
                     'quad_pred': q, 'dE_over_t2': ratio_t2,
                     'dE_over_quad': ratio_q})
    return rows


# ---------------------------------------------------------------------------
# numeric layer (numpy/pyscf imported lazily)
# ---------------------------------------------------------------------------

def build_mol_mf_nlc(cell, settings):
    '''UKS + DF + VV10 object with the batch/LC-65 matched settings; fails
    closed when the xc does not activate the NLC layer (pyscf activation:
    mf.nlc == '' means "determined by xc" via KohnShamDFT.do_nlc() --
    is_nlc(xc) True for wb97mv/wb97m-v; the NLC response, energy and
    nlcgrids all key on that, NOT on the mf.nlc string).'''
    from pyscf import gto, dft
    from pyscf.dft import libxc as pyscf_libxc
    mol = gto.M(atom=cell['geom'], basis=cell['basis'],
                charge=cell['charge'], spin=cell['spin'],
                unit='Angstrom', verbose=0)
    mf = dft.UKS(mol, xc=cell['xc']).density_fit(
        auxbasis=settings['auxbasis'])
    if not pyscf_libxc.is_nlc(cell['xc']):
        raise RuntimeError(
            f'xc {cell["xc"]!r} does not carry NLC (is_nlc=False); the '
            'NLC-inclusive operator requires a VV10 functional')
    if not mf.do_nlc():
        raise RuntimeError(
            f'mf.do_nlc() is False for xc {cell["xc"]!r} '
            f'(mf.nlc={mf.nlc!r}); the NLC layer must be active')
    mf.grids.level = settings['grids_level']
    mf.nlcgrids.level = settings['nlcgrids_level']
    mf.conv_tol = settings['conv_tol']
    mf.direct_scf_tol = settings['direct_scf_tol']
    mf.with_df.direct_scf_tol = settings['direct_scf_tol']
    mf.max_cycle = settings['max_cycle']
    mf.init_guess = settings['init_guess']
    return mol, mf


def gen_g_hop_uhf_nlc(mf, mo_coeff, mo_occ):
    '''LC-65 validated clone of gen_g_hop_uhf with the VV10/NLC response
    added to vind (pyscf.hessian.rks.get_vnlc_resp supports unrestricted
    density matrices).  Returns (g, h_op, h_diag, dims).'''
    from functools import reduce
    import numpy as np
    occidxa = np.where(mo_occ[0] > 0)[0]
    occidxb = np.where(mo_occ[1] > 0)[0]
    viridxa = np.where(mo_occ[0] == 0)[0]
    viridxb = np.where(mo_occ[1] == 0)[0]
    nocca, noccb = len(occidxa), len(occidxb)
    nvira, nvirb = len(viridxa), len(viridxb)
    orboa = mo_coeff[0][:, occidxa]
    orbob = mo_coeff[1][:, occidxb]
    orbva = mo_coeff[0][:, viridxa]
    orbvb = mo_coeff[1][:, viridxb]

    h1e = mf.get_hcore()
    dm0 = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = mf.get_fock(h1e, dm=dm0)
    focka = reduce(np.dot, (mo_coeff[0].conj().T, fock_ao[0], mo_coeff[0]))
    fockb = reduce(np.dot, (mo_coeff[1].conj().T, fock_ao[1], mo_coeff[1]))
    fooa = focka[occidxa[:, None], occidxa]
    fvva = focka[viridxa[:, None], viridxa]
    foob = fockb[occidxb[:, None], occidxb]
    fvvb = fockb[viridxb[:, None], viridxb]

    g = np.hstack((focka[viridxa[:, None], occidxa].ravel(),
                   fockb[viridxb[:, None], occidxb].ravel()))
    h_diaga = fvva.diagonal().real[:, None] - fooa.diagonal().real
    h_diagb = fvvb.diagonal().real[:, None] - foob.diagonal().real
    h_diag = np.hstack((h_diaga.reshape(-1), h_diagb.reshape(-1)))

    vind = mf.gen_response(mo_coeff, mo_occ, hermi=1, with_nlc=True)

    def h_op(x):
        x1a = x[:nvira * nocca].reshape(nvira, nocca)
        x1b = x[nvira * nocca:].reshape(nvirb, noccb)
        x2a = np.einsum('pr,rq->pq', fvva, x1a)
        x2a -= np.einsum('sq,ps->pq', fooa, x1a)
        x2b = np.einsum('pr,rq->pq', fvvb, x1b)
        x2b -= np.einsum('sq,ps->pq', foob, x1b)

        d1a = reduce(np.dot, (orbva, x1a, orboa.conj().T))
        d1b = reduce(np.dot, (orbvb, x1b, orbob.conj().T))
        dm1 = np.array((d1a + d1a.conj().T, d1b + d1b.conj().T))
        v1 = vind(dm1)
        x2a += reduce(np.dot, (orbva.conj().T, v1[0], orboa))
        x2b += reduce(np.dot, (orbvb.conj().T, v1[1], orbob))
        return np.hstack((x2a.ravel(), x2b.ravel()))

    return g, h_op, h_diag, (nvira, nocca, nvirb, noccb)


def save_dense_ckpt(path, H, n_done):
    import numpy as np
    with open(path + '.tmp', 'wb') as f:
        np.savez(f, H=H, n_done=np.int64(n_done))
    os.replace(path + '.tmp', path)


def dense_stability_nlc(mf, log, ckpt=None, ckpt_every=100,
                        davidson=True):
    '''Dense exact eigh of the NLC-inclusive internal-stability Hessian at
    the mf state (stability() convention: H = 2 * h_op), column build
    resumable via the npz checkpoint.'''
    import numpy as np
    from pyscf import lib as pyscf_lib

    g, hop, hdiag, dims = gen_g_hop_uhf_nlc(mf, mf.mo_coeff, mf.mo_occ)
    hdiag = hdiag * 2
    dim = int(g.size)

    H = None
    n_done = 0
    if ckpt and os.path.isfile(ckpt):
        z = np.load(ckpt)
        H = np.asarray(z['H'], dtype=float)
        n_done = int(z['n_done'])
        if H.shape != (dim, dim) or not 0 <= n_done <= dim:
            raise RuntimeError(f'checkpoint {ckpt} does not match the '
                               f'operator dimension {dim}')
        log(f'dense resume from {ckpt}: {n_done}/{dim} columns')
    if H is None:
        H = np.zeros((dim, dim))
        n_done = 0

    t0 = time.time()
    for i in columns_to_run(n_done, dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H[:, i] = hop(e).real * 2
        if ckpt and ckpt_due(i + 1, ckpt_every):
            save_dense_ckpt(ckpt, H, i + 1)
            log(f'  dense {i + 1}/{dim} columns '
                f'({time.time() - t0:.0f}s since resume)')
    if ckpt and n_done < dim:
        save_dense_ckpt(ckpt, H, dim)
    wall = time.time() - t0      # this invocation's column work only

    asym = float(np.abs(H - H.T).max())
    w, v = np.linalg.eigh(H)
    r1 = float(np.linalg.norm(H @ v[:, 0] - w[0] * v[:, 0]))
    hdiag_dev = float(np.abs(np.diag(H) - hdiag).max())

    e_dav = None
    if davidson:
        def hop2(x):
            return hop(x).real * 2

        def precond(dx, e, x0):
            hd = hdiag - e
            hd[abs(hd) < 1e-8] = 1e-8
            return dx / hd

        x0 = np.zeros_like(g)
        x0[g != 0] = 1.0 / hdiag[g != 0]
        e_dav, _ = pyscf_lib.davidson(hop2, x0, precond, tol=1e-9,
                                      nroots=3, verbose=0)
        e_dav = np.sort(np.asarray(e_dav).ravel())[:3]

    log(f'dense NLC stability: dim={dim} '
        f'asym={asym:.2e} softest-5='
        f'{np.array2string(w[:5], precision=8)} '
        f'davidson-3={np.array2string(e_dav, precision=8)}')
    return {
        'dim': dim,
        'wall_s': wall,
        'build_resumed_from': n_done,
        'operator_asymmetry': asym,
        'orbital_gradient_norm': float(np.linalg.norm(g)),
        'eigenvalues': w,
        'eigenvectors': v,
        'softest_residual_norm': r1,
        'hdiag_vs_dense_dev': hdiag_dev,
        'davidson_softest3': e_dav.tolist() if e_dav is not None else None,
    }


def spectrum_summary(spec):
    import numpy as np
    w = np.asarray(spec['eigenvalues'])
    return {
        'dim': spec['dim'],
        'softest_10': w[:10].tolist(),
        'hardest': float(w[-1]),
        'separation_ratio_l2_over_abs_l1':
            float(abs(w[1]) / abs(w[0])) if w[0] != 0 else None,
        'operator_asymmetry': spec['operator_asymmetry'],
        'softest_residual_norm': spec['softest_residual_norm'],
        'hdiag_vs_dense_dev': spec['hdiag_vs_dense_dev'],
        'davidson_softest3': spec['davidson_softest3'],
        'orbital_gradient_norm': spec['orbital_gradient_norm'],
        'build_resumed_from': spec['build_resumed_from'],
        'wall_s_this_invocation': spec['wall_s'],
    }


def quad_prediction_np(H, kappa):
    '''numpy quadratic-model dE = kappa^T H kappa / 2 (cross-checks the
    eigen-decomposed quad_prediction).'''
    import numpy as np
    kappa = np.asarray(kappa)
    return float(0.5 * kappa @ (np.asarray(H) @ kappa))


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def run(plan, out_json, out_md, log):
    import numpy as np

    base.verify_inputs(plan)
    cell, settings = plan['cell'], plan['settings']
    ckpt_dir = plan['spectra_ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    mol, mf = build_mol_mf_nlc(cell, settings)
    log(f'cell {cell["tag"]}: nao={mol.nao_nr()} nelec={mol.nelec} '
        f'spin={cell["spin"]} nlc={getattr(mf, "nlc", None)} '
        f'nlcgrids.level={settings["nlcgrids_level"]}')

    dms = {p: base.dm_from_npz(plan['family_dms'][p]['path'])
           for p in base.FAMILY_POINTS}

    def as_ukp_spin2(x):
        if isinstance(x, tuple) and len(x) == 2:
            return x
        if isinstance(x, np.ndarray) and x.ndim == 3 and x.shape[0] == 2:
            return x
        return None

    for p in base.FAMILY_POINTS:
        if as_ukp_spin2(dms[p]) is None:
            raise RuntimeError(
                f'family point {p} density is not a 2-spin UKS dm: '
                f'{type(dms[p])} {getattr(dms[p], "shape", None)}')
    inputs = {p: {
        'path': plan['family_dms'][p]['path'],
        'sha256': base.sha256_file(plan['family_dms'][p]['path']),
        'anchor_e': plan['family_dms'][p]['anchor_e'],
        'anchor_source': plan['family_dms'][p]['anchor_source'],
    } for p in base.FAMILY_POINTS}

    # cross-implementation anchors: direct energy evaluation at each dm
    energy_at_dm = {}
    for p in base.FAMILY_POINTS:
        e = base.energy_at(mf, dms[p])
        energy_at_dm[p] = {
            'e_cpu': e,
            'anchor_e': inputs[p]['anchor_e'],
            'diff': e - inputs[p]['anchor_e'],
        }
        log(f'anchor {p}: E_cpu(dm)={e:.12f} '
            f'dE_vs_gpu={e - inputs[p]["anchor_e"]:+.3e}')

    # every family point gets its own fresh seeded SCF (stationarity; the
    # seeds must stay in place) and keeps its object for its spectrum
    states = {'R': mf}
    seeded_family = {}
    for p in base.FAMILY_POINTS:
        mf_p = mf if p == 'R' else build_mol_mf_nlc(cell, settings)[1]
        states[p] = mf_p
        seeded_family[p] = base.seeded_scf(mf_p, dms[p], p, log)

    # CPU nuclear gradient of the R seeded state vs the batch R doc
    g_R = base.gradient_of_state(states['R'])
    grad_anchor_path = plan['family_dms']['R'].get('anchor_gradient_path')
    grad_anchor = None
    if grad_anchor_path:
        with open(grad_anchor_path, encoding='utf-8') as f:
            doc = json.load(f)
        g_ref = np.asarray(doc['rows'][0]['result']['gradient'], dtype=float)
        grad_anchor = {
            'path': grad_anchor_path,
            'sha256': base.sha256_file(grad_anchor_path),
            'max_abs_diff': float(np.abs(g_R - g_ref).max()),
            'cpu_grad_norm': float(np.linalg.norm(g_R)),
        }
        log(f'gradient anchor R: max|dg| vs batch R doc = '
            f'{grad_anchor["max_abs_diff"]:.3e}')

    # rotation space: kappa of each displaced point in R's MO basis
    mo, mo_occ = mf.mo_coeff, mf.mo_occ
    s = mol.intor_symmetric('int1e_ovlp')
    dm_R = dms['R']
    kappas = {}
    rot_info = {}
    for p in ('R0_AUTO', 'C', 'B'):
        dd = [np.asarray(dms[p][k]) - np.asarray(dm_R[k]) for k in range(2)]
        dm_mo = [base.mo_space_dm(dd[k], mo[k], s) for k in range(2)]
        occ_a, virt_a = base.occ_virt(mo_occ[0])
        occ_b, virt_b = base.occ_virt(mo_occ[1])
        ka, ra = base.extract_rotation(dm_mo[0], occ_a, virt_a)
        kb, rb = base.extract_rotation(dm_mo[1], occ_b, virt_b)
        kappas[p] = base.flatten_x(ka, kb)
        rot_info[p] = {
            'kappa_norm': float(np.linalg.norm(kappas[p])),
            'purity_residual_fraction': (ra + rb) / 2.0,
            'ao_dm_norm': float(sum(np.linalg.norm(x) for x in dd)),
            'anchor_dE': inputs[p]['anchor_e'] - inputs['R']['anchor_e'],
        }
        log(f'rotation {p}: |kappa|={rot_info[p]["kappa_norm"]:.4f} '
            f'residual={rot_info[p]["purity_residual_fraction"]:.3f} '
            f'dE_anchor={rot_info[p]["anchor_dE"]:+.3e}')

    def ao_dir(p):
        return tuple(np.asarray(dms[p][k]) - np.asarray(dm_R[k])
                     for k in range(2))

    family_cos = {}
    pairs = (('B', 'C'), ('B', 'R0_AUTO'), ('C', 'R0_AUTO'))
    for a, b in pairs:
        family_cos[f'{a}_{b}'] = {
            'rotation_space': base.cosine(kappas[a], kappas[b]),
            'ao_space': base.ao_cosine(ao_dir(a), ao_dir(b)),
        }

    # dense NLC-inclusive spectra at every plan point
    spectra = {}
    spectra_summaries = {}
    for p in plan['spectra_points']:
        ck = os.path.join(ckpt_dir, f'dense-H-{p}.npz')
        spectra[p] = dense_stability_nlc(states[p], log, ckpt=ck)
        spectra_summaries[p] = spectrum_summary(spectra[p])
        log(f'spectrum {p}: lambda1='
            f'{spectra_summaries[p]["softest_10"][0]:+.6e}')

    # mode overlaps: R's softest vs each other spectrum point's softest
    mode_overlap = {}
    for p in plan['spectra_points']:
        if p == 'R':
            continue
        mode_overlap[f'R_{p}'] = {
            'ao_density_perturbation_cos': base.mode_overlaps(
                spectra['R'], spectra[p], states['R'], states[p]),
            'lambda1_R': float(spectra['R']['eigenvalues'][0]),
            'lambda1_' + p: float(spectra[p]['eigenvalues'][0]),
        }

    # alignment of the family displacements with R's modes + the
    # quadratic-model prediction on R's full spectrum
    w_r = np.asarray(spectra['R']['eigenvalues'])
    v_r = np.asarray(spectra['R']['eigenvectors'])
    H_R = v_r @ np.diag(w_r) @ v_r.T
    v1 = v_r[:, 0]
    alignment = {}
    for p in ('R0_AUTO', 'C', 'B'):
        c = base.cosine(kappas[p], v1)
        quad_eig = quad_prediction(
            w_r.tolist(), [v_r[:, i].tolist() for i in range(w_r.size)],
            kappas[p].tolist())
        quad_mat = quad_prediction_np(H_R, kappas[p])
        alignment[p] = {
            'cos_with_v1': c,
            'spectral_weight_v1': base.spectral_weight(c)
            if c is not None else None,
            't_proj': float(kappas[p] @ v1),
            'top5_weights': base.top5_weights(kappas[p], v_r),
            'quad_pred_eig': quad_eig,
            'quad_pred_matrix': quad_mat,
            'quad_pred_consistency': (quad_eig - quad_mat)
            if (quad_eig is not None and quad_mat is not None) else None,
        }
        log(f'alignment {p}: cos_v1={c} w1='
            f'{alignment[p]["spectral_weight_v1"]} '
            f'quad_pred={quad_mat:+.3e} vs dE_anchor='
            f'{rot_info[p]["anchor_dE"]:+.3e}')

    # profile lines
    fracs = plan['profile']['fracs']
    profiles = {}
    t_proj_max = max((abs(alignment[p]['t_proj'])
                      for p in ('R0_AUTO', 'C', 'B')
                      if alignment[p]['t_proj']), default=0.0)
    for line in plan['profile_lines']:
        if line == 'v1':
            t_ref = t_proj_max or max(rot_info[p]['kappa_norm']
                                      for p in ('R0_AUTO', 'C', 'B'))
            x1a, x1b = base.split_x(v1, mo_occ[0], mo_occ[1])
            src = 'max family |t_proj| on v1 (fallback max kappa norm)'
        else:
            p = line[len('kappa_'):]
            kn = rot_info[p]['kappa_norm']
            if kn == 0.0:
                raise RuntimeError(
                    f'profile line {line!r} requested but kappa_{p} '
                    'vanishes; the line direction is undefined')
            t_ref = kn
            x1a, x1b = base.split_x(kappas[p] / kn, mo_occ[0], mo_occ[1])
            src = f'|kappa_{p}|'
        prof = base.profile_scan(mf, mo, mo_occ, x1a, x1b, t_ref, fracs,
                                 log)
        profiles[line] = {
            't_ref': t_ref,
            't_ref_source': src,
            'points': {f'{t:.6g}': e for t, e in prof.items()},
            'curvatures': base.curvature_series(prof),
        }

    # family placement on the v1 line + the extended ladder
    prof_v1 = {float(k): v for k, v in profiles['v1']['points'].items()}
    e0_v1 = prof_v1[0.0]
    placement = {}
    ladder_points = []
    for p in ('R0_AUTO', 'C', 'B'):
        tp = alignment[p]['t_proj']
        e_line = base.interp_line(prof_v1, tp)
        placement[p] = {
            't_proj': tp,
            'E_line': e_line,
            'dE_line': (e_line - e0_v1) if e_line is not None else None,
            'dE_anchor': rot_info[p]['anchor_dE'],
        }
        ladder_points.append({
            'label': p, 'dE': rot_info[p]['anchor_dE'], 't_proj': tp,
            'quad_pred': alignment[p]['quad_pred_matrix'],
        })
    ladder = ladder_rows_nlc(ladder_points)

    report = {
        'schema_version': SAMPLE_SCHEMA,
        'unit': plan['unit'],
        'amendment_id': plan['amendment_id'],
        'batch_id': plan['batch_id'],
        'generated_at_utc': base.utc_now(),
        'operator': 'lc65-gen_g_hop_uhf_nlc-clone (with_nlc=True, x2 '
                    'stability convention)',
        'inputs': inputs,
        'cell': cell,
        'settings': settings,
        'spectra_points': plan['spectra_points'],
        'energy_at_dm': energy_at_dm,
        'seeded_scf_family': seeded_family,
        'gradient_anchor_R': grad_anchor,
        'rotation_space': {
            'per_point': rot_info,
            'family_cos': family_cos,
            'alignment_with_v1': alignment,
        },
        'spectra': spectra_summaries,
        'mode_overlap': mode_overlap,
        'profile': profiles,
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


def render_md(report):
    lines = [f"# Softmode-NLC analysis {report['amendment_id']}",
             f"generated {report['generated_at_utc']} (mechanical "
             "output; verdicts live in the session docs)",
             f"operator: {report['operator']}", '']
    lines.append('## cross-implementation anchors')
    for p, r in report['energy_at_dm'].items():
        lines.append(f"- {p}: E_cpu(dm)={r['e_cpu']:.12f} "
                     f"dE_vs_gpu={r['diff']:+.3e}")
    for p, r in report['seeded_scf_family'].items():
        lines.append(f"- seeded {p}: E={r['e_tot']:.12f} conv="
                     f"{r['converged']} cyc={r['cycles']}")
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
    lines.append('## NLC-inclusive spectra (dense exact eigh)')
    for p, s in report['spectra'].items():
        lines.append(f"### {p}")
        lines.append(f"- dim={s['dim']} softest-5="
                     f"{[round(x, 8) for x in s['softest_10'][:5]]}")
        lines.append(f"- separation |l2|/|l1|="
                     f"{s['separation_ratio_l2_over_abs_l1']}")
        lines.append(f"- asym={s['operator_asymmetry']:.2e} "
                     f"davidson-3={s['davidson_softest3']}")
    lines.append('')
    lines.append('## alignment with R softest mode + quadratic model')
    for p, r in report['rotation_space']['alignment_with_v1'].items():
        lines.append(
            f"- {p}: cos={r['cos_with_v1']} w1={r['spectral_weight_v1']} "
            f"t_proj={r['t_proj']} quad_pred={r['quad_pred_matrix']:+.3e} "
            f"(eig-consistency {r['quad_pred_consistency']:+.1e})")
    for pair, r in report['mode_overlap'].items():
        lines.append(f"- mode overlap {pair}: "
                     f"{r['ao_density_perturbation_cos']}")
    lines.append('')
    lines.append('## profiles')
    for name, prof in report['profile'].items():
        lines.append(f"### {name} (t_ref={prof['t_ref']:.4f}, "
                     f"source {prof['t_ref_source']})")
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
    lines.append('## quadratic ladder (dE vs quad model)')
    for row in report['quadratic_ladder']:
        lines.append(f"- {row['label']}: dE={row['dE']:+.3e} "
                     f"t={row['t_proj']} dE/t^2={row['dE_over_t2']} "
                     f"quad={row['quad_pred']:+.3e} "
                     f"dE/quad={row['dE_over_quad']}")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='LC-208 wb97mv multi-basin soft-mode analysis '
                    '(CPU, NLC-inclusive operator)')
    ap.add_argument('--plan', required=True)
    ap.add_argument('--out-json', required=True)
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--log', default=None)
    ap.add_argument('--ckpt-every', type=int, default=100)
    args = ap.parse_args(argv)

    plan = load_nlc_plan(args.plan)
    log_path = args.log or os.path.join(
        os.path.dirname(os.path.abspath(args.out_json)),
        'softmode-nlc.log')
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    lines = []

    def log(msg):
        line = f'[{base.utc_now()}] {msg}'
        print(line, flush=True)
        lines.append(line)

    log(f'plan {args.plan} sha256={base.sha256_file(args.plan)}')
    report = run(plan, args.out_json, args.out_md, log)
    log(f'wrote {args.out_json}')
    with open(log_path + '.tmp', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    os.replace(log_path + '.tmp', log_path)
    del report
    return 0


if __name__ == '__main__':
    sys.exit(main())

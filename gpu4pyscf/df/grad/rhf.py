# Copyright 2021-2025 The PySCF Developers. All Rights Reserved.
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

import ctypes
import os
import numpy as np
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import precision
from gpu4pyscf.lib.cupy_helper import (
    contract, asarray, ndarray, cholesky, eigh, transpose_sum, get_avail_mem)
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc,
    int2c2e_ip1_per_atom, int3c2e_scheme,
    _taskgran_pair_batch, _taskgran_aux16_offsets,
    _taskgran_resplit_active, _taskgran_resplit_splits,
    _molecular_int3c2e_opt,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e)
from gpu4pyscf.df import df
from gpu4pyscf.df.df_jk import factorize_dm, _DFHF
from gpu4pyscf.gto.mole import SortedMole

__all__ = ['Gradients']

# LC-42: opt-in fused launch of the sum_ejk_int3c2e_ip1 kernel on the J+K
# (hybrid-functional) gradient path.  The default path serializes one
# kernel launch per aux l-batch (grid (n_shell_pair_blocks, 1), 1 block/SM
# at ~95 KB shared memory -> 16-28% SM occupancy on the 128-SM RTX 4090;
# measured 8-12 sequential launches per gradient, 17-40% of the fp64
# gradient phase at 43-222 AO).  The kernel's launcher has always accepted
# a 2D grid; the J-only path (_j_energy_per_atom) has used it in
# production since upstream.  Fusing the existing batch boundaries into
# one launch does not change any block's arithmetic or memory addressing:
# each y-block computes exactly what its per-batch launch computed (same
# shl_pair x, same ksh_offsets slice relative to its own start, same
# compressed columns, same atomicAdd order per block), so the result
# differs from the per-batch path only by the atomic interleaving order
# across blocks (the same nondeterminism class as any occupancy change;
# the per-batch path itself is not deterministic across runs at the last
# bits).  Sub-batch y-chunking was rejected by static analysis: the
# exchange branch indexes the compressed buffer as
# dm[pair_offset*naux + k0 + k*nksh] with nksh = ksh1-ksh0, i.e. the
# column layout of the batch is interleaved by y-block shell count --
# changing block boundaries inside a batch would read wrong columns.
_IP1_FUSED = None   # cached env value; None = not yet read

def _ip1_fused():
    '''Whether to fuse the per-batch sum_ejk_int3c2e_ip1 launches (opt-in).

    GPU4PYSCF_DF_IP1_FUSED=1 enables the fused launch when the full-width
    compressed buffer fits the memory guard; anything else keeps the
    upstream per-batch path byte-for-byte.
    '''
    global _IP1_FUSED
    if _IP1_FUSED is None:
        raw = os.environ.get('GPU4PYSCF_DF_IP1_FUSED', '')
        if raw == '1':
            _IP1_FUSED = True
        elif raw:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_IP1_FUSED={raw!r} ignored; '
                          'expected 1 to enable the fused ip1 launch')
            _IP1_FUSED = False
        else:
            _IP1_FUSED = False
    return _IP1_FUSED

_IP1_F32 = None     # cached env value; None = not yet read

def _ip1_f32():
    '''Whether to run the ip1 integral kernel in float32 on the fp64 lane
    (opt-in precision-policy prototype, LC-86).

    GPU4PYSCF_DF_IP1_F32=1 selects sum_ejk_int3c2e_ip1_f32 for the DF
    gradient ip1 kernel regardless of the global precision mode.  The
    existing fp32 discipline is preserved exactly: only the integral
    kernel and its pseudo-DM/dm+auxvec inputs go float32; the j3c@dm
    contractions, the j2c metric solve and the atomic ejk/ejk_aux
    accumulation targets stay float64 (the metric solve amplifies
    float32 relative error by the auxiliary-basis condition number
    straight into the gradient).  On lanes that already run the gradient
    in fp32 (the resolved 'auto' policy) this is a no-op.
    '''
    global _IP1_F32
    if _IP1_F32 is None:
        raw = os.environ.get('GPU4PYSCF_DF_IP1_F32', '')
        if raw == '1':
            _IP1_F32 = True
        elif raw:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_IP1_F32={raw!r} ignored; '
                          'expected 1 to enable the fp32 ip1 kernel')
            _IP1_F32 = False
        else:
            _IP1_F32 = False
    return _IP1_F32

_RESP_F32 = None     # cached env value; None = not yet read
_RESP_F32_CALLS = 0  # armed response-f32 grad engagements (evidence counter)

def _resp_f32():
    '''Whether to run the phase-2 response contractions of the DF-gradient
    J+K pseudo-DM build in float32 (opt-in precision-policy prototype,
    DFJK-RESPGEMM-IMPL-01; BENCH-03 V1 basis, entry 368: 19.92x/28.83x
    per-chain at the 819/1042 cells, operand numerics 4.6e-7/5.3e-7).

    GPU4PYSCF_DFGRAD_RESP_F32=1 casts the response-chain operands (dm_oo,
    dm_factor_l/r) to float32 ONCE per gradient call and runs the two
    phase-2 contractions -- the transform and back-transform feeding the
    ip1 pseudo-DM tensor -- in float32 buffers.  Phase-1 (the j3c@dm_factor
    loop), the j2c metric solve, the J-part elementwise build and the
    auxbasis-response contraction keep the deployed float64 discipline
    verbatim (see the amplification note at the top of _jk_energy_per_atom;
    the metric solve amplifies float32 relative error by the
    auxiliary-basis condition number straight into the gradient).  dm_oo
    itself stays float64 for the auxresp contraction; only the fp32 twin
    is added.  On the float64 kernel lane the fp32 pseudo-DM tensor is
    reassembled to float64 before the pair compression, so the compressed
    buffer and the ip1 kernel keep their exact dtypes; on the fp32 kernel
    lane (global fp32 mode or GPU4PYSCF_DF_IP1_F32=1) the fp32 dm_tensor
    feeds the kernel directly and the redundant cast is skipped.
    Default OFF.  The strict fp64-reference context
    (numint.strict_fp64_reference_ctx) forces the knob inert.
    '''
    global _RESP_F32
    if _RESP_F32 is None:
        raw = os.environ.get('GPU4PYSCF_DFGRAD_RESP_F32', '')
        if raw == '1':
            _RESP_F32 = True
        elif raw in ('', '0'):
            _RESP_F32 = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DFGRAD_RESP_F32={raw!r} ignored; '
                          'expected 1 to enable the fp32 response chain '
                          'or 0 (fp64 everywhere, the default)')
            _RESP_F32 = False
    return _RESP_F32

def _strict_fp64_reference_active():
    '''Whether the current call runs under the strict fp64-reference
    context (dft.numint.strict_fp64_reference_ctx, armed by
    stability_verify around its exact-fp64 sibling trajectories).  The
    response-f32 knob stays inert there: the sibling must reproduce the
    plain DEFAULT-lane solution.  Lazy import so df.grad does not pay
    the dft import at module load; attribute read at call time (the
    flag flips inside the context manager).'''
    from gpu4pyscf.dft import numint as _numint
    return bool(_numint._strict_fp64_reference_active)

def _arm_resp_f32(dm_oo, dm_factor_l, dm_factor_r):
    '''DFJK-RESPGEMM-IMPL-01 armed-path operand staging, shared by the
    RHF and UHF gradient twins.  Bumps the engagement counter and returns
    the one-time fp32 operand casts (l/r aliased when the factor arrays
    are the same object, mirroring _factorize_dm's hermi=1 aliasing).
    Returns (None, None, None) when the knob is off or the strict
    fp64-reference context is active -- the caller's resp32 flag is
    simply ``dm_oo32 is not None``.'''
    if not (_resp_f32() and not _strict_fp64_reference_active()):
        return None, None, None
    global _RESP_F32_CALLS
    _RESP_F32_CALLS += 1
    dm_factor_l32 = dm_factor_l.astype(cp.float32)
    dm_factor_r32 = (dm_factor_l32 if dm_factor_l is dm_factor_r
                     else dm_factor_r.astype(cp.float32))
    dm_oo32 = dm_oo.astype(cp.float32)
    return dm_oo32, dm_factor_l32, dm_factor_r32

_AUXRESP_F32 = None     # cached env value; None = not yet read
_AUXRESP_F32_GRADS = 0  # armed auxresp-f32 grad engagements (evidence counter)
_AUXRESP_F32_STATS = {}  # realized armed/naux state of the last call either lane

def _auxresp_f32():
    '''Whether to quantize the j-only auxbasis-response operand to the
    float32 lane (opt-in harmonization prototype, GRADAUX-F32-CANCEL-01).

    GPU4PYSCF_DF_GRAD_AUXRESP_F32=1 routes the 2-center auxbasis response
    of the j-only gradient (_j_energy_per_atom) through the same float32
    quantization the f32 ip1 J kernel applies to its operands: dm_aux is
    rebuilt as the float32 outer product of the float32-cast auxvec (the
    same one-time cast sum_ejk_int3c2e_ip1_f32 consumes on its armed
    lane) and reassembled to float64 for the deployed e_int2c2e_ip1
    kernel, whose arithmetic stays float64 verbatim.
    M5-DRIFT-ATTRIB-01 measured that on the D1-ON f32 ip1 lane the fp64
    2-center response decorrelates from the f32-quantized ip1 aux
    accumulation and breaks the elementwise aux-response cancellation
    behind the ~1e-6 Eh/Bohr k-repeat gradient drift; this knob is the
    harmonization candidate for that mechanism.  The only arithmetic
    delta is the float32 rounding of each dm_aux entry (one rounding of
    the c_P*c_Q product of float32-quantized coefficients); the
    ej/ej_aux accumulation targets and the 2c kernel arithmetic stay
    float64.  Intended use is the GPU4PYSCF_DF_IP1_F32 (or global fp32)
    lane; the knob is independent of that selection so the paired
    experiment can arm it alone.  Default OFF.  The strict
    fp64-reference context (numint.strict_fp64_reference_ctx) forces
    the knob inert.
    '''
    global _AUXRESP_F32
    if _AUXRESP_F32 is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_AUXRESP_F32', '')
        if raw == '1':
            _AUXRESP_F32 = True
        elif raw:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_AUXRESP_F32={raw!r} ignored; '
                          'expected 1 to enable the fp32 aux-response '
                          'quantization')
            _AUXRESP_F32 = False
        else:
            _AUXRESP_F32 = False
    return _AUXRESP_F32

def _arm_auxresp_f32(auxvec, dm_aux):
    '''GRADAUX-F32-CANCEL-01 armed-path operand staging of the j-only
    auxbasis response, in _j_energy_per_atom (the single J-only site --
    the UKS k_factor==0 route delegates here, so the RKS and UKS lanes
    share it verbatim).  Returns the float32-quantized dm_aux twin --
    the float32 outer product of the float32-cast auxvec, reassembled
    to float64 for the deployed e_int2c2e_ip1 kernel -- or the input
    dm_aux verbatim when the knob is off or the strict fp64-reference
    context is active.  Bumps the engagement counter and records the
    realized armed/naux state of the last call either lane (the
    home-module pattern; no uhf.py twin exists for this site).'''
    armed = _auxresp_f32() and not _strict_fp64_reference_active()
    global _AUXRESP_F32_GRADS
    if armed:
        _AUXRESP_F32_GRADS += 1
        auxvec32 = cp.asarray(auxvec, dtype=cp.float32)
        dm_aux = (auxvec32[:,None] * auxvec32).astype(cp.float64)
    _AUXRESP_F32_STATS.update(armed=bool(armed), naux=len(auxvec))
    return dm_aux

_JONLY_AUXFP64 = None    # cached env value; None = not yet read
_JONLY_AUXFP64_GRADS = 0 # armed j-only auxfp64 grad engagements (evidence counter)
_JONLY_AUXFP64_STATS = {}  # realized armed/site state of the last call either lane

def _jonly_auxfp64():
    '''Whether to keep the deployed fp64 ip1 kernel at the J-only
    gradient site (opt-in precision-scope knob, D1-GRAD-JONLY-AUXFP64-01).

    GPU4PYSCF_DF_GRAD_JONLY_AUXFP64=1 keeps _j_energy_per_atom -- the
    single J-only site in df/grad/rhf.py, shared verbatim by the RKS and
    UKS k_factor==0 routes -- on the deployed fp64 sum_ejk_int3c2e_ip1
    kernel with its fp64 dm/auxvec operands, even when
    GPU4PYSCF_DF_IP1_F32 (the D1-ON lane) would have selected the f32
    twin there.  M5-DRIFT-ATTRIB-01 (entry 645) measured that at the
    tamoxifen r2scan3c UKS jonly 687 cell the f32 twin's aux accumulation
    decorrelates from the fp64 2-center response and breaks the
    elementwise aux-response cancellation behind the ~1e-6 Eh/Bohr
    k-repeat gradient drift (the fp64 R lane passes via correlated
    cancellation, total ~1e-12), and GRADAUX-F32-CANCEL-01 (entry 646)
    measured that operand-level harmonization does NOT restore it --
    the decorrelation sits in the kernels' own accumulation behavior.
    This knob scopes the FP32/FP64 division per site: the
    determinism-sensitive j-only aux sub-chain stays fp64 (the deployed
    verbatim path; bitwise the ip1-f32-off lane), the rest of the
    composition keeps its measured configuration.  The only delta is the
    reverted kernel selection and operand casts at this one site.  The
    knob never fires on the global fp32 ('auto') lane and stays inert
    under the strict fp64-reference context (numint.
    strict_fp64_reference_ctx).  Default OFF.
    '''
    global _JONLY_AUXFP64
    if _JONLY_AUXFP64 is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_JONLY_AUXFP64', '')
        if raw == '1':
            _JONLY_AUXFP64 = True
        elif raw in ('', '0'):
            _JONLY_AUXFP64 = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_JONLY_AUXFP64={raw!r} ignored; '
                          'expected 1 to keep the fp64 ip1 kernel at the '
                          'J-only gradient site or 0 (off, the default)')
            _JONLY_AUXFP64 = False
    return _JONLY_AUXFP64

def _arm_jonly_auxfp64(mol, naux):
    '''D1-GRAD-JONLY-AUXFP64-01 armed-path decision, called from
    _j_energy_per_atom on EVERY pass through the site (right after its
    fp32 kernel-selection line), so the realized stats are recorded on
    the plain-fp64 lanes too where the knob is inert by construction.
    Returns True when the knob flips this site back to the deployed
    fp64 ip1 kernel (fp32 was True only via the opt-in GPU4PYSCF_DF_IP1_F32
    prototype, precision is not the global fp32 mode, and the strict
    fp64-reference context is inactive); the caller then keeps its fp64
    branch verbatim.  Bumps the engagement counter and records the
    realized armed/natm/naux state of the last call either lane
    (the home-module pattern; no uhf.py twin exists for this site --
    the UKS k_factor==0 route delegates here).'''
    armed = (precision.get_precision() != 'fp32'
             and _ip1_f32() and _jonly_auxfp64()
             and not _strict_fp64_reference_active())
    global _JONLY_AUXFP64_GRADS
    if armed:
        _JONLY_AUXFP64_GRADS += 1
    _JONLY_AUXFP64_STATS.update(
        armed=bool(armed), natm=mol.natm, naux=int(naux),
        site_fp32=bool(precision.get_precision() == 'fp32'))
    return armed

_JONLY_AUXVEC = None   # cached env value; None = not yet read
_JONLY_AUXVEC_GRADS = 0    # armed direct-route grad engagements (counter)
_JONLY_AUXVEC_STATS = {}   # realized armed/pack state of the last pass

def _jonly_auxvec():
    '''Whether to compute the J-only fall-through auxvec directly from the
    retained cderi pack (opt-in, GRAD-JONLY-AUXVEC-DIRECT-01).

    GPU4PYSCF_DF_GRAD_JONLY_AUXVEC=1 replaces, at the single RKS
    closed-shell k_factor==0 fall-through of _jk_energy_per_atom (the
    GRAD-J3C-REUSE-JONLY-01 armed route, pack retained), the pack
    construction -- the X = j3c@dm-factor GEMM pair + symmetric scatter
    (the e636 arm.grad.jk.p1 band, 0.3017 s at the 687 cell) and the
    dm_oo = aux_coef@X tail einsum (the resid.gemm band, 0.140 s) --
    with the direct two-GEMV identity trace(aux_coef @ X) ==
    aux_coef @ (cderi @ dm_sparse): the trace is linear, and
    rhoj = cderi @ dm_sparse is EXACTLY the SCF get_jk packed-pair
    convention (df_jk.py: off-diagonal pairs 2x, the duplicate
    in-block pairs and the true diagonal halved -- the cderi_idx[1]
    indices the pack now carries).  At k_factor==0 the X/dm_oo chain's
    only live product IS that trace (the response chain and the
    auxresp Gram both carry alpha == -.5*k_factor == 0 there, so their
    GEMMs are skipped as identically-zero contributions; the UKS twin
    short-circuits to _j_energy_per_atom before the pack, uhf.py, so
    this knob never touches the UKS lane).  The value delta is
    reassociation-only -- the same class the fall-through itself
    declares (5.4e-13 Eh/Bohr max|dg| at the 63-AO probe); no dtype,
    grid, screening or convergence change anywhere; the J-part
    multiply, the ip1 kernel entry mix, the 2c response and the dm_aux
    expression sequences stay verbatim.  Default OFF; the pack capture
    already refuses the strict fp64-reference context, non-fp64 or
    host-resident cderi and VRAM-admission failures (all keep the
    deployed fall-through).
    '''
    global _JONLY_AUXVEC
    if _JONLY_AUXVEC is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_JONLY_AUXVEC', '')
        if raw == '1':
            _JONLY_AUXVEC = True
        elif raw in ('', '0'):
            _JONLY_AUXVEC = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_JONLY_AUXVEC={raw!r} ignored; '
                          'expected 1 for the direct jonly auxvec route or '
                          '0 (the deployed pack construction, the default)')
            _JONLY_AUXVEC = False
    return _JONLY_AUXVEC

def _arm_jonly_auxvec_direct(mol, dm, pack):
    '''GRAD-JONLY-AUXVEC-DIRECT-01 armed-path decision, called on EVERY
    pass through the k_factor==0 fall-through (after the jonly-reuse
    engagement note, with the pack already captured), so the realized
    stats cover the inert passes too.  Returns True when the direct
    two-GEMV auxvec route runs; bumps the engagement counter and
    records the realized armed/nF/naux/pack state either lane (the
    home-module pattern; no uhf.py twin -- the UKS k_factor==0 route
    delegates before the pack).  Symmetrize-tagged TDDFT amplitude dms
    stay on the deployed pack construction (the direct identity contracts
    the plain dense dm; the hstack factor semantics of the symmetrize
    convention are out of its scope).'''
    cderi, _, aux_coef = pack[0], pack[1], pack[2]
    armed = (_jonly_auxvec() and not _strict_fp64_reference_active()
             and not hasattr(dm, 'symmetrize'))
    global _JONLY_AUXVEC_GRADS
    if armed:
        _JONLY_AUXVEC_GRADS += 1
    _JONLY_AUXVEC_STATS.update(
        armed=bool(armed), natm=mol.natm, nF=int(cderi.shape[0]),
        naux=int(aux_coef.shape[0]), npair=int(cderi.shape[1]),
        symmetrize_refused=bool(hasattr(dm, 'symmetrize')))
    return armed

_JONLY_DEADSKIP = None   # cached env value; None = not yet read
_JONLY_DEADSKIP_GRADS = 0    # armed dead-skip grad engagements (counter)
_JONLY_DEADSKIP_STATS = {}   # realized armed/subsumed state of the last pass

def _jonly_deadskip():
    '''Whether to skip the identically-zero exchange contractions at the
    J-only fall-through (opt-in, GRAD-JONLY-DEADSKIP-01).

    GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP=1 skips, at the single RKS
    closed-shell k_factor==0 fall-through of _jk_energy_per_atom (the
    GRAD-J3C-REUSE-JONLY-01 armed route, pack retained), the two
    contractions the deployed fall-through launches whose exchange
    contribution is identically zero there: the response-chain GEMM pair
    (its second GEMM carries alpha == -.5*k_factor == 0, computing
    dm_tensor = 0*(...) + beta*dm_tensor) and the auxresp Gram
    contraction (alpha == 0, computing dm_aux = 0*Gram + beta*dm_aux).
    The skip applies exactly the beta scales the GEMM beta paths apply
    -- 0*AB + beta*C == beta*C bitwise for finite operands, the same
    one-rounding multiply in the same dtype -- so the value is
    bitwise-identical to the deployed fall-through by construction.
    Unlike GRAD-JONLY-AUXVEC-DIRECT-01 (which additionally replaces the
    pack construction with the two-GEMV reassociation and measured a
    stable amplified envelope delta at the band cells, e664), this knob
    keeps the X build, the dm_oo tail einsum and the trace auxvec
    VERBATIM (dm_oo stays live up to the trace) and never touches the
    resp-f32 arming or any dtype path (the skip takes precedence over
    the response GEMM branch on both dtype lanes, so the fp32-lane
    dm_tensor keeps the one-rounding fp32 product the deployed fp32 GEMM
    beta path would consume).  It never arms when the direct route is
    armed (that route already skips the dead GEMMs -- recorded as
    subsumed).  Default OFF; the pack capture's refusals (strict
    fp64-reference context, non-fp64 or host-resident cderi, shape and
    VRAM-admission failures) keep the verbatim early return; the UKS
    k_factor==0 twin delegates to _j_energy_per_atom before the pack,
    so this knob never touches the UKS lane; precision-neutral (no
    dtype change anywhere).
    '''
    global _JONLY_DEADSKIP
    if _JONLY_DEADSKIP is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP', '')
        if raw == '1':
            _JONLY_DEADSKIP = True
        elif raw in ('', '0'):
            _JONLY_DEADSKIP = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_JONLY_DEADSKIP={raw!r} ignored; '
                          'expected 1 for the dead-chain skip or 0 (the '
                          'deployed contractions, the default)')
            _JONLY_DEADSKIP = False
    return _JONLY_DEADSKIP

def _arm_jonly_deadskip(mol, direct_armed):
    '''GRAD-JONLY-DEADSKIP-01 armed-path decision, called on EVERY pass
    through the k_factor==0 fall-through (right after the jonly-auxvec
    decision, with the pack already captured), so the realized stats
    cover the inert passes too (the 2faa55d pattern).  Returns True when
    the dead-chain skip runs; bumps the engagement counter and records
    the realized armed/natm/subsumed state either lane (the home-module
    pattern; no uhf.py twin -- the UKS k_factor==0 route delegates
    before the pack).'''
    armed = (_jonly_deadskip() and not _strict_fp64_reference_active()
             and not direct_armed)
    global _JONLY_DEADSKIP_GRADS
    if armed:
        _JONLY_DEADSKIP_GRADS += 1
    _JONLY_DEADSKIP_STATS.update(
        armed=bool(armed), natm=mol.natm,
        subsumed_by_direct=bool(direct_armed))
    return armed

_AUXDET = None        # cached env value; None = not yet read
_AUXDET_GRADS = 0     # armed auxdet grad engagements (evidence counter)
_AUXDET_STATS = {}    # realized armed/site state of the last call either lane

def _auxdet():
    '''Whether to run the j-only aux sub-chain's kernel accumulation in the
    deterministic atomic-free mode (opt-in, GRADAUX-DET-REDUCTION-01).

    GPU4PYSCF_DF_GRAD_AUXDET=1 replaces, at the single J-only site
    (_j_energy_per_atom, shared verbatim by the RKS and UKS k_factor==0
    routes), the atomicAdd reassociation of BOTH aux sub-chain kernels --
    the f32 ip1 twin's ejk/ejk_aux accumulation and the 2-center
    e_int2c2e_ip1 auxbasis response -- with the same-value fixed-order
    segmented reduction (per-block private partial rows folded by a
    single-writer fixed-order reduce kernel): the multiset of addends is
    exactly the atomicAdd multiset, so the only arithmetic delta is the
    deterministic reassociation.  M5-DRIFT-ATTRIB-01 (entry 645) measured
    that on the D1-ON GPU4PYSCF_DF_IP1_F32 lane the two carriers'
    differently-reassociated accumulations decorrelate and break the
    elementwise aux-response cancellation behind the ~1e-6 Eh/Bohr
    k-repeat gradient drift (the fp64 R lane passes via correlated
    cancellation, total ~1e-12); GRADAUX-F32-CANCEL-01 (entry 646)
    measured that operand-level harmonization does NOT restore it (the
    decorrelation sits in the kernels' accumulation behavior); D1-GRAD-
    JONLY-AUXFP64-01 (entry 649) restores the cancellation by retreating
    the site to the deployed fp64 kernel at a measured +4.1% e2e price.
    This knob instead makes the accumulation deterministic while KEEPING
    the f32 twin: both pieces become bitwise reproducible run-to-run, so
    the assembled j-only aux response no longer wobbles at all.  The knob
    arms only on the opt-in ip1-f32 site lane (after the e649
    jonly-auxfp64 flip decision: if that knob reverts the site to fp64
    this one stays inert), never on the global fp32 ('auto') lane, and
    stays inert under the strict fp64-reference context (numint.
    strict_fp64_reference_ctx).  The hybrid J+K site and the nuclear
    response site (measured non-carrier, ~1e-14 drift) keep their
    deployed paths.  Default OFF.
    '''
    global _AUXDET
    if _AUXDET is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_AUXDET', '')
        if raw == '1':
            _AUXDET = True
        elif raw in ('', '0'):
            _AUXDET = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_AUXDET={raw!r} ignored; '
                          'expected 1 to enable the deterministic aux '
                          'sub-chain reduction or 0 (off, the default)')
            _AUXDET = False
    return _AUXDET

def _arm_auxdet(mol, naux, site_fp32, nbx, nby):
    '''GRADAUX-DET-REDUCTION-01 armed-path decision, called from
    _j_energy_per_atom on EVERY pass through the site (right after the
    e649 jonly-auxfp64 flip decision, so the recorded site_fp32 is the
    site's realized kernel selection), so the realized stats are recorded
    on the plain-fp64 and jonly-reverted lanes too where the knob is
    inert by construction.  Returns True when BOTH aux sub-chain pieces
    of this call run the deterministic entries: the f32-twin 3c kernel
    (site_fp32 is True only via the opt-in GPU4PYSCF_DF_IP1_F32
    prototype, precision is not the global fp32 mode, knob on, strict
    fp64-reference context inactive) and, on the armed lane only, the
    2-center response.  Bumps the engagement counter and records the
    realized armed/natm/naux/nblocks state of the last call either lane
    (the home-module pattern; no uhf.py twin exists for this site --
    the UKS k_factor==0 route delegates here).'''
    armed = (site_fp32
             and precision.get_precision() != 'fp32'
             and _auxdet()
             and not _strict_fp64_reference_active())
    global _AUXDET_GRADS
    if armed:
        _AUXDET_GRADS += 1
    _AUXDET_STATS.update(
        armed=bool(armed), natm=mol.natm, naux=int(naux),
        nblocks=int(nbx)*int(nby) if armed else 0,
        site_fp32=bool(site_fp32))
    return armed


_J3C_REUSE = None     # cached env value; None = not yet read
_J3C_REUSE_GRADS = 0  # armed cderi-reuse gradient engagements (evidence counter)
_J3C_REUSE_JONLY_GRADS = 0  # armed pure-functional (k_factor==0) engagements

def _note_j3c_reuse_engaged():
    '''Bump the engagement counter from either the RHF or the UHF twin
    (a ``global`` statement in uhf.py would touch uhf's namespace, not
    this counter).'''
    global _J3C_REUSE_GRADS
    _J3C_REUSE_GRADS += 1

def _note_j3c_reuse_jonly_engaged():
    '''Bump the J-only-route engagement counter from either the RHF lane
    or a future UHF twin (same home-module pattern as
    _note_j3c_reuse_engaged: a ``global`` in uhf.py would fork the
    counter into uhf's namespace).'''
    global _J3C_REUSE_JONLY_GRADS
    _J3C_REUSE_JONLY_GRADS += 1

def _j3c_reuse():
    '''Whether the DF-gradient K-chain may consume the SCF-built cderi
    instead of re-evaluating fp64 j3c blocks (opt-in, GRAD-J3C-REUSE-01;
    at pure functionals the armed pack also serves the J-only route,
    GRAD-J3C-REUSE-JONLY-01: k_factor == 0 falls through to the joint
    machinery instead of re-evaluating contract_dm + the j2c metric).

    GPU4PYSCF_DF_GRAD_J3C_REUSE=1 keeps the with_df cderi (and its retained
    aux-side j2c^{-1/2} factor, DF._cderi_aux_coef) alive across the
    SCF->gradient phase boundary and rebuilds the pseudo-DM as
    dm_oo = (dm_factor_l . cderi . dm_factor_r) . aux_coef^T -- the exact
    associativity of the deployed chain's j3c-eval + j2c-metric route
    (differences are only the _decompose_j2c discarded subspace and fp64
    reassociation rounding).  This removes the per-batch plain-j3c Rys
    re-evaluation (the measured fill3c member) and the fresh j2c build +
    metric solve from the gradient.  The verbatim path runs whenever the
    knob is off, the cderi is absent/host-resident, shapes disagree, or
    retention would break the free-memory headroom; strict fp64-reference
    lanes never arm.  Default OFF.
    '''
    global _J3C_REUSE
    if _J3C_REUSE is None:
        raw = os.environ.get('GPU4PYSCF_DF_GRAD_J3C_REUSE', '')
        if raw == '1':
            _J3C_REUSE = True
        elif raw in ('', '0'):
            _J3C_REUSE = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DF_GRAD_J3C_REUSE={raw!r} ignored; '
                          'expected 1 to enable cderi reuse in the DF '
                          'gradient or 0 (re-evaluate j3c, the default)')
            _J3C_REUSE = False
    return _J3C_REUSE

_P1_F32 = None       # cached env value; None = not yet read
_P1_F32_GRADS = 0    # armed phase-1-f32 grad engagements (evidence counter)

def _note_p1_f32_engaged():
    '''Bump the phase-1-f32 engagement counter from either the RHF lane or
    a future UHF twin (same home-module pattern as
    _note_j3c_reuse_engaged: a ``global`` in uhf.py would fork the
    counter into uhf's namespace).'''
    global _P1_F32_GRADS
    _P1_F32_GRADS += 1

def _p1_f32():
    '''Whether the phase-1 j3c@dm_factor contractions of the reuse-armed
    DF-gradient K-chain may run in float32 (opt-in precision-policy
    prototype, GRAD-P1F32-01).

    GPU4PYSCF_DFGRAD_P1_F32=1 casts the phase-1 operands and buffers --
    the dense symmetric scatter destination, the two per-block GEMMs and
    the X buffer -- to float32 ONCE per pack (the scatter converts
    in-kernel from the retained fp64 cderi; the dm factors cast once per
    gradient call); X is cast back to float64 before the aux_coef
    einsum, which keeps float64, as does everything downstream (the
    response chain, the J-part build, the ip1 kernel and the
    auxbasis-response contraction are the armed parent's verbatim).

    Discipline distinction: the float64 phase-1 + metric-solve note at
    the top of _jk_energy_per_atom binds the VERBATIM path, where the
    j2c^-1 metric solve amplifies float32 j3c rounding by the
    auxiliary-basis condition number straight into the gradient.  The
    reuse-armed path this knob composes with has NO metric solve:
    dm_oo = aux_coef @ X applies the j2c^{-1/2} factor (sqrt of the
    condition number), and the downstream float32 tolerance is the
    deployed resp32 precedent (GPU4PYSCF_DFGRAD_RESP_F32, entry 369:
    dg 4.6e-7/3.5e-7 Eh/Bohr at the band).  The verbatim path is
    therefore never touched: the knob is inert -- warn + ignore --
    whenever the reuse pack is absent, and the strict fp64-reference
    context forces it inert.  RKS/RHF lane only; a UKS twin is a
    successor unit.  Default OFF.
    '''
    global _P1_F32
    if _P1_F32 is None:
        raw = os.environ.get('GPU4PYSCF_DFGRAD_P1_F32', '')
        if raw == '1':
            _P1_F32 = True
        elif raw in ('', '0'):
            _P1_F32 = False
        else:
            import warnings
            warnings.warn(f'GPU4PYSCF_DFGRAD_P1_F32={raw!r} ignored; '
                          'expected 1 to enable the fp32 phase-1 lane on '
                          'the cderi-reuse path or 0 (fp64, the default)')
            _P1_F32 = False
    return _P1_F32

def _capture_grad_j3c_reuse(mf, omega):
    '''Collect the retained-cderi reuse pack from mf.with_df, or None.

    The SR (omega None/0) pack comes from with_df itself; a range-separated
    pack lives on the per-omega RSH-DF object created by DF.range_coulomb
    (the same object whose lazy first-use build the SCF already paid for).
    Returns (cderi_device_array (nF, nao_pair), pair_addresses (original
    AO order, matching cderi rows), aux_coef (sorted-aux rows x nF),
    nao_original) or None with a debug reason when any precondition fails.
    '''
    wdf = getattr(mf, 'with_df', None)
    if wdf is None:
        return None
    holder = wdf
    if omega is not None and omega != 0:
        holder = wdf._rsh_df.get('%.6f' % omega)
        if holder is None:
            logger.debug1(mf.mol, 'j3c-reuse inert: no RSH-DF object for '
                          'omega=%s', omega)
            return None
    cderi_store = getattr(holder, '_cderi', None)
    aux_coef = getattr(holder, '_cderi_aux_coef', None)
    cderi_idx = getattr(holder, '_cderi_idx', None)
    if (cderi_store is None or aux_coef is None or cderi_idx is None):
        logger.debug1(mf.mol, 'j3c-reuse inert: cderi/aux_coef not retained')
        return None
    cderi = cderi_store[0] if isinstance(cderi_store, (list, tuple)) \
        else cderi_store
    # only the on-GPU store is in scope; the host-mapped build stays verbatim
    if not isinstance(cderi, cp.ndarray) or cderi.ndim != 2:
        logger.debug1(mf.mol, 'j3c-reuse inert: cderi not on device')
        return None
    # fp32-stored cderi (global fp32 cderi mode) is out of scope: the same
    # metric-amplification discipline that keeps the verbatim chain fp64
    if cderi.dtype != cp.float64 or aux_coef.dtype != cp.float64:
        logger.debug1(mf.mol, 'j3c-reuse inert: non-fp64 cderi/aux_coef')
        return None
    pair_addresses = cderi_idx[0]
    if cderi.shape[1] != len(pair_addresses):
        logger.debug1(mf.mol, 'j3c-reuse inert: cderi/pair shape mismatch')
        return None
    if aux_coef.ndim != 2 or aux_coef.shape[1] != cderi.shape[0]:
        logger.debug1(mf.mol, 'j3c-reuse inert: aux_coef/F mismatch')
        return None
    nao = mf.mol.nao
    if int(pair_addresses.max()) >= nao * nao:
        logger.debug1(mf.mol, 'j3c-reuse inert: pair addresses exceed nao^2')
        return None
    # admission: keeping the (already resident) cderi allocated must leave
    # as much free memory as releasing it would, plus the standing headroom
    # policy; else release and take the verbatim path.
    free = get_avail_mem(exclude_memory_pool=True)
    if free < cderi.nbytes + 2048 * 2**20:
        logger.debug1(mf.mol, 'j3c-reuse inert: free %.0f MiB < cderi %.0f '
                      'MiB + headroom', free / 2**20, cderi.nbytes / 2**20)
        return None
    # GRAD-JONLY-AUXVEC-DIRECT-01: the pack carries the diagonal-block
    # pair indices (cderi_idx[1], the ground-truth convention the SCF
    # get_jk packed-pair dm_sparse halving uses) so the direct jonly
    # auxvec route needs no local re-derivation of the duplicate-pair
    # convention.  The deployed X/dm_oo consumers ignore this element.
    return cderi, pair_addresses, aux_coef, nao, cderi_idx[1]

def _gen_metric_solver(int2c, decompose_j2c='CD', lindep=df.LINEAR_DEP_THR):
    ''' generate a solver to solve Ax = b, RHS must be in (n,....) '''
    if decompose_j2c.upper() == 'CD':
        try:
            j2c = cholesky(int2c)
            def j2c_solver(b):
                b_ = b.reshape(j2c.shape[0], -1)
                y = solve_triangular(j2c, b_, lower=True, overwrite_b=False)
                x = solve_triangular(j2c.conj().T, y, lower=False, overwrite_b=False)
                return cp.asarray(x.reshape(b.shape), order='A')
            return j2c_solver
        except RuntimeError:
            pass

    w, v = eigh(int2c)
    mask = w > lindep
    inv_w = 1 / w[mask]
    v1 = v[:,mask]
    def j2c_solver(b): # noqa: F811
        x = v1.conj().T @ b.reshape(int2c.shape[0],-1)
        x = x * inv_w[:,None]
        x = v1 @ x
        return x.reshape(b.shape)
    return j2c_solver

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=0,
                        auxbasis_response=True, verbose=None,
                        omega=None, lr_factor=None, sr_factor=None,
                        j3c_reuse=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    j3c_reuse (GRAD-J3C-REUSE-01, opt-in): a pack from
    _capture_grad_j3c_reuse -- (cderi (nF, nao_pair, original-AO pair
    rows), pair_addresses, aux_coef (sorted-aux x nF), nao_original,
    diag_indices (the cderi_idx[1] diagonal-block pair columns,
    GRAD-JONLY-AUXVEC-DIRECT-01, unused by the deployed X/dm_oo route)) --
    rebuilds dm_oo as (dm_factor_l . cderi . dm_factor_r) . aux_coef^T,
    the exact associativity of the verbatim j3c-evaluation + j2c-metric
    route (differences: the _decompose_j2c discarded subspace + fp64
    reassociation rounding only).  None everywhere by default.
    '''
    if hermi == 2:
        j_factor = 0
    # GRAD-JONLY-AUXVEC-DIRECT-01: True only on the armed k_factor==0
    # fall-through below (the RKS closed-shell pure-functional route with
    # the reuse pack retained); every other path keeps it False so the
    # X/dm_oo pack construction and the exchange-side contractions run
    # verbatim.
    jonly_direct = False
    jonly_deadskip = False
    if k_factor == 0:
        assert omega is None or omega == 0
        if j3c_reuse is None or j_factor == 0:
            return _j_energy_per_atom(int3c2e_opt, dm, hermi,
                                      auxbasis_response, verbose) * j_factor
        # GRAD-J3C-REUSE-JONLY-01 armed pure-functional route: with the pack
        # retained, k_factor == 0 falls through to the joint machinery below,
        # where every exchange contribution is identically zero (the
        # -.5*k_factor scales and the response-chain alpha) and the J
        # derivative is computed through the same production-verified pack
        # construction hybrids use, replacing the J-only re-evaluation of
        # the 3c stream (contract_dm) + fresh j2c build + metric solve.
        # The k_factor==0 joint math equals the J-only math up to the shared
        # metric-route reassociation (measured 5.4e-13 Eh/Bohr max|dg| at
        # the 63-AO probe cell); at small cells the pack construction can
        # exceed the verbatim cost (0.17 vs 0.13 s/call at 63 AO), so the
        # knob stays opt-in with the unchanged capture/admission guards.
        # Pack capture upstream already refuses the strict fp64-reference
        # context, non-fp64 or host-resident cderi, shape mismatches and
        # VRAM-admission failures; those all keep the verbatim early return.
        _note_j3c_reuse_jonly_engaged()
        # GRAD-JONLY-AUXVEC-DIRECT-01: the armed direct-auxvec decision,
        # recorded on every pass through the fall-through (the 2faa55d
        # unconditional-stats pattern; mol is bound only below the
        # dispatch, so the site mol is used here).
        jonly_direct = _arm_jonly_auxvec_direct(int3c2e_opt.mol, dm,
                                                j3c_reuse)
        # GRAD-JONLY-DEADSKIP-01: the dead-chain skip decision, recorded
        # on every fall-through pass (the 2faa55d pattern); never armed
        # when the direct route is (that route already skips the dead
        # GEMMs -- subsumed).
        jonly_deadskip = _arm_jonly_deadskip(int3c2e_opt.mol,
                                             jonly_direct)

    if j_factor != 0 and (omega is not None and omega != 0):
        raise RuntimeError(
            'Cannot compute J contributions with range-separated Coulomb operator.')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = _factorize_dm(mol, dm, hermi)
    nao, nocc = dm_factor_l.shape
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - naux*nocc**2*8 - nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True,
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
    aux_batches = len(aux_offsets) - 1

    # DF-GRAD-TASKGRAN-967-01 (c): GPU4PYSCF_DF_TASKGRAN=2 caps the
    # launch capacity at the evaluator's largest aux batch (upstream
    # #967 -- the same cap our df/hessian sites already carry) and
    # bounds the blksize/scratch tensors by that capacity; knob-off
    # lanes keep the verbatim expressions value-identically.
    resplit = _taskgran_resplit_active()
    if resplit:
        batch_size = min(batch_size, int((aux_offsets[1:]-aux_offsets[:-1]).max()))

    blksize = max(1, min(batch_size if resplit else naux,
                         int(mem_avail*.4/(nao*nao*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    # Note: the j3c @ dm_factor contractions and the metric solve stay in
    # float64 even in fp32 mode. The j2c^-1 metric solve amplifies relative
    # errors of the j3c blocks (~1e-7 in float32) by the auxiliary-basis
    # condition number, which lands around 1e-3 in the final gradient -
    # far above the ~1e-5 obtained from the integral kernel alone.
    if j3c_reuse is None:
        if _p1_f32() and not _strict_fp64_reference_active():
            # GRAD-P1F32-01 value-only knob: no armed pack to compose with
            import warnings
            warnings.warn('GPU4PYSCF_DFGRAD_P1_F32=1 ignored: the fp32 '
                          'phase-1 lane requires GPU4PYSCF_DF_GRAD_J3C_REUSE'
                          '=1 with an armed cderi pack; running the fp64 '
                          'phase-1 verbatim')
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        buf1 = cp.empty((blksize, nocc, nao))
        j3c_oo = cp.empty((naux, nocc, nocc))
        for kbatch in range(aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:,:,:dk]
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm_factor_r, out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l, out=j3c_oo[aux0:aux1])
        j3c_full = buf = buf1 = eval_j3c = j3c = tmp = compressed = None
        t0 = log.timer_debug1('contract dm', *t0)

        aux_coeff = cp.asarray(auxmol.ctr_coeff)
        aux_coeff, tmp = cp.empty_like(aux_coeff), aux_coeff
        aux_coeff[aux_sorting] = tmp
        tmp = None

        j2c = int2c2e(auxmol, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
        if mol.omega <= 0 and not auxmol.mol.cart:
            metric = aux_coeff.dot(_gen_metric_solver(j2c, 'CD')(aux_coeff.T))
        else:
            metric = aux_coeff.dot(_gen_metric_solver(j2c, 'ED')(aux_coeff.T))
        j2c = aux_coeff = None
        dm_oo = cp.einsum('uv,vij->uij', metric, j3c_oo)
        metric = j3c_oo = None
    else:
        # GRAD-J3C-REUSE-01 armed path: same fp64 discipline, no fp32.
        # The cderi rows are original-AO pairs (the convention the SCF's
        # own DF machinery consumes), so the dense symmetric scatter and
        # the two GEMMs run in the original AO basis with untransformed
        # dm factors; aux_coef maps the reduced F axis onto the sorted-aux
        # dm_oo rows the downstream ip1/auxresp tail expects.
        _note_j3c_reuse_engaged()
        cderi, pair_addr0, aux_coef, nao0, diag0 = j3c_reuse
        i_addr0, j_addr0 = divmod(pair_addr0, nao0)
        if jonly_direct:
            # GRAD-JONLY-AUXVEC-DIRECT-01: at k_factor == 0 the X/dm_oo
            # chain below has exactly ONE live product --
            # auxvec = trace(dm_oo) -- and the trace is linear:
            # trace(aux_coef @ X) == aux_coef @ (cderi @ dm_sparse),
            # with rhoj = cderi @ dm_sparse the EXACT SCF get_jk
            # packed-pair convention (df_jk.py: off-diagonal pairs 2x,
            # the duplicate in-block pairs and the true diagonal halved
            # -- diag0, the cderi_idx[1] indices the pack carries).  Two
            # GEMVs replace the nocc^2-factorized pack construction (the
            # e636-measured p1 + tail einsum band); the delta vs the
            # deployed route is summation order only (the reassociation
            # class the fall-through itself declares).  dm is still the
            # caller's original-basis matrix here (the sorted-basis
            # reconstruction happens later, below).
            dm0 = cp.asarray(dm)
            dm_sparse = dm0[i_addr0, j_addr0]
            if hermi == 0:
                dm_sparse = dm_sparse + dm0[j_addr0, i_addr0]
            else:
                dm_sparse = dm_sparse * 2
            if diag0 is not None and len(diag0):
                dm_sparse[diag0] *= .5
            auxvec = aux_coef.dot(cderi.dot(dm_sparse))
            dm_oo = None
            t0 = log.timer_debug1('contract dm (j3c-reuse direct)', *t0)
        else:
            dm_factor_l0, dm_factor_r0 = factorize_dm(dm, hermi)
            if dm_factor_r0 is None:
                # mirror _factorize_dm's hermi=0 aliasing
                dm_factor_r0 = dm_factor_l0
            if hasattr(dm, 'symmetrize'):
                # mirror _factorize_dm's convention on the original basis
                if dm.symmetrize == 1:
                    dm_factor_l0, dm_factor_r0 = (
                        cp.hstack([dm_factor_l0, dm_factor_r0]),
                        cp.hstack([dm_factor_r0, dm_factor_l0]))
                elif dm.symmetrize == 2:
                    dm_factor_l0, dm_factor_r0 = (
                        cp.hstack([dm_factor_l0, dm_factor_r0]),
                        cp.hstack([dm_factor_r0, -dm_factor_l0]))
            nocc0 = dm_factor_l0.shape[1]
            assert nocc0 == nocc and dm_factor_l0.shape[0] == nao0
            nF = cderi.shape[0]
            # GRAD-P1F32-01 opt-in fp32 phase-1 lane (armed path only; see
            # _p1_f32 for the sqrt-conditioning distinction vs the verbatim
            # path's metric-solve discipline): fp32 scatter destination, GEMM
            # operands and X buffer; the aux_coef einsum below stays fp64.
            if _p1_f32() and not _strict_fp64_reference_active():
                _note_p1_f32_engaged()
                p1_dt = cp.float32
                dm_factor_r_g = dm_factor_r0.astype(cp.float32)
                dm_factor_l_g = (dm_factor_r_g if dm_factor_l0 is dm_factor_r0
                                 else dm_factor_l0.astype(cp.float32))
            else:
                p1_dt = cp.float64
                dm_factor_r_g = dm_factor_r0
                dm_factor_l_g = dm_factor_l0
            blksize0 = max(1, min(nF, int(mem_avail*.4/(nao0*nao0*2*8))//8*8))
            X = cp.empty((nF, nocc, nocc), dtype=p1_dt)
            j3c_full = cp.zeros((nao0, nao0, blksize0), dtype=p1_dt)
            buf1 = cp.empty((blksize0, nocc, nao0), dtype=p1_dt)
            for k0, k1 in lib.prange(0, nF, blksize0):
                dk = k1 - k0
                j3c = j3c_full[:,:,:dk]
                # cderi is stored F-major (nF, nao_pair); the transposed view
                # feeds the same symmetric scatter the verbatim path uses
                # (on the fp32 lane the assignment converts in-kernel)
                j3c[j_addr0,i_addr0] = j3c[i_addr0,j_addr0] = cderi[k0:k1].T
                tmp = ndarray((nocc, nao0, dk), buffer=buf1, dtype=p1_dt)
                contract('pqr,pi->iqr', j3c, dm_factor_r_g, out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l_g, out=X[k0:k1])
            j3c_full = buf1 = j3c = tmp = None
            t0 = log.timer_debug1('contract dm (j3c-reuse)', *t0)
            if p1_dt != cp.float64:
                X = X.astype(cp.float64)
            dm_oo = cp.einsum('kv,vij->kij', aux_coef, X)
            X = None
    if j_factor != 0 and dm_oo is not None:
        auxvec = dm_oo.trace(axis1=1, axis2=2)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=mol.omega<0, gout_width=54, deriv=(1,0,0))
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    # DF-GRAD-TASKGRAN-01 (a): GPU4PYSCF_DF_TASKGRAN=1 (and 2) caps the
    # ip1 sum-kernel pair blocks at min(nsp*4, max(nsp//4, 16))
    # (upstream #970) instead of the verbatim nsp*4; knob-off lanes are
    # value-identical.  len(shl_pair_offsets)-1 is this launch's grid
    # x-dimension (the launch-count instrumentation knob).
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 4))
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    if j_factor != 0:
        dm = dm_factor_l.dot(dm_factor_r.T)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    # LC-86: the opt-in ip1-f32 prototype selects the fp32 kernel on the
    # fp64 lane too; the fp32 discipline (fp64 pre-contractions and metric
    # solve, fp32 kernel + pseudo-DM) is the one the global fp32 mode
    # already uses -- see _ip1_f32().
    fp32 = precision.get_precision() == 'fp32' or _ip1_f32()
    if fp32:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1_f32
    else:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1
    assert lr_factor is None and sr_factor is None
    # DFJK-RESPGEMM-IMPL-01: opt-in fp32 PHASE-2 response chain (BENCH-03
    # V1, entry 368: 19.92x/28.83x per-chain at 819/1042; one-time operand
    # casts 2.8 ms at 819, excluded from the chain wall there and counted
    # into t_grad here).  The fp32 casts below plus the fp32 tmp/dm_tensor
    # buffers are the entire armed delta; phase-1, the metric solve, the
    # J-part elementwise build and auxresp keep their fp64 operands and
    # buffers (deployed amplification discipline).  dm_oo stays fp64 for
    # auxresp; dm_oo32 coexists for one grad call (~52 MB at the 819
    # cell, measured feasible at both bench cells).
    if jonly_direct:
        # GRAD-JONLY-AUXVEC-DIRECT-01: no dm_oo exists on the direct route
        # and the exchange-side response chain is skipped below (its
        # contribution is identically zero at k_factor == 0), so the resp32
        # arming never has an operand to cast; the knob's engagement
        # semantics at the jonly fall-through become honestly inert (the
        # UKS jonly twin never reaches this site at all).
        dm_oo32 = dm_factor_l32 = dm_factor_r32 = None
        resp32 = False
    else:
        # the DEADSKIP lane passes through this verbatim arming on purpose:
        # buf1/buf2/dm_tensor dtypes and the one-rounding product discipline
        # must stay identical to the deployed lane (the skip below applies
        # the same-dtype beta multiply the GEMM beta path would apply); its
        # dm_oo32 cast is simply unconsumed when the skip fires.
        dm_oo32, dm_factor_l32, dm_factor_r32 = _arm_resp_f32(
            dm_oo, dm_factor_l, dm_factor_r)
        resp32 = dm_oo32 is not None
    # LC-42: fused-launch guard.  The per-batch path allocates
    # nao_pair*batch_size and reuses it batch by batch; the fused path
    # needs the full nao_pair*naux compressed matrix resident at once.
    # Require it to fit in 25% of the currently available device memory,
    # else keep the upstream per-batch behavior exactly.
    fused = _ip1_fused() and aux_batches > 1 and \
        nao_pair*naux*(4 if fp32 else 8) < get_avail_mem()*0.25
    aux0 = aux1 = 0
    if fused:
        buf = cp.empty((nao_pair*naux), dtype=cp.float32 if fp32 else cp.float64)
    else:
        buf = cp.empty((nao_pair*batch_size), dtype=cp.float32 if fp32 else cp.float64)
    buf1 = cp.empty((blksize, nao, nao),
                    dtype=cp.float32 if resp32 else cp.float64)
    buf2 = cp.empty((blksize, nocc, nao),
                    dtype=cp.float32 if resp32 else cp.float64)
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    # DF-GRAD-TASKGRAN-967-01 (c): armed level 2 packs consecutive aux
    # groups up to batch_size functions per launch (upstream #967) and
    # passes the packed group count to the kernel; knob-off lanes
    # iterate value-identically (kend == kbatch+1, group count 1).  The
    # fused path already covers all groups in one launch, so the
    # packing is skipped there.  len(aux_splits)-1 is the armed launch
    # count vs the verbatim aux_batches (launch-count instrumentation).
    if fused:
        launch_units = [(0, aux_batches)]
    else:
        aux_splits = _taskgran_resplit_splits(aux_offsets, batch_size)
        launch_units = list(zip(aux_splits[:-1], aux_splits[1:]))
    for kbatch, kend in launch_units:
        if fused:
            naux_in_batch = naux
            aux_ao_offset = 0
            compressed = ndarray((nao_pair, naux),
                                 dtype=cp.float32 if fp32 else cp.float64, buffer=buf)
        else:
            naux_in_batch = aux_offsets[kend] - aux_offsets[kbatch]
            aux_ao_offset = aux_loc[ksh_offsets_cpu[kbatch]]
            compressed = ndarray((nao_pair, naux_in_batch),
                                 dtype=cp.float32 if fp32 else cp.float64, buffer=buf)
        for k0, k1 in lib.prange(0, naux_in_batch, blksize):
            dk = k1 - k0
            aux0, aux1 = aux1, aux1 + dk
            dm_tensor = ndarray((nao,nao,dk), dtype=buf1.dtype, buffer=buf1)
            tmp = ndarray((nocc,nao,dk), dtype=buf2.dtype, buffer=buf2)
            beta = 0
            if j_factor != 0:
                # fp64 inputs into the (possibly fp32) tensor: one rounding
                # of the exact fp64 product, at least as accurate as casting
                # the inputs first
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            if jonly_direct or jonly_deadskip:
                # GRAD-JONLY-AUXVEC-DIRECT-01 / GRAD-JONLY-DEADSKIP-01: the
                # exchange-side response chain's contribution is identically
                # zero at k_factor == 0 (alpha == -.5*k_factor == 0); the
                # deployed second GEMM would compute dm_tensor =
                # 0*(...) + beta*dm_tensor, so the skip applies exactly the
                # beta scale the GEMM's beta path applies (one rounded
                # multiply; bitwise the same value).  The skip takes
                # precedence over the resp32 branch on the DEADSKIP lane:
                # that lane keeps the resp-f32 arming and the dm_tensor
                # dtype verbatim, so this is the same-dtype multiply the
                # fp32 GEMM's beta path would apply (dtype-parity bitwise).
                dm_tensor *= beta
            elif resp32:
                # the fp32 response chain (bench V1 mechanism verbatim:
                # fp32 operands, fp32 tmp/dm_tensor buffers)
                contract('rji,qj->iqr', dm_oo32[aux0:aux1], dm_factor_l32,
                         out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r32, -.5*k_factor,
                         beta, out=dm_tensor)
            else:
                contract('rji,qj->iqr', dm_oo[aux0:aux1], dm_factor_l, out=tmp)
                contract('iqr,pi->pqr', tmp, dm_factor_r, -.5*k_factor, beta, out=dm_tensor)
            # the pseudo-DM handed to the integral kernel is cast to float32
            # when the kernel runs in fp32; the pre-contractions above stay
            # float64 for accuracy.  The armed resp32 chain already holds
            # dm_tensor in the fp32 lane's dtype; on the fp64 kernel lane it
            # reassembles to fp64 so the pair compression and the kernel
            # consume their exact dtypes (a dtype leak raises in cp.take,
            # fail-closed).
            if fp32 and not resp32:
                dm_tensor = dm_tensor.astype(cp.float32)
            elif resp32 and not fp32:
                dm_tensor = dm_tensor.astype(cp.float64)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            else:
                dm_tensor1 = cp.ascontiguousarray(dm_tensor.transpose(1,0,2))
                dm_tensor1 = dm_tensor1 + dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            dm_tensor = ndarray((nao,nao,dk), dtype=buf1.dtype, buffer=buf1)
        err = kern(
            ctypes.cast(ejk.data.ptr, ctypes.c_void_p),
            ctypes.cast(ejk_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed.data.ptr, ctypes.c_void_p),
            lib.c_null_ptr(),
            ctypes.c_int(1),
            ctypes.byref(int3c2e_envs),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(len(ksh_offsets_cpu) - 1 if fused else kend-kbatch),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu.data.ptr if fused
                        else ksh_offsets_gpu[kbatch:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao), ctypes.c_int(nao_pair),
            ctypes.c_int(naux_in_batch), ctypes.c_int(mol.natm))
        if err != 0:
            raise RuntimeError('int3c2e_ejk_ip1 failed')
    buf = buf1 = buf2 = compressed = dm_tensor = dm_tensor1 = tmp = None
    # release the armed fp32 twins before the fp64 auxresp contraction
    dm_oo32 = dm_factor_l32 = dm_factor_r32 = None
    if hermi == 1:
        ejk *= 2
        ejk_aux *= 2
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    if auxbasis_response:
        # (d/dX P|Q) contributions
        if j_factor == 0:
            dm_aux = None
        else:
            dm_aux = auxvec[:,None] * auxvec
        if dm_oo is None or jonly_deadskip:
            # GRAD-JONLY-AUXVEC-DIRECT-01 / GRAD-JONLY-DEADSKIP-01: at
            # k_factor == 0 the Gram term carries alpha == -.5*k_factor ==
            # 0, so the deployed contraction returns exactly j_factor *
            # (auxvec outer auxvec) (0*Gram + beta*out == beta*out bitwise
            # for finite operands); the skip applies that beta scale
            # without launching the dead GEMM.  The DIRECT route has no
            # dm_oo at all; the DEADSKIP lane keeps dm_oo (live up to the
            # trace) and skips only this dead contraction.
            dm_aux *= j_factor
        elif dm_factor_l is dm_factor_r:
            dm_aux = contract('rij,sij->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('rij,sji->rs', dm_oo, dm_oo,
                              alpha=-.5*k_factor, beta=j_factor, out=dm_aux)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        #ejk_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        ejk_aux -= cp.asarray(
            int2c2e_ip1_per_atom(auxmol, dm_aux, omega, lr_factor, sr_factor))
        t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
        ejk += ejk_aux

    ejk = ejk.get()
    return ejk

def _j_energy_per_atom(int3c2e_opt, dm, hermi=0, auxbasis_response=True, verbose=None):
    '''
    Computes the first-order derivatives of the Coulomb energy
    '''
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm = mol.apply_C_mat_CT(dm)
    if hermi != 1:
        dm = transpose_sum(dm, inplace=True)
        dm[:] *= .5
    auxvec = int3c2e_opt.contract_dm(dm, hermi=1)
    naux = len(auxvec)
    t0 = log.timer_debug1('contract dm', *t0)
    j2c = int2c2e(auxmol)

    auxvec = auxmol.CT_dot_mat(auxvec)
    if mol.omega <= 0 and not auxmol.mol.cart:
        auxvec = _gen_metric_solver(j2c, 'CD')(auxvec)
    else:
        auxvec = _gen_metric_solver(j2c, 'ED')(auxvec)
    auxvec = auxmol.C_dot_mat(auxvec)
    j2c = None

    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=mol.omega<0, gout_width=54, deriv=(1,0,0))
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, nsp_per_block[0]*16)
    # DF-GRAD-TASKGRAN-01 (b): GPU4PYSCF_DF_TASKGRAN=1 subdivides the
    # homogeneous aux groups into <=16-shell tasks (upstream #970) for
    # this launch's y-grid; knob-off lanes are value-identical.
    # len(ksh_offsets_cpu)-1 is the y-grid dimension.
    ksh_offsets_cpu = _taskgran_aux16_offsets(auxmol, laux)
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    # Only the integral kernel runs in float32; the dm contraction and the
    # j2c metric solve above stay in float64 (see the note in
    # _jk_energy_per_atom on the metric solve amplifying float32 error).
    # LC-86: the opt-in ip1-f32 prototype applies the same selection on
    # the fp64 lane (see _ip1_f32()).
    fp32 = precision.get_precision() == 'fp32' or _ip1_f32()
    # the arm check runs (and records the realized stats) on EVERY call
    # through this site, including the plain-fp64 lanes where the knob
    # is inert by construction -- the engagement evidence counter must
    # see armed=False there, not an unrecorded gap
    jonly_auxfp64_armed = _arm_jonly_auxfp64(mol, naux)
    if fp32 and jonly_auxfp64_armed:
        # D1-GRAD-JONLY-AUXFP64-01: the opt-in per-site scope keeps this
        # single J-only site on the deployed fp64 ip1 kernel with its fp64
        # dm/auxvec operands (bitwise the ip1-f32-off lane), restoring the
        # R-lane fp64 3c/2c aux-response cancellation structure that
        # entries 645/646 measured the f32 twin breaks at the drifted
        # cells; everything else in the composition keeps its lane.
        fp32 = False
    # GRADAUX-DET-REDUCTION-01: the deterministic-reduction arming runs
    # (and records the realized stats) on EVERY pass through this site --
    # after the e649 flip decision, so site_fp32 is the realized kernel
    # selection (the auxdet knob stays inert on the jonly-reverted,
    # plain-fp64, global-fp32 and strict-reference lanes)
    auxdet_armed = _arm_auxdet(mol, naux, fp32,
                               len(shl_pair_offsets) - 1,
                               len(ksh_offsets_cpu) - 1)
    if fp32:
        if auxdet_armed:
            kern = libvhf_rys.sum_ejk_int3c2e_ip1_f32_det
        else:
            kern = libvhf_rys.sum_ejk_int3c2e_ip1_f32
        dm_kern = cp.asarray(dm, dtype=cp.float32, order='C')
        auxvec_kern = cp.asarray(auxvec, dtype=cp.float32)
    else:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1
        dm_kern = dm
        auxvec_kern = auxvec
    ej = cp.zeros((mol.natm, 3))
    if auxbasis_response:
        ej_aux = cp.zeros_like(ej)
        ej_aux_ptr = ctypes.cast(ej_aux.data.ptr, ctypes.c_void_p)
    else:
        ej_aux_ptr = lib.c_null_ptr()

    # GRADAUX-DET-REDUCTION-01: the armed lane's per-block staging buffer
    # (rows = the flattened blockIdx.x + blockIdx.y*gridDim.x blocks of
    # the launch below; columns = the ejk slots [0, natm*3) followed by
    # the ejk_aux slots).  Zeroed here: the kernel's staged lane
    # accumulates into its own row with non-atomic read-modify-write and
    # the wrapper's fixed-order reduce kernel folds it into ej/ej_aux
    # (both pre-zeroed above).  The buffer stays referenced until the
    # ej.get() below synchronizes the stream, so no later allocation can
    # reuse it while the kernels read it.
    extra_args = ()
    if auxdet_armed:
        partial = cp.zeros(((len(shl_pair_offsets) - 1)
                            * (len(ksh_offsets_cpu) - 1),
                            2*mol.natm*3))
        extra_args = (ctypes.cast(partial.data.ptr, ctypes.c_void_p),)

    err = kern(
        ctypes.cast(ej.data.ptr, ctypes.c_void_p), ej_aux_ptr,
        ctypes.cast(dm_kern.data.ptr, ctypes.c_void_p),
        ctypes.cast(auxvec_kern.data.ptr, ctypes.c_void_p),
        ctypes.c_int(1),
        ctypes.byref(int3c2e_envs),
        ctypes.c_int(shm_size_max),
        ctypes.c_int(len(shl_pair_offsets) - 1),
        ctypes.c_int(len(ksh_offsets_cpu) - 1),
        ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.cast(ksh_offsets_gpu.data.ptr, ctypes.c_void_p),
        ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
        lib.c_null_ptr(),
        ctypes.c_int(0),
        ctypes.c_int(0), ctypes.c_int(0),
        ctypes.c_int(naux), ctypes.c_int(mol.natm), *extra_args)
    if err != 0:
        raise RuntimeError('int3c2e_ejk_ip1 failed')
    ej *= 2
    ej = ej.get()
    t0 = log.timer_debug1('contract int3c2e_ejk_ip1', *t0)

    # (d/dX P|Q) contributions
    if auxbasis_response:
        ej_aux *= 2
        #ej_aux += .5*contract_h1e_dm(auxmol, auxmol.intor('int2c2e_ip1'), dm_aux)
        dm_aux = auxvec[:,None] * auxvec
        # GRADAUX-F32-CANCEL-01: the opt-in harmonization quantization of
        # the 2-center operand to the f32 ip1 J kernel's lane (see
        # _auxresp_f32); knob-off/strict-reference lanes get the deployed
        # fp64 dm_aux back verbatim.
        dm_aux = _arm_auxresp_f32(auxvec, dm_aux)
        # GRADAUX-DET-REDUCTION-01: the armed lane routes the 2-center
        # auxbasis response through the deterministic entry as well -- both
        # carriers of the e645-measured aux sub-chain wobble become bitwise
        # reproducible run-to-run (the knob-off/strict lanes keep the
        # deployed e_int2c2e_ip1 verbatim)
        ej_aux -= cp.asarray(int2c2e_ip1_per_atom(
            auxmol, dm_aux, det_reduction=auxdet_armed))
        ej += ej_aux.get()
    t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
    return ej

def _factorize_dm(mol, dm, hermi):
    dm_factor_l, dm_factor_r = factorize_dm(dm, hermi)
    axis = dm_factor_l.ndim - 2
    dm_factor_l = mol.apply_C_dot(dm_factor_l, axis=axis)
    if dm_factor_r is None:
        dm_factor_r = dm_factor_l
    else:
        dm_factor_r = mol.apply_C_dot(dm_factor_r, axis=axis)
    if hasattr(dm, 'symmetrize'):
        # See the convention in _make_factorized_dm provided by df_jk.py
        if dm.symmetrize == 1:
            dm_factor_l, dm_factor_r = (cp.hstack([dm_factor_l, dm_factor_r]),
                                        cp.hstack([dm_factor_r, dm_factor_l]))
        elif dm.symmetrize == 2:
            dm_factor_l, dm_factor_r = (cp.hstack([dm_factor_l, dm_factor_r]),
                                        cp.hstack([dm_factor_r,-dm_factor_l]))
    return dm_factor_l, dm_factor_r

class Gradients(rhf_grad.Gradients):

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df.df_jk._DFHF)

    def energy_ee(self, mol, dm):
        return self.jk_energy_per_atom(dm, hermi=1)

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, *,
                           omega=None, lr_factor=None, sr_factor=None,
                           hermi=0, verbose=None):
        r'''Compute the first-order derivatives of the Coulomb and exchange
        energies with respect to atomic positions.

            j_factor * dE_J / dR - k_factor * dE_K / dR

        Parameters
        ----------
        dm : ndarray or sequence of ndarray, optional
            Density matrix or (for UKS) a sequence of density matrices.
        j_factor : float, optional
            Scaling factor applied to the Coulomb (J) contribution.
        k_factor : float, optional
            Scaling factor applied to the exchange (K) contribution.
        omega : float, optional
            Range-separation parameter. Together with ``lr_factor`` and
            ``sr_factor``, defines the range-separated Coulomb operator used in
            exchange (K) matrix evaluation:
                lr_factor * erf(omega * r) / r + sr_factor * erfc(omega * r) / r.
        lr_factor : float, optional
            Scaling factor for the long-range interaction,
        sr_factor : float, optional
            Scaling factor for the short-range interaction,
        hermi : int, optional
            Symmetry of the density matrix:
            - ``0``: no symmetry
            - ``1``: Hermitian
            - ``2``: anti-Hermitian

        Notes
        -----
        When omega, lr_factor, and sr_factor parameters are specified, the K
        contribution is evaluated using the attenuated Coulomb operator.

        Returns
        -------
        Array of shape ``(natm, 3)`` containing the derivative of the weighted
        Coulomb and exchange energy with respect to the Cartesian coordinates of
        each atom.
        '''
        mf = self.base
        if dm is None: dm = mf.make_rdm1()
        mol = mf.with_df.mol
        auxmol = mf.with_df.auxmol
        # GRAD-J3C-REUSE-01: capture the retained cderi BEFORE the reset
        # releases it; when the knob cannot arm (off/strict-reference/
        # missing or host-resident cderi/shape or VRAM-admission failure)
        # the pack is None and the verbatim reset + re-evaluation runs.
        j3c_reuse = None
        if (_j3c_reuse() and not _strict_fp64_reference_active()
                and isinstance(dm, cp.ndarray) and dm.ndim == 2):
            j3c_reuse = _capture_grad_j3c_reuse(mf, omega)
        if j3c_reuse is None:
            mf.with_df.reset() # Release GPU memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            # CDERI-B2D-HOSTPATH-01 (C1): the identity-keyed cache serves the
            # opt the SCF DF.build already constructed for this Mole/auxmol
            # pair (decontracted variant).  The token is read inside the
            # range-coulomb context, so an active omega context (RSH cells)
            # fails closed to the verbatim fresh build below.
            int3c2e_opt = _molecular_int3c2e_opt(
                mol, auxmol, decontract=True)
            if int3c2e_opt is None:
                sorted_mol = SortedMole.from_mol(mol, decontract=True)
                int3c2e_opt = Int3c2eOpt(sorted_mol, auxmol).build()
        return _jk_energy_per_atom(
            int3c2e_opt, dm, j_factor, k_factor, hermi=hermi,
            auxbasis_response=self.auxbasis_response, verbose=verbose,
            omega=omega, lr_factor=lr_factor, sr_factor=sr_factor,
            j3c_reuse=j3c_reuse)

Grad = Gradients

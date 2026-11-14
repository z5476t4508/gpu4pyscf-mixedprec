# Copyright 2025 The PySCF Developers. All Rights Reserved.
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
3-center 2-electron Coulomb integral helper functions
'''

import ctypes
import math
import os
import weakref
import numpy as np
import cupy as cp
import warnings
from pyscf import lib
from pyscf.lib.parameters import ANGULAR
from pyscf import gto
from pyscf.gto.mole import (
    ANG_OF, ATOM_OF, PTR_COORD, PTR_EXP, NPRIM_OF, NCTR_OF, conc_env)
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    load_library, contract, dist_matrix, asarray, hermi_triu, transpose_sum,
    ndarray, get_avail_mem)
from gpu4pyscf.lib.utils import splits_by_blocksize
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.gto.mole import (
    PTR_BAS_COORD, SortedMole, RysIntEnvVars, extract_pgto_params, groupby,
    _split_l_ctr_groups)
from gpu4pyscf.scf.jk import (
    _nearest_power2, _scale_sp_ctr_coeff, _check_rsh_factors, SHM_SIZE,
    libvhf_rys)
from gpu4pyscf.__config__ import props as gpu_specs

__all__ = [
    'aux_e2',
]

LMAX = 4
L_AUX_MAX = 6
GOUT_WIDTH = 54
THREADS = 256
POOL_SIZE = 25600

libvhf_rys.pair_recontraction_info.restype = ctypes.c_int
libvhf_rys.recontract_ao_pair.restype = ctypes.c_int

# DF-GRAD-TASKGRAN-01 (Phase A, default off): opt-in port of the
# upstream #970 (8475f60) host-side task-granularity mechanisms into
# our diverged df machinery, per the entry-517 FUNDED disposition.
# (a) the pair cap min(nsp*4, max(nsp//4, 16)) at the four
#     aggregate_shl_pairs sites of this module
#     (int3c2e_evaluator/contract_dm/contract_auxvec/
#     _int3c2e_ip1_evaluator) and df/grad/rhf.py's ip1-sum-kernel site;
# (b) the fixed <=16-shell aux subdivision of the J-only gradient
#     ksh_offsets (df/grad/rhf.py _j_energy_per_atom, via
#     _taskgran_aux16_offsets below; _split_l_ctr_groups is already
#     ours at gto/mole.py);
# (c) DF-GRAD-TASKGRAN-967-01, GPU4PYSCF_DF_TASKGRAN=2 additionally
#     arms the upstream #967 uniform splits_by_blocksize re-split +
#     multi-group launch count at the _int3c2e_ip1_evaluator fill
#     machinery and df/grad/rhf.py's ip1 sum-kernel loop (see
#     _taskgran_resplit_splits below).  The Int3c2eOpt.int3c2e_evaluator
#     aux packing (upstream's bdiv:216 hunk) is a DISCLOSED NON-PORT of
#     (c): it changes aux_offsets semantics for every aux_batch_size
#     caller (uhf/tdrhf/hessian/mp/pbc) whose consumer loops this card
#     does not port.
# Long indivisible tasks (whole homogeneous aux groups, 4B/16B-pair
# blocks) leave workers idle as a launch finishes; the smaller ranges
# expose more independently schedulable work.  Armed differences are
# launch-geometry only: the ip1/j3c atomicAdd interleaving class.  Our
# VRAM-driven _split_l_ctr_pattern bounds (rhf.py:432 lineage, bdiv:746
# whole-group reject) are untouched by every level.
_TASKGRAN = None            # cached env value; None = not yet read
_TASKGRAN_PAIRCAP_SETS = 0  # armed pair-cap geometries differing from verbatim
_TASKGRAN_AUX16_SPLITS = 0  # armed 16-shell aux splits differing from verbatim
_TASKGRAN_RESPLITS = 0      # armed #967 repacks differing from verbatim units

def _taskgran():
    '''The task-granularity level (opt-in, DF-GRAD-TASKGRAN-01/-967-01;
    default off = verbatim).

    GPU4PYSCF_DF_TASKGRAN=1 arms the #970 mechanisms (a)/(b); =2 arms
    them plus the #967 uniform re-split (c); anything else keeps every
    site's verbatim expression value-identical.
    '''
    global _TASKGRAN
    if _TASKGRAN is None:
        raw = os.environ.get('GPU4PYSCF_DF_TASKGRAN', '')
        if raw == '1':
            _TASKGRAN = 1
        elif raw == '2':
            _TASKGRAN = 2
        elif raw in ('', '0'):
            _TASKGRAN = 0
        else:
            warnings.warn(f'GPU4PYSCF_DF_TASKGRAN={raw!r} ignored; '
                          'expected 1 (#970) or 2 (#970 + the #967 '
                          'uniform re-split) to arm the task-granularity '
                          'launch geometry or 0 (verbatim, the default)')
            _TASKGRAN = 0
    return _TASKGRAN

def _taskgran_pair_batch(nsp0, mult):
    '''(a) The #970 pair cap for one aggregate_shl_pairs site; returns
    the npairs value the caller passes on.  nsp0 is the site's
    ``nsp_per_block[0]`` -- the per-(li,lj) capacity table for s-type
    aux (a 2-D ndarray at the gout_width sites; a scalar in unit use)
    -- so the cap applies elementwise, exactly the shape class the
    verbatim ``nsp0*mult`` expression and upstream #970 both pass.

    Knob off (or the strict fp64-reference context active): the
    verbatim ``nsp0*mult`` value with zero engagement.  Armed:
    ``min(nsp0*4, max(nsp0//4, 16))`` elementwise (upstream #970); when
    the capped table equals the verbatim one (e.g. all capacities 4 at
    a 4x site) the geometry is identical and no engagement is counted,
    so drivers can assert the counter exactly.  mult is the site's
    verbatim multiplier -- 4 at the upstream-mirror sites, 16 at
    contract_dm, which the card arms with the same 4B-based cap
    (upstream leaves 16x sites raw; disclosed deviation, default off).
    '''
    from gpu4pyscf.dft import numint as _numint
    if _taskgran() >= 1 and not _numint._strict_fp64_reference_active:
        cap = np.minimum(nsp0*4, np.maximum(nsp0//4, 16))
        if not np.array_equal(cap, nsp0*mult):
            global _TASKGRAN_PAIRCAP_SETS
            _TASKGRAN_PAIRCAP_SETS += 1
            return cap
    return nsp0*mult

def _taskgran_aux16_offsets(auxmol, laux):
    '''(b) The #970 fixed <=16-shell aux subdivision of the J-only
    gradient ksh_offsets; returns the CPU offsets array.

    Knob off / strict reference: the verbatim
    ``np.append(0, np.cumsum(auxmol.l_ctr_counts))`` values.  Armed:
    ``_split_l_ctr_groups(..., [16]*(laux+1))`` -- at most 16 shells
    per task regardless of memory (a scheduling bound; the VRAM-driven
    _split_l_ctr_pattern stays the memory bound on the hybrid route).
    The engagement counter registers only real offset changes.
    '''
    from gpu4pyscf.dft import numint as _numint
    raw = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    if _taskgran() >= 1 and not _numint._strict_fp64_reference_active:
        _, aux_counts = _split_l_ctr_groups(
            auxmol.uniq_l_ctr, auxmol.l_ctr_counts, [16]*(laux+1))
        offsets = np.append(0, np.cumsum(aux_counts))
        if len(offsets) != len(raw) or not np.array_equal(offsets, raw):
            global _TASKGRAN_AUX16_SPLITS
            _TASKGRAN_AUX16_SPLITS += 1
        return offsets
    return raw

def _taskgran_resplit_active():
    '''(c) Whether the #967 uniform re-split is armed for this call
    (DF-GRAD-TASKGRAN-967-01; level >= 2 and not the strict
    fp64-reference context).'''
    from gpu4pyscf.dft import numint as _numint
    return _taskgran() >= 2 and not _numint._strict_fp64_reference_active

def _taskgran_resplit_splits(aux_offsets, batch_size):
    '''(c) The #967 uniform splits_by_blocksize re-split of the launch
    aux batches; returns the split-index list over aux_offsets.

    Knob level < 2 (or the strict fp64-reference context active): the
    verbatim one-group-per-launch units ``list(range(n+1))`` --
    value-identical iteration.  Armed: consecutive aux groups are packed
    up to batch_size functions per launch (upstream #967); every packed
    unit spans whole groups (group boundaries and function ordering are
    preserved), so when the input groups each span at most batch_size
    functions -- the post-_split_l_ctr_pattern invariant at both armed
    sites -- every packed unit also spans at most batch_size functions:
    the VRAM admission bound of the verbatim one-group-per-launch
    buffers cannot grow.  The engagement counter registers only real
    unit-count changes.
    '''
    n_units = len(aux_offsets) - 1
    if _taskgran_resplit_active():
        splits = splits_by_blocksize(np.asarray(aux_offsets), batch_size)
        if len(splits) - 1 != n_units:
            global _TASKGRAN_RESPLITS
            _TASKGRAN_RESPLITS += 1
        return splits
    return list(range(n_units + 1))

def aux_e2(mol, auxmol, omega=None, lr_factor=None, sr_factor=None):
    r'''
    3-center integrals (ij|k). The auxiliary basis functions are
    placed at the second electron.
    '''
    int3c2e_opt = _molecular_int3c2e_opt(mol, auxmol)
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
        reorder_aux=True, cart=mol.cart,
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)[:2]
    aux_coef = int3c2e_opt.aux_coeff
    aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
    aux_coef[aux_sorting] = tmp
    j3c = eval_j3c()
    j3c = j3c.dot(aux_coef)

    nao = mol.nao
    naux = auxmol.nao
    pair_address = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)[0]
    rows, cols = divmod(pair_address, nao)
    out = cp.zeros((nao, nao, naux))
    out[cols,rows] = j3c
    out[rows,cols] = j3c
    return out

def compressed_aux_e2(mol, auxmol, omega=None, lr_factor=None, sr_factor=None):
    r'''
    Returns compressed_int3c, rows, cols. The compressed_int3c stores the
    3-center integrals (ij|k) compressed on the orbital-pair dimensions.
    The addresses of the non-zero pairs are stored in the rows and cols indices.
    The 3-center integral tensor can be restored by:
        int3c[rows,cols] = compressed_int3c
    '''
    int3c2e_opt = _molecular_int3c2e_opt(mol, auxmol)
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    eval_j3c, aux_sorting = int3c2e_opt.int3c2e_evaluator(
        reorder_aux=True, cart=mol.cart,
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)[:2]
    aux_coef = int3c2e_opt.auxmol.ctr_coeff
    aux_coef, tmp = cp.empty_like(aux_coef), aux_coef
    aux_coef[aux_sorting] = tmp
    j3c = eval_j3c()
    j3c = j3c.dot(aux_coef)
    pair_address = int3c2e_opt.pair_and_diag_indices(cart=mol.cart)[0]
    rows, cols = divmod(pair_address, mol.nao)
    return j3c, rows, cols

def contract_int3c2e_dm(mol, auxmol, dm, hermi=0, int3c2e_opt=None):
    if int3c2e_opt is None:
        int3c2e_opt = _molecular_int3c2e_opt(mol, auxmol)
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    dm = int3c2e_opt.mol.apply_C_mat_CT(dm)
    auxvec = int3c2e_opt.contract_dm(dm, hermi)
    return int3c2e_opt.auxmol.apply_CT_dot(auxvec, axis=-1)

def contract_int3c2e_auxvec(mol, auxmol, auxvec, sort_output=True,
                            int3c2e_opt=None):
    if int3c2e_opt is None:
        int3c2e_opt = _molecular_int3c2e_opt(mol, auxmol)
    if int3c2e_opt is None:
        int3c2e_opt = Int3c2eOpt(mol, auxmol).build()
    auxvec = int3c2e_opt.auxmol.apply_C_dot(auxvec, axis=-1)
    vj = int3c2e_opt.contract_auxvec(auxvec)
    if sort_output:
        vj = int3c2e_opt.mol.apply_CT_mat_C(vj)
    return vj

def _sort_pair_ij_order(ish0, nish, jsh0, njsh, nbas, tile=4):
    '''Host reproduction of PBCsort_pair_ij's enumeration order
    (lib/pbc/supmol_sr_estimator.cu: sort_pair_ij_kernel).  Returns
    ish*nbas+jsh as int64 in the exact order the device kernel writes,
    including partial i/j tiles; the tiled order is preserved so the
    q-cond estimator input array stays bit-identical to the reference
    assembly path.
    '''
    parts = []
    for i_tile in range((nish + tile - 1) // tile):
        nish_rem = min(tile, nish - i_tile * tile)
        for j_tile in range((njsh + tile - 1) // tile):
            njsh_rem = njsh - j_tile * tile
            if njsh_rem >= tile:
                seg = np.arange(nish_rem * tile)
                ir, jr = seg // tile, seg - seg // tile * tile
            else:
                seg = np.arange(nish_rem * njsh_rem)
                ir, jr = seg // njsh_rem, seg - seg // njsh_rem * njsh_rem
            i = ish0 + i_tile * tile + ir
            j = jsh0 + j_tile * tile + jr
            parts.append(i.astype(np.int64) * nbas + j)
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)

def _screen_bas_pairs_qcond(mol, rys_envs, precision=1e-14, tile=4, tril=True):
    '''Screened shell-pair lists for Int3c2eOpt.build: returns
    {group_key: uint32 pair array} equivalent to
    {k: v[0] for k, v in _cache_q_cond_and_non0pairs(...).items()}.

    Int3c2eOpt.build re-sorts the pairs (cp.sort) and discards the
    q-cond values, so the descending-q ordering _group_by_split_points
    computes for the JK screening engines is wasted work here (measured
    2026-10-06: 41-73 ms of a 60-102 ms build on consumer cells).  This
    path keeps only the screening decision:

    * pair assembly runs host-side in PBCsort_pair_ij's tiled order
      (bit-identical input to the estimator kernel);
    * the SAME int2e_qcond_estimator kernel is called with the same
      scheme/omega factors as the reference;
    * the keep predicate q_cond > log(precision) is exactly the
      bin >= 1 subset _group_by_split_points retains
      (cp.searchsorted(split_points, q, side='left') >= 1
      <=> q > split_points[0] = log(precision)).

    The result is the identical SET of surviving pairs per group; the
    groups are returned pre-sorted host-side (CDERI-B2D-HOSTPATH-01 C2),
    bit-identical to the reference path's device sort.
    '''
    from gpu4pyscf.pbc.gto import int1e
    omega = mol.omega
    ls = np.arange(LMAX+1)
    li = ls[:,None]
    lj = ls
    lij = li + lj
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nroots = lij + 1
    if omega < 0:
        nroots *= 2

    SIZEOF_FLOAT = ctypes.sizeof(ctypes.c_float)
    gout_width = 29
    unit = (li+1)*(lj+1)*2 + (li+1)*(lj+1)*(lij+1) + 6 + nroots*2
    shm_size = 1024 * 48 - 1024
    nsp_max = _nearest_power2(shm_size // (unit*SIZEOF_FLOAT))
    gout_size = nfi * nfj
    gout_stride = (gout_size+gout_width-1) // gout_width
    gout_stride = _nearest_power2(gout_stride, return_leq=False)
    nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_per_block < nsp_max, nsp_per_block, nsp_max)
    gout_stride = THREADS // nsp_per_block
    gout_stride_dev = cp.asarray(gout_stride, dtype=np.int32)
    shm_size = nsp_per_block * (unit*SIZEOF_FLOAT)
    # (pp|pp) requires more shm than this estimation. 5888 is the required size
    max_shm_size = max(shm_size.max(), 5888*SIZEOF_FLOAT)

    l_ctr_offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
    nbas = mol.nbas
    uniq_l = mol.uniq_l_ctr[:,0]
    n_groups = np.count_nonzero(uniq_l <= LMAX)
    if tril:
        pair_keys = [(i, j) for i in range(n_groups) for j in range(i+1)]
    else:
        pair_keys = [(i, j) for i in range(n_groups) for j in range(n_groups)]

    ovlp_mask = int1e._shell_overlap_mask(mol, precision=precision**2)
    if tril:
        ovlp_mask = cp.tril(ovlp_mask)
    ovlp_mask = cp.asnumpy(ovlp_mask.ravel())

    # Host-side pair assembly in the device kernel's tiled order; integer
    # indices only, so the values are bit-identical to the elementwise
    # device assembly.
    parts = []
    for i, j in pair_keys:
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        pij = _sort_pair_ij_order(ish0, ish1-ish0, jsh0, jsh1-jsh0, nbas, tile)
        parts.append(pij[ovlp_mask[pij]])
    if parts:
        # CDERI-B2D-HOSTPATH-01 (C2): keep the host mirror at the SAME
        # uint32 dtype the device array carries (the tiled pair assembly
        # yields int64; values are non-negative and < 2^32 so the cast is
        # value-identical to the device-side cast below).
        all_pairs_np = np.concatenate(parts).astype(np.uint32)
        all_pairs = cp.asarray(all_pairs_np, dtype=np.uint32)
    else:
        all_pairs_np = np.empty(0, dtype=np.uint32)
        all_pairs = cp.empty(0, dtype=np.uint32)
    offsets = np.append(0, np.cumsum([len(p) for p in parts]))

    q_out = None
    if len(all_pairs) > 0:
        shl_pair_offsets = []
        sp0 = sp1 = 0
        for (i, j), part in zip(pair_keys, parts):
            sp0, sp1 = sp1, sp1 + len(part)
            nspb = THREADS // int(gout_stride[uniq_l[i], uniq_l[j]]) * 8
            shl_pair_offsets.append(np.arange(sp0, sp1, nspb, dtype=np.int32))
        shl_pair_offsets.append(np.int32(sp1))
        shl_pair_offsets = cp.asarray(np.hstack(shl_pair_offsets), dtype=np.int32)

        lr_factor = sr_factor = 1
        if omega < 0:
            lr_factor = 0
        if omega > 0:
            sr_factor = 0
        q_out = cp.empty(len(all_pairs), dtype=np.float32)
        libvhf_rys.int2e_qcond_estimator.restype = ctypes.c_int
        err = libvhf_rys.int2e_qcond_estimator(
            ctypes.cast(q_out.data.ptr, ctypes.c_void_p),
            ctypes.byref(rys_envs),
            ctypes.c_int(max_shm_size),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.cast(all_pairs.data.ptr, ctypes.c_void_p),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride_dev.data.ptr, ctypes.c_void_p),
            ctypes.c_double(omega),
            ctypes.c_double(lr_factor),
            ctypes.c_double(sr_factor))
        if err != 0:
            raise RuntimeError('int2e_qcond_estimator kernel failed')

    threshold = math.log(precision)
    # CDERI-B2D-HOSTPATH-01 (C2): the keep-mask and the per-group sort run
    # host-side on the numpy mirror of all_pairs (q_out pulled to the host
    # once) instead of per-group cupy compare/getitem/sort kernel chains.
    # Integer arrays only, so the per-group results are bit-identical to the
    # elementwise device path.
    if q_out is not None:
        q_host = cp.asnumpy(q_out)
    else:
        q_host = None
    bas_ij_cache = {}
    for k, key in enumerate(pair_keys):
        seg = all_pairs_np[offsets[k]:offsets[k+1]]
        if q_host is not None and len(seg) > 0:
            seg = seg[q_host[offsets[k]:offsets[k+1]] > threshold]
        bas_ij_cache[key] = np.sort(seg)
    return bas_ij_cache

class Int3c2eOpt:
    def __init__(self, mol, auxmol):
        self.mol = mol
        self.auxmol = auxmol
        self._int3c2e_envs = None
        self.bas_ij_cache = None
        self._bas_ij_np = None

    def build(self, cutoff=1e-14, tril=True):
        mol = self.mol = SortedMole.from_cell(self.mol)
        auxmol = self.auxmol = SortedMole.from_cell(self.auxmol)
        _atm, _bas, _env = conc_env(
            mol._atm, mol._bas, _scale_sp_ctr_coeff(mol),
            auxmol._atm, auxmol._bas, _scale_sp_ctr_coeff(auxmol))
        #NOTE: PTR_BAS_COORD is not updated in conc_env()
        off = _bas[mol.nbas,PTR_EXP] - auxmol._bas[0,PTR_EXP]
        _bas[mol.nbas:,PTR_BAS_COORD] += off
        ao_loc = mol.ao_loc
        aux_loc = auxmol.ao_loc
        ao_loc = cp.asarray(_conc_locs(ao_loc, aux_loc), dtype=np.int32)
        self._int3c2e_envs = RysIntEnvVars.new(
            mol.natm, mol.nbas, _atm, _bas, _env, ao_loc)

        bas_pair_cache = _screen_bas_pairs_qcond(
            mol, self._int3c2e_envs, cutoff, tril=tril)
        # CDERI-B2D-HOSTPATH-01 (C2): the parts arrive pre-sorted host-side;
        # one concatenate + ONE device transfer replaces the per-group
        # cp.sort kernel chains.  Values and dtypes are bit-identical
        # (integer sort of unique pair ids); the per-key device arrays are
        # views of the single transferred block, exactly the slice-of-one-
        # array structure the previous path produced.  The concatenation,
        # the slicing offsets and the final dict all follow the SAME
        # iter_pair_by_l (L-sorted) order the previous comprehension used.
        ordered = list(mol.iter_pair_by_l(bas_pair_cache))
        if ordered:
            all_np = np.concatenate([v for _, v in ordered])
        else:
            all_np = np.empty(0, dtype=np.uint32)
        all_gpu = cp.asarray(all_np)
        bas_ij_cache = {}
        bas_ij_np = {}
        p0 = 0
        for key, part in ordered:
            bas_ij_cache[key] = all_gpu[p0:p0 + len(part)]
            bas_ij_np[key] = part
            p0 += len(part)
        self.bas_ij_cache = bas_ij_cache
        self._bas_ij_np = bas_ij_np
        return self

    @property
    def int3c2e_envs(self):
        _int3c2e_envs = self._int3c2e_envs
        if _int3c2e_envs is None or cp.cuda.device.get_device_id() == _int3c2e_envs.device:
            return self._int3c2e_envs
        return _int3c2e_envs.copy()

    def int3c2e_evaluator(self, ao_pair_batch_size=None, aux_batch_size=None,
                          reorder_aux=False, cart=None, pair_batch_by_l=False,
                          return_clone_context=False, clone_context=None,
                          omega=None, lr_factor=None, sr_factor=None):
        if self._int3c2e_envs is None:
            self.build()
        mol = self.mol
        auxmol = self.auxmol
        omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=omega<0, gout_width=54, cache_cart_idx=True)
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

        if clone_context is None:
            if cart is None:
                cart = mol.mol.cart

            if pair_batch_by_l:
                bas_ij_cache = {k: v for k, v in mol.iter_pair_by_l(self.bas_ij_cache)}
            else:
                bas_ij_cache = self.bas_ij_cache
            bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
                bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 4))
            ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], bas_ij_cache, cart)
            nao_pair = ao_pair_loc[-1].get()

            if ao_pair_batch_size is None or nao_pair <= ao_pair_batch_size:
                pair_splits = np.array([0, len(shl_pair_offsets)-1])
                ao_pair_offsets = np.array([0, nao_pair])

            elif pair_batch_by_l:
                uniq_l = mol.uniq_l_ctr[:,0].tolist()
                pair_batch_sizes = []
                last_key = None
                for (i, j), bas_ij in bas_ij_cache.items():
                    key = (uniq_l[i], uniq_l[j])
                    if key != last_key:
                        last_key = key
                        pair_batch_sizes.append(bas_ij.size)
                    else:
                        pair_batch_sizes[-1] += bas_ij.size
                ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
                pair_splits = np.append(0, np.searchsorted(
                    shl_pair_offsets.get(), np.cumsum(pair_batch_sizes), side='left'))
                ao_pair_offsets = ao_pair_offsets[pair_splits]
                assert max(ao_pair_offsets[1:]-ao_pair_offsets[:-1]) < ao_pair_batch_size

            else:
                ao_pair_offsets = ao_pair_loc[shl_pair_offsets].get()
                pair_splits = splits_by_blocksize(ao_pair_offsets, ao_pair_batch_size)
                ao_pair_offsets = ao_pair_offsets[pair_splits]

            l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
            uniq_l_ctr_aux = auxmol.uniq_l_ctr
            aux_loc = auxmol.ao_loc
            if aux_batch_size is None:
                ksh_offsets_cpu = l_ctr_aux_offsets
                aux_splits = [0, len(ksh_offsets_cpu)-1]
            else:
                l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
                    l_ctr_aux_offsets, uniq_l_ctr_aux, aux_batch_size)
                ksh_offsets_cpu = l_ctr_aux_offsets
                aux_splits = range(len(ksh_offsets_cpu))
            aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
            if reorder_aux:
                aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
            else:
                aux_sorting = slice(aux_loc[-1])

            # Save the context for rebuilding evaluate_j3c on multiple GPUs
            clone_context = (bas_ij_idx, shl_pair_offsets, pair_splits, cart,
                             ksh_offsets_cpu, aux_splits, aux_sorting)
        else:
            # Restore the context for running evaluate_j3c on multiple GPUs
            bas_ij_idx, shl_pair_offsets, pair_splits, cart, \
                    ksh_offsets_cpu, aux_splits, aux_sorting = clone_context

            bas_ij_idx = cp.asarray(bas_ij_idx)
            shl_pair_offsets = cp.asarray(shl_pair_offsets)

            ish, jsh = divmod(bas_ij_idx, mol.nbas)
            ls = mol._bas[:,ANG_OF]
            if cart:
                nf = cp.asarray((ls + 1) * (ls + 2) // 2)
            else:
                nf = cp.asarray(ls * 2 + 1)
            ao_pair_counts = nf[ish] * nf[jsh]
            ao_pair_loc = cp.asarray(cp.append(np.int32(0), ao_pair_counts.cumsum()), dtype=np.int32)
            ao_pair_offsets = ao_pair_loc[shl_pair_offsets[pair_splits]].get()

            aux_loc = auxmol.ao_loc
            aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
            if isinstance(aux_sorting, (np.ndarray, cp.ndarray)):
                # aux_sorting can be a slice() instance
                aux_sorting = cp.asarray(aux_sorting)

        ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
        shl_pair_batches = len(ao_pair_offsets) - 1
        aux_batches = len(aux_offsets) - 1
        logger.debug1(self.mol, 'sp_batches = %d, ksh_batches = %d',
                      shl_pair_batches, aux_batches)

        workers = gpu_specs['multiProcessorCount']
        pool = cp.empty(workers * POOL_SIZE + 1)
        kern = libvhf_rys.fill_int3c2e
        int3c2e_envs = self.int3c2e_envs

        def evaluate_j3c(shl_pair_batch_id=0, aux_batch_id=0, out=None):
            pair_split0 = pair_splits[shl_pair_batch_id]
            pair_split1 = pair_splits[shl_pair_batch_id+1]
            ao_pair_offset = ao_pair_offsets[shl_pair_batch_id]
            nao_pair = ao_pair_offsets[shl_pair_batch_id+1] - ao_pair_offset

            aux_split0 = aux_splits[aux_batch_id]
            aux_split1 = aux_splits[aux_batch_id+1]
            ksh0 = ksh_offsets_cpu[aux_split0]
            ksh1 = ksh_offsets_cpu[aux_split1]
            aux_ao_offset = aux_loc[ksh0]
            naux = aux_loc[ksh1] - aux_ao_offset
            out = ndarray((nao_pair, naux), buffer=out)
            if not cart:
                out[:] = 0.
            if out.size == 0:
                return out
            err = kern(
                ctypes.cast(out.data.ptr, ctypes.c_void_p),
                ctypes.byref(int3c2e_envs),
                ctypes.cast(pool.data.ptr, ctypes.c_void_p),
                ctypes.c_double(omega),
                ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
                ctypes.c_int(shm_size_max),
                ctypes.c_int(pair_split1 - pair_split0),
                ctypes.c_int(aux_split1 - aux_split0),
                ctypes.cast(shl_pair_offsets[pair_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
                ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
                ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
                ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
                ctypes.c_int(ao_pair_offset), ctypes.c_int(aux_ao_offset),
                ctypes.c_int(naux), ctypes.c_int(reorder_aux),
                ctypes.c_int(not cart))
            if err != 0:
                raise RuntimeError('fill_int3c2e kernel failed')
            return out


        if return_clone_context:
            return (evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets,
                    clone_context)
        else:
            return evaluate_j3c, aux_sorting, ao_pair_offsets, aux_offsets

    def int3c2e_bdiv_generator(self, cutoff=1e-14, batch_size=None, verbose=None):
        '''An iterator to generate eri3c blocks using the block-divergent
        integral parallelism
        '''
        warnings.warn('int3c2e_bdiv_generator is deprecated')
        evaluate_j3c, _, _, aux_offsets = self.int3c2e_evaluator(
            aux_batch_size=batch_size, reorder_aux=False)
        aux_batches = len(aux_offsets) - 1
        for batch_id in range(aux_batches):
            yield evaluate_j3c(aux_batch_id=batch_id)

    def create_ao_pair_mapping(self):
        warnings.warn('create_ao_pair_mapping is deprecated')
        return self.pair_and_diag_indices(original_ao_order=True)[0]

    @property
    def coeff(self):
        return self.mol.ctr_coeff

    @property
    def aux_coeff(self):
        return self.auxmol.ctr_coeff

    def contract_dm(self, dm, hermi=0):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        nao = mol.nao
        auxmol = self.auxmol
        assert dm.shape[-1] == nao, 'Requires transforming dm: mol.apply_C_mat_CT(dm)'
        assert dm.dtype == np.float64
        assert dm.flags.c_contiguous
        nbas_aux = auxmol.nbas
        if hermi != 1:
            dm = transpose_sum(dm)

        dm_ndim = dm.ndim
        if dm_ndim == 2:
            dm = dm[None]
        n_dm = len(dm)
        dm = cp.asarray(dm.reshape(n_dm,nao,nao), order='C')

        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=mol.omega<0, cache_cart_idx=True)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 16))
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        int3c2e_envs = self.int3c2e_envs
        naux = auxmol.nao
        vj_aux = cp.zeros((n_dm, naux))
        err = libvhf_rys.contract_int3c2e_dm(
            ctypes.cast(vj_aux.data.ptr, ctypes.c_void_p),
            ctypes.cast(dm.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(naux),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(nbas_aux),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('contract_int3c2e_dm failed')
        if hermi == 1:
            vj_aux *= 2
        if dm_ndim == 2:
            vj_aux = vj_aux[0]
        log.timer_debug1('processing contract_int3c2e_dm', *t0)
        return vj_aux

    def contract_auxvec(self, auxvec):
        if self._int3c2e_envs is None:
            self.build()
        log = logger.new_logger(self.mol)
        t0 = log.init_timer()
        mol = self.mol
        auxmol = self.auxmol
        naux = auxmol.nao
        assert auxvec.shape[-1] == naux

        auxvec_ndim = auxvec.ndim
        auxvec = cp.asarray(auxvec.reshape(-1,naux), order='C')
        n_dm = len(auxvec)
        lmax = mol.uniq_l_ctr[:,0].max()
        laux = auxmol.uniq_l_ctr[:,0].max()

        dm_batch_size = 4
        nfk_max = (laux + 1) * (laux + 2) // 2
        sizeof_float64 = 8
        shm_size = SHM_SIZE - dm_batch_size*nfk_max * sizeof_float64
        nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
            short_range=mol.omega<0, shm_size=shm_size,
            gout_width=36, cache_cart_idx=True)
        shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()
        shm_size_max += dm_batch_size*nfk_max * sizeof_float64
        bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
            self.bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 4))
        gout_stride = cp.asarray(gout_stride, dtype=np.int32)

        l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
        ksh_offsets = cp.asarray(l_ctr_aux_offsets + mol.nbas, dtype=np.int32)
        log.debug1('sp_blocks = %d, ksh_blocks = %d, shm_size = %d B',
                   len(shl_pair_offsets)-1, len(ksh_offsets)-1, shm_size_max)

        int3c2e_envs = self.int3c2e_envs
        nao = mol.nao
        vj = cp.zeros((n_dm, nao, nao))
        err = libvhf_rys.contract_int3c2e_auxvec(
            ctypes.cast(vj.data.ptr, ctypes.c_void_p),
            ctypes.cast(auxvec.data.ptr, ctypes.c_void_p),
            ctypes.c_int(n_dm), ctypes.c_int(naux),
            ctypes.byref(int3c2e_envs), ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(len(ksh_offsets) - 1),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError('contract_int3c2e_auxvec kernel failed')
        log.timer_debug1('processing contract_int3c2e_auxvec', *t0)

        vj = hermi_triu(vj, inplace=True)
        if auxvec_ndim == 1:
            vj = vj[0]
        return vj

    def orbital_pair_cart2sph(self, compressed_eri3c):
        '''Transforms the AO of the compressed eri3c from Cartesian to spherical basis'''
        mol = self.mol
        uniq_l = mol.uniq_l_ctr[:,0]
        bas_ij_cache = {k: v for k, v in mol.iter_pair_by_l(self.bas_ij_cache)}
        bas_ij_idx = mol.aggregate_shl_pairs(bas_ij_cache, 1000000000)[0]
        cart_pair_loc = get_ao_pair_loc(uniq_l, bas_ij_cache, cart=True)
        assert compressed_eri3c.shape[0] == cart_pair_loc[-1].get(), \
                'compressed_eri3c might be already transformed into spherical GTOs'

        sph_pair_loc = get_ao_pair_loc(uniq_l, bas_ij_cache, cart=False)
        nao_pair = sph_pair_loc[-1].get()
        naux = compressed_eri3c.shape[1]
        out = cp.zeros((nao_pair, naux))
        int3c2e_envs = self.int3c2e_envs
        compressed = 1
        libvhf_rys.int3c2e_cart2sph(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(compressed_eri3c.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(sph_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(cart_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(len(bas_ij_idx)),
            ctypes.c_int(naux), ctypes.c_int(mol.nbas),
            ctypes.c_int(0), ctypes.c_int(compressed))
        return out

    def pair_and_diag_indices(self, cart=None, original_ao_order=True):
        '''
        original_ao_order:
            controls whether to produce pair addresses corresponding to the
            original Mole (without sorting basis).
        '''
        mol = self.mol
        if cart is None:
            cart = mol.mol.cart
        nbas = mol.nbas
        ao_loc = mol.ao_loc_nr(cart=cart)
        nao = ao_loc[-1]
        uniq_l = mol.uniq_l_ctr[:,0]
        if cart:
            nf = (uniq_l + 1) * (uniq_l + 2) // 2
        else:
            nf = uniq_l * 2 + 1
        # CDERI-B2D-HOSTPATH-01 (C2): the address/diag index construction is
        # integer-only, so it runs host-side on the numpy mirror of the pair
        # cache (one D2H per group only when the mirror is absent) and
        # transfers the two results once each; dtypes and values are
        # bit-identical to the cupy elementwise path.
        ao_loc = np.asarray(ao_loc)
        carts = [np.arange(n) for n in nf]
        # diag stores the indices for cderi_row that corresponds to
        # the diagonal blocks. Note this index array can contain some of
        # the off-diagonal elements which happen to be the off-diagonal
        # elements while within the diagonal blocks.
        offset = 0
        diag = []
        ao_pair_addresses = []
        parts = self._bas_ij_np
        for (i, j), bas_ij in mol.iter_pair_by_l(self.bas_ij_cache):
            if parts is not None and (i, j) in parts:
                bas_ij = parts[(i, j)]
            else:
                bas_ij = cp.asnumpy(bas_ij)
            ish, jsh = divmod(bas_ij, nbas)
            iaddr = ao_loc[ish,None] + carts[i]
            jaddr = ao_loc[jsh,None] + carts[j]
            ao_pair_addresses.append((iaddr[:,None,:] * nao + jaddr[:,:,None]).ravel())
            if i == j: # the diagonal blocks
                nfi = nf[i]
                idx = np.where(ish == jsh)[0]
                addr = offset + idx[:,None] * (nfi*nfi) + np.arange(nfi*nfi)
                diag.append(addr.ravel())
            offset += len(bas_ij) * nf[i] * nf[j]
        addresses_np = np.hstack(ao_pair_addresses)
        if original_ao_order:
            ao_idx_mapping = np.empty(nao, dtype=int)
            ao_idx_mapping[mol.get_ao_idx(cart)] = np.arange(nao)
            i, j = divmod(addresses_np, nao)
            addresses_np = ao_idx_mapping[i] * nao + ao_idx_mapping[j]
        ao_pair_addresses = cp.asarray(addresses_np)
        diag = cp.asarray(np.hstack(diag))

        return ao_pair_addresses, diag

def _molecular_int3c2e_opt(mol, auxmol, decontract=False):
    '''Identity-keyed molecular Int3c2eOpt cache (CDERI-B2D-HOSTPATH-01 C1).

    Measured 2026-11-14 at HEAD 99b7f4e (b3lyp/testosterone 434-AO, default
    fp64 e2e): Int3c2eOpt.build ran 4x per energy+force -- the SCF get_hcore
    contract_int3c2e_auxvec opt (original mol), the SCF DF.build
    decontracted-sorted opt, the gradient _grad_nuc_without_ecp
    decontracted+fakemol opt, and the gradient jk_energy_per_atom opt, the
    last being a VERBATIM rebuild of the DF.build one.  This cache serves
    byte-identical rebuilds from the first construction: the token is the
    passed mol's _bas/_env/_atm bytes + cart flag (plus the key Mole's own
    token and natm/nao when a sorted view is handed in, mirroring the
    LC-35 ``_int2c2e_opt_cache`` keying) together with the auxmol token and
    the construction variant (original-mol vs decontracted-sorted-mol).
    Any in-place env mutation -- e.g. an active
    ``mol.with_range_coulomb(omega)`` context at a range-separated gradient
    site -- changes the token and fails closed to a fresh build, so an opt
    built outside the context is never served inside it.  The cache lives
    in a module-level WeakKeyDictionary keyed on the passed mol object (NOT
    as an attribute: Mole.dumps() json-serializes __dict__ and the byte
    tokens would break it), so entries die with their Mole.  Returns None
    for non-molecular (periodic) inputs; callers then build fresh exactly
    as before.
    '''
    from gpu4pyscf.gto.mole import SortedGTO
    if not isinstance(mol, gto.Mole) or getattr(mol, 'lattice', None) is not None:
        return None
    if not isinstance(auxmol, gto.Mole) or getattr(auxmol, 'lattice', None) is not None:
        return None
    key = mol.mol if isinstance(mol, SortedGTO) else mol

    def _token(cell):
        return (cell._bas.tobytes(), cell._env.tobytes(),
                cell._atm.tobytes(), bool(cell.cart))

    token = _token(key) + _token(auxmol)
    if mol is not key:
        token = token + _token(mol) + (mol.natm, mol.nao)
    token = (token, bool(decontract))
    entry = _INT3C2E_OPT_CACHE.get(key)
    if entry is not None:
        for tok, opt in entry:
            if tok == token:
                return opt
    if decontract:
        opt = Int3c2eOpt(mol, auxmol)
        opt.mol = SortedMole.from_mol(mol, decontract=True)
        opt.build()
    else:
        opt = Int3c2eOpt(mol, auxmol).build()
    _INT3C2E_OPT_CACHE.setdefault(key, []).append((token, opt))
    return opt

_INT3C2E_OPT_CACHE = weakref.WeakKeyDictionary()

def _conc_locs(ao_loc1, ao_loc2):
    return np.append(ao_loc1[:-1], ao_loc1[-1] + ao_loc2)

class Int3c2eEnvVars(ctypes.Structure):
    _fields_ = [
        ('natm', ctypes.c_int),
        ('nbas', ctypes.c_int),
        ('atm', ctypes.c_void_p),
        ('bas', ctypes.c_void_p),
        ('env', ctypes.c_void_p),
        ('ao_loc', ctypes.c_void_p),
        ('log_cutoff', ctypes.c_float),
    ]

    @classmethod
    def new(cls, natm, nbas, atm, bas, env, ao_loc, log_cutoff):
        obj = Int3c2eEnvVars(natm, nbas, atm.data.ptr, bas.data.ptr,
                             env.data.ptr, ao_loc.data.ptr, log_cutoff)
        # Keep a reference to these arrays, prevent releasing them upon returning
        obj._env_ref_holder = (atm, bas, env, ao_loc)
        obj._device = cp.cuda.device.get_device_id()
        return obj

    def copy(self):
        atm, bas, env, ao_loc = self._env_ref_holder
        atm = cp.asarray(atm)
        bas = cp.asarray(bas)
        env = cp.asarray(env)
        ao_loc = cp.asarray(ao_loc)
        return Int3c2eEnvVars.new(self.natm, self.nbas, atm, bas, env, ao_loc,
                                  self.log_cutoff)

def int3c2e_scheme(*, short_range=False, shm_size=SHM_SIZE, gout_width=None,
                   gout_ndim='ijk', deriv=None, cache_cart_idx=False,
                   angular_inc=None):
    if deriv is None:
        deriv = (0, 0, 0)
    if angular_inc is None:
        angular_inc = sum(deriv)
    i_inc, j_inc, k_inc = deriv

    li = np.arange(LMAX+1)[:,None]
    lj = np.arange(LMAX+1)
    lk = np.arange(L_AUX_MAX+1)[:,None,None]
    nfi = (li + 1) * (li + 2) // 2
    nfj = (lj + 1) * (lj + 2) // 2
    nfk = (lk + 1) * (lk + 2) // 2
    order = angular_inc + li + lj + lk
    nroots = order//2 + 1
    if short_range:
        nroots *= 2
    g_size = (li+1+i_inc)*(lj+1+j_inc)*(lk+1+k_inc)
    unit = g_size*3 + nroots*2 + 6
    shm_size = shm_size - 1024
    nsp_max = _nearest_power2(shm_size // (unit*8))
    nsp_per_block = THREADS
    if gout_width is not None:
        if gout_ndim == 'ij':
            gout_size = nfi * nfj
        elif gout_ndim == 'k':
            gout_size = nfk
        else:
            gout_size = nfi * nfj * nfk
        gout_stride = (gout_size + gout_width-1) // gout_width
        # Round up to the next 2^n
        gout_stride = _nearest_power2(gout_stride, return_leq=False)
        nsp_per_block = THREADS // gout_stride
    nsp_per_block = np.where(nsp_max < nsp_per_block, nsp_max, nsp_per_block)
    gout_stride = cp.asarray(THREADS // nsp_per_block, dtype=np.int32)
    shm_size = nsp_per_block * (unit*8)
    if cache_cart_idx:
        shm_size += (nfi + nfj + nfk) * 3 * 4
    return nsp_per_block, gout_stride, shm_size

def _split_l_ctr_pattern(l_ctr_offsets, uniq_l_ctr, batch_size):
    '''
    Split l_ctr patterns into smaller chunks.
    '''
    l = uniq_l_ctr[:,0]
    nf = (l + 1) * (l + 2) // 2
    l_ctr_counts = l_ctr_offsets[1:] - l_ctr_offsets[:-1]
    if any(l_ctr_counts * nf > batch_size):
        counts = l_ctr_counts.tolist()
        repeats = []
        for i, count in enumerate(l_ctr_counts):
            mxshl_in_batch = max(batch_size // nf[i], 1)
            repeat, remainder = divmod(count, mxshl_in_batch)
            expand = [mxshl_in_batch] * repeat
            if remainder != 0:
                expand.append(remainder)
                repeat += 1
            counts[i] = expand
            repeats.append(repeat)
        l_ctr_counts = np.hstack(counts)
        l_ctr_offsets = np.append(0, np.cumsum(l_ctr_counts))
        uniq_l_ctr = np.repeat(uniq_l_ctr, repeats, axis=0)
    return l_ctr_offsets, uniq_l_ctr

def argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux):
    '''
    The auxiliary functions are sorted to
    [s,s,s,...,px,px,px,...,py,py,py,...,pz,pz,pz,...] than the
    conventional order [s,s,...,px,py,pz,px,py,pz,pz,...]. This function returns
    aux_sorting which maps the addresses of the two storge formats.
    Specifically, array_sss_pxpypz_pxpypz = array_sss_pxpx_pypy_pzpz[aux_sorting]
    '''
    l = uniq_l_ctr_aux[:,0]
    nf = (l + 1) * (l + 2) // 2
    aux0 = aux1 = 0
    aux_sorting = []
    nksh = l_ctr_aux_offsets[1:] - l_ctr_aux_offsets[:-1]
    for k, lk, in enumerate(uniq_l_ctr_aux[:,0]):
        aux0, aux1 = aux1, aux1 + nf[k] * nksh[k]
        aux_sorting.append(np.arange(aux0, aux1).reshape(nf[k], nksh[k]).T.ravel())
    return cp.asarray(np.hstack(aux_sorting))

def get_ao_pair_loc(uniq_l, bas_ij_cache, cart=True):
    '''
    For each primitive shell-pair in bas_ij_idx, ao_pair_loc points to the
    addresses of first element for the contracted pair-GTOs. In each
    shell-pair, there are nfij elements. Note, the nfij elements are
    sorted as [nfj,nfi] (in F-order).
    '''
    if cart:
        nf = (uniq_l + 1) * (uniq_l + 2) // 2
    else:
        nf = uniq_l * 2 + 1
    ao_pair_loc = []
    p0 = p1 = 0
    for (i, j), bas_ij in bas_ij_cache.items():
        nfij = nf[i] * nf[j]
        p0, p1 = p1, p1 + nfij * len(bas_ij)
        # CDERI-B2D-HOSTPATH-01 (C2): host-side aranges; the concatenation
        # moves to the host and transfers once below (bit-identical int32).
        ao_pair_loc.append(np.arange(p0, p1, nfij, dtype=np.int32))
    ao_pair_loc.append(np.int32(p1))
    ao_pair_loc = cp.asarray(np.hstack(ao_pair_loc), dtype=np.int32)
    return ao_pair_loc

def int2c2e(mol, sort_output=True,
            omega=None, lr_factor=None, sr_factor=None):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e
    return int2c2e(mol, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)

def int2c2e_ip1(mol, sort_output=True,
                omega=None, lr_factor=None, sr_factor=None):
    '''2c2e Coulomb integrals for the auxiliary basis set'''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1
    return int2c2e_ip1(mol, sort_output=sort_output,
                       omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)

def int2c2e_ip1_per_atom(auxmol, dm, omega=None, lr_factor=None, sr_factor=None):
    '''Computes the first-order derivatives of the Coulomb energy per atom for
    the auxiliary basis set'''
    from gpu4pyscf.pbc.df.int2c2e import int2c2e_ip1_per_atom
    return int2c2e_ip1_per_atom(
        auxmol, dm, omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)

def _create_pair_recontraction(mol, int3c2e_context):
    assert isinstance(mol, SortedMole)
    recontract_bas = cp.asnumpy(mol.recontract_bas)
    recontract_coef = cp.asnumpy(mol.recontract_coef)
    recontraction_idx = cp.asnumpy(mol.recontraction_idx)

    orig_shell_for_sorted_bas = np.empty(mol.nbas, dtype=np.int32)
    orig_shell_for_sorted_bas[recontraction_idx] = \
            np.repeat(np.arange(mol.mol.nbas), recontract_bas[:,NPRIM_OF])

    # Generate np.hstack([np.arange(i) for i in recontract_bas[:,NPRIM_OF]])
    nprims = recontract_bas[:,NPRIM_OF]
    prim_offsets = np.append(0, np.cumsum(nprims))
    prim_id_within_shell = np.empty(mol.nbas, dtype=np.int32)
    prim_id_within_shell[recontraction_idx] = \
            np.arange(mol.nbas) - np.repeat(prim_offsets[:-1], nprims)

    nctrs = recontract_bas[orig_shell_for_sorted_bas, NCTR_OF]

    bas_ij_idx, shl_pair_offsets, pair_splits, cart = int3c2e_context[:4]
    bas_ij_splits = shl_pair_offsets[pair_splits].get()
    bas_ij_batches = cp.split(bas_ij_idx, bas_ij_splits[1:-1])

    l = mol._bas[:,ANG_OF]
    if cart:
        nf = (l + 1) * (l + 2) // 2 * nctrs
    else:
        nf = (l * 2 + 1) * nctrs

    ao_pair_counts = []
    contracted_ao_pair_counts = []
    recontraction_params = []

    nbas_sorted = mol.nbas
    ao_loc = mol.mol.ao_loc
    nao = int(ao_loc[-1])
    output_lut = np.full((nao, nao), -1, dtype=np.int32)
    pair_addresses = np.empty(nao**2, dtype=np.int32)
    offset = 0
    for batch_id, bas_ij_idx in enumerate(bas_ij_batches):
        bas_ij_idx = cp.asnumpy(bas_ij_idx)
        i, j = divmod(bas_ij_idx, nbas_sorted)
        # For diagonal blocks of mol, only the low-triangular part of the
        # corresonding integrals in sorted_mol are generated. They and their
        # transpose are both used in the contraction expansion.
        # *2 to ensure sufficient buffer size.
        size = np.dot(nf[i], nf[j]) * 2
        inp_idx = np.empty(size, dtype=np.int32)
        out_idx = np.empty(size, dtype=np.int32)
        coef = np.empty(size, dtype=np.float64)
        count = ctypes.c_int(0)
        cderi_npairs = libvhf_rys.pair_recontraction_info(
            inp_idx.ctypes, out_idx.ctypes, coef.ctypes, ctypes.byref(count),
            pair_addresses[offset:].ctypes, output_lut.ctypes,
            bas_ij_idx.ctypes, ctypes.c_int(len(bas_ij_idx)),
            orig_shell_for_sorted_bas.ctypes, prim_id_within_shell.ctypes,
            recontract_bas.ctypes, recontract_coef.ctypes,
            ao_loc.ctypes,
            ctypes.c_int(nbas_sorted),
            ctypes.c_int(nao))
        count = count.value
        ao_pair_counts.append(count)
        contracted_ao_pair_counts.append(cderi_npairs)
        recontraction_params.append(
            (inp_idx[:count], out_idx[:count], coef[:count]))
        offset += cderi_npairs
    pair_addresses = pair_addresses[:offset]

    def recontract(batch_id, j3c, out=None):
        naux = j3c.shape[1]
        out_count = contracted_ao_pair_counts[batch_id]
        out = ndarray((out_count, naux), buffer=out)
        if out_count == 0:
            return out
        out[:] = 0.
        inp_idx, out_idx, coef = recontraction_params[batch_id]
        inp_idx = asarray(inp_idx)
        out_idx = asarray(out_idx)
        coef = asarray(coef)
        count = len(inp_idx)
        err = libvhf_rys.recontract_ao_pair(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.cast(j3c.data.ptr, ctypes.c_void_p),
            ctypes.cast(out_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(inp_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(coef.data.ptr, ctypes.c_void_p),
            ctypes.c_int(naux), ctypes.c_int(count))
        if err != 0:
            raise RuntimeError('recontract_ao_pair kernel failed')
        return out
    return recontract, ao_pair_counts, contracted_ao_pair_counts, pair_addresses

def int3c2e_scheme_ip1(omega=0, gout_width=None):
    if omega is None:
        omega = 0
    return int3c2e_scheme(
        short_range=omega<0, gout_width=gout_width, deriv=(1,0,0))

def int3c2e_scheme_ipaux(omega=0, gout_width=None):
    if omega is None:
        omega = 0
    return int3c2e_scheme(
        short_range=omega<0, gout_width=gout_width, deriv=(0,0,1))

def _int3c2e_ip1_evaluator(int3c2e_opt, scheme, batch_size,
                           kern='fill_int3c2e_ip1',
                           omega=None, lr_factor=None, sr_factor=None):
    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)
    nsp_per_block, gout_stride, shm_size = scheme
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 4))
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    nao_pair = int(ao_pair_loc[-1].get())

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    uniq_l_ctr_aux = auxmol.uniq_l_ctr
    aux_loc = auxmol.ao_loc
    # The per-batch j3c tensor is 3 * nao_pair * naux_in_batch * 8 bytes;
    # a fixed shell count per batch OOM'd at ~3300 AO (measured 2026-09-13,
    # Azadirachtin def2-qzvp: ~52 GB requested). Bound naux per batch to a
    # quarter of free VRAM.
    mem_avail = get_avail_mem(exclude_memory_pool=True)
    fns_per_shell_max = (laux + 1) * (laux + 2) // 2   # cartesian
    naux_batch_max = max(1, int(mem_avail * .25) // (3 * nao_pair * 8)
                         // fns_per_shell_max)
    batch_size = min(batch_size, naux_batch_max)
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, uniq_l_ctr_aux, batch_size)
    aux_sorting = argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux)
    # assert cp.array_equal(aux_sorting, argsort_aux(l_ctr_aux_offsets, uniq_l_ctr_aux))

    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    # DF-GRAD-TASKGRAN-967-01 (c): GPU4PYSCF_DF_TASKGRAN=2 packs
    # consecutive aux groups up to batch_size functions per launch
    # (upstream #967) instead of one group per launch; the kernel's
    # group-count argument below carries the packed count.  Knob-off
    # lanes iterate value-identically (aux_split1 - aux_split0 == 1).
    aux_splits = _taskgran_resplit_splits(aux_loc[ksh_offsets_cpu], batch_size)
    aux_offsets = aux_loc[ksh_offsets_cpu[aux_splits]]
    kern = getattr(libvhf_rys, kern)
    int3c2e_envs = int3c2e_opt.int3c2e_envs

    def evaluate_j3c(batch_id, out=None):
        aux_split0 = aux_splits[batch_id]
        aux_split1 = aux_splits[batch_id+1]
        ksh0 = ksh_offsets_cpu[aux_split0]
        ksh1 = ksh_offsets_cpu[aux_split1]
        aux_ao_offset = aux_loc[ksh0]
        naux = aux_loc[ksh1] - aux_ao_offset
        out = ndarray((3, nao_pair, naux), buffer=out)
        if out.size == 0:
            return out

        err = kern(
            ctypes.cast(out.data.ptr, ctypes.c_void_p),
            ctypes.byref(int3c2e_envs),
            ctypes.c_double(omega),
            ctypes.c_double(lr_factor), ctypes.c_double(sr_factor),
            ctypes.c_int(shm_size_max),
            ctypes.c_int(len(shl_pair_offsets) - 1),
            ctypes.c_int(aux_split1 - aux_split0),
            ctypes.cast(shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(ksh_offsets_gpu[aux_split0:].data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(ao_pair_loc.data.ptr, ctypes.c_void_p),
            ctypes.c_int(0),
            ctypes.c_int(aux_ao_offset),
            ctypes.c_int(nao_pair),
            ctypes.c_int(naux))
        if err != 0:
            raise RuntimeError(f'{kern} failed')
        return out
    return evaluate_j3c, aux_sorting, aux_offsets

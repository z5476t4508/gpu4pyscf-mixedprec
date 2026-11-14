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
from pyscf import lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import precision
from gpu4pyscf.lib.cupy_helper import contract, asarray, ndarray, get_avail_mem
from gpu4pyscf.grad import uhf as uhf_grad
from gpu4pyscf.df.grad.rhf import (
    int3c2e_scheme, _factorize_dm, _gen_metric_solver, _ip1_fused, _ip1_f32,
    _arm_resp_f32, _j3c_reuse, _capture_grad_j3c_reuse,
    _strict_fp64_reference_active, _note_j3c_reuse_engaged)
from gpu4pyscf.df.int3c2e_bdiv import (
    _split_l_ctr_pattern, argsort_aux, get_ao_pair_loc, int2c2e_ip1_per_atom,
    _taskgran_pair_batch, _taskgran_resplit_splits, _molecular_int3c2e_opt,
    SHM_SIZE, LMAX, L_AUX_MAX, THREADS, libvhf_rys, Int3c2eOpt, int2c2e)
from gpu4pyscf.df import df_jk
from gpu4pyscf.gto.mole import SortedMole

__all__ = ['Gradients']

def _jk_energy_per_atom(int3c2e_opt, dm, j_factor=1, k_factor=1, hermi=0,
                        auxbasis_response=True, verbose=None,
                        omega=None, lr_factor=None, sr_factor=None,
                        j3c_reuse=None):
    '''
    Computes the first-order derivatives of the energy contributions from
    J and K terms per atom.

    j3c_reuse (GRAD-J3C-REUSE-01 UKS twin): the rhf capture pack; rebuilds
    the spin-stacked dm_oo as (dm_factor_l . cderi . dm_factor_r) .
    aux_coef^T per spin -- the exact associativity of the verbatim route.
    None everywhere by default.
    '''
    assert dm.ndim == 3
    if hermi == 2:
        j_factor = 0
    if k_factor == 0:
        from gpu4pyscf.df.grad.rhf import _j_energy_per_atom
        assert omega is None or omega == 0
        return _j_energy_per_atom(int3c2e_opt, dm[0]+dm[1], hermi,
                                  auxbasis_response, verbose) * j_factor

    if j_factor != 0 and (omega is not None and omega != 0):
        raise RuntimeError(
            'Cannot compute J contributions with range-separated Coulomb operator.')

    mol = int3c2e_opt.mol
    auxmol = int3c2e_opt.auxmol
    log = logger.new_logger(mol, verbose)
    t0 = log.init_timer()

    dm_factor_l, dm_factor_r = _factorize_dm(mol, dm, hermi)
    nao, nocc = dm_factor_l.shape[1:]
    log.debug1('dm_factor shape %s', dm_factor_l.shape)

    pair_addresses = int3c2e_opt.pair_and_diag_indices(
        cart=True, original_ao_order=False)[0]
    i_addr, j_addr = divmod(pair_addresses, nao)
    nao_pair = len(pair_addresses)
    naux = auxmol.nao

    mem_free = get_avail_mem(exclude_memory_pool=True)
    mem_avail = mem_free - 2*naux*nocc**2*8 - nao**2*8
    batch_size = max(1, min(naux, int(mem_avail*.5/(nao_pair*8))))
    eval_j3c, aux_sorting, _, aux_offsets = int3c2e_opt.int3c2e_evaluator(
        aux_batch_size=batch_size, reorder_aux=True, cart=True,
        omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
    aux_batches = len(aux_offsets) - 1

    blksize = max(1, min(naux, int(mem_avail*.4/(nao*nao*2*8))//8*8))
    log.debug1('%.3f GB free memory. nao_pair=%d naux=%d batch_size=%d blksize=%d',
               mem_free*1e-9, nao_pair, naux, batch_size, blksize)

    aux0 = aux1 = 0
    if j3c_reuse is None:
        j3c_full = cp.zeros((nao, nao, blksize))
        buf = cp.empty((batch_size, nao_pair))
        buf1 = cp.empty((blksize, nocc, nao))
        j3c_oo = cp.empty((2, naux, nocc, nocc))
        for kbatch in range(aux_batches):
            compressed = eval_j3c(aux_batch_id=kbatch, out=buf)
            naux_in_batch = compressed.shape[1]
            for k0, k1 in lib.prange(0, naux_in_batch, blksize):
                dk = k1 - k0
                aux0, aux1 = aux1, aux1 + dk
                j3c = j3c_full[:,:,:dk]
                j3c[j_addr,i_addr] = j3c[i_addr,j_addr] = compressed[:,k0:k1]
                tmp = ndarray((nocc, nao, dk), buffer=buf1)
                contract('pqr,pi->iqr', j3c, dm_factor_r[0], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[0], out=j3c_oo[0,aux0:aux1])
                contract('pqr,pi->iqr', j3c, dm_factor_r[1], out=tmp)
                contract('iqr,qj->rij', tmp, dm_factor_l[1], out=j3c_oo[1,aux0:aux1])
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
        dm_oo = contract('uv,nvij->nuij', metric, j3c_oo)
        metric = j3c_oo = None
    else:
        # GRAD-J3C-REUSE-01 UKS twin (armed): the rhf pack semantics with
        # spin-stacked factors -- original-AO-basis scatter + per-spin GEMM
        # pairs, aux_coef mapping the reduced F axis onto sorted-aux rows.
        _note_j3c_reuse_engaged()
        cderi, pair_addr0, aux_coef, nao0 = j3c_reuse
        i_addr0, j_addr0 = divmod(pair_addr0, nao0)
        dm_factor_l0, dm_factor_r0 = df_jk.factorize_dm(dm, hermi)
        if dm_factor_r0 is None:
            dm_factor_r0 = dm_factor_l0
        if hasattr(dm, 'symmetrize'):
            if dm.symmetrize == 1:
                dm_factor_l0, dm_factor_r0 = (
                    cp.concatenate([dm_factor_l0, dm_factor_r0], axis=-1),
                    cp.concatenate([dm_factor_r0, dm_factor_l0], axis=-1))
            elif dm.symmetrize == 2:
                dm_factor_l0, dm_factor_r0 = (
                    cp.concatenate([dm_factor_l0, dm_factor_r0], axis=-1),
                    cp.concatenate([dm_factor_r0, -dm_factor_l0], axis=-1))
        nocc0 = dm_factor_l0.shape[-1]
        assert dm_factor_l0.shape[-2] == nao0 and nocc0 == nocc
        nF = cderi.shape[0]
        blksize0 = max(1, min(nF, int(mem_avail*.4/(nao0*nao0*2*8))//8*8))
        X = cp.empty((2, nF, nocc, nocc))
        j3c_full = cp.zeros((nao0, nao0, blksize0))
        buf1 = cp.empty((blksize0, nocc, nao0))
        for k0, k1 in lib.prange(0, nF, blksize0):
            dk = k1 - k0
            j3c = j3c_full[:,:,:dk]
            j3c[j_addr0,i_addr0] = j3c[i_addr0,j_addr0] = cderi[k0:k1].T
            tmp = ndarray((nocc, nao0, dk), buffer=buf1)
            contract('pqr,pi->iqr', j3c, dm_factor_r0[0], out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l0[0], out=X[0,k0:k1])
            contract('pqr,pi->iqr', j3c, dm_factor_r0[1], out=tmp)
            contract('iqr,qj->rij', tmp, dm_factor_l0[1], out=X[1,k0:k1])
        j3c_full = buf1 = j3c = tmp = None
        t0 = log.timer_debug1('contract dm (j3c-reuse)', *t0)
        dm_oo = contract('kv,nvij->nkij', aux_coef, X)
        X = None
    if j_factor != 0:
        auxvec = dm_oo.trace(axis1=2, axis2=3).sum(axis=0)

    # contract the derivatives and the pseudo DM/rho
    nsp_per_block, gout_stride, shm_size = int3c2e_scheme(
        short_range=mol.omega<0, gout_width=54, deriv=(1,0,0))
    gout_stride = cp.asarray(gout_stride, dtype=np.int32)
    lmax = mol.uniq_l_ctr[:,0].max()
    laux = auxmol.uniq_l_ctr[:,0].max()
    shm_size_max = shm_size[:laux+1,:lmax+1,:lmax+1].max()

    # DF-GRAD-TASKGRAN-UKS-PORT-01 (a): GPU4PYSCF_DF_TASKGRAN>=1 caps the
    # ip1 sum-kernel pair blocks at min(nsp*4, max(nsp//4, 16))
    # elementwise (upstream #970 arms this uhf-local site; the RKS twin
    # is df/grad/rhf.py).  Knob-off lanes are value-identical.
    # len(shl_pair_offsets)-1 is this launch's grid x-dimension
    # (launch-count instrumentation).
    bas_ij_idx, shl_pair_offsets = mol.aggregate_shl_pairs(
        int3c2e_opt.bas_ij_cache, _taskgran_pair_batch(nsp_per_block[0], 4))
    ao_pair_loc = get_ao_pair_loc(mol.uniq_l_ctr[:,0], int3c2e_opt.bas_ij_cache)
    aux_loc = auxmol.ao_loc

    l_ctr_aux_offsets = np.append(0, np.cumsum(auxmol.l_ctr_counts))
    l_ctr_aux_offsets, uniq_l_ctr_aux = _split_l_ctr_pattern(
        l_ctr_aux_offsets, auxmol.uniq_l_ctr, batch_size)
    ksh_offsets_cpu = l_ctr_aux_offsets
    ksh_offsets_gpu = cp.asarray(ksh_offsets_cpu+mol.nbas, dtype=np.int32)
    aux_offsets = aux_loc[ksh_offsets_cpu]
    aux_batches = len(aux_offsets) - 1

    if j_factor != 0:
        dm = mol.apply_C_mat_CT(dm[0]+dm[1])

    int3c2e_envs = int3c2e_opt.int3c2e_envs
    # Same fp32 discipline as the closed-shell path (df/grad/rhf.py): only
    # the int3c2e_ip1 integral kernel runs in float32 on the compressed
    # pair tensor; the j3c@dm contractions and the j2c metric solve above
    # stay in float64 (the metric solve amplifies float32 relative error by
    # the auxiliary-basis condition number straight into the gradient).
    # LC-86: the opt-in ip1-f32 prototype applies the same selection on
    # the fp64 lane (see _ip1_f32()).
    fp32 = precision.get_precision() == 'fp32' or _ip1_f32()
    if fp32:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1_f32
    else:
        kern = libvhf_rys.sum_ejk_int3c2e_ip1
    # DFJK-RESPGEMM-IMPL-01: the fp32 PHASE-2 response-chain twins of the
    # closed-shell path (see _arm_resp_f32 in df/grad/rhf.py; identical
    # arming, counter and fp64 phase-1/metric-solve/auxresp discipline;
    # the spin-stacked operands cast once per grad call)
    dm_oo32, dm_factor_l32, dm_factor_r32 = _arm_resp_f32(
        dm_oo, dm_factor_l, dm_factor_r)
    resp32 = dm_oo32 is not None
    aux0 = aux1 = 0
    # LC-42: opt-in fused ip1 launch (see df/grad/rhf.py for the full
    # rationale and the memory guard semantics); the UKS path shares the
    # same per-batch serialization as the closed-shell J+K path.
    fused = _ip1_fused() and aux_batches > 1 and \
        nao_pair*naux*(4 if fp32 else 8) < get_avail_mem()*0.25
    if fused:
        buf = cp.empty((nao_pair*naux), dtype=cp.float32 if fp32 else cp.float64)
    else:
        buf = cp.empty((nao_pair*batch_size), dtype=cp.float32 if fp32 else cp.float64)
    buf1 = cp.empty((blksize, nao, nao),
                    dtype=cp.float32 if resp32 else cp.float64)
    buf2 = cp.empty(blksize*nao*max(2*nocc, nao),
                    dtype=cp.float32 if resp32 else cp.float64)
    ejk = cp.zeros((mol.natm, 3))
    ejk_aux = cp.zeros((mol.natm, 3))
    # DF-GRAD-TASKGRAN-UKS-PORT-01 (c): armed GPU4PYSCF_DF_TASKGRAN=2
    # packs consecutive aux groups up to batch_size functions per launch
    # (the #967 uniform re-split, the df/grad/rhf.py twin; fork extension
    # -- the upstream #967 uhf-loop status was not re-verifiable on the
    # network-blocked upstream-watch axis).  Knob-off lanes iterate
    # value-identically (kend == kbatch+1, one group per launch).  The
    # fused path already covers all groups in one launch, so the packing
    # is skipped there.  Every packed unit spans whole groups and
    # <= batch_size functions (the post-_split_l_ctr_pattern invariant
    # above), so the per-batch buffer admission bound is unchanged.
    # len(aux_splits)-1 is the armed launch count vs the verbatim
    # aux_batches (launch-count instrumentation).
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
            tmp = ndarray((2,nocc,nao,dk), dtype=buf2.dtype, buffer=buf2)
            beta = 0
            if j_factor != 0:
                # fp64 inputs into the (possibly fp32) tensor: one rounding
                # of the exact fp64 product (same as the closed-shell twin)
                cp.multiply(dm[:,:,None], auxvec[aux0:aux1], out=dm_tensor)
                beta = j_factor
            if resp32:
                # the fp32 response chain (bench V1 mechanism verbatim)
                contract('nrji,nqj->niqr', dm_oo32[:,aux0:aux1],
                         dm_factor_l32, out=tmp)
                contract('niqr,npi->pqr', tmp, dm_factor_r32, -k_factor,
                         beta, out=dm_tensor)
            else:
                contract('nrji,nqj->niqr', dm_oo[:,aux0:aux1], dm_factor_l, out=tmp)
                contract('niqr,npi->pqr', tmp, dm_factor_r, -k_factor, beta, out=dm_tensor)
            # the pseudo-DM handed to the integral kernel is cast to float32
            # when the kernel runs in fp32; the pre-contractions above stay
            # float64 for accuracy.  The armed resp32 chain already holds
            # dm_tensor in the fp32 lane's dtype; on the fp64 kernel lane it
            # reassembles to fp64 (dtype-consistent take, fail-closed on any
            # leak).  The buf2-recycled non-hermi branch below only serves
            # the all-fp64 path -- armed runs materialize dm_tensor1 fresh
            # so the reassembled fp64 values are never re-rounded fp32.
            if fp32 and not resp32:
                dm_tensor = dm_tensor.astype(cp.float32)
            elif resp32 and not fp32:
                dm_tensor = dm_tensor.astype(cp.float64)
            if hermi == 1:
                cp.take(dm_tensor.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            elif fp32 or resp32:
                dm_tensor1 = cp.ascontiguousarray(dm_tensor.transpose(1,0,2))
                dm_tensor1 = dm_tensor1 + dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
            else:
                dm_tensor1 = ndarray((nao,nao,dk), dtype=buf2.dtype, buffer=buf2)
                dm_tensor1[:] = dm_tensor.transpose(1,0,2)
                dm_tensor1[:] += dm_tensor
                cp.take(dm_tensor1.reshape(-1,dk), pair_addresses, axis=0,
                        out=compressed[:,k0:k1])
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
        if dm_factor_l is dm_factor_r:
            dm_aux = contract('nrij,nsij->rs', dm_oo, dm_oo,
                              alpha=-k_factor, beta=j_factor, out=dm_aux)
        else:
            dm_aux = contract('nrij,nsji->rs', dm_oo, dm_oo,
                              alpha=-k_factor, beta=j_factor, out=dm_aux)
        dm_aux = dm_aux[aux_sorting[:,None], aux_sorting]
        ejk_aux -= cp.asarray(
            int2c2e_ip1_per_atom(auxmol, dm_aux, omega, lr_factor, sr_factor))
        t0 = log.timer_debug1('contract int2c2e_ip1', *t0)
        ejk += ejk_aux

    ejk = ejk.get()
    return ejk

class Gradients(uhf_grad.Gradients):
    '''Unrestricted density-fitting Hartree-Fock gradients'''

    _keys = {'with_df', 'auxbasis_response'}

    auxbasis_response = True

    def check_sanity(self):
        assert isinstance(self.base, df_jk._DFHF)

    def energy_ee(self, mol, dm):
        return self.jk_energy_per_atom(dm, hermi=1)

    def jk_energy_per_atom(self, dm=None, j_factor=1, k_factor=1, *,
                           omega=None, lr_factor=None, sr_factor=None,
                           hermi=0, verbose=None):
        '''
        Computes the first-order derivatives of the energy per atom for
        j_factor * J_derivatives - k_factor * K_derivatives
        '''
        mf = self.base
        if dm is None: dm = mf.make_rdm1()
        mol = mf.with_df.mol
        auxmol = mf.with_df.auxmol
        # GRAD-J3C-REUSE-01 UKS twin: capture the retained cderi BEFORE the
        # reset releases it (shared rhf knob/capture; inert fallback verbatim)
        j3c_reuse = None
        if (_j3c_reuse() and not _strict_fp64_reference_active()
                and isinstance(dm, cp.ndarray) and dm.ndim == 3):
            j3c_reuse = _capture_grad_j3c_reuse(mf, omega)
        if j3c_reuse is None:
            mf.with_df.reset() # Release GPU memory
        with mol.with_range_coulomb(omega), auxmol.with_range_coulomb(omega):
            # CDERI-B2D-UHFTD-EXT-01: the identity-keyed cache serves the
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

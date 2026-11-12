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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
# Modified by Xiaojie Wu <wxj6000@gmail.com>

import copy
import os
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
import cupy
import numpy
import cupy as cp
from cupyx.scipy.linalg import solve_triangular
from pyscf import lib
from pyscf.scf import dhf
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import cutensor as _cutensor
from gpu4pyscf.lib.cupy_helper import (
    contract, transpose_sum, tag_array, CPArrayWithTag, cholesky)
from gpu4pyscf.dft import rks, uks, numint, gks
from gpu4pyscf.scf import hf, uhf, rohf
from gpu4pyscf.scf.jk import _check_rsh_factors
from gpu4pyscf.scf import ghf
from gpu4pyscf.lib.cupy_helper import asarray, ndarray, get_avail_mem
from gpu4pyscf.lib import multi_gpu
from gpu4pyscf.lib import precision
num_devices = multi_gpu.num_devices

def density_fit(mf, auxbasis=None, with_df=None, only_dfj=False):
    '''For the given SCF object, update the J, K matrix constructor with
    corresponding density fitting integrals.
    Args:
        mf : an SCF object
    Kwargs:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.  If auxbasis is
            None, optimal auxiliary basis based on AO basis (if possible) or
            even-tempered Gaussian basis will be used.
        only_dfj : str
            Compute Coulomb integrals only and no approximation for HF
            exchange. Same to RIJONX in ORCA
    Returns:
        An SCF object with a modified J, K matrix constructor which uses density
        fitting integrals to compute J and K
    Examples:
    '''
    from gpu4pyscf.df import df
    assert isinstance(mf, hf.SCF)

    if with_df is None:
        if isinstance(mf, dhf.UHF):
            with_df = df.DF4C(mf.mol)
        else:
            with_df = df.DF(mf.mol)
        with_df.max_memory = mf.max_memory
        with_df.stdout = mf.stdout
        with_df.verbose = mf.verbose
        with_df.auxbasis = auxbasis

    if isinstance(mf, _DFHF):
        if mf.with_df is None:
            mf.with_df = with_df
        elif getattr(mf.with_df, 'auxbasis', None) != auxbasis:
            #logger.warn(mf, 'DF might have been initialized twice.')
            mf = mf.copy()
            mf.with_df = with_df
            mf.only_dfj = only_dfj
        return mf

    dfmf = _DFHF(mf, with_df, only_dfj)
    return lib.set_class(dfmf, (_DFHF, mf.__class__))

from gpu4pyscf.lib import utils
class _DFHF:
    '''
    Density fitting SCF class
    Attributes for density-fitting SCF:
        auxbasis : str or basis dict
            Same format to the input attribute mol.basis.
            The default basis 'weigend+etb' means weigend-coulomb-fit basis
            for light elements and even-tempered basis for heavy elements.
        with_df : DF object
            Set mf.with_df = None to switch off density fitting mode.
    '''
    to_gpu = utils.to_gpu
    device = utils.device
    __name_mixin__ = 'DF'
    _keys = {'disp', 'screen_tol', 'with_df', 'only_dfj'}

    def __init__(self, mf, dfobj, only_dfj):
        self.__dict__.update(mf.__dict__)
        self._eri = None
        self.direct_scf = False
        self.with_df = dfobj
        self.only_dfj = only_dfj

    def undo_df(self):
        '''Remove the DFHF Mixin'''
        obj = lib.view(self, lib.drop_class(self.__class__, _DFHF))
        del obj.with_df, obj.only_dfj
        return obj

    def reset(self, mol=None):
        self.with_df.reset(mol)
        return super().reset(mol)

    def get_j(self, mol=None, dm=None, hermi=1, omega=None):
        return self.get_jk(mol, dm, hermi, with_j=True, with_k=False, omega=omega)[0]

    def get_k(self, mol=None, dm=None, hermi=1,
              omega=None, lr_factor=None, sr_factor=None):
        omega, lr_factor, sr_factor = _check_rsh_factors(mol, omega, lr_factor, sr_factor)
        if sr_factor == 0:
            # Only LR
            vk = self.get_jk(mol, dm, hermi, False, True, omega=omega)[1]
            vk *= lr_factor
            return vk

        vk = self.get_jk(mol, dm, hermi, False, True)[1]

        if sr_factor != 1:
            vk *= sr_factor

        if lr_factor != sr_factor:
            vklr = self.get_jk(mol, dm, hermi, False, True, omega=abs(omega))[1]
            vklr *= lr_factor - sr_factor
            vk += vklr
        return vk

    def get_jk(self, mol=None, dm=None, hermi=1, with_j=True, with_k=True,
               omega=None, lr_factor=None, sr_factor=None):
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()

        if isinstance(self, ghf.GHF):
            nao = mol.nao
            if with_k and not self.only_dfj:
                factor_l, factor_r = factorize_dm(dm, hermi)
                if factor_r is None:
                    factor_r = factor_l
                n2c, nocc = factor_l.shape[-2:]
                nao = n2c // 2
                factor_l = factor_l.reshape(-1, n2c, nocc)
                factor_r = factor_r.reshape(-1, n2c, nocc)
                la = factor_l[:,:nao]
                lb = factor_l[:,nao:]
                ra = factor_r[:,:nao]
                rb = factor_r[:,nao:]
                is_real = dm.dtype == cp.float64
                if dm.ndim == 2:
                    n_dm = 1
                else:
                    n_dm = len(dm)

            def get_jk_spin_free(mf, mol, dm, hermi, with_j=True, with_k=True,
                                 omega=None, lr_factor=None, sr_factor=None):
                vj = vk = None
                if with_j:
                    vj = mf.with_df.get_jk(dm, hermi, with_k=False)[0]

                if not with_k:
                    return vj, vk

                if self.only_dfj:
                    vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega,
                                      lr_factor=lr_factor, sr_factor=sr_factor)
                    return vj, vk

                if is_real:
                    # leading dimension of dm corresponds to aa,bb,ab,ba
                    factor_l = cp.vstack([la, lb, la, lb])
                    factor_r = cp.vstack([ra, rb, rb, ra])
                else:
                    # leading dimension of dm corresponds to
                    # aaR,bbR,abR,baR,aaI,bbI,abI,baI
                    factor_l = cp.stack([
                        la.real, la.imag, # laR*raR - laI*raI
                        lb.real, lb.imag, # lbR*rbR - lbI*rbI
                        la.real, la.imag, # laR*rbR - laI*rbI
                        lb.real, lb.imag, # lbR*raR - lbI*raI
                        la.real, la.imag, # laR*raI + laI*raR
                        lb.real, lb.imag, # lbR*rbI + lbI*rbR
                        la.real, la.imag, # laR*rbI + laI*rbR
                        lb.real, lb.imag, # lbR*raI + lbI*raR
                    ]).reshape(8,2, n_dm,nao,nocc).transpose(0,2,3,1,4).reshape(8*n_dm, nao, 2*nocc)
                    factor_r = cp.stack([
                        ra.real, -ra.imag,
                        rb.real, -rb.imag,
                        rb.real, -rb.imag,
                        ra.real, -ra.imag,
                        ra.imag, ra.real,
                        rb.imag, rb.real,
                        rb.imag, rb.real,
                        ra.imag, ra.real,
                    ]).reshape(8,2, n_dm,nao,nocc).transpose(0,2,3,1,4).reshape(8*n_dm, nao, 2*nocc)
                dm = tag_array(dm, factor_l=factor_l, factor_r=factor_r)
                vk = self.with_df.get_jk(dm, hermi=0, with_j=False, omega=omega,
                                         lr_factor=lr_factor, sr_factor=sr_factor)[1]
                return vj, vk

            return ghf._get_jk(self, mol, dm, hermi, with_j, with_k,
                               jkbuild=get_jk_spin_free, omega=omega)

        vj = vk = None
        if self.only_dfj:
            if with_j:
                vj = self.with_df.get_jk(mol, dm, hermi, with_k=False)[0]
            if with_k:
                vk = hf.SCF.get_k(self, mol, dm, hermi, omega=omega,
                                  lr_factor=lr_factor, sr_factor=sr_factor)
        else:
            # Full DF mode (DF J + DF K)
            vj, vk = self.with_df.get_jk(dm, hermi, with_j, with_k, omega=omega,
                                         lr_factor=lr_factor, sr_factor=sr_factor)
        return vj, vk

    def Gradients(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Gradients of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
        from gpu4pyscf.dft import ucdft
        if isinstance(self, ucdft.CDFT_UKS):
            from gpu4pyscf.df.grad import ucdft as ucdft_grad
            return ucdft_grad.Gradients(self)
        if isinstance(self, rks.RKS):
            from gpu4pyscf.df.grad import rks as rks_grad
            return rks_grad.Gradients(self)
        if isinstance(self, hf.RHF):
            from gpu4pyscf.df.grad import rhf as rhf_grad
            return rhf_grad.Gradients(self)
        if isinstance(self, uks.UKS):
            from gpu4pyscf.df.grad import uks as uks_grad
            return uks_grad.Gradients(self)
        if isinstance(self, uhf.UHF):
            from gpu4pyscf.df.grad import uhf as uhf_grad
            return uhf_grad.Gradients(self)
        raise NotImplementedError()

    def Hessian(self):
        if self.istype('_Solvation'):
            raise NotImplementedError(
                'Hessian of solvent are not computed. '
                'Solvent must be applied after density fitting method, e.g.\n'
                'mf = mol.RKS().to_gpu().density_fit().PCM()')
        from gpu4pyscf.dft.rks import KohnShamDFT
        if isinstance(self, hf.RHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import rks as rks_hess
                return rks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import rhf as rhf_hess
                return rhf_hess.Hessian(self)
        elif isinstance(self, uhf.UHF):
            if isinstance(self, KohnShamDFT):
                from gpu4pyscf.df.hessian import uks as uks_hess
                return uks_hess.Hessian(self)
            else:
                from gpu4pyscf.df.hessian import uhf as uhf_hess
                return uhf_hess.Hessian(self)
        else:
            raise NotImplementedError

    @property
    def auxbasis(self):
        return getattr(self.with_df, 'auxbasis', None)

    def get_veff(self, mol=None, dm=None, dm_last=None, vhf_last=None, hermi=1):
        '''
        effective potential
        '''
        if mol is None: mol = self.mol
        if dm is None: dm = self.make_rdm1()
        assert not self.direct_scf
        log = logger.new_logger(self)

        if isinstance(self, rohf.ROHF):
            if getattr(dm, 'mo_coeff', None) is not None:
                mo_coeff = cupy.repeat(dm.mo_coeff[None], 2, axis=0)
                mo_occ = cupy.stack([dm.mo_occ>0, dm.mo_occ==2]).astype(numpy.double)
                if dm.ndim == 2:  # RHF DM
                    dm = cupy.repeat(dm[None]*.5, 2, axis=0)
                dm = tag_array(dm, mo_coeff=mo_coeff, mo_occ=mo_occ)
            elif dm.ndim == 2:  # RHF DM
                dm = cupy.repeat(dm[None]*.5, 2, axis=0)

        # for DFT
        if isinstance(self, rks.KohnShamDFT):
            t0 = log.init_timer()
            ni = self._numint
            if isinstance(self, (uhf.UHF, rohf.ROHF)): # UKS
                if self.grids.coords is None:
                    rks.initialize_grids(self, mol, dm[0]+dm[1])
                n, exc, vxc = ni.nr_uks(mol, self.grids, self.xc, dm)
                log.debug('nelec by numeric integration = %s', n)
                if self.do_nlc() and not numint._scf_head_nlc_suppressed():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm[0]+dm[1], hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vj = vj[0] + vj[1]
                    vxc += vj
                    vk *= hyb
                    if abs(omega) > 1e-10:
                        vklr = self.get_k(mol, dm, hermi, omega=omega)
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk
                    exc -= float(cupy.einsum('sij,sji->', dm, vk).real.get()) * .5
                ecoul = float(cupy.einsum('sij,ji->', dm, vj).real.get()) * .5

            elif isinstance(self, hf.RHF):
                rks.initialize_grids(self, mol, dm)
                n, exc, vxc = ni.nr_rks(mol, self.grids, self.xc, dm)
                log.debug('nelec by numeric integration = %s', n)
                if self.do_nlc() and not numint._scf_head_nlc_suppressed():
                    if ni.libxc.is_nlc(self.xc):
                        xc = self.xc
                    else:
                        assert ni.libxc.is_nlc(self.nlc)
                        xc = self.nlc
                    n, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm)
                    exc += enlc
                    vxc += vnlc
                    log.debug('nelec with nlc grids = %s', n)
                t0 = log.timer('vxc', *t0)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vj = self.get_j(mol, dm, hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    vj, vk = self.get_jk(mol, dm, hermi)
                    vxc += vj
                    vk *= hyb
                    if omega != 0:
                        vklr = self.get_k(mol, dm, hermi, omega=abs(omega))
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc -= vk * .5
                    exc -= float(cupy.einsum('ij,ji->', dm, vk).real.get()) * .25
                ecoul = float(cupy.einsum('ij,ji->', dm, vj).real.get()) * .5
            elif isinstance(self, ghf.GHF):
                if hermi == 2:  # because rho = 0
                    n, exc, vxc = 0, 0, 0
                else:
                    max_memory = self.max_memory - lib.current_memory()[0]
                    if ni.collinear[0].lower() != 'm':
                        raise NotImplementedError('Only multi-colinear GKS is implemented for DF')
                    if self.grids.coords is None:
                        self.initialize_grids(mol, dm)
                    
                    if self.grids.coords is None:
                        self.initialize_grids(mol, dm)
                    
                    n, exc, vxc = ni.get_vxc(mol, self.grids, self.xc, dm,
                                             hermi=hermi, max_memory=max_memory)
                    log.debug('nelec by numeric integration = %s', n)
                    t0 = log.timer('vxc', *t0)
                    
                    if self.do_nlc() and not numint._scf_head_nlc_suppressed():
                        if ni.libxc.is_nlc(self.xc):
                            xc = self.xc
                        else:
                            assert ni.libxc.is_nlc(self.nlc)
                            xc = self.nlc
                        n_nlc, enlc, vnlc = ni.nr_nlc_vxc(mol, self.nlcgrids, xc, dm,
                                                      hermi=hermi, max_memory=max_memory)
                        exc += enlc
                        vxc += vnlc
                        log.debug('nelec with nlc grids = %s', n_nlc)

                if not ni.libxc.is_hybrid_xc(self.xc):
                    vk = None
                    vj = self.get_j(mol, dm, hermi)
                    vxc += vj
                else:
                    omega, alpha, hyb = ni.rsh_and_hybrid_coeff(self.xc, spin=mol.spin)
                    if omega == 0:
                        vj, vk = self.get_jk(mol, dm, hermi)
                        vk *= hyb
                    elif alpha == 0:
                        vj = self.get_j(mol, dm, hermi)
                        vk = self.get_k(mol, dm, hermi, omega=-omega)
                        vk *= hyb
                    elif hyb == 0:
                        vj = self.get_j(mol, dm, hermi)
                        vk = self.get_k(mol, dm, hermi, omega=omega)
                        vk *= alpha
                    else:
                        vj, vk = self.get_jk(mol, dm, hermi)
                        vk *= hyb
                        vklr = self.get_k(mol, dm, hermi, omega=omega)
                        vklr *= (alpha - hyb)
                        vk += vklr
                    vxc += vj - vk
                    exc -= cupy.einsum('ij,ji->', dm, vk).real * .5
                    ecoul = cupy.einsum('ij,ji->', dm, vj).real * .5
            else:
                raise NotImplementedError("DF only supports R/U/RO KS.")
            t0 = log.timer('veff', *t0)
            return tag_array(vxc, ecoul=ecoul, exc=exc)

        if isinstance(self, (uhf.UHF, rohf.ROHF)):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vj = vj[0] + vj[1]
            vhf = vj - vk
            ecoul = float(cp.einsum('sij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        elif isinstance(self, hf.RHF):
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vhf = vj - vk * .5
            ecoul = float(cp.einsum('ij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        elif isinstance(self, ghf.GHF): # GHF branch
            vj, vk = self.get_jk(mol, dm, hermi=hermi)
            vhf = vj - vk
            ecoul = float(cp.einsum('ij,ji->', dm, vj).real.get()) * .5
            return tag_array(vhf, ecoul=ecoul)
        else:
            raise NotImplementedError("DF only supports R/U/RO/G HF.")

    def to_cpu(self):
        obj = self.undo_df().to_cpu().density_fit()
        obj = utils.to_cpu(self, obj)
        if hasattr(self, 'collinear'):
            obj.collinear = self.collinear
        if hasattr(self, 'spin_samples'):
            obj.spin_samples = self.spin_samples
        return obj

def _eager_plan_warmup(dfobj, dm_sparse, factor_l, vj, vk, blksize,
                       nao, nocc, n_dm, nspin, buf, buf_dtype,
                       dm_batch_size, scope='all'):
    '''WARMUP-PRELOAD-01 M1 (opt-in GPU4PYSCF_DFJK_EAGER_PLAN_WARMUP,
    default off): populate the contraction plan cache at the EXACT
    per-block shapes/strides/dtypes the upcoming first block loop will
    contract, before the first real contraction.  The cold cutensor
    descriptor/plan host build measured inside the first jk_j_build
    calls (23.6/6.4% clean t_scf, entry 331) then happens here, in an
    explicitly-timed step, and every first-call contraction is served
    from a warmed entry (same plan object -> bitwise results).

    No contraction executes here: plan construction is host-side
    metadata only, and the tril block views come from
    loop(blksize, unpack=False) (pure slicing -- no fill_symmetric, no
    device compute paid twice).  The K-side unpacked operand is
    reproduced from one dummy (nao, nao, blksize) work buffer (the
    real work buffer's transposed view strides), and rhok/vk from the
    real buf/vk views.  Latched once per DF object per device;
    fail-closed on any error (per-call build path untouched, never
    retried, never raised).

    WARMUP-PRELOAD-02 scope='jbuild' (knob value 'jbuild'): warm ONLY
    the j_build family ('np,Lp->nL') -- the entry-332 per-family
    cold-cost table's single expensive family (629/653 ms vs 1-4 ms
    for j_accum/k_build/k_accum, whose warm cost over-paid and failed
    the v1 t_scf gate).  No j_accum operand is allocated and the K
    side is not touched at all in that scope.
    '''
    t0 = time.perf_counter()
    device_id = None
    done = getattr(dfobj, '_plan_warmup_devices', None)
    if done is None:
        done = dfobj._plan_warmup_devices = set()
    try:
        device_id = cp.cuda.Device().id
        if device_id in done:
            return
        cderi_sparse = dfobj._cderi[device_id]
        if not isinstance(cderi_sparse, cp.ndarray):
            # host-resident cderi: loop(unpack=False) would drive the
            # prefetch host->device copies; v1 stays inert there
            logger.debug1(dfobj.mol, 'eager plan warmup skipped: '
                          'host-resident cderi (device %d)', device_id)
            return
        tril_blocks = [blk for _, blk in
                       dfobj.loop(blksize=blksize, unpack=False)]
        if dm_sparse is not None and (scope == 'jbuild' or vj is not None):
            # J side at the real tril views
            dtype_j = numpy.result_type(dm_sparse.dtype, cderi_sparse.dtype)
            for cderi_tril in tril_blocks:
                L = cderi_tril.shape[0]
                _cutensor.warm_contraction('np,Lp->nL', dm_sparse,
                                           cderi_tril)
                if scope == 'jbuild':
                    # scoped: the j_accum family's cold cost is ~1-4 ms
                    # (entry 332); warming it over-pays -- stop here
                    continue
                auxvec = cp.empty((dm_sparse.shape[0], L), dtype=dtype_j)
                _cutensor.warm_contraction('nL,Lp->np', auxvec, cderi_tril,
                                           out=vj)
        if (scope != 'jbuild' and factor_l is not None
                and vk is not None and buf is not None):
            # K side (dm_factor_mode 0): reproduce the unpacked
            # work-buffer view and the first dm-batch rhok/vk views
            work = cp.empty((nao, nao, blksize), dtype=cderi_sparse.dtype)
            i0, i1 = next(iter(lib.prange(0, n_dm, dm_batch_size)))
            for s in range(nspin):
                fac = factor_l[s, i0:i1]
                vacc = vk[s, i0:i1]
                for cderi_tril in tril_blocks:
                    L = cderi_tril.shape[0]
                    cderi_view = work[:, :, :L].transpose(2, 0, 1)
                    rhok = ndarray((i1-i0, nao, nocc, L), dtype=buf_dtype,
                                   buffer=buf)
                    _cutensor.warm_contraction('Lij,njk->nikL', cderi_view,
                                               fac, out=rhok)
                    _cutensor.warm_contraction('nikL,njkL->nij', rhok, rhok,
                                               out=vacc)
        done.add(device_id)
    except Exception:
        if device_id is not None:
            done.add(device_id)   # fail-closed once; never retry, never raise
        _cutensor._warmup_stats['failures'] += 1
        logger.debug(dfobj.mol, 'eager plan warmup failed (fail-closed)')
    finally:
        _cutensor._warmup_stats['host_ms'] += \
            (time.perf_counter() - t0) * 1e3


# SCFJK-KBUILD-IMPL-01 (2026-11-05): the DF K-transform response GEMMs
# of get_jk ('Lij,njk->nikL' jk_k_build + 'nikL,njkL->nij' jk_k_accum)
# carry 0.4225/0.4712 of the jk wall at 819/1042 AO (entry 370) -- the
# largest named interior mechanism above 819 on the SCF side; the same
# GEMM family the delivered grad-side GPU4PYSCF_DFGRAD_RESP_F32 knob
# addresses per-grad-call.  The opt-in GPU4PYSCF_SCF_JK_F32 knob
# (DEFAULT OFF, parse mirroring GPU4PYSCF_SCF_VMAT_F32 /
# GPU4PYSCF_SCF_RHO_F32) arms the EXISTING fp32 branch of get_jk below
# (the exact discipline the auto policy's fp32 head runs: fp32 MO
# factors, fp32 cderi per-block casts or the SP-28 capped mirror, fp32
# rhok buffer and K accumulation, fp64 upcast on return) on the SCF
# head of PURE-fp64 lanes.  'headtail' flips the lane off (single
# trigger) at the same cycle-end |dE| tolerance in scf.hf._kernel_body
# the vmat/rho knobs use, so the convergence tail keeps the exact fp64
# K map (the LC-115 fixed-point restoration argument: an fp32 tail
# displaces the SCF fixed point at large AO; the head/tail split
# preserves it).  '1' parses for parity (lane armed the whole kernel)
# but is NOT the adoption candidate.  Armed by scf.hf._kernel on
# pure-fp64 lanes only (not the auto family, whose fp32 head already
# runs this branch and whose fp64 tail must stay bitwise fp64), never
# on stability_verify's strict fp64-reference siblings; standalone
# get_jk callers (TDDFT/hessian/grad) keep the flag's default False.
# _scf_jk_f32_head_calls counts lane-armed engagements only (knob-off,
# auto-lane and strict deltas are exactly 0).
# SCFJK-WINDOW-01 (2026-11-06): the opt-in companion
# GPU4PYSCF_SCF_JK_F32_ARM_AFTER=N (DEFAULT 0 = arm at kernel entry,
# the SCFJK-KBUILD-IMPL-01 semantics; N >= 1 = the jk lane stays off
# through cycle N so the fp64 prefix is bitwise the knob-off
# trajectory, and scf.hf._kernel_body arms it once at that cycle-end
# latch; the e_diff < switch-tol tail flip is UNCHANGED) tests whether
# the +1-cycle penalty of the entry-armed knob (entry 372: per-cycle
# win -4.7/-7.2% but +1 SCF cycle) originates in the early
# DIIS-building cycles.  Defined for the 'headtail' knob value only
# (warn + 0 otherwise); the vmat/rho knobs keep arm-at-entry.
_SCF_JK_F32 = None       # cached env value; None = not yet read;
                         # False | True ('1') | 'headtail'
_SCF_JK_F32_ARM_AFTER = None   # cached; None = not yet read; int >= 0
_scf_jk_f32_lane = False
_scf_jk_f32_head_calls = 0


def _scf_jk_f32():
    '''Whether to run the DF K-transform contractions of get_jk in
    float32 on the SCF head of pure-fp64 SCF lanes (opt-in
    precision-policy knob, SCFJK-KBUILD-IMPL-01; parse mirrors
    _scf_vmat_f32/_scf_rho_f32).

    GPU4PYSCF_SCF_JK_F32=headtail: the existing fp32 branch of get_jk
    (fp32 MO factors and cderi operands, fp32 K accumulation, fp64
    upcast on return) serves the SCF head cycles only; the tail flips
    back to the exact fp64 K map at the same scf.hf trigger as
    GPU4PYSCF_SCF_VMAT_F32=headtail.  With the knob on, the fp64-lane
    SCF is no longer the bitwise strict fp64 reference (its head
    trajectory differs); strict adjudication keeps knob-off records.'''
    global _SCF_JK_F32
    if _SCF_JK_F32 is None:
        raw = os.environ.get('GPU4PYSCF_SCF_JK_F32', '')
        if raw == '1':
            _SCF_JK_F32 = True
        elif raw == 'headtail':
            _SCF_JK_F32 = 'headtail'
        elif raw in ('', '0'):
            _SCF_JK_F32 = False
        else:
            warnings.warn(f'GPU4PYSCF_SCF_JK_F32={raw!r} ignored; '
                          'expected headtail (fp32 K-transform head, '
                          'exact-fp64 tail) or 1 on pure-fp64 lanes, or '
                          '0 (fp64 everywhere, the default)')
            _SCF_JK_F32 = False
    return _SCF_JK_F32


def _scf_jk_f32_arm_after():
    '''Delayed-arming cycle count for the GPU4PYSCF_SCF_JK_F32=headtail
    knob (SCFJK-WINDOW-01, opt-in precision-policy companion, DEFAULT
    0 = arm at kernel entry).

    GPU4PYSCF_SCF_JK_F32_ARM_AFTER=N (N >= 1) keeps the jk lane off
    through cycle N (the fp64 prefix is bitwise the knob-off
    trajectory) and arms it once at that cycle-end latch in
    scf.hf._kernel_body; the e_diff < switch-tol tail flip is
    unchanged.  Defined for the 'headtail' knob value only: any other
    value (including '1') warns and ignores the delay (arm at entry).
    Non-integer or negative values warn and parse as 0.
    '''
    global _SCF_JK_F32_ARM_AFTER
    if _SCF_JK_F32_ARM_AFTER is None:
        raw = os.environ.get('GPU4PYSCF_SCF_JK_F32_ARM_AFTER', '')
        try:
            value = int(raw) if raw else 0
            if value < 0:
                raise ValueError(raw)
        except ValueError:
            warnings.warn(f'GPU4PYSCF_SCF_JK_F32_ARM_AFTER={raw!r} '
                          'ignored; expected a non-negative integer '
                          'cycle count (0 = arm at kernel entry)')
            value = 0
        if value and _scf_jk_f32() != 'headtail':
            warnings.warn('GPU4PYSCF_SCF_JK_F32_ARM_AFTER is defined '
                          'only for GPU4PYSCF_SCF_JK_F32=headtail; '
                          'ignoring the delayed arming')
            value = 0
        _SCF_JK_F32_ARM_AFTER = value
    return _SCF_JK_F32_ARM_AFTER


# SCFJK-KACC-SYRK-IMPL-01 (2026-11-06): the k_accum back-transform
# contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk) assembles the
# symmetric Gram K = rhok . rhok over the DENSE (nao,nao) output, so
# every off-diagonal duplicates its mirror's MACs (~2x waste; tril
# fraction 0.5006/0.5005 at 819/1042 AO; k_accum = 0.3564/0.3790 of
# the jk wall, entry 370).  BENCH-02 (entry 377) measured the fix at
# the deployed shapes: replacing the dense contraction with a cuBLAS
# dsyrk tril beta-accumulation per block plus ONE mirror fill runs the
# chain 1.3351x/1.5300x faster at 819/1042 while the dense dgemm
# engine control stays engine-neutral (1.0035/1.0019) -- the win is
# purely the symmetry MAC cut -- at reassociation-level numerics
# (max_rel 6.1e-15/5.5e-15; the fp64 contraction-reorder class entry
# 329 measured cycle-preserving at 1.14e-12/4.55e-13 Ha).  The opt-in
# GPU4PYSCF_SCF_JK_KACC_SYRK knob (DEFAULT OFF) engages that form on
# the SCF single-DM lane only: fp64 operands (never the global-fp32
# mode or the armed SCFJK-KBUILD fp32 lane), dm_factor_mode 0 and
# n_dm == 1 (per-spin), dense contract fallback everywhere else; the
# strict fp64-reference context forces it inert.  The returned K is
# mirrored to exact symmetry (the deployed dense output is itself
# bitwise symmetric at the band cells, entry 377 informative);
# _scf_jk_kacc_syrk_calls counts armed engagements only (knob-off and
# inert deltas are exactly 0).  Adoption/manual-decision gates: paired
# FULL-SCF t_scf + cycle-count equality at the band cells (Phase B of
# the card); this knob is NOT a default.
_SCF_JK_KACC_SYRK = None       # cached env value; None = not yet read
_SCF_JK_KACC_SYRK_CALLS = 0    # armed k_accum-syrk engagements


def _scf_jk_kacc_syrk():
    '''Whether to accumulate the DF K back-transform with cuBLAS dsyrk
    on the symmetric K triangle plus a mirror fill (opt-in structural
    knob, SCFJK-KACC-SYRK-IMPL-01; BENCH-02 basis, entry 377).

    GPU4PYSCF_SCF_JK_KACC_SYRK=1 swaps the per-block
    contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk) for
    cupy.cublas.syrk('N', rhok2d, out=vk, lower=True, beta=1) on the
    fp64 single-DM lane (n_dm == 1, dm_factor_mode 0, any nspin) and
    mirrors the accumulated triangle once after the block loop --
    identical values up to fp64 reduction-order reassociation, ~2x
    fewer MACs.  Inert on the global-fp32 mode, the armed fp32
    K-transform lane, multi-DM calls (TDDFT/hessian), other
    dm_factor_modes, and the strict fp64-reference context; the
    deployed contract path is unchanged in all of those.
    '''
    global _SCF_JK_KACC_SYRK
    if _SCF_JK_KACC_SYRK is None:
        raw = os.environ.get('GPU4PYSCF_SCF_JK_KACC_SYRK', '')
        if raw == '1':
            _SCF_JK_KACC_SYRK = True
        elif raw in ('', '0'):
            _SCF_JK_KACC_SYRK = False
        else:
            warnings.warn(f'GPU4PYSCF_SCF_JK_KACC_SYRK={raw!r} ignored; '
                          'expected 1 to enable the symmetric-triangle '
                          'K accumulation, or 0 (the dense contract, '
                          'the default)')
            _SCF_JK_KACC_SYRK = False
    return _SCF_JK_KACC_SYRK


def _dfjk_strict_fp64_reference_active():
    '''Whether the current call runs under the strict fp64-reference
    context (dft.numint.strict_fp64_reference_ctx, armed by
    stability_verify around its exact-fp64 sibling trajectories).  The
    k_accum-syrk knob stays inert there: the sibling must reproduce
    the plain DEFAULT-lane solution.  Lazy import so df.df_jk does not
    pay the dft import at module load; attribute read at call time.'''
    from gpu4pyscf.dft import numint as _numint
    return bool(_numint._strict_fp64_reference_active)


def get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True,
           omega=None, lr_factor=None, sr_factor=None):
    '''
    get jk with density fitting
    '''
    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(omega=omega, lr_factor=lr_factor, sr_factor=sr_factor)
        t1 = log.timer_debug1('init jk', *t0)

    fp32 = precision.get_precision() == 'fp32'
    # SCFJK-KBUILD-IMPL-01: the opt-in lane flag arms the SAME fp32
    # branch below on pure-fp64 SCF lanes (armed/cleared by scf.hf only;
    # default False everywhere else -- standalone callers, the auto
    # family's fp64 tail and strict references are bitwise unchanged).
    # The counter records the lane-armed engagement itself.
    if _scf_jk_f32_lane:
        global _scf_jk_f32_head_calls
        _scf_jk_f32_head_calls += 1
        fp32 = True
    out_cupy = isinstance(dms, cp.ndarray)
    dm_factor_l, dm_factor_r = factorize_dm(dms, hermi)
    symmetrize = getattr(dms, 'symmetrize', 0)

    nspin = 1
    if dm_factor_l.ndim == 4: # for UHF-TDDFT and UHF-hessian
        assert dms.ndim == 4
        nspin = 2

    if dm_factor_r is None:
        dm_factor_mode = 0
    elif dm_factor_l.ndim == dm_factor_r.ndim:
        dm_factor_mode = 1
    elif dm_factor_l.ndim < dm_factor_r.ndim:
        dm_factor_mode = 2
    else: # dm_factor_l.ndim > dm_factor_r.ndim:
        dm_factor_mode = 3

    nao, nocc = dm_factor_l.shape[-2:]
    dms_3d = cp.asarray(dms).reshape(-1,nao,nao)
    n_dm = dms_3d.shape[0] // nspin

    if nocc == 0:
        # dms equals to 0. vj and vk must be all zeros.
        return dms, dms

    if with_j:
        pair_addresses, diags = dfobj._cderi_idx
        rows, cols = divmod(cp.asarray(pair_addresses), nao)
        dm_sparse = dms_3d[:,rows,cols]
        if hermi == 0:
            dm_sparse += dms_3d[:,cols,rows]
        else:
            dm_sparse *= 2
        dm_sparse[:,diags] *= .5

    def proc():
        factor_l = cp.asarray(dm_factor_l).reshape(nspin,-1,nao,nocc)
        factor_r = dm_factor_r
        if factor_r is not None:
            factor_r = cp.asarray(factor_r).reshape(nspin,-1,nao,nocc)
        if fp32:
            factor_l = factor_l.astype(cp.float32)
            if factor_r is not None:
                factor_r = factor_r.astype(cp.float32)

        vj = vk = None
        if with_j:
            _dm_sparse = cp.asarray(dm_sparse)
            if fp32:
                _dm_sparse = _dm_sparse.astype(cp.float32)
            vj = cp.zeros_like(_dm_sparse)

        blksize = dfobj.get_blksize(mem_fraction=0.4)
        if with_k:
            buf_dtype = cp.float32 if fp32 else cp.float64
            vk = cupy.zeros((nspin, n_dm, nao, nao),
                            dtype=buf_dtype)
            mem_avail = get_avail_mem(exclude_memory_pool=True)
            dm_batch_size = int(mem_avail * 0.6 / (blksize*nao*nocc * 8))
            if dm_factor_mode == 1:
                dm_batch_size = dm_batch_size // 2
            if fp32:
                # half word size per buffer entry
                dm_batch_size *= 2
            dm_batch_size = min(dm_batch_size, n_dm)
            assert dm_batch_size > 0
            log.debug1('blksize=%d, dm_batch_size=%d (fp32=%s)',
                       blksize, dm_batch_size, fp32)

            if dm_factor_mode == 0:
                buf = cp.empty((dm_batch_size, blksize * nao*nocc), dtype=buf_dtype)
            elif dm_factor_mode == 1:
                buf = cp.empty((2 * dm_batch_size, blksize * nao*nocc), dtype=buf_dtype)
            else:
                buf = cp.empty((dm_batch_size+1, blksize * nao*nocc), dtype=buf_dtype)
                buf1 = buf[-1]

        # SCFJK-KACC-SYRK-IMPL-01: opt-in symmetric-triangle K
        # accumulation (see the knob block above): fp64 single-DM
        # unsymmetrized lane only; the dense contract stays everywhere
        # else.  syrk_armed records actual engagements (>= 1 block).
        use_syrk_acc = (not fp32 and dm_factor_mode == 0 and n_dm == 1
                        and symmetrize == 0 and _scf_jk_kacc_syrk()
                        and not _dfjk_strict_fp64_reference_active())
        syrk_armed = False
        if use_syrk_acc:
            import cupy.cublas as _cp_cublas
            iu = cp.triu_indices(nao, k=1)

        # SP-28: with the persistent fp32 cderi block cache admitted,
        # iterate its blocks (fp32 tril blocks for J, fp32-unpacked views
        # for K) and skip the per-block casts; bitwise-identical to the
        # per-call _cast_cderi path, fail-closed to it otherwise
        if fp32 and dfobj._get_cderi_f32_blocks(blksize) is not None:
            blocks = dfobj.loop_f32(blksize=blksize, unpack=with_k)
            cast_blocks = False
        else:
            blocks = dfobj.loop(blksize=blksize, unpack=with_k)
            cast_blocks = fp32

        # WARMUP-PRELOAD-01 M1 (opt-in, default off; fp64 clean path,
        # dm_factor_mode 0 only): warm the plan cache at the real
        # first-call shapes before the first real contraction.
        # WARMUP-PRELOAD-02: knob value 'jbuild' restricts the warm to
        # the j_build family (scope), any other enabled value warms all
        # four families.
        if (not fp32 and dm_factor_mode == 0
                and _cutensor._eager_warmup_enabled()):
            _eager_plan_warmup(
                dfobj,
                _dm_sparse if with_j else None,
                factor_l,
                vj if with_j else None,
                vk if with_k else None,
                blksize, nao, nocc, n_dm, nspin,
                buf if with_k else None, buf_dtype, dm_batch_size,
                scope=_cutensor._eager_warmup_scope())

        for cderi, cderi_tril in blocks:
            if cast_blocks and cderi_tril is not None:
                cderi_tril = _cast_cderi(cderi_tril)
            if with_j:
                auxvec = contract('np,Lp->nL', _dm_sparse, cderi_tril)
                contract('nL,Lp->np', auxvec, cderi_tril, beta=1, out=vj)

            if with_k:
                if cast_blocks and cderi is not None:
                    cderi = _cast_cderi(cderi)
                nL = len(cderi)
                for s in range(nspin):
                    if dm_factor_mode == 0:
                        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                            rhok = ndarray((i1-i0,nao,nocc,nL), dtype=buf_dtype, buffer=buf)
                            contract('Lij,njk->nikL', cderi, factor_l[s,i0:i1], out=rhok)
                            if use_syrk_acc:
                                # BENCH-02 form (entry 377): accumulate
                                # the symmetric K triangle via cuBLAS
                                # dsyrk on the 2-D views (row-major
                                # lower incl. diagonal; beta=1 across
                                # blocks); the mirror fill runs once
                                # after the block loop below.
                                _cp_cublas.syrk(
                                    'N', rhok[0].reshape(nao, nocc*nL),
                                    out=vk[s,0], alpha=1.0, beta=1.0,
                                    lower=True)
                                syrk_armed = True
                            else:
                                contract('nikL,njkL->nij', rhok, rhok, beta=1, out=vk[s,i0:i1])
                    elif dm_factor_mode == 1:
                        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                            rhok, rhok1 = ndarray((2,i1-i0,nao,nocc,nL), dtype=buf_dtype, buffer=buf)
                            contract('Lij,njk->nikL', cderi, factor_l[s,i0:i1], out=rhok)
                            contract('Lij,njk->nikL', cderi, factor_r[s,i0:i1], out=rhok1)
                            contract('nikL,njkL->nij', rhok, rhok1, beta=1, out=vk[s,i0:i1])
                    elif dm_factor_mode == 2:
                        rhok = ndarray((nao,nocc,nL), dtype=buf_dtype, buffer=buf1)
                        contract('Lij,jk->ikL', cderi, factor_l[s,0], out=rhok)
                        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                            rhok1 = ndarray((i1-i0,nao,nocc,nL), dtype=buf_dtype, buffer=buf)
                            contract('Lij,njk->nikL', cderi, factor_r[s,i0:i1], out=rhok1)
                            contract('nikL,jkL->nij', rhok, rhok1, beta=1, out=vk[s,i0:i1])
                    else:
                        rhok1 = ndarray((nao,nocc,nL), dtype=buf_dtype, buffer=buf1)
                        contract('Lij,jk->ikL', cderi, factor_r[s,0], out=rhok1)
                        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                            rhok = ndarray((i1-i0,nao,nocc,nL), dtype=buf_dtype, buffer=buf)
                            contract('Lij,njk->nikL', cderi, factor_l[s,i0:i1], out=rhok)
                            contract('nikL,jkL->nij', rhok, rhok1, beta=1, out=vk[s,i0:i1])
                rhok1 = rhok = None
        if syrk_armed:
            # the ONE mirror fill per armed spin plane (upper <- the
            # accumulated lower triangle); index the 2-D view -- the
            # 3-D fancy-index form is the smoke-caught defect class
            # (entry 377 repairs[])
            for s in range(nspin):
                vk[s,0][iu[0], iu[1]] = vk[s,0][iu[1], iu[0]]
            global _SCF_JK_KACC_SYRK_CALLS
            _SCF_JK_KACC_SYRK_CALLS += 1
        return vj, vk

    results = multi_gpu.run(proc, non_blocking=True)

    vj = vk = None
    if with_j:
        vj_sparse = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        if fp32:
            # keep the returned dtype independent of the precision mode, and
            # consistent with vk below
            vj_sparse = vj_sparse.astype(cp.float64)
        vj = cp.zeros_like(dms_3d)
        vj[:,cols,rows] = vj[:,rows,cols] = vj_sparse
        vj = vj.reshape(dms.shape)
        if not out_cupy: vj = vj.get()

    if with_k:
        vk = multi_gpu.array_reduce([x[1] for x in results], inplace=True)
        if fp32:
            vk = vk.astype(cp.float64)
        if symmetrize != 0:
            vk = transpose_sum(vk.reshape(-1,nao,nao), hermi=symmetrize)
        vk = vk.reshape(dms.shape)
        if not out_cupy: vk = vk.get()
    t1 = log.timer_debug1('vj and vk', *t1)
    return vj, vk

def _cast_cderi(cderi):
    '''Cast a CDERI block to float32, avoiding the slow non-coalesced copy
    on the (nL,nao,nao) transposed views yielded by DF.loop().  Casting the
    C-contiguous (nao,nao,nL) parent buffer and re-transposing is several
    times faster; cutensor accepts the resulting strided tensor natively.
    '''
    base = cderi.base
    if (cderi.ndim == 3 and cderi.strides[0] == cderi.itemsize
            and base is not None and base.ndim == 3
            and base.flags['C_CONTIGUOUS']
            and base.shape[2] == cderi.shape[0]
            and base.shape[:2] == cderi.shape[1:]):
        return base.astype(cp.float32).transpose(2, 0, 1)
    return cderi.astype(cp.float32)

def get_j(dfobj, dm, hermi=1):
    from gpu4pyscf.df.int3c2e_bdiv import int2c2e
    from gpu4pyscf.df import j_engine_3c2e
    if dfobj.intopt is None:
        dfobj.build(build_cderi=False)

    if dfobj._cd_j2c is None:
        j2c = int2c2e(dfobj.auxmol)
        try:
            dfobj._cd_j2c = cholesky(j2c), 'cd'
        except RuntimeError:
            dfobj._cd_j2c = j2c, None

    dm_shape = dm.shape
    intopt = dfobj.intopt
    if dm.ndim == 2 or len(dm) == 1:
        if dfobj.j_engine is None:
            dfobj.j_engine = j_engine_3c2e.Int3c2eOpt(dfobj.mol, dfobj.auxmol).build()
        intopt = dfobj.j_engine

    mol = intopt.mol
    auxmol = intopt.auxmol
    dm = mol.apply_C_mat_CT(dm)
    rhoj = intopt.contract_dm(dm, hermi)
    rhoj = auxmol.apply_CT_dot(rhoj, axis=-1)

    j2c, tag = dfobj._cd_j2c
    if tag == 'cd':
        rhoj = solve_triangular(j2c, rhoj.T, lower=True)
        rhoj = solve_triangular(j2c.T, rhoj, lower=False).T
    else:
        rhoj = cp.linalg.solve(j2c, rhoj.T).T

    rhoj = auxmol.apply_C_dot(rhoj, axis=-1)
    vj = intopt.contract_auxvec(rhoj)
    vj = mol.apply_CT_mat_C(vj)
    return vj.reshape(dm_shape)

def factorize_dm(dm, hermi=0):
    '''
    Factorize density matrices to the product of two low-rank tensors.

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix.
            When the input dm contains the mo_coeff attribute, orbol stores
            eigenvectors * sqrt(occupancies).
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies).
            When the input dm contains the mo_coeff attribute, orbor is None
    '''
    if isinstance(dm, CPArrayWithTag):
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = cp.asarray(dm.mo_coeff)
            mo_occ = cp.asarray(dm.mo_occ)
            assert mo_coeff.ndim == mo_occ.ndim + 1
            if mo_coeff.ndim == 2:
                mask = mo_occ > 0
                dm_factor = mo_coeff[:,mask]
                dm_factor *= cp.sqrt(mo_occ[mask])
            elif mo_coeff.ndim == 3:
                mask = (mo_occ > 0).any(axis=0)
                dm_factor = mo_coeff[:,:,mask]
                dm_factor *= cp.sqrt(mo_occ[:,None,mask])
            else:
                mask = (mo_occ > 0).any(axis=(0, 1))
                dm_factor = mo_coeff[:,:,:,mask]
                dm_factor *= cp.sqrt(mo_occ[:,:,None,mask])
            return dm_factor, None
        if hasattr(dm, 'factor_l'):
            return dm.factor_l, dm.factor_r

    dm = cp.asarray(dm)
    shape = dm.shape
    if len(shape) > 3:
        dm = dm.reshape(-1, *shape[-2:])
    l, r = decompose_rdm1_svd(dm, hermi)
    if len(shape) > 3:
        shape = shape[:-2] + l.shape[-2:]
        l = l.reshape(shape)
        r = r.reshape(shape)
    return l, r

def decompose_rdm1_svd(dm, hermi=0):
    '''Decompose density matrix as U.Vh using SVD

    Args:
        dm : ndarray or sequence of ndarrays of shape (*,nao,nao)
            Density matrices

    Returns:
        orbol : list of ndarrays of shape (nao,*)
            Contains non-null eigenvectors of density matrix
        orbor : list of ndarrays of shape (nao,*)
            Contains orbol * eigenvalues (occupancies)
    '''
    if hermi == 1:
        s, u = cp.linalg.eigh(dm)
        mask = abs(s) > 1e-8
        if dm.ndim == 2:
            c = u[:,mask]
            return c, contract('i,pi->pi', s[mask], c).conj()
        else:
            mask = mask.any(axis=0)
            c = u[:,:,mask]
            return c, contract('si,spi->spi', s[:,mask], c).conj()

    u, s, vh = cp.linalg.svd(dm)
    mask = s > 1e-8
    if dm.ndim == 2:
        mask = cp.where(mask)[0]
        return u[:,mask], contract('i,ip->pi', s[mask], vh[mask])
    else:
        mask = cp.where(mask.any(axis=0))[0]
        return u[:,:,mask], contract('si,sip->spi', s[:,mask], vh[:,mask])

def _make_factorized_dm(factor_l, factor_r, symmetrize=1):
    if factor_l.ndim == 4 and factor_r.ndim == 3:
        dm = contract('snpi,sqi->snpq', factor_l, factor_r)
    else:
        dm = cp.matmul(factor_l, factor_r.swapaxes(-1, -2))
    if symmetrize == 1 or symmetrize == 2:
        nao = dm.shape[-1] # dm1 may have dimensions > 3
        transpose_sum(dm.reshape(-1,nao,nao), inplace=True, hermi=symmetrize)
    return tag_array(dm, factor_l=factor_l, factor_r=factor_r, symmetrize=symmetrize)

def _tag_factorize_dm(dm, hermi=0):
    if hasattr(dm, 'symmetrize'):
        # This dm should be created by the _make_factorized_dm
        return dm
    l, r = factorize_dm(dm, hermi)
    return tag_array(dm, factor_l=l, factor_r=r, symmetrize=0)

def _transpose_dm(dm):
    dm_T = dm.T
    if hasattr(dm, 'symmetrize'):
        dm_T.factor_l = dm.factor_r
        dm_T.factor_r = dm.factor_l
        dm_T.symmetrize = dm.symmetrize
    else:
        dm_T = dm_T.view(cp.ndarray)
    return dm_T

def _aggregate_dm_factor_l(dms):
    factor_l = cp.stack([x.factor_l for x in dms])
    factor_r = dms[0].factor_r
    assert all(x.symmetrize == 0 for x in dms)
    return tag_array(cp.stack(dms), factor_l=factor_l, factor_r=factor_r,
                     symmetrize=0)

def _stack_uhf_occ_oribtals(moa, mob):
    nocca = moa.shape[-1]
    noccb = mob.shape[-1]
    assert nocca >= noccb
    mo_ab = cp.empty((2,) + moa.shape, dtype=moa.dtype)
    mo_ab[0] = moa
    mo_ab[1,...,:noccb] = mob
    if nocca > noccb:
        mo_ab[1,...,noccb:] = 0.
    return mo_ab

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
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#
# modified by Xiaojie Wu <wxj6000@gmail.com>; Zhichen Pu <hoshishin@163.com>

"""
DIIS
"""

import os
import numpy
import cupy as cp
import scipy.linalg
import scipy.optimize
import pyscf.scf.diis as cpu_diis
import gpu4pyscf.lib as lib
from gpu4pyscf.lib import logger
from gpu4pyscf.lib.cupy_helper import (
    contract, eigh, sandwich_dot, pack_tril, unpack_tril, get_avail_mem,
    asarray)

# J. Mol. Struct. 114, 31-34 (1984); DOI:10.1016/S0022-2860(84)87198-7
# PCCP, 4, 11 (2002); DOI:10.1039/B108658H
# GEDIIS, JCTC, 2, 835 (2006); DOI:10.1021/ct050275a
# C2DIIS, IJQC, 45, 31 (1993); DOI:10.1002/qua.560450106
# SCF-EDIIS, JCP 116, 8255 (2002); DOI:10.1063/1.1470195

# error vector = SDF-FDS
# error vector = F_ai ~ (S-SDS)*S^{-1}FDS = FDS - SDFDS ~ FDS-SDF in converge
class CDIIS(lib.diis.DIIS):
    incore = None

    def __init__(self, mf=None, filename=None):
        lib.diis.DIIS.__init__(self, mf, filename)
        self.rollback = False
        self.Corth = None
        self.space = 8

    def update(self, s, d, f, *args, **kwargs):
        if d.dtype == cp.complex128:
            s = s.astype(cp.complex128)
        errvec = self._sdf_err_vec(s, d, f)
        if self.incore is None:
            mem_avail = get_avail_mem()
            self.incore = errvec.nbytes*2 * (20+self.space) < mem_avail
            if not self.incore:
                logger.debug(self, 'Large system detected. DIIS intermediates '
                             'are saved in the host memory')
        if self.Corth.ndim == 3:
            nao, nmo = self.Corth.shape[-2:]
        else:
            assert self.Corth.ndim == 2
            nao, nmo = self.Corth.shape
        errvec = pack_tril(errvec.reshape(-1,nmo,nmo))
        f_tril = pack_tril(f.reshape(-1,nao,nao))
        xnew = lib.diis.DIIS.update(self, f_tril, xerr=errvec)
        if self.rollback > 0 and len(self._bookkeep) == self.space:
            self._bookkeep = self._bookkeep[-self.rollback:]
        return unpack_tril(xnew).reshape(f.shape)

    def get_num_vec(self):
        if self.rollback:
            return self._head
        else:
            return len(self._bookkeep)

    def _sdf_err_vec(self, s, d, f):
        '''error vector = SDF - FDS'''
        if f.ndim == s.ndim+1: # UHF
            assert len(f) == 2
            if s.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f[0], s)[1]
                sdf = cp.empty_like(f)
                s.dot(d[0]).dot(f[0], out=sdf[0])
                s.dot(d[1]).dot(f[1], out=sdf[1])
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().transpose(0,2,1)
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f[0], s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                Corth = asarray(self.Corth)
                sdf = cp.empty_like(f)
                tmp = None
                tmp = contract('Kij,Kjk->Kik', d[0], f[0], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[0])
                tmp = contract('Kpq,Kqj->Kpj', sdf[0], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[0])

                tmp = contract('Kij,Kjk->Kik', d[1], f[1], out=tmp)
                contract('Kij,Kjk->Kik', s, tmp, out=sdf[1])
                tmp = contract('Kpq,Kqj->Kpj', sdf[1], Corth, out=tmp)
                contract('Kpj,Kpi->Kij', tmp, Corth.conj(), out=sdf[1])
                errvec = sdf - sdf.conj().transpose(0,1,3,2)
        else: # RHF
            assert f.ndim == s.ndim
            if f.ndim == 2: # molecular SCF or single k-point
                if self.Corth is None:
                    self.Corth = eigh(f, s)[1]
                sdf = s.dot(d).dot(f)
                sdf = sandwich_dot(sdf, self.Corth)
                errvec = sdf - sdf.conj().T
            else: # k-points
                if self.Corth is None:
                    self.Corth = cp.empty_like(s)
                    for k, (fk, sk) in enumerate(zip(f, s)):
                        self.Corth[k] = eigh(fk, sk)[1]
                sd = contract('Kij,Kjk->Kik', s, d)
                sdf = contract('Kij,Kjk->Kik', sd, f)
                Corth = asarray(self.Corth)
                sdf = contract('Kpq,Kqj->Kpj', sdf, Corth)
                sdf = contract('Kpj,Kpi->Kij', sdf, Corth.conj())
                errvec = sdf - sdf.conj().transpose(0,2,1)
        return errvec.ravel()

SCFDIIS = SCF_DIIS = DIIS = CDIIS


def _adiis_head_tail_armed(rec):
    '''ADIAS-HEAD-01 (GPU4PYSCF_DIIS_MODE=adiis, default off): opt-in
    ADIIS head-phase Fock extrapolation with CDIIS tail handover.  Inert
    under the strict fp64 reference lane (R anchors are never altered) and
    for every other knob value.'''
    if getattr(rec, '_strict_fp64_reference', False):
        return False
    return os.environ.get('GPU4PYSCF_DIIS_MODE', '').strip().lower() == 'adiis'


def _adiis_minimize(df, idnewest):
    '''Host minimizer over the ADIIS energy model on a precomputed df
    matrix.  The cost/gradient/BFGS arithmetic mirrors pyscf 2.14
    pyscf.scf.diis.adiis_minimize verbatim; only the df assembly (device
    einsum in ADIISHeadTail._adiis_extrapolate vs host einsum in pyscf)
    differs, which pins GPU-vs-CPU update-path parity to solver tolerance.
    Ref: JCP 132, 054109 (2010); DOI:10.1063/1.3304922'''
    nx = df.shape[0]
    d_fn = df[:,idnewest]
    dn_f = df[idnewest]
    dn_fn = df[idnewest,idnewest]
    dd_fn = d_fn - dn_fn
    df = df - d_fn[:,None] - dn_f + dn_fn

    def costf(x):
        c = x**2 / (x**2).sum()
        return (numpy.einsum('i,i', c, dd_fn) * 2 +
                numpy.einsum('i,ij,j', c, df, c))

    def grad(x):
        x2sum = (x**2).sum()
        c = x**2 / x2sum
        fc = 2*dd_fn
        fc += numpy.einsum('j,kj->k', c, df)
        fc += numpy.einsum('i,ik->k', c, df)
        cx = numpy.diag(x*x2sum) - numpy.einsum('k,n->kn', x**2, x)
        cx *= 2/x2sum**2
        return numpy.einsum('k,kn->n', fc, cx)

    res = scipy.optimize.minimize(costf, numpy.ones(nx), method='BFGS',
                                  jac=grad, tol=1e-9)
    return res.fun, (res.x**2)/(res.x**2).sum()


class ADIISHeadTail(CDIIS):
    '''ADIAS-HEAD-01 opt-in (GPU4PYSCF_DIIS_MODE=adiis, default off):
    ADIIS head-phase Fock extrapolation with permanent handover to the
    deployed CDIIS tail.

    While the Corth-basis SDF-FDS error rms stays above adiis_switch_rms
    the update stores the (dm, fock) history in a ring buffer of size
    `space` and returns the ADIIS model extrapolation (JCP 132, 054109
    (2010)); at/below the threshold -- or after adiis_max_fallbacks failed
    minimizations -- it latches to the parent CDIIS.update for the rest of
    the SCF.  The handover cycle is recorded in mf.scf_summary
    (adias_switch_cycle / adias_head_cycles) for engagement evidence.
    Update-path change only: the converged fixed point, the Fock build and
    the precision map are untouched, so the converged energy is identical
    up to the convergence tolerance.'''

    def __init__(self, mf=None, filename=None):
        CDIIS.__init__(self, mf, filename)
        self.mf = mf
        # 1e-5 per amendment-1 (2026-11-05): the measured melatonin rms
        # trajectory (4.9e-3 at cycle 0) put the 1e-2 latch on a knife
        # edge after ONE ADIIS cycle; 1e-5 holds the measured head
        # (entry-269 boundary maps to rms ~6e-6..2e-6) and keeps ADIIS
        # off the asymptotic CDIIS tail.
        self.adiis_switch_rms = 1e-5
        self.adiis_max_fallbacks = 3
        self.adiis_active = True
        self.adiis_cycle = 0
        self.adiis_switch_cycle = None
        self.adiis_fallbacks = 0
        self._adiis_buffer = {}
        self._adiis_head = 0

    def _record(self, key, value):
        mf = self.mf
        if mf is not None and hasattr(mf, 'scf_summary'):
            mf.scf_summary[key] = value

    def _latch_cdiis(self, rms=None):
        self.adiis_active = False
        self.adiis_switch_cycle = self.adiis_cycle
        self._record('adias_switch_cycle', self.adiis_cycle)
        logger.debug(self, 'ADIAS-HEAD-01 handover to CDIIS at head cycle %d (rms %s)',
                     self.adiis_cycle, rms)

    def _adiis_extrapolate(self, d, f):
        if not self._adiis_buffer:
            shape = (self.space,) + d.shape
            self._adiis_buffer['dm'] = cp.zeros(shape, dtype=f.dtype)
            self._adiis_buffer['fock'] = cp.zeros(shape, dtype=f.dtype)
        head = self._adiis_head
        self._adiis_buffer['dm'][head] = d
        self._adiis_buffer['fock'][head] = f
        ds = self._adiis_buffer['dm']
        fs = self._adiis_buffer['fock']
        nx, nao = ds.shape[0], ds.shape[-1]
        df = cp.einsum('inpq,jnqp->ij',
                       ds.reshape(nx, -1, nao, nao),
                       fs.reshape(nx, -1, nao, nao)).real
        fun, c = _adiis_minimize(cp.asnumpy(df), head)
        if not numpy.all(numpy.isfinite(c)) or not numpy.isfinite(fun):
            raise RuntimeError('ADIAS-HEAD-01 non-finite minimizer output')
        self._adiis_head = head + 1
        if self._adiis_head >= self.space:
            self._adiis_head = 0
        return cp.einsum('i,i...pq->...pq', cp.asarray(c, dtype=f.dtype), fs)

    def update(self, s, d, f, *args, **kwargs):
        if not self.adiis_active:
            return CDIIS.update(self, s, d, f, *args, **kwargs)
        errvec = self._sdf_err_vec(s, d, f)
        rms = float(cp.linalg.norm(errvec)) / errvec.size ** .5
        if self._adiis_head > 0 and rms < self.adiis_switch_rms:
            self._latch_cdiis(rms)
            return CDIIS.update(self, s, d, f, *args, **kwargs)
        try:
            fock = self._adiis_extrapolate(d, f)
        except Exception:
            self.adiis_fallbacks += 1
            logger.debug(self, 'ADIAS-HEAD-01 minimizer fallback %d',
                         self.adiis_fallbacks)
            if self.adiis_fallbacks >= self.adiis_max_fallbacks:
                self._latch_cdiis(rms)
            else:
                self.adiis_cycle += 1
                self._record('adias_head_cycles', self.adiis_cycle)
            return CDIIS.update(self, s, d, f, *args, **kwargs)
        self.adiis_cycle += 1
        self._record('adias_head_cycles', self.adiis_cycle)
        return fock

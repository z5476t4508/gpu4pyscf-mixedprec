"""FP32 J/K prototype for gpu4pyscf DF-RHF (mixed-precision SCF).

Monkey-patches gpu4pyscf.df.df_jk.get_jk with a dtype-parameterized copy.
When PRECISION_MODE == 'fp32', the density factors, sparse DM, CDERI blocks
and all contraction buffers are cast to float32; cutensor contractions then
run in FP32 (on RTX 5090: FP32 ~64x faster than the FP64 default).
Outputs vj/vk are returned as float64. The SCF driver flips the mode to
'fp64' for the final iterations (see step2_proto.py).
"""
import cupy
import cupy as cp
from pyscf import lib
from gpu4pyscf.lib import logger, multi_gpu
from gpu4pyscf.lib.cupy_helper import (
    contract, ndarray, get_avail_mem, transpose_sum)
from gpu4pyscf.df.df_jk import factorize_dm

PRECISION_MODE = 'fp64'  # 'fp32' or 'fp64'; flipped by the SCF callback

def get_jk(dfobj, dms, hermi=0, with_j=True, with_k=True, omega=None):
    '''get jk with density fitting (mixed-precision prototype)'''
    log = logger.new_logger(dfobj.mol, dfobj.verbose)
    t1 = t0 = log.init_timer()
    if dfobj._cderi is None:
        log.debug('Build CDERI ...')
        dfobj.build(omega=omega)
        t1 = log.timer_debug1('init jk', *t0)

    fp32 = PRECISION_MODE == 'fp32'
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
        if fp32:  # FP32: cast the density factors
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
            vk = cupy.zeros((nspin, n_dm, nao, nao),
                            dtype=cp.float32 if fp32 else cp.float64)
            mem_avail = get_avail_mem(exclude_memory_pool=True)
            dm_batch_size = int(mem_avail * 0.6 / (blksize*nao*nocc * 8))
            if dm_factor_mode == 1:
                dm_batch_size = dm_batch_size // 2
            dm_batch_size = min(dm_batch_size, n_dm)
            assert dm_batch_size > 0
            log.debug1('blksize=%d, dm_batch_size=%d (fp32=%s)',
                       blksize, dm_batch_size, fp32)

            buf_dtype = cp.float32 if fp32 else cp.float64
            if dm_factor_mode == 0:
                buf = cp.empty((dm_batch_size, blksize * nao*nocc), dtype=buf_dtype)
            elif dm_factor_mode == 1:
                buf = cp.empty((2 * dm_batch_size, blksize * nao*nocc), dtype=buf_dtype)
            else:
                buf = cp.empty((dm_batch_size+1, blksize * nao*nocc), dtype=buf_dtype)
                buf1 = buf[-1]

        for cderi, cderi_tril in dfobj.loop(blksize=blksize, unpack=with_k):
            if fp32:  # FP32: cast CDERI blocks
                # cderi (unpacked) is a (nL,nao,nao) view with L contiguous
                # (stride 8); astype on that layout is ~7x slower than on the
                # underlying (nao,nao,nL) C-order buffer. Cast via the parent
                # buffer and re-transpose; cutensor handles the resulting
                # strided view natively (strided fp32 contract measured faster
                # than contiguous here).
                if cderi is not None and cderi.ndim == 3:
                    base = cderi.base
                    if (cderi.strides[0] == cderi.itemsize and
                            base is not None and base.ndim == 3 and
                            base.flags['C_CONTIGUOUS'] and
                            base.shape[2] == cderi.shape[0] and
                            base.shape[:2] == cderi.shape[1:]):
                        cderi = base.astype(cp.float32).transpose(2, 0, 1)
                    else:
                        cderi = cderi.astype(cp.float32)
                elif cderi is not None:
                    cderi = cderi.astype(cp.float32)
                if cderi_tril is not None:
                    cderi_tril = cderi_tril.astype(cp.float32)

            if with_j:
                auxvec = contract('np,Lp->nL', _dm_sparse, cderi_tril)
                contract('nL,Lp->np', auxvec, cderi_tril, beta=1, out=vj)

            if with_k:
                nL = len(cderi)
                for s in range(nspin):
                    if dm_factor_mode == 0:
                        for i0, i1 in lib.prange(0, n_dm, dm_batch_size):
                            rhok = ndarray((i1-i0,nao,nocc,nL), dtype=buf_dtype, buffer=buf)
                            contract('Lij,njk->nikL', cderi, factor_l[s,i0:i1], out=rhok)
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
        return vj, vk

    results = multi_gpu.run(proc, non_blocking=True)

    vj = vk = None
    if with_j:
        vj_sparse = multi_gpu.array_reduce([x[0] for x in results], inplace=True)
        vj = cp.zeros_like(dms_3d)  # float64; scatter auto-casts
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


def install():
    '''Patch the single hook point: df.py:117 resolves df_jk.get_jk at call
    time. (df_jk.py:138 get_jk is a class method, unaffected.)'''
    import gpu4pyscf.df.df_jk as df_jk_mod
    df_jk_mod.get_jk = get_jk
    return get_jk

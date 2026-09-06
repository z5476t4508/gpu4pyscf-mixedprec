/*
 * Copyright 2026 The PySCF Developers. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Float32 variant of the ejk_int3c2e_ip1 gradient kernel.
 *
 * The heavy interior of sum_ejk_int3c2e_ip1_kernel (Rys quadrature, the
 * gxyz recurrence in shared memory, and the per-thread dm_tensor cache)
 * runs in float32 which doubles the shared-memory residency per block and
 * the register throughput.  The ejk/ejk_aux accumulators stay in float64:
 * they are reduced with atomicAdd over thousands of blocks and carry the
 * final gradient, so their precision is preserved.
 *
 * The Rys roots/weights are interpolated in float32 (values are O(1));
 * this contributes ~1e-6 relative error to the integrals which lands
 * around 1e-9 absolute on the gradient.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gvhf-rys/vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "build_rys_gxyz.cuh"

#define THREADS         256
#define BLOCK_SIZE      16
#define GOUT_WIDTH      54

// float32 Rys roots (roots and weights are O(1); interpolation in float)
__device__ __forceinline__
static void rys_roots_f32(int nroots, float x, float *rw,
                          int block_size, int rt_id, int stride)
{
    float *r = rw;
    float *w = rw + block_size;
    int block_size2 = block_size * 2;
    if (x < 3.e-7f){
        int off = nroots * (nroots - 1) / 2;
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = ROOT_SMALLX_R0[off+i] + ROOT_SMALLX_R1[off+i] * x;
            w[i*block_size2] = ROOT_SMALLX_W0[off+i] + ROOT_SMALLX_W1[off+i] * x;
        }
        return;
    }

    if (x > 35.f+nroots*5.f) {
        int off = nroots * (nroots - 1) / 2;
        float t = sqrtf(PIE4/x);
        for (int i = rt_id; i < nroots; i += stride)  {
            r[i*block_size2] = ROOT_LARGEX_R_DATA[off+i] / x;
            w[i*block_size2] = ROOT_LARGEX_W_DATA[off+i] * t;
        }
        return;
    }

    if (nroots == 1) {
        float tt = sqrtf(x);
        float fmt0 = SQRTPIE4 / tt * erff(tt);
        w[0] = fmt0;
        float e = __expf(-x);
        float b = .5f / x;
        float fmt1 = b * (fmt0 - e);
        r[0] = fmt1 / fmt0;
        return;
    }

    // ROOT_RW_DATA is a double[] table; it must NOT be aliased as float --
    // that reinterprets the bits rather than converting the values.  The
    // recurrence stays in double: measured, running it in float buys no time
    // (it is amortised over the GOUT_WIDTH loop below), so there is nothing
    // to trade the precision for.  The result is narrowed into a float rw[].
    double *datax_d = ROOT_RW_DATA + DEGREE1*INTERVALS * nroots*(nroots-1);
    int it = (int)(x * .4f);
    double u = (x - it * 2.5f) * 0.8f - 1.;
    double u2 = u * 2.;
    for (int i = rt_id; i < nroots*2; i += stride) {
        double *c = datax_d + i * DEGREE1 * INTERVALS;
        double c0 = c[it + DEGREE   *INTERVALS];
        double c1 = c[it +(DEGREE-1)*INTERVALS];
        double c2, c3;
#pragma unroll
        for (int n = DEGREE-2; n > 0; n-=2) {
            c2 = c[it + n   *INTERVALS] - c1;
            c3 = c0 + c1*u2;
            c1 = c2 + c3*u2;
            c0 = c[it +(n-1)*INTERVALS] - c3;
        }
        if (DEGREE % 2 == 0) {
            c2 = c[it] - c1;
            c3 = c0 + c1*u2;
            rw[i*block_size] = c2 + c3*u;
        } else {
            rw[i*block_size] = c0 + c1*u;
        }
    }
}

__device__ __forceinline__
static void rys_roots_rs_f32(int nroots, float theta, float rr, double omega,
                             float *rw, int block_size, int rt_id, int stride)
{
    float theta_rr = theta * rr;
    if (omega == 0) {
        rys_roots_f32(nroots, theta_rr, rw, block_size, rt_id, stride);
    } else if (omega > 0) {
        float theta_fac = omega * omega / (omega * omega + theta);
        rys_roots_f32(nroots, theta_fac*theta_rr, rw, block_size, rt_id, stride);
        if (stride != 1) __syncthreads();
        float sqrt_theta_fac = sqrtf(theta_fac);
        for (int irys = rt_id; irys < nroots; irys+=stride) {
            rw[ irys*2   *block_size] *= theta_fac;
            rw[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    } else {
        int _nroots = nroots / 2;
        rys_roots_f32(_nroots, theta_rr, rw, block_size, rt_id, stride);
        float theta_fac = omega * omega / (omega * omega + theta);
        float *rw1 = rw + nroots*block_size;
        rys_roots_f32(_nroots, theta_fac*theta_rr, rw1, block_size, rt_id, stride);
        if (stride != 1) __syncthreads();
        float sqrt_theta_fac = -sqrtf(theta_fac);
        for (int irys = rt_id; irys < _nroots; irys+=stride) {
            rw1[ irys*2   *block_size] *= theta_fac;
            rw1[(irys*2+1)*block_size] *= sqrt_theta_fac;
        }
    }
}

// float32 BUILD_3C_GXYZ: same recurrence as build_rys_gxyz.cuh but with
// float gx / rjri / Rpq shared-memory arrays.  The rt weights are read
// as float from rw.
#define BUILD_3C_GXYZ_F32(lj, lk, rjri_stride, active) \
        __syncthreads(); \
        int nst = nst_per_block; \
        int gx_len = nst * g_size; \
        if (gout_id == 0) { \
            gx[gx_len*2] = rw[(irys*2+1)*nst]; \
        } \
        float rt = rw[irys*2*nst]; \
        float rt_aa = rt / (aij + ak); \
        float s0x, s1x, s2x; \
        if (lij > 0) { \
            float rt_aij = rt_aa * ak; \
            float b10 = .5f/aij * (1 - rt_aij); \
            __syncthreads(); \
            for (int n = gout_id; n < 3; n += gout_stride) { \
                float *_gx = gx + n * gx_len; \
                float Rpa = rjri[n*rjri_stride] * aj_aij; \
                float c0x = Rpa - rt_aij * Rpq[n*nst]; \
                s0x = _gx[0]; \
                s1x = c0x * s0x; \
                _gx[nst] = s1x; \
                for (int i = 1; i < lij; ++i) { \
                    s2x = c0x * s1x + i * b10 * s0x; \
                    _gx[(i+1)*nst] = s2x; \
                    s0x = s1x; \
                    s1x = s2x; \
                } \
            } \
        } \
        if (lk > 0) { \
            float rt_ak  = rt_aa * aij; \
            float b00 = .5f * rt_aa; \
            float b01 = .5f/ak  * (1 - rt_ak ); \
            int lij3 = (lij+1)*3; \
            for (int n = gout_id; n < lij3+gout_id; n += gout_stride) { \
                __syncthreads(); \
                int i = n / 3; \
                int _ix = n - i * 3; \
                float *_gx = gx + (i + _ix * g_size) * nst; \
                float cpx = rt_ak * Rpq[_ix*nst]; \
                if (n < lij3) { \
                    s0x = _gx[0]; \
                    s1x = cpx * s0x; \
                    if (i > 0) { \
                        s1x += i * b00 * _gx[-nst]; \
                    } \
                    _gx[stride_k*nst] = s1x; \
                } \
                for (int k = 1; k < lk; ++k) { \
                    __syncthreads(); \
                    if (n < lij3) { \
                        s2x = cpx*s1x + k*b01*s0x; \
                        if (i > 0) { \
                            s2x += i * b00 * _gx[(k*stride_k-1)*nst]; \
                        } \
                        _gx[(k*stride_k+stride_k)*nst] = s2x; \
                        s0x = s1x; \
                        s1x = s2x; \
                    } \
                } \
            } \
        } \
        if (lj > 0) { \
            __syncthreads(); \
            if (active) { \
                int lk3 = (lk+1)*3; \
                for (int m = gout_id; m < lk3; m += gout_stride) { \
                    int k = m / 3; \
                    int _ix = m - k * 3; \
                    float xjxi = rjri[_ix*rjri_stride]; \
                    float *_gx = gx + (_ix*g_size + k*stride_k) * nst; \
                    for (int j = 0; j < lj; ++j) { \
                        int ij = (lij-j) + j*stride_j; \
                        s1x = _gx[ij*nst]; \
                        for (--ij; ij >= j*stride_j; --ij) { \
                            s0x = _gx[ij*nst]; \
                            _gx[(ij+stride_j)*nst] = s1x - xjxi * s0x; \
                            s1x = s0x; \
                        } \
                    } \
                } \
            } \
        } \
        __syncthreads()

__global__ static
void sum_ejk_int3c2e_ip1_kernel_f32(double *ejk, double *ejk_aux,
                            float *dm, float *density_auxvec, int n_dm,
                            RysIntEnvVars envs, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                            int *ksh_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int naux)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    extern __shared__ float shared_memory[];
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nsp_per_block;

    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    double omega = env[PTR_RANGE_OMEGA];
    if (thread_id == 0) {
        shl_pair0 = shl_pair_offsets[sp_block_id];
        shl_pair1 = shl_pair_offsets[sp_block_id+1];
        uint32_t bas_ij0 = bas_ij_idx[shl_pair0];
        int ish0 = bas_ij0 / nbas;
        int jsh0 = bas_ij0 - nbas * ish0;
        ksh0 = ksh_offsets[ksh_block_id];
        ksh1 = ksh_offsets[ksh_block_id+1];
        nksh = ksh1 - ksh0;
        li = bas[ish0*BAS_SLOTS+ANG_OF];
        lj = bas[jsh0*BAS_SLOTS+ANG_OF];
        lk = bas[ksh0*BAS_SLOTS+ANG_OF];
        int lij = li + lj + 1;
        nroots = (lij + lk) / 2 + 1;
        if (omega < 0) {
            nroots *= 2;
        }
        iprim = bas[ish0*BAS_SLOTS+NPRIM_OF];
        jprim = bas[jsh0*BAS_SLOTS+NPRIM_OF];
        kprim = bas[ksh0*BAS_SLOTS+NPRIM_OF];
        nao = envs.ao_loc[nbas];
        int nfi = c_nf[li];
        int nfj = c_nf[lj];
        int nfk = c_nf[lk];
        int nfij = nfi * nfj;
        nf = nfij * nfk;
        int stride_j = li + 2;
        int stride_k = stride_j * (lj + 1);
        g_size = stride_k * (lk + 1);
        gout_stride = gout_stride_lookup[lk*LMAX1*LMAX1+li*LMAX1+lj];
        nst_per_block = THREADS / gout_stride;
        aux_per_block = min(nst_per_block, BLOCK_SIZE);
        nsp_per_block = nst_per_block / aux_per_block;
    }
    __syncthreads();

    register int gout_id = thread_id / nst_per_block;
    register int st_id = thread_id - gout_id * nst_per_block;
    register int sp_id = st_id / aux_per_block;
    register int aux_id = st_id - sp_id * aux_per_block;

    int gx_len = g_size * nst_per_block;
    float *rjri = shared_memory + sp_id;
    float *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    float *gx = shared_memory + nst_per_block * 6 + st_id;
    float *rw = shared_memory + nst_per_block * (g_size*3+6) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        // The cross-thread reduction below runs through a float shared-memory
        // buffer, so a double per-thread accumulator is narrowed to float
        // immediately anyway -- it bought one extra rounding and cost nine
        // 1/64-rate fp64 adds in the innermost (GOUT_WIDTH) loop.
        float v_ix = 0;
        float v_iy = 0;
        float v_iz = 0;
        float v_jx = 0;
        float v_jy = 0;
        float v_jz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        __syncthreads();
        if (gout_id == 0 && aux_id == 0) {
            float xjxi = env[rj+0] - env[ri+0];
            float yjyi = env[rj+1] - env[ri+1];
            float zjzi = env[rj+2] - env[ri+2];
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];
            float dm_tensor[GOUT_WIDTH];
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                if (density_auxvec == NULL) {
                    int nfi = c_nf[li];
                    int nfj = c_nf[lj];
                    float div_nfi = c_div_nf[li];
                    float div_nfj = c_div_nf[lj];
                    float div_nfij = div_nfi * div_nfj;
                    int k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                    size_t pair_offset = ao_pair_loc[pair_ij];
                    float *dm_local = dm + pair_offset * naux + k0;
#pragma unroll
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        uint32_t ijk = n*gout_stride+gout_id;
                        if (ijk >= nf) break;
                        uint32_t k = ijk * div_nfij;
                        uint32_t ij = ijk - k * nfi*nfj;
                        dm_tensor[n] = dm_local[ij*naux + k*nksh];
                    }
                } else {
                    for (int n = 0; n < GOUT_WIDTH; ++n) {
                        dm_tensor[n] = 0;
                    }
                    int nfi = c_nf[li];
                    int nfj = c_nf[lj];
                    int i0 = envs.ao_loc[ish];
                    int j0 = envs.ao_loc[jsh];
                    int k0 = envs.ao_loc[ksh] - nao;
                    for (int i_dm = 0; i_dm < n_dm; ++i_dm) {
                        float *dm_local = dm + (i_dm * nao + j0) * (size_t)nao + i0;
#pragma unroll
                        for (int n = 0; n < GOUT_WIDTH; ++n) {
                            uint32_t ijk = n*gout_stride+gout_id;
                            if (ijk >= nf) break;
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            uint32_t jk = ijk * div_nfi;
                            uint32_t i = ijk - jk * nfi;
                            uint32_t k = jk * div_nfj;
                            uint32_t j = jk - k * nfj;
                            dm_tensor[n] += dm_local[j*nao+i] * density_auxvec[i_dm*naux+k0+k];
                        }
                    }
                }
            }

            float v_kx = 0;
            float v_ky = 0;
            float v_kz = 0;
            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                float ai = env[expi+ip];
                float aj = env[expj+jp];
                float aij = ai + aj;
                float aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    float theta_ij = ai * aj_aij;
                    float xjxi = rjri[0*nsp_per_block];
                    float yjyi = rjri[1*nsp_per_block];
                    float zjzi = rjri[2*nsp_per_block];
                    float rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    float Kab = theta_ij * rr_ij;
                    float fac_ij = PI_FAC;
                    if (ish == jsh) {
                        fac_ij *= .5f;
                    } else if (ish < jsh) {
                        fac_ij = 0;
                    }
                    float cicj = fac_ij * env[ci+ip] * env[cj+jp];
                    gx[gx_len] = cicj * __expf(-Kab);
                    float xij = xjxi * aj_aij + env[ri+0];
                    float yij = yjyi * aj_aij + env[ri+1];
                    float zij = zjzi * aj_aij + env[ri+2];
                    float xk = env[rk+0];
                    float yk = env[rk+1];
                    float zk = env[rk+2];
                    float xpq = xij - xk;
                    float ypq = yij - yk;
                    float zpq = zij - zk;
                    Rpq[0*nst_per_block] = xpq;
                    Rpq[1*nst_per_block] = ypq;
                    Rpq[2*nst_per_block] = zpq;
                }
                for (int kp = 0; kp < kprim; ++kp) {
                    float ak = env[expk+kp];
                    float theta = aij * ak / (aij + ak);
                    __syncthreads();
                    if (gout_id == 0) {
                        gx[0] = env[ck+kp] / (aij*ak*sqrtf(aij+ak));
                    }
                    float xpq = Rpq[0*nst_per_block];
                    float ypq = Rpq[1*nst_per_block];
                    float zpq = Rpq[2*nst_per_block];
                    float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    rys_roots_rs_f32(nroots, theta, rr, omega,
                                     rw, nst_per_block, gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int lij = li + lj + 1;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 1);
                        BUILD_3C_GXYZ_F32(lj, lk, nsp_per_block, pair_ij < shl_pair1 && kidx < ksh1);
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            int nsp = nsp_per_block;
                            int nfi = c_nf[li];
                            int nfj = c_nf[lj];
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            int i_1 =          nst;
                            int j_1 = stride_j*nst;
                            float ai2 = ai * 2;
                            float aj2 = aj * 2;
#pragma unroll
                            for (int n = 0; n < GOUT_WIDTH; ++n) {
                                uint32_t ijk = n*gout_stride+gout_id;
                                if (ijk >= nf) break;
                                uint32_t jk = ijk * div_nfi;
                                uint32_t i = ijk - jk * nfi;
                                uint32_t k = jk * div_nfj;
                                uint32_t j = jk - k * nfj;
                                int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                                int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                                int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                                int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                                int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                                int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                                int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                                int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                                int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                                float dm_ijk = dm_tensor[n];
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst;
                                float Ix = gx[addrx];
                                float Iy = gx[addry];
                                float Iz = gx[addrz];
                                float prod_xy = Ix * Iy * dm_ijk;
                                float prod_xz = Ix * Iz * dm_ijk;
                                float prod_yz = Iy * Iz * dm_ijk;
                                float goutx, gouty, goutz;
                                float gix = gx[addrx+i_1];
                                float giy = gx[addry+i_1];
                                float giz = gx[addrz+i_1];
                                float fix = ai2 * gix; if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                                float fiy = ai2 * giy; if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                                float fiz = ai2 * giz; if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }
                                goutx = fix * prod_yz;
                                gouty = fiy * prod_xz;
                                goutz = fiz * prod_xy;
                                v_ix += goutx;
                                v_iy += gouty;
                                v_iz += goutz;
                                v_kx -= goutx;
                                v_ky -= gouty;
                                v_kz -= goutz;
                                float gjx = gix - rjri[0*nsp] * Ix;
                                float gjy = giy - rjri[1*nsp] * Iy;
                                float gjz = giz - rjri[2*nsp] * Iz;
                                float fjx = aj2 * gjx; if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                                float fjy = aj2 * gjy; if (jy > 0) { fjy -= jy * gx[addry-j_1]; }
                                float fjz = aj2 * gjz; if (jz > 0) { fjz -= jz * gx[addrz-j_1]; }
                                goutx = fjx * prod_yz;
                                gouty = fjy * prod_xz;
                                goutz = fjz * prod_xy;
                                v_jx += goutx;
                                v_jy += gouty;
                                v_jz += goutz;
                                v_kx -= goutx;
                                v_ky -= gouty;
                                v_kz -= goutz;
                            }
                        }
                    }
                }
            }
            if (ejk_aux != NULL) {
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                float *reduce = shared_memory + nsp_per_block * 3 + thread_id;
                __syncthreads();
                reduce[0*THREADS] = v_kx;
                reduce[1*THREADS] = v_ky;
                reduce[2*THREADS] = v_kz;
                for (int i = gout_stride/2; i > 0; i >>= 1) {
                    __syncthreads();
                    if (gout_id < i && pair_ij < shl_pair1 && kidx < ksh1) {
#pragma unroll
                        for (int m = 0; m < 3; ++m) {
                            reduce[m*THREADS] += reduce[m*THREADS+i*nst_per_block];
                        }
                    }
                }
                if (gout_id == 0 && pair_ij < shl_pair1 && kidx < ksh1) {
                    atomicAdd(ejk_aux+ka*3+0, reduce[0*THREADS]);
                    atomicAdd(ejk_aux+ka*3+1, reduce[1*THREADS]);
                    atomicAdd(ejk_aux+ka*3+2, reduce[2*THREADS]);
                }
            }
        }
        int ia = bas[ish*BAS_SLOTS+ATOM_OF];
        int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
        float *reduce = shared_memory + nsp_per_block * 3 + thread_id;
        __syncthreads();
        // (\nabla i,j|k) + (i,\nabla j|k) + (ij|\nabla k) = 0
        reduce[0*THREADS] = v_ix;
        reduce[1*THREADS] = v_iy;
        reduce[2*THREADS] = v_iz;
        reduce[3*THREADS] = v_jx;
        reduce[4*THREADS] = v_jy;
        reduce[5*THREADS] = v_jz;
        for (int i = gout_stride/2; i > 0; i >>= 1) {
            __syncthreads();
            if (gout_id < i && pair_ij < shl_pair1) {
#pragma unroll
                for (int m = 0; m < 6; ++m) {
                    reduce[m*THREADS] += reduce[m*THREADS+i*nst_per_block];
                }
            }
        }
        if (gout_id == 0 && pair_ij < shl_pair1) {
            atomicAdd(ejk+ia*3+0, reduce[0*THREADS]);
            atomicAdd(ejk+ia*3+1, reduce[1*THREADS]);
            atomicAdd(ejk+ia*3+2, reduce[2*THREADS]);
            atomicAdd(ejk+ja*3+0, reduce[3*THREADS]);
            atomicAdd(ejk+ja*3+1, reduce[4*THREADS]);
            atomicAdd(ejk+ja*3+2, reduce[5*THREADS]);
        }
    }
}

extern "C" {
// Float32 counterpart of sum_ejk_int3c2e_ip1 (see ejk_int3c2e_ip1.cu).
// dm and density_auxvec are float32 arrays; ejk/ejk_aux remain float64.
// For exchange energy (density_auxvec==NULL), n_dm must be 1
int sum_ejk_int3c2e_ip1_f32(double *ejk, double *ejk_aux,
                    float *dm, float *density_auxvec, int n_dm,
                    RysIntEnvVars *envs, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset,
                    int nao, int npairs, int naux, int natm)
{
    cudaFuncSetAttribute(sum_ejk_int3c2e_ip1_kernel_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    sum_ejk_int3c2e_ip1_kernel_f32<<<blocks, THREADS, shm_size>>>(
            ejk, ejk_aux, dm, density_auxvec, n_dm, *envs,
            shl_pair_offsets, bas_ij_idx, ksh_offsets, gout_stride_lookup,
            ao_pair_loc, aux_offset, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in sum_ejk_int3c2e_ip1_f32: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}

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
 * Float32 variant of the ejk_int3c2e_ip2 Hessian kernel.
 *
 * Measured share of the work it is meant to remove: on B3LYP/def2-SVP/Tamoxifen
 * the JK half of partial_hess_elec is 34.88s, of which this kernel is 17.31s
 * (49.6%) over 13 launches.  The ceiling is the 2.7x its fully-ported sibling
 * ejk_int3c2e_ip1_f32 measured (3.181s -> 1.18s), not the card's 64:1
 * fp32:fp64 ratio: these Rys kernels are bound by shared-memory and global
 * traffic, not by fp64 arithmetic.
 *
 * Transformation, following ejk_int3c2e_ip1_f32.cu:
 *   - the gxyz recurrence, rjri/Rpq/rw and the whole quadrature interior run in
 *     float32; shared_memory is a float array, so the same element offsets
 *     address half the bytes;
 *   - dm / density_auxvec are float32 arrays (the caller casts the pseudo-DM
 *     after building it in float64);
 *   - ejk stays float64: it is the Hessian, reduced with atomicAdd over
 *     thousands of blocks.
 *
 * One deliberate precision decision that the ip1 port did not have to make.
 * ip1 reduces its per-thread accumulators through a float shared-memory buffer,
 * so making them float there cost nothing -- the extra precision was discarded
 * one statement later.  ip2 has no such buffer: each thread atomicAdds its 45
 * accumulators straight into the float64 ejk, so float accumulators here are a
 * real reduction in precision, not a free cleanup.  They are float in this file
 * because that is where most of this kernel's cost is: 45 fp64 accumulators are
 * 90 registers of pressure and 45 1/64-rate adds per iteration of the innermost
 * (nf/gout_stride) loop.  Switching the 45 declarations to double is the other
 * variant measured below.
 *
 * Measured, B3LYP/def2-SVP against a float64 reference (mixedprec/step8g):
 *
 *                        JK block    Hessian     VitC max   Tam max   Tam rms
 *   float64                35.2s      285.2s        --         --        --
 *   float accumulators     18.8s       68.3s      0.587      0.909     0.177
 *   double accumulators    27.3s       76.5s      0.219      3.400     0.432
 *   (unported lane)         35.2s      84.4s      0.006      0.006     0.001
 *
 * frequencies in cm^-1.  The kernel itself is correct: relative error on the
 * raw ejk block is 1.9e-5 (float acc) / 3.4e-6 (double acc), and the exact
 * translational sum rule sum_A d2E/dR_A dR_B = 0 holds to 1.8e-5 / 2.0e-7
 * against 3.6e-12 in float64.
 *
 * Note the two molecules rank the variants *oppositely* on frequencies while
 * the raw block error ranks them consistently: the frequency error is partly
 * cancelling against the fp32 error already present in make_h1/solve_mo1, so
 * it cannot be predicted from the kernel's own error.  That unpredictability,
 * plus 1.10-1.24x costing 150-500x the lane's frequency error, is why the
 * dispatch is not enabled -- see mixedprec/STATUS.md.
 *
 * Range separation is not ported.  The host entry returns 2 unless
 * omega == 0 && lr_factor == 1, and the Python dispatch checks the same
 * condition, so range-separated hybrids keep using the float64 kernel.  That
 * confines this file to the omega == 0 branch of rys_roots_for_k, i.e. plain
 * rys_roots, and covers the hybrids the port was written for (B3LYP, PBE0, HF).
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vhf.cuh"
#include "gvhf-rys/rys_roots.cu"
#include "gvhf-rys/rys_contract_k.cuh"
#include "build_rys_gxyz.cuh"

#define THREADS         256
#define BLOCK_SIZE      16

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
    // recurrence stays in double: measured on the ip1 port, running it in
    // float buys no time (it is amortised over the inner loop below), so
    // there is nothing to trade the precision for.
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

// float32 BUILD_3C_GXYZ: same recurrence as build_rys_gxyz.cuh but with
// float gx / rjri / Rpq shared-memory arrays.
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
void ejk_int3c2e_ip2_kernel_f32(double *ejk, float *dm, float *density_auxvec,
                            RysIntEnvVars envs, int *shl_pair_offsets,
                            uint32_t *bas_ij_idx, int *ksh_offsets, int *gout_stride_lookup,
                            int *ao_pair_loc, int aux_offset, int naux)
{
    // For better load balance, consume blocks in the reversed order
    int thread_id = threadIdx.x;
    int sp_block_id = gridDim.x - blockIdx.x - 1;
    int ksh_block_id = gridDim.y - blockIdx.y - 1;
    int nbas = envs.nbas;
    int *bas = envs.bas;
    double *env = envs.env;
    __shared__ int shl_pair0, shl_pair1;
    __shared__ int ksh0, ksh1, nksh;
    __shared__ int li, lj, lk, nroots, nf;
    __shared__ int iprim, jprim, kprim;
    __shared__ int g_size;
    __shared__ int nao;
    __shared__ int gout_stride, nst_per_block, aux_per_block, nsp_per_block;
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
        int lij = li + lj + 2;
        // omega < 0 (short range) is not dispatched to this kernel, so nroots
        // is never doubled here
        nroots = (lij + lk) / 2 + 1;
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
        int stride_k = stride_j * (lj + 2);
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
    extern __shared__ float shared_memory[];
    float *rjri = shared_memory + sp_id;
    float *Rpq = shared_memory + nsp_per_block * 3 + st_id;
    float *gx = shared_memory + nst_per_block * 6 + st_id;
    float *rw = shared_memory + nst_per_block * (g_size*3+6) + st_id;
    int idx_i = lex_xyz_offset(li);
    int idx_j = lex_xyz_offset(lj);
    int idx_k = lex_xyz_offset(lk);

    for (int pair_ij = shl_pair0+sp_id; pair_ij < shl_pair1+sp_id; pair_ij += nsp_per_block) {
        // float accumulators: see the note at the top of this file -- unlike
        // ip1 this is a precision decision, not a free one
        float v_ixx = 0;
        float v_ixy = 0;
        float v_ixz = 0;
        float v_iyy = 0;
        float v_iyz = 0;
        float v_izz = 0;
        float v_jxx = 0;
        float v_jxy = 0;
        float v_jxz = 0;
        float v_jyy = 0;
        float v_jyz = 0;
        float v_jzz = 0;
        float v1xx = 0;
        float v1xy = 0;
        float v1xz = 0;
        float v1yx = 0;
        float v1yy = 0;
        float v1yz = 0;
        float v1zx = 0;
        float v1zy = 0;
        float v1zz = 0;
        int bas_ij;
        if (pair_ij < shl_pair1) {
            bas_ij = bas_ij_idx[pair_ij];
        } else {
            bas_ij = bas_ij_idx[shl_pair0];
        }
        int ish = bas_ij / nbas;
        int jsh = bas_ij - nbas * ish;
        int i0 = envs.ao_loc[ish];
        int j0 = envs.ao_loc[jsh];
        int expi = bas[ish*BAS_SLOTS+PTR_EXP];
        int expj = bas[jsh*BAS_SLOTS+PTR_EXP];
        int ci = bas[ish*BAS_SLOTS+PTR_COEFF];
        int cj = bas[jsh*BAS_SLOTS+PTR_COEFF];
        int ri = bas[ish*BAS_SLOTS+PTR_BAS_COORD];
        int rj = bas[jsh*BAS_SLOTS+PTR_BAS_COORD];
        float xjxi = env[rj+0] - env[ri+0];
        float yjyi = env[rj+1] - env[ri+1];
        float zjzi = env[rj+2] - env[ri+2];
        __syncthreads();
        if (gout_id == 0 && aux_id == 0) {
            rjri[0*nsp_per_block] = xjxi;
            rjri[1*nsp_per_block] = yjyi;
            rjri[2*nsp_per_block] = zjzi;
        }
        for (int kidx = ksh0+aux_id; kidx < ksh1+aux_id; kidx += aux_per_block) {
            int ksh = kidx;
            if (kidx >= ksh1) {
                ksh = ksh0;
            }
            int k0;
            float *dm_tensor;
            if (density_auxvec == NULL) {
                k0 = envs.ao_loc[ksh0] - nao - aux_offset + ksh - ksh0;
                size_t pair_offset = ao_pair_loc[pair_ij];
                dm_tensor = dm + pair_offset * naux + k0;
            } else {
                k0 = envs.ao_loc[ksh] - nao;
                dm_tensor = dm + j0 * nao + i0;
            }

            float v_kxx = 0;
            float v_kxy = 0;
            float v_kxz = 0;
            float v_kyy = 0;
            float v_kyz = 0;
            float v_kzz = 0;
            float v_ixkx = 0;
            float v_ixky = 0;
            float v_ixkz = 0;
            float v_iykx = 0;
            float v_iyky = 0;
            float v_iykz = 0;
            float v_izkx = 0;
            float v_izky = 0;
            float v_izkz = 0;
            float v_jxkx = 0;
            float v_jxky = 0;
            float v_jxkz = 0;
            float v_jykx = 0;
            float v_jyky = 0;
            float v_jykz = 0;
            float v_jzkx = 0;
            float v_jzky = 0;
            float v_jzkz = 0;

            int expk = bas[ksh*BAS_SLOTS+PTR_EXP];
            int ck = bas[ksh*BAS_SLOTS+PTR_COEFF];
            int rk = bas[ksh*BAS_SLOTS+PTR_BAS_COORD];

            for (int ijp = 0; ijp < iprim*jprim; ++ijp) {
                int ip = ijp / jprim;
                int jp = ijp - jprim * ip;
                float ai = env[expi+ip];
                float aj = env[expj+jp];
                float aij = ai + aj;
                float aj_aij = aj / aij;
                __syncthreads();
                if (gout_id == 0) {
                    // The primitive prefactor is computed in double and only
                    // then narrowed into the float gx.  It is evaluated once
                    // per (primitive pair, thread), outside the inner loop, so
                    // the fp64 cost is negligible -- while a relative error
                    // here scales that primitive's entire contribution, and in
                    // a contracted basis the primitives cancel against each
                    // other.  __expf's reduced accuracy is the wrong thing to
                    // save time on at this position.
                    double theta_ij = ai * aj_aij;
                    double xjxi = rjri[0*nsp_per_block];
                    double yjyi = rjri[1*nsp_per_block];
                    double zjzi = rjri[2*nsp_per_block];
                    double rr_ij = xjxi*xjxi + yjyi*yjyi + zjzi*zjzi;
                    double Kab = theta_ij * rr_ij;
                    double fac_ij = PI_FAC;
                    if (ish == jsh) {
                        fac_ij *= .5;
                    } else if (ish < jsh) {
                        fac_ij = 0;
                    }
                    double cicj = fac_ij * env[ci+ip] * env[cj+jp];
                    gx[gx_len] = cicj * exp(-Kab);
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
                        // same reasoning as the ij prefactor above
                        double akd = env[expk+kp];
                        double aijd = aij;
                        gx[0] = env[ck+kp] / (aijd*akd*sqrt(aijd+akd));
                    }
                    float xpq = Rpq[0*nst_per_block];
                    float ypq = Rpq[1*nst_per_block];
                    float zpq = Rpq[2*nst_per_block];
                    float rr = xpq*xpq + ypq*ypq + zpq*zpq;
                    // omega == 0 && lr_factor == 1 is enforced by the host
                    // entry, i.e. the omega==0 branch of rys_roots_for_k with
                    // no long-range rescaling
                    rys_roots_f32(nroots, theta*rr, rw, nst_per_block,
                                  gout_id, gout_stride);
                    for (int irys = 0; irys < nroots; ++irys) {
                        int lij = li + lj + 2;
                        int stride_j = li + 2;
                        int stride_k = stride_j * (lj + 2);
                        BUILD_3C_GXYZ_F32(lj+1, lk, nsp_per_block, pair_ij < shl_pair1 && kidx < ksh1);
                        if (pair_ij < shl_pair1 && kidx < ksh1) {
                            int nsp = nsp_per_block;
                            int i_1 =          nst_per_block;
                            int j_1 = stride_j*nst_per_block;
                            int nfi = c_nf[li];
                            int nfj = c_nf[lj];
                            int nfij = nfi * nfj;
                            float div_nfi = c_div_nf[li];
                            float div_nfj = c_div_nf[lj];
                            float div_nfij = div_nfi * div_nfj;
                            float ai2 = ai * 2;
                            float aj2 = aj * 2;
                            for (int n = gout_id; n < nf; n+=gout_stride) {
                                uint32_t k = n * div_nfij;
                                uint32_t ij = n - k * nfij;
                                uint32_t j = ij * div_nfi;
                                uint32_t i = ij - j * nfi;
                                int ix = _c_cartesian_lexical_xyz[idx_i + i*3+0];
                                int iy = _c_cartesian_lexical_xyz[idx_i + i*3+1];
                                int iz = _c_cartesian_lexical_xyz[idx_i + i*3+2];
                                int jx = _c_cartesian_lexical_xyz[idx_j + j*3+0];
                                int jy = _c_cartesian_lexical_xyz[idx_j + j*3+1];
                                int jz = _c_cartesian_lexical_xyz[idx_j + j*3+2];
                                int kx = _c_cartesian_lexical_xyz[idx_k + k*3+0];
                                int ky = _c_cartesian_lexical_xyz[idx_k + k*3+1];
                                int kz = _c_cartesian_lexical_xyz[idx_k + k*3+2];
                                float dm_ijk;
                                if (density_auxvec == NULL) {
                                    dm_ijk = dm_tensor[ij*naux + k*nksh];
                                } else {
                                    dm_ijk = dm_tensor[j*nao+i] * density_auxvec[k0+k];
                                }
                                int addrx = (ix + jx*stride_j + kx*stride_k) * nst;
                                int addry = (iy + jy*stride_j + ky*stride_k + g_size) * nst;
                                int addrz = (iz + jz*stride_j + kz*stride_k + g_size*2) * nst;
                                float Ix = gx[addrx];
                                float Iy = gx[addry];
                                float Iz = gx[addrz];
                                float Ix_d = Ix * dm_ijk;
                                float Iy_d = Iy * dm_ijk;
                                float Iz_d = Iz * dm_ijk;
                                float prod_yz = Iy * Iz_d;
                                float prod_xz = Ix * Iz_d;
                                float prod_xy = Ix * Iy_d;
                                float gix = gx[addrx+i_1];
                                float giy = gx[addry+i_1];
                                float giz = gx[addrz+i_1];
                                float gjx = gx[addrx+j_1];
                                float gjy = gx[addry+j_1];
                                float gjz = gx[addrz+j_1];

                                float f3x, f3y, f3z;
                                float fkkx, fkky, fkkz;
                                float goutx, gouty, goutz;
                                float _gx_inc2, _gy_inc2, _gz_inc2;
                                float fjx = aj2 * gjx;
                                float fjy = aj2 * gjy;
                                float fjz = aj2 * gjz;
                                if (jx > 0) { fjx -= jx * gx[addrx-j_1]; }
                                if (jy > 0) { fjy -= jy * gx[addry-j_1]; }
                                if (jz > 0) { fjz -= jz * gx[addrz-j_1]; }

                                float fix = ai2 * gix;
                                float fiy = ai2 * giy;
                                float fiz = ai2 * giz;
                                if (ix > 0) { fix -= ix * gx[addrx-i_1]; }
                                if (iy > 0) { fiy -= iy * gx[addry-i_1]; }
                                if (iz > 0) { fiz -= iz * gx[addrz-i_1]; }

                                float gijx = gx[addrx+i_1+j_1];
                                float gijy = gx[addry+i_1+j_1];
                                float gijz = gx[addrz+i_1+j_1];
                                f3x = ai2 * gijx;
                                f3y = ai2 * gijy;
                                f3z = ai2 * gijz;
                                if (ix > 0) { f3x -= ix * gx[addrx-i_1+j_1]; }
                                if (iy > 0) { f3y -= iy * gx[addry-i_1+j_1]; }
                                if (iz > 0) { f3z -= iz * gx[addrz-i_1+j_1]; }
                                f3x *= aj2;
                                f3y *= aj2;
                                f3z *= aj2;
                                if (jx > 0) {
                                    float fx = ai2 * gx[addrx+i_1-j_1];
                                    if (ix > 0) { fx -= ix * gx[addrx-i_1-j_1]; }
                                    f3x -= jx * fx;
                                }
                                if (jy > 0) {
                                    float fy = ai2 * gx[addry+i_1-j_1];
                                    if (iy > 0) { fy -= iy * gx[addry-i_1-j_1]; }
                                    f3y -= jy * fy;
                                }
                                if (jz > 0) {
                                    float fz = ai2 * gx[addrz+i_1-j_1];
                                    if (iz > 0) { fz -= iz * gx[addrz-i_1-j_1]; }
                                    f3z -= jz * fz;
                                }
                                fkkx = f3x * 2;
                                fkky = f3y * 2;
                                fkkz = f3z * 2;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v1xx += goutx;
                                v1yy += gouty;
                                v1zz += goutz;
                                v_ixkx -= goutx; // ixjx in ixkx = -ixix - ixjx
                                v_iyky -= gouty;
                                v_izkz -= goutz;
                                v_jxkx -= goutx; // jxix in jxkx = -jxix - jxjx
                                v_jyky -= gouty;
                                v_jzkz -= goutz;
                                float goutxy = fix * fjy * Iz_d;
                                float goutxz = fix * fjz * Iy_d;
                                float goutyx = fiy * fjx * Iz_d;
                                float goutyz = fiy * fjz * Ix_d;
                                float goutzx = fiz * fjx * Iy_d;
                                float goutzy = fiz * fjy * Ix_d;
                                v1xy += goutxy;
                                v1xz += goutxz;
                                v1yx += goutyx;
                                v1yz += goutyz;
                                v1zx += goutzx;
                                v1zy += goutzy;
                                v_ixky -= goutxy; // ixky = -ixiy - ixjy
                                v_ixkz -= goutxz;
                                v_iykx -= goutyx;
                                v_iykz -= goutyz;
                                v_izkx -= goutzx;
                                v_izky -= goutzy;
                                v_jxky -= goutyx; // jxky = -jxiy - jxjy
                                v_jxkz -= goutzx;
                                v_jykx -= goutxy;
                                v_jykz -= goutzy;
                                v_jzkx -= goutxz;
                                v_jzky -= goutyz;

                                float xjxi = rjri[0*nsp];
                                float yjyi = rjri[1*nsp];
                                float zjzi = rjri[2*nsp];
                                _gx_inc2 = gijx - gjx * xjxi;
                                _gy_inc2 = gijy - gjy * yjyi;
                                _gz_inc2 = gijz - gjz * zjzi;
                                f3x = aj2 * (aj2 * _gx_inc2 - (2*jx+1) * Ix);
                                f3y = aj2 * (aj2 * _gy_inc2 - (2*jy+1) * Iy);
                                f3z = aj2 * (aj2 * _gz_inc2 - (2*jz+1) * Iz);
                                if (jx > 1) { f3x += jx*(jx-1) * gx[addrx-j_1*2]; }
                                if (jy > 1) { f3y += jy*(jy-1) * gx[addry-j_1*2]; }
                                if (jz > 1) { f3z += jz*(jz-1) * gx[addrz-j_1*2]; }
                                fkkx += f3x;
                                fkky += f3y;
                                fkkz += f3z;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v_jxx += goutx;
                                v_jyy += gouty;
                                v_jzz += goutz;
                                v_jxkx -= goutx; // jxjx in jxkx = -jxix - jxjx
                                v_jyky -= gouty;
                                v_jzkz -= goutz;
                                goutz = fjx * fjy * Iz_d;
                                gouty = fjx * fjz * Iy_d;
                                goutx = fjy * fjz * Ix_d;
                                v_jxy += goutz;
                                v_jxz += gouty;
                                v_jyz += goutx;
                                v_jxky -= goutz; // ixky = -ixiy - ixjy
                                v_jxkz -= gouty;
                                v_jykx -= goutz;
                                v_jykz -= goutx;
                                v_jzkx -= gouty;
                                v_jzky -= goutx;

                                _gx_inc2 = gijx + gix * xjxi;
                                _gy_inc2 = gijy + giy * yjyi;
                                _gz_inc2 = gijz + giz * zjzi;
                                f3x = ai2 * (ai2 * _gx_inc2 - (2*ix+1) * Ix);
                                f3y = ai2 * (ai2 * _gy_inc2 - (2*iy+1) * Iy);
                                f3z = ai2 * (ai2 * _gz_inc2 - (2*iz+1) * Iz);
                                if (ix > 1) { f3x += ix*(ix-1) * gx[addrx-i_1*2]; }
                                if (iy > 1) { f3y += iy*(iy-1) * gx[addry-i_1*2]; }
                                if (iz > 1) { f3z += iz*(iz-1) * gx[addrz-i_1*2]; }
                                fkkx += f3x;
                                fkky += f3y;
                                fkkz += f3z;
                                goutx = f3x * prod_yz;
                                gouty = f3y * prod_xz;
                                goutz = f3z * prod_xy;
                                v_ixx += goutx;
                                v_iyy += gouty;
                                v_izz += goutz;
                                v_ixkx -= goutx; // ixix in ixkx = -ixix - ixjx
                                v_iyky -= gouty;
                                v_izkz -= goutz;
                                goutz = fix * fiy * Iz_d;
                                gouty = fix * fiz * Iy_d;
                                goutx = fiy * fiz * Ix_d;
                                v_ixy += goutz;
                                v_ixz += gouty;
                                v_iyz += goutx;
                                v_ixky -= goutz; // ixky = -ixiy - ixjy
                                v_ixkz -= gouty;
                                v_iykx -= goutz;
                                v_iykz -= goutx;
                                v_izkx -= gouty;
                                v_izky -= goutx;

                                float fkx = -fix - fjx;
                                float fky = -fiy - fjy;
                                float fkz = -fiz - fjz;
                                v_kxx += fkkx * prod_yz;
                                v_kyy += fkky * prod_xz;
                                v_kzz += fkkz * prod_xy;
                                v_kxy += fkx * fky * Iz_d;
                                v_kxz += fkx * fkz * Iy_d;
                                v_kyz += fky * fkz * Ix_d;
                            }
                        }
                    }
                }
            }
            if (pair_ij < shl_pair1 && kidx < ksh1) {
                int ia = bas[ish*BAS_SLOTS+ATOM_OF];
                int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
                int ka = bas[ksh*BAS_SLOTS+ATOM_OF] - envs.natm;
                int natm = envs.natm;
                atomicAdd(ejk + (ka*natm+ka)*9 + 0, v_kxx * .5);
                atomicAdd(ejk + (ka*natm+ka)*9 + 3, v_kxy     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 4, v_kyy * .5);
                atomicAdd(ejk + (ka*natm+ka)*9 + 6, v_kxz     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 7, v_kyz     );
                atomicAdd(ejk + (ka*natm+ka)*9 + 8, v_kzz * .5);

                atomicAdd(ejk + (ia*natm+ka)*9 + 0, v_ixkx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 1, v_ixky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 2, v_ixkz);
                atomicAdd(ejk + (ia*natm+ka)*9 + 3, v_iykx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 4, v_iyky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 5, v_iykz);
                atomicAdd(ejk + (ia*natm+ka)*9 + 6, v_izkx);
                atomicAdd(ejk + (ia*natm+ka)*9 + 7, v_izky);
                atomicAdd(ejk + (ia*natm+ka)*9 + 8, v_izkz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 0, v_jxkx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 1, v_jxky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 2, v_jxkz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 3, v_jykx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 4, v_jyky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 5, v_jykz);
                atomicAdd(ejk + (ja*natm+ka)*9 + 6, v_jzkx);
                atomicAdd(ejk + (ja*natm+ka)*9 + 7, v_jzky);
                atomicAdd(ejk + (ja*natm+ka)*9 + 8, v_jzkz);
            }
        }
        if (pair_ij < shl_pair1) {
            int ia = bas[ish*BAS_SLOTS+ATOM_OF];
            int ja = bas[jsh*BAS_SLOTS+ATOM_OF];
            int natm = envs.natm;
            atomicAdd(ejk + (ia*natm+ja)*9 + 0, v1xx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 1, v1xy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 2, v1xz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 3, v1yx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 4, v1yy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 5, v1yz);
            atomicAdd(ejk + (ia*natm+ja)*9 + 6, v1zx);
            atomicAdd(ejk + (ia*natm+ja)*9 + 7, v1zy);
            atomicAdd(ejk + (ia*natm+ja)*9 + 8, v1zz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 0, v_ixx*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 3, v_ixy);
            atomicAdd(ejk + (ia*natm+ia)*9 + 4, v_iyy*.5);
            atomicAdd(ejk + (ia*natm+ia)*9 + 6, v_ixz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 7, v_iyz);
            atomicAdd(ejk + (ia*natm+ia)*9 + 8, v_izz*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 0, v_jxx*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 3, v_jxy);
            atomicAdd(ejk + (ja*natm+ja)*9 + 4, v_jyy*.5);
            atomicAdd(ejk + (ja*natm+ja)*9 + 6, v_jxz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 7, v_jyz);
            atomicAdd(ejk + (ja*natm+ja)*9 + 8, v_jzz*.5);
        }
    }
}

extern "C" {
// Float32 counterpart of ejk_int3c2e_ip2 (see ejk_int3c2e_ip2.cu).
// dm and density_auxvec are float32 arrays; ejk remains float64.
// omega/lr_factor/sr_factor are kept in the signature so that the Python
// dispatch can hand both kernels the same argument list, but only the plain
// Coulomb operator is implemented: anything else returns 2 rather than
// silently computing the wrong integrals.
int ejk_int3c2e_ip2_f32(double *ejk, float *dm, float *density_auxvec,
                    RysIntEnvVars *envs, double omega, double lr_factor,
                    double sr_factor, int shm_size, int nbatches_shl_pair,
                    int nbatches_ksh, int *shl_pair_offsets, uint32_t *bas_ij_idx,
                    int *ksh_offsets, int *gout_stride_lookup,
                    int *ao_pair_loc, int aux_offset, int naux)
{
    if (omega != 0 || lr_factor != 1) {
        fprintf(stderr, "ejk_int3c2e_ip2_f32: range separation is not ported "
                "(omega=%g, lr_factor=%g); use the float64 kernel\n",
                omega, lr_factor);
        return 2;
    }
    cudaFuncSetAttribute(ejk_int3c2e_ip2_kernel_f32, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_size);
    dim3 blocks(nbatches_shl_pair, nbatches_ksh);
    ejk_int3c2e_ip2_kernel_f32<<<blocks, THREADS, shm_size>>>(
            ejk, dm, density_auxvec, *envs,
            shl_pair_offsets, bas_ij_idx, ksh_offsets,
            gout_stride_lookup, ao_pair_loc, aux_offset, naux);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error in ejk_int3c2e_ip2_f32: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}

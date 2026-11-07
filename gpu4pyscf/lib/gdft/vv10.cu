/*
 * Copyright 2021-2024 The PySCF Developers. All Rights Reserved.
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

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "gint/gint.h"
#include "gint/cuda_alloc.cuh"
#include "nr_eval_gto.cuh"
#include "contract_rho.cuh"

#define NG_PER_BLOCK      128

// Block-wide tree reduction of three doubles (used by the multi-warp VV10
// UWE kernel). blockDim.x is a power of two in every caller, and every
// thread enters with __syncthreads() balanced, so the barriers below are
// uniform across the block.
__device__
static void _vv10_block_reduce3(double& U, double& W, double& E)
{
    __shared__ double s_U[NG_PER_BLOCK];
    __shared__ double s_W[NG_PER_BLOCK];
    __shared__ double s_E[NG_PER_BLOCK];
    s_U[threadIdx.x] = U;
    s_W[threadIdx.x] = W;
    s_E[threadIdx.x] = E;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_U[threadIdx.x] += s_U[threadIdx.x + stride];
            s_W[threadIdx.x] += s_W[threadIdx.x + stride];
            s_E[threadIdx.x] += s_E[threadIdx.x + stride];
        }
        __syncthreads();
    }
    U = s_U[0];
    W = s_W[0];
    E = s_E[0];
}

// Issue #10 candidate C1 (rework): the REFERENCE kernel below (warp1)
// assigns one grid point i to each THREAD and one 128-thread block to 128
// consecutive i points; it stages each 128-point j chunk into shared memory
// ONCE per block and reuses it for all 128 i points of the block (minimum
// possible global read amplification: each j element is read once per
// ngrids/128 blocks). Its limitation is parallelism, not traffic: with one
// thread per i the kernel exposes only ngrids threads, so small grids
// (e.g. ~58k points -> 452 blocks) under-occupy the GPU and run the fp64
// pipes at a fraction of the achievable rate, while large grids already
// sit near the fp64 roofline.
//
// The multi-warp variant (warp4, kept below for continuity) is the opposite
// trade: one block per i with all 128 threads cooperating on that i's
// j-loop, which restores parallelism but restages the whole j grid in every
// block (128x global read amplification) and turns L2 bandwidth into the
// binding limit at benzene-scale grids.
//
// The multi-i tile kernel (this file, selected via GPU4PYSCF_VV10_UWE_KERNEL
// = tile8/tile16/tile32) interpolates between the two: one block serves
// I_TILE i points, and the 128 threads are split into groups of
// 128/I_TILE that cooperatively walk each i's j-loop over the block-shared
// staged j chunk. Global read amplification becomes 128/I_TILE (16x/8x/4x
// for tile8/tile16/tile32 respectively, vs warp4's 128x and the reference's
// 1x) while exposed parallelism becomes ngrids * 128/I_TILE threads. The
// per-pair fp64
// arithmetic expression is IDENTICAL to the reference kernel; the result
// differs from the reference only in fp64 summation order (per-i partials
// over 128/I_TILE strided j subsets combined by a within-warp shuffle
// tree reduction instead of one serial chain), the same class of
// reduction-order noise already probed for warp4 (issue #10 precision
// probe: <= 4.3e-16 kernel-level, <= 5.7e-14 Ha SCF-level).
//
// Boundary handling: I_TILE i points per block means the last block may
// carry fewer than I_TILE active i points (partial tile) and the last j
// chunk may be shorter than 128 (tail chunk). No thread ever exits early:
// inactive i groups keep executing the uniform chunk loop (their loads are
// skipped and their accumulators stay zero) so every __syncthreads() in the
// loop is reached by all 128 threads, and staged shared slots beyond the
// tail are never read because the compute loop bound is the same uniform
// block_upper_bound used by the reference kernel. Ownership: exactly one
// thread (group lane 0) writes each active i's U/W/E, so there is no
// write race between groups.
//
// Selected at runtime via GPU4PYSCF_VV10_UWE_KERNEL (unset/empty = warp1
// reference, the default; 'warp4' = multi-warp; 'tile8'/'tile16'/'tile32'
// = multi-i tile); the default never changes without that opt-in.
template <int I_TILE>
__global__
static void vv10_fock_eval_UWE_multii_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                             const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                             const double* __restrict__ omega, const double* __restrict__ kappa,
                                             const int ngrids)
{
    constexpr int THREADS_PER_I = NG_PER_BLOCK / I_TILE;
    static_assert(THREADS_PER_I >= 4 && THREADS_PER_I <= 32,
                  "multi-i tile keeps the per-i reduction inside one warp "
                  "(I_TILE in 4..32; I_TILE=128 is the reference kernel)");
    const int group = threadIdx.x / THREADS_PER_I;
    const int lane_in_group = threadIdx.x % THREADS_PER_I;
    const int i = blockIdx.x * I_TILE + group;
    const bool active = i < ngrids;

    double omega_i = NAN;
    double kappa_i = NAN;
    double3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        // group lanes walk the staged chunk with stride THREADS_PER_I, so
        // the group's lanes together cover every staged j for their i
        for (int j_in_block = lane_in_group; j_in_block < block_upper_bound; j_in_block += THREADS_PER_I) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    // per-i reduction inside the owning warp: THREADS_PER_I is 4..32 and
    // groups are aligned contiguous lane ranges (NG_PER_BLOCK is a
    // multiple of 32 and THREADS_PER_I divides 32), so a shuffle tree
    // reduction over THREADS_PER_I lanes needs no shared memory and no
    // barrier. All threads reach here (no early return above), so the
    // full-warp sync mask is uniform.
    #pragma unroll
    for (int stride = THREADS_PER_I / 2; stride > 0; stride >>= 1) {
        U_i += __shfl_down_sync(0xffffffffu, U_i, stride);
        W_i += __shfl_down_sync(0xffffffffu, W_i, stride);
        E_i += __shfl_down_sync(0xffffffffu, E_i, stride);
    }
    if (lane_in_group == 0 && active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

// The fp32-safe mixed-precision UWE kernel (LC-80, review 2026-10-12
// continuation): the SAME warp1 block/j-staging structure as the
// reference kernel, but the per-pair arithmetic runs in fp32 with the
// numerically safe expression measured in LC-79
// (tasks/issue-9/2026-10-13-lc79-vv10-lever):
//   * omega/kappa/rho_weight arrive as fp64 (computed by the fp64
//     omega_derivative kernel); they are cast to fp32 ONCE at staging.
//     This keeps the rho^-4 overflow (measured 8.5e39 at nonzero-rho
//     ~1e-10 grid points) OUT of the fp32 domain -- a fully-fp32 VV10
//     is structurally NaN there.
//   * the three reciprocals are computed as separate 1/g factors; the
//     reference's 1/(g_ij*g_ji*g_sum) overflows fp32 on far pairs
//     (measured 154k pairs with g-prod above FLT_MAX on benzene/tzvp).
//   * accumulation: per-j-chunk (128-term) partial sums in fp32,
//     combined into an fp64 accumulator. A pure serial fp32 chain over
//     the full ngrids (127k terms on benzene/tzvp) was measured at
//     dE +1.5e-6 Ha (compiler reassociation makes the serial order
//     uncontrollable), while chunked partials + fp64 combine measured
//     dE -2.6e-11 Ha on the same data (LC-80 diag_accum.py); the
//     cupy-tree equivalent measured +8.5e-10. The chunk partial is a
//     plain fp32 sum of 128 contiguous terms (reassociation within a
//     chunk is bounded by 128 eps against terms of one chunk).
//   * coordinates are staged relative to the first grid point: the
//     origin is subtracted in fp64 BEFORE the fp32 cast (independent
//     review 2026-10-17; absolute-coordinate staging broke rigid-
//     translation invariance, measured +5.27e-6 Ha at 1e6 Bohr).
// Selected via GPU4PYSCF_VV10_UWE_KERNEL=fp32 (opt-in; default and all
// other selectors unchanged).
__global__
static void vv10_fock_eval_UWE_fp32safe_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                               const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                               const double* __restrict__ omega, const double* __restrict__ kappa,
                                               const int ngrids)
{
    const int i = blockIdx.x * NG_PER_BLOCK + threadIdx.x;
    const bool active = i < ngrids;

    // Translation-safe staging (independent review 2026-10-17): the fp32
    // cast must never see ABSOLUTE coordinates.  At a distant origin
    // (measured on a rigid 1e6-Bohr translation of every NLC grid point,
    // rho/weights/omega/kappa unchanged) the fp32 mantissa quantizes the
    // low bits of each coordinate and the pair differences inherit the
    // quantization: dE_NLC +5.27e-6 Ha on methanol/def2-SVP/wb97m-v,
    // while the fp64 kernels stayed at ~7e-15.  Subtracting the first
    // grid point in fp64 before the cast keeps every staged value
    // molecule-scale (the grid spans one molecule, so the origin is
    // always near) and restores rigid-translation invariance: a common
    // shift moves the origin with the grid and cancels in the fp64
    // subtraction, leaving only fp64-rounding ~1e-10 Bohr before the
    // cast -- the same error class as the unshifted molecule.
    const double origin_x = grid_coord[0];
    const double origin_y = grid_coord[1];
    const double origin_z = grid_coord[2];

    float omega_i = NAN;
    float kappa_i = NAN;
    float3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = __double2float_rn(omega[i]);
        kappa_i = __double2float_rn(kappa[i]);
        r_i.x = __double2float_rn(grid_coord[i * 3 + 0] - origin_x);
        r_i.y = __double2float_rn(grid_coord[i * 3 + 1] - origin_y);
        r_i.z = __double2float_rn(grid_coord[i * 3 + 2] - origin_z);
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ float3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ float3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = __double2float_rn(omega[j]);
            shared_omega_kappa_rhow_j[threadIdx.x].y = __double2float_rn(kappa[j]);
            shared_omega_kappa_rhow_j[threadIdx.x].z = __double2float_rn(rho_weight[j]);
            shared_r_j[threadIdx.x].x = __double2float_rn(grid_coord[j * 3 + 0] - origin_x);
            shared_r_j[threadIdx.x].y = __double2float_rn(grid_coord[j * 3 + 1] - origin_y);
            shared_r_j[threadIdx.x].z = __double2float_rn(grid_coord[j * 3 + 2] - origin_z);
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        float U_chunk = 0;
        float W_chunk = 0;
        float E_chunk = 0;
        for (int j_in_block = 0; j_in_block < block_upper_bound; ++j_in_block) {
            const float omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const float kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const float3 r_j = shared_r_j[j_in_block];
            const float rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const float dx = r_i.x - r_j.x;
            const float dy = r_i.y - r_j.y;
            const float dz = r_i.z - r_j.z;
            const float r_ij2 = dx * dx + dy * dy + dz * dz;
            const float g_ij = omega_i * r_ij2 + kappa_i;
            const float g_ji = omega_j * r_ij2 + kappa_j;
            const float g_sum = g_ij + g_ji;
            // three separate reciprocals: 1/(g_ij*g_ji*g_sum) overflows
            // fp32 on far pairs (measured; see the kernel comment above)
            const float inv_gij = 1.0f / g_ij;
            const float inv_gji = 1.0f / g_ji;
            const float inv_gsum = 1.0f / g_sum;
            const float Phi_ij = -(inv_gij * inv_gji * inv_gsum);
            const float E_ij = rho_weight_j * Phi_ij;
            const float U_ij = E_ij * (g_sum + g_ij) * g_ji * inv_gij * inv_gji * inv_gsum;
            const float W_ij = U_ij * r_ij2;

            U_chunk += U_ij;
            W_chunk += W_ij;
            E_chunk += E_ij;
        }
        // chunk partials (<= 128 fp32 terms) combined in fp64: bounds
        // any within-chunk reassociation and keeps the accumulator
        // error at the fp64 level (measured dE -2.6e-11 Ha benzene/tzvp)
        U_i += double(U_chunk);
        W_i += double(W_chunk);
        E_i += double(E_chunk);
        __syncthreads();
    }

    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

// The multi-warp variant (issue #10 candidate C1 round 1): one block per
// grid point i, with all NG_PER_BLOCK threads cooperating on that single
// i's j-loop (thread t takes every NG_PER_BLOCK-th j; per-i sums via block
// reduction). Compared with the reference kernel this raises exposed
// parallelism from ngrids threads to ngrids*128 threads but restages the
// entire j grid in every block (128x global read amplification), which
// makes L2 bandwidth the binding limit at benzene-scale grids; measured
// end-to-end it was 0.944x-1.152x by cell, so it stays opt-in. The fp64
// arithmetic expression per pair is identical to the reference kernel;
// only the fp64 summation order differs (block tree reduction of 128
// partials instead of one serial chain).
__global__
static void vv10_fock_eval_UWE_multiwarp_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                                const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                const double* __restrict__ omega, const double* __restrict__ kappa,
                                                const int ngrids)
{
    const int i = blockIdx.x;
    const bool active = i < ngrids;

    double omega_i = NAN;
    double kappa_i = NAN;
    double3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        // thread t takes j_in_block = t, t+blockDim.x, ... within this
        // 128-j chunk: all four warps cooperate on the j-loop
        for (int j_in_block = threadIdx.x; j_in_block < block_upper_bound; j_in_block += blockDim.x) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    _vv10_block_reduce3(U_i, W_i, E_i);
    if (threadIdx.x == 0 && active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

__global__
static void vv10_fock_eval_UWE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ E,
                                      const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                      const double* __restrict__ omega, const double* __restrict__ kappa,
                                      const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < ngrids;

    double omega_i = NAN;
    double kappa_i = NAN;
    double3 r_i = { NAN, NAN, NAN };
    if (active) {
        omega_i = omega[i];
        kappa_i = kappa[i];
        r_i.x = grid_coord[i * 3 + 0];
        r_i.y = grid_coord[i * 3 + 1];
        r_i.z = grid_coord[i * 3 + 2];
    }

    double U_i = 0;
    double W_i = 0;
    double E_i = 0;

    __shared__ double3 shared_omega_kappa_rhow_j[NG_PER_BLOCK];
    __shared__ double3 shared_r_j[NG_PER_BLOCK];

    for (int j_block_offset = 0; j_block_offset < ngrids; j_block_offset += NG_PER_BLOCK) {
        const int j = j_block_offset + threadIdx.x;
        if (j < ngrids) {
            shared_omega_kappa_rhow_j[threadIdx.x].x = omega[j];
            shared_omega_kappa_rhow_j[threadIdx.x].y = kappa[j];
            shared_omega_kappa_rhow_j[threadIdx.x].z = rho_weight[j];
            shared_r_j[threadIdx.x].x = grid_coord[j * 3 + 0];
            shared_r_j[threadIdx.x].y = grid_coord[j * 3 + 1];
            shared_r_j[threadIdx.x].z = grid_coord[j * 3 + 2];
        }
        __syncthreads();

        const int block_upper_bound = min(NG_PER_BLOCK, ngrids - j_block_offset);
        for (int j_in_block = 0; j_in_block < block_upper_bound; j_in_block++) {
            const double omega_j = shared_omega_kappa_rhow_j[j_in_block].x;
            const double kappa_j = shared_omega_kappa_rhow_j[j_in_block].y;
            const double3 r_j = shared_r_j[j_in_block];
            const double rho_weight_j = shared_omega_kappa_rhow_j[j_in_block].z;

            const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
            const double g_ij = omega_i * r_ij2 + kappa_i;
            const double g_ji = omega_j * r_ij2 + kappa_j;
            const double g_sum = g_ij + g_ji;
            const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
            const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

            const double E_ij = rho_weight_j * Phi_ij;
            const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
            const double W_ij = U_ij * r_ij2;

            U_i += U_ij;
            W_i += W_ij;
            E_i += E_ij;
        }
        __syncthreads();
    }

    if (active) {
        U[i] = -U_i * 1.5;
        W[i] = -W_i * 1.5;
        E[i] = E_i * 1.5;
    }
}

__global__
static void vv10_fock_eval_omega_derivative_kernel(double* __restrict__ omega, double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega1 = sqrt(omega2);
    const double omega_1 = 1 / omega1;

    omega[i] = omega1;
    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;
}

__global__
static void vv10_grad_eval_E_grid_response_offdiagonal_kernel(double* __restrict__ Egr,
                                                              const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                              const double* __restrict__ omega, const double* __restrict__ kappa,
                                                              const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                              const int natoms, const int i_grid_begin, const int ngrids)
{
    const int i_unoffset = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i_unoffset >= ngrids || B_atom >= natoms)
        return;
    const int i = i_unoffset + i_grid_begin;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }
    if (i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ij = { r_i.x - r_j.x, r_i.y - r_j.y, r_i.z - r_j.z };
        const double r_ij2 = r_ij.x * r_ij.x + r_ij.y * r_ij.y + r_ij.z * r_ij.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = (omega_i * g_ji * g_sum + omega_j * g_ij * g_sum + (omega_i + omega_j) * g_ij * g_ji) * g_ij_ji_sum_1;
        const double Egr_ij = E_ij * dPhi_drj_over_Phi;

        Egr_i.x += Egr_ij * r_ij.x;
        Egr_i.y += Egr_ij * r_ij.y;
        Egr_i.z += Egr_ij * r_ij.z;
    }

    Egr[B_atom * 3 * ngrids + 0 * ngrids + i_unoffset] = 1.5 * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i_unoffset] = 1.5 * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i_unoffset] = 1.5 * Egr_i.z;
}

__global__
static void vv10_hess_eval_UWABCE_kernel(double* __restrict__ U, double* __restrict__ W, double* __restrict__ A, double* __restrict__ B, double* __restrict__ C, double* __restrict__ E,
                                         const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                         const double* __restrict__ omega, const double* __restrict__ kappa,
                                         const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double U_i = 0;
    double W_i = 0;
    double A_i = 0;
    double B_i = 0;
    double C_i = 0;
    double E_i = 0;

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_sum_1 = g_ji * g_ij_ji_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
        const double U_ij = E_ij * (g_ij + g_sum) * g_ij_sum_1;
        const double W_ij = U_ij * r_ij2;
        const double A_ij = E_ij * (g_ij * g_ij + g_ij * g_sum + g_sum * g_sum) * g_ij_sum_1 * g_ij_sum_1;
        const double B_ij = A_ij * r_ij2;
        const double C_ij = B_ij * r_ij2;

        U_i += U_ij;
        W_i += W_ij;
        A_i += A_ij;
        B_i += B_ij;
        C_i += C_ij;
        E_i += E_ij;
    }

    U[i] = -1.5 * U_i;
    W[i] = -1.5 * W_i;
    A[i] = (1.5 * 2) * A_i;
    B[i] = (1.5 * 2) * B_i;
    C[i] = (1.5 * 2) * C_i;
    E[i] = 1.5 * E_i;
}

__global__
static void vv10_hess_eval_omega_derivative_kernel(double* __restrict__ omega, double* __restrict__ domega_drho, double* __restrict__ domega_dgamma,
                                                   double* __restrict__ d2omega_drho2, double* __restrict__ d2omega_dgamma2, double* __restrict__ d2omega_drho_dgamma,
                                                   const double* __restrict__ rho, const double* __restrict__ gamma, const double C_factor,
                                                   const int ngrids)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double gamma_i = gamma[i];

    const double rho_1 = 1 / rho_i;
    const double rho_2 = rho_1 * rho_1;
    const double rho_3 = rho_1 * rho_2;
    const double rho_4 = rho_2 * rho_2;
    const double rho_5 = rho_1 * rho_4;
    const double gamma2 = gamma_i * gamma_i;
    constexpr double four_pi_over_three = 4.0 / 3.0 * M_PI;
    const double omega2 = C_factor * gamma2 * rho_4 + four_pi_over_three * rho_i;
    const double omega1 = sqrt(omega2);
    const double omega_1 = 1 / omega1;

    omega[i] = omega1;
    domega_drho[i] = 0.5 * (four_pi_over_three - 4 * C_factor * gamma2 * rho_5) * omega_1;
    domega_dgamma[i] = C_factor * gamma_i * rho_4 * omega_1;

    const double omega_3 = omega_1 / omega2;
    d2omega_drho2[i] = (-0.25 * four_pi_over_three * four_pi_over_three
                        + 12 * four_pi_over_three * C_factor * gamma2 * rho_5
                        + 6 * C_factor * C_factor * gamma2 * gamma2 * rho_5 * rho_5) * omega_3;
    d2omega_dgamma2[i] = four_pi_over_three * C_factor * rho_3 * omega_3;
    d2omega_drho_dgamma[i] = -C_factor * gamma_i * (4.5 * four_pi_over_three * rho_4
                                                    + 2 * C_factor * gamma2 * rho_4 * rho_5) * omega_3;
}

template <int n_trial_per_thread>
__global__
static void vv10_hess_eval_f_t_offdiagonal_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                                  const double* __restrict__ grid_coord, const double* __restrict__ grid_weight,
                                                  const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                  const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                                  const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                                  const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_trial_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_trial_per_thread;
    if (i >= ngrids || i_trial_start >= ntrial)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    double f_rho_t_i[n_trial_per_thread] {0};
    double f_gamma_t_i[n_trial_per_thread] {0};
    // #pragma unroll
    // for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
    //     f_rho_t_i  [i_trial] = 0;
    //     f_gamma_t_i[i_trial] = 0;
    // }

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double domega_drho_j = domega_drho[j];
        const double domega_dgamma_j = domega_dgamma[j];
        const double dkappa_drho_j = dkappa_drho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double rho_dgdrho_i = rho_i * (r_ij2 * domega_drho_i + dkappa_drho_i);
        const double rho_dgdrho_j = rho_j * (r_ij2 * domega_drho_j + dkappa_drho_j);
        const double d2Phi_dgij_dgji_over_Phi = 2 * (g_sum_1 * g_sum_1 + g_ij_ji_1);

        const double f_rho_rho_ij = Phi_ij * (rho_dgdrho_i * rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi
                                              - rho_dgdrho_i * (g_sum_1 + g_ij_1)
                                              - rho_dgdrho_j * (g_sum_1 + g_ji_1) + 1);
        const double rho_domegadgamma_i_rij2 = rho_i * domega_dgamma_i * r_ij2;
        const double f_gamma_rho_ij = rho_domegadgamma_i_rij2 * Phi_ij * (rho_dgdrho_j * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ij_1));
        const double rho_domegadgamma_j_rij2 = rho_j * domega_dgamma_j * r_ij2;
        const double f_rho_gamma_ij = rho_domegadgamma_j_rij2 * Phi_ij * (rho_dgdrho_i * d2Phi_dgij_dgji_over_Phi - (g_sum_1 + g_ji_1));
        const double f_gamma_gamma_ij = rho_domegadgamma_i_rij2 * rho_domegadgamma_j_rij2 * Phi_ij * d2Phi_dgij_dgji_over_Phi;

        const double weight_j = grid_weight[j];

        #pragma unroll
        for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
            if (i_trial + i_trial_start >= ntrial) continue;
            const double   rho_t_j =   rho_t[(i_trial + i_trial_start) * ngrids + j];
            const double gamma_t_j = gamma_t[(i_trial + i_trial_start) * ngrids + j];
            f_rho_t_i  [i_trial] += weight_j * (  f_rho_rho_ij * rho_t_j +   f_rho_gamma_ij * gamma_t_j);
            f_gamma_t_i[i_trial] += weight_j * (f_gamma_rho_ij * rho_t_j + f_gamma_gamma_ij * gamma_t_j);
        }
    }

    #pragma unroll
    for (int i_trial = 0; i_trial < n_trial_per_thread; i_trial++) {
        if (i_trial + i_trial_start >= ntrial) continue;
        f_rho_t  [(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_rho_t_i  [i_trial];
        f_gamma_t[(i_trial + i_trial_start) * ngrids + i] = 1.5 * f_gamma_t_i[i_trial];
    }
}

__global__
static void vv10_hess_eval_f_t_diagonal_kernel(double* __restrict__ f_rho_t, double* __restrict__ f_gamma_t,
                                               const double* __restrict__ rho,
                                               const double* __restrict__ U, const double* __restrict__ W, const double* __restrict__ A, const double* __restrict__ B, const double* __restrict__ C,
                                               const double* __restrict__ domega_drho, const double* __restrict__ domega_dgamma, const double* __restrict__ dkappa_drho,
                                               const double* __restrict__ d2omega_drho2, const double* __restrict__ d2omega_dgamma2, const double* __restrict__ d2omega_drho_dgamma, const double* __restrict__ d2kappa_drho2,
                                               const double* __restrict__ rho_t, const double* __restrict__ gamma_t,
                                               const int ngrids, const int ntrial)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ngrids)
        return;

    const double rho_i = rho[i];
    const double domega_drho_i = domega_drho[i];
    const double domega_dgamma_i = domega_dgamma[i];
    const double dkappa_drho_i = dkappa_drho[i];

    const double U_i = U[i];
    const double W_i = W[i];
    const double A_i = A[i];
    const double B_i = B[i];
    const double C_i = C[i];
    const double d2omega_drho2_i = d2omega_drho2[i];
    const double d2omega_dgamma2_i = d2omega_dgamma2[i];
    const double d2omega_drho_dgamma_i = d2omega_drho_dgamma[i];
    const double d2kappa_drho2_i = d2kappa_drho2[i];

    const double f_rho_rho_ii = 2 * domega_drho_i * W_i + 2 * dkappa_drho_i * U_i
                                + rho_i * (d2omega_drho2_i * W_i + d2kappa_drho2_i * U_i + dkappa_drho_i * dkappa_drho_i * A_i
                                           + domega_drho_i * domega_drho_i * C_i + 2 * domega_drho_i * dkappa_drho_i * B_i);
    const double f_gamma_rho_ii = domega_dgamma_i * W_i + rho_i * (d2omega_drho_dgamma_i * W_i
                                                                   + domega_dgamma_i * (dkappa_drho_i * B_i + domega_drho_i * C_i));
    const double f_rho_gamma_ii = f_gamma_rho_ii;
    const double f_gamma_gamma_ii = rho_i * (d2omega_dgamma2_i * W_i + domega_dgamma_i * domega_dgamma_i * C_i);

    for (int i_trial = 0; i_trial < ntrial; i_trial++) {
        const double rho_t_i   =   rho_t[i_trial * ngrids + i];
        const double gamma_t_i = gamma_t[i_trial * ngrids + i];

        f_rho_t  [i_trial * ngrids + i] += (  f_rho_rho_ii * rho_t_i +   f_rho_gamma_ii * gamma_t_i);
        f_gamma_t[i_trial * ngrids + i] += (f_gamma_rho_ii * rho_t_i + f_gamma_gamma_ii * gamma_t_i);
    }
}

__global__
static void vv10_hess_eval_EUW_grid_response_offdiagonal_kernel(double* __restrict__ Egr, double* __restrict__ Ugr, double* __restrict__ Wgr,
                                                                const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                                const double* __restrict__ omega, const double* __restrict__ kappa,
                                                                const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                                const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0 || i_associated_atom == B_atom) {
        Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] = 0;
        Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double3 Egr_i = { 0, 0, 0 };
    double3 Ugr_i = { 0, 0, 0 };
    double3 Wgr_i = { 0, 0, 0 };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];

        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later
        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double E_ij = rho_weight_j * Phi_ij;
        const double dPhi_drj_over_Phi = omega_i * g_ij_1 + omega_j * g_ji_1 + (omega_i + omega_j) * g_sum_1;
        const double d2Phi_dgij_drj_over_Phi = omega_i * g_ij_1 * g_ij_1 + (omega_i + omega_j) * g_sum_1 * g_sum_1;
        const double dPhi_dgij_over_Phi = g_sum_1 + g_ij_1;

        const double Egr_ij = E_ij * dPhi_drj_over_Phi;
        const double Ugr_ij = E_ij * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi);
        const double Wgr_ij = E_ij * (r_ij2 * (dPhi_drj_over_Phi * dPhi_dgij_over_Phi + d2Phi_dgij_drj_over_Phi) - dPhi_dgij_over_Phi);

        Egr_i.x += Egr_ij * r_ji.x;
        Egr_i.y += Egr_ij * r_ji.y;
        Egr_i.z += Egr_ij * r_ji.z;
        Ugr_i.x += Ugr_ij * r_ji.x;
        Ugr_i.y += Ugr_ij * r_ji.y;
        Ugr_i.z += Ugr_ij * r_ji.z;
        Wgr_i.x += Wgr_ij * r_ji.x;
        Wgr_i.y += Wgr_ij * r_ji.y;
        Wgr_i.z += Wgr_ij * r_ji.z;
    }

    // Factor of \pm 2 from Egr,Ugr,Wgr definition, 1.5 from phi definiton
    Egr[B_atom * 3 * ngrids + 0 * ngrids + i] = -(2 * 1.5) * Egr_i.x;
    Egr[B_atom * 3 * ngrids + 1 * ngrids + i] = -(2 * 1.5) * Egr_i.y;
    Egr[B_atom * 3 * ngrids + 2 * ngrids + i] = -(2 * 1.5) * Egr_i.z;
    Ugr[B_atom * 3 * ngrids + 0 * ngrids + i] =  (2 * 1.5) * Ugr_i.x;
    Ugr[B_atom * 3 * ngrids + 1 * ngrids + i] =  (2 * 1.5) * Ugr_i.y;
    Ugr[B_atom * 3 * ngrids + 2 * ngrids + i] =  (2 * 1.5) * Ugr_i.z;
    Wgr[B_atom * 3 * ngrids + 0 * ngrids + i] =  (2 * 1.5) * Wgr_i.x;
    Wgr[B_atom * 3 * ngrids + 1 * ngrids + i] =  (2 * 1.5) * Wgr_i.y;
    Wgr[B_atom * 3 * ngrids + 2 * ngrids + i] =  (2 * 1.5) * Wgr_i.z;
}

template <int n_derivative_per_thread>
__global__
static void vv10_hess_eval_EUW_with_weight1_kernel(double* __restrict__ Ew, double* __restrict__ Uw, double* __restrict__ Ww,
                                                   const double* __restrict__ grid_coord, const double* __restrict__ grid_weight1,
                                                   const double* __restrict__ rho, const double* __restrict__ omega, const double* __restrict__ kappa,
                                                   const int ngrids, const int nderivative)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_derivative_start = (blockIdx.y * blockDim.y + threadIdx.y) * n_derivative_per_thread;
    if (i >= ngrids || i_derivative_start >= nderivative)
        return;

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double Ew_i[n_derivative_per_thread] {0};
    double Uw_i[n_derivative_per_thread] {0};
    double Ww_i[n_derivative_per_thread] {0};

    for (int j = 0; j < ngrids; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_j = rho[j];

        const double r_ij2 = (r_i.x - r_j.x) * (r_i.x - r_j.x) + (r_i.y - r_j.y) * (r_i.y - r_j.y) + (r_i.z - r_j.z) * (r_i.z - r_j.z);
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double E_ij = rho_j * Phi_ij;
        const double U_ij = E_ij * (g_sum + g_ij) * g_ji * g_ij_ji_sum_1;
        const double W_ij = U_ij * r_ij2;

        #pragma unroll
        for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
            if (i_derivative + i_derivative_start >= nderivative) continue;
            const double weight_j = grid_weight1[(i_derivative + i_derivative_start) * ngrids + j];
            Ew_i[i_derivative] += weight_j * E_ij;
            Uw_i[i_derivative] += weight_j * U_ij;
            Ww_i[i_derivative] += weight_j * W_ij;
        }
    }

    #pragma unroll
    for (int i_derivative = 0; i_derivative < n_derivative_per_thread; i_derivative++) {
        if (i_derivative + i_derivative_start >= nderivative) continue;
        Ew[(i_derivative + i_derivative_start) * ngrids + i] =  1.5 * Ew_i[i_derivative];
        Uw[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Uw_i[i_derivative];
        Ww[(i_derivative + i_derivative_start) * ngrids + i] = -1.5 * Ww_i[i_derivative];
    }
}

__global__
static void vv10_hess_eval_D_B_in_double_grid_response_offdiagonal_kernel(double* __restrict__ D_B,
                                                                          const double* __restrict__ grid_coord, const double* __restrict__ rho_weight,
                                                                          const double* __restrict__ omega, const double* __restrict__ kappa,
                                                                          const int* __restrict__ grid_associated_atom, const int* __restrict__ grid_offsets_of_atom,
                                                                          const int ngrids, const int natoms)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int B_atom = blockIdx.y;
    if (i >= ngrids || B_atom >= natoms)
        return;
    const int i_associated_atom = grid_associated_atom[i];
    if (i_associated_atom < 0 || i_associated_atom == B_atom) {
        D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = 0;
        D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = 0;
        return;
    }

    const double omega_i = omega[i];
    const double kappa_i = kappa[i];
    const double3 r_i = { grid_coord[i * 3 + 0], grid_coord[i * 3 + 1], grid_coord[i * 3 + 2] };

    double D_B_i[9] { 0,0,0, 0,0,0, 0,0,0, };

    const int j_B_atom_start = grid_offsets_of_atom[B_atom];
    const int j_B_atom_end = grid_offsets_of_atom[B_atom + 1];

    for (int j = j_B_atom_start; j < j_B_atom_end; j++) {
        const double omega_j = omega[j];
        const double kappa_j = kappa[j];
        const double3 r_j = { grid_coord[j * 3 + 0], grid_coord[j * 3 + 1], grid_coord[j * 3 + 2] };
        const double rho_weight_j = rho_weight[j];


        const double3 r_ji = { r_j.x - r_i.x, r_j.y - r_i.y, r_j.z - r_i.z };
        const double r_ij2 = r_ji.x * r_ji.x + r_ji.y * r_ji.y + r_ji.z * r_ji.z;
        const double g_ij = omega_i * r_ij2 + kappa_i;
        const double g_ji = omega_j * r_ij2 + kappa_j;
        const double g_sum = g_ij + g_ji;
        const double g_ij_ji_sum_1 = 1 / (g_ij * g_ji * g_sum);
        const double Phi_ij = -g_ij_ji_sum_1; // Prefactor of 1.5 is applied later

        const double g_ij_ji_1 = g_sum * g_ij_ji_sum_1;
        const double g_ij_1 = g_ji * g_ij_ji_1;
        const double g_ji_1 = g_ij * g_ij_ji_1;
        const double g_sum_1 = g_ij * g_ji * g_ij_ji_sum_1;

        const double omega_i_over_g_ij = omega_i * g_ij_1;
        const double omega_j_over_g_ji = omega_j * g_ji_1;
        const double omega_sum_over_g_sum = (omega_i + omega_j) * g_sum_1;
        const double omega_over_g_three_term_sum = omega_i_over_g_ij + omega_j_over_g_ji + omega_sum_over_g_sum;

        const double E_ij = rho_weight_j * Phi_ij;
        const double outer_product_prefactor = 2 * E_ij * (
            omega_over_g_three_term_sum * omega_over_g_three_term_sum
            + omega_i_over_g_ij * omega_i_over_g_ij
            + omega_j_over_g_ji * omega_j_over_g_ji
            + omega_sum_over_g_sum * omega_sum_over_g_sum
        );
        const double identity_prefactor = -E_ij * omega_over_g_three_term_sum;

        D_B_i[0] += outer_product_prefactor * r_ji.x * r_ji.x + identity_prefactor;
        D_B_i[1] += outer_product_prefactor * r_ji.x * r_ji.y;
        D_B_i[2] += outer_product_prefactor * r_ji.x * r_ji.z;
        D_B_i[3] += outer_product_prefactor * r_ji.y * r_ji.x;
        D_B_i[4] += outer_product_prefactor * r_ji.y * r_ji.y + identity_prefactor;
        D_B_i[5] += outer_product_prefactor * r_ji.y * r_ji.z;
        D_B_i[6] += outer_product_prefactor * r_ji.z * r_ji.x;
        D_B_i[7] += outer_product_prefactor * r_ji.z * r_ji.y;
        D_B_i[8] += outer_product_prefactor * r_ji.z * r_ji.z + identity_prefactor;
    }

    // Factor of 2 from D_B definition, 1.5 from phi definiton
    D_B[B_atom * 9 * ngrids + 0 * ngrids + i] = -(2 * 1.5) * D_B_i[0];
    D_B[B_atom * 9 * ngrids + 1 * ngrids + i] = -(2 * 1.5) * D_B_i[1];
    D_B[B_atom * 9 * ngrids + 2 * ngrids + i] = -(2 * 1.5) * D_B_i[2];
    D_B[B_atom * 9 * ngrids + 3 * ngrids + i] = -(2 * 1.5) * D_B_i[3];
    D_B[B_atom * 9 * ngrids + 4 * ngrids + i] = -(2 * 1.5) * D_B_i[4];
    D_B[B_atom * 9 * ngrids + 5 * ngrids + i] = -(2 * 1.5) * D_B_i[5];
    D_B[B_atom * 9 * ngrids + 6 * ngrids + i] = -(2 * 1.5) * D_B_i[6];
    D_B[B_atom * 9 * ngrids + 7 * ngrids + i] = -(2 * 1.5) * D_B_i[7];
    D_B[B_atom * 9 * ngrids + 8 * ngrids + i] = -(2 * 1.5) * D_B_i[8];
}

// Issue #10 candidate C1: runtime kernel selection for the VV10 UWE
// evaluation. The reference 'warp1' kernel (one thread per i, one
// 128-thread block per 128 i, j chunk staged once and shared by the
// block's 128 i points) is the default and remains bit-for-bit the
// historical path. 'warp4' routes to the multi-warp variant (one block
// per i, four warps cooperating on each i's j-loop; 128x global read
// amplification; kept from round 1, opt-in). 'tile8'/'tile16'/'tile32'
// route to the multi-i tile kernel with I_TILE = 8/16/32 i points per
// block (global read amplification 16x/8x/4x; per-pair fp64 arithmetic
// identical to the reference, summation-order-only difference, see the
// kernel comment). 'fp32' routes to the fp32-safe mixed-precision kernel
// (LC-80: fp32 pair arithmetic with separate reciprocals; measured
// dE <= 8.6e-10 Ha on wb97mv benzene/methanol -- see that kernel's
// comment). Selection is a process-lifetime decision read from
// GPU4PYSCF_VV10_UWE_KERNEL once; anything unrecognized (including
// unset) keeps the reference kernel, and an unknown non-empty value is
// reported and ignored rather than guessed. Returns 0 = warp1,
// 1 = warp4, 2 = fp32-safe, or the I_TILE of the selected multi-i tile.
//
// MR1 (mainline repair 2026-10-21): g_uwe_kernel_override, set through the
// exported VV10_uwe_kernel_set_override below, is honored ahead of the
// env-cached selection.  gpu4pyscf.dft.numint.strict_fp64_reference_ctx sets
// it to 0 (the reference warp1 kernel) while a stability_verify strict
// fp64-reference sibling SCF runs, so arming
// GPU4PYSCF_VV10_UWE_KERNEL=fp32 in the candidate process cannot leak the
// fp32 kernel into the reference trajectory.  -1 = no override (env-cached
// behavior, the previous semantics).
static int g_uwe_kernel_override = -1;

static int vv10_uwe_kernel_variant()
{
    if (g_uwe_kernel_override >= 0) {
        return g_uwe_kernel_override;
    }
    static int cached = -1;
    if (cached < 0) {
        const char* env = getenv("GPU4PYSCF_VV10_UWE_KERNEL");
        if (env != nullptr && strcmp(env, "warp4") == 0) {
            cached = 1;
        } else if (env != nullptr && strcmp(env, "fp32") == 0) {
            cached = 2;
        } else if (env != nullptr && strcmp(env, "tile8") == 0) {
            cached = 8;
        } else if (env != nullptr && strcmp(env, "tile16") == 0) {
            cached = 16;
        } else if (env != nullptr && strcmp(env, "tile32") == 0) {
            cached = 32;
        } else {
            if (env != nullptr && strlen(env) > 0 && strcmp(env, "warp1") != 0) {
                fprintf(stderr, "vv10 UWE kernel selector: unknown value "
                        "'%s' (expected warp1|warp4|fp32|tile8|tile16|tile32); "
                        "using the reference warp1 kernel\n", env);
            }
            cached = 0;
        }
    }
    return cached;
}

extern "C" {
// MR1 (mainline repair 2026-10-21): strict fp64-reference kernel override.
// Sets the UWE kernel selection override (0 = warp1 reference kernel) and
// returns the previous override value (-1 = follow the env-cached
// selection).  See g_uwe_kernel_override above.
__host__
int VV10_uwe_kernel_set_override(int variant)
{
    int prev = g_uwe_kernel_override;
    g_uwe_kernel_override = variant;
    return prev;
}

__host__
int VXC_vv10nlc_fock_eval_UWE(const cudaStream_t stream,
                              double* U, double* W, double* E,
                              const double* grid_coord, const double* rho_weight,
                              const double* omega, const double* kappa,
                              const int ngrids)
{
    const int variant = vv10_uwe_kernel_variant();
    const dim3 threads(NG_PER_BLOCK);
    if (variant == 1) {
        // warp4: one block per grid point; the j-loop inside is cooperative
        const dim3 blocks(ngrids);
        vv10_fock_eval_UWE_multiwarp_kernel<<<blocks, threads, 0, stream>>>(
            U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
    } else if (variant == 2) {
        // fp32-safe mixed-precision kernel (LC-80): same warp1 structure,
        // fp32 pair arithmetic; see the kernel comment for the measured
        // numerical-safety structure and gate margins
        const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
        vv10_fock_eval_UWE_fp32safe_kernel<<<blocks, threads, 0, stream>>>(
            U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
    } else if (variant >= 4) {
        // multi-i tile: one block per I_TILE grid points; the per-i
        // reduction lives inside one warp (see kernel comment)
        const dim3 blocks((ngrids + variant - 1) / variant);
        if (variant == 8) {
            vv10_fock_eval_UWE_multii_kernel<8><<<blocks, threads, 0, stream>>>(
                U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
        } else if (variant == 16) {
            vv10_fock_eval_UWE_multii_kernel<16><<<blocks, threads, 0, stream>>>(
                U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
        } else {
            vv10_fock_eval_UWE_multii_kernel<32><<<blocks, threads, 0, stream>>>(
                U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
        }
    } else {
        const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
        vv10_fock_eval_UWE_kernel<<<blocks, threads, 0, stream>>>(
            U, W, E, grid_coord, rho_weight, omega, kappa, ngrids);
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_UWE: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_fock_eval_omega_derivative(const cudaStream_t stream,
                                           double* omega, double* domega_drho, double* domega_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_fock_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(omega, domega_drho, domega_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 fock eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_grad_eval_E_grid_response_offdiagonal(const cudaStream_t stream,
                                                      double* Egr,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                      const int natm, const int i_grid_begin, const int ngrids)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_grad_eval_E_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        Egr, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, natm, i_grid_begin, ngrids
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 grad eval_E_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_UWABCE(const cudaStream_t stream,
                                 double* U, double* W, double* A, double* B, double* C, double* E,
                                 const double* grid_coord, const double* rho_weight,
                                 const double* omega, const double* kappa,
                                 const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_UWABCE_kernel<<<blocks, threads, 0, stream>>>(U, W, A, B, C, E,
                                                                 grid_coord, rho_weight, omega, kappa, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_UWABCE: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_omega_derivative(const cudaStream_t stream,
                                           double* omega, double* domega_drho, double* domega_dgamma,
                                           double* d2omega_drho2, double* d2omega_dgamma2, double* d2omega_drho_dgamma,
                                           const double* rho, const double* gamma, const double C_factor,
                                           const int ngrids)
{
    const dim3 threads(NG_PER_BLOCK);
    const dim3 blocks((ngrids+NG_PER_BLOCK-1)/NG_PER_BLOCK);
    vv10_hess_eval_omega_derivative_kernel<<<blocks, threads, 0, stream>>>(omega, domega_drho, domega_dgamma,
                                                                           d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma,
                                                                           rho, gamma, C_factor, ngrids);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_omega_derivative: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_f_t(const cudaStream_t stream,
                              double* f_rho_t, double* f_gamma_t,
                              const double* grid_coord, const double* grid_weight,
                              const double* rho, const double* omega, const double* kappa,
                              const double* U, const double* W, const double* A, const double* B, const double* C,
                              const double* domega_drho, const double* domega_dgamma, const double* dkappa_drho,
                              const double* d2omega_drho2, const double* d2omega_dgamma2, const double* d2omega_drho_dgamma, const double* d2kappa_drho2,
                              const double* rho_t, const double* gamma_t,
                              const int ngrids, const int ntrial)
{
    {
        constexpr int n_trial_per_thread = 6; // Notice: ntrial is likely a multiple of 3
        const dim3 threads(NG_PER_BLOCK, 1);
        const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                        (ntrial + n_trial_per_thread - 1) / n_trial_per_thread);
        vv10_hess_eval_f_t_offdiagonal_kernel<n_trial_per_thread> <<<blocks, threads, 0, stream>>> (
            f_rho_t, f_gamma_t,
            grid_coord, grid_weight, rho, omega, kappa,
            domega_drho, domega_dgamma, dkappa_drho,
            rho_t, gamma_t, ngrids, ntrial
        );
    }
    {
        const dim3 threads(NG_PER_BLOCK);
        const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK);
        vv10_hess_eval_f_t_diagonal_kernel<<<blocks, threads, 0, stream>>> (
            f_rho_t, f_gamma_t,
            rho,
            U, W, A, B, C,
            domega_drho, domega_dgamma, dkappa_drho,
            d2omega_drho2, d2omega_dgamma2, d2omega_drho_dgamma, d2kappa_drho2,
            rho_t, gamma_t, ngrids, ntrial
        );
    }
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_f_t: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_grid_response_offdiagonal(const cudaStream_t stream,
                                                        double* Egr, double* Ugr, double* Wgr,
                                                        const double* grid_coord, const double* rho_weight,
                                                        const double* omega, const double* kappa,
                                                        const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                        const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_hess_eval_EUW_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        Egr, Ugr, Wgr, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, ngrids, natm);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_grid_response: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_EUW_with_weight1(const cudaStream_t stream,
                                           double* Ew, double* Uw, double* Ww,
                                           const double* grid_coord, const double* grid_weight1,
                                           const double* rho, const double* omega, const double* kappa,
                                           const int ngrids, const int nderivative)
{
    constexpr int n_derivative_per_thread = 6; // Notice: ntrial is always a multiple of 3
    const dim3 threads(NG_PER_BLOCK, 1);
    const dim3 blocks((ngrids + NG_PER_BLOCK - 1) / NG_PER_BLOCK,
                      (nderivative + n_derivative_per_thread - 1) / n_derivative_per_thread);
    vv10_hess_eval_EUW_with_weight1_kernel<n_derivative_per_thread> <<<blocks, threads, 0, stream>>> (
        Ew, Uw, Ww,
        grid_coord, grid_weight1, rho, omega, kappa,
        ngrids, nderivative
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_EUW_with_weight1: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

__host__
int VXC_vv10nlc_hess_eval_D_B_in_double_grid_response_offdiagonal(const cudaStream_t stream,
                                                      double* D_B,
                                                      const double* grid_coord, const double* rho_weight,
                                                      const double* omega, const double* kappa,
                                                      const int* grid_associated_atom, const int* grid_offsets_of_atom,
                                                      const int ngrids, const int natm)
{
    constexpr int n_grids_per_block = 128;
    const dim3 threads(n_grids_per_block, 1);
    const dim3 blocks((ngrids + n_grids_per_block - 1) / n_grids_per_block, natm);
    vv10_hess_eval_D_B_in_double_grid_response_offdiagonal_kernel<<<blocks, threads, 0, stream>>>(
        D_B, grid_coord, rho_weight, omega, kappa, grid_associated_atom, grid_offsets_of_atom, ngrids, natm
    );
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error of vv10 hess eval_E_grgr_AB: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
}

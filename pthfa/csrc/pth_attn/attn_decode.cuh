// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// attn_decode.cuh: the decode kernel, a swap-AB, d-split variant of the
// attention kernel for row tiles of at most 32 rows (q * G <= 32: decode and
// short verify). ADR 0012 Amendment 2 has the measurements behind it; this
// comment has the design.
//
// Why a second kernel. attn_kernel.cuh puts the 16 query rows on the MMA's M
// dimension (m16n8k16: M = 16 rows, N = 8 tokens). A decode row tile has
// q * G rows (6 at q = 1, 24 at q = 4 with G = 6), so 62% (q = 1) and 25%
// (q = 4) of the tensor-core work is padding, and at WM = 2 two warps
// dequantise the same K and V bytes. ncu on the shipped kernel (docs/adr
// 0012 Amendment 2): the HMMA pipe is the busiest unit, and under the 220 W
// cap the SM clock drops to ~1.1 GHz, so the MMA count, not the memory system,
// sets the time at q = 4.
//
// Swap-AB. Here the MMA computes S^T = K . Q^T and O^T = V^T . P'':
//   * M = 16 tokens (QK) or 16 d values (PV): always full;
//   * N = 8 rows of the row tile: NT = ceil(rows / 8) n-tiles, so q = 1 uses
//     one n-tile (8 rows for 6) and q = 4 uses three (24 rows for 24);
//   * K = d (QK) or 16 tokens (PV).
// The operand fragments are the shipped kernel's, with the roles exchanged:
//   * A fragment of K (rows g, g + 8 = tokens, k = d pairs) holds the same
//     bytes as the old B fragment (col g = token, k = d pairs): the same
//     ldmatrix.x4 on the int8 stage, the same cvt4 and the same K
//     permutation pi (ADR 0008);
//   * B fragment of Q^T (k = d pairs, col g = row of the tile) holds the same
//     elements as the old A fragment (row g, k = d pairs): one ldmatrix of
//     8 rows per matrix;
//   * A fragment of V^T (rows g, g + 8 = d, k = token pairs) is the old
//     B fragment of two adjacent n-tiles: the same LDS.32 + PRMT pairing +
//     cvt4 of ADR 0009, two d-permuted columns per register pair;
//   * B fragment of P'' (k = token pairs, col g = row) is read back from a
//     small shared buffer P[row][token] with ldmatrix; the buffer is how the
//     four warps share P'' (below).
//
// d-split. The O^T accumulators of all of D are D/16 x NT x 4 registers per
// warp (192 at D = 256, NT = 3): too many. Instead warp w owns d in
// [D w / 4, D (w + 1) / 4) for ALL 64 tokens of a tile (4 m-tiles x NT x 4 =
// 16 NT registers) and every V byte is dequantised exactly once, at any NT.
// QK is split by tokens (warp w: tokens 16 w .. 16 w + 15), each K byte again
// dequantised once. The price is two CTA-wide barriers per KV tile (among the
// four compute warps; the producers never join): one to exchange the row max
// and one to exchange P''.
//
// One tile step (all four warps, 64 tokens):
//   1  S^T = K_w Q^T            16 tokens x 8 NT rows, fp32;
//   2  s = fp32(fp32(S ks) c)   per token ks, per row c; causal mask;
//   3  per-row max of the warp's 16 tokens (3 xor shuffles over lanes g),
//      written to shared memory; barrier A; the tile max is the max of the
//      four (each warp computes the same value);
//   4  online softmax as softmax.cuh: alpha, c_new from the 64 tokens' vs,
//      O^T rescale, p = 2^(s - m), P'' = fp16(p vs 2^-c_new); l is a thread
//      partial over the warp's own tokens (summed over lanes and warps once,
//      at the end);
//   5  P'' (transposed with movmatrix) to shared memory; barrier B;
//   6  O^T += V^T_w P'' for the 64 tokens of the tile.
// The arithmetic is emulate.py's with sub_tile = 64 and one warp per tile
// (Tiling(64, 64, 1, S, bq)): one max, one V-scale exponent and one
// normalisation per 64-token tile. Its l is summed in a different order (a
// 32-term tree at the end instead of a quad sum); the bound counts l's
// roundings per token, which this does not change (EB sec. 3.4, bound.py
// n_l).

#pragma once

#include "attn_kernel.cuh"

namespace pth_attn {

// Rows per n-tile (the MMA's N) and the largest row tile of this kernel.
constexpr int kDecRowsPerNt = kMmaN;
constexpr int kDecMaxNt = 4;
constexpr int kDecMaxRows = kDecMaxNt * kDecRowsPerNt;  // 32
// Tokens per warp in the QK step and per softmax step (a whole load tile).
constexpr int kDecQkTokens = kLoadTileTokens / kWarps;  // 16
// P'' buffer row: 64 fp16 tokens + 16 B so that the 8 rows of an ldmatrix
// matrix start in 8 different 16-B bank groups (pitch 144 = 9 x 16 B).
constexpr int kDecPPitch = kLoadTileTokens * kF16Bytes + kSmemChunkBytes;
// m-tiles (16 d each) per warp: D / 4 d per warp.
PTH_HD constexpr int dec_mtiles(int d) { return d / kWarps / kMmaM; }

// Shared memory: Q (8 NT rows), two stages in the loader's layout, the P''
// buffer, the row-max and row-sum exchange ([row][warp] f32) and the per-row
// score factors.
template <int D, int NT, bool kCell>
struct DecSmem {
  using R = std::conditional_t<kCell, CellStageReader<D>, SepStageReader<D>>;
  static constexpr int kRows = kDecRowsPerNt * NT;
  static constexpr int kQ = 0;
  static constexpr int kQBytes = kRows * kF16Bytes * D;
  static constexpr int kStageBytes = R::kBytes;
  PTH_HD static constexpr int stage_offset(int s) {
    return kQBytes + s * kStageBytes;
  }
  static constexpr int kP = kQBytes + kStages * kStageBytes;
  static constexpr int kPBytes = kRows * kDecPPitch;
  static constexpr int kXchgMax = kP + kPBytes;
  static constexpr int kXchgBytes = kRows * kWarps * kAccElemBytes;
  static constexpr int kXchgSum = kXchgMax + kXchgBytes;
  static constexpr int kRowFactor = kXchgSum + kXchgBytes;
  static constexpr int kBytes = kRowFactor + kRows * kScaleBytes;
  static_assert(kQBytes % kSmemLineBytes == 0 &&
                    kStageBytes % kSmemLineBytes == 0 &&
                    kP % kSmemLineBytes == 0 && kXchgMax % 16 == 0,
                "regions must start on 128-B lines (16 B for the exchange)");
  static_assert(kBytes + kSmemReservedPerBlock <= kSmemPerSm &&
                    kBytes <= kSmemPerBlockLimit,
                "one CTA per SM must fit");
};

#if defined(__CUDACC__)

// movmatrix.sync.aligned.m8n8.trans.b16 (PTX ISA 9.7.16.5.16): the 8 x 8
// matrix of 16-bit elements held in the ldmatrix fragment layout (lane l:
// row l / 4, columns 2 (l % 4), 2 (l % 4) + 1) comes back transposed in the
// same layout.
PTH_DEVICE uint32_t movmatrix_trans(uint32_t a) {
  uint32_t d;
  asm("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(d) : "r"(a));
  return d;
}

// S^T[n] (n = 0..NT-1) = K[tokens tok0 .. tok0 + 15] . Q~^T[rows 8n .. 8n + 7]
// as fp32 accumulators in the C layout of m16n8k16: c0, c1 = (token g, rows
// 8n + 2q, +1), c2, c3 = (token g + 8, same rows). k = d under pi (ADR 0008).
template <int D, int NT, class R>
__device__ __forceinline__ void dec_qk(float (&s)[NT][4], uint32_t q_smem,
                                       uint32_t stage_smem, int tok0,
                                       int lane) {
  #pragma unroll
  for (int n = 0; n < NT; ++n) s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f;
  const int mi = lane >> 3, l8 = lane & 7;
  #pragma unroll
  for (int ks2 = 0; ks2 < D / (2 * kMmaK); ++ks2) {
    // Matrix mi of the K ldmatrix.x4: tokens tok0 + 8 (mi & 1) + 0..7 at the
    // 16-B chunk 2 ks2 + (mi >> 1), i.e. k-step e = mi >> 1 of this pair.
    uint32_t kb[4];
    ldmatrix_x4(kb, stage_smem + R::k(tok0 + 8 * (mi & 1) + l8,
                                      (2 * ks2 + (mi >> 1)) * kSmemChunkBytes));
    F16x2Pair f[4];
  #pragma unroll
    for (int m = 0; m < 4; ++m) f[m] = cvt4_s8_f16x2(kb[m]);
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
      // Matrix mi of the Q ldmatrix.x4: rows 8 n + 0..7 at the 8-element
      // chunk 4 ks2 + mi: (k-step e, half h) = (mi >> 1, mi & 1), which is
      // B register h of k-step e (b0 = k 0..7 of the step, b1 = k 8..15).
      uint32_t qb[4];
      ldmatrix_x4(qb, q_smem + q_swz(8 * n + l8, 8 * (4 * ks2 + mi), D));
  #pragma unroll
      for (int e = 0; e < 2; ++e) {
        // A = (row g, k 2q..) a0; (row g + 8, k 2q..) a1; (g, 2q + 8) a2;
        // (g + 8, 2q + 8) a3: tokens 0..7 are kb[2e], tokens 8..15 kb[2e + 1].
        const uint32_t a[4] = {f[2 * e].lo, f[2 * e + 1].lo, f[2 * e].hi,
                               f[2 * e + 1].hi};
        const uint32_t b[2] = {qb[2 * e], qb[2 * e + 1]};
        mma_m16n8k16_f16f32(s[n], a, b);
      }
    }
  }
}

// O^T[mt][n] += V^T[d of m-tile mt] . P''^T for the 64 tokens of a tile.
// Warp `warp` owns the d groups D / 32 / 4 * warp ..: m-tile mt = 2 a + h of
// its group a has rows g <-> d = 32 ga + 4 g + 2 h and g + 8 <-> + 1 (ADR
// 0009's d-permutation: B column g of n-tile n_sub is d = 32 ga + 4 g + n_sub,
// so an m-tile is the pair of n-tiles (2h, 2h + 1)). The accumulator is in the
// C layout: c0, c1 = (d row g, rows 8 n + 2q, +1), c2, c3 = (d row g + 8,
// same). p_smem is P'' as [row][token] fp16 with pitch kDecPPitch.
template <int D, int NT, class R>
__device__ __forceinline__ void dec_pv(float (&o)[dec_mtiles(D)][NT][4],
                                       uint32_t p_smem, const uint8_t* stage,
                                       int warp, int lane) {
  constexpr int kGroups = D / kVGroupD / kWarps;  // 32-d groups per warp
  static_assert(2 * kGroups == dec_mtiles(D), "two m-tiles per 32-d group");
  const int g = lane >> 2, q = lane & 3;
  const int mi = lane >> 3, l8 = lane & 7;
  #pragma unroll
  for (int j = 0; j < kLoadTileTokens / kMmaK; ++j) {
    // B fragments of P'': matrix mi = (n-tile np + (mi >> 1), half mi & 1) at
    // token chunk j: rows 8 n + 0..7, 16 B at byte 32 j + 16 h.
    uint32_t bp[NT][2];
  #pragma unroll
    for (int np = 0; np < NT; np += 2) {
      const int nm = np + (mi >> 1) < NT ? np + (mi >> 1) : NT - 1;
      uint32_t r[4];
      ldmatrix_x4(r,
                  p_smem + (8 * nm + l8) * kDecPPitch + 32 * j + 16 * (mi & 1));
      bp[np][0] = r[0];
      bp[np][1] = r[1];
      if (np + 1 < NT) {
        bp[np + 1][0] = r[2];
        bp[np + 1][1] = r[3];
      }
    }
  #pragma unroll
    for (int a = 0; a < kGroups; ++a) {
      uint32_t w[4];
  #pragma unroll
      for (int i = 0; i < 4; ++i) {  // tokens 2q, 2q+1, 2q+8, 2q+9 of the step
        const int tok = kMmaK * j + 2 * q + v_token_offset(i);
        w[i] = *reinterpret_cast<const uint32_t*>(
            stage + R::v(tok, kVGroupD * (kGroups * warp + a) + 4 * g));
      }
      uint32_t b[4][2];
      v_b_fragments(w, b);  // b[n_sub][r]: d = 32 ga + 4 g + n_sub
  #pragma unroll
      for (int h = 0; h < 2; ++h) {
        // A: a0 = (row g, tokens 2q..2q+1), a1 = (row g + 8, ..), a2, a3 the
        // tokens + 8; rows g <-> n_sub 2h, g + 8 <-> n_sub 2h + 1.
        const uint32_t am[4] = {b[2 * h][0], b[2 * h + 1][0], b[2 * h][1],
                                b[2 * h + 1][1]};
  #pragma unroll
        for (int n = 0; n < NT; ++n) {
          mma_m16n8k16_f16f32(o[2 * a + h][n], am, bp[n]);
        }
      }
    }
  }
}

// Stores 4 floats (4 contiguous d) at dst in `dtype`, round to nearest even
// for the 16-bit types (as store8).
__device__ __forceinline__ void store4(char* dst, int32_t dtype,
                                       const float (&v)[4]) {
  if (dtype == PTH_ATTN_F32) {
    *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
    return;
  }
  uint32_t w[2];
  #pragma unroll
  for (int i = 0; i < 2; ++i) {
    if (dtype == PTH_ATTN_BF16) {
      const __nv_bfloat162 b = __floats2bfloat162_rn(v[2 * i], v[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&b);
    } else {
      const __half2 h = __floats2half2_rn(v[2 * i], v[2 * i + 1]);
      w[i] = *reinterpret_cast<const uint32_t*>(&h);
    }
  }
  *reinterpret_cast<uint2*>(dst) = make_uint2(w[0], w[1]);
}

// Row max / row sum of a thread's row g of S^T over the 8 lanes that share
// q (the tokens g = 0..7): xor shuffles over the lane bits 2, 3, 4.
__device__ __forceinline__ float octet_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
}
__device__ __forceinline__ float octet_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 16);
}

template <int D, int NT, typename QT, bool kCell>
__device__ __forceinline__ void attn_dec_body(const AttnParams& ap) {
  using Sm = DecSmem<D, NT, kCell>;
  using R = typename Sm::R;
  constexpr int kRows = Sm::kRows;
  constexpr int kMt = dec_mtiles(D);
  constexpr int kGroups = D / kVGroupD / kWarps;
  extern __shared__ __align__(128) uint8_t smem[];
  const KernelParams& p = ap.k;
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
  const int32_t kvh = static_cast<int32_t>(blockIdx.y);
  const int32_t split = static_cast<int32_t>(blockIdx.z);

  RowTile rt{};
  if (!row_tile(p.query_start_loc, p.seq_lens, p.bq_div, p.bq, p.num_reqs,
                static_cast<int32_t>(blockIdx.x), rt)) {
    return;  // surplus q-tile of the padded grid
  }
  const int32_t kv_end = kv_range_end(rt);
  const int32_t min_limit = kv_min_limit(rt);
  int32_t tb = 0, te = 0;
  split_tile_range(split, p.num_splits, num_kv_tiles(kv_end, kLoadTileTokens),
                   tb, te);

  if (warp >= kWarps) {
    run_producer_warp<D, kCell, Sm>(ap, rt, kvh, tb, te, kv_end, smem,
                                    warp - kWarps, lane);
    return;
  }

  float* row_factor = reinterpret_cast<float*>(smem + Sm::kRowFactor);
  stage_q<D, kRows, QT>(p, rt, kvh, smem + Sm::kQ, row_factor, warp, lane);
  consumer_sync<kAttnProducerWarps<D>>();

  const uint32_t smem_base = smem_u32(smem);
  const uint32_t q_smem = smem_base + Sm::kQ;
  const uint32_t p_smem = smem_base + Sm::kP;
  float* xchg_max = reinterpret_cast<float*>(smem + Sm::kXchgMax);
  float* xchg_sum = reinterpret_cast<float*>(smem + Sm::kXchgSum);
  uint8_t* p_buf = smem + Sm::kP;

  const int g = frag_g(lane), q = frag_q(lane);
  // A lane's S^T columns: rows 8 n + 2q + j of the row tile (n < NT, j < 2).
  float row_scale[NT][2];
  int32_t limit[NT][2];
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r = 8 * n + 2 * q + j;
      row_scale[n][j] = row_factor[r];
      int32_t tok = 0, hh = 0;
      limit[n][j] = row_to_token_head(static_cast<uint32_t>(r), p.group,
                                      rt.num_tok, tok, hh)
                        ? kv_limit(rt.q_len, rt.seq_len, rt.tok_begin + tok)
                        : -1;
    }
  }

  float m[NT][2], l[NT][2];
  float o[kMt][NT][4];
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
    m[n][0] = m[n][1] = -INFINITY;
    l[n][0] = l[n][1] = 0.f;
  }
  #pragma unroll
  for (int mt = 0; mt < kMt; ++mt) {
  #pragma unroll
    for (int n = 0; n < NT; ++n)
      o[mt][n][0] = o[mt][n][1] = o[mt][n][2] = o[mt][n][3] = 0.f;
  }
  int c = 0;
  const int tok0 = kDecQkTokens * warp;

  #pragma unroll 1
  for (int32_t i = tb; i < te; ++i) {
    const int buf = (i - tb) & 1;
    named_bar_sync(kBarFull0 + buf, kAttnThreadsWs);  // tile i has landed
    const uint32_t stage = smem_base + Sm::stage_offset(buf);
    const uint8_t* stage_ptr = smem + Sm::stage_offset(buf);

    // V-scale maximum of the tile's 64 tokens (every warp computes it).
    float vmax = fmaxf(
        *reinterpret_cast<const float*>(stage_ptr + R::vs(lane)),
        *reinterpret_cast<const float*>(stage_ptr + R::vs(lane + kWarpSize)));
  #pragma unroll
    for (int off = kWarpSize / 2; off > 0; off /= 2) {
      vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffffu, vmax, off));
    }
    // This lane's token scales: tokens tok0 + g and tok0 + g + 8.
    const float ks_lo =
        *reinterpret_cast<const float*>(stage_ptr + R::ks(tok0 + g));
    const float ks_hi =
        *reinterpret_cast<const float*>(stage_ptr + R::ks(tok0 + g + 8));
    const float vs_lo =
        *reinterpret_cast<const float*>(stage_ptr + R::vs(tok0 + g));
    const float vs_hi =
        *reinterpret_cast<const float*>(stage_ptr + R::vs(tok0 + g + 8));

    float s[NT][4];
    dec_qk<D, NT, R>(s, q_smem, stage, tok0, lane);
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
      // s = fp32(fp32(S ks) c_row) as the shipped kernel (EB sec. 3.1).
      s[n][0] = (s[n][0] * ks_lo) * row_scale[n][0];
      s[n][1] = (s[n][1] * ks_lo) * row_scale[n][1];
      s[n][2] = (s[n][2] * ks_hi) * row_scale[n][0];
      s[n][3] = (s[n][3] * ks_hi) * row_scale[n][1];
    }
    const int32_t abs0 = i * kLoadTileTokens;
    if (subtile_needs_mask(abs0, kLoadTileTokens, min_limit)) {
      const int32_t pos = abs0 + tok0 + g;
  #pragma unroll
      for (int n = 0; n < NT; ++n) {
  #pragma unroll
        for (int j = 0; j < 2; ++j) {
          if (pos > limit[n][j]) s[n][j] = -INFINITY;
          if (pos + 8 > limit[n][j]) s[n][2 + j] = -INFINITY;
        }
      }
    }

    // Row max: this warp's 16 tokens, then the four warps' via shared memory.
    float mx[NT][2];
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
  #pragma unroll
      for (int j = 0; j < 2; ++j) {
        mx[n][j] = octet_max(fmaxf(s[n][j], s[n][2 + j]));
        if (g == 0) xchg_max[(8 * n + 2 * q + j) * kWarps + warp] = mx[n][j];
      }
    }
    consumer_sync<kAttnProducerWarps<D>>();  // barrier A
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
  #pragma unroll
      for (int j = 0; j < 2; ++j) {
        const float4 v = *reinterpret_cast<const float4*>(
            xchg_max + (8 * n + 2 * q + j) * kWarps);
        mx[n][j] = fmaxf(m[n][j], fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
      }
    }

    // The online softmax step (softmax.cuh softmax_step, one step per tile).
    const int c_new = vmax > 0.f ? ceil_log2_pos(vmax) : c;
    const float p_norm = pow2_clamped(-c_new);
    const float c_shift = pow2_clamped(c - c_new);
    float m_use[NT][2], alpha_o[NT][2];
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
  #pragma unroll
      for (int j = 0; j < 2; ++j) {
        m_use[n][j] = mx[n][j] == -INFINITY ? 0.f : mx[n][j];
        const float alpha =
            m[n][j] == -INFINITY ? 0.f : ex2_approx(m[n][j] - m_use[n][j]);
        alpha_o[n][j] = alpha * c_shift;
        l[n][j] *= alpha;
        m[n][j] = mx[n][j];
      }
    }
    c = c_new;
  #pragma unroll
    for (int mt = 0; mt < kMt; ++mt) {
  #pragma unroll
      for (int n = 0; n < NT; ++n) {
        o[mt][n][0] *= alpha_o[n][0];
        o[mt][n][1] *= alpha_o[n][1];
        o[mt][n][2] *= alpha_o[n][0];
        o[mt][n][3] *= alpha_o[n][1];
      }
    }
  #pragma unroll
    for (int n = 0; n < NT; ++n) {
      const float p00 = ex2_approx(s[n][0] - m_use[n][0]);
      const float p01 = ex2_approx(s[n][1] - m_use[n][1]);
      const float p10 = ex2_approx(s[n][2] - m_use[n][0]);
      const float p11 = ex2_approx(s[n][3] - m_use[n][1]);
      l[n][0] += p00 + p10;
      l[n][1] += p01 + p11;
      // P'' = fp16_rn(fp32(p vs) 2^-c_new); the C fragment holds (token g,
      // rows 2q, 2q + 1) as one register, and movmatrix turns each 8 x 8
      // block into (row g, tokens 2q, 2q + 1): the layout of P[row][token].
      const uint32_t lo = movmatrix_trans(
          pack_f16x2_rn((p00 * vs_lo) * p_norm, (p01 * vs_lo) * p_norm));
      const uint32_t hi = movmatrix_trans(
          pack_f16x2_rn((p10 * vs_hi) * p_norm, (p11 * vs_hi) * p_norm));
      uint8_t* row = p_buf + (8 * n + g) * kDecPPitch;
      *reinterpret_cast<uint32_t*>(row + kF16Bytes * (tok0 + 2 * q)) = lo;
      *reinterpret_cast<uint32_t*>(row + kF16Bytes * (tok0 + 8 + 2 * q)) = hi;
    }
    consumer_sync<kAttnProducerWarps<D>>();  // barrier B

    dec_pv<D, NT, R>(o, p_smem, stage_ptr, warp, lane);
    // Release the stage for tile i + 2 (the producers wait for exactly these
    // arrivals, so every arrival is matched).
    if (i + kStages < te) named_bar_arrive(kBarEmpty0 + buf, kAttnThreadsWs);
  }

  // ---- row sums: lanes, then warps (fixed order, the same in every warp) ----
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int j = 0; j < 2; ++j) {
      l[n][j] = octet_sum(l[n][j]);
      if (g == 0) xchg_sum[(8 * n + 2 * q + j) * kWarps + warp] = l[n][j];
    }
  }
  consumer_sync<kAttnProducerWarps<D>>();
  const float c_unit = pow2_clamped(c);
  #pragma unroll
  for (int n = 0; n < NT; ++n) {
  #pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int r = 8 * n + 2 * q + j;
      const float4 v = *reinterpret_cast<const float4*>(xchg_sum + r * kWarps);
      const float l_tot = ((v.x + v.y) + v.z) + v.w;
      int32_t tok = 0, hh = 0;
      if (!row_to_token_head(static_cast<uint32_t>(r), p.group, rt.num_tok, tok,
                             hh)) {
        continue;  // padding row: never stored
      }
      const int64_t t_abs =
          static_cast<int64_t>(rt.q_start) + rt.tok_begin + tok;
      const int32_t head = kvh * p.group_size + hh;
      const bool live = l_tot > 0.f;
      const float inv_l = live ? rcp_approx(l_tot) : 0.f;
      char* dst_row;
      int32_t dtype;
      int64_t elem;
      if (p.num_splits == 1) {
        dst_row = static_cast<char*>(p.out) + out_offset(p, t_abs, head);
        dtype = ap.out_dtype;
        elem = dtype == PTH_ATTN_F32 ? kAccElemBytes : kF16Bytes;
      } else {
        dst_row =
            static_cast<char*>(p.ws_o) + ws_o_offset(p, split, t_abs, head);
        dtype = PTH_ATTN_F32;
        elem = kAccElemBytes;
      }
  #pragma unroll
      for (int a = 0; a < kGroups; ++a) {
        // Output d = 32 ga + 4 g + {0, 1, 2, 3}: (m-tile 2a, c0/c1), (2a,
        // c2/c3), (2a + 1, c0/c1), (2a + 1, c2/c3) of column j.
        const float v4[4] = {(o[2 * a][n][j] * inv_l) * c_unit,
                             (o[2 * a][n][2 + j] * inv_l) * c_unit,
                             (o[2 * a + 1][n][j] * inv_l) * c_unit,
                             (o[2 * a + 1][n][2 + j] * inv_l) * c_unit};
        store4(dst_row + (kVGroupD * (kGroups * warp + a) + 4 * g) * elem,
               dtype, v4);
      }
      if (p.num_splits > 1 && warp == 0 && g == 0) {
        const float lse = live ? m[n][j] + lg2_approx(l_tot) : -INFINITY;
        *reinterpret_cast<float*>(static_cast<char*>(p.ws_lse) +
                                  ws_lse_offset(p, split, t_abs, head)) = lse;
      }
    }
  }
}

// One CTA per SM: 4 compute warps + 4 producer warps, as attn_fwd_kernel at
// D = 256.
template <int D, int NT, typename QT, bool kCell>
__global__ void __launch_bounds__(kAttnThreads<D>, ctas_per_sm(D))
    attn_dec_kernel(const __grid_constant__ AttnParams ap) {
  attn_dec_body<D, NT, QT, kCell>(ap);
}

template <int D, int NT, typename QT, bool kCell>
cudaError_t launch_dec_one(const AttnLaunch& L, cudaStream_t stream) {
  return launch_with_smem<attn_dec_kernel<D, NT, QT, kCell>,
                          DecSmem<D, NT, kCell>::kBytes, kAttnThreads<D>>(
      L.grid, stream, L.ap);
}

template <int D, int NT>
cudaError_t launch_dec_nt(const AttnLaunch& L, cudaStream_t stream) {
  const bool bf16 = L.q_dtype == PTH_ATTN_BF16;
  if (L.cell) {
    return bf16 ? launch_dec_one<D, NT, __nv_bfloat16, true>(L, stream)
                : launch_dec_one<D, NT, __half, true>(L, stream);
  }
  return bf16 ? launch_dec_one<D, NT, __nv_bfloat16, false>(L, stream)
              : launch_dec_one<D, NT, __half, false>(L, stream);
}

// Dispatch on the number of n-tiles of the row tile (rows = bq * G).
template <int D>
cudaError_t launch_dec(const AttnLaunch& L, cudaStream_t stream) {
  switch ((L.rows + kDecRowsPerNt - 1) / kDecRowsPerNt) {
    case 1:
      return launch_dec_nt<D, 1>(L, stream);
    case 2:
      return launch_dec_nt<D, 2>(L, stream);
    case 3:
      return launch_dec_nt<D, 3>(L, stream);
    case 4:
      return launch_dec_nt<D, 4>(L, stream);
    default:
      return cudaErrorInvalidValue;
  }
}

#endif  // __CUDACC__

}  // namespace pth_attn

// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
//
// ptx.cuh: the instruction layer. Every inline-PTX statement of the kernel
// lives here, one thin wrapper per instruction, and nothing else does.
//
// Source for every wrapper: PTX ISA 9.4,
// https://docs.nvidia.com/cuda/parallel-thread-execution/ (section numbers
// below; they drift between ISA versions, the anchors do not). The numpy
// simulator (tests/sim/ptxsim/isa.py) models the same instructions per lane
// and proves the kernel's use of them (docs/walkthrough/05, 06).
//
// Two kinds of wrapper:
//
// 1. Device-only (cp.async, ldmatrix, mma): warp-collective or asynchronous
//    memory operations with no meaningful single-thread host behaviour. They
//    exist only under nvcc (__CUDACC__). Their CPU evidence is the simulator
//    plus the SASS gate (tests/cpu/test_primitives_golden.py); their hardware
//    evidence is the G1 one-warp test (build ladder step 12).
//
// 2. Host-emulated (prmt, sub.f16x2, cvt.rn.f16x2.f32 and the f32 bit
//    casts): pure functions of their inputs. On the device they are the
//    instruction; on the host (g++, or nvcc's host pass) they are a C++
//    emulation written from the ISA semantics. That lets the bit math built on
//    them (dequant.cuh) run under g++ and be compared bit for bit with the
//    simulator's golden tables (tests/cpp/test_primitives.cu). The emulation is
//    the "host shim": it is only as trustworthy as its agreement with the
//    simulator, which is what that test checks for every input that matters.
//    NaN payloads are not modelled (the kernel never feeds NaN to these).

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

// Shared with cache_geometry.cuh (T2): the identical text, guarded, so the
// headers can be included together. `inline` keeps non-template functions
// ODR-safe when several g++ translation units include them; __forceinline__
// keeps device code free of calls.
#ifndef PTH_HD
  #if defined(__CUDACC__)
    #define PTH_HD __host__ __device__ __forceinline__
  #else
    #define PTH_HD inline
  #endif
#endif

#ifndef PTH_DEVICE
  #if defined(__CUDACC__)
    #define PTH_DEVICE __device__ __forceinline__
  #endif
#endif

namespace pth_attn {

// ---------------------------------------------------------------------------
// Host emulation helpers (binary16 <-> double). Host only.
// ---------------------------------------------------------------------------
namespace host_emu {

// binary16: 1 sign, 5 exponent (bias 15), 10 fraction bits.
constexpr int kF16FracBits = 10;
constexpr int kF16ExpMask = 0x1F;
constexpr int kF16FracMask = 0x3FF;
constexpr int kF16Bias = 15;
constexpr int kF16EminNormal = -14;  // 2^-14, smallest normal
constexpr double kF16Max = 65504.0;  // (2 - 2^-10) * 2^15
constexpr uint16_t kF16SignBit = 0x8000;
constexpr uint16_t kF16Inf = 0x7C00;
constexpr uint16_t kF16QuietNaN = 0x7E00;  // payload not modelled

// Exact: every binary16 value is a double.
inline double f16_bits_to_double(uint16_t h) {
  const int exp = (h >> kF16FracBits) & kF16ExpMask;
  const int frac = h & kF16FracMask;
  double v;
  if (exp == 0) {
    // Subnormal: frac * 2^-24 (= 2^(1 - 15 - 10)).
    v = std::ldexp(static_cast<double>(frac), 1 - kF16Bias - kF16FracBits);
  } else if (exp == kF16ExpMask) {
    v = frac ? NAN : INFINITY;
  } else {
    // (2^10 + frac) * 2^(exp - 15 - 10): the hidden bit made explicit.
    v = std::ldexp(static_cast<double>((1 << kF16FracBits) + frac),
                   exp - kF16Bias - kF16FracBits);
  }
  return (h & kF16SignBit) ? -v : v;
}

// Round a double to the nearest binary16, ties to even (IEEE .rn). The caller
// passes the *exact* real result (see sub_f16x2 / pack_f16x2_rn for why the
// double is exact), so this is the one and only rounding, as in hardware.
inline uint16_t double_to_f16_bits_rn(double x) {
  if (std::isnan(x)) return kF16QuietNaN;
  const uint16_t sign = std::signbit(x) ? kF16SignBit : 0;
  const double ax = std::fabs(x);
  if (ax == 0.0) return sign;
  if (std::isinf(ax)) return sign | kF16Inf;
  int e2;
  std::frexp(ax, &e2);  // ax = m * 2^e2, m in [0.5, 1): floor(log2 ax) = e2-1
  // Grid spacing ("quantum") around ax: 2^(exp - 10) for normals, the fixed
  // subnormal spacing 2^-24 below 2^-14.
  const int exp = e2 - 1 < kF16EminNormal ? kF16EminNormal : e2 - 1;
  const int q = exp - kF16FracBits;
  // nearbyint rounds half to even in the default floating-point environment;
  // scaling by a power of two is exact, so this is the IEEE rounding.
  const double v = std::ldexp(std::nearbyint(std::ldexp(ax, -q)), q);
  if (v > kF16Max) return sign | kF16Inf;  // rounds to >= 65520
  if (v == 0.0) return sign;               // rounded below 2^-25: +-0
  int ev;
  std::frexp(v, &ev);
  if (ev - 1 < kF16EminNormal) {
    // Subnormal (or the rounding stayed below 2^-14): frac = v / 2^-24.
    return sign |
           static_cast<uint16_t>(std::ldexp(v, kF16Bias + kF16FracBits - 1));
  }
  const int frac = static_cast<int>(std::ldexp(v, -(ev - 1) + kF16FracBits)) -
                   (1 << kF16FracBits);
  return sign |
         static_cast<uint16_t>(((ev - 1 + kF16Bias) << kF16FracBits) | frac);
}

}  // namespace host_emu

// ---------------------------------------------------------------------------
// IEEE binary32 bit fields. Defined once here, the bottom of the include
// graph, for every header that takes f32 values apart (dequant.cuh,
// softmax.cuh, combine_kernel.cuh): two definitions of the same name in one
// translation unit do not compile.
// ---------------------------------------------------------------------------

constexpr int kF32MantBits = 23;               // stored fraction bits
constexpr int kF32ExpBias = 127;               // exponent field of 1.0
constexpr uint32_t kF32AbsMask = 0x7FFFFFFFu;  // all bits but the sign
constexpr int kF32ExpFieldMask = 0xFF;  // 8 exponent bits, after the shift

// ---------------------------------------------------------------------------
// Bit casts (host-emulated)
// ---------------------------------------------------------------------------

PTH_HD uint32_t f32_as_u32(float x) {
#if defined(__CUDA_ARCH__)
  return __float_as_uint(x);
#else
  uint32_t u;
  std::memcpy(&u, &x, sizeof(u));
  return u;
#endif
}

PTH_HD float u32_as_f32(uint32_t u) {
#if defined(__CUDA_ARCH__)
  return __uint_as_float(u);
#else
  float x;
  std::memcpy(&x, &u, sizeof(x));
  return x;
#endif
}

// ---------------------------------------------------------------------------
// prmt.b32 (PTX ISA 9.7.10.7), copy-only selectors (host-emulated)
// ---------------------------------------------------------------------------

// Bit 3 of a prmt selector nibble asks for sign replication instead of a
// byte copy. Every selector this project uses copies bytes, and for those
// prmt.b32 == CUDA's __byte_perm (which reads only 3 bits per nibble). The
// dequant constants are static_assert-ed with this.
constexpr uint32_t kPrmtSignModeBits = 0x8888;
constexpr uint32_t kPrmtSelectorMask = 0xFFFF;

PTH_HD constexpr bool prmt_selector_is_copy_only(uint32_t sel) {
  return (sel & ~kPrmtSelectorMask) == 0 && (sel & kPrmtSignModeBits) == 0;
}

// d.byte[i] = {b, a}.byte[sel.nibble[i]]: the 8 source bytes are a's 0..3 and
// b's 4..7. Compiles to one PRMT; __byte_perm is used on the device so the
// compiler knows the semantics (constant folding, scheduling).
PTH_HD uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
#if defined(__CUDA_ARCH__)
  return __byte_perm(a, b, sel);
#else
  const uint64_t src = (static_cast<uint64_t>(b) << 32) | a;
  uint32_t d = 0;
  for (int i = 0; i < 4; ++i) {
    const uint32_t byte_index = (sel >> (4 * i)) & 0x7;
    d |= static_cast<uint32_t>((src >> (8 * byte_index)) & 0xFF) << (8 * i);
  }
  return d;
#endif
}

// ---------------------------------------------------------------------------
// sub.rn.f16x2 (PTX ISA 9.7.4.2) (host-emulated)
// ---------------------------------------------------------------------------

// d.half[i] = RN(a.half[i] - b.half[i]), i = 0 (bits 0-15), 1 (bits 16-31).
// No .ftz: subnormals are kept, as the error bound assumes. Explicit .rn so
// the optimiser never fuses it with a neighbouring multiply.
PTH_HD uint32_t sub_f16x2(uint32_t a, uint32_t b) {
#if defined(__CUDA_ARCH__)
  uint32_t d;
  asm("sub.rn.f16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
#else
  // The exact difference of two binary16 values needs at most 40 significant
  // bits (2^15 down to 2^-24), so the double subtraction is exact and the
  // only rounding is double_to_f16_bits_rn.
  uint32_t d = 0;
  for (int i = 0; i < 2; ++i) {
    const auto x = static_cast<uint16_t>(a >> (16 * i));
    const auto y = static_cast<uint16_t>(b >> (16 * i));
    const double diff =
        host_emu::f16_bits_to_double(x) - host_emu::f16_bits_to_double(y);
    d |= static_cast<uint32_t>(host_emu::double_to_f16_bits_rn(diff))
         << (16 * i);
  }
  return d;
#endif
}

// ---------------------------------------------------------------------------
// cvt.rn.f16x2.f32 (PTX ISA 9.7.10.24) (host-emulated)
// ---------------------------------------------------------------------------

// Packs RN(lo) into bits 0-15 and RN(hi) into bits 16-31. Trap: the PTX
// instruction `cvt.rn.f16x2.f32 d, a, b` puts *a* in the upper half, so the
// operands are passed as (hi, lo).
PTH_HD uint32_t pack_f16x2_rn(float lo, float hi) {
#if defined(__CUDA_ARCH__)
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
#else
  // float -> double is exact, so this is a single rounding to binary16.
  return static_cast<uint32_t>(
             host_emu::double_to_f16_bits_rn(static_cast<double>(lo))) |
         (static_cast<uint32_t>(
              host_emu::double_to_f16_bits_rn(static_cast<double>(hi)))
          << 16);
#endif
}

#if defined(__CUDACC__)
// ---------------------------------------------------------------------------
// Device-only wrappers
// ---------------------------------------------------------------------------

// 32-bit shared-state-space address of a generic pointer into shared memory
// (the operand form of cp.async / ldmatrix / ld.shared).
PTH_DEVICE uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async (PTX ISA 9.7.10.28.3.1). Copies W bytes global -> shared without
// a register round trip; completes asynchronously (see commit/wait below).
// Constraints: W in {4, 8, 16}; dst and src aligned to W; sm_80+. Only W = 16
// may use .cg (bypass L1); 4 and 8 must use .ca. The "memory" clobber stops
// the compiler moving other shared-memory accesses across the issue.
template <int W>
PTH_DEVICE void cp_async(uint32_t dst_smem, const void* src) {
  static_assert(W == 4 || W == 8 || W == 16, "cp.async size is 4, 8 or 16");
  if constexpr (W == 16) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst_smem),
                 "l"(src)
                 : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst_smem),
                 "l"(src), "n"(W)
                 : "memory");
  }
}

// cp.async with src-size: copies src_bytes (<= W) and zero-fills the rest of
// the W destination bytes; src_bytes = 0 writes W zero bytes without reading
// src. For tokens past the end of the KV range, so stale shared memory (e.g.
// a NaN-looking scale) never reaches the math. Same constraints as cp_async.
template <int W>
PTH_DEVICE void cp_async_zfill(uint32_t dst_smem, const void* src,
                               uint32_t src_bytes) {
  static_assert(W == 4 || W == 8 || W == 16, "cp.async size is 4, 8 or 16");
  if constexpr (W == 16) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst_smem),
        "l"(src), "r"(src_bytes)
        : "memory");
  } else {
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2, %3;\n" ::"r"(dst_smem),
        "l"(src), "n"(W), "r"(src_bytes)
        : "memory");
  }
}

// cp.async.commit_group (PTX ISA 9.7.10.28.3.2): closes the group of all
// cp.async issued by this thread since the last commit.
PTH_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

// cp.async.wait_group N (PTX ISA 9.7.10.28.3.3): waits until at most N of
// this thread's committed groups are pending. It orders only the calling
// thread's copies; other threads' data needs a __syncthreads() afterwards.
template <int N>
PTH_DEVICE void cp_async_wait() {
  static_assert(N >= 0, "wait_group count is non-negative");
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ldmatrix.sync.aligned.m8n8.x4{.trans}.shared.b16 (PTX ISA 9.7.16.5.15).
// Lanes 8m..8m+7 supply the 8 row addresses of matrix m (16 contiguous bytes
// each, 16-B aligned); matrix m arrives in r[m]. Lane l receives row l/4,
// 16-bit columns 2(l%4), 2(l%4)+1 (Figure 107); with .trans the transposed
// element pair. Warp-collective: all 32 lanes must execute it, convergent.
// The data type is "16 bits": on int8 data each element is a byte pair, which
// is how int8 K is loaded (walkthrough 06 section 3).
template <bool Trans = false>
PTH_DEVICE void ldmatrix_x4(uint32_t (&r)[4], uint32_t smem_addr) {
  if constexpr (Trans) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, "
        "[%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr));
  } else {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(smem_addr));
  }
}

// mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 (PTX ISA 9.7.16.5.8):
// acc += A * B with A 16x16 f16 (a[0..3] = (a0,a1)..(a6,a7), Figure 82), B
// 16x8 f16 (b[0..1] = (b0,b1),(b2,b3), Figure 84), acc 16x8 f32 (c0..c3,
// Figure 86). Warp-collective. Not volatile: a pure function of its
// operands, so the compiler may schedule it freely.
PTH_DEVICE void mma_m16n8k16_f16f32(float (&acc)[4], const uint32_t (&a)[4],
                                    const uint32_t (&b)[2]) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
#endif  // __CUDACC__

}  // namespace pth_attn

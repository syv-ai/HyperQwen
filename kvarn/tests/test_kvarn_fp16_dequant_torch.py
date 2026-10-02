#!/usr/bin/env python3
"""Numerics gate for KVARN_FP16_DEQUANT (pure torch — runs without a venv
or a GPU; any torch build works, CPU is fine).

The fp16 variant of the fused KVarN decode kernels changes exactly one
thing: every dequant op (q codes, per-channel and per-row scales, p tiles)
is done in fp16 instead of fp32, and the two ``tl.dot`` calls run
fp16 x fp16 -> fp32-accumulate (the standard Triton-FA pattern).

This test quantizes random K ([D, G], 4-bit) and V ([G, D], 2-bit) tiles
with the SAME store math the Triton store kernels implement (sinkhorn
variance-normalize from kvarn/files/.../kvarn/sinkhorn.py, then asymmetric
per-row RTN, then scale absorption — mirroring kvarn_store.py), stores the
scales in fp16 (the cache format), and compares two reconstructions:

  R32  : the kernel's fp32 path today  — (q * s_col + zp) * s_row, fp32
  R16  : the new fp16 path            — same, each op rounded to fp16

What the metrics mean (and why NOT a plain per-element relative error):
after variance normalization the dequantized values are ~N(0, 1), and the
fp16 pipeline's error is an *absolute* ~2^-11 of the O(1) intermediate
(q*s + zp) — the same intermediate can round to a final value arbitrarily
close to zero (RTN's mid codes cancel zp), so a relative metric explodes
on ~1% of elements and measures the denominator, not the error. The
physically meaningful gates are:

  * abs p99 of |R16 - R32|  < 1e-2  (unit-variance tile)
  * abs p99  /  p99 of the 4-bit/2-bit RTN step  < 0.1
    (the fp16 wobble is an order of magnitude inside one quantization step)
  * rel-L2 of R16 vs R32 < 1e-2  (the global energy statement)
  * 256-wide score dot: rel err of the two pipelines' dot < 5e-2
  * mini-attention (softmax over 128 keys, fp32 softmax in both pipelines):
    output rel-L2 < 5e-2 — the number that maps onto the perplexity gate
    in docs/kvarn-decode-speedup.md.

Usage:  python kvarn/tests/test_kvarn_fp16_dequant_torch.py
Exits 0 on pass, 1 on fail.
"""

import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = os.path.normpath(os.path.join(HERE, "..", "files"))
sys.path.insert(0, FILES)

try:
    from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (  # noqa: F401
        variance_normalize,
    )
except ImportError:
    # No vllm package visible: load the pure-torch module straight from the
    # files tree (it has no vllm imports of its own).
    import importlib.util
    _sp = importlib.util.spec_from_file_location(
        "kvarn_sinkhorn_ref",
        os.path.join(
            FILES,
            "vllm/model_executor/layers/quantization/kvarn/sinkhorn.py"),
    )
    _mod = importlib.util.module_from_spec(_sp)
    _sp.loader.exec_module(_mod)
    variance_normalize = _mod.variance_normalize

D = 256
G = 128
SEED = 20260904
FAIL = []


def _check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name} {detail}")
    if not cond:
        FAIL.append(name)


def _p99(x):
    s, _ = x.flatten().abs().sort(descending=True)
    return s[max(0, int(0.01 * s.numel()) - 1)].item()


def _sinkhorn_k(x):
    """K tile [D, G]: returns (balanced, s_chan [D], s_tok [G]) matching
    kvarn_store.kvarn_store_tile_k's orientation (s_row = per-channel,
    s_col = per-token)."""
    balanced, s_col_s, s_row_s = variance_normalize(x, iterations=8)
    return balanced, s_row_s.squeeze(-1), s_col_s.squeeze(0)


def _sinkhorn_v(x):
    """V tile [G, D]: (balanced, s_chan [D], s_tok [G]) matching
    kvarn_store.kvarn_store_tile_v (s_col = per-channel, s_row = per token)."""
    balanced, s_col_s, s_row_s = variance_normalize(x, iterations=8)
    return balanced, s_col_s.squeeze(0), s_row_s.squeeze(-1)


def _rtn_rows(t, bits):
    """Asymmetric per-row RTN (dim=-1), same math as kvarn_store."""
    qmax = (1 << bits) - 1
    lo = t.amin(dim=-1, keepdim=True)
    hi = t.amax(dim=-1, keepdim=True)
    scale = ((hi - lo) / qmax).clamp_min(1e-10)
    zp = lo
    q = torch.clamp(torch.round((t - zp) / scale), 0, qmax).to(torch.int32)
    return q, scale.squeeze(-1), zp.squeeze(-1)


def _fp16_round(x):
    """One Triton fp16 op: the product/sum's exact (<=22-bit) result,
    rounded once to fp16 — exactly what the fp16 hardware does, emulated
    through fp32 bookkeeping."""
    return x.to(torch.float16).float()


def quantize_k(x):
    """Full K store math. Returns R32, R16 [D, G] and the RTN step [D]."""
    balanced, s_chan, s_tok = _sinkhorn_k(x)
    q, rtn_scale, rtn_zp = _rtn_rows(balanced, bits=4)
    s_col_K = (s_chan * rtn_scale).to(torch.float16)   # [D]
    zp_K = (s_chan * rtn_zp).to(torch.float16)        # [D]
    s_row_K = s_tok.to(torch.float16)                 # [G]
    sc, zpk, sr = s_col_K.float(), zp_K.float(), s_row_K.float()
    r32 = (q.float() * sc[:, None] + zpk[:, None]) * sr[None, :]
    # fp16 pipeline: every op's exact result rounded once to fp16.
    q16 = q.to(torch.float16).float()
    r16 = (_fp16_round(_fp16_round(q16 * sc[:, None]) + zpk[:, None])
           * sr[None, :])
    return r32, r16, rtn_scale


def quantize_v(x):
    """V tile [G, D], 2-bit V: (q * s_row_V + zp_V) * s_col_V."""
    balanced, s_chan, s_tok = _sinkhorn_v(x)
    q, rtn_scale, rtn_zp = _rtn_rows(balanced, bits=2)
    s_row_V = (s_tok * rtn_scale).to(torch.float16)   # [G]
    zp_V = (s_tok * rtn_zp).to(torch.float16)         # [G]
    s_col_V = s_chan.to(torch.float16)                # [D]
    sr, zpv, sc = s_row_V.float(), zp_V.float(), s_col_V.float()
    r32 = (q.float() * sr[:, None] + zpv[:, None]) * sc[None, :]
    q16 = q.to(torch.float16).float()
    r16 = (_fp16_round(_fp16_round(q16 * sr[:, None]) + zpv[:, None])
           * sc[None, :])
    return r32, r16, rtn_scale


def main():
    torch.manual_seed(SEED)
    print(f"KVarN fp16 dequant numerics test (D={D}, G={G}, "
          f"device={torch.device('cpu')})")

    # K: realistic rotated-K magnitudes (per-channel scale spread).
    xk = torch.randn(D, G, dtype=torch.float64)
    xk *= (0.5 + 1.5 * torch.rand(D, 1, dtype=torch.float64))
    xk = xk.to(torch.float32)
    r32k, r16k, stepk = quantize_k(xk)
    diffk = (r16k - r32k).abs()
    k_p99, k_max = _p99(diffk), diffk.max().item()
    k_step_p99 = _p99(stepk)
    k_l2 = (r16k - r32k).norm().item() / (r32k.norm().item() + 1e-9)
    print(f"\nK (4-bit, [D,G], unit-variance tile after normalization):")
    _check("K abs p99 < 1e-2", k_p99 < 1e-2, f"(abs p99={k_p99:.3e})")
    _check("K abs max < 5e-2", k_max < 5e-2, f"(abs max={k_max:.3e})")
    _check("K abs p99 < 0.1x the 4-bit RTN step",
            k_p99 < 0.1 * k_step_p99,
            f"({k_p99:.3e} vs step p99 {k_step_p99:.3e})")
    _check("K rel-L2 < 1e-2", k_l2 < 1e-2, f"({k_l2:.3e})")

    # V: 2-bit values.
    xv = torch.randn(G, D, dtype=torch.float64)
    xv *= (0.5 + 1.5 * torch.rand(1, D, dtype=torch.float64))
    xv = xv.to(torch.float32)
    r32v, r16v, stepv = quantize_v(xv)
    diffv = (r16v - r32v).abs()
    v_p99, v_max = _p99(diffv), diffv.max().item()
    v_step_p99 = _p99(stepv)
    v_l2 = (r16v - r32v).norm().item() / (r32v.norm().item() + 1e-9)
    print(f"\nV (2-bit, [G,D]):")
    _check("V abs p99 < 1e-2", v_p99 < 1e-2, f"(abs p99={v_p99:.3e})")
    _check("V abs max < 5e-2", v_max < 5e-2, f"(abs max={v_max:.3e})")
    _check("V abs p99 < 0.1x the 2-bit RTN step",
            v_p99 < 0.1 * v_step_p99,
            f"({v_p99:.3e} vs step p99 {v_step_p99:.3e})")
    _check("V rel-L2 < 1e-2", v_l2 < 1e-2, f"({v_l2:.3e})")

    # Score dots: q [D] against each of 16 token columns of K, fp32 pipeline
    # vs fp16-input emulation (the kernel's tl.dot(fp16,fp16,fp32-acc) is
    # exactly the float matmul of the fp16 inputs, since fp16 products are
    # exact in fp32). The dot is ~N(0, sqrt(D))-sized, so a plain relative
    # error is well-defined here.
    print(f"\nScore dot (D={D}, 16 sample keys):")
    worst = 0.0
    for i in range(16):
        qv_1 = torch.randn(D, dtype=torch.float32)
        k32 = r32k[:, i]
        k16 = r16k[:, i]
        s32 = float((qv_1 * k32).sum())
        s16 = float((qv_1 * k16).sum())
        rel = abs(s16 - s32) / (abs(s32) + abs(s16) + 1e-3)
        worst = max(worst, rel)
    _check("score dot rel err < 5e-2", worst < 5e-2, f"(worst={worst:.3e})")

    # End-to-end mini-attention: softmax over 128 keys (fp32 softmax in
    # both pipelines — the kernel's softmax is fp32 either way), output
    # [D] = V @ p. This is the number that maps onto the quality gate.
    qv_1 = torch.randn(D, dtype=torch.float32)
    scores32 = torch.stack([torch.dot(qv_1, r32k[:, t]) for t in range(G)])
    scores16 = torch.stack([torch.dot(qv_1, r16k[:, t]) for t in range(G)])
    p32 = torch.softmax(scores32 * 0.1, dim=0)
    p16 = torch.softmax(scores16 * 0.1, dim=0)
    o32 = r32v.T @ p32          # [D] = sum over tokens of p[t] * V[t, d]
    o16 = r16v.T @ p16
    arel = (o32 - o16).norm().item() / (o32.norm().item() + 1e-9)
    print(f"\nMini-attention output (G={G} keys, D={D}):")
    _check("output rel-L2 < 5e-2", arel < 5e-2, f"({arel:.3e})")

    print()
    if FAIL:
        print(f"RESULT: FAIL ({len(FAIL)}): {', '.join(FAIL)}")
        return 1
    print("RESULT: PASS — fp16 dequant is an order of magnitude inside the "
          "quantization step (see docs/kvarn-decode-speedup.md P0-1 gates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

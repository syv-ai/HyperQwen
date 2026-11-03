# SPDX-License-Identifier: Apache-2.0
"""Triton kernels of the LSA read on a paged KV cache.

One layer's read is five launches' worth of steps, every one static-shaped and sync-free so a
decode step is capturable in a CUDA graph:

    _score_kernel            bucket scores of each query over its visible buckets (tiles of queries
                             of one request share the centroid loads)
    _radix_*                 exact top-p of each row by radix select over the scores' ordered keys,
                             a row split over many programs; ties go to the lower bucket, as the
                             reference's stable sort
    _expand_*                the picked buckets' keys, best bucket first, up to top_k, beside the sinks
                             and the tail, written straight to physical KV slots
    _slot_attn_kernel        FlashAttention's online softmax over each query's slot list, the G query
                             heads of a group as the rows of one tile (every head of a query reads the
                             same list, as in the paper: `query_heads: all`); split-KV for decode
    _combine_kernel          the split-KV merge

The attention is unchanged: the dense layer's q, k, v and scale over the keys the selection names.
KV caches are vLLM's Triton-backend layout [blocks, block_size, kv_heads, head_size] as strided views,
bf16/fp16, or int8 with per-(token, head) fp32 scale views (`int8_per_token_head`), whose scales are
folded into the logit and the probability (exact: a scale is constant along the head dim).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

SCALE_NONE = 0
SCALE_PER_TOKEN_HEAD = 2


@triton.jit
def _score_kernel(
    QIDX, sq,                 # [N, 64] fp32 index queries (global token rows)
    QSL,                      # [R + 1] int32 query start locations
    VIS,                      # [N] int32 visible bucket count of each token
    BT, sbt,                  # [R, max_blocks] int32 block table
    CEN,                      # [rows, 64] bf16 centroids
    S, ss,                    # [t1 - t0, nb_max] fp32 scores out
    t0, t1, nb_max, block_size, max_positions,
    BQ: tl.constexpr, BB: tl.constexpr, DIM: tl.constexpr,
):
    r = tl.program_id(0)
    qt = tl.program_id(1)
    bb = tl.program_id(2)
    q_lo = tl.load(QSL + r)
    q_hi = tl.load(QSL + r + 1)
    tok = q_lo + qt * BQ + tl.arange(0, BQ)
    tok_ok = (tok < q_hi) & (tok >= t0) & (tok < t1)
    if tl.sum(tok_ok.to(tl.int32)) == 0:
        return
    vis = tl.load(VIS + tok, mask=tok_ok, other=0)
    b = bb * BB + tl.arange(0, BB)
    b_ok = (b < nb_max)
    out_ptr = S + (tok - t0)[:, None] * ss + b[None, :]
    if bb * BB >= tl.max(vis):
        return
    pos = b * 8
    load_ok = b_ok & (pos < max_positions) & (b < tl.max(vis))
    blk = tl.load(BT + r * sbt + pos // block_size, mask=load_ok, other=0).to(tl.int64)
    rows = (blk * block_size + pos % block_size) // 8
    d = tl.arange(0, DIM)
    cen = tl.load(CEN + rows[:, None] * DIM + d[None, :], mask=load_ok[:, None], other=0.0)
    q = tl.load(QIDX + tok[:, None] * sq + d[None, :], mask=tok_ok[:, None], other=0.0)
    s = tl.dot(q, tl.trans(cen.to(tl.float32)), input_precision="ieee")
    s = tl.where(b[None, :] < vis[:, None], s, float("-inf"))
    tl.store(out_ptr, s, mask=tok_ok[:, None] & b_ok[None, :])


def score_buckets(qidx: torch.Tensor, query_start_loc: torch.Tensor, visible: torch.Tensor,
                  block_table: torch.Tensor, centroids: torch.Tensor, t0: int, t1: int,
                  nb_max: int, block_size: int, num_reqs: int, max_query_len: int) -> torch.Tensor:
    out = torch.empty(t1 - t0, nb_max, dtype=torch.float32, device=qidx.device)
    bq = 16 if max_query_len <= 16 else 64
    bb = 64
    grid = (num_reqs, triton.cdiv(max_query_len, bq), triton.cdiv(nb_max, bb))
    _score_kernel[grid](
        qidx, qidx.stride(0), query_start_loc, visible, block_table, block_table.stride(0),
        centroids, out, out.stride(0), t0, t1, nb_max, block_size,
        block_table.shape[1] * block_size, BQ=bq, BB=bb, DIM=qidx.shape[1], num_warps=4)
    return out


@triton.jit
def _combine_kernel(PM, PL, PACC, O, so_t, so_h, HQ, NSPLIT: tl.constexpr, D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, NSPLIT)
    d = tl.arange(0, D)
    base = (t * HQ + h) * NSPLIT
    m = tl.load(PM + base + s)
    l = tl.load(PL + base + s)
    m_all = tl.max(m, 0)
    w = tl.where(m > float("-inf"), tl.exp2(m - m_all), 0.0)
    acc = tl.load(PACC + (base + s)[:, None] * D + d[None, :])
    out = tl.sum(acc * w[:, None], 0) / tl.sum(l * w, 0)
    tl.store(O + t * so_t + h * so_h + d, out.to(O.dtype.element_ty))




@triton.jit
def _slot_attn_kernel(
    Q, sq_t, sq_h,
    K, sk_slot, sk_h,
    V, sv_slot, sv_h,
    KS, VS, ss_slot, ss_h,
    O, so_t, so_h,
    PM, PL, PACC,
    SLOTS, sslot, LENS,
    scale,
    G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    SCALE_MODE: tl.constexpr, NSPLIT: tl.constexpr,
):
    """The read over a materialized list of physical slots (one indirection a key)."""
    t = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    total = tl.load(LENS + t)
    per = tl.cdiv(tl.cdiv(total, NSPLIT), BLOCK_N) * BLOCK_N
    lo = split * per
    hi = tl.minimum(lo + per, total)
    m_rows = tl.arange(0, BLOCK_M)
    d = tl.arange(0, D)
    heads = kvh * G + m_rows
    row_ok = m_rows < G
    q = tl.load(Q + t * sq_t + heads[:, None] * sq_h + d[None, :], mask=row_ok[:, None], other=0.0)
    qk_scale = scale * 1.4426950408889634
    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D), tl.float32)
    for start in range(lo, hi, BLOCK_N):
        i = start + tl.arange(0, BLOCK_N)
        ok = i < hi
        slot = tl.load(SLOTS + t * sslot + i, mask=ok, other=0).to(tl.int64)
        k = tl.load(K + slot[:, None] * sk_slot + kvh * sk_h + d[None, :], mask=ok[:, None], other=0.0)
        s = tl.dot(q, tl.trans(k.to(q.dtype))) * qk_scale
        if SCALE_MODE == 2:
            ks = tl.load(KS + slot * ss_slot + kvh * ss_h, mask=ok, other=0.0)
            s = s * ks[None, :]
        s = tl.where(ok[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        pr = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        v = tl.load(V + slot[:, None] * sv_slot + kvh * sv_h + d[None, :], mask=ok[:, None], other=0.0)
        if SCALE_MODE == 2:
            vs = tl.load(VS + slot * ss_slot + kvh * ss_h, mask=ok, other=0.0)
            pv = pr * vs[None, :]
        else:
            pv = pr
        acc = acc * alpha[:, None] + tl.dot(pv.to(q.dtype), v.to(q.dtype))
        m_i = m_new
    if NSPLIT == 1:
        out = acc / l_i[:, None]
        tl.store(O + t * so_t + heads[:, None] * so_h + d[None, :], out.to(O.dtype.element_ty),
                 mask=row_ok[:, None])
    else:
        hq = G * tl.num_programs(1)
        base = (t * hq + heads) * NSPLIT + split
        tl.store(PM + base, m_i, mask=row_ok)
        tl.store(PL + base, l_i, mask=row_ok)
        tl.store(PACC + base[:, None] * D + d[None, :], acc, mask=row_ok[:, None])


def slot_attention(query, key_cache, value_cache, k_scales, v_scales, output, slots, lens, scale,
                   num_splits=1, block_n=32, num_warps=4, num_stages=2):
    n, hq, dim = query.shape
    hkv = key_cache.shape[2]
    group = hq // hkv
    kflat = torch.as_strided(key_cache, (key_cache.shape[0] * key_cache.shape[1], hkv, key_cache.shape[3]),
                             (key_cache.stride(1), key_cache.stride(2), key_cache.stride(3)))
    vflat = torch.as_strided(value_cache, (value_cache.shape[0] * value_cache.shape[1], hkv, value_cache.shape[3]),
                             (value_cache.stride(1), value_cache.stride(2), value_cache.stride(3)))
    assert key_cache.stride(0) == key_cache.shape[1] * key_cache.stride(1)
    mode = SCALE_NONE if k_scales is None else SCALE_PER_TOKEN_HEAD
    if k_scales is None:
        ksf = vsf = query
        ss = (0, 0)
    else:
        assert k_scales.stride(0) == k_scales.shape[1] * k_scales.stride(1)
        ksf = torch.as_strided(k_scales, (k_scales.shape[0] * k_scales.shape[1], hkv), (k_scales.stride(1), k_scales.stride(2)))
        vsf = torch.as_strided(v_scales, (v_scales.shape[0] * v_scales.shape[1], hkv), (v_scales.stride(1), v_scales.stride(2)))
        ss = (ksf.stride(0), ksf.stride(1))
    if num_splits > 1:
        pm = torch.empty(n, hq, num_splits, device=query.device)
        pl = torch.empty(n, hq, num_splits, device=query.device)
        pacc = torch.empty(n, hq, num_splits, dim, device=query.device)
    else:
        pm = pl = pacc = output
    _slot_attn_kernel[(n, hkv, num_splits)](
        query, query.stride(0), query.stride(1),
        kflat, kflat.stride(0), kflat.stride(1),
        vflat, vflat.stride(0), vflat.stride(1),
        ksf, vsf, ss[0], ss[1],
        output, output.stride(0), output.stride(1),
        pm, pl, pacc, slots, slots.stride(0), lens, scale,
        G=group, D=dim, BLOCK_M=max(16, triton.next_power_of_2(group)), BLOCK_N=block_n,
        SCALE_MODE=mode, NSPLIT=num_splits, num_warps=num_warps, num_stages=num_stages)
    if num_splits > 1:
        _combine_kernel[(n, hq)](pm, pl, pacc, output, output.stride(0), output.stride(1), hq,
                                 NSPLIT=num_splits, D=dim, num_warps=4)


@triton.jit
def _ordered_key(x):
    """A float's bits as an int64 in [0, 2^32) that orders like the float."""
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(bits >= 0, bits, bits ^ 0x7FFFFFFF).to(tl.int64) + 2147483648


# ---------------------------------------------------------------------------------------------
# Multi-CTA top-p: the same radix select, each row's visible columns split over P programs


@triton.jit
def _radix_hist_kernel(S, ss, VIS, t0, PREFIX, HIST, PHIST, P: tl.constexpr, PASS: tl.constexpr,
                       CH: tl.constexpr, LAST: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    vis = tl.load(VIS + t0 + row)
    span = tl.cdiv(tl.cdiv(vis, P), CH) * CH
    lo = part * span
    hi = tl.minimum(lo + span, vis)
    shift = 24 - 8 * PASS
    prefix = tl.load(PREFIX + row)
    hist = tl.zeros((256,), tl.int32)
    for c0 in range(lo, hi, CH):
        c = c0 + tl.arange(0, CH)
        ok = c < hi
        key = _ordered_key(tl.load(S + row * ss + c, mask=ok, other=0.0))
        if PASS > 0:
            ok = ok & ((key >> (shift + 8)) == prefix)
        hist += tl.histogram(((key >> shift) & 255).to(tl.int32), 256, mask=ok)
    d = tl.arange(0, 256)
    tl.atomic_add(HIST + row * 256 + d, hist, sem="relaxed")
    if LAST:
        tl.store(PHIST + (row * P + part) * 256 + d, hist)


@triton.jit
def _radix_pick_kernel(HIST, PREFIX, REM, DIGIT, ALL, K: tl.constexpr, FIRST: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, 256)
    hist = tl.load(HIST + row * 256 + d)
    tl.store(HIST + row * 256 + d, tl.zeros((256,), tl.int32))
    if FIRST:
        remaining = tl.full((), K, tl.int32)
        prefix = tl.zeros((), tl.int64)
        tl.store(ALL + row, (tl.sum(hist, 0) <= K).to(tl.int32))
    else:
        remaining = tl.load(REM + row)
        prefix = tl.load(PREFIX + row)
    above = tl.sum(hist, 0) - tl.cumsum(hist, 0)
    fits = (above < remaining) & (above + hist >= remaining)
    digit = tl.max(tl.where(fits, d, -1), 0)
    tl.store(REM + row, remaining - tl.sum(tl.where(d == digit, above, 0), 0))
    tl.store(PREFIX + row, prefix * 256 + tl.maximum(digit, 0))
    tl.store(DIGIT + row, tl.maximum(digit, 0))


@triton.jit
def _radix_compact_kernel(S, ss, VIS, t0, PREFIX, REM, DIGIT, ALL, PHIST, COUNT, CV, CI,
                          K: tl.constexpr, P: tl.constexpr, CH: tl.constexpr):
    row = tl.program_id(0)
    part = tl.program_id(1)
    vis = tl.load(VIS + t0 + row)
    span = tl.cdiv(tl.cdiv(vis, P), CH) * CH
    lo = part * span
    hi = tl.minimum(lo + span, vis)
    take_all = tl.load(ALL + row) != 0
    kth = tl.where(take_all, -1, tl.load(PREFIX + row))
    remaining = tl.load(REM + row)
    digit = tl.load(DIGIT + row)
    q = tl.arange(0, P)
    tie_counts = tl.load(PHIST + (row * P + q) * 256 + digit)
    ties = tl.sum(tl.where(q < part, tie_counts, 0), 0)          # ties in the parts before this one
    for c0 in range(lo, hi, CH):
        c = c0 + tl.arange(0, CH)
        ok = c < hi
        x = tl.load(S + row * ss + c, mask=ok, other=0.0)
        key = _ordered_key(x)
        tie = ok & (key == kth)
        tie_rank = ties + tl.cumsum(tie.to(tl.int32), 0)
        take = (ok & (key > kth)) | (tie & (tie_rank <= remaining))
        n_take = tl.sum(take.to(tl.int32), 0)
        base = tl.atomic_add(COUNT + row, n_take, sem="relaxed")
        slot = base + tl.cumsum(take.to(tl.int32), 0) - 1
        tl.store(CV + row * K + slot, x, mask=take)
        tl.store(CI + row * K + slot, c, mask=take)
        ties += tl.sum(tie.to(tl.int32), 0)


@triton.jit
def _radix_sort_kernel(S, ss, COUNT, CV, CI, PICK, PSC, K: tl.constexpr):
    row = tl.program_id(0)
    kept = tl.load(COUNT + row)
    k = tl.arange(0, K)
    live = k < kept
    v = tl.load(CV + row * K + k, mask=live, other=0.0)
    i = tl.load(CI + row * K + k, mask=live, other=0)
    sort_key = (_ordered_key(v) << 31) | (2147483647 - i).to(tl.int64)
    sort_key = tl.where(live, sort_key, -1)
    sort_key = tl.sort(sort_key, descending=True)
    idx = (2147483647 - (sort_key & 2147483647)).to(tl.int32)
    sc = tl.load(S + row * ss + idx, mask=live, other=float("-inf"))
    tl.store(PICK + row * K + k, tl.where(live, idx, 0))
    tl.store(PSC + row * K + k, tl.where(live, sc, float("-inf")))


def top_p(scores: torch.Tensor, visible: torch.Tensor, t0: int, k: int, parts: int = 16):
    """top_p over `parts` programs a row: four histogram passes, a compaction, a sort."""
    assert k & (k - 1) == 0, "top_p must be a power of two"
    n = scores.shape[0]
    dev = scores.device
    hist = torch.zeros(n, 256, dtype=torch.int32, device=dev)
    phist = torch.empty(n, parts, 256, dtype=torch.int32, device=dev)
    prefix = torch.zeros(n, dtype=torch.int64, device=dev)
    rem = torch.empty(n, dtype=torch.int32, device=dev)
    digit = torch.empty(n, dtype=torch.int32, device=dev)
    take_all = torch.empty(n, dtype=torch.int32, device=dev)
    count = torch.zeros(n, dtype=torch.int32, device=dev)
    cv = torch.empty(n, k, dtype=torch.float32, device=dev)
    ci = torch.empty(n, k, dtype=torch.int32, device=dev)
    pick = torch.empty(n, k, dtype=torch.int32, device=dev)
    psc = torch.empty(n, k, dtype=torch.float32, device=dev)
    for p in range(4):
        _radix_hist_kernel[(n, parts)](scores, scores.stride(0), visible, t0, prefix, hist, phist,
                                       P=parts, PASS=p, CH=512, LAST=p == 3, num_warps=4)
        _radix_pick_kernel[(n,)](hist, prefix, rem, digit, take_all, K=k, FIRST=p == 0, num_warps=1)
    _radix_compact_kernel[(n, parts)](scores, scores.stride(0), visible, t0, prefix, rem, digit, take_all,
                                      phist, count, cv, ci, K=k, P=parts, CH=512, num_warps=4)
    _radix_sort_kernel[(n,)](scores, scores.stride(0), count, cv, ci, pick, psc, K=k, num_warps=8)
    return psc, pick


@triton.jit
def _expand_chunk(PICK, spick, PSCORE, MEM, BT, sbt, req, tail, n_sink, block_size, t, chunk,
                  top_p, CB: tl.constexpr, CAP: tl.constexpr, CLUSTER: tl.constexpr):
    """The members of one chunk of a row's picked buckets: positions and which are counted."""
    cols = chunk * CB + tl.arange(0, CB)
    col_ok = cols < top_p
    b = tl.load(PICK + t * spick + cols, mask=col_ok, other=0).to(tl.int64)
    sc = tl.load(PSCORE + t * spick + cols, mask=col_ok, other=float("-inf"))
    live = col_ok & (sc > float("-inf"))
    bpos = b * 8
    bblk = tl.load(BT + req * sbt + bpos // block_size, mask=live, other=0).to(tl.int64)
    row = (bblk * block_size + bpos % block_size) // 8
    k = tl.arange(0, CAP)
    local = tl.load(MEM + row[:, None] * CAP + k[None, :], mask=live[:, None], other=-1).to(tl.int64)
    p = (bpos // CLUSTER * CLUSTER)[:, None] + local
    valid = live[:, None] & (local >= 0) & (p >= n_sink) & (p < tail)
    return tl.reshape(valid, (CB * CAP,)), tl.reshape(p, (CB * CAP,))


@triton.jit
def _expand_count_kernel(PICK, spick, PSCORE, MEM, BT, sbt, TREQ, TAIL, COUNTS, top_p, n_sink,
                         block_size, NCH: tl.constexpr, CB: tl.constexpr, CAP: tl.constexpr,
                         CLUSTER: tl.constexpr):
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    req = tl.load(TREQ + t)
    tail = tl.load(TAIL + t)
    valid, _ = _expand_chunk(PICK, spick, PSCORE, MEM, BT, sbt, req, tail, n_sink, block_size, t,
                             chunk, top_p, CB, CAP, CLUSTER)
    tl.store(COUNTS + t * NCH + chunk, tl.sum(valid.to(tl.int32), 0))


@triton.jit
def _expand_write_kernel(PICK, spick, PSCORE, MEM, BT, sbt, TREQ, TPOS, TAIL, COUNTS, SLOTS, sslot,
                         LENS, top_p, top_k, n_sink, block_size, NCH: tl.constexpr, CB: tl.constexpr,
                         CAP: tl.constexpr, CLUSTER: tl.constexpr, TAIL_BLOCK: tl.constexpr):
    t = tl.program_id(0)
    chunk = tl.program_id(1)
    req = tl.load(TREQ + t)
    tail = tl.load(TAIL + t)
    sinks = tl.minimum(n_sink, tail)
    out = SLOTS + t * sslot
    q = tl.arange(0, NCH)
    counts = tl.load(COUNTS + t * NCH + q)
    if chunk < NCH:
        offset = tl.sum(tl.where(q < chunk, counts, 0), 0)
        if offset < top_k:
            valid, p = _expand_chunk(PICK, spick, PSCORE, MEM, BT, sbt, req, tail, n_sink, block_size,
                                     t, chunk, top_p, CB, CAP, CLUSTER)
            idx = offset + tl.cumsum(valid.to(tl.int32), 0) - 1
            keep = valid & (idx < top_k)
            pblk = tl.load(BT + req * sbt + p // block_size, mask=keep, other=0).to(tl.int64)
            tl.store(out + sinks + idx, (pblk * block_size + p % block_size).to(tl.int32), mask=keep)
    else:
        pos = tl.load(TPOS + t)
        count = tl.minimum(tl.sum(counts, 0), top_k)
        j = tl.arange(0, 16)
        sink_ok = j < sinks
        sblk = tl.load(BT + req * sbt + j // block_size, mask=sink_ok, other=0).to(tl.int64)
        tl.store(out + j, (sblk * block_size + j % block_size).to(tl.int32), mask=sink_ok)
        base = sinks + count
        tail_len = pos - tail + 1
        for s0 in range(0, tail_len, TAIL_BLOCK):
            ti = s0 + tl.arange(0, TAIL_BLOCK)
            tail_ok = ti < tail_len
            tp = tail + ti
            tblk = tl.load(BT + req * sbt + tp // block_size, mask=tail_ok, other=0).to(tl.int64)
            tl.store(out + base + ti, (tblk * block_size + tp % block_size).to(tl.int32), mask=tail_ok)
        tl.store(LENS + t, (base + tail_len).to(tl.int32))


def expand_slots(picked, picked_scores, members, block_table, token_requests, token_positions,
                          tail_starts, top_k, n_sink, block_size, width, cb=128):
    n, top_p = picked.shape
    nch = triton.cdiv(top_p, cb)
    assert nch & (nch - 1) == 0
    slots = torch.empty(n, width, dtype=torch.int32, device=picked.device)
    lens = torch.empty(n, dtype=torch.int32, device=picked.device)
    counts = torch.empty(n, nch, dtype=torch.int32, device=picked.device)
    common = dict(CB=cb, CAP=members.shape[1], CLUSTER=512)
    _expand_count_kernel[(n, nch)](picked, picked.stride(0), picked_scores, members, block_table,
                                   block_table.stride(0), token_requests, tail_starts, counts, top_p,
                                   n_sink, block_size, NCH=nch, num_warps=4, **common)
    _expand_write_kernel[(n, nch + 1)](picked, picked.stride(0), picked_scores, members, block_table,
                                       block_table.stride(0), token_requests, token_positions, tail_starts,
                                       counts, slots, slots.stride(0), lens, top_p, top_k, n_sink, block_size,
                                       NCH=nch, TAIL_BLOCK=128, num_warps=4, **common)
    return slots, lens

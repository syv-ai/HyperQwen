"""Split-KV paged attention for speculative-decode batches (a few query tokens per
request, GQA), Triton.

Neither vLLM's FlashAttention-2 path nor its Triton unified attention split the KV
sequence across SMs when a request has more than one query token: with MTP k=4
(5 queries) on a 24-head model that leaves 24 (FA) or ~8 (Triton) thread blocks on an
82-SM RTX 3090 and the attention layer takes ~57 us for a 1.5k-token context.
This kernel gives every (request, kv-head, query tile) NUM_SEGMENTS blocks, each
computing an online-softmax partial over its slice of the KV cache for all its query
rows at once (G = query heads per kv head), followed by a tiny combine kernel.

Long blocks: the query rows of one request (q_len x G of them) are split into tiles of
BLOCK_M rows, so the kernel is not capped at BLOCK_M // G query tokens. That cap is what
made lookup-augmented drafting unaffordable past a block of 7 -- a 16-token verify fell
back to FA2 and doubled the step at 25k context. Each tile re-reads the KV segment, so
BLOCK_M is chosen to keep the tile count at 1 where the register budget allows.

Layout matches vLLM's FLASH_ATTN backend: q [T, Hq, D], key/value cache
[num_blocks, block_size, Hkv, D] (block_size any multiple of 16), block_table
[num_reqs, max_blocks], seqused_k [num_reqs] = kv length including the new tokens,
cu_seqlens_q [num_reqs + 1]. Query token i of a request sits at kv position
seqused_k - q_len + i and attends causally.

int8 KV (QUANT=1): the same kernel reads TRITON_ATTN's `int8_per_token_head` cache. That
backend hands out K/V views of shape [num_blocks, block_size, Hkv, D + 4] -- the head dim
is padded so one float32 scale per (token, head) sits inline after the data -- plus f32
scale views [num_blocks, block_size, Hkv]. Only the element type and the scale multiply
change here: the padded head dim is already carried by stride_kh, so the addressing is
untouched. Both scales are per (token, head), i.e. constant along D, so folding them in
after the dot is exact rather than an approximation:
  s   = (q . k_int8) * k_scale        instead of  q . (k_int8 * k_scale)
  acc = (p * v_scale) . v_int8        instead of  p . (v_int8 * v_scale)
int8 -> bf16 is itself exact (bf16 has 8 mantissa bits, int8 needs 7), so the only new
rounding is the one bf16 multiply the stock Triton kernel also does.

Halving the bytes matters because this kernel is bandwidth-bound at long context: at a
128k context and 8 query tokens it reads 524 MB of bf16 KV in 1,198 us, which is 437 GB/s
of the 3090's ~936.

Restrictions: q_len <= QMAX_TOKENS per request, D a power of two <= 256, no sliding
window / softcap / alibi.
"""
import vllm.envs as envs

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

NUM_SEGMENTS = 16
TARGET_BLOCKS = 64  # blocks enough to cover the SMs at the shapes we see; more only adds overhead
BLOCK_M = 64       # query rows (q_len * G) per program at the default register budget
BLOCK_M_BIG = 128  # ... and with 8 warps, which keeps a 16-token block in one tile
QMAX_TOKENS = 64   # query tokens per request the caller may ask for


@triton.jit
def _spec_attn_partial(
    q_ptr, k_ptr, v_ptr, bt_ptr, seqused_ptr, cu_q_ptr,
    part_o_ptr, part_m_ptr, part_l_ptr,
    ks_ptr, vs_ptr, kd_ptr, vd_ptr, qd_ptr,
    scale,
    stride_qt, stride_qh,
    stride_kb, stride_ks, stride_kh,
    stride_vb, stride_vs, stride_vh,
    stride_bt,
    stride_ksb, stride_kss, stride_ksh,
    stride_vsb, stride_vss, stride_vsh,
    G: tl.constexpr, Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr, TILE: tl.constexpr, NSEG: tl.constexpr, QT: tl.constexpr,
    NTILE: tl.constexpr, QUANT: tl.constexpr, FP8: tl.constexpr, Q_FP8: tl.constexpr,
    CDT: tl.constexpr,
):
    pid = tl.program_id(0)
    req = pid // NTILE
    qtile = pid % NTILE
    kvh = tl.program_id(1)
    seg = tl.program_id(2)

    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    kv_len = tl.load(seqused_ptr + req)

    # rows: r = i * G + g  -> query token qtile * QT + i (0..q_len-1), head kvh*G + g
    r = tl.arange(0, BLOCK_M)
    ri = qtile * QT + r // G
    rg = r % G
    row_ok = (r < QT * G) & (ri < q_len)
    q_pos = kv_len - q_len + ri                      # kv position of each query row
    d = tl.arange(0, D)
    q_ptrs = q_ptr + (q_start + ri)[:, None] * stride_qt + (kvh * G + rg)[:, None] * stride_qh + d[None, :]
    q = tl.load(q_ptrs, mask=row_ok[:, None], other=0.0)
    if Q_FP8:
        # the fp8 cache path quantizes the query too (layer._q_scale); undo it once here
        q = q.to(tl.float32) * tl.load(qd_ptr)

    # this segment's key range
    tiles_total = (kv_len + TILE - 1) // TILE
    tiles_per_seg = (tiles_total + NSEG - 1) // NSEG
    t0 = seg * tiles_per_seg
    t1 = tl.minimum(t0 + tiles_per_seg, tiles_total)

    m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qs = (q * scale).to(CDT)
    if FP8:
        # fp8 e4m3 data with one f32 scale per layer tensor (vLLM's `fp8` cache): fp8 -> bf16 is exact
        # (3 mantissa bits into 8), and the scalar scales fold in after each dot, the way the int8
        # per-token scales do, so the only new rounding is the bf16 dot itself.
        k_dsc = tl.load(kd_ptr)
        v_dsc = tl.load(vd_ptr)

    for t in range(t0, t1):
        pos = t * TILE + tl.arange(0, TILE)
        k_ok = pos < kv_len
        blk = tl.load(bt_ptr + req * stride_bt + pos // BLOCK_SIZE, mask=k_ok, other=0)
        # int64: blk * stride_kb (and the int8 scale strides) overflow int32 for high block ids in a
        # large KV pool (issue #86: a 480k-token pool on a 64 GB card).
        blk = blk.to(tl.int64)
        slot = pos % BLOCK_SIZE
        k_ptrs = k_ptr + blk[:, None] * stride_kb + slot[:, None] * stride_ks + kvh * stride_kh + d[None, :]
        v_ptrs = v_ptr + blk[:, None] * stride_vb + slot[:, None] * stride_vs + kvh * stride_vh + d[None, :]
        k = tl.load(k_ptrs, mask=k_ok[:, None], other=0.0)
        v = tl.load(v_ptrs, mask=k_ok[:, None], other=0.0)
        if QUANT:
            # int8 data, one f32 scale per (token, head); both constant along D.
            k = k.to(CDT)
            v = v.to(CDT)
            k_sc = tl.load(ks_ptr + blk * stride_ksb + slot * stride_kss + kvh * stride_ksh,
                           mask=k_ok, other=0.0)
            v_sc = tl.load(vs_ptr + blk * stride_vsb + slot * stride_vss + kvh * stride_vsh,
                           mask=k_ok, other=0.0)
            s = tl.dot(qs, tl.trans(k)).to(tl.float32) * k_sc[None, :]
        elif FP8:
            k = k.to(CDT)
            v = v.to(CDT)
            s = tl.dot(qs, tl.trans(k)).to(tl.float32) * k_dsc
        else:
            s = tl.dot(qs, tl.trans(k)).to(tl.float32)            # [BLOCK_M, TILE]
        allowed = k_ok[None, :] & (pos[None, :] <= q_pos[:, None]) & row_ok[:, None]
        s = tl.where(allowed, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(tl.where(m_i == float("-inf"), float("-inf"), m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, 1)
        if QUANT:
            acc = acc * alpha[:, None] + tl.dot((p * v_sc[None, :]).to(CDT), v).to(tl.float32)
        elif FP8:
            acc = acc * alpha[:, None] + tl.dot(p.to(CDT), v).to(tl.float32) * v_dsc
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(CDT), v).to(tl.float32)
        m_i = m_new

    # store partials at flat index ((req*Hq + head)*QMAX + i)*NSEG + seg
    hrow = kvh * G + rg
    pidx = ((req * Hq + hrow) * QMAX + ri) * NSEG + seg
    tl.store(part_o_ptr + pidx[:, None] * D + d[None, :], acc, mask=row_ok[:, None])
    tl.store(part_m_ptr + pidx, m_i, mask=row_ok)
    tl.store(part_l_ptr + pidx, l_i, mask=row_ok)


@triton.jit
def _spec_attn_combine(
    part_o_ptr, part_m_ptr, part_l_ptr, out_ptr, cu_q_ptr,
    stride_ot, stride_oh,
    Hq: tl.constexpr, QMAX: tl.constexpr, D: tl.constexpr, NSEG: tl.constexpr,
):
    req = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    q_start = tl.load(cu_q_ptr + req)
    q_len = tl.load(cu_q_ptr + req + 1) - q_start
    if i < q_len:
        base = ((req * Hq + h) * QMAX + i) * NSEG
        segs = tl.arange(0, NSEG)
        m = tl.load(part_m_ptr + base + segs)
        l = tl.load(part_l_ptr + base + segs)
        m_max = tl.max(m, 0)
        m_max = tl.where(m_max == float("-inf"), 0.0, m_max)
        w = tl.exp(m - m_max)                       # segments with -inf give 0
        l_tot = tl.sum(l * w, 0)
        d = tl.arange(0, D)
        o = tl.load(part_o_ptr + (base + segs)[:, None] * D + d[None, :])   # [NSEG, D]
        o = tl.sum(o * w[:, None], 0) / tl.maximum(l_tot, 1e-30)
        tl.store(out_ptr + (q_start + i) * stride_ot + h * stride_oh + d, o.to(out_ptr.dtype.element_ty))



# (device, BLOCK_M, D, cache dtype) -> the KV tile that launched there. Filled on first use,
# so a device with less shared memory than sm86's pays the failed launch once.
_TILE_FITS = {}

class SpecDecodeAttention:
    """Holds the partial buffers; call .run(...) per layer."""

    def __init__(self, max_num_reqs, num_heads, head_dim, device, qmax,
                 num_segments=NUM_SEGMENTS):
        # The partial buffers are allocated once, for the longest query block the caller
        # will ever pass (qmax): a CUDA graph captures their addresses, so growing them
        # later would leave the captured decode graph pointing at freed memory.
        self.nseg = num_segments
        self.qmax = qmax
        self.max_num_reqs = max_num_reqs
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        n = max_num_reqs * num_heads * qmax * self.nseg
        self.part_o = torch.empty(n, head_dim, dtype=torch.float32, device=device)
        self.part_m = torch.empty(n, dtype=torch.float32, device=device)
        self.part_l = torch.empty(n, dtype=torch.float32, device=device)

    def _segments(self, num_reqs, ntile, Hkv):
        """Active KV segments. Chosen from the (captured) batch size only, so it is
        fixed for the life of a CUDA graph and safe to vary between graphs. More
        segments than the SMs can hold only add combine work: at batch 8 the table
        is as fast with 2 segments as with 16."""
        override = envs.VLLM_SPEC_ATTN_NSEG
        if override:
            n = max(1, min(self.nseg, override))
        else:
            n = max(1, min(self.nseg, triton.cdiv(TARGET_BLOCKS, num_reqs * ntile * Hkv)))
        # The combine kernel's tl.arange(0, NSEG) needs a power of two, and HQ's
        # capture sizes are not all powers of two (24, 48), so round down.
        return 1 << (n.bit_length() - 1)

    def _plan(self, q_len, G, D):
        """(BLOCK_M, tokens per tile, tile count, warps). One tile is preferred: every
        extra tile re-reads this request's KV segment."""
        override = envs.VLLM_SPEC_ATTN_BLOCK_M
        rows = q_len * G
        if override:
            block_m = override
        elif rows <= 32:
            block_m = 32
        elif rows <= 64:
            block_m = 64
        else:
            block_m = BLOCK_M_BIG
        block_m = max(block_m, G if G > 1 else 1)
        block_m = 1 << (block_m - 1).bit_length()
        qt = max(1, block_m // G)
        # D=256 makes the [BLOCK_M, D] accumulator too large for 4 warps.
        warps = 8 if (block_m >= 128 or D > 128) else 4
        return block_m, qt, triton.cdiv(q_len, qt), warps

    def run(self, q, key_cache, value_cache, out, cu_seqlens_q, seqused_k, block_table, scale,
            num_reqs, max_query_len, k_scale_cache=None, v_scale_cache=None,
            k_descale=None, v_descale=None, q_descale=None):
        """key/value_cache: [num_blocks, block_size, Hkv, D] bf16, or [.., D + pad] int8
        with k/v_scale_cache [num_blocks, block_size, Hkv] float32 (per-token-head int8), or
        [.., D] fp8 e4m3 with k/v_descale one-element float32 tensors (vLLM's per-tensor `fp8`)."""
        Hq, D = q.shape[1], q.shape[2]
        Hkv = key_cache.shape[2]
        G = Hq // Hkv
        quant = k_scale_cache is not None
        fp8 = key_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        q_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        if fp8:
            assert not quant and k_descale is not None and v_descale is not None, "fp8 cache needs per-tensor descales"
        if q_fp8:
            assert q_descale is not None, "fp8 query needs its descale"
        assert max_query_len <= self.qmax, "too many query tokens per request for this kernel"
        assert num_reqs <= self.max_num_reqs
        # shared memory on sm86 is 99 KB: q tile + one K and one V tile + scores must fit.
        # Turing (sm75) has 64 KB, where a 64-token tile at D=256 needs 98,304 bytes; a tile
        # that does not launch is halved below, and the size that did is remembered.
        block_m, qt, ntile, warps = self._plan(max_query_len, G, D)
        # 170HX (GA100) has 163 KB smem per block, not sm86's 99 KB, so a
        # 64-token tile fits at D=256 for the production 8-query shape. The
        # OutOfResources retry below still halves it on any shape that does not.
        tile = 64 if (block_m <= 32 or D <= 128 or (D == 256 and block_m <= 64)) else 32
        fit = (q.device, block_m, D, key_cache.dtype)
        tile = _TILE_FITS.get(fit, tile)
        if isinstance(tile, tuple):
            # A smaller BLOCK_M was needed to launch on this device (see the retry below).
            block_m, tile = tile
            qt = max(1, block_m // G)
            ntile = triton.cdiv(max_query_len, qt)
            warps = 8 if block_m >= 128 else 4
        nseg = self._segments(num_reqs, ntile, Hkv)
        if envs.VLLM_SPEC_ATTN_DEBUG:
            self._dbg_calls = getattr(self, "_dbg_calls", 0) + 1
            if self._dbg_calls in (1, 2, 500, 2000):
                import time as _t
                torch.cuda.synchronize(); _t0 = _t.perf_counter()
                print(f"syv spec-attn run#{self._dbg_calls}: q {q.dtype} {tuple(q.shape)} strides {q.stride()} contig={q.is_contiguous()} | "
                      f"kc {key_cache.dtype} {tuple(key_cache.shape)} strides {key_cache.stride()} | out {out.dtype} strides {out.stride()} | "
                      f"quant={quant} fp8={fp8} q_fp8={q_fp8} kd={None if k_descale is None else float(k_descale.reshape(-1)[0])} "
                      f"vd={None if v_descale is None else float(v_descale.reshape(-1)[0])} qd={None if q_descale is None else float(q_descale.reshape(-1)[0])} | "
                      f"num_reqs={num_reqs} qlen={max_query_len} plan block_m={block_m} qt={qt} ntile={ntile} warps={warps} tile={tile} nseg={nseg} "
                      f"seqused={seqused_k[:num_reqs].tolist()}", flush=True)
                self._dbg_t0 = _t0
        # Dot-operand dtype follows the model: bf16 where it is native, fp16 on Turing (sm75),
        # whose tensor cores have no bf16 MMA. int8 and fp8 codes convert to either exactly.
        cdt = tl.float16 if out.dtype == torch.float16 else tl.bfloat16
        grid = (num_reqs * ntile, Hkv, nseg)
        block_m0 = block_m
        while True:
            try:
                _spec_attn_partial[grid](
                    q, key_cache, value_cache, block_table, seqused_k, cu_seqlens_q,
                    self.part_o, self.part_m, self.part_l,
                    k_scale_cache, v_scale_cache, k_descale, v_descale, q_descale,
                    scale,
                    q.stride(0), q.stride(1),
                    key_cache.stride(0), key_cache.stride(1), key_cache.stride(2),
                    value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                    block_table.stride(0),
                    *(k_scale_cache.stride() if quant else (0, 0, 0)),
                    *(v_scale_cache.stride() if quant else (0, 0, 0)),
                    G=G, Hq=Hq, QMAX=self.qmax, D=D, BLOCK_SIZE=key_cache.shape[1], BLOCK_M=block_m,
                    TILE=tile, NSEG=nseg, QT=qt, NTILE=ntile, QUANT=quant, FP8=fp8, Q_FP8=q_fp8,
                    CDT=cdt,
                    num_warps=warps, num_stages=1,
                )
                break
            except OutOfResources:
                if tile > 16:
                    tile //= 2
                    continue
                # The KV tile is at its floor and the launch still does not fit: the rest is
                # the query tile and the [BLOCK_M, D] fp32 accumulator, which scale with
                # BLOCK_M (Turing, 64 KB: 64 rows x D=256 needs 80 KB at a 16-token tile).
                # Halve BLOCK_M (more query tiles per request) and restore the tile.
                if block_m // 2 < max(16, G):
                    raise
                block_m //= 2
                qt = max(1, block_m // G)
                ntile = triton.cdiv(max_query_len, qt)
                warps = 8 if block_m >= 128 else 4
                grid = (num_reqs * ntile, Hkv, nseg)
                tile = 64 if (block_m <= 32 or D <= 128) else 32
        _TILE_FITS[fit] = tile if block_m == block_m0 else (block_m, tile)
        _spec_attn_combine[(num_reqs, Hq, max_query_len)](
            self.part_o, self.part_m, self.part_l, out, cu_seqlens_q,
            out.stride(0), out.stride(1),
            Hq=Hq, QMAX=self.qmax, D=D, NSEG=nseg, num_warps=4,
        )
        if envs.VLLM_SPEC_ATTN_DEBUG and getattr(self, "_dbg_t0", None) is not None:
            import time as _t
            torch.cuda.synchronize(); print(f"syv spec-attn run#{self._dbg_calls}: kernel pair took {(_t.perf_counter() - self._dbg_t0) * 1e6:.0f} us", flush=True)
            self._dbg_t0 = None
        return out

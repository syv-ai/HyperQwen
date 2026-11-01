# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""Host-side launch planner for ``pth_flash_attn_fwd``.

This module is the single source of truth for every launch-geometry decision:
warps along M (``wm``), query tokens per CTA (``bq``), the grid's q-tile upper
bound, the split-KV count ``num_splits`` and the workspace size. The vLLM
metadata builder, the benchmarks and the tests all call it; nothing else
re-derives these numbers.

Why it is pure Python with no torch import:
Every input is a host ``int`` taken from the *padded batch descriptor* that
vLLM already holds on the CPU. A CUDA graph captures one grid per descriptor
and replays it for every later step with that descriptor, while sequence
lengths keep growing on the device. If any output here depended on a sequence
length, a replay would silently run the grid planned for a different length.
So the functions cannot even *receive* a length: there is no such parameter,
all parameters are keyword-only (no ``*args``/``**kwargs`` to smuggle one in),
and every argument must be a real ``int``/``bool`` (a 0-d tensor would pass
``operator.index`` and force a device-to-host sync; it is rejected here).
The SM count and the CTAs per SM are inputs or compile-time mirrors, never
device queries, so this file runs on a CPU-only CI box.

Derivations:
A CTA has 4 warps x 16 MMA rows = 64 query rows. With GQA packing a row is one
(query token, query head of the group), so a CTA holds at most
``b_max = 64 // G`` query tokens of one request.

Decode-like launch (every request has q <= the reorder threshold)::

    rt = ceil(q_max / b_max)          row tiles per request (fewest possible)
    wm = smallest of {1, 2, 4} with 16*wm >= ceil(q_max / rt) * G
    bq = floor(16 * wm / G)           tokens per tile, the kernel's own rule
                                      (cache_geometry.cuh); q=16, G=6: 10+6
    S  = ceil(C_target / (Hkv * rt))  splits so ONE request fills the GPU

``wm`` is the fewest warps that hold q_max in rt tiles; the kernel then fills
each tile with as many tokens as its 16*wm rows allow. The op balances the tiles
itself for a decode-like launch: it takes the smaller of this bq and
``max_query_len`` as the tile height, so the plan carries
no bq argument.

Prefill launch::

    bq = b_max                        (G=6: 10 tokens = 60 of 64 rows)
    S  = ceil(C_target / (Hkv * ceil(T / bq)))

Both::

    C_target      = num_sms * CTAS_PER_SM[D]
    S             = clamp(S, 1, ws_token_splits // T)   so S > 1 => S*T <= W_ts
    q_tiles_upper = T // bq + num_reqs  (>= sum_i ceil(q_i / bq); each CTA
                                         binary-searches query_start_loc)

Sliding window W > 0::

    S = min(S, ceil((W + q_max - 2) / 64) + 1)

A row tile of at most q_max tokens sees at most n = W + q_max - 1 keys,
causal or not (its queries lie within the last q_max positions of the
request). n consecutive keys touch at most ceil((n - 1) / 64) + 1 KV tiles
(n - 1 key steps after the first key, at the worst alignment), so the cap is
the most tiles one row tile can have: at the cap no split is empty at the
worst alignment, and every split it removes would have been empty
(lse = -inf, an extra combine input). W is a layer constant, not a length, so
the plan stays a pure function of the descriptor.

Workspace in token-splits::

    S_alone = ceil(C_target / Hkv)
    W_ts    = min(max_num_seqs, S_alone) * q_dec_max * S_alone
    bytes   = W_ts * Hq * (D + 1) * 4   (fp32 O partials + fp32 log2-LSE)
"""

from __future__ import annotations

from dataclasses import dataclass

# Compile-time mirrors of the kernel's traits.

# Query rows per MMA slice: the M of mma.sync.m16n8k16. One warp owns 16 rows.
MMA_ROWS_PER_WARP = 16
# Warp layouts along M that the kernel instantiates (WM x WN = 4).
WM_CHOICES = (1, 2, 4)
# Rows per CTA = 16 * max(WM) = 64; this is also the GQA packing limit G <= 64.
MAX_ROWS_PER_CTA = MMA_ROWS_PER_WARP * WM_CHOICES[-1]
# CTAs resident on one SM, by head size (register/smem probe):
# D=256 needs 99,328 B smem (1/SM); D=128 needs 50,176 B smem and 198 registers
# (2/SM); D=64 needs 167 registers x 128 threads x 3 = 64,128 <= 65,536 (3/SM).
# The GPU tests check these against the occupancy API, not here.
CTAS_PER_SM = {64: 3, 128: 2, 256: 1}
SUPPORTED_HEAD_SIZES = tuple(sorted(CTAS_PER_SM))
# Head size with a windowed (sliding-window / non-causal) kernel variant; the
# op refuses window > 0 or causal = False at any other D.
WINDOWED_HEAD_SIZES = (128,)
# KV tokens per tile: kLoadTileTokens in csrc/pth_attn/tile_traits.cuh.
KV_TILE_TOKENS = 64
# O partials and the per-row log2-LSE are fp32.
PARTIAL_BYTES_PER_ELEMENT = 4

# The only parameters plan_launch may ever accept (a test enforces this).
DESCRIPTOR_FIELDS = (
    "num_tokens",
    "num_reqs",
    "max_query_len",
    "G",
    "Hkv",
    "D",
    "num_sms",
    "is_prefill",
    "ws_token_splits",
    "window",
)


def _ceil_div(a: int, b: int) -> int:
    # Exact integer ceil(a / b) for a >= 0, b > 0: no float rounding at 2**53.
    return -(-a // b)


def _require_int(name: str, value: object, minimum: int) -> int:
    # bool is an int subclass (is_prefill=True passed as G would be a bug), and
    # a 0-d torch.Tensor or numpy scalar is not an int: accepting either would
    # make a planner call a hidden device-to-host sync or a silent type change.
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be a host int, got {type(value).__name__}")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _require_head_size(D: object) -> int:
    d = _require_int("D", D, 1)
    if d not in CTAS_PER_SM:
        raise ValueError(f"D must be one of {SUPPORTED_HEAD_SIZES}, got {d}")
    return d


def _require_group_size(G: object) -> int:
    g = _require_int("G", G, 1)
    if g > MAX_ROWS_PER_CTA:
        # One query-head group must fit one 64-row tile.
        raise ValueError(f"G must be <= {MAX_ROWS_PER_CTA}, got {g}")
    return g


def c_target(*, D: int, num_sms: int) -> int:
    """Return the CTA count that fills the GPU in one wave: SMs x CTAs/SM(D)."""
    d = _require_head_size(D)
    return _require_int("num_sms", num_sms, 1) * CTAS_PER_SM[d]


def max_tokens_per_tile(G: int) -> int:
    """Return b_max = 64 // G, the query tokens of one request per row tile.

    It is also the base of the reorder (decode) threshold:
    ``_init_reorder_batch_threshold(max_tokens_per_tile(G), ...)``.
    """
    return MAX_ROWS_PER_CTA // _require_group_size(G)


def wm_for_rows(rows: int) -> int:
    """Return the smallest WM in {1, 2, 4} whose 16*WM MMA rows cover rows."""
    r = _require_int("rows", rows, 1)
    for wm in WM_CHOICES:
        if r <= MMA_ROWS_PER_WARP * wm:
            return wm
    raise ValueError(f"{r} rows exceed one {MAX_ROWS_PER_CTA}-row CTA tile")


def split_tile_range(*, split: int, num_splits: int, num_tiles: int) -> tuple[int, int]:
    """Return the KV-tile range [start, end) that split ``split`` processes.

    The kernel computes the same partition on the device from the device-side
    sequence length: ``[floor(s*T/S), floor((s+1)*T/S))``. Any two
    splits differ by at most one tile, the ranges are disjoint and cover all
    tiles, and when T < S some splits are empty (they write lse = -inf and the
    combine skips them). This mirror exists so the property is tested on CPU.
    """
    s_count = _require_int("num_splits", num_splits, 1)
    s = _require_int("split", split, 0)
    tiles = _require_int("num_tiles", num_tiles, 0)
    if s >= s_count:
        raise ValueError(f"split {s} out of range for num_splits={s_count}")
    return s * tiles // s_count, (s + 1) * tiles // s_count


@dataclass(frozen=True)
class LaunchPlan:
    """Everything one ``pth_flash_attn_fwd`` launch needs from the host.

    The first four fields are the frozen interface to the op (docs/api.md). The
    remaining fields record the descriptor the plan was made for, so the grid and the
    workspace view shapes are derived here and nowhere else.
    """

    wm: int  # warps along M; rows per CTA <= 16 * wm (C++ checks G <= 16*wm)
    bq: int  # query tokens per CTA (row tile) = floor(16*wm/G), as the kernel
    q_tiles_upper: int  # grid.x: upper bound on the number of row tiles
    num_splits: int  # S: KV splits per (row tile, KV head); grid.z
    num_tokens: int  # T of this launch (padded, from the descriptor)
    G: int  # query heads per KV head
    Hkv: int  # KV heads on this rank; grid.y
    D: int  # head size
    is_prefill: bool

    @property
    def Hq(self) -> int:
        """Query heads on this rank, G * Hkv."""
        return self.G * self.Hkv

    @property
    def rows_per_cta(self) -> int:
        """MMA rows one CTA computes: bq tokens x G heads (<= 16 * wm)."""
        return self.bq * self.G

    @property
    def grid(self) -> tuple[int, int, int]:
        """Return (q-tile upper bound, KV heads, splits); x varies fastest.

        x fastest keeps the row tiles of one (head, split) co-resident, so they
        can share K/V through L2 (an assumption of the design, not measured
        separately).
        """
        return (self.q_tiles_upper, self.Hkv, self.num_splits)

    @property
    def ws_token_splits_used(self) -> int:
        """Token-splits of workspace written; 0 when S == 1 (no combine)."""
        return self.num_splits * self.num_tokens if self.num_splits > 1 else 0

    @property
    def ws_o_shape(self) -> tuple[int, int, int, int]:
        """Shape of the fp32 O partials [S, T, Hq, D]; S is 0 when S == 1."""
        s = self.num_splits if self.num_splits > 1 else 0
        return (s, self.num_tokens, self.Hq, self.D)

    @property
    def ws_lse_shape(self) -> tuple[int, int, int]:
        """Shape of the fp32 log2-LSE partials [S, T, Hq]; S is 0 when S == 1."""
        s = self.num_splits if self.num_splits > 1 else 0
        return (s, self.num_tokens, self.Hq)


def plan_launch(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    G: int,
    Hkv: int,
    D: int,
    num_sms: int,
    is_prefill: bool,
    ws_token_splits: int,
    window: int = 0,
) -> LaunchPlan:
    """Plan one launch from the padded batch descriptor of its slice.

    There is deliberately no sequence-length parameter (see the module
    docstring): the result is a pure function of these host integers, so a
    CUDA graph captured for a descriptor stays valid however long the
    sequences grow.

    Args:
        num_tokens: query tokens in this launch (padded, as captured).
        num_reqs: requests in this launch (padded, as captured).
        max_query_len: host bound on any request's query length.
        G: query heads per KV head (1..64).
        Hkv: KV heads on this rank.
        D: head size, one of 64/128/256.
        num_sms: SM count of the device (an input; never queried here).
        is_prefill: True for the prefill launch, False for the decode-like one.
        ws_token_splits: W_ts reserved at load (``workspace_token_splits``).
        window: the layer's sliding window W in tokens, 0 for none (the op's
            ``window``). A layer constant, not a length: W > 0 caps S at the
            KV tiles one row tile can see (module docstring).

    Returns:
        The frozen LaunchPlan for this launch.
    """
    t = _require_int("num_tokens", num_tokens, 1)
    reqs = _require_int("num_reqs", num_reqs, 1)
    q_max = _require_int("max_query_len", max_query_len, 1)
    g = _require_group_size(G)
    hkv = _require_int("Hkv", Hkv, 1)
    d = _require_head_size(D)
    sms = _require_int("num_sms", num_sms, 1)
    w_ts = _require_int("ws_token_splits", ws_token_splits, 0)
    w = _require_int("window", window, 0)
    if not isinstance(is_prefill, bool):
        raise TypeError(f"is_prefill must be a bool, got {type(is_prefill).__name__}")
    if q_max > t:
        raise ValueError(f"max_query_len={q_max} > num_tokens={t}")
    if t > reqs * q_max:
        raise ValueError(
            f"num_tokens={t} cannot come from {reqs} requests of at most {q_max} tokens"
        )

    b_max = max_tokens_per_tile(g)
    if is_prefill:
        wm = WM_CHOICES[-1]
        # bq = floor(64 / G) = b_max. sum_i ceil(q_i / bq) >= ceil(T / bq), so
        # planning with the lower bound ceil(T / bq) never under-fills the GPU.
        tiles_for_split = _ceil_div(t, b_max)
    else:
        rt = _ceil_div(q_max, b_max)  # fewest tiles that hold q_max tokens
        # Per REQUEST, not per batch: a lone long request among short ones must
        # still fill the GPU, and the planner cannot know which one is long.
        tiles_for_split = rt
        # Fewest warps whose rows hold q_max in rt tiles. A smaller wm would
        # leave floor(16*wm/G) < ceil(q_max/rt), i.e. need more than rt tiles.
        wm = wm_for_rows(_ceil_div(q_max, rt) * g)
    # The kernel derives its tile height from wm (no bq in the op schema), so
    # the plan reports exactly that: floor(16*wm/G) >= ceil(q_max/rt), hence
    # the request still takes rt tiles and the split count below is the
    # kernel's. In the prefill launch this is floor(64 / G) = b_max.
    bq = MMA_ROWS_PER_WARP * wm // g

    s_want = _ceil_div(c_target(D=d, num_sms=sms), hkv * tiles_for_split)
    s_cap = w_ts // t  # S * T must fit the workspace reserved at load
    num_splits = max(1, min(s_want, s_cap))
    if w > 0:
        # The most KV tiles one row tile can touch at any alignment: its
        # W + q_max - 1 keys span ceil((W + q_max - 2) / 64) + 1 tiles at worst.
        num_splits = min(num_splits, _ceil_div(w + q_max - 2, KV_TILE_TOKENS) + 1)

    return LaunchPlan(
        wm=wm,
        bq=bq,
        q_tiles_upper=t // bq + reqs,
        num_splits=num_splits,
        num_tokens=t,
        G=g,
        Hkv=hkv,
        D=d,
        is_prefill=is_prefill,
    )


def s_alone(*, Hkv: int, D: int, num_sms: int) -> int:
    """Return the splits that let ONE row tile fill the GPU: ceil(C_target/Hkv)."""
    hkv = _require_int("Hkv", Hkv, 1)
    return _ceil_div(c_target(D=D, num_sms=num_sms), hkv)


def workspace_token_splits(
    *, max_num_seqs: int, q_dec_max: int, Hkv: int, D: int, num_sms: int
) -> int:
    """Return W_ts = min(max_num_seqs, S_alone) * q_dec_max * S_alone.

    Sized so that up to S_alone concurrent decode requests of up to
    ``q_dec_max`` tokens can each take the full S_alone splits. Beyond S_alone
    requests the GPU is filled by requests rather than splits, and the clamp in
    ``plan_launch`` lowers S instead of growing the workspace.
    """
    seqs = _require_int("max_num_seqs", max_num_seqs, 1)
    q_dec = _require_int("q_dec_max", q_dec_max, 1)
    alone = s_alone(Hkv=Hkv, D=D, num_sms=num_sms)
    return min(seqs, alone) * q_dec * alone


def workspace_shapes(
    *, ws_token_splits: int, Hq: int, D: int
) -> tuple[tuple[int, int, int], tuple[int, int]]:
    """Return the fp32 shapes to reserve at Impl ``__init__``.

    O partials [W_ts, Hq, D] and log2-LSE partials [W_ts, Hq]. Any launch view
    [S, T, Hq, D] / [S, T, Hq] with S*T <= W_ts fits inside them.
    """
    w_ts = _require_int("ws_token_splits", ws_token_splits, 0)
    hq = _require_int("Hq", Hq, 1)
    d = _require_head_size(D)
    return (w_ts, hq, d), (w_ts, hq)


def workspace_bytes(*, ws_token_splits: int, Hq: int, D: int) -> int:
    """Return W_ts * Hq * (D + 1) * 4, the bytes of both fp32 partial buffers."""
    o_shape, lse_shape = workspace_shapes(ws_token_splits=ws_token_splits, Hq=Hq, D=D)
    numel = o_shape[0] * o_shape[1] * o_shape[2] + lse_shape[0] * lse_shape[1]
    return numel * PARTIAL_BYTES_PER_ELEMENT

"""Opt-in FA2 prefill with a 64 MiB K/V slab; native PTH handles decode.

The FlashAttention implementation is unchanged. Only paged-cache gathering
and the stable merge of normalized partial outputs are local Triton kernels.
"""
import os

import torch
from vllm.triton_utils import tl, triton

from pth_flash_attn.fa2_slab_plan import QUERY_CAPACITY, SLAB_TOKENS, slabs, workspace_bytes


@triton.jit(do_not_specialize=["START", "N"])
def _gather(K, V, KS, VS, BT, OK, OV, CU, START, N,
            K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr, K3: tl.constexpr,
            V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr, V3: tl.constexpr,
            S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr,
            T0: tl.constexpr, T1: tl.constexpr, T2: tl.constexpr,
            PAGE: tl.constexpr, TILE: tl.constexpr = 16):
    tile, head = tl.program_id(0), tl.program_id(1)
    local = tile * TILE + tl.arange(0, TILE)
    token = START + local
    valid = local < N
    dim = tl.arange(0, 256)
    page = tl.load(BT + token // PAGE, valid, other=0).to(tl.int64)
    slot = token % PAGE
    kp = page[:, None]*K0 + slot[:, None]*K1 + head*K2 + dim[None, :]*K3
    vp = page[:, None]*V0 + slot[:, None]*V1 + head*V2 + dim[None, :]*V3
    ks = tl.load(KS + page*S0 + slot*S1 + head*S2, valid, other=0)
    vs = tl.load(VS + page*T0 + slot*T1 + head*T2, valid, other=0)
    k = tl.load(K + kp, valid[:, None], other=0).to(tl.float32) * ks[:, None]
    v = tl.load(V + vp, valid[:, None], other=0).to(tl.float32) * vs[:, None]
    offset = local[:, None]*1024 + head*256 + dim[None, :]
    tl.store(OK + offset, k, valid[:, None])
    tl.store(OV + offset, v, valid[:, None])
    if (tile == 0) & (head == 0):
        tl.store(CU, 0)
        tl.store(CU + 1, N)


@triton.jit
def _merge(PART, LSE, ACC, STATE, OUT,
           L0: tl.constexpr, L1: tl.constexpr,
           O0: tl.constexpr, O1: tl.constexpr, O2: tl.constexpr,
           FIRST: tl.constexpr, FINAL: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, 256)
    offset = token*6144 + head*256 + dim
    p = tl.load(PART + offset).to(tl.float32)
    new_lse = tl.load(LSE + head*L0 + token*L1)
    state_offset = head*2048 + token
    if FIRST:
        merged, merged_lse = p, new_lse
    else:
        old_lse = tl.load(STATE + state_offset)
        maximum = tl.maximum(old_lse, new_lse)
        a = tl.where(old_lse == -float("inf"), 0., tl.exp(old_lse - maximum))
        b = tl.where(new_lse == -float("inf"), 0., tl.exp(new_lse - maximum))
        denominator = a + b
        inverse = 1. / tl.where(denominator > 0., denominator, 1.)
        merged = (tl.load(ACC + offset)*a + p*b) * inverse
        merged_lse = tl.where(denominator > 0., maximum + tl.log(denominator), -float("inf"))
    if FINAL:
        tl.store(OUT + token*O0 + head*O1 + dim*O2, merged)
    else:
        tl.store(ACC + offset, merged)
        tl.store(STATE + state_offset, merged_lse)


class SlabWorkspace:
    def __init__(self, device):
        self.device = torch.device(device)
        self.k = torch.empty((SLAB_TOKENS, 4, 256), dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)
        self.partial = torch.empty((QUERY_CAPACITY, 24, 256), dtype=torch.bfloat16, device=device)
        self.acc = torch.empty(self.partial.shape, dtype=torch.float32, device=device)
        self.lse = torch.empty((24, QUERY_CAPACITY), dtype=torch.float32, device=device)
        self.cu_k = torch.empty(2, dtype=torch.int32, device=device)
        self.allocation_stream = torch.cuda.current_stream(device)
        self.owner_stream = None

    def claim_stream(self):
        stream = torch.cuda.current_stream(self.device)
        if self.owner_stream is None:
            stream.wait_stream(self.allocation_stream)
            self.owner_stream = stream.cuda_stream
        return self.owner_stream == stream.cuda_stream

    def run(self, query, caches, block_table, cu_q, sequence_len, scale, output):
        from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

        if query.dtype != torch.bfloat16 or query.shape[1:] != (24, 256):
            raise ValueError("FA2 slab requires BF16 queries with 24 heads of size 256")
        if output.shape != query.shape or output.dtype != query.dtype:
            raise ValueError("Output must match the query")
        if query.stride(-1) != 1 or output.stride(-1) != 1:
            raise ValueError("Query/output head dimension must be contiguous")
        plan = slabs(sequence_len, query.shape[0])
        k, v, ks, vs = caches
        if (k.dtype != torch.int8 or v.dtype != torch.int8 or k.shape != v.shape
                or k.ndim != 4 or k.shape[2:] != (4, 256)
                or ks.dtype != torch.float32 or vs.dtype != torch.float32
                or ks.shape != k.shape[:3] or vs.shape != k.shape[:3]
                or block_table.ndim != 2 or block_table.shape[0] != 1
                or block_table.stride(-1) != 1
                or block_table.shape[1]*k.shape[1] < sequence_len or cu_q.numel() != 2):
            raise ValueError("Invalid single-request INT8 paged cache")
        if any(t.device != self.device for t in (*caches, block_table, cu_q, query, output)):
            raise ValueError("Cache, query and workspace must share a device")
        if torch.cuda.is_current_stream_capturing() or not self.claim_stream():
            raise RuntimeError("FA2 slab prefill requires its owner stream outside graph capture")
        qlen = len(query)
        partial = self.partial[:qlen]
        for i, slab in enumerate(plan):
            size = slab.end - slab.start
            _gather[(triton.cdiv(size, 16), 4)](
                k, v, ks, vs, block_table, self.k, self.v, self.cu_k, slab.start, size,
                *k.stride(), *v.stride(), *ks.stride(), *vs.stride(),
                PAGE=k.shape[1], num_warps=4, num_stages=1)
            single = len(plan) == 1
            result = flash_attn_varlen_func(
                q=query, k=self.k[:size], v=self.v[:size],
                cu_seqlens_q=cu_q, cu_seqlens_k=self.cu_k,
                max_seqlen_q=qlen, max_seqlen_k=size, softmax_scale=scale,
                causal=slab.causal, fa_version=2, num_splits=1,
                out=output if single else partial, return_softmax_lse=not single)
            if not single:
                _, lse = result
                _merge[(qlen, 24)](
                    partial, lse, self.acc, self.lse, output,
                    *lse.stride(), *output.stride(), FIRST=i == 0,
                    FINAL=i == len(plan) - 1, num_warps=4)
        return output


_WORKSPACES = {}


def prepare_workspace(impl, config):
    flag = os.environ.get("PTH_FA2_SLAB_PREFILL", "0")
    if flag not in ("0", "1"):
        raise ValueError("PTH_FA2_SLAB_PREFILL must be 0 or 1")
    if flag == "0" or config is None:
        return None
    if (impl.num_heads, impl.num_kv_heads, impl.head_size, impl.window) != (24, 4, 256, 0):
        return None
    if (config.scheduler_config.max_num_seqs != 1
            or config.model_config.dtype != torch.bfloat16):
        return None
    from vllm.v1.attention.backends.fa_utils import is_fa_version_supported
    if torch.cuda.get_device_capability() != (8, 6) or not is_fa_version_supported(2):
        return None
    device = torch.device("cuda", torch.cuda.current_device())
    key = (id(config.model_config), device.index)
    if key not in _WORKSPACES:
        _WORKSPACES[key] = SlabWorkspace(device)
        from vllm.logger import init_logger
        init_logger("vllm.pth_flash_attn.fa2_slab").info(
            "FA2 slab workspace: 64 MiB K/V, %.3f MiB persistent total, 16384 KV / 2048 query tokens",
            sum(workspace_bytes().values())/2**20)
    return _WORKSPACES[key]


def try_prefill(impl, workspace, query, output, caches, metadata):
    if workspace is None or metadata.prefill is None or metadata.decode is not None:
        return False
    n = metadata.num_actual_tokens
    if (metadata.causal is not True or metadata.use_cascade
            or not 16 < n <= QUERY_CAPACITY or metadata.max_query_len != n
            or metadata.seq_lens.numel() != 1 or metadata.query_start_loc.numel() != 2
            or metadata.prefill.token_start != 0 or metadata.prefill.token_end != n
            or query.dtype != torch.bfloat16 or torch.cuda.is_current_stream_capturing()):
        return False
    if not workspace.claim_stream():
        return False
    workspace.run(query[:n], caches, metadata.block_table, metadata.query_start_loc,
                  metadata.max_seq_len, impl.scale, output[:n])
    if not getattr(workspace, "route_logged", False):
        workspace.route_logged = True
        from vllm.logger import init_logger
        init_logger("vllm.pth_flash_attn.fa2_slab").info(
            "FA2 slab prefill engaged: unchanged FA2, 64 MiB K/V, qlen=%d seq=%d slabs=%d",
            n, metadata.max_seq_len, len(slabs(metadata.max_seq_len, n)))
    return True

"""Check FlashAttention KV writes, FP8 attention and graph replay against FP32."""
from dataclasses import dataclass
import math

import torch
from fa2_fp8kv_vllm.backend import Fp8Attention, Fp8Metadata


@dataclass(frozen=True)
class Scales:
    _k_scale: torch.Tensor
    _v_scale: torch.Tensor


def check(queries, length, dim, heads, kv_heads, causal, window):
    torch.manual_seed(20260913)
    block_size = 16
    blocks = math.ceil((length + 8) / block_size)
    capacity = blocks * block_size
    query = torch.randn(queries, heads, dim, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(capacity, kv_heads, dim, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    cache = torch.empty(blocks, block_size, kv_heads, 2 * dim,
                        device="cuda", dtype=torch.float8_e4m3fn).transpose(1, 2)
    table = torch.randperm(blocks, device="cuda", dtype=torch.int64).to(torch.int32).unsqueeze(0)
    logical = torch.arange(capacity, device="cuda")
    slots = table[0, logical // block_size].to(torch.int64) * block_size + logical % block_size
    layer = Scales(torch.tensor(0.5, device="cuda"), torch.tensor(0.25, device="cuda"))
    impl = Fp8Attention(heads, dim, dim ** -0.5, kv_heads,
                        sliding_window=window, kv_cache_dtype="fp8_e4m3")
    impl.do_kv_cache_update(layer, key, value, cache, slots)
    expected_k = (key.float() / layer._k_scale).to(torch.float8_e4m3fn)
    expected_v = (value.float() / layer._v_scale).to(torch.float8_e4m3fn)
    physical = cache.transpose(1, 2).reshape(capacity, kv_heads, 2 * dim)
    assert torch.equal(physical[slots, :, :dim].view(torch.uint8), expected_k.view(torch.uint8))
    assert torch.equal(physical[slots, :, dim:].view(torch.uint8), expected_v.view(torch.uint8))
    metadata = Fp8Metadata(
        num_actual_tokens=queries, max_query_len=queries,
        query_start_loc=torch.tensor([0, queries], dtype=torch.int32, device="cuda"),
        max_seq_len=length, seq_lens=torch.tensor([length], dtype=torch.int32, device="cuda"),
        block_table=table, slot_mapping=slots[:queries], use_cascade=False,
        common_prefix_len=0, cu_prefix_query_lens=None, prefix_kv_lens=None,
        suffix_kv_lens=None, causal=causal,
        prefill_context=length if causal and queries > 64 else 0,
        maximum_context=capacity,
    )
    output = torch.empty_like(query)

    def reference(tokens):
        k = (expected_k[:tokens].float() * layer._k_scale).repeat_interleave(heads // kv_heads, dim=1)
        v = (expected_v[:tokens].float() * layer._v_scale).repeat_interleave(heads // kv_heads, dim=1)
        scores = torch.einsum("qhd,khd->hqk", query.float(), k) * impl.scale
        rows = torch.arange(queries, device="cuda")[:, None] + tokens - queries
        columns = torch.arange(tokens, device="cuda")[None, :]
        mask = torch.ones(queries, tokens, dtype=torch.bool, device="cuda")
        if causal:
            mask &= columns <= rows
        if window is not None:
            mask &= columns >= rows - window + 1
        scores.masked_fill_(~mask, float("-inf"))
        return torch.einsum("hqk,khd->qhd", scores.softmax(-1), v)

    impl.forward(layer, query, key[:queries], value[:queries], cache, metadata, output)
    expected = reference(length)
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    maximum_error = (output.float() - expected).abs().max().item()
    if queries <= 64:
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                impl.forward(layer, query, key[:queries], value[:queries], cache, metadata, output)
        torch.cuda.current_stream().wait_stream(stream)
        with torch.cuda.graph(graph):
            impl.forward(layer, query, key[:queries], value[:queries], cache, metadata, output)
        metadata.seq_lens.fill_(length + 7)
        graph.replay()
        torch.testing.assert_close(output.float(), reference(length + 7), atol=0.02, rtol=0.02)
    print(f"PASS q={queries} k={length} d={dim} h={heads}/{kv_heads} causal={causal} window={window} max_error={maximum_error}", flush=True)


for case in [(8, 37, 256, 12, 2, True, None),
             (16, 2111, 128, 32, 4, False, 2048),
             (65, 131, 256, 12, 2, True, None)]:
    check(*case)

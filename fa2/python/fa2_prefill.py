"""Causal FA2 prefill with bounded, one-pass FP8 KV unpacking."""
from dataclasses import dataclass
import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func


@dataclass(frozen=True)
class Fa2Prefill:
    chunk_tokens: int = 65536

    def forward(self, query, keys, values, table, key_scale, value_scale,
                context_length, softmax_scale, output):
        query_length, query_heads, dim = query.shape
        kv_heads = keys.shape[2]
        if query_heads % kv_heads or context_length < query_length:
            raise ValueError("Invalid causal prefill geometry")
        starts_q = torch.tensor([0, query_length], device=query.device, dtype=torch.int32)
        group = query_heads // kv_heads
        prefix_length = context_length - query_length
        chunks = [(start, min(self.chunk_tokens, prefix_length - start), False)
                  for start in range(0, prefix_length, self.chunk_tokens)]
        chunks.append((prefix_length, query_length, True))
        for head in range(kv_heads):
            q = query[:, head * group:(head + 1) * group]
            for start, length, causal in chunks:
                unpacked = torch.ops.fa2_fp8kv_prefill.gather(
                    keys, values, table, key_scale, value_scale, start, length, head)
                starts_k = torch.tensor([0, length], device=query.device, dtype=torch.int32)
                partial, lse = flash_attn_varlen_func(
                    q, unpacked[0], unpacked[1], query_length, starts_q, length,
                    cu_seqlens_k=starts_k, softmax_scale=softmax_scale,
                    causal=causal, return_softmax_lse=True, num_splits=1, fa_version=2)
                if start == 0:
                    accumulated = partial.float()
                    total_lse = lse
                else:
                    merged_lse = torch.logaddexp(total_lse, lse)
                    fraction = torch.exp(lse - merged_lse).transpose(0, 1).unsqueeze(-1)
                    accumulated = torch.lerp(accumulated, partial.float(), fraction)
                    total_lse = merged_lse
                del unpacked
            output[:, head * group:(head + 1) * group].copy_(accumulated)
        return output

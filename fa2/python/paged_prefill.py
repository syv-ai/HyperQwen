"""Bounded-query fallback using the original paged FP8 FA2 operation."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PagedPrefill:
    query_tokens: int = 2048

    def forward(self, query, keys, values, table, key_scale, value_scale,
                context_length, softmax_scale, output):
        total = query.shape[0]
        if context_length < total:
            raise ValueError("Invalid causal prefill length")
        lengths = torch.tensor([context_length], device=query.device, dtype=torch.int32)
        return self.forward_batch(query, keys, values, table, key_scale, value_scale,
            (0, total), lengths, context_length, softmax_scale, output)

    def forward_batch(self, query, keys, values, table, key_scale, value_scale,
                      query_starts, context_lengths, maximum_context, softmax_scale, output):
        if not 1 <= self.query_tokens <= 2048:
            raise ValueError("Paged prefill query chunks must contain 1..2048 tokens")
        for row, (begin, end) in enumerate(zip(query_starts, query_starts[1:])):
            for start in range(begin, end, self.query_tokens):
                count = min(self.query_tokens, end - start)
                # Keep the causal offset at the chunk's original position. GPU
                # lengths are exact even when async spec decode's CPU bound is not.
                lengths = context_lengths[row:row + 1] - (end - start - count)
                starts = torch.tensor([0, count], device=query.device, dtype=torch.int32)
                torch.ops.fa2_fp8kv.forward(query[start:start + count], keys, values,
                    output[start:start + count], starts, lengths, table[row:row + 1],
                    key_scale, value_scale, count, maximum_context,
                    True, -1, -1, softmax_scale, 1, False)
        return output

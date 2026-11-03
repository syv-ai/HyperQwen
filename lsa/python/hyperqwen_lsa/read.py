# SPDX-License-Identifier: Apache-2.0
"""One layer's LSA read for a batch: score, pick, expand, attend."""
from __future__ import annotations

import torch

from hyperqwen_lsa import core, kernels


def tail_starts(token_positions: torch.Tensor, clustered_end: torch.Tensor, window: int) -> torch.Tensor:
    """First position of each query's dense tail: its window, widened to the last clustered boundary."""
    return torch.minimum(token_positions + 1 - window, clustered_end).clamp_min(0).int()


def visible_buckets(token_positions: torch.Tensor, clustered_end: torch.Tensor) -> torch.Tensor:
    """Buckets a query may score: those of the clustered blocks closed before its own block."""
    block_floor = token_positions // core.CLUSTER_BLOCK * core.CLUSTER_BLOCK
    return (torch.minimum(block_floor, clustered_end) // core.BUCKET_SIZE).int()


def lsa_attention(query: torch.Tensor, w_q: torch.Tensor, kv_heads: int, rotary_dim: int,
                  tables: core.LayerTables, key_cache: torch.Tensor, value_cache: torch.Tensor,
                  k_scales, v_scales, output: torch.Tensor,
                  query_start_loc: torch.Tensor, block_table: torch.Tensor,
                  token_requests: torch.Tensor, token_positions: torch.Tensor,
                  clustered_end_per_token: torch.Tensor, num_reqs: int, max_query_len: int,
                  nb_max: int, scale: float, design: core.Design,
                  chunk: int = 1024, num_splits: int = 1, partials=None) -> None:
    """LSA attention for query rows [0, N); every argument is a device tensor except the sizes.

    `chunk` bounds the [rows, nb_max] score matrix and the [rows, top_p * 16] expansion; a decode
    batch is one chunk, so the call is capturable in a CUDA graph.
    """
    n = query.shape[0]
    block_size = key_cache.shape[1]
    qidx = core.index_queries(query, w_q, kv_heads, rotary_dim)
    tails = tail_starts(token_positions, clustered_end_per_token, design.window)
    visible = visible_buckets(token_positions, clustered_end_per_token)
    width = design.sink + design.top_k + design.window + core.CLUSTER_BLOCK
    for t0 in range(0, n, chunk):
        t1 = min(n, t0 + chunk)
        scores = kernels.score_buckets(qidx, query_start_loc, visible, block_table, tables.centroids,
                                       t0, t1, nb_max, block_size, num_reqs, max_query_len)
        best, picked = kernels.top_p(scores, visible, t0, design.top_p, parts=32)
        slots, lens = kernels.expand_slots(picked, best, tables.members, block_table,
                                                    token_requests[t0:t1], token_positions[t0:t1],
                                                    tails[t0:t1], design.top_k, design.sink, block_size,
                                                    width, cb=64)
        kernels.slot_attention(query[t0:t1], key_cache, value_cache, k_scales, v_scales, output[t0:t1],
                               slots, lens, scale, num_splits=num_splits, block_n=16, num_warps=2,
                               num_stages=2)

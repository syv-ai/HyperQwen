# SPDX-License-Identifier: Apache-2.0
"""D-LSA (flat) for a paged KV cache: index keys, Ward buckets, selection.

A port of the flat design of Latent Sparse Attention (Kubicek et al., arXiv 2610.42147,
github.com/Prefrontal-AI/latent-sparse-attention at the commit in ../../PIN) onto vLLM's paged cache.
`lsa.core.reference` and `lsa.core.ward` are the specification; tests/check_core.py holds this file
to them.

What a layer keeps, all paged by the KV cache's own slot ids, so the allocator, prefix caching and
preemption need no new bookkeeping:

- index keys, one per slot: normalize(k[:, :, rot:] flattened @ W_K)          [slots + 1, 64] bf16
- bucket rows, owned by the KV page containing a cluster block's final key. A page reserves
  64 rows per cluster that can end in it. A shared prefix page must never hold buckets that
  depend on a different request's private suffix -- the centroid and block-local members.

The extra row of each table is a sink for writes that must go nowhere (padding, masked blocks), so
nothing here needs a host sync to decide whether to write.

A query at position p reads: the sinks [0, 4), the window [p - 511, p] widened down to the last
clustered block boundary (keys in no bucket yet are always read), and up to top_k keys of the
top_p buckets it scores best among the cluster blocks closed before its own. That is the reference
read, except that the widening makes it a superset whenever a block has closed but not yet been
clustered (at most one decode step, see backend.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# The paper's flat design, the defaults of lsa.sglang.settings.design.IndexerConfig, which the
# Qwen3.8-27B lsa_flat configurations do not override.
POOL_SIZE = 4
BUCKET_SIZE = 8
BUCKET_CAP = 16
WINDOW = 512            # also the cluster block size
SINK = 4
TOP_K = 4096
TOP_P = 1024
INDEX_DIM = 64
CLUSTER_BLOCK = WINDOW
POOLS_PER_BLOCK = CLUSTER_BLOCK // POOL_SIZE        # 128
BUCKETS_PER_BLOCK = CLUSTER_BLOCK // BUCKET_SIZE    # 64
CAP_POOLS = BUCKET_CAP // POOL_SIZE                 # 4


@dataclass
class Design:
    top_k: int = TOP_K
    top_p: int = TOP_P
    sink: int = SINK
    window: int = WINDOW


@dataclass
class LayerTables:
    """One layer's index, paged by KV slot. The last row of each tensor is the write sink."""

    index_keys: torch.Tensor   # [slots + 1, 64] bf16
    centroids: torch.Tensor    # [pages * bucket_rows_per_page(block_size) + 1, 64] bf16
    members: torch.Tensor      # [same rows, 16] int16, block-local positions, -1 padded

    @classmethod
    def allocate(cls, num_slots: int, device: torch.device, block_size: int) -> "LayerTables":
        assert block_size > 0 and block_size % BUCKET_SIZE == 0
        assert num_slots % block_size == 0
        rows = num_slots // block_size * bucket_rows_per_page(block_size)
        return cls(
            torch.zeros(num_slots + 1, INDEX_DIM, dtype=torch.bfloat16, device=device),
            torch.zeros(rows + 1, INDEX_DIM, dtype=torch.bfloat16, device=device),
            torch.full((rows + 1, BUCKET_CAP), -1, dtype=torch.int16, device=device),
        )

    @staticmethod
    def bytes_per_slot(block_size: int) -> float:
        return INDEX_DIM * 2 + (INDEX_DIM * 2 + BUCKET_CAP * 2) * bucket_rows_per_page(block_size) / block_size

    @property
    def key_sink(self) -> int:
        return self.index_keys.shape[0] - 1

    @property
    def row_sink(self) -> int:
        return self.centroids.shape[0] - 1


# ---------------------------------------------------------------------------------------------
# Index vectors


def index_keys(key: torch.Tensor, w_k: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """normalize(concat_h k_h[rot:] @ W_K) in fp32, stored bf16 -- [T, 64]."""
    t = key.shape[0]
    flat = key[:, :, rotary_dim:].reshape(t, -1).float()
    return F.normalize(flat @ w_k, dim=-1).to(torch.bfloat16)


def index_queries(query: torch.Tensor, w_q: torch.Tensor, kv_heads: int, rotary_dim: int) -> torch.Tensor:
    """normalize(concat_g sum_{h in g} q_h[rot:] @ W_Q) in fp32 -- [T, 64].

    Query head h reads key head h // G, so each group's query heads are summed into the slice of
    the key head they read (lsa.sglang.models.qwen3_5.index_space.key_and_query).
    """
    t, heads, dim = query.shape
    group = heads // kv_heads
    grouped = query.view(t, kv_heads, group, dim)[..., rotary_dim:].sum(2, dtype=torch.float32)
    return F.normalize(grouped.reshape(t, -1) @ w_q, dim=-1)


def write_index_keys(tables: LayerTables, keys: torch.Tensor, slot_mapping: torch.Tensor) -> None:
    n = slot_mapping.shape[0]
    slots = slot_mapping.long()
    slots = torch.where(slots < 0, tables.key_sink, slots)
    tables.index_keys.index_copy_(0, slots, keys[:n])


# ---------------------------------------------------------------------------------------------
# Ward clustering of a cluster block's pools (lsa.core.ward.ward_rounds), without host syncs


def ward_labels(pool_keys: torch.Tensor, bucket_count: int, cap_pools: int,
                max_rounds: int = 128, rounds_per_check: int = 8) -> torch.Tensor:
    """Each pool's bucket, numbered from 0 within its block in the order of its first pool.

    Every pool holds the same number of keys, so sizes are counted in pools and the cap is
    `bucket_cap / pool_size` pools. Rounds of mutual-cheapest merges, the cheapest pairs first in
    the round that reaches the count; a block that has reached its count stops merging, so running
    extra rounds is a no-op and the loop only checks for completion every `rounds_per_check`.

    Args:
        pool_keys: [G, P, d] fp32.
    Returns:
        [G, P] int64 labels.
    """
    blocks, pools, dim = pool_keys.shape
    device = pool_keys.device
    key_sums = pool_keys.clone()
    sizes = torch.ones(blocks, pools, device=device)
    alive = torch.ones(blocks, pools, dtype=torch.bool, device=device)
    index = torch.arange(pools, device=device).expand(blocks, pools)
    parent = index.clone()
    off_diagonal = ~torch.eye(pools, dtype=torch.bool, device=device)
    trash = torch.full((blocks, 1), 0, dtype=torch.long, device=device)
    rounds = 0
    while rounds < max_rounds:
        for _ in range(rounds_per_check):
            merges_left = alive.sum(1) - bucket_count
            gram = torch.bmm(key_sums, key_sums.transpose(1, 2))
            squares = gram.diagonal(dim1=1, dim2=2)
            merged = (squares[:, :, None] + squares[:, None, :] + 2 * gram).clamp_min(0).sqrt()
            alignments = squares.clamp_min(0).sqrt()
            costs = alignments[:, :, None] + alignments[:, None, :] - merged
            allowed = alive[:, :, None] & alive[:, None, :] & off_diagonal
            allowed &= sizes[:, :, None] + sizes[:, None, :] <= cap_pools
            best_costs, best = costs.masked_fill(~allowed, float("inf")).min(2)
            mutual = (best_costs < float("inf")) & (best.gather(1, best) == index) & (index < best)
            rank = best_costs.masked_fill(~mutual, float("inf")).argsort(1).argsort(1)
            mutual &= rank < merges_left.clamp_min(0)[:, None]
            m = mutual.float()
            key_sums = key_sums + m[:, :, None] * key_sums.gather(1, best[:, :, None].expand(-1, -1, dim))
            sizes = sizes + m * sizes.gather(1, best)
            # Absorbed pools: the partners of the kept side. Non-merging rows write to a trash column.
            target = torch.where(mutual, best, pools)
            absorbed = torch.zeros(blocks, pools + 1, dtype=torch.bool, device=device)
            absorbed.scatter_(1, target, True)
            alive &= ~absorbed[:, :pools]
            parent = torch.cat([parent, trash], 1).scatter_(1, target, index)[:, :pools]
        rounds += rounds_per_check
        if bool((alive.sum(1) <= bucket_count).all()):
            break
    else:
        raise RuntimeError(f"Ward did not reach {bucket_count} buckets in {max_rounds} rounds")
    for _ in range(max(1, (pools - 1).bit_length()) + 1):
        parent = parent.gather(1, parent)
    number = torch.cumsum(alive.long(), 1) - 1
    return number.gather(1, parent)


def bucket_tables(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Ward-bucket closed cluster blocks (lsa.core.reference.build_buckets, per block).

    Args:
        keys: [G, 512, 64] fp32 unit index keys, one cluster block a row.
    Returns:
        centroids [G, 64, 64] fp32 and members [G, 64, 16] int64 (block-local positions in
        position order, -1 padded).
    """
    blocks = keys.shape[0]
    device = keys.device
    pools = keys.view(blocks, POOLS_PER_BLOCK, POOL_SIZE, INDEX_DIM).sum(2)
    labels = ward_labels(pools, BUCKETS_PER_BLOCK, CAP_POOLS)
    key_labels = labels.repeat_interleave(POOL_SIZE, dim=1)                        # [G, 512]
    sums = torch.zeros(blocks, BUCKETS_PER_BLOCK, INDEX_DIM, device=device)
    sums.scatter_add_(1, key_labels[:, :, None].expand(-1, -1, INDEX_DIM), keys)
    centroids = sums / sums.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    positions = torch.arange(CLUSTER_BLOCK, device=device)
    order = torch.argsort(key_labels * CLUSTER_BLOCK + positions, dim=1)           # by (bucket, position)
    sorted_labels = key_labels.gather(1, order)
    counts = torch.zeros(blocks, BUCKETS_PER_BLOCK, dtype=torch.long, device=device)
    counts.scatter_add_(1, key_labels, torch.ones_like(key_labels))
    starts = torch.cumsum(counts, 1) - counts
    rank = positions[None, :] - starts.gather(1, sorted_labels)
    members = torch.full((blocks, BUCKETS_PER_BLOCK * BUCKET_CAP), -1, dtype=torch.long, device=device)
    members.scatter_(1, sorted_labels * BUCKET_CAP + rank, order)
    return centroids, members.view(blocks, BUCKETS_PER_BLOCK, BUCKET_CAP)


def slots_of(block_table: torch.Tensor, requests: torch.Tensor, positions: torch.Tensor,
             block_size: int) -> torch.Tensor:
    """KV slot of each (request, position); `requests` broadcasts against `positions`."""
    blocks = block_table[requests, positions // block_size].long()
    return blocks * block_size + positions % block_size


def bucket_rows_per_page(block_size: int) -> int:
    return (block_size + CLUSTER_BLOCK - 1) // CLUSTER_BLOCK * BUCKETS_PER_BLOCK


def bucket_rows(block_table: torch.Tensor, requests: torch.Tensor, bucket_positions: torch.Tensor,
                block_size: int, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Address buckets through their cluster's last key, never a partial shared prefix.

    `valid` masks padding buckets before indexing the block table: their otherwise
    arbitrary positions may name a cluster beyond the request's cache capacity.
    """
    last = (bucket_positions // CLUSTER_BLOCK + 1) * CLUSTER_BLOCK - 1
    if valid is not None:
        last = torch.where(valid, last, 0)
    pages = block_table[requests, last // block_size].long()
    cluster_in_page = last.remainder(block_size) // CLUSTER_BLOCK
    bucket_in_cluster = bucket_positions.remainder(CLUSTER_BLOCK) // BUCKET_SIZE
    return pages * bucket_rows_per_page(block_size) + cluster_in_page * BUCKETS_PER_BLOCK + bucket_in_cluster


def cluster_blocks(tables: LayerTables, block_table: torch.Tensor, requests: torch.Tensor,
                   starts: torch.Tensor, valid: torch.Tensor, block_size: int) -> None:
    """Bucket the cluster blocks [starts, starts + 512) of `requests` and write their rows.

    Masked-off blocks (`valid` False) are computed on harmless slots and written to the sink row,
    so the call needs no host sync to skip them.
    """
    if requests.numel() == 0:
        return
    offsets = torch.arange(CLUSTER_BLOCK, device=starts.device)
    starts = torch.where(valid, starts, 0)
    slots = slots_of(block_table, requests[:, None], starts[:, None] + offsets[None, :], block_size)
    keys = tables.index_keys[slots].float()
    keys = torch.where(valid[:, None, None], keys, F.normalize(torch.ones_like(keys), dim=-1))
    centroids, members = bucket_tables(keys)
    rows = bucket_rows(block_table, requests[:, None], starts[:, None] + offsets[None, ::BUCKET_SIZE],
                       block_size, valid[:, None])                                  # [G, 64]
    rows = torch.where(valid[:, None], rows, tables.row_sink).flatten()
    tables.centroids.index_copy_(0, rows, centroids.reshape(-1, INDEX_DIM).to(torch.bfloat16))
    tables.members.index_copy_(0, rows, members.reshape(-1, BUCKET_CAP).to(torch.int16))


# ---------------------------------------------------------------------------------------------
# Selection


def expand(tables: LayerTables, block_table: torch.Tensor, token_requests: torch.Tensor,
           token_positions: torch.Tensor, tail_starts: torch.Tensor, scores: torch.Tensor,
           block_size: int, design: Design) -> tuple[torch.Tensor, torch.Tensor]:
    """Pick each query's top_p buckets and read them as keys, best bucket first, up to top_k.

    lsa.core.reference.expand_buckets: a key in the sinks or at or past the tail start is not
    counted (the read takes those anyway), and the last bucket is cut short.

    Args:
        scores: [n, nb] fp32, -inf for buckets a query may not see. Column b is the bucket stored
            at position 8b.
    Returns:
        selected positions [n, top_k] int32 (padding unspecified past the count) and counts [n].
    """
    n, nb = scores.shape
    top_p = min(design.top_p, nb)
    best_scores, picked = torch.topk(scores, top_p, dim=1, sorted=True)            # [n, p]
    bucket_positions = picked * BUCKET_SIZE
    rows = bucket_rows(block_table, token_requests[:, None], bucket_positions, block_size,
                       torch.isfinite(best_scores))
    local = tables.members[rows].long()                                              # [n, p, 16]
    base = (bucket_positions // CLUSTER_BLOCK) * CLUSTER_BLOCK
    positions = base[:, :, None] + local
    keep = (local >= 0) & torch.isfinite(best_scores)[:, :, None]
    keep &= (positions >= design.sink) & (positions < tail_starts[:, None, None])
    keep = keep.view(n, -1)
    positions = positions.view(n, -1)
    rank = torch.cumsum(keep.int(), 1)
    keep &= rank <= design.top_k
    destination = torch.where(keep, rank - 1, design.top_k).long()
    selected = torch.zeros(n, design.top_k + 1, dtype=torch.int32, device=scores.device)
    selected.scatter_(1, destination, positions.int())
    counts = rank[:, -1].clamp_max(design.top_k).int()
    return selected[:, : design.top_k], counts

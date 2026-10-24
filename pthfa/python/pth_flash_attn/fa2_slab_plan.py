"""CPU-only bounds for attention over a fixed-size dequantization workspace."""
from dataclasses import dataclass

SLAB_TOKENS = 16384
QUERY_CAPACITY = 2048


@dataclass(frozen=True)
class Slab:
    start: int
    end: int
    causal: bool


def slabs(sequence_len: int, query_len: int) -> tuple[Slab, ...]:
    """Cover all keys; only the final slab intersects the causal boundary.

    Balanced slabs avoid very short FA2 calls at a slab boundary. For a
    multi-slab call each slab has at least 8192 keys, exceeding query capacity.
    All earlier slabs are therefore visible to every query. FA2's bottom-right
    causal alignment is applied only to the final slab, which ends at the
    complete sequence length.
    """
    if not 1 <= query_len <= min(sequence_len, QUERY_CAPACITY):
        raise ValueError("Query length must fit the sequence and 2048-token workspace")
    count = (sequence_len + SLAB_TOKENS - 1) // SLAB_TOKENS
    size, extra = divmod(sequence_len, count)
    result = []
    start = 0
    for i in range(count):
        end = start + size + (i < extra)
        result.append(Slab(start, end, i == count - 1))
        start = end
    return tuple(result)


def workspace_bytes() -> dict[str, int]:
    return dict(
        kv=SLAB_TOKENS * 4 * 256 * 2 * 2,
        partial=QUERY_CAPACITY * 24 * 256 * 2,
        accumulator=QUERY_CAPACITY * 24 * 256 * 4,
        lse=24 * QUERY_CAPACITY * 4,
        cu_k=2 * 4,
    )

# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend `CUSTOM` that serves D-LSA (flat) on the full-attention layers.

Selected with `--attention-backend CUSTOM` once `register()` has run with LSA=1. It is vLLM's Triton
backend with three changes, so the cache spec, page layout, scale views (int8_per_token_head),
writer and dense read are the code vLLM itself runs:

* `do_kv_cache_update` also writes each token's index key, into a table paged by the same slots.
* The metadata builder adds each token's request and position and each request's clustered end
  (device tensors in persistent buffers, so a FULL decode graph replays them), and buckets the
  cluster blocks that closed during the previous decode step.
* `forward` of a layer the map covers reads its keys through the LSA selection (read.py) instead of
  all of them. A layer the map does not cover -- the MTP drafter's -- runs the Triton read.

When blocks are clustered. A block is bucketed as soon as all of its 512 keys are known to be
final: a prefill chunk buckets the blocks it closes in its own forward, before its queries select;
a block closed by decode tokens, which may be rejected drafts, is bucketed by the next step's
builder, once they have been accepted. Until then the query's dense tail reaches down to the last
bucketed boundary, so nothing is missed in the meantime -- the read is a superset of the paper's.
Bucket rows live in KV slots, so prefix caching and preemption carry them with the KV pages; the
prefill rule also re-buckets the block a prefix hit ends in, which covers a block whose decode
closure never reached a next step.

Dense where the design is dense. A query whose context fits the budget (sink + top_k + window) reads
every key under LSA too, so an eager batch whose longest sequence fits it takes the Triton read,
exactly as the paper's server is exact there. Decode always takes the LSA read, so a captured graph
has one path.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import KVQuantMode

from hyperqwen_lsa import core, read, settings

logger = init_logger(f"vllm.{__name__}")


def _bases():
    """The backend LSA extends: pthfa's int8 kernel when installed (its dense read is the fast one on
    sm8x, for the drafter's layer, short prefills and the dense fallback), vLLM's Triton backend
    otherwise. Both keep Triton's cache spec, layout and writer, which is all the LSA read relies on.
    LSA_BASE=triton forces the latter."""
    if settings.base() != "triton":
        try:
            from pth_flash_attn import vllm_backend as pth
            return (pth.PthFlashAttnBackend, pth.PthFlashAttnMetadataBuilder, pth.PthFlashAttnImpl,
                    pth.PthAttnMetadata, "pth_flash_attn")
        except ImportError:
            if settings.base() == "pthfa":
                raise
    return (TritonAttentionBackend, TritonAttentionMetadataBuilder, TritonAttentionImpl,
            TritonAttentionMetadata, "triton")


BaseBackend, BaseBuilder, BaseImpl, BaseMetadata, BASE_NAME = _bases()

LAYER_INDEX = re.compile(r"layers\.(\d+)\.")


class Registry:
    """The process's LSA layers: projections by decoder index, and each layer's tables by name."""

    maps: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None
    tables: dict[str, core.LayerTables] = {}
    slots: dict[str, int] = {}
    announced: int | bool = False

    @classmethod
    def load_maps(cls, device: torch.device) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if cls.maps is None:
            blob = torch.load(settings.map_path(), map_location="cpu", weights_only=True)
            maps: dict[int, dict[str, torch.Tensor]] = {}
            for name, tensor in blob.items():
                layer, _, kind = name.partition(".")
                maps.setdefault(int(layer), {})[kind] = tensor.float()
            cls.maps = {layer: (pair["W_Q_idx"].to(device), pair["W_K_idx"].to(device))
                        for layer, pair in maps.items()}
        return cls.maps


def layer_index(layer_name: str) -> int | None:
    match = LAYER_INDEX.search(layer_name)
    return int(match.group(1)) if match else None


def rotary_dim_of(vllm_config: VllmConfig, head_size: int) -> int:
    config = vllm_config.model_config.hf_text_config
    factor = getattr(config, "partial_rotary_factor", None)
    if factor is None:
        factor = (getattr(config, "rope_parameters", None) or {}).get("partial_rotary_factor")
    if factor is None or not 0 < float(factor) < 1:
        raise NotImplementedError(
            f"LSA indexes the position-free dims of a partially rotated head; this model's "
            f"partial_rotary_factor is {factor}")
    return int(head_size * float(factor))


def decode_query_len(vllm_config: VllmConfig) -> int:
    spec = vllm_config.speculative_config
    return 1 + (int(spec.num_speculative_tokens) if spec and spec.num_speculative_tokens else 0)


@dataclass
class LsaMetadata(BaseMetadata):
    token_requests: torch.Tensor | None = None       # [N] int64
    token_positions: torch.Tensor | None = None      # [N] int64
    token_clustered_end: torch.Tensor | None = None  # [N] int64
    num_reqs: int = 0
    decode_only: bool = True
    # Prefill blocks to bucket in each layer's forward: request rows, block starts, validity
    prefill_blocks: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None


class LsaMetadataBuilder(BaseBuilder):
    # Decode batches replay FULL graphs; a batch with a prefill runs eager (PIECEWISE).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.decode_q = decode_query_len(vllm_config)
        tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
        seqs = int(vllm_config.scheduler_config.max_num_seqs)
        self._token_requests = torch.zeros(tokens, dtype=torch.long, device=device)
        self._token_positions = torch.zeros(tokens, dtype=torch.long, device=device)
        self._token_clustered_end = torch.zeros(tokens, dtype=torch.long, device=device)
        self._arange = torch.arange(tokens, dtype=torch.long, device=device)
        self._capturing = False
        del seqs

    def build_for_cudagraph_capture(self, common_attn_metadata):
        self._capturing = True
        try:
            return super().build_for_cudagraph_capture(common_attn_metadata)
        finally:
            self._capturing = False

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False) -> LsaMetadata:
        base = super().build(common_prefix_len, common_attn_metadata, fast_build)
        cam = common_attn_metadata
        n = cam.num_actual_tokens
        r = cam.num_reqs
        qsl = cam.query_start_loc[: r + 1].long()
        q_len = qsl[1:] - qsl[:-1]
        seq_lens = cam.seq_lens[:r].long()
        ctx = seq_lens - q_len
        tokens = self._arange[:n]
        req = torch.searchsorted(qsl[1:], tokens, right=True).clamp_max(max(r - 1, 0))
        pos = (ctx[req] + tokens - qsl[req]).clamp_min(0)
        clusterable = torch.where(q_len > self.decode_q, seq_lens, ctx)
        clustered_end = clusterable // core.CLUSTER_BLOCK * core.CLUSTER_BLOCK
        self._token_requests[:n].copy_(req)
        self._token_positions[:n].copy_(pos)
        self._token_clustered_end[:n].copy_(clustered_end[req])
        decode_only = cam.max_query_len <= self.decode_q
        metadata = LsaMetadata(**vars(base), token_requests=self._token_requests[:n],
                               token_positions=self._token_positions[:n],
                               token_clustered_end=self._token_clustered_end[:n],
                               num_reqs=r, decode_only=decode_only)
        if self._capturing:
            return metadata
        q_len_cpu = (cam.query_start_loc_cpu[1: r + 1] - cam.query_start_loc_cpu[:r]).tolist()
        upper = cam.seq_lens_cpu_upper_bound
        seq_cpu = (upper[:r] if upper is not None else cam.seq_lens[:r].cpu()).tolist()
        self._cluster_decode_closures(cam, ctx, q_len_cpu, seq_cpu)
        if not decode_only:
            metadata.prefill_blocks = self._prefill_blocks(ctx, seq_lens, q_len_cpu, seq_cpu)
        return metadata

    def _lsa_tables(self) -> list[core.LayerTables]:
        return [Registry.tables[name] for name in self.layer_names if name in Registry.tables]

    def _cluster_decode_closures(self, cam, ctx, q_len_cpu, seq_cpu) -> None:
        """Bucket the blocks that the previous step's accepted decode tokens closed."""
        tables = self._lsa_tables()
        if not tables:
            return
        reach = 2 * self.decode_q
        rows, starts = [], []
        for row, (q, s) in enumerate(zip(q_len_cpu, seq_cpu)):
            upper_ctx = s - q
            end = upper_ctx // core.CLUSTER_BLOCK * core.CLUSTER_BLOCK
            if end >= core.CLUSTER_BLOCK and end > upper_ctx - reach:
                rows.append(row)
                starts.append(end - core.CLUSTER_BLOCK)
        if not rows:
            return
        device = ctx.device
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        starts_t = torch.tensor(starts, dtype=torch.long, device=device)
        ends = starts_t + core.CLUSTER_BLOCK
        c = ctx[rows_t]
        valid = (ends <= c) & (ends > c - self.decode_q)
        for table in tables:
            core.cluster_blocks(table, cam.block_table_tensor, rows_t, starts_t, valid, self._block_size())

    def _prefill_blocks(self, ctx, seq_lens, q_len_cpu, seq_cpu):
        rows, starts = [], []
        for row, (q, s) in enumerate(zip(q_len_cpu, seq_cpu)):
            if q <= self.decode_q:
                continue
            first = max(0, (s - q - self.decode_q) // core.CLUSTER_BLOCK)
            for block in range(first, s // core.CLUSTER_BLOCK):
                rows.append(row)
                starts.append(block * core.CLUSTER_BLOCK)
        device = ctx.device
        rows_t = torch.tensor(rows, dtype=torch.long, device=device)
        starts_t = torch.tensor(starts, dtype=torch.long, device=device)
        ends = starts_t + core.CLUSTER_BLOCK
        valid = (ends <= seq_lens[rows_t]) & (ends > ctx[rows_t] - self.decode_q)
        return rows_t, starts_t, valid

    def _block_size(self) -> int:
        return int(self.kv_cache_spec.block_size)


class LsaAttentionImpl(BaseImpl):
    _lsa_ready = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from vllm.config import get_current_vllm_config
        # Only the constructor runs inside set_current_vllm_config; forward does not.
        self._lsa_vllm_config = get_current_vllm_config()

    def _lsa_setup(self, layer, kv_cache: torch.Tensor) -> core.LayerTables | None:
        """The layer's tables, sized to the cache it is handed; None if the layer is not indexed.

        vLLM hands a layer more than one cache in its life (the CUDA-graph memory profile runs on a
        small one before the real pool is allocated), so the tables follow the cache's storage: a
        new storage, a new set of tables. Allocation happens before any graph is captured on the
        real cache, so a graph only ever holds the final tables.
        """
        if self._lsa_ready is None:          # not indexed
            return None
        if kv_cache.numel() == 0:
            return None
        key = (kv_cache.untyped_storage().data_ptr(), kv_cache.shape[0] * kv_cache.shape[2])
        if self._lsa_ready and self._lsa_cache_key == key:
            return self._lsa_tables
        name = layer.layer_name
        vllm_config = self._lsa_vllm_config
        if not self._lsa_ready:
            index = layer_index(name)
            maps = Registry.load_maps(kv_cache.device)
            if index is None or index not in maps or name.startswith("mtp."):
                logger.info("LSA: %s is not in the map, it keeps the dense Triton read", name)
                self._lsa_ready = None
                return None
            self.lsa_w_q, self.lsa_w_k = maps[index]
            self.lsa_rotary = rotary_dim_of(vllm_config, self.head_size)
            key_dim = self.num_kv_heads * (self.head_size - self.lsa_rotary)
            if tuple(self.lsa_w_k.shape) != (key_dim, core.INDEX_DIM):
                raise ValueError(f"LSA map for layer {index} is {tuple(self.lsa_w_k.shape)}, expected "
                                 f"[{key_dim}, {core.INDEX_DIM}]")
            if self._kv_quant_mode not in (KVQuantMode.NONE, KVQuantMode.INT8_PER_TOKEN_HEAD):
                raise NotImplementedError(f"LSA reads bf16 or int8_per_token_head KV, not {self.kv_cache_dtype}")
            self.lsa_design = settings.design()
            self.lsa_dense_below = settings.dense_below(self.lsa_design)
            self.lsa_decode_q = decode_query_len(vllm_config)
            self.lsa_nb_max = vllm_config.model_config.max_model_len // core.BUCKET_SIZE + 1
            self.lsa_splits = settings.decode_splits()
        num_slots = key[1]
        Registry.tables.pop(name, None)
        self._lsa_tables = None
        tables = core.LayerTables.allocate(num_slots, kv_cache.device)
        Registry.tables[name] = tables
        self._lsa_tables, self._lsa_cache_key, self._lsa_ready = tables, key, True
        Registry.slots[name] = num_slots
        if Registry.slots and len(set(Registry.slots.values())) == 1 and Registry.announced != num_slots:
            Registry.announced = num_slots
            logger.info(
                "LSA armed: D-LSA flat on %s, map %s, top_k %d top_p %d window %d sink %d, dense below %d "
                "tokens; index tables %.1f MiB a layer (%d slots)", BASE_NAME, settings.map_path(),
                self.lsa_design.top_k, self.lsa_design.top_p, self.lsa_design.window,
                self.lsa_design.sink, self.lsa_dense_below,
                num_slots * core.LayerTables.bytes_per_slot() / 2**20, num_slots)
        return tables

    def fused_rope_kvcache_supported(self):
        return False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        tables = self._lsa_setup(layer, kv_cache)
        if tables is not None:
            n = slot_mapping.shape[0]
            core.write_index_keys(tables, core.index_keys(key[:n], self.lsa_w_k, self.lsa_rotary), slot_mapping)

    def _lsa_caches(self, kv_cache):
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            hs = self.head_size
            return key_cache[..., :hs], value_cache[..., :hs], self._k_scale_cache, self._v_scale_cache
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        return key_cache, value_cache, None, None

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                output_scale=None, output_block_scale=None):
        if attn_metadata is None:
            return output.fill_(0)
        tables = self._lsa_setup(layer, kv_cache)
        if tables is None or not isinstance(attn_metadata, LsaMetadata) or output_scale is not None:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                                   output_scale, output_block_scale)
        m = attn_metadata
        n = m.num_actual_tokens
        block_size = kv_cache.shape[2]
        decode = m.decode_only
        if not decode and m.prefill_blocks is not None and m.prefill_blocks[0].numel():
            rows, starts, valid = m.prefill_blocks
            core.cluster_blocks(tables, m.block_table, rows, starts, valid, block_size)
        if not decode and m.max_seq_len <= self.lsa_dense_below:
            return super().forward(layer, query, key, value, kv_cache, attn_metadata, output,
                                   output_scale, output_block_scale)
        key_cache, value_cache, k_scales, v_scales = self._lsa_caches(kv_cache)
        out = output[:n].view(n, self.num_heads, self.head_size)
        read.lsa_attention(
            query[:n].view(n, self.num_heads, self.head_size), self.lsa_w_q, self.num_kv_heads,
            self.lsa_rotary, tables, key_cache, value_cache, k_scales, v_scales, out,
            m.query_start_loc, m.block_table, m.token_requests, m.token_positions,
            m.token_clustered_end, m.num_reqs, m.max_query_len, self.lsa_nb_max, self.scale,
            self.lsa_design, chunk=n if decode else settings.prefill_chunk(),
            num_splits=self.lsa_splits if decode else 1)
        return output


class LsaAttentionBackend(BaseBackend):
    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls():
        return LsaAttentionImpl

    @staticmethod
    def get_builder_cls():
        return LsaMetadataBuilder


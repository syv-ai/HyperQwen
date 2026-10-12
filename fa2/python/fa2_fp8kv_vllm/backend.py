# SPDX-License-Identifier: Apache-2.0
"""FP8 E4M3 extension using FlashAttention metadata and KV-cache storage."""
from dataclasses import dataclass
import os
from typing import ClassVar

import torch
from fa2_prefill import Fa2Prefill
from paged_prefill import PagedPrefill
from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionImpl, AttentionType
from vllm.v1.attention.backends import flash_attn as native
from vllm.v1.kv_cache_layout import KVCacheLayout


def extension_selected(cache_dtype):
    return (current_platform.get_device_capability() in ((8, 6), (8, 9), (12, 0))
            and cache_dtype in ("fp8", "fp8_e4m3"))


if current_platform.get_device_capability() in ((8, 6), (8, 9), (12, 0)):
    torch.ops.load_library(os.environ["FA2_FP8KV_LIBRARY"])
    torch.ops.load_library(os.environ["FA2_FP8KV_PREFILL_LIBRARY"])


@dataclass
class Fp8Metadata(native.FlashAttentionMetadata):
    prefill_context: int = 0
    maximum_context: int = 0
    prefill_query_starts: tuple[int, ...] = ()


class Fp8MetadataBuilder(native.FlashAttentionMetadataBuilder):
    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        common = common_attn_metadata
        if self.dcp_world_size != 1 or not isinstance(common.causal, bool):
            raise NotImplementedError("FP8 FA2 requires DCP=1 and uniform causality")
        metadata = super().build(0, common, fast_build)
        context = 0
        if (common.causal and common.max_query_len > 64 and common.num_reqs == 1
                and common.seq_lens_cpu_upper_bound is not None):
            context = int(common.seq_lens_cpu_upper_bound[0].item())
        query_starts = ()
        if common.causal and common.max_query_len > 2048:
            query_starts = tuple(common.query_start_loc_cpu[:common.num_reqs + 1].tolist())
        return Fp8Metadata(**vars(metadata), prefill_context=context,
                           maximum_context=self.model_config.max_model_len,
                           prefill_query_starts=query_starts)

    def use_cascade_attention(self, *args, **kwargs):
        return False


# vLLM assigns distributed ranks in AttentionImplBase.__new__; the framework
# requires a mutable instance during that initialization.
@dataclass
class Fp8Attention(AttentionImpl[Fp8Metadata]):
    num_heads: int
    head_size: int
    scale: float
    num_kv_heads: int | None = None
    alibi_slopes: list[float] | None = None
    sliding_window: int | None = None
    kv_cache_dtype: str = "auto"
    logits_soft_cap: float | None = None
    attn_type: str = AttentionType.DECODER
    kv_sharing_target_layer_name: str | None = None
    sinks: torch.Tensor | None = None
    supports_dcp: ClassVar[bool] = False
    supports_quant_query_input: ClassVar[bool] = False

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        # The stock FlashAttention layout is (B, H, N, 2D).
        keys, values = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        native.reshape_and_cache_flash(key, value, keys, values, slot_mapping,
                                       self.kv_cache_dtype, layer._k_scale, layer._v_scale)

    def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                output, output_scale=None, output_block_scale=None):
        if attn_metadata is None or attn_metadata.num_actual_tokens == 0:
            return output.fill_(0)
        window_left = self.sliding_window - 1 if self.sliding_window is not None else -1
        kv_heads = self.num_kv_heads or self.num_heads
        if not (extension_selected(self.kv_cache_dtype)
                and self.dcp_world_size == 1 and self.pcp_world_size == 1
                and self.head_size in (128, 256) and query.dtype == torch.bfloat16
                and window_left in (-1, 2047) and not self.logits_soft_cap
                and self.alibi_slopes is None and self.sinks is None
                and self.attn_type == AttentionType.DECODER
                and kv_cache.dtype in (torch.uint8, torch.float8_e4m3fn)):
            raise NotImplementedError("Unsupported FP8 FA2 attention configuration")
        assert output_scale is None and output_block_scale is None
        assert isinstance(attn_metadata, Fp8Metadata)
        count = attn_metadata.num_actual_tokens
        keys, values = kv_cache.view(torch.float8_e4m3fn).transpose(1, 2).split(self.head_size, dim=-1)
        max_query = attn_metadata.max_query_len
        if (attn_metadata.prefill_context >= max_query > 64
                and attn_metadata.causal and window_left == -1
                and not torch.cuda.is_current_stream_capturing()):
            try:
                Fa2Prefill().forward(query[:count], keys, values,
                    attn_metadata.block_table, layer._k_scale, layer._v_scale,
                    attn_metadata.prefill_context, self.scale, output[:count])
                native.logger.info_once("FP8 FA2: bounded BF16 prefill with persistent FP8 KV")
                return output
            except torch.OutOfMemoryError:
                native.logger.warning_once("FP8 FA2 prefill workspace exhausted; using paged FA2")
            PagedPrefill().forward(query[:count], keys, values,
                attn_metadata.block_table, layer._k_scale, layer._v_scale,
                attn_metadata.prefill_context, self.scale, output[:count])
            return output
        maximum = min(attn_metadata.maximum_context,
                      attn_metadata.block_table.shape[1] * keys.shape[1])
        if (attn_metadata.prefill_query_starts and attn_metadata.causal
                and window_left == -1 and not torch.cuda.is_current_stream_capturing()):
            PagedPrefill().forward_batch(query[:count], keys, values,
                attn_metadata.block_table, layer._k_scale, layer._v_scale,
                attn_metadata.prefill_query_starts, attn_metadata.seq_lens,
                maximum, self.scale, output[:count])
            return output
        grouped = (attn_metadata.causal and window_left == -1
                   and max_query * (query.shape[1] // kv_heads) <= 64)
        splits = (128 if attn_metadata.causal else 32) if max_query <= 64 else 1
        torch.ops.fa2_fp8kv.forward(query[:count], keys, values, output[:count],
            attn_metadata.query_start_loc, attn_metadata.seq_lens,
            attn_metadata.block_table, layer._k_scale, layer._v_scale,
            max_query, maximum, attn_metadata.causal, window_left, -1,
            self.scale, splits, grouped)
        return output


class Fp8FlashAttentionBackend(native.FlashAttentionBackend):
    @staticmethod
    def get_impl_cls():
        if extension_selected(get_current_vllm_config().cache_config.cache_dtype):
            return Fp8Attention
        return native.FlashAttentionImpl

    @staticmethod
    def get_builder_cls():
        if extension_selected(get_current_vllm_config().cache_config.cache_dtype):
            return Fp8MetadataBuilder
        return native.FlashAttentionMetadataBuilder

    @classmethod
    def supported_kv_cache_layouts(cls):
        return (KVCacheLayout.LBNHC,)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype):
        return extension_selected(kv_cache_dtype) or super().supports_kv_cache_dtype(kv_cache_dtype)

    @classmethod
    def supports_combination(cls, head_size, dtype, kv_cache_dtype, block_size,
                             use_mla, has_sink, use_sparse, use_mm_prefix, device_capability):
        if extension_selected(kv_cache_dtype):
            if (head_size not in (128, 256) or dtype != torch.bfloat16
                    or use_mla or has_sink or use_sparse or use_mm_prefix):
                return "FP8 FA2 requires BF16 queries, D128/D256 and plain decoder attention"
            return None
        return super().supports_combination(head_size, dtype, kv_cache_dtype, block_size,
            use_mla, has_sink, use_sparse, use_mm_prefix, device_capability)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""High-Performance Triton-only Attention layer."""

from dataclasses import dataclass, replace
from typing import ClassVar

import os
import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform

# 170HX opt: the split-KV verify path's batch gate is read from the environment
# directly (no envs.py registration, so the overlay cannot drift from the
# image's envs.py - KVarN, Turing knobs, the next pin).
_SPEC_ATTN_MIN_BATCH = int(os.environ.get("VLLM_SPEC_ATTN_MIN_BATCH", "4"))
from vllm.platforms.interface import DeviceCapability
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    compute_mm_prefix_range_tensor,
    get_num_attention_heads_from_layers,
    set_triton_ptr_range,
)
from vllm.v1.attention.ops.triton_prefill_attention import context_attention_fwd
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.attention.backends.flash_attn import (  # syv patch
    _spec_attn_enabled,
    _spec_attn_qmax,
    _spec_attn_run,
    _spec_attn_run_fp8,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)

logger = init_logger(__name__)


# constants
MIN_LAUNCH_GRID_SIZE_2D = 128  # Minimum launch grid size of 2D kernel
NUM_PAR_SOFTMAX_SEGMENTS = 16  # Number of parallel tiled softmax segments
# Max query tokens PER SEQUENCE eligible for the 3D path. ONE source: it bounds both
# eligibility (max_seqlen_q) and allocation (scratch_token_capacity_3d), so the two can
# never drift apart the way a literal in the dispatch site did.
MAX_QUERY_LEN_3D = 16


# ---- SCRATCH INSIDE THE MEMORY BUDGET -----------------------------------------
# The metadata builder that owns the 3D scratch is constructed in
# initialize_attn_backend, which initialize_kv_cache runs AFTER
# determine_available_memory has fixed the KV budget. Buffers allocated there are
# invisible to the memory profile: they come out of the headroom the profile
# reserved for activations (measured on the reference config: +182 MiB resident at
# max_num_seqs=64 with a byte-identical KV pool). The attention Impl is constructed
# during load_model, inside the window whose consumption the profile does count
# (memory_profiling: total_consumed = free memory at worker init minus free memory
# after the profile; "Model loading took" reports it). So the first Impl of a given
# geometry allocates the scratch into this pool and every builder of that geometry
# adopts it: the bytes show up in the model-load figure and the KV budget shrinks
# by exactly that much. The profiling-time builders
# (initialize_kv_cache(..., is_profiling=True) under profile_cudagraph_memory)
# adopt too, so the scratch is allocated once per geometry per process, not twice.
#
# resolve_cudagraph_mode_and_sizes can re-round cudagraph_capture_sizes for spec
# decode after the model is loaded, so a builder may derive a capacity the Impl did
# not: smaller adopts the larger pool (the declared capacity is the pool's), larger
# allocates again and says so at WARNING. Either way the boot record shows it.
#
# The derivation is the one the builder used; it moved, it did not change. That
# includes where the geometry comes from: head size and KV-head count are the
# MODEL CONFIG's, as the builder reads them, not the layer's own arguments. The
# two differ for a speculative drafter (its layers carry their own head size and
# KV heads; its builder still sizes rows by the target config), and a pool keyed
# on the layer's numbers is one the drafter's builder can never adopt.


@dataclass(frozen=True)
class Mq3dScratchPlan:
    num_heads_q: int
    num_heads_kv: int
    headdim_padded: int
    seq_threshold_3D: int          # SEQUENCES  (policy)
    max_query_len_3d: int          # QUERY TOKENS PER SEQUENCE  (eligibility)
    capacity: int                  # TOTAL QUERY TOKENS  (allocation)
    row_bytes: int
    max_num_batched_tokens: int
    max_num_seqs: int


def mq3d_scratch_plan(vllm_config: VllmConfig, num_heads_q: int) -> Mq3dScratchPlan:
    model_config = vllm_config.model_config
    num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
    headdim = model_config.get_head_size()
    # The launch grid for the 2D kernel is defined as (num_q_blocks, num_heads_kv).
    # A lower bound for num_q_blocks is the number of sequences.
    # To ensure the minimum launch grid size is achieved, the number of sequences
    # must be at least equal to the threshold below.
    # If this threshold is not reached (i.e., the batch size is not large enough),
    # the 3D kernel will be selected instead.
    seq_threshold_3D = MIN_LAUNCH_GRID_SIZE_2D // num_heads_kv
    compilation_config = vllm_config.compilation_config
    if compilation_config.cudagraph_mode in (
        CUDAGraphMode.FULL_AND_PIECEWISE,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL,
    ):
        capture_sizes = compilation_config.cudagraph_capture_sizes
        assert capture_sizes, "CUDA Graphs enabled but no capture sizes specified."
        # Select the CUDA Graph capture size closest to seq_threshold_3D as
        # threshold. This ensures that each captured graph covers the correct
        # execution path.
        seq_threshold_3D = min(capture_sizes, key=lambda x: abs(x - seq_threshold_3D))
    sched = vllm_config.scheduler_config
    # Eligibility already requires num_seqs <= seq_threshold_3D, and no batch can
    # carry more query tokens than the scheduler will place in one, so this is the
    # smallest capacity that can hold every policy-eligible batch.
    capacity = min(
        sched.max_num_batched_tokens,
        min(sched.max_num_seqs, seq_threshold_3D) * MAX_QUERY_LEN_3D,
    )
    headdim_padded = next_power_of_2(headdim)
    row_bytes = 4 * (
        num_heads_q * NUM_PAR_SOFTMAX_SEGMENTS * headdim_padded
        + 2 * num_heads_q * NUM_PAR_SOFTMAX_SEGMENTS
    )
    return Mq3dScratchPlan(
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        headdim_padded=headdim_padded,
        seq_threshold_3D=seq_threshold_3D,
        max_query_len_3d=MAX_QUERY_LEN_3D,
        capacity=capacity,
        row_bytes=row_bytes,
        max_num_batched_tokens=sched.max_num_batched_tokens,
        max_num_seqs=sched.max_num_seqs,
    )


class Mq3dScratch:
    """The three 3D-softmax scratch buffers for one geometry, and where they came from."""

    def __init__(self, plan: Mq3dScratchPlan, device: torch.device, origin: str):
        rows = plan.capacity
        self.output = torch.empty(
            (rows, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, plan.headdim_padded),
            dtype=torch.float32,
            device=device,
        )
        self.max = torch.empty(
            (rows, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS),
            dtype=torch.float32,
            device=device,
        )
        self.expsum = torch.empty_like(self.max)
        self.plan = plan
        self.origin = origin
        self.adopters = 0   # builders using this set

    @property
    def capacity(self) -> int:
        return self.output.shape[0]


# (device type, device index, num_heads_q, segments, headdim_padded) -> scratch
_MQ3D_SCRATCH_POOL: dict[tuple, Mq3dScratch] = {}


def mq3d_scratch_acquire(
    plan: Mq3dScratchPlan, device: torch.device, origin: str, exclusive: bool = False
) -> Mq3dScratch:
    """origin is "model load" (inside the memory profile) or "metadata builder"
    (after it). A builder that finds nothing to adopt still gets its buffers, and
    the boot record says so at WARNING: that is the unaccounted case.

    Builders of one geometry share a set: an attention call writes and reduces the
    scratch within the call, on one stream, which is the assumption the sharing
    across a group's layers already makes. Under ubatching (DBO) the builders of a
    group run on separate streams, so a builder passes exclusive=True and adopts
    only a set no builder has adopted; the rest allocate their own, after the
    profile, and say so."""
    dev = torch.device(device)
    key = (dev.type, dev.index, plan.num_heads_q, NUM_PAR_SOFTMAX_SEGMENTS, plan.headdim_padded)
    have = _MQ3D_SCRATCH_POOL.get(key)
    fits = have is not None and have.capacity >= plan.capacity
    if fits and not (exclusive and have.adopters > 0):
        if origin != "model load":
            have.adopters += 1
            logger.info(
                "int4 3D scratch: capacity=%d query tokens adopted by the %s "
                "(derived %d for this instance; allocated at %s)",
                have.capacity, origin, plan.capacity, have.origin,
            )
        return have
    inside = origin == "model load"
    log = logger.info if inside else logger.warning
    # Bind the inputs and the resulting cost into the boot record: a capacity that
    # cannot be read back at boot is a capacity nobody can audit later.
    log(
        "int4 3D scratch: capacity=%d query tokens (min of max_num_batched_tokens=%d, "
        "min(max_num_seqs=%d, seq_threshold_3D=%d) * max_query_len_3d=%d); "
        "%d B/row, %.2f MiB, allocated at %s%s",
        plan.capacity, plan.max_num_batched_tokens, plan.max_num_seqs,
        plan.seq_threshold_3D, plan.max_query_len_3d,
        plan.row_bytes, plan.row_bytes * plan.capacity / 2**20, origin,
        " (inside the memory profile)" if inside
        else " (AFTER the memory profile: these bytes are not in the KV budget"
        + ("; ubatching: builders do not share a set)" if fits else ")"),
    )
    scratch = Mq3dScratch(plan, dev, origin)
    if not inside:
        scratch.adopters = 1
    if not fits:
        _MQ3D_SCRATCH_POOL[key] = scratch   # an exclusive extra set is nobody else's to adopt
    return scratch


@dataclass
class TritonAttentionMetadata:
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    seq_threshold_3D: int          # SEQUENCES  (policy)
    max_query_len_3d: int          # QUERY TOKENS PER SEQUENCE  (eligibility)
    scratch_token_capacity_3d: int  # TOTAL QUERY TOKENS  (allocation)
    num_par_softmax_segments: int
    softmax_segm_output: torch.Tensor
    softmax_segm_max: torch.Tensor
    softmax_segm_expsum: torch.Tensor

    causal: bool | torch.Tensor

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None
    mm_prefix_range_tensor: torch.Tensor | None = None
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None


class TritonAttentionMetadataBuilder(AttentionMetadataBuilder[TritonAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS
    # Step-dependent fields reference persistent input buffers directly.
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        model_config = vllm_config.model_config
        # Compatible with models with non-uniform per-layer head counts.
        self.num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.headdim = model_config.get_head_size()

        # Check if CUDA Graphs are enabled for decode
        self.decode_cudagraph_enabled = (
            self.vllm_config.compilation_config.cudagraph_mode
            in (
                CUDAGraphMode.FULL_AND_PIECEWISE,
                CUDAGraphMode.FULL_DECODE_ONLY,
                CUDAGraphMode.FULL,
            )
        )

        self.num_par_softmax_segments = NUM_PAR_SOFTMAX_SEGMENTS
        # (syv) upstream's SM120 64-segment mode (#58482) stays off while the 3D scratch
        # comes from the budgeted pool (mq3d_scratch_plan), which is sized for 16 segments,
        # the layout every card ran before #58482. build()'s 16-segment fallback never fires.
        self.max_seqs_64_segments = 0
        # ---- UNIT SPLIT ------------------------------------------------------------
        # These three quantities were one variable, in three different units. The
        # scratch buffers are indexed by QUERY TOKEN, but were sized by
        # seq_threshold_3D, a SEQUENCE policy -- and in capture-enabled mode that
        # policy is snapped to the nearest decode capture size, which on this config
        # shrinks it 32 -> 8. The result was a silent 2D fallback for every
        # multi-query batch of 9..16 tokens, in capture-enabled mode only, i.e. the
        # only mode that ships. Measured: q=9 falls back with q_token_capacity_failed
        # in capture-enabled and runs 3D fully eager, same config, same request.
        #
        #   self.seq_threshold_3D        -- SEQUENCES.  Policy. Unchanged here (the
        #                                   capture-snap question is a separate change).
        #   self.max_query_len_3d        -- QUERY TOKENS PER SEQUENCE. Eligibility.
        #   self.scratch_token_capacity_3d -- TOTAL QUERY TOKENS. Allocation.
        #
        # The bound is derived from the engine's own config objects, not from an
        # observed run, the environment, or a launcher default. Eligibility already
        # requires num_seqs <= seq_threshold_3D, and no batch can carry more query
        # tokens than the scheduler will place in one, so this is the smallest
        # capacity that can hold every policy-eligible batch.
        plan = mq3d_scratch_plan(self.vllm_config, self.num_heads_q)
        assert (plan.num_heads_kv, plan.headdim_padded) == (
            self.num_heads_kv, next_power_of_2(self.headdim)
        ), "int4 3D scratch plan geometry disagrees with the builder's"
        self.seq_threshold_3D = plan.seq_threshold_3D
        self.max_query_len_3d = plan.max_query_len_3d
        scratch = mq3d_scratch_acquire(
            plan, device, "metadata builder",
            exclusive=bool(getattr(self.vllm_config.parallel_config, "use_ubatching", False)),
        )
        self.softmax_segm_output = scratch.output
        self.softmax_segm_max = scratch.max
        self.softmax_segm_expsum = scratch.expsum
        # The declared capacity is the pool's, never below the derived one. All
        # three buffers must agree with it: one oversized buffer can otherwise mask
        # another undersized one until a different reducer path is taken.
        self.scratch_token_capacity_3d = scratch.capacity
        assert (
            self.softmax_segm_output.shape[0]
            == self.softmax_segm_max.shape[0]
            == self.softmax_segm_expsum.shape[0]
            == self.scratch_token_capacity_3d
            >= plan.capacity
        ), "int4 3D scratch buffers disagree with the declared token capacity"
        self.rswa_window = model_config.rswa_window
        # R-SWA restages prefix lengths in build(), which the no-op
        # update_draft_decode_metadata() does not refresh.
        self.supports_draft_decode_metadata_update = self.rswa_window is None
        self.persistent_rswa_prefix_lens: torch.Tensor | None = None
        if self.rswa_window is not None:
            self.persistent_rswa_prefix_lens = torch.empty(
                vllm_config.scheduler_config.max_num_seqs,
                dtype=torch.int32,
                device=device,
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> TritonAttentionMetadata:
        attn_metadata = self.build(0, common_attn_metadata)
        # When doing full graph capture, setting seq_lens to
        # max_model_len will cause graph capture to be extremely
        # slow, so here we set it to 1.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TritonAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        attn_metadata = TritonAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=common_attn_metadata.causal,
            seq_threshold_3D=self.seq_threshold_3D,
            max_query_len_3d=self.max_query_len_3d,
            scratch_token_capacity_3d=self.scratch_token_capacity_3d,
            num_par_softmax_segments=self.num_par_softmax_segments,
            softmax_segm_output=self.softmax_segm_output,
            softmax_segm_max=self.softmax_segm_max,
            softmax_segm_expsum=self.softmax_segm_expsum,
        )
        if self.max_seqs_64_segments > 0 and (
            self.max_seqs_64_segments < seq_lens.shape[0]
            or self.softmax_segm_max.shape[0] < num_actual_tokens
        ):
            # The 16-segment grid fills the SMs, or the 64-segment rows cannot
            # hold every query token: same scratch, 16-segment layout.
            segments = NUM_PAR_SOFTMAX_SEGMENTS
            attn_metadata.num_par_softmax_segments = segments
            attn_metadata.softmax_segm_output = self.softmax_segm_output.view(
                -1, self.num_heads_q, segments, self.softmax_segm_output.shape[-1]
            )
            attn_metadata.softmax_segm_max = self.softmax_segm_max.view(
                -1, self.num_heads_q, segments
            )
            attn_metadata.softmax_segm_expsum = self.softmax_segm_expsum.view(
                -1, self.num_heads_q, segments
            )

        mm_ranges = common_attn_metadata.mm_req_doc_ranges
        if mm_ranges is not None:
            attn_metadata.mm_prefix_range = mm_ranges
            attn_metadata.mm_prefix_range_tensor = compute_mm_prefix_range_tensor(
                mm_ranges, num_reqs, seq_lens.device
            )

        rswa_prefix_lens = common_attn_metadata.rswa_prefix_lens
        if self.rswa_window is not None and rswa_prefix_lens is not None:
            assert self.persistent_rswa_prefix_lens is not None
            rswa_prefix_lens = rswa_prefix_lens.to(
                device=self.device, dtype=torch.int32, non_blocking=True
            )
            persistent_prefix_lens = self.persistent_rswa_prefix_lens[:num_reqs]
            persistent_prefix_lens.copy_(rswa_prefix_lens[:num_reqs])
            attn_metadata.rswa_prefix_lens = persistent_prefix_lens
            attn_metadata.rswa_window = self.rswa_window

        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: TritonAttentionMetadata) -> None:
        pass


class TritonAttentionBackend(AttentionBackend):
    @classmethod
    def customize_spec(cls, spec: "AttentionSpec") -> "AttentionSpec":
        """Per-token-head modes pack inline fp32 scales after each head's
        data, so the content is (data + one scale) per K/V side."""
        mode = spec.kv_quant_mode
        if spec.state_content_bytes is not None or not mode.is_per_token_head:
            return spec
        hs_k, hs_v = spec.head_size, spec.head_size_v
        if mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            hs_k, hs_v = hs_k // 2, hs_v // 2
        scale_bytes = get_dtype_size(torch.float32)
        content = (hs_k + hs_v) * get_dtype_size(spec.dtype) + 2 * scale_bytes
        return replace(spec, state_content_bytes=content)

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
        "int4_per_token_head",
        "int8_per_token_head",
        "fp8_per_token_head",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    forward_includes_kv_cache_update: bool = False

    @classmethod
    def supports_non_causal(cls) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @classmethod
    def supports_rswa(cls) -> bool:
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        return True

    @staticmethod
    def get_impl_cls() -> type["TritonAttentionImpl"]:
        return TritonAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @classmethod
    def supports_sink(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        return True

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True


class TritonAttentionImpl(AttentionImpl):
    # Per-token-head quant: scale views carved from inline head padding.
    _k_scale_cache: torch.Tensor | None = None
    _v_scale_cache: torch.Tensor | None = None

    def _ensure_scale_caches(self, kv_cache: torch.Tensor) -> None:
        """Extract per-head scale views from the padded content dimension.

        The KV cache is packed as logical shape
        ``(num_blocks, nkv, block_size, 2 * (hs + pad))`` where
        ``pad = sizeof(float32) / sizeof(cache_dtype)``.  The content dim holds
        ``[K(hs) | K_scale(pad) | V(hs) | V_scale(pad)]`` per (head, slot); the
        last ``pad`` elements of each half hold one float32 scale.  We create
        strided float32 views over those bytes.  ``kv_cache`` must be the
        packed logical tensor (call before any transpose), but may have HND or
        NHD physical strides.

        Scale shape: ``(num_blocks, block_size, num_kv_heads)``
        """
        if self._k_scale_cache is not None:
            return
        from vllm.utils.torch_utils import get_dtype_size

        num_blocks, nkv, block_size, content = kv_cache.shape
        dtype_sz = kv_cache.element_size()
        scale_pad = get_dtype_size(torch.float32) // dtype_sz  # e.g. 4
        padded_hs = content // 2
        hs = padded_hs - scale_pad

        raw = kv_cache.untyped_storage()
        base_f32 = torch.tensor([], dtype=torch.float32, device=kv_cache.device).set_(
            raw
        )

        def to_f32_units(elements: int) -> int:
            nbytes = elements * dtype_sz
            assert nbytes % 4 == 0
            return nbytes // 4

        # Actual strides (in float32 units) from the tensor. The logical cache
        # may be physically NHD, so do not assume C-contiguous HND layout.
        strides = kv_cache.stride()
        block_f32 = to_f32_units(strides[0])
        head_f32 = to_f32_units(strides[1])
        slot_f32 = to_f32_units(strides[2])
        # Scale sits at byte offset hs within each (K, then V) content half.
        base_off_f32 = to_f32_units(kv_cache.storage_offset())
        k_scale_off_f32 = base_off_f32 + to_f32_units(hs)
        v_scale_off_f32 = base_off_f32 + to_f32_units(padded_hs + hs)

        # K scales (first content half)
        self._k_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=k_scale_off_f32,
        )

        # V scales (second content half)
        self._v_scale_cache = torch.as_strided(
            base_f32,
            size=(num_blocks, block_size, nkv),
            stride=(block_f32, slot_f32, head_f32),
            storage_offset=v_scale_off_f32,
        )

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        sinks: torch.Tensor | None = None,
        use_alibi_sqrt: bool = False,
        chunk_lookback: int = -1,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type in (AttentionType.ENCODER, AttentionType.ENCODER_ONLY):
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if current_platform.is_cuda():
            cap = current_platform.get_device_capability()
            cap_str = cap.as_version_str() if cap is not None else "unknown"
            dev = current_platform.get_device_name()
            if self.kv_cache_dtype.startswith("fp8") and not (
                current_platform.has_device_capability(89)
            ):
                suggested = (
                    "float16" if (cap is None or cap.to_int() < 80) else "bfloat16"
                )
                raise ValueError(
                    f"FP8 KV cache is not supported by the Triton attention backend "
                    f"on {dev} (compute capability {cap_str}); native FP8 (fp8e4nv) "
                    f"requires SM89+. Re-run with --kv-cache-dtype {suggested}."
                )
            if self.kv_cache_dtype == "bfloat16" and not (
                current_platform.has_device_capability(80)
            ):
                raise ValueError(
                    f"bfloat16 KV cache is not supported on {dev} (compute capability "
                    f"{cap_str}); bfloat16 requires SM80+. Re-run with "
                    f"--kv-cache-dtype float16."
                )
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.fp8_dtype = current_platform.fp8_dtype()

        self.sinks = sinks
        if sinks is not None:
            assert sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                f"heads in the layer. Sinks shape: {sinks.shape}, "
                f"num_heads: {num_heads}."
            )
        self.use_alibi_sqrt = use_alibi_sqrt
        self.chunk_lookback = chunk_lookback
        self.supports_quant_query_input = current_platform.is_cuda()
        if current_platform.is_rocm():
            # Enable GPUs with fp8 MMA support
            from vllm.platforms.rocm import on_gfx11, on_gfx90a

            self.supports_quant_query_input = not (on_gfx90a() or on_gfx11())

        self._kv_quant_mode = get_kv_quant_mode(kv_cache_dtype)
        self._is_per_token_head_quant = self._kv_quant_mode.is_per_token_head

        # Enable tensor descriptors for Q/K/V load/store on platforms that
        # benefit from HW 2D block reads (Intel XPU).  The dead branch
        # is eliminated at Triton compile time, so other platforms see
        # zero cost when TD is off.
        #
        # ``VLLM_TRITON_USE_TD`` is tri-state:
        #   - unset (None): auto-select (TD on for XPU, off elsewhere),
        #   - ``1``: force TD on regardless of platform,
        #   - ``0``: force TD off regardless of platform (useful for A/B).
        td_override = envs.VLLM_TRITON_USE_TD
        if td_override is None:
            self.use_td = current_platform.is_xpu()
        else:
            self.use_td = td_override
        self._set_kv_ptr_range = current_platform.is_rocm()

        # Allocate the 3D scratch for this layer's head count while the memory
        # profile can still see it (Mq3dScratch above); the metadata builder
        # adopts it later. Only num_heads is the layer's: head size and KV heads
        # are the model config's, as the builder will derive them. Constructed
        # without a current config (unit tests), there is nothing to size from
        # and the builder allocates, unaccounted, as before.
        vllm_config = get_current_vllm_config_or_none()
        if vllm_config is not None and current_platform.is_cuda_alike():
            mq3d_scratch_acquire(
                mq3d_scratch_plan(vllm_config, num_heads),
                torch.device("cuda", torch.cuda.current_device()),
                "model load",
            )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with Paged Attention impl. in Triton.

        Args:
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

        Returns:
            shape = [num_tokens, num_heads * head_size]

        """
        if output_block_scale is not None:
            raise NotImplementedError(
                "fused block_scale output quantization is not yet supported"
                " for TritonAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # KV cache arrives in logical (B, H, N, 2*hs) order.
        # Per-token-head quantized KV cache: handled by the core unified
        # kernel, which dequantizes per-(token, head) inline via constexpr
        # branches (INT8 / FP8) and dispatches to the packed INT4 kernel.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            q_descale = k_descale = v_descale = None
        # FP8 per-tensor / auto path (original flow).
        else:
            kv_cache = kv_cache.transpose(1, 2)
            hs = self.head_size
            key_cache, value_cache = kv_cache.split(hs, dim=-1)
            if (
                is_quantized_kv_cache(self.kv_cache_dtype)
                and key_cache.dtype != self.fp8_dtype
            ):
                key_cache = key_cache.view(self.fp8_dtype)
                value_cache = value_cache.view(self.fp8_dtype)
            if self._set_kv_ptr_range:
                set_triton_ptr_range(kv_cache, key_cache, value_cache)
            descale_shape = (
                attn_metadata.query_start_loc.shape[0] - 1,
                key_cache.shape[2],
            )
            q_descale = (
                layer._q_scale
                if (
                    self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
                    and query.dtype == self.fp8_dtype
                )
                else None
            )
            k_descale = layer._k_scale.expand(descale_shape)
            v_descale = layer._v_scale.expand(descale_shape)
            k_scale_cache = None
            v_scale_cache = None

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        max_query_len_3d = attn_metadata.max_query_len_3d
        scratch_token_capacity_3d = attn_metadata.scratch_token_capacity_3d
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        # syv patch: the split-KV Triton verify attention (patches/spec-decode-attn.patch),
        # which vLLM's own unified attention cannot do here -- it refuses to split the KV
        # sequence whenever max_seqlen_q > 1 (triton_unified_attention.py, `use_3d`), and
        # every DFlash2 step is a multi-query verify. Measured at a 128k context, 8 query
        # tokens: 1.3 ms per layer against unified attention's 7.4 ms.
        # int8 per-token-head only: the kernel folds the per-(token, head) scales in after
        # each dot, which is exact because they are constant along the head dim. The
        # drafter's own sliding-window layers are excluded by the window test below.
        # fp8 per-tensor (vLLM's `fp8` cache, sm89+): the same kernel reads fp8 e4m3 and folds the
        # layer's scalar k/v scales in after each dot; it is what keeps FULL graphs and the split
        # verify together on the 4090 (issue 87: stock Triton verify decays 2.3x by 90K).
        spec_fp8 = (
            self._kv_quant_mode == KVQuantMode.FP8_PER_TENSOR
            and key_cache.dtype == self.fp8_dtype
        )
        if envs.VLLM_SPEC_ATTN_DEBUG and max_seqlen_q > 1 and not getattr(self, "_spec_dbg_logged", False):
            self._spec_dbg_logged = True
            logger.info(
                "syv spec-attn gate: enabled=%s quant_mode=%s cache_dtype=%s fp8_dtype=%s q_descale=%s spec_fp8=%s "
                "max_seqlen_q=%d qmax=%d causal=%s window=%s softcap=%s alibi=%s sinks=%s lookback=%s mm_prefix=%s rswa=%s out_scale=%s",
                _spec_attn_enabled(), self._kv_quant_mode, key_cache.dtype, self.fp8_dtype, q_descale is not None, spec_fp8,
                max_seqlen_q, _spec_attn_qmax(self.num_heads // self.num_kv_heads), attn_metadata.causal, self.sliding_window,
                self.logits_soft_cap, self.alibi_slopes is not None, self.sinks is not None, self.chunk_lookback,
                mm_prefix_range_tensor is not None, attn_metadata.rswa_prefix_lens is not None, output_scale is not None)
        if (
            _spec_attn_enabled()
            # 170HX opt: split-KV only once enough requests fill the SMs; 1-2
            # requests stay on unified attention because the extra combine launch
            # loses there. Batch size is a captured constant under FULL cudagraph.
            and (cu_seqlens_q.shape[0] - 1) >= _SPEC_ATTN_MIN_BATCH
            and (
                spec_fp8
                or (
                    k_scale_cache is not None
                    and self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD
                )
            )
            and 1 < max_seqlen_q <= _spec_attn_qmax(self.num_heads // self.num_kv_heads)
            and attn_metadata.causal
            and (self.sliding_window is None or self.sliding_window == (-1, -1))
            and not self.logits_soft_cap
            and self.alibi_slopes is None
            and self.sinks is None
            and self.chunk_lookback == -1
            and mm_prefix_range_tensor is None
            and attn_metadata.rswa_prefix_lens is None
            and output_scale is None
        ):
            if spec_fp8 and not getattr(self, "_spec_fp8_logged", False):
                self._spec_fp8_logged = True
                logger.info("syv spec-attn: split-KV verify on the fp8 per-tensor cache engaged (max_seqlen_q=%d)", max_seqlen_q)
            if spec_fp8:
                _spec_attn_run_fp8(
                    self,
                    query[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    cu_seqlens_q,
                    seqused_k,
                    block_table,
                    max_seqlen_q,
                    layer._k_scale,
                    layer._v_scale,
                    q_descale=q_descale,
                )
                return output
            _spec_attn_run(
                self,
                query[:num_actual_tokens],
                key_cache,
                value_cache,
                output[:num_actual_tokens],
                cu_seqlens_q,
                seqused_k,
                block_table,
                max_seqlen_q,
                k_scale_cache,
                v_scale_cache,
            )
            return output

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            alibi_slopes=self.alibi_slopes,
            use_alibi_sqrt=self.use_alibi_sqrt,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            seq_threshold_3D=seq_threshold_3D,
            max_query_len_3d=max_query_len_3d,
            scratch_token_capacity_3d=scratch_token_capacity_3d,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            sinks=self.sinks,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
            rswa_prefix_lens=attn_metadata.rswa_prefix_lens,
            rswa_window=attn_metadata.rswa_window,
            kv_quant_mode=self._kv_quant_mode,
            k_scale_cache=k_scale_cache,
            v_scale_cache=v_scale_cache,
            chunk_lookback=self.chunk_lookback,
            use_td=self.use_td,
            mm_prefix_clamp_sliding_window=getattr(
                layer, "mm_prefix_clamp_sliding_window", False
            ),
        )

        return output

    def _pth_key_value_caches(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token-head K/V cache views (ensures scale caches; FP8 retyped)."""
        self._ensure_scale_caches(kv_cache)
        padded_hs = kv_cache.shape[-1] // 2
        key_cache, value_cache = kv_cache.transpose(1, 2).split(padded_hs, dim=-1)
        if self._kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        return key_cache, value_cache

    def _forward_encoder_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Forward pass for encoder attention without KV cache.

        Args:
            query: shape = [num_encoder_tokens, num_heads, head_size]
            key: shape = [num_encoder_tokens, num_kv_heads, head_size]
            value: shape = [num_encoder_tokens, num_kv_heads, head_size]
            output: shape = [num_encoder_tokens, num_heads, head_size]
            attn_metadata: Encoder attention metadata
            layer: The attention layer

        """
        # Quantized KV cache is not supported for encoder attention.
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "quantized KV cache is not supported for encoder attention"
            )

        # Use encoder-specific metadata for sequence information
        query_start_loc = attn_metadata.query_start_loc
        seq_lens = attn_metadata.seq_lens
        max_query_len = attn_metadata.max_query_len

        # Call flash attention directly on Q, K, V tensors
        context_attention_fwd(
            q=query,
            k=key,
            v=value,
            o=output,
            b_start_loc=query_start_loc,
            b_seq_len=seq_lens,
            max_input_len=max_query_len,
            is_causal=False,  # Encoder attention is bidirectional
            softmax_scale=self.scale,
            sliding_window_q=self.sliding_window[0],
            sliding_window_k=self.sliding_window[1],
            sinks=self.sinks,
        )
        return output

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return
        # Reshape the input keys and values and store them in the cache.
        if self._is_per_token_head_quant:
            key_cache, value_cache = self._pth_key_value_caches(kv_cache)
            k_scale_cache = self._k_scale_cache
            v_scale_cache = self._v_scale_cache
            triton_reshape_and_cache_flash_per_token_head_quant(
                key,
                value,
                key_cache,
                value_cache,
                k_scale_cache,
                v_scale_cache,
                slot_mapping,
                kv_quant_mode=self._kv_quant_mode,
            )
            return
        # For decoder and cross-attention, use KV cache as before.
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        if is_quantized_kv_cache(self.kv_cache_dtype):
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)
        if self._set_kv_ptr_range:
            set_triton_ptr_range(kv_cache, key_cache, value_cache)
        triton_reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def fused_rope_kvcache_supported(self):
        if self._is_per_token_head_quant:
            return False
        return rocm_aiter_ops.is_enabled()

    def do_rope_and_kv_cache_update(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        is_neox: bool,
        kv_cache: torch.Tensor,
        layer_slot_mapping: torch.Tensor,
    ):
        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        flash_layout = True

        is_fp8_kv_cache = is_quantized_kv_cache(self.kv_cache_dtype)
        if is_fp8_kv_cache:
            key_cache = key_cache.view(self.fp8_dtype)
            value_cache = value_cache.view(self.fp8_dtype)

        rocm_aiter_ops.triton_rope_and_cache(
            query,
            key,
            value,
            positions,
            cos_sin_cache,
            is_neox,
            key_cache,
            value_cache,
            layer_slot_mapping,
            layer._k_scale,
            layer._v_scale,
            flash_layout,
            is_fp8_kv_cache,
        )

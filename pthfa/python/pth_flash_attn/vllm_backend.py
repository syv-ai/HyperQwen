# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""vLLM attention backend ``CUSTOM`` that runs ``pth_flash_attn_fwd``.

Selected with ``--attention-backend CUSTOM`` once the ``vllm.general_plugins``
entry point (``plugin.py``) has registered this class. It serves the paged
``int8_per_token_head`` cache of vLLM 0.30 and 0.31 on sm_80/86/89 and is built from
three pieces:

* **Cache spec, layout and writer are Triton's.** ``PthFlashAttnBackend``
  subclasses ``TritonAttentionBackend`` and ``PthFlashAttnImpl`` subclasses
  ``TritonAttentionImpl``, so ``customize_spec`` (inline fp32 scales), the
  page layout, the scale views and ``do_kv_cache_update`` (the Triton writer)
  are the code vLLM itself runs for this cache. Only the read side changes.
* **Its own metadata builder**. Triton's builder allocates scratch
  in ``__init__``, and the HyperQwen image patches that allocation; inheriting
  it would inherit the patch. ``PthFlashAttnMetadataBuilder`` allocates one
  tiny persistent buffer and plans launches with ``planner.plan_launch``.
* **Workspace reserved by the Impl at model load**: the Impl is
  constructed inside the window whose memory the KV budget accounts for, the
  builder is not.

Per forward there are at most two op calls: the decode-like slice (every request
with ``q_len <= reorder_batch_threshold``, MTP verify included: spec-as-decode)
and the prefill slice. The batch is ordered decode first (Model Runner V2
sorts it that way; a batch in another order is still correct, only slower, because
the prefill launch handles any ``q_len``).

Sliding window and non-causal attention are served at D = 128 only (the
windowed kernel exists at that head size), the DFlash2 drafter's layers (window 2048,
non-causal). The window is a layer constant: the Impl passes its own
``sliding_window`` to the op, and the builder, which is per KV-cache group, takes
it from the group's spec (vLLM groups layers by window) to cap the split count.
``causal`` comes per group from ``CommonAttentionMetadata.causal``. A causal
layer without a window calls the op with ``window=0, causal=True``: the causal
kernel and the same launch plan as without these arguments.

Nothing here reads device data on the host, allocates in ``forward`` or calls
``torch.cuda`` (device questions go through ``current_platform``): the decode
launch is capturable in a CUDA graph (``UNIFORM_BATCH``), and ``q_len`` of
every request is read from the device ``query_start_loc`` by the kernel, so
adaptive verification (device ``q_len`` smaller than the host bound) works.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from vllm.config import VllmConfig, get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
)
from vllm.v1.attention.backends.utils import (
    get_num_attention_heads_from_layers,
    split_decodes_and_prefills,
)
from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode
from vllm.v1.worker.ubatch_utils import get_num_ubatches
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

from pth_flash_attn import ops
from pth_flash_attn.planner import (
    MAX_ROWS_PER_CTA,
    SUPPORTED_HEAD_SIZES,
    WINDOWED_HEAD_SIZES,
    LaunchPlan,
    max_tokens_per_tile,
    plan_launch,
    workspace_shapes,
    workspace_token_splits,
)

# Under "vllm.": vLLM configures handlers and the INFO level only for that
# namespace, so a logger named "pth_flash_attn..." drops the INFO lines (such as
# the workspace-reservation line of the boot log).
logger = init_logger(f"vllm.{__name__}")

# The one backend name vLLM accepts for an out-of-tree backend:
# AttentionBackendEnum[backend.get_name()] must resolve, and CUSTOM is the only
# member reserved for plugins.
BACKEND_NAME = "CUSTOM"
# The only cache dtype served.
KV_CACHE_DTYPE = "int8_per_token_head"
# Kernel block sizes are multiples of 16 (the cp.async widths and the 64-token
# tile need no more); any larger multiple, such as the hybrid group's 816,
# is used as is, so no block-table conversion is needed.
KERNEL_BLOCK_MULTIPLE = 16
# The kernels target compute capability 8.x (sm_80/86/89).
SUPPORTED_CC_MAJOR = 8
# Verify length of a decode request without speculation.
NO_SPEC_QUERY_LEN = 1


def verify_query_len(vllm_config: VllmConfig) -> int:
    """Return q_dec_max, the verify length ``1 + k`` that sizes the workspace.

    The workspace is sized for the speculative verify length, not for
    the reorder threshold. A decode slice with a longer ``q_len`` (a short
    prefill extension) still works: the planner clamps its split count.
    """
    spec = vllm_config.speculative_config
    k = 0
    if spec is not None and spec.num_speculative_tokens:
        k = int(spec.num_speculative_tokens)
    return NO_SPEC_QUERY_LEN + k


def _num_sms(device: torch.device | None = None) -> int:
    """Return the SM count of ``device`` (default: the current device).

    From ``current_platform`` (never ``torch.cuda``). An input to the planner,
    which is pure host arithmetic and never queries a device itself.
    """
    index = device.index if device is not None else None
    if index is None:
        index = torch.accelerator.current_device_index()
    return int(current_platform.num_compute_units(index))


def reserved_token_splits(
    vllm_config: VllmConfig, *, num_kv_heads: int, head_size: int, num_sms: int
) -> int:
    """Return W_ts, the token-splits of workspace the Impl reserves at load.

    The Impl (reservation) and the builder (planner clamp) both call this with
    the same inputs, so a plan can never ask for more than was reserved.
    """
    return workspace_token_splits(
        max_num_seqs=int(vllm_config.scheduler_config.max_num_seqs),
        q_dec_max=verify_query_len(vllm_config),
        Hkv=num_kv_heads,
        D=head_size,
        num_sms=num_sms,
    )


@dataclass(frozen=True)
class PthLaunchSlice:
    """One op call: a contiguous run of tokens and requests of the batch.

    All tensors are views of persistent device buffers (or of the builder's
    own persistent rebase buffer) made once per step in ``build``, so ``forward``
    only slices ``query``/``output`` by the token range.
    """

    plan: LaunchPlan
    token_start: int
    token_end: int
    max_query_len: int
    query_start_loc: torch.Tensor  # int32 [num_reqs + 1], starts at 0
    seq_lens: torch.Tensor  # int32 [num_reqs]
    block_table: torch.Tensor  # int32 [>= num_reqs, max_blocks]


@dataclass
class PthAttnMetadata:
    """Attention metadata of one step: the Triton field names plus two slices.

    The Triton field names are kept because vLLM's runners and speculators read
    them (``slot_mapping``, ``seq_lens``, ``max_query_len`` ...).
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    causal: bool
    decode: PthLaunchSlice | None
    prefill: PthLaunchSlice | None
    use_cascade: bool = False
    common_prefix_len: int = 0


def _group_size(num_heads_q: int, num_kv_heads: int) -> int:
    if num_kv_heads < 1 or num_heads_q % num_kv_heads != 0:
        raise ValueError(
            f"pth_flash_attn: Hq = {num_heads_q} must be a positive multiple of "
            f"Hkv = {num_kv_heads}"
        )
    group = num_heads_q // num_kv_heads
    if group > MAX_ROWS_PER_CTA:
        raise ValueError(
            f"pth_flash_attn: G = Hq/Hkv = {group} exceeds {MAX_ROWS_PER_CTA}"
        )
    return group


def _no_windowed_kernel(what: str, head_size: int) -> str:
    """Reason text for a windowed/non-causal layer at a head size without kernel."""
    return (
        f"{what} at head size {head_size} (the windowed kernels exist for "
        f"D = {', '.join(map(str, WINDOWED_HEAD_SIZES))} only)"
    )


def group_window(kv_cache_spec: AttentionSpec) -> int:
    """Return the sliding window of a KV-cache group's layers, 0 for none.

    vLLM puts only layers of one window into a group: a ``SlidingWindowSpec``
    carries it, and so does a ``FullAttentionSpec`` when the hybrid allocator
    is off and windowed layers are kept as full attention. Specs without the
    field have no window. The builder uses it only to cap the split count;
    the mask the kernel applies comes from each layer's Impl.
    """
    window = getattr(kv_cache_spec, "sliding_window", None)
    return int(window) if window else 0


class PthFlashAttnMetadataBuilder(AttentionMetadataBuilder[PthAttnMetadata]):
    """Builds the two launch slices of a step; allocates only a small buffer."""

    # Decode batches of one uniform q_len (1, or 1 + k for MTP verify) replay as
    # FULL graphs; a batch with a prefill runs PIECEWISE (attention eager).
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    # Every step-dependent field is a persistent device tensor, and the host
    # integers are the same for every step of a q = 1 draft loop, so a fused
    # multi-step draft needs no per-step update (as Triton's builder).
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        """Plan inputs from the group's spec; raise for what v1 does not serve."""
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        model_config = vllm_config.model_config
        if get_num_ubatches(vllm_config.parallel_config) > 1:
            raise NotImplementedError(
                "pth_flash_attn: dual-batch overlap (DBO) is not supported: the "
                "workspace is one buffer per ubatch lane"
            )
        if model_config.rswa_window is not None:
            raise NotImplementedError("pth_flash_attn: R-SWA is not supported")
        num_heads_q = get_num_attention_heads_from_layers(
            vllm_config, layer_names
        ) or model_config.get_num_attention_heads(vllm_config.parallel_config)
        self.num_kv_heads = int(kv_cache_spec.num_kv_heads)
        self.head_size = int(kv_cache_spec.head_size)
        self.group = _group_size(num_heads_q, self.num_kv_heads)
        if self.head_size not in SUPPORTED_HEAD_SIZES:
            raise NotImplementedError(
                f"pth_flash_attn: head size {self.head_size} is not one of "
                f"{SUPPORTED_HEAD_SIZES}"
            )
        self.window = group_window(kv_cache_spec)
        if self.window and self.head_size not in WINDOWED_HEAD_SIZES:
            raise NotImplementedError(
                "pth_flash_attn: "
                + _no_windowed_kernel("sliding window", self.head_size)
            )
        # The decode threshold is the most query tokens one row tile holds
        # (floor(64 / G)), raised to the verify length by spec-as-decode (a
        # verify of 1 + k tokens is a decode, not a prefill).
        self._init_reorder_batch_threshold(
            max_tokens_per_tile(self.group), supports_spec_as_decode=True
        )
        self.num_sms = _num_sms(device)
        self.ws_token_splits = reserved_token_splits(
            vllm_config,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            num_sms=self.num_sms,
        )
        # query_start_loc of the prefill slice, rebased to 0 on the device. One
        # int32 per request plus one: persistent so the step allocates nothing,
        # and not in the KV budget because it is a few bytes.
        self._prefill_qsl = torch.zeros(
            int(vllm_config.scheduler_config.max_num_seqs) + 1,
            dtype=torch.int32,
            device=device,
        )

    @classmethod
    def get_varlen_cudagraph_max_query_len(
        cls,
        vllm_config: VllmConfig,
        kv_cache_spec: AttentionSpec,  # type: ignore[override]
    ) -> int:
        """Return the longest decode ``q_len`` a FULL graph covers (vLLM 0.31+).

        Harmless on 0.30.0 (never called). The same value as the builder's
        reorder threshold, from the model config instead of a live builder.
        """
        model_config = vllm_config.model_config
        heads = model_config.get_num_attention_heads(vllm_config.parallel_config)
        group = _group_size(heads, int(kv_cache_spec.num_kv_heads))
        return max(max_tokens_per_tile(group), verify_query_len(vllm_config))

    def _slice(
        self,
        cam: CommonAttentionMetadata,
        *,
        req_start: int,
        req_end: int,
        token_start: int,
        token_end: int,
        max_query_len: int,
        is_prefill: bool,
    ) -> PthLaunchSlice:
        plan = plan_launch(
            num_tokens=token_end - token_start,
            num_reqs=req_end - req_start,
            max_query_len=max_query_len,
            G=self.group,
            Hkv=self.num_kv_heads,
            D=self.head_size,
            num_sms=self.num_sms,
            is_prefill=is_prefill,
            ws_token_splits=self.ws_token_splits,
            window=self.window,
        )
        n = req_end - req_start
        qsl = cam.query_start_loc
        if req_start == 0:
            # Decode slice: tokens start at 0, no rebase.
            query_start_loc = qsl if qsl.shape[0] == n + 1 else qsl[: n + 1]
        else:
            # Prefill slice: rebase on the device (the host copy of qsl can
            # differ from the device one under adaptive verification, so the
            # device value of the first prefill request is subtracted).
            if n + 1 > self._prefill_qsl.shape[0]:
                raise RuntimeError(
                    f"pth_flash_attn: {n} prefill requests exceed max_num_seqs = "
                    f"{self._prefill_qsl.shape[0] - 1}"
                )
            buf = self._prefill_qsl[: n + 1]
            torch.sub(
                qsl[req_start : req_end + 1], qsl[req_start : req_start + 1], out=buf
            )
            query_start_loc = buf
        seq_lens = cam.seq_lens[req_start:req_end]
        block_table = cam.block_table_tensor[req_start:req_end]
        return PthLaunchSlice(
            plan=plan,
            token_start=token_start,
            token_end=token_end,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> PthAttnMetadata:
        """Split the batch into its decode-like and prefill slices and plan each."""
        cam = common_attn_metadata
        if common_prefix_len > 0:
            raise NotImplementedError("pth_flash_attn: cascade attention")
        causal = cam.causal
        if not isinstance(causal, bool):
            # vLLM types it bool | Tensor; a tensor is a per-request mask.
            raise NotImplementedError(
                "pth_flash_attn: per-request causal masks are not supported (got "
                f"{type(causal).__name__}); causal must be one bool per group"
            )
        if not causal and self.head_size not in WINDOWED_HEAD_SIZES:
            raise NotImplementedError(
                "pth_flash_attn: "
                + _no_windowed_kernel("non-causal attention", self.head_size)
            )
        if cam.rswa_prefix_lens is not None:
            raise NotImplementedError("pth_flash_attn: R-SWA is not supported")
        if cam.mm_req_doc_ranges and any(cam.mm_req_doc_ranges.values()):
            raise NotImplementedError(
                "pth_flash_attn: bidirectional multimodal prefix ranges are not "
                "supported; Qwen does not use them"
            )
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                cam, decode_threshold=self.reorder_batch_threshold or 1
            )
        )
        num_decodes = int(num_decodes)
        num_decode_tokens = int(num_decode_tokens)
        num_reqs = int(cam.num_reqs)
        num_tokens = int(cam.num_actual_tokens)
        decode = prefill = None
        if num_decodes > 0 and num_decode_tokens > 0:
            if num_prefills == 0:
                q_max = int(cam.max_query_len)
            else:
                # Host bound of the decode part alone, from the CPU copy of
                # query_start_loc (a CPU tensor: no device sync).
                qsl_cpu = cam.query_start_loc_cpu
                q_max = int(
                    (qsl_cpu[1 : num_decodes + 1] - qsl_cpu[:num_decodes]).max()
                )
            decode = self._slice(
                cam,
                req_start=0,
                req_end=num_decodes,
                token_start=0,
                token_end=num_decode_tokens,
                max_query_len=max(q_max, 1),
                is_prefill=False,
            )
        if num_prefills > 0 and num_tokens - num_decode_tokens > 0:
            prefill = self._slice(
                cam,
                req_start=num_decodes,
                req_end=num_reqs,
                token_start=num_decode_tokens,
                token_end=num_tokens,
                max_query_len=int(cam.max_query_len),
                is_prefill=True,
            )
        return PthAttnMetadata(
            num_actual_tokens=num_tokens,
            max_query_len=int(cam.max_query_len),
            query_start_loc=cam.query_start_loc,
            max_seq_len=int(cam.max_seq_len),
            seq_lens=cam.seq_lens,
            block_table=cam.block_table_tensor,
            slot_mapping=cam.slot_mapping,
            causal=causal,
            decode=decode,
            prefill=prefill,
        )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> PthAttnMetadata:
        """Build for a graph capture; seq_lens = 1 keeps the warm-up launch short."""
        attn_metadata = self.build(0, common_attn_metadata)
        # Same as Triton's builder: with seq_lens = max_model_len a capture run
        # walks the whole cache. The planned grid does not depend on seq_lens.
        attn_metadata.seq_lens.fill_(1)
        return attn_metadata

    def update_draft_decode_metadata(self, _metadata: PthAttnMetadata) -> None:
        """Nothing to update: every step-dependent field is a persistent buffer."""


class PthFlashAttnBackend(TritonAttentionBackend):
    """``CUSTOM``: Triton's cache and writer, this project's kernel and builder."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [KV_CACHE_DTYPE]  # type: ignore[list-item]

    @staticmethod
    def get_name() -> str:
        """Return ``CUSTOM``: vLLM resolves the name with ``AttentionBackendEnum``."""
        return BACKEND_NAME

    @staticmethod
    def get_impl_cls() -> type[PthFlashAttnImpl]:
        """Return the Impl class."""
        return PthFlashAttnImpl

    @staticmethod
    def get_builder_cls() -> type[PthFlashAttnMetadataBuilder]:  # type: ignore[override]
        """Return this backend's builder (not Triton's)."""
        return PthFlashAttnMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes(
        kv_cache_spec: AttentionSpec | None = None,  # type: ignore[override]
    ) -> list[int | MultipleOf]:
        """Return the kernel block sizes; the argument exists on vLLM 0.31+ only."""
        return [MultipleOf(KERNEL_BLOCK_MULTIPLE)]

    @classmethod
    def supports_block_size(cls, block_size: int | None) -> bool:
        """Accept any block size that is a multiple of 16."""
        return block_size is None or block_size % KERNEL_BLOCK_MULTIPLE == 0

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        """Accept the three instantiated head sizes."""
        return head_size in SUPPORTED_HEAD_SIZES

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """Decoder self-attention only."""
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_non_causal(cls) -> bool:
        """Yes, at D = 128 (``validate_configuration`` refuses other sizes)."""
        return True

    @classmethod
    def supports_sliding_window(cls) -> bool:
        """Yes, at D = 128 (``validate_configuration`` refuses other sizes)."""
        return True

    @classmethod
    def supports_rswa(cls) -> bool:
        """No R-SWA: prefix tokens that stay globally visible are not modelled."""
        return False

    @classmethod
    def validate_configuration(cls, *args: Any, **kwargs: Any) -> list[str]:
        """Return vLLM's reasons, plus: windowed or non-causal needs D = 128.

        ``supports_sliding_window``/``supports_non_causal`` take no head size,
        so the refusal at D = 64 and 256 (no windowed kernel there) is added
        here, where an explicit selection gets it as a reason instead of a
        load-time error.
        The arguments are bound to the base signature (it differs across vLLM
        versions), so its defaults apply and keyword or positional calls agree.
        """
        reasons = super().validate_configuration(*args, **kwargs)
        bound = inspect.signature(super().validate_configuration).bind(*args, **kwargs)
        bound.apply_defaults()
        arg = bound.arguments
        head_size = arg["head_size"]
        if head_size in SUPPORTED_HEAD_SIZES and head_size not in WINDOWED_HEAD_SIZES:
            if arg["has_sliding_window"]:
                reasons.append(_no_windowed_kernel("sliding window", head_size))
            if arg["use_non_causal"]:
                reasons.append(_no_windowed_kernel("non-causal attention", head_size))
        return reasons

    @classmethod
    def supports_sink(cls) -> bool:
        """No attention sinks."""
        return False

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """No bidirectional multimodal prefix."""
        return False

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:
        """No ALiBi."""
        return False

    @classmethod
    def supports_batch_invariance(cls) -> bool:
        """Not batch invariant: the split count depends on the batch shape."""
        return False

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        """sm_8x with the op library built."""
        return capability.major == SUPPORTED_CC_MAJOR and ops.is_available()


class PthFlashAttnImpl(TritonAttentionImpl):
    """Reads the int8 cache with ``pth_flash_attn_fwd``; writes it as Triton does."""

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
        """Reject what the kernel does not serve, then reserve the workspace."""
        # Checked before anything is allocated: an invalid explicit selection
        # must fail at load with its reason, not after a reservation.
        unsupported = []
        if alibi_slopes is not None or use_alibi_sqrt:
            unsupported.append("ALiBi")
        if sliding_window is not None:
            if sliding_window < 1:
                unsupported.append(f"sliding window {sliding_window}")
            elif head_size not in WINDOWED_HEAD_SIZES:
                unsupported.append(_no_windowed_kernel("sliding window", head_size))
        if logits_soft_cap:
            unsupported.append("logits soft cap")
        if sinks is not None:
            unsupported.append("attention sinks")
        if chunk_lookback != -1:
            unsupported.append("chunk lookback")
        if attn_type != AttentionType.DECODER:
            unsupported.append(f"attention type {attn_type}")
        if kv_cache_dtype != KV_CACHE_DTYPE:
            unsupported.append(f"kv_cache_dtype {kv_cache_dtype}")
        if head_size not in SUPPORTED_HEAD_SIZES:
            unsupported.append(f"head size {head_size}")
        if unsupported:
            raise NotImplementedError(
                "pth_flash_attn does not support: " + ", ".join(unsupported)
            )
        _group_size(num_heads, num_kv_heads)
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
            use_alibi_sqrt,
            chunk_lookback,
        )
        if self.dcp_world_size > 1 or self.pcp_world_size > 1:
            raise NotImplementedError("pth_flash_attn: DCP/PCP are not supported")
        if self._kv_quant_mode != KVQuantMode.INT8_PER_TOKEN_HEAD:
            raise NotImplementedError("pth_flash_attn: int8_per_token_head only")
        # The query stays 16-bit: there is no quantised-query input.
        self.supports_quant_query_input = False
        # The op's window: Triton's ``self.sliding_window`` is a (left, right)
        # tuple that vLLM's helpers read, so the int is kept separately.
        self.window = 0 if sliding_window is None else int(sliding_window)
        self._cache_views: tuple[int, tuple[torch.Tensor, ...]] | None = None
        self._ws_token_splits = 0
        self._no_workspace: torch.Tensor | None = None
        self._reserve_workspace(get_current_vllm_config_or_none())
        from pth_flash_attn.fa2_slab import prepare_workspace
        self._fa2_slab = prepare_workspace(self, get_current_vllm_config_or_none())

    def _reserve_workspace(self, vllm_config: VllmConfig | None) -> None:
        """Reserve W_ts token-splits of fp32 partials in the shared workspace.

        Done here, at Impl construction during model load, so the bytes are in
        the memory profile that sizes the KV cache. With no
        config (unit tests) nothing is reserved and forward asks the manager
        for what it needs.
        """
        if vllm_config is None or not is_workspace_manager_initialized():
            return
        self._ws_token_splits = reserved_token_splits(
            vllm_config,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            num_sms=_num_sms(),
        )
        o_shape, lse_shape = workspace_shapes(
            ws_token_splits=self._ws_token_splits, Hq=self.num_heads, D=self.head_size
        )
        ws_o, _ = current_workspace_manager().get_simultaneous(
            (o_shape, torch.float32), (lse_shape, torch.float32)
        )
        # S == 1 launches pass an empty tensor (the op reads no workspace); one
        # per Impl, created here so forward allocates nothing, on the device
        # the workspace lives on.
        self._no_workspace = torch.empty(0, dtype=torch.float32, device=ws_o.device)
        logger.info(
            "pth_flash_attn: reserved %d token-splits of workspace "
            "(Hq=%d, D=%d, %.2f MiB) inside the model-load memory window",
            self._ws_token_splits,
            self.num_heads,
            self.head_size,
            (o_shape[0] * o_shape[1] * (o_shape[2] + 1) * 4) / 2**20,
        )

    def _caches(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (k, v, k_scale, v_scale) views of this layer's cache.

        Built once per cache tensor: the layer's cache never moves, and a view
        per call would cost CPU time in the piecewise path. K and V are cut to
        the head size (Triton's views include the 4 scale bytes of padding).
        """
        ptr = kv_cache.data_ptr()
        cached = self._cache_views
        if cached is not None and cached[0] == ptr:
            return cached[1]  # type: ignore[return-value]
        key_cache, value_cache = self._pth_key_value_caches(kv_cache)
        hs = self.head_size
        views = (
            key_cache[..., :hs],
            value_cache[..., :hs],
            self._k_scale_cache,
            self._v_scale_cache,
        )
        self._cache_views = (ptr, views)  # type: ignore[assignment]
        return views  # type: ignore[return-value]

    def _run_slice(
        self,
        part: PthLaunchSlice,
        query: torch.Tensor,
        output: torch.Tensor,
        caches: tuple[torch.Tensor, ...],
        causal: bool,
    ) -> None:
        """One op call for one slice; workspace views of this launch's size."""
        plan = part.plan
        if plan.num_splits > 1:
            if (
                self._ws_token_splits
                and plan.num_splits * plan.num_tokens > self._ws_token_splits
            ):
                raise RuntimeError(
                    f"pth_flash_attn: launch needs {plan.num_splits * plan.num_tokens}"
                    f" token-splits, {self._ws_token_splits} were reserved"
                )
            ws_o, ws_lse = current_workspace_manager().get_simultaneous(
                (plan.ws_o_shape, torch.float32), (plan.ws_lse_shape, torch.float32)
            )
        else:
            assert self._no_workspace is not None
            ws_o = ws_lse = self._no_workspace
        t0, t1 = part.token_start, part.token_end
        k, v, ks, vs = caches
        ops.pth_flash_attn_fwd(
            output if (t0 == 0 and t1 == output.shape[0]) else output[t0:t1],
            query if (t0 == 0 and t1 == query.shape[0]) else query[t0:t1],
            k,
            v,
            ks,
            vs,
            part.block_table,
            part.seq_lens,
            part.query_start_loc,
            part.max_query_len,
            plan.wm,
            plan.num_splits,
            self.scale,
            ws_o,
            ws_lse,
            self.window,
            causal,
        )

    def forward(
        self,
        layer: torch.nn.Module,  # type: ignore[override]
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PthAttnMetadata,  # type: ignore[override]
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decode slice and the prefill slice of the batch (<= 2 op calls).

        Args:
            layer: The attention layer (unused: no per-tensor scales here).
            query: ``[num_tokens, num_heads, head_size]``.
            key: Unused: the cache write is ``do_kv_cache_update``.
            value: Unused, as ``key``.
            kv_cache: ``[num_blocks, num_kv_heads, block_size, 2 * (hs + pad)]``.
            attn_metadata: Built by ``PthFlashAttnMetadataBuilder``; None in the
                memory-profile run.
            output: ``[num_tokens, num_heads, head_size]``, written in place.
            output_scale: Fused output quantisation, refused.
            output_block_scale: Fused block-scale output quantisation, refused.

        Returns:
            ``output``.
        """
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "pth_flash_attn: fused output quantisation is not supported"
            )
        if attn_metadata is None:
            # Memory-profile run: no attention is executed.
            return output.fill_(0)
        caches = self._caches(kv_cache)
        if self._fa2_slab is not None:
            from pth_flash_attn.fa2_slab import try_prefill
            if try_prefill(self, self._fa2_slab, query, output, caches, attn_metadata):
                return output
        if attn_metadata.decode is not None:
            self._run_slice(
                attn_metadata.decode, query, output, caches, attn_metadata.causal
            )
        if attn_metadata.prefill is not None:
            self._run_slice(
                attn_metadata.prefill, query, output, caches, attn_metadata.causal
            )
        return output

"""Register the FP8 extension through vLLM's attention-backend plugin API."""
from fa2_fp8kv_vllm.paths import enabled


def register():
    if not enabled():
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.FLASH_ATTN,
                     "fa2_fp8kv_vllm.backend.Fp8FlashAttentionBackend")

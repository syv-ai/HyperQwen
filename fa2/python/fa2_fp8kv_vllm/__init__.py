"""Register the FP8 extension through vLLM's attention-backend plugin API."""


def register():
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.FLASH_ATTN,
                     "fa2_fp8kv_vllm.backend.Fp8FlashAttentionBackend")

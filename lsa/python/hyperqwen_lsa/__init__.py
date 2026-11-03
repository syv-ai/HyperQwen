# SPDX-License-Identifier: Apache-2.0
"""Register the LSA backend as AttentionBackendEnum.CUSTOM when LSA=1; a no-op otherwise.

With LSA=1 this takes the CUSTOM slot even if pthfa's plugin registered it first (the launcher
refuses LSA=1 with PTH_FA=1, so the two never both mean to own it).
"""
from hyperqwen_lsa.settings import enabled

BACKEND_CLASS_PATH = "hyperqwen_lsa.backend.LsaAttentionBackend"


def register():
    if not enabled():
        return
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS_PATH)

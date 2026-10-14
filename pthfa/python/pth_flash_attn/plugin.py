# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 noon-at-cgn
"""The ``vllm.general_plugins`` entry point: registers the ``CUSTOM`` backend.

vLLM loads every ``vllm.general_plugins`` entry point in each process it starts
(the API server's engine core and every tensor-parallel worker), before the
attention backend is selected. ``register`` records the backend's class path
under ``AttentionBackendEnum.CUSTOM`` and nothing else; the CUDA op library and
vLLM's Triton modules are imported only when the backend is actually selected
(``--attention-backend CUSTOM``), so a server that does not select it pays
nothing and cannot be broken by this package.

Why ``CUSTOM`` and not a name of our own: the attention layer resolves the
selected class back to the enum with ``AttentionBackendEnum[get_name()]``
(plan V10), and ``CUSTOM`` is the only member reserved for plugins. If another
plugin already registered a different class there, this one stays out of the
way and says so: silently replacing it would change what ``--attention-backend
CUSTOM`` means for that user.
"""

from __future__ import annotations

import logging

BACKEND_CLASS_PATH = "pth_flash_attn.vllm_backend.PthFlashAttnBackend"

logger = logging.getLogger(__name__)


def register() -> None:
    """Register ``PthFlashAttnBackend`` as ``AttentionBackendEnum.CUSTOM``."""
    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    custom = AttentionBackendEnum.CUSTOM
    if custom.is_overridden() and custom.get_path() != BACKEND_CLASS_PATH:
        logger.warning(
            "pth_flash_attn: AttentionBackendEnum.CUSTOM is already registered "
            "as %s; leaving it unchanged",
            custom.get_path(),
        )
        return
    register_backend(custom, BACKEND_CLASS_PATH)

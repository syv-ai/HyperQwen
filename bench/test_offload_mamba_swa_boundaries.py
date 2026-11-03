#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regression: real native store/lookup must agree at Mamba replay boundaries.

Uses actual scheduler, masks and CPU manager; GPU block IDs are metadata-only
stand-ins and no model or GPU data transfer runs. Run in the patched image.
Setup follows vLLM's tests/v1/kv_connector/unit/offloading_connector scaffolding.
"""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec, MambaSpec, SlidingWindowSpec, KVCacheConfig,
    KVCacheGroupSpec, get_mamba_prefill_checkpoint_position,
)
from vllm.v1.kv_offload.base import OffloadingKVEventsConfig
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


class ReplayBoundaryTests(unittest.TestCase):
    def replay(self, length, retention, eagle, with_swa=True):
        block = 832
        attention = dict(block_size=block, num_kv_heads=1, head_size=1, dtype=torch.bfloat16)
        full = FullAttentionSpec(**attention)
        mamba = MambaSpec(block_size=block, shapes=((1, 1), (1, 1, 1)),
                          dtypes=(torch.bfloat16, torch.float32), mamba_cache_mode='align')
        swa = SlidingWindowSpec(**attention, sliding_window=2048)
        # The reported topology: two full-attention, six recurrent and one SWA group.
        specs = [full, full] + [mamba] * 6 + ([swa] if with_swa else [])
        groups = [KVCacheGroupSpec([f'layer{i}'], value) for i, value in enumerate(specs)]
        manager = CPUOffloadingManager(num_chunks=10000)
        spec = NS(config=NS(groups=tuple(NS(group_id=i, tokens_per_block=block) for i in range(len(groups)))),
                  tokens_per_block=(block,) * len(groups), tokens_per_hash=block,
                  blocks_per_chunk=1, offload_prompt_only=True,
                  kv_events_config=OffloadingKVEventsConfig(enable_kv_cache_events=False, self_describing_kv_events=False),
                  get_manager=lambda: manager)
        config = NS(speculative_config=NS(use_eagle_block_drop=lambda: eagle),
                    cache_config=NS(enable_prefix_caching=True, prefix_cache_retention_interval=retention),
                    parallel_config=NS(world_size=1, decode_context_parallel_size=1))
        scheduler = OffloadingConnectorScheduler(
            spec, config, KVCacheConfig(num_blocks=10000, kv_cache_tensors=[], kv_cache_groups=groups))

        def request(name, tokens):
            value = MagicMock()
            value.request_id = name
            value.kv_transfer_params = value.kv_hints = None
            value.num_tokens = value.num_prompt_tokens = tokens
            value.num_computed_tokens = value.shared_prefix_boundary = 0
            value.skip_reading_prefix_cache = False
            value.is_finished.return_value = False
            value.block_hashes = [f'h{i}'.encode() for i in range(tokens // block)]
            scheduler.on_new_request(value)
            state = scheduler._req_status[name]
            state.update_offload_keys()
            return value, state

        original, state = request('original', length)
        count = length // block
        for group in state.group_states:
            # Store scheduling consumes IDs only; this test does not transfer tensors.
            group.block_ids = list(range(1, count + 1))
        jobs = scheduler._build_store_jobs(NS(num_scheduled_tokens={'original': length}, finished_req_ids=set()))
        self.assertTrue(jobs)
        stored = set()
        for job_id in jobs:
            keys = scheduler._jobs[job_id].keys
            manager.complete_store(keys, state.req_context)
            stored.update(keys)
        # The existing CoW handoff stores the exact Mamba boundary separately.
        # Feed those real keys so this regression isolates the SWA store mask.
        boundary = get_mamba_prefill_checkpoint_position(length, block, eagle)
        recurrent_keys = [state.group_states[i].offload_keys[boundary // block - 1] for i in range(2, 8)]
        prepared = manager.prepare_store(recurrent_keys, state.req_context)
        self.assertIsNotNone(prepared)
        self.assertFalse(prepared.evicted_keys, 'the tier must retain every stored key')
        manager.complete_store(prepared.keys_to_store, state.req_context)
        stored.update(recurrent_keys)
        replay, resumed = request('replay', length + 1046)
        result = scheduler._lookup_complete_chunks(resumed)
        self.assertEqual(result, boundary,
                         f'retention={retention}, eagle={eagle}, length={length}: '
                         'stored Mamba state must have the SWA history needed at its boundary')
        if with_swa:
            window = scheduler.config.kv_group_configs[-1].load_window_size_in_chunks(boundary)
            keys = resumed.group_states[-1].offload_keys
            self.assertTrue(set(keys[boundary // block - window:boundary // block]).issubset(stored))

    def test_sparse_dflash_store_preserves_mamba_replay_window(self):
        for length in (8000, 32768, 600000):
            for retention in (0, 832 * 8):
                with self.subTest(length=length, retention=retention):
                    self.replay(length, retention, True)

    def test_dense_and_non_eagle_replays_still_restore(self):
        for length in (8000, 32768):
            for retention, eagle in ((None, True), (None, False), (0, False)):
                with self.subTest(length=length, retention=retention, eagle=eagle):
                    self.replay(length, retention, eagle)

    def test_mtp_style_topology_without_swa_still_restores(self):
        self.replay(32768, 0, True, with_swa=False)


if __name__ == '__main__':
    unittest.main()

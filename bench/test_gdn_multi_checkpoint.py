#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU gate for real GDN multi-checkpoint allocation, metadata and cache lookup.

Run in the patched image, without GPU or downloaded weights. This tests cache
metadata, not model inference. Exit 0 on pass, 1 on failure.
"""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from vllm.model_executor.layers.mamba import checkpoint as cp
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.attention.backends import gdn_attn as gdn
from vllm.third_party.flash_linear_attention.ops import index as fla_index
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import MambaSpec, get_mamba_periodic_prefill_checkpoints
from vllm.v1.request import Request


def transfer(values, dtype=None, device=None):
    return torch.tensor(values, dtype=dtype, device=device)


def fixture(speculative=0, interval=256, drop=False):
    spec = MambaSpec(
        block_size=128,
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_speculative_blocks=speculative,
        num_prefill_checkpoint_blocks=9 if interval else 1,
        prefill_checkpoint_alignment=1,
        prefill_checkpoint_interval=interval,
    )
    pool = BlockPool(num_gpu_blocks=512, enable_caching=True, hash_block_size=128)
    manager = MambaManager(
        spec, pool, enable_caching=True, kv_cache_group_id=0, scheduler_block_size=128
    )
    manager.drop_eagle_checkpoint_block = drop
    config = NS(
        num_hash_lookahead_tokens=0,
        cache_config=NS(
            mamba_cache_mode="align",
            hash_block_size=128,
            cache_hit_alignment_tokens=128,
        ),
        speculative_config=NS(use_eagle_block_drop=lambda: drop) if drop else None,
    )
    return manager, config


def request(tokens=1423):
    init_none_hash(sha256)
    return Request(
        request_id="r",
        prompt_token_ids=list(range(tokens)),
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        block_hasher=get_request_block_hasher(128, sha256),
    )


def allocate(manager, req, start, end):
    manager.remove_skipped_blocks("r", start, req.num_prompt_tokens)
    before = [b.block_id for b in manager.req_to_blocks["r"]]
    estimate = manager.get_num_blocks_to_allocate(
        "r", end, [], start, start, end, prefill_end=req.num_prompt_tokens
    )
    free = manager.block_pool.get_num_free_blocks()
    new = manager.allocate_new_blocks("r", end, end)
    actual = free - manager.block_pool.get_num_free_blocks()
    if actual != estimate:
        raise AssertionError(("allocation estimate", start, end, estimate, actual))
    table = manager.req_to_blocks["r"]
    if [b.block_id for b in new] != [b.block_id for b in table[len(before) :]]:
        raise AssertionError("worker append table differs from allocator")
    for col in manager._periodic_checkpoints.get("r", []):
        if col < len(before) and before[col]:
            assert before[col] == table[col].block_id, (
                "export moved an existing worker column"
            )
    return table


def metadata_plan(manager, config, start, end):
    table = torch.tensor([[b.block_id for b in manager.req_to_blocks["r"]]])
    m = NS(
        seq_lens_cpu_upper_bound=torch.tensor([end]),
        query_start_loc_cpu=torch.tensor([0, end - start], dtype=torch.int32),
        query_start_loc=torch.tensor([0, end - start], dtype=torch.int32),
        block_table_tensor=table,
    )
    builder = object.__new__(gdn.GDNAttentionMetadataBuilder)
    builder.gdn_prefill_backend = "triton"
    with (
        patch.object(cp, "async_tensor_h2d", transfer),
        patch.object(gdn, "async_tensor_h2d", transfer),
        patch.object(fla_index, "async_tensor_h2d", transfer),
    ):
        metadata = cp.MambaPrefillCheckpointBuilder(
            config, manager.kv_cache_spec
        ).build(m, [0])
        plan = builder._build_checkpoint_plan(
            metadata, 0, 0, m.query_start_loc_cpu, torch.device("cpu")
        )
    return metadata, plan


class MultiCheckpointTests(unittest.TestCase):
    def test_coordinator_keeps_the_real_prefill_boundary_for_periodic_exports(self):
        for retention in (None, 256):
            manager, _ = fixture()
            coordinator = NS(
                retention_interval=retention, single_type_managers=[manager]
            )
            KVCacheCoordinator.get_num_blocks_to_allocate(
                coordinator, "r", 1423, ([],), 0, 0, 0, 1423, prefill_end=1423
            )
            self.assertTrue(manager._periodic_checkpoints["r"])
            # Once decode begins no periodic export may be allocated: the GDN
            # speculative/decode path does not write those prefill snapshots.
            KVCacheCoordinator.get_num_blocks_to_allocate(
                coordinator, "r", 2048, ([],), 0, 1500, 1500, 2048, prefill_end=1423
            )
            self.assertEqual(manager._periodic_checkpoints["r"], [])

    def test_deep_divergence_has_states_missing_from_single_checkpoint(self):
        for drop in (False, True):
            with self.subTest(drop=drop):
                hits = []
                for interval in (0, 256):
                    manager, config = fixture(interval=interval, drop=drop)
                    req = request()
                    table = allocate(manager, req, 0, req.num_prompt_tokens)
                    metadata, plan = metadata_plan(
                        manager, config, 0, req.num_prompt_tokens
                    )
                    written = {
                        int(block): length
                        for (_, length), block in zip(
                            plan.source_slices, plan.state_indices.tolist()
                        )
                    }
                    manager.cache_blocks(
                        req,
                        req.num_prompt_tokens,
                        retention_interval=256,
                        replay_boundaries=[1280],
                    )
                    # Every key must name the exact state the worker exports.
                    for col, block in enumerate(table):
                        if block.block_hash is not None:
                            self.assertEqual(
                                block.block_hash_num_tokens, written[block.block_id]
                            )
                    _, hit = manager.find_longest_cache_hit(
                        req.block_hashes,
                        768,
                        [0],
                        manager.block_pool,
                        manager.kv_cache_spec,
                        drop,
                        128,
                    )
                    hits.append(hit)
                self.assertEqual(hits, [0, 768])

    def test_chunked_and_speculative_allocation_matches_worker_plan(self):
        for speculative in (0, 1, 3, 7):
            for drop in (False, True):
                for chunks in (
                    [1423],
                    [384, 1423],
                    [640, 1152, 1423],
                    [333, 777, 1423],
                ):
                    with self.subTest(
                        speculative=speculative, drop=drop, chunks=chunks
                    ):
                        manager, config = fixture(speculative, drop=drop)
                        req, start = request(), 0
                        for end in chunks:
                            manager.new_step_starts()
                            table = allocate(manager, req, start, end)
                            meta, plan = metadata_plan(manager, config, start, end)
                            actual = (
                                dict(
                                    zip(
                                        plan.state_indices.tolist(),
                                        (start + n for _, n in plan.source_slices),
                                    )
                                )
                                if plan
                                else {}
                            )
                            expected = {
                                table[col].block_id: (col + 1) * 128
                                for col in manager._periodic_checkpoints["r"]
                            }
                            if "r" in manager._checkpoints:
                                pos, col = manager._checkpoints["r"]
                                expected[table[col].block_id] = pos
                            self.assertEqual(actual, expected)
                            self.assertNotIn(
                                manager.block_pool.null_block.block_id, actual
                            )
                            manager.cache_blocks(
                                req,
                                end,
                                retention_interval=256,
                                replay_boundaries=[1280],
                            )
                            req.num_computed_tokens = start = end
                        manager.free("r")
                        self.assertEqual(manager.block_pool.get_num_free_blocks(), 511)
                        self.assertNotIn("r", manager._periodic_checkpoints)

    def test_periodic_geometry_never_aliases_input_or_final_checkpoint(self):
        for start in (0, 128, 333, 512):
            for end in (777, 1423, 2048):
                if end <= start:
                    continue
                values = get_mamba_periodic_prefill_checkpoints(
                    start, end, 256, 128, 128, 1, 5
                )
                for pos, col in values:
                    self.assertTrue(start < pos < end)
                    self.assertEqual(pos, (col + 1) * 128)
                    self.assertGreater(col, (start - 1) // 128)
                    self.assertNotEqual(col, 5)

    def test_admission_probe_does_not_replace_pending_exports(self):
        manager, _ = fixture()
        req = request()
        allocate(manager, req, 0, 384)
        before = (list(manager._periodic_checkpoints["r"]), dict(manager._checkpoints))
        manager.get_num_blocks_to_allocate(
            "r",
            100000,
            [],
            384,
            384,
            100000,
            apply_admission_cap=True,
            prefill_end=100000,
        )
        self.assertEqual(
            before, (manager._periodic_checkpoints["r"], manager._checkpoints)
        )

    def test_multi_page_memory_budget_counts_each_in_flight_batch(self):
        for concurrent in (1, 2):
            manager, _ = fixture(speculative=3)
            manager.kv_cache_spec = replace(
                manager.kv_cache_spec, num_prefill_checkpoint_blocks=5
            )
            spec = manager.kv_cache_spec
            config = NS(
                cache_config=NS(mamba_cache_mode="align"),
                max_concurrent_batches=concurrent,
            )
            req = request(16000)
            free = manager.block_pool.get_num_free_blocks()
            peak = 0
            for end in range(1024, 12289, 1024):
                start = end - 1024
                # The previous batch may still be writing its exported states.
                committed = max(0, start - (concurrent - 1) * 1024)
                manager.remove_skipped_blocks("r", committed, req.num_prompt_tokens)
                manager.get_num_blocks_to_allocate(
                    "r", end, [], start, start, end, prefill_end=req.num_prompt_tokens
                )
                manager.allocate_new_blocks("r", end, end)
                peak = max(peak, free - manager.block_pool.get_num_free_blocks())
                manager.cache_blocks(
                    req, end, retention_interval=256, replay_boundaries=[]
                )
                req.num_computed_tokens = end
                manager.new_step_starts()
            self.assertLessEqual(
                peak * spec.page_size_bytes, spec.max_memory_usage_bytes(config)
            )
            if concurrent > 1:
                # Counting only one batch of checkpoints underestimates this
                # actual allocation trace, even though it covers current state.
                old = replace(spec, prefill_checkpoint_interval=0)
                self.assertGreater(
                    peak * spec.page_size_bytes, old.max_memory_usage_bytes(config)
                )
            manager.free("r")
            self.assertEqual(manager.block_pool.get_num_free_blocks(), free)


class ComparatorTests(unittest.TestCase):
    def compare(self, off_hit=0, mutate=None):
        from test_gdn_checkpoint_model import compare_results

        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for variant, hit in (("enabled", 768), ("default-off", off_hit)):
                records = [
                    {
                        "label": str(i),
                        "prompt_sha256": str(i),
                        "token_ids": [42],
                        "elapsed_seconds": 1.0,
                        "cached_tokens": hit if i == 2 else 0,
                        "memory_per_rank": [{"peak_allocated": 42}],
                    }
                    for i in range(4)
                ]
                data = {
                    "passed": True,
                    "records": records,
                    "parameters": {"kv_cache_memory_bytes": 4_000_000_000},
                    "model_files": {"config": "same"},
                    "probe_sha256": "same",
                }
                if variant == "default-off" and mutate:
                    mutate(data)
                (directory / f"gdn-model-{variant}.json").write_text(json.dumps(data))
            compare_results(directory, False, True)

    def test_comparison_accepts_an_enabled_only_hit(self):
        self.compare()

    def test_comparison_rejects_the_old_equal_hit_result(self):
        with self.assertRaisesRegex(AssertionError, "criterion was not observed"):
            self.compare(off_hit=768)

    def test_comparison_rejects_missing_records_or_changed_prompts(self):
        for mutate in (
            lambda data: data["records"].pop(),
            lambda data: data["records"][2].update(prompt_sha256="changed"),
            lambda data: data["records"][2].update(token_ids=[99]),
            lambda data: data["parameters"].update(kv_cache_memory_bytes=1),
        ):
            with self.assertRaises(AssertionError):
                self.compare(mutate=mutate)


if __name__ == "__main__":
    unittest.main()

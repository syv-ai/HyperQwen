"""CPU regressions for async Mamba admission, capacity and allocation replay.

Run after patches/apply.sh with the pinned vLLM runtime:
  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_async_mamba_kv_sizing.py
The fixture contains allocation counters from a 163K-token cold prefill.
"""

import json
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch
from vllm.v1.core import kv_cache_utils as utils
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)

HERE = Path(__file__).resolve().parent
PAGE = 1730560
BLOCK = 17 * PAGE
OLD_BUDGET = 6442450944
NEW_BUDGET = 6576668672


def config(depth=2, mode="align", max_len=163840):
    return NS(
        max_concurrent_batches=depth,
        num_prefill_lookahead_tokens=1,
        model_config=NS(max_model_len=max_len, original_max_model_len=max_len),
        parallel_config=NS(decode_context_parallel_size=1, tensor_parallel_size=1),
        attention_config=NS(hisparse_config=None),
        cache_config=NS(
            mamba_cache_mode=mode,
            num_gpu_blocks_override=None,
            prefix_cache_retention_interval=None,
            get_resolved_kv_cache_layout=lambda: KVCacheLayout.BLHNC,
        ),
    )


def mamba_spec(drafts=4, checkpoints=0):
    # Only the padded state-page size matters to this allocator regression.
    return MambaSpec(
        block_size=832,
        shapes=((PAGE,),),
        dtypes=(torch.int8,),
        mamba_cache_mode="align",
        num_speculative_blocks=drafts,
        num_prefill_checkpoint_blocks=checkpoints,
    )


def groups(aggregated=True):
    attn = FullAttentionSpec(
        block_size=832,
        num_kv_heads=4,
        head_size=256,
        dtype=torch.int8,
        state_content_bytes=520,
    )
    result = []
    for gid, (spec, count) in enumerate([(attn, 17)] + [(mamba_spec(), 16)] * 3):
        layers = {f"group{gid}.layer{i}": spec for i in range(count)}
        merged = (
            UniformTypeKVCacheSpecs(block_size=832, kv_cache_specs=layers)
            if aggregated
            else spec
        )
        result.append(KVCacheGroupSpec(list(layers), merged))
    return result


def cache(blocks, aggregated=True):
    return KVCacheConfig(
        num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=groups(aggregated)
    )


def admit(budget, cfg=None):
    cfg = cfg or config()
    gs = groups()
    # These are the real admission/auto-fit functions used after grouping in
    # get_kv_cache_configs, including its existing null-block byte reservation.
    usable = budget - utils._pool_bytes_per_block(gs)
    utils._check_enough_kv_cache_memory(
        usable,
        lambda: utils._max_memory_usage_bytes_from_groups(cfg, gs),
        cfg.model_config.max_model_len,
        lambda am: utils._estimate_max_model_len_from_groups(cfg, gs, am),
    )


class Pool:
    num_gpu_blocks = 223
    hash_block_size = 832

    def __init__(self):
        self.used = 0

    def get_new_blocks(self, count):
        self.used += count
        return [NS(is_null=False, block_hash=None, ref_cnt=1) for _ in range(count)]

    def free_blocks(self, blocks):
        self.used -= len(blocks)

    def is_block_writable(self, block):
        return True


def replay():
    snapshots = json.loads((HERE / "fixtures/async-mamba-prefill.json").read_text())[
        "snapshots"
    ]
    m = MambaManager.__new__(MambaManager)
    m.kv_cache_spec = mamba_spec()
    m.mamba_cache_mode = "align"
    m.num_speculative_blocks = 4
    m.has_prefill_checkpoint_blocks = False
    m.drop_eagle_checkpoint_block = False
    m.block_size = 832
    m.block_pool = Pool()
    m._null_block = NS(is_null=True)
    m.req_to_blocks = defaultdict(list)
    m._allocated_block_reqs = set()
    m._checkpoints = {}
    m._partial_hit_reqs = {}
    m.last_state_block_idx = {}
    m._num_retired_blocks = {}
    m._deferred_state_blocks = {}
    m.cached_blocks_this_step = set()
    rows = []
    for row in snapshots:
        end, computed, processed = row["end"], row["computed"], row["processed"]
        m.remove_skipped_blocks("cold", processed, 163328)
        m.get_num_blocks_to_allocate(
            "cold", end, [], computed, computed, end, prefill_end=163328
        )
        m.allocate_new_blocks("cold", end, end)
        actual = utils.cdiv(end + 4, 832) + 3 * m.block_pool.used + 1
        if actual != row["physical_blocks"]:
            raise AssertionError((row, actual))
        rows.append(
            dict(end=end, physical_blocks=actual, mamba_per_group=m.block_pool.used)
        )
    return dict(
        snapshots_matched=len(rows), peak=max(rows, key=lambda r: r["physical_blocks"])
    )


def report():
    cfg, gs = config(), groups()
    needed = utils._max_memory_usage_bytes_from_groups(cfg, gs)
    result = dict(
        context=163840,
        kv_bytes=NEW_BUDGET,
        bytes_per_pool_block=BLOCK,
        physical_blocks=NEW_BUDGET // BLOCK,
        null_blocks=1,
        required_request_blocks=needed // BLOCK,
        required_physical_blocks=needed // BLOCK + 1,
        minimum_budget_bytes=needed + BLOCK,
        spare_blocks=NEW_BUDGET // BLOCK - needed // BLOCK - 1,
        reported_capacity_tokens=utils.get_kv_cache_capacity(
            cfg, cache(NEW_BUDGET // BLOCK)
        )[0],
        allocator_replay=replay(),
        cuda_initialized=torch.cuda.is_initialized(),
    )
    return result


class SizingTests(unittest.TestCase):
    def test_async_depth_accounts_for_inflight_states(self):
        for depth in (1, 2, 3, 5):
            with self.subTest(depth=depth):
                self.assertEqual(
                    mamba_spec().max_memory_usage_bytes(config(depth)),
                    PAGE * (1 + depth + 4),
                )

    def test_internal_checkpoint_reservation_is_additive(self):
        self.assertEqual(
            mamba_spec(checkpoints=1).max_memory_usage_bytes(config()), 8 * PAGE
        )

    def test_no_speculation_still_accounts_for_async_overlap(self):
        self.assertEqual(
            mamba_spec(drafts=0).max_memory_usage_bytes(config()), 3 * PAGE
        )

    def test_non_align_modes_preserve_original_sizing(self):
        spec = mamba_spec()
        self.assertEqual(spec.max_memory_usage_bytes(config(mode="all")), 201 * PAGE)
        self.assertEqual(spec.max_memory_usage_bytes(config(mode="none")), 5 * PAGE)

    def test_worker_and_scheduler_capacity_agree(self):
        for aggregated in (False, True):
            self.assertAlmostEqual(
                utils.get_max_concurrency_for_kv_cache_config(
                    config(), cache(223, aggregated)
                ),
                222 / 218,
            )

    def test_null_block_is_unavailable_even_for_full_attention_only(self):
        cfg = cache(198)
        cfg.kv_cache_groups = cfg.kv_cache_groups[:1]
        self.assertEqual(
            utils.get_max_concurrency_for_kv_cache_config(config(), cfg), 1
        )

    def test_pool_with_only_null_block_has_zero_capacity(self):
        self.assertEqual(
            utils.get_max_concurrency_for_kv_cache_config(config(), cache(1)), 0
        )

    def test_old_six_gib_budget_fails_admission(self):
        with self.assertRaisesRegex(ValueError, "max seq len"):
            admit(OLD_BUDGET)

    def test_boundary_budget_accepts_exactly_219_blocks(self):
        admit(219 * BLOCK)
        with self.assertRaises(ValueError):
            admit(219 * BLOCK - 1)

    def test_tested_budget_has_four_spare_blocks(self):
        admit(NEW_BUDGET)
        r = report()
        self.assertEqual(
            (r["physical_blocks"], r["required_physical_blocks"], r["spare_blocks"]),
            (223, 219, 4),
        )

    def test_auto_fit_uses_whole_blocks_and_restores_config(self):
        cfg = config(max_len=200000)
        self.assertEqual(
            utils._estimate_max_model_len_from_groups(
                cfg, groups(), OLD_BUDGET - BLOCK
            ),
            163072,
        )
        self.assertEqual(
            utils._estimate_max_model_len_from_groups(
                cfg, groups(), NEW_BUDGET - BLOCK
            ),
            167232,
        )
        self.assertEqual(cfg.model_config.max_model_len, 200000)

    def test_legacy_admission_uses_group_padding_and_async_peak(self):
        cfg, gs = config(), groups()
        specs = {
            name: spec
            for g in gs
            for name, spec in g.kv_cache_spec.kv_cache_specs.items()
        }
        with patch.object(utils, "get_kv_cache_groups", return_value=gs):
            with self.assertRaises(ValueError):
                utils.check_enough_kv_cache_memory(cfg, specs, OLD_BUDGET)
            utils.check_enough_kv_cache_memory(cfg, specs, NEW_BUDGET)

    def test_engine_startup_rejects_small_pool_and_small_override(self):
        cfg, gs = config(), groups()
        specs = {
            name: spec
            for g in gs
            for name, spec in g.kv_cache_spec.kv_cache_specs.items()
        }
        # Grouping is supplied from the measured layout; admission, pool-byte
        # rounding, override handling, and config construction run installed code.
        with patch.object(utils, "get_kv_cache_groups", return_value=gs):
            with self.assertRaises(ValueError):
                utils.get_kv_cache_configs(cfg, [specs], [OLD_BUDGET])
            cfg.cache_config.num_gpu_blocks_override = 218
            with self.assertRaises(ValueError):
                utils.get_kv_cache_configs(cfg, [specs], [NEW_BUDGET])

    def test_engine_startup_builds_the_223_block_pool(self):
        cfg, gs = config(), groups()
        specs = {
            name: spec
            for g in gs
            for name, spec in g.kv_cache_spec.kv_cache_specs.items()
        }
        with patch.object(utils, "get_kv_cache_groups", return_value=gs):
            (built,) = utils.get_kv_cache_configs(cfg, [specs], [NEW_BUDGET])
        self.assertEqual(built.num_blocks, 223)
        self.assertTrue(all(t.size == 223 * BLOCK for t in built.kv_cache_tensors))
        self.assertEqual(utils.get_kv_cache_capacity(cfg, built)[0], 166846)

    def test_real_allocator_reproduces_all_99_snapshots(self):
        r = replay()
        self.assertEqual(r["snapshots_matched"], 99)
        self.assertEqual(
            r["peak"], dict(end=163328, physical_blocks=219, mamba_per_group=7)
        )

    def test_no_gpu_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)

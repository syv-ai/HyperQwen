"""CPU check of the KV cache seat budget line (patches/kv-cache-seat-budget.patch).

log_kv_cache_seat_budget must report how many requests the pool holds at ~1k, 32k and
max_model_len from each device group's admission cap, using vLLM's own spec classes. The groups
below are the shape of the reference stack (Qwen3.8-27B + DFlash2 drafter: 6 GDN groups, 2
full-attention groups, 1 sliding-window group, group size 8). Each arm in ARMS is a boot on the
reference RTX 3090 (vLLM 0.31.0, batch 5 Q2, prefix caching on, MAX_LEN 65536):
  - the old cost per request is recovered from its captured concurrency metric;
  - corrected async reservations add one state to each of six GDN groups;
  - the new conservative residency estimates are pinned exactly.
Captured short-request peaks remain historical data, not an upper bound on
worst-case async reservations: not every resident request peaks together.
No GPU. Run inside the vLLM venv after the patches are applied:
  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_kv_cache_seat_budget.py
"""
import re
import types
import unittest
from unittest import mock

import torch
from vllm.v1.core import kv_cache_utils as kcu
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

ns = types.SimpleNamespace


def vllm_config(max_model_len, mamba_cache_mode="align", block_size=448, max_num_seqs=16):
    # The fields the specs and the log read. max_in_flight_tokens: async scheduling (2 batches)
    # x max_num_batched_tokens 2048, the launcher's setting.
    return ns(
        model_config=ns(max_model_len=max_model_len),
        cache_config=ns(mamba_cache_mode=mamba_cache_mode, block_size=block_size),
        scheduler_config=ns(max_num_seqs=max_num_seqs, max_num_batched_tokens=2048),
        parallel_config=ns(decode_context_parallel_size=1, prefill_context_parallel_size=1),
        max_in_flight_tokens=4096,
        max_concurrent_batches=2,
    )


def stack(block, k, num_blocks, kv_dtype=torch.bfloat16, host_extra=False):
    """The reference stack's KV cache groups at one block size and k."""
    mamba = MambaSpec(
        block_size=block,
        shapes=((10240, 3 + k), (48, 128, 128)),
        dtypes=(torch.bfloat16, torch.float16),
        num_speculative_blocks=k,
    )
    full = FullAttentionSpec(block_size=block, num_kv_heads=4, head_size=256, dtype=kv_dtype)
    window = SlidingWindowSpec(
        block_size=block, num_kv_heads=8, head_size=128, dtype=kv_dtype, sliding_window=2048
    )
    groups = [KVCacheGroupSpec([f"gdn.{i}"], mamba) for i in range(6)]
    groups += [KVCacheGroupSpec([f"attn.{i}"], full) for i in range(2)]
    groups += [KVCacheGroupSpec(["draft.sw"], window)]
    if host_extra:
        groups += [KVCacheGroupSpec(["host"], full, host_resident=True)]
    return KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups)


def logged(cfg, kv):
    with mock.patch.object(kcu.logger, "info_once") as info:
        kcu.log_kv_cache_seat_budget(cfg, kv)
    if not info.called:
        return None
    msg, *args = info.call_args.args
    return msg % tuple(args)


def residency(line):
    """{context length: (low, high)} from the line."""
    tail = line.split("Resident requests by context length: ")[1]
    out = {}
    for low, high, n in re.findall(r"(\d+)(?:-(\d+))? at ([\d,]+)", tail):
        out[int(n.replace(",", ""))] = (int(low), int(high or low))
    return out


# arm: (block, k, kv dtype, num_gpu_blocks, the engine's kv_cache_max_concurrency at 65536,
#       the line's residency at 1024 / 32768 / 65536, held at ~1k / ~32.8k / ~61.5k)
ARMS = {
    "Q2-a int8 k=7": (864, 7, torch.int8, 382, 1.7767441860465116, ((6, 7), (2, 2), (1, 1)), (7, 2, 1)),
    "Q2-b int8 k=3": (816, 3, torch.int8, 404, 2.009950248756219, ((10, 13), (3, 3), (2, 2)), (13, 3, 2)),
    "Q2-c int8 k=7, KV_MEM -1 GiB": (864, 7, torch.int8, 308, 1.4325581395348836, ((5, 5), (2, 2), (1, 1)), (5, 2, 1)),
    "Q2-d bf16 k=7": (448, 7, torch.bfloat16, 380, 1.046831955922865, ((5, 7), (1, 1), (1, 1)), (6, 1, 1)),
}


ASYNC_RESIDENCY = {
    "Q2-a int8 k=7": ((5, 6), (2, 2), (1, 1)),
    "Q2-b int8 k=3": ((9, 11), (3, 3), (1, 1)),
    "Q2-c int8 k=7, KV_MEM -1 GiB": ((4, 5), (2, 2), (1, 1)),
    "Q2-d bf16 k=7": ((5, 6), (1, 1), (1, 1)),
}


class MeasuredBoots(unittest.TestCase):
    def test_each_arm(self):
        for name, (block, k, dtype, blocks, old_concurrency, _, _) in ARMS.items():
            with self.subTest(name=name):
                cfg = vllm_config(65536, block_size=block)
                line = logged(cfg, stack(block, k, blocks, kv_dtype=dtype))
                # Six GDN groups each gain the third in-flight async state.
                old_blocks = round(blocks / old_concurrency)
                self.assertIn(f"{6 * (3 + k)} blocks that do not grow with context, "
                              f"{old_blocks + 6} in all at 65,536 tokens", line)
                self.assertIn(f"{blocks - 1} usable blocks of {block} tokens", line)
                r = residency(line)
                self.assertEqual((r[1024], r[32768], r[65536]), ASYNC_RESIDENCY[name])

    def test_ctx_long_boot(self):
        # 2026-10-16 H3: int8 CTX=long, k=7, max_model_len 131072, num_gpu_blocks 382, block 864.
        # The engine reported 1.04x concurrency (382 / 367) and held 7 at ~1k and 2 at ~32k.
        line = logged(vllm_config(131072, block_size=864), stack(864, 7, 382, kv_dtype=torch.int8))
        self.assertIn("381 usable blocks of 864 tokens; a request holds 60 blocks that do not grow "
                      "with context, 373 in all at 131,072 tokens", line)
        r = residency(line)
        self.assertEqual((r[1024], r[32768], r[131072]), ((5, 6), (2, 2), (1, 1)))

    def test_update_kv_cache_capacity_logs_the_budget(self):
        cfg = vllm_config(65536)
        cfg.device_config = ns(device_type="cuda")
        with mock.patch.object(kcu.logger, "info_once") as info:
            kcu.update_kv_cache_capacity(cfg, stack(448, 7, 380))
        self.assertTrue(any("KV cache seat budget" in c.args[0] for c in info.call_args_list))


class Behaviour(unittest.TestCase):
    def test_state_term_follows_the_mamba_cache_mode(self):
        line = logged(vllm_config(65536, mamba_cache_mode="none"), stack(448, 7, 380))
        self.assertIn("48 blocks that do not grow with context", line)

    def test_no_32k_entry_below_32k_and_max_model_len_restored(self):
        cfg = vllm_config(16384)
        line = logged(cfg, stack(448, 7, 380))
        self.assertEqual(sorted(residency(line)), [1024, 16384])
        self.assertEqual(cfg.model_config.max_model_len, 16384)

    def test_host_resident_groups_do_not_count(self):
        self.assertEqual(
            logged(vllm_config(65536), stack(448, 7, 380, host_extra=True)),
            logged(vllm_config(65536), stack(448, 7, 380)),
        )

    def test_attention_only_model_prints_no_range(self):
        full = FullAttentionSpec(block_size=16, num_kv_heads=8, head_size=128, dtype=torch.bfloat16)
        kv = KVCacheConfig(num_blocks=1001, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["a"], full)])
        r = residency(logged(vllm_config(4096, block_size=16), kv))
        self.assertEqual(r, {1024: (15, 15), 4096: (3, 3)})

    def test_empty_pool_logs_nothing(self):
        self.assertIsNone(logged(vllm_config(65536), stack(448, 7, 1)))


if __name__ == "__main__":
    unittest.main()

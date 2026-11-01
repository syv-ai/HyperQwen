#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scale views must preserve shared hybrid pages and already-written KV scales.

Runs against the installed, patched vLLM and PTH_FA plugin. CPU is the default
and needs no model/GPU. --device cuda:0 also checks the real INT8 cache writer.
The implementations' view methods are called directly; constructors and their
unrelated distributed/workspace setup are bypassed.
"""
import argparse
import unittest

import torch
from pth_flash_attn.vllm_backend import PthFlashAttnImpl
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
from vllm.v1.kv_cache_interface import KVQuantMode

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--device", default="cpu", help="cpu (default), cuda:0 or cuda:1")
args, remaining = parser.parse_known_args()
DEVICE = torch.device(args.device)
if DEVICE.type == "cuda":
    torch.cuda.set_device(DEVICE)
MODES = (("int4", torch.uint8, KVQuantMode.INT4_PER_TOKEN_HEAD),
         ("int8", torch.int8, KVQuantMode.INT8_PER_TOKEN_HEAD),
         ("fp8", torch.uint8, KVQuantMode.FP8_PER_TOKEN_HEAD))


def cache_for(kind, dtype, layout, offset, head_size=64):
    packed = head_size // 2 if kind == "int4" else head_size
    content = 2 * (packed + 4)
    shape = (4, 2, 16, content)
    size = 4 * 2 * 16 * content
    # The pool includes prefix/suffix guards; nonzero offsets must be honoured.
    pool = torch.full((size // 4 + offset + 8,), 0.25, dtype=torch.float32, device=DEVICE)
    strides = ((2 * 16 * content, 16 * content, content, 1) if layout == "HND"
               else (16 * 2 * content, content, 2 * content, 1))
    cache = torch.as_strided(pool.view(dtype), shape, strides, storage_offset=offset * 4)
    return pool, cache


def implementation(cls, mode, head_size=64):
    obj = cls.__new__(cls)
    obj._k_scale_cache = obj._v_scale_cache = None
    obj._kv_quant_mode = mode
    obj.fp8_dtype = torch.float8_e4m3fn
    obj.head_size = head_size
    if cls is PthFlashAttnImpl:
        obj._cache_views = None
    return obj


class ScaleViewTests(unittest.TestCase):
    def test_first_triton_views_preserve_shared_pool(self):
        for kind, dtype, mode in MODES:
            for layout in ("HND", "NHD"):
                for offset in (0, 8):
                    with self.subTest(kind=kind, layout=layout, offset=offset):
                        pool, cache = cache_for(kind, dtype, layout, offset)
                        before = pool.view(torch.uint8).clone()
                        obj = implementation(TritonAttentionImpl, mode)
                        obj._pth_key_value_caches(cache)
                        torch.testing.assert_close(pool.view(torch.uint8), before, rtol=0, atol=0)

    def test_first_pth_plugin_views_preserve_shared_pool(self):
        self.assertIs(PthFlashAttnImpl._ensure_scale_caches, TritonAttentionImpl._ensure_scale_caches)
        for layout in ("HND", "NHD"):
            for offset in (0, 8):
                with self.subTest(layout=layout, offset=offset):
                    pool, cache = cache_for("int8", torch.int8, layout, offset)
                    before = pool.view(torch.uint8).clone()
                    obj = implementation(PthFlashAttnImpl, KVQuantMode.INT8_PER_TOKEN_HEAD)
                    views = obj._caches(cache)
                    self.assertEqual(tuple(views[0].shape), (4, 16, 2, 64))
                    self.assertIs(obj._caches(cache), views)
                    torch.testing.assert_close(pool.view(torch.uint8), before, rtol=0, atol=0)

    def test_new_layer_alias_preserves_existing_scales(self):
        for kind, dtype, mode in MODES:
            for layout in ("HND", "NHD"):
                with self.subTest(kind=kind, layout=layout):
                    pool, cache = cache_for(kind, dtype, layout, 8)
                    first = implementation(TritonAttentionImpl, mode)
                    first._pth_key_value_caches(cache)
                    first._k_scale_cache.fill_(0.375)
                    first._v_scale_cache.fill_(0.625)
                    before = pool.view(torch.uint8).clone()
                    second = implementation(TritonAttentionImpl, mode)
                    second._pth_key_value_caches(cache)
                    torch.testing.assert_close(pool.view(torch.uint8), before, rtol=0, atol=0)

    def test_scale_views_alias_only_the_intended_inline_bytes(self):
        # A detached replacement would pass preservation but fail this check.
        for kind, dtype, mode in MODES:
            for layout in ("HND", "NHD"):
                with self.subTest(kind=kind, layout=layout):
                    pool, cache = cache_for(kind, dtype, layout, 8)
                    obj = implementation(TritonAttentionImpl, mode)
                    obj._pth_key_value_caches(cache)
                    expected = pool.view(torch.uint8).clone()
                    packed = 32 if kind == "int4" else 64
                    for scale, index, value in ((obj._k_scale_cache, packed, 2.5),
                                                (obj._v_scale_cache, 2 * packed + 4, 0.125)):
                        address = cache[2, 1, 3, index:index + 4].data_ptr() - pool.data_ptr()
                        raw = torch.tensor([value], dtype=torch.float32, device=DEVICE).view(torch.uint8)
                        expected[address:address + 4].copy_(raw)
                        scale[2, 3, 1] = value
                    torch.testing.assert_close(pool.view(torch.uint8), expected, rtol=0, atol=0)

    @unittest.skipUnless(DEVICE.type == "cuda", "GPU cache-writer check; CPU view checks run above")
    def test_gpu_int8_writer_initializes_valid_slots(self):
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_per_token_head_quant,
        )
        for head_size in (64, 256):
            for layout in ("HND", "NHD"):
                with self.subTest(head_size=head_size, layout=layout):
                    _, cache = cache_for("int8", torch.int8, layout, 8, head_size)
                    obj = implementation(TritonAttentionImpl, KVQuantMode.INT8_PER_TOKEN_HEAD, head_size)
                    key_cache, value_cache = obj._pth_key_value_caches(cache)
                    generator = torch.Generator(device=DEVICE).manual_seed(17)
                    key = torch.randn((3, 2, head_size), generator=generator, device=DEVICE, dtype=torch.float16)
                    value = torch.randn((3, 2, head_size), generator=generator, device=DEVICE, dtype=torch.float16)
                    slots = torch.tensor([0, 17, -1], dtype=torch.int64, device=DEVICE)
                    triton_reshape_and_cache_flash_per_token_head_quant(
                        key, value, key_cache, value_cache, obj._k_scale_cache, obj._v_scale_cache, slots,
                        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                    )
                    for row, block, slot in ((0, 0, 0), (1, 1, 1)):
                        for original, quantized, scales in ((key, key_cache, obj._k_scale_cache),
                                                            (value, value_cache, obj._v_scale_cache)):
                            scale = scales[block, slot]
                            self.assertTrue(bool(torch.all(torch.isfinite(scale) & (scale > 0))))
                            restored = quantized[block, slot, :, :head_size].float() * scale[:, None]
                            torch.testing.assert_close(restored, original[row].float(), rtol=0, atol=0.04)
                    torch.testing.assert_close(obj._k_scale_cache[0, 1], torch.full((2,), 0.25, device=DEVICE))


if __name__ == "__main__":
    unittest.main(argv=[__file__, *remaining])

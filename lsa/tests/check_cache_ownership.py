# SPDX-License-Identifier: Apache-2.0
"""CPU page-sharing invariants; --device cuda also checks actual sparse-read isolation."""
import argparse
import os
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from hyperqwen_lsa import core

DEVICE = torch.device("cpu")


class CacheOwnershipTests(unittest.TestCase):
    def fork(self, block_size, total=None):
        torch.manual_seed(483729)
        end = (block_size // core.CLUSTER_BLOCK + 1) * core.CLUSTER_BLOCK
        total = total or end
        # Generous independent storage keeps this isolation test applicable to
        # both row-address implementations; it does not assume where rows go.
        pages = 5
        rows = pages * ((block_size + 511) // 512) * 64
        tables = core.LayerTables(
            torch.zeros(pages * block_size + 1, 64, dtype=torch.bfloat16, device=DEVICE),
            torch.zeros(rows + 1, 64, dtype=torch.bfloat16, device=DEVICE),
            torch.full((rows + 1, 16), -1, dtype=torch.int16, device=DEVICE))
        a = torch.tensor([[1, 2]], dtype=torch.int32, device=DEVICE)
        b = torch.tensor([[1, 3]], dtype=torch.int32, device=DEVICE)
        prefix = F.normalize(torch.randn(block_size, 64), dim=-1).to(DEVICE, torch.bfloat16)
        for bt, shift in ((a, 0), (b, 2)):
            suffix = F.normalize(torch.randn(total - block_size, 64) + shift, dim=-1)
            keys = torch.cat((prefix, suffix.to(DEVICE, torch.bfloat16)))
            positions = torch.arange(total, device=DEVICE)
            slots = core.slots_of(bt, torch.zeros_like(positions), positions, block_size)
            core.write_index_keys(tables, keys, slots)
        starts = torch.arange(0, end, 512, dtype=torch.long, device=DEVICE)
        request = torch.zeros_like(starts)
        valid = torch.ones_like(starts, dtype=torch.bool)
        return tables, a, b, request, starts, valid

    def test_fork_does_not_mutate_cached_metadata(self):
        for bs in (448, 816, 864, 880):
            with self.subTest(block_size=bs):
                tables, a, b, req, starts, valid = self.fork(bs)
                core.cluster_blocks(tables, a, req, starts, valid, bs)
                owned = tables.members.ne(-1).any(1)
                centroids = tables.centroids[owned].clone()
                members = tables.members[owned].clone()
                core.cluster_blocks(tables, b, req, starts, valid, bs)
                self.assertTrue(torch.equal(tables.centroids[owned], centroids), "another request changed cached centroids")
                self.assertTrue(torch.equal(tables.members[owned], members), "another request changed cached members")

    def test_bucket_group_belongs_to_final_key_page(self):
        for bs in (16, 64, 448, 512, 816, 832, 864, 880, 1024, 1536):
            with self.subTest(block_size=bs):
                pages = 5120 // bs + 2
                bt = torch.randperm(pages, device=DEVICE)[None].int()
                positions = torch.arange(0, 5120, 8, device=DEVICE)
                rows = core.bucket_rows(bt, torch.zeros_like(positions), positions, bs)
                per_page = core.bucket_rows_per_page(bs)
                self.assertEqual(rows.unique().numel(), rows.numel())
                self.assertTrue(bool((rows < pages * per_page).all()))
                for block in range(10):
                    last = (block + 1) * 512 - 1
                    owner = int(bt[0, last // bs])
                    group = rows[block * 64:(block + 1) * 64]
                    self.assertTrue(bool((group // per_page == owner).all()))
                    self.assertTrue(torch.equal(group[1:] - group[:-1], torch.ones_like(group[1:])))

    def test_shared_complete_cluster_keeps_same_rows_and_values(self):
        tables, a, b, req, starts, valid = self.fork(816)
        anchors = torch.arange(0, 512, 8, device=DEVICE)
        ra = core.bucket_rows(a, torch.zeros_like(anchors), anchors, 816)
        rb = core.bucket_rows(b, torch.zeros_like(anchors), anchors, 816)
        self.assertTrue(torch.equal(ra, rb))
        core.cluster_blocks(tables, a, req, starts, valid, 816)
        before = tables.centroids[ra].clone()
        core.cluster_blocks(tables, b, req, starts, valid, 816)
        self.assertTrue(torch.equal(tables.centroids[ra], before))

    def test_allocation_and_accounting_include_page_rounding(self):
        for bs in (16, 448, 512, 816, 864, 1024):
            with self.subTest(block_size=bs):
                slots = 5 * bs
                table = core.LayerTables.allocate(slots, DEVICE, bs)
                rows = 5 * ((bs + 511) // 512) * 64
                self.assertEqual(table.centroids.shape, (rows + 1, 64))
                self.assertEqual(table.members.shape, (rows + 1, 16))
                measured = sum(x.numel() * x.element_size() for x in
                               (table.index_keys, table.centroids, table.members))
                self.assertAlmostEqual(measured - 288, slots * table.bytes_per_slot(bs), places=6)

    def test_padding_buckets_do_not_index_incomplete_cluster_pages(self):
        table = core.LayerTables.allocate(32, DEVICE, 16)
        bt = torch.tensor([[1]], dtype=torch.int32, device=DEVICE)
        query = torch.tensor([0], device=DEVICE)
        scores = torch.full((1, 128), -torch.inf, device=DEVICE)
        _, counts = core.expand(table, bt, query, query + 1, query,
                                scores, 16, core.Design(top_p=64, top_k=4))
        self.assertEqual(counts.tolist(), [0])

    def test_layout_negotiation_requires_layer_compact_nhd(self):
        os.environ["LSA_BASE"] = "triton"
        from hyperqwen_lsa.backend import LsaAttentionBackend
        from vllm.v1.attention.backends.utils import get_supported_kv_cache_layouts
        from vllm.v1.kv_cache_interface import KVCacheLayout
        self.assertEqual(get_supported_kv_cache_layouts([LsaAttentionBackend]), [KVCacheLayout.LBNHC])

    def test_gpu_fork_preserves_sparse_read_outputs(self):
        if DEVICE.type != "cuda":
            self.skipTest("actual sparse-read kernels require --device cuda")
        from hyperqwen_lsa import read
        bs, total = 816, 1408
        tables, a, b, req, starts, valid = self.fork(bs, total)
        core.cluster_blocks(tables, a, req, starts, valid, bs)
        key = torch.randn(5, bs, 4, 256, device=DEVICE, dtype=torch.bfloat16)
        value = torch.randn_like(key)
        query = torch.randn(4, 24, 256, device=DEVICE, dtype=torch.bfloat16)
        wq = torch.randn(768, 64, device=DEVICE)
        qpos = torch.tensor([1300, 1333, 1377, 1407], device=DEVICE)
        qsl = torch.tensor([0, 4], dtype=torch.int32, device=DEVICE)
        token_req = torch.zeros(4, dtype=torch.long, device=DEVICE)
        end = torch.full((4,), 1024, device=DEVICE)
        design = core.Design(top_k=32, top_p=16, window=16)

        def outputs():
            result = []
            for split in (1, 16):
                out = torch.empty_like(query)
                read.lsa_attention(query, wq, 4, 64, tables, key, value, None, None, out,
                                   qsl, a, token_req, qpos, end, 1, 4, 177, 256 ** -0.5,
                                   design, chunk=4, num_splits=split)
                self.assertTrue(bool(torch.isfinite(out).all()))
                result.append(out.clone())
            return result

        before = outputs()
        core.cluster_blocks(tables, b, req, starts, valid, bs)
        after = outputs()
        for x, y in zip(before, after):
            self.assertTrue(torch.equal(x, y), "another request changed the existing request's sparse output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cpu")
    args, rest = parser.parse_known_args()
    DEVICE = torch.device(args.device)
    torch.set_num_threads(2)
    unittest.main(argv=[sys.argv[0], *rest])

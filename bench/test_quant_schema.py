#!/usr/bin/env python3
"""CPU checks for what prepare/quant_schema.py writes for every in-place quant script: the
packed tensors, the killed-run check, the index entries and the config group.

  venv/bin/python bench/test_quant_schema.py   # needs torch and compressed-tensors
"""
import sys
import unittest
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized.base import unpack_from_int32

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "prepare"))
from quant_schema import GROUP, GROUPS, MTP_LINEARS, clone_group, index_packed, pack, packed_already


class QuantSchemaTests(unittest.TestCase):
    def weight(self):
        w = (torch.randn(10, 2 * GROUP, generator=torch.Generator().manual_seed(0)) * 0.02).to(torch.bfloat16)
        w[0, :GROUP] = 0
        return w

    def test_pack_rounds_each_group_to_nearest(self):
        w = self.weight()
        for bits in (8, 4):
            qmax = 2 ** (bits - 1) - 1
            out, err = pack("x", w, bits, torch.float32, rows=3)
            self.assertEqual(list(out), ["x.weight_packed", "x.weight_scale", "x.weight_shape"])
            self.assertEqual(out["x.weight_shape"].tolist(), [10, 2 * GROUP])
            q = unpack_from_int32(out["x.weight_packed"], bits, torch.Size([10, 2 * GROUP])).float()
            s = out["x.weight_scale"].repeat_interleave(GROUP, dim=1)
            self.assertLessEqual(q.abs().max().item(), qmax)
            self.assertTrue(((q * s - w.float()).abs() <= s / 2 + 1e-9).all(), bits)
            self.assertEqual(out["x.weight_scale"][0, 0].item(), torch.tensor(1e-10).item())
            self.assertEqual(q[0, :GROUP].abs().sum().item(), 0)
            self.assertAlmostEqual(err, ((q * s - w.float()).norm() / w.float().norm()).item(), places=6)

    def test_pack_bytes_do_not_depend_on_rows(self):
        w = self.weight()
        whole, _ = pack("x", w, 4, torch.float16)
        for k, t in pack("x", w, 4, torch.float16, rows=3)[0].items():
            self.assertTrue(torch.equal(t, whole[k]), k)

    def test_pack_of_an_all_zero_weight_reports_nan(self):
        _, err = pack("x", torch.zeros(2, GROUP, dtype=torch.bfloat16), 8, torch.float16)
        self.assertNotEqual(err, err)

    def test_pack_keeps_the_scale_dtype_it_is_given(self):
        for dt in (torch.float16, torch.bfloat16):
            out, _ = pack("x", self.weight(), 8, dt)
            self.assertEqual(out["x.weight_scale"].dtype, dt)
            self.assertEqual(out["x.weight_packed"].dtype, torch.int32)

    def test_packed_already(self):
        packed = [f"x.{s}" for s in ("weight_packed", "weight_scale", "weight_shape")]
        self.assertFalse(packed_already({"x.weight", "y"}, "x", "s.safetensors", ".bak"))
        self.assertTrue(packed_already(set(packed), "x", "s.safetensors", ".bak"))
        for names in (set(), set(packed[:2])):
            with self.assertRaises(SystemExit) as e:
                packed_already(names, "x", "s.safetensors", ".bak-mtp")
            self.assertIn("restore it from s.safetensors.bak-mtp", str(e.exception.code))

    def test_index_packed_replaces_the_weight_entry(self):
        wm = {"a": "1", "x.weight": "2", "b": "1"}
        index_packed(wm, "x", "3")
        want = {"a": "1", "b": "1", "x.weight_packed": "3", "x.weight_scale": "3", "x.weight_shape": "3"}
        self.assertEqual(list(wm.items()), list(want.items()))
        index_packed(wm, "x", "3")
        self.assertEqual(list(wm.items()), list(want.items()))

    def test_clone_group_declares_symmetric_whatever_group_0_says(self):
        awq = {"num_bits": 4, "type": "int", "symmetric": False, "group_size": 32, "strategy": "group",
               "zp_dtype": "torch.int8"}
        qc = {"config_groups": {"group_0": {"targets": ["Linear"], "weights": dict(awq)}}}
        want = {"lm_head": ("group_1", ["re:.*lm_head$"]), "embed": ("group_2", ["re:.*embed_tokens$"]),
                "mtp": ("group_3", ["re:^mtp\\..*"]), "mtp-keep-fc": ("group_3", ["re:^mtp\\.layers\\..*"])}
        for part, bits in (("lm_head", 8), ("embed", 8), ("mtp", 8), ("mtp-keep-fc", 4)):
            clone_group(qc, part, bits)
            name, targets = want[part]
            g = qc["config_groups"][name]
            self.assertEqual(g["targets"], targets)
            self.assertEqual(g["weights"], {"num_bits": bits, "type": "int", "symmetric": True,
                                            "group_size": GROUP, "strategy": "group", "zp_dtype": None})
            g["targets"].append("mutated")
            self.assertNotIn("mutated", GROUPS[part][1])
        self.assertEqual(qc["config_groups"]["group_0"]["weights"], awq)

    def test_mtp_linears(self):
        self.assertEqual(MTP_LINEARS, ("mtp.fc", *(f"mtp.layers.0.{m}" for m in (
            "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj",
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"))))


if __name__ == "__main__":
    unittest.main()

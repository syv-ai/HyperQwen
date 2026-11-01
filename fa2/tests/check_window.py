"""Check noncausal sliding-window graph replay with discarded KV pages."""
import argparse
import math
import json
import torch
from fa2_fp8kv_vllm import paths

parser = argparse.ArgumentParser()
parser.add_argument("library", nargs="?", default=str(paths.library()))
args = parser.parse_args()
torch.ops.load_library(args.library)
torch.cuda.set_device(0)
torch.set_float32_matmul_precision("highest")
torch.manual_seed(97)
with torch.inference_mode():
    query = torch.randn((8, 16, 128), device="cuda", dtype=torch.bfloat16)
    keys = torch.randn((132, 16, 4, 128), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    values = torch.randn_like(keys.float()).to(torch.float8_e4m3fn)
    keys[0] = float("nan")
    values[0] = float("nan")
    starts = torch.tensor([0, 8], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([2055], device="cuda", dtype=torch.int32)
    table = torch.zeros((1, 16384), device="cuda", dtype=torch.int32)
    scale_k = torch.tensor([0.5], device="cuda")
    scale_v = torch.tensor([1.75], device="cuda")
    for splits in (1, 32, 128):
        output = torch.empty_like(query)
        graph = torch.cuda.CUDAGraph()
        for _ in range(3):
            torch.ops.fa2_fp8kv.forward(query, keys, values, output, starts, lengths,
                table, scale_k, scale_v, 8, 262144, False, 2047, -1, 128**-0.5, splits, False)
        with torch.cuda.graph(graph):
            _, lse = torch.ops.fa2_fp8kv.forward(query, keys, values, output, starts, lengths,
                table, scale_k, scale_v, 8, 262144, False, 2047, -1, 128**-0.5, splits, False)
        for length in (2055, 32771, 262143, 2055):
            first_block = max(0, (length - 8 - 2047) // 16)
            blocks = math.ceil(length / 16) - first_block
            table.zero_()
            table[0, first_block:first_block + blocks] = torch.arange(1, blocks + 1, device="cuda", dtype=torch.int32)
            lengths.fill_(length)
            query.normal_()
            graph.replay()
            torch.cuda.synchronize()
            count = length - first_block * 16
            k = keys[1:blocks + 1].float().reshape(-1, 4, 128)[:count].repeat_interleave(4, dim=1) * 0.5
            v = values[1:blocks + 1].float().reshape(-1, 4, 128)[:count].repeat_interleave(4, dim=1) * 1.75
            scores = torch.einsum("qhd,khd->hqk", query.float(), k) * 128**-0.5
            positions = torch.arange(first_block * 16, length, device="cuda")
            query_positions = torch.arange(length - 8, length, device="cuda")
            scores.masked_fill_(positions[None, None] < query_positions[None, :, None] - 2047, -torch.inf)
            expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v).bfloat16()
            torch.testing.assert_close(output, expected, atol=0.025, rtol=0.025)
            torch.testing.assert_close(lse, scores.logsumexp(-1), atol=0.025, rtol=0.005)
            print(json.dumps({"splits": splits, "length": length, "result": "PASS"}), flush=True)
        graph.reset()

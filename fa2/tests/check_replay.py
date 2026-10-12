"""Check ragged batches and changing metadata in captured FA2 graphs."""

import argparse
import json
import math

import torch
from fa2_fp8kv_vllm import paths

parser = argparse.ArgumentParser()
parser.add_argument("library", nargs="?", default=str(paths.library()))
args = parser.parse_args()
torch.ops.load_library(args.library)
torch.cuda.set_device(0)
torch.cuda.set_per_process_memory_fraction(0.20)
torch.set_float32_matmul_precision("highest")
torch.manual_seed(83)

with torch.inference_mode():
    query = torch.randn((5, 8, 256), device="cuda", dtype=torch.bfloat16)
    # Noncontiguous block strides with aligned, contiguous rows.
    keys = torch.randn((9, 2, 16, 1, 256), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)[:, 0]
    values = torch.randn((9, 2, 16, 1, 256), device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)[:, 0]
    table = torch.tensor([[2, 1, 0], [5, 4, 3], [8, 7, 6]], device="cuda", dtype=torch.int32)
    cu_q = torch.tensor([0, 1, 5, 5], device="cuda", dtype=torch.int32)
    seq_k = torch.tensor([17, 31, 0], device="cuda", dtype=torch.int32)
    k_scale = torch.tensor([0.5], device="cuda")
    v_scale = torch.tensor([1.75], device="cuda")
    for grouped in (False, True):
        for splits in (1, 4, 128):
            guarded = torch.full((7, 2, 8, 256), 11, device="cuda", dtype=torch.bfloat16)
            output = guarded[1:-1, 0]
            for _ in range(3):
                torch.ops.fa2_fp8kv.forward(
                    query, keys, values, output, cu_q, seq_k, table, k_scale, v_scale,
                    4, 48, True, -1, -1, 1 / 16, splits, grouped,
                )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                _, captured_lse = torch.ops.fa2_fp8kv.forward(
                    query, keys, values, output, cu_q, seq_k, table, k_scale, v_scale,
                    4, 48, True, -1, -1, 1 / 16, splits, grouped,
                )
            for lengths, starts, rotate in [
                ([17, 31, 0], [0, 1, 5, 5], False),
                ([23, 37, 0], [0, 3, 5, 5], True),
                ([17, 31, 0], [0, 1, 5, 5], False),
            ]:
                cu_q.copy_(torch.tensor(starts, device="cuda", dtype=torch.int32))
                seq_k.copy_(torch.tensor(lengths, device="cuda", dtype=torch.int32))
                pages = [[0, 2, 1], [4, 3, 5], [8, 7, 6]] if rotate else [[2, 1, 0], [5, 4, 3], [8, 7, 6]]
                table.copy_(torch.tensor(pages, device="cuda", dtype=torch.int32))
                query.normal_()
                graph.replay()
                torch.cuda.synchronize()
                expected, expected_lse = [], []
                for row in range(2):
                    n = lengths[row]
                    count = starts[row + 1] - starts[row]
                    ids = table[row, :math.ceil(n / 16)].long()
                    k = keys.view(torch.uint8)[ids].view(torch.float8_e4m3fn).float().reshape(-1, 256)[:n] * 0.5
                    v = values.view(torch.uint8)[ids].view(torch.float8_e4m3fn).float().reshape(-1, 256)[:n] * 1.75
                    q = query[starts[row]:starts[row + 1]].float()
                    scores = torch.einsum("qhd,kd->hqk", q, k) / 16
                    mask = torch.arange(n, device="cuda")[None] > (
                        n - count + torch.arange(count, device="cuda")[:, None]
                    )
                    scores.masked_fill_(mask[None], -torch.inf)
                    expected.append(torch.einsum("hqk,kd->qhd", scores.softmax(-1), v))
                    expected_lse.append(scores.logsumexp(-1))
                reference = torch.cat(expected).bfloat16()
                torch.testing.assert_close(output, reference, atol=0.025, rtol=0.025)
                torch.testing.assert_close(captured_lse, torch.cat(expected_lse, dim=1), atol=0.025, rtol=0.005)
                assert bool((guarded[[0, -1]] == 11).all())
                assert bool((guarded[:, 1] == 11).all())
            graph.reset()
            print(json.dumps({"grouped": grouped, "splits": splits, "replays": 3, "result": "PASS"}), flush=True)

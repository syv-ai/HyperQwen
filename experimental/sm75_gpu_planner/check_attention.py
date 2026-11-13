"""Test overprovisioned CPU plans with exact GPU attention metadata."""

from pathlib import Path
import json, torch, argparse
from hq_gpu_plan import update
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

BASE = Path(__file__).resolve().parent
torch.manual_seed(42)
ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", type=Path, default=Path("/tmp/hq-gpu-plan-tests"))
args = ap.parse_args()
BASE = args.output_dir
BASE.mkdir(parents=True, exist_ok=True)
label = "gpu-plan-exact"
device = "cuda"
page = 32
max_pages = 1700
results = []
for causal, window, nheads, nkv, hd in [
    (True, -1, 12, 2, 256),
    (False, 2047, 16, 4, 128),
]:
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    cache = (
        torch.randn(max_pages, 2, page, nkv, hd, dtype=torch.float16, device=device)
        * 0.2
    )
    for batch in [1, 2]:
        q = torch.randn(batch * 8, nheads, hd, dtype=torch.float16, device=device)
        qo = torch.arange(batch + 1, dtype=torch.int32) * 8
        buffers = {
            "qo": torch.empty(batch + 1, dtype=torch.int32, device=device),
            "ptr": torch.empty(batch + 1, dtype=torch.int32, device=device),
            "idx": torch.empty(max_pages, dtype=torch.int32, device=device),
            "last": torch.empty(batch, dtype=torch.int32, device=device),
        }
        graph_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=buffers["qo"],
            paged_kv_indptr_buf=buffers["ptr"],
            paged_kv_indices_buf=buffers["idx"],
            paged_kv_last_page_len_buf=buffers["last"],
            backend="fa2",
        )
        eager = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            use_cuda_graph=True,
            qo_indptr_buf=torch.empty_like(buffers["qo"]),
            paged_kv_indptr_buf=torch.empty_like(buffers["ptr"]),
            paged_kv_indices_buf=torch.empty_like(buffers["idx"]),
            paged_kv_last_page_len_buf=torch.empty_like(buffers["last"]),
            backend="fa2",
        )
        permutation = torch.randperm(max_pages, device=device, dtype=torch.int32)

        def metadata(lengths):
            lengths = torch.tensor(lengths, dtype=torch.int32)
            counts = (lengths + page - 1) // page
            ptr = torch.cat([torch.zeros(1, dtype=torch.int32), counts.cumsum(0).int()])
            last = (lengths - 1) % page + 1
            return ptr, last, permutation[: int(ptr[-1])]

        def plan(wrapper, lengths):
            ptr, last, idx = metadata(lengths)
            wrapper.plan(
                qo,
                ptr,
                idx,
                last,
                nheads,
                nkv,
                hd,
                page,
                causal=causal,
                window_left=window,
                q_data_type=torch.float16,
                kv_data_type=torch.float16,
                seq_lens=torch.tensor(lengths, dtype=torch.int32),
                disable_split_kv=False,
                fixed_split_size=-1,
            )

        plan(graph_wrapper, [8] * batch)
        output = torch.empty_like(q)
        for _ in range(3):
            graph_wrapper.run(q, cache, out=output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_wrapper.run(q, cache, out=output)
        for n in [
            8,
            17,
            31,
            32,
            33,
            65,
            1024,
            2047,
            2048,
            2049,
            4095,
            4096,
            4097,
            8191,
            8192,
            8193,
            16383,
            16384,
            16385,
            23700,
            65,
            8,
        ]:
            actual = [n] if batch == 1 else [n, max(8, n // 2 + 3)]
            # Include both within-page and cross-page overestimates.
            for slack in [0, 7, 15, 63]:
                upper = [
                    v + slack for v in actual
                ]  # no CPU replan after initialization
                ptr, last, idx = metadata(actual)
                buffers["ptr"].copy_(ptr)
                buffers["last"].copy_(last)
                buffers["idx"][: idx.numel()].copy_(idx)
                graph_wrapper._kv_lens_buffer[:batch].copy_(
                    torch.tensor(actual, dtype=torch.int32)
                )
                update(
                    graph_wrapper,
                    torch.tensor(actual, device=device, dtype=torch.int32),
                    8,
                    page,
                    nheads,
                    nkv,
                    window,
                )
                graph.replay()
                got = output.clone()
                plan(eager, actual)
                expected = eager.run(q, cache)
                error = float((got - expected).abs().max())
                same = torch.equal(got, expected)
                row = {
                    "causal": causal,
                    "window": window,
                    "batch": batch,
                    "lengths": actual,
                    "slack": slack,
                    "max_abs_error": error,
                    "pass": same,
                }
                results.append(row)
                if not same:
                    (BASE / ("bounded-plan-" + label + ".json")).write_text(
                        json.dumps(results, indent=2)
                    )
                    raise AssertionError(row)
        print(
            "PASS",
            causal,
            window,
            batch,
            "changing-length cases completed; cumulative count",
            len(results),
            flush=True,
        )
    del workspace, cache, q, output, graph_wrapper, eager, graph, buffers, got, expected
    torch.cuda.empty_cache()
(BASE / ("bounded-plan-" + label + ".json")).write_text(json.dumps(results, indent=2))
print("COMPLETE", len(results), flush=True)

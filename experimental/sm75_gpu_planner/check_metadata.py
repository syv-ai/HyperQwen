"""Compare GPU scheduler integer arrays to native CPU planner through full capacity."""

from pathlib import Path
import torch, json
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from hq_gpu_plan import update

BASE = Path("/tmp/hq-gpu-plan-tests")
BASE.mkdir(parents=True, exist_ok=True)
results = []
for page in [16, 32, 64, 128, 512]:
    for heads, kvheads, hd, window in [(12, 2, 256, -1), (16, 4, 128, 2047)]:
        ws = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        maxpages = 2 * ((262136 + page - 1) // page)
        for B in [1, 2]:

            def wrapper():
                return BatchPrefillWithPagedKVCacheWrapper(
                    ws,
                    "NHD",
                    use_cuda_graph=True,
                    qo_indptr_buf=torch.empty(B + 1, dtype=torch.int32, device="cuda"),
                    paged_kv_indptr_buf=torch.empty(
                        B + 1, dtype=torch.int32, device="cuda"
                    ),
                    paged_kv_indices_buf=torch.empty(
                        maxpages, dtype=torch.int32, device="cuda"
                    ),
                    paged_kv_last_page_len_buf=torch.empty(
                        B, dtype=torch.int32, device="cuda"
                    ),
                    backend="fa2",
                )

            gpu = wrapper()
            cpu = wrapper()
            idx = torch.zeros(maxpages, dtype=torch.int32, device="cuda")
            qo = torch.arange(B + 1, dtype=torch.int32) * 8

            def plan(w, lens):
                lens = torch.tensor(lens, dtype=torch.int32)
                count = (lens + page - 1) // page
                ptr = torch.cat(
                    [torch.zeros(1, dtype=torch.int32), count.cumsum(0).int()]
                )
                last = torch.where(lens > 0, (lens - 1) % page + 1, 0)
                w.plan(
                    qo,
                    ptr,
                    idx[: int(ptr[-1])],
                    last,
                    heads,
                    kvheads,
                    hd,
                    page,
                    causal=window < 0,
                    window_left=window,
                    q_data_type=torch.float16,
                    kv_data_type=torch.float16,
                    seq_lens=lens,
                )

            plan(gpu, [8] * B)
            cases = [
                [x] if B == 1 else [x, max(0, x // 2 - 1)]
                for x in [0, 8, 31, 32, 33, 1024, 8192, 23700, 65536, 262136]
            ]
            if B == 2:
                cases += [[262136, 262136], [8, 262136], [0, 262136], [262136, 0]]
            for lens in cases:
                plan(cpu, lens)
                assert gpu._plan_info == cpu._plan_info
                update(
                    gpu,
                    torch.tensor(lens, dtype=torch.int32, device="cuda"),
                    8,
                    page,
                    heads,
                    kvheads,
                    window,
                )
                info = gpu._plan_info
                gi = gpu._int_workspace_buffer.view(torch.int32)
                ci = cpu._int_workspace_buffer.view(torch.int32)
                gm = gpu._int_workspace_buffer[info[12] : info[12] + info[0]]
                cm = cpu._int_workspace_buffer[info[12] : info[12] + info[0]]
                assert torch.equal(gm, cm), ("mask", page, B, lens)
                valid = cm.bool()
                for field in [4, 5, 6]:
                    offset = info[field] // 4
                    assert torch.equal(
                        gi[offset : offset + info[0]][valid],
                        ci[offset : offset + info[0]][valid],
                    ), ("tiles", field, page, B, lens)
                for field, size in [(7, B * 8 + 1), (8, B + 1), (9, 1), (2, 1)]:
                    offset = info[field] // 4
                    assert torch.equal(
                        gi[offset : offset + size], ci[offset : offset + size]
                    ), ("ptr/chunk", field, page, B, lens)
                results.append(
                    {
                        "page": page,
                        "heads": heads,
                        "kv_heads": kvheads,
                        "window": window,
                        "batch": B,
                        "lengths": lens,
                        "pass": True,
                    }
                )
            del gpu, cpu, idx
        del ws
        torch.cuda.empty_cache()
        print("PASS metadata page", page, "geometry", heads, kvheads, hd, flush=True)
(BASE / "metadata-results.json").write_text(json.dumps(results, indent=2))
print("COMPLETE", len(results), flush=True)

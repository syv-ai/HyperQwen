# Copyright (c) 2023 by FlashInfer team.
# Copyright (c) 2026 HyperQwen contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Experimental native-FA2 work planner for uniform speculative queries, batch<=2.
Integer scheduling mirrors FlashInfer PrefillSplitQOKVIndptr (Apache-2.0).
"""

import torch
from vllm.triton_utils import triton, tl


@triton.jit
def _plan(
    seq,
    iw,
    raw,
    ptr,
    last,
    kv_lens,
    B: tl.constexpr,
    Q: tl.constexpr,
    PAGE: tl.constexpr,
    GQA: tl.constexpr,
    TILEQ: tl.constexpr,
    MAX_SPLIT: tl.constexpr,
    PAD: tl.constexpr,
    WINDOW: tl.constexpr,
    REQ: tl.constexpr,
    QT: tl.constexpr,
    KT: tl.constexpr,
    MERGE: tl.constexpr,
    OUT: tl.constexpr,
    CHUNK: tl.constexpr,
    ROWS: tl.constexpr,
    MASK: tl.constexpr,
    S: tl.constexpr,
    A: tl.constexpr,
    R: tl.constexpr,
):
    requests = tl.arange(0, S)
    length = tl.load(seq + requests, requests < B, 0)
    pages = (length + PAGE - 1) // PAGE
    effective = pages
    if WINDOW >= 0:
        effective = tl.minimum(pages, (WINDOW + TILEQ + PAGE - 1) // PAGE)
    tiles = (Q * GQA + TILEQ - 1) // TILEQ
    low = tl.full((), max(128 // PAGE, 1), tl.int32)
    high = tl.maximum(tl.max(effective, 0), 1)
    while low < high:
        mid = (low + high) // 2
        work = tl.sum(
            tl.where(
                requests < B, tiles * ((tl.maximum(effective, 1) + mid - 1) // mid), 0
            ),
            0,
        )
        low = tl.where(work > MAX_SPLIT, mid + 1, low)
        high = tl.where(work > MAX_SPLIT, high, mid)
    chunks = (tl.maximum(effective, 1) + low - 1) // low
    slot = tl.arange(0, A)
    tl.store(iw + REQ + slot, 0, slot < PAD)
    tl.store(iw + QT + slot, 0, slot < PAD)
    tl.store(iw + KT + slot, 0, slot < PAD)
    work_base = tl.full((), 0, tl.int32)
    out_base = tl.full((), 0, tl.int32)
    page_base = tl.full((), 0, tl.int32)
    rows = tl.arange(0, R)
    for r in tl.static_range(B):
        n = tl.sum(tl.where(requests == r, chunks, 0), 0)
        npages = tl.sum(tl.where(requests == r, pages, 0), 0)
        nlen = tl.sum(tl.where(requests == r, length, 0), 0)
        own = (slot >= work_base) & (slot < work_base + tiles * n) & (slot < PAD)
        local = slot - work_base
        tl.store(iw + REQ + slot, r, own)
        tl.store(iw + QT + slot, local // n, own)
        tl.store(iw + KT + slot, local % n, own)
        tl.store(iw + OUT + r, out_base)
        tl.store(iw + MERGE + r * Q + rows, out_base + rows * n, rows < Q)
        tl.store(ptr + r, page_base)
        tl.store(last + r, tl.where(nlen > 0, (nlen - 1) % PAGE + 1, 0))
        tl.store(kv_lens + r, nlen)
        work_base += tiles * n
        out_base += Q * n
        page_base += npages
    tl.store(iw + OUT + B, out_base)
    tl.store(iw + MERGE + B * Q, out_base)
    tl.store(ptr + B, page_base)
    tl.store(iw + CHUNK, low * PAGE)
    tl.store(iw + ROWS, B * Q)
    tl.store(raw + MASK + slot, slot < work_base, slot < PAD)


def update(wrapper, lengths, query_width, page_size, heads, kv_heads, window):
    info = wrapper._plan_info
    assert len(info) == 15 and info[13] and info[14], info
    B = wrapper._batch_size
    assert B in (1, 2)
    _plan[(1,)](
        lengths,
        wrapper._int_workspace_buffer.view(torch.int32),
        wrapper._int_workspace_buffer,
        wrapper._paged_kv_indptr_buf,
        wrapper._paged_kv_last_page_len_buf,
        wrapper._kv_lens_buffer,
        B,
        query_width,
        page_size,
        heads // kv_heads,
        info[3],
        2
        * torch.cuda.get_device_properties(lengths.device).multi_processor_count
        // kv_heads,
        info[0],
        window,
        info[4] // 4,
        info[5] // 4,
        info[6] // 4,
        info[7] // 4,
        info[8] // 4,
        info[9] // 4,
        info[2] // 4,
        info[12],
        triton.next_power_of_2(B),
        triton.next_power_of_2(info[0]),
        triton.next_power_of_2(query_width),
        num_warps=4,
    )

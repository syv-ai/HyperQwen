"""Correctness tests for the LABD kernels against a plain-Python reference.

  venv/bin/python bench/test_lookup_kernels.py    # needs a GPU with ~200 MB free
"""
import sys

import torch

# run inside the vLLM venv: venv/bin/python bench/test_lookup_kernels.py
from vllm.v1.worker.gpu.spec_decode.dflash2.lookup import (  # noqa: E402
    fuse_draft,
    suffix_lookup,
)


def ref_lookup(seq, k, nmin, nmax):
    """(tokens, match_len, valid) for the suffix of `seq`, longest match then most recent."""
    t = len(seq) - 1
    best = (-1, -1)
    for e in range(nmin - 1, t):
        n = 0
        while n < nmax and e - n >= 0 and t - n >= 0 and seq[e - n] == seq[t - n]:
            n += 1
        if n >= nmin and (n, e) > best:
            best = (n, e)
    if best[0] < 0:
        return [0] * k, 0, 0
    n, e = best
    valid = min(k, t - e)
    toks = list(seq[e + 1 : e + 1 + valid]) + [0] * (k - valid)
    return toks, n, valid


def run_case(seq, k, nmin, nmax, device="cuda"):
    max_len = len(seq) + 8
    all_ids = torch.zeros((4, max_len), dtype=torch.int32, device=device)
    all_ids[1, : len(seq)] = torch.tensor(seq, dtype=torch.int32, device=device)
    total = torch.zeros(4, dtype=torch.int32, device=device)
    total[1] = len(seq)
    # one request, mapped to request state 1; stride k as the DFlash speculators use
    idx_map = torch.full((1 * k,), -1, dtype=torch.int32, device=device)
    idx_map[:] = 1
    toks, mlen, valid = suffix_lookup(all_ids, total, idx_map, 1, k,
                                      idx_mapping_stride=k, nmax=nmax, nmin=nmin)
    return toks[0].tolist(), int(mlen[0]), int(valid[0])


def main():
    torch.manual_seed(0)
    fails = 0
    cases = []
    # random sequences over a small vocabulary, so matches are common
    for vocab, n in ((6, 200), (20, 500), (3, 120), (60, 900)):
        cases.append(torch.randint(1, vocab, (n,)).tolist())
    # a periodic pattern: the match must be allowed to overlap the suffix
    cases.append(([5, 6, 7, 8] * 30))
    # a verbatim quote that reappears near the end
    body = torch.randint(1, 50, (300,)).tolist()
    cases.append(body + body[40:80] )
    # no match at all
    cases.append(list(range(1, 200)))
    for k, nmin, nmax in ((7, 4, 32), (15, 4, 32), (31, 6, 12), (15, 8, 64), (7, 2, 8)):
        for ci, seq in enumerate(cases):
            got = run_case(seq, k, nmin, nmax)
            want = ref_lookup(seq, k, nmin, nmax)
            # tokens beyond `valid` are don't-care
            v = want[2]
            ok = got[1] == want[1] and got[2] == want[2] and got[0][:v] == want[0][:v]
            if not ok:
                fails += 1
                print(f"MISMATCH k={k} nmin={nmin} nmax={nmax} case={ci} len={len(seq)}")
                print(f"  got  mlen={got[1]} valid={got[2]} toks={got[0][:8]}")
                print(f"  want mlen={want[1]} valid={want[2]} toks={want[0][:8]}")
    print(f"suffix_lookup: {'OK' if not fails else str(fails) + ' FAILURES'}")

    # fuse: a strong match takes the block; a weak one only with drafter agreement
    k = 7
    dev = "cuda"
    for match_len, agree, expect in ((12, 0, True), (4, 0, False), (4, 2, True), (2, 7, False)):
        draft = torch.arange(100, 100 + k, dtype=torch.int64, device=dev).view(1, k)
        look = torch.arange(200, 200 + k, dtype=torch.int32, device=dev).view(1, k)
        look[0, :agree] = draft[0, :agree].to(torch.int32)
        use = torch.zeros((1, k), dtype=torch.int32, device=dev)
        hits = torch.zeros((), dtype=torch.int64, device=dev)
        fuse_draft(draft, look, torch.tensor([match_len], dtype=torch.int32, device=dev),
                   torch.tensor([k], dtype=torch.int32, device=dev), use,
                   torch.ones(k, dtype=torch.int32, device=dev), hits, 1, k,
                   idx_mapping_stride=k, nmin=4, nstrong=8, agree_min=2)
        took = bool(use.any().item())
        status = "ok" if took == expect else "FAIL"
        if took != expect:
            fails += 1
        print(f"fuse match_len={match_len:2d} agree={agree} -> took={took} ({status})")

    # Decoupled block: the drafter proposed only the first `draft_block` positions, so the
    # tail must always be marked used (it needs a point mass) and must take the lookup's
    # continuation when the two sources agree on the head.
    for match_len, agree, block, want_tail_lookup in ((12, 0, 3, True), (4, 0, 3, False),
                                                      (6, 3, 3, True), (4, 1, 3, False)):
        draft = torch.arange(100, 100 + k, dtype=torch.int64, device=dev).view(1, k)
        look = torch.arange(200, 200 + k, dtype=torch.int32, device=dev).view(1, k)
        look[0, :agree] = draft[0, :agree].to(torch.int32)
        use = torch.zeros((1, k), dtype=torch.int32, device=dev)
        hits = torch.zeros((), dtype=torch.int64, device=dev)
        fuse_draft(draft, look, torch.tensor([match_len], dtype=torch.int32, device=dev),
                   torch.tensor([k], dtype=torch.int32, device=dev), use,
                   torch.ones(k, dtype=torch.int32, device=dev), hits, 1, k,
                   draft_block=block, idx_mapping_stride=block, nmin=4, nstrong=8, agree_min=2)
        tail_used = bool(use[0, block:].all().item())
        tail_lookup = bool((draft[0, block:] == look[0, block:].to(torch.int64)).all().item())
        ok = tail_used and tail_lookup == want_tail_lookup
        fails += 0 if ok else 1
        print(f"fuse block={block} match_len={match_len:2d} agree={agree} -> "
              f"tail_used={tail_used} tail_from_lookup={tail_lookup} "
              f"({'ok' if ok else 'FAIL'})")
    # take_flags: "this request has something to put in the tail". Whether the long block
    # is worth its step time is the host's decision (next_num_draft_tokens).
    for match_len, valid_n, block, prev, want in (
        (12, k, 3, 0, True),     # long match with a full tail: there is something to fill
        (12, 2, 3, 0, False),    # nothing left to put in the tail
        (2, k, 3, 0, False),     # match too short
    ):
        draft = torch.arange(100, 100 + k, dtype=torch.int64, device=dev).view(1, k)
        look = torch.arange(200, 200 + k, dtype=torch.int32, device=dev).view(1, k)
        use = torch.zeros((1, k), dtype=torch.int32, device=dev)
        hits = torch.zeros((), dtype=torch.int64, device=dev)
        flags = torch.full((1,), -1, dtype=torch.int32, device=dev)
        fuse_draft(draft, look, torch.tensor([match_len], dtype=torch.int32, device=dev),
                   torch.tensor([valid_n], dtype=torch.int32, device=dev), use,
                   torch.ones(k, dtype=torch.int32, device=dev), hits, 1, k,
                   draft_block=block, idx_mapping_stride=block, nmin=4, nstrong=8,
                   agree_min=2, long_min=6, take_flags=flags)
        got = bool(flags[0].item())
        ok = got == want
        fails += 0 if ok else 1
        print(f"take_flags match_len={match_len:2d} valid={valid_n:2d} block={block} -> "
              f"{got} ({'ok' if ok else 'FAIL'})")
    print("FAILURES:", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())

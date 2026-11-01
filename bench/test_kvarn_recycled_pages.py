"""CPU check of KVarN's recycled-page drop (#208, kvarn/kvarn-recycled-pages-0.31.0.patch).

note_scheduled_blocks must hand each KVarN builder the pages that went to OTHER KV-cache groups
(never its own, never the null block), and _drop_recycled_pages must release what the pool holds
for them -- a finished request's pending block, a retired sink -- without touching the builder's
own blocks or anything written this step. No GPU: the GPU check is bench/concurrent_collapse.py.

  CUDA_VISIBLE_DEVICES= venv/bin/python bench/test_kvarn_recycled_pages.py [path/to/kvarn_attn.py]
"""
import importlib.util, os, sys, types
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "..", "kvarn", "files", "vllm", "v1", "attention", "backends", "kvarn_attn.py")
spec = importlib.util.spec_from_file_location("vllm.v1.attention.backends.kvarn_attn", PATH)
m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
B = m.KVarNMetadataBuilder
ns = types.SimpleNamespace

def builder(layers):
    b = object.__new__(B)
    b._layer_names_set = set(layers); b._group = 128
    b._recycled_pages = set(); b._page_tokens = 0; b._recycled_dropped = 0
    b._block_fill = {}; b._retired_sinks = {}
    for n in layers: B._owner[n] = b
    return b

B._owner.clear()
attn = builder(["model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn"])
draft = builder(["drafter.layers.0.attn"])
groups = [ns(layer_names=["model.layers.3.self_attn.attn", "model.layers.7.self_attn.attn"], kv_cache_spec=ns(block_size=2048)),
          ns(layer_names=["model.layers.0.linear_attn"], kv_cache_spec=ns(block_size=2048)),
          ns(layer_names=["drafter.layers.0.attn"], kv_cache_spec=ns(block_size=2048))]
kvc = ns(kv_cache_groups=groups)
so = ns(scheduled_new_reqs=[ns(block_ids=([5, 6], [7, 0], [9]))],
        scheduled_cached_reqs=ns(new_block_ids=[([12], [11], []), None]))
m.note_scheduled_blocks(so, kvc)
assert attn._recycled_pages == {7, 9, 11}, attn._recycled_pages       # mamba + drafter pages, not its own, not null
assert draft._recycled_pages == {5, 6, 7, 11, 12}, draft._recycled_pages
assert attn._page_tokens == 2048 and attn._kv_group_ids == [0]

# attn's pool: page 7 holds a finished request's pending last block, page 9 a retired sink,
# page 5 its own pending block, page 11 a block written this step (must survive).
dm = {7 * 16 + 15: 0, 9 * 16: 1, 5 * 16 + 3: 2, 11 * 16 + 2: 3}
attn._block_fill = {7 * 16 + 15: 128, 5 * 16 + 3: 128}
attn._retired_sinks = {9 * 16: None}
sinks = {9 * 16}
b2s = torch.full((256,), -1, dtype=torch.int32); is_sink = torch.zeros(256, dtype=torch.bool)
for bid, s in dm.items(): b2s[bid] = s
is_sink[9 * 16] = True
free = []
attn._drop_recycled_pages({11 * 16 + 2}, dm, free, b2s, is_sink, sinks)
assert set(dm) == {5 * 16 + 3, 11 * 16 + 2}, dm
assert sorted(free) == [0, 1]
assert b2s[7 * 16 + 15] == -1 and b2s[9 * 16] == -1 and b2s[5 * 16 + 3] == 2
assert not is_sink[9 * 16] and 9 * 16 not in sinks and not attn._retired_sinks
assert 7 * 16 + 15 not in attn._block_fill and attn._recycled_pages == set()
assert attn._recycled_dropped == 2
# single-group model: nothing to do
before = set(draft._recycled_pages)
m.note_scheduled_blocks(so, ns(kv_cache_groups=groups[:1]))
assert draft._recycled_pages == before
print("OK")

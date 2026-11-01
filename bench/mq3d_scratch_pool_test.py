"""The int4 3D scratch pool (spec-decode-scratch-within-budget.patch), on CPU tensors: a model-load
allocation is adopted by builders, builders of one geometry share it, a smaller derived capacity
adopts the larger pool, a larger one allocates and replaces, an exclusive (ubatching) builder never
shares, and a different geometry is a different key. Ends with a negative control: with the
exclusive flag off, the second builder DOES share, which is the case the guard exists to prevent.

Runs inside the image, no GPU:  /app/venv/bin/python bench/mq3d_scratch_pool_test.py
Exit 0 = every assertion held (assert raises otherwise)."""
import logging, sys
import torch
import vllm
import vllm.v1.attention.backends.triton_attn as m
logging.getLogger("vllm").setLevel(logging.ERROR)
P = m.Mq3dScratchPlan
def plan(cap, heads=4, hd=64):
    row = 4 * (heads * m.NUM_PAR_SOFTMAX_SEGMENTS * hd + 2 * heads * m.NUM_PAR_SOFTMAX_SEGMENTS)
    return P(num_heads_q=heads, num_heads_kv=2, headdim_padded=hd, seq_threshold_3D=32,
             max_query_len_3d=16, capacity=cap, row_bytes=row, max_num_batched_tokens=2048, max_num_seqs=4)
dev = torch.device("cpu")
pool = m._MQ3D_SCRATCH_POOL; pool.clear()
a = m.mq3d_scratch_acquire(plan(64), dev, "model load")
assert a.origin == "model load" and a.adopters == 0 and len(pool) == 1, "load allocates, pooled"
b = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder")
c = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder")
assert b is a and c is a and a.adopters == 2, "two builders share the load-time set"
d = m.mq3d_scratch_acquire(plan(32), dev, "metadata builder")
assert d is a and d.capacity == 64, "smaller derived adopts the larger pool"
e = m.mq3d_scratch_acquire(plan(128), dev, "metadata builder")
assert e is not a and e.capacity == 128 and pool[next(iter(pool))] is e and e.adopters == 1, "larger derived allocates and replaces"
pool.clear()
x = m.mq3d_scratch_acquire(plan(64), dev, "model load")
y = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder", exclusive=True)
z = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder", exclusive=True)
assert y is x and z is not x and z.capacity == 64 and pool[next(iter(pool))] is x, "exclusive: first adopts, second allocates its own, pool keeps the first"
w = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder", exclusive=False)
assert w is x, "non-exclusive still shares"
g = m.mq3d_scratch_acquire(plan(64, heads=8), dev, "metadata builder")
assert g is not x and len(pool) == 2, "different geometry is a different key"
print("POOL UNIT TEST OK: 9 assertions")
# negative control: the exclusive guard removed -> sharing across exclusive builders would be wrong
pool.clear()
x = m.mq3d_scratch_acquire(plan(64), dev, "model load")
m.mq3d_scratch_acquire(plan(64), dev, "metadata builder", exclusive=True)
shared_when_it_must_not = m.mq3d_scratch_acquire(plan(64), dev, "metadata builder", exclusive=False) is x
print("NEGATIVE CONTROL (exclusive=False on the second builder shares):", shared_when_it_must_not)

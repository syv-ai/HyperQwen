"""Numerical check: Marlin W4A8-INT8 with zero-point (uint4) weights against the W4A16
path on the same packed weights, at the shapes this checkpoint uses.

  CUDA_VISIBLE_DEVICES=0 venv/bin/python bench/test_marlin_int8_asym.py

Exit 0 when the int8 path is within 5% of the float reference at every shape (measured:
0.9-1.05% on an RTX 3090, the W4A16 path 0.26%). Needs patches/marlin-int8-asym-zp.patch;
without it the int8 arm dies with "W8A8 is not supported by marlin kernel".

Builds a random asymmetric int4 g128 weight in the compressed-tensors pack-quantized
layout (signed nibbles, +8 offset on packing, zero points packed along the output
dim), runs vLLM's MarlinLinearKernel once with bf16 activations and once with int8
activations, and compares both to the float reference of the dequantized weight.
"""
import sys, torch
from vllm.model_executor.kernels.linear.mixed_precision import MPLinearLayerConfig
from vllm.model_executor.kernels.linear.mixed_precision.marlin import MarlinLinearKernel
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32

dev = "cuda"
GROUP = 128

# vLLM's parameter classes ask for the TP rank at construction: bring up a 1-process group
# under a default VllmConfig, the way vLLM's own kernel tests do.
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import init_distributed_environment, ensure_model_parallel_initialized
_cfg_ctx = set_current_vllm_config(VllmConfig())
_cfg_ctx.__enter__()
init_distributed_environment(world_size=1, rank=0, distributed_init_method="tcp://127.0.0.1:29517", local_rank=0)
ensure_model_parallel_initialized(1, 1)

def make_layer(K, N, act_type, seed):
    torch.manual_seed(seed)
    w = torch.randn(N, K, dtype=torch.float32) * 0.02
    g = w.reshape(N, K // GROUP, GROUP)
    wmin, wmax = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    s = (wmax - wmin).clamp(min=1e-8) / 15.0
    zp_u = torch.round(-wmin / s).clamp(0, 15)
    q_u = torch.clamp(torch.round(g / s) + zp_u, 0, 15)
    deq = ((q_u - zp_u) * s).reshape(N, K)
    q_s = (q_u.reshape(N, K) - 8).to(torch.int8)                 # signed nibbles, as the compressor stores them
    zp_s = (zp_u.reshape(N, K // GROUP) - 8).to(torch.int8)
    packed = pack_to_int32(q_s, 4, packed_dim=1)                 # [N, K/8]
    zp_packed = pack_to_int32(zp_s, 4, packed_dim=0)             # [N/8, K/G]
    scale = s.reshape(N, K // GROUP).to(torch.bfloat16)          # [N, K/G]

    layer = torch.nn.Module()
    wl = lambda *a, **k: None
    layer.weight_packed = PackedvLLMParameter(data=packed.contiguous().to(dev), input_dim=1, output_dim=0,
                                              packed_dim=1, packed_factor=8, weight_loader=wl)
    layer.weight_scale = GroupQuantScaleParameter(data=scale.contiguous().to(dev), output_dim=0, input_dim=1, weight_loader=wl)
    layer.weight_zero_point = PackedvLLMParameter(data=zp_packed.contiguous().to(dev), input_dim=1, output_dim=0,
                                                  packed_dim=0, packed_factor=8, weight_loader=wl)
    cfg = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N),
                              weight_type=scalar_types.uint4, act_type=act_type,
                              group_size=GROUP, zero_points=True, has_g_idx=False)
    ok, why = MarlinLinearKernel.can_implement(cfg)
    assert ok, why
    k = MarlinLinearKernel(cfg, w_q_param_name="weight_packed", w_s_param_name="weight_scale",
                           w_zp_param_name="weight_zero_point", w_gidx_param_name="weight_g_idx")
    k.process_weights_after_loading(layer)
    return layer, k, deq.to(dev)

def run(K, N, M, seed):
    torch.manual_seed(seed + 100)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=dev)
    l16, k16, deq = make_layer(K, N, torch.bfloat16, seed)
    l8, k8, _ = make_layer(K, N, torch.int8, seed)
    ref = x.float() @ deq.t()
    y16 = k16.apply_weights(l16, x).float()
    y8 = k8.apply_weights(l8, x).float()
    e16 = ((y16 - ref).norm() / ref.norm()).item()
    e8 = ((y8 - ref).norm() / ref.norm()).item()
    e8_16 = ((y8 - y16).norm() / y16.norm()).item()
    print(f"K={K:6d} N={N:6d} M={M:3d}  W4A16 rel err {e16:.4f}   W4A8-int8 rel err {e8:.4f}   int8 vs bf16 {e8_16:.4f}", flush=True)
    return e8

worst = 0.0
for i, (K, N) in enumerate([(5120, 17408), (17408, 5120), (5120, 12288), (5120, 6144), (5120, 5120), (5120, 1024)]):
    for M in (1, 5, 16, 256):
        worst = max(worst, run(K, N, M, i))
print("worst W4A8-int8 rel err:", worst)
sys.exit(0 if worst < 0.05 else 1)

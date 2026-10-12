// SPDX-License-Identifier: Apache-2.0
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>

namespace fa2_prefill {
template<int Dim>
__global__ void gather_pages(
    const unsigned char* keys, const unsigned char* values,
    const int* table, const float* key_scale, const float* value_scale,
    __nv_bfloat16* out, int start, int length, int head, int page_size,
    int64_t kb, int64_t kr, int64_t kh, int64_t vb, int64_t vr, int64_t vh) {
    const int flat = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int row = flat / Dim;
    const int col = flat % Dim;
    if (row >= length) return;
    const int logical = start + row;
    const int page = table[logical / page_size];
    const int offset = logical % page_size;
    const int64_t ki = int64_t(page) * kb + offset * kr + head * kh + col;
    const int64_t vi = int64_t(page) * vb + offset * vr + head * vh + col;
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        __nv_fp8_e4m3 key;
        __nv_fp8_e4m3 value;
        key.__x = keys[ki + lane];
        value.__x = values[vi + lane];
        out[flat + lane] = __float2bfloat16_rn(static_cast<float>(key) * key_scale[0]);
        out[int64_t(length) * Dim + flat + lane] = __float2bfloat16_rn(static_cast<float>(value) * value_scale[0]);
    }
}

at::Tensor gather(const at::Tensor& keys, const at::Tensor& values,
                  const at::Tensor& table, const at::Tensor& key_scale,
                  const at::Tensor& value_scale, int64_t start,
                  int64_t length, int64_t head) {
    TORCH_CHECK(keys.is_cuda() && keys.dim() == 4 && values.sizes() == keys.sizes(), "Expected paged CUDA K/V");
    TORCH_CHECK((keys.scalar_type() == at::kByte || keys.scalar_type() == at::kFloat8_e4m3fn)
                && values.scalar_type() == keys.scalar_type(), "Expected E4M3 KV bytes");
    for (const auto& tensor : {values, table, key_scale, value_scale}) {
        TORCH_CHECK(tensor.device() == keys.device(), "Tensor devices must match");
    }
    TORCH_CHECK(table.dim() == 2 && table.size(0) == 1 && table.stride(1) == 1
                && table.scalar_type() == at::kInt, "Expected one int32 page-table row");
    TORCH_CHECK(keys.stride(3) == 1 && values.stride(3) == 1, "Head dimension must be contiguous");
    TORCH_CHECK((keys.size(3) == 128 || keys.size(3) == 256) && head >= 0 && head < keys.size(2), "Unsupported head geometry");
    TORCH_CHECK(start >= 0 && length > 0 && start + length <= table.size(1) * keys.size(1), "Invalid KV range");
    TORCH_CHECK(key_scale.scalar_type() == at::kFloat && value_scale.scalar_type() == at::kFloat
                && key_scale.numel() == 1 && value_scale.numel() == 1, "Expected per-tensor float32 scales");
    const c10::cuda::CUDAGuard guard(keys.device());
    auto out = at::empty({2, length, 1, keys.size(3)}, keys.options().dtype(at::kBFloat16));
    const int blocks = (length * keys.size(3) + 1023) / 1024;
    const auto stream = c10::cuda::getCurrentCUDAStream(keys.get_device());
    auto launch = [&](auto dim) {
        gather_pages<decltype(dim)::value><<<blocks, 256, 0, stream>>>(
            static_cast<const unsigned char*>(keys.data_ptr()),
            static_cast<const unsigned char*>(values.data_ptr()), table.data_ptr<int>(),
            key_scale.data_ptr<float>(), value_scale.data_ptr<float>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), start, length, head,
            keys.size(1), keys.stride(0), keys.stride(1), keys.stride(2),
            values.stride(0), values.stride(1), values.stride(2));
    };
    if (keys.size(3) == 256) launch(std::integral_constant<int, 256>{});
    else launch(std::integral_constant<int, 128>{});
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
}

TORCH_LIBRARY(fa2_fp8kv_prefill, m) {
    m.def("gather(Tensor keys, Tensor values, Tensor table, Tensor key_scale, Tensor value_scale, int start, int length, int head) -> Tensor");
}
TORCH_LIBRARY_IMPL(fa2_fp8kv_prefill, CUDA, m) {
    m.impl("gather", &fa2_prefill::gather);
}

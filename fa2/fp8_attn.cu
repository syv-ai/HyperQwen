// SPDX-License-Identifier: Apache-2.0
// FP8-cache adapter for the pinned FlashAttention 2 split-KV kernels.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <flashinfer/vec_dtypes.cuh>

#define FLASH_NAMESPACE fa2_fp8kv_kernel
namespace fa2_fp8kv_kernel {}
namespace flash = fa2_fp8kv_kernel;
#include "flash_fwd_launch_template.h"

namespace fa2_fp8kv {

template<int HeadDim>
struct Fp8Traits : Flash_fwd_kernel_traits<HeadDim, 64, (HeadDim == 256 ? 32 : 64), 4, false, false, cutlass::bfloat16_t> {
    using Base = Flash_fwd_kernel_traits<HeadDim, 64, (HeadDim == 256 ? 32 : 64), 4, false, false, cutlass::bfloat16_t>;
    using ElementKV = cutlass::float_e4m3_t;
    static constexpr int kBaseSmemSize = Base::kSmemSize;
    static constexpr int kSmemSize = kBaseSmemSize + (FA2_FP8KV_PIPELINE
        ? 2 * cute::size(typename Base::SmemLayoutKV{}) * sizeof(ElementKV) : 0);
};

template<int HeadDim, bool Causal, bool Local, bool Split>
void launch(fa2_fp8kv_kernel::Flash_fwd_params& params, cudaStream_t stream) {
    using Traits = Fp8Traits<HeadDim>;
    const dim3 grid((params.seqlen_q + 63) / 64,
                   Split ? params.num_splits : params.b,
                   Split ? params.b * params.h : params.h);
    auto kernel = fa2_fp8kv_kernel::flash_fwd_splitkv_kernel<
        Traits, Causal, Local, false, false, true, false, Split, false>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Traits::kSmemSize));
    kernel<<<grid, Traits::kNThreads, Traits::kSmemSize, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if constexpr (Split) {
        const dim3 combine((int64_t(params.b) * params.h * params.seqlen_q + 3) / 4);
        fa2_fp8kv_kernel::flash_fwd_splitkv_combine_kernel<Traits, 4, 7, true>
            <<<combine, Traits::kNThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template<int HeadDim, bool Split>
void dispatch(fa2_fp8kv_kernel::Flash_fwd_params& params, cudaStream_t stream) {
    if (params.is_causal) {
        launch<HeadDim, true, false, Split>(params, stream);
    } else if (params.window_size_left >= 0 || params.window_size_right >= 0) {
        launch<HeadDim, false, true, Split>(params, stream);
    } else {
        launch<HeadDim, false, false, Split>(params, stream);
    }
}

std::tuple<at::Tensor, at::Tensor> forward(
    const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
    at::Tensor out, const at::Tensor& cu_q, const at::Tensor& seq_k,
    const at::Tensor& table, const at::Tensor& k_scale, const at::Tensor& v_scale,
    int64_t max_q, int64_t max_k, bool causal, int64_t window_left,
    int64_t window_right, double softmax_scale, int64_t splits, bool pack_gqa) {
    TORCH_CHECK(q.is_cuda() && q.dim() == 3 && q.scalar_type() == at::kBFloat16,
                "FP8 FA2 requires CUDA BF16 Q [tokens, heads, dim]");
    TORCH_CHECK(k.dim() == 4 && v.sizes() == k.sizes(), "K/V must have matching paged shapes");
    TORCH_CHECK((k.scalar_type() == at::kByte || k.scalar_type() == at::kFloat8_e4m3fn)
                && v.scalar_type() == k.scalar_type(), "K/V must contain FP8 E4M3FN bytes");
    for (const auto& tensor : {k, v, out, cu_q, seq_k, table, k_scale, v_scale}) {
        TORCH_CHECK(tensor.device() == q.device(), "All tensors must be on Q's device");
    }
    TORCH_CHECK(out.sizes() == q.sizes() && out.scalar_type() == q.scalar_type(), "Invalid output tensor");
    const int head_dim = q.size(2);
    const int heads = q.size(1);
    const int kv_heads = k.size(2);
    TORCH_CHECK((head_dim == 128 || head_dim == 256) && k.size(3) == head_dim,
                "This port supports head dimensions 128 and 256");
    TORCH_CHECK(kv_heads > 0 && heads % kv_heads == 0, "Invalid GQA head counts");
    TORCH_CHECK(k.size(1) >= 16 && k.size(1) % 16 == 0, "Cache block size must be a multiple of 16");
    TORCH_CHECK(q.stride(2) == 1 && out.stride(2) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
                "Head dimensions must be contiguous");
    TORCH_CHECK(q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0
                && out.stride(0) % 8 == 0 && out.stride(1) % 8 == 0, "Q/output strides must be 16-byte aligned");
    TORCH_CHECK(cu_q.scalar_type() == at::kInt && seq_k.scalar_type() == at::kInt
                && table.scalar_type() == at::kInt, "Sequence metadata must be int32");
    TORCH_CHECK(cu_q.dim() == 1 && seq_k.dim() == 1 && table.dim() == 2
                && cu_q.is_contiguous() && seq_k.is_contiguous() && table.stride(1) == 1,
                "Invalid sequence metadata layout");
    const int batch = seq_k.numel();
    TORCH_CHECK(batch > 0 && cu_q.numel() == batch + 1 && table.size(0) == batch, "Invalid batch metadata");
    TORCH_CHECK(max_q > 0 && max_q <= 2048 && max_k > 0 && max_k <= 262144
                && q.size(0) <= batch * max_q, "Sequence bounds exceed this port's supported envelope");
    TORCH_CHECK(table.size(1) >= (max_k + k.size(1) - 1) / k.size(1), "Block table is too short");
    TORCH_CHECK(k_scale.scalar_type() == at::kFloat && v_scale.scalar_type() == at::kFloat
                && k_scale.numel() == 1 && v_scale.numel() == 1, "Only per-tensor float32 KV scales are supported");
    TORCH_CHECK(window_left >= -1 && window_right >= -1, "Invalid attention window");
    TORCH_CHECK(splits >= 0 && splits <= 128, "Split count must be between 0 and 128");

    const c10::cuda::CUDAGuard guard(q.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    cudaDeviceProp properties;
    C10_CUDA_CHECK(cudaGetDeviceProperties(&properties, q.get_device()));
    const int sm = properties.major * 10 + properties.minor;
    TORCH_CHECK(sm == 86 || sm == 89 || sm == 120,
                "This kernel build targets SM86, SM89 and SM120");
    if (splits == 0) {
        // Keep prefill scratch bounded; short speculative queries benefit from split-KV.
        const int64_t active_k = window_left >= 0 ? std::min(max_k, window_left + max_q) : max_k;
        splits = max_q <= 64
            ? std::min<int64_t>(128, std::max<int64_t>(1, (active_k + 127) / 128)) : 1;
    }
    if (causal) window_right = 0;
    const bool ordinary_causal = window_left < 0 && window_right == 0;
    if (window_left < 0 && window_right >= 0) window_left = max_k;
    if (window_right < 0 && window_left >= 0) window_right = max_k;

    const int groups = pack_gqa ? heads / kv_heads : 1;
    TORCH_CHECK(groups == 1 || (max_q * groups <= 64 && ordinary_causal),
                "GQA packing currently supports one causal query tile");
    const int kernel_heads = heads / groups;
    const int kernel_max_q = max_q * groups;
    // Both reshapes are zero-copy when there is one KV head per rank.
    const auto kernel_q = groups == 1 ? q : q.reshape({q.size(0), kv_heads, groups, head_dim})
        .permute({0, 2, 1, 3}).contiguous().reshape({q.size(0) * groups, kv_heads, head_dim});
    const bool packed_output_view = kv_heads == 1 && out.is_contiguous();
    auto kernel_out = groups == 1 ? out : (packed_output_view
        ? out.reshape({q.size(0) * groups, kv_heads, head_dim}) : at::empty_like(kernel_q));
    const auto kernel_cu_q = groups == 1 ? cu_q : cu_q * groups;
    auto lse = at::empty({kernel_heads, kernel_q.size(0)}, q.options().dtype(at::kFloat));
    fa2_fp8kv_kernel::Flash_fwd_params params{};
    params.q_ptr = kernel_q.data_ptr(); params.k_ptr = k.data_ptr(); params.v_ptr = v.data_ptr();
    params.o_ptr = kernel_out.data_ptr(); params.softmax_lse_ptr = lse.data_ptr();
    params.q_row_stride = kernel_q.stride(0); params.q_head_stride = kernel_q.stride(1);
    params.o_row_stride = kernel_out.stride(0); params.o_head_stride = kernel_out.stride(1);
    params.k_batch_stride = k.stride(0); params.k_row_stride = k.stride(1); params.k_head_stride = k.stride(2);
    params.v_batch_stride = v.stride(0); params.v_row_stride = v.stride(1); params.v_head_stride = v.stride(2);
    params.b = batch; params.h = kernel_heads; params.h_k = kv_heads; params.h_h_k_ratio = kernel_heads / kv_heads;
    params.query_group_size = groups;
    params.d = head_dim; params.d_rounded = head_dim; params.total_q = kernel_q.size(0);
    params.seqlen_q = kernel_max_q; params.seqlen_k = max_k;
    params.seqlen_q_rounded = (kernel_max_q + 127) / 128 * 128;
    params.seqlen_k_rounded = (max_k + 127) / 128 * 128;
    params.cu_seqlens_q = kernel_cu_q.data_ptr<int>(); params.seqused_k = seq_k.data_ptr<int>();
    params.block_table = table.data_ptr<int>(); params.block_table_batch_stride = table.stride(0);
    params.page_block_size = k.size(1);
    params.k_descale_ptr = k_scale.data_ptr<float>(); params.v_descale_ptr = v_scale.data_ptr<float>();
    params.scale_softmax = softmax_scale; params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    params.p_dropout = 1.0f; params.rp_dropout = 1.0f;
    params.scale_softmax_rp_dropout = softmax_scale;
    params.window_size_left = window_left; params.window_size_right = window_right;
    params.is_bf16 = true; params.is_causal = ordinary_causal; params.unpadded_lse = true;
    params.num_splits = splits;

    // PR #190's initialization is required for empty splits in ragged causal batches.
    auto lse_accum = at::full({splits > 1 ? splits : 0, batch, kernel_heads, kernel_max_q},
                             -std::numeric_limits<float>::infinity(), q.options().dtype(at::kFloat));
    auto out_accum = at::zeros({splits > 1 ? splits : 0, batch, kernel_heads, kernel_max_q, head_dim},
                              q.options().dtype(at::kFloat));
    if (splits > 1) {
        params.softmax_lseaccum_ptr = lse_accum.data_ptr();
        params.oaccum_ptr = out_accum.data_ptr();
    }
    if (head_dim == 128) {
        if (splits > 1) dispatch<128, true>(params, stream);
        else dispatch<128, false>(params, stream);
    } else {
        if (splits > 1) dispatch<256, true>(params, stream);
        else dispatch<256, false>(params, stream);
    }
    if (groups > 1) {
        if (!packed_output_view) {
            out.copy_(kernel_out.reshape({q.size(0), groups, kv_heads, head_dim})
                .permute({0, 2, 1, 3}).reshape_as(out));
        }
        lse = lse.reshape({kv_heads, q.size(0), groups}).permute({0, 2, 1})
            .contiguous().reshape({heads, q.size(0)});
    }
    return {out, lse};
}
}  // namespace fa2_fp8kv

TORCH_LIBRARY(fa2_fp8kv, m) {
    m.def("forward(Tensor q, Tensor k, Tensor v, Tensor(a!) out, Tensor cu_q, Tensor seq_k, Tensor table, Tensor k_scale, Tensor v_scale, int max_q, int max_k, bool causal, int window_left, int window_right, float softmax_scale, int splits=0, bool pack_gqa=False) -> (Tensor(a!), Tensor)");
}
TORCH_LIBRARY_IMPL(fa2_fp8kv, CUDA, m) {
    m.impl("forward", &fa2_fp8kv::forward);
}

// difflogic_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

template <typename T> T ceil_div(const T x, const T y) { return x / y + !!(x % y); }

/**********************************************************************************************************************/
// AIG-Style Forward Kernel:
// Each neuron computes: out = ( (flag_a ? ~a : a) & (flag_b ? ~b : b) )
template <typename scalar_t>
__global__ void aig_logic_layer_cuda_forward_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x, // [in_dim, batch]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,   // [num_neurons]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,   // [num_neurons]
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> neg_flags, // [num_neurons, 2]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y        // [num_neurons, batch]
) {
    for (auto row = blockIdx.x * blockDim.x + threadIdx.x; row < y.size(1); row += blockDim.x * gridDim.x) {
        for (auto col = blockIdx.y * blockDim.y + threadIdx.y; col < y.size(0); col += blockDim.y * gridDim.y) {
            const int64_t idx_a = a[col];
            const int64_t idx_b = b[col];
            scalar_t a_val = x[idx_a][row];
            scalar_t b_val = x[idx_b][row];

            // Read negation flags for neuron col.
            uint8_t flag_a = neg_flags[col][0];
            uint8_t flag_b = neg_flags[col][1];

            if (flag_a) { a_val = ~a_val; }
            if (flag_b) { b_val = ~b_val; }

            y[col][row] = a_val & b_val;
        }
    }
}

/**********************************************************************************************************************/
// AIG-Style Backward Kernel for Inputs (simple straight-through pass).
template <typename scalar_t>
__global__ void aig_logic_layer_cuda_backward_x_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x, // [in_dim, batch]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,   // [num_neurons]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,   // [num_neurons]
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> neg_flags, // [num_neurons, 2]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y, // [num_neurons, batch]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_x, // [in_dim, batch]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y_start,
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> given_x_indices_of_y
) {
    for (auto row = blockIdx.x * blockDim.x + threadIdx.x; row < grad_x.size(1); row += blockDim.x * gridDim.x) {
        for (auto col = blockIdx.y * blockDim.y + threadIdx.y; col < grad_x.size(0); col += blockDim.y * gridDim.y) {
            scalar_t grad_val = 0;
            const auto start = given_x_indices_of_y_start[col];
            const auto end = given_x_indices_of_y_start[col + 1];
            for (int cur = start; cur < end; ++cur) {
                const auto idx_y = given_x_indices_of_y[cur];
                grad_val += grad_y[idx_y][row]; // straight-through gradient pass.
            }
            grad_x[col][row] = grad_val;
        }
    }
}

/**********************************************************************************************************************/
// AIG-Style Backward Kernel for Negation Parameters (stub version).
template <typename scalar_t>
__global__ void aig_logic_layer_cuda_backward_neg_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x, // [in_dim, batch]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,   // [num_neurons]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,   // [num_neurons]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_y, // [num_neurons, batch]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> grad_neg // [num_neurons, 2]
) {
    for (auto col = blockIdx.x * blockDim.x + threadIdx.x; col < grad_y.size(0); col += blockDim.x * gridDim.x) {
        scalar_t grad_neg_a = 0;
        scalar_t grad_neg_b = 0;
        const int64_t idx_a = a[col];
        const int64_t idx_b = b[col];
        for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < grad_y.size(1); row += blockDim.y * gridDim.y) {
            grad_neg_a += - (x[idx_a][row]) * grad_y[col][row];
            grad_neg_b += - (x[idx_b][row]) * grad_y[col][row];
        }
        grad_neg[col][0] = grad_neg_a;
        grad_neg[col][1] = grad_neg_b;
    }
}

/**********************************************************************************************************************/
// AIG-Style Eval Kernel (same as forward, using hard negation flags).
template <typename scalar_t>
__global__ void aig_logic_layer_cuda_eval_kernel(
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> x, // [in_dim, batch]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> a,   // [num_neurons]
    torch::PackedTensorAccessor64<int64_t, 1, torch::RestrictPtrTraits> b,   // [num_neurons]
    torch::PackedTensorAccessor64<uint8_t, 2, torch::RestrictPtrTraits> neg_flags, // [num_neurons, 2]
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> y        // [num_neurons, batch]
) {
    for (auto row = blockIdx.x * blockDim.x + threadIdx.x; row < y.size(1); row += blockDim.x * gridDim.x) {
        for (auto col = blockIdx.y * blockDim.y + threadIdx.y; col < y.size(0); col += blockDim.y * gridDim.y) {
            const int64_t idx_a = a[col];
            const int64_t idx_b = b[col];
            scalar_t a_val = x[idx_a][row];
            scalar_t b_val = x[idx_b][row];
            uint8_t flag_a = neg_flags[col][0];
            uint8_t flag_b = neg_flags[col][1];
            if (flag_a) { a_val = ~a_val; }
            if (flag_b) { b_val = ~b_val; }
            y[col][row] = a_val & b_val;
        }
    }
}

/**********************************************************************************************************************/
// Wrapper functions.
torch::Tensor aig_logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(neg_flags);
    const auto batch_size = x.size(1);
    const auto out_size = neg_flags.size(0);
    auto y = torch::empty({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));
    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min((int64_t)65535, ceil_div(batch_size, (int64_t)threads_per_block.x)),
        min((int64_t)65535, ceil_div(out_size, (int64_t)threads_per_block.y))
    );
    AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "aig_logic_layer_cuda_forward", ([&] {
        aig_logic_layer_cuda_forward_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            neg_flags.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
            y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    cudaDeviceSynchronize();
    return y;
}

torch::Tensor aig_logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(neg_flags);
    CHECK_INPUT(grad_y);
    CHECK_INPUT(given_x_indices_of_y_start);
    CHECK_INPUT(given_x_indices_of_y);
    auto grad_x = torch::empty_like(x);
    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min((int64_t)65535, ceil_div(x.size(1), (int64_t)threads_per_block.x)),
        min((int64_t)65535, ceil_div(x.size(0), (int64_t)threads_per_block.y))
    );
    AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "aig_logic_layer_cuda_backward_x", ([&] {
        aig_logic_layer_cuda_backward_x_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            neg_flags.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
            grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            given_x_indices_of_y_start.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            given_x_indices_of_y.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>()
        );
    }));
    cudaDeviceSynchronize();
    return grad_x;
}

torch::Tensor aig_logic_layer_cuda_backward_neg(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(grad_y);
    auto grad_neg = torch::zeros({grad_y.size(0), 2}, torch::dtype(x.dtype()).device(x.device()));
    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min((int64_t)65535, ceil_div(grad_y.size(0), (int64_t)threads_per_block.x)),
        min((int64_t)65535, ceil_div(grad_y.size(1), (int64_t)threads_per_block.y))
    );
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "aig_logic_layer_cuda_backward_neg", ([&] {
        aig_logic_layer_cuda_backward_neg_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            grad_y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            grad_neg.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    cudaDeviceSynchronize();
    return grad_neg;
}

torch::Tensor aig_logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags
) {
    CHECK_INPUT(x);
    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(neg_flags);
    const auto batch_size = x.size(1);
    const auto out_size = neg_flags.size(0);
    auto y = torch::zeros({out_size, batch_size}, torch::dtype(x.dtype()).device(x.device()));
    dim3 threads_per_block(32, 32);
    const dim3 blocks_per_grid(
        min((int64_t)65535, ceil_div(batch_size, (int64_t)threads_per_block.x)),
        min((int64_t)65535, ceil_div(out_size, (int64_t)threads_per_block.y))
    );
    AT_DISPATCH_INTEGRAL_TYPES(x.scalar_type(), "aig_logic_layer_cuda_eval", ([&] {
        aig_logic_layer_cuda_eval_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            x.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
            a.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            b.packed_accessor64<int64_t, 1, torch::RestrictPtrTraits>(),
            neg_flags.packed_accessor64<uint8_t, 2, torch::RestrictPtrTraits>(),
            y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>()
        );
    }));
    cudaDeviceSynchronize();
    return y;
}

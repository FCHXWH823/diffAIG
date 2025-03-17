// difflogic.cpp
#include <pybind11/numpy.h>
#include <torch/extension.h>
#include <vector>

namespace py = pybind11;

// Declare the AIG-style kernel functions.
torch::Tensor aig_logic_layer_cuda_forward(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags  // Shape: [num_neurons, 2], each element 0 or 1.
);

torch::Tensor aig_logic_layer_cuda_backward_neg(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor grad_y
);

torch::Tensor aig_logic_layer_cuda_backward_x(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags,
    torch::Tensor grad_y,
    torch::Tensor given_x_indices_of_y_start,
    torch::Tensor given_x_indices_of_y
);

torch::Tensor aig_logic_layer_cuda_eval(
    torch::Tensor x,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor neg_flags
);

std::tuple<torch::Tensor, int> tensor_packbits_cuda(
    torch::Tensor t,
    const int bit_count
);

torch::Tensor groupbitsum(
    torch::Tensor b,
    const int pad_len,
    const int k
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor neg_flags) {
            return aig_logic_layer_cuda_forward(x, a, b, neg_flags);
        },
        "AIG logic layer forward (CUDA)"
    );
    m.def(
        "backward_neg",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor grad_y) {
            return aig_logic_layer_cuda_backward_neg(x, a, b, grad_y);
        },
        "AIG logic layer backward for negation parameters (CUDA)"
    );
    m.def(
        "backward_x",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor neg_flags,
           torch::Tensor grad_y, torch::Tensor given_x_indices_of_y_start, torch::Tensor given_x_indices_of_y) {
            return aig_logic_layer_cuda_backward_x(x, a, b, neg_flags, grad_y,
                                                   given_x_indices_of_y_start, given_x_indices_of_y);
        },
        "AIG logic layer backward for inputs (CUDA)"
    );
    m.def(
        "eval",
        [](torch::Tensor x, torch::Tensor a, torch::Tensor b, torch::Tensor neg_flags) {
            return aig_logic_layer_cuda_eval(x, a, b, neg_flags);
        },
        "AIG logic layer eval (CUDA)"
    );
    m.def(
        "tensor_packbits_cuda",
        [](torch::Tensor t, const int bit_count) {
            return tensor_packbits_cuda(t, bit_count);
        },
        "tensor_packbits_cuda (CUDA)"
    );
    m.def(
        "groupbitsum",
        [](torch::Tensor b, const int pad_len, const unsigned int k) {
            if (b.size(0) % k != 0) {
                throw py::value_error("in_dim (" + std::to_string(b.size(0)) +
                                      ") has to be divisible by k (" + std::to_string(k) + ") but it is not");
            }
            return groupbitsum(b, pad_len, k);
        },
        "groupbitsum (CUDA)"
    );
}

#include <torch/extension.h>

std::vector<torch::Tensor> forward_1(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_1",
        torch::wrap_pybind_function(forward_1),
        "forward_1"
    );
}

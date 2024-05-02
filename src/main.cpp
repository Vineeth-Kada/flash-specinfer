#include <torch/extension.h>

std::vector<torch::Tensor> forward_1(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_2(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_3(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);
std::vector<torch::Tensor> forward_4(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward_1",
        torch::wrap_pybind_function(forward_1),
        "forward_1"
    );
    m.def(
        "forward_2",
        torch::wrap_pybind_function(forward_2),
        "forward_2"
    );
    m.def(
        "forward_3",
        torch::wrap_pybind_function(forward_3),
        "forward_3"
    );
    m.def(
        "forward_4",
        torch::wrap_pybind_function(forward_4),
        "forward_4"
    );
}

import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6' # For A6000

from tree import generate_random_trees

# Load the CUDA kernel as a python module
minimal_attn = load(
    name='minimal_attn',
    sources = list(map(lambda x: '../src/' + x, ['main.cpp', 'forward_1.cu', 'forward_2.cu'])),
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 10
n_head = 10
head_embd = 64
num_tree_nodes = 2**10 - 40
prompt_length = 40

seq_len = num_tree_nodes + prompt_length
IsTree = True

q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

# Tree creation for SpecInfer
start_times, end_times, causal_masks = generate_random_trees(batch_size, num_tree_nodes, prompt_length)
# print(causal_masks)
# print(causal_masks.shape)
# exit()

print('\n\n=== profiling torch attention === ')
# os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '1'
def torch_attention(q, k, v, mask):
    output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return output

with torch.no_grad():
    causal_mask = causal_masks.unsqueeze(1)#.broadcast_to((batch_size, n_head, seq_len, seq_len)).contiguous()
    if not IsTree:
        causal_mask = None
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    manual_result_torch = torch_attention(q, k, v, causal_mask)
    end.record()
    torch.cuda.synchronize()
print("Time taken in ms: ", start.elapsed_time(end))

print('\n\n=== profiling minimal flash attention (forward pass) === ')
with torch.no_grad():
    (minimal_result,) = minimal_attn.forward_2(q, k, v, start_times, end_times, IsTree)
print(
    'attn values sanity check:',
    torch.allclose(minimal_result, manual_result_torch, rtol=0, atol=1e-02),
)
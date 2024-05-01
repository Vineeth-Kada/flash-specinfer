import math

import torch
from torch.nn import functional as F
from torch.utils.cpp_extension import load

import os
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6' # For A6000

# Load the CUDA kernel as a python module
minimal_attn = load(
    name='minimal_attn',
    sources = list(map(lambda x: '../src/' + x, ['main.cpp', 'forward_1.cu'])),
    extra_cuda_cflags=['-O3', '--use_fast_math']
)

# Use small model params, otherwise slower than manual attention. See caveats in README.
batch_size = 10
n_head = 2
seq_len = 1024
head_embd = 64

q = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
k = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()
v = torch.randn(batch_size, n_head, seq_len, head_embd, requires_grad=True).cuda()

print('\n\n=== profiling torch attention === ')
# os.environ['TORCHINDUCTOR_COORDINATE_DESCENT_TUNING'] = '1'
def torch_attention(q, k, v):
    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return output
# torch_attention_script = torch.jit.script(torch_attention)
with torch.no_grad():
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    manual_result_torch = torch_attention(q, k, v)
    end.record()
    torch.cuda.synchronize()
print("Time taken in ms: ", start.elapsed_time(end))

print('\n\n=== profiling minimal flash attention (forward pass) === ')
with torch.no_grad():
    (minimal_result,) = minimal_attn.forward_1(q, k, v)
print(
    'attn values sanity check:',
    torch.allclose(minimal_result, manual_result_torch, rtol=0, atol=1e-02),
)

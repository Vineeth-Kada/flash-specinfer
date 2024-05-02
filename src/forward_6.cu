#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define D 64
__global__
void forward_6_kernel(
    const float* Q,
    const float* K,
    const float* V,
    const int N, // Sequence Length
    const int d, // Hidden Dimension per Head
    const int Tc, // Tc = ceil(N / Bc)
    const int Tr, // Tr = ceil(N / Br)
    const int Bc,
    const int Br,
    const float softmax_scale, // = 1 / sqrt(d)
    float* O, // Output Tensor
    const float* startT, // (B * N) start Time
    const float* endT, // (B * N) end Time
    const bool IsTree // If true then use tree causality
) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;  // batch, head, tile row index

    // Offset into Q,K,V,O - different for each batch and head
    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);  // gridDim.y = nh

    // Define SRAM for Q,K,V,S
    extern __shared__ float sram[];
    int tile_size = Bc * d;  // size of Qi, Kj, Vj
    float* QiT = sram;
    float* Kj = &sram[tile_size];
    float* Vj = &sram[tile_size * 2];
    // float* OiT = &sram[tile_size * 3];
    float* startI = &sram[tile_size * 4];
    float* endI = &sram[tile_size * 4 + Bc];
    float* startJ = &sram[tile_size * 4 + 2 * Bc];
    float* endJ = &sram[tile_size * 4 + 2 * Bc + Br];
    float* S = &sram[tile_size * 4 + 2 * Bc + 2 * Br];
    
    int i = bz;

    // Load Qi from HBM to SRAM, l and m to registers
    // Populate Qi - (32, {64, 128}) - Think like 1D 32*d array
    for (int start = 0; start < 32 * d; start += 32){
        // OiT[start + tx] = 0;
        int row = (start + tx)/d;
        int col = (start + tx)%d;
        // FIXME: Bank conflict while storing
        QiT[col * 32 + row] = Q[qkv_offset + (tile_size * i) + start + tx]; // Storing transposed
    }
    
    startI[tx] = startT[(bx * N) + i * Br + tx];
    endI[tx] = endT[(bx * N) + i * Br + tx];
    float row_m_prev = -INFINITY;
    float row_l_prev = 0;

    // Causal mask: j <= i
    for (int j = 0; j < Tc; ++j) { // j is the column tile index
        __syncthreads();
        
        // Load Kj, Vj from HBM to SRAM
        // Populate Kj, Vj - (32, {64, 128}) - Think like 1D 32*64 array
        for(int start = 0; start < 32 * d; start += 32){
            Kj[start + tx] = K[qkv_offset + (tile_size * j) + start + tx];
            Vj[start + tx] = V[qkv_offset + (tile_size * j) + start + tx];
        }
        
        startJ[tx] = startT[(bx * N) + j * Bc + tx];
        endJ[tx] = endT[(bx * N) + j * Bc + tx];
        __syncthreads();

        bool mask[32];
        for(int y = 0; y < 32; y++){
            mask[y] = (!IsTree) || ((startI[tx] >= startJ[y]) && (endI[tx] <= endJ[y]));
        }

        // S_i^j = softmax_scale * QiKj^T
        // S_i^j[tx][y] = softmax_scale * Sum_{x = 0}^{d-1} Qi[tx][x] * Kj[y][x]
        float row_m = -INFINITY;
        for (int y = 0; y < Bc; y++) {
            if (mask[y]){ // FIXME: Thread divergence
                float sum = 0;
                for (int x = 0; x < d; x++)
                    // FIXME: Also here we are not using register tiling for matmul
                    // before coding comment out the matmul entire and check whether you are getting speedups
                    // FIXME: We can have extra threads in the block just for matmul QK^t rest of the time they are idle
                    sum += QiT[(x * 32) + tx] * Kj[(y * d) + x];
                sum *= softmax_scale;
                S[(Bc * y) + tx] = sum;

                if (sum > row_m)
                    row_m = sum;
            }
        }

        // m_i^j = max(m_i^j-1, row_max(S_i^j))
        float new_row_m = max(row_m_prev, row_m);

        // P_i^j = exp(S_i^j - m_i^j)
        // P_i^j[tx][y] = exp(S_i^j[tx][y] - m_i^j)
        float row_l = 0;
        for (int y = 0; y < Bc; y++) {
            // causal mask
            if (mask[y]){ // FIXME: Thread divergence
                S[(Bc * y) + tx] = __expf(S[(Bc * y) + tx] - new_row_m);
                row_l += S[(Bc * y) + tx];
            }
        }

        // l_i^j = (exp(m_i^j-1 - m_i^j) * l_i^j-1) + row_sum(P_i^j)
        float row_m_exp = __expf(row_m_prev - new_row_m);
        float new_row_l = (row_m_exp * row_l_prev) + row_l;

        // O_i^j = diag(exp(m_i^j-1 - m_i^j))^-1 * O_i^j-1 + P_i^jVj
        for (int x = 0; x < d; x++) {
            float pv = 0;  // Pij * Vj
            for (int y = 0; y < Bc; y++) {
                if (mask[y]){ // FIXME: Thread divergence
                    pv += S[(Bc * y) + tx] * Vj[(y * d) + x];
                }
            }
            // int temp = OiT[x * 32 + tx];
            // temp = row_m_exp * temp + pv;
            // OiT[x * 32 + tx] = temp;
            O[qkv_offset + (tile_size * i) + (tx * d) + x] = \
                row_m_exp * O[qkv_offset + (tile_size * i) + (tx * d) + x] + pv;
        }

        // Update m and l
        row_m_prev = new_row_m;
        row_l_prev = new_row_l;
    }
    
    __syncthreads();
    
    // O_i = diag(l_i^{Tc})^-1 * O_i^{Tc}
    for (int x = 0; x < d; x++){
        O[qkv_offset + (tile_size * i) + (tx * d) + x] /= row_l_prev;
    }
    // for(int start = 0; start < 32 * d; start += 32){
    //     int row = (start + tx)/d;
    //     int col = (start + tx)%d;
    //     O[qkv_offset + (tile_size * i) + start + tx] = OiT[col * 32 + row] / row_l_prev;
    // }
}

std::vector<torch::Tensor> forward_6(torch::Tensor Q, torch::Tensor K, torch::Tensor V,
                                torch::Tensor StartTimes, torch::Tensor EndTimes, bool IsTree) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); 
    
    printf("SMs: %d\n", prop.multiProcessorCount);
    printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Max Block Size: %d x %d x %d\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
    printf("Max Share Memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Max Share Memory per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
    
    // TODO: determine Bc, Br dynamically
    const int Bc = 32;
    const int Br = 32;
    assert(Br == Bc);
    assert(Bc == 32);

    const int B = Q.size(0); const int nh = Q.size(1);
    const int N = Q.size(2); const int d = Q.size(3);
    assert(d%32 == 0);
    assert(d == D);

    const int Tc = ceil((float) N / Bc); const int Tr = ceil((float) N / Br);
    const float softmax_scale = 1.0 / sqrt(d);

    // Initialize O to HBM
    auto O = torch::zeros_like(Q);
    torch::Device device(torch::kCUDA);

    // Calculate SRAM size needed per block
    int col_tile_size = Bc * d;  // size of Kj, Vj
    int row_tile_size = Br * d;  // size of Qi
    const int sram_size =
        (2 * col_tile_size * sizeof(float))  // SRAM size for Kj, Vj
        + (2 * row_tile_size * sizeof(float))  // SRAM size for Qi, Oi
        + (2 * Bc * sizeof(float)) // SRAM for startI, endI
        + (2 * Bc * sizeof(float)) // SRAM for startJ, endJ
        + (Bc * Br * sizeof(float)); // SRAM size for S
    
    int max_sram_size = prop.sharedMemPerBlock;
    printf("Max shared memory: %d, requested shared memory: %d \n", max_sram_size, sram_size);

    dim3 grid_dim(B, nh, Tr);  // batch_size x num_heads x seq_length
    dim3 block_dim(Br); // FIXME

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    forward_6_kernel<<<grid_dim, block_dim, sram_size>>>(
        Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
        N, d, Tc, Tr, Bc, Br, softmax_scale, O.data_ptr<float>(),
        StartTimes.data_ptr<float>(), EndTimes.data_ptr<float>(), IsTree
    );

    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float milliseconds = 0; cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    return {O};
}

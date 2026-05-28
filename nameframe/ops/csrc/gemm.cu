// CUDA implementation of GEMM (General Matrix Multiply).

#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ C,
    float* __restrict__ D,
    int M, int N, int K,
    float alpha, float beta
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; ++t) {
        // load A tile [TILE_SIZE, TILE_SIZE] from global memory
        if (row < M && (t * TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        // load B tile [TILE_SIZE, TILE_SIZE] from global memory
        if ((t * TILE_SIZE + ty) < K && col < N)
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // accumulate partial dot product
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }

    if (row < M && col < N) {
        float c_val = (C != nullptr) ? C[row * N + col] : 0.0f;
        D[row * N + col] = alpha * sum + beta * c_val;
    }
}

torch::Tensor gemm_cuda(
    torch::Tensor A,
    torch::Tensor B,
    c10::optional<torch::Tensor> C,
    float alpha,
    float beta
) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "B must be float32");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.shape[1] must equal B.shape[0]");

    A = A.contiguous();
    B = B.contiguous();

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = A.options();
    auto D = torch::empty({M, N}, options);

    torch::Tensor C_contig;  // keep storage alive until kernel finishes
    const float* C_ptr = nullptr;
    if (C.has_value()) {
        auto C_tensor = C.value();
        TORCH_CHECK(C_tensor.is_cuda(), "C must be a CUDA tensor");
        TORCH_CHECK(C_tensor.dtype() == torch::kFloat32, "C must be float32");
        TORCH_CHECK(C_tensor.dim() == 2, "C must be 2D");
        TORCH_CHECK(C_tensor.size(0) == M && C_tensor.size(1) == N,
                    "C must have shape (M, N)");
        C_contig = C_tensor.contiguous();
        C_ptr = C_contig.data_ptr<float>();
    }

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    gemm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C_ptr,
        D.data_ptr<float>(),
        M, N, K,
        alpha, beta
    );

    return D;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gemm", &gemm_cuda, "GEMM CUDA kernel: D = alpha * A @ B + beta * C",
          py::arg("A"), py::arg("B"), py::arg("C") = py::none(),
          py::arg("alpha") = 1.0f, py::arg("beta") = 1.0f);
}

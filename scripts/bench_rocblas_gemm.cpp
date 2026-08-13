// Minimal rocBLAS BF16-GEMM throughput probe — compiled against each ROCm
// version's librocblas to compare gfx1151 GEMM-kernel maturity directly.
// Used to predict whether vLLM/BF16 on 7.14 (compute-bound at concurrency)
// would beat 7.2.1, WITHOUT building vLLM. vLLM BF16 inference is GEMM-dominated.
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include <cstdio>
#include <chrono>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main() {
    rocblas_handle h;
    if (rocblas_create_handle(&h) != rocblas_status_success) { printf("handle fail\n"); return 1; }
    rocblas_set_pointer_mode(h, rocblas_pointer_mode_host);

    // Representative large BF16 GEMMs (column-major C[M,N] = A[M,K] @ B[K,N]).
    // Chosen to be compute-heavy (>> memory traffic) and well above launch overhead.
    long shapes[][3] = {{4096, 4096, 4096}, {8192, 8192, 4096}, {12288, 12288, 12288}};
    const int warmup = 5, iters = 30;
    float alpha = 1.0f, beta = 0.0f;

    for (auto& s : shapes) {
        int M = (int)s[0], N = (int)s[1], K = (int)s[2];
        hip_bfloat16 *A, *B; float *C;
        hipMalloc(&A, (size_t)M * K * sizeof(hip_bfloat16));
        hipMalloc(&B, (size_t)K * N * sizeof(hip_bfloat16));
        hipMalloc(&C, (size_t)M * N * sizeof(float));
        auto gemm = [&]() {
            return rocblas_gemm_ex(h, rocblas_operation_none, rocblas_operation_none,
                M, N, K, &alpha, A, rocblas_datatype_bf16_r, M,
                B, rocblas_datatype_bf16_r, K, &beta,
                C, rocblas_datatype_f32_r, M, C, rocblas_datatype_f32_r, M,
                rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0);
        };
        for (int i = 0; i < warmup; ++i) gemm();
        hipDeviceSynchronize();
        double t0 = now_s();
        for (int i = 0; i < iters; ++i) gemm();
        hipDeviceSynchronize();
        double dt = (now_s() - t0) / iters;
        double tflops = 2.0 * M * N * K / dt / 1e12;
        printf("GEMM M=%-6d N=%-6d K=%-6d bf16 : %7.3f ms  %7.1f TFLOPS\n", M, N, K, dt * 1000.0, tflops);
        hipFree(A); hipFree(B); hipFree(C);
    }
    rocblas_destroy_handle(h);
    return 0;
}

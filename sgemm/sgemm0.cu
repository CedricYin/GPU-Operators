#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/cdefs.h>
#include <cublas_v2.h>
#include <cassert>

using namespace std;

// A: M * K; B: K * N
__global__ void sgemm_v0(const float *__restrict__ a, const float *__restrict__ b, float *c, 
                        int M, int N, int K,
                        float alpha, float beta) {
    int grow = blockDim.y * blockIdx.y + threadIdx.y;
    int gcol = blockDim.x * blockIdx.x + threadIdx.x;

    if (grow < M && gcol < N) {
        float val = 0.f;
        for (int k = 0; k < K; k++)
            val += a[grow * K + k] * b[k * N + gcol];

        c[grow * N + gcol] = alpha * val + beta * c[grow * N + gcol];
    }
}

void init(float *a, float *b, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) {
        a[i] = 1.f;
        b[i] = 1.f;
    }
}

bool check_ans(float *truth, float *c, int M, int N) {
    for (int i = 0; i < M * N; i++)
        if (fabs(c[i] - truth[i]) > 0.5) {
            printf("truth: %f, output: %f\n", truth[i], c[i]);
            return false;
        }
    return true;
}

int main(int argc, char **argv) {
    // ncu: nvcc -std=c++17 -lineinfo -DPROFILE -o sgemm sgemm.cu        ./sgemm 0 1
    // compute-santi: nvcc -std=c++17 -o sgemm sgemm.cu                  ./sgemm 0 1 256 256 128
    // v100:          nvcc -std=c++17 -o sgemm sgemm.cu                  ./sgemm
    int M = 8192;
    int N = 8192;
    int K = 4096;
    int nWarmup = 2;
    int nIters = 50;
    assert(argc == 1 || argc == 3 || argc == 6);
    if (argc >= 3) {
        nWarmup = atoi(argv[1]);
        nIters = atoi(argv[2]);
    }
    if (argc >= 6) {
        M = atoi(argv[3]);
        N = atoi(argv[4]);
        K = atoi(argv[5]);
    }
    const float alpha = 1.f;
    const float beta = 0.f;
    const int BLOCKDIM = 32;
    float elapsed_my = 0.f;
    float elapsed_cublas = 0.f;

    size_t size = sizeof(float) * M * N;
    float *a_h = (float *) malloc(size);
    float *b_h = (float *) malloc(size);
    float *c_h = (float *) malloc(size);
    float *c_truth = (float *) malloc(size);
    init(a_h, b_h, M, N, K);

    float *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    dim3 gridDim(ceil(1.f * M / BLOCKDIM), ceil(1.f * N / BLOCKDIM), 1);
    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // my sgemm
    for (int i = 0; i < nIters + nWarmup; i++) {
        cudaEventRecord(start);
        sgemm_v0<<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i >= nWarmup) {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << i - nWarmup << ": " << ms << " ms\n";
            elapsed_my += ms;
        }
    }
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    #ifndef PROFILE
    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < nIters; i++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_d, N, a_d, K, &beta, c_d, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        elapsed_cublas += ms;
    }
    cublasDestroy(handle);
    cudaMemcpy(c_truth, c_d, size, cudaMemcpyDeviceToHost);

    // check
    if (!check_ans(c_truth, c_h, M, N)) {
        cerr << "result is wrong!" << endl;
        return -1;
    }
    cout << "result is right" << endl;

    // output
    const int64_t flop = int64_t(M) * int64_t(N) * int64_t(K) * 2;
    double gflops_my = flop / ((elapsed_my / nIters) / 1000) / 1e9;
    double gflops_cublas = flop / ((elapsed_cublas / nIters) / 1000) / 1e9;
    cout << "mysgemm: " << gflops_my << "GFLOPS (" << flop << " flop, " << (elapsed_my / nIters) / 1000 << "s)\n";
    cout << "cublas: " << gflops_cublas << "GFLOPS (" << flop << " flop, " << (elapsed_cublas / nIters) / 1000 << "s)\n";
    cout << "% of cublas: " << gflops_my / gflops_cublas * 100 << "%" << endl;
    #endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_truth);

    return 0;
}
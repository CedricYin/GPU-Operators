#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>

using namespace std;

#define M 1024
#define N 1024
#define K 1024
#define BLOCKDIM 32

// A: M * K; B: K * N
__global__ void gemm0_baseline(float *a, float *b, float *c) {
    int grow = blockDim.y * blockIdx.y + threadIdx.y;
    int gcol = blockDim.x * blockIdx.x + threadIdx.x;

    if (grow < M && gcol < N) {
        float val = 0.f;
        for (int k = 0; k < K; k++)
            val += a[grow * K + k] * b[k * N + gcol];

        c[grow * N + gcol] = val;
    }
}

float* init(float *a, float *b) {
    for (int i = 0; i < M * N; i++) {
        a[i] = 1.f;
        b[i] = 1.f;
    }
    float *c = (float *) calloc(M * N, sizeof(float));
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                c[i * N + j] += a[i * K + k] * b[k * N + j];
    
    return c;
}

bool check_ans(float *truth, float *c) {
    for (int i = 0; i < M * N; i++)
        if (fabs(c[i] - truth[i]) > 0.5) {
            printf("truth: %f, output: %f\n", truth[i], c[i]);
            return false;
        }
    return true;
}

int main() {
    size_t size = sizeof(float) * M * N;
    float *a_h = (float *) malloc(size);
    float *b_h = (float *) malloc(size);
    float *c_h = (float *) malloc(size);
    float *c_truth = init(a_h, b_h);

    float *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed = 0.f;
    const int nWarmup = 2;
    const int nIters = 3;

    dim3 gridDim(ceil(1.f * M / BLOCKDIM), ceil(1.f * N / BLOCKDIM), 1);
    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);  // block 的y维缩小COARSENING_FACTOR倍

    for (int i = 0; i < nIters + nWarmup; i++) {
        cudaEventRecord(start);
        gemm0_baseline<<<gridDim, blockDim>>>(a_d, b_d, c_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i < nWarmup) {
            if (i == 0) {
                cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
                if (!check_ans(c_truth, c_h)) {
                    cerr << "result is wrong!" << endl;
                    return -1;
                }
                cout << "result is right" << endl;
            }
        } else {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << i - nWarmup << ": " << ms << " ms\n";
            elapsed += ms;
        }
    }
    const int64_t flop = int64_t(M) * int64_t(N) * int64_t(K) * 2;
    double gflops = flop / ((elapsed / nIters) / 1000) / 1e9;
    cout << "kernel: " << gflops << "GFLOPS (" << flop << " flop, " << (elapsed / nIters) / 1000 << "s)\n";

    return 0;
}
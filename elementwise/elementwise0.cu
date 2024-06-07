#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define N 8192*8192
#define BLOCKDIM 1024

__global__ void elementwise0(const float *__restrict__ a, const float *__restrict__ b, float *c) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

bool check_ans(float *arr) {
    for (int i = 0; i < N; i++)
        if (fabs(arr[i] - 3.f) > FLT_EPSILON)
            return false;
    return true;
}

int main() {
    size_t size = N * sizeof(float);
    float *a_h = (float *) malloc(size);
    float *b_h = (float *) malloc(size);
    float *c_h = (float *) malloc(size);
    for (int i = 0; i < N; i++) {
        a_h[i] = 1.f;
        b_h[i] = 2.f;
    }

    float *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    dim3 gridDim(ceil(1.f * N / BLOCKDIM));
    dim3 blockDim(BLOCKDIM);

    const int nWarmup = 2;
    const int nIter = 10;
    float elapsedTime = 0;
    for (int i = 0; i < nWarmup + nIter; i++) {
        cudaEventRecord(start);
        elementwise0<<<gridDim, blockDim>>>(a_d, b_d, c_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i == 0) {
            cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);
            if (!check_ans(c_h)) {
                cerr << "answer is wrong!" << endl;
                return -1;
            }
        }
        if (i >= nWarmup) {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << i - nWarmup << ": " << ms << " ms\n";
            elapsedTime += ms;
        }
    }

    const unsigned int nbytes = 3 * size;
    double bw = 1. * nbytes / (elapsedTime / nIter / 1000) / 1e9;
    cout << "effective bandwidth: " << bw << "GB/s\n";
    cout << "% of V100 peak bandwidth: " << bw / 900 * 100 << "%" << endl;
    return 0;
}
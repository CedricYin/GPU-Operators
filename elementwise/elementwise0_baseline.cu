#include <cstddef>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

#define N 1024 * 1024 * 32
#define BLOCKDIM 1024

__global__ void elementwise0_baseline(float *a, float *b, float *c) {
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
    cudaEventRecord(start);
    elementwise0_baseline<<<gridDim, blockDim>>>(a_d, b_d, c_d);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    if (!check_ans(c_h)) {
        cerr << "answer is wrong!" << endl;
        return -1;
    }

    cout << "elapsedTime: " << elapsedTime * 1000 << " ns" << endl;
    return 0;
}
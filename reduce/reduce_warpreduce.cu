#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 256 * 1024 * 1024
#define BLOCKDIM 1024
#define COARSE_FACTOR 2

using namespace std;

__device__ void warp_reduce(volatile int *input, unsigned tid) {
    input[tid] += input[tid + 32];
    input[tid] += input[tid + 16];
    input[tid] += input[tid + 8];
    input[tid] += input[tid + 4];
    input[tid] += input[tid + 2];
    input[tid] += input[tid + 1];
}

// using warp reduce
__global__ void reduce_warp_reduce(int *input, int *output) {
    extern __shared__ int input_s[];
    unsigned start_idx = COARSE_FACTOR * 2 * BLOCKDIM * blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned i = start_idx + tid;

    int sum = input[i];  // local var
    for (unsigned tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        sum += input[i + tile * BLOCKDIM];
    }
    input_s[tid] = sum;
    __syncthreads();

    for (unsigned stride = BLOCKDIM / 2; stride > 32; stride /= 2) {
        if (tid < stride)
            input_s[tid] += input_s[tid + stride];
        __syncthreads();
    }
    
    if (tid < 32) {
        warp_reduce(input_s, tid);
    }

    if (tid == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main() {
    int *input_h = NULL;
    int *output_h = NULL;
    int *input_d = NULL;
    int *output_d = NULL;

    input_h = (int *) malloc(N * sizeof(int));
    output_h = (int *) malloc(sizeof(int));

    for (int i = 0; i < N; i++)
        input_h[i] = 1;

    cudaMalloc((void **) &input_d, N * sizeof(int));
    cudaMalloc((void **) &output_d, sizeof(int));

    cudaMemcpy(input_d, input_h, N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 gridDim(N / BLOCKDIM / 2 / COARSE_FACTOR);
    dim3 blockDim(BLOCKDIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_warp_reduce<<<gridDim, blockDim, BLOCKDIM * sizeof(int)>>>(input_d, output_d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    cudaMemcpy(output_h, output_d, sizeof(int), cudaMemcpyDeviceToHost);

    int result = 0;
    for (int i = 0; i < N; i++)
        result += input_h[i];
    cout << "expected output: " << result << endl;
    cout << "output: " << *output_h << endl;
    assert(result == *output_h);
    cout << "time consumed: " << elapsedTime << "ms" << endl;

    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
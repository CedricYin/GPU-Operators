#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 1024 * 1024
#define BLOCKDIM 1024

using namespace std;

// Hierarchical reduction for arbitrary input length
__global__ void reduce_arbitrary_length(int *input, int *output) {
    extern __shared__ int input_s[];
    unsigned start_idx = 2 * BLOCKDIM * blockIdx.x;
    unsigned tid = threadIdx.x;
    unsigned i = start_idx + tid;

    input_s[tid] = input[i] + input[i + BLOCKDIM];
    for (unsigned stride = BLOCKDIM / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (tid < stride)
            input_s[tid] += input_s[tid + stride];
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

    dim3 gridDim(N / BLOCKDIM / 2);
    dim3 blockDim(BLOCKDIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_arbitrary_length<<<gridDim, blockDim, BLOCKDIM * sizeof(int)>>>(input_d, output_d);

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
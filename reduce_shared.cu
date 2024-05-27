#include <__clang_cuda_builtin_vars.h>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 2048

using namespace std;

// Minimizing global memory accesses
__global__ void reduce_shared(int *input, int *output) {
    unsigned i = threadIdx.x;
    extern __shared__ int input_s[];

    input_s[i] = input[i] + input[i + blockDim.x];
    for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride)
            input_s[i] += input_s[i + stride];
    }

    if (threadIdx.x == 0) {
        *output = input_s[0];
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

    dim3 gridDim(1);
    dim3 blockDim(N / 2);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_shared<<<gridDim, blockDim, N / 2 * sizeof(int)>>>(input_d, output_d);

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
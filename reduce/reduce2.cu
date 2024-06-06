#include <cfloat>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// Minimizing global memory accesses
__global__ void reduce2(float *input, float *output) {
    const int start_idx = 2 * blockDim.x * blockIdx.x;
    const int i = start_idx + threadIdx.x;
    const int tx = threadIdx.x;
    extern __shared__ float input_s[];

    input_s[tx] = input[i] + input[i + blockDim.x];
    for (unsigned stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride)
            input_s[tx] += input_s[tx + stride];
    }

    if (threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}

int main(int argc, char **argv) {
    constexpr int N = 8192 * 8192;
    constexpr int BLOCKDIM = 256;
    constexpr float result = N * 1.f;
    int nWarmup = 2;
    int nIters = 10;
    assert(argc == 1 || argc == 3);
    if (argc == 3) {
        nWarmup = atoi(argv[1]);
        nIters = atoi(argv[2]);
    }
    float elapsedTime;
    float *input_h = NULL;
    float *output_h = NULL;
    float *input_d = NULL;
    float *output_d = NULL;

    input_h = (float *) malloc(N * sizeof(float));
    output_h = (float *) malloc(sizeof(float));

    for (int i = 0; i < N; i++)
        input_h[i] = 1.f;

    cudaMalloc((void **) &input_d, N * sizeof(float));
    cudaMalloc((void **) &output_d, sizeof(float));

    cudaMemcpy(input_d, input_h, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 gridDim(N / BLOCKDIM / 2);
    dim3 blockDim(BLOCKDIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < nWarmup + nIters; i++) {
        cudaEventRecord(start);
        reduce2<<<gridDim, blockDim, sizeof(float) * BLOCKDIM>>>(input_d, output_d);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i == 0) {
            cudaMemcpy(output_h, output_d, sizeof(float), cudaMemcpyDeviceToHost);
            cout << "result: " << result << ", output: " << *output_h << '\n';
            assert(fabs(result - (*output_h)) < FLT_EPSILON);
            cout << "result is right\n";
        }
        if (i >= nWarmup) {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << i - nWarmup << ": " << ms << " ms\n";
            elapsedTime += ms;
        }
    }

    double bw = 4. * N / (elapsedTime / nIters / 1000) / 1e9;
    cout << "effective bandwidth: " << bw << "GB/s" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    free(input_h);
    free(output_h);
    cudaFree(input_d);
    cudaFree(output_d);
    return 0;
}
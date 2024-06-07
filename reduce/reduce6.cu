#include <cassert>
#include <cstdlib>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>

using namespace std;

template<int BLOCKDIM>
__device__ __forceinline__ float warp_reduce(volatile float *input, unsigned tx) {
    float x = input[tx];
    if (BLOCKDIM >= 64) {x += input[tx + 32]; __syncwarp();}
    if (BLOCKDIM >= 32) x += __shfl_down_sync(0xffffffff, x, 16);
    if (BLOCKDIM >= 16) x += __shfl_down_sync(0xffffffff, x,  8);
    if (BLOCKDIM >=  8) x += __shfl_down_sync(0xffffffff, x,  4);
    if (BLOCKDIM >=  4) x += __shfl_down_sync(0xffffffff, x,  2);
    if (BLOCKDIM >=  2) x += __shfl_down_sync(0xffffffff, x,  1);
    return x;
}

// using block reduce
template<int BLOCKDIM, int COARSE_FACTOR>
__global__ void reduce6(float *input, float *output) {
    extern __shared__ float input_s[];
    const int start_idx = COARSE_FACTOR * 2 * BLOCKDIM * blockIdx.x;
    const int tx = threadIdx.x;
    const int i = start_idx + tx;

    float sum = input[i];  // local var
    for (unsigned tile = 1; tile < COARSE_FACTOR * 2; tile++) {
        sum += input[i + tile * BLOCKDIM];
    }
    input_s[tx] = sum;
    __syncthreads();

    // unrolling: just like block reduce
    if (BLOCKDIM >= 1024) {
      if (tx < 512) {
        input_s[tx] += input_s[tx + 512];
      }
      __syncthreads();
    }
    if (BLOCKDIM >= 512) {
      if (tx < 256) {
        input_s[tx] += input_s[tx + 256];
      }
      __syncthreads();
    }
    if (BLOCKDIM >= 256) {
      if (tx < 128) {
        input_s[tx] += input_s[tx + 128];
      }
      __syncthreads();
    }
    if (BLOCKDIM >= 128) {
      if (tx < 64) {
        input_s[tx] += input_s[tx + 64];
      }
      __syncthreads();
    }
    
    float result;
    if (tx < 32) {
        result = warp_reduce<BLOCKDIM>(input_s, tx);
    }

    if (tx == 0) {
        atomicAdd(output, result);
    }
}

int main(int argc, char **argv) {
    constexpr int N = 8192 * 8192;
    constexpr int BLOCKDIM = 256;
    constexpr int COARSE_FACTOR = 2;
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

    dim3 gridDim(N / BLOCKDIM / 2 / COARSE_FACTOR);
    dim3 blockDim(BLOCKDIM);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (int i = 0; i < nWarmup + nIters; i++) {
        cudaEventRecord(start);
        reduce6<BLOCKDIM, COARSE_FACTOR><<<gridDim, blockDim, sizeof(float) * BLOCKDIM>>>(input_d, output_d);
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
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
#define VECTOR_FACTOR 4

#define FETCH_VEC4(ptr) (((float4 *) (ptr))[0])

__global__ void elementwise2(const float *__restrict__ a, const float *__restrict__ b, float *c) {
    int i = (blockDim.x * blockIdx.x + threadIdx.x) * VECTOR_FACTOR;
    float4 vec4_a = FETCH_VEC4(&a[i]);
    float4 vec4_b = FETCH_VEC4(&b[i]);
    float4 vec4_c;
    vec4_c.x = vec4_a.x + vec4_b.x;
    vec4_c.y = vec4_a.y + vec4_b.y;
    vec4_c.z = vec4_a.z + vec4_b.z;
    vec4_c.w = vec4_a.w + vec4_b.w;
    FETCH_VEC4(&c[i]) = vec4_c;
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
    dim3 gridDim(ceil(1.f * N / BLOCKDIM / VECTOR_FACTOR));
    dim3 blockDim(BLOCKDIM);
    const int nWarmup = 2;
    const int nIter = 10;
    float elapsedTime = 0;
    for (int i = 0; i < nWarmup + nIter; i++) {
        cudaEventRecord(start);
        elementwise2<<<gridDim, blockDim>>>(a_d, b_d, c_d);
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

    const unsigned int nbytes = 3 * N * 4;
    double bw = 1. * nbytes / (elapsedTime / nIter / 1000) / 1e9;
    cout << "effective bandwidth: " << bw << "GB/s\n";
    cout << "% of V100 peak bandwidth: " << bw / 900 * 100 << "%" << endl;
    return 0;
}
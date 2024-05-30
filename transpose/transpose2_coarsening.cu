#include <cstddef>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>

using namespace std;

#define checkCudaErrors(func)				                                            \
{									                                                    \
    cudaError_t e = (func);			                                                    \
    if(e != cudaSuccess)						                                        \
        printf ("%s %d CUDA: %s\n", __FILE__,  __LINE__, cudaGetErrorString(e));		\
}


template<int COARSE_FACTOR>
__global__ void transpose2_coarsening(float *a, float *b, size_t rows, size_t cols) {
    unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned row = blockDim.y * blockIdx.y * COARSE_FACTOR + threadIdx.y;
    unsigned idx_a = row * cols + col;
    unsigned idx_b = col * rows + row;

    if (row < rows && col < cols) {
        #pragma unroll
        for (int i = 0; i < COARSE_FACTOR; i++)
            b[idx_b + i * blockDim.y] = a[idx_a + i * blockDim.y * cols];
    }
}

static float *truth;
bool check_ans(float *arr, size_t size) {
    for (int i = 0; i < size; i++) {
        if (fabs(truth[i] - arr[i]) > FLT_EPSILON) {
            return false;
        }
    }
    return true;
}

int main() {
    const int ROWS = 4096;
    const int COLS = 4096 * 32;
    const int N = ROWS * COLS;
    const int BLOCKDIM = 64;
    const int COARSE_FACTOR = 4;

    size_t size = N * sizeof(float);
    float *a_h = (float *) malloc(size);
    for (int i = 0; i < N; i++) {
        a_h[i] = 1.f * i;
    }
    float *b_h = (float *) malloc(size);
    truth = (float *) malloc(size);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++)
            truth[j * ROWS + i] = a_h[i * COLS + j];
    }

    float *a_d, *b_d;
    checkCudaErrors(cudaMalloc((void **) &a_d, size));
    checkCudaErrors(cudaMalloc((void **) &b_d, size));
    checkCudaErrors(cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice));
    
    dim3 gridDim(ceil(1.f * COLS / BLOCKDIM), ceil(1.f * ROWS / BLOCKDIM), 1);
    dim3 blockDim(BLOCKDIM, BLOCKDIM / COARSE_FACTOR, 1);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    const int nWarmup = 1;
    const int nIter = 10;
    float elapsed = 0.f;

    for (int i = 0; i < nWarmup + nIter; i++) {
        checkCudaErrors(cudaEventRecord(start));
        transpose2_coarsening<COARSE_FACTOR><<<gridDim, blockDim>>>(a_d, b_d, ROWS, COLS);
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaEventSynchronize(stop));
        if (i < nWarmup) {
            if (i == 0) {
                checkCudaErrors(cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost));
                if (!check_ans(b_h, N)) {
                    cerr << "result is wrong!" << endl;
                    return -1;
                }
                cout << "result is right" << endl;
            }
        } else {
            float ms;
            checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
            elapsed += ms;
        }
    }
    cout << "average elapsed time: " << elapsed / nIter << "ms" << endl;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(a_d));
    checkCudaErrors(cudaFree(b_d));
    free(a_h);
    free(b_h);
    free(truth);
    return 0;
}
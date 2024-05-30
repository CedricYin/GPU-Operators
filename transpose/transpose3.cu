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


template<int TILEDIM>
__global__ void transpose3_shared(float *input, float *output, size_t rows, size_t cols) {
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int igy_start = blockDim.y * blockIdx.y;
    int igx_start = blockDim.x * blockIdx.x;
    int ogy_start = blockDim.y * blockIdx.x;
    int ogx_start = blockDim.x * blockIdx.y;
    __shared__ float tile[TILEDIM][TILEDIM];

    if (igy_start + ty < rows && igx_start + tx < cols) {
        tile[ty][tx] = input[(igy_start + ty) * cols + (igx_start + tx)];
        __syncthreads();

        output[(ogy_start + ty) * rows + (ogx_start + tx)] = tile[tx][ty];  // 转置后的leading dimension变成了rows
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
    const int BLOCKDIM = 32;
    const int TILEDIM = 32;
    const size_t accessed_bytes = N * sizeof(float) * 2;

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
    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    const int nWarmup = 1;
    const int nIter = 10;
    float elapsed = 0.f;

    for (int i = 0; i < nWarmup + nIter; i++) {
        checkCudaErrors(cudaEventRecord(start));
        transpose3_shared<TILEDIM><<<gridDim, blockDim>>>(a_d, b_d, ROWS, COLS);
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
    cout << "average elapsed time: " << elapsed / nIter << "ms\n";
    double bw = accessed_bytes / (elapsed / nIter / 1000) / 1e9;
    cout << "bandwidth: " << bw << "GB/s, " << bw / 900 * 100 << "% of peak bandwidth" << endl;

    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(a_d));
    checkCudaErrors(cudaFree(b_d));
    free(a_h);
    free(b_h);
    free(truth);
    return 0;
}
#include <cstddef>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>

using namespace std;

#define ROWS 4096
#define COLS 4096 * 32
#define N (ROWS * COLS)  // 要加括号，因为有空格
#define BLOCKDIM 32

// read by row and write by col
__global__ void transpose0_baseline(float *a, float *b, size_t rows, size_t cols) {
    unsigned col = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned row = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned idx_a = row * cols + col;
    unsigned idx_b = col * rows + row;

    if (row < rows && col < cols) {
        b[idx_b] = a[idx_a];
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

void print_matrix(float *matrix, int row, int col) {
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
            cout << matrix[i * col + j] << ' ';
        cout << '\n';
    }
    cout << '\n';
}

int main() {
    size_t size = N * sizeof(float);
    float *a_h = (float *) malloc(size);
    for (size_t i = 0; i < N; i++) {
        a_h[i] = 1.f * i;
    }

    float *b_h = (float *) malloc(size);
    truth = (float *) malloc(size);
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++)
            truth[j * ROWS + i] = a_h[i * COLS + j];
    }

    float *a_d, *b_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    
    dim3 gridDim(ceil(1.f * COLS / BLOCKDIM), ceil(1.f * ROWS / BLOCKDIM), 1);
    dim3 blockDim(BLOCKDIM, BLOCKDIM, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int nWarmup = 1;
    const int nIter = 10;
    float elapsed = 0.f;

    for (int i = 0; i < nWarmup + nIter; i++) {
        cudaEventRecord(start);
        transpose0_baseline<<<gridDim, blockDim>>>(a_d, b_d, ROWS, COLS);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i < nWarmup) {
            if (i == 0) {
                cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);
                if (!check_ans(b_h, N)) {
                    cerr << "result is wrong!" << endl;
                    return -1;
                }
                cout << "result is right" << endl;
            }
        } else {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            elapsed += ms;
        }
    }
    cout << "average elapsed time: " << elapsed / nIter << "ms" << endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    free(a_h);
    free(b_h);
    free(truth);
    return 0;
}
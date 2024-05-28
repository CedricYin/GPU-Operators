#include <cstddef>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <cstdlib>

using namespace std;

#define ROWS 1024
#define COLS 1024 * 32
#define N (ROWS * COLS)  // 要加括号，因为有空格
#define BLOCKDIM 1024

// read by col and write by row
// 优化思路：使用L1cache来优化read data，期盼cache中有下一行的数据；write data由于必须要写回显存，无法用cache优化
// 该方案和baseline几乎没什么区别
__global__ void transpose1_colread(float *a, float *b, size_t rows, size_t cols) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_a = col * rows + row;  // read by col
    int idx_b = row * cols + col;  // write by row

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

int main() {
    size_t size = N * sizeof(float);
    float *a_h = (float *) malloc(size);
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
    
    dim3 gridDim(ceil(1.f * N / BLOCKDIM));
    dim3 blockDim(BLOCKDIM);
    transpose1_colread<<<gridDim, blockDim>>>(a_d, b_d, ROWS, COLS);
    cudaMemcpy(b_h, b_d, size, cudaMemcpyDeviceToHost);

    if (!check_ans(b_h, N)) {
        for (int i = 0; i < N; i++) {
            cout << b_h[i] << ' ';
        }
        cout << endl;
        cerr << "result is wrong!" << endl;
        return -1;
    }

    return 0;
}
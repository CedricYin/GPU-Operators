#include <__clang_cuda_builtin_vars.h>
#include <cfloat>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// #define OFFSET(row, col, ld) ((row) * (ld) + (col))

void spmv_csr_cpu(const uint *A_offset, const uint *A_col_index,
                  const float *A_val, const uint M, const uint N,
                  const float *x, float *y) {
  for (uint i = 0; i < M; i++) {
    float sum = 0.f;
    for (uint j = A_offset[i]; j < A_offset[i + 1]; j++) {
      sum += A_val[j] * x[A_col_index[j]];
    }
    y[i] = sum;
  }
}

template <uint width = 32>
__device__ __forceinline__ float warp_reduce(float x) {
  if (width >= 32)
    x += __shfl_down_sync(0xffffffff, x, 16, width);
  if (width >= 16)
    x += __shfl_down_sync(0xffffffff, x, 8, width);
  if (width >= 8)
    x += __shfl_down_sync(0xffffffff, x, 4, width);
  if (width >= 4)
    x += __shfl_down_sync(0xffffffff, x, 2, width);
  if (width >= 2)
    x += __shfl_down_sync(0xffffffff, x, 1, width);
  return x;
}

// blockDim(32, 4)
// gridDim(M / 4)
__global__ void spmv_csr(const uint *A_offset, const uint *A_col_index,
                         const float *A_val, const uint M, const uint N,
                         const float *x, float *y) {

  const uint bx = blockIdx.x;
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  const uint row_id = bx * blockDim.y + ty;
  const uint row_start = A_offset[row_id];
  const uint row_end = A_offset[row_id + 1];

  float sum = 0.f;
#pragma unroll
  // 当前row的元素个数可能比32(blockDim.x)多，也可能比32少，所以用以下for循环处理这些情况
  for (int i = row_start + tx; i < row_end; i += blockDim.x) {
    sum += A_val[i] * x[A_col_index[i]];
  }
  sum = warp_reduce(sum);

  if (tx == 0)
    y[row_id] = sum;
}
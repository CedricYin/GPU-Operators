#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

#define CHECK(func)                                                            \
  {                                                                            \
    cudaError_t e = (func);                                                    \
    if (e != cudaSuccess)                                                      \
      printf("%s %d CUDA: %s\n", __FILE__, __LINE__, cudaGetErrorString(e));   \
  }

template <uint width> __device__ __forceinline__ float warp_reduce(float x) {
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

__global__ void sgemv2(float *__restrict__ A, float *__restrict__ b, float *c,
                       const uint M, const uint N) {
  const uint bx = blockIdx.x;
  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;
  uint row = bx * blockDim.y * 2 + ty * 2 + tx / 16;
  uint col = tx & (16U - 1);
  float pval = A[OFFSET(row, col, N)] * b[col];
  float val = warp_reduce<16>(pval);
  if (tx & 0xf) {
    c[row] = val;
  }
}

void init(float *A, float *b, uint M, uint N) {
  for (uint i = 0; i < M * N; i++) {
    A[i] = 1.f;
    if (i < N)
      b[i] = 1.f;
  }
}

void check_ans(const float *a, const float *b, const uint M) {
  for (uint i = 0; i < M; i++) {
    if (fabs(a[i] - b[i]) > FLT_EPSILON) {
      cout << "result is wrong" << endl;
      exit(1);
    }
  }
  cout << "result is right" << endl;
}

int main(int argc, char **argv) {
  uint M = 16384;
  uint N = 16;
  const uint64_t flop = 2L * N * M;
  uint nWarmup = 5;
  uint nIters = 1000;
  float elapsed_my = 0.f;
  float elapsed_cublas = 0.f;
  uint sizeA = sizeof(float) * M * N;
  uint sizeb = sizeof(float) * N * 1;
  uint sizec = sizeof(float) * M * 1;
  float *A_h = (float *)malloc(sizeA);
  float *b_h = (float *)malloc(sizeb);
  float *c_h = (float *)malloc(sizec);
  float *c_truth = (float *)malloc(sizec);
  float *A_d = NULL, *b_d = NULL, *c_d = NULL;
  CHECK(cudaMalloc((void **)&A_d, sizeA));
  CHECK(cudaMalloc((void **)&b_d, sizeb));
  CHECK(cudaMalloc((void **)&c_d, sizec));
  init(A_h, b_h, M, N);
  CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d, b_h, sizeb, cudaMemcpyHostToDevice));
  dim3 blockDim(32, 4);
  dim3 gridDim(M / 4 / 2);

  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  cublasHandle_t handle;
  cublasCreate(&handle);
  float alpha = 1.f;
  float beta = 0.f;

  for (uint i = 0; i < nWarmup + nIters; i++) {
    CHECK(cudaEventRecord(start));
    sgemv2<<<gridDim, blockDim>>>(A_d, b_d, c_d, M, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    if (i == 0) {
      CHECK(cudaMemcpy(c_h, c_d, sizec, cudaMemcpyDeviceToHost));
      CHECK(cudaMemset(c_d, 0, sizec));
      cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, A_d, N, b_d, 1, &beta, c_d,
                  1);
      CHECK(cudaMemcpy(c_truth, c_d, sizec, cudaMemcpyDeviceToHost));
      check_ans(c_h, c_truth, M);
    }
    if (i >= nWarmup) {
      float ms;
      cudaEventElapsedTime(&ms, start, stop);
      cout << i - nWarmup << ": " << ms << " ms\n";
      elapsed_my += ms;
    }
  }

  for (uint i = 0; i < nIters; i++) {
    CHECK(cudaEventRecord(start));
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha, A_d, N, b_d, 1, &beta, c_d,
                1);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    elapsed_cublas += ms;
  }

  double gflops_my = flop / ((elapsed_my / nIters) / 1000) / 1e9;
  double gflops_cublas = flop / ((elapsed_cublas / nIters) / 1000) / 1e9;
  cout << "mysgemm: " << gflops_my << "GFLOPS (" << flop << " flop, "
       << (elapsed_my / nIters) / 1000 << "s)\n";
  cout << "cublas: " << gflops_cublas << "GFLOPS (" << flop << " flop, "
       << (elapsed_cublas / nIters) / 1000 << "s)\n";
  cout << "% of cublas: " << gflops_my / gflops_cublas * 100 << "%" << endl;

  cublasDestroy(handle);
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  CHECK(cudaFree(A_d));
  CHECK(cudaFree(b_d));
  CHECK(cudaFree((c_d)));
  free(A_h);
  free(b_h);
  free(c_h);
  free(c_truth);
  return 0;
}
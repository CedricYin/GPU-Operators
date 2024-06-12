#include <cstdint>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

// 0.295232 ms
__global__ void copy_if1(int * dst, const int *src, int *len, const uint N) {
  const uint i = blockDim.x * blockIdx.x + threadIdx.x;
  const uint tx = threadIdx.x;
  __shared__ int klen_s;
  if (tx == 0) {
    klen_s = 0;
  }
  __syncthreads();

  if (i >= N) return ;

  int d = src[i];
  int kpos;
  if (d > 0) {
    kpos = atomicAdd(&klen_s, 1);
  }
  __syncthreads();

  if (tx == 0) {
    klen_s = atomicAdd(len, klen_s);
  }
  __syncthreads();

  if (d > 0) {
    dst[klen_s + kpos] = d;
  }
  
}
bool CheckResult(int *out, int groudtruth, int n) {
  if (*out != groudtruth) {
    return false;
  }
  return true;
}

int main() {
  float milliseconds = 0;
  int N = 25600000;
  const int blockSize = 256;
  int GridSize = (int)ceil(1.f * N / blockSize);

  int *src_h = (int *)malloc(N * sizeof(int));
  int *dst_h = (int *)malloc(N * sizeof(int));
  int *nres_h = (int *)malloc(1 * sizeof(int));
  int *dst, *nres;
  int *src;
  cudaMalloc((void **)&src, N * sizeof(int));
  cudaMalloc((void **)&dst, N * sizeof(int));
  cudaMalloc((void **)&nres, 1 * sizeof(int));

  for (int i = 0; i < N; i++) {
    src_h[i] = 1;
  }

  int groudtruth = 0;
  for (int j = 0; j < N; j++) {
    if (src_h[j] > 0) {
      groudtruth += 1;
    }
  }

  cudaMemcpy(src, src_h, N * sizeof(int), cudaMemcpyHostToDevice);

  dim3 Grid(GridSize);
  dim3 Block(blockSize);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  copy_if1<<<Grid, Block>>>(dst, src, nres, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaMemcpy(nres_h, nres, 1 * sizeof(int), cudaMemcpyDeviceToHost);
  bool is_right = CheckResult(nres_h, groudtruth, N);
  if (is_right) {
    printf("the ans is right\n");
  } else {
    printf("the ans is wrong\n");
    printf("%d ", *nres_h);
    printf("\n");
  }
  printf("filter_k latency = %f ms\n", milliseconds);

  cudaFree(src);
  cudaFree(dst);
  cudaFree(nres);
  free(src_h);
  free(dst_h);
  free(nres_h);
}
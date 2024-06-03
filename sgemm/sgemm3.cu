#include <cstddef>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <sys/cdefs.h>
#include <cublas_v2.h>

#define OFFSET(row, col, stride) ((row) * (stride) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

using namespace std;

template<int TM, int TN, int TK, int RM, int RN>
__global__ void sgemm_v3(float *__restrict__ A, float *__restrict__ B, float *C, 
                        int M, int N, int K,
                        float alpha, float beta) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    constexpr int thread_nums = (TM / RM) * (TN / RN);
    // (y_st, x_st) 是每个线程所负责的RM*RN块的左上角坐标
    const int x_st = (threadIdx.x % (TN / RN)) * RN;
    const int y_st = (threadIdx.x / (TM / RM)) * RM;
    
    __shared__ float As[TK][TM];  // 为了便于float4读取，这里As进行转置
    __shared__ float Bs[TK][TN];

    // 移动到当前要处理的C block，以及A和B的起始位置
    A = &A[by * TM * K];
    B = &B[bx * TN];
    C = &C[by * TM * N + bx * TN];

    // 重新将一维的线程组织成二维的:
    // 组织成有TK列的线程，用来搬运数据到As
    const int a_tile_y = threadIdx.x / (TK / 4);
    const int a_tile_x = threadIdx.x % (TK / 4) * 4;
    constexpr int a_tile_stride = thread_nums / (TK / 4); // 每轮跨越的行数
    // 组织成有TN列的线程，用来搬运数据到Bs
    const int b_tile_y = threadIdx.x / (TN / 4);
    const int b_tile_x = threadIdx.x % (TN / 4) * 4;
    constexpr int b_tile_stride = thread_nums / (TN / 4);

    float pval[RM][RN] = {0.f};  // 每个线程负责RM*RN个位置
    // float a_tmp[4 * (TM * TK) / thread_nums / 4] = {0.f};
    
    for (int phase = 0; phase < K; phase += TK) {
        // global to shared
        for (int i = 0; i < TM; i += a_tile_stride) {
            // int a_tmp_idx = i / a_tile_stride * 4;
            // FETCH_FLOAT4(a_tmp[a_tmp_idx]) = FETCH_FLOAT4(A[OFFSET(a_tile_y + i, a_tile_x, K)]);
            // As[a_tile_x][a_tile_y + i] = a_tmp[a_tmp_idx];
            // As[a_tile_x + 1][a_tile_y + i] = a_tmp[a_tmp_idx + 1];
            // As[a_tile_x + 2][a_tile_y + i] = a_tmp[a_tmp_idx + 2];
            // As[a_tile_x + 3][a_tile_y + i] = a_tmp[a_tmp_idx + 3];
            float a_tmp[4] = {0.f};
            FETCH_FLOAT4(a_tmp[0]) = FETCH_FLOAT4(A[OFFSET(a_tile_y + i, a_tile_x, K)]);
            As[a_tile_x][a_tile_y + i] = a_tmp[0];
            As[a_tile_x + 1][a_tile_y + i] = a_tmp[1];
            As[a_tile_x + 2][a_tile_y + i] = a_tmp[2];
            As[a_tile_x + 3][a_tile_y + i] = a_tmp[3];
        }
        for (int i = 0; i < TK; i += b_tile_stride) {
            FETCH_FLOAT4(Bs[b_tile_y + i][b_tile_x]) = FETCH_FLOAT4(B[OFFSET(b_tile_y + i, b_tile_x, N)]);
        }
        __syncthreads();

        // 移动到下一个迭代的位置
        A += TK;
        B += TK * N;

        // partial dot product
        for (int k = 0; k < TK; k++) {
            // shared to register
            float Areg[RM], Breg[RN];
            for (int m = 0; m < RM; m += 4) FETCH_FLOAT4(Areg[m]) = FETCH_FLOAT4(As[k][y_st + m]);
            for (int n = 0; n < RN; n += 4) FETCH_FLOAT4(Breg[n]) = FETCH_FLOAT4(Bs[k][x_st + n]);

            for (int m = 0; m < RM; m++)
                for (int n = 0; n < RN; n++)
                    pval[m][n] += Areg[m] * Breg[n];
        }
        __syncthreads();
    }

    float c_tmp[4] = {0.f};
    for (int m = 0; m < RM; m++) {
        for (int n = 0; n < RN; n += 4) {
            FETCH_FLOAT4(c_tmp[0]) = FETCH_FLOAT4(C[OFFSET(y_st + m, x_st + n, N)]);
            c_tmp[0] = alpha * pval[m][n] + beta * c_tmp[0];
            c_tmp[1] = alpha * pval[m][n + 1] + beta * c_tmp[1];
            c_tmp[2] = alpha * pval[m][n + 2] + beta * c_tmp[2];
            c_tmp[3] = alpha * pval[m][n + 3] + beta * c_tmp[3];
            FETCH_FLOAT4(C[OFFSET(y_st + m, x_st + n, N)]) = FETCH_FLOAT4(c_tmp[0]);
        }
    }
}

void init(float *a, float *b, int M, int N, int K) {
    for (int i = 0; i < M * N; i++) {
        a[i] = 1.f;
        b[i] = 1.f;
    }
}

bool check_ans(float *truth, float *c, int M, int N) {
    for (int i = 0; i < M * N; i++)
        if (fabs(c[i] - truth[i]) > 0.5) {
            printf("truth: %f, output: %f\n", truth[i], c[i]);
            return false;
        }
    return true;
}

int main(int argc, char **argv) {
    // ncu: ./sgemm 0 1
    // compute-santi: ./sgemm 0 1 256 256 128
    // v100: ./sgemm
    int M = 8192;
    int N = 8192;
    int K = 4096;
    constexpr int TM = 128;
    constexpr int TN = 128;
    constexpr int TK = 8;
    constexpr int RM = 8;
    constexpr int RN = 8;
    int nWarmup = 2;
    int nIters = 50;
    assert(argc == 1 || argc == 3 || argc == 6);
    if (argc >= 3) {
        nWarmup = atoi(argv[1]);
        nIters = atoi(argv[2]);
    }
    if (argc >= 6) {
        M = atoi(argv[3]);
        N = atoi(argv[4]);
        K = atoi(argv[5]);
    }
    const float alpha = 1.f;
    const float beta = 0.f;
    float elapsed_my = 0.f;
    float elapsed_cublas = 0.f;
    size_t size = sizeof(float) * M * N;
    float *a_h = (float *) malloc(size);
    float *b_h = (float *) malloc(size);
    float *c_h = (float *) malloc(size);
    float *c_truth = (float *) malloc(size);
    init(a_h, b_h, M, N, K);

    float *a_d, *b_d, *c_d;
    cudaMalloc((void **) &a_d, size);
    cudaMalloc((void **) &b_d, size);
    cudaMalloc((void **) &c_d, size);
    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);
    dim3 gridDim(ceil(1.f * N / TN), ceil(1.f * M / TM), 1);
    dim3 blockDim((TN / RN) * (TM / RM), 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // my sgemm
    for (int i = 0; i < nIters + nWarmup; i++) {
        cudaEventRecord(start);
        sgemm_v3<TM, TN, TK, RM, RN><<<gridDim, blockDim>>>(a_d, b_d, c_d, M, N, K, alpha, beta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        if (i >= nWarmup) {
            float ms;
            cudaEventElapsedTime(&ms, start, stop);
            cout << i - nWarmup << ": " << ms << " ms\n";
            elapsed_my += ms;
        }
    }
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    #ifndef PROFILE
    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    for (int i = 0; i < nIters; i++) {
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, b_d, N, a_d, K, &beta, c_d, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        elapsed_cublas += ms;
    }
    cublasDestroy(handle);
    cudaMemcpy(c_truth, c_d, size, cudaMemcpyDeviceToHost);

    // check
    if (!check_ans(c_truth, c_h, M, N)) {
        cerr << "result is wrong!" << endl;
        return -1;
    }
    cout << "result is right" << endl;

    // output
    const int64_t flop = int64_t(M) * int64_t(N) * int64_t(K) * 2;
    double gflops_my = flop / ((elapsed_my / nIters) / 1000) / 1e9;
    double gflops_cublas = flop / ((elapsed_cublas / nIters) / 1000) / 1e9;
    cout << "mysgemm: " << gflops_my << "GFLOPS (" << flop << " flop, " << (elapsed_my / nIters) / 1000 << "s)\n";
    cout << "cublas: " << gflops_cublas << "GFLOPS (" << flop << " flop, " << (elapsed_cublas / nIters) / 1000 << "s)\n";
    cout << "% of cublas: " << gflops_my / gflops_cublas * 100 << "%" << endl;
    #endif

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);
    free(a_h);
    free(b_h);
    free(c_h);
    free(c_truth);

    return 0;
}
// Copyright (c) 2024-present, Junyeol Ryu, junyeol@aces.snu.ac.kr

#include <stdio.h>
#include <time.h>
#include <cassert>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

#define EPS 1e-3 
#define CHECK_CUDA(e) \
  if ((e) != cudaSuccess) { \
    printf("[%s:%d CudaError]: %s\n", \
        __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE);                          \
  } 
#define CHECK_CUBLAS(e)                                  \
  if ((e) != CUBLAS_STATUS_SUCCESS) {                    \
    printf("[%s:%d CublasError]\n", __FILE__, __LINE__); \
    exit(EXIT_FAILURE);                                  \
  }

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

#define WARMUP

half *A, *B;
float *C, *C_ans;
half *A_cuda, *B_cuda;
float *C_cuda, *C_cublas;

constexpr int M = 4096;
constexpr int K = 4096;
constexpr int N = 4096;

//////////////////////////////////////////// DO NOT MODIFY ////////////////////////////////////////////
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;

constexpr int BLOCK_TILE_M = 128; // d_bm
constexpr int BLOCK_TILE_N = 128; // d_bn
constexpr int BLOCK_TILE_K = 16;  // MODIFIABLE

/* SMEM load */
constexpr int A_N = MAX(MIN(BLOCK_TILE_K, BLOCK_N), 1); // A tile row is loaded by BLOCK_N threads, and can take multiple iterations
constexpr int A_M = (BLOCK_M * BLOCK_N) / A_N;    // Number of A tile rows loaded by a thread block in a single iteration
constexpr int B_N = MAX(MIN(BLOCK_TILE_N, BLOCK_N), 1); // B tile row is loaded by BLOCK_N threads, and can take multiple iterations
constexpr int B_M = (BLOCK_M * BLOCK_N) / B_N;    // Number of A tile rows loaded by a thread block in a single iteration

/* Warps for a block tile */
constexpr int WARP_TILE_M = 64; // d_wm
constexpr int WARP_TILE_N = 32; // d_wn

static_assert(BLOCK_TILE_M % WARP_TILE_M == 0);
static_assert(BLOCK_TILE_N % WARP_TILE_N == 0);

constexpr int NUM_WARPS_PER_BLOCK_TILE_M = BLOCK_TILE_M / WARP_TILE_M;  // NUM_WARPS_Y
constexpr int NUM_WARPS_PER_BLOCK_TILE_N = BLOCK_TILE_N / WARP_TILE_N;  // NUM_WARPS_X

constexpr int WMMA_TILE_M = 16;
constexpr int WMMA_TILE_K = 16;
constexpr int WMMA_TILE_N = 16;

static_assert(WARP_TILE_M % WMMA_TILE_M == 0);
static_assert(BLOCK_TILE_K % WMMA_TILE_K == 0);
static_assert(WARP_TILE_N % WMMA_TILE_N == 0);
////////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void mm16tc(half *A, half *B, float *C, const int M, const int K, const int N)
{
  if (blockIdx.x * BLOCK_TILE_N >= N || blockIdx.y * BLOCK_TILE_M >= M) return;

  const half ZERO = { 0.f };

  __shared__ half A_shared[BLOCK_TILE_M][BLOCK_TILE_K];  // TODO: Use transposed? // TODO: Add padding?
  __shared__ half B_shared[BLOCK_TILE_K][BLOCK_TILE_N];  // TODO: Add padding?

  constexpr int NUM_WMMA_TILES_PER_WARP_TILE_M = WARP_TILE_M / WMMA_TILE_M;
  constexpr int NUM_WMMA_TILES_PER_WARP_TILE_N = WARP_TILE_N / WMMA_TILE_N;
  static_assert(WARP_TILE_M % (NUM_WMMA_TILES_PER_WARP_TILE_M * WMMA_TILE_M) == 0);
  static_assert(WARP_TILE_N % (NUM_WMMA_TILES_PER_WARP_TILE_N * WMMA_TILE_N) == 0);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> a_frag[NUM_WMMA_TILES_PER_WARP_TILE_M];
  wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> b_frag[NUM_WMMA_TILES_PER_WARP_TILE_N];
  wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> c_frag[NUM_WMMA_TILES_PER_WARP_TILE_M][NUM_WMMA_TILES_PER_WARP_TILE_N];
  for (int ii = 0; ii < NUM_WMMA_TILES_PER_WARP_TILE_M; ++ii) {
    for (int jj = 0; jj < NUM_WMMA_TILES_PER_WARP_TILE_N; ++jj) {
      wmma::fill_fragment(c_frag[ii][jj], 0.f);
    }
  }

  /* Loading */
  const int ay = (blockDim.x * threadIdx.y + threadIdx.x) / A_N;
  const int ax = (blockDim.x * threadIdx.y + threadIdx.x) % A_N;
  const int by = (blockDim.x * threadIdx.y + threadIdx.x) / B_N;
  const int bx = (blockDim.x * threadIdx.y + threadIdx.x) % B_N;

  /* Computing */
  const int THREAD_IDX = threadIdx.y * blockDim.x + threadIdx.x;  // tid in block
  const int WARP_IDX = THREAD_IDX / 32; // warp id in block
  const int WARP_Y = WARP_IDX / NUM_WARPS_PER_BLOCK_TILE_N; // warp y id in computing block tile
  const int WARP_X = WARP_IDX % NUM_WARPS_PER_BLOCK_TILE_N; // warp x id in computing block tile

  for (int tk = 0; tk < K; tk += BLOCK_TILE_K) {
    // load A
    for (int ii = 0; ii < BLOCK_TILE_M; ii += A_M) {
      int li = ii + ay; // which row in shared A
      int Ai = BLOCK_TILE_M * blockIdx.y + li;  // which row in A
      for (int kk = 0; kk < BLOCK_TILE_K; kk += A_N) {  // load A row iteratively
        int lk = kk + ax; // which col in shared A
        int Ak = tk + lk;
        A_shared[li][lk] = (Ai < M && Ak < K) ? A[Ai * K + Ak] : ZERO;
      }
    }

    // load B
    for (int kk = 0; kk < BLOCK_TILE_K; kk += B_M) {
      int lk = kk + by; // which row in shared B
      int Bk = tk + lk;  // which row in B
      for (int jj = 0; jj < BLOCK_TILE_N; jj += B_N) {  // load B row iteratively
        int lj = jj + bx; // which col in shared B
        int Bj = blockIdx.x * BLOCK_TILE_N + lj;  // which col in B
        B_shared[lk][lj] = (Bk < K && Bj < N) ? B[Bk * N + Bj] : ZERO;
      }
    }
    // sync after load
    __syncthreads();

    // validate A
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //   int Ai = BLOCK_TILE_M * blockIdx.y;
    //   for (int ii = 0; ii < BLOCK_TILE_M; ++ii) {
    //     int Ak = tk;
    //     for (int kk = 0; kk < BLOCK_TILE_K; ++kk) {
    //       assert(A_shared[ii][kk] == ((Ai + ii < M && Ak + kk < K) ? A[(Ai + ii) * K + Ak + kk] : ZERO));
    //     }
    //   }
    // }

    // validate B
    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //   int Bk = tk;
    //   for (int kk = 0; kk < BLOCK_TILE_K; ++kk) {
    //     int Bj = BLOCK_TILE_N * blockIdx.x;
    //     for (int jj = 0; jj < BLOCK_TILE_N; ++jj) {
    //       assert(B_shared[kk][jj] == ((Bk + kk < K && Bj + jj < N) ? B[(Bk + kk) * N + Bj + jj] : ZERO));
    //     }
    //   }
    // }

    for (int k = 0; k < BLOCK_TILE_K; k += WMMA_TILE_K) {

      // wmma tile repeat row
      for (int wtr = 0; wtr < NUM_WMMA_TILES_PER_WARP_TILE_M; ++wtr) {
        // warp tile row + wtr stride
        const int wtri = WARP_Y * WARP_TILE_M \
                         + wtr * (WARP_TILE_M / NUM_WMMA_TILES_PER_WARP_TILE_M);
        wmma::load_matrix_sync(a_frag[wtr], &A_shared[wtri][k], BLOCK_TILE_K);  // TODO: ldm correct?
      }

      // wmma tile repeat col
      for (int wtc = 0; wtc < NUM_WMMA_TILES_PER_WARP_TILE_N; ++wtc) {
        // warp tile col + wtc stride
        const int wtcj = WARP_X * WARP_TILE_N \
                         + wtc * (WARP_TILE_N / NUM_WMMA_TILES_PER_WARP_TILE_N);
        wmma::load_matrix_sync(b_frag[wtc], &B_shared[k][wtcj], BLOCK_TILE_N);  // TODO: ldm correct?
      }

      // outer product
      for (int wtr = 0; wtr < NUM_WMMA_TILES_PER_WARP_TILE_M; ++wtr) {
        for (int wtc = 0; wtc < NUM_WMMA_TILES_PER_WARP_TILE_N; ++wtc) {
          wmma::mma_sync(c_frag[wtr][wtc], a_frag[wtr], b_frag[wtc], c_frag[wtr][wtc]);
        }
      }

    }
    // sync after use
    __syncthreads();
  }

  // copy back
  for (int wtr = 0; wtr < NUM_WMMA_TILES_PER_WARP_TILE_M; ++wtr) {
    for (int wtc = 0; wtc < NUM_WMMA_TILES_PER_WARP_TILE_N; ++wtc) {
      const int i = blockIdx.y * BLOCK_TILE_M + WARP_Y * WARP_TILE_M + wtr * (WARP_TILE_M / NUM_WMMA_TILES_PER_WARP_TILE_M);
      const int j = blockIdx.x * BLOCK_TILE_N + WARP_X * WARP_TILE_N + wtc * (WARP_TILE_N / NUM_WMMA_TILES_PER_WARP_TILE_N);
      wmma::store_matrix_sync(&C[i * N + j], c_frag[wtr][wtc], N, wmma::mem_row_major);
    }
  }
}

int main(int argc, char *argv[]) {
  A = (half *)malloc(M * K * sizeof(half));
  B = (half *)malloc(K * N * sizeof(half));
  C = (float *)malloc(M * N * sizeof(float));
  C_ans = (float *)malloc(M * N * sizeof(float));

  CHECK_CUDA(cudaMalloc(&A_cuda, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&B_cuda, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&C_cuda, M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_cublas, M * N * sizeof(float)));

  for (int i = 0; i < M * K; ++i) {
    A[i] = 2 * (rand() / (double)RAND_MAX);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = 2 * (rand() / (double)RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(A_cuda, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_cuda, B, K * N * sizeof(half), cudaMemcpyHostToDevice));

  printf("Running kernel\n");
#ifdef WARMUP
  {
    for (int ii = 0; ii < 10; ++ii) {
      dim3 blockDim{ BLOCK_N, BLOCK_M, 1 };
      dim3 gridDim{ (unsigned int)(N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (unsigned int)(M + BLOCK_TILE_M - 1) / BLOCK_TILE_M, 1 };
      mm16tc << < gridDim, blockDim >> > (A_cuda, B_cuda, C_cuda, M, K, N);
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  struct timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  {
    dim3 blockDim{ BLOCK_N, BLOCK_M, 1 };
    dim3 gridDim{ (unsigned int)(N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (unsigned int)(M + BLOCK_TILE_M - 1) / BLOCK_TILE_M, 1 };
    mm16tc << < gridDim, blockDim >> > (A_cuda, B_cuda, C_cuda, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  clock_gettime(CLOCK_MONOTONIC, &e);
  double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
  double bw = 2.0 * M * K * N / 1000000000. / elapsed;
  printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, bw);
  CHECK_CUDA(cudaMemcpy(C, C_cuda, M * N * sizeof(float), cudaMemcpyDeviceToHost));

  // cublas verify
  if (argc == 2) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

  // NOTE: not warming up cublas gives unexpected slow performance
#ifdef WARMUP
    {
      for (int ii = 0; ii < 10; ++ii) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_cuda, CUDA_R_16F, N, 
                                  A_cuda, CUDA_R_16F, K, &beta, C_cublas, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      }
      CHECK_CUDA(cudaDeviceSynchronize());
    }
#endif

    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    {
      printf("Running cublas\n");
      float alpha = 1.f;
      float beta = 0.f;
      CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_cuda, CUDA_R_16F, N, 
                                A_cuda, CUDA_R_16F, K, &beta, C_cublas, CUDA_R_32F, N, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
    double cublas_bw = 2.0 * M * K * N / 1000000000. / elapsed;
    printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, cublas_bw);

    printf("Kernel / cuBlas = %lf / %lf = %lf %%\n", bw, cublas_bw, bw / cublas_bw * 100);
    CHECK_CUDA(cudaMemcpy(C_ans, C_cublas, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs((C[i * N + j] - C_ans[i * N + j]) / C[i * N + j]) >= EPS) {
          printf("Validation Failed! C[%d, %d]: %f %f\n", i, j, C[i * N + j], C_ans[i * N + j]);
          exit(1);
        }
      }
    }
    printf("Verification Success!\n");
  }
}
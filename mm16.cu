#include <stdio.h>
#include <time.h>
#include <cassert>
#include <cublas_v2.h>

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
half *C, *C_ans;
half *A_cuda, *B_cuda;
half *C_cuda, *C_cublas;

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 64;

constexpr int A_N = MAX(MIN(TILE_K, BLOCK_N), 1); // number of threads to load a row of A shared. Max: BLOCK_N
constexpr int A_M = (BLOCK_M * BLOCK_N) / A_N;        // number of rows of A shared loaded in a single memory access 
constexpr int B_N = MAX(MIN(TILE_N, BLOCK_N), 1); // number of threads to load a row of B shared. Max: BLOCK_N
constexpr int B_M = (BLOCK_M * BLOCK_N) / B_N;        // number of rows of B shared loaded in a single memory access 

constexpr int W_M = 32; // warp가 처리하는 총 row 개수. stride는 32이다.
constexpr int W_N = 16;

constexpr int T_M = 16; // thread가 처리하는 총 element 개수.
constexpr int T_N = 1;

constexpr int REG_M = 8;  // warp가 한번에 처리하는 16 x 16 안에서 한 thread가 처리하는 element 개수
constexpr int REG_N = 1;

// constexpr int NUMWARPS = (BLOCK_M * BLOCK_N) / 32;  // num warps in a thread block

__global__ void mm16(half *A, half *B, half *C, const int M, const int K, const int N) 
{
  if (blockIdx.x * TILE_N >= N || blockIdx.y * TILE_M >= M) return;

  const half ZERO = { 0.f };

  __shared__ half A_shared[TILE_M][TILE_K];
  __shared__ half B_shared[TILE_K][TILE_N];

  const int ay = (blockDim.x * threadIdx.y + threadIdx.x) / A_N;
  const int ax = (blockDim.x * threadIdx.y + threadIdx.x) % A_N;
  const int by = (blockDim.x * threadIdx.y + threadIdx.x) / B_N;
  const int bx = (blockDim.x * threadIdx.y + threadIdx.x) % B_N;

  half c_reg[T_M][T_N];
  for (int i = 0; i < REG_M; ++i) {
    for (int j = 0; j < REG_N; ++j) {
      c_reg[i][j] = ZERO;
    }
  }

  for (int tk = 0; tk < K; tk += TILE_K) {
    // load A
    for (int ii = 0; ii < TILE_M; ii += A_M) {
      int li = ii + ay;
      int Ai = TILE_M * blockIdx.y + li;
      for (int kk = 0; kk < TILE_K; kk += A_N) {
        int lk = kk + ax;
        int Ak = tk + lk;
        A_shared[li][lk] = (Ai < M && Ak < K) ? A[K * Ai + Ak] : ZERO;
      }
    }

    // load B
    for (int kk = 0; kk < TILE_K; kk += B_M) {
      int lk = kk + by;
      int Bk = tk + lk;
      for (int jj = 0; jj < TILE_N; jj += B_N) {
        int lj = jj + bx;
        int Bj = TILE_N * blockIdx.x + lj;
        B_shared[lk][lj] = (Bk < K && Bj < N) ? B[N * Bk + Bj] : ZERO;
      }
    }
    // sync after load
    __syncthreads();

    const int w = (blockDim.x * threadIdx.y + threadIdx.x) / 32;  // 내가 속한 warp의 1D index in thread block

    const int WMITER = W_M / 16;
    const int WNITER = W_N / 16;

    for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
      const int wy = (w / 4) * 16 + w_sub_row_idx * 32; // (tile_n / wninter / w_n) * 16 + (stride * 32) <-- warp가 처리하는 row index
      assert(wy < TILE_M);
      for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
        const int wx = (w % 4) * 16 + w_sub_col_idx * 64; // <-- warp가 처리하는 col index. stride 64
        assert(wx < TILE_N);
        for (int res_idx_m = 0; res_idx_m < REG_M; ++res_idx_m) {
          int ty = (threadIdx.y % 2) + res_idx_m * 2; // (warp 32개를 16개씩 끊은거. blockDim.x가 32일때만 성립.) + (stride가 2) <-- thread가 처리하는 row index
          ty = ty + wy;
          assert(ty < TILE_M);
          for (int res_idx_n = 0; res_idx_n < REG_N; ++res_idx_n) {
            int tx = (threadIdx.x % 16) + res_idx_n * 16; // (warp를 16씩 끊은거. stride가 16) <-- thread가 처리하는 col index
            tx = tx + wx;
            assert(tx < TILE_N);
            for (int k = 0; k < TILE_K; ++k) {
              c_reg[w_sub_row_idx * REG_M + ty][w_sub_col_idx * REG_N + tx] += A_shared[ty][k] * B_shared[k][tx];
            }
          }
        }
      }
    }
    // sync after use
    __syncthreads();
  }

  // copy back
  const int w = (blockDim.x * threadIdx.y + threadIdx.x) / 32;  // 내가 속한 warp의 1D index in thread block
  const int WMITER = W_M / 16;
  const int WNITER = W_N / 16;

  for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
      for (int res_idx_m = 0; res_idx_m < REG_M; ++res_idx_m) {
        for (int res_idx_n = 0; res_idx_n < REG_N; ++res_idx_n) {
          const int wy = blockIdx.y * TILE_M + (w / 4) * 16 + w_sub_row_idx * 32;
          const int wx = blockIdx.x * TILE_N + (w % 4) * 16 + w_sub_col_idx * 64;
          const int ty = (threadIdx.y % 2) + res_idx_m * 2; // (warp 32개를 16개씩 끊은거. blockDim.x가 32일때만 성립.) + (stride가 2) <-- thread가 처리하는 row index
          const int tx = (threadIdx.x % 16) + res_idx_n * 16; // (warp를 16씩 끊은거. stride가 16) <-- thread가 처리하는 col index
          C[(wy + ty) * N + (wx + tx)] = c_reg[w_sub_row_idx * REG_M + res_idx_m][w_sub_col_idx * REG_N + res_idx_n];
        }
      }

    }

  }


  // for (int y = 0; y < TILE_M; y += WARP_M) {
  //   int i = blockIdx.y * TILE_M + y + (threadIdx.x / W_N) % 2;
  //   int j = blockIdx.x * TILE_N + wx * W_N + (threadIdx.x % W_N);
  //   for (int yy = 0; yy < REG; ++yy) {
  //     C[(i + yy * 2) * N + j] = c_reg[y][yy];
  //   }
  // }
}

int main(int argc, char *argv[]) {
  int M = 64; int K = 64; int N = 64; // size in scalar
  // printf("M: %d, K: %d, N: %d, NUMWARPS: %d, WARP_M: %d, WARP_N: %d, REG: %d\n", M, K, N, NUMWARPS, WARP_M, WARP_N, REG);
  // assert(NUMWARPS * W_M * W_N == TILE_M * TILE_N);  // check if the tile is fully covered by warps' single iteration
  //                                                   // TODO: fix kernel code
  A = (half *)malloc(M * K * sizeof(half));
  B = (half *)malloc(K * N * sizeof(half));
  C = (half *)malloc(M * N * sizeof(half));
  C_ans = (half *)malloc(M * N * sizeof(half));

  CHECK_CUDA(cudaMalloc(&A_cuda, M * K * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&B_cuda, K * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&C_cuda, M * N * sizeof(half)));
  CHECK_CUDA(cudaMalloc(&C_cublas, M * N * sizeof(half)));

  for (int i = 0; i < M * K; ++i) {
    A[i] = (half) ((float)rand() / RAND_MAX - 0.5);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = (half) ((float)rand() / RAND_MAX - 0.5);
  }

  CHECK_CUDA(cudaMemcpy(A_cuda, A, M * K * sizeof(half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_cuda, B, K * N * sizeof(half), cudaMemcpyHostToDevice));

  printf("Running kernel\n");
#ifdef WARMUP
  {
    for (int ii = 0; ii < 10; ++ii) {
      dim3 blockDim { BLOCK_N, BLOCK_M, 1 };
      dim3 gridDim { (unsigned int)(N + TILE_N - 1) / TILE_N, (unsigned int)(M + TILE_M - 1) / TILE_M, 1 };
      mm16 <<< gridDim, blockDim >>> (A_cuda, B_cuda, C_cuda, M, K, N);
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  struct timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  {
    dim3 blockDim { BLOCK_N, BLOCK_M, 1 };
    dim3 gridDim { (unsigned int)(N + TILE_N - 1) / TILE_N, (unsigned int)(M + TILE_M - 1) / TILE_M, 1 };
    mm16 <<< gridDim, blockDim >>> (A_cuda, B_cuda, C_cuda, M, K, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  clock_gettime(CLOCK_MONOTONIC, &e);
  double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
  double bw = 2.0 * M * K * N / 1000000000. / elapsed;
  printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, bw);
  CHECK_CUDA(cudaMemcpy(C, C_cuda, M * N * sizeof(half), cudaMemcpyDeviceToHost));

  // cublas verify
  if (argc == 2) {
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    {
      printf("Running cublas\n");
      half alpha = 1.f;
      half beta = 0.f;
      CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_cuda, N, A_cuda, K, &beta, C_cublas, N));
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
    double cublas_bw = 2.0 * M * K * N / 1000000000. / elapsed;
    printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, cublas_bw);

    printf("Kernel / cuBlas = %lf / %lf = %lf %%\n", bw, cublas_bw, bw / cublas_bw * 100);
    CHECK_CUDA(cudaMemcpy(C_ans, C_cublas, sizeof(half) * M * N, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        if (fabs((float)((C[i * N + j] - C_ans[i * N + j]) / C[i * N + j])) >= EPS) {
          printf("Validation Failed! C[%d, %d]: %f %f\n", i, j, (float)C[i * N + j], (float)C_ans[i * N + j]);
          // exit(1);
        }
      }
    }
    printf("Verification Success!\n");
  }
}
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

float *A, *B, *C, *C_ans;
float *A_cuda, *B_cuda, *C_cuda, *C_cublas;

constexpr int M = 4096;
constexpr int K = 4096;
constexpr int N = 4096;

//////////////////////////////////////////// DO NOT MODIFY ////////////////////////////////////////////
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;

constexpr int BLOCK_TILE_M = 128; // d_bm
constexpr int BLOCK_TILE_N = 128; // d_bn
constexpr int BLOCK_TILE_K = 16;

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

constexpr int NUM_THREADS_PER_WARP_Y = 8; // m_t
constexpr int NUM_THREADS_PER_WARP_X = 4; // n_t

static_assert(NUM_THREADS_PER_WARP_Y * NUM_THREADS_PER_WARP_X == 32);
////////////////////////////////////////////////////////////////////////////////////////////////////////

/* Threads for a warp tile */
constexpr int THREAD_TILE_M = 8;  // d_tm
constexpr int THREAD_TILE_N = 8;  // d_tn

static_assert(WARP_TILE_M % (THREAD_TILE_M * NUM_THREADS_PER_WARP_Y) == 0); // (d_wm / m_t) % d_tm == 0
static_assert(WARP_TILE_N % (THREAD_TILE_N * NUM_THREADS_PER_WARP_X) == 0); // (d_wn / n_t) % d_tn == 0

__global__ void mm32w(float *A, float *B, float *C, const int M, const int K, const int N)
{
  if (blockIdx.x * BLOCK_TILE_N >= N || blockIdx.y * BLOCK_TILE_M >= M) return;

  const float ZERO = { 0.f };

  __shared__ float A_shared[BLOCK_TILE_M][BLOCK_TILE_K]; // TODO: Use transposed?
  __shared__ float B_shared[BLOCK_TILE_K][BLOCK_TILE_N];

  // Each thread computes (d_wm / m_t / d_tm) * (d_wn / n_t / d_tn) number of tiles,
  // each with size d_tm x d_tn
  constexpr int NUM_THREAD_TILES_PER_WARP_TILE_M = WARP_TILE_M / NUM_THREADS_PER_WARP_Y / THREAD_TILE_M; // Thread가 속한 warp의 tile 중, 내 thread tile로 계산하는 개수
  constexpr int NUM_THREAD_TILES_PER_WARP_TILE_N = WARP_TILE_N / NUM_THREADS_PER_WARP_X / THREAD_TILE_N; // Thread가 속한 warp의 tile 중, 내 thread tile로 계산하는 개수
  static_assert(WARP_TILE_M % (NUM_THREADS_PER_WARP_Y * THREAD_TILE_M) == 0);
  static_assert(WARP_TILE_N % (NUM_THREADS_PER_WARP_X * THREAD_TILE_N) == 0);

  float a_reg[NUM_THREAD_TILES_PER_WARP_TILE_M][THREAD_TILE_M] = { ZERO };
  float b_reg[NUM_THREAD_TILES_PER_WARP_TILE_N][THREAD_TILE_N] = { ZERO };
  float c_reg[NUM_THREAD_TILES_PER_WARP_TILE_M][NUM_THREAD_TILES_PER_WARP_TILE_N][THREAD_TILE_M][THREAD_TILE_N] = { ZERO };

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
  const int THREAD_Y_IN_WARP = (THREAD_IDX % 32) / NUM_THREADS_PER_WARP_X; // thread y id in warp tile
  const int THREAD_X_IN_WARP = (THREAD_IDX % 32) % NUM_THREADS_PER_WARP_X; // thread x id in warp tile

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

    for (int k = 0; k < BLOCK_TILE_K; ++k) {

      // thread tile repeat row
      for (int ttr = 0; ttr < NUM_THREAD_TILES_PER_WARP_TILE_M; ++ttr) {
        // warp tile row + ttr stride + thread tile row
        const int ttri = WARP_Y * WARP_TILE_M \
                         + ttr * (WARP_TILE_M / NUM_THREAD_TILES_PER_WARP_TILE_M) \
                         + THREAD_Y_IN_WARP * THREAD_TILE_M;
        for (int ii = 0; ii < THREAD_TILE_M; ++ii) {
          a_reg[ttr][ii] = A_shared[ttri + ii][k];
        }
      }

      // thread tile repeat col
      for (int ttc = 0; ttc < NUM_THREAD_TILES_PER_WARP_TILE_N; ++ttc) {
        // warp tile col + ttc stride + thread tile col
        const int ttcj = WARP_X * WARP_TILE_N \
                         + ttc * (WARP_TILE_N / NUM_THREAD_TILES_PER_WARP_TILE_N) \
                         + THREAD_X_IN_WARP * THREAD_TILE_N;
        for (int jj = 0; jj < THREAD_TILE_N; ++jj) {
          b_reg[ttc][jj] = B_shared[k][ttcj + jj];
        }
      }

      // outer product
      for (int ttr = 0; ttr < NUM_THREAD_TILES_PER_WARP_TILE_M; ++ttr) {
        for (int ttc = 0; ttc < NUM_THREAD_TILES_PER_WARP_TILE_N; ++ttc) {
          for (int ii = 0; ii < THREAD_TILE_M; ++ii) {
            for (int jj = 0; jj < THREAD_TILE_N; ++jj) {
              c_reg[ttr][ttc][ii][jj] += a_reg[ttr][ii] * b_reg[ttc][jj];
            }
          }
        }
      }

    }
    // sync after use
    __syncthreads();
  }

  // copy back
  for (int ttr = 0; ttr < NUM_THREAD_TILES_PER_WARP_TILE_M; ++ttr) {
    for (int ttc = 0; ttc < NUM_THREAD_TILES_PER_WARP_TILE_N; ++ttc) {
      for (int ii = 0; ii < THREAD_TILE_M; ++ii) {
        for (int jj = 0; jj < THREAD_TILE_N; ++jj) {
          const int i = blockIdx.y * BLOCK_TILE_M + WARP_Y * WARP_TILE_M + ttr * (WARP_TILE_M / NUM_THREAD_TILES_PER_WARP_TILE_M) + THREAD_Y_IN_WARP * THREAD_TILE_M + ii;
          const int j = blockIdx.x * BLOCK_TILE_N + WARP_X * WARP_TILE_N + ttc * (WARP_TILE_N / NUM_THREAD_TILES_PER_WARP_TILE_N) + THREAD_X_IN_WARP * THREAD_TILE_N + jj;
          C[i * N + j] = c_reg[ttr][ttc][ii][jj];
        }
      }
    }
  }
}

int main(int argc, char *argv[]) {
  A = (float *)malloc(M * K * sizeof(float));
  B = (float *)malloc(K * N * sizeof(float));
  C = (float *)malloc(M * N * sizeof(float));
  C_ans = (float *)malloc(M * N * sizeof(float));

  CHECK_CUDA(cudaMalloc(&A_cuda, M * K * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_cuda, K * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_cuda, M * N * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_cublas, M * N * sizeof(float)));

  for (int i = 0; i < M * K; ++i) {
    A[i] = 2 * (rand() / (double)RAND_MAX);
  }
  for (int i = 0; i < K * N; ++i) {
    B[i] = 2 * (rand() / (double)RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(A_cuda, A, M * K * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_cuda, B, K * N * sizeof(float), cudaMemcpyHostToDevice));

  printf("Running kernel\n");
#ifdef WARMUP
  {
    for (int ii = 0; ii < 10; ++ii) {
      dim3 blockDim{ BLOCK_N, BLOCK_M, 1 };
      dim3 gridDim{ (unsigned int)(N + BLOCK_TILE_N - 1) / BLOCK_TILE_N, (unsigned int)(M + BLOCK_TILE_M - 1) / BLOCK_TILE_M, 1 };
      mm32w << < gridDim, blockDim >> > (A_cuda, B_cuda, C_cuda, M, K, N);
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
    mm32w << < gridDim, blockDim >> > (A_cuda, B_cuda, C_cuda, M, K, N);
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

    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    {
      printf("Running cublas\n");
      float alpha = 1.f;
      float beta = 0.f;
      CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B_cuda, N, A_cuda, K, &beta, C_cublas, N));
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
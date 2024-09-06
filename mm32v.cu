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

#define V (4)

constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;

/* SMEM load */
constexpr int A_N = MAX(MIN(TILE_K / V, BLOCK_N), 1); // number of threads to load a row of A shared. Max: BLOCK_N
constexpr int A_M = (BLOCK_M * BLOCK_N) / A_N;        // number of rows of A shared loaded in a single memory access 
constexpr int B_N = MAX(MIN(TILE_N / V, BLOCK_N), 1); // number of threads to load a row of B shared. Max: BLOCK_N
constexpr int B_M = (BLOCK_M * BLOCK_N) / B_N;        // number of rows of B shared loaded in a single memory access 

/* a thread computes REG_M * (REG_N vector) C elements */
constexpr int REG_M = (TILE_M + BLOCK_M - 1) / BLOCK_M; // number of C row elements per thread
constexpr int REG_N = ((TILE_N / V) + BLOCK_N - 1) / BLOCK_N; // number of C col **vector** elements per thread
                                                              // a thread computes REG_M * REG_N C elements

__global__ void mm32(float4 *A, float4 *B, float4 *C, const int M, const int K, const int N)
{
  if (blockIdx.x * TILE_N >= N || blockIdx.y * TILE_M >= M) return;
  
  const int K_V = K / V;
  const int N_V = N / V;

  const float4 ZERO = { 0.f };

  __shared__ float4 A_shared[TILE_M][TILE_K / V];
  __shared__ float4 B_shared[TILE_K][TILE_N / V];
  
  const int ay = (blockDim.x * threadIdx.y + threadIdx.x) / A_N;
  const int ax = (blockDim.x * threadIdx.y + threadIdx.x) % A_N;
  const int by = (blockDim.x * threadIdx.y + threadIdx.x) / B_N;
  const int bx = (blockDim.x * threadIdx.y + threadIdx.x) % B_N;

  float4 c_reg[REG_M][REG_N];

  // init c_reg
  for (int i = 0; i < REG_M; ++i) {
    for (int j = 0; j < REG_N; ++j) {
      c_reg[i][j] = ZERO;
    }
  }

  for (int tk = 0; tk < K; tk += TILE_K) {
    // load A
    for (int ii = 0; ii < TILE_M; ii += A_M) {
      int li = ii + ay; // which row in shared A
      int Ai = TILE_M * blockIdx.y + li;  // which row in A
      for (int kk = 0; kk < TILE_K / V; kk += A_N) {  // load A row iteratively
        int lk = kk + ax; // which col in shared A
        int Ak = (tk / V) + lk;
        A_shared[li][lk] = (Ai < M && Ak < K_V) ? A[Ai * K_V + Ak] : ZERO;
      }
    }

    // load B
    for (int kk = 0; kk < TILE_K; kk += B_M) {
      int lk = kk + by; // which row in shared B
      int Bk = tk + lk;  // which row in B. My prev. wrong answer was: TILE_K * blockIdx.y + lk
      for (int jj = 0; jj < TILE_N / V; jj += B_N) {  // load B row iteratively
        int lj = jj + bx;
        int Bj = blockIdx.x * (TILE_N / V) + lj;
        B_shared[lk][lj] = (Bk < K && Bj < N_V) ? B[Bk * N_V + Bj] : ZERO;
      }
    }
    // sync after load
    __syncthreads();

    for (int y = 0; y < REG_M; ++y) {
      int i = threadIdx.y + y * BLOCK_M;  // last operand NOT * REG_M
      for (int x = 0; x < REG_N; ++x) {
        int j = threadIdx.x + x * BLOCK_N;  // last operand NOT * REG_N
        for (int k = 0; k < TILE_K / V; ++k) {
          c_reg[y][x].x += A_shared[i][k].x * B_shared[V * k + 0][j].x;
          c_reg[y][x].y += A_shared[i][k].x * B_shared[V * k + 0][j].y;
          c_reg[y][x].z += A_shared[i][k].x * B_shared[V * k + 0][j].z;
          c_reg[y][x].w += A_shared[i][k].x * B_shared[V * k + 0][j].w;

          c_reg[y][x].x += A_shared[i][k].y * B_shared[V * k + 1][j].x;
          c_reg[y][x].y += A_shared[i][k].y * B_shared[V * k + 1][j].y;
          c_reg[y][x].z += A_shared[i][k].y * B_shared[V * k + 1][j].z;
          c_reg[y][x].w += A_shared[i][k].y * B_shared[V * k + 1][j].w;

          c_reg[y][x].x += A_shared[i][k].z * B_shared[V * k + 2][j].x;
          c_reg[y][x].y += A_shared[i][k].z * B_shared[V * k + 2][j].y;
          c_reg[y][x].z += A_shared[i][k].z * B_shared[V * k + 2][j].z;
          c_reg[y][x].w += A_shared[i][k].z * B_shared[V * k + 2][j].w;

          c_reg[y][x].x += A_shared[i][k].w * B_shared[V * k + 3][j].x;
          c_reg[y][x].y += A_shared[i][k].w * B_shared[V * k + 3][j].y;
          c_reg[y][x].z += A_shared[i][k].w * B_shared[V * k + 3][j].z;
          c_reg[y][x].w += A_shared[i][k].w * B_shared[V * k + 3][j].w;
        }
      }
    }
    // sync after use
    __syncthreads();
  }

  // copy back
  for (int y = 0; y < REG_M; ++y) {
    int i = blockIdx.y * TILE_M + threadIdx.y + y * BLOCK_M;  // last operand NOT * REG_M
    for (int x = 0; x < REG_N; ++x) {
      int j = blockIdx.x * (TILE_N / V) + threadIdx.x + x * BLOCK_N;  // last operand NOT * REG_N
      C[i * N_V + j] = c_reg[y][x];
    }
  }
}

int main(int argc, char *argv[]) {
  assert(K % V == 0); assert(N % V == 0);

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
      dim3 blockDim { BLOCK_N, BLOCK_M, 1 };
      dim3 gridDim { (unsigned int)(N + TILE_N - 1) / TILE_N, (unsigned int)(M + TILE_M - 1) / TILE_M, 1 };
      mm32 <<< gridDim, blockDim >>> ((float4*)A_cuda, (float4*)B_cuda, (float4*)C_cuda, M, K, N);
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
    mm32 <<< gridDim, blockDim >>> ((float4*)A_cuda, (float4*)B_cuda, (float4*)C_cuda, M, K, N);
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
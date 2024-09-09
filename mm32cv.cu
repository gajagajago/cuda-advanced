// Copyright (c) 2024-present, Junyeol Ryu, junyeol@aces.snu.ac.kr

#include <stdio.h> 
#include <time.h> 

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

#define VA (4)
#define VB (4)
#define VC (4)

#if (VA == 1)
typedef float1 type_a;
#elif (VA == 2)
typedef float2 type_a;
#else
typedef float4 type_a;
#endif

#if (VB == 1)
typedef float1 type_b;
#elif (VB == 2)
typedef float2 type_b;
#else
typedef float4 type_b;
#endif

#if (VC == 1)
typedef float1 type_c;
#elif (VC == 2)
typedef float2 type_c;
#else
typedef float4 type_c;
#endif

#define MAX(a, b) (((a) < (b)) ? (b) : (a))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

constexpr int M_ = 4096;
constexpr int K_ = 4096;
constexpr int N_ = 4096;

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 32;
constexpr int BLOCK_M = 16;
constexpr int BLOCK_N = 16;
constexpr int REG_M = ((TILE_M + BLOCK_M - 1) / BLOCK_M); // 한 스레드가 output tile에서 세로 방향에서 처리하는 개수
constexpr int REG_N = (((TILE_N / VC) + BLOCK_N - 1) / BLOCK_N);  // 한 스레드가 output tile에서 가로 방향에서 처리하는 vector 개수
constexpr int A_N = MAX(MIN(TILE_K / VA, BLOCK_N), 1);  // A tile 한 row load 당 thread 개수
constexpr int A_M = (BLOCK_M * BLOCK_N) / A_N;  // 한 threadblock이 한번에 load하는 A tile의 row 개수
constexpr int B_N = MAX(MIN(TILE_K / VB, BLOCK_N), 1); // B tile 한 row load 당 thread의 개수
constexpr int B_M = (BLOCK_M * BLOCK_N) / B_N; // B tile 한 row load 당 thread의 개수
constexpr int TILE_K_VA = TILE_K / VA;
constexpr int TILE_K_VB = TILE_K / VB;
constexpr int TILE_N_VC = TILE_N / VC;

#define WARMUP 


/* GEMM
 * @param [in1] A: [M, K]
 * @param [in2] B: [N, K]          
 * @param [out] C: [M, N]
 */
__global__ void mm32cv(const type_a* A, const type_b* B, type_c* C, const int M, const int N, const int K) {
  const int _K_VA = K / VA;
  const int _K_VB = K / VB;
  const int _N_VC = N / VC;

  if (blockIdx.x * TILE_N >= N || blockIdx.y * TILE_M >= M) return;

  __shared__ type_a Ashared[TILE_M][TILE_K_VA];
  __shared__ type_c Bshared[TILE_K][TILE_N_VC + 1]; // !!!

  const type_a ZEROA = { 0.f };
  const type_b ZEROB = { 0.f };
  const type_c ZEROC = { 0.f };

  type_c creg[REG_M][REG_N];
  for (int y = 0; y < REG_M; ++y) {
    for (int x = 0; x < REG_N; ++x) {
      creg[y][x] = ZEROC;
    }
  }

  const int ax = (threadIdx.y * blockDim.x + threadIdx.x) % A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표 (여러번하면 움직임)
  const int ay = (threadIdx.y * blockDim.x + threadIdx.x) / A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표
  const int bx = (threadIdx.y * blockDim.x + threadIdx.x) % B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표
  const int by = (threadIdx.y * blockDim.x + threadIdx.x) / B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표


  for (int tk = 0; tk < K; tk += TILE_K) {

#pragma unroll
    for (int ii = 0; ii < TILE_M / A_M; ++ii) { // A tile의 가로-wise 여러번 걸림
      int li = A_M * ii + ay;
      int Ai = blockIdx.y * TILE_M + li;
      // printf("ii: %d tid(%d,%d) li: %d, Ai: %d\n", ii, threadIdx.x, threadIdx.y, li, Ai);
#pragma unroll
      for (int kk = 0; kk < TILE_K_VA / A_N; ++kk) {  // A tile의 세로-wise 여러번 걸림 
        int lk = A_N * kk + ax;
        int Ak = (tk / VA) + lk;
        // printf("\tkk: %d tid(%d,%d) lk: %d, Ak: %d\n", kk, threadIdx.x, threadIdx.y, lk, Ak);

        type_a val = (Ai < M && Ak < _K_VA) ? A[Ai * _K_VA + Ak] : ZEROA;
        Ashared[li][lk] = val;
      }
    }

#pragma unroll
    for (int jj = 0; jj < TILE_N / B_M; ++jj) { // B tile의 가로-wise 여러번 걸림
      int lj = B_M * jj + by;
      // int Bj = blockIdx.y * TILE_N + lj;  // 중요! blockIdx.y 맞음?
      int Bj = blockIdx.x * TILE_N + lj;  // 중요! blockIdx.y 했다가 아니어서 blockIdx.x로 바꿈 -> 맞음. TODO: Why?


      // printf("jj: %d tid(%d,%d) lj: %d, Bj: %d\n", jj, threadIdx.x, threadIdx.y, lj, Bj);
#pragma unroll
      for (int kk = 0; kk < TILE_K_VB / B_N; ++kk) { // B tile의 세로-wise 여러번 걸림 // TILE_K 가 아니라 TILE_K_VB!!!
        int lk = B_N * kk + bx;
        int Bk = (tk / VB) + lk;  // 중요!
        // printf("\tkk: %d tid(%d,%d) lk: %d, Bk: %d\n", kk, threadIdx.x, threadIdx.y, lk, Bk);

        type_b val = (Bj < N && Bk < _K_VB) ? B[Bj * _K_VB + Bk] : ZEROB;
        // Bshared[lj][lk] = val;

        if (lj % VC == 0) {
          Bshared[VB * lk + 0][lj / VC].x = val.x;
          Bshared[VB * lk + 1][lj / VC].x = val.y;
          Bshared[VB * lk + 2][lj / VC].x = val.z;
          Bshared[VB * lk + 3][lj / VC].x = val.w;
        } else if (lj % VC == 1) {
          Bshared[VB * lk + 0][lj / VC].y = val.x;
          Bshared[VB * lk + 1][lj / VC].y = val.y;
          Bshared[VB * lk + 2][lj / VC].y = val.z;
          Bshared[VB * lk + 3][lj / VC].y = val.w;
        } else if (lj % VC == 2) {
          Bshared[VB * lk + 0][lj / VC].z = val.x;
          Bshared[VB * lk + 1][lj / VC].z = val.y;
          Bshared[VB * lk + 2][lj / VC].z = val.z;
          Bshared[VB * lk + 3][lj / VC].z = val.w;
        } else {
          Bshared[VB * lk + 0][lj / VC].w = val.x;
          Bshared[VB * lk + 1][lj / VC].w = val.y;
          Bshared[VB * lk + 2][lj / VC].w = val.z;
          Bshared[VB * lk + 3][lj / VC].w = val.w;
        }
      }
    }

    __syncthreads();
    // printf("shmem init!\n");

#pragma unroll
    for (int y = 0; y < REG_M; ++y) {
      int si = threadIdx.y + y * BLOCK_M; // 내가 계산할 A tile의 y좌표
#pragma unroll
      for (int x = 0; x < REG_N; ++x) {
        int sj = threadIdx.x + x * BLOCK_N; // 내가 계산할 B tile의 x좌표
#pragma unroll
        for (int lk = 0; lk < TILE_K / VA; ++lk) {
          // A tile의 한 원소는 벡터 (4개) -> 한 원소는 B 타일의 x좌표에 해당하는 곳에서 (가로) 위부터 가로로 4개씩 곱해서 creg로 더함 

          creg[y][x].x += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].x;
          creg[y][x].y += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].y;
          creg[y][x].z += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].z;
          creg[y][x].w += Ashared[si][lk].x * Bshared[VC * lk + 0][sj].w;

          creg[y][x].x += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].x;
          creg[y][x].y += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].y;
          creg[y][x].z += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].z;
          creg[y][x].w += Ashared[si][lk].y * Bshared[VC * lk + 1][sj].w;

          creg[y][x].x += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].x;
          creg[y][x].y += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].y;
          creg[y][x].z += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].z;
          creg[y][x].w += Ashared[si][lk].z * Bshared[VC * lk + 2][sj].w;

          creg[y][x].x += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].x;
          creg[y][x].y += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].y;
          creg[y][x].z += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].z;
          creg[y][x].w += Ashared[si][lk].w * Bshared[VC * lk + 3][sj].w;

        }
      }
    }
    __syncthreads();
  }


// 내가 처리해야 할 creg 값 16개 처리
#pragma unroll
  for (int y = 0; y < REG_M; ++y) {
#pragma unroll
    for (int x = 0; x < REG_N; ++x) {
      int i = blockIdx.y * TILE_M + threadIdx.y + y * BLOCK_M;
      int j = blockIdx.x * (TILE_N / VC) + threadIdx.x + x * BLOCK_N;
      C[i * _N_VC + j] = creg[y][x];
    }
  }
}




/* GEMM
 * @param [in1] A: [M, K]
 * @param [in2] B: [N, K]          
 * @param [out] C: [M, N]
 */
__global__ void mm32cv_bank_conflict(const type_a* A, const type_b* B, type_c* C, const int M, const int N, const int K) {
  const int _K_VA = K / VA;
  const int _K_VB = K / VB;
  const int _N_VC = N / VC;

  if (blockIdx.x * TILE_N >= N || blockIdx.y * TILE_M >= M) return;

  __shared__ type_a Ashared[TILE_M][TILE_K_VA];
  __shared__ type_b Bshared[TILE_N][TILE_K_VB+1]; // 1 padding to avoid bank conflicts in compute loop
  // printf("Ashared[%d][%d] Bshared[%d][%d]\n", TILE_M, TILE_K_VA, TILE_N, TILE_K_VB);
  const type_a ZEROA = { 0.f };
  const type_b ZEROB = { 0.f };
  const type_c ZEROC = { 0.f };

  type_c creg[REG_M][REG_N];
  for (int y = 0; y < REG_M; ++y) {
    for (int x = 0; x < REG_N; ++x) {
      creg[y][x] = ZEROC;
    }
  }

  const int ax = (threadIdx.y * blockDim.x + threadIdx.x) % A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표 (여러번하면 움직임)
  const int ay = (threadIdx.y * blockDim.x + threadIdx.x) / A_N;  // A tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표
  const int bx = (threadIdx.y * blockDim.x + threadIdx.x) % B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 x좌표
  const int by = (threadIdx.y * blockDim.x + threadIdx.x) / B_N; // B tile load 시 (여러번 걸릴 수 있음) thread 당 y좌표


  for (int tk = 0; tk < K; tk += TILE_K) {

#pragma unroll
    for (int ii = 0; ii < TILE_M / A_M; ++ii) { // A tile의 가로-wise 여러번 걸림
      int li = A_M * ii + ay;
      int Ai = blockIdx.y * TILE_M + li;
      // printf("ii: %d tid(%d,%d) li: %d, Ai: %d\n", ii, threadIdx.x, threadIdx.y, li, Ai);
#pragma unroll
      for (int kk = 0; kk < TILE_K_VA / A_N; ++kk) {  // A tile의 세로-wise 여러번 걸림 
        int lk = A_N * kk + ax;
        int Ak = (tk / VA) + lk;
        // printf("\tkk: %d tid(%d,%d) lk: %d, Ak: %d\n", kk, threadIdx.x, threadIdx.y, lk, Ak);

        type_a val = (Ai < M && Ak < _K_VA) ? A[Ai * _K_VA + Ak] : ZEROA;
        Ashared[li][lk] = val;
      }
    }

#pragma unroll
    for (int jj = 0; jj < TILE_N / B_M; ++jj) { // B tile의 가로-wise 여러번 걸림
      int lj = B_M * jj + by;
      // int Bj = blockIdx.y * TILE_N + lj;  // 중요! blockIdx.y 맞음?
      int Bj = blockIdx.x * TILE_N + lj;  // 중요! blockIdx.y 했다가 아니어서 blockIdx.x로 바꿈 -> 맞음. TODO: Why?


      // printf("jj: %d tid(%d,%d) lj: %d, Bj: %d\n", jj, threadIdx.x, threadIdx.y, lj, Bj);
#pragma unroll
      for (int kk = 0; kk < TILE_K_VB / B_N; ++kk) { // B tile의 세로-wise 여러번 걸림 // TILE_K 가 아니라 TILE_K_VB!!!
        int lk = B_N * kk + bx;
        int Bk = (tk / VB) + lk;  // 중요!
        // printf("\tkk: %d tid(%d,%d) lk: %d, Bk: %d\n", kk, threadIdx.x, threadIdx.y, lk, Bk);

        type_b val = (Bj < N && Bk < _K_VB) ? B[Bj * _K_VB + Bk] : ZEROB;
        Bshared[lj][lk] = val;
      }
    }

    __syncthreads();

#pragma unroll
    for (int y = 0; y < REG_M; ++y) {
      int si = threadIdx.y + y * BLOCK_M; // 내가 계산할 A tile y좌표
#pragma unroll
      for (int x = 0; x < REG_N; ++x) {
        int sj = threadIdx.x + x * BLOCK_N; // 내가 계산할 B tile y좌표
#pragma unroll
        for (int lk = 0; lk < TILE_K / VA; ++lk) { // 일단 VA == VB (== VC) 가정.

          // Bshared access pattern is definitely a problem...
          // w/o Bshared padding 3TFLOPS
          // w/ padding 6TFLOPS
          // ideal 13TFLOPS

          // ncu

          // VB times due to 4 elements in a column for Bshared should be accessed to compute x y z w of a C vector
          creg[y][x].x += Ashared[si][lk].x * Bshared[VB * sj][lk].x;
          creg[y][x].x += Ashared[si][lk].y * Bshared[VB * sj][lk].y;
          creg[y][x].x += Ashared[si][lk].z * Bshared[VB * sj][lk].z;
          creg[y][x].x += Ashared[si][lk].w * Bshared[VB * sj][lk].w;

          creg[y][x].y += Ashared[si][lk].x * Bshared[VB * sj+1][lk].x;
          creg[y][x].y += Ashared[si][lk].y * Bshared[VB * sj+1][lk].y;
          creg[y][x].y += Ashared[si][lk].z * Bshared[VB * sj+1][lk].z;
          creg[y][x].y += Ashared[si][lk].w * Bshared[VB * sj+1][lk].w;

          creg[y][x].z += Ashared[si][lk].x * Bshared[VB * sj+2][lk].x;
          creg[y][x].z += Ashared[si][lk].y * Bshared[VB * sj+2][lk].y;
          creg[y][x].z += Ashared[si][lk].z * Bshared[VB * sj+2][lk].z;
          creg[y][x].z += Ashared[si][lk].w * Bshared[VB * sj+2][lk].w;

          creg[y][x].w += Ashared[si][lk].x * Bshared[VB * sj+3][lk].x;
          creg[y][x].w += Ashared[si][lk].y * Bshared[VB * sj+3][lk].y;
          creg[y][x].w += Ashared[si][lk].z * Bshared[VB * sj+3][lk].z;
          creg[y][x].w += Ashared[si][lk].w * Bshared[VB * sj+3][lk].w;
        }
      }
    }
    __syncthreads();
  }


#pragma unroll
  for (int y = 0; y < REG_M; ++y) {
#pragma unroll
    for (int x = 0; x < REG_N; ++x) {
      int i = blockIdx.y * TILE_M + threadIdx.y + y * BLOCK_M;
      int j = blockIdx.x * (TILE_N / VC) + threadIdx.x + x * BLOCK_N;
      C[i * _N_VC + j] = creg[y][x];
    }
  }
}

int main(int argc, char* argv[])
{
  float* a = (float*)malloc(sizeof(float) * M_ * K_);
  float* b = (float*)malloc(sizeof(float) * N_ * K_);
  float* c = (float*)malloc(sizeof(float) * M_ * N_);
  float* c_ans = (float*)malloc(sizeof(float) * M_ * N_);

  float* d_a;
  float* d_b;
  float* d_c;
  float* d_c_ans;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(float) * M_ * K_));
  CHECK_CUDA(cudaMalloc(&d_b, sizeof(float) * N_ * K_));
  CHECK_CUDA(cudaMalloc(&d_c, sizeof(float) * M_ * N_));
  CHECK_CUDA(cudaMalloc(&d_c_ans, sizeof(float) * M_ * N_));


  for (int i = 0; i < M_ * K_; ++i) {
    a[i] = 2 * (rand() / (double)RAND_MAX);
  }
  for (int i = 0; i < N_ * K_; ++i) {
    b[i] = 2 * (rand() / (double)RAND_MAX);
  }

  CHECK_CUDA(cudaMemcpy(d_a, a, sizeof(float) * M_ * K_, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_b, b, sizeof(float) * N_ * K_, cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));

  printf("Running kernel\n");
#ifdef WARMUP
  {
    for (int ii = 0; ii < 10; ++ii) {
      dim3 threadDims = { BLOCK_N, BLOCK_M, 1 };
      dim3 blockDims = {
        (unsigned int)((N_ + TILE_N - 1) / (TILE_N)),
        (unsigned int)((M_ + TILE_M - 1) / (TILE_M)),
        1,
      };
      mm32cv << <blockDims, threadDims >> > (
        (type_a*)d_a,
        (type_b*)d_b,
        (type_c*)d_c,
        M_,
        N_,
        K_
        );
      CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  struct timespec s, e;
  clock_gettime(CLOCK_MONOTONIC, &s);
  {
    dim3 threadDims = { BLOCK_N, BLOCK_M, 1 };
    dim3 blockDims = {
      (unsigned int)((N_ + TILE_N - 1) / (TILE_N)),
      (unsigned int)((M_ + TILE_M - 1) / (TILE_M)),
      1,
    };

    mm32cv << <blockDims, threadDims >> > (
      (type_a*)d_a,
      (type_b*)d_b,
      (type_c*)d_c,
      M_,
      N_,
      K_
      );
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaDeviceSynchronize());
  }

  clock_gettime(CLOCK_MONOTONIC, &e);
  double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
  double bw = 2.0 * M_ * K_ * N_ / 1000000000. / elapsed;
  printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, bw);
  CHECK_CUDA(cudaMemcpy(c, d_c, sizeof(float) * M_ * N_, cudaMemcpyDeviceToHost));

  if (argc == 2) {
    struct timespec s, e;
    clock_gettime(CLOCK_MONOTONIC, &s);
    {
      printf("Running cublas\n");
      float alpha = 1.f;
      float beta = 0.f;
      CHECK_CUBLAS(cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N_, M_, K_,
        &alpha,
        d_b, K_,
        d_a, K_,
        &beta,
        d_c_ans, N_)
      );
      CHECK_CUDA(cudaDeviceSynchronize());
    }
    clock_gettime(CLOCK_MONOTONIC, &e);
    double elapsed = (e.tv_sec - s.tv_sec) + ((double)e.tv_nsec - s.tv_nsec) / 1000000000.;
    double cublas_bw = 2.0 * M_ * K_ * N_ / 1000000000. / elapsed;
    printf("elapsed time: %lfs, bandwidth: %lf GB/s\n", elapsed, cublas_bw);

    printf("Kernel / cuBlas = %lf / %lf = %lf %%\n",
      bw, cublas_bw, bw / cublas_bw * 100);
    CHECK_CUDA(cudaMemcpy(c_ans, d_c_ans, sizeof(float) * M_ * N_, cudaMemcpyDeviceToHost));

    for (int i = 0; i < M_; ++i) {
      for (int j = 0; j < N_; ++j) {
        if (fabs((c[i * N_ + j] - c_ans[i * N_ + j]) / c[i * N_ + j]) >= EPS) {
          printf("Validation Failed! C[%d, %d]: %f %f\n", i, j, c[i * N_ + j], c_ans[i * N_ + j]);
          exit(1);
        }
      }
    }
    printf("Verification Success!\n");
  }
  return 0;
}

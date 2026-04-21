#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_seidel_2d kernel_seidel_2d_dut
#include "seidel-2d.c"
#undef kernel_seidel_2d

static void kernel_seidel_2d_ref(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int t, i, j;

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
    for (i = 1; i<= _PB_N - 2; i++)
      for (j = 1; j <= _PB_N - 2; j++)
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/SCALAR_VAL(9.0);
#pragma endscop

}

static int tb_equal_data(DATA_TYPE lhs, DATA_TYPE rhs) {
    double diff = fabs((double)lhs - (double)rhs);
    double scale = fabs((double)lhs);
    double rhs_scale = fabs((double)rhs);
    if (rhs_scale > scale) {
        scale = rhs_scale;
    }
    return diff <= 1e-6 * (1.0 + scale);
}

static int run_case(int case_id) {
    int tsteps = TSTEPS;
    int n = N;

    DATA_TYPE A_ref[N][N];
    DATA_TYPE A_dut[N][N];

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        A_ref[i0][i1] = (DATA_TYPE)((((7) * (i0 + 1) + (11) * (i1 + 1) + 5 * (case_id + 1)) % 17) + 1);
      }
    }
    memcpy(A_dut, A_ref, sizeof(A_ref));

    kernel_seidel_2d_dut(tsteps, n, A_dut);
    kernel_seidel_2d_ref(tsteps, n, A_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (!tb_equal_data(A_ref[i0][i1], A_dut[i0][i1])) {
          return 0;
        }
      }
    }
    return 1;
}

int main(void) {
    if (!run_case(0)) {
        printf("fail\n");
        return 1;
    }
    if (!run_case(1)) {
        printf("fail\n");
        return 1;
    }
    printf("pass\n");
    return 0;
}

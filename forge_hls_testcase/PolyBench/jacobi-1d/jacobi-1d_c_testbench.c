#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_jacobi_1d kernel_jacobi_1d_dut
#include "jacobi-1d.c"
#undef kernel_jacobi_1d

static void kernel_jacobi_1d_ref(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
      for (i = 1; i < _PB_N - 1; i++)
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
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

    DATA_TYPE A_ref[N];
    DATA_TYPE A_dut[N];
    DATA_TYPE B_ref[N];
    DATA_TYPE B_dut[N];

    for (int i0 = 0; i0 < N; ++i0) {
      A_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 5 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      B_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 6 * (case_id + 1)) % 17) + 1);
    }
    memcpy(A_dut, A_ref, sizeof(A_ref));
    memcpy(B_dut, B_ref, sizeof(B_ref));

    kernel_jacobi_1d_dut(tsteps, n, A_dut, B_dut);
    kernel_jacobi_1d_ref(tsteps, n, A_ref, B_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(A_ref[i0], A_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(B_ref[i0], B_dut[i0])) {
        return 0;
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

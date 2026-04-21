#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_gesummv kernel_gesummv_dut
#include "gesummv.c"
#undef kernel_gesummv

static void kernel_gesummv_ref(int n,
		    DATA_TYPE alpha,
		    DATA_TYPE beta,
		    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		    DATA_TYPE POLYBENCH_2D(B,N,N,n,n),
		    DATA_TYPE POLYBENCH_1D(tmp,N,n),
		    DATA_TYPE POLYBENCH_1D(x,N,n),
		    DATA_TYPE POLYBENCH_1D(y,N,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    {
      tmp[i] = SCALAR_VAL(0.0);
      y[i] = SCALAR_VAL(0.0);
      for (j = 0; j < _PB_N; j++)
	{
	  tmp[i] = A[i][j] * x[j] + tmp[i];
	  y[i] = B[i][j] * x[j] + y[i];
	}
      y[i] = alpha * tmp[i] + beta * y[i];
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
    int n = N;
    DATA_TYPE alpha = (DATA_TYPE)(1.25 + 0.25 * case_id);
    DATA_TYPE beta = (DATA_TYPE)(0.75 + 0.50 * case_id);

    DATA_TYPE A_ref[N][N];
    DATA_TYPE A_dut[N][N];
    DATA_TYPE B_ref[N][N];
    DATA_TYPE B_dut[N][N];
    DATA_TYPE tmp_ref[N];
    DATA_TYPE tmp_dut[N];
    DATA_TYPE x_ref[N];
    DATA_TYPE x_dut[N];
    DATA_TYPE y_ref[N];
    DATA_TYPE y_dut[N];

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        A_ref[i0][i1] = (DATA_TYPE)((((7) * (i0 + 1) + (11) * (i1 + 1) + 5 * (case_id + 1)) % 17) + 1);
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        B_ref[i0][i1] = (DATA_TYPE)((((7) * (i0 + 1) + (11) * (i1 + 1) + 6 * (case_id + 1)) % 17) + 1);
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      tmp_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 7 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      x_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 8 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      y_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 9 * (case_id + 1)) % 17) + 1);
    }
    memcpy(A_dut, A_ref, sizeof(A_ref));
    memcpy(B_dut, B_ref, sizeof(B_ref));
    memcpy(tmp_dut, tmp_ref, sizeof(tmp_ref));
    memcpy(x_dut, x_ref, sizeof(x_ref));
    memcpy(y_dut, y_ref, sizeof(y_ref));

    kernel_gesummv_dut(n, alpha, beta, A_dut, B_dut, tmp_dut, x_dut, y_dut);
    kernel_gesummv_ref(n, alpha, beta, A_ref, B_ref, tmp_ref, x_ref, y_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (!tb_equal_data(A_ref[i0][i1], A_dut[i0][i1])) {
          return 0;
        }
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (!tb_equal_data(B_ref[i0][i1], B_dut[i0][i1])) {
          return 0;
        }
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(tmp_ref[i0], tmp_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(x_ref[i0], x_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(y_ref[i0], y_dut[i0])) {
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

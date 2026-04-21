#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_gemver kernel_gemver_dut
#include "gemver.c"
#undef kernel_gemver

static void kernel_gemver_ref(int n,
		   DATA_TYPE alpha,
		   DATA_TYPE beta,
		   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		   DATA_TYPE POLYBENCH_1D(u1,N,n),
		   DATA_TYPE POLYBENCH_1D(v1,N,n),
		   DATA_TYPE POLYBENCH_1D(u2,N,n),
		   DATA_TYPE POLYBENCH_1D(v2,N,n),
		   DATA_TYPE POLYBENCH_1D(w,N,n),
		   DATA_TYPE POLYBENCH_1D(x,N,n),
		   DATA_TYPE POLYBENCH_1D(y,N,n),
		   DATA_TYPE POLYBENCH_1D(z,N,n))
{
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      w[i] = w[i] +  alpha * A[i][j] * x[j];

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
    DATA_TYPE u1_ref[N];
    DATA_TYPE u1_dut[N];
    DATA_TYPE v1_ref[N];
    DATA_TYPE v1_dut[N];
    DATA_TYPE u2_ref[N];
    DATA_TYPE u2_dut[N];
    DATA_TYPE v2_ref[N];
    DATA_TYPE v2_dut[N];
    DATA_TYPE w_ref[N];
    DATA_TYPE w_dut[N];
    DATA_TYPE x_ref[N];
    DATA_TYPE x_dut[N];
    DATA_TYPE y_ref[N];
    DATA_TYPE y_dut[N];
    DATA_TYPE z_ref[N];
    DATA_TYPE z_dut[N];

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        A_ref[i0][i1] = (DATA_TYPE)((((7) * (i0 + 1) + (11) * (i1 + 1) + 5 * (case_id + 1)) % 17) + 1);
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      u1_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 6 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      v1_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 7 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      u2_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 8 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      v2_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 9 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      w_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 10 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      x_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 11 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      y_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 12 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      z_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 13 * (case_id + 1)) % 17) + 1);
    }
    memcpy(A_dut, A_ref, sizeof(A_ref));
    memcpy(u1_dut, u1_ref, sizeof(u1_ref));
    memcpy(v1_dut, v1_ref, sizeof(v1_ref));
    memcpy(u2_dut, u2_ref, sizeof(u2_ref));
    memcpy(v2_dut, v2_ref, sizeof(v2_ref));
    memcpy(w_dut, w_ref, sizeof(w_ref));
    memcpy(x_dut, x_ref, sizeof(x_ref));
    memcpy(y_dut, y_ref, sizeof(y_ref));
    memcpy(z_dut, z_ref, sizeof(z_ref));

    kernel_gemver_dut(n, alpha, beta, A_dut, u1_dut, v1_dut, u2_dut, v2_dut, w_dut, x_dut, y_dut, z_dut);
    kernel_gemver_ref(n, alpha, beta, A_ref, u1_ref, v1_ref, u2_ref, v2_ref, w_ref, x_ref, y_ref, z_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (!tb_equal_data(A_ref[i0][i1], A_dut[i0][i1])) {
          return 0;
        }
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(u1_ref[i0], u1_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(v1_ref[i0], v1_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(u2_ref[i0], u2_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(v2_ref[i0], v2_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(w_ref[i0], w_dut[i0])) {
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
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(z_ref[i0], z_dut[i0])) {
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

#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_mvt kernel_mvt_dut
#include "mvt.c"
#undef kernel_mvt

static void kernel_mvt_ref(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
    for (j = 0; j < _PB_N; j++)
      x2[i] = x2[i] + A[j][i] * y_2[j];
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

    DATA_TYPE x1_ref[N];
    DATA_TYPE x1_dut[N];
    DATA_TYPE x2_ref[N];
    DATA_TYPE x2_dut[N];
    DATA_TYPE y_1_ref[N];
    DATA_TYPE y_1_dut[N];
    DATA_TYPE y_2_ref[N];
    DATA_TYPE y_2_dut[N];
    DATA_TYPE A_ref[N][N];
    DATA_TYPE A_dut[N][N];

    for (int i0 = 0; i0 < N; ++i0) {
      x1_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 5 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      x2_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 6 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      y_1_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 7 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      y_2_ref[i0] = (DATA_TYPE)((((7) * (i0 + 1) + 8 * (case_id + 1)) % 17) + 1);
    }
    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        A_ref[i0][i1] = (DATA_TYPE)((((7) * (i0 + 1) + (11) * (i1 + 1) + 9 * (case_id + 1)) % 17) + 1);
      }
    }
    memcpy(x1_dut, x1_ref, sizeof(x1_ref));
    memcpy(x2_dut, x2_ref, sizeof(x2_ref));
    memcpy(y_1_dut, y_1_ref, sizeof(y_1_ref));
    memcpy(y_2_dut, y_2_ref, sizeof(y_2_ref));
    memcpy(A_dut, A_ref, sizeof(A_ref));

    kernel_mvt_dut(n, x1_dut, x2_dut, y_1_dut, y_2_dut, A_dut);
    kernel_mvt_ref(n, x1_ref, x2_ref, y_1_ref, y_2_ref, A_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(x1_ref[i0], x1_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(x2_ref[i0], x2_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(y_1_ref[i0], y_1_dut[i0])) {
        return 0;
      }
    }
    for (int i0 = 0; i0 < N; ++i0) {
      if (!tb_equal_data(y_2_ref[i0], y_2_dut[i0])) {
        return 0;
      }
    }
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

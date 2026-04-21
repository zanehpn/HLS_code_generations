#include <math.h>
#include <stdio.h>
#include <string.h>

#define MINI_DATASET
#define kernel_floyd_warshall kernel_floyd_warshall_dut
#include "floyd-warshall.c"
#undef kernel_floyd_warshall

static void kernel_floyd_warshall_ref(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
      for(i = 0; i < _PB_N; i++)
	for (j = 0; j < _PB_N; j++)
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
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

    DATA_TYPE path_ref[N][N];
    DATA_TYPE path_dut[N][N];

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (i0 == i1) {
          path_ref[i0][i1] = (DATA_TYPE)0;
        } else {
          path_ref[i0][i1] = (DATA_TYPE)(((i0 + 1) * (i1 + 2) + case_id) % 9 + 1);
        }
      }
    }
    memcpy(path_dut, path_ref, sizeof(path_ref));

    kernel_floyd_warshall_dut(n, path_dut);
    kernel_floyd_warshall_ref(n, path_ref);

    for (int i0 = 0; i0 < N; ++i0) {
      for (int i1 = 0; i1 < N; ++i1) {
        if (!tb_equal_data(path_ref[i0][i1], path_dut[i0][i1])) {
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

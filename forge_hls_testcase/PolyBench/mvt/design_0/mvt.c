/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* mvt.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "mvt.h"



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_mvt(int n,
		DATA_TYPE POLYBENCH_1D(x1,N,n),
		DATA_TYPE POLYBENCH_1D(x2,N,n),
		DATA_TYPE POLYBENCH_1D(y_1,N,n),
		DATA_TYPE POLYBENCH_1D(y_2,N,n),
		DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=y_2 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=y_1 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=x2 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=x1 type=cyclic dim=1 factor=2
  int i, j;

#pragma scop
  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
    for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
      x1[i] = x1[i] + A[i][j] * y_1[j];
  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
    for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
      x2[i] = x2[i] + A[j][i] * y_2[j];
#pragma endscop

}

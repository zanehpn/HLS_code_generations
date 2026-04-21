/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "gemver.h"

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(int n,
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
#pragma HLS ARRAY_PARTITION variable=z type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=y type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=x type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=w type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=v2 type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=u2 type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=v1 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=u1 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=2
  int i, j;

#pragma scop

  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
    for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
    for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
      x[i] = x[i] + beta * A[j][i] * y[j];

  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
    x[i] = x[i] + z[i];

  for (i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
    for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
      w[i] = w[i] +  alpha * A[i][j] * x[j];

#pragma endscop
}

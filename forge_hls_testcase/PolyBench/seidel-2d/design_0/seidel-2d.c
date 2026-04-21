/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* seidel-2d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "seidel-2d.h"



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_seidel_2d(int tsteps,
		      int n,
		      DATA_TYPE POLYBENCH_2D(A,N,N,n,n))
{
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=2 factor=4
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
  int t, i, j;

#pragma scop
  for (t = 0; t <= _PB_TSTEPS - 1; t++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
    for (i = 1; i<= _PB_N - 2; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
      for (j = 1; j <= _PB_N - 2; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
	A[i][j] = (A[i-1][j-1] + A[i-1][j] + A[i-1][j+1]
		   + A[i][j-1] + A[i][j] + A[i][j+1]
		   + A[i+1][j-1] + A[i+1][j] + A[i+1][j+1])/SCALAR_VAL(9.0);
#pragma endscop

}


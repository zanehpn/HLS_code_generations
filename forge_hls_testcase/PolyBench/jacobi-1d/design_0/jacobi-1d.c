/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* jacobi-1d.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "jacobi-1d.h"



/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_jacobi_1d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_1D(A,N,n),
			    DATA_TYPE POLYBENCH_1D(B,N,n))
{
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=2
  int t, i;

#pragma scop
  for (t = 0; t < _PB_TSTEPS; t++)
    {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
      for (i = 1; i < _PB_N - 1; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
	B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
      for (i = 1; i < _PB_N - 1; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
	A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
#pragma endscop

}


/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* floyd-warshall.c: this file is part of PolyBench/C */

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include "floyd-warshall.h"


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_floyd_warshall(int n,
			   DATA_TYPE POLYBENCH_2D(path,N,N,n,n))
{
#pragma HLS ARRAY_PARTITION variable=path type=cyclic dim=2 factor=2
#pragma HLS ARRAY_PARTITION variable=path type=cyclic dim=1 factor=4
  int i, j, k;

#pragma scop
  for (k = 0; k < _PB_N; k++)
    {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
      for(i = 0; i < _PB_N; i++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
	for (j = 0; j < _PB_N; j++)
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
	  path[i][j] = path[i][j] < path[i][k] + path[k][j] ?
	    path[i][j] : path[i][k] + path[k][j];
    }
#pragma endscop

}


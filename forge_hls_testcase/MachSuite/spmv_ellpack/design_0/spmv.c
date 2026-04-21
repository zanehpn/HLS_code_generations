/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"

void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
#pragma HLS ARRAY_PARTITION variable=out type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=vec type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=cols type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=nzval type=cyclic dim=1 factor=1
    int i, j;
    TYPE Si;

    ellpack_1 : for (i=0; i<N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        TYPE sum = out[i];
        ellpack_2 : for (j=0; j<L; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
                Si = nzval[j + i*L] * vec[cols[j + i*L]];
                sum += Si;
        }
        out[i] = sum;
    }
}

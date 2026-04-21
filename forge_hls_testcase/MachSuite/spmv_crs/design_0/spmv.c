/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/

#include "spmv.h"

void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    int i, j;
#pragma HLS ARRAY_PARTITION variable=out type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=vec type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=rowDelimiters type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=cols type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=val type=cyclic dim=1 factor=1
    TYPE sum, Si;

    spmv_1 : for(i = 0; i < N; i++){
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        sum = 0; Si = 0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i+1];
        spmv_2 : for (j = tmp_begin; j < tmp_end; j++){
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
        #pragma HLS loop_tripcount min=1 max=NNZ avg=5
            Si = val[j] * vec[cols[j]];
            sum = sum + Si;
        }
        out[i] = sum;
    }
}



#include <ap_int.h>

#define SIZE 1024

void sr_latch(ap_uint<1> S[SIZE], ap_uint<1> R[SIZE], ap_uint<1> Q[SIZE], ap_uint<1> Qn[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=Qn type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=Q type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=R type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=S type=cyclic dim=1 factor=1
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        if (S[i] == 1 && R[i] == 0) {
            Q[i] = 1;
            Qn[i] = 0;
        } else if (S[i] == 0 && R[i] == 1) {
            Q[i] = 0;
            Qn[i] = 1;
        } else if (S[i] == 0 && R[i] == 0) {
            if (i == 0) {
                Q[i] = 0;
                Qn[i] = 1;
            } else {
                Q[i] = Q[i-1];
                Qn[i] = Qn[i-1];
            }
        } else {
            Q[i] = 0;
            Qn[i] = 0;
        }
    }
}

// Top function name: sr_latch

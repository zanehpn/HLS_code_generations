#include <ap_int.h>

#define SIZE 1024

void master_slave_d_latch(ap_uint<1> D[SIZE], ap_uint<1> CLK[SIZE], ap_uint<1> Q[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=Q type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=CLK type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=D type=cyclic dim=1 factor=16
    ap_uint<1> master_latch = 0;
    ap_uint<1> slave_latch = 0;

    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        if (CLK[i] == 1) {
            master_latch = D[i];
        } else {
            slave_latch = master_latch;
        }
        Q[i] = slave_latch;
    }
}

// Top function name: master_slave_d_latch

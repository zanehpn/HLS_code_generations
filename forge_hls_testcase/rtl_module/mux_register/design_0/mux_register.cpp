#include <ap_int.h>

#define SIZE 1024

void mux_register(ap_uint<1> sel, ap_uint<32> in1[SIZE], ap_uint<32> in2[SIZE], ap_uint<32> out[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=out type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=in2 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=in1 type=cyclic dim=1 factor=4
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        if (sel == 0) {
            out[i] = in1[i];
        } else {
            out[i] = in2[i];
        }
    }
}

// Top function name: mux_register

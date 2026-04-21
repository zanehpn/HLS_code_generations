#include <ap_int.h>

#define ARRAY_SIZE 1024

void mux_4_to_1(ap_uint<2> sel[ARRAY_SIZE], ap_uint<8> in0[ARRAY_SIZE], ap_uint<8> in1[ARRAY_SIZE], ap_uint<8> in2[ARRAY_SIZE], ap_uint<8> in3[ARRAY_SIZE], ap_uint<8> out[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=out type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=in3 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=in2 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=in1 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=in0 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=sel type=cyclic dim=1 factor=8
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        switch (sel[i]) {
            case 0:
                out[i] = in0[i];
                break;
            case 1:
                out[i] = in1[i];
                break;
            case 2:
                out[i] = in2[i];
                break;
            case 3:
                out[i] = in3[i];
                break;
            default:
                out[i] = 0; // Default case, should not occur
                break;
        }
    }
}

// Top function name: mux_4_to_1

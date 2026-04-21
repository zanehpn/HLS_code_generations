#include <ap_int.h>

#define ARRAY_SIZE 1024

void xor_gate(ap_uint<1> input1[ARRAY_SIZE], ap_uint<1> input2[ARRAY_SIZE], ap_uint<1> output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=16
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        output[i] = input1[i] ^ input2[i];
    }
}

// Top function name: xor_gate

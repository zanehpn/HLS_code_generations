#include <ap_int.h>

#define SIZE 1024

void pll_2x(ap_int<16> input[SIZE], ap_int<16> output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        output[i] = input[i] * 2;
    }
}

// Top function name: pll_2x

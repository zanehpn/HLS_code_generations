#include <ap_int.h>

#define SIZE 1024

void single_supply_comparator(ap_int<16> input1[SIZE], ap_int<16> input2[SIZE], ap_int<1> output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=8
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output[i] = (input1[i] > input2[i]) ? 1 : 0;
    }
}

// Top function name: single_supply_comparator

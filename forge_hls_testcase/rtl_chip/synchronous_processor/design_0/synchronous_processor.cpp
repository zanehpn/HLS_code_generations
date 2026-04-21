#include <ap_int.h>

#define SIZE 1024

void synchronous_processor(ap_int<16> input1[SIZE], ap_int<16> input2[SIZE], ap_int<16> output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=16
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        output[i] = input1[i] + input2[i];
    }
}

// Top function name: synchronous_processor

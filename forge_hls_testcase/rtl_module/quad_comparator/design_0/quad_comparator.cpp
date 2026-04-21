#include <ap_int.h>

#define ARRAY_SIZE 1024

void quad_comparator(ap_int<16> input1[ARRAY_SIZE], ap_int<16> input2[ARRAY_SIZE], ap_int<16> input3[ARRAY_SIZE], ap_int<16> input4[ARRAY_SIZE], ap_int<16> output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input4 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input3 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=16
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        ap_int<16> max_val = input1[i];
        if (input2[i] > max_val) max_val = input2[i];
        if (input3[i] > max_val) max_val = input3[i];
        if (input4[i] > max_val) max_val = input4[i];
        output[i] = max_val;
    }
}

// Top function name: quad_comparator

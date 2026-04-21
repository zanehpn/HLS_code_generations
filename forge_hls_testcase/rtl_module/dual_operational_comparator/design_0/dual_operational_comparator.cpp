#include <ap_int.h>

#define ARRAY_SIZE 1024

void dual_operational_comparator(ap_int<16> input1[ARRAY_SIZE], ap_int<16> input2[ARRAY_SIZE], ap_int<16> output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=8
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        if (input1[i] > input2[i]) {
            output[i] = input1[i] - input2[i];
        } else {
            output[i] = input2[i] - input1[i];
        }
    }
}

// Top function name: dual_operational_comparator

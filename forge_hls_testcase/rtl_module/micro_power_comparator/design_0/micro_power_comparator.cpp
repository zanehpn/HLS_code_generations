#include <ap_int.h>

#define ARRAY_SIZE 1024

void micro_power_comparator(ap_int<16> input1[ARRAY_SIZE], ap_int<16> input2[ARRAY_SIZE], ap_int<1> output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=2
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        if (input1[i] > input2[i]) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

// Top function name: micro_power_comparator

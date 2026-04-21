#include <ap_int.h>

#define SIZE 1024

void ittyBittyComparator(ap_int<32> input1[SIZE], ap_int<32> input2[SIZE], ap_int<1> output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=16
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        if (input1[i] > input2[i]) {
            output[i] = 1;
        } else {
            output[i] = 0;
        }
    }
}

// Top function name: ittyBittyComparator

#include <ap_int.h>

#define SIZE 1024

void data_forwarding(ap_int<32> input[SIZE], ap_int<32> output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=4
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        output[i] = input[i];
    }
}

// Top function name: data_forwarding

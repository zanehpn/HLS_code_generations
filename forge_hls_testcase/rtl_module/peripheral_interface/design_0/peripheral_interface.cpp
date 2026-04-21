#include <ap_int.h>

#define ARRAY_SIZE 1024

void peripheral_interface(ap_int<32> input_data[ARRAY_SIZE], ap_int<32> output_data[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output_data type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input_data type=cyclic dim=1 factor=1
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        output_data[i] = input_data[i] + 1;
    }
}

// Top function name: peripheral_interface

#include <ap_int.h>

#define SIZE 1024

void power_divider(ap_int<32> input_array[SIZE], ap_int<32> output_array[SIZE], ap_int<32> power, ap_int<32> divisor) {
#pragma HLS ARRAY_PARTITION variable=output_array type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input_array type=cyclic dim=1 factor=8
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        ap_int<32> temp = input_array[i];
        for (int j = 1; j < power; j++) {
            temp *= input_array[i];
        }
        output_array[i] = temp / divisor;
    }
}

// Top function name: power_divider

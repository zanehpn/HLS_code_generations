#include <ap_int.h>

#define ARRAY_SIZE 1024

void single_comparator(ap_int<16> input_array[ARRAY_SIZE], ap_int<16> &max_value) {
#pragma HLS ARRAY_PARTITION variable=input_array type=cyclic dim=1 factor=16
    max_value = input_array[0];
    for (int i = 1; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        if (input_array[i] > max_value) {
            max_value = input_array[i];
        }
    }
}

// Top function name: single_comparator

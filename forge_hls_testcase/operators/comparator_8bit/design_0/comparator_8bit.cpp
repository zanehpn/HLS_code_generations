#include <ap_int.h>

#define ARRAY_SIZE 1024

void comparator_8bit(ap_uint<8> input1[ARRAY_SIZE], ap_uint<8> input2[ARRAY_SIZE], bool result[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=result type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=2
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        result[i] = (input1[i] > input2[i]);
    }
}

// Top function name: comparator_8bit

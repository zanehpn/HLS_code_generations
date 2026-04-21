#include <stdint.h>

#define SIZE 1024

void data_bus_32bit(uint32_t input[SIZE], uint32_t output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=4
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        output[i] = input[i] + 1;
    }
}

// Top function name: data_bus_32bit

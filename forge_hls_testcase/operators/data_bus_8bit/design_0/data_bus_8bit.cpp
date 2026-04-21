#include <stdint.h>

#define DATA_SIZE 1024

void data_bus_8bit(uint8_t input[DATA_SIZE], uint8_t output[DATA_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    for (int i = 0; i < DATA_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        output[i] = input[i];
    }
}

// Top function name: data_bus_8bit

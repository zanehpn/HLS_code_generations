#include <stdint.h>

#define ARRAY_SIZE 1024

void sram_cell(uint16_t input_array[ARRAY_SIZE], uint16_t output_array[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output_array type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input_array type=cyclic dim=1 factor=8
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        // Simple operation on SRAM cell
        output_array[i] = input_array[i] + 1;
    }
}

// Top function name: sram_cell

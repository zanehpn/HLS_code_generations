#include <stdint.h>

#define ARRAY_SIZE 1024

void dac_24bit(uint32_t input[ARRAY_SIZE], uint8_t output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        uint32_t value = input[i];
        uint8_t msb = (value >> 16) & 0xFF; // Extract the most significant byte
        uint8_t mid = (value >> 8) & 0xFF;  // Extract the middle byte
        uint8_t lsb = value & 0xFF;         // Extract the least significant byte
        output[i] = (msb + mid + lsb) / 3;  // Simple averaging for DAC output
    }
}

// Top function name: dac_24bit

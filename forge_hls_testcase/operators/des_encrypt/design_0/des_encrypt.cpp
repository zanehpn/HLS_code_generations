#include <stdint.h>

#define DATA_SIZE 1024

void des_encrypt(uint64_t input[DATA_SIZE], uint64_t output[DATA_SIZE], uint64_t key) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=4
    uint64_t subkeys[16];
    int i, j;

    // Key schedule
    for (i = 0; i < 16; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        subkeys[i] = key; // Simplified key schedule for demonstration
    }

    // Initial permutation (simplified)
    for (i = 0; i < DATA_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        uint64_t data = input[i];
        uint64_t left = data >> 32;
        uint64_t right = data & 0xFFFFFFFF;

        // 16 rounds of DES
        for (j = 0; j < 16; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
            uint64_t temp = right;
            right = left ^ (right + subkeys[j]); // Simplified round function
            left = temp;
        }

        // Final permutation (simplified)
        output[i] = (right << 32) | left;
    }
}

// Top function name: des_encrypt

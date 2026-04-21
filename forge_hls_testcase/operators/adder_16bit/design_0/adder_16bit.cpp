#include <cstdint>

#define ARRAY_SIZE 1024

void adder_16bit(uint16_t A[ARRAY_SIZE], uint16_t B[ARRAY_SIZE], uint16_t C[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        C[i] = A[i] + B[i];
    }
}

// Top function name: adder_16bit

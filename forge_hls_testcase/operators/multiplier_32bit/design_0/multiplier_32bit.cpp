#include <stdint.h>

#define ARRAY_SIZE 1024

void multiplier_32bit(uint32_t A[ARRAY_SIZE], uint32_t B[ARRAY_SIZE], uint32_t C[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=4
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        C[i] = A[i] * B[i];
    }
}

// Top function name: multiplier_32bit

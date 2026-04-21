#include <ap_fixed.h>

#define ARRAY_SIZE 1024

typedef ap_fixed<16, 8> fixed_point_t;

void fixed_point_addition(fixed_point_t A[ARRAY_SIZE], fixed_point_t B[ARRAY_SIZE], fixed_point_t C[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        C[i] = A[i] + B[i];
    }
}

// Top function name: fixed_point_addition

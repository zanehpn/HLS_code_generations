#include <ap_int.h>

#define ARRAY_SIZE 1024

void high_speed_6bit_adder(ap_uint<6> A[ARRAY_SIZE], ap_uint<6> B[ARRAY_SIZE], ap_uint<6> C[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=16
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        C[i] = A[i] + B[i];
    }
}

// Top function name: high_speed_6bit_adder

#include <ap_int.h>

#define SIZE 1024

void nand_gate(ap_uint<1> A[SIZE], ap_uint<1> B[SIZE], ap_uint<1> C[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        C[i] = ~(A[i] & B[i]);
    }
}

// Top function name: nand_gate

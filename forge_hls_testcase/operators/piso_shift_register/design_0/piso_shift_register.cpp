#include <ap_int.h>

#define SIZE 1024

void piso_shift_register(ap_uint<1> parallel_in[SIZE], ap_uint<1> &serial_out) {
#pragma HLS ARRAY_PARTITION variable=parallel_in type=cyclic dim=1 factor=1
    static ap_uint<1> shift_reg[SIZE];
    int i;

    // Load parallel input into shift register
    for (i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        shift_reg[i] = parallel_in[i];
    }

    // Shift out the bits serially
    for (i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        serial_out = shift_reg[i];
    }
}

// Top function name: piso_shift_register

#include <ap_int.h>

#define ARRAY_SIZE 1024

void true_random_number_generator(ap_uint<32> seed, ap_uint<32> output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
    ap_uint<32> lfsr = seed;
    ap_uint<32> bit;

    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        bit = ((lfsr >> 0) ^ (lfsr >> 2) ^ (lfsr >> 3) ^ (lfsr >> 5)) & 1;
        lfsr = (lfsr >> 1) | (bit << 31);
        output[i] = lfsr;
    }
}

// Top function name: true_random_number_generator

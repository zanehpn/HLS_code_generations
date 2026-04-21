#include <ap_int.h>

#define ARRAY_SIZE 1024

void pll_clock(ap_uint<32> input_signal[ARRAY_SIZE], ap_uint<32> output_signal[ARRAY_SIZE], ap_uint<32> multiplier, ap_uint<32> divider) {
#pragma HLS ARRAY_PARTITION variable=output_signal type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input_signal type=cyclic dim=1 factor=2
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        ap_uint<64> temp = input_signal[i];
        temp = (temp * multiplier) / divider;
        output_signal[i] = temp;
    }
}

// Top function name: pll_clock

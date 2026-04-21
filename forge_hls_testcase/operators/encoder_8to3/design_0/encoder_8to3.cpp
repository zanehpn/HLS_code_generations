#include <ap_int.h>

#define INPUT_SIZE 1024

void encoder_8to3(ap_uint<8> input[INPUT_SIZE], ap_uint<3> output[INPUT_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    for (int i = 0; i < INPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        ap_uint<8> in = input[i];
        ap_uint<3> out = 0;
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            if (in[j] == 1) {
                out = j;
                break;
            }
        }
        output[i] = out;
    }
}

// Top function name: encoder_8to3

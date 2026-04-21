#include <ap_int.h>

#define INPUT_SIZE 1024

void decoder_2to4(ap_uint<2> input[INPUT_SIZE], ap_uint<4> output[INPUT_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=16
    for (int i = 0; i < INPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        switch (input[i]) {
            case 0:
                output[i] = 1; // 0001
                break;
            case 1:
                output[i] = 2; // 0010
                break;
            case 2:
                output[i] = 4; // 0100
                break;
            case 3:
                output[i] = 8; // 1000
                break;
            default:
                output[i] = 0; // Should never happen
                break;
        }
    }
}

// Top function name: decoder_2to4

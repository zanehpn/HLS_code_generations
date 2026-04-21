#include <ap_int.h>

#define ARRAY_SIZE 1024

void micro_power_quad_comparator(
    ap_int<16> input1[ARRAY_SIZE],
    ap_int<16> input2[ARRAY_SIZE],
    ap_int<16> input3[ARRAY_SIZE],
    ap_int<16> input4[ARRAY_SIZE],
    ap_int<1> output1[ARRAY_SIZE],
    ap_int<1> output2[ARRAY_SIZE],
    ap_int<1> output3[ARRAY_SIZE],
    ap_int<1> output4[ARRAY_SIZE]
) {
#pragma HLS ARRAY_PARTITION variable=output4 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=output3 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=output2 type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=output1 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input4 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input3 type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=4
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output1[i] = (input1[i] > input2[i]) ? 1 : 0;
        output2[i] = (input2[i] > input3[i]) ? 1 : 0;
        output3[i] = (input3[i] > input4[i]) ? 1 : 0;
        output4[i] = (input4[i] > input1[i]) ? 1 : 0;
    }
}

// Top function name: micro_power_quad_comparator

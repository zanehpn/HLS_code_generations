#include <iostream>

#define ARRAY_SIZE 1024

void instruction_pipeline(int input[ARRAY_SIZE], int output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    int stage1[ARRAY_SIZE];
    int stage2[ARRAY_SIZE];
    int stage3[ARRAY_SIZE];

    // Stage 1: Simple transformation
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        stage1[i] = input[i] + 1;
    }

    // Stage 2: Another transformation
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        stage2[i] = stage1[i] * 2;
    }

    // Stage 3: Final transformation
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        stage3[i] = stage2[i] - 3;
    }

    // Output the result
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        output[i] = stage3[i];
    }
}

// Top function name: instruction_pipeline

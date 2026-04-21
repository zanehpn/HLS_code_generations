#include <stdint.h>

#define ARRAY_SIZE 1024

void hazard_detection_resolution(int32_t input1[ARRAY_SIZE], int32_t input2[ARRAY_SIZE], int32_t output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=2
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        int32_t temp1 = input1[i];
        int32_t temp2 = input2[i];
        
        // Hazard detection and resolution logic
        if (temp1 > temp2) {
            output[i] = temp1 - temp2;
        } else if (temp1 < temp2) {
            output[i] = temp2 - temp1;
        } else {
            output[i] = 0; // No hazard
        }
    }
}

// Top function name: hazard_detection_resolution

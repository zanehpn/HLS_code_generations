#include <stdint.h>

#define TIMER_SIZE 1024

void timer_32bit(uint32_t input[TIMER_SIZE], uint32_t output[TIMER_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    for (int i = 0; i < TIMER_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        output[i] = input[i] + 1;
    }
}

// Top function name: timer_32bit

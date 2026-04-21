#include <stdint.h>

#define TIMER_SIZE 1024

void timer_16bit(uint16_t start_value, uint16_t increment, uint16_t timer[TIMER_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=timer type=cyclic dim=1 factor=1
    for (int i = 0; i < TIMER_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        timer[i] = start_value + i * increment;
    }
}

// Top function name: timer_16bit

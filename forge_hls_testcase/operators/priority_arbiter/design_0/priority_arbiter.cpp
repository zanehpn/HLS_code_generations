#include <stdint.h>

#define NUM_REQUESTS 1024

void priority_arbiter(uint8_t requests[NUM_REQUESTS], uint8_t grants[NUM_REQUESTS]) {
#pragma HLS ARRAY_PARTITION variable=grants type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=requests type=cyclic dim=1 factor=1
    for (int i = 0; i < NUM_REQUESTS; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        grants[i] = 0;
    }

    for (int i = 0; i < NUM_REQUESTS; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        if (requests[i] == 1) {
            grants[i] = 1;
            break;
        }
    }
}

// Top function name: priority_arbiter

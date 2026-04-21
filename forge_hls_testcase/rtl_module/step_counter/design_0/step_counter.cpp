#include <ap_int.h>

#define ARRAY_SIZE 1024

void step_counter(ap_int<32> steps[ARRAY_SIZE], ap_int<32> threshold, ap_int<32> &count) {
#pragma HLS ARRAY_PARTITION variable=steps type=cyclic dim=1 factor=2
    count = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        if (steps[i] > threshold) {
            count++;
        }
    }
}

// Top function name: step_counter

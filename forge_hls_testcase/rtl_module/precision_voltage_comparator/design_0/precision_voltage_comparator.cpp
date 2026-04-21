#include <ap_int.h>

#define ARRAY_SIZE 1024

void precision_voltage_comparator(ap_int<16> voltage1[ARRAY_SIZE], ap_int<16> voltage2[ARRAY_SIZE], ap_int<1> result[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=result type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=voltage2 type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=voltage1 type=cyclic dim=1 factor=8
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        if (voltage1[i] > voltage2[i]) {
            result[i] = 1;
        } else {
            result[i] = 0;
        }
    }
}
// Top function name: precision_voltage_comparator

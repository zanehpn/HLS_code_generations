#include <ap_fixed.h>

#define ARRAY_SIZE 1024

void voltage_comparator(ap_fixed<16, 8> input_voltage[ARRAY_SIZE], ap_fixed<16, 8> reference_voltage, bool output_result[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output_result type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input_voltage type=cyclic dim=1 factor=16
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output_result[i] = (input_voltage[i] > reference_voltage);
    }
}

// Top function name: voltage_comparator

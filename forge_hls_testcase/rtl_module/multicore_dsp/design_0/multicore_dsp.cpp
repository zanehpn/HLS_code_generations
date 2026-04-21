#include <ap_fixed.h>
#include <ap_int.h>

#define ARRAY_SIZE 1024

typedef ap_fixed<16, 8> fixed_point_t;
typedef float floating_point_t;

void multicore_dsp(fixed_point_t input_fixed[ARRAY_SIZE], floating_point_t input_float[ARRAY_SIZE], fixed_point_t output_fixed[ARRAY_SIZE], floating_point_t output_float[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output_float type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=output_fixed type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input_float type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input_fixed type=cyclic dim=1 factor=4
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        // Fixed-point processing
        fixed_point_t temp_fixed = input_fixed[i] * input_fixed[i];
        output_fixed[i] = temp_fixed + input_fixed[i];

        // Floating-point processing
        floating_point_t temp_float = input_float[i] * input_float[i];
        output_float[i] = temp_float + input_float[i];
    }
}

// Top function name: multicore_dsp

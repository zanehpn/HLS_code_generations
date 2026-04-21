#include <cmath>

#define N 1024

void carradio_signal_processor(float input_signal[N], float output_signal[N], float gain, float offset) {
#pragma HLS ARRAY_PARTITION variable=output_signal type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input_signal type=cyclic dim=1 factor=16
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        // Apply gain
        float amplified_signal = input_signal[i] * gain;

        // Apply offset
        float shifted_signal = amplified_signal + offset;

        // Apply a simple non-linear transformation (e.g., squaring)
        float processed_signal = shifted_signal * shifted_signal;

        // Store the result
        output_signal[i] = processed_signal;
    }
}

// Top function name: carradio_signal_processor

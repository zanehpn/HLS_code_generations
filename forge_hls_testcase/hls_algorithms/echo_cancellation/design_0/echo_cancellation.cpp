#include <iostream>
#include <cmath>

#define N 1024

void echo_cancellation(float input_signal[N], float echo_signal[N], float output_signal[N], float filter_coeff[N]) {
#pragma HLS ARRAY_PARTITION variable=filter_coeff type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=output_signal type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=echo_signal type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input_signal type=cyclic dim=1 factor=2
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        float estimated_echo = 0.0;
        for (int j = 0; j < N; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            if (i - j >= 0) {
                estimated_echo += filter_coeff[j] * input_signal[i - j];
            }
        }
        output_signal[i] = input_signal[i] - estimated_echo + echo_signal[i];
    }
}

// Top function name: echo_cancellation

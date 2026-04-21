#include <cmath>
#include <cstdint>

#define N 1024

void digital_phase_shifter(int16_t input_real[N], int16_t input_imag[N], int16_t output_real[N], int16_t output_imag[N], float phase_shift) {
#pragma HLS ARRAY_PARTITION variable=output_imag type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=output_real type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input_imag type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input_real type=cyclic dim=1 factor=4
    float cos_val = cos(phase_shift);
    float sin_val = sin(phase_shift);

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        output_real[i] = static_cast<int16_t>(input_real[i] * cos_val - input_imag[i] * sin_val);
        output_imag[i] = static_cast<int16_t>(input_real[i] * sin_val + input_imag[i] * cos_val);
    }
}

// Top function name: digital_phase_shifter

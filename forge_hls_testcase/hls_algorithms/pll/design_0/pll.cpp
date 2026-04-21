#include <cmath>
#include <cstdint>

#define N 1024

void pll(float input_signal[N], float output_signal[N], float ref_frequency, float loop_bandwidth, float damping_factor) {
#pragma HLS ARRAY_PARTITION variable=output_signal type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input_signal type=cyclic dim=1 factor=16
    float phase_error[N] = {0};
    float vco_phase[N] = {0};
    float vco_frequency[N] = {0};
    float integrator[N] = {0};
    float proportional_gain = 2.0f * damping_factor * loop_bandwidth;
    float integrator_gain = loop_bandwidth * loop_bandwidth;

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        // Phase detector
        phase_error[i] = input_signal[i] - vco_phase[i];

        // Loop filter
        integrator[i] = integrator[i] + integrator_gain * phase_error[i];
        float control_signal = proportional_gain * phase_error[i] + integrator[i];

        // Voltage Controlled Oscillator (VCO)
        vco_frequency[i] = ref_frequency + control_signal;
        if (i > 0) {
            vco_phase[i] = vco_phase[i-1] + vco_frequency[i];
        } else {
            vco_phase[i] = vco_frequency[i];
        }

        // Output signal
        output_signal[i] = sin(vco_phase[i]);
    }
}

// Top function name: pll

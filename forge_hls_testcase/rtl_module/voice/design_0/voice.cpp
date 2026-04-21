#include <cmath>
#include <cstdint>

#define N 1024

void voice(float input[N], float output[N]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    float sum = 0.0;
    float mean = 0.0;
    float variance = 0.0;
    float stddev = 0.0;

    // Calculate mean
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        sum += input[i];
    }
    mean = sum / N;

    // Calculate variance
    sum = 0.0;
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        sum += (input[i] - mean) * (input[i] - mean);
    }
    variance = sum / N;

    // Calculate standard deviation
    stddev = std::sqrt(variance);

    // Normalize input
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        output[i] = (input[i] - mean) / stddev;
    }
}

// Top function name: voice

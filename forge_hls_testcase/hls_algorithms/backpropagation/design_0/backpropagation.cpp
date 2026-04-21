#include <cmath>

#define INPUT_SIZE 1024
#define HIDDEN_SIZE 512
#define OUTPUT_SIZE 256

void backpropagation(
    float input[INPUT_SIZE],
    float hidden_weights[INPUT_SIZE][HIDDEN_SIZE],
    float output_weights[HIDDEN_SIZE][OUTPUT_SIZE],
    float hidden_bias[HIDDEN_SIZE],
    float output_bias[OUTPUT_SIZE],
    float target[OUTPUT_SIZE],
    float learning_rate)
{
#pragma HLS ARRAY_PARTITION variable=target type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=output_bias type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=hidden_bias type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=output_weights type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=output_weights type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=hidden_weights type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=hidden_weights type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    float hidden_layer[HIDDEN_SIZE];
    float output_layer[OUTPUT_SIZE];
    float hidden_delta[HIDDEN_SIZE];
    float output_delta[OUTPUT_SIZE];

    // Forward pass
    for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        hidden_layer[i] = hidden_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
            hidden_layer[i] += input[j] * hidden_weights[j][i];
        }
        hidden_layer[i] = 1.0 / (1.0 + exp(-hidden_layer[i])); // Sigmoid activation
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        output_layer[i] = output_bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            output_layer[i] += hidden_layer[j] * output_weights[j][i];
        }
        output_layer[i] = 1.0 / (1.0 + exp(-output_layer[i])); // Sigmoid activation
    }

    // Backward pass
    for (int i = 0; i < OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
        output_delta[i] = (target[i] - output_layer[i]) * output_layer[i] * (1.0 - output_layer[i]);
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        hidden_delta[i] = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
            hidden_delta[i] += output_delta[j] * output_weights[i][j];
        }
        hidden_delta[i] *= hidden_layer[i] * (1.0 - hidden_layer[i]);
    }

    // Update weights and biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output_bias[i] += learning_rate * output_delta[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            output_weights[j][i] += learning_rate * hidden_layer[j] * output_delta[i];
        }
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        hidden_bias[i] += learning_rate * hidden_delta[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
            hidden_weights[j][i] += learning_rate * input[j] * hidden_delta[i];
        }
    }
}

// Top function name: backpropagation

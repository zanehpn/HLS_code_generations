#include <cstdint>

#define INPUT_SIZE 1024

void priority_encoder(uint32_t input[INPUT_SIZE], uint32_t output[INPUT_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=1
    for (int i = 0; i < INPUT_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        uint32_t in_val = input[i];
        uint32_t out_val = 0;
        for (int j = 0; j < 32; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            if (in_val & (1 << j)) {
                out_val = j;
                break;
            }
        }
        output[i] = out_val;
    }
}

// Top function name: priority_encoder

#include <stdint.h>

#define WIDTH 1024
#define HEIGHT 1024

void video_processor(uint8_t input[HEIGHT][WIDTH], uint8_t output[HEIGHT][WIDTH]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=16
    for (int i = 0; i < HEIGHT; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        for (int j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            uint8_t pixel = input[i][j];
            // Example processing: simple inversion
            output[i][j] = 255 - pixel;
        }
    }
}

// Top function name: video_processor

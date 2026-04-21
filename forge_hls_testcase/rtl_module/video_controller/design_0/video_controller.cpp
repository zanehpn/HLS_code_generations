#include <ap_int.h>

#define WIDTH 1920
#define HEIGHT 1080

void video_controller(ap_uint<8> input_frame[HEIGHT][WIDTH], ap_uint<8> output_frame[HEIGHT][WIDTH]) {
#pragma HLS ARRAY_PARTITION variable=output_frame type=cyclic dim=2 factor=1
#pragma HLS ARRAY_PARTITION variable=output_frame type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input_frame type=cyclic dim=2 factor=8
#pragma HLS ARRAY_PARTITION variable=input_frame type=cyclic dim=1 factor=1
    int i, j;
    for (i = 0; i < HEIGHT; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        for (j = 0; j < WIDTH; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
            // Example processing: Invert the pixel value
            output_frame[i][j] = 255 - input_frame[i][j];
        }
    }
}

// Top function name: video_controller

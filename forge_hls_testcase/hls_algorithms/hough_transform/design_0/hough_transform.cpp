#include <cmath>
#include <ap_int.h>

#define WIDTH 1024
#define HEIGHT 1024
#define THETA_SIZE 180
#define RHO_SIZE 1448

void hough_transform(ap_uint<8> image[HEIGHT][WIDTH], ap_uint<32> accumulator[THETA_SIZE][RHO_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=accumulator type=cyclic dim=2 factor=16
#pragma HLS ARRAY_PARTITION variable=accumulator type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=image type=cyclic dim=2 factor=16
#pragma HLS ARRAY_PARTITION variable=image type=cyclic dim=1 factor=2
    int x, y, theta;
    int rho;
    float theta_radians;
    float cos_theta, sin_theta;

    // Initialize the accumulator array
    for (theta = 0; theta < THETA_SIZE; theta++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        for (rho = 0; rho < RHO_SIZE; rho++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
            accumulator[theta][rho] = 0;
        }
    }

    // Perform the Hough Transform
    for (y = 0; y < HEIGHT; y++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
        for (x = 0; x < WIDTH; x++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
            if (image[y][x] > 0) { // Edge pixel
                for (theta = 0; theta < THETA_SIZE; theta++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
                    theta_radians = theta * M_PI / 180.0;
                    cos_theta = cos(theta_radians);
                    sin_theta = sin(theta_radians);
                    rho = (int)(x * cos_theta + y * sin_theta + (RHO_SIZE / 2));
                    if (rho >= 0 && rho < RHO_SIZE) {
                        accumulator[theta][rho]++;
                    }
                }
            }
        }
    }
}

// Top function name: hough_transform

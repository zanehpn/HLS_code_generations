#include <ap_int.h>

#define IMG_SIZE 1024

void otsu_thresholding(ap_uint<8> image[IMG_SIZE], ap_uint<8> thresholded_image[IMG_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=thresholded_image type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=image type=cyclic dim=1 factor=2
    ap_uint<32> histogram[256] = {0};
    ap_uint<32> total_pixels = IMG_SIZE;
    ap_uint<32> sum = 0;
    ap_uint<32> sumB = 0;
    ap_uint<32> wB = 0;
    ap_uint<32> wF = 0;
    ap_uint<8> threshold = 0;
    ap_uint<32> max_variance = 0;

    // Compute histogram
    for (int i = 0; i < IMG_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
        histogram[image[i]]++;
    }

    // Compute sum of all pixel values
    for (int t = 0; t < 256; t++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        sum += t * histogram[t];
    }

    // Find optimal threshold
    for (int t = 0; t < 256; t++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        wB += histogram[t];
        if (wB == 0) continue;
        wF = total_pixels - wB;
        if (wF == 0) break;

        sumB += t * histogram[t];
        ap_uint<32> mB = sumB / wB;
        ap_uint<32> mF = (sum - sumB) / wF;

        ap_uint<32> variance = wB * wF * (mB - mF) * (mB - mF);
        if (variance > max_variance) {
            max_variance = variance;
            threshold = t;
        }
    }

    // Apply threshold to the image
    for (int i = 0; i < IMG_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        thresholded_image[i] = (image[i] > threshold) ? 255 : 0;
    }
}

// Top function name: otsu_thresholding

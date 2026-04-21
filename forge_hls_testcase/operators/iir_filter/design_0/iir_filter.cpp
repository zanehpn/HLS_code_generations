#include <ap_fixed.h>

#define N 1024

void iir_filter(ap_fixed<16, 8> input[N], ap_fixed<16, 8> output[N], ap_fixed<16, 8> a1, ap_fixed<16, 8> a2, ap_fixed<16, 8> b0, ap_fixed<16, 8> b1, ap_fixed<16, 8> b2) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    ap_fixed<16, 8> x1 = 0, x2 = 0, y1 = 0, y2 = 0;

    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        ap_fixed<16, 8> x0 = input[i];
        ap_fixed<16, 8> y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;

        output[i] = y0;

        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
    }
}

// Top function name: iir_filter

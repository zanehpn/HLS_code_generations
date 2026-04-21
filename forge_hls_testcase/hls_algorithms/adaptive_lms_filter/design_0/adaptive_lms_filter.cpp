#include <cmath>

#define N 1024

void adaptive_lms_filter(float x[N], float d[N], float y[N], float e[N], float w[N], float mu, int filter_length) {
#pragma HLS ARRAY_PARTITION variable=w type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=e type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=y type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=d type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=x type=cyclic dim=1 factor=8
    float y_n;
    for (int n = 0; n < N; n++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        y_n = 0.0;
        for (int k = 0; k < filter_length; k++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
            if (n - k >= 0) {
                y_n += w[k] * x[n - k];
            }
        }
        y[n] = y_n;
        e[n] = d[n] - y_n;
        for (int k = 0; k < filter_length; k++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            if (n - k >= 0) {
                w[k] += 2 * mu * e[n] * x[n - k];
            }
        }
    }
}

// Top function name: adaptive_lms_filter

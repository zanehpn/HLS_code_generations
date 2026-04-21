#include <ap_int.h>

#define N 1024

void binary_rate_multiplier(ap_uint<1> input[N], ap_uint<1> rate[N], ap_uint<1> output[N]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=rate type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=16
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output[i] = input[i] & rate[i];
    }
}

// Top function name: binary_rate_multiplier

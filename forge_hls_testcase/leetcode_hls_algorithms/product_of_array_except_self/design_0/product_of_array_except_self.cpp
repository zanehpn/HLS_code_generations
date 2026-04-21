#include <iostream>

#define ARRAY_SIZE 1024

void product_of_array_except_self(int input[ARRAY_SIZE], int output[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    int left_products[ARRAY_SIZE];
    int right_products[ARRAY_SIZE];

    left_products[0] = 1;
    for (int i = 1; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        left_products[i] = left_products[i - 1] * input[i - 1];
    }

    right_products[ARRAY_SIZE - 1] = 1;
    for (int i = ARRAY_SIZE - 2; i >= 0; i--) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        right_products[i] = right_products[i + 1] * input[i + 1];
    }

    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        output[i] = left_products[i] * right_products[i];
    }
}

// Top function name: product_of_array_except_self

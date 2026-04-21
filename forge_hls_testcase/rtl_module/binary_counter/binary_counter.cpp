#include <ap_int.h>

#define SIZE 1024

void binary_counter(ap_uint<32> input[SIZE], ap_uint<32> output[SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        ap_uint<32> count = 0;
        ap_uint<32> value = input[i];
        for (int j = 0; j < 32; j++) {
            count += value[j];
        }
        output[i] = count;
    }
}

// Top function name: binary_counter

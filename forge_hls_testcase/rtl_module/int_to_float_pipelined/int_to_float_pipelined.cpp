#include <ap_int.h>
#include <ap_fixed.h>

#define SIZE 1024

void int_to_float_pipelined(ap_int<32> input[SIZE], ap_fixed<32,16> output[SIZE]) {
    for (int i = 0; i < SIZE; i++) {
        output[i] = (ap_fixed<32,16>)input[i];
    }
}

// Top function name: int_to_float_pipelined

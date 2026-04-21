#include <iostream>

#define SIZE 1024

void pci_algorithm(int input1[SIZE], int input2[SIZE], int output[SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input2 type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input1 type=cyclic dim=1 factor=4
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        output[i] = input1[i] + input2[i];
    }
}

// Top function name: pci_algorithm

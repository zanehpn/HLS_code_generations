#include <ap_int.h>

#define ARRAY_SIZE 1024

void insertion_sort(ap_int<32> array[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=array type=cyclic dim=1 factor=2
    for (int i = 1; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        ap_int<32> key = array[i];
        int j = i - 1;
        for (; j >= 0 && array[j] > key; j--) {
            array[j + 1] = array[j];
        }
        array[j + 1] = key;
    }
}

// Top function name: insertion_sort

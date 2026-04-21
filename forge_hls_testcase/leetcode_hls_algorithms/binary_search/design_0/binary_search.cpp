#include <ap_int.h>

#define ARRAY_SIZE 1024

void binary_search(ap_int<32> array[ARRAY_SIZE], ap_int<32> key, ap_int<32> *index) {
#pragma HLS ARRAY_PARTITION variable=array type=cyclic dim=1 factor=2
    ap_int<32> low = 0;
    ap_int<32> high = ARRAY_SIZE - 1;
    ap_int<32> mid;
    *index = -1;

    for (ap_int<32> i = 0; i < ARRAY_SIZE; i++) {
        if (low <= high) {
            mid = (low + high) / 2;
            if (array[mid] == key) {
                *index = mid;
                break;
            } else if (array[mid] < key) {
                low = mid + 1;
            } else {
                high = mid - 1;
            }
        }
    }
}

// Top function name: binary_search

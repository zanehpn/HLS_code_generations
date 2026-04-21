#include <stdint.h>

#define ARRAY_SIZE 1024

void ip_checksum(uint16_t data[ARRAY_SIZE], uint16_t& checksum) {
#pragma HLS ARRAY_PARTITION variable=data type=cyclic dim=1 factor=1
    uint32_t sum = 0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
        sum += data[i];
        if (sum > 0xFFFF) {
            sum = (sum & 0xFFFF) + (sum >> 16);
        }
    }
    checksum = ~((uint16_t)sum);
}

// Top function name: ip_checksum

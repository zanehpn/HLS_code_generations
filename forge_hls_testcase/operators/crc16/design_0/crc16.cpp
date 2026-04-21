#include <stdint.h>

#define ARRAY_SIZE 1024

void crc16(uint16_t data[ARRAY_SIZE], uint16_t crc[ARRAY_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=crc type=cyclic dim=1 factor=16
#pragma HLS ARRAY_PARTITION variable=data type=cyclic dim=1 factor=4
    uint16_t polynomial = 0x1021;
    for (int i = 0; i < ARRAY_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        uint16_t current_crc = 0xFFFF;
        for (int j = 0; j < 16; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            bool bit = ((data[i] >> (15 - j) & 1) == 1);
            bool c15 = ((current_crc >> 15 & 1) == 1);
            current_crc <<= 1;
            if (c15 ^ bit) {
                current_crc ^= polynomial;
            }
        }
        crc[i] = current_crc;
    }
}

// Top function name: crc16

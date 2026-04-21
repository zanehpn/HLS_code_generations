#include <stdint.h>

#define DATA_SIZE 1024

void crc8(uint8_t data[DATA_SIZE], uint8_t* crc_out) {
#pragma HLS ARRAY_PARTITION variable=data type=cyclic dim=1 factor=16
    uint8_t crc = 0x00;
    uint8_t polynomial = 0x07;

    for (int i = 0; i < DATA_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            if (crc & 0x80) {
                crc = (crc << 1) ^ polynomial;
            } else {
                crc <<= 1;
            }
        }
    }
    *crc_out = crc;
}

// Top function name: crc8

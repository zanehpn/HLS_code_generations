#include <ap_int.h>

#define DATA_SIZE 1024

void hamming_encoder(ap_uint<8> data_in[DATA_SIZE], ap_uint<12> data_out[DATA_SIZE]) {
    for (int i = 0; i < DATA_SIZE; i++) {
        ap_uint<8> d = data_in[i];
        ap_uint<12> encoded = 0;

        // Assign data bits
        encoded[2] = d[0];
        encoded[4] = d[1];
        encoded[5] = d[2];
        encoded[6] = d[3];
        encoded[8] = d[4];
        encoded[9] = d[5];
        encoded[10] = d[6];
        encoded[11] = d[7];

        // Calculate parity bits
        encoded[0] = d[0] ^ d[1] ^ d[3] ^ d[4] ^ d[6];
        encoded[1] = d[0] ^ d[2] ^ d[3] ^ d[5] ^ d[6];
        encoded[3] = d[1] ^ d[2] ^ d[3] ^ d[7];
        encoded[7] = d[4] ^ d[5] ^ d[6] ^ d[7];

        data_out[i] = encoded;
    }
}

// Top function name: hamming_encoder

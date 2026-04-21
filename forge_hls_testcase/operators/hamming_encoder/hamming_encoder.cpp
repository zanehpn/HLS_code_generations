#include <ap_int.h>

#define DATA_SIZE 1024

void hamming_encoder(ap_uint<16> input[DATA_SIZE], ap_uint<21> output[DATA_SIZE]) {
    for (int idx = 0; idx < DATA_SIZE; idx++) {
        ap_uint<16> data = input[idx];
        ap_uint<21> encoded = 0;

        // Assign data bits to encoded positions
        encoded[2] = data[0];
        encoded[4] = data[1];
        encoded[5] = data[2];
        encoded[6] = data[3];
        encoded[8] = data[4];
        encoded[9] = data[5];
        encoded[10] = data[6];
        encoded[11] = data[7];
        encoded[12] = data[8];
        encoded[13] = data[9];
        encoded[14] = data[10];
        encoded[16] = data[11];
        encoded[17] = data[12];
        encoded[18] = data[13];
        encoded[19] = data[14];
        encoded[20] = data[15];

        // Calculate parity bits
        encoded[0] = encoded[2] ^ encoded[4] ^ encoded[6] ^ encoded[8] ^ encoded[10] ^ encoded[12] ^ encoded[14] ^ encoded[16] ^ encoded[18] ^ encoded[20];
        encoded[1] = encoded[2] ^ encoded[5] ^ encoded[6] ^ encoded[9] ^ encoded[10] ^ encoded[13] ^ encoded[14] ^ encoded[17] ^ encoded[18];
        encoded[3] = encoded[4] ^ encoded[5] ^ encoded[6] ^ encoded[11] ^ encoded[12] ^ encoded[13] ^ encoded[14] ^ encoded[19] ^ encoded[20];
        encoded[7] = encoded[8] ^ encoded[9] ^ encoded[10] ^ encoded[11] ^ encoded[12] ^ encoded[13] ^ encoded[14];
        encoded[15] = encoded[16] ^ encoded[17] ^ encoded[18] ^ encoded[19] ^ encoded[20];

        output[idx] = encoded;
    }
}

// Top function name: hamming_encoder

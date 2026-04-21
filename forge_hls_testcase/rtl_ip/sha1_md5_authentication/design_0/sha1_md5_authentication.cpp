#include <stdint.h>

#define DATA_SIZE 1024

void sha1_md5_authentication(uint8_t input[DATA_SIZE], uint8_t output[20]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    uint32_t w[80];
    uint32_t a, b, c, d, e, f, k, temp;

    for (int i = 0; i < DATA_SIZE; i += 64) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        for (int j = 0; j < 16; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
            w[j] = (input[i + 4 * j] << 24) | (input[i + 4 * j + 1] << 16) | (input[i + 4 * j + 2] << 8) | (input[i + 4 * j + 3]);
        }

        for (int j = 16; j < 80; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
            w[j] = (w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16]);
            w[j] = (w[j] << 1) | (w[j] >> 31);
        }

        a = h0;
        b = h1;
        c = h2;
        d = h3;
        e = h4;

        for (int j = 0; j < 80; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
            if (j < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (j < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (j < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            temp = ((a << 5) | (a >> 27)) + f + e + k + w[j];
            e = d;
            d = c;
            c = (b << 30) | (b >> 2);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    output[0] = (h0 >> 24) & 0xFF;
    output[1] = (h0 >> 16) & 0xFF;
    output[2] = (h0 >> 8) & 0xFF;
    output[3] = h0 & 0xFF;
    output[4] = (h1 >> 24) & 0xFF;
    output[5] = (h1 >> 16) & 0xFF;
    output[6] = (h1 >> 8) & 0xFF;
    output[7] = h1 & 0xFF;
    output[8] = (h2 >> 24) & 0xFF;
    output[9] = (h2 >> 16) & 0xFF;
    output[10] = (h2 >> 8) & 0xFF;
    output[11] = h2 & 0xFF;
    output[12] = (h3 >> 24) & 0xFF;
    output[13] = (h3 >> 16) & 0xFF;
    output[14] = (h3 >> 8) & 0xFF;
    output[15] = h3 & 0xFF;
    output[16] = (h4 >> 24) & 0xFF;
    output[17] = (h4 >> 16) & 0xFF;
    output[18] = (h4 >> 8) & 0xFF;
    output[19] = h4 & 0xFF;
}

// Top function name: sha1_md5_authentication

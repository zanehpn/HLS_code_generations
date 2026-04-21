#include <cstdint>
#include <cstring>

#define INPUT_SIZE 1024

const uint32_t k[64] = {
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391
};

const uint32_t r[] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
};

void md5(const uint8_t input[INPUT_SIZE], uint8_t output[16]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=2
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xefcdab89;
    uint32_t h2 = 0x98badcfe;
    uint32_t h3 = 0x10325476;

    uint8_t data[64];
    uint32_t bit_len = INPUT_SIZE * 8;
    uint32_t new_len = ((bit_len + 64) / 512 + 1) * 512 / 8;

    for (uint32_t i = 0; i < new_len; i += 64) {
        if (i + 64 <= INPUT_SIZE) {
            memcpy(data, input + i, 64);
        } else {
            memset(data, 0, 64);
            if (i < INPUT_SIZE) {
                memcpy(data, input + i, INPUT_SIZE - i);
                data[INPUT_SIZE - i] = 0x80;
            } else if (i == INPUT_SIZE) {
                data[0] = 0x80;
            }
            if (i + 64 - 8 <= INPUT_SIZE) {
                uint32_t bits = bit_len;
                memcpy(data + 56, &bits, 4);
            }
        }

        uint32_t a = h0, b = h1, c = h2, d = h3, f, g;
        for (uint32_t j = 0; j < 64; j++) {
            if (j < 16) {
                f = (b & c) | ((~b) & d);
                g = j;
            } else if (j < 32) {
                f = (d & b) | ((~d) & c);
                g = (5 * j + 1) % 16;
            } else if (j < 48) {
                f = b ^ c ^ d;
                g = (3 * j + 5) % 16;
            } else {
                f = c ^ (b | (~d));
                g = (7 * j) % 16;
            }
            uint32_t temp = d;
            d = c;
            c = b;
            b = b + ((a + f + k[j] + ((uint32_t*)data)[g]) << r[j] | (a + f + k[j] + ((uint32_t*)data)[g]) >> (32 - r[j]));
            a = temp;
        }
        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
    }

    memcpy(output, &h0, 4);
    memcpy(output + 4, &h1, 4);
    memcpy(output + 8, &h2, 4);
    memcpy(output + 12, &h3, 4);
}

// Top function name: md5

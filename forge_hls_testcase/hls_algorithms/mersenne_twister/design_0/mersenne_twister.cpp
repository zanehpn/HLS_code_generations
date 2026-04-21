#include <stdint.h>

#define N 1024

void mersenne_twister(uint32_t seed, uint32_t output[N]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=8
    const uint32_t w = 32;
    const uint32_t n = 624;
    const uint32_t m = 397;
    const uint32_t r = 31;
    const uint32_t a = 0x9908B0DF;
    const uint32_t u = 11;
    const uint32_t d = 0xFFFFFFFF;
    const uint32_t s = 7;
    const uint32_t b = 0x9D2C5680;
    const uint32_t t = 15;
    const uint32_t c = 0xEFC60000;
    const uint32_t l = 18;
    const uint32_t f = 1812433253;

    uint32_t MT[n];
    uint32_t index = n + 1;

    // Initialize the generator from a seed
    MT[0] = seed;
    for (uint32_t i = 1; i < n; ++i) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        MT[i] = (f * (MT[i - 1] ^ (MT[i - 1] >> (w - 2))) + i);
    }

    // Generate an array of n untempered numbers
    for (uint32_t i = 0; i < n - m; ++i) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
        uint32_t x = (MT[i] & ((1 << r) - 1)) + (MT[i + 1] & ~((1 << r) - 1));
        uint32_t xA = x >> 1;
        if (x % 2 != 0) {
            xA ^= a;
        }
        MT[i] = MT[i + m] ^ xA;
    }
    for (uint32_t i = n - m; i < n - 1; ++i) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        uint32_t x = (MT[i] & ((1 << r) - 1)) + (MT[i + 1] & ~((1 << r) - 1));
        uint32_t xA = x >> 1;
        if (x % 2 != 0) {
            xA ^= a;
        }
        MT[i] = MT[i + (m - n)] ^ xA;
    }
    uint32_t x = (MT[n - 1] & ((1 << r) - 1)) + (MT[0] & ~((1 << r) - 1));
    uint32_t xA = x >> 1;
    if (x % 2 != 0) {
        xA ^= a;
    }
    MT[n - 1] = MT[m - 1] ^ xA;

    // Tempering
    for (uint32_t i = 0; i < N; ++i) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        uint32_t y = MT[i % n];
        y ^= (y >> u) & d;
        y ^= (y << s) & b;
        y ^= (y << t) & c;
        y ^= (y >> l);
        output[i] = y;
    }
}

// Top function name: mersenne_twister

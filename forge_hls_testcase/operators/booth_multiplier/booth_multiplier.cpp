#include <ap_int.h>

#define N 1024

void booth_multiplier(ap_int<32> x, ap_int<32> y, ap_int<64> &product) {
    ap_int<64> A = x;
    ap_int<64> S = -x;
    ap_int<64> P = 0;
    P.range(63, 32) = y;
    
    for (int i = 0; i < 32; i++) {
        ap_int<2> last_two_bits = P.range(1, 0);
        if (last_two_bits == 1) {
            P.range(63, 32) = P.range(63, 32) + A;
        } else if (last_two_bits == 2) {
            P.range(63, 32) = P.range(63, 32) + S;
        }
        P = P >> 1;
    }
    
    product = P;
}

// Top function name: booth_multiplier

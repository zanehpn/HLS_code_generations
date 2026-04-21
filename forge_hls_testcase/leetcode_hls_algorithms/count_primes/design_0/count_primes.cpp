#include <ap_int.h>

#define N 1024

void count_primes(int limit, int &count) {
    bool is_prime[N];
    int i, j;

    // Initialize the is_prime array
    for (i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        is_prime[i] = true;
    }

    is_prime[0] = is_prime[1] = false; // 0 and 1 are not prime numbers

    // Sieve of Eratosthenes algorithm
    for (i = 2; i * i < limit; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
        if (is_prime[i]) {
            for (j = i * i; j < limit; j += i) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
                is_prime[j] = false;
            }
        }
    }

    // Count the number of prime numbers
    count = 0;
    for (i = 2; i < limit; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
        if (is_prime[i]) {
            count++;
        }
    }
}

// Top function name: count_primes

#include <ap_int.h>

#define CACHE_SIZE 1024
#define ADDR_SIZE 2048

void cache_replacement_lru_fifo(ap_uint<32> addresses[ADDR_SIZE], ap_uint<32> cache[CACHE_SIZE], bool is_lru) {
#pragma HLS ARRAY_PARTITION variable=cache type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=addresses type=cyclic dim=1 factor=8
    ap_uint<32> lru_tracker[CACHE_SIZE];
    ap_uint<32> fifo_tracker = 0;
    ap_uint<32> time = 0;

    for (int i = 0; i < ADDR_SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        ap_uint<32> addr = addresses[i];
        bool hit = false;

        for (int j = 0; j < CACHE_SIZE; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=16
            if (cache[j] == addr) {
                hit = true;
                if (is_lru) {
                    lru_tracker[j] = time;
                }
                break;
            }
        }

        if (!hit) {
            ap_uint<32> replace_index = 0;
            if (is_lru) {
                ap_uint<32> min_time = time;
                for (int j = 0; j < CACHE_SIZE; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
                    if (lru_tracker[j] < min_time) {
                        min_time = lru_tracker[j];
                        replace_index = j;
                    }
                }
            } else {
                replace_index = fifo_tracker;
                fifo_tracker = (fifo_tracker + 1) % CACHE_SIZE;
            }

            cache[replace_index] = addr;
            if (is_lru) {
                lru_tracker[replace_index] = time;
            }
        }

        time++;
    }
}

// Top function name: cache_replacement_lru_fifo

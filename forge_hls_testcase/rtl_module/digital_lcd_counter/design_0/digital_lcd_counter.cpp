#include <ap_int.h>

#define COUNTER_SIZE 1024

void digital_lcd_counter(ap_uint<8> input[COUNTER_SIZE], ap_uint<8> output[COUNTER_SIZE]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=4
    for (int i = 0; i < COUNTER_SIZE; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
        ap_uint<8> digit = input[i];
        ap_uint<8> lcd_representation = 0;

        switch (digit) {
            case 0: lcd_representation = 0b00111111; break; // 0
            case 1: lcd_representation = 0b00000110; break; // 1
            case 2: lcd_representation = 0b01011011; break; // 2
            case 3: lcd_representation = 0b01001111; break; // 3
            case 4: lcd_representation = 0b01100110; break; // 4
            case 5: lcd_representation = 0b01101101; break; // 5
            case 6: lcd_representation = 0b01111101; break; // 6
            case 7: lcd_representation = 0b00000111; break; // 7
            case 8: lcd_representation = 0b01111111; break; // 8
            case 9: lcd_representation = 0b01101111; break; // 9
            default: lcd_representation = 0b00000000; break; // Invalid input
        }

        output[i] = lcd_representation;
    }
}

// Top function name: digital_lcd_counter

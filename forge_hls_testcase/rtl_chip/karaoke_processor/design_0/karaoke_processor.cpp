#include <iostream>
#include <cmath>

#define SIZE 1024

void karaoke_processor(float input_audio[SIZE], float output_audio[SIZE], float vocal_gain, float music_gain) {
#pragma HLS ARRAY_PARTITION variable=output_audio type=cyclic dim=1 factor=8
#pragma HLS ARRAY_PARTITION variable=input_audio type=cyclic dim=1 factor=1
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
        float vocal = input_audio[i] * vocal_gain;
        float music = input_audio[i] * music_gain;
        output_audio[i] = vocal + music;
    }
}

// Top function name: karaoke_processor

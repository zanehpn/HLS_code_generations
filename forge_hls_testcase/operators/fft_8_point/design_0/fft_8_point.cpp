#include <cmath>
#include <complex>

#define N 1024

typedef std::complex<float> Complex;

void fft_8_point(Complex input[N], Complex output[N]) {
#pragma HLS ARRAY_PARTITION variable=output type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=input type=cyclic dim=1 factor=4
    const float PI = 3.14159265358979323846;
    Complex W[N];
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=8
        float theta = -2 * PI * i / N;
        W[i] = Complex(cos(theta), sin(theta));
    }

    for (int i = 0; i < N; i += 8) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        Complex temp[8];
        for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=4
            temp[j] = input[i + j];
        }

        Complex stage1[8];
        for (int j = 0; j < 4; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=8
            stage1[j] = temp[j] + temp[j + 4];
            stage1[j + 4] = (temp[j] - temp[j + 4]) * W[j * N / 8];
        }

        Complex stage2[8];
        for (int j = 0; j < 2; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=16
            stage2[j] = stage1[j] + stage1[j + 2];
            stage2[j + 2] = (stage1[j] - stage1[j + 2]) * W[j * N / 4];
            stage2[j + 4] = stage1[j + 4] + stage1[j + 6];
            stage2[j + 6] = (stage1[j + 4] - stage1[j + 6]) * W[j * N / 4];
        }

        for (int j = 0; j < 8; j += 4) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
            output[i + j] = stage2[j] + stage2[j + 1];
            output[i + j + 1] = (stage2[j] - stage2[j + 1]) * W[j * N / 8];
            output[i + j + 2] = stage2[j + 2] + stage2[j + 3];
            output[i + j + 3] = (stage2[j + 2] - stage2[j + 3]) * W[j * N / 8];
        }
    }
}

// Top function name: fft_8_point

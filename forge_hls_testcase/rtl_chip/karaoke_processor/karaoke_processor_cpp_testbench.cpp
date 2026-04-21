#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include "karaoke_processor.cpp"

namespace ref {
#define karaoke_processor ref_karaoke_processor
#include "karaoke_processor.cpp"
#undef karaoke_processor
}

template <typename T>
static bool close_scalar(const T& lhs, const T& rhs) {
    return lhs == rhs;
}

static bool close_scalar(const float& lhs, const float& rhs) {
    return std::fabs(lhs - rhs) <= 1e-4f;
}

static bool close_scalar(const double& lhs, const double& rhs) {
    return std::fabs(lhs - rhs) <= 1e-6;
}

template <typename T>
static bool close_scalar(const std::complex<T>& lhs, const std::complex<T>& rhs) {
    return std::fabs(lhs.real() - rhs.real()) <= static_cast<T>(1e-4) &&
           std::fabs(lhs.imag() - rhs.imag()) <= static_cast<T>(1e-4);
}

template <typename T>
static bool compare_value(const T& lhs, const T& rhs) {
    return close_scalar(lhs, rhs);
}

template <typename T, size_t Extent>
static bool compare_value(const T (&lhs)[Extent], const T (&rhs)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        if (!compare_value(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

static void fill_case(int case_id,
                      float input_audio[SIZE],
                      float output_audio[SIZE],
                      float& vocal_gain,
                      float& music_gain) {
    // case 0: zero baseline; case 1: small patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < SIZE; ++i0) {
        input_audio[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input_audio[i0] = static_cast<float>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input_audio[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < SIZE; ++i0) {
        output_audio[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            output_audio[i0] = static_cast<float>((((2 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            output_audio[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 2)) * (1.5 + 0.25 * case_id));
        }
    }

    vocal_gain = static_cast<float>(0);
    if (case_id == 1) {
        vocal_gain = static_cast<float>((((3 + case_id + 1) % 9) - 4) * 0.5);
    } else if (case_id == 2) {
        vocal_gain = static_cast<float>(1.0 + 0.5 * case_id);
    }

    music_gain = static_cast<float>(0);
    if (case_id == 1) {
        music_gain = static_cast<float>((((4 + case_id + 1) % 9) - 4) * 0.5);
    } else if (case_id == 2) {
        music_gain = static_cast<float>(1.0 + 0.5 * case_id);
    }
}

static bool run_case(int case_id) {
    static float dut_input_audio[SIZE] = {};
    static float ref_input_audio[SIZE] = {};
    static float dut_output_audio[SIZE] = {};
    static float ref_output_audio[SIZE] = {};
    static float dut_vocal_gain = {};
    static float ref_vocal_gain = {};
    static float dut_music_gain = {};
    static float ref_music_gain = {};

    fill_case(case_id,
              dut_input_audio,
              dut_output_audio,
              dut_vocal_gain,
              dut_music_gain);

    std::memcpy(ref_input_audio, dut_input_audio, sizeof(dut_input_audio));
    std::memcpy(ref_output_audio, dut_output_audio, sizeof(dut_output_audio));
    ref_vocal_gain = dut_vocal_gain;
    ref_music_gain = dut_music_gain;

    karaoke_processor(dut_input_audio, dut_output_audio, dut_vocal_gain, dut_music_gain);
    ref::ref_karaoke_processor(ref_input_audio, ref_output_audio, ref_vocal_gain, ref_music_gain);

    if (!compare_value(dut_input_audio, ref_input_audio)) return false;
    if (!compare_value(dut_output_audio, ref_output_audio)) return false;
    if (!compare_value(dut_vocal_gain, ref_vocal_gain)) return false;
    if (!compare_value(dut_music_gain, ref_music_gain)) return false;
    return true;
}

int main() {
    for (int case_id = 0; case_id < 3; ++case_id) {
        if (!run_case(case_id)) {
            std::printf("case %d failed\n", case_id);
            return 1;
        }
    }

    std::printf("pass\n");
    return 0;
}

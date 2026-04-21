#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include "video_processor.cpp"

namespace ref {
#define video_processor ref_video_processor
#include "video_processor.cpp"
#undef video_processor
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
                      uint8_t input[HEIGHT][WIDTH],
                      uint8_t output[HEIGHT][WIDTH]) {
    // case 0: zero baseline; case 1: small patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < HEIGHT; ++i0) {
        for (int i1 = 0; i1 < WIDTH; ++i1) {
            input[i0][i1] = static_cast<uint8_t>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                input[i0][i1] = static_cast<uint8_t>((1 + i0 + i1 * 7) % 251);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                input[i0][i1] = static_cast<uint8_t>((1 * 7 + i0 + i1 * 7 * 3 + case_id) % 251);
            }
        }
    }

    for (int i0 = 0; i0 < HEIGHT; ++i0) {
        for (int i1 = 0; i1 < WIDTH; ++i1) {
            output[i0][i1] = static_cast<uint8_t>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                output[i0][i1] = static_cast<uint8_t>((2 + i0 + i1 * 7) % 251);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                output[i0][i1] = static_cast<uint8_t>((2 * 7 + i0 + i1 * 7 * 3 + case_id) % 251);
            }
        }
    }
}

static bool run_case(int case_id) {
    static uint8_t dut_input[HEIGHT][WIDTH] = {};
    static uint8_t ref_input[HEIGHT][WIDTH] = {};
    static uint8_t dut_output[HEIGHT][WIDTH] = {};
    static uint8_t ref_output[HEIGHT][WIDTH] = {};

    fill_case(case_id,
              dut_input,
              dut_output);

    std::memcpy(ref_input, dut_input, sizeof(dut_input));
    std::memcpy(ref_output, dut_output, sizeof(dut_output));

    video_processor(dut_input, dut_output);
    ref::ref_video_processor(ref_input, ref_output);

    if (!compare_value(dut_input, ref_input)) return false;
    if (!compare_value(dut_output, ref_output)) return false;
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

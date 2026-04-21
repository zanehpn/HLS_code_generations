#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include "synchronous_processor.cpp"

namespace ref {
#define synchronous_processor ref_synchronous_processor
#include "synchronous_processor.cpp"
#undef synchronous_processor
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
                      ap_int<16> input1[SIZE],
                      ap_int<16> input2[SIZE],
                      ap_int<16> output[SIZE]) {
    // case 0: zero baseline; case 1: small patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < SIZE; ++i0) {
        input1[i0] = static_cast<ap_int<16>>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input1[i0] = static_cast<ap_int<16>>(((1 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input1[i0] = static_cast<ap_int<16>>(((1 * 3 + i0 * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < SIZE; ++i0) {
        input2[i0] = static_cast<ap_int<16>>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input2[i0] = static_cast<ap_int<16>>(((2 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            input2[i0] = static_cast<ap_int<16>>(((2 * 3 + i0 * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < SIZE; ++i0) {
        output[i0] = static_cast<ap_int<16>>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            output[i0] = static_cast<ap_int<16>>(((3 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < SIZE; ++i0) {
            output[i0] = static_cast<ap_int<16>>(((3 * 3 + i0 * 2 + case_id) % 17) - 8);
        }
    }
}

static bool run_case(int case_id) {
    static ap_int<16> dut_input1[SIZE] = {};
    static ap_int<16> ref_input1[SIZE] = {};
    static ap_int<16> dut_input2[SIZE] = {};
    static ap_int<16> ref_input2[SIZE] = {};
    static ap_int<16> dut_output[SIZE] = {};
    static ap_int<16> ref_output[SIZE] = {};

    fill_case(case_id,
              dut_input1,
              dut_input2,
              dut_output);

    std::memcpy(ref_input1, dut_input1, sizeof(dut_input1));
    std::memcpy(ref_input2, dut_input2, sizeof(dut_input2));
    std::memcpy(ref_output, dut_output, sizeof(dut_output));

    synchronous_processor(dut_input1, dut_input2, dut_output);
    ref::ref_synchronous_processor(ref_input1, ref_input2, ref_output);

    if (!compare_value(dut_input1, ref_input1)) return false;
    if (!compare_value(dut_input2, ref_input2)) return false;
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

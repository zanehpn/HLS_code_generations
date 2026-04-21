#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <cstddef>

#include "carradio_signal_processor.cpp"

namespace ref {
#define carradio_signal_processor ref_carradio_signal_processor
#include "carradio_signal_processor.cpp"
#undef carradio_signal_processor
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
                      float input_signal[N],
                      float output_signal[N],
                      float& gain,
                      float& offset) {
    // case 0: zero baseline; case 1: small patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < N; ++i0) {
        input_signal[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_signal[i0] = static_cast<float>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_signal[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        output_signal[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_signal[i0] = static_cast<float>((((2 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_signal[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 2)) * (1.5 + 0.25 * case_id));
        }
    }

    gain = static_cast<float>(0);
    if (case_id == 1) {
        gain = static_cast<float>((((3 + case_id + 1) % 9) - 4) * 0.5);
    } else if (case_id == 2) {
        gain = static_cast<float>(1.0 + 0.5 * case_id);
    }

    offset = static_cast<float>(0);
    if (case_id == 1) {
        offset = static_cast<float>((((4 + case_id + 1) % 9) - 4) * 0.5);
    } else if (case_id == 2) {
        offset = static_cast<float>(1.0 + 0.5 * case_id);
    }
}

static bool run_case(int case_id) {
    static float dut_input_signal[N] = {};
    static float ref_input_signal[N] = {};
    static float dut_output_signal[N] = {};
    static float ref_output_signal[N] = {};
    static float dut_gain = {};
    static float ref_gain = {};
    static float dut_offset = {};
    static float ref_offset = {};

    fill_case(case_id,
              dut_input_signal,
              dut_output_signal,
              dut_gain,
              dut_offset);

    std::memcpy(ref_input_signal, dut_input_signal, sizeof(dut_input_signal));
    std::memcpy(ref_output_signal, dut_output_signal, sizeof(dut_output_signal));
    ref_gain = dut_gain;
    ref_offset = dut_offset;

    carradio_signal_processor(dut_input_signal, dut_output_signal, dut_gain, dut_offset);
    ref::ref_carradio_signal_processor(ref_input_signal, ref_output_signal, ref_gain, ref_offset);

    if (!compare_value(dut_input_signal, ref_input_signal)) return false;
    if (!compare_value(dut_output_signal, ref_output_signal)) return false;
    if (!compare_value(dut_gain, ref_gain)) return false;
    if (!compare_value(dut_offset, ref_offset)) return false;
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

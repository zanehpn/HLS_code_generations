#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "echo_cancellation.cpp"

namespace ref {
#define echo_cancellation ref_echo_cancellation
#include "echo_cancellation.cpp"
#undef echo_cancellation
}

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename T, typename = void>
struct has_equal : std::false_type {};

template <typename T>
struct has_equal<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>> : std::true_type {};

template <typename T>
static bool compare_value(const T& lhs, const T& rhs) {
    if constexpr (std::is_floating_point_v<T>) {
        if (std::isnan(lhs) && std::isnan(rhs)) {
            return true;
        }
        if (std::isinf(lhs) || std::isinf(rhs)) {
            return lhs == rhs;
        }
        return std::fabs(lhs - rhs) <= static_cast<T>(1e-4);
    } else if constexpr (is_std_complex<T>::value) {
        using element_type = typename T::value_type;
        return std::fabs(lhs.real() - rhs.real()) <= static_cast<element_type>(1e-4) &&
               std::fabs(lhs.imag() - rhs.imag()) <= static_cast<element_type>(1e-4);
    } else if constexpr (has_equal<T>::value) {
        return lhs == rhs;
    } else if constexpr (std::is_trivially_copyable_v<T>) {
        return std::memcmp(&lhs, &rhs, sizeof(T)) == 0;
    } else {
        return false;
    }
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

template <typename L, typename R>
static bool compare_storage(const L& lhs, const R& rhs) {
    if constexpr (std::is_same_v<L, R>) {
        return compare_value(lhs, rhs);
    } else if constexpr (std::is_trivially_copyable_v<L> &&
                         std::is_trivially_copyable_v<R> &&
                         sizeof(L) == sizeof(R)) {
        return std::memcmp(&lhs, &rhs, sizeof(L)) == 0;
    } else {
        return false;
    }
}

template <typename L, typename R, size_t Extent>
static bool compare_storage(const L (&lhs)[Extent], const R (&rhs)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        if (!compare_storage(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

static void fill_case(int case_id,
                      float input_signal[N],
                      float echo_signal[N],
                      float output_signal[N],
                      float filter_coeff[N]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
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
        echo_signal[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            echo_signal[i0] = static_cast<float>((((2 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            echo_signal[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 2)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        output_signal[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_signal[i0] = static_cast<float>((((3 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_signal[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 3)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        filter_coeff[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            filter_coeff[i0] = static_cast<float>((((4 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            filter_coeff[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 4)) * (1.5 + 0.25 * case_id));
        }
    }
}

static bool run_case(int case_id) {
    static float dut_input_signal[N] = {};
    static float ref_input_signal[N] = {};
    static float dut_echo_signal[N] = {};
    static float ref_echo_signal[N] = {};
    static float dut_output_signal[N] = {};
    static float ref_output_signal[N] = {};
    static float dut_filter_coeff[N] = {};
    static float ref_filter_coeff[N] = {};

    fill_case(case_id,
              dut_input_signal,
              dut_echo_signal,
              dut_output_signal,
              dut_filter_coeff);

    std::memcpy(ref_input_signal, dut_input_signal, sizeof(dut_input_signal));
    std::memcpy(ref_echo_signal, dut_echo_signal, sizeof(dut_echo_signal));
    std::memcpy(ref_output_signal, dut_output_signal, sizeof(dut_output_signal));
    std::memcpy(ref_filter_coeff, dut_filter_coeff, sizeof(dut_filter_coeff));

    echo_cancellation(dut_input_signal, dut_echo_signal, dut_output_signal, dut_filter_coeff);
    ref::ref_echo_cancellation(ref_input_signal, ref_echo_signal, ref_output_signal, ref_filter_coeff);

    if (!compare_storage(dut_input_signal, ref_input_signal)) return false;
    if (!compare_storage(dut_echo_signal, ref_echo_signal)) return false;
    if (!compare_storage(dut_output_signal, ref_output_signal)) return false;
    if (!compare_storage(dut_filter_coeff, ref_filter_coeff)) return false;
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

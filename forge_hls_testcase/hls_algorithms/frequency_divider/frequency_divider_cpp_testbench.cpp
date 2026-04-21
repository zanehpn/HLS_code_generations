#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "frequency_divider.cpp"

namespace ref {
#define frequency_divider ref_frequency_divider
#include "frequency_divider.cpp"
#undef frequency_divider
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
                      int input_freq[ARRAY_SIZE],
                      int& divisor,
                      int output_freq[ARRAY_SIZE]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        input_freq[i0] = static_cast<int>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input_freq[i0] = static_cast<int>(8 + ((1 + i0) % 11));
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input_freq[i0] = static_cast<int>(12 + ((1 * 3 + (i0) * 2 + case_id) % 17));
        }
    }

    divisor = static_cast<int>(2);
    if (case_id == 1) {
        divisor = static_cast<int>(3 + (case_id));
    } else if (case_id == 2) {
        divisor = static_cast<int>(6 + (case_id + 1));
    }

    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        output_freq[i0] = static_cast<int>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            output_freq[i0] = static_cast<int>(((3 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            output_freq[i0] = static_cast<int>(((3 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }
}

static bool run_case(int case_id) {
    static int dut_input_freq[ARRAY_SIZE] = {};
    static int ref_input_freq[ARRAY_SIZE] = {};
    static int dut_divisor = {};
    static int ref_divisor = {};
    static int dut_output_freq[ARRAY_SIZE] = {};
    static int ref_output_freq[ARRAY_SIZE] = {};

    fill_case(case_id,
              dut_input_freq,
              dut_divisor,
              dut_output_freq);

    std::memcpy(ref_input_freq, dut_input_freq, sizeof(dut_input_freq));
    ref_divisor = dut_divisor;
    std::memcpy(ref_output_freq, dut_output_freq, sizeof(dut_output_freq));

    frequency_divider(dut_input_freq, dut_divisor, dut_output_freq);
    ref::ref_frequency_divider(ref_input_freq, ref_divisor, ref_output_freq);

    if (!compare_storage(dut_input_freq, ref_input_freq)) return false;
    if (!compare_storage(dut_divisor, ref_divisor)) return false;
    if (!compare_storage(dut_output_freq, ref_output_freq)) return false;
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

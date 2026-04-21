#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "hazard_detection_resolution.cpp"

namespace ref {
#define hazard_detection_resolution ref_hazard_detection_resolution
#include "hazard_detection_resolution.cpp"
#undef hazard_detection_resolution
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
                      int32_t input1[ARRAY_SIZE],
                      int32_t input2[ARRAY_SIZE],
                      int32_t output[ARRAY_SIZE]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        input1[i0] = static_cast<int32_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input1[i0] = static_cast<int32_t>(((1 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input1[i0] = static_cast<int32_t>(((1 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        input2[i0] = static_cast<int32_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input2[i0] = static_cast<int32_t>(((2 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            input2[i0] = static_cast<int32_t>(((2 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        output[i0] = static_cast<int32_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            output[i0] = static_cast<int32_t>(((3 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            output[i0] = static_cast<int32_t>(((3 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }
}

static bool run_case(int case_id) {
    static int32_t dut_input1[ARRAY_SIZE] = {};
    static int32_t ref_input1[ARRAY_SIZE] = {};
    static int32_t dut_input2[ARRAY_SIZE] = {};
    static int32_t ref_input2[ARRAY_SIZE] = {};
    static int32_t dut_output[ARRAY_SIZE] = {};
    static int32_t ref_output[ARRAY_SIZE] = {};

    fill_case(case_id,
              dut_input1,
              dut_input2,
              dut_output);

    std::memcpy(ref_input1, dut_input1, sizeof(dut_input1));
    std::memcpy(ref_input2, dut_input2, sizeof(dut_input2));
    std::memcpy(ref_output, dut_output, sizeof(dut_output));

    hazard_detection_resolution(dut_input1, dut_input2, dut_output);
    ref::ref_hazard_detection_resolution(ref_input1, ref_input2, ref_output);

    if (!compare_storage(dut_input1, ref_input1)) return false;
    if (!compare_storage(dut_input2, ref_input2)) return false;
    if (!compare_storage(dut_output, ref_output)) return false;
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

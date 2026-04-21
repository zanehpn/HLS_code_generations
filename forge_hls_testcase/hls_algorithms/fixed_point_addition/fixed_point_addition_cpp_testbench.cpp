#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "fixed_point_addition.cpp"

namespace ref {
#define fixed_point_addition ref_fixed_point_addition
#include "fixed_point_addition.cpp"
#undef fixed_point_addition
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
                      fixed_point_t A[ARRAY_SIZE],
                      fixed_point_t B[ARRAY_SIZE],
                      fixed_point_t C[ARRAY_SIZE]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        A[i0] = static_cast<fixed_point_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            A[i0] = static_cast<fixed_point_t>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            A[i0] = static_cast<fixed_point_t>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        B[i0] = static_cast<fixed_point_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            B[i0] = static_cast<fixed_point_t>((((2 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            B[i0] = static_cast<fixed_point_t>(std::sin(0.05 * static_cast<double>(i0 + 2)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        C[i0] = static_cast<fixed_point_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            C[i0] = static_cast<fixed_point_t>((((3 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            C[i0] = static_cast<fixed_point_t>(std::sin(0.05 * static_cast<double>(i0 + 3)) * (1.5 + 0.25 * case_id));
        }
    }
}

static bool run_case(int case_id) {
    static fixed_point_t dut_A[ARRAY_SIZE] = {};
    static fixed_point_t ref_A[ARRAY_SIZE] = {};
    static fixed_point_t dut_B[ARRAY_SIZE] = {};
    static fixed_point_t ref_B[ARRAY_SIZE] = {};
    static fixed_point_t dut_C[ARRAY_SIZE] = {};
    static fixed_point_t ref_C[ARRAY_SIZE] = {};

    fill_case(case_id,
              dut_A,
              dut_B,
              dut_C);

    std::memcpy(ref_A, dut_A, sizeof(dut_A));
    std::memcpy(ref_B, dut_B, sizeof(dut_B));
    std::memcpy(ref_C, dut_C, sizeof(dut_C));

    fixed_point_addition(dut_A, dut_B, dut_C);
    ref::ref_fixed_point_addition(ref_A, ref_B, ref_C);

    if (!compare_storage(dut_A, ref_A)) return false;
    if (!compare_storage(dut_B, ref_B)) return false;
    if (!compare_storage(dut_C, ref_C)) return false;
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

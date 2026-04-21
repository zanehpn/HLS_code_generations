#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "hough_transform.cpp"

namespace ref {
#define hough_transform ref_hough_transform
#include "hough_transform.cpp"
#undef hough_transform
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
                      ap_uint<8> image[HEIGHT][WIDTH],
                      ap_uint<32> accumulator[THETA_SIZE][RHO_SIZE]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < HEIGHT; ++i0) {
        for (int i1 = 0; i1 < WIDTH; ++i1) {
            image[i0][i1] = static_cast<ap_uint<8>>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                image[i0][i1] = static_cast<ap_uint<8>>((1 + i0 + i1 * 7) % 251);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < HEIGHT; ++i0) {
            for (int i1 = 0; i1 < WIDTH; ++i1) {
                image[i0][i1] = static_cast<ap_uint<8>>((1 * 7 + (i0 + i1 * 7) * 3 + case_id) % 251);
            }
        }
    }

    for (int i0 = 0; i0 < THETA_SIZE; ++i0) {
        for (int i1 = 0; i1 < RHO_SIZE; ++i1) {
            accumulator[i0][i1] = static_cast<ap_uint<32>>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < THETA_SIZE; ++i0) {
            for (int i1 = 0; i1 < RHO_SIZE; ++i1) {
                accumulator[i0][i1] = static_cast<ap_uint<32>>((2 + i0 + i1 * 7) % 251);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < THETA_SIZE; ++i0) {
            for (int i1 = 0; i1 < RHO_SIZE; ++i1) {
                accumulator[i0][i1] = static_cast<ap_uint<32>>((2 * 7 + (i0 + i1 * 7) * 3 + case_id) % 251);
            }
        }
    }
}

static bool run_case(int case_id) {
    static ap_uint<8> dut_image[HEIGHT][WIDTH] = {};
    static ap_uint<8> ref_image[HEIGHT][WIDTH] = {};
    static ap_uint<32> dut_accumulator[THETA_SIZE][RHO_SIZE] = {};
    static ap_uint<32> ref_accumulator[THETA_SIZE][RHO_SIZE] = {};

    fill_case(case_id,
              dut_image,
              dut_accumulator);

    std::memcpy(ref_image, dut_image, sizeof(dut_image));
    std::memcpy(ref_accumulator, dut_accumulator, sizeof(dut_accumulator));

    hough_transform(dut_image, dut_accumulator);
    ref::ref_hough_transform(ref_image, ref_accumulator);

    if (!compare_storage(dut_image, ref_image)) return false;
    if (!compare_storage(dut_accumulator, ref_accumulator)) return false;
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

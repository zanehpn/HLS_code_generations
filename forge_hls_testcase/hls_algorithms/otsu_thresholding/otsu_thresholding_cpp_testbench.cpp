#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "otsu_thresholding.cpp"

namespace ref {
#define otsu_thresholding ref_otsu_thresholding
#include "otsu_thresholding.cpp"
#undef otsu_thresholding
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
                      ap_uint<8> image[IMG_SIZE],
                      ap_uint<8> thresholded_image[IMG_SIZE]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
        image[i0] = static_cast<ap_uint<8>>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
            image[i0] = static_cast<ap_uint<8>>((1 + i0) % 251);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
            image[i0] = static_cast<ap_uint<8>>((1 * 7 + (i0) * 3 + case_id) % 251);
        }
    }

    for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
        thresholded_image[i0] = static_cast<ap_uint<8>>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
            thresholded_image[i0] = static_cast<ap_uint<8>>((2 + i0) % 251);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < IMG_SIZE; ++i0) {
            thresholded_image[i0] = static_cast<ap_uint<8>>((2 * 7 + (i0) * 3 + case_id) % 251);
        }
    }
}

static bool run_case(int case_id) {
    static ap_uint<8> dut_image[IMG_SIZE] = {};
    static ap_uint<8> ref_image[IMG_SIZE] = {};
    static ap_uint<8> dut_thresholded_image[IMG_SIZE] = {};
    static ap_uint<8> ref_thresholded_image[IMG_SIZE] = {};

    fill_case(case_id,
              dut_image,
              dut_thresholded_image);

    std::memcpy(ref_image, dut_image, sizeof(dut_image));
    std::memcpy(ref_thresholded_image, dut_thresholded_image, sizeof(dut_thresholded_image));

    otsu_thresholding(dut_image, dut_thresholded_image);
    ref::ref_otsu_thresholding(ref_image, ref_thresholded_image);

    if (!compare_storage(dut_image, ref_image)) return false;
    if (!compare_storage(dut_thresholded_image, ref_thresholded_image)) return false;
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

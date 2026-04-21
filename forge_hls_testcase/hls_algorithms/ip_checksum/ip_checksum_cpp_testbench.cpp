#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "ip_checksum.cpp"

namespace ref {
#define ip_checksum ref_ip_checksum
#include "ip_checksum.cpp"
#undef ip_checksum
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
                      uint16_t data[ARRAY_SIZE],
                      uint16_t& checksum) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
        data[i0] = static_cast<uint16_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            data[i0] = static_cast<uint16_t>((1 + i0) % 251);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < ARRAY_SIZE; ++i0) {
            data[i0] = static_cast<uint16_t>((1 * 7 + (i0) * 3 + case_id) % 251);
        }
    }

    checksum = static_cast<uint16_t>(0);
    if (case_id == 1) {
        checksum = static_cast<uint16_t>(6 + (case_id));
    } else if (case_id == 2) {
        checksum = static_cast<uint16_t>(15 + (case_id + 1));
    }
}

static bool run_case(int case_id) {
    static uint16_t dut_data[ARRAY_SIZE] = {};
    static uint16_t ref_data[ARRAY_SIZE] = {};
    static uint16_t dut_checksum = {};
    static uint16_t ref_checksum = {};

    fill_case(case_id,
              dut_data,
              dut_checksum);

    std::memcpy(ref_data, dut_data, sizeof(dut_data));
    ref_checksum = dut_checksum;

    ip_checksum(dut_data, dut_checksum);
    ref::ref_ip_checksum(ref_data, ref_checksum);

    if (!compare_storage(dut_data, ref_data)) return false;
    if (!compare_storage(dut_checksum, ref_checksum)) return false;
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

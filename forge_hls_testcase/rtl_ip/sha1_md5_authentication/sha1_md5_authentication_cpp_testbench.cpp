#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <type_traits>

#include "sha1_md5_authentication.cpp"

namespace ref {
void sha1_md5_authentication(uint8_t input[DATA_SIZE], uint8_t output[20]) {
    uint32_t h0 = 0x67452301;
    uint32_t h1 = 0xEFCDAB89;
    uint32_t h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476;
    uint32_t h4 = 0xC3D2E1F0;

    uint32_t w[80];
    uint32_t a, b, c, d, e, f, k, temp;

    for (int i = 0; i < DATA_SIZE; i += 64) {
        for (int j = 0; j < 16; j++) {
            w[j] = (input[i + 4 * j] << 24) | (input[i + 4 * j + 1] << 16) | (input[i + 4 * j + 2] << 8) | (input[i + 4 * j + 3]);
        }

        for (int j = 16; j < 80; j++) {
            w[j] = (w[j - 3] ^ w[j - 8] ^ w[j - 14] ^ w[j - 16]);
            w[j] = (w[j] << 1) | (w[j] >> 31);
        }

        a = h0;
        b = h1;
        c = h2;
        d = h3;
        e = h4;

        for (int j = 0; j < 80; j++) {
            if (j < 20) {
                f = (b & c) | ((~b) & d);
                k = 0x5A827999;
            } else if (j < 40) {
                f = b ^ c ^ d;
                k = 0x6ED9EBA1;
            } else if (j < 60) {
                f = (b & c) | (b & d) | (c & d);
                k = 0x8F1BBCDC;
            } else {
                f = b ^ c ^ d;
                k = 0xCA62C1D6;
            }

            temp = ((a << 5) | (a >> 27)) + f + e + k + w[j];
            e = d;
            d = c;
            c = (b << 30) | (b >> 2);
            b = a;
            a = temp;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
    }

    output[0] = (h0 >> 24) & 0xFF;
    output[1] = (h0 >> 16) & 0xFF;
    output[2] = (h0 >> 8) & 0xFF;
    output[3] = h0 & 0xFF;
    output[4] = (h1 >> 24) & 0xFF;
    output[5] = (h1 >> 16) & 0xFF;
    output[6] = (h1 >> 8) & 0xFF;
    output[7] = h1 & 0xFF;
    output[8] = (h2 >> 24) & 0xFF;
    output[9] = (h2 >> 16) & 0xFF;
    output[10] = (h2 >> 8) & 0xFF;
    output[11] = h2 & 0xFF;
    output[12] = (h3 >> 24) & 0xFF;
    output[13] = (h3 >> 16) & 0xFF;
    output[14] = (h3 >> 8) & 0xFF;
    output[15] = h3 & 0xFF;
    output[16] = (h4 >> 24) & 0xFF;
    output[17] = (h4 >> 16) & 0xFF;
    output[18] = (h4 >> 8) & 0xFF;
    output[19] = h4 & 0xFF;
}
}

template <typename T>
struct is_std_complex : std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : std::true_type {};

template <typename T, typename = void>
struct has_equal : std::false_type {};

template <typename T>
struct has_equal<T, std::void_t<decltype(std::declval<const T&>() == std::declval<const T&>())>> : std::true_type {};

template <typename T, typename = void>
struct can_static_cast_from_int : std::false_type {};

template <typename T>
struct can_static_cast_from_int<T, std::void_t<decltype(static_cast<T>(1))>> : std::true_type {};

template <typename T>
void zero_value(T& value) {
    value = T{};
}

template <typename T, size_t Extent>
void zero_value(T (&value)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        zero_value(value[i]);
    }
}

template <typename T>
void copy_value(T& dst, const T& src) {
    dst = src;
}

template <typename T, size_t Extent>
void copy_value(T (&dst)[Extent], const T (&src)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        copy_value(dst[i], src[i]);
    }
}

template <typename T>
void seed_scalar(T& value, int case_id, int salt) {
    const int base = ((case_id + 1) * (salt + 2)) % 7;
    if constexpr (std::is_same_v<T, bool>) {
        value = (base & 1) != 0;
    } else if constexpr (is_std_complex<T>::value) {
        using element_type = typename T::value_type;
        value = T(static_cast<element_type>(base), static_cast<element_type>(base - 2));
    } else if constexpr (std::is_floating_point_v<T>) {
        value = static_cast<T>(0.25 * (base + 1));
    } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
        value = static_cast<T>(base);
    } else if constexpr (can_static_cast_from_int<T>::value) {
        value = static_cast<T>(base);
    } else {
        zero_value(value);
    }
}

template <typename T>
void seed_value(T& value, int case_id, int salt) {
    zero_value(value);
    seed_scalar(value, case_id, salt);
}

template <typename T, size_t Extent>
void seed_value(T (&value)[Extent], int case_id, int salt) {
    zero_value(value);
    const size_t active = Extent < static_cast<size_t>(case_id + 2) ? Extent : static_cast<size_t>(case_id + 2);
    for (size_t i = 0; i < active; ++i) {
        seed_value(value[i], case_id, salt + static_cast<int>(i) + 1);
    }
}

template <typename T>
bool compare_value(const T& lhs, const T& rhs) {
    if constexpr (std::is_floating_point_v<T>) {
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
bool compare_value(const T (&lhs)[Extent], const T (&rhs)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        if (!compare_value(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

template <typename L, typename R>
bool compare_storage(const L& lhs, const R& rhs) {
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
bool compare_storage(const L (&lhs)[Extent], const R (&rhs)[Extent]) {
    for (size_t i = 0; i < Extent; ++i) {
        if (!compare_storage(lhs[i], rhs[i])) {
            return false;
        }
    }
    return true;
}

static void set_stack_limit() {
    rlimit rl;
    rl.rlim_cur = RLIM_INFINITY;
    rl.rlim_max = RLIM_INFINITY;
    setrlimit(RLIMIT_STACK, &rl);
}

static int run_case(int case_id) {
static uint8_t dut_input[DATA_SIZE] = {};
static uint8_t ref_input[DATA_SIZE] = {};
static uint8_t dut_output[20] = {};
static uint8_t ref_output[20] = {};
    seed_value(dut_input, case_id, 1);
    seed_value(dut_output, case_id, 2);
    copy_value(ref_input, dut_input);
    copy_value(ref_output, dut_output);

    sha1_md5_authentication(dut_input, dut_output);
    ref::sha1_md5_authentication(ref_input, ref_output);

    if (!compare_storage(dut_input, ref_input)) return 0;
    if (!compare_storage(dut_output, ref_output)) return 0;
    return 1;
}

int main() {
    set_stack_limit();

    for (int case_id = 0; case_id < 3; ++case_id) {
        if (!run_case(case_id)) {
            std::printf("fail\n");
            return 1;
        }
    }

    std::printf("pass\n");
    return 0;
}

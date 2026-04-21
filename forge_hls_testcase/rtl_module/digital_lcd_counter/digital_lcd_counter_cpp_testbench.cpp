#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <type_traits>

#include "digital_lcd_counter.cpp"

namespace ref {
void digital_lcd_counter(ap_uint<8> input[COUNTER_SIZE], ap_uint<8> output[COUNTER_SIZE]) {
    for (int i = 0; i < COUNTER_SIZE; i++) {
        ap_uint<8> digit = input[i];
        ap_uint<8> lcd_representation = 0;

        switch (digit) {
            case 0: lcd_representation = 0b00111111; break; // 0
            case 1: lcd_representation = 0b00000110; break; // 1
            case 2: lcd_representation = 0b01011011; break; // 2
            case 3: lcd_representation = 0b01001111; break; // 3
            case 4: lcd_representation = 0b01100110; break; // 4
            case 5: lcd_representation = 0b01101101; break; // 5
            case 6: lcd_representation = 0b01111101; break; // 6
            case 7: lcd_representation = 0b00000111; break; // 7
            case 8: lcd_representation = 0b01111111; break; // 8
            case 9: lcd_representation = 0b01101111; break; // 9
            default: lcd_representation = 0b00000000; break; // Invalid input
        }

        output[i] = lcd_representation;
    }
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
static ap_uint<8> dut_input[COUNTER_SIZE] = {};
static ap_uint<8> ref_input[COUNTER_SIZE] = {};
static ap_uint<8> dut_output[COUNTER_SIZE] = {};
static ap_uint<8> ref_output[COUNTER_SIZE] = {};
    seed_value(dut_input, case_id, 1);
    seed_value(dut_output, case_id, 2);
    copy_value(ref_input, dut_input);
    copy_value(ref_output, dut_output);

    digital_lcd_counter(dut_input, dut_output);
    ref::digital_lcd_counter(ref_input, ref_output);

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

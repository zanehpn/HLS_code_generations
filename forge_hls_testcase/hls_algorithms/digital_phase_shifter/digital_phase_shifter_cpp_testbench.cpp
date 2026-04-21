#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "digital_phase_shifter.cpp"

namespace ref {
#define digital_phase_shifter ref_digital_phase_shifter
#include "digital_phase_shifter.cpp"
#undef digital_phase_shifter
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
                      int16_t input_real[N],
                      int16_t input_imag[N],
                      int16_t output_real[N],
                      int16_t output_imag[N],
                      float& phase_shift) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < N; ++i0) {
        input_real[i0] = static_cast<int16_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_real[i0] = static_cast<int16_t>(((1 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_real[i0] = static_cast<int16_t>(((1 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        input_imag[i0] = static_cast<int16_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_imag[i0] = static_cast<int16_t>(((2 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            input_imag[i0] = static_cast<int16_t>(((2 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        output_real[i0] = static_cast<int16_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_real[i0] = static_cast<int16_t>(((3 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_real[i0] = static_cast<int16_t>(((3 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        output_imag[i0] = static_cast<int16_t>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_imag[i0] = static_cast<int16_t>(((4 + i0) % 11) - 5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            output_imag[i0] = static_cast<int16_t>(((4 * 3 + (i0) * 2 + case_id) % 17) - 8);
        }
    }

    phase_shift = static_cast<float>(0);
    if (case_id == 1) {
        phase_shift = static_cast<float>((5 + (case_id)) * 0.5);
    } else if (case_id == 2) {
        phase_shift = static_cast<float>((8 + (case_id + 1)) * 0.5);
    }
}

static bool run_case(int case_id) {
    static int16_t dut_input_real[N] = {};
    static int16_t ref_input_real[N] = {};
    static int16_t dut_input_imag[N] = {};
    static int16_t ref_input_imag[N] = {};
    static int16_t dut_output_real[N] = {};
    static int16_t ref_output_real[N] = {};
    static int16_t dut_output_imag[N] = {};
    static int16_t ref_output_imag[N] = {};
    static float dut_phase_shift = {};
    static float ref_phase_shift = {};

    fill_case(case_id,
              dut_input_real,
              dut_input_imag,
              dut_output_real,
              dut_output_imag,
              dut_phase_shift);

    std::memcpy(ref_input_real, dut_input_real, sizeof(dut_input_real));
    std::memcpy(ref_input_imag, dut_input_imag, sizeof(dut_input_imag));
    std::memcpy(ref_output_real, dut_output_real, sizeof(dut_output_real));
    std::memcpy(ref_output_imag, dut_output_imag, sizeof(dut_output_imag));
    ref_phase_shift = dut_phase_shift;

    digital_phase_shifter(dut_input_real, dut_input_imag, dut_output_real, dut_output_imag, dut_phase_shift);
    ref::ref_digital_phase_shifter(ref_input_real, ref_input_imag, ref_output_real, ref_output_imag, ref_phase_shift);

    if (!compare_storage(dut_input_real, ref_input_real)) return false;
    if (!compare_storage(dut_input_imag, ref_input_imag)) return false;
    if (!compare_storage(dut_output_real, ref_output_real)) return false;
    if (!compare_storage(dut_output_imag, ref_output_imag)) return false;
    if (!compare_storage(dut_phase_shift, ref_phase_shift)) return false;
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

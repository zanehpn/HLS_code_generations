#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "backpropagation.cpp"

namespace ref {
#define backpropagation ref_backpropagation
#include "backpropagation.cpp"
#undef backpropagation
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
                      float input[INPUT_SIZE],
                      float hidden_weights[INPUT_SIZE][HIDDEN_SIZE],
                      float output_weights[HIDDEN_SIZE][OUTPUT_SIZE],
                      float hidden_bias[HIDDEN_SIZE],
                      float output_bias[OUTPUT_SIZE],
                      float target[OUTPUT_SIZE],
                      float& learning_rate) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
        input[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
            input[i0] = static_cast<float>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
            input[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
        for (int i1 = 0; i1 < HIDDEN_SIZE; ++i1) {
            hidden_weights[i0][i1] = static_cast<float>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
            for (int i1 = 0; i1 < HIDDEN_SIZE; ++i1) {
                hidden_weights[i0][i1] = static_cast<float>((((2 + i0 + i1 * 7) % 9) - 4) * 0.5);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < INPUT_SIZE; ++i0) {
            for (int i1 = 0; i1 < HIDDEN_SIZE; ++i1) {
                hidden_weights[i0][i1] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + i1 * 7 + 2)) * (1.5 + 0.25 * case_id));
            }
        }
    }

    for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
        for (int i1 = 0; i1 < OUTPUT_SIZE; ++i1) {
            output_weights[i0][i1] = static_cast<float>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
            for (int i1 = 0; i1 < OUTPUT_SIZE; ++i1) {
                output_weights[i0][i1] = static_cast<float>((((3 + i0 + i1 * 7) % 9) - 4) * 0.5);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
            for (int i1 = 0; i1 < OUTPUT_SIZE; ++i1) {
                output_weights[i0][i1] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + i1 * 7 + 3)) * (1.5 + 0.25 * case_id));
            }
        }
    }

    for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
        hidden_bias[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
            hidden_bias[i0] = static_cast<float>((((4 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < HIDDEN_SIZE; ++i0) {
            hidden_bias[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 4)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
        output_bias[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
            output_bias[i0] = static_cast<float>((((5 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
            output_bias[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 5)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
        target[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
            target[i0] = static_cast<float>((((6 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < OUTPUT_SIZE; ++i0) {
            target[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 6)) * (1.5 + 0.25 * case_id));
        }
    }

    learning_rate = static_cast<float>(0);
    if (case_id == 1) {
        learning_rate = static_cast<float>(2.25 + 0.25 * (case_id));
    } else if (case_id == 2) {
        learning_rate = static_cast<float>(3.0 + 0.25 * (case_id + 1));
    }
}

static bool run_case(int case_id) {
    static float dut_input[INPUT_SIZE] = {};
    static float ref_input[INPUT_SIZE] = {};
    static float dut_hidden_weights[INPUT_SIZE][HIDDEN_SIZE] = {};
    static float ref_hidden_weights[INPUT_SIZE][HIDDEN_SIZE] = {};
    static float dut_output_weights[HIDDEN_SIZE][OUTPUT_SIZE] = {};
    static float ref_output_weights[HIDDEN_SIZE][OUTPUT_SIZE] = {};
    static float dut_hidden_bias[HIDDEN_SIZE] = {};
    static float ref_hidden_bias[HIDDEN_SIZE] = {};
    static float dut_output_bias[OUTPUT_SIZE] = {};
    static float ref_output_bias[OUTPUT_SIZE] = {};
    static float dut_target[OUTPUT_SIZE] = {};
    static float ref_target[OUTPUT_SIZE] = {};
    static float dut_learning_rate = {};
    static float ref_learning_rate = {};

    fill_case(case_id,
              dut_input,
              dut_hidden_weights,
              dut_output_weights,
              dut_hidden_bias,
              dut_output_bias,
              dut_target,
              dut_learning_rate);

    std::memcpy(ref_input, dut_input, sizeof(dut_input));
    std::memcpy(ref_hidden_weights, dut_hidden_weights, sizeof(dut_hidden_weights));
    std::memcpy(ref_output_weights, dut_output_weights, sizeof(dut_output_weights));
    std::memcpy(ref_hidden_bias, dut_hidden_bias, sizeof(dut_hidden_bias));
    std::memcpy(ref_output_bias, dut_output_bias, sizeof(dut_output_bias));
    std::memcpy(ref_target, dut_target, sizeof(dut_target));
    ref_learning_rate = dut_learning_rate;

    backpropagation(dut_input, dut_hidden_weights, dut_output_weights, dut_hidden_bias, dut_output_bias, dut_target, dut_learning_rate);
    ref::ref_backpropagation(ref_input, ref_hidden_weights, ref_output_weights, ref_hidden_bias, ref_output_bias, ref_target, ref_learning_rate);

    if (!compare_storage(dut_input, ref_input)) return false;
    if (!compare_storage(dut_hidden_weights, ref_hidden_weights)) return false;
    if (!compare_storage(dut_output_weights, ref_output_weights)) return false;
    if (!compare_storage(dut_hidden_bias, ref_hidden_bias)) return false;
    if (!compare_storage(dut_output_bias, ref_output_bias)) return false;
    if (!compare_storage(dut_target, ref_target)) return false;
    if (!compare_storage(dut_learning_rate, ref_learning_rate)) return false;
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

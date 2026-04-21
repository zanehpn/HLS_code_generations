#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "adaptive_lms_filter.cpp"

namespace ref {
#define adaptive_lms_filter ref_adaptive_lms_filter
#include "adaptive_lms_filter.cpp"
#undef adaptive_lms_filter
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
                      float x[N],
                      float d[N],
                      float y[N],
                      float e[N],
                      float w[N],
                      float& mu,
                      int& filter_length) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < N; ++i0) {
        x[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            x[i0] = static_cast<float>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            x[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        d[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            d[i0] = static_cast<float>((((2 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            d[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 2)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        y[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            y[i0] = static_cast<float>((((3 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            y[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 3)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        e[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            e[i0] = static_cast<float>((((4 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            e[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 4)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        w[i0] = static_cast<float>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            w[i0] = static_cast<float>((((5 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            w[i0] = static_cast<float>(std::sin(0.05 * static_cast<double>(i0 + 5)) * (1.5 + 0.25 * case_id));
        }
    }

    mu = static_cast<float>(0);
    if (case_id == 1) {
        mu = static_cast<float>((6 + (case_id)) * 0.5);
    } else if (case_id == 2) {
        mu = static_cast<float>((9 + (case_id + 1)) * 0.5);
    }

    filter_length = static_cast<int>(0);
    if (case_id == 1) {
        filter_length = static_cast<int>(7 + (case_id));
    } else if (case_id == 2) {
        filter_length = static_cast<int>(10 + (case_id + 1));
    }
}

static bool run_case(int case_id) {
    static float dut_x[N] = {};
    static float ref_x[N] = {};
    static float dut_d[N] = {};
    static float ref_d[N] = {};
    static float dut_y[N] = {};
    static float ref_y[N] = {};
    static float dut_e[N] = {};
    static float ref_e[N] = {};
    static float dut_w[N] = {};
    static float ref_w[N] = {};
    static float dut_mu = {};
    static float ref_mu = {};
    static int dut_filter_length = {};
    static int ref_filter_length = {};

    fill_case(case_id,
              dut_x,
              dut_d,
              dut_y,
              dut_e,
              dut_w,
              dut_mu,
              dut_filter_length);

    std::memcpy(ref_x, dut_x, sizeof(dut_x));
    std::memcpy(ref_d, dut_d, sizeof(dut_d));
    std::memcpy(ref_y, dut_y, sizeof(dut_y));
    std::memcpy(ref_e, dut_e, sizeof(dut_e));
    std::memcpy(ref_w, dut_w, sizeof(dut_w));
    ref_mu = dut_mu;
    ref_filter_length = dut_filter_length;

    adaptive_lms_filter(dut_x, dut_d, dut_y, dut_e, dut_w, dut_mu, dut_filter_length);
    ref::ref_adaptive_lms_filter(ref_x, ref_d, ref_y, ref_e, ref_w, ref_mu, ref_filter_length);

    if (!compare_storage(dut_x, ref_x)) return false;
    if (!compare_storage(dut_d, ref_d)) return false;
    if (!compare_storage(dut_y, ref_y)) return false;
    if (!compare_storage(dut_e, ref_e)) return false;
    if (!compare_storage(dut_w, ref_w)) return false;
    if (!compare_storage(dut_mu, ref_mu)) return false;
    if (!compare_storage(dut_filter_length, ref_filter_length)) return false;
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

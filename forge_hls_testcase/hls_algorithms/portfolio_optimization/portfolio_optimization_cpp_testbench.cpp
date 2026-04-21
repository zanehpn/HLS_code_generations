#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <type_traits>

#include "portfolio_optimization.cpp"

namespace ref {
#define portfolio_optimization ref_portfolio_optimization
#include "portfolio_optimization.cpp"
#undef portfolio_optimization
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
                      double returns[N],
                      double cov_matrix[N][N],
                      double weights[N],
                      double& risk_free_rate,
                      double& target_return,
                      double optimized_weights[N]) {
// case 0: zero baseline; case 1: discrete patterned values; case 2: smoothly varying values
    for (int i0 = 0; i0 < N; ++i0) {
        returns[i0] = static_cast<double>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            returns[i0] = static_cast<double>((((1 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            returns[i0] = static_cast<double>(std::sin(0.05 * static_cast<double>(i0 + 1)) * (1.5 + 0.25 * case_id));
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        for (int i1 = 0; i1 < N; ++i1) {
            cov_matrix[i0][i1] = static_cast<double>(0);
        }
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            for (int i1 = 0; i1 < N; ++i1) {
                cov_matrix[i0][i1] = static_cast<double>((((2 + i0 + i1 * 7) % 9) - 4) * 0.5);
            }
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            for (int i1 = 0; i1 < N; ++i1) {
                cov_matrix[i0][i1] = static_cast<double>(std::sin(0.05 * static_cast<double>(i0 + i1 * 7 + 2)) * (1.5 + 0.25 * case_id));
            }
        }
    }

    for (int i0 = 0; i0 < N; ++i0) {
        weights[i0] = static_cast<double>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            weights[i0] = static_cast<double>((((3 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            weights[i0] = static_cast<double>(std::sin(0.05 * static_cast<double>(i0 + 3)) * (1.5 + 0.25 * case_id));
        }
    }

    risk_free_rate = static_cast<double>(0);
    if (case_id == 1) {
        risk_free_rate = static_cast<double>((4 + (case_id)) * 0.5);
    } else if (case_id == 2) {
        risk_free_rate = static_cast<double>((7 + (case_id + 1)) * 0.5);
    }

    target_return = static_cast<double>(0);
    if (case_id == 1) {
        target_return = static_cast<double>((5 + (case_id)) * 0.5);
    } else if (case_id == 2) {
        target_return = static_cast<double>((8 + (case_id + 1)) * 0.5);
    }

    for (int i0 = 0; i0 < N; ++i0) {
        optimized_weights[i0] = static_cast<double>(0);
    }
    if (case_id == 1) {
        for (int i0 = 0; i0 < N; ++i0) {
            optimized_weights[i0] = static_cast<double>((((6 + i0) % 9) - 4) * 0.5);
        }
    } else if (case_id == 2) {
        for (int i0 = 0; i0 < N; ++i0) {
            optimized_weights[i0] = static_cast<double>(std::sin(0.05 * static_cast<double>(i0 + 6)) * (1.5 + 0.25 * case_id));
        }
    }
}

static bool run_case(int case_id) {
    static double dut_returns[N] = {};
    static double ref_returns[N] = {};
    static double dut_cov_matrix[N][N] = {};
    static double ref_cov_matrix[N][N] = {};
    static double dut_weights[N] = {};
    static double ref_weights[N] = {};
    static double dut_risk_free_rate = {};
    static double ref_risk_free_rate = {};
    static double dut_target_return = {};
    static double ref_target_return = {};
    static double dut_optimized_weights[N] = {};
    static double ref_optimized_weights[N] = {};

    fill_case(case_id,
              dut_returns,
              dut_cov_matrix,
              dut_weights,
              dut_risk_free_rate,
              dut_target_return,
              dut_optimized_weights);

    std::memcpy(ref_returns, dut_returns, sizeof(dut_returns));
    std::memcpy(ref_cov_matrix, dut_cov_matrix, sizeof(dut_cov_matrix));
    std::memcpy(ref_weights, dut_weights, sizeof(dut_weights));
    ref_risk_free_rate = dut_risk_free_rate;
    ref_target_return = dut_target_return;
    std::memcpy(ref_optimized_weights, dut_optimized_weights, sizeof(dut_optimized_weights));

    portfolio_optimization(dut_returns, dut_cov_matrix, dut_weights, dut_risk_free_rate, dut_target_return, dut_optimized_weights);
    ref::ref_portfolio_optimization(ref_returns, ref_cov_matrix, ref_weights, ref_risk_free_rate, ref_target_return, ref_optimized_weights);

    if (!compare_storage(dut_returns, ref_returns)) return false;
    if (!compare_storage(dut_cov_matrix, ref_cov_matrix)) return false;
    if (!compare_storage(dut_weights, ref_weights)) return false;
    if (!compare_storage(dut_risk_free_rate, ref_risk_free_rate)) return false;
    if (!compare_storage(dut_target_return, ref_target_return)) return false;
    if (!compare_storage(dut_optimized_weights, ref_optimized_weights)) return false;
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

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <type_traits>

#include "combined.cpp"

namespace ref {
/*
 * Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
 * Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
 * Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef FXP_SQRT_TOP_H_

/*
 * Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
 * Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __FXP_SQRT_H__

using namespace std;

// Fixed point square-root template
//
// Basic usage: fxp_sqrt(root_var, radicand_var);
//          or: fxp_sqrt<W2,IW2,W1,IW1>(root_var, radicand_var);
// where root_var and radicand_var are ap_ufixed<> variables and W2 & IW2
// are the total and iteger widths for root_var and W1/IW1 those for
// radicand_var.
//
// Despcription:
// This template function implements a non-restoring algorithm to calculate
// the square-root of a fixed point variable (ap_ufixed<>) of arbitrary
// format and assign it to another fixed point vaiable of arbitrary format.
//
// The template parameters specify the ap_ufixed<> width parameters to be used
// for input and output formats (see documentation for ap_fixed<> for details).
// These parameters do not need to be explicitly specified, as they will be
// inferred by the types passed to the function call.
//
// As this function is only intended for use with real numbers, the input
// and output variables are expected to be unsigned, i.e. ap_ufixed<>.  If
// signed types are passed, it is the user's responsibility to ensure the
// input variable does not go negative, as no checks are made for such;
// also note that passing a signed variable may result in slighty sub-optimal
// HLS due to a wider format than necessary.  Explicit casting is preferred.
//
// This function returns a precise (rounded to one ulp) result
// as long as the output format has at least half (rounded up) as many
// fractional bits as the input format.  The output format must have enough
// whole (integer) bit to hold thd full range of the result (i.e. half
// rounded up), otherwise an assertion failure will result at runtime (of SW
// model and/or RTL co-simulation).
//
// Note that integer square-roots may be accomplished with this function by
// use of ap_ufixed<> intermediate variables with no fractional portion, e.g.
//    uint16_t a;
//    uint8_t a_sqrt;
//    ...
//    ap_ufixed<16,16> a_fxp = ap_ufixed<16,16>(a);
//    ap_ufixed<8,8> a_sqrt_fxp;
//    fxp_sqrt(a_sqrt_fxp, a_fxp);
//    a_sqrt = a_sqrt_fxp.toint();
//
// Which should result in no more hardware area than having an integer-
// specific version of this function.

template <int W2, int IW2, int W1, int IW1>
void fxp_sqrt(ap_ufixed<W2, IW2>& result, ap_ufixed<W1, IW1>& in_val) {
    enum { QW = (IW1 + 1) / 2 + (W2 - IW2) + 1 }; // derive max root width
    enum {
        SCALE = (W2 - W1) - (IW2 - (IW1 + 1) / 2)
    }; // scale (shift) to adj initial remainer value
    enum { ROOT_PREC = QW - (IW1 % 2) };
    assert((IW1 + 1) / 2 <=
           IW2); // Check that output format can accommodate full result

    ap_uint<QW> q = 0;      // partial sqrt
    ap_uint<QW> q_star = 0; // diminished partial sqrt
    ap_int<QW + 2> s; // scaled remainder initialized to extracted input bits
    if (SCALE >= 0)
        s = in_val.range(W1 - 1, 0) << (SCALE);
    else
        s = ((in_val.range(W1 - 1, 0) >> (0 - (SCALE + 1))) + 1) >> 1;

    // Non-restoring square-root algorithm
    for (int i = 0; i <= ROOT_PREC; i++) {
        if (s >= 0) {
            s = 2 * s - (((ap_int<QW + 2>(q) << 2) | 1) << (ROOT_PREC - i));
            q_star = q << 1;
            q = (q << 1) | 1;
        } else {
            s = 2 * s +
                (((ap_int<QW + 2>(q_star) << 2) | 3) << (ROOT_PREC - i));
            q = (q_star << 1) | 1;
            q_star <<= 1;
        }
    }
    // Round result by "extra iteration" method
    if (s > 0)
        q = q + 1;
    // Truncate excess bit and assign to output format
    result.range(W2 - 1, 0) = ap_uint<W2>(q >> 1);
}

#endif //__FXP_SQRT_H__ not defined


// typedefs for top-level input and output fixed-point formats
typedef ap_ufixed<IN_BW, IN_IW> in_data_t;
typedef ap_ufixed<OUT_BW, OUT_IW> out_data_t;

// Top level wrapper function - calls the core template function w/ above types
out_data_t fxp_sqrt_top(in_data_t& in_val);

#endif // FXP_SQRT_TOP_H_ not defined

out_data_t fxp_sqrt_top(in_data_t& in_val) {
    out_data_t result;
    fxp_sqrt(result, in_val);
    return result;
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
static in_data_t dut_in_val = {};
static in_data_t ref_in_val = {};
    seed_value(dut_in_val, case_id, 1);
    copy_value(ref_in_val, dut_in_val);

    fxp_sqrt_top(dut_in_val);
    ref::fxp_sqrt_top(ref_in_val);

    if (!compare_storage(dut_in_val, ref_in_val)) return 0;
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

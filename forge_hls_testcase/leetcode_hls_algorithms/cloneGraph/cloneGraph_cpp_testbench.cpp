#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <sys/resource.h>
#include <type_traits>

#include "cloneGraph.cpp"

namespace ref {
struct Node {
    int val;
    int neighbors[N];
    int num_neighbors;
};

void cloneGraph(Node graph_in[N], Node graph_out[N]) {
    for (int i = 0; i < N; i++) {
        graph_out[i].val = graph_in[i].val;
        graph_out[i].num_neighbors = graph_in[i].num_neighbors;
        for (int j = 0; j < graph_in[i].num_neighbors; j++) {
            graph_out[i].neighbors[j] = graph_in[i].neighbors[j];
        }
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

void seed_value(Node& value, int case_id, int salt) {
    zero_value(value);
    seed_value(value.val, case_id, salt + 1);
    seed_value(value.neighbors, case_id, salt + 2);
    value.num_neighbors = static_cast<int>(0);
}

void seed_value(ref::Node& value, int case_id, int salt) {
    zero_value(value);
    seed_value(value.val, case_id, salt + 1);
    seed_value(value.neighbors, case_id, salt + 2);
    value.num_neighbors = static_cast<int>(0);
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

static void zero_graph(Node graph[N]) {
    for (int i = 0; i < N; ++i) {
        graph[i].val = 0;
        graph[i].num_neighbors = 0;
        for (int j = 0; j < N; ++j) {
            graph[i].neighbors[j] = 0;
        }
    }
}

static void zero_graph(ref::Node graph[N]) {
    for (int i = 0; i < N; ++i) {
        graph[i].val = 0;
        graph[i].num_neighbors = 0;
        for (int j = 0; j < N; ++j) {
            graph[i].neighbors[j] = 0;
        }
    }
}

static void fill_case(int case_id, Node graph_in[N], Node graph_out[N]) {
    zero_graph(graph_in);
    zero_graph(graph_out);

    if (case_id == 0) {
        graph_in[0].val = 7;
        return;
    }

    graph_in[0].val = 1;
    graph_in[0].num_neighbors = 2;
    graph_in[0].neighbors[0] = 1;
    graph_in[0].neighbors[1] = 2;
    graph_in[1].val = 2;
    graph_in[1].num_neighbors = 1;
    graph_in[1].neighbors[0] = 2;
    graph_in[2].val = 3;

    if (case_id == 1) {
        return;
    }

    graph_in[2].num_neighbors = 1;
    graph_in[2].neighbors[0] = 0;
}

static void copy_graph(ref::Node dst[N], const Node src[N]) {
    for (int i = 0; i < N; ++i) {
        dst[i].val = src[i].val;
        dst[i].num_neighbors = src[i].num_neighbors;
        for (int j = 0; j < N; ++j) {
            dst[i].neighbors[j] = src[i].neighbors[j];
        }
    }
}

static int run_case(int case_id) {
static Node dut_graph_in[N] = {};
static ref::Node ref_graph_in[N] = {};
static Node dut_graph_out[N] = {};
static ref::Node ref_graph_out[N] = {};
    fill_case(case_id, dut_graph_in, dut_graph_out);
    copy_graph(ref_graph_in, dut_graph_in);
    copy_graph(ref_graph_out, dut_graph_out);

    cloneGraph(dut_graph_in, dut_graph_out);
    ref::cloneGraph(ref_graph_in, ref_graph_out);

    if (!compare_storage(dut_graph_in, ref_graph_in)) return 0;
    if (!compare_storage(dut_graph_out, ref_graph_out)) return 0;
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

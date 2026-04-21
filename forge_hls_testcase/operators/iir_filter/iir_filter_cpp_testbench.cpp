
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <type_traits>
#include <cmath>
#include <unistd.h>
#include "hls_stream.h"

#define TB_PTR_BYTES 4096
#define TB_ITERS 3
#define TB_STREAM_ELEMS 8

#define iir_filter ref_model_impl
#include "iir_filter.cpp"
#undef iir_filter
static void tb_fill(unsigned char* p, size_t n, unsigned int* seed) {
    memset(p, 0, n);
    unsigned int x = *seed;
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
    }
    *seed = x;
}

template <typename T>
static void tb_init_obj(T& v, unsigned int* seed) {
    using U = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    unsigned int x = *seed;
    *seed = x * 1664525u + 1013904223u;
    if constexpr (std::is_array<U>::value) {
        for (size_t i = 0; i < std::extent<U>::value; ++i) {
            tb_init_obj(v[i], seed);
        }
    } else if constexpr (std::is_same<U, bool>::value) {
        v = (x & 1u) != 0;
    } else if constexpr (std::is_integral<U>::value || std::is_enum<U>::value) {
        v = static_cast<U>(1u + (x % 4u));
    } else if constexpr (std::is_floating_point<U>::value) {
        v = static_cast<U>(0.25 * double(1u + (x % 4u)));
    } else if constexpr (std::is_constructible<U, int>::value) {
        v = U(static_cast<int>(1u + (x % 4u)));
    } else if constexpr (std::is_constructible<U, double>::value) {
        v = U(0.25 * double(1u + (x % 4u)));
    } else {
        memset(&v, 0, sizeof(v));
    }
}

template <typename T>
static int tb_equal_obj(const T& a, const T& b) {
    using U = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    if constexpr (std::is_array<U>::value) {
        for (size_t i = 0; i < std::extent<U>::value; ++i) {
            if (!tb_equal_obj(a[i], b[i])) {
                return 0;
            }
        }
        return 1;
    } else if constexpr (std::is_floating_point<U>::value) {
        const double da = static_cast<double>(a);
        const double db = static_cast<double>(b);
        const double diff = std::fabs(da - db);
        const double scale = 1.0 + std::fabs(da) + std::fabs(db);
        return diff <= 1e-4 * scale;
    } else {
        const unsigned char* a_bytes = reinterpret_cast<const unsigned char*>(&a);
        const unsigned char* b_bytes = reinterpret_cast<const unsigned char*>(&b);
        for (size_t i = 0; i < sizeof(U); ++i) {
            if (a_bytes[i] != b_bytes[i]) {
                return 0;
            }
        }
        return 1;
    }
}


template <typename T>
struct tb_stream_value;

template <typename U>
struct tb_stream_value<hls::stream<U>> { using type = U; };

template <typename U>
struct tb_stream_value<hls::stream<U>&> { using type = U; };

template <typename U>
struct tb_stream_value<const hls::stream<U>&> { using type = U; };

template <typename StreamT>
static void tb_init_stream(StreamT& s, unsigned int seed) {
    using S = typename std::remove_reference<StreamT>::type;
    using Elem = typename tb_stream_value<S>::type;
    for (int i = 0; i < TB_STREAM_ELEMS; ++i) {
        Elem v{};
        tb_init_obj(v, &seed);
        s.write(v);
    }
}

template <typename StreamT>
static int tb_equal_stream(StreamT& a, StreamT& b) {
    while (!a.empty() && !b.empty()) {
        auto va = a.read();
        auto vb = b.read();
        if (!tb_equal_obj(va, vb)) {
            return 0;
        }
    }
    return a.empty() && b.empty();
}

int main(int argc, char** argv) {
    const char* candidate = (argc > 1) ? argv[1] : "../../../../forgehls/kernels/kernels/operators/iir_filter/iir_filter.cpp";
    char so_path[512];
    char wrap_path[512];
    snprintf(so_path, sizeof(so_path), "/tmp/tb_cand_%ld_%d.so", (long)getpid(), rand());
    snprintf(wrap_path, sizeof(wrap_path), "/tmp/tb_wrap_%ld_%d.cpp", (long)getpid(), rand());
    FILE* wf = fopen(wrap_path, "w");
    if (!wf) { printf("fail\n"); return 1; }
    fprintf(wf, "#define iir_filter __tb_dut_impl\n");
    fprintf(wf, "#include \"%s\"\n", candidate);
    fprintf(wf, "#undef iir_filter\n");
    fprintf(wf, "extern \"C\" void __tb_entry(ap_fixed<16, 8> input[N], ap_fixed<16, 8> output[N], ap_fixed<16, 8> a1, ap_fixed<16, 8> a2, ap_fixed<16, 8> b0, ap_fixed<16, 8> b1, ap_fixed<16, 8> b2) { __tb_dut_impl(input, output, a1, a2, b0, b1, b2); }\n");
    fclose(wf);
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "gcc -x c++ -std=gnu++17 -O2 -w -shared -fPIC -I\"../../../../forgehls/kernels/kernels/_tb_compat\" -I\"../../../../forgehls/kernels/kernels/operators/iir_filter\" \"%s\" -o \"%s\" -lm >/dev/null 2>&1", wrap_path, so_path);
    if (system(cmd) != 0) { remove(wrap_path); printf("fail\n"); return 1; }
    void* handle = dlopen(so_path, RTLD_NOW);
    if (!handle) { remove(wrap_path); remove(so_path); printf("fail\n"); return 1; }
    __typeof__(&ref_model_impl) dut_fn = (__typeof__(&ref_model_impl))dlsym(handle, "__tb_entry");
    if (!dut_fn) { dlclose(handle); remove(wrap_path); remove(so_path); printf("fail\n"); return 1; }
    int ok = 1;
    for (int it = 0; it < TB_ITERS && ok; ++it) {
        unsigned int seed = (unsigned int)(0x12345678u + (unsigned int)it * 911u);
      static ap_fixed<16, 8> ref_input[N];
      static ap_fixed<16, 8> dut_input[N];
        tb_init_obj(ref_input, &seed);
        memcpy(&dut_input, &ref_input, sizeof(ref_input));
      static ap_fixed<16, 8> ref_output[N];
      static ap_fixed<16, 8> dut_output[N];
        tb_init_obj(ref_output, &seed);
        memcpy(&dut_output, &ref_output, sizeof(ref_output));
        ap_fixed<16, 8> ref_a1;
        ap_fixed<16, 8> dut_a1;
        tb_init_obj(ref_a1, &seed);
        memcpy(&dut_a1, &ref_a1, sizeof(ref_a1));
        ap_fixed<16, 8> ref_a2;
        ap_fixed<16, 8> dut_a2;
        tb_init_obj(ref_a2, &seed);
        memcpy(&dut_a2, &ref_a2, sizeof(ref_a2));
        ap_fixed<16, 8> ref_b0;
        ap_fixed<16, 8> dut_b0;
        tb_init_obj(ref_b0, &seed);
        memcpy(&dut_b0, &ref_b0, sizeof(ref_b0));
        ap_fixed<16, 8> ref_b1;
        ap_fixed<16, 8> dut_b1;
        tb_init_obj(ref_b1, &seed);
        memcpy(&dut_b1, &ref_b1, sizeof(ref_b1));
        ap_fixed<16, 8> ref_b2;
        ap_fixed<16, 8> dut_b2;
        tb_init_obj(ref_b2, &seed);
        memcpy(&dut_b2, &ref_b2, sizeof(ref_b2));
        ref_model_impl(ref_input, ref_output, ref_a1, ref_a2, ref_b0, ref_b1, ref_b2);
        dut_fn(dut_input, dut_output, dut_a1, dut_a2, dut_b0, dut_b1, dut_b2);
        if (!tb_equal_obj(ref_input, dut_input)) ok = 0;
        if (!tb_equal_obj(ref_output, dut_output)) ok = 0;
        if (!tb_equal_obj(ref_a1, dut_a1)) ok = 0;
        if (!tb_equal_obj(ref_a2, dut_a2)) ok = 0;
        if (!tb_equal_obj(ref_b0, dut_b0)) ok = 0;
        if (!tb_equal_obj(ref_b1, dut_b1)) ok = 0;
        if (!tb_equal_obj(ref_b2, dut_b2)) ok = 0;
    }
    dlclose(handle);
    remove(wrap_path);
    remove(so_path);
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

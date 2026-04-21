
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
#define TB_CASES 3
#define TB_STREAM_ELEMS 8

#define mem_bottleneck_resolved ref_model_impl
#include "combined.cpp"
#undef mem_bottleneck_resolved
static unsigned int tb_mix(unsigned int case_id, unsigned int salt) {
    unsigned int x = 0x9e3779b9u * (case_id + 1u) + 0x85ebca6bu * (salt + 1u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static void tb_fill(unsigned char* p, size_t n, int case_id, unsigned int salt) {
    unsigned int x = tb_mix((unsigned int)case_id, salt);
    for (size_t i = 0; i < n; ++i) {
        x = x * 1664525u + 1013904223u;
        p[i] = (unsigned char)(x >> 24);
    }
}

template <typename T>
static void tb_init_obj(T& v, int case_id, unsigned int salt = 0) {
    using U = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
    const unsigned int x = tb_mix((unsigned int)case_id, salt);
    if constexpr (std::is_array<U>::value) {
        for (size_t i = 0; i < std::extent<U>::value; ++i) {
            tb_init_obj(v[i], case_id, salt + 131u * (unsigned int)(i + 1));
        }
    } else if constexpr (std::is_same<U, bool>::value) {
        v = ((x >> (case_id + 1)) & 1u) != 0;
    } else if constexpr (std::is_integral<U>::value || std::is_enum<U>::value) {
        unsigned int base = (case_id == 0) ? (1u + (x % 4u)) : ((case_id == 1) ? (1u + (x % 5u)) : (2u + (x % 5u)));
        v = static_cast<U>(base);
    } else if constexpr (std::is_floating_point<U>::value) {
        double base = (case_id == 0) ? (0.25 * double(1u + (x % 4u)))
                                     : ((case_id == 1) ? (0.1 * double(1u + (x % 5u)))
                                                       : (0.15 * double(1u + (x % 6u))));
        v = static_cast<U>(base);
    } else if constexpr (std::is_constructible<U, int>::value) {
        int base = (case_id == 0) ? int(1u + (x % 4u)) : ((case_id == 1) ? int(1u + (x % 5u)) : int(2u + (x % 5u)));
        v = U(base);
    } else if constexpr (std::is_constructible<U, double>::value) {
        double base = (case_id == 0) ? (0.25 * double(1u + (x % 4u)))
                                     : ((case_id == 1) ? (0.1 * double(1u + (x % 5u)))
                                                       : (0.15 * double(1u + (x % 6u))));
        v = U(base);
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
static void tb_init_stream(StreamT& s, int case_id, unsigned int salt = 0) {
    using S = typename std::remove_reference<StreamT>::type;
    using Elem = typename tb_stream_value<S>::type;
    for (int i = 0; i < TB_STREAM_ELEMS; ++i) {
        Elem v{};
        tb_init_obj(v, case_id, salt + 97u * (unsigned int)(i + 1));
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
    const char* candidate = (argc > 1) ? argv[1] : "../../../../forgehls/kernels/kernels/Vitis-HLS-Introductory-Examples-flatten/modified/combined.cpp";
    char so_path[512];
    char wrap_path[512];
    snprintf(so_path, sizeof(so_path), "/tmp/tb_cand_%ld_%d.so", (long)getpid(), rand());
    snprintf(wrap_path, sizeof(wrap_path), "/tmp/tb_wrap_%ld_%d.cpp", (long)getpid(), rand());
    FILE* wf = fopen(wrap_path, "w");
    if (!wf) { printf("fail\n"); return 1; }
    fprintf(wf, "#define mem_bottleneck_resolved __tb_dut_impl\n");
    fprintf(wf, "#include \"%s\"\n", candidate);
    fprintf(wf, "#undef mem_bottleneck_resolved\n");
    fprintf(wf, "extern \"C\" dout_t __tb_entry(din_t mem[N]) { return __tb_dut_impl(mem); }\n");
    fclose(wf);
    char cmd[8192];
    snprintf(cmd, sizeof(cmd), "gcc -x c++ -std=gnu++17 -O2 -w -shared -fPIC -I\"../../../../forgehls/kernels/kernels/_tb_compat\" -I\"../../../../forgehls/kernels/kernels/Vitis-HLS-Introductory-Examples-flatten/modified\" \"%s\" -o \"%s\" -lm >/dev/null 2>&1", wrap_path, so_path);
    if (system(cmd) != 0) { remove(wrap_path); printf("fail\n"); return 1; }
    void* handle = dlopen(so_path, RTLD_NOW);
    if (!handle) { remove(wrap_path); remove(so_path); printf("fail\n"); return 1; }
    __typeof__(&ref_model_impl) dut_fn = (__typeof__(&ref_model_impl))dlsym(handle, "__tb_entry");
    if (!dut_fn) { dlclose(handle); remove(wrap_path); remove(so_path); printf("fail\n"); return 1; }
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
      static din_t ref_mem[N];
      static din_t dut_mem[N];
        tb_init_obj(ref_mem, case_id, 0u);
        memcpy(&dut_mem, &ref_mem, sizeof(ref_mem));
        dout_t ref_ret = ref_model_impl(ref_mem);
        dout_t dut_ret = dut_fn(dut_mem);
        if (!tb_equal_obj(ref_ret, dut_ret)) ok = 0;
        if (!tb_equal_obj(ref_mem, dut_mem)) ok = 0;
    }
    dlclose(handle);
    remove(wrap_path);
    remove(so_path);
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

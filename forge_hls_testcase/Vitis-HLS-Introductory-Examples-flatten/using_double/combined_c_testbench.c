
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TB_PTR_BYTES 4096
#define TB_CASES 3

#define pointer_double ref_model_impl
#include "combined.c"
#undef pointer_double

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

static int tb_is_cpp_path(const char* p) {
    const char* d = strrchr(p, '.');
    if (!d) return 0;
    return strcmp(d, ".cpp") == 0 || strcmp(d, ".cc") == 0 || strcmp(d, ".cxx") == 0;
}

int main(int argc, char** argv) {
    const char* candidate = (argc > 1) ? argv[1] : "../../../../forgehls/kernels/kernels/Vitis-HLS-Introductory-Examples-flatten/using_double/combined.c";
    char so_path[512];
    snprintf(so_path, sizeof(so_path), "/tmp/tb_cand_%ld_%d.so", (long)getpid(), rand());
    char cmd[8192];
    if (tb_is_cpp_path(candidate)) {
        snprintf(cmd, sizeof(cmd), "gcc -x c++ -std=gnu++17 -O2 -w -shared -fPIC -I\"../../../../forgehls/kernels/kernels/_tb_compat\" -I\"../../../../forgehls/kernels/kernels/Vitis-HLS-Introductory-Examples-flatten/using_double\" \"%s\" -o \"%s\" -lm >/dev/null 2>&1", candidate, so_path);
    } else {
        snprintf(cmd, sizeof(cmd), "gcc -x c -std=gnu11 -O2 -w -shared -fPIC -I\"../../../../forgehls/kernels/kernels/_tb_compat\" -I\"../../../../forgehls/kernels/kernels/Vitis-HLS-Introductory-Examples-flatten/using_double\" \"%s\" -o \"%s\" -lm >/dev/null 2>&1", candidate, so_path);
    }
    if (system(cmd) != 0) { printf("fail\n"); return 1; }
    void* handle = dlopen(so_path, RTLD_NOW);
    if (!handle) { printf("fail\n"); remove(so_path); return 1; }
    __typeof__(&ref_model_impl) dut_fn = (__typeof__(&ref_model_impl))dlsym(handle, "pointer_double");
    if (!dut_fn) { dlclose(handle); remove(so_path); printf("fail\n"); return 1; }
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
        data_t ref_pos;
        data_t dut_pos;
        tb_fill((unsigned char*)&ref_pos, sizeof(ref_pos), case_id, 0u);
        memcpy(&dut_pos, &ref_pos, sizeof(ref_pos));
        data_t init_pos;
        memcpy(&init_pos, &ref_pos, sizeof(ref_pos));
        data_t ref_x;
        data_t dut_x;
        tb_fill((unsigned char*)&ref_x, sizeof(ref_x), case_id, 1u);
        memcpy(&dut_x, &ref_x, sizeof(ref_x));
        data_t init_x;
        memcpy(&init_x, &ref_x, sizeof(ref_x));
        unsigned char ref_flag_storage[TB_PTR_BYTES];
        unsigned char dut_flag_storage[TB_PTR_BYTES];
        data_t *ref_flag;
        data_t *dut_flag;
        ref_flag = (__typeof__(ref_flag))(ref_flag_storage);
        dut_flag = (__typeof__(dut_flag))(dut_flag_storage);
        tb_fill(ref_flag_storage, sizeof(ref_flag_storage), case_id, 2u);
        memcpy(dut_flag_storage, ref_flag_storage, sizeof(ref_flag_storage));
        unsigned char init_flag_storage[TB_PTR_BYTES];
        memcpy(init_flag_storage, ref_flag_storage, sizeof(ref_flag_storage));
        data_t ref_ret = ref_model_impl(ref_pos, ref_x, ref_flag);
        data_t dut_ret = dut_fn(dut_pos, dut_x, dut_flag);
        if (memcmp(&ref_ret, &dut_ret, sizeof(ref_ret)) != 0) ok = 0;
        if (memcmp(&ref_pos, &init_pos, sizeof(ref_pos)) != 0 || memcmp(&dut_pos, &init_pos, sizeof(ref_pos)) != 0) {
            if (memcmp(&ref_pos, &dut_pos, sizeof(ref_pos)) != 0) ok = 0;
        }
        if (memcmp(&ref_x, &init_x, sizeof(ref_x)) != 0 || memcmp(&dut_x, &init_x, sizeof(ref_x)) != 0) {
            if (memcmp(&ref_x, &dut_x, sizeof(ref_x)) != 0) ok = 0;
        }
        if (memcmp(ref_flag_storage, init_flag_storage, sizeof(ref_flag_storage)) != 0 || memcmp(dut_flag_storage, init_flag_storage, sizeof(ref_flag_storage)) != 0) {
            if (memcmp(ref_flag_storage, dut_flag_storage, sizeof(ref_flag_storage)) != 0) ok = 0;
        }
    }
    dlclose(handle);
    remove(so_path);
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

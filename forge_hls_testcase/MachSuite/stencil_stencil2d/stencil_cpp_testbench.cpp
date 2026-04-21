#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

#define TB_CASES 3

#include "stencil.c"

namespace reference_model_ns {
void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c, k1, k2;
    TYPE temp, mul;

    stencil_label1:for (r=0; r<row_size-2; r++) {
        stencil_label2:for (c=0; c<col_size-2; c++) {
            temp = (TYPE)0;
            stencil_label3:for (k1=0;k1<3;k1++){
                stencil_label4:for (k2=0;k2<3;k2++){
                    mul = filter[k1*3 + k2] * orig[(r+k1)*col_size + c+k2];
                    temp += mul;
                }
            }
            sol[(r*col_size) + c] = temp;
        }
    }
}

}

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

int main() {
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
        struct bench_args_t ref_args;
        struct bench_args_t dut_args;
        struct bench_args_t init_args;
        tb_fill((unsigned char*)&ref_args.orig, sizeof(ref_args.orig), case_id, 0u);
        tb_fill((unsigned char*)&ref_args.sol, sizeof(ref_args.sol), case_id, 1u);
        tb_fill((unsigned char*)&ref_args.filter, sizeof(ref_args.filter), case_id, 2u);
        memcpy(&dut_args, &ref_args, sizeof(ref_args));
        memcpy(&init_args, &ref_args, sizeof(ref_args));
        reference_model_ns::stencil(ref_args.orig, ref_args.sol, ref_args.filter);
        ::stencil(dut_args.orig, dut_args.sol, dut_args.filter);
        if (memcmp(&ref_args.orig, &init_args.orig, sizeof(ref_args.orig)) != 0 || memcmp(&dut_args.orig, &init_args.orig, sizeof(ref_args.orig)) != 0) {
            if (memcmp(&ref_args.orig, &dut_args.orig, sizeof(ref_args.orig)) != 0) ok = 0;
        }
        if (memcmp(&ref_args.sol, &init_args.sol, sizeof(ref_args.sol)) != 0 || memcmp(&dut_args.sol, &init_args.sol, sizeof(ref_args.sol)) != 0) {
            if (memcmp(&ref_args.sol, &dut_args.sol, sizeof(ref_args.sol)) != 0) ok = 0;
        }
        if (memcmp(&ref_args.filter, &init_args.filter, sizeof(ref_args.filter)) != 0 || memcmp(&dut_args.filter, &init_args.filter, sizeof(ref_args.filter)) != 0) {
            if (memcmp(&ref_args.filter, &dut_args.filter, sizeof(ref_args.filter)) != 0) ok = 0;
        }

    }
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

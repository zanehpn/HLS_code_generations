#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

#define TB_CASES 3

#include "spmv.c"

namespace reference_model_ns {
/*
Based on algorithm described here:
http://www.cs.berkeley.edu/~mhoemmen/matrix-seminar/slides/UCB_sparse_tutorial_1.pdf
*/


void ellpack(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N])
{
    int i, j;
    TYPE Si;

    ellpack_1 : for (i=0; i<N; i++) {
        TYPE sum = out[i];
        ellpack_2 : for (j=0; j<L; j++) {
                Si = nzval[j + i*L] * vec[cols[j + i*L]];
                sum += Si;
        }
        out[i] = sum;
    }
}

}

static void tb_setup_case(TYPE nzval[N*L], int32_t cols[N*L], TYPE vec[N], TYPE out[N], int case_id) {
    int i;
    memset(nzval, 0, sizeof(TYPE) * N * L);
    memset(cols, 0, sizeof(int32_t) * N * L);
    memset(out, 0, sizeof(TYPE) * N);
    for (i = 0; i < N; ++i) vec[i] = (TYPE)(0.2 * ((i % 9) + 1 + case_id));
    for (i = 0; i < 4; ++i) {
        nzval[i * L + 0] = (TYPE)(1.0 + 0.1 * (i + case_id));
        cols[i * L + 0] = i;
        nzval[i * L + 1] = (TYPE)(0.5 + 0.05 * (i + case_id));
        cols[i * L + 1] = (i + 1) % N;
    }
}

static int tb_equal_out(const TYPE a[N], const TYPE b[N]) {
    return memcmp(a, b, sizeof(TYPE) * N) == 0;
}

int main() {
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
        static TYPE ref_nzval[N*L], dut_nzval[N*L], ref_vec[N], dut_vec[N], ref_out[N], dut_out[N];
        static int32_t ref_cols[N*L], dut_cols[N*L];
        tb_setup_case(ref_nzval, ref_cols, ref_vec, ref_out, case_id);
        memcpy(dut_nzval, ref_nzval, sizeof(ref_nzval));
        memcpy(dut_cols, ref_cols, sizeof(ref_cols));
        memcpy(dut_vec, ref_vec, sizeof(ref_vec));
        memcpy(dut_out, ref_out, sizeof(ref_out));
        reference_model_ns::ellpack(ref_nzval, ref_cols, ref_vec, ref_out);
        ::ellpack(dut_nzval, dut_cols, dut_vec, dut_out);
        if (!tb_equal_out(ref_out, dut_out)) ok = 0;
    }
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

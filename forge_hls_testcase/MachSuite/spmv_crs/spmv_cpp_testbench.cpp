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


void spmv(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N+1], TYPE vec[N], TYPE out[N]){
    int i, j;
    TYPE sum, Si;

    spmv_1 : for(i = 0; i < N; i++){
        sum = 0; Si = 0;
        int tmp_begin = rowDelimiters[i];
        int tmp_end = rowDelimiters[i+1];
        spmv_2 : for (j = tmp_begin; j < tmp_end; j++){
        #pragma HLS loop_tripcount min=1 max=NNZ avg=5
            Si = val[j] * vec[cols[j]];
            sum = sum + Si;
        }
        out[i] = sum;
    }
}

}

static void tb_setup_case(TYPE val[NNZ], int32_t cols[NNZ], int32_t rowDelimiters[N + 1], TYPE vec[N], TYPE out[N], int case_id) {
    int i, nz = 0;
    memset(val, 0, sizeof(TYPE) * NNZ);
    memset(cols, 0, sizeof(int32_t) * NNZ);
    memset(rowDelimiters, 0, sizeof(int32_t) * (N + 1));
    memset(out, 0, sizeof(TYPE) * N);
    for (i = 0; i < N; ++i) vec[i] = (TYPE)(0.25 * ((i % 7) + 1 + case_id));
    rowDelimiters[0] = 0;
    for (i = 0; i < 4; ++i) {
        val[nz] = (TYPE)(1.0 + 0.1 * (i + case_id)); cols[nz++] = i;
        val[nz] = (TYPE)(0.5 + 0.05 * (i + case_id)); cols[nz++] = (i + 1) % N;
        rowDelimiters[i + 1] = nz;
    }
    for (i = 4; i < N; ++i) rowDelimiters[i + 1] = nz;
}

static int tb_equal_out(const TYPE a[N], const TYPE b[N]) {
    return memcmp(a, b, sizeof(TYPE) * N) == 0;
}

int main() {
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
        static TYPE ref_val[NNZ], dut_val[NNZ], ref_vec[N], dut_vec[N], ref_out[N], dut_out[N];
        static int32_t ref_cols[NNZ], dut_cols[NNZ], ref_rowDelimiters[N+1], dut_rowDelimiters[N+1];
        tb_setup_case(ref_val, ref_cols, ref_rowDelimiters, ref_vec, ref_out, case_id);
        memcpy(dut_val, ref_val, sizeof(ref_val));
        memcpy(dut_cols, ref_cols, sizeof(ref_cols));
        memcpy(dut_rowDelimiters, ref_rowDelimiters, sizeof(ref_rowDelimiters));
        memcpy(dut_vec, ref_vec, sizeof(ref_vec));
        memcpy(dut_out, ref_out, sizeof(ref_out));
        reference_model_ns::spmv(ref_val, ref_cols, ref_rowDelimiters, ref_vec, ref_out);
        ::spmv(dut_val, dut_cols, dut_rowDelimiters, dut_vec, dut_out);
        if (!tb_equal_out(ref_out, dut_out)) ok = 0;
    }
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

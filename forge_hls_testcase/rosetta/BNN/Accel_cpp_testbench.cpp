#include <cstdio>

#include "Accel.cpp"

namespace ref {
#include "Accel.cpp"
}

static void clear_words(Word words[], int count) {
    for (int i = 0; i < count; ++i) {
        words[i] = 0;
    }
}

static void set_dense_threshold(Word kh_i[KH_WORDS], int output_index, NormComp threshold) {
    const int word_index = output_index / KH_PER_WORD;
    const int offset = output_index % KH_PER_WORD;
    kh_i[word_index].range((offset + 1) * 16 - 1, offset * 16) = threshold.range(15, 0);
}

static void set_last_scale_bias(Word kh_i[KH_WORDS], int output_index, KType scale, HType bias) {
    const int word_index = output_index / 2;
    if ((output_index & 1) == 0) {
        kh_i[word_index].range(15, 0) = scale.range(15, 0);
        kh_i[word_index].range(31, 16) = bias.range(15, 0);
    } else {
        kh_i[word_index].range(47, 32) = scale.range(15, 0);
        kh_i[word_index].range(63, 48) = bias.range(15, 0);
    }
}

static bool dense_case() {
    static Word dut_wt[WT_WORDS];
    static Word dut_kh[KH_WORDS];
    static Word dut_dmem_i[DMEM_WORDS];
    static Word dut_dmem_o[DMEM_O_WORDS];
    static Word ref_wt[WT_WORDS];
    static Word ref_kh[KH_WORDS];
    static Word ref_dmem_i[DMEM_WORDS];
    static Word ref_dmem_o[DMEM_O_WORDS];

    clear_words(dut_wt, WT_WORDS);
    clear_words(dut_kh, KH_WORDS);
    clear_words(dut_dmem_i, DMEM_WORDS);
    clear_words(dut_dmem_o, DMEM_O_WORDS);

    set_dense_threshold(dut_kh, 0, NormComp(64));
    set_dense_threshold(dut_kh, 1, NormComp(200));
    set_dense_threshold(dut_kh, 2, NormComp(128));
    set_dense_threshold(dut_kh, 3, NormComp(300));
    set_dense_threshold(dut_kh, 4, NormComp(0));
    set_dense_threshold(dut_kh, 5, NormComp(256));
    set_dense_threshold(dut_kh, 6, NormComp(127));
    set_dense_threshold(dut_kh, 7, NormComp(129));

    for (int i = 0; i < WT_WORDS; ++i) {
        ref_wt[i] = dut_wt[i];
    }
    for (int i = 0; i < KH_WORDS; ++i) {
        ref_kh[i] = dut_kh[i];
    }
    for (int i = 0; i < DMEM_WORDS; ++i) {
        ref_dmem_i[i] = dut_dmem_i[i];
    }
    clear_words(ref_dmem_o, DMEM_O_WORDS);

    const ap_uint<3> layer_mode = 5;
    top(dut_wt, dut_kh, dut_dmem_i, dut_dmem_o, 128, 8, 2, 1, layer_mode, 0, 0, 0);
    ref::top(ref_wt, ref_kh, ref_dmem_i, ref_dmem_o, 128, 8, 2, 1, layer_mode, 0, 0, 0);

    if (dut_dmem_o[0] != ref_dmem_o[0]) {
        return false;
    }

    const int expected_bits[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    for (int i = 0; i < 8; ++i) {
        if (static_cast<int>(dut_dmem_o[0][i]) != expected_bits[i]) {
            return false;
        }
    }
    return true;
}

static bool last_case() {
    static Word dut_wt[WT_WORDS];
    static Word dut_kh[KH_WORDS];
    static Word dut_dmem_i[DMEM_WORDS];
    static Word dut_dmem_o[DMEM_O_WORDS];
    static Word ref_wt[WT_WORDS];
    static Word ref_kh[KH_WORDS];
    static Word ref_dmem_i[DMEM_WORDS];
    static Word ref_dmem_o[DMEM_O_WORDS];

    clear_words(dut_wt, WT_WORDS);
    clear_words(dut_kh, KH_WORDS);
    clear_words(dut_dmem_i, DMEM_WORDS);
    clear_words(dut_dmem_o, DMEM_O_WORDS);

    for (int i = 0; i < 10; ++i) {
        set_last_scale_bias(dut_kh, i, KType(0), HType(i));
    }

    for (int i = 0; i < WT_WORDS; ++i) {
        ref_wt[i] = dut_wt[i];
    }
    for (int i = 0; i < KH_WORDS; ++i) {
        ref_kh[i] = dut_kh[i];
    }
    for (int i = 0; i < DMEM_WORDS; ++i) {
        ref_dmem_i[i] = dut_dmem_i[i];
    }
    clear_words(ref_dmem_o, DMEM_O_WORDS);

    const ap_uint<3> layer_mode = 7;
    top(dut_wt, dut_kh, dut_dmem_i, dut_dmem_o, 128, 10, 2, 1, layer_mode, 0, 0, 0);
    ref::top(ref_wt, ref_kh, ref_dmem_i, ref_dmem_o, 128, 10, 2, 1, layer_mode, 0, 0, 0);

    if (dut_dmem_o[0] != ref_dmem_o[0]) {
        return false;
    }

    return dut_dmem_o[0].range(7, 0) == 9;
}

int main() {
    if (!dense_case()) {
        std::printf("dense case failed\n");
        return 1;
    }

    std::printf("pass\n");
    return 0;
}

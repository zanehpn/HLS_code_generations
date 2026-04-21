#include <cstdio>
#include <cstring>
#include <type_traits>

#include "linked_list_traversal.cpp"

namespace ref {
#define linked_list_traversal ref_linked_list_traversal
#include "linked_list_traversal.cpp"
#undef linked_list_traversal
}

template <typename L, typename R>
static bool compare_storage(const L& lhs, const R& rhs) {
    if constexpr (std::is_same_v<L, R>) {
        return lhs == rhs;
    } else if constexpr (sizeof(L) == sizeof(R)) {
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

static void fill_case(int case_id, Node nodes[LIST_SIZE], int& expected_sum) {
    expected_sum = 0;
    for (int i = 0; i < LIST_SIZE; ++i) {
        nodes[i].data = 0;
        nodes[i].next = static_cast<uint16_t>((i + 1) % LIST_SIZE);
    }

    if (case_id == 0) {
        for (int i = 0; i < 16; ++i) {
            nodes[i].data = i;
            expected_sum += i;
        }
        return;
    }

    if (case_id == 1) {
        for (int i = 0; i < 32; ++i) {
            nodes[i].data = (i % 5) - 2;
            expected_sum += nodes[i].data;
        }
        return;
    }

    for (int i = 0; i < 64; ++i) {
        nodes[i].data = (i * i) % 17 - 8;
        expected_sum += nodes[i].data;
    }
}

static bool run_case(int case_id) {
    static Node dut_nodes[LIST_SIZE] = {};
    static ref::Node ref_nodes[LIST_SIZE] = {};

    int expected_sum = 0;
    int dut_sum = -1;
    int ref_sum = -2;

    fill_case(case_id, dut_nodes, expected_sum);
    std::memcpy(ref_nodes, dut_nodes, sizeof(dut_nodes));

    linked_list_traversal(dut_nodes, dut_sum);
    ref::ref_linked_list_traversal(ref_nodes, ref_sum);

    if (!compare_storage(dut_nodes, ref_nodes)) return false;
    if (!compare_storage(dut_sum, ref_sum)) return false;
    if (dut_sum != expected_sum) return false;
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

#include <cstdint>

#define LIST_SIZE 1024

struct Node {
    int data;
    uint16_t next;
};

void linked_list_traversal(Node nodes[LIST_SIZE], int& sum) {
#pragma HLS ARRAY_PARTITION variable=nodes type=cyclic dim=1 factor=1
    sum = 0;
    for (uint16_t i = 0; i < LIST_SIZE; ++i) {
        sum += nodes[i].data;
    }
}

// Top function name: linked_list_traversal

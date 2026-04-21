#include <iostream>

#define N 1024

struct Node {
    int val;
    int neighbors[N];
    int num_neighbors;
};

void cloneGraph(Node graph_in[N], Node graph_out[N]) {
#pragma HLS ARRAY_PARTITION variable=graph_out type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=graph_in type=cyclic dim=1 factor=2
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=1
        graph_out[i].val = graph_in[i].val;
        graph_out[i].num_neighbors = graph_in[i].num_neighbors;
        for (int j = 0; j < graph_in[i].num_neighbors; j++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            graph_out[i].neighbors[j] = graph_in[i].neighbors[j];
        }
    }
}

// Top function name: cloneGraph

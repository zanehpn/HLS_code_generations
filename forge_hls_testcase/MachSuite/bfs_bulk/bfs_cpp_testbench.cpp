#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

#define TB_CASES 3

#include "bfs.c"

namespace reference_model_ns {
/*
Implementations based on:
Harish and Narayanan. "Accelerating large graph algorithms on the GPU using CUDA." HiPC, 2007.
Hong, Oguntebi, Olukotun. "Efficient Parallel Graph Exploration on Multi-Core CPU and GPU." PACT, 2011.
*/


void bfs(node_t nodes[N_NODES], edge_t edges[N_EDGES],
            node_index_t starting_node, level_t level[N_NODES],
            edge_index_t level_counts[N_LEVELS])
{
  node_index_t n;
  edge_index_t e;
  level_t horizon;
  edge_index_t cnt;

  level[starting_node] = 0;
  level_counts[0] = 1;

  loop_horizons: for( horizon=0; horizon<N_LEVELS; horizon++ ) {
    #pragma HLS loop_tripcount min=1 max=N_LEVELS
    cnt = 0;
    // Add unmarked neighbors of the current horizon to the next horizon
    loop_nodes: for( n=0; n<N_NODES; n++ ) {
      if( level[n]==horizon ) {
        edge_index_t tmp_begin = nodes[n].edge_begin;
        edge_index_t tmp_end = nodes[n].edge_end;
        loop_neighbors: for( e=tmp_begin; e<tmp_end; e++ ) {
          #pragma HLS loop_tripcount min=1 max=N_NODES
          node_index_t tmp_dst = edges[e].dst;
          level_t tmp_level = level[tmp_dst];

          if( tmp_level ==MAX_LEVEL ) { // Unmarked
            level[tmp_dst] = horizon+1;
            ++cnt;
          }
        }
      }
    }
    if( (level_counts[horizon+1]=cnt)==0 )
      break;
  }
}

}

static void tb_setup_case(node_t nodes[N_NODES], edge_t edges[N_EDGES],
                          node_index_t* starting_node, level_t level[N_NODES],
                          edge_index_t level_counts[N_LEVELS], int case_id) {
    int i;
    int edge_pos = 0;
    memset(nodes, 0, sizeof(node_t) * N_NODES);
    memset(edges, 0, sizeof(edge_t) * N_EDGES);
    for (i = 0; i < N_NODES; ++i) level[i] = MAX_LEVEL;
    for (i = 0; i < N_LEVELS; ++i) level_counts[i] = 0;
    *starting_node = 0;
    if (case_id == 0) {
        for (i = 0; i < 6; ++i) {
            nodes[i].edge_begin = edge_pos;
            if (i < 5) edges[edge_pos++].dst = (node_index_t)(i + 1);
            nodes[i].edge_end = edge_pos;
        }
    } else if (case_id == 1) {
        nodes[0].edge_begin = edge_pos; edges[edge_pos++].dst = 1; edges[edge_pos++].dst = 2; nodes[0].edge_end = edge_pos;
        nodes[1].edge_begin = edge_pos; edges[edge_pos++].dst = 3; nodes[1].edge_end = edge_pos;
        nodes[2].edge_begin = edge_pos; edges[edge_pos++].dst = 3; edges[edge_pos++].dst = 4; nodes[2].edge_end = edge_pos;
        nodes[3].edge_begin = edge_pos; edges[edge_pos++].dst = 5; nodes[3].edge_end = edge_pos;
        nodes[4].edge_begin = edge_pos; edges[edge_pos++].dst = 5; nodes[4].edge_end = edge_pos;
        nodes[5].edge_begin = edge_pos; nodes[5].edge_end = edge_pos;
    } else {
        for (i = 0; i < 8; ++i) {
            nodes[i].edge_begin = edge_pos;
            if (i + 1 < 8) edges[edge_pos++].dst = (node_index_t)(i + 1);
            if (i + 2 < 8) edges[edge_pos++].dst = (node_index_t)(i + 2);
            nodes[i].edge_end = edge_pos;
        }
    }
}

static int tb_equal_level_array(const level_t a[N_NODES], const level_t b[N_NODES]) {
    return memcmp(a, b, sizeof(level_t) * N_NODES) == 0;
}

static int tb_equal_count_array(const edge_index_t a[N_LEVELS], const edge_index_t b[N_LEVELS]) {
    return memcmp(a, b, sizeof(edge_index_t) * N_LEVELS) == 0;
}

int main() {
    int ok = 1;
    for (int case_id = 0; case_id < TB_CASES && ok; ++case_id) {
        static node_t ref_nodes[N_NODES], dut_nodes[N_NODES];
        static edge_t ref_edges[N_EDGES], dut_edges[N_EDGES];
        static level_t ref_level[N_NODES], dut_level[N_NODES];
        static edge_index_t ref_level_counts[N_LEVELS], dut_level_counts[N_LEVELS];
        node_index_t ref_starting_node, dut_starting_node;
        tb_setup_case(ref_nodes, ref_edges, &ref_starting_node, ref_level, ref_level_counts, case_id);
        memcpy(dut_nodes, ref_nodes, sizeof(ref_nodes));
        memcpy(dut_edges, ref_edges, sizeof(ref_edges));
        dut_starting_node = ref_starting_node;
        memcpy(dut_level, ref_level, sizeof(ref_level));
        memcpy(dut_level_counts, ref_level_counts, sizeof(ref_level_counts));
        reference_model_ns::bfs(ref_nodes, ref_edges, ref_starting_node, ref_level, ref_level_counts);
        ::bfs(dut_nodes, dut_edges, dut_starting_node, dut_level, dut_level_counts);
        if (!tb_equal_level_array(ref_level, dut_level)) ok = 0;
        if (!tb_equal_count_array(ref_level_counts, dut_level_counts)) ok = 0;
    }
    if (ok) { printf("pass\n"); return 0; }
    printf("fail\n");
    return 1;
}

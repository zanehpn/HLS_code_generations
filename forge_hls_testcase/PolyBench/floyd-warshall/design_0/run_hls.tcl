
open_project -reset project
set_top kernel_floyd_warshall
add_files polybench.h
add_files floyd-warshall.h
add_files floyd-warshall.c
add_files hls_design_meta.h
add_files mapper_kernel_floyd_warshall.cpp
add_files hls_design_meta.cpp
add_files apatb_kernel_floyd_warshall.cpp
add_files floyd-warshall.pp.0.c
add_files kernel_floyd_warshall_mac_muladd_12ns_12ns_12ns_23_4_1.h

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

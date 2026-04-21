
open_project -reset project
set_top kernel_gemver
add_files polybench.h
add_files gemver.h
add_files gemver.c
add_files apatb_kernel_gemver.cpp
add_files gemver.pp.0.c
add_files hls_design_meta.h
add_files hls_design_meta.cpp
add_files mapper_kernel_gemver.cpp
add_files kernel_gemver_mac_muladd_11ns_11ns_11ns_22_4_1.h

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

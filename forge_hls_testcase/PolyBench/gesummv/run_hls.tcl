
open_project -reset project
set_top kernel_gesummv
add_files gesummv.c
add_files gesummv.h
add_files polybench.h
add_files gesummv.pp.0.c
add_files mapper_kernel_gesummv.cpp
add_files hls_design_meta.h
add_files apatb_kernel_gesummv.cpp
add_files hls_design_meta.cpp
add_files kernel_gesummv_mac_muladd_11ns_11ns_11ns_21_4_1.h

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

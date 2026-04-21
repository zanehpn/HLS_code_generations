
open_project -reset project
set_top kernel_mvt
add_files mvt.c
add_files mvt.h
add_files polybench.h
add_files hls_design_meta.h
add_files mvt.pp.0.c
add_files mapper_kernel_mvt.cpp
add_files hls_design_meta.cpp
add_files apatb_kernel_mvt.cpp
add_files kernel_mvt_mac_muladd_11ns_11ns_11ns_22_4_1.h

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit


open_project -reset project
set_top kernel_jacobi_1d
add_files jacobi-1d.h
add_files jacobi-1d.c
add_files polybench.h
add_files mapper_kernel_jacobi_1d.cpp
add_files apatb_kernel_jacobi_1d.cpp
add_files jacobi-1d.pp.0.c
add_files hls_design_meta.h
add_files hls_design_meta.cpp

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

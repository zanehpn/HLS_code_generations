
open_project -reset project
set_top stencil
add_files stencil.c
add_files support.h
add_files stencil.h

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

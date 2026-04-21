
open_project -reset project
set_top int_to_float_pipelined
add_files int_to_float_pipelined.cpp

open_solution -reset "solution1"
set_part xcu280-fsvh2892-2L-e
create_clock -period 10
csynth_design
close_solution
exit

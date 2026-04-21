#include "stencil.h"

void stencil (TYPE orig[row_size * col_size], TYPE sol[row_size * col_size], TYPE filter[f_size]){
    int r, c, k1, k2;
#pragma HLS ARRAY_PARTITION variable=filter type=cyclic dim=1 factor=4
#pragma HLS ARRAY_PARTITION variable=sol type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=orig type=cyclic dim=1 factor=1
    TYPE temp, mul;

    stencil_label1:for (r=0; r<row_size-2; r++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        stencil_label2:for (c=0; c<col_size-2; c++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=2
            temp = (TYPE)0;
            stencil_label3:for (k1=0;k1<3;k1++){
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
                stencil_label4:for (k2=0;k2<3;k2++){
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
                    mul = filter[k1*3 + k2] * orig[(r+k1)*col_size + c+k2];
                    temp += mul;
                }
            }
            sol[(r*col_size) + c] = temp;
        }
    }
}

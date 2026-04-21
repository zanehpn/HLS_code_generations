/*
 * Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
 * Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
 * Copyright 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _ARRAY_MEM_PERFORM_H_
#define _ARRAY_MEM_PERFORM_H_

#include <fstream>
#include <iostream>
using namespace std;

#include "ap_int.h"
#define N 128

typedef ap_int<8> din_t;
typedef ap_int<16> dout_t;

dout_t mem_bottleneck_resolved(din_t mem[N]);

#endif

dout_t mem_bottleneck_resolved(din_t mem[N]) {
#pragma HLS ARRAY_PARTITION variable=mem type=cyclic dim=1 factor=2

    din_t tmp0, tmp1, tmp2;
    dout_t sum = 0;
    int i;

    tmp0 = mem[0];
    tmp1 = mem[1];
SUM_LOOP:
    for (i = 2; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        tmp2 = mem[i];
        sum += tmp2 + tmp1 + tmp0;
        tmp0 = tmp1;
        tmp1 = tmp2;
    }

    return sum;
}

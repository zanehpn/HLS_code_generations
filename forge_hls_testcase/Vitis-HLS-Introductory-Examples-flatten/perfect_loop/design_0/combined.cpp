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

#ifndef _LOOP_PERFECT_H_
#define _LOOP_PERFECT_H_

#include <fstream>
#include <iostream>
using namespace std;

#include "ap_int.h"
#define N 20

typedef ap_int<5> din_t;
typedef ap_int<12> dint_t;
typedef ap_int<6> dout_t;

void loop_perfect(din_t A[N], dout_t B[N]);

#endif

void loop_perfect(din_t A[N], dout_t B[N]) {
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=1
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=1
    int i, j;
    dint_t acc;

LOOP_I:
    for (i = 0; i < 20; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=4
    LOOP_J:
        for (j = 0; j < 20; j++) {
#pragma HLS PIPELINE 
#pragma HLS UNROLL factor=2
            if (j == 0)
                acc = 0;
            acc += A[j] * j;
            if (j == 19) {
                if (i % 2 == 0)
                    B[i] = acc / 20;
                else
                    B[i] = 0;
            }
        }
    }
}

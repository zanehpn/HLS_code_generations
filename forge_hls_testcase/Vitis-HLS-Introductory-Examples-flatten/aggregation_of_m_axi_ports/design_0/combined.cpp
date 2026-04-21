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

#include <iostream>
#define N 10

struct A {
    char foo;
    short bar;
};

// Top function
int dut(A* arr);

int dut(A* arr) {
#pragma HLS ARRAY_PARTITION variable=arr type=cyclic dim=1 factor=2
#pragma HLS interface m_axi port = arr depth = 10
#pragma HLS interface s_axilite port = arr
#pragma HLS aggregate variable = arr compact = auto
    int sum = 0;
    for (unsigned i = 0; i < N; i++) {
#pragma HLS PIPELINE OFF
#pragma HLS UNROLL factor=1
        auto tmp = arr[i];
        sum += tmp.foo + tmp.bar;
    }
    return sum;
}

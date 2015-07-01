/*
 * Copyright 2015 Baidu USA, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


//Simple test to determine that the kernels can be loaded and run
//with some basic sanity checking.

#include <cuda.h>
#include <iostream>
#include <vector>
#include <type_traits>
#include "nervana_c_api.h"

void test_sconv()
{
    int n = 8; //minibatch
    int c = 1; //channels
    int k = 8; //output filters

    int d = 1;
    int h = 5;
    int w = 5;

    int t = 1;
    int r = 3;
    int s = 3;

    int pad_d = 0;
    int pad_h = 0;
    int pad_w = 0;

    int str_d = 1;
    int str_h = 1;
    int str_w = 1;


    int input_size = n * c * d * h * w;
    int filter_size = c * k * t * r * s;
    int output_size = 3 * 3 * k * n;

    float *h_input = (float *)malloc(sizeof(float) * input_size);
    float *h_filter = (float *)malloc(sizeof(float) * filter_size);
    float *h_output = (float *)malloc(sizeof(float) * output_size);

    for (int i = 0; i < input_size; ++i)
        h_input[i] = (i % 2) / 5.f;

    for (int i = 0; i < filter_size; ++i) {
        h_filter[i] = (float)i / filter_size;
    }


    float *d_output, *d_input, *d_filter;
    cudaMalloc(&d_output, sizeof(float) * 3 * 3 * k * n);
    cudaMalloc(&d_input, sizeof(float) * input_size);
    cudaMalloc(&d_filter, sizeof(float) * filter_size);

    cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter, sizeof(float) * filter_size, cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, sizeof(float) * output_size);

    nervana_sconv_fprop(NULL, d_output, d_input, d_filter, 1.f, n, c, k, d, h, w, t, r, s, pad_d,
                        pad_h, pad_w, str_d, str_h, str_w, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(h_output, d_output, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; ++i)
        printf("%f\n", h_output[i]);


    for (int i = 0; i < output_size; ++i) {
        h_output[i] = (float)i / output_size;
    }

    cudaMemcpy(d_output, h_output, sizeof(float) * output_size, cudaMemcpyHostToDevice);
    cudaMemset(d_input, 0, sizeof(float) * input_size);

    nervana_sconv_bprop(NULL, d_input, d_filter, d_output, 1.f, n, c, k, d, h, w, t, r, s, pad_d,
                        pad_h, pad_w, str_d, str_h, str_w, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(h_input, d_input, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; ++i)
        printf("%f\n", h_input[i]);

    cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output, sizeof(float) * output_size, cudaMemcpyHostToDevice);

    nervana_sconv_bprop(NULL, d_filter, d_input, d_output, 1.f, n, c, k, d, h, w, t, r, s, pad_d,
                        pad_h, pad_w, str_d, str_h, str_w, 0);

    cudaDeviceSynchronize();
    cudaMemcpy(h_filter, d_filter, sizeof(float) * 9, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 9; ++i)
        printf("%f\n", h_filter[i]);

}

void test_hgemm(short* d_a, short* d_b, short* d_c, bool a_t, bool b_t, int size) {
    if (!nervana_hgemm(d_a, d_b, d_c, a_t, b_t, size, size, size, size, size, size, 1.0, 0.0, NULL, false, false, 0)) {
        std::cerr << "Error in kernel" << std::endl;
    }

    short* h_c = (short *)malloc(sizeof(short) * size * size);
    cudaMemcpy(h_c, d_c, sizeof(short) * size * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size * size; ++i) {
        if (h_c[i] != 0)
            std::cout << "Mismatch at " << i << " " << h_c[i] << " " << 0 << std::endl;
    }

    free(h_c);
}

void test_sgemm(float* d_a, float* d_b, float* d_c, bool a_t, bool b_t, int size) {
    if (!nervana_sgemm(d_a, d_b, d_c, a_t, b_t, size, size, size, size, size, size, 1.0, 0.0, NULL, false, false, 0)) {
        std::cerr << "Error in kernel" << std::endl;
    }

    float* h_c = (float *)malloc(sizeof(float) * size * size);
    cudaMemcpy(h_c, d_c, sizeof(float) * size * size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size * size; ++i) {
        if (h_c[i] != size)
            std::cout << "Mismatch at " << i << " " << h_c[i] << " " << size << std::endl;
    }

    free(h_c);
}

int main() {
    cudaError_t res = cudaFree(0);
    if (res != cudaSuccess) {
        std::cout << "CUDA did not initialize correctly" << std::endl;
        exit(1);
    }

    nervana_loadKernels("../cubin/");

    //make sure we load and run all different blocking and vector variants
    std::vector<int> sizes {257, 256, 255, 129, 128, 127, 65, 64, 17, 16, 15};

    {
        float *d_a, *d_b, *d_c;
        int size = sizes[0];

        std::vector<float> h_a(size * size, 1.f);
        std::vector<float> h_b(size * size, 1.f);

        cudaMalloc(&d_a, sizeof(float) * size * size);
        cudaMalloc(&d_b, sizeof(float) * size * size);
        cudaMalloc(&d_c, sizeof(float) * size * size);

        cudaMemcpy(d_a, h_a.data(), sizeof(float) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), sizeof(float) * size * size, cudaMemcpyHostToDevice);

        for (auto size : sizes) {
            test_sgemm(d_a, d_b, d_c, false, false, size);
            test_sgemm(d_a, d_b, d_c, false, true, size);
            test_sgemm(d_a, d_b, d_c, true, false, size);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    {
        short *d_a, *d_b, *d_c;
        int size = sizes[0];

        std::vector<short> h_a(size * size, 0);
        std::vector<short> h_b(size * size, 0);

        cudaMalloc(&d_a, sizeof(short) * size * size);
        cudaMalloc(&d_b, sizeof(short) * size * size);
        cudaMalloc(&d_c, sizeof(short) * size * size);

        cudaMemcpy(d_a, h_a.data(), sizeof(short) * size * size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b.data(), sizeof(short) * size * size, cudaMemcpyHostToDevice);

        for (auto size : sizes) {
            test_hgemm(d_a, d_b, d_c, false, false, size);
            test_hgemm(d_a, d_b, d_c, false, true, size);
            test_hgemm(d_a, d_b, d_c, true, false, size);
        }

        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    {
        test_sconv();
    }

    nervana_unloadKernels();
    return 0;
}

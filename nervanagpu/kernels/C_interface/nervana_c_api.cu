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


#include <vector>
#include <string>
#include <map>
#include <cuda.h>
#include <iostream>
#include <sstream>
#include <mutex>
#include <utility>
#include <tuple>
#include <stdint.h>
#include "nervana_c_api.h"

std::map<CUdevice, int> nervana_sm_counts_;
std::map<std::string, CUfunction> nervana_kernels_;
std::vector<CUmodule> nervana_modules_;

//for when we need to modify the above data structures
std::mutex nervana_load_kernels_mutex_;
std::mutex nervana_sm_count_mutex_;

extern "C" bool nervana_loadKernels(const char* const base_path_cstr) {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);

    //better would be a vector<string>, but there is a bug in nvcc that prevents this
    // (bug report filed)
    std::string names[38] = {
        "hgemm_nn_vec_128x128",
        "hgemm_nn_128x128",
        "hgemm_nt_vec_128x128",
        "hgemm_nt_128x128",
        "hgemm_tn_vec_128x128",
        "hgemm_tn_128x128",
        "hgemm_nn_vec_128x64",
        "hgemm_nn_128x64",
        "hgemm_tn_vec_128x64",
        "hgemm_tn_128x64",
        "hgemm_nn_vec_128x32",
        "hgemm_nn_128x32",
        "hgemm_tn_vec_128x32",
        "hgemm_tn_128x32",
        "hconv_fprop_K64_N64",
        "hconv_bprop_C128_N64",
        "hconv_updat_C128_K128",
        "hconv_updat_C64_K64",
        "hpool_max",
        "sgemm_nn_vec_128x128",
        "sgemm_nn_128x128",
        "sgemm_nt_vec_128x128",
        "sgemm_nt_128x128",
        "sgemm_tn_vec_128x128",
        "sgemm_tn_128x128",
        "sgemm_nn_vec_128x64",
        "sgemm_nn_128x64",
        "sgemm_tn_vec_128x64",
        "sgemm_tn_128x64",
        "sgemm_nn_vec_128x32",
        "sgemm_nn_128x32",
        "sgemm_tn_vec_128x32",
        "sgemm_tn_128x32",
        "sconv_fprop_K64_N64",
        "sconv_bprop_C128_N64",
        "sconv_updat_C128_K128",
        "sconv_updat_C128_K64",
        "spool_max"
    };

    std::string base_path(base_path_cstr);

    for (auto kernel : names) {
        if (nervana_kernels_.count(kernel) > 0)
            continue;

        CUmodule module;

        std::string path = base_path + kernel + std::string(".cubin");
        CUresult res = cuModuleLoad(&module, path.c_str());

        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to load: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_modules_.push_back(module);

        CUfunction function;
        res = cuModuleGetFunction(&function, module, kernel.c_str());
        if (res != CUDA_SUCCESS) {
            std::cerr << "Failed to extract: " << kernel << " " << res << std::endl;
            return false;
        }

        nervana_kernels_.insert(std::make_pair(kernel, function));
    }

    return true;
}

extern "C" bool nervana_unloadKernels() {
    std::lock_guard<std::mutex> lock(nervana_load_kernels_mutex_);
    while(nervana_modules_.size() > 0) {
        auto module = nervana_modules_.back();
        CUresult res = cuModuleUnload(module);

        nervana_modules_.pop_back();

        if (res != CUDA_SUCCESS)
            return false;
    }

    nervana_kernels_.clear();

    return true;
}

extern "C" size_t nervana_randStateSizeBytes() {
    return 2048 * 32 * sizeof(int);
}

extern "C" bool nervana_sgemm(float *A, float *B, float *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    int gridA = m / 128 + (m % 128 != 0);

    int gridB, threads;
    std::string name = "sgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    if ( (trans == "tn" && m % 4 == 0 && n % 4 == 0) ||
         (trans == "nn" && k % 8 == 0 && n % 4 == 0) ||
         (trans == "nt" && k % 16 == 0)) {
         name += "_vec";
    }

    int size = 0;
    if (trans == "nt")
        size = 128;

    if (size == 0) {
        if (n < 384 - 16) {
            int n128 = n % 128;
            if (n128 > 0 && n128 < 112) {
                if (n128 > 48 && n128 <= 64) {
                    int n64 = n / 64;
                    n64 *= gridA / sm_count;
                    if (n64 > 1 || trans == "tn") {
                        size = 64;
                    }
                    else {
                        size = 32;
                    }
                }
                else {
                    size = 32;
                }
            }
            else {
                size = 128;
            }
        }
        else {
            size = 128;
        }
    }

    gridB = n / size + (n % size != 0);
    threads = size == 128 ? 256 : 128;
    std::stringstream ss;
    ss << "_128x" << size;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    void *args[13] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags};

    CUresult res = cuLaunchKernel(nervana_kernels_[name],
                                  gridA, gridB, 1,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

extern "C" bool nervana_hgemm(short *A, short *B, short *C,
                              bool a_t, bool b_t,
                              int m, int n, int k,
                              int lda, int ldb, int ldc,
                              float alpha, float beta,
                              unsigned int *rand_state,
                              bool stochastic_round, bool apply_relu,
                              CUstream stream
                             )
{
    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    int gridA = m / 128 + (m % 128 != 0);

    int gridB, threads;
    std::string name = "hgemm_";

    std::string trans;
    trans += a_t ? 't' : 'n';
    trans += b_t ? 't' : 'n';

    name += trans;

    if ( (trans == "tn" && m % 8 == 0 && n % 8 == 0) ||
         (trans == "nn" && k % 16 == 0 && n % 8 == 0) ||
         (trans == "nt" && k % 16 == 0)) {
         name += "_vec";
    }

    int size = 0;
    if (trans == "nt")
        size = 128;

    if (size == 0) {
        if (n < 384 - 16) {
            int n128 = n % 128;
            if (n128 > 0 && n128 < 112) {
                if (n128 > 48 && n128 <= 64) {
                    int n64 = n / 64;
                    n64 *= gridA / sm_count;
                    if (n64 > 1 || trans == "tn") {
                        size = 64;
                    }
                    else {
                        size = 32;
                    }
                }
                else {
                    size = 32;
                }
            }
            else {
                size = 128;
            }
        }
        else {
            size = 128;
        }
    }

    gridB = n / size + (n % size != 0);
    threads = size == 128 ? 256 : 128;
    std::stringstream ss;
    ss << "_128x" << size;
    name += ss.str();

    int flags = 0;
    flags |= (stochastic_round << 0);
    flags |= (apply_relu << 1);

    void *args[13] = {&rand_state, &A, &B, &C, &lda, &ldb, &ldc, &m, &n, &k, &alpha, &beta, &flags};

    CUresult res = cuLaunchKernel(nervana_kernels_[name],
                                  gridA, gridB, 1,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << name << " " << res << std::endl;
        return false;
    }

    return true;
}

//implementation adapted from the python version
//it upcasts intermediate results to avoid possible numerical problems
std::pair<uint32_t, uint32_t> magic32(uint32_t nmax_, uint32_t d_) {
    uint64_t nmax = nmax_;
    uint64_t d    = d_;

    uint64_t nc = ((nmax + 1) / d) * d - 1;
    uint64_t nbits = 32 - __builtin_clz(nmax);
    for (uint64_t p = 0; p < 2 * nbits + 1; ++p) {
        uint64_t two_powp = 1UL << p;
        if (two_powp > nc * (d - 1 - (two_powp - 1) % d)) {
            int m = (two_powp + d - 1 - two_powp % d) / d;
            return std::make_pair(m, p);
        }
    }
    return std::make_pair(0, 0);
}

extern "C"
bool nervana_sconv_fprop(unsigned int *rand_state,
                         float *O, float *I, float *F,
                         float alpha,
                         int N, int C, int K, int D, int H, int W, int T, int R, int S,
                         int pad_d, int pad_h, int pad_w, int str_d, int str_h, int str_w,
                         CUstream stream
                         )
{
    if(N % 8 != 0 || K % 8 != 0) {
        std::cerr << "sconv_fprop N and K must be multiples of 8" << std::endl;
        return false;
    }

    int M = ceil(static_cast<float>(D - T + 1 + 2 * pad_d) / str_d);
    int P = ceil(static_cast<float>(H - R + 1 + 2 * pad_h) / str_h);
    int Q = ceil(static_cast<float>(W - S + 1 + 2 * pad_w) / str_w);

    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int RS = R * S;
    int RST = RS * T;
    int CRST = C * RST;
    int PQ = P * Q;
    int PM = P * M;
    int PQM = P * Q * M;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;

    if (PQM >= (1 << 16) || CRST + 8 >= (1 << 16)) {
        std::cerr << "Dimensions require more than 16-bits, currently not supported" << std::endl;
        return false;
    }

    int grid_N64 = N / 64 + (N % 64 != 0);
    int grid_K64 = K / 64 + (K % 64 != 0);
    int grid_C64 = CRST / 64 + (CRST % 64 != 0);

    int grid_N128 = N / 128 + (N % 128 != 0);
    int grid_K128 = K / 128 + (K % 128 != 0);
    int grid_C128 = CRST / 128 + (CRST % 128 != 0);

    int grid_P = P;
    int grid_Q = Q / 4;

    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    if (sm_count == 24) {
        int grid_PQ = grid_P * grid_Q;
        if (grid_PQ < 30) {
            grid_P = 6;
            grid_Q = 4;
        }
        else if (grid_PQ < 54) {
            grid_P = 8;
            grid_Q = 6;
        }
        else if (grid_PQ <  78) {
            grid_P = 9;
            grid_Q = 8;
        }
        else if (grid_PQ < 108) {
            grid_P = 12;
            grid_Q = 8;
        }
    }

    grid_P = min(grid_P, P);
    grid_Q = min(grid_Q, Q);

    int grid_PQ = grid_P * grid_Q;
    int grid_PQM = grid_PQ * M;

    int magic_RST_m, magic_RST_p;
    int magic_RS_m, magic_RS_p;
    int magic_S_m, magic_S_p;
    int magic_Q_m, magic_Q_p;
    int magic_PQ_m, magic_PQ_p;

    std::tie(magic_RST_m, magic_RST_p) = magic32(CRST + 8, RST);
    std::tie(magic_RS_m, magic_RS_p) = magic32(RST + 32, RS);
    std::tie(magic_S_m, magic_S_p) = magic32(RS + 32, S);
    std::tie(magic_Q_m, magic_Q_p) = magic32(PQ, P);
    std::tie(magic_PQ_m, magic_PQ_p) = magic32(PQM, PQ);

    int lut_size = (RST / 32 + (RST + 32 != 0)) * 32 * 4;

    int flags = 0;

    void *args[44] = {&rand_state, &O, &I, &F, &alpha, &flags, &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
                                   &C, &CRST, &RST, &magic_RST_m, &magic_RST_p,
                                   &RS, &magic_RS_m, &magic_RS_p,
                                   &S, &magic_S_m, &magic_S_p,
                                   &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
                                   &P, &Q, &PQ, &QN, &PQN, &MPQN,
                                   &magic_Q_m, &magic_Q_p,
                                   &magic_PQ_m, &magic_PQ_p,
                                   &grid_P, &grid_Q, &grid_PQ};


    CUresult res = cuLaunchKernel(nervana_kernels_[std::string("sconv_fprop_K64_N64")],
                                  PQM, grid_K64, grid_N64,
                                  64, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel sconv_fprop_K64_N64" << " " << res << std::endl;
        return false;
    }

    return true;
}

//grad_I = output
//F = filters
//E = deltas from previous layer

extern "C"
bool nervana_sconv_bprop(unsigned int *rand_state,
                         float *grad_I, float *F, float *E,
                         float alpha,
                         int N, int C, int K, int D, int H, int W, int T, int R, int S,
                         int pad_d, int pad_h, int pad_w, int str_d, int str_h, int str_w,
                         CUstream stream
                         )
{
    if(N % 8 != 0 || K % 8 != 0) {
        std::cerr << "sconv_bprop N and K must be multiples of 8" << std::endl;
        return false;
    }

    int M = ceil(static_cast<float>(D - T + 1 + 2 * pad_d) / str_d);
    int P = ceil(static_cast<float>(H - R + 1 + 2 * pad_h) / str_h);
    int Q = ceil(static_cast<float>(W - S + 1 + 2 * pad_w) / str_w);

    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int RS = R * S;
    int RST = RS * T;
    int CRST = C * RST;
    int PQ = P * Q;
    int PM = P * M;
    int PQM = P * Q * M;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;

    if (PQM >= (1 << 16) || CRST + 8 >= (1 << 16)) {
        std::cerr << "Dimensions require more than 16-bits, currently not supported" << std::endl;
        return false;
    }

    int grid_N64 = N / 64 + (N % 64 != 0);
    int grid_K64 = K / 64 + (K % 64 != 0);
    int grid_C64 = CRST / 64 + (CRST % 64 != 0);

    int grid_N128 = N / 128 + (N % 128 != 0);
    int grid_K128 = K / 128 + (K % 128 != 0);
    int grid_C128 = CRST / 128 + (CRST % 128 != 0);

    int grid_P = P;
    int grid_Q = Q / 4;

    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    if (sm_count == 24) {
        int grid_PQ = grid_P * grid_Q;
        if (grid_PQ < 30) {
            grid_P = 6;
            grid_Q = 4;
        }
        else if (grid_PQ < 54) {
            grid_P = 8;
            grid_Q = 6;
        }
        else if (grid_PQ <  78) {
            grid_P = 9;
            grid_Q = 8;
        }
        else if (grid_PQ < 108) {
            grid_P = 12;
            grid_Q = 8;
        }
    }

    grid_P = min(grid_P, P);
    grid_Q = min(grid_Q, Q);

    int grid_PQ = grid_P * grid_Q;
    int grid_PQM = grid_PQ * M;

    int magic_RST_m, magic_RST_p;
    int magic_RS_m, magic_RS_p;
    int magic_S_m, magic_S_p;
    int magic_Q_m, magic_Q_p;
    int magic_PQ_m, magic_PQ_p;

    std::tie(magic_RST_m, magic_RST_p) = magic32(CRST + 8, RST);
    std::tie(magic_RS_m, magic_RS_p) = magic32(RST + 32, RS);
    std::tie(magic_S_m, magic_S_p) = magic32(RS + 32, S);
    std::tie(magic_Q_m, magic_Q_p) = magic32(PQ, P);
    std::tie(magic_PQ_m, magic_PQ_p) = magic32(PQM, PQ);

    int lut_size = (RST / 32 + (RST + 32 != 0)) * 32 * 4;

    int flags = 0;

    void *args[44] = {&rand_state, &grad_I, &F, &E, &alpha, &flags, &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
                                   &C, &CRST, &RST, &magic_RST_m, &magic_RST_p,
                                   &RS, &magic_RS_m, &magic_RS_p,
                                   &S, &magic_S_m, &magic_S_p,
                                   &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
                                   &P, &Q, &PQ, &QN, &PQN, &MPQN,
                                   &magic_Q_m, &magic_Q_p,
                                   &magic_PQ_m, &magic_PQ_p,
                                   &grid_P, &grid_Q, &grid_PQ};


    CUresult res = cuLaunchKernel(nervana_kernels_[std::string("sconv_bprop_C128_N64")],
                                  PQM, grid_C128, grid_N64,
                                  128, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel sconv_fprop_K64_N64" << " " << res << std::endl;
        return false;
    }

    return true;
}

//F = update to filters (or updated filters, not sure)
//I = input
//E = deltas

extern "C"
bool nervana_sconv_updat(unsigned int *rand_state,
                         float *F, float *I, float *E,
                         float alpha,
                         int N, int C, int K, int D, int H, int W, int T, int R, int S,
                         int pad_d, int pad_h, int pad_w, int str_d, int str_h, int str_w,
                         CUstream stream
                         )
{
    if(N % 8 != 0 || K % 8 != 0) {
        std::cerr << "sconv_updat N and K must be multiples of 8" << std::endl;
        return false;
    }

    int M = ceil(static_cast<float>(D - T + 1 + 2 * pad_d) / str_d);
    int P = ceil(static_cast<float>(H - R + 1 + 2 * pad_h) / str_h);
    int Q = ceil(static_cast<float>(W - S + 1 + 2 * pad_w) / str_w);

    int WN = W * N;
    int HWN = H * WN;
    int DHWN = D * HWN;
    int RS = R * S;
    int RST = RS * T;
    int CRST = C * RST;
    int PQ = P * Q;
    int PM = P * M;
    int PQM = P * Q * M;
    int QN = Q * N;
    int PQN = P * QN;
    int MPQN = M * PQN;

    if (PQM >= (1 << 16) || CRST + 8 >= (1 << 16)) {
        std::cerr << "Dimensions require more than 16-bits, currently not supported" << std::endl;
        return false;
    }

    int grid_N64 = N / 64 + (N % 64 != 0);
    int grid_K64 = K / 64 + (K % 64 != 0);
    int grid_C64 = CRST / 64 + (CRST % 64 != 0);

    int grid_N128 = N / 128 + (N % 128 != 0);
    int grid_K128 = K / 128 + (K % 128 != 0);
    int grid_C128 = CRST / 128 + (CRST % 128 != 0);

    int grid_P = P;
    int grid_Q = Q / 4;

    int sm_count;
    {
        std::lock_guard<std::mutex> lock(nervana_sm_count_mutex_);

        CUdevice device;
        CUresult res = cuCtxGetDevice(&device);
        if (res != CUDA_SUCCESS) {
            return false;
        }
        auto count = nervana_sm_counts_.find(device);
        if (count != nervana_sm_counts_.end()) {
            sm_count = count->second;
        }
        else {
            int pi;
            res = cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
            if (res != CUDA_SUCCESS) {
                return false;
            }
            sm_count = pi;
            nervana_sm_counts_[device] = pi;
        }
    }

    if (sm_count == 24) {
        int grid_PQ = grid_P * grid_Q;
        if (grid_PQ < 30) {
            grid_P = 6;
            grid_Q = 4;
        }
        else if (grid_PQ < 54) {
            grid_P = 8;
            grid_Q = 6;
        }
        else if (grid_PQ <  78) {
            grid_P = 9;
            grid_Q = 8;
        }
        else if (grid_PQ < 108) {
            grid_P = 12;
            grid_Q = 8;
        }
    }

    grid_P = min(grid_P, P);
    grid_Q = min(grid_Q, Q);

    int grid_PQ = grid_P * grid_Q;
    int grid_PQM = grid_PQ * M;

    dim3 grid;
    int threads;
    std::string update_size;
    if (CRST <= 64 ||  K <= 64 || (K % 64 == 0 && K % 128 != 0)) {
        grid.x = grid_PQM;
        grid.y = grid_C128;
        grid.z = grid_K64;
        update_size = "C128_K64";
        threads = 128;
    }
    else {
        grid.x = grid_PQM;
        grid.y = grid_C128;
        grid.z = grid_K128;
        update_size = "C128_K128";
        threads = 256;
    }

    int magic_RST_m, magic_RST_p;
    int magic_RS_m, magic_RS_p;
    int magic_S_m, magic_S_p;
    int magic_Q_m, magic_Q_p;
    int magic_PQ_m, magic_PQ_p;
    int magic_Qu_m, magic_Qu_p;
    int magic_PQu_m, magic_PQu_p;

    std::tie(magic_RST_m, magic_RST_p) = magic32(CRST + 8, RST);
    std::tie(magic_RS_m, magic_RS_p) = magic32(RST + 32, RS);
    std::tie(magic_S_m, magic_S_p) = magic32(RS + 32, S);
    std::tie(magic_Q_m, magic_Q_p) = magic32(PQ, P);
    std::tie(magic_PQ_m, magic_PQ_p) = magic32(PQM, PQ);
    std::tie(magic_Qu_m, magic_Qu_p) = magic32(grid_PQ, grid_Q);
    std::tie(magic_PQu_m, magic_PQu_p) = magic32(grid_PQM, grid_PQ);

    int lut_size = (RST / 32 + (RST + 32 != 0)) * 32 * 4;

    int flags = 0;

    void *args[44] = {&rand_state, &F, &I, &E, &alpha, &flags,
                                   &N, &K, &D, &H, &W, &WN, &HWN, &DHWN,
                                   &C, &CRST, &RST, &magic_RST_m, &magic_RST_p,
                                   &RS, &magic_RS_m, &magic_RS_p,
                                   &S, &magic_S_m, &magic_S_p,
                                   &pad_d, &pad_h, &pad_w, &str_d, &str_h, &str_w,
                                   &P, &Q, &PQ, &QN, &PQN, &MPQN,
                                   &magic_Qu_m, &magic_Qu_p,
                                   &magic_PQu_m, &magic_PQu_p,
                                   &grid_P, &grid_Q, &grid_PQ};


    std::string kernel_name = std::string("sconv_updat_") + update_size;
    CUresult res = cuLaunchKernel(nervana_kernels_[kernel_name],
                                  grid.x, grid.y, grid.z,
                                  threads, 1, 1,
                                  0,
                                  stream, args, NULL);

    if (res != CUDA_SUCCESS) {
        std::cerr << "Error launching kernel " << kernel_name << " " << res << std::endl;
        return false;
    }

    return true;
}

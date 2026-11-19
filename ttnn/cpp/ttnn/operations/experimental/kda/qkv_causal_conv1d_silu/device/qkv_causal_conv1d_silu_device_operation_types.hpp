// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    uint32_t channel_chunk_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // QWEN36_GDN_CONV_KDA2 (cfx patches/tt-metal/0035), read per call from the environment by qkv_causal_conv1d_silu()
    // and hashed with the other attributes, so both paths can be cached in one process (the warmup self-test runs
    // both): 0 = the stock programs; 1 = read-once kernels with the stock silu_tile; 2 = read-once kernels with
    // silu_tile<false> (matches the c98 compute overlay on the stock path).
    uint32_t reads_once = 0;
    // QWEN36_GDN_CONV_KDA2_MUSTFAIL=1 with reads_once != 0: the read-once reader stages every row one row late.
    uint32_t reads_once_mustfail = 0;
};

struct QkvCausalConv1dSiluInputs {
    Tensor input;
    Tensor history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
};

}  // namespace ttnn::experimental::prim

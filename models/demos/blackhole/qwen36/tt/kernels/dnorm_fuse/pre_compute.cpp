// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_FUSE (dnorm_fuse.py): compute half of the fused PRE pass, one 32-row tile row per core.
// The same LLK sequence as the image's rmsnorm_pre_allgather.cpp (1D factory, bf16 input, no residual, HiFi2 with
// fp32 DEST): x**2 = mul_tiles(h, h) packed tile by tile into the fp32 CB_X2, then the FPU row reduce (AVG with the
// reader's 1.0 scaler, BulkWaitBulkPop over WT tiles, INPUT_AND_OUTPUT reconfig, Fast fp32 mode) packed as bf16 into
// CB_ST. Per-tile results do not depend on how tiles are grouped per DEST acquire, and the reduce accumulates the WT
// tiles in the same order, so the stats are the same bytes.
// h is read in place from CB_H (the hs shard); chunk j's arrival is signalled on counter CB_C0 (even j) / CB_C1 (odd j).
//
// Compile args: {CB_H, CB_C0, CB_C1, CB_SC, CB_X2, CB_ST, WT, CHUNK, BLK}

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t CB_H = get_compile_time_arg_val(0);
    constexpr uint32_t CB_C0 = get_compile_time_arg_val(1);
    constexpr uint32_t CB_C1 = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(3);
    constexpr uint32_t CB_X2 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(5);
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(7);
    constexpr uint32_t BLK = get_compile_time_arg_val(8);
    static_assert(CHUNK % BLK == 0, "a DEST block never straddles a chunk");
    constexpr uint32_t NCHUNK = (WT + CHUNK - 1) / CHUNK;

    compute_kernel_hw_startup(CB_H, CB_SC, CB_X2);

    reconfig_data_format(CB_H, CB_H);
    pack_reconfig_data_format(CB_X2);
    mul_init(CB_H, CB_H);
    for (uint32_t wt = 0; wt < WT; wt += BLK) {
        if (wt % CHUNK == 0) {
            const uint32_t j = wt / CHUNK;
            cb_wait_front((j & 1) ? CB_C1 : CB_C0, (j >> 1) + 1);  // cumulative: chunk j has landed
        }
        const uint32_t n = (wt + BLK <= WT) ? BLK : WT - wt;
        cb_reserve_back(CB_X2, n);
        tile_regs_acquire();
        for (uint32_t i = 0; i < n; ++i) {
            mul_tiles(CB_H, CB_H, wt + i, wt + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i, CB_X2);
        }
        tile_regs_release();
        cb_push_back(CB_X2, n);
    }

    compute_kernel_lib::reduce<
        PoolType::AVG,
        ReduceDim::REDUCE_ROW,
        CB_X2,
        CB_SC,
        CB_ST,
        compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
        compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
        ReduceFp32Mode::Fast>(compute_kernel_lib::ReduceInputBlockShape::row(WT));

    cb_pop_front(CB_SC, 1);
    cb_pop_front(CB_C0, (NCHUNK + 1) / 2);
    cb_pop_front(CB_C1, NCHUNK / 2);
}

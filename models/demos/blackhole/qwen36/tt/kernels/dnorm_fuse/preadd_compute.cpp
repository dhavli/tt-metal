// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_ADD (dnorm_fuse.py): compute of the PRE pass with the residual add folded in.
//   h   = add_tiles(a, b)  chunk by chunk from the two roles' CB pairs, packed (bf16) into CB_H = the hs shard.
//         ttsim: add_tiles with fp32 or 16-bit DEST is bitwise equal to ttnn.add (bf16+bf16 and bf16+bf8, 2026-10-19).
//   x^2 = mul_tiles(h, h) -> fp32 CB_X2, then the image's FPU row reduce -> bf16 CB_ST (as pre_compute.cpp).
//
// Compile args: {CB_A0, CB_B0, CB_A1, CB_B1, CB_H, CB_SC, CB_X2, CB_ST, WT, CHUNK, BLK}

#include <cstdint>

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr uint32_t CB_A0 = get_compile_time_arg_val(0);
    constexpr uint32_t CB_B0 = get_compile_time_arg_val(1);
    constexpr uint32_t CB_A1 = get_compile_time_arg_val(2);
    constexpr uint32_t CB_B1 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_H = get_compile_time_arg_val(4);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(5);
    constexpr uint32_t CB_X2 = get_compile_time_arg_val(6);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(7);
    constexpr uint32_t WT = get_compile_time_arg_val(8);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(9);
    constexpr uint32_t BLK = get_compile_time_arg_val(10);
    static_assert(CHUNK % BLK == 0, "a DEST block never straddles a chunk");
    constexpr uint32_t NCHUNK = (WT + CHUNK - 1) / CHUNK;

    compute_kernel_hw_startup(CB_A0, CB_B0, CB_H);

    for (uint32_t j = 0; j < NCHUNK; ++j) {
        const uint32_t ca = (j & 1) ? CB_A1 : CB_A0;
        const uint32_t cb = (j & 1) ? CB_B1 : CB_B0;
        const uint32_t t0 = j * CHUNK;
        const uint32_t n = (t0 + CHUNK < WT) ? CHUNK : WT - t0;
        reconfig_data_format(ca, cb);
        pack_reconfig_data_format(CB_H);
        add_tiles_init(ca, cb);
        cb_wait_front(ca, n);
        cb_wait_front(cb, n);
        cb_reserve_back(CB_H, n);
        for (uint32_t i0 = 0; i0 < n; i0 += BLK) {
            const uint32_t m = (i0 + BLK <= n) ? BLK : n - i0;
            tile_regs_acquire();
            for (uint32_t i = 0; i < m; ++i) {
                add_tiles(ca, cb, i0 + i, i0 + i, i);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < m; ++i) {
                pack_tile(i, CB_H);
            }
            tile_regs_release();
        }
        cb_push_back(CB_H, n);
        cb_pop_front(ca, n);
        cb_pop_front(cb, n);
    }

    reconfig_data_format(CB_H, CB_H);
    pack_reconfig_data_format(CB_X2);
    mul_init(CB_H, CB_H);
    cb_wait_front(CB_H, WT);
    for (uint32_t wt = 0; wt < WT; wt += BLK) {
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
}

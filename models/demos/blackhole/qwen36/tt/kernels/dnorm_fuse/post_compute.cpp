// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_FUSE (dnorm_fuse.py): compute half of the fused POST pass, one 32-row tile row per core.
// The same LLK sequence as the image's rmsnorm_post_allgather (gamma, no beta, HiFi2, fp32 DEST, non-legacy rsqrt):
//   var   = reduce<AVG, REDUCE_ROW>(stats[STATS_COLS])      (default WaitAndPopPerTile, INPUT_AND_OUTPUT, Fast)
//   recip = rsqrt(var + eps)                                (add_tiles + rsqrt_tile, packed fp32)
//   xn    = mul_tiles_bcast_cols(h, recip)                  (packed fp32 into CB_XN, all WT tiles)
//   out   = mul_tiles_bcast_rows(xn, gamma)                 (packed in the output dtype)
// h is read in place from CB_H (this core's hs shard, written by the PRE pass); it is never waited on or popped.
//
// Compile args: {CB_H, CB_STATS, CB_G, CB_EPS, CB_SC, CB_VAR, CB_RECIP, CB_XN, CB_OUT, WT, BLK, STATS_COLS}

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

ALWI void ACQ() {
    tile_regs_acquire();
    tile_regs_wait();
}
ALWI void REL() {
    tile_regs_commit();
    tile_regs_release();
}

void kernel_main() {
    constexpr uint32_t CB_H = get_compile_time_arg_val(0);
    constexpr uint32_t CB_STATS = get_compile_time_arg_val(1);
    constexpr uint32_t CB_G = get_compile_time_arg_val(2);
    constexpr uint32_t CB_EPS = get_compile_time_arg_val(3);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(4);
    constexpr uint32_t CB_VAR = get_compile_time_arg_val(5);
    constexpr uint32_t CB_RECIP = get_compile_time_arg_val(6);
    constexpr uint32_t CB_XN = get_compile_time_arg_val(7);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(8);
    constexpr uint32_t WT = get_compile_time_arg_val(9);
    constexpr uint32_t BLK = get_compile_time_arg_val(10);
    constexpr uint32_t STATS_COLS = get_compile_time_arg_val(11);
    constexpr bool LEGACY_RSQRT = false;
    static_assert(WT % BLK == 0, "BLK divides WT");

    compute_kernel_hw_startup(CB_H, CB_H, CB_VAR);

    cb_wait_front(CB_SC, 1);
    cb_wait_front(CB_EPS, 1);

    compute_kernel_lib::reduce<PoolType::AVG, ReduceDim::REDUCE_ROW, CB_STATS, CB_SC, CB_VAR>(
        compute_kernel_lib::ReduceInputBlockShape::row(STATS_COLS));

    // 1/sqrt(var + eps)
    cb_wait_front(CB_VAR, 1);
    cb_reserve_back(CB_RECIP, 1);
    reconfig_data_format(CB_VAR, CB_EPS);
    pack_reconfig_data_format(CB_RECIP);
    add_init(CB_VAR, CB_EPS);
    ACQ();
    add_tiles(CB_VAR, CB_EPS, 0, 0, 0);
    rsqrt_tile_init<LEGACY_RSQRT>();
    rsqrt_tile<LEGACY_RSQRT>(0);
    pack_tile(0, CB_RECIP);
    REL();
    cb_push_back(CB_RECIP, 1);
    cb_pop_front(CB_VAR, 1);

    // xn = h * recip (bcast cols)
    reconfig_data_format(CB_H, CB_RECIP);
    pack_reconfig_data_format(CB_XN);
    mul_bcast_cols_init(CB_H, CB_RECIP);
    cb_wait_front(CB_RECIP, 1);
    for (uint32_t wt = 0; wt < WT; wt += BLK) {
        cb_reserve_back(CB_XN, BLK);
        ACQ();
        for (uint32_t i = 0; i < BLK; ++i) {
            mul_tiles_bcast_cols(CB_H, CB_RECIP, wt + i, 0, i);
            pack_tile(i, CB_XN);
        }
        REL();
        cb_push_back(CB_XN, BLK);
    }
    cb_pop_front(CB_RECIP, 1);

    // out = xn * gamma (bcast rows)
    reconfig_data_format(CB_XN, CB_G);
    pack_reconfig_data_format(CB_OUT);
    cb_wait_front(CB_G, WT);
    mul_bcast_rows_init(CB_XN, CB_G);
    for (uint32_t wt = 0; wt < WT; wt += BLK) {
        cb_wait_front(CB_XN, BLK);
        cb_reserve_back(CB_OUT, BLK);
        ACQ();
        for (uint32_t i = 0; i < BLK; ++i) {
            mul_tiles_bcast_rows(CB_XN, CB_G, i, wt + i, i);
            pack_tile(i, CB_OUT);
        }
        REL();
        cb_push_back(CB_OUT, BLK);
        cb_pop_front(CB_XN, BLK);
    }

    cb_pop_front(CB_EPS, 1);
    cb_pop_front(CB_SC, 1);
    cb_pop_front(CB_G, WT);
}

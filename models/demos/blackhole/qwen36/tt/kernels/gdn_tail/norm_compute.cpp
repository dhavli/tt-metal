// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_TAIL (gdn_tail.py): compute half of the fused GDN per-head RMSNorm + head->token relayout.
// The same LLK sequence as the image's layernorm.cpp on the path ttnn.rms_norm(o, weight=norm_w, eps) takes for the
// GDN scan output (interleaved factory, RMSNORM, FUSE_GAMMA, no beta, no pre-add, not large-tensor; rms_norm's default
// compute config: HiFi4, approx, fp32 DEST off, so the intermediates are Float16_b and block_size is 8):
//   xmm2 = mul_tiles(x, x)                                     (packed bf16)
//   ex2  = reduce<SUM, REDUCE_ROW>(xmm2, scaler 1.0) * (1/W)   (numeric::row_wise_mean: mul_unary_tile epilogue)
//   r    = rsqrt(ex2 + eps)                                    (add_tiles + rsqrt_tile<non-legacy>, packed bf16)
//   xn   = mul_tiles_bcast_cols(x, r)                          (packed bf16 into CB_FU)
//   out  = mul_tiles_bcast_rows(xn, gamma)                     (packed in the output format, fp32 here)
// with the same reconfig_data_format / pack_reconfig_data_format calls in the same order, so the same bytes. Each
// tile row is independent; the relayout lives in the writer's addressing (gdn_tail norm_dm.cpp ROLE 1).
//
// Compile args: {CB_IN, CB_SC, CB_EPS, CB_G, CB_X2, CB_EX2, CB_EX2PE, CB_FU, CB_OUT, WT, W}
// Runtime args: {nrows}

#include <cstdint>

#define BCAST_LLKOP EltwiseBinaryType::ELWMUL
#define BCAST_DIM BroadcastType::COL

#include "api/compute/compute_kernel_api.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_EPS = get_compile_time_arg_val(2);
    constexpr uint32_t CB_G = get_compile_time_arg_val(3);
    constexpr uint32_t CB_X2 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_EX2 = get_compile_time_arg_val(5);
    constexpr uint32_t CB_EX2PE = get_compile_time_arg_val(6);
    constexpr uint32_t CB_FU = get_compile_time_arg_val(7);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(8);
    constexpr uint32_t WT = get_compile_time_arg_val(9);
    constexpr uint32_t W = get_compile_time_arg_val(10);
    constexpr bool LEGACY_RSQRT = false;
    static_assert(WT <= 8, "one block (block_size 8 with fp32 DEST off)");
    static_assert(W == WT * 32, "full tiles only");

    const uint32_t nrows = get_arg_val<uint32_t>(0);

    union {
        float f;
        uint32_t u;
    } inv_w;
    inv_w.f = 1.0f / W;

    // RMSNORM without a fused pre-add: xmm IS the input FIFO.
    compute_kernel_hw_startup(CB_IN, CB_IN, CB_X2);

    cb_wait_front(CB_EPS, 1);

    for (uint32_t row = 0; row < nrows; ++row) {
        reconfig_data_format(CB_IN, CB_IN);
        pack_reconfig_data_format(CB_X2);

        // x**2
        mul_init(CB_IN, CB_IN);
        cb_wait_front(CB_IN, WT);
        tile_regs_acquire();
        for (uint32_t i = 0; i < WT; ++i) {
            mul_tiles(CB_IN, CB_IN, i, i, i);
        }
        tile_regs_commit();
        cb_reserve_back(CB_X2, WT);
        tile_regs_wait();
        for (uint32_t i = 0; i < WT; ++i) {
            pack_tile(i, CB_X2);
        }
        tile_regs_release();
        cb_push_back(CB_X2, WT);
        reconfig_data_format(CB_IN, CB_X2, CB_IN, CB_SC);

        // E[x**2] (numeric::row_wise_mean, FullBlockWithPopPolicy)
        cb_wait_front(CB_SC, 1);
        reconfig_data_format(CB_X2, CB_SC);
        tile_regs_acquire();
        reconfig_data_format(CB_SC, CB_X2);  // swap_operands (REDUCE_ROW SUM)
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_X2, CB_SC, CB_EX2);
        cb_wait_front(CB_X2, WT);
        for (uint32_t j = 0; j < WT; ++j) {
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(CB_X2, CB_SC, j, 0, 0);
        }
        cb_pop_front(CB_X2, WT);
        reduce_uninit();
        reconfig_data_format(CB_X2, CB_SC);
        binop_with_scalar_tile_init();
        mul_unary_tile(0, inv_w.u);
        tile_regs_commit();
        tile_regs_wait();
        cb_reserve_back(CB_EX2, 1);
        pack_reconfig_data_format(CB_EX2);
        pack_tile(0, CB_EX2);
        tile_regs_release();
        cb_push_back(CB_EX2, 1);
        cb_wait_front(CB_EX2, 1);

        // 1/sqrt(E[x**2] + eps)
        cb_wait_front(CB_EX2, 1);
        reconfig_data_format(CB_EX2, CB_EPS);
        tile_regs_acquire();
        add_init(CB_EX2, CB_EPS);
        add_tiles(CB_EX2, CB_EPS, 0, 0, 0);
        rsqrt_tile_init<LEGACY_RSQRT>();
        rsqrt_tile<LEGACY_RSQRT>(0);
        tile_regs_commit();
        cb_pop_front(CB_EX2, 1);
        cb_reserve_back(CB_EX2PE, 1);
        pack_reconfig_data_format(CB_EX2PE);
        tile_regs_wait();
        pack_tile(0, CB_EX2PE);
        tile_regs_release();
        cb_push_back(CB_EX2PE, 1);

        // x * r (bcast cols) -> CB_FU, then * gamma (bcast rows) -> CB_OUT
        cb_wait_front(CB_EX2PE, 1);
        reconfig_data_format(CB_IN, CB_EX2PE);
        pack_reconfig_data_format(CB_FU);
        cb_reserve_back(CB_FU, WT);
        reconfig_data_format_srca(CB_FU, CB_IN);
        tile_regs_acquire();
        mul_bcast_cols_init(CB_IN, CB_EX2PE);
        for (uint32_t i = 0; i < WT; ++i) {
            mul_tiles_bcast_cols(CB_IN, CB_EX2PE, i, 0, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < WT; ++i) {
            pack_tile(i, CB_FU);
        }
        tile_regs_release();
        cb_push_back(CB_FU, WT);
        reconfig_data_format_srca(CB_IN, CB_FU);

        pack_reconfig_data_format(CB_OUT);
        reconfig_data_format_srcb(CB_EX2PE, CB_G);
        cb_wait_front(CB_G, WT);
        cb_wait_front(CB_FU, WT);
        tile_regs_acquire();
        mul_bcast_rows_init(CB_FU, CB_G);
        for (uint32_t i = 0; i < WT; ++i) {
            mul_tiles_bcast_rows(CB_FU, CB_G, i, i, i);
        }
        tile_regs_commit();
        cb_pop_front(CB_FU, WT);
        cb_reserve_back(CB_OUT, WT);
        tile_regs_wait();
        for (uint32_t i = 0; i < WT; ++i) {
            pack_tile(i, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, WT);

        cb_pop_front(CB_EX2PE, 1);
        cb_pop_front(CB_IN, WT);
    }
    cb_pop_front(CB_SC, 1);
    cb_pop_front(CB_EPS, 1);
    cb_pop_front(CB_G, WT);
}

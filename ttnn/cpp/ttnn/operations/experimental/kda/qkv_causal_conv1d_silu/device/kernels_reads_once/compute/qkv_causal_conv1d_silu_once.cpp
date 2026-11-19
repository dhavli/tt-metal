// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// QWEN36_GDN_CONV_KDA2 (cfx patches/tt-metal/0035): the stock compute's per-tile op sequence, fed by the read-once
// reader. Two changes, both in what is waited on and popped, none in the arithmetic:
//   - The taps arrive once per channel block per core (work item `work` is (block work / Mt, tile row work % Mt))
//     and stay in the weights DFB until the block changes.
//   - act_rm holds row-sized entries; the reader pushes 35 rows per item (three look-back rows plus the tile row).
//     Tap t tilizes stage rows t..t+31: the asymmetric-page tilize reads 32 rows at the read pointer, and taps 0..2
//     pop one row each, tap 3 pops the remaining 32.
// silu_fast = 1 selects silu_tile<false> (the cfx c98 bf16-accuracy sigmoid that the served compute overlay
// bench/probes/kda_conv_fid/compute.fast.cpp uses on the stock path), 0 the stock silu_tile.

#include "api/compute/bcast.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

template <uint32_t block_ct, uint32_t Mt, uint32_t silu_fast>
TT_KERNEL void compute(uint32_t wi_start, uint32_t wi_count) {
    // Kimi-K3 uses a fixed four-tap causal convolution, with three preceding rows supplied by history.
    constexpr uint32_t tap_count = 4;
    constexpr uint32_t history_rows = 3;
    constexpr uint32_t stage_rows = 32 + history_rows;
    constexpr uint32_t no_block = 0xFFFFFFFFu;
    compute_kernel_hw_startup(dfb::act_rm, dfb::act_tile, dfb::output);
    DataflowBuffer stage(dfb::act_rm);
    DataflowBuffer activation(dfb::act_tile);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer partial(dfb::partial);
    DataflowBuffer output(dfb::output);
    silu_tile_init();

    uint32_t cur_blk = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t blk = (wi_start + item) / Mt;
        if (blk != cur_blk) {
            // Release the previous block's taps first: that pop is what lets the reader load this block's into the
            // one-block-deep weights DFB.
            if (cur_blk != no_block) {
                weights.pop_front(tap_count * block_ct);
            }
            weights.wait_front(tap_count * block_ct);
            cur_blk = blk;
        }
        // The whole stage is resident before the first window is tilized.
        stage.wait_front(stage_rows);
        for (uint32_t tap = 0; tap < tap_count; ++tap) {
            // Window `tap` = stage rows tap..tap+31 (the read pointer sits at row `tap`). Pop 1 row after taps 0..2,
            // the remaining 32 after tap 3: 35 per item.
            constexpr uint32_t last_pop = stage_rows - (tap_count - 1);
            compute_kernel_lib::tilize<block_ct, dfb::act_rm, dfb::act_tile>(1, tap + 1 == tap_count ? last_pop : 1);
            activation.wait_front(block_ct);

            const bool is_final_tap = tap + 1 == tap_count;
            const uint32_t destination_dfb = is_final_tap ? dfb::output : dfb::partial;
            if (tap != 0) {
                partial.wait_front(block_ct);
            }

            reconfig_data_format_srca(dfb::act_tile);
            reconfig_data_format_srcb(dfb::weights);
            if (tap == 0) {
                mul_bcast_rows_init(dfb::act_tile, dfb::weights);
            }
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                if (is_final_tap) {
                    output.reserve_back(1);
                } else {
                    partial.reserve_back(1);
                }
                tile_regs_acquire();
                if (tap != 0) {
                    mul_bcast_rows_init(dfb::act_tile, dfb::weights);
                }
                mul_tiles_bcast_rows(dfb::act_tile, dfb::weights, ct, tap * block_ct + ct, 0);

                if (tap != 0) {
                    reconfig_data_format_srca(dfb::partial);
                    add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial);
                    add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::partial, 0, 0);
                    // The partial add binds srcA to the accumulator; restore activation for the next multiply.
                    reconfig_data_format_srca(dfb::act_tile);
                }
                if (is_final_tap) {
                    if constexpr (silu_fast != 0) {
                        silu_tile<false>(0);
                    } else {
                        silu_tile(0);
                    }
                }
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(0, destination_dfb);
                if (is_final_tap) {
                    output.push_back(1);
                } else {
                    partial.push_back(1);
                }
                if (tap != 0) {
                    partial.pop_front(1);
                }
                tile_regs_release();
            }
            activation.pop_front(block_ct);
        }
    }
    if (cur_blk != no_block) {
        weights.pop_front(tap_count * block_ct);
    }
}

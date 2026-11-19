// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// QWEN36_GDN_CONV_KDA2 (cfx patches/tt-metal/0035, the read-once idea of upstream tt-metal #58493, without its NoC
// loopback copies). The stock reader reads every activation row four times (once per tap shift) and the four tap
// rows of a channel block once per item; here:
//   - Work item `work` is (channel block blk = work / Mt, tile row mt = work % Mt). distribute_prep gives each core a
//     contiguous range, so a core walks consecutive tile rows of ONE block and crosses a block boundary at most once
//     per Mt items. The block's taps are read when the block changes and stay in the weights DFB; the compute pops
//     them when the block changes.
//   - Each item's 35 source rows (three look-back rows plus the tile row) are read from DRAM ONCE into a stage of
//     35 row-sized entries of act_rm (ring of two stages). The compute tilizes its four tap-shifted 32-row windows
//     straight out of the stage by popping one row between taps (asymmetric-page tilize), so no L1->L1 copy is made
//     (own-core NoC loopback is a dead idiom on our cards: LEVERS c91).
// The compute sees, per tap t, exactly the rows it saw before (source rows mt*32 + row + t - 3), so its tilize and
// arithmetic are unchanged and the outputs are bit-identical.
// mustfail = 1: every stage row is read one row late (clamped to the last row), so every output differs: a timing-
// free proof that these ELFs ran, and the input the warmup self-test must refuse.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <uint32_t block_ct, typename Tap0Accessor, typename Tap1Accessor, typename Tap2Accessor, typename Tap3Accessor>
FORCE_INLINE void load_weight_block(
    Noc& noc,
    DataflowBuffer& weights,
    const Tap0Accessor& tap0,
    const Tap1Accessor& tap1,
    const Tap2Accessor& tap2,
    const Tap3Accessor& tap3,
    uint32_t tile_bytes,
    uint32_t ct_start) {
    weights.reserve_back(4 * block_ct);
    for (uint32_t ct = 0; ct < block_ct; ++ct) {
        const uint32_t source_ct = ct_start + ct;
        // The weight DFB is laid out as [tap][channel tile].
        noc.async_read(tap0, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = ct * tile_bytes});
        noc.async_read(
            tap1, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (block_ct + ct) * tile_bytes});
        noc.async_read(
            tap2, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (2 * block_ct + ct) * tile_bytes});
        noc.async_read(
            tap3, weights, tile_bytes, {.page_id = source_ct}, {.offset_bytes = (3 * block_ct + ct) * tile_bytes});
    }
    noc.async_read_barrier();
    weights.push_back(4 * block_ct);
}

template <uint32_t block_ct, uint32_t Mt, uint32_t mustfail>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer stage(dfb::act_rm);  // entries are rows of the block (block_ct * 64 bytes)
    Noc noc;

    constexpr uint32_t tile_width = 32;
    constexpr uint32_t tile_height = 32;
    constexpr uint32_t history_rows = 3;
    constexpr uint32_t stage_rows = tile_height + history_rows;
    constexpr uint32_t block_row_bytes = block_ct * tile_width * sizeof(uint16_t);
    constexpr uint32_t block_offset_scale = tile_width * sizeof(uint16_t);
    constexpr int32_t last_row = static_cast<int32_t>(Mt * tile_height) - 1;
    constexpr uint32_t no_block = 0xFFFFFFFFu;
    const uint32_t tile_bytes = weights.get_entry_size();

    uint32_t cur_blk = no_block;
    for (uint32_t item = 0; item < wi_count; ++item) {
        const uint32_t work = wi_start + item;
        const uint32_t blk = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = blk * block_ct;

        // Stage first: it needs only a free stage slot (the compute frees one per finished item), so the next item's
        // rows land while the compute still works on this one.
        stage.reserve_back(stage_rows);
        for (uint32_t j = 0; j < stage_rows; ++j) {
            int32_t source_row = static_cast<int32_t>(mt * tile_height + j) - static_cast<int32_t>(history_rows);
            if constexpr (mustfail != 0) {
                source_row = source_row + 1 > last_row ? last_row : source_row + 1;
            }
            if (source_row < 0) {
                noc.async_read(
                    history,
                    stage,
                    block_row_bytes,
                    {.page_id = static_cast<uint32_t>(source_row + static_cast<int32_t>(history_rows)),
                     .offset_bytes = ct_start * block_offset_scale},
                    {.offset_bytes = j * block_row_bytes});
            } else {
                noc.async_read(
                    input,
                    stage,
                    block_row_bytes,
                    {.page_id = static_cast<uint32_t>(source_row), .offset_bytes = ct_start * block_offset_scale},
                    {.offset_bytes = j * block_row_bytes});
            }
        }
        noc.async_read_barrier();
        stage.push_back(stage_rows);

        if (blk != cur_blk) {
            // The weights DFB holds one block: this reserve waits until the compute has popped the previous block's
            // taps, which it does at the start of its first item of the new block. That item's stage was pushed just
            // above, so nothing the compute needs waits on us.
            load_weight_block<block_ct>(noc, weights, tap0, tap1, tap2, tap3, tile_bytes, ct_start);
            cur_blk = blk;
        }
    }
}

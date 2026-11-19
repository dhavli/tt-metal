// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_ATTN_POST_FOLD (attention/attn_post_fold.py): reader half (RISCV_0). Core b = batch row b. Reads the row's
// WT attention tiles ([1, B, NH->32, HD] bf16 TILE, L1 interleaved: tile id b*WT + t) into CB_ATTN and the row's WT
// gate tiles (same shape, DRAM) into CB_GATE, one tile row each, and hands them to compute.
//
// Compile args: {CB_ATTN, CB_GATE, WT} + TensorAccessorArgs(attn) + TensorAccessorArgs(gate)
// Runtime args: {attn_addr, gate_addr, b}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_ATTN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_GATE = get_compile_time_arg_val(1);
    constexpr uint32_t WT = get_compile_time_arg_val(2);
    constexpr auto a_args = TensorAccessorArgs<3>();
    constexpr auto g_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t gate_addr = get_arg_val<uint32_t>(1);
    const uint32_t b = get_arg_val<uint32_t>(2);

    const uint32_t tile_bytes = get_tile_size(CB_ATTN);
    const auto attn = TensorAccessor(a_args, attn_addr, tile_bytes);
    const auto gate = TensorAccessor(g_args, gate_addr, tile_bytes);

    cb_reserve_back(CB_ATTN, WT);
    cb_reserve_back(CB_GATE, WT);
    uint32_t a_l1 = get_write_ptr(CB_ATTN);
    uint32_t g_l1 = get_write_ptr(CB_GATE);
    for (uint32_t t = 0; t < WT; ++t) {
        noc_async_read_page(b * WT + t, attn, a_l1 + t * tile_bytes);
        noc_async_read_page(b * WT + t, gate, g_l1 + t * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(CB_ATTN, WT);
    cb_push_back(CB_GATE, WT);
}

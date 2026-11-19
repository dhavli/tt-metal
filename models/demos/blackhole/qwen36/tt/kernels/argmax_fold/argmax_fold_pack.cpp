// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_ARGMAX_FOLD (argmax_fold.py): pack kernel, one core, data movement only. Replaces ttnn.gather +
// to_layout(idx) + typecast(idx, f32) + to_layout(val) + the second CCL: builds the per-shard [B, 2] pair tile the
// ONE all-gather carries: column 0 = the shard argmax index (raw uint32 bits), column 1 = the shard max logit
// (bf16 bits << 16, i.e. its exact float32 bit pattern). The tensor is declared float32 (the CCL dtype the served
// path gathers) but only ever read as bits (argmax_fold_combine.cpp).
//
// Inputs: idx [1, 1, B, 1] uint32 ROW_MAJOR (ttnn.argmax, one page per row), rm logits [1, 1, B, per] bf16
// ROW_MAJOR (one page per row). Output: [1, 1, B -> 32, 2 -> 32] float32 TILE, one 4 KiB page, zero-filled.
//
// Compile args: {CB_SCR, B, IDX_PAGE, RM_PAGE} + TensorAccessorArgs(idx) + (rm) + (out)
// Runtime args: {idx_addr, rm_addr, out_addr}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(2);
    constexpr uint32_t RM_PAGE = get_compile_time_arg_val(3);  // one logits row (bytes): the page the banks interleave
    constexpr auto i_args = TensorAccessorArgs<4>();
    constexpr auto r_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto o_args = TensorAccessorArgs<r_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 4096;  // 32 x 32 float32
    constexpr uint32_t FACE_BYTES = 1024;
    constexpr uint32_t CHUNK = 64;

    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t rm_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);

    const auto ix = TensorAccessor(i_args, idx_addr, IDX_PAGE);
    // The accessor maps page b to its bank by the TRUE page size (a 64-byte page size put rows >= the bank count on
    // the wrong bank: ttsim B=16 differed in 5 rows); each read is a 64-byte chunk at an offset inside the row page.
    const auto rm = TensorAccessor(r_args, rm_addr, RM_PAGE);
    const auto out = TensorAccessor(o_args, out_addr, TILE_BYTES);

    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t tile_l1 = scr;              // the output tile
    const uint32_t idx_l1 = scr + TILE_BYTES;  // B index pages, 64 B apart
    const uint32_t val_l1 = idx_l1 + B * CHUNK;  // B logit chunks, 64 B apart

    volatile tt_l1_ptr uint32_t* tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_l1);
    for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
        tile[i] = 0;
    }
    for (uint32_t b = 0; b < B; ++b) {
        noc_async_read_page(b, ix, idx_l1 + b * CHUNK);
    }
    noc_async_read_barrier();
    for (uint32_t b = 0; b < B; ++b) {
        const uint32_t idx = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1 + b * CHUNK);
        const uint32_t byte_off = idx * 2;
        noc_async_read(rm.get_noc_addr(b, byte_off & ~(CHUNK - 1)), val_l1 + b * CHUNK, CHUNK);
    }
    noc_async_read_barrier();
    for (uint32_t b = 0; b < B; ++b) {
        const uint32_t idx = *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1 + b * CHUNK);
        const uint32_t byte_off = idx * 2;
        const uint32_t v16 = *reinterpret_cast<volatile tt_l1_ptr uint16_t*>(val_l1 + b * CHUNK + (byte_off & (CHUNK - 1)));
        // tile element (row b, col c): face (b >= 16) * 2 (cols 0, 1 are in the left faces), row b % 16, col c
        const uint32_t e = ((b >= 16 ? 2u : 0u) * FACE_BYTES + (b & 15) * 64) / 4;
        tile[e + 0] = idx;
        tile[e + 1] = v16 << 16;
    }
    noc_async_write_page(0, out, tile_l1);
    noc_async_write_barrier();
}

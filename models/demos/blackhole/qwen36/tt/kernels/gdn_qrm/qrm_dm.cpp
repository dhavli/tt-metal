// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_QKV_RM (gdn_qrm.py): data movement. Block b of this core's run [blk0, blk0 + nblocks) is token
// tile row r = b / G, column group g = b % G (G = C_T / NB groups of NB tiles over the first C_T tile columns of the
// TILE source, whose rows are SRC_WT tiles wide).
// ROLE 0 (reader): the NB source tiles r * SRC_WT + g * NB + j (j < NB), one barrier per block. G_SHIFT (MUSTFAIL)
//   reads column group (g + 1) % G instead.
// ROLE 1 (writer): the untilized block is 32 rows of NB * 32 bf16; row i goes to row-major page r * 32 + i (one page =
//   one token row of C_T * 32 bf16) at byte offset g * NB * 64.
// Compile args: {CB_IN, CB_OUT, NB, G, SRC_WT, ROLE} + TensorAccessorArgs(src) + TensorAccessorArgs(dst)
// Runtime args: {src_addr, dst_addr, blk0, nblocks, g_shift}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    constexpr uint32_t G = get_compile_time_arg_val(3);
    constexpr uint32_t SRC_WT = get_compile_time_arg_val(4);
    constexpr uint32_t ROLE = get_compile_time_arg_val(5);
    constexpr auto s_args = TensorAccessorArgs<6>();
    constexpr auto d_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t blk0 = get_arg_val<uint32_t>(2);
    const uint32_t nblocks = get_arg_val<uint32_t>(3);
    const uint32_t g_shift = get_arg_val<uint32_t>(4);

    if constexpr (ROLE == 0) {
        constexpr uint32_t tsz = 2048;  // bf16 32x32 tile (get_tile_size is not declared in the served image's dataflow API: jit-check)
        const auto S = TensorAccessor(s_args, src_addr, tsz);
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t b = blk0 + k;
            const uint32_t r = b / G;
            const uint32_t g = (b - r * G + g_shift) % G;
            cb_reserve_back(CB_IN, NB);
            const uint32_t l1 = get_write_ptr(CB_IN);
            const uint32_t t0 = r * SRC_WT + g * NB;
            for (uint32_t j = 0; j < NB; ++j) {
                noc_async_read_page(t0 + j, S, l1 + j * tsz);
            }
            noc_async_read_barrier();
            cb_push_back(CB_IN, NB);
        }
    } else {
        constexpr uint32_t ROW_BYTES = NB * 32 * 2;
        const auto D = TensorAccessor(d_args, dst_addr, G * ROW_BYTES);
        for (uint32_t k = 0; k < nblocks; ++k) {
            const uint32_t b = blk0 + k;
            const uint32_t r = b / G;
            const uint32_t g = b - r * G;
            cb_wait_front(CB_OUT, NB);
            uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t i = 0; i < 32; ++i) {
                noc_async_write(l1, D.get_noc_addr(r * 32 + i, g * ROW_BYTES), ROW_BYTES);
                l1 += ROW_BYTES;
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, NB);
        }
    }
}

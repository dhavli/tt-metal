// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_STATE_SLOT: per-row GDN conv-tap commit by slot index. Reader half.
//
// Each conv tap i is a TILE [1, Bmax, C] DRAM-interleaved tensor: the batch rows are the rows of ONE
// tile row (Bw <= 32), so row r of tile page p sits inside the page (face (r/16)*2 + {0,1}, face row
// r%16, 16 elements each). The candidates for tap i are snap[0][i] .. snap[T-2][i] (the taps after
// 1..T-1 verify tokens) and the live tap itself (after T tokens, candidate T-1), exactly the
// where-chain's cands list in TPGatedDeltaNet.mtp_commit.
//
// Work unit u = (tap i = u / CT, tile column p = u % CT). The live page is read into CB_TILE; for each
// candidate s < T-1 that some row r < BW selects, the snap page is read (full page) into a private
// scratch page and rows r with idx[r] == s are copied over the live page's row r with core-side L1
// word copies (volatile + barriers). Rows >= BW and rows selecting T-1 keep the live values.
// Sub-page NoC transfers are never used (they silently fail on this stack).
//
// Compile args: {T, NTAP, CT, BW, CB_TILE, CB_SCR, CB_IDX, PAGE, IDX_PAGE} + TensorAccessorArgs(conv) + TensorAccessorArgs(idx)
// Runtime args: {idx_addr, u_start, n_u, live_addr[0..NTAP-1], snap_addr[s*NTAP + i] for s < T-1}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t NTAP = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t BW = get_compile_time_arg_val(3);
    constexpr uint32_t CB_TILE = get_compile_time_arg_val(4);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(5);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(6);
    constexpr uint32_t PAGE = get_compile_time_arg_val(7);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(8);
    constexpr auto st_args = TensorAccessorArgs<9>();
    constexpr auto ix_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();
    constexpr uint32_t ELEM = PAGE / 1024;             // bytes per element (TILE 32x32)
    constexpr uint32_t ROW_WORDS = 16 * ELEM / 4;      // one 16-element face row
    static_assert(BW <= 32, "the bucket rows must fit one tile row");

    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t u_start = get_arg_val<uint32_t>(1);
    const uint32_t n_u = get_arg_val<uint32_t>(2);

    cb_reserve_back(CB_IDX, 1);
    const uint32_t idx_l1 = get_write_ptr(CB_IDX);
    const auto ix = TensorAccessor(ix_args, idx_addr, IDX_PAGE);
    noc_async_read_page(0, ix, idx_l1);
    noc_async_read_barrier();
    cb_push_back(CB_IDX, 1);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    // Which candidates < T-1 are selected by any row (read each such snap page once per unit).
    uint32_t need = 0;
    for (uint32_t r = 0; r < BW; ++r) {
        const uint32_t s = idx[r];
        if (s + 1 < T) {
            need |= 1u << s;
        }
    }

    cb_reserve_back(CB_SCR, 1);  // private scratch page, never pushed
    const uint32_t scr = get_write_ptr(CB_SCR);

    for (uint32_t u = u_start; u < u_start + n_u; ++u) {
        const uint32_t i = u / CT;
        const uint32_t p = u % CT;
        cb_reserve_back(CB_TILE, 1);
        const uint32_t base = get_write_ptr(CB_TILE);
        const auto live = TensorAccessor(st_args, get_arg_val<uint32_t>(3 + i), PAGE);
        noc_async_read_page(p, live, base);
        noc_async_read_barrier();
        for (uint32_t s = 0; s + 1 < T; ++s) {
            if (!(need & (1u << s))) {
                continue;
            }
            const auto snap = TensorAccessor(st_args, get_arg_val<uint32_t>(3 + NTAP + s * NTAP + i), PAGE);
            noc_async_read_page(p, snap, scr);
            noc_async_read_barrier();
            asm volatile("" ::: "memory");
            for (uint32_t r = 0; r < BW; ++r) {
                if (idx[r] != s) {
                    continue;
                }
                const uint32_t f0 = (r / 16) * 2;
                const uint32_t off0 = (f0 * 256 + (r % 16) * 16) * ELEM;
                const uint32_t off1 = off0 + 256 * ELEM;  // face f0 + 1, same face row
                volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off0);
                volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + off0);
                volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off1);
                volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + off1);
                for (uint32_t w = 0; w < ROW_WORDS; ++w) {
                    d0[w] = s0[w];
                    d1[w] = s1[w];
                }
            }
            asm volatile("" ::: "memory");
        }
        cb_push_back(CB_TILE, 1);
    }
}

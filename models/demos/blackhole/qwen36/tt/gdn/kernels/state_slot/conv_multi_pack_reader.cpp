// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_COMMIT_PACK (docs/MTP-BATCH.md section 19): the CONV_MULTI conv-tap commit (conv_multi_reader.cpp) of
// ALL GDN layers in ONE program. Reader half.
//
// Per work unit it does EXACTLY what conv_multi_reader.cpp does (the four live pages read, rows < BW taken from window
// row min(idx[r], T-1)*BW + r, rows >= BW shifted T times with zero input), byte moves only. The only difference: unit
// gu = l * U + u carries its layer l, whose four live-tap and four window addresses come from the runtime args (a
// core's contiguous unit range spans at most a few layers; the wrapper passes only those).
//
// Compile args: {T, BW, CT, CB_TILE, CB_SCR, CB_IDX, PAGE, IDX_PAGE} + TensorAccessorArgs(live) + TensorAccessorArgs(P)
//               + TensorAccessorArgs(idx)    (every layer's taps / windows have the same accessor args: wrapper-checked)
// Runtime args: {idx_addr, gu_start, n_u, U, l_first, nl, then per layer lo < nl: live_addr[0..3], p_addr[0..3]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t BW = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t CB_TILE = get_compile_time_arg_val(3);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(4);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(5);
    constexpr uint32_t PAGE = get_compile_time_arg_val(6);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(7);
    constexpr auto lv_args = TensorAccessorArgs<8>();
    constexpr auto p_args = TensorAccessorArgs<lv_args.next_compile_time_args_offset()>();
    constexpr auto ix_args = TensorAccessorArgs<p_args.next_compile_time_args_offset()>();
    constexpr uint32_t NTAP = 4;
    constexpr uint32_t ELEM = PAGE / 1024;
    constexpr uint32_t ROW_WORDS = 16 * ELEM / 4;
    static_assert(T >= 1 && T * BW <= 32, "the verify rows must fit one tile row");

    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t gu_start = get_arg_val<uint32_t>(1);
    const uint32_t n_u = get_arg_val<uint32_t>(2);
    const uint32_t U = get_arg_val<uint32_t>(3);
    const uint32_t l_first = get_arg_val<uint32_t>(4);
    constexpr uint32_t ARG_ADDR = 6;
    constexpr uint32_t PER_LAYER = 2 * NTAP;

    cb_reserve_back(CB_IDX, 1);
    const uint32_t idx_l1 = get_write_ptr(CB_IDX);
    const auto ix = TensorAccessor(ix_args, idx_addr, IDX_PAGE);
    noc_async_read_page(0, ix, idx_l1);
    noc_async_read_barrier();
    cb_push_back(CB_IDX, 1);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    cb_reserve_back(CB_SCR, 1);  // private scratch page, never pushed
    const uint32_t scr = get_write_ptr(CB_SCR);

    auto row_off = [](uint32_t r) { return (((r / 16) * 2) * 256 + (r % 16) * 16) * ELEM; };
    auto copy_row = [&](uint32_t sbase, uint32_t sr, uint32_t dbase, uint32_t dr) {
        volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbase + row_off(sr));
        volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sbase + row_off(sr) + 256 * ELEM);
        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbase + row_off(dr));
        volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbase + row_off(dr) + 256 * ELEM);
        for (uint32_t w = 0; w < ROW_WORDS; ++w) {
            d0[w] = s0[w];
            d1[w] = s1[w];
        }
    };
    auto zero_row = [&](uint32_t dbase, uint32_t dr) {
        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbase + row_off(dr));
        volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dbase + row_off(dr) + 256 * ELEM);
        for (uint32_t w = 0; w < ROW_WORDS; ++w) {
            d0[w] = 0u;
            d1[w] = 0u;
        }
    };

    for (uint32_t gu = gu_start; gu < gu_start + n_u; ++gu) {
        const uint32_t l = gu / U;
        const uint32_t u = gu % U;
        const uint32_t abase = ARG_ADDR + (l - l_first) * PER_LAYER;
        const uint32_t bt = u / CT;
        const uint32_t p = u % CT;
        const uint32_t page = bt * CT + p;
        cb_reserve_back(CB_TILE, NTAP);
        const uint32_t base = get_write_ptr(CB_TILE);
        for (uint32_t k = 0; k < NTAP; ++k) {
            const auto live = TensorAccessor(lv_args, get_arg_val<uint32_t>(abase + k), PAGE);
            noc_async_read_page(page, live, base + k * PAGE);
        }
        noc_async_read_barrier();
        asm volatile("" ::: "memory");
        for (uint32_t k = 0; k < NTAP; ++k) {
            const uint32_t tk = base + k * PAGE;
            if (bt == 0) {
                const auto pw = TensorAccessor(p_args, get_arg_val<uint32_t>(abase + NTAP + k), PAGE);
                noc_async_read_page(p, pw, scr);
                noc_async_read_barrier();
                asm volatile("" ::: "memory");
                for (uint32_t r = 0; r < BW; ++r) {
                    uint32_t s = idx[r];
                    if (s >= T) {
                        s = T - 1;
                    }
                    copy_row(scr, s * BW + r, tk, r);
                }
            }
            for (uint32_t r = (bt == 0 ? BW : 0); r < 32; ++r) {
                if (k + T < NTAP) {
                    copy_row(base + (k + T) * PAGE, r, tk, r);
                } else {
                    zero_row(tk, r);
                }
            }
            asm volatile("" ::: "memory");
        }
        cb_push_back(CB_TILE, NTAP);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_CONV_MULTI: exact row gather between TILE tensors whose rows all sit in tile-row 0. Reader half.
//
// Every source and destination is a TILE [1, R, W] tensor with the same dtype and the same width (CT tile columns);
// the rows a program reads or writes are < 32, so tile column p of tile-row 0 is page p of every tensor. Destination
// row r of dst d is either a copy of row `sr` of source `s` or zero, from a per-row map. Pure data movement: whole
// 16-element face rows are copied core-side in L1 (volatile + barriers), no arithmetic, no format change, so the
// result is bit-exact. Sub-page NoC transfers are never used (they silently fail on this stack).
//
// Work unit u = (dst d = u / CT, tile column p = u % CT): zero the output page, then for each source that dst d
// references read that source's page p (full page) into a private scratch page and copy the mapped rows.
//
// Compile args: {NSRC, NDST, CT, CB_OUT, CB_SCR, PAGE} + TensorAccessorArgs(class 0) + TensorAccessorArgs(class 1)
// Runtime args: {u_start, n_u, cls_mask, src_addr[NSRC], need[NDST], map[NDST * 16]}
//   cls_mask bit s: source s uses the class-1 accessor (e.g. L1 vs DRAM interleaved)
//   need[d]: bitmask of the sources dst d reads
//   map: 16-bit entries, two per word (row r of dst d at word d*16 + r/2, low half for even r):
//        0xFFFF = zero row, else (s << 5) | sr

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NSRC = get_compile_time_arg_val(0);
    constexpr uint32_t NDST = get_compile_time_arg_val(1);
    constexpr uint32_t CT = get_compile_time_arg_val(2);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(3);
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(4);
    constexpr uint32_t PAGE = get_compile_time_arg_val(5);
    constexpr auto c0_args = TensorAccessorArgs<6>();
    constexpr auto c1_args = TensorAccessorArgs<c0_args.next_compile_time_args_offset()>();
    constexpr uint32_t ELEM = PAGE / 1024;         // bytes per element (TILE 32x32)
    constexpr uint32_t ROW_WORDS = 16 * ELEM / 4;  // one 16-element face row
    static_assert(NSRC <= 32 && NDST <= 16, "row gather: at most 32 sources and 16 destinations");

    const uint32_t u_start = get_arg_val<uint32_t>(0);
    const uint32_t n_u = get_arg_val<uint32_t>(1);
    const uint32_t cls = get_arg_val<uint32_t>(2);
    constexpr uint32_t A_SRC = 3;
    constexpr uint32_t A_NEED = A_SRC + NSRC;
    constexpr uint32_t A_MAP = A_NEED + NDST;

    cb_reserve_back(CB_SCR, 1);  // private scratch page, never pushed
    const uint32_t scr = get_write_ptr(CB_SCR);

    for (uint32_t u = u_start; u < u_start + n_u; ++u) {
        const uint32_t d = u / CT;
        const uint32_t p = u % CT;
        cb_reserve_back(CB_OUT, 1);
        const uint32_t base = get_write_ptr(CB_OUT);
        {
            volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
            for (uint32_t w = 0; w < PAGE / 4; ++w) {
                z[w] = 0u;
            }
        }
        const uint32_t need = get_arg_val<uint32_t>(A_NEED + d);
        for (uint32_t s = 0; s < NSRC; ++s) {
            if (!(need & (1u << s))) {
                continue;
            }
            const uint32_t addr = get_arg_val<uint32_t>(A_SRC + s);
            if (cls & (1u << s)) {
                const auto src = TensorAccessor(c1_args, addr, PAGE);
                noc_async_read_page(p, src, scr);
            } else {
                const auto src = TensorAccessor(c0_args, addr, PAGE);
                noc_async_read_page(p, src, scr);
            }
            noc_async_read_barrier();
            asm volatile("" ::: "memory");
            for (uint32_t r = 0; r < 32; ++r) {
                const uint32_t word = get_arg_val<uint32_t>(A_MAP + d * 16 + r / 2);
                const uint32_t e = (r & 1) ? (word >> 16) : (word & 0xFFFFu);
                if (e == 0xFFFFu || (e >> 5) != s) {
                    continue;
                }
                const uint32_t sr = e & 31u;
                const uint32_t so = (((sr / 16) * 2) * 256 + (sr % 16) * 16) * ELEM;
                const uint32_t dof = (((r / 16) * 2) * 256 + (r % 16) * 16) * ELEM;
                volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + so);
                volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + so + 256 * ELEM);
                volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + dof);
                volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + dof + 256 * ELEM);
                for (uint32_t w = 0; w < ROW_WORDS; ++w) {
                    d0[w] = s0[w];
                    d1[w] = s1[w];
                }
            }
            asm volatile("" ::: "memory");
        }
        cb_push_back(CB_OUT, 1);
    }
}

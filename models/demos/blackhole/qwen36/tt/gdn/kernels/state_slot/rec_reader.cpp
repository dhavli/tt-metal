// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_STATE_SLOT (docs/MTP-BATCH.md section 10): per-row GDN recurrent-state commit by slot index.
// Reader half. dst[b] <- cand[idx[b]][b] for every (b, h) head instance this core owns.
//
// Layout: every candidate and the destination is a TILE [B*, Nv, Dk, Dv] DRAM-interleaved tensor of
// one dtype. Dk, Dv are 32-multiples, so head instance bh = b*Nv + h owns the KV contiguous pages
// [bh*KV, bh*KV + KV) in every one of them (dst has Bmax >= Bw rows; rows >= Bw are never touched).
// Full-page reads only: sub-page NoC reads silently fail on this stack (decode_gated_delta_rule reader).
//
// idx: one DRAM page of uint32 (ROW_MAJOR [1, 32]), entry b = the verify depth to commit for row b.
// An entry >= T means "leave row b as it is" (no read, no write). The page is read ONCE, pushed to
// cb_idx and consumed by the writer (one producer, one consumer), so both halves skip the same rows.
//
// Compile args: {T, KV, NV, CB_DATA, CB_IDX, PAGE, IDX_PAGE} + TensorAccessorArgs(state) + TensorAccessorArgs(idx)
// Runtime args: {idx_addr, bh_start, n_inst, cand_addr[0..T-1]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t KV = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t CB_DATA = get_compile_time_arg_val(3);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(4);
    constexpr uint32_t PAGE = get_compile_time_arg_val(5);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(6);
    constexpr auto st_args = TensorAccessorArgs<7>();
    constexpr auto ix_args = TensorAccessorArgs<st_args.next_compile_time_args_offset()>();

    const uint32_t idx_addr = get_arg_val<uint32_t>(0);
    const uint32_t bh_start = get_arg_val<uint32_t>(1);
    const uint32_t n_inst = get_arg_val<uint32_t>(2);

    cb_reserve_back(CB_IDX, 1);
    const uint32_t idx_l1 = get_write_ptr(CB_IDX);
    const auto ix = TensorAccessor(ix_args, idx_addr, IDX_PAGE);
    noc_async_read_page(0, ix, idx_l1);
    noc_async_read_barrier();
    cb_push_back(CB_IDX, 1);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(idx_l1);

    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        const uint32_t s = idx[bh / NV];
        if (s >= T) {
            continue;  // keep this row: the writer skips it too
        }
        const auto src = TensorAccessor(st_args, get_arg_val<uint32_t>(3 + s), PAGE);
        cb_reserve_back(CB_DATA, KV);
        const uint32_t l1 = get_write_ptr(CB_DATA);
        const uint32_t first = bh * KV;
        for (uint32_t t = 0; t < KV; ++t) {
            noc_async_read_page(first + t, src, l1 + t * PAGE);
        }
        noc_async_read_barrier();
        cb_push_back(CB_DATA, KV);
    }
}

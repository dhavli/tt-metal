// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_STATE_SLOT: writer half of the recurrent-state slot commit (see rec_reader.cpp).
// Writes each committed head instance's KV pages straight into the persistent state (full pages).
//
// Compile args: {T, KV, NV, CB_DATA, CB_IDX, PAGE} + TensorAccessorArgs(state)
// Runtime args: {dst_addr, bh_start, n_inst}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t T = get_compile_time_arg_val(0);
    constexpr uint32_t KV = get_compile_time_arg_val(1);
    constexpr uint32_t NV = get_compile_time_arg_val(2);
    constexpr uint32_t CB_DATA = get_compile_time_arg_val(3);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(4);
    constexpr uint32_t PAGE = get_compile_time_arg_val(5);
    constexpr auto st_args = TensorAccessorArgs<6>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t bh_start = get_arg_val<uint32_t>(1);
    const uint32_t n_inst = get_arg_val<uint32_t>(2);
    const auto dst = TensorAccessor(st_args, dst_addr, PAGE);

    cb_wait_front(CB_IDX, 1);
    volatile tt_l1_ptr uint32_t* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_IDX));

    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        if (idx[bh / NV] >= T) {
            continue;
        }
        cb_wait_front(CB_DATA, KV);
        const uint32_t l1 = get_read_ptr(CB_DATA);
        const uint32_t first = bh * KV;
        for (uint32_t t = 0; t < KV; ++t) {
            noc_async_write_page(first + t, dst, l1 + t * PAGE);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_DATA, KV);
    }
    noc_async_write_barrier();
    cb_pop_front(CB_IDX, 1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_STATE_SLOT: writer half of the conv-tap slot commit (see conv_reader.cpp). Writes each
// composed tile page back over the live tap (full-page writes; each page has exactly one owner core).
//
// Compile args: {NTAP, CT, CB_TILE, CB_IDX, PAGE} + TensorAccessorArgs(conv)
// Runtime args: {u_start, n_u, live_addr[0..NTAP-1]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NTAP = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t CB_TILE = get_compile_time_arg_val(2);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(3);
    constexpr uint32_t PAGE = get_compile_time_arg_val(4);
    constexpr auto st_args = TensorAccessorArgs<5>();

    const uint32_t u_start = get_arg_val<uint32_t>(0);
    const uint32_t n_u = get_arg_val<uint32_t>(1);

    for (uint32_t u = u_start; u < u_start + n_u; ++u) {
        const uint32_t i = u / CT;
        const uint32_t p = u % CT;
        const auto dst = TensorAccessor(st_args, get_arg_val<uint32_t>(2 + i), PAGE);
        cb_wait_front(CB_TILE, 1);
        noc_async_write_page(p, dst, get_read_ptr(CB_TILE));
        noc_async_writes_flushed();
        cb_pop_front(CB_TILE, 1);
    }
    noc_async_write_barrier();
    // The reader pushed the idx page; consume it so the CB has its single producer/consumer pair.
    cb_wait_front(CB_IDX, 1);
    cb_pop_front(CB_IDX, 1);
}

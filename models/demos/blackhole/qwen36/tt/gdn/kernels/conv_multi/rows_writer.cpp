// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_CONV_MULTI: writer half of the exact row gather (see rows_reader.cpp). Writes each composed page to
// page p of its destination (full-page writes; every destination page has exactly one owner core).
//
// Compile args: {NDST, CT, CB_OUT, PAGE} + TensorAccessorArgs(dst)
// Runtime args: {u_start, n_u, dst_addr[NDST]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t NDST = get_compile_time_arg_val(0);
    constexpr uint32_t CT = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
    constexpr uint32_t PAGE = get_compile_time_arg_val(3);
    constexpr auto d_args = TensorAccessorArgs<4>();
    static_assert(NDST <= 16, "row gather: at most 16 destinations");

    const uint32_t u_start = get_arg_val<uint32_t>(0);
    const uint32_t n_u = get_arg_val<uint32_t>(1);

    for (uint32_t u = u_start; u < u_start + n_u; ++u) {
        const uint32_t d = u / CT;
        const uint32_t p = u % CT;
        const auto dst = TensorAccessor(d_args, get_arg_val<uint32_t>(2 + d), PAGE);
        cb_wait_front(CB_OUT, 1);
        noc_async_write_page(p, dst, get_read_ptr(CB_OUT));
        noc_async_writes_flushed();
        cb_pop_front(CB_OUT, 1);
    }
    noc_async_write_barrier();
}

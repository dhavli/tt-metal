// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_KV_DEVICE_INIT self-test: copy whole pages of a DRAM-interleaved buffer, byte for byte, into the
// pages of a same-page-size ROW_MAJOR uint32 buffer, so the host reads the RAW device bytes (bfp8/bfp4 tiles:
// exponent section + mantissas) instead of a decoded float view in which a non-canonical zero would hide.
// Full-page reads only (sub-page NoC reads silently fail on this stack).
//
// Compile args: {CB, PAGE} + TensorAccessorArgs(src) + TensorAccessorArgs(dst)
// Runtime args: {src_addr, dst_addr, n_runs, (src_page, dst_page, len) x n_runs}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t n_runs = get_arg_val<uint32_t>(2);
    const auto src = TensorAccessor(src_args, src_addr, PAGE);
    const auto dst = TensorAccessor(dst_args, dst_addr, PAGE);

    cb_reserve_back(CB, 1);
    const uint32_t l1 = get_write_ptr(CB);
    for (uint32_t r = 0; r < n_runs; ++r) {
        const uint32_t s0 = get_arg_val<uint32_t>(3 + 3 * r);
        const uint32_t d0 = get_arg_val<uint32_t>(4 + 3 * r);
        const uint32_t len = get_arg_val<uint32_t>(5 + 3 * r);
        for (uint32_t i = 0; i < len; ++i) {
            noc_async_read_page(s0 + i, src, l1);
            noc_async_read_barrier();
            noc_async_write_page(d0 + i, dst, l1);
            noc_async_write_barrier();
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_KV_DEVICE_INIT: fill every page this core owns of a DRAM-interleaved buffer with one 32-bit word,
// straight from L1. The paged KV pool used to be torch.zeros on the host, packed to bf8/bf4 and uploaded
// (~1.9 s per cache, 65 s per boot). Here the buffer is ttnn.empty and this writer stores the bytes of
// host-packed zero tiles (all-zero: shared exponents and mantissas, checked at boot against the host packer
// by kv_devinit.selftest) into every page. Pure data movement, no format conversion.
//
// Compile args: {CB, PAGE, FILL} + TensorAccessorArgs(dst)
// Runtime args: {dst_addr, page_start, n_pages}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB = get_compile_time_arg_val(0);
    constexpr uint32_t PAGE = get_compile_time_arg_val(1);
    constexpr uint32_t FILL = get_compile_time_arg_val(2);
    constexpr auto dst_args = TensorAccessorArgs<3>();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t start = get_arg_val<uint32_t>(1);
    const uint32_t n = get_arg_val<uint32_t>(2);
    const auto dst = TensorAccessor(dst_args, dst_addr, PAGE);

    cb_reserve_back(CB, 1);
    const uint32_t l1 = get_write_ptr(CB);
    volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(l1);
    for (uint32_t i = 0; i < PAGE / 4; ++i) {
        p[i] = FILL;
    }
    // The L1 page is never modified again, so every write may source it while earlier ones are in flight.
    for (uint32_t id = start; id < start + n; ++id) {
        noc_async_write_page(id, dst, l1);
    }
    noc_async_write_barrier();
}

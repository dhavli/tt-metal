// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_COMMIT_PACK (docs/MTP-BATCH.md section 19): writer half of the packed conv-tap commit (see
// conv_multi_pack_reader.cpp). Per unit what conv_multi_writer.cpp does: the four composed pages back over the unit's
// layer's live taps (full-page writes; each page has exactly one owner core, the one that read it).
//
// Compile args: {CT, CB_TILE, CB_IDX, PAGE} + TensorAccessorArgs(live)
// Runtime args: {gu_start, n_u, U, l_first, nl, then per layer lo < nl: live_addr[0..3]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CT = get_compile_time_arg_val(0);
    constexpr uint32_t CB_TILE = get_compile_time_arg_val(1);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(2);
    constexpr uint32_t PAGE = get_compile_time_arg_val(3);
    constexpr auto lv_args = TensorAccessorArgs<4>();
    constexpr uint32_t NTAP = 4;
    static_assert(CT > 0, "conv taps must have at least one tile column");

    const uint32_t gu_start = get_arg_val<uint32_t>(0);
    const uint32_t n_u = get_arg_val<uint32_t>(1);
    const uint32_t U = get_arg_val<uint32_t>(2);
    const uint32_t l_first = get_arg_val<uint32_t>(3);
    constexpr uint32_t ARG_ADDR = 5;

    for (uint32_t gu = gu_start; gu < gu_start + n_u; ++gu) {
        const uint32_t l = gu / U;
        const uint32_t page = gu % U;  // unit u = (tile-row u / CT, column u % CT) is page u of every live tap
        const uint32_t abase = ARG_ADDR + (l - l_first) * NTAP;
        cb_wait_front(CB_TILE, NTAP);
        const uint32_t src = get_read_ptr(CB_TILE);
        for (uint32_t k = 0; k < NTAP; ++k) {
            const auto dst = TensorAccessor(lv_args, get_arg_val<uint32_t>(abase + k), PAGE);
            noc_async_write_page(page, dst, src + k * PAGE);
        }
        noc_async_writes_flushed();
        cb_pop_front(CB_TILE, NTAP);
    }
    noc_async_write_barrier();
    // The reader pushed the idx page; consume it so the CB has its single producer/consumer pair.
    cb_wait_front(CB_IDX, 1);
    cb_pop_front(CB_IDX, 1);
}

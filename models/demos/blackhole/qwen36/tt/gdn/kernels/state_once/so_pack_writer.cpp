// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_COMMIT_PACK (docs/MTP-BATCH.md section 19): writer half of the packed STATE_ONCE commit (see
// so_pack_reader.cpp). Per instance it does what so_writer.cpp MODE 1 does: the reader's token count n (cb_wcnt);
// n > 0: the kv committed state pages from cb_sout over this instance's layer's rec_state pages [bh*kv, bh*kv + kv).
//
// Compile args: {Kt, Vt} + TensorAccessorArgs(state of layer 0) (every layer's state has the same accessor args)
// Runtime args: {gi_start, n_inst, BH, l_first, nl, state_addr[nl]}

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"

constexpr uint32_t cb_sout = 18, cb_wcnt = 32;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr auto st_a = TensorAccessorArgs<2>();

    const uint32_t gi_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t BH = get_arg_val<uint32_t>(2);
    const uint32_t l_first = get_arg_val<uint32_t>(3);
    constexpr uint32_t ARG_ADDR = 5;

    const uint32_t tb_io = get_tile_size(cb_sout);
    constexpr uint32_t kv = Kt * Vt;

    Noc noc;

    for (uint32_t gi = gi_start; gi < gi_start + n_inst; ++gi) {
        const uint32_t l = gi / BH;
        const uint32_t bh = gi % BH;
        CircularBuffer ccb(cb_wcnt);
        ccb.wait_front(1);
        asm volatile("" ::: "memory");
        const uint32_t n = CoreLocalMem<volatile uint32_t>(ccb.get_read_ptr())[0];
        ccb.pop_front(1);
        if (n == 0) {
            continue;
        }
        const auto s_acc = TensorAccessor(st_a, get_arg_val<uint32_t>(ARG_ADDR + (l - l_first)), tb_io);
        CircularBuffer cb(cb_sout);
        cb.wait_front(kv);
        const uint32_t base_page = bh * kv;
        auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
        for (uint32_t t = 0; t < kv; t++) {
            noc.async_write(src, s_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = base_page + t});
        }
        noc.async_write_barrier();
        cb.pop_front(kv);
    }
}

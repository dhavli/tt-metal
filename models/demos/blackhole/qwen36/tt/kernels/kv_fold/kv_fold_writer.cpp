// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_KV_UPDATE_FOLD (attention/kv_fold.py): writer half (RISCV_1) of the one-program paged K + V update (see
// kv_fold_reader.cpp). Per cache head it does what paged_update_cache's writer
// (writer_update_cache_interleaved_start_id.cpp) does for one row: wait for compute's untilized cache block, copy the
// row's untilized input row (head h) over row idx % 32 of it, hand it back (CB_UC2 aliases CB_UC), wait for the
// re-tilized block and write its Wt tiles back over the cache. After the K (V) tiles are written (write barrier), it
// increments SEM_K (SEM_V) on the core of row u + Bw, the next candidate of the same user, if there is one.
//
// MF (compile arg, QWEN36_KV_UPDATE_FOLD_MUSTFAIL): the input row lands on tile row (idx + 1) % 32 instead of idx % 32
// (same tile, wrong position): the self-test must REFUSE it.
//
// Compile args: {CB_OK, CB_OV, CB_UC, CB_UC2, CB_UINK, CB_UINV, CB_IDX, CB_PT, WT, WBYTES, NH, BLOCK, BLOCK_T, SEM_K,
//                SEM_V, MF} + TensorAccessorArgs(k cache) + (v cache)
// Runtime args: {kc_addr, vc_addr, my_row, send, next_x, next_y}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

template <uint32_t CB_OUT, uint32_t CB_UIN, uint32_t CB_UC, uint32_t CB_UC2, uint32_t WT, uint32_t WBYTES, uint32_t NH,
          uint32_t HEAD_OFFSET_T, typename Acc>
FORCE_INLINE void update_one_cache(const Acc& acc, uint32_t cache_id, uint32_t row_offset_B, bool skip) {
    const uint32_t out_tile = get_tile_size(CB_OUT);
    cb_wait_front(CB_UIN, WT);
    uint32_t in_l1 = get_read_ptr(CB_UIN);
    for (uint32_t h = 0; h < NH; ++h) {
        cb_wait_front(CB_UC, WT);
        cb_reserve_back(CB_UC2, WT);
        const uint32_t dst = get_read_ptr(CB_UC) + row_offset_B;
        noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], in_l1), dst, WBYTES);
        noc_async_read_barrier();
        cb_push_back(CB_UC2, WT);
        cb_pop_front(CB_UC, WT);

        cb_wait_front(CB_OUT, WT);
        if (!skip) {
            uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t t = 0; t < WT; ++t) {
                noc_async_write_page(cache_id + t, acc, l1);
                l1 += out_tile;
            }
            noc_async_writes_flushed();
        }
        cb_pop_front(CB_OUT, WT);
        if (!skip) {
            noc_async_write_barrier();
        }
        in_l1 += WBYTES;
        cache_id += HEAD_OFFSET_T;
    }
    cb_pop_front(CB_UIN, WT);
}

void kernel_main() {
    constexpr uint32_t CB_OK = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OV = get_compile_time_arg_val(1);
    constexpr uint32_t CB_UC = get_compile_time_arg_val(2);
    constexpr uint32_t CB_UC2 = get_compile_time_arg_val(3);
    constexpr uint32_t CB_UINK = get_compile_time_arg_val(4);
    constexpr uint32_t CB_UINV = get_compile_time_arg_val(5);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(6);
    constexpr uint32_t CB_PT = get_compile_time_arg_val(7);
    constexpr uint32_t WT = get_compile_time_arg_val(8);
    constexpr uint32_t WBYTES = get_compile_time_arg_val(9);
    constexpr uint32_t NH = get_compile_time_arg_val(10);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(11);
    constexpr uint32_t BLOCK_T = get_compile_time_arg_val(12);
    constexpr uint32_t SEM_K = get_compile_time_arg_val(13);
    constexpr uint32_t SEM_V = get_compile_time_arg_val(14);
    constexpr uint32_t MF = get_compile_time_arg_val(15);
    constexpr auto kc_args = TensorAccessorArgs<16>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t HEAD_OFFSET_T = WT * BLOCK_T;

    const uint32_t kc_addr = get_arg_val<uint32_t>(0);
    const uint32_t vc_addr = get_arg_val<uint32_t>(1);
    const uint32_t my_row = get_arg_val<uint32_t>(2);
    const bool send = get_arg_val<uint32_t>(3) == 1;
    const uint32_t next_x = get_arg_val<uint32_t>(4);
    const uint32_t next_y = get_arg_val<uint32_t>(5);

    const auto kc = TensorAccessor(kc_args, kc_addr, get_tile_size(CB_OK));
    const auto vc = TensorAccessor(vc_args, vc_addr, get_tile_size(CB_OV));

    cb_wait_front(CB_IDX, 1);
    const uint32_t idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_IDX))[my_row];
    const bool skip = idx == 0xFFFFFFFFu;
    uint32_t cache_id = 0;
    uint32_t row_offset_B = 0;
    if (!skip) {
        cb_wait_front(CB_PT, 1);
        const uint32_t phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(CB_PT))[idx / BLOCK];
        cache_id = phys * NH * BLOCK_T * WT + ((idx % BLOCK) / TILE_H) * WT;
        cb_pop_front(CB_PT, 1);
        row_offset_B = ((idx + MF) % TILE_H) * WBYTES;
    }
    cb_pop_front(CB_IDX, 1);

    update_one_cache<CB_OK, CB_UINK, CB_UC, CB_UC2, WT, WBYTES, NH, HEAD_OFFSET_T>(kc, cache_id, row_offset_B, skip);
    if (send) {
        noc_semaphore_inc(get_noc_addr(next_x, next_y, get_semaphore(SEM_K)), 1);
    }
    update_one_cache<CB_OV, CB_UINV, CB_UC, CB_UC2, WT, WBYTES, NH, HEAD_OFFSET_T>(vc, cache_id, row_offset_B, skip);
    if (send) {
        noc_semaphore_inc(get_noc_addr(next_x, next_y, get_semaphore(SEM_V)), 1);
        noc_async_atomic_barrier();
    }
}

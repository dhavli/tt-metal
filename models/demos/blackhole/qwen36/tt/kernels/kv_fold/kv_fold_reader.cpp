// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_KV_UPDATE_FOLD (attention/kv_fold.py): the paged K AND V cache update of all T*Bw verify rows (or the B
// rows of plain decode, T=1) in ONE program. Reader half (RISCV_0).
//
// Per core = per row u = j*Bw + b (candidate-major, the order fold_pos and the k/v height shard already have), this
// does what ttnn.experimental.paged_update_cache's reader (reader_update_cache_interleaved_start_id.cpp) does for
// one row, for the K cache and then the V cache:
//   - push the row's sharded K and V inputs (globally allocated CBs) to compute;
//   - read the index page (fold_pos) into CB_IDX for the writer; idx = fold_pos[u]; idx == -1 -> skip;
//   - read page-table row (u % Bw) -- the candidates of user b share user b's page-table row -- into CB_PT;
//   - per cache head: read the Wt tiles of the 32-row tile block holding position idx.
// The only additions: rows of one user at consecutive positions share cache tiles, so their read-modify-writes are
// chained in candidate order: row u >= Bw waits on SEM_K (SEM_V) until row u - Bw's writer has written its K (V)
// tiles back (the writer of row u - Bw increments them). Row order per tile is then exactly the T sequential
// paged_update_cache calls' order (candidate 0 first), so the cache bytes are the same.
//
// Compile args: {CB_CK, CB_CV, CB_INK, CB_INV, CB_IDX, CB_PT, WT, NH, BLOCK, BLOCK_T, IDX_PAGE, PT_PAGE, SEM_K, SEM_V,
//                N_IN_TILES} + TensorAccessorArgs(k cache) + (v cache) + (idx) + (page table)
// Runtime args: {kc_addr, vc_addr, idx_addr, pt_addr, my_row, pt_row, wait}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_CK = get_compile_time_arg_val(0);
    constexpr uint32_t CB_CV = get_compile_time_arg_val(1);
    constexpr uint32_t CB_INK = get_compile_time_arg_val(2);
    constexpr uint32_t CB_INV = get_compile_time_arg_val(3);
    constexpr uint32_t CB_IDX = get_compile_time_arg_val(4);
    constexpr uint32_t CB_PT = get_compile_time_arg_val(5);
    constexpr uint32_t WT = get_compile_time_arg_val(6);
    constexpr uint32_t NH = get_compile_time_arg_val(7);
    constexpr uint32_t BLOCK = get_compile_time_arg_val(8);
    constexpr uint32_t BLOCK_T = get_compile_time_arg_val(9);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(10);
    constexpr uint32_t PT_PAGE = get_compile_time_arg_val(11);
    constexpr uint32_t SEM_K = get_compile_time_arg_val(12);
    constexpr uint32_t SEM_V = get_compile_time_arg_val(13);
    constexpr uint32_t N_IN_TILES = get_compile_time_arg_val(14);
    constexpr auto kc_args = TensorAccessorArgs<15>();
    constexpr auto vc_args = TensorAccessorArgs<kc_args.next_compile_time_args_offset()>();
    constexpr auto ix_args = TensorAccessorArgs<vc_args.next_compile_time_args_offset()>();
    constexpr auto pt_args = TensorAccessorArgs<ix_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t HEAD_OFFSET_T = WT * BLOCK_T;

    const uint32_t kc_addr = get_arg_val<uint32_t>(0);
    const uint32_t vc_addr = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t pt_addr = get_arg_val<uint32_t>(3);
    const uint32_t my_row = get_arg_val<uint32_t>(4);
    const uint32_t pt_row = get_arg_val<uint32_t>(5);
    const bool wait = get_arg_val<uint32_t>(6) == 1;

    const uint32_t ck_tile = get_tile_size(CB_CK);
    const uint32_t cv_tile = get_tile_size(CB_CV);
    const auto kc = TensorAccessor(kc_args, kc_addr, ck_tile);
    const auto vc = TensorAccessor(vc_args, vc_addr, cv_tile);
    const auto ix = TensorAccessor(ix_args, idx_addr, IDX_PAGE);
    const auto pt = TensorAccessor(pt_args, pt_addr, PT_PAGE);

    // The row's K and V shards (globally allocated on the inputs' L1 buffers): hand them to compute.
    cb_reserve_back(CB_INK, N_IN_TILES);
    cb_push_back(CB_INK, N_IN_TILES);
    cb_reserve_back(CB_INV, N_IN_TILES);
    cb_push_back(CB_INV, N_IN_TILES);

    cb_reserve_back(CB_IDX, 1);
    const uint32_t ix_l1 = get_write_ptr(CB_IDX);
    noc_async_read_page(0, ix, ix_l1);
    noc_async_read_barrier();
    cb_push_back(CB_IDX, 1);
    const uint32_t idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ix_l1)[my_row];
    const bool skip = idx == 0xFFFFFFFFu;

    uint32_t cache_id = 0;
    if (!skip) {
        cb_reserve_back(CB_PT, 1);
        const uint32_t pt_l1 = get_write_ptr(CB_PT);
        noc_async_read_page(pt_row, pt, pt_l1);
        noc_async_read_barrier();
        cb_push_back(CB_PT, 1);
        const uint32_t phys = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(pt_l1)[idx / BLOCK];
        cache_id = phys * NH * BLOCK_T * WT + ((idx % BLOCK) / TILE_H) * WT;
    }

    // ---- K
    if (wait) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_K));
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    uint32_t id = cache_id;
    for (uint32_t h = 0; h < NH; ++h) {
        cb_reserve_back(CB_CK, WT);
        if (!skip) {
            uint32_t l1 = get_write_ptr(CB_CK);
            for (uint32_t t = 0; t < WT; ++t) {
                noc_async_read_page(id + t, kc, l1);
                l1 += ck_tile;
            }
            noc_async_read_barrier();
        }
        cb_push_back(CB_CK, WT);
        id += HEAD_OFFSET_T;
    }

    // ---- V (same tile ids: the V cache has the K cache's shape and paging)
    if (wait) {
        volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM_V));
        noc_semaphore_wait(sem, 1);
        noc_semaphore_set(sem, 0);
    }
    id = cache_id;
    for (uint32_t h = 0; h < NH; ++h) {
        cb_reserve_back(CB_CV, WT);
        if (!skip) {
            uint32_t l1 = get_write_ptr(CB_CV);
            for (uint32_t t = 0; t < WT; ++t) {
                noc_async_read_page(id + t, vc, l1);
                l1 += cv_tile;
            }
            noc_async_read_barrier();
        }
        cb_push_back(CB_CV, WT);
        id += HEAD_OFFSET_T;
    }
}

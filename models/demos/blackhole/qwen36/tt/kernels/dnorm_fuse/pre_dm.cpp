// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_FUSE (dnorm_fuse.py): the data-movement half of the fused distributed-RMSNorm PRE pass.
// One core per 32-row tile row r. Both RISCs run this source (ROLE 0 on one NoC, ROLE 1 on the other): they read the
// row's WT bf16 tiles of h (DRAM interleaved) straight into CB_H, which is backed by the core's shard of the L1
// HEIGHT-sharded output hs (so the tiles stay resident for the POST pass, no second DRAM read). Chunk j (CHUNK tiles)
// is read by ROLE j % 2, with ONE barrier per chunk (the image's reader barriers per tile), and announced to compute by
// one page pushed on that role's counter CB.
// ROLE 0 also builds the reduce scaler tile (SUM, 1.0, the image's pre reader); ROLE 1 writes the row's stats tile.
//
// Compile args: {CB_H, CB_CNT, CB_SC, CB_ST, WT, CHUNK, ROLE} + TensorAccessorArgs(h) + TensorAccessorArgs(stats)
// Runtime args: {h_addr, stats_addr, row}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t CB_H = get_compile_time_arg_val(0);
    constexpr uint32_t CB_CNT = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(2);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(3);
    constexpr uint32_t WT = get_compile_time_arg_val(4);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(5);
    constexpr uint32_t ROLE = get_compile_time_arg_val(6);
    constexpr auto h_args = TensorAccessorArgs<7>();
    constexpr auto s_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t h_addr = get_arg_val<uint32_t>(0);
    const uint32_t stats_addr = get_arg_val<uint32_t>(1);
    const uint32_t row = get_arg_val<uint32_t>(2);

    if constexpr (ROLE == 0) {
        // The image's pre reader: calculate_and_prepare_reduce_scaler<reduce, SUM, REDUCE_ROW>() (scaler 1.0).
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<CB_SC, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }

    const uint32_t tsz = get_tile_size(CB_H);
    const auto h = TensorAccessor(h_args, h_addr, tsz);
    const uint32_t base_l1 = get_write_ptr(CB_H);
    const uint32_t base_tile = row * WT;
    constexpr uint32_t NCHUNK = (WT + CHUNK - 1) / CHUNK;
    for (uint32_t j = ROLE; j < NCHUNK; j += 2) {
        const uint32_t t0 = j * CHUNK;
        const uint32_t t1 = (t0 + CHUNK < WT) ? t0 + CHUNK : WT;
        cb_reserve_back(CB_CNT, 1);
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_read_page(base_tile + t, h, base_l1 + t * tsz);
        }
        noc_async_read_barrier();
        cb_push_back(CB_CNT, 1);
    }

    if constexpr (ROLE == 1) {
        const uint32_t ssz = get_tile_size(CB_ST);
        const auto st = TensorAccessor(s_args, stats_addr, ssz);
        cb_wait_front(CB_ST, 1);
        noc_async_write_page(row, st, get_read_ptr(CB_ST));
        noc_async_write_barrier();
        cb_pop_front(CB_ST, 1);
    }
}

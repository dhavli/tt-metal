// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_ADD (dnorm_fuse.py): data movement of the PRE pass with the residual add folded in
// (h = a + b, then the RMSNorm statistics of h). One core per 32-row tile row r; both RISCs run this source.
// ROLE j % 2 reads chunk j (CHUNK tiles) of a and of b into its OWN pair of streaming CBs (CB_A0/CB_B0 or CB_A1/CB_B1,
// so each CB has one producer), one barrier per chunk. Compute adds them straight into CB_H (backed by this core's shard
// of the L1 height-sharded hs, which the POST pass reads). After its reads, each role writes its chunks of h from CB_H
// to the interleaved DRAM output h (the residual stream the next add needs), waiting on CB_H's cumulative count; CB_H
// is never popped. ROLE 0 builds the reduce scaler; ROLE 1 writes the row's stats tile.
//
// Compile args: {CB_A, CB_B, CB_H, CB_SC, CB_ST, WT, CHUNK, ROLE}
//               + TensorAccessorArgs(a) + TensorAccessorArgs(b) + TensorAccessorArgs(h) + TensorAccessorArgs(stats)
// Runtime args: {a_addr, b_addr, h_addr, stats_addr, row}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t CB_A = get_compile_time_arg_val(0);
    constexpr uint32_t CB_B = get_compile_time_arg_val(1);
    constexpr uint32_t CB_H = get_compile_time_arg_val(2);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(3);
    constexpr uint32_t CB_ST = get_compile_time_arg_val(4);
    constexpr uint32_t WT = get_compile_time_arg_val(5);
    constexpr uint32_t CHUNK = get_compile_time_arg_val(6);
    constexpr uint32_t ROLE = get_compile_time_arg_val(7);
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto h_args = TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
    constexpr auto s_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_arg_val<uint32_t>(1);
    const uint32_t h_addr = get_arg_val<uint32_t>(2);
    const uint32_t stats_addr = get_arg_val<uint32_t>(3);
    const uint32_t row = get_arg_val<uint32_t>(4);

    if constexpr (ROLE == 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<CB_SC, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();
    }

    const uint32_t asz = get_tile_size(CB_A), bsz = get_tile_size(CB_B), hsz = get_tile_size(CB_H);
    const auto a = TensorAccessor(a_args, a_addr, asz);
    const auto b = TensorAccessor(b_args, b_addr, bsz);
    const auto h = TensorAccessor(h_args, h_addr, hsz);
    const uint32_t base_tile = row * WT;
    constexpr uint32_t NCHUNK = (WT + CHUNK - 1) / CHUNK;
    for (uint32_t j = ROLE; j < NCHUNK; j += 2) {
        const uint32_t t0 = j * CHUNK;
        const uint32_t n = (t0 + CHUNK < WT) ? CHUNK : WT - t0;
        cb_reserve_back(CB_A, n);
        cb_reserve_back(CB_B, n);
        const uint32_t la = get_write_ptr(CB_A), lb = get_write_ptr(CB_B);
        for (uint32_t i = 0; i < n; ++i) {
            noc_async_read_page(base_tile + t0 + i, a, la + i * asz);
            noc_async_read_page(base_tile + t0 + i, b, lb + i * bsz);
        }
        noc_async_read_barrier();
        cb_push_back(CB_A, n);
        cb_push_back(CB_B, n);
    }

    const uint32_t hbase = get_read_ptr(CB_H);
    for (uint32_t j = ROLE; j < NCHUNK; j += 2) {
        const uint32_t t0 = j * CHUNK;
        const uint32_t t1 = (t0 + CHUNK < WT) ? t0 + CHUNK : WT;
        cb_wait_front(CB_H, t1);  // cumulative: chunk j has been added
        for (uint32_t t = t0; t < t1; ++t) {
            noc_async_write_page(base_tile + t, h, hbase + t * hsz);
        }
    }
    noc_async_write_barrier();

    if constexpr (ROLE == 1) {
        const uint32_t ssz = get_tile_size(CB_ST);
        const auto st = TensorAccessor(s_args, stats_addr, ssz);
        cb_wait_front(CB_ST, 1);
        noc_async_write_page(row, st, get_read_ptr(CB_ST));
        noc_async_write_barrier();
        cb_pop_front(CB_ST, 1);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_DNORM_FUSE (dnorm_fuse.py): data-movement half of the fused POST pass, one 32-row tile row per
// core. h is NOT read: it is still in this core's shard of hs (CB_H), left there by the PRE pass.
// ROLE 0 (reader): the reduce scaler (AVG over REDUCE_FACTOR, the image's post reader), the eps column tile
//   (generate_bcast_col_scalar's bytes), the row's STATS_COLS gathered stats tiles (tile ids row * STATS_COLS + i)
//   and the WT gamma tiles, each built from one 32-datum row-major stick exactly as the image's post reader does
//   (async_read_row_to_tile<ROW_MAJOR>: 64 B to the tile start, then datums 16..31 copied to face 1 row 0).
// ROLE 1 (writer): drains CB_OUT in BLK-tile blocks into y (interleaved), tile id row * WT + t.
//
// Compile args: {CB_SC, CB_EPS, CB_STATS, CB_G, CB_OUT, WT, BLK, STATS_COLS, ROLE}
//               + TensorAccessorArgs(stats) + TensorAccessorArgs(gamma) + TensorAccessorArgs(y)
// Runtime args: {stats_addr, gamma_addr, y_addr, row, gamma_shift, eps_bits}  (gamma_shift 1 = the MUSTFAIL: tile t
//               gets gamma stick (t + 1) % WT)

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

#ifndef REDUCE_FACTOR
#error "REDUCE_FACTOR (logical W x num_devices) must be defined"
#endif

void kernel_main() {
    constexpr uint32_t CB_SC = get_compile_time_arg_val(0);
    constexpr uint32_t CB_EPS = get_compile_time_arg_val(1);
    constexpr uint32_t CB_STATS = get_compile_time_arg_val(2);
    constexpr uint32_t CB_G = get_compile_time_arg_val(3);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(4);
    constexpr uint32_t WT = get_compile_time_arg_val(5);
    constexpr uint32_t BLK = get_compile_time_arg_val(6);
    constexpr uint32_t STATS_COLS = get_compile_time_arg_val(7);
    constexpr uint32_t ROLE = get_compile_time_arg_val(8);
    constexpr auto s_args = TensorAccessorArgs<9>();
    constexpr auto g_args = TensorAccessorArgs<s_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t stats_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t y_addr = get_arg_val<uint32_t>(2);
    const uint32_t row = get_arg_val<uint32_t>(3);
    const uint32_t gamma_shift = get_arg_val<uint32_t>(4);
    const uint32_t eps_bits = get_arg_val<uint32_t>(5);

    if constexpr (ROLE == 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            CB_SC, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW, REDUCE_FACTOR>();

        // generate_bcast_col_scalar(eps): column 0 of faces 0 and 2 (the rest of the tile is not read downstream).
        {
            const uint16_t v = eps_bits >> 16;
            cb_reserve_back(CB_EPS, 1);
            volatile tt_l1_ptr uint16_t* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(CB_EPS));
            for (int k = 0; k < 4; k += 2) {
                uint32_t idx = k << 8;
                for (int j = 0; j < 256; j += 16) {
                    p[idx + j] = v;
                }
            }
            cb_push_back(CB_EPS, 1);
        }

        const uint32_t ssz = get_tile_size(CB_STATS);
        const auto st = TensorAccessor(s_args, stats_addr, ssz);
        cb_reserve_back(CB_STATS, STATS_COLS);
        uint32_t l1 = get_write_ptr(CB_STATS);
        for (uint32_t i = 0; i < STATS_COLS; ++i) {
            noc_async_read_page(row * STATS_COLS + i, st, l1 + i * ssz);
        }
        noc_async_read_barrier();
        cb_push_back(CB_STATS, STATS_COLS);

        // gamma: one 64-byte stick per tile (bf16, ROW_MAJOR [.., WT, 32]).
        constexpr uint32_t ROW_BYTES = 32 * 2, FACE_BYTES = 256 * 2, HALF_ROW_BYTES = 16 * 2;
        const auto g = TensorAccessor(g_args, gamma_addr, ROW_BYTES);
        cb_reserve_back(CB_G, WT);
        const uint32_t g0 = get_write_ptr(CB_G);
        const uint32_t gsz = get_tile_size(CB_G);
        for (uint32_t t = 0; t < WT; ++t) {
            noc_async_read(g.get_noc_addr((t + gamma_shift) % WT), g0 + t * gsz, ROW_BYTES);
        }
        noc_async_read_barrier();
        for (uint32_t t = 0; t < WT; ++t) {
            const uint32_t a = g0 + t * gsz;
            noc_async_read(get_noc_addr(my_x[noc_index], my_y[noc_index], a + HALF_ROW_BYTES), a + FACE_BYTES, ROW_BYTES);
        }
        noc_async_read_barrier();
        cb_push_back(CB_G, WT);
    } else {
        const uint32_t osz = get_tile_size(CB_OUT);
        const auto y = TensorAccessor(y_args, y_addr, osz);
        const uint32_t base_tile = row * WT;
        for (uint32_t t = 0; t < WT; t += BLK) {
            const uint32_t n = (t + BLK <= WT) ? BLK : WT - t;
            cb_wait_front(CB_OUT, n);
            uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write_page(base_tile + t + i, y, l1 + i * osz);
            }
            noc_async_write_barrier();
            cb_pop_front(CB_OUT, n);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_TAIL (gdn_tail.py): data-movement half of the fused GDN per-head RMSNorm + head->token relayout.
// The input o is head-major [NH, T, D] (TILE, fp32); tile row R = h * TT + r holds head h, token tile r. This core
// owns rows [row0, row0 + nrows).
// ROLE 0 (reader): the reduce scaler (SUM, REDUCE_ROW, factor 1: the image reader's
//   calculate_and_prepare_reduce_scaler), the eps column tile (generate_bcast_col_scalar's bytes), the WT gamma tiles
//   (tile ids 0..WT-1 of the [1, 1, D] TILE weight, as the image reader reads them; gamma_shift 1 is the MUSTFAIL:
//   tile t gets gamma tile (t + 1) % WT), then each row's WT input tiles (tile id R * WT + j).
// ROLE 1 (writer): drains CB_OUT, WT tiles per row, into the token-major [T, NH * D] output: row R's tile j goes to
//   tile id r * (NH * WT) + h * WT + j, which is nlp_concat_heads' placement, so no relayout op is needed.
//
// Compile args: {CB_IN, CB_SC, CB_EPS, CB_G, CB_OUT, WT, TT, NH, ROLE}
//               + TensorAccessorArgs(o) + TensorAccessorArgs(gamma) + TensorAccessorArgs(out)
// Runtime args: {o_addr, gamma_addr, out_addr, row0, nrows, gamma_shift, eps_bits}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_SC = get_compile_time_arg_val(1);
    constexpr uint32_t CB_EPS = get_compile_time_arg_val(2);
    constexpr uint32_t CB_G = get_compile_time_arg_val(3);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(4);
    constexpr uint32_t WT = get_compile_time_arg_val(5);
    constexpr uint32_t TT = get_compile_time_arg_val(6);
    constexpr uint32_t NH = get_compile_time_arg_val(7);
    constexpr uint32_t ROLE = get_compile_time_arg_val(8);
    constexpr auto o_args = TensorAccessorArgs<9>();
    constexpr auto g_args = TensorAccessorArgs<o_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

    const uint32_t o_addr = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t row0 = get_arg_val<uint32_t>(3);
    const uint32_t nrows = get_arg_val<uint32_t>(4);
    const uint32_t gamma_shift = get_arg_val<uint32_t>(5);
    const uint32_t eps_bits = get_arg_val<uint32_t>(6);

    if constexpr (ROLE == 0) {
        dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
            CB_SC, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

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

        const uint32_t gsz = get_tile_size(CB_G);
        const auto g = TensorAccessor(g_args, gamma_addr, gsz);
        cb_reserve_back(CB_G, WT);
        const uint32_t g0 = get_write_ptr(CB_G);
        for (uint32_t t = 0; t < WT; ++t) {
            noc_async_read_page((t + gamma_shift) % WT, g, g0 + t * gsz);
        }
        noc_async_read_barrier();
        cb_push_back(CB_G, WT);

        const uint32_t isz = get_tile_size(CB_IN);
        const auto src = TensorAccessor(o_args, o_addr, isz);
#ifdef GTL_DM
        // QWEN36_PREFILL_GDN_TAIL_DM: 2 rows per read barrier (CB_IN holds 4 rows); compute still takes one row
        constexpr uint32_t RR = 2;
#else
        constexpr uint32_t RR = 1;
#endif
        for (uint32_t k = 0; k < nrows; k += RR) {
            const uint32_t m = (k + RR <= nrows) ? RR : nrows - k;
            cb_reserve_back(CB_IN, m * WT);
            const uint32_t l1 = get_write_ptr(CB_IN);
            for (uint32_t q = 0; q < m; ++q) {
                const uint32_t R = row0 + k + q;
                for (uint32_t j = 0; j < WT; ++j) {
                    noc_async_read_page(R * WT + j, src, l1 + (q * WT + j) * isz);
                }
            }
            noc_async_read_barrier();
            cb_push_back(CB_IN, m * WT);
        }
    } else {
        const uint32_t osz = get_tile_size(CB_OUT);
        const auto y = TensorAccessor(y_args, out_addr, osz);
        for (uint32_t k = 0; k < nrows; ++k) {
            const uint32_t R = row0 + k;
            const uint32_t h = R / TT;
            const uint32_t r = R - h * TT;
            const uint32_t base = r * (NH * WT) + h * WT;
            cb_wait_front(CB_OUT, WT);
            const uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t j = 0; j < WT; ++j) {
                noc_async_write_page(base + j, y, l1 + j * osz);
            }
#ifdef GTL_DM
            noc_async_writes_flushed();
#else
            noc_async_write_barrier();
#endif
            cb_pop_front(CB_OUT, WT);
        }
#ifdef GTL_DM
        noc_async_write_barrier();
#endif
    }
}

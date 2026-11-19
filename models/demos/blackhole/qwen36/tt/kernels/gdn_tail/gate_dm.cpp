// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_TAIL (gdn_tail.py): data movement of the fused GDN out-gate. This core owns output tiles
// [t0, t0 + ntiles) of the [T, C] token-major tensors (a fp32, z bf16, out bf16 share the tile grid).
// ROLE 0 (reader): a and z tiles, N_PER at a time, one barrier per group. z may be a wider tensor read at a column tile
//   offset (z tile = r * Z_WT + Z_C0 + c for out tile r * C_T + c).
// ROLE 1 (writer): out tiles, N_PER at a time.
//
// Compile args: {CB_A, CB_Z, CB_OUT, N_PER, C_T, Z_WT, Z_C0, ROLE} + TensorAccessorArgs(a) + (z) + (out)
// Runtime args: {a_addr, z_addr, out_addr, t0, ntiles}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_A = get_compile_time_arg_val(0);
    constexpr uint32_t CB_Z = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
    constexpr uint32_t N_PER = get_compile_time_arg_val(3);
    constexpr uint32_t C_T = get_compile_time_arg_val(4);
    constexpr uint32_t Z_WT = get_compile_time_arg_val(5);
    constexpr uint32_t Z_C0 = get_compile_time_arg_val(6);
    constexpr uint32_t ROLE = get_compile_time_arg_val(7);
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto z_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    constexpr auto y_args = TensorAccessorArgs<z_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t z_addr = get_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_arg_val<uint32_t>(2);
    const uint32_t t0 = get_arg_val<uint32_t>(3);
    const uint32_t ntiles = get_arg_val<uint32_t>(4);

    if constexpr (ROLE == 0) {
        const uint32_t asz = get_tile_size(CB_A), zsz = get_tile_size(CB_Z);
        const auto A = TensorAccessor(a_args, a_addr, asz);
        const auto Z = TensorAccessor(z_args, z_addr, zsz);
#ifdef GTL_DM
        // QWEN36_PREFILL_GDN_TAIL_DM: GTL_DM tiles per read barrier (the CBs hold 2 * GTL_DM); compute still
        // consumes N_PER at a time.
        constexpr uint32_t RB = GTL_DM;
#else
        constexpr uint32_t RB = N_PER;
#endif
        for (uint32_t t = 0; t < ntiles; t += RB) {
            const uint32_t n = (t + RB <= ntiles) ? RB : ntiles - t;
            cb_reserve_back(CB_A, n);
            cb_reserve_back(CB_Z, n);
            const uint32_t la = get_write_ptr(CB_A), lz = get_write_ptr(CB_Z);
            for (uint32_t i = 0; i < n; ++i) {
                const uint32_t id = t0 + t + i;
                const uint32_t r = id / C_T, c = id - r * C_T;
                noc_async_read_page(id, A, la + i * asz);
                noc_async_read_page(r * Z_WT + Z_C0 + c, Z, lz + i * zsz);
            }
            noc_async_read_barrier();
            cb_push_back(CB_A, n);
            cb_push_back(CB_Z, n);
        }
    } else {
        const uint32_t osz = get_tile_size(CB_OUT);
        const auto Y = TensorAccessor(y_args, out_addr, osz);
#ifdef GTL_DM
        constexpr uint32_t WB = GTL_DM;
#else
        constexpr uint32_t WB = N_PER;
#endif
        for (uint32_t t = 0; t < ntiles; t += WB) {
            const uint32_t n = (t + WB <= ntiles) ? WB : ntiles - t;
            cb_wait_front(CB_OUT, n);
            const uint32_t l1 = get_read_ptr(CB_OUT);
            for (uint32_t i = 0; i < n; ++i) {
                noc_async_write_page(t0 + t + i, Y, l1 + i * osz);
            }
#ifdef GTL_DM
            noc_async_writes_flushed();  // the L1 source may be reused once the writes have left; acks at the end
#else
            noc_async_write_barrier();
#endif
            cb_pop_front(CB_OUT, n);
        }
#ifdef GTL_DM
        noc_async_write_barrier();
#endif
    }
}

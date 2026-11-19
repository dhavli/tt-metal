// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_TAIL (gdn_tail.py): compute half of the fused GDN out-gate, out = a * silu(z) in ONE pass.
// Today: s = ttnn.silu(z) (unary, bf16 z, fp32 DEST off, HiFi4, not approx: silu_tile<false> rounds its result to bf16
// RNE in the SFPU, which the 16-bit DEST stores and the packer writes unchanged) then ttnn.multiply(a fp32, s bf16,
// dtype=bf16) (binary_ng SFPU path: fp32 DEST, UnpackToDestFp32 on both inputs, mul_binary_tile, packed bf16).
// Here, in one fp32-DEST kernel: z is unpacked to DEST as fp32 (exact for bf16), silu_tile<false> runs the SAME SFPU
// code (its result is bf16-exact, so DEST fp32 holds what the 16-bit DEST held and what unpacking s gives the
// multiply), then mul_binary_tile on a (UnpackToDestFp32) and that value, the same typecast_tile, packed bf16. Same bytes, no s round trip.
//
// Compile args: {CB_A, CB_Z, CB_OUT, N_PER}  (N_PER tiles per DEST acquire; each tile uses DEST slots 2i, 2i+1)
// Runtime args: {ntiles}

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/typecast.h"
#include "api/dataflow/circular_buffer.h"

#ifndef GATE_MUSTFAIL
#define GATE_MUSTFAIL 0
#endif

void kernel_main() {
    constexpr uint32_t CB_A = get_compile_time_arg_val(0);
    constexpr uint32_t CB_Z = get_compile_time_arg_val(1);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(2);
    constexpr uint32_t N_PER = get_compile_time_arg_val(3);
    const uint32_t ntiles = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(CB_A, CB_OUT);
    copy_init(CB_A);

    for (uint32_t t = 0; t < ntiles; t += N_PER) {
        const uint32_t n = (t + N_PER <= ntiles) ? N_PER : ntiles - t;
        cb_wait_front(CB_A, n);
        cb_wait_front(CB_Z, n);
        cb_reserve_back(CB_OUT, n);
        tile_regs_acquire();
        reconfig_data_format_srca(CB_Z, CB_A);
        copy_init(CB_A);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_A, i, i * 2);
        }
        reconfig_data_format_srca(CB_A, CB_Z);
        copy_init(CB_Z);
        for (uint32_t i = 0; i < n; ++i) {
            copy_tile(CB_Z, i, i * 2 + 1);
#if GATE_MUSTFAIL
            // wrong-input branch (self-test / QWEN36_PREFILL_GDN_TAIL_MUSTFAIL): no silu, out = a * z; differs
            // wherever silu(z) != z
#else
            silu_tile_init();
            silu_tile<false>(i * 2 + 1);
#endif
            mul_binary_tile_init();
            mul_binary_tile(i * 2, i * 2 + 1, i * 2);
            // binary_ng's bf16-output post-activation: SFPU fp32 -> bf16 RNE (the packer alone rounds ties differently)
            typecast_tile_init<0u, 5u>();
            typecast_tile<0u, 5u>(i * 2);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < n; ++i) {
            pack_tile(i * 2, CB_OUT);
        }
        tile_regs_release();
        cb_push_back(CB_OUT, n);
        cb_pop_front(CB_A, n);
        cb_pop_front(CB_Z, n);
    }
}

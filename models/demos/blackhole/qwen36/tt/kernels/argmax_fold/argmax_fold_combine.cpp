// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_ARGMAX_FOLD (argmax_fold.py): combine kernel, one core, data movement only. Replaces the served 4 slices,
// gt, typecast, subtract, add, multiply, add, typecast (10 programs) after the all-gather. Input: the gathered pair
// tiles, logical [1, 1, B, 2 * NDEV] float32 TILE whose elements are raw bits (argmax_fold_pack.cpp): device d's
// (index, bf16 max << 16) at columns d*2, d*2 + 1 when the gather packed the logical columns into one tile
// (NT == 1), or at tile d columns 0, 1 when it concatenated whole tiles (NT == NDEV). Output: [1, 1, B -> 32, 1 -> 32]
// uint32 TILE, zero-filled, element (b, 0) = the global greedy id.
//
// Tie rule = the served chain's (model.py _greedy_argmax_device, vLLM greedy, lowest index): shard 1 wins only when
// its max is STRICTLY greater (as float; NaN never wins, -0 == +0), else shard 0. tok = win ? i1 + PER : i0, in
// integers (the served fp32 chain is exact below 2^24, so the same value).
//
// MF (compile arg, QWEN36_ARGMAX_FOLD_MUSTFAIL): tok + 1: the self-test must REFUSE it.
//
// Compile args: {CB_SCR, B, NDEV, NT, PER, MF} + TensorAccessorArgs(gathered) + (out)
// Runtime args: {g_addr, out_addr}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Ordered key of a bf16 bit pattern for a strict float comparison: sign-magnitude -> signed; +-0 -> 0; NaN -> flagged.
FORCE_INLINE int32_t bf16_key(uint32_t h, bool& nan) {
    const uint32_t mag = h & 0x7fffu;
    nan = mag > 0x7f80u;
    if (mag == 0) {
        return 0;
    }
    return (h & 0x8000u) ? -static_cast<int32_t>(mag) : static_cast<int32_t>(mag);
}

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(0);
    constexpr uint32_t B = get_compile_time_arg_val(1);
    constexpr uint32_t NDEV = get_compile_time_arg_val(2);
    constexpr uint32_t NT = get_compile_time_arg_val(3);
    constexpr uint32_t PER = get_compile_time_arg_val(4);
    constexpr uint32_t MF = get_compile_time_arg_val(5);
    constexpr auto g_args = TensorAccessorArgs<6>();
    constexpr auto o_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();
    constexpr uint32_t TILE_BYTES = 4096;
    constexpr uint32_t FACE_BYTES = 1024;

    const uint32_t g_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const auto gath = TensorAccessor(g_args, g_addr, TILE_BYTES);
    const auto out = TensorAccessor(o_args, out_addr, TILE_BYTES);

    const uint32_t scr = get_write_ptr(CB_SCR);
    const uint32_t in_l1 = scr;                     // NT gathered tiles
    const uint32_t out_l1 = scr + NT * TILE_BYTES;  // the output tile
    for (uint32_t k = 0; k < NT; ++k) {
        noc_async_read_page(k, gath, in_l1 + k * TILE_BYTES);
    }
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_l1);
    for (uint32_t i = 0; i < TILE_BYTES / 4; ++i) {
        o[i] = 0;
    }
    noc_async_read_barrier();

    auto elem = [&](uint32_t b, uint32_t d, uint32_t c) -> uint32_t {
        uint32_t tile = 0, col = d * 2 + c;
        if constexpr (NT > 1) {
            tile = d;
            col = c;
        }
        const uint32_t off = tile * TILE_BYTES + ((b >= 16 ? 2u : 0u) + (col >= 16 ? 1u : 0u)) * FACE_BYTES +
                             (b & 15) * 64 + (col & 15) * 4;
        return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in_l1 + off);
    };
    for (uint32_t b = 0; b < B; ++b) {
        uint32_t tok = elem(b, 0, 0);
        if constexpr (NDEV > 1) {
            bool n0, n1;
            const int32_t k0 = bf16_key(elem(b, 0, 1) >> 16, n0);
            const int32_t k1 = bf16_key(elem(b, 1, 1) >> 16, n1);
            if (!n0 && !n1 && k1 > k0) {
                tok = elem(b, 1, 0) + PER;
            }
        }
        if constexpr (MF) {
            tok += 1;
        }
        o[((b >= 16 ? 2u : 0u) * FACE_BYTES + (b & 15) * 64) / 4] = tok;
    }
    noc_async_write_page(0, out, out_l1);
    noc_async_write_barrier();
}

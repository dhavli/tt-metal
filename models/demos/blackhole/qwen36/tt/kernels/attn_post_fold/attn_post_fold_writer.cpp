// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_ATTN_POST_FOLD (attention/attn_post_fold.py): writer half (RISCV_1). Core b = batch row b. The gated row
// sits in CB_OUT as WT bf16 tiles ([32 (heads, NH used), HD]). The output is [1, 1, B -> 32, NH*HD] bf16 TILE
// (interleaved, tile id = column tile since it is one tile row). Head h of user b goes to output columns
// h*HD .. h*HD + HD - 1, row b: for each input tile t (columns t*32..) and each of its two column faces f, the
// 16-element row h (32 bytes at face (h >= 16) * 2 + f, row h % 16) lands in output tile h*WT + t, face
// (b >= 16) * 2 + f, row b % 16. The same 32-byte moves nlp_concat_heads_decode's reader makes, straight to the
// interleaved output. Rows >= B are not written (as the served op leaves them).
//
// MF (compile arg, QWEN36_ATTN_POST_FOLD_MUSTFAIL): head h is written to head slot (h + 1) % NH: the self-test must
// REFUSE it.
//
// Compile args: {CB_OUT, WT, NH, MF} + TensorAccessorArgs(out)
// Runtime args: {out_addr, b}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(0);
    constexpr uint32_t WT = get_compile_time_arg_val(1);
    constexpr uint32_t NH = get_compile_time_arg_val(2);
    constexpr uint32_t MF = get_compile_time_arg_val(3);
    constexpr auto o_args = TensorAccessorArgs<4>();
    constexpr uint32_t FACE_BYTES = 512;  // 16 x 16 bf16
    constexpr uint32_t ROW_BYTES = 32;    // 16 bf16

    const uint32_t out_addr = get_arg_val<uint32_t>(0);
    const uint32_t b = get_arg_val<uint32_t>(1);

    const uint32_t tile_bytes = get_tile_size(CB_OUT);
    const auto out = TensorAccessor(o_args, out_addr, tile_bytes);

    cb_wait_front(CB_OUT, WT);
    const uint32_t src_base = get_read_ptr(CB_OUT);
    const uint32_t dst_face0 = (b >= 16 ? 2u : 0u) * FACE_BYTES + (b & 15) * ROW_BYTES;
    for (uint32_t h = 0; h < NH; ++h) {
        const uint32_t slot = MF ? (h + 1) % NH : h;
        const uint32_t src_h = (h >= 16 ? 2u : 0u) * FACE_BYTES + (h & 15) * ROW_BYTES;
        for (uint32_t t = 0; t < WT; ++t) {
            const uint32_t src_t = src_base + t * tile_bytes + src_h;
            const uint32_t out_tile = slot * WT + t;
            for (uint32_t f = 0; f < 2; ++f) {
                noc_async_write(src_t + f * FACE_BYTES, out.get_noc_addr(out_tile, dst_face0 + f * FACE_BYTES), ROW_BYTES);
            }
        }
    }
    noc_async_write_barrier();
    cb_pop_front(CB_OUT, WT);
}

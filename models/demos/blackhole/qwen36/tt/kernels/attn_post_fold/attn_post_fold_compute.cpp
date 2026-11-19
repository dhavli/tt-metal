// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_ATTN_POST_FOLD (attention/attn_post_fold.py): compute half. Per tile t of the row:
//   1. sigmoid(gate_t): copy_tile to DEST, sigmoid_tile_init<false>() + sigmoid_tile<VectorMode::RC, false>(0) -- the
//      calls ttnn.sigmoid's eltwise_sfpu.cpp emits for SigmoidMode::ACCURATE (unary_op_utils.cpp: param1 = 0), at the
//      unary op's HiFi4 / bf16 DEST (fp32_dest_acc_en false for bf16 in/out); packed to CB_SIG as bf16;
//   2. attn_t * sig_t: copy_tile both to DEST (fp32 unpack, as binary_ng configures the source CBs of an SFPU op),
//      mul_binary_tile_init() + mul_binary_tile(0, 1, 0) -- binary_ng's eltwise_binary_sfpu_no_bcast.cpp, the
//      kernel ttnn.multiply runs for bf16 x bf16 (is_binary_sfpu_op: MUL is SFPU unless fast_and_approximate_mode;
//      measured in ttsim: the FPU mul_tiles differs in 27% of elements, the SFPU product == RNE(a*b)); packed to
//      CB_OUT as bf16.
// Same LLK calls, DEST and unpack modes as the two ttnn ops, so the same bytes (self-tested bitwise).
//
// Compile args: {CB_ATTN, CB_GATE, CB_SIG, CB_OUT, WT}

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"

void kernel_main() {
    constexpr uint32_t CB_ATTN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_GATE = get_compile_time_arg_val(1);
    constexpr uint32_t CB_SIG = get_compile_time_arg_val(2);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(3);
    constexpr uint32_t WT = get_compile_time_arg_val(4);

    compute_kernel_hw_startup(CB_GATE, CB_SIG);

    // 1. sigmoid(gate) -> CB_SIG (eltwise_sfpu.cpp, SFPU_OP_CHAIN_0 = sigmoid accurate)
    cb_wait_front(CB_GATE, WT);
    cb_reserve_back(CB_SIG, WT);
    copy_init(CB_GATE);
    for (uint32_t t = 0; t < WT; ++t) {
        tile_regs_acquire();
        copy_tile(CB_GATE, t, 0);
        sigmoid_tile_init<false>();
        sigmoid_tile<VectorMode::RC, false>(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_SIG);
        tile_regs_release();
    }
    cb_push_back(CB_SIG, WT);
    cb_pop_front(CB_GATE, WT);

    // 2. attn * sig -> CB_OUT (eltwise_binary_sfpu_no_bcast.cpp, BINARY_SFPU_OP = mul_binary_tile)
    cb_wait_front(CB_ATTN, WT);
    cb_wait_front(CB_SIG, WT);
    cb_reserve_back(CB_OUT, WT);
    mul_binary_tile_init();
    pack_reconfig_data_format(CB_OUT);
    for (uint32_t t = 0; t < WT; ++t) {
        tile_regs_acquire();
        reconfig_data_format_srca(CB_SIG, CB_ATTN);
        copy_init(CB_ATTN);
        copy_tile(CB_ATTN, t, 0);
        reconfig_data_format_srca(CB_ATTN, CB_SIG);
        copy_init(CB_SIG);
        copy_tile(CB_SIG, t, 1);
        mul_binary_tile(0, 1, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, CB_OUT);
        tile_regs_release();
    }
    cb_push_back(CB_OUT, WT);
    cb_pop_front(CB_ATTN, WT);
    cb_pop_front(CB_SIG, WT);
}

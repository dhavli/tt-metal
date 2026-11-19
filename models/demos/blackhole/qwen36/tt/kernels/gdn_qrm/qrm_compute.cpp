// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_PREFILL_GDN_QKV_RM (gdn_qrm.py): compute half. The image's untilize helper (the same pack_untilize LLK
// ttnn.to_layout(ROW_MAJOR) runs for bf16 TILE input), NB tiles per block; bf16 in, bf16 out: a pure data move.
// Compile args: {CB_IN, CB_OUT, NB}   Runtime args: {nblocks}

#include <cstdint>
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t CB_IN = get_compile_time_arg_val(0);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(1);
    constexpr uint32_t NB = get_compile_time_arg_val(2);
    const uint32_t nblocks = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_IN, CB_OUT);
    compute_kernel_lib::untilize<
        NB,
        CB_IN,
        CB_OUT,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(nblocks);
}

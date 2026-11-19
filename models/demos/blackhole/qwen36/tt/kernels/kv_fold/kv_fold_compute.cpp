// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_KV_UPDATE_FOLD (attention/kv_fold.py): compute half of the one-program paged K + V update (see
// kv_fold_reader.cpp). The same helper calls as paged_update_cache's compute kernel (update_cache.cpp), once for the
// K cache and once for the V cache: untilize the row's input shard; per cache head untilize the cache block, then
// (after the writer patched one row) tilize it back into the cache's format. The helpers' default register
// reconfiguration switches the unpack / pack formats between the K (bf8) and V (bf4) caches.
//
// Compile args: {CB_CK, CB_CV, CB_INK, CB_INV, CB_UC, CB_UC2, CB_UINK, CB_UINV, CB_OK, CB_OV, WT, NH}

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tilize.h"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

void kernel_main() {
    constexpr uint32_t CB_CK = get_compile_time_arg_val(0);
    constexpr uint32_t CB_CV = get_compile_time_arg_val(1);
    constexpr uint32_t CB_INK = get_compile_time_arg_val(2);
    constexpr uint32_t CB_INV = get_compile_time_arg_val(3);
    constexpr uint32_t CB_UC = get_compile_time_arg_val(4);
    constexpr uint32_t CB_UC2 = get_compile_time_arg_val(5);
    constexpr uint32_t CB_UINK = get_compile_time_arg_val(6);
    constexpr uint32_t CB_UINV = get_compile_time_arg_val(7);
    constexpr uint32_t CB_OK = get_compile_time_arg_val(8);
    constexpr uint32_t CB_OV = get_compile_time_arg_val(9);
    constexpr uint32_t WT = get_compile_time_arg_val(10);
    constexpr uint32_t NH = get_compile_time_arg_val(11);

    compute_kernel_hw_startup(CB_INK, CB_UINK);

    // Untilize both input shards (the first exactly as update_cache.cpp: no register reconfig after the startup).
    compute_kernel_lib::untilize<
        WT,
        CB_INK,
        CB_UINK,
        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(1);
    compute_kernel_lib::untilize<WT, CB_INV, CB_UINV>(1);

    for (uint32_t h = 0; h < NH; ++h) {
        compute_kernel_lib::untilize<WT, CB_CK, CB_UC>(1);
        compute_kernel_lib::tilize<WT, CB_UC2, CB_OK>(1);
    }
    for (uint32_t h = 0; h < NH; ++h) {
        compute_kernel_lib::untilize<WT, CB_CV, CB_UC>(1);
        compute_kernel_lib::tilize<WT, CB_UC2, CB_OV>(1);
    }
}

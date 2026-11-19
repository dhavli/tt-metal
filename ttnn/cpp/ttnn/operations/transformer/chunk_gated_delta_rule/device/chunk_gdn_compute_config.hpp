// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

// The one compute configuration for every chunk_gated_delta_rule program (mono, phased, fused). The
// kernels keep the recurrent state and the WY inverse in fp32 DEST. HiFi4 is the validated default; cfx 0031
// also admits HiFi3 and HiFi2 (still fp32 DEST, no approx) so the model can screen fidelity per call behind
// QWEN36_GDN_SCAN_FID; LoFi and bf16 DEST stay rejected. dst_full_sync_en passes through, other knobs are unused.
inline tt::tt_metal::ComputeConfigDescriptor gdn_compute_config(const DeviceComputeKernelConfig& cfg) {
    using tt::tt_metal::MathFidelity;
    TT_FATAL(
        (cfg.math_fidelity == MathFidelity::HiFi4 || cfg.math_fidelity == MathFidelity::HiFi3 ||
         cfg.math_fidelity == MathFidelity::HiFi2) &&
            cfg.fp32_dest_acc_en && !cfg.math_approx_mode,
        "chunk_gated_delta_rule runs HiFi2..HiFi4 with fp32 destination accumulation and no approx mode on every "
        "path (its recurrent state and WY inverse are fp32); compute_kernel_config asked for math_fidelity {} "
        "(HiFi4 = {}), fp32_dest_acc_en {}, math_approx_mode {}",
        static_cast<int>(cfg.math_fidelity),
        static_cast<int>(MathFidelity::HiFi4),
        cfg.fp32_dest_acc_en,
        cfg.math_approx_mode);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = cfg.math_fidelity,
        .fp32_dest_acc_en = cfg.fp32_dest_acc_en,
        .dst_full_sync_en = cfg.dst_full_sync_en,
        .math_approx_mode = cfg.math_approx_mode};
}

}  // namespace ttnn::prim

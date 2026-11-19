// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<CoreRangeSet>& core_range_set = std::nullopt);
};

// tt-stack 0040 (lift of tenstorrent/tt-metal #58922): the multi-core DRAM-sharded decode variant, selected only
// when MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::cores_per_bank > 0. A Metal 2.0 ProgramSpec factory.
struct MatmulDRAMShardedMulticoreProgramFactory {
    static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
        const ttnn::prim::MatmulParams& operation_attributes,
        const ttnn::prim::MatmulInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

}  // namespace ttnn::prim

#include "tt-metalium/tt_stack_public_defaults.hpp"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "qkv_causal_conv1d_silu_device_operation.hpp"

#include <array>
#include <cstdlib>
#include <string_view>
#include <utility>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

QkvCausalConv1dSiluOperation::program_factory_t QkvCausalConv1dSiluOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return QkvCausalConv1dSiluProgramFactory{};
}

void QkvCausalConv1dSiluOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr std::string_view operation_name = "qkv_causal_conv1d_silu";
    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::ROW_MAJOR, operation_name, "input");
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    check_allocated_device_tensor(in.history, operation_name, "history");
    check_layout(in.history, Layout::ROW_MAJOR, operation_name, "history");
    check_dtype(in.history, DataType::BFLOAT16, operation_name, "history");
    check_interleaved(in.history, operation_name, "history");
    check_allocated_device_tensor(in.tap0, operation_name, "tap0");
    check_layout(in.tap0, Layout::TILE, operation_name, "tap0");
    check_dtype(in.tap0, DataType::BFLOAT16, operation_name, "tap0");
    check_interleaved(in.tap0, operation_name, "tap0");
    check_allocated_device_tensor(in.tap1, operation_name, "tap1");
    check_layout(in.tap1, Layout::TILE, operation_name, "tap1");
    check_dtype(in.tap1, DataType::BFLOAT16, operation_name, "tap1");
    check_interleaved(in.tap1, operation_name, "tap1");
    check_allocated_device_tensor(in.tap2, operation_name, "tap2");
    check_layout(in.tap2, Layout::TILE, operation_name, "tap2");
    check_dtype(in.tap2, DataType::BFLOAT16, operation_name, "tap2");
    check_interleaved(in.tap2, operation_name, "tap2");
    check_allocated_device_tensor(in.tap3, operation_name, "tap3");
    check_layout(in.tap3, Layout::TILE, operation_name, "tap3");
    check_dtype(in.tap3, DataType::BFLOAT16, operation_name, "tap3");
    check_interleaved(in.tap3, operation_name, "tap3");
    check_same_device(in.input, in.history, operation_name, "history");
    check_same_device(in.input, in.tap0, operation_name, "tap0");
    check_same_device(in.input, in.tap1, operation_name, "tap1");
    check_same_device(in.input, in.tap2, operation_name, "tap2");
    check_same_device(in.input, in.tap3, operation_name, "tap3");

    TT_FATAL(
        attrs.q_width > 0 && attrs.k_width > 0 && attrs.v_width > 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be positive");
    TT_FATAL(
        attrs.q_width % tt::constants::TILE_WIDTH == 0 && attrs.k_width % tt::constants::TILE_WIDTH == 0 &&
            attrs.v_width % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: Q/K/V widths must be tile aligned");
    const uint64_t channels =
        static_cast<uint64_t>(attrs.q_width) + static_cast<uint64_t>(attrs.k_width) + attrs.v_width;
    TT_FATAL(attrs.channel_chunk_size > 0, "qkv_causal_conv1d_silu: channel_chunk_size must be positive");
    TT_FATAL(
        attrs.channel_chunk_size % tt::constants::TILE_WIDTH == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must be tile aligned");
    TT_FATAL(
        attrs.channel_chunk_size <= channels, "qkv_causal_conv1d_silu: channel_chunk_size must not exceed Q+K+V width");
    TT_FATAL(
        channels % attrs.channel_chunk_size == 0,
        "qkv_causal_conv1d_silu: channel_chunk_size must divide Q+K+V width exactly");

    const auto& input_shape = in.input.logical_shape();
    const auto& history_shape = in.history.logical_shape();
    TT_FATAL(
        input_shape.rank() == 3 && input_shape[0] == 1 && input_shape[1] == attrs.sequence &&
            input_shape[2] == channels,
        "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    TT_FATAL(
        history_shape.rank() == 3 && history_shape[0] == 1 && history_shape[1] == 3 && history_shape[2] == channels,
        "qkv_causal_conv1d_silu: history must be [1,3,Q+K+V]");
    TT_FATAL(
        attrs.sequence > 0 && attrs.sequence % tt::constants::TILE_HEIGHT == 0,
        "qkv_causal_conv1d_silu: sequence must be positive and tile aligned");

    for (const auto& [tensor, name] : std::array{
             std::pair{&in.tap0, "tap0"},
             std::pair{&in.tap1, "tap1"},
             std::pair{&in.tap2, "tap2"},
             std::pair{&in.tap3, "tap3"}}) {
        TT_FATAL(
            tensor->logical_shape()[-1] == channels,
            "qkv_causal_conv1d_silu: {} last dimension must equal Q+K+V",
            name);
        TT_FATAL(
            tensor->logical_volume() == channels, "qkv_causal_conv1d_silu: {} logical volume must equal Q+K+V", name);
    }
    check_output_interleaved(attrs.output_mem_config, operation_name);
    check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(
        !attrs.compute_kernel_config.math_approx_mode,
        "qkv_causal_conv1d_silu: math_approx_mode=true is unsupported because silu_tile always uses precise sigmoid");
}

QkvCausalConv1dSiluOperation::spec_return_value_t QkvCausalConv1dSiluOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    return {
        TensorSpec(Shape({1, attrs.sequence, attrs.q_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.k_width}), layout),
        TensorSpec(Shape({1, attrs.sequence, attrs.v_width}), layout)};
}

QkvCausalConv1dSiluOperation::tensor_return_value_t QkvCausalConv1dSiluOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    return {
        create_device_tensor(specs[0], in.input.device()),
        create_device_tensor(specs[1], in.input.device()),
        create_device_tensor(specs[2], in.input.device())};
}

namespace {

// QWEN36_GDN_CONV_KDA2 (cfx patches/tt-metal/0035), read on EVERY call (not once per process) so the model's warmup
// self-test can run the stock and the read-once programs back to back in one process:
//   unset / "" / "0"  stock programs (byte-identical to the unpatched op)
//   "1"               read-once kernels, stock silu_tile
//   "fast"            read-once kernels, silu_tile<false> (pairs with the c98 compute overlay on the stock path)
// QWEN36_GDN_CONV_KDA2_MUSTFAIL=1 additionally stages every row one row late (wrong outputs on purpose).
// Anything else is an error, so a typo cannot silently run the stock arm.
std::pair<uint32_t, uint32_t> conv_kda2_mode() {
    const char* raw = tt_stack_public::getenv("QWEN36_GDN_CONV_KDA2");
    const std::string_view v = raw == nullptr ? std::string_view{} : std::string_view{raw};
    uint32_t mode = 0;
    if (v.empty() || v == "0") {
        mode = 0;
    } else if (v == "1") {
        mode = 1;
    } else if (v == "fast") {
        mode = 2;
    } else {
        TT_THROW("QWEN36_GDN_CONV_KDA2 must be unset, 0, 1 or fast; got '{}'", v);
    }
    const char* mf_raw = tt_stack_public::getenv("QWEN36_GDN_CONV_KDA2_MUSTFAIL");
    const std::string_view mf = mf_raw == nullptr ? std::string_view{} : std::string_view{mf_raw};
    TT_FATAL(mf.empty() || mf == "0" || mf == "1", "QWEN36_GDN_CONV_KDA2_MUSTFAIL must be unset, 0 or 1; got '{}'", mf);
    return {mode, (mode != 0 && mf == "1") ? 1u : 0u};
}

}  // namespace

std::vector<Tensor> qkv_causal_conv1d_silu(
    const Tensor& input,
    const Tensor& history,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    const auto& input_shape = input.logical_shape();
    TT_FATAL(input_shape.rank() == 3, "qkv_causal_conv1d_silu: input must be [1,T,Q+K+V]");
    const auto [reads_once, reads_once_mustfail] = conv_kda2_mode();
    return ttnn::device_operation::launch<QkvCausalConv1dSiluOperation>(
        QkvCausalConv1dSiluParams{
            .sequence = static_cast<uint32_t>(input_shape[1]),
            .q_width = q_width,
            .k_width = k_width,
            .v_width = v_width,
            .channel_chunk_size = channel_chunk_size,
            .output_mem_config = output_mem_config,
            .compute_kernel_config = compute_kernel_config,
            .reads_once = reads_once,
            .reads_once_mustfail = reads_once_mustfail},
        QkvCausalConv1dSiluInputs{
            .input = input, .history = history, .tap0 = tap0, .tap1 = tap1, .tap2 = tap2, .tap3 = tap3});
}

}  // namespace ttnn::experimental::prim

#include "tt-metalium/tt_stack_public_defaults.hpp"
// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <string_view>
#include <utility>
#include <set>
#include <tuple>
#include <map>
#include <string>
#include "ttnn/metal_v2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include <tt-logger/tt-logger.hpp>

#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/compute_throttle_utils.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operations/matmul/shared_with_host/activation_type.hpp"

using namespace tt;

using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

using tt::tt_metal::CBDescriptor;
using tt::tt_metal::CBFormatDescriptor;
using tt::tt_metal::ComputeConfigDescriptor;
using tt::tt_metal::DataMovementConfigDescriptor;
using tt::tt_metal::KernelDescriptor;
using tt::tt_metal::ProgramDescriptor;

namespace ttnn::prim {
namespace reuse_dram_sharded_optimized_helpers {

using dram_sharded_helpers::get_dram_bank_reader_assignments;
using dram_sharded_helpers::get_max_page_size_and_num_pages;
using dram_sharded_helpers::move_common_entries;
using dram_sharded_helpers::validate_num_workers_per_dram_bank;

// TT_MR_IN1_DUAL_NOC (read once per process): split each multi-reader (num_workers_per_dram_bank > 1)
// bank's in1 stream across NOC0 and NOC1 by reader, not by k-row (rev 3). Every reader whose
// worker_index is odd and which is a pure in0 mcast receiver (never an in0 storage core) runs the
// unchanged in1 kernel on NOC1 and its in0 receiver kernel on NOC0; the RISC assignment and
// DM_DEDICATED_NOC are unchanged everywhere, so no kernel shares a NoC with the other RISC.
//   unset / "" / "0"  off: byte-identical programs to the single-NoC factory
//   "1"               on
//   "mustfail"        on, and every k-row of a NOC1 reader is read twice (same numerics, slower) to
//                     prove the NOC1 ELF ran
// Anything else is an error, so a typo cannot silently run the default arm.
enum class MrIn1DualNoc { OFF, ON, MUSTFAIL };

static MrIn1DualNoc mr_in1_dual_noc_mode() {
    static const MrIn1DualNoc mode = [] {
        const char* raw = tt_stack_public::getenv("TT_MR_IN1_DUAL_NOC");
        const std::string_view v = raw == nullptr ? std::string_view{} : std::string_view{raw};
        if (v.empty() || v == "0") {
            return MrIn1DualNoc::OFF;
        }
        if (v == "1") {
            return MrIn1DualNoc::ON;
        }
        if (v == "mustfail") {
            return MrIn1DualNoc::MUSTFAIL;
        }
        TT_THROW("TT_MR_IN1_DUAL_NOC must be unset, 0, 1 or mustfail; got '{}'", v);
    }();
    return mode;
}

static ProgramDescriptor create_program_dram_sharded_descriptor(
    tt::tt_metal::IDevice* device,
    const CoreRangeSet& input_all_storage_cores,
    const CoreRangeSet& output_all_storage_cores,
    MathFidelity math_fidelity,
    bool fp32_dest_acc_en,
    bool math_approx_mode,
    bool packer_l1_acc,
    bool dst_full_sync_en,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t B,
    uint32_t /* M */,
    uint32_t N,
    uint32_t K,
    uint32_t in0_block_w,
    uint32_t in0_last_ktile_w,
    uint32_t per_core_M,
    uint32_t per_core_N_storage,
    uint32_t workers_per_bank,
    std::optional<UnaryWithParam> fused_activation,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& bias_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat bias_data_format,
    tt::DataFormat output_data_format,
    bool untilize_out,
    bool skip_compute,
    bool skip_in0_mcast,
    bool skip_write_back,
    bool row_broadcast_bias) {
    using namespace tt;

    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias;
    if (bias_tensor.has_value()) {
        bias = *bias_tensor;
    }

    // currently only support transpose of the full tile
    bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    TT_FATAL(
        in1_tensor.shard_spec()->orientation == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_mcast_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    CoreCoord top_left_core = {(std::size_t)start_core_x, (std::size_t)start_core_y};
    CoreCoord bottom_right_core = {
        (std::size_t)start_core_x + compute_with_storage_grid_size.x - 1,
        (std::size_t)start_core_y + compute_with_storage_grid_size.y - 1};
    auto top_left_core_physical = device->worker_core_from_logical_core(top_left_core);
    auto bottom_right_core_physical = device->worker_core_from_logical_core(bottom_right_core);

    // in1 is the reader of weights/output writer, and we choose to make it use the optimized reader noc
    tt_metal::NOC in0_noc = tt::tt_metal::detail::preferred_noc_for_dram_write(device->arch());
    tt_metal::NOC in1_noc = tt::tt_metal::detail::preferred_noc_for_dram_read(device->arch());

    CoreCoord start_core_noc = top_left_core_physical;
    CoreCoord end_core_noc = bottom_right_core_physical;
    if (in0_noc == tt::tt_metal::NOC::NOC_1) {
        std::swap(start_core_noc, end_core_noc);
    }

    validate_num_workers_per_dram_bank(workers_per_bank);
    TT_FATAL(
        workers_per_bank == 1 || device->arch() == tt::ARCH::BLACKHOLE,
        "Multiple workers per DRAM bank are currently supported only on Blackhole");
    TT_FATAL(
        workers_per_bank == 1 || in1_noc == tt::tt_metal::NOC::NOC_0,
        "Multiple workers per DRAM bank currently require a NOC0 data-movement kernel");

    // Dual-NoC in1 (TT_MR_IN1_DUAL_NOC, rev 3): reader placement and the NOC0 requirement above are
    // unchanged; the odd-indexed secondary readers of each bank are moved to NOC1 below, once the
    // in0 mcast receivers are known. Everything stays in DM_DEDICATED_NOC (rev 1/2 used DM_DYNAMIC_NOC
    // and hung on hardware; see the patch header).
    const MrIn1DualNoc in1_dual_noc_mode = workers_per_bank > 1 ? mr_in1_dual_noc_mode() : MrIn1DualNoc::OFF;
    const bool in1_dual_noc = in1_dual_noc_mode != MrIn1DualNoc::OFF;
    if (in1_dual_noc) {
        TT_FATAL(
            device->arch() == tt::ARCH::BLACKHOLE && in1_noc == tt::tt_metal::NOC::NOC_0 &&
                in0_noc == tt::tt_metal::NOC::NOC_1,
            "TT_MR_IN1_DUAL_NOC needs Blackhole with in1 on NOC0 and in0 on NOC1");
    }
    const tt_metal::NOC_MODE dm_noc_mode = tt_metal::NOC_MODE::DM_DEDICATED_NOC;

    auto reader_assignments =
        get_dram_bank_reader_assignments(device, in1_noc, workers_per_bank, input_all_storage_cores);
    uint32_t num_dram_banks = reader_assignments.size() / workers_per_bank;

    // Remove cores assigned to padding-only DRAM banks from the workers category
    uint32_t in1_shard_width_tiles = in1_tensor.shard_spec()->shape[1] / in1_tile.get_tile_shape()[1];
    uint32_t in1_tensor_padded_width_tiles = in1_shard_width_tiles * num_dram_banks;

    if (in1_tensor_padded_width_tiles > N) {
        uint32_t padding_width_tiles = in1_tensor_padded_width_tiles - N;
        uint32_t only_padding_banks = padding_width_tiles / in1_shard_width_tiles;
        TT_FATAL(
            only_padding_banks < num_dram_banks,
            "Padding banks count {} must be less than DRAM bank count {}",
            only_padding_banks,
            num_dram_banks);
        num_dram_banks -= only_padding_banks;
        reader_assignments.resize(num_dram_banks * workers_per_bank);
    }

    TT_FATAL(
        in1_shard_width_tiles % workers_per_bank == 0,
        "DRAM-sharded matmul weight shard width {} tiles per bank must be divisible by workers_per_bank {}",
        in1_shard_width_tiles,
        workers_per_bank);

    std::vector<CoreCoord> all_worker_cores_ordered;
    all_worker_cores_ordered.reserve(reader_assignments.size());
    std::set<CoreRange> worker_ranges;
    for (const auto& assignment : reader_assignments) {
        all_worker_cores_ordered.push_back(assignment.worker_core);
        worker_ranges.insert(CoreRange(assignment.worker_core));
    }
    CoreRangeSet all_worker_cores(worker_ranges);
    const uint32_t num_workers = all_worker_cores_ordered.size();

    uint32_t per_core_N_compute = div_up(N, num_workers);
    uint32_t per_core_N_in1_sender = per_core_N_compute;
    TT_FATAL(
        workers_per_bank == 1 || in1_shard_width_tiles == workers_per_bank * per_core_N_in1_sender,
        "Multi-reader DRAM-sharded matmul requires weight shard width {} tiles to equal {} workers times {} reader "
        "tiles",
        in1_shard_width_tiles,
        workers_per_bank,
        per_core_N_in1_sender);

    auto subblock_hw = operations::matmul::bmm_op_utils::get_matmul_subblock_params(
        per_core_M, per_core_N_compute, false, false, fp32_dest_acc_en);
    auto out_subblock_h = std::get<0>(subblock_hw);
    auto out_subblock_w = std::get<1>(subblock_hw);

    uint32_t max_subblock_w = fp32_dest_acc_en ? 4 : 8;
    if (out_subblock_h == 1 and out_subblock_w < max_subblock_w) {
        uint32_t num_subblock_w_per_core_N = per_core_N_compute / out_subblock_w;
        uint32_t num_iter = max_subblock_w - out_subblock_w;
        uint32_t new_out_subblock_w = out_subblock_w;
        uint32_t preferred_out_subblock_w = out_subblock_w;

        for (uint32_t i = 0; i < num_iter; ++i) {
            new_out_subblock_w += 1;
            uint32_t new_num_subblock_w_per_core_N = (per_core_N_compute + new_out_subblock_w - 1) / new_out_subblock_w;

            if (new_num_subblock_w_per_core_N < num_subblock_w_per_core_N) {
                num_subblock_w_per_core_N = new_num_subblock_w_per_core_N;
                preferred_out_subblock_w = new_out_subblock_w;
            }
        }
        out_subblock_w = preferred_out_subblock_w;
        per_core_N_compute = out_subblock_w * num_subblock_w_per_core_N;
    }

    // Number of in1 columns in the last subblock that are actually backed by reader-pushed tiles.
    // When the subblock-width optimization above pads per_core_N_compute beyond per_core_N_in1_sender,
    // the last subblock has out_subblock_w lanes total but only this many lanes correspond to in1
    // tiles the reader pushed into cb_in1; the rest are padded columns that the output writer drops.
    // The compute kernel uses this to narrow the matmul_block call for the last subblock so it never
    // reads cb_in1 tile indices that were not produced for the current block.
    // When no padding occurs (per_core_N_compute == per_core_N_in1_sender) this equals out_subblock_w.
    uint32_t last_subblock_w_valid = out_subblock_w - (per_core_N_compute - per_core_N_in1_sender);

    uint32_t num_blocks = K / in0_block_w;
    bool packer_l1_acc_en = packer_l1_acc && num_blocks > 1;

    tt::DataFormat interm0_data_format = packer_l1_acc_en
                                             ? (fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b)
                                             : (fp32_dest_acc_en ? tt::DataFormat::Float32 : output_data_format);

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    uint32_t bias_single_tile_size = bias_tile.get_tile_size(bias_data_format);
    uint32_t output_single_tile_size = output_tile.get_tile_size(output_data_format);
    uint32_t interm0_single_tile_size = output_tile.get_tile_size(interm0_data_format);

    // in1/bias are DRAM sharded with one tile per page; the allocator pads each page to the DRAM
    // alignment (e.g. bfp8 32x16 tile = 544B padded to 576B on Blackhole's 64B alignment). The
    // reader copies blocks contiguously from DRAM, so the CB must hold tiles at the padded stride.
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t in1_aligned_tile_size = tt::align(in1_single_tile_size, dram_alignment);
    uint32_t bias_aligned_tile_size = tt::align(bias_single_tile_size, dram_alignment);

    uint32_t in0_block_tiles = per_core_M * in0_block_w;
    uint32_t in0_CB_tiles = in0_block_tiles;
    if (B * num_blocks > 1) {
        in0_CB_tiles = in0_CB_tiles * 2;  // double buffer
    }
    uint32_t in0_CB_size = in0_CB_tiles * in0_single_tile_size;
    uint32_t in1_block_tiles = per_core_N_in1_sender * in0_block_w;
    uint32_t in1_CB_tiles = in1_block_tiles;
    if (B * num_blocks > 1) {
        in1_CB_tiles = in1_CB_tiles * 3;  // triple buffer
    }
    uint32_t in1_CB_size = in1_CB_tiles * in1_aligned_tile_size;

    uint32_t out_block_tiles = per_core_M * per_core_N_compute;
    uint32_t out_CB_tiles = out_block_tiles;
    uint32_t out_CB_size = out_CB_tiles * output_single_tile_size;
    uint32_t interm0_CB_size = out_CB_tiles * interm0_single_tile_size;

    uint32_t out_reshard_block_tiles = per_core_M * per_core_N_storage;
    uint32_t out_reshard_CB_tiles = out_reshard_block_tiles;
    uint32_t out_reshard_CB_size = out_reshard_CB_tiles * output_single_tile_size;

    uint32_t in0_shard_width_in_tiles = in0_tensor.shard_spec()->shape[1] / in0_tile.get_tile_shape()[1];
    uint32_t in2_block_tiles = per_core_M * in0_shard_width_in_tiles;
    uint32_t in2_CB_tiles = in2_block_tiles;
    uint32_t in2_CB_size = in2_CB_tiles * in0_single_tile_size;

    uint32_t in3_CB_tiles = per_core_N_compute;
    uint32_t in3_CB_size = in3_CB_tiles * bias_aligned_tile_size;

    // get the max page size based on num tiles
    uint32_t in1_buffer_page_size, in1_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, in1_block_tiles, in1_aligned_tile_size, in1_buffer_page_size, in1_buffer_num_pages);

    uint32_t bias_buffer_page_size, bias_buffer_num_pages;
    get_max_page_size_and_num_pages(
        device, per_core_N_in1_sender, bias_aligned_tile_size, bias_buffer_page_size, bias_buffer_num_pages);

    uint32_t num_worker_cores = num_workers;

    // move conflict coord from mcast receiver to mcast sender
    std::vector<CoreCoord> input_all_storage_cores_vec = corerange_to_cores(input_all_storage_cores);
    std::vector<CoreCoord> all_worker_cores_vec = corerange_to_cores(all_worker_cores);
    std::vector<CoreCoord> storage_worker_common;
    move_common_entries(input_all_storage_cores_vec, all_worker_cores_vec, storage_worker_common);

    std::vector<CoreRange> input_all_storage_cores_range;
    input_all_storage_cores_range.reserve(input_all_storage_cores_vec.size());
    std::transform(
        input_all_storage_cores_vec.begin(),
        input_all_storage_cores_vec.end(),
        std::back_inserter(input_all_storage_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::vector<CoreRange> all_worker_cores_range;
    all_worker_cores_range.reserve(all_worker_cores_vec.size());
    std::transform(
        all_worker_cores_vec.begin(),
        all_worker_cores_vec.end(),
        std::back_inserter(all_worker_cores_range),
        [](const CoreCoord& coord) { return CoreRange(coord); });

    std::set<CoreRange> input_all_storage_cores_set(
        input_all_storage_cores_range.begin(), input_all_storage_cores_range.end());
    std::set<CoreRange> all_worker_cores_set(all_worker_cores_range.begin(), all_worker_cores_range.end());
    CoreRangeSet mcast_senders = CoreRangeSet(input_all_storage_cores_set);
    CoreRangeSet mcast_receivers = CoreRangeSet(all_worker_cores_set);

    // all cores
    std::set<CoreRange> all_cores_set;
    all_cores_set.insert(mcast_senders.ranges().begin(), mcast_senders.ranges().end());
    all_cores_set.insert(mcast_receivers.ranges().begin(), mcast_receivers.ranges().end());
    CoreRangeSet all_cores = CoreRangeSet(all_cores_set);

    // grid bounding box
    CoreRange bounding_box = all_cores.bounding_box();
    std::set<CoreRange> bounding_box_set;
    bounding_box_set.insert(bounding_box);
    CoreRangeSet all_cores_in_rect_grid(bounding_box_set);
    std::vector<CoreCoord> all_cores_in_rect_grid_vec = corerange_to_cores(all_cores_in_rect_grid);

    // TT_MR_IN1_DUAL_NOC: readers that move to NOC1. all_worker_cores_vec now holds only the workers that
    // are not in0 storage cores (move_common_entries above), i.e. the in0 type-3 receivers, whose in0
    // kernel issues no multicast and so may run on NOC0.
    std::set<CoreCoord> in1_noc1_cores;
    if (in1_dual_noc) {
        const std::set<CoreCoord> receiver_cores(all_worker_cores_vec.begin(), all_worker_cores_vec.end());
        for (const auto& assignment : reader_assignments) {
            if ((assignment.worker_index & 1) != 0 && receiver_cores.contains(assignment.worker_core)) {
                in1_noc1_cores.insert(assignment.worker_core);
            }
        }
        static std::once_flag engaged_banner;
        std::call_once(engaged_banner, [&] {
            log_info(
                tt::LogOp,
                "TT_MR_IN1_DUAL_NOC engaged mode={} rev=3 (DRAM-sharded multi-reader in1: odd-indexed readers "
                "run in1 on NOC1 and in0 on NOC0, DM_DEDICATED_NOC; first program workers_per_bank={} readers={} "
                "on_noc1={})",
                in1_dual_noc_mode == MrIn1DualNoc::MUSTFAIL ? "mustfail" : "1",
                workers_per_bank,
                reader_assignments.size(),
                in1_noc1_cores.size());
        });
    }
    std::set<CoreRange> in1_noc1_ranges;
    for (const auto& core : in1_noc1_cores) {
        in1_noc1_ranges.insert(CoreRange(core));
    }
    const CoreRangeSet in1_noc1_grid(in1_noc1_ranges);
    const CoreRangeSet in1_noc0_grid =
        in1_noc1_cores.empty() ? all_cores_in_rect_grid : all_cores_in_rect_grid.subtract(in1_noc1_grid);
    auto on_noc1 = [&](const CoreCoord& core) { return in1_noc1_cores.contains(core); };

    // Semaphore IDs (manually assigned, matching CreateSemaphore sequential allocation)
    uint32_t in0_mcast_sender_semaphore_id = 0;
    uint32_t in0_mcast_receiver_semaphore_id = 1;
    uint32_t in0_mcast_sender_valid_semaphore_id = 2;

    uint32_t in0_num_subblocks = (per_core_M / out_subblock_h);
    uint32_t in0_block_num_tiles = out_subblock_h * in0_block_w * in0_num_subblocks;

    uint32_t num_blocks_per_shard = num_blocks / input_all_storage_cores_vec.size();
    if (per_core_M > 1) {
        TT_FATAL(
            num_blocks_per_shard == 1,
            "currently not support per_core_M larger than 1, while split one shard into multiple blocks (per_core_M "
            "{}, num_blocks_per_shard {})",
            per_core_M,
            num_blocks_per_shard);
    }

    std::vector<uint32_t> in0_sender_compile_time_args = {
        (std::uint32_t)in0_block_num_tiles,                         // in0_block_num_tiles
        (std::uint32_t)in0_block_num_tiles * in0_single_tile_size,  // in0_block_size_bytes
        (std::uint32_t)in0_last_ktile_w,                            // in0_last_ktile_w
        (std::uint32_t)0,                                           // in0_last_ktile_h (transpose not supported)
        // in0 mcast args
        (std::uint32_t)in0_mcast_sender_semaphore_id,
        (std::uint32_t)in0_mcast_receiver_semaphore_id,
        (std::uint32_t)num_worker_cores,  // in0_mcast_num_dests
        (std::uint32_t)num_mcast_cores,   // in0_mcast_num_cores
        // block
        (std::uint32_t)num_blocks,
        // mcast noc coords
        (std::uint32_t)start_core_noc.x,
        (std::uint32_t)start_core_noc.y,
        (std::uint32_t)end_core_noc.x,
        (std::uint32_t)end_core_noc.y,
        // semaphore valid
        (std::uint32_t)in0_mcast_sender_valid_semaphore_id,
        //
        (std::uint32_t)num_blocks_per_shard,
        (std::uint32_t)in0_block_w};

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {
        (std::uint32_t)in1_buffer_page_size,
        (std::uint32_t)in1_buffer_num_pages,
        // in1 block args
        (std::uint32_t)per_core_N_compute,                   // in1_block_w (padded, used only for bias CB)
        (std::uint32_t)per_core_N_in1_sender * in0_block_w,  // in1_block_num_tiles
        // in0/in1 common args
        (std::uint32_t)num_blocks,                                    // num_blocks
        (std::uint32_t)out_block_tiles,                               // out_block_num_tiles
        (std::uint32_t)per_core_N_compute * output_single_tile_size,  // out_tensor_stride_w_bytes
        (std::uint32_t)per_core_N_storage * output_single_tile_size,  // out_reshard_tensor_stride_w_bytes
        (std::uint32_t)per_core_M,
        (std::uint32_t)workers_per_bank,
        (std::uint32_t)in1_shard_width_tiles,
        (std::uint32_t)per_core_N_in1_sender};
    if (bias.has_value()) {
        in1_sender_writer_compile_time_args.push_back(bias_buffer_page_size);
        in1_sender_writer_compile_time_args.push_back(bias_buffer_num_pages);
        in1_sender_writer_compile_time_args.push_back((std::uint32_t)1);
    }
    std::map<std::string, std::string> mm_kernel_defines;
    std::map<std::string, std::string> mm_kernel_in0_sender_define;
    std::map<std::string, std::string> mm_kernel_in1_sender_writer_defines;
    if (bias.has_value()) {
        mm_kernel_defines["FUSE_BIAS"] = "1";
        mm_kernel_in1_sender_writer_defines["FUSE_BIAS"] = "1";
    }
    if (workers_per_bank > 1) {
        mm_kernel_in1_sender_writer_defines["SPLIT_DRAM_BANK"] = "1";
    }
    if (fused_activation.has_value()) {
        if (fused_activation.value().op_type == UnaryOpType::RELU) {
            mm_kernel_defines["PACK_RELU"] = "1";
        } else {
            mm_kernel_defines["SFPU_ACTIVATION"] = "1";
        }
    }
    if (packer_l1_acc_en) {
        mm_kernel_defines["PACKER_L1_ACC"] = "1";
    }
    if (fp32_dest_acc_en) {
        mm_kernel_defines["FP32_DEST_ACC_EN"] = "1";
    }
    mm_kernel_in1_sender_writer_defines["OUT_SHARDED"] = "1";
    mm_kernel_in1_sender_writer_defines["SKIP_MCAST"] = "1";

    if (skip_compute) {
        mm_kernel_defines["SKIP_COMPUTE"] = "1";
    }
    if (skip_in0_mcast) {
        mm_kernel_in0_sender_define["SKIP_MCAST"] = "1";
    }
    if (skip_write_back) {
        mm_kernel_in1_sender_writer_defines["SKIP_WRITE_BACK"] = "1";
    }
    mm_kernel_defines["MATMUL_DRAM_SHARDED"] = "1";
    if (in1_transpose_tile) {
        mm_kernel_defines["IN1_TRANSPOSE_TILE"] = "1";
    }

    const uint32_t num_compute_cores = all_cores_in_rect_grid.num_cores();
    ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
        device->arch(), num_compute_cores, mm_kernel_defines);
    ttnn::operations::compute_throttle_utils::throttle_mm_perf(
        device->arch(), num_compute_cores, mm_kernel_defines, throttle_level);

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)
    auto map_to_defines = [](const std::map<std::string, std::string>& m) -> KernelDescriptor::Defines {
        KernelDescriptor::Defines result;
        result.reserve(m.size());
        for (const auto& [k, v] : m) {
            result.emplace_back(k, v);
        }
        return result;
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Build Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////

    // in0 sender kernel (reader - RISCV_1)
    KernelDescriptor in0_sender_kernel_desc;
    in0_sender_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded.cpp";
    in0_sender_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in0_sender_kernel_desc.core_ranges = in1_noc0_grid;
    in0_sender_kernel_desc.compile_time_args = in0_sender_compile_time_args;
    in0_sender_kernel_desc.defines = map_to_defines(mm_kernel_in0_sender_define);
    in0_sender_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
    };
    in0_sender_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in0_noc, .noc_mode = dm_noc_mode};

    // in1 sender/writer kernel (writer - RISCV_0)
    KernelDescriptor in1_sender_writer_kernel_desc;
    in1_sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded.cpp";
    in1_sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    in1_sender_writer_kernel_desc.core_ranges = in1_noc0_grid;
    in1_sender_writer_kernel_desc.compile_time_args = in1_sender_writer_compile_time_args;
    in1_sender_writer_kernel_desc.defines = map_to_defines(mm_kernel_in1_sender_writer_defines);
    in1_sender_writer_kernel_desc.named_compile_time_args = {
        {"cb_in1", tt::CBIndex::c_1},
        {"cb_bias", tt::CBIndex::c_3},
        {"cb_out", tt::CBIndex::c_4},
        {"cb_out_reshard", tt::CBIndex::c_6},
    };
    in1_sender_writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in1_noc, .noc_mode = dm_noc_mode};

    // TT_MR_IN1_DUAL_NOC: the same two kernels on the NOC1 readers, with the NoCs swapped (same RISCs).
    // Only the in0 receiver path (worker_core_type 3: semaphore wait and a semaphore up to the sender)
    // runs there, and it does not use the NoC-ordered mcast rectangle compile args.
    KernelDescriptor in0_noc0_kernel_desc = in0_sender_kernel_desc;
    in0_noc0_kernel_desc.core_ranges = in1_noc1_grid;
    in0_noc0_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = in1_noc, .noc_mode = dm_noc_mode};
    KernelDescriptor in1_noc1_kernel_desc = in1_sender_writer_kernel_desc;
    in1_noc1_kernel_desc.core_ranges = in1_noc1_grid;
    if (in1_dual_noc_mode == MrIn1DualNoc::MUSTFAIL) {
        in1_noc1_kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/matmul/device/kernels_mr_dual_noc/dataflow/"
            "reader_bmm_tile_layout_in1_sender_dram_sharded_dual_noc.cpp";
        auto mustfail_defines = mm_kernel_in1_sender_writer_defines;
        mustfail_defines["MR_IN1_DUAL_NOC_MUSTFAIL"] = "1";
        in1_noc1_kernel_desc.defines = map_to_defines(mustfail_defines);
    }
    in1_noc1_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = in0_noc, .noc_mode = dm_noc_mode};

    // Compute kernel
    uint32_t in0_subblock_num_tiles = out_subblock_h * in0_block_w;
    uint32_t in1_num_subblocks = (per_core_N_compute / out_subblock_w);
    uint32_t in1_per_core_w = per_core_N_in1_sender;
    uint32_t out_subblock_num_tiles = out_subblock_h * out_subblock_w;

    std::vector<uint32_t> compute_kernel_args = {
        in0_block_w,             // in0_block_w
        in0_num_subblocks,       // in0_num_subblocks
        in0_block_num_tiles,     // in0_block_num_tiles
        in0_subblock_num_tiles,  // in0_subblock_num_tiles

        in1_num_subblocks,  // in1_num_subblocks
        in1_block_tiles,    // in1_block_num_tiles
        in1_per_core_w,     // in1_per_core_w

        num_blocks,  // num_blocks
        1,           // out_num_blocks_x
        1,           // out_num_blocks_y

        out_subblock_h,          // out_subblock_h
        out_subblock_w,          // out_subblock_w
        out_subblock_num_tiles,  // out_subblock_num_tiles
        B,                       // batch
        out_block_tiles,         // out_block_num_tiles

        untilize_out,  // untilize_out
        false,         // get_batch_from_reader
        false,         // in0_transpose_tile
    };
    if (bias.has_value()) {
        compute_kernel_args.push_back(row_broadcast_bias ? 1u : 0u);
    }

    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores_in_rect_grid;
    compute_kernel_desc.compile_time_args = compute_kernel_args;
    compute_kernel_desc.defines = map_to_defines(mm_kernel_defines);
    {
        KernelDescriptor::NamedCompileTimeArgs named_compile_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in1", tt::CBIndex::c_1},
            {"cb_bias", tt::CBIndex::c_3},
            {"cb_out", tt::CBIndex::c_4},
            {"cb_intermed0", tt::CBIndex::c_5},
            {"cb_in0_intermediate", tt::CBIndex::c_8},
            {"cb_in1_intermediate", tt::CBIndex::c_9},
            {"cb_in0_transposed", tt::CBIndex::c_10},
            {"bias_ntiles", per_core_N_compute},
            {"last_subblock_w_valid", last_subblock_w_valid},
        };
        if (fused_activation.has_value() && fused_activation.value().op_type != UnaryOpType::RELU) {
            using ttnn::operations::matmul::utilities::get_activation_params;
            const auto params = get_activation_params(fused_activation.value());
            named_compile_args.push_back({"activation_type", static_cast<uint32_t>(params.type)});
            named_compile_args.push_back({"activation_param0", params.param0});
            named_compile_args.push_back({"activation_param1", params.param1});
            named_compile_args.push_back({"activation_param2", params.param2});
        }
        compute_kernel_desc.named_compile_time_args = std::move(named_compile_args);
    }
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .math_approx_mode = math_approx_mode,
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      Build CBDescriptors
    ////////////////////////////////////////////////////////////////////////////
    ProgramDescriptor desc;

    tt::tt_metal::TileDescriptor in0_tile_desc{in0_tile};
    tt::tt_metal::TileDescriptor in1_tile_desc{in1_tile};
    tt::tt_metal::TileDescriptor bias_tile_desc{bias_tile};
    tt::tt_metal::TileDescriptor output_tile_desc{output_tile};

    // CB 0: in0
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in0_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_0,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 1: in1
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in1_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_1,
            .data_format = in1_data_format,
            .page_size = in1_aligned_tile_size,
            .tile = in1_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 2: in0 sharded (backed by in0_buffer)
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = in2_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_2,
            .data_format = in0_data_format,
            .page_size = in0_single_tile_size,
            .tile = in0_tile_desc});
        cb_desc.tensor = &in0_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 4 and CB 5: output and intermediate
    if ((interm0_data_format != output_data_format) || (untilize_out && (in1_num_subblocks > 1))) {
        // Separate output and intermediate CBs
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = out_CB_size;
            cb_desc.core_ranges = all_cores_in_rect_grid;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_4,
                .data_format = output_data_format,
                .page_size = output_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
        {
            CBDescriptor cb_desc;
            cb_desc.total_size = interm0_CB_size;
            cb_desc.core_ranges = all_cores_in_rect_grid;
            cb_desc.format_descriptors.push_back(CBFormatDescriptor{
                .buffer_index = tt::CBIndex::c_5,
                .data_format = interm0_data_format,
                .page_size = interm0_single_tile_size,
                .tile = output_tile_desc});
            desc.cbs.push_back(std::move(cb_desc));
        }
    } else {
        // share buffer
        CBDescriptor cb_desc;
        cb_desc.total_size = out_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_4,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_5,
            .data_format = interm0_data_format,
            .page_size = interm0_single_tile_size,
            .tile = output_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 6: resharded output (backed by out_buffer)
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = out_reshard_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_6,
            .data_format = output_data_format,
            .page_size = output_single_tile_size,
            .tile = output_tile_desc});
        cb_desc.tensor = &out_tensor;
        desc.cbs.push_back(std::move(cb_desc));
    }

    // CB 3: bias
    if (bias.has_value()) {
        CBDescriptor cb_desc;
        cb_desc.total_size = in3_CB_size;
        cb_desc.core_ranges = all_cores_in_rect_grid;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = tt::CBIndex::c_3,
            .data_format = bias_data_format,
            .page_size = bias_aligned_tile_size,
            .tile = bias_tile_desc});
        desc.cbs.push_back(std::move(cb_desc));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Semaphore Descriptors
    ////////////////////////////////////////////////////////////////////////////
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_sender_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_receiver_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = INVALID});
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = in0_mcast_sender_valid_semaphore_id, .core_ranges = all_cores_in_rect_grid, .initial_value = VALID});

    ////////////////////////////////////////////////////////////////////////////
    //                      Runtime Args (per-core loop)
    ////////////////////////////////////////////////////////////////////////////

    std::vector<uint32_t> in0_mcast_sender_noc_x;
    std::vector<uint32_t> in0_mcast_sender_noc_y;
    std::vector<CoreCoord> mcast_senders_coords = corerange_to_cores(mcast_senders);
    std::sort(mcast_senders_coords.begin(), mcast_senders_coords.end(), [](const CoreCoord& a, const CoreCoord& b) {
        if (a.y != b.y) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });
    in0_mcast_sender_noc_x.reserve(mcast_senders_coords.size());
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    in0_mcast_sender_noc_y.reserve(mcast_senders_coords.size());
    for (auto core : mcast_senders_coords) {
        in0_mcast_sender_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    // in0 sender runtime args (mcast senders)
    uint32_t sender_id = 0;
    for (auto core : mcast_senders_coords) {
        std::vector<uint32_t> mm_in0_sender_args;
        mm_in0_sender_args.reserve(3 + in0_mcast_sender_noc_x.size() + in0_mcast_sender_noc_y.size());

        uint32_t worker_core_type;
        if (find(storage_worker_common.begin(), storage_worker_common.end(), core) != storage_worker_common.end()) {
            worker_core_type = 2;
        } else {
            worker_core_type = 1;
        }

        mm_in0_sender_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_sender_args.push_back((std::uint32_t)sender_id);
        mm_in0_sender_args.push_back(
            (std::uint32_t)((core == input_all_storage_cores_vec.back()) and (in0_last_ktile_w > 0)));
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_sender_args.insert(
            mm_in0_sender_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        in0_sender_kernel_desc.runtime_args.emplace_back(core, std::move(mm_in0_sender_args));
        sender_id++;
    }

    // in0 sender runtime args (mcast receivers)
    std::vector<CoreCoord> mcast_receiver_coords = corerange_to_cores(mcast_receivers);
    for (auto core : mcast_receiver_coords) {
        std::vector<uint32_t> mm_in0_receiver_args;
        mm_in0_receiver_args.reserve(3 + in0_mcast_sender_noc_x.size() + in0_mcast_sender_noc_y.size());
        uint32_t worker_core_type = 3;
        mm_in0_receiver_args.push_back((std::uint32_t)worker_core_type);
        mm_in0_receiver_args.push_back((std::uint32_t)0);
        mm_in0_receiver_args.push_back((std::uint32_t)0);  // in0_last_ktile_w
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_x.begin(), in0_mcast_sender_noc_x.end());
        mm_in0_receiver_args.insert(
            mm_in0_receiver_args.end(), in0_mcast_sender_noc_y.begin(), in0_mcast_sender_noc_y.end());

        (on_noc1(core) ? in0_noc0_kernel_desc : in0_sender_kernel_desc)
            .runtime_args.emplace_back(core, std::move(mm_in0_receiver_args));
    }

    // in0 sender runtime args (idle cores in rect grid)
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(mcast_senders_coords.begin(), mcast_senders_coords.end(), core) == mcast_senders_coords.end() and
            std::find(mcast_receiver_coords.begin(), mcast_receiver_coords.end(), core) ==
                mcast_receiver_coords.end()) {
            std::vector<uint32_t> mm_in0_idle_args;
            uint32_t worker_core_type = 0;
            mm_in0_idle_args.push_back((std::uint32_t)worker_core_type);

            in0_sender_kernel_desc.runtime_args.emplace_back(core, std::move(mm_in0_idle_args));
        }
    }

    // Compute and in1 sender/writer runtime args
    std::vector<uint32_t> bank_ids;
    bank_ids.reserve(all_worker_cores_ordered.size());
    uint32_t curr_storage_core_idx = 0;
    uint32_t per_core_N_storage_curr_stride = 0;

    uint32_t worker_core_stride = 0;
    uint32_t storage_core_stride = 0;
    uint32_t curr_worker_core = 0;
    uint32_t curr_storage_core = 0;

    std::vector<uint32_t> output_noc_x;
    std::vector<uint32_t> output_noc_y;
    std::vector<CoreCoord> output_coords = corerange_to_cores(output_all_storage_cores, std::nullopt, true);
    output_noc_x.reserve(output_coords.size());
    for (auto core : output_coords) {
        output_noc_x.push_back((std::uint32_t)device->worker_core_from_logical_core(core).x);
    }
    output_noc_y.reserve(output_coords.size());
    for (auto core : output_coords) {
        output_noc_y.push_back((std::uint32_t)device->worker_core_from_logical_core(core).y);
    }

    uint32_t num_cores_written_back = (N + per_core_N_storage - 1) / per_core_N_storage;
    uint32_t expected_max_total_width = num_cores_written_back * per_core_N_storage;
    uint32_t total_tensor_width_written_back = 0;

    // For all cores in the rect grid, set compute and in1 rt args for non-worker cores
    for (auto core : all_cores_in_rect_grid_vec) {
        if (std::find(all_worker_cores.ranges().begin(), all_worker_cores.ranges().end(), core) ==
            all_worker_cores.ranges().end()) {  // not worker
            bool is_worker_core = false;
            std::vector<uint32_t> mm_in1_sender_writer_args;
            mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
            in1_sender_writer_kernel_desc.runtime_args.emplace_back(core, std::move(mm_in1_sender_writer_args));

            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);
            compute_kernel_desc.runtime_args.emplace_back(core, std::move(mm_compute_args));
        } else {
            bool is_worker_core = true;
            std::vector<uint32_t> mm_compute_args;
            mm_compute_args.push_back((std::uint32_t)is_worker_core);
            compute_kernel_desc.runtime_args.emplace_back(core, std::move(mm_compute_args));
        }
    }

    // Worker cores: in1 sender/writer runtime args
    constexpr std::size_t num_shards_to_write_back_arg_index = 6;
    constexpr std::size_t fixed_writer_arg_count = 11;
    for (uint32_t i = 0; i < all_worker_cores_ordered.size(); ++i) {
        auto core = all_worker_cores_ordered[i];
        const auto& reader_assignment = reader_assignments[i];
        const uint32_t bank_id = reader_assignment.bank_id;

        bool is_worker_core = true;
        std::vector<uint32_t> mm_in1_sender_writer_args;
        mm_in1_sender_writer_args.push_back((std::uint32_t)is_worker_core);
        mm_in1_sender_writer_args.push_back(in1_tensor.address());                     // [1] smuggled-rta-ok: rebound
        mm_in1_sender_writer_args.push_back(bias.has_value() ? bias->address() : 0u);  // [2] smuggled-rta-ok

        uint32_t vc = bank_id & 0x3;
        bank_ids.push_back(bank_id);
        for (uint32_t j = 0; j < i; ++j) {
            auto core_prev = all_worker_cores_ordered[j];

            if (core_prev.y == core.y and ((bank_id & 0x3) == (bank_ids[j] & 0x3))) {
                vc = (vc + 1) & 0x3;
                break;
            }
        }
        mm_in1_sender_writer_args.push_back((std::uint32_t)bank_id);
        mm_in1_sender_writer_args.push_back((std::uint32_t)vc);
        mm_in1_sender_writer_args.push_back(reader_assignment.worker_index);

        if (per_core_N_in1_sender < per_core_N_storage) {
            TT_FATAL(curr_storage_core_idx < num_cores_written_back, "Worker {} has no storage area assigned", core);

            uint32_t remaining_per_core_N_storage = (per_core_N_storage - per_core_N_storage_curr_stride);
            uint32_t per_core_N_reshard_1 = (remaining_per_core_N_storage > per_core_N_in1_sender)
                                                ? per_core_N_in1_sender
                                                : remaining_per_core_N_storage;
            uint32_t per_core_N_reshard_2 = per_core_N_in1_sender - per_core_N_reshard_1;

            if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                mm_in1_sender_writer_args.push_back(2);
            } else {
                mm_in1_sender_writer_args.push_back(1);
            }

            mm_in1_sender_writer_args.push_back(
                per_core_N_storage_curr_stride * output_single_tile_size);  // reshard_tensor_start_offset
            mm_in1_sender_writer_args.push_back(
                per_core_N_reshard_1 * output_single_tile_size);                       // per_core_N_reshard_bytes_1
            mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core_idx]);  // output_noc_x
            mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core_idx]);  // output_noc_y

            total_tensor_width_written_back += per_core_N_reshard_1;

            if (per_core_N_reshard_2 != 0 and (curr_storage_core_idx + 1) < num_cores_written_back) {
                mm_in1_sender_writer_args.push_back(
                    per_core_N_reshard_2 * output_single_tile_size);  // per_core_N_reshard_bytes_2
                mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core_idx + 1]);  // output_noc_x
                mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core_idx + 1]);  // output_noc_y

                total_tensor_width_written_back += per_core_N_reshard_2;
            }

            curr_storage_core_idx += (per_core_N_storage_curr_stride + per_core_N_in1_sender) / per_core_N_storage;
            per_core_N_storage_curr_stride =
                (per_core_N_storage_curr_stride + per_core_N_in1_sender) % per_core_N_storage;
        } else {
            uint32_t num_cores_write_back = 0;

            if (curr_storage_core < num_cores_written_back) {
                num_cores_write_back++;

                worker_core_stride = per_core_N_storage - storage_core_stride;

                mm_in1_sender_writer_args.push_back(
                    storage_core_stride * output_single_tile_size);  // reshard_tensor_start_offset
                mm_in1_sender_writer_args.push_back(
                    worker_core_stride * output_single_tile_size);                     // per_core_N_reshard
                mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core]);  // output_noc_x
                mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core]);  // output_noc_y

                curr_storage_core += (storage_core_stride + worker_core_stride) / per_core_N_storage;
                storage_core_stride = (storage_core_stride + worker_core_stride) % per_core_N_storage;

                if (worker_core_stride >= per_core_N_in1_sender) {
                    curr_worker_core += 1;
                }

                total_tensor_width_written_back += worker_core_stride;

                while (curr_worker_core <= i and curr_storage_core < num_cores_written_back) {
                    num_cores_write_back++;

                    bool increment_worker_core = (worker_core_stride + per_core_N_storage) >= per_core_N_in1_sender;
                    uint32_t current_worker_stride_total =
                        increment_worker_core ? per_core_N_in1_sender : worker_core_stride + per_core_N_storage;
                    uint32_t current_worker_write_back_tiles = current_worker_stride_total - worker_core_stride;

                    if (increment_worker_core) {
                        curr_worker_core += 1;
                    }

                    mm_in1_sender_writer_args.push_back(
                        current_worker_write_back_tiles * output_single_tile_size);        // per_core_N_reshard
                    mm_in1_sender_writer_args.push_back(output_noc_x[curr_storage_core]);  // output_noc_x
                    mm_in1_sender_writer_args.push_back(output_noc_y[curr_storage_core]);  // output_noc_y

                    total_tensor_width_written_back += current_worker_write_back_tiles;

                    storage_core_stride = current_worker_write_back_tiles % per_core_N_storage;
                    curr_storage_core += current_worker_write_back_tiles / per_core_N_storage;
                    worker_core_stride = current_worker_stride_total;
                }
            }

            mm_in1_sender_writer_args.insert(
                mm_in1_sender_writer_args.begin() + num_shards_to_write_back_arg_index, num_cores_write_back);
        }

        // A worker can legitimately have no output storage shard to reshard, for example when the reader count exceeds
        // the output shard count. The kernel still materializes its fixed writer-argument views before the zero-count
        // write-back loop, so provide neutral placeholders for those slots.
        if (mm_in1_sender_writer_args.size() < fixed_writer_arg_count) {
            mm_in1_sender_writer_args.resize(fixed_writer_arg_count, 0);
        }

        // Build variant args: positions [1] and [2] are buffer addresses
        std::vector<std::variant<uint32_t, std::reference_wrapper<const tt::tt_metal::MeshTensor>>> in1_writer_args(
            mm_in1_sender_writer_args.begin(), mm_in1_sender_writer_args.end());
        in1_writer_args[1] = in1_tensor;
        if (bias.has_value()) {
            in1_writer_args[2] = *bias;
        }
        (on_noc1(core) ? in1_noc1_kernel_desc : in1_sender_writer_kernel_desc)
            .emplace_runtime_args(core, in1_writer_args);
    }

    TT_FATAL(
        total_tensor_width_written_back <= expected_max_total_width,
        "more datums written back to sharded tensor, L1 corruption, expected: {}, actual: {}",
        expected_max_total_width,
        total_tensor_width_written_back);

    ////////////////////////////////////////////////////////////////////////////
    //                      Push Kernel Descriptors
    ////////////////////////////////////////////////////////////////////////////
    desc.kernels.push_back(std::move(in0_sender_kernel_desc));
    desc.kernels.push_back(std::move(in1_sender_writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    if (!in1_noc1_cores.empty()) {
        desc.kernels.push_back(std::move(in0_noc0_kernel_desc));
        desc.kernels.push_back(std::move(in1_noc1_kernel_desc));
    }

    return desc;
}

// ---------------------------------------------------------------------------------------------------------
// tt-stack 0040: lift of tenstorrent/tt-metal #58922 (head 5e6d4950, "matmul: multi-core DRAM-sharded decode
// variant (cores_per_bank)") onto our pin. Selected only by MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig::
// cores_per_bank > 0 (its own program factory struct, MatmulDRAMShardedMulticoreProgramFactory); with 0 every
// program above is byte-identical. Differences from upstream: our Metal 2.0 host API spells the hardware configs as
// DataMovementGen1Config / ComputeGen1Config variants; only CK == 1 (cores_per_bank <= the per-bank shard width) is
// carried, so the in-group reduce-scatter (which needs api/semaphore.h, absent on our pin) is TT_FATAL'd instead;
// the bank anchors come from the same IDevice query the multi-reader path already uses on the 1x2 mesh.
//
// The single-reader program above runs one compute core per DRAM bank, issues its weight reads on one NoC, and
// accumulates in0_block_w-wide blocks through L1. At M = one tile a core cannot retire tiles as fast as its bank
// delivers them. This variant:
//   - places cores_per_bank cores next to each bank's NOC_0-optimal worker and splits the bank's columns over them
//     (Nbt * c / C .. Nbt * (c + 1) / C, so C need not divide the shard width);
//   - streams the weight on both data-movement RISCs (alternate K blocks, one per NoC, each with its own
//     transaction-id ring);
//   - reads the activation rows a core needs straight from their L1 shards with lookahead;
//   - keeps each column pass's whole K in fp32 Dest (full sync, <= 8 tiles, P passes);
//   - writes the output tiles straight into the caller's width-sharded output.
//
// TT_MM_DSMC_MUSTFAIL (read once per process): unset / "" / "0" off; "1" compiles DSMC_MUSTFAIL into the compute
// kernel, which skips the matmul of the last streamed K block (a specific wrong result: the output misses one K
// slice) while keeping every wait/pop, so the program still completes. Anything else is an error.
static bool dsmc_mustfail() {
    static const bool mf = [] {
        const char* raw = tt_stack_public::getenv("TT_MM_DSMC_MUSTFAIL");
        const std::string_view v = raw == nullptr ? std::string_view{} : std::string_view{raw};
        if (v.empty() || v == "0") {
            return false;
        }
        if (v == "1") {
            return true;
        }
        TT_THROW("TT_MM_DSMC_MUSTFAIL must be unset, 0 or 1; got '{}'", v);
    }();
    return mf;
}

static ttnn::device_operation::ProgramArtifacts create_program_dram_sharded_multicore_spec(
    tt::tt_metal::IDevice* device,
    const tt::tt_metal::MeshTensor& in0_tensor,
    const tt::tt_metal::MeshTensor& in1_tensor,
    const tt::tt_metal::MeshTensor& out_tensor,
    tt::tt_metal::experimental::ComputeHardwareConfig compute_hw,
    ttnn::operations::compute_throttle_utils::ThrottleLevel throttle_level,
    uint32_t Kt,
    uint32_t Nt,
    uint32_t k_block,
    uint32_t cores_per_bank,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const tt::tt_metal::Tile& output_tile,
    tt::DataFormat in0_data_format,
    tt::DataFormat in1_data_format,
    tt::DataFormat output_data_format) {
    using namespace tt::tt_metal::experimental;
    using tt::tt_metal::CoreCoord;
    using tt::tt_metal::CoreRange;
    using tt::tt_metal::CoreRangeSet;
    using tt::tt_metal::ShardOrientation;

    const auto arch = device->arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "cores_per_bank (tt-stack 0040) is lifted for Blackhole only");
    // Blackhole: both data-movement RISCs read the weight.
    const bool two_streams = true;
    constexpr uint32_t dst_tiles = 8;  // fp32 Dest, full sync
    const uint32_t depth = 3;          // weight blocks in flight per stream
    constexpr uint32_t x_lookahead = 6;  // activation blocks in flight ahead of the weight stream
    const bool mustfail = dsmc_mustfail();

    const auto in1_shard_spec = in1_tensor.shard_spec().value();
    TT_FATAL(in1_shard_spec.orientation == ShardOrientation::ROW_MAJOR, "Only ROW_MAJOR sharding is supported");
    const uint32_t Nbt = in1_shard_spec.shape[1] / in1_tile.get_tile_shape()[1];
    TT_FATAL(
        in1_shard_spec.shape[0] == Kt * in1_tile.get_tile_shape()[0],
        "cores_per_bank: the weight shard must hold the whole K ({} rows), got {}",
        Kt * in1_tile.get_tile_shape()[0],
        in1_shard_spec.shape[0]);
    const auto anchors = device->get_optimal_dram_bank_to_logical_worker_assignment(tt::tt_metal::NOC::NOC_0);
    const uint32_t num_dram_banks = anchors.size();
    TT_FATAL(
        in1_shard_spec.grid.num_cores() == num_dram_banks,
        "cores_per_bank: the weight must be width-sharded over all {} DRAM banks, got {} shards",
        num_dram_banks,
        in1_shard_spec.grid.num_cores());
    const uint32_t num_banks = tt::div_up(Nt, Nbt);  // banks holding real columns
    TT_FATAL(
        num_banks <= num_dram_banks, "cores_per_bank: N ({} tiles) exceeds {} DRAM shards of {} tiles", Nt,
        num_dram_banks, Nbt);

    // Layout: CN column groups per bank, P column passes per core. CK (K row groups) is fixed at 1 in this lift.
    TT_FATAL(
        cores_per_bank <= Nbt,
        "cores_per_bank {} exceeds the per-bank shard width {} tiles; K row groups (#58922 CK > 1) are not lifted",
        cores_per_bank,
        Nbt);
    const uint32_t CN = cores_per_bank;
    const uint32_t CK = 1;
    const uint32_t ncmax = tt::div_up(Nbt, CN);
    const uint32_t P = tt::div_up(ncmax, dst_tiles);
    const uint32_t ncp = tt::div_up(ncmax, P);
    // Every core needs work in every pass, or the pass waits on a weight block nobody streams.
    TT_FATAL(
        Nbt / CN >= P,
        "cores_per_bank {}: {} columns per bank leave a core with fewer columns than its {} passes",
        cores_per_bank,
        Nbt,
        P);
    TT_FATAL(k_block >= 1, "cores_per_bank: in0_block_w must be at least 1");
    const uint32_t KB = std::min(k_block, Kt);
    const uint32_t max_blocks = tt::div_up(Kt, KB);

    // Cores and work items: wi = (bank * CN + cn) * CK + kidx; role = wi + 1 (0 = idle).
    const auto cores =
        dram_sharded_helpers::get_dram_bank_adjacent_workers(device, tt::tt_metal::NOC::NOC_0, CN * CK, num_banks);
    const auto grid = device->compute_with_storage_grid_size();
    std::vector<uint32_t> role(grid.x * grid.y, 0);
    std::vector<CoreCoord> work_cores;
    uint32_t max_own = 0;
    for (uint32_t bank = 0; bank < num_banks; ++bank) {
        for (uint32_t cn = 0; cn < CN; ++cn) {
            const uint32_t n0 = Nbt * cn / CN, n1 = Nbt * (cn + 1) / CN;
            for (uint32_t kidx = 0; kidx < CK; ++kidx) {
                const CoreCoord core = cores[bank][(cn * CK) + kidx];
                role[(core.y * grid.x) + core.x] = work_cores.size() + 1;
                work_cores.push_back(core);
                max_own = std::max(max_own, ((n1 - n0) * (kidx + 1) / CK) - ((n1 - n0) * kidx / CK));
            }
        }
    }
    TT_FATAL(work_cores.size() < 256, "cores_per_bank: {} work items exceed the one-byte role table", work_cores.size());
    std::set<CoreRange> work_ranges;
    for (const auto& core : work_cores) {
        work_ranges.insert(CoreRange(core));
    }
    const CoreRangeSet rect(CoreRangeSet(work_ranges).bounding_box());

    auto virt = [&](const CoreCoord& c) {
        const auto v = device->worker_core_from_logical_core(c);
        return static_cast<uint32_t>((v.y << 16) | v.x);
    };
    auto join = [](const std::vector<uint32_t>& v) {
        std::string s;
        for (std::size_t i = 0; i < v.size(); ++i) {
            s += (i ? "," : "") + std::to_string(v[i]);
        }
        return s;
    };
    const auto in0_shard_spec = in0_tensor.shard_spec().value();
    const auto out_shard_spec = out_tensor.shard_spec().value();
    TT_FATAL(
        in0_shard_spec.orientation == ShardOrientation::ROW_MAJOR &&
            out_shard_spec.orientation == ShardOrientation::ROW_MAJOR,
        "cores_per_bank: activation and output shards must be ROW_MAJOR");
    std::vector<uint32_t> x_cores, out_cores;
    for (const auto& c : corerange_to_cores(in0_shard_spec.grid, std::nullopt, true)) {
        x_cores.push_back(virt(c));
    }
    for (const auto& c : corerange_to_cores(out_shard_spec.grid, std::nullopt, true)) {
        out_cores.push_back(virt(c));
    }
    const uint32_t x_shard_w = in0_shard_spec.shape[1] / in0_tile.get_tile_shape()[1];
    const uint32_t out_shard_w = out_shard_spec.shape[1] / output_tile.get_tile_shape()[1];
    TT_FATAL(
        x_shard_w * x_cores.size() >= Kt,
        "cores_per_bank: the activation shards ({} x {} tiles) do not cover K ({} tiles)",
        x_cores.size(),
        x_shard_w,
        Kt);

    const uint32_t x_tile_size = in0_tile.get_tile_size(in0_data_format);
    const uint32_t w_tile_size =
        tt::align(in1_tile.get_tile_size(in1_data_format), tt::tt_metal::hal::get_dram_alignment());
    const uint32_t out_tile_size = output_tile.get_tile_size(output_data_format);

    {
        // Banner, once per distinct program shape (a handful per model: one per MR role).
        static std::mutex mu;
        static std::set<std::tuple<uint32_t, uint32_t, uint32_t, uint32_t>> seen;
        std::lock_guard<std::mutex> lock(mu);
        if (seen.insert({Kt, Nt, cores_per_bank, KB}).second) {
            log_info(
                tt::LogOp,
                "TT_MM_DSMC engaged cores_per_bank={} mode={} (tt-stack 0040 / tt-metal #58922 multi-core DRAM-sharded "
                "decode matmul; Kt={} Nt={} Nbt={} banks={} CN={} P={} KB={} depth={} work_cores={} rect={} "
                "x_shards={}x{} out_shards={}x{} two_streams={})",
                cores_per_bank,
                mustfail ? "mustfail" : "1",
                Kt,
                Nt,
                Nbt,
                num_banks,
                CN,
                P,
                KB,
                depth,
                work_cores.size(),
                rect.str(),
                x_cores.size(),
                x_shard_w,
                out_cores.size(),
                out_shard_w,
                two_streams);
        }
    }

    const KernelSpecName STREAM0{"stream0"};
    const KernelSpecName STREAM1{"stream1"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName X_DFB{"x"};
    const DFBSpecName W0_DFB{"w0"};
    const DFBSpecName W1_DFB{"w1"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName IN0{"in0"};
    const TensorParamName IN1{"in1"};
    const TensorParamName OUTPUT{"output"};

    Group<DataflowBufferSpec> dataflow_buffers;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = X_DFB,
        .entry_size = x_tile_size,
        .num_entries = max_blocks * KB,
        .data_format_metadata = in0_data_format,
        .tile_format_metadata = in0_tile,
    });
    for (const auto& name : std::vector<DFBSpecName>{W0_DFB, W1_DFB}) {
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = name,
            .entry_size = w_tile_size,
            .num_entries = depth * KB * ncp,
            .data_format_metadata = in1_data_format,
            .tile_format_metadata = in1_tile,
        });
    }
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = out_tile_size,
        .num_entries = max_own,
        .data_format_metadata = output_data_format,
        .tile_format_metadata = output_tile,
    });

    // Shared compile-time layout; every kernel derives its work item from these.
    const std::vector<std::pair<std::string, uint32_t>> layout_args = {
        {"Nbt", Nbt},
        {"KT", Kt},
        {"KB", KB},
        {"CN", CN},
        {"CK", CK},
        {"P", P},
        {"NCP", ncp},
        {"NGRP", num_banks * CN},
        {"ROT", 1u},
        {"GX", static_cast<uint32_t>(grid.x)},
        {"MAXOWN", max_own},
    };
    const std::string role_table = join(role);

    auto make_stream = [&](const KernelSpecName& name, uint32_t par) {
        const bool stream1 = par == 1;
        KernelSpec k{
            .unique_id = name,
            .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_dram_sharded_multicore.cpp",
            .hw_config = DataMovementHardwareConfig{DataMovementGen1Config{
                .processor = stream1 ? tt::tt_metal::DataMovementProcessor::RISCV_0
                                     : tt::tt_metal::DataMovementProcessor::RISCV_1,
                .noc = stream1 ? tt::tt_metal::NOC::NOC_1 : tt::tt_metal::NOC::NOC_0,
            }},
        };
        auto& defines = k.compiler_options.defines;
        defines["DSMC_ROLE"] = role_table;
        for (const auto& [arg, value] : layout_args) {
            k.compile_time_args.insert({arg, value});
        }
        k.compile_time_args.insert({"D", depth});
        k.compile_time_args.insert({"WT", w_tile_size});
        k.compile_time_args.insert({"PAR", par});
        k.compile_time_args.insert({"WSINGLE", 0u});
        defines["DSMC_STREAMS_W"] = "1";
        k.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = stream1 ? W1_DFB : W0_DFB,
            .accessor_name = "w",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = IN1, .accessor_name = "in1"});
        if (stream1) {
            defines["DSMC_STREAM1"] = "1";
            defines["DSMC_XC"] = join(x_cores);
            defines["DSMC_OC"] = join(out_cores);
            k.compile_time_args.insert({"XT", x_tile_size});
            k.compile_time_args.insert({"SW", x_shard_w});
            k.compile_time_args.insert({"XL", x_lookahead});
            k.compile_time_args.insert({"OT", out_tile_size});
            k.compile_time_args.insert({"OW", out_shard_w});
            k.compile_time_args.insert({"OCAP", static_cast<uint32_t>(out_cores.size()) * out_shard_w});
            k.dfb_bindings.push_back(
                DFBBinding{.dfb_spec_name = X_DFB, .accessor_name = "x", .endpoint_type = DFBEndpointType::PRODUCER});
            k.dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = IN0, .accessor_name = "in0"});
            k.tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"});
        }
        return k;
    };

    // The design keeps the whole K in fp32 Dest at full-sync capacity, whatever the caller's compute kernel config
    // says about Dest; fidelity and approximation modes are the caller's.
    enable_32_bit_dest(compute_hw) = true;
    double_buffer_dest(compute_hw) = false;
    auto& um = unpack_modes(compute_hw);
    um = ComputeUnpackModes{};
    um.emplace(X_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    um.emplace(W0_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    um.emplace(W1_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_dram_sharded_multicore.cpp",
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .hw_config = std::move(compute_hw),
    };
    compute.compiler_options.defines["DSMC_ROLE"] = role_table;
    {
        // The same opt-in di/dt mitigations as the single-reader path (TT_MM_STAGGER_TYPE, TT_MM_THROTTLE_PERF).
        std::map<std::string, std::string> throttle_defines;
        ttnn::operations::compute_throttle_utils::add_stagger_defines_if_needed(
            arch, static_cast<int>(work_cores.size()), throttle_defines);
        ttnn::operations::compute_throttle_utils::throttle_mm_perf(
            arch, static_cast<int>(work_cores.size()), throttle_defines, throttle_level);
        for (const auto& [name, value] : throttle_defines) {
            compute.compiler_options.defines[name] = value;
        }
    }
    compute.compiler_options.defines["DSMC_TWO_STREAMS"] = "1";
    if (mustfail) {
        compute.compiler_options.defines["DSMC_MUSTFAIL"] = "1";
    }
    for (const auto& [arg, value] : layout_args) {
        compute.compile_time_args.insert({arg, value});
    }
    compute.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = X_DFB, .accessor_name = "x", .endpoint_type = DFBEndpointType::CONSUMER});
    compute.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = W0_DFB, .accessor_name = "w0", .endpoint_type = DFBEndpointType::CONSUMER});
    compute.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = W1_DFB, .accessor_name = "w1", .endpoint_type = DFBEndpointType::CONSUMER});
    compute.dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER});

    Group<KernelSpec> kernels;
    kernels.push_back(make_stream(STREAM0, 0));
    kernels.push_back(make_stream(STREAM1, 1));
    kernels.push_back(std::move(compute));

    Group<TensorParameter> tensor_parameters;
    tensor_parameters.push_back(TensorParameter{.unique_id = IN0, .spec = in0_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = IN1, .spec = in1_tensor.tensor_spec()});
    tensor_parameters.push_back(TensorParameter{.unique_id = OUTPUT, .spec = out_tensor.tensor_spec()});

    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{
        .name = "bank_cores_rect",
        .kernels = {STREAM0, STREAM1, COMPUTE},
        .target_nodes = rect,
    });
    ProgramSpec spec{
        .name = "matmul_dram_sharded_multicore",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    // No kernel has runtime args (every range derives from the work index), so no KernelRunArgs entries.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {IN0, in0_tensor},
        {IN1, in1_tensor},
        {OUTPUT, out_tensor},
    };
    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace reuse_dram_sharded_optimized_helpers

ProgramDescriptor MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory::create_descriptor(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const std::optional<CoreRangeSet>& /*core_range_set*/) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& optional_input_tensors = tensor_args.optional_input_tensors;
    const auto& output_tensors = tensor_return_value;

    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1).mesh_tensor();
    const auto& bias = optional_input_tensors.at(0);
    const auto& output = output_tensors.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});

    // CB dataformats
    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    ttsl::optional_reference<const tt::tt_metal::MeshTensor> bias_mesh;
    tt::DataFormat bias_data_format = tt::DataFormat::Bfp8_b;
    if (bias.has_value()) {
        const auto& c = bias.value();
        TT_FATAL(
            c.storage_type() == StorageType::DEVICE,
            "Bias tensor must be on device, got storage type: {}",
            c.storage_type());
        TT_FATAL(a.device() == c.device(), "Operands to matmul need to be on the same device!");

        bias_mesh = c.mesh_tensor();
        bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(c.dtype());
    }

    const bool row_broadcast_bias = operations::matmul::utilities::fused_matmul_bias_row_broadcastable(bias);

    tt::tt_metal::IDevice* device = a.device();

    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    CoreRangeSet input_all_cores_storage = a.shard_spec().value().grid;
    CoreRangeSet output_all_cores_storage = output.shard_spec().value().grid;

    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    const auto& a_mesh = a.mesh_tensor();
    TT_FATAL(
        a_mesh.mesh_buffer().device_local_size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        a_mesh.mesh_buffer().device_local_size(),
        in0_single_tile_size);
    TT_FATAL(
        b.mesh_buffer().device_local_size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        b.mesh_buffer().device_local_size(),
        in1_single_tile_size);

    TT_FATAL(
        ashape[-1] == bshape[-2],
        "Dimension K (A.shape[-1] = {}, B.shape[-2] = {}) must match for matmul",
        ashape[-1],
        bshape[-2]);
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        ashape[-1],
        in0_tile_shape[1]);
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        bshape[-1],
        in1_tile_shape[1]);

    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        operation_attributes.program_config.value());
    const auto& in0_block_w = program_config.in0_block_w;
    const auto& per_core_M = program_config.per_core_M;
    const auto& per_core_N = program_config.per_core_N;
    const auto& workers_per_bank = program_config.num_workers_per_dram_bank;
    const auto& fused_activation = program_config.fused_activation;

    const auto& untilize_out = operation_attributes.untilize_out;
    const bool skip_compute = false;
    const bool skip_in0_mcast = false;
    const bool skip_write_back = false;

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    uint32_t B = 1;
    uint32_t Mt = get_batch_size(ashape) * ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
    uint32_t in0_last_ktile_w = a.logical_shape()[-1] % in0_tile_shape[1];

    TT_FATAL(Kt % in0_block_w == 0, "Kt ({}) must be divisible by in0_block_w ({})", Kt, in0_block_w);

    const auto& output_mesh = output.mesh_tensor();

    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded_descriptor(
        device,
        input_all_cores_storage,
        output_all_cores_storage,
        math_fidelity,
        fp32_dest_acc_en,
        math_approx_mode,
        packer_l1_acc,
        dst_full_sync_en,
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config),
        B,
        Mt,
        Nt,
        Kt,
        in0_block_w,
        in0_last_ktile_w,
        per_core_M,
        per_core_N,
        workers_per_bank,
        fused_activation,
        a_mesh,
        b,
        bias_mesh,
        output_mesh,
        in0_tile,
        in1_tile,
        bias.has_value() ? bias->tensor_spec().tile() : output_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        bias_data_format,
        output_data_format,
        untilize_out,
        skip_compute,
        skip_in0_mcast,
        skip_write_back,
        row_broadcast_bias);
}

// tt-stack 0040 (#58922): the multi-core decode variant, cores_per_bank > 0 only (select_program_factory).
ttnn::device_operation::ProgramArtifacts MatmulDRAMShardedMulticoreProgramFactory::create_program_artifacts(
    const ttnn::prim::MatmulParams& operation_attributes,
    const ttnn::prim::MatmulInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value) {
    const auto& input_tensors = tensor_args.input_tensors;
    const auto& bias = tensor_args.optional_input_tensors.at(0);
    const auto& a = input_tensors.at(0);
    const auto& b = input_tensors.at(1);
    const auto& output = tensor_return_value.at(0);
    const auto& ashape = a.padded_shape();
    const auto& bshape = b.padded_shape();
    const auto in0_tile = a.tensor_spec().tile();
    const auto in1_tile = b.tensor_spec().tile();
    const auto output_tile = tt::tt_metal::Tile({in0_tile.get_tile_shape()[0], in1_tile.get_tile_shape()[1]});
    const tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(b.dtype());
    const tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    TT_FATAL(!bias.has_value(), "cores_per_bank does not support a fused bias");
    TT_FATAL(
        a.shard_spec().has_value() && output.shard_spec().has_value(), "Both input A and output must have shard specs");
    TT_FATAL(ashape[-1] == bshape[-2], "Dimension K (A {} vs B {}) must match for matmul", ashape[-1], bshape[-2]);
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config.value();
    const auto& program_config = std::get<operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>(
        operation_attributes.program_config.value());
    TT_FATAL(program_config.cores_per_bank > 0, "MatmulDRAMShardedMulticoreProgramFactory needs cores_per_bank > 0");
    tt::tt_metal::IDevice* device = a.device();
    const uint32_t Kt = ashape[-1] / in0_tile.get_tile_shape()[1];
    const uint32_t Nt = bshape[-1] / in1_tile.get_tile_shape()[1];
    return reuse_dram_sharded_optimized_helpers::create_program_dram_sharded_multicore_spec(
        device,
        a.mesh_tensor(),
        b.mesh_tensor(),
        output.mesh_tensor(),
        ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config),
        ttnn::get_throttle_level(operation_attributes.compute_kernel_config),
        Kt,
        Nt,
        program_config.in0_block_w,
        program_config.cores_per_bank,
        in0_tile,
        in1_tile,
        output_tile,
        in0_data_format,
        in1_data_format,
        output_data_format);
}

}  // namespace ttnn::prim

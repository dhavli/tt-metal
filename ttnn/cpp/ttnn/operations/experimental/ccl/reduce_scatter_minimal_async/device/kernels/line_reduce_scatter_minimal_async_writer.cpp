// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

// QWEN36_PREFILL_RS_BANK (tt-stack bench/probes/rs_bank): BANK-CONTIGUOUS fabric packets in the line RS writer.
//
// The remote intermediate is DRAM-INTERLEAVED. The interleaved address generator (InterleavedAddrGen::get_noc_addr,
// tt_metal/hw/inc/internal/dataflow/dataflow_api_addrgen.h:292-297 with get_bank_offset_index/get_bank_index at :18-41)
// maps page id t to bank b = t mod NUM_DRAM_BANKS and to address
//     bank_base_address + (t div NUM_DRAM_BANKS) * aligned_page_size + bank_to_dram_offset[b]
// on core dram_bank_to_noc_xy[noc][b]. Inside one slice row the tile ids are consecutive, so the columns c, c+NB,
// c+2NB, ... of a row segment sit in ONE bank at consecutive per-bank pages: one NoC write of k pages to the first
// one's address carries all k of them (when aligned_page_size == page_size). The paired reader overlay
// (line_reader.bank.cpp) fills the sender CB in that bank order, inside each sync group (the tiles between two
// out_ready semaphore increments, cut at a channel end), so the set of tiles that has landed when the reducer's
// semaphore wait returns is exactly the stock set: only the order inside a group changes.
//
// This writer recomputes every tile's destination with the SAME address generator the stock writer uses
// (addrgen_detail::get_noc_address), and merges a tile into the previous chunk only when its NoC address is exactly
// the previous chunk's end. A packet carries up to 4 such chunks (NOC_SCATTER_WRITE_MAX_CHUNKS; 1 chunk = a plain
// unicast write) and at most (mux channel buffer - packet header) / page_size tiles: 8 bf8 tiles = 8704 B at the
// served QWEN36_FABRIC_PAYLOAD=8704. So the bank mapping is never ASSUMED for correctness: a layout where the bank
// order does not give contiguous pages only costs packets. Transport only: every tile lands at the same remote
// address as before; the reduction is untouched (tier 1, bitwise).
//
// rs_bank_mustfail = true is the ttsim-only must-fail build: it merges every next tile into the chunk regardless of
// its address, i.e. it assumes consecutive tiles in L1 are consecutive in DRAM (the free/u-ccl rs_uni error), which
// corrupts the bank mapping; the self-test must REFUSE it. Served only by arms that mount BOTH overlay files; the
// Python side (rs_bank.py) refuses a boot whose flag and mounted kernels disagree (marker + version on both files).
#define QWEN36_PREFILL_RS_BANK_KERNEL 1
#define QWEN36_PREFILL_RS_BANK_ORDER_VERSION 1
constexpr bool rs_bank_mustfail = false;

using address_t = uint32_t;
using ttnn::ccl::Topology;
using namespace tt::tt_fabric::linear::experimental;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t cb_compute_output_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_reader_output_id = get_compile_time_arg_val(2);
constexpr uint32_t tile_granularity = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(5);
constexpr uint32_t input_num_pages = get_compile_time_arg_val(6);
constexpr uint32_t input_batch_num_pages = get_compile_time_arg_val(7);
constexpr uint32_t input_channel_num_pages = get_compile_time_arg_val(8);
constexpr uint32_t output_batch_num_pages = get_compile_time_arg_val(9);
constexpr uint32_t output_channel_num_pages = get_compile_time_arg_val(10);
constexpr uint32_t input_tensor_B = get_compile_time_arg_val(11);
constexpr uint32_t input_tensor_Wt = get_compile_time_arg_val(12);
constexpr uint32_t slice_C = get_compile_time_arg_val(13);
constexpr uint32_t slice_Ht = get_compile_time_arg_val(14);
constexpr uint32_t slice_Wt = get_compile_time_arg_val(15);
constexpr uint32_t dim = get_compile_time_arg_val(16);
constexpr bool sync_with_other_direction = get_compile_time_arg_val(17);
constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(18);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(19);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(20);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(21);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(22);
constexpr uint32_t barrier_target_count = get_compile_time_arg_val(23);

constexpr uint32_t num_ct_args = 24;

constexpr ccl_routing_utils::line_unicast_route_info_t forward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<num_ct_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t forward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args>();
constexpr ccl_routing_utils::line_unicast_route_info_t backward_unicast_route_info =
    ccl_routing_utils::get_line_unicast_route_info_from_args<
        num_ct_args + ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();
constexpr ccl_routing_utils::line_multicast_route_info_t backward_multicast_route_info =
    ccl_routing_utils::get_line_multicast_route_info_from_args<
        num_ct_args + 2 * ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args>();

// The packet cap: one packet's payload must fit the worker's mux channel buffer slot after the packet header
// (the same clamp as the shipped rs4 overlay, bench/probes/rs_pkt4/line_writer.pkt4.cpp).
constexpr uint32_t rs_bank_cap_tiles =
    (fabric_mux_channel_buffer_size_bytes > sizeof(PACKET_HEADER_TYPE) + page_size)
        ? (uint32_t)((fabric_mux_channel_buffer_size_bytes - sizeof(PACKET_HEADER_TYPE)) / page_size)
        : 1u;
static_assert(rs_bank_cap_tiles >= 1, "rs_bank: a packet must carry at least one page");
static_assert((uint64_t)rs_bank_cap_tiles * page_size <= 0xFFFFu, "rs_bank: chunk/payload sizes are uint16 fields");
static_assert(tile_granularity >= 1 && tile_granularity <= 64, "rs_bank: per-round address array is on the stack");

// ---- rs_bank::GroupIter: BYTE-IDENTICAL in line_reader.bank.cpp and line_writer.bank.cpp (the ttsim gate diffs it).
namespace rs_bank {
constexpr uint32_t NB = NUM_DRAM_BANKS;
// Visits the n tiles of one sync group, which starts at column col0 of the slice row at row_off (a tile offset, a
// multiple of the row stride) and runs on in row-major order, in BANK order: inside each row segment the columns
// are taken class by class, class k = the columns at offsets k, k+NB, k+2NB, ... from the segment start (one DRAM
// bank each, since the tile ids of a segment are consecutive), each class in ascending column order (= ascending
// per-bank page). The class order is rotated by (row index + worker rotation) mod NB so that the 8 workers of a
// direction do not all stream the same bank at the same time. tile() is relative to the slice's tile_id_start.
struct GroupIter {
    uint32_t row_off;  // tile offset of the current segment's row
    uint32_t row_idx;  // slice row index of the current segment (rotation input)
    uint32_t c0;       // first column of the current segment
    uint32_t len;      // columns in the current segment
    uint32_t rest;     // tiles of the group after the current segment
    uint32_t rot;      // class rotation of the current segment
    uint32_t q;        // class slot being visited (0..NB-1)
    uint32_t c;        // current column offset inside the segment
    uint32_t wrot;     // per-worker rotation
    uint32_t stride;   // row stride in tiles (input_tensor_Wt)
    uint32_t width;    // slice width in tiles (slice_Wt)

    bool seek_class() {
        for (; q < NB; ++q) {
            uint32_t cls = q + rot;
            if (cls >= NB) {
                cls -= NB;
            }
            if (cls < len) {
                c = cls;
                return true;
            }
        }
        return false;
    }
    void start_segment() {
        rot = (row_idx + wrot) % NB;
        q = 0;
        seek_class();
    }
    void init(uint32_t row_off_, uint32_t row_idx_, uint32_t col0, uint32_t n, uint32_t wrot_, uint32_t stride_,
              uint32_t width_) {
        stride = stride_;
        width = width_;
        wrot = wrot_;
        row_off = row_off_;
        row_idx = row_idx_;
        c0 = col0;
        len = (width - col0 < n) ? (width - col0) : n;
        rest = n - len;
        start_segment();
    }
    uint32_t tile() const { return row_off + c0 + c; }
    void advance() {
        c += NB;
        if (c < len) {
            return;
        }
        ++q;
        if (seek_class()) {
            return;
        }
        if (rest == 0) {
            return;  // group exhausted; the caller never reads past n tiles
        }
        row_off += stride;
        row_idx++;
        c0 = 0;
        len = (rest < width) ? rest : width;
        rest -= len;
        start_segment();
    }
};
}  // namespace rs_bank
// ---- end rs_bank::GroupIter

namespace detail {

inline bool do_forward_sync(const bool is_forward) {
    if constexpr (!sync_with_other_direction) {
        return false;
    }

    return is_forward;
}
}  // namespace detail
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    uint32_t arg_idx = 0;
    address_t intermediate_address = get_arg_val<address_t>(arg_idx++);
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    Semaphore<> fwd_bwd_sem(get_arg_val<uint32_t>(arg_idx++));
    uint32_t opposite_core_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t opposite_core_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    bool use_barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    size_t barrier_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool is_forward = get_arg_val<uint32_t>(arg_idx++);
    const bool is_first_device_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t num_targets_in_direction = get_arg_val<uint32_t>(arg_idx++);
    const bool do_final_reduction = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_pages_read_in_row = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_row_offset = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_read = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t start_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    bool mux_connection_valid = get_arg_val<uint32_t>(arg_idx++) == 1;
    const bool is_termination_master = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_y = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_channel_base_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_info_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_connection_handshake_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_flow_control_address = get_arg_val<uint32_t>(arg_idx++);
    const size_t fabric_mux_buffer_index_address = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t fabric_mux_channel_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_id = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_sync_address = get_semaphore(termination_sync_id);
    uint32_t local_fabric_mux_status_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_flow_control_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_teardown_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t local_buffer_index_address = get_semaphore(get_arg_val<uint32_t>(arg_idx++));
    uint32_t termination_master_noc_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t termination_master_noc_y = get_arg_val<uint32_t>(arg_idx++);

    const auto& unicast_route_info = (is_forward) ? forward_unicast_route_info : backward_unicast_route_info;
    [[maybe_unused]] const auto& multicast_route_info =
        (is_forward) ? forward_multicast_route_info : backward_multicast_route_info;

    constexpr uint32_t ct_idx =
        num_ct_args + 2 * (ccl_routing_utils::num_line_unicast_args + ccl_routing_utils::num_line_multicast_args);

#ifdef INTERMEDIATE_IS_SHARDED
    constexpr uint32_t ct_offset = 7;

    using intermediate_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx),       // Memory layout
        get_compile_time_arg_val(ct_idx + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + 3),   // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(ct_idx + 4),   // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(ct_idx + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + 6)>;  // pages_per_shard_y

    const auto [intermediate_mapping_table, intermediate_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<intermediate_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<intermediate_tensor_shard_info> intermediate_addrgen = {
        .bank_base_address = intermediate_address, .shard_array = intermediate_mapping_table};

    arg_idx += intermediate_rt_increment;

#else
    constexpr auto intermediate_tensor_args = TensorAccessorArgs<ct_idx>();
    constexpr uint32_t ct_offset = intermediate_tensor_args.num_compile_time_args();
    auto intermediate_addrgen = TensorAccessor(intermediate_tensor_args, intermediate_address);
#endif

#ifdef OUTPUT_IS_SHARDED
    using output_tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(ct_idx + ct_offset),       // Memory layout
        get_compile_time_arg_val(ct_idx + ct_offset + 1),   // The number of sharding cores
        get_compile_time_arg_val(ct_idx + ct_offset + 2),   // The page size we offset each write to
        get_compile_time_arg_val(ct_idx + ct_offset + 3),   // The number of pages in each sharding row not including
                                                            // padding pages
        get_compile_time_arg_val(ct_idx + ct_offset + 4),   // This defines times when contiguous pages can't be
                                                            // calculated
        get_compile_time_arg_val(ct_idx + ct_offset + 5),   // pages_per_shard_x
        get_compile_time_arg_val(ct_idx + ct_offset + 6)>;  // pages_per_shard_y

    const auto [output_mapping_table, output_rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<output_tensor_shard_info>(get_arg_addr(arg_idx));
    experimental::ShardedAddrGen<output_tensor_shard_info> output_addrgen = {
        .bank_base_address = output_address, .shard_array = output_mapping_table};

    arg_idx += output_rt_increment;
#else
    constexpr auto output_tensor_args = TensorAccessorArgs<ct_idx + ct_offset>();
    auto output_addrgen = TensorAccessor(output_tensor_args, output_address);
#endif

    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel>* mux_connection_handle;
    tt::tt_fabric::WorkerToFabricMuxSender<fabric_mux_num_buffers_per_channel> mux_connection;
    if (mux_connection_valid) {
        mux_connection = tt::tt_fabric::build_connection_to_fabric_endpoint<fabric_mux_num_buffers_per_channel>(
            fabric_mux_x,
            fabric_mux_y,
            fabric_mux_channel_id,
            fabric_mux_num_buffers_per_channel,
            fabric_mux_channel_buffer_size_bytes,
            fabric_mux_channel_base_address,
            fabric_mux_connection_info_address,
            fabric_mux_connection_handshake_address,
            fabric_mux_flow_control_address,
            fabric_mux_buffer_index_address,
            local_flow_control_address,
            local_teardown_address,
            local_buffer_index_address);
        mux_connection_handle = &mux_connection;
    } else {
        mux_connection_handle = nullptr;
    }

    if (mux_connection_valid) {
        // need to wait for fabric mux to be ready to accept connections
        tt::tt_fabric::wait_for_fabric_endpoint_ready(
            fabric_mux_x, fabric_mux_y, fabric_mux_status_address, local_fabric_mux_status_address);
        tt::tt_fabric::fabric_client_connect_start(*mux_connection_handle);
    }

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_connect_finish(*mux_connection_handle);
    }

    Noc noc_obj;
    CircularBuffer cb_compute_output(cb_compute_output_id);
    CircularBuffer cb_reader_output(cb_reader_output_id);

    auto pkt_scatter_hdr = PacketHeaderPool::allocate_header();
    auto pkt_unicast_hdr = PacketHeaderPool::allocate_header();
    auto pkt_hdr_seminc = PacketHeaderPool::allocate_header();

    if (use_barrier_sem) {
        if (num_targets_in_direction) {
            // Use neighbor unicast instead of multicast to support reshaped 'logical linear' mesh devices
            ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);
            fabric_unicast_noc_unicast_atomic_inc_set_state<
                UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
                pkt_hdr_seminc,
                static_cast<uint8_t>(unicast_route_info.distance_in_hops),
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                    0,                           // ignore
                    static_cast<uint32_t>(1)});  // increment 1

            uint64_t opposite_direction_barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, barrier_sem, 0);
            fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                mux_connection_handle,
                pkt_hdr_seminc,
                tt::tt_fabric::NocUnicastAtomicIncCommandHeader{opposite_direction_barrier_sem_noc_addr_in_pkt, 0});

            // Exactly one increment arrives: the two workers sharing a link (Dk forward and Dk+1
            // backward) handshake with each other, so the target is structurally 1 and does not
            // depend on barrier_target_count.
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 1);
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(barrier_sem), 0);
        }
    }

    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem, 0);

    // only initialize if we're actually going to send anything over fabric
    if (num_targets_in_direction) {
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_scatter_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_unicast_hdr, unicast_route_info);
        ccl_routing_utils::fabric_set_line_unicast_route(pkt_hdr_seminc, unicast_route_info);

        fabric_unicast_noc_scatter_write_set_state<
            UnicastScatterWriteUpdateMask::ChunkSizes | UnicastScatterWriteUpdateMask::PayloadSize>(
            pkt_scatter_hdr,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            NocUnicastScatterCommandHeader({0, 0}, {static_cast<uint16_t>(page_size)}),
            page_size * 2);

        fabric_unicast_noc_unicast_write_set_state<UnicastWriteUpdateMask::PayloadSize>(
            pkt_unicast_hdr, static_cast<uint8_t>(unicast_route_info.distance_in_hops), nullptr, page_size);

        fabric_unicast_noc_unicast_atomic_inc_set_state<
            UnicastAtomicIncUpdateMask::Val | UnicastAtomicIncUpdateMask::Flush>(
            pkt_hdr_seminc,
            static_cast<uint8_t>(unicast_route_info.distance_in_hops),
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                0,                           // ignore
                static_cast<uint32_t>(1)});  // increment 1
    }

    const uint32_t intermediate_full_offset = is_forward ? 0 : input_num_pages;

    // per-worker class rotation (rs_bank::GroupIter): the worker's ordinal in its direction, from its tile range
    const uint32_t rs_bank_span = start_tiles_to_read - start_tiles_read;
    const uint32_t rs_bank_wrot = (rs_bank_span > 0 ? start_tiles_read / rs_bank_span : 0) % rs_bank::NB;

    uint32_t chunk_count = 0;
    for (uint32_t b = 0; b < input_tensor_B; b++) {
        int slice_idx = is_forward ? ring_size - 1 : 0;

        uint32_t batch_offset = input_batch_num_pages * b;
        for (uint32_t iter = 0; iter < num_targets_in_direction; ++iter) {
            CircularBuffer& cb_output = is_first_device_in_direction ? cb_reader_output : cb_compute_output;
            chunk_count = 0;

            uint32_t intermediate_tile_id_start;
            if constexpr (dim == 3) {
                intermediate_tile_id_start = slice_idx * slice_Wt + batch_offset + intermediate_full_offset;
            } else if constexpr (dim == 2) {
                intermediate_tile_id_start = slice_idx * slice_Ht * slice_Wt + batch_offset + intermediate_full_offset;
            } else if constexpr (dim == 1) {
                intermediate_tile_id_start =
                    slice_idx * slice_C * slice_Ht * slice_Wt + batch_offset + intermediate_full_offset;
            } else {
                ASSERT(false);
            }
            for (uint32_t c = 0; c < slice_C; ++c) {
                // linear (stock-order) position of the next sync group's first tile
                uint32_t intermediate_pages_read_in_row = start_pages_read_in_row;
                uint32_t intermediate_row_offset = start_row_offset;
                uint32_t intermediate_row_idx = start_row_offset / input_tensor_Wt;

                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                rs_bank::GroupIter group;
                uint32_t group_left = 0;

                // Write to remote intermediate buffer
                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    if (group_left == 0) {
                        // a sync group = the rounds up to the next out_ready increment, cut at the channel end
                        const uint32_t rounds_to_sync = chunks_per_sync - (chunk_count % chunks_per_sync);
                        const uint32_t n = std::min(rounds_to_sync * tile_granularity, tiles_remaining_to_read);
                        group.init(
                            intermediate_row_offset,
                            intermediate_row_idx,
                            intermediate_pages_read_in_row,
                            n,
                            rs_bank_wrot,
                            input_tensor_Wt,
                            slice_Wt);
                        group_left = n;
                        intermediate_pages_read_in_row += n;
                        while (intermediate_pages_read_in_row >= slice_Wt) {
                            intermediate_pages_read_in_row -= slice_Wt;
                            intermediate_row_offset += input_tensor_Wt;
                            intermediate_row_idx++;
                        }
                    }

                    // destination of every tile of this round, in the CB (bank) order the reader filled it
                    uint64_t dst[tile_granularity];
                    for (uint32_t k = 0; k < num_pages_to_read; ++k) {
                        dst[k] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                            intermediate_addrgen, intermediate_tile_id_start + group.tile(), 0);
                        group.advance();
                    }
                    group_left -= num_pages_to_read;

                    cb_output.wait_front(tile_granularity);
                    size_t l1_read_addr = cb_output.get_read_ptr();
                    for (uint32_t j = 0; j < num_pages_to_read;) {
                        // one packet: up to NOC_SCATTER_WRITE_MAX_CHUNKS address-contiguous chunks, <= cap tiles
                        uint64_t chunk_addr[NOC_SCATTER_WRITE_MAX_CHUNKS];
                        uint16_t chunk_bytes[NOC_SCATTER_WRITE_MAX_CHUNKS];
                        uint32_t num_chunks = 0;
                        uint32_t pkt_tiles = 0;
                        while (j < num_pages_to_read && pkt_tiles < rs_bank_cap_tiles) {
                            if (num_chunks > 0 &&
                                (rs_bank_mustfail ||
                                 dst[j] == chunk_addr[num_chunks - 1] + chunk_bytes[num_chunks - 1])) {
                                chunk_bytes[num_chunks - 1] += page_size;
                            } else {
                                if (num_chunks == NOC_SCATTER_WRITE_MAX_CHUNKS) {
                                    break;
                                }
                                chunk_addr[num_chunks] = dst[j];
                                chunk_bytes[num_chunks] = page_size;
                                num_chunks++;
                            }
                            pkt_tiles++;
                            j++;
                        }
                        const uint16_t pkt_bytes = static_cast<uint16_t>(pkt_tiles * page_size);
                        if (num_chunks == 1) {
                            fabric_unicast_noc_unicast_write_with_state<
                                UnicastWriteUpdateMask::DstAddr | UnicastWriteUpdateMask::PayloadSize>(
                                mux_connection_handle,
                                pkt_unicast_hdr,
                                l1_read_addr,
                                NocUnicastCommandHeader{chunk_addr[0]},
                                pkt_bytes);
                        } else {
                            fabric_unicast_noc_scatter_write_with_state<
                                UnicastScatterWriteUpdateMask::DstAddrs | UnicastScatterWriteUpdateMask::ChunkSizes |
                                UnicastScatterWriteUpdateMask::PayloadSize>(
                                mux_connection_handle,
                                pkt_scatter_hdr,
                                l1_read_addr,
                                NocUnicastScatterCommandHeader(
                                    chunk_addr, chunk_bytes, static_cast<uint8_t>(num_chunks)),
                                pkt_bytes);
                        }
                        // the header is rewritten for the next packet: its last send must have left L1
                        noc_obj.async_writes_flushed();
                        l1_read_addr += pkt_bytes;
                        tiles_read += pkt_tiles;
                    }
                    cb_output.pop_front(tile_granularity);

                    chunk_count++;
                    if (chunk_count % chunks_per_sync == 0) {
                        // 2. unicast output ready semaphore
                        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                            mux_connection_handle,
                            pkt_hdr_seminc,
                            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
                    }
                }
                intermediate_tile_id_start += input_channel_num_pages;
            }

            if (chunk_count % chunks_per_sync != 0) {
                // 2. unicast output ready semaphore
                fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
                    mux_connection_handle,
                    pkt_hdr_seminc,
                    tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0});
            }

            // Next slice idx
            if (is_forward) {
                slice_idx--;
            } else {
                slice_idx++;
            }
        }

        // Do write of final reduction and sync local FWD/BWD cores
        if (do_final_reduction) {
            // Write output
            uint32_t output_tile_id_start = b * output_batch_num_pages;
            for (uint32_t c = 0; c < slice_C; ++c) {
                uint32_t tiles_read = start_tiles_read;
                uint32_t tiles_to_read = start_tiles_to_read;

                while (tiles_read < tiles_to_read) {
                    uint32_t tiles_remaining_to_read = tiles_to_read - tiles_read;
                    uint32_t num_pages_to_read = std::min(tiles_remaining_to_read, tile_granularity);

                    cb_compute_output.wait_front(tile_granularity);
                    uint32_t l1_read_addr = cb_compute_output.get_read_ptr();
                    for (uint32_t j = 0; j < num_pages_to_read; ++j) {
                        uint32_t output_tile_id = output_tile_id_start + tiles_read;
                        uint64_t noc_write_addr = output_addrgen.get_noc_addr(output_tile_id);
                        noc_async_write(l1_read_addr, noc_write_addr, page_size);
                        l1_read_addr += page_size;
                        tiles_read++;
                    }

                    if (detail::do_forward_sync(is_forward)) {
                        noc_obj.async_write_barrier();
                    } else {
                        noc_obj.async_writes_flushed();
                    }
                    cb_compute_output.pop_front(tile_granularity);
                    if (detail::do_forward_sync(is_forward)) {
                        // Tell local backwards reader that it can proceed
                        fwd_bwd_sem.up(noc_obj, opposite_core_sem_noc0_x, opposite_core_sem_noc0_y, 1);
                    }
                }
                output_tile_id_start += output_channel_num_pages;
            }
            noc_obj.async_write_barrier();
        }
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();

    if (mux_connection_valid) {
        tt::tt_fabric::fabric_client_disconnect(*mux_connection_handle);

        if (is_termination_master) {
            Semaphore<> termination_sync(termination_sync_id);
            termination_sync.wait(num_mux_clients - 1);
            tt::tt_fabric::fabric_endpoint_terminate(fabric_mux_x, fabric_mux_y, fabric_mux_termination_signal_address);
        } else {
            uint64_t dest_addr =
                safe_get_noc_addr(termination_master_noc_x, termination_master_noc_y, termination_sync_address, 0);
            noc_semaphore_inc(dest_addr, 1);
            noc_obj.async_atomic_barrier();
        }
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
}

// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// cfx 0063 (QWEN36_AGMM_FWO): the in0 forward to the peer chip, offloaded from the in0 chain to an idle core.
//
// Stock Linear AGMM: one core per in0 row chain (the chain's fabric sender) forwards each LOCAL in0 K block to the peer
// with forward_half_block_to_fabric_neighbor() INSIDE its receive loop, so the whole row chain waits for the forward
// (~12.9 us per K block vs 6.7-10 us of compute) during the local half. This kernel runs on one free core per row
// (row full_grid.y - 2), is that row's mux client instead of the chain core, reads the same local K blocks straight
// from the local input tensor (in3, the injector's READ_FROM_LOCAL_INPUT source) into two L1 slots (the next block's
// read overlaps the current forward) and calls forward_half_block_to_fabric_neighbor() with the same tile coordinates,
// the same L1 row layout (K_block_tiles row stride) and the same one-semaphore-increment-per-block, in the same
// (m_block, k_block) order. The peer's gather-buffer writes and out_ready_sem increments are unchanged in content and
// count; the chain cores run the no-fabric kernel (no forward at all).
//
// AGMM_FWO_MUSTFAIL: the last tile of every forwarded block is zeroed in L1 before the forward (the peer's gathered
// in0 is wrong -> its output differs -> the model's self-test must REFUSE). ttsim only.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "matmul_dataflow_common.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
constexpr uint32_t K_block_tiles = get_compile_time_arg_val(3);
constexpr uint32_t M_blocks_per_core = get_compile_time_arg_val(4);
constexpr uint32_t in0_tile_size = get_compile_time_arg_val(5);
constexpr uint32_t num_devices = get_compile_time_arg_val(6);
constexpr uint32_t my_rank = get_compile_time_arg_val(7);
constexpr uint32_t in3_tile_size = get_compile_time_arg_val(8);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(9);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(10);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(11);
constexpr uint32_t K_tiles_per_device = get_compile_time_arg_val(12);
constexpr uint32_t K_block_tail_tiles = get_compile_time_arg_val(13);
constexpr uint32_t cb_id_fwd = get_compile_time_arg_val(14);

constexpr uint8_t fabric_mux_num_buffers_per_channel = get_compile_time_arg_val(15);
constexpr size_t fabric_mux_channel_buffer_size_bytes = get_compile_time_arg_val(16);
constexpr size_t fabric_mux_status_address = get_compile_time_arg_val(17);
constexpr size_t fabric_mux_termination_signal_address = get_compile_time_arg_val(18);
constexpr uint32_t num_mux_clients = get_compile_time_arg_val(19);

constexpr uint32_t mux_arg_count = 20;

constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_forward =
    ccl_routing_utils::get_line_unicast_route_info_from_args<mux_arg_count>();
constexpr ccl_routing_utils::line_unicast_route_info_t unicast_route_info_backward =
    ccl_routing_utils::get_line_unicast_route_info_from_args<mux_arg_count + ccl_routing_utils::num_line_unicast_args>();

constexpr uint32_t ct_arg_count = mux_arg_count + 2 * ccl_routing_utils::num_line_unicast_args;

// The factory refuses FWO for anything but a two-device Linear ring, where the forwarded blocks are exactly the
// K_blocks_per_device local ones (k_block_iter < K_num_blocks - K_blocks_per_device in dm_in0_sender.cpp).
static_assert(num_devices == 2, "QWEN36_AGMM_FWO: two devices only");

constexpr uint32_t K_blocks_per_device = (K_tiles_per_device + K_block_tiles - 1) / K_block_tiles;
constexpr uint32_t in0_block_num_tiles = M_block_tiles * K_block_tiles;
constexpr uint32_t slot_bytes = in0_block_num_tiles * in0_tile_size;
constexpr uint32_t blocks_total = M_blocks_per_core * K_blocks_per_device;

// The local in0 block (m_block, k_block) into one L1 slot, laid out exactly as the chain's in0 CB slot that the stock
// forward reads: row stride K_block_tiles tiles, the k_left_tiles valid tiles first. Rows >= M_tiles are not read (the
// stock injector does not read them either, and the forward stops at M_tiles).
template <typename LocalAccessor>
FORCE_INLINE void issue_block_reads(
    Noc noc, const LocalAccessor& in3_reader, uint32_t slot_addr, uint32_t m_tile, uint32_t m_tile_end, uint32_t kb) {
    const uint32_t k_local_start = kb * K_block_tiles;
    const uint32_t k_left_tiles = (kb == K_blocks_per_device - 1) ? K_block_tail_tiles : K_block_tiles;
    for (uint32_t i = m_tile; i < m_tile_end && i < M_tiles; i++) {
        uint32_t write_ptr = slot_addr + (i - m_tile) * K_block_tiles * in0_tile_size;
        uint32_t tile_id = i * K_tiles_per_device + k_local_start;
        for (uint32_t j = 0; j < k_left_tiles; j++) {
            noc.async_read(in3_reader, CoreLocalMem<uint8_t>(write_ptr), in0_tile_size, {.page_id = tile_id}, {});
            write_ptr += in0_tile_size;
            tile_id++;
        }
    }
}

void kernel_main() {
    uint32_t cargidx = 0;
    const uint32_t in0_addr = get_common_arg_val<uint32_t>(cargidx++);
    cargidx++;  // in2_addr (bias): unused here
    const uint32_t in3_addr = get_common_arg_val<uint32_t>(cargidx++);
    cargidx++;  // out_ready_sem_backward: Linear signals the forward semaphore only
    const size_t out_ready_sem_forward = get_common_arg_val<uint32_t>(cargidx++);

    uint32_t argidx = 0;
    const uint32_t M_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t M_end_tile = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_injector_noc0_x = get_arg_val<uint32_t>(argidx++);
    const uint8_t out_ready_sem_injector_noc0_y = get_arg_val<uint32_t>(argidx++);

    // in0_reader: the gather buffer (the forward's destination address generator, as in the stock sender);
    // in3_reader: the local input (the forward's source data).
    constexpr auto in0_args = TensorAccessorArgs<ct_arg_count>();
    const auto in0_reader = TensorAccessor(in0_args, in0_addr);
    constexpr auto in3_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto in3_reader = TensorAccessor(in3_args, in3_addr);

    MuxConnection<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes> mux =
        parse_mux_connection_args<fabric_mux_num_buffers_per_channel, fabric_mux_channel_buffer_size_bytes>(
            argidx, 0, 0);
    auto* mux_connection_handle = mux.build_and_connect(fabric_mux_status_address);

    // Same headers / route as the stock sender: device 0 (no backward neighbour) long-sends with the forward route,
    // every other device short-sends with the backward route.
    PacketHeaders pkt_hdrs;
    if constexpr (num_targets_backward_direction == 0) {
        pkt_hdrs = allocate_and_init_packet_headers(
            num_targets_forward_direction > 0,
            unicast_route_info_forward,
            in0_reader,
            num_tiles_to_write_per_packet,
            in3_tile_size);
    } else {
        pkt_hdrs = allocate_and_init_packet_headers(
            true, unicast_route_info_backward, in0_reader, num_tiles_to_write_per_packet, in3_tile_size);
    }
    const uint64_t out_ready_sem_injector_noc_addr_forward_in_pkt =
        safe_get_noc_addr(out_ready_sem_injector_noc0_x, out_ready_sem_injector_noc0_y, out_ready_sem_forward, 0);

    Noc noc_obj;
    CircularBuffer cb_fwd(cb_id_fwd);
    const uint32_t slot_base = cb_fwd.get_write_ptr();

    const bool do_forward = (num_targets_backward_direction != 0) || (num_targets_forward_direction > 0);
    if (do_forward && mux_connection_handle != nullptr && blocks_total > 0) {
        // prologue: block 0 -> slot 0
        {
            const uint32_t m_tile = M_start_tile;
            const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
            issue_block_reads(noc_obj, in3_reader, slot_base, m_tile, m_tile_end, 0);
        }
        uint32_t slot = 0;
        for (uint32_t m_block_iter = 0; m_block_iter < M_blocks_per_core; m_block_iter++) {
            const uint32_t m_tile = M_start_tile + m_block_iter * M_block_tiles;
            const uint32_t m_tile_end = std::min(m_tile + M_block_tiles, M_end_tile);
            const uint32_t current_M_block_tiles = m_tile_end - m_tile;
            for (uint32_t kb = 0; kb < K_blocks_per_device; kb++) {
                const uint32_t slot_addr = slot_base + slot * slot_bytes;
                noc_obj.async_read_barrier();  // this block's reads (the only ones in flight) have landed
                // next block -> other slot, in flight while this one is forwarded. The other slot's previous forward
                // is complete: forward_half_block_to_fabric_neighbor flushes every packet's source read.
                {
                    uint32_t nm = m_block_iter;
                    uint32_t nkb = kb + 1;
                    if (nkb == K_blocks_per_device) {
                        nkb = 0;
                        nm++;
                    }
                    if (nm < M_blocks_per_core) {
                        const uint32_t n_tile = M_start_tile + nm * M_block_tiles;
                        const uint32_t n_tile_end = std::min(n_tile + M_block_tiles, M_end_tile);
                        issue_block_reads(
                            noc_obj, in3_reader, slot_base + (slot ^ 1u) * slot_bytes, n_tile, n_tile_end, nkb);
                    }
                }
                const uint32_t k_left_tiles = (kb == K_blocks_per_device - 1) ? K_block_tail_tiles : K_block_tiles;
                const uint32_t k_right_tiles = K_block_tiles - k_left_tiles;
                const uint32_t k_block_left_tile = my_rank * K_tiles_per_device + kb * K_block_tiles;
#ifdef AGMM_FWO_MUSTFAIL
                {
                    const uint32_t row_end = std::min(m_tile_end, M_tiles);
                    const uint32_t rows = row_end > m_tile ? row_end - m_tile : 0;
                    if (rows > 0) {
                        volatile tt_l1_ptr uint32_t* t = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            slot_addr + ((rows - 1) * K_block_tiles + (k_left_tiles - 1)) * in0_tile_size);
                        for (uint32_t w = 0; w < in0_tile_size / sizeof(uint32_t); w++) {
                            t[w] = 0;
                        }
                    }
                }
#endif
                forward_half_block_to_fabric_neighbor(
                    noc_obj,
                    m_tile,
                    k_block_left_tile,
                    current_M_block_tiles,
                    k_left_tiles,
                    k_right_tiles,
                    num_tiles_to_write_per_packet,
                    slot_addr,
                    K_tiles,
                    in0_reader,
                    mux_connection_handle,
                    pkt_hdrs,
                    in0_tile_size,
                    out_ready_sem_injector_noc_addr_forward_in_pkt,
                    true,
                    M_tiles,
                    true);
                slot ^= 1u;
            }
        }
        noc_obj.async_read_barrier();
    }

    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();

    if (mux.connection_valid) {
        close_mux(
            noc_obj,
            mux_connection_handle,
            mux.is_termination_master,
            mux.termination_sync_address,
            mux.num_mux_clients,
            mux.fabric_mux_x,
            mux.fabric_mux_y,
            fabric_mux_termination_signal_address,
            mux.termination_master_noc_x,
            mux.termination_master_noc_y);
    }
    noc_obj.async_write_barrier();
}

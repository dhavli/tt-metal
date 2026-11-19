// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::transformer {

struct SDPAProgramConfig {
    tt::tt_metal::CoreCoord compute_with_storage_grid_size;
    std::optional<CoreRangeSet> sub_core_grids;
    std::size_t q_chunk_size;
    std::size_t k_chunk_size;
    std::optional<bool> exp_approx_mode;
    uint32_t max_cores_per_head_batch = 16;
    // cfx (tt-stack 0025): causal/chunked prefill only -- request the GQA K/V multicast schedule
    // (tt-metal #56119). Unlike TT_SDPA_GQA_MCAST=1 this is per call and the op TT_FATALs when the
    // schedule cannot be applied to the call (never a silent fallback). Ignored by decode SDPA.
    bool gqa_kv_mcast = false;
    // cfx (tt-stack 0026): with gqa_kv_mcast, allow the zigzag-pair placement when the (head, q_chunk) units exceed
    // the cores but the default schedule gives every busy core exactly one causal zigzag pair (e.g. q=k=128 for
    // 12 q-heads x 2048 tokens). Each core then runs the default's exact work at the same chunk size; only core
    // placement and the K/V source (one multicast stream per group) change. No effect without gqa_kv_mcast.
    bool gqa_kv_mcast_pairs = false;
    // cfx (tt-stack 0027): SDPA prefill reader depth. Data movement only -- the bytes that land in each CB and the
    // order compute consumes them are unchanged, so outputs are bit-identical by construction.
    //   reader_barrier_tiles: tiles issued between partial NoC read barriers in the prefill reader (K, V, Q, mask).
    //     0 = the kernel's default (get_barrier_read_threshold: 4 tiles in flight for bf8 on 110 cores; 8 on the
    //     GQA-mcast injectors, TT_SDPA_GQA_MCAST_BARRIER). >= the chunk's tile count = one barrier per chunk.
    //   kv_cb_slots: K and V circular-buffer depth in chunks. 0 = default (2; TT_SDPA_GQA_MCAST_KV_SLOTS in GQA-mcast
    //     mode). 3..4 lets the reader run further ahead of compute. L1 cost: one K + one V chunk per extra slot.
    uint32_t reader_barrier_tiles = 0;
    uint32_t kv_cb_slots = 0;
};

// Paired geometry for an HMA-shared paged K/V cache (chunked prefill SDPA and
// paged decode SDPA). When the physical buffer was allocated for a different layer's
// (num_kv_heads, block_size, head_dim) view — e.g. vLLM hybrid kv-cache-groups — the
// reader must address it with this call's block_size / num_kv_heads (Q drives head_dim).
// Presence of the override is the outer std::optional on the op API; both fields are
// required plain values whenever the struct is provided. Default {0, 0} means inactive
// after the optional is collapsed into operation attributes.
struct PagedCacheGeometryOverride {
    uint32_t block_size = 0;
    uint32_t num_kv_heads = 0;

    [[nodiscard]] bool active() const { return block_size != 0 || num_kv_heads != 0; }
};

}  // namespace ttnn::operations::transformer

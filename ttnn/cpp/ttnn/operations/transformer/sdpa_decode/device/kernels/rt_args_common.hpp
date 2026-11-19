// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/constants.hpp>
#include <optional>
#include <tuple>

inline uint32_t nearest_n(uint32_t x, uint32_t n) { return ((x + n - 1) / n) * n; }

template <uint8_t max>
inline uint8_t nearest_pow_of_2_up_to_8(uint32_t x) {
    if (x == 0) {
        return 1;  // Handle edge case when x is 0
    }

    // Round up to nearest power of 2
    --x;          // Decrease x by 1 to handle exact powers of 2
    x |= x >> 1;  // Propagate the highest set bit
    x |= x >> 2;  // Propagate the highest set bit

    // Uncomment if want to support full range of uin32_t
    // x |= x >> 4;  // Propagate the highest set bit
    // x |= x >> 8;  // Propagate the highest set bit
    // x |= x >> 16;  // Propagate the highest set bit

    uint32_t result = x + 1;  // Add 1 to get the next power of 2

    // Cap the result at max
    return (result > max) ? max : result;
}

inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_workload_for_core(
    int cur_pos,
    int cur_batch,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    std::optional<uint32_t> sliding_window_size = std::nullopt) {
    uint32_t window_start = 0;
    uint32_t window_start_unaligned = 0;  // Keep track of the actual window start for masking
    uint32_t valid_seq_len;

    if (sliding_window_size.has_value() && sliding_window_size.value() > 0) {
        // Calculate actual window bounds
        uint32_t window_end = cur_pos + 1;  // exclusive end
        window_start_unaligned =
            (window_end > sliding_window_size.value()) ? (window_end - sliding_window_size.value()) : 0;

        // Round window_start down to chunk boundary to ensure we capture the full window
        uint32_t window_start_aligned = (window_start_unaligned / k_chunk_size) * k_chunk_size;

        // Round window_end up to chunk boundary to ensure we capture the full window
        uint32_t window_end_aligned = nearest_n(window_end, k_chunk_size);

        // Calculate valid_seq_len based on the sliding window range
        valid_seq_len = window_end_aligned - window_start_aligned;
        window_start = window_start_aligned;  // Use aligned start for chunk calculations
    } else {
        // Standard behavior: process from beginning up to cur_pos
        valid_seq_len = nearest_n(cur_pos + 1, k_chunk_size);
        window_start = 0;
        window_start_unaligned = 0;
    }

    uint32_t pst_value = valid_seq_len / tt::constants::TILE_HEIGHT;
    uint32_t window_start_chunk = window_start / k_chunk_size;
    uint32_t num_chunks_value = valid_seq_len / k_chunk_size;

    uint32_t k_chunk_start = window_start_chunk;
    uint32_t k_chunk_end = window_start_chunk;

    // Distribute active chunks among cores
    if (num_cores_per_batch > int(num_chunks_value)) {
        int chunks_per_core = (core_num < int(num_chunks_value)) ? 1 : 0;
        k_chunk_start = window_start_chunk + (num_chunks_value - core_num - 1) * chunks_per_core;
        k_chunk_end = window_start_chunk + (num_chunks_value - core_num) * chunks_per_core;
    } else {
        int chunks_per_core = num_chunks_value / num_cores_per_batch;
        int residuals = num_chunks_value % num_cores_per_batch;
        int reversed_core_num = num_cores_per_batch - core_num - 1;
        k_chunk_start =
            window_start_chunk + reversed_core_num * chunks_per_core + std::min(residuals, reversed_core_num);
        k_chunk_end = k_chunk_start + chunks_per_core;
        if (reversed_core_num < residuals) {
            k_chunk_end += 1;
        }
    }

    return {pst_value, num_chunks_value, k_chunk_start, k_chunk_end, window_start_unaligned, window_start_chunk};
}

template <uint32_t Sk_chunk_t, uint32_t max_size>
inline uint32_t get_dynamic_Sk_chunk_t(int cur_pos) {
    if constexpr (Sk_chunk_t == 0) {
        // Cur_pos + 1 for position, but -1 for divup, so cancels out
        // ie. divup(a, b) = (a - 1) / b + 1
        uint32_t seq_len_in_tiles = cur_pos / tt::constants::TILE_HEIGHT + 1;

        // Use nearest power of 2 to nicely divide total CB size which is some factor of max_size
        // Technically, should not be an issue but seeing PCC issues when using 3 tiles eg.
        // - Can switch to nearest tile if this is fixed
        return nearest_pow_of_2_up_to_8<max_size>(seq_len_in_tiles);
    }
    return Sk_chunk_t;
}

/******************************************************************************
 * cfx (tt-stack 0034): speculative-verify multi-position fold (SDPA_SPEC_FOLD_T)
 *
 * One program serves T candidate positions of each of B users, reading each user's KV once. Layout is candidate-major,
 * exactly the concatenation of the T legacy per-candidate calls: Q [1, T*B, NQH, DH] (row u = j*B + b), cur_pos
 * [T*B] (entry j*B + b), page table [B, ...], output [1, T*B, NQH, vDH]. Candidate j of user b attends [0, pos(j,b)].
 * Inside a core, row-tile r of the Q block belongs to candidate r / PNHt_cand.
 *
 * Workload: the KV scan runs to the group's LARGEST valid position (pos_max), but the chunk split across the
 * cores_per_head is the legacy split of the SMALLEST valid position (pos_min), with the chunks past it (at most one:
 * T-1 < 32 <= chunk positions) appended to core 0, which always holds that split's last chunk. So every core's range
 * starts at or before pos_min: no row of any core is masked over its whole range (a fully masked row would give
 * max = -inf and exp(-inf - -inf) = NaN), and a candidate whose bound lies in the pos_min chunk sees exactly the
 * legacy partial softmaxes plus, on core 0, one chunk that contributes exp(-inf) = 0 to its sum and output.
 * Entries of UINT32_MAX (skip) are ignored; a group with no valid entry is skipped like a legacy skip row.
 ******************************************************************************/
// Program defines set by the factory only on a fold program; every other program compiles the legacy code.
#ifndef SDPA_SPEC_FOLD_T
#define SDPA_SPEC_FOLD_T 0
#endif
#ifndef SDPA_SPEC_FOLD_B
#define SDPA_SPEC_FOLD_B 1
#endif
#ifndef SDPA_SPEC_FOLD_PNHT_CAND
#define SDPA_SPEC_FOLD_PNHT_CAND 1
#endif
#ifndef SDPA_SPEC_FOLD_MUSTFAIL
#define SDPA_SPEC_FOLD_MUSTFAIL 0
#endif

struct SpecFoldPositions {
    uint32_t pos_min;
    uint32_t pos_max;
    bool skip;
};

template <uint32_t T, typename GetPos>
inline SpecFoldPositions spec_fold_positions(uint32_t cur_batch, uint32_t B, GetPos get_pos) {
    SpecFoldPositions p{UINT32_MAX, 0, true};
    for (uint32_t j = 0; j < T; ++j) {
        const uint32_t v = get_pos(j * B + cur_batch);
        if (v == UINT32_MAX) {
            continue;
        }
        p.skip = false;
        p.pos_min = v < p.pos_min ? v : p.pos_min;
        p.pos_max = v > p.pos_max ? v : p.pos_max;
    }
    return p;
}

// get_workload_for_core, or (T > 0) its fold form: legacy split of pos_min's chunks, core 0 extended to pos_max's.
// k_num_chunks stays pos_min's count (the "core has data" predicate the tree reduction uses on every RISC).
template <uint32_t T>
inline std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> get_workload_for_core_spec_fold(
    uint32_t cur_pos,
    uint32_t pos_min,
    int cur_batch,
    int core_num,
    int num_cores_per_batch,
    uint32_t k_chunk_size,
    std::optional<uint32_t> sliding_window_size = std::nullopt) {
    if constexpr (T == 0) {
        return get_workload_for_core(cur_pos, cur_batch, core_num, num_cores_per_batch, k_chunk_size, sliding_window_size);
    } else {
        auto [pst, n_min, start, end, wsu, wsc] =
            get_workload_for_core(pos_min, cur_batch, core_num, num_cores_per_batch, k_chunk_size, std::nullopt);
        const uint32_t n_max = nearest_n(cur_pos + 1, k_chunk_size) / k_chunk_size;
        if (core_num == 0 && n_max > n_min) {
            end = n_max;
            pst = n_max * k_chunk_size / tt::constants::TILE_HEIGHT;
        }
        return {pst, n_min, start, end, wsu, wsc};
    }
}

// A chunk needs the per-candidate causal mask iff its last position is past the smallest candidate bound. With the
// fold workload only core 0 can own such chunks (at most two). Writer and compute evaluate this same predicate in
// the same ascending chunk order: one mask block per qualifying chunk, pushed by the writer, popped by compute.
inline bool spec_fold_chunk_needs_mask(uint32_t k_chunk, uint32_t k_chunk_size, uint32_t pos_min) {
    return (k_chunk * k_chunk_size + k_chunk_size - 1) > pos_min;
}

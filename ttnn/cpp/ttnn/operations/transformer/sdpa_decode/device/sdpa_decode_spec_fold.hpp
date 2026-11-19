#include "tt-metalium/tt_stack_public_defaults.hpp"
// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// cfx (tt-stack patches/tt-metal/0034): speculative-verify multi-position fold for paged decode SDPA.
//
// TT_SDPA_DEC_SPEC_FOLD (read once per process):
//   unset / "" / "0"  off: every call is the stock op (a fold-shaped call keeps the stock cur_pos TT_FATAL)
//   "1"               a call whose Q batch is T x the page-table rows (T >= 2) and whose cur_pos tensor has T x rows
//                     entries runs as ONE program that reads each user's KV once for all T candidates
//   "mustfail"        as "1", with every valid candidate's causal bound one position too far (candidate j also sees
//                     candidate j+1): equivalence against the per-candidate calls must FAIL
//   anything else     TT_THROW
//
// Fold layout, candidate-major: exactly the T legacy calls concatenated on the batch dim.
//   Q        [1, T*B, NQH, DH]   row u = j*B + b: candidate j of user b (bf16, TILE, DRAM interleaved)
//   cur_pos  [T*B] int32         entry j*B + b; -1 skips that candidate
//   page     [B, max_blocks]     int32 DRAM interleaved (one row per user; all candidates share it)
//   output   [1, T*B, NQH, vDH]  row u as Q
// The KV of user b / kv head h is read once per (b, h) core group; the group's T candidates are T*PNHt_cand
// row-tiles of one Q block, each with its own causal bound (per-chunk mask blocks from the writer).
// Legacy shapes (Q batch == page-table rows) never take this path, whatever the env says.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <string_view>

#include <tt_stl/assert.hpp>

#include "sdpa_decode_device_operation_types.hpp"

namespace ttnn::prim::sdpa_spec_fold {

enum class Mode { OFF, ON, MUSTFAIL };

inline Mode mode() {
    static const Mode m = [] {
        const char* raw = tt_stack_public::getenv("TT_SDPA_DEC_SPEC_FOLD");
        const std::string_view v = raw == nullptr ? std::string_view{} : std::string_view{raw};
        if (v.empty() || v == "0") {
            return Mode::OFF;
        }
        if (v == "1") {
            return Mode::ON;
        }
        if (v == "mustfail") {
            return Mode::MUSTFAIL;
        }
        TT_THROW("TT_SDPA_DEC_SPEC_FOLD must be unset, 0, 1 or mustfail; got '{}'", v);
    }();
    return m;
}

inline const char* mode_name() { return mode() == Mode::MUSTFAIL ? "mustfail" : (mode() == Mode::ON ? "1" : "0"); }

// T (>= 2) when this call is a fold call, else 0. Pure function of the mode and of hashed tensor shapes, so a fold
// program and a legacy program never share a program-cache entry.
inline uint32_t fold_T(const SdpaDecodeParams& attrs, const SdpaDecodeInputs& args) {
    if (mode() == Mode::OFF || !attrs.paged_attention || !attrs.is_causal) {
        return 0;
    }
    if (!args.page_table_tensor.has_value() || args.page_table_tensor->is_sharded()) {
        return 0;
    }
    if (!args.cur_pos_tensor.has_value() || args.cur_pos_tensor->is_sharded()) {
        return 0;
    }
    const uint32_t B = args.page_table_tensor->padded_shape()[0];
    const uint32_t qB = args.q.padded_shape()[1];
    if (B == 0 || qB <= B || qB % B != 0) {
        return 0;
    }
    if (args.cur_pos_tensor->padded_shape()[-1] != qB) {
        return 0;
    }
    return qB / B;
}

}  // namespace ttnn::prim::sdpa_spec_fold

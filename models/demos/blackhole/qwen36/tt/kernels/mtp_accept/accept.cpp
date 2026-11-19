// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_DEVACCEPT (docs/MTP-BATCH.md section 25): the MTP-at-batch accept on device, one core, one RISC, data
// movement and scalar integer work only. The first program of the draft trace (cd under SO1_CSPLIT, else C2 / C). It
// reads the verify's outputs (the ids; under the served exact rule also the rule's accept flags and residual ids), the
// verify tokens (rows j >= 1 are this step's drafts) and one plan-only page the host wrote before V, and writes every
// input of the commit/draft traces that the host accept used to write after reading the verify back: the slot-index
// commit page, the draft anchor rows, t0n, the K draft KV / RoPE positions, (SPEC) the K chain-noise rows, plus an m
// page (m_b, then t0n_b) the host checks at the next drain. mtp_batch_host.da_reference is this kernel in Python.
//
// Row u = j*BW + b (j-major). Per row b < BW (n = usable drafts, 0 for inactive rows):
//   exact row (SPEC and plan EXACT[b]): m = longest prefix i < min(n, K) with accept[i*BW + b] > 0.5 (fp32 bits:
//            0x3F000000 < (int32)bits <= 0x7F800000, the same truth value as the host's float compare for every
//            pattern: NaN, zeros, negatives and 0.5 fail; +inf passes)
//            tail = m < n ? resid[m*BW + b] : ids[m*BW + b]
//   else     m = longest prefix i < min(n, K) with v_tok[(i+1)*BW + b] == ids[i*BW + b];   tail = ids[m*BW + b]
//   (must-fail flag: row 0, if active, m+1 or m-1 at m == T-1)
//   gslot  = active ? (commit_plus1 ? min(m+1, T-1) : m) : 0;   gslot[b >= BW] = 0xFFFF (SKIP)
//   anchor = m*BW + b;   t0n = active ? tail : 0
//   d_pos[k] / d_rpos[k] = kvbase + m + k / rbase + m + k   if active and m + k < lim, else -1
//   (SPEC) d_sgum[k] row b = noise table k row m*BW + b   (the host wrote the noise for every possible m)
//
// All transfers are FULL aligned pages (the TensorAccessor page size is the buffer's aligned page size: sub-page NoC
// reads silently fail on this stack). Every staged output page is zeroed first, so the padding past a page's payload
// is written as zeros. Every L1 offset is 64-byte aligned. RISC stores to L1 are scalars only.
//
// The accept flags / residual ids are read either as one 32x32 TILE (A_TILE = 1: element (r, 0) at face (r >= 16 ? 2
// : 0), row r & 15; 32-bit faces of 1024 B) or as ROW_MAJOR [M, 1] pages (A_TILE = 0).
//
// Compile args: {BW, T, SPEC, A_TILE, P_IDS, P_VTOK, P_PLAN, P_GSLOT, P_ANCHOR, P_T0N, P_DPOS, P_M, P_ACC, P_RES,
//                P_NZ, CB_IN, CB_OUT}
//               + TensorAccessorArgs(host-written class: plan, v_tok, every output, the noise tables)
//               + TensorAccessorArgs(ids) + TensorAccessorArgs(accept) + TensorAccessorArgs(resid)
//               (SPEC = 0: the accept / resid slots repeat the ids' args; never used)
// Runtime args: {ids, v_tok, plan, gslot, anchor, t0n, m, accept, resid, d_pos[K], d_rpos[K], (SPEC) tab[K], nz[K]}
// SPEC with P_NZ == 0 (QWEN36_MTP_SPEC_SAMPLE_SSP): the exact accept without the chain-noise copy (no tab / nz).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t BW = get_compile_time_arg_val(0);
    constexpr uint32_t T = get_compile_time_arg_val(1);
    constexpr uint32_t SPEC = get_compile_time_arg_val(2);
    constexpr uint32_t A_TILE = get_compile_time_arg_val(3);
    constexpr uint32_t P_IDS = get_compile_time_arg_val(4);
    constexpr uint32_t P_VTOK = get_compile_time_arg_val(5);
    constexpr uint32_t P_PLAN = get_compile_time_arg_val(6);
    constexpr uint32_t P_GSLOT = get_compile_time_arg_val(7);
    constexpr uint32_t P_ANCHOR = get_compile_time_arg_val(8);
    constexpr uint32_t P_T0N = get_compile_time_arg_val(9);
    constexpr uint32_t P_DPOS = get_compile_time_arg_val(10);
    constexpr uint32_t P_M = get_compile_time_arg_val(11);
    constexpr uint32_t P_ACC = get_compile_time_arg_val(12);
    constexpr uint32_t P_RES = get_compile_time_arg_val(13);
    constexpr uint32_t P_NZ = get_compile_time_arg_val(14);
    constexpr uint32_t CB_IN = get_compile_time_arg_val(15);
    constexpr uint32_t CB_OUT = get_compile_time_arg_val(16);
    constexpr auto h_args = TensorAccessorArgs<17>();
    constexpr auto i_args = TensorAccessorArgs<h_args.next_compile_time_args_offset()>();
    constexpr auto a_args = TensorAccessorArgs<i_args.next_compile_time_args_offset()>();
    constexpr auto r_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    constexpr uint32_t K = T - 1;
    constexpr uint32_t M = BW * T;
    constexpr uint32_t SLOT = 64;   // one id / token / flag page per 64-byte L1 slot
    constexpr uint32_t OSLOT = 256; // one single-page output per 256-byte L1 slot
    static_assert(M <= 32 && T >= 2 && K <= 15, "device accept: M = BW*T <= 32, 1 <= K <= 15");
    static_assert(P_IDS <= SLOT && P_VTOK <= SLOT && P_T0N <= SLOT, "device accept: id pages must fit a 64 B slot");
    static_assert(P_GSLOT <= OSLOT && P_ANCHOR <= OSLOT && P_DPOS <= OSLOT && P_M <= OSLOT, "device accept: output page > 256 B");
    static_assert(P_PLAN >= 1024 && P_PLAN % 64 == 0, "device accept: plan page must hold 256 words");
    static_assert(SPEC == 0 || A_TILE == 1 || (P_ACC <= SLOT && P_RES <= SLOT), "device accept: RM flag pages > 64 B");
    static_assert(SPEC == 0 || A_TILE == 0 || (P_ACC == 4096 && P_RES == 4096), "device accept: 32-bit 32x32 tiles");
    static_assert(SPEC == 0 || (P_NZ % 64 == 0 && P_NZ <= 4096), "device accept: noise row page");
    // plan page words (mtp_batch_host.da_plan_page)
    constexpr uint32_t W_FLAGS = 0, W_N = 32, W_ACT = 64, W_KVB = 96, W_RB = 128, W_LIM = 160, W_EXACT = 192;
    constexpr uint32_t F_COMMIT_PLUS1 = 1, F_MUSTFAIL = 2;
    constexpr uint32_t F32_HALF = 0x3F000000u, F32_INF = 0x7F800000u;
    // L1 layout (input CB)
    constexpr uint32_t I_VTOK = P_PLAN;
    constexpr uint32_t I_IDS = I_VTOK + M * SLOT;
    constexpr uint32_t I_ACC = I_IDS + M * SLOT;
    constexpr uint32_t A_BYTES = SPEC ? (A_TILE ? 4096 : M * SLOT) : 0;
    constexpr uint32_t I_RES = I_ACC + A_BYTES;
    constexpr uint32_t I_NZ = I_RES + A_BYTES;
    constexpr uint32_t I_END = I_NZ + (SPEC ? K * BW * P_NZ : 0);
    // L1 layout (output CB)
    constexpr uint32_t O_GSLOT = 0, O_ANCHOR = OSLOT, O_M = 2 * OSLOT, O_DPOS = 3 * OSLOT;
    constexpr uint32_t O_DRPOS = O_DPOS + K * OSLOT;
    constexpr uint32_t O_T0N = O_DRPOS + K * OSLOT;
    constexpr uint32_t O_END = O_T0N + BW * SLOT;
    static_assert(I_END <= 64 * 1024 && O_END <= 32 * 1024, "device accept: L1 scratch");

    const uint32_t a_ids = get_arg_val<uint32_t>(0);
    const uint32_t a_vtok = get_arg_val<uint32_t>(1);
    const uint32_t a_plan = get_arg_val<uint32_t>(2);
    const uint32_t a_gslot = get_arg_val<uint32_t>(3);
    const uint32_t a_anchor = get_arg_val<uint32_t>(4);
    const uint32_t a_t0n = get_arg_val<uint32_t>(5);
    const uint32_t a_m = get_arg_val<uint32_t>(6);
    const uint32_t a_acc = get_arg_val<uint32_t>(7);
    const uint32_t a_res = get_arg_val<uint32_t>(8);
    constexpr uint32_t A_DPOS = 9;
    constexpr uint32_t A_DRPOS = A_DPOS + K;
    constexpr uint32_t A_TAB = A_DRPOS + K;
    constexpr uint32_t A_NZ = A_TAB + K;

    // Both CBs are private scratch (reserved, never pushed).
    cb_reserve_back(CB_IN, 1);
    cb_reserve_back(CB_OUT, 1);
    const uint32_t in = get_write_ptr(CB_IN);
    const uint32_t out = get_write_ptr(CB_OUT);

    {
        const auto plan = TensorAccessor(h_args, a_plan, P_PLAN);
        const auto vtok = TensorAccessor(h_args, a_vtok, P_VTOK);
        const auto ids = TensorAccessor(i_args, a_ids, P_IDS);
        noc_async_read_page(0, plan, in);
        for (uint32_t u = 0; u < M; ++u) {
            noc_async_read_page(u, vtok, in + I_VTOK + u * SLOT);
            noc_async_read_page(u, ids, in + I_IDS + u * SLOT);
        }
        if constexpr (SPEC) {
            const auto acc = TensorAccessor(a_args, a_acc, P_ACC);
            const auto res = TensorAccessor(r_args, a_res, P_RES);
            if constexpr (A_TILE) {
                noc_async_read_page(0, acc, in + I_ACC);
                noc_async_read_page(0, res, in + I_RES);
            } else {
                for (uint32_t u = 0; u < M; ++u) {
                    noc_async_read_page(u, acc, in + I_ACC + u * SLOT);
                    noc_async_read_page(u, res, in + I_RES + u * SLOT);
                }
            }
        }
    }
    {
        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out);
        for (uint32_t w = 0; w < O_END / 4; ++w) {
            z[w] = 0u;
        }
    }
    noc_async_read_barrier();
    asm volatile("" ::: "memory");

    volatile tt_l1_ptr uint32_t* pw = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in);
    auto vt = [&](uint32_t u) -> uint32_t { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + I_VTOK + u * SLOT); };
    auto id = [&](uint32_t u) -> uint32_t { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + I_IDS + u * SLOT); };
    auto flag_off = [&](uint32_t base, uint32_t u) -> uint32_t {
        if constexpr (A_TILE) {
            return base + ((u >= 16 ? 2u : 0u) * 1024u) + (u & 15u) * 64u;
        } else {
            return base + u * SLOT;
        }
    };
    auto acc = [&](uint32_t u) -> uint32_t { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + flag_off(I_ACC, u)); };
    auto res = [&](uint32_t u) -> uint32_t { return *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in + flag_off(I_RES, u)); };
    auto ow = [&](uint32_t off, uint32_t w) -> volatile tt_l1_ptr uint32_t* {
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out + off) + w;
    };

    const uint32_t flags = pw[W_FLAGS];
    for (uint32_t b = BW; b < 32; ++b) {
        *ow(O_GSLOT, b) = 0xFFFFu;
    }
    uint32_t mrow[32];
    for (uint32_t b = 0; b < BW; ++b) {
        const uint32_t n = pw[W_N + b];
        const bool act = pw[W_ACT + b] != 0u;
        const bool ex = SPEC && pw[W_EXACT + b] != 0u;
        const int32_t kvb = static_cast<int32_t>(pw[W_KVB + b]);
        const int32_t rb = static_cast<int32_t>(pw[W_RB + b]);
        const int32_t lim = static_cast<int32_t>(pw[W_LIM + b]);
        uint32_t m = 0;
        if (ex) {
            while (m < n && m < K) {
                const int32_t s = static_cast<int32_t>(acc(m * BW + b));
                if (!(s > static_cast<int32_t>(F32_HALF) && s <= static_cast<int32_t>(F32_INF))) {
                    break;
                }
                ++m;
            }
        } else {
            while (m < n && m < K && vt((m + 1) * BW + b) == id(m * BW + b)) {
                ++m;
            }
        }
        if (b == 0 && act && (flags & F_MUSTFAIL)) {
            m = (m < T - 1) ? m + 1 : m - 1;  // QWEN36_MTP_DEVACCEPT_MUSTFAIL
        }
        uint32_t commit = 0;
        if (act) {
            commit = (flags & F_COMMIT_PLUS1) ? ((m + 1 < T - 1) ? m + 1 : T - 1) : m;
        }
        const uint32_t tail = (ex && m < n) ? res(m * BW + b) : id(m * BW + b);
        const uint32_t t0 = act ? tail : 0u;
        *ow(O_GSLOT, b) = commit;
        *ow(O_ANCHOR, b) = m * BW + b;
        *ow(O_T0N + b * SLOT, 0) = t0;
        *ow(O_M, b) = m;
        *ow(O_M, 32 + b) = t0;
        mrow[b] = m;
        for (uint32_t k = 0; k < K; ++k) {
            const bool ok = act && static_cast<int32_t>(m + k) < lim;
            *ow(O_DPOS + k * OSLOT, b) = ok ? static_cast<uint32_t>(kvb + static_cast<int32_t>(m + k)) : 0xFFFFFFFFu;
            *ow(O_DRPOS + k * OSLOT, b) = ok ? static_cast<uint32_t>(rb + static_cast<int32_t>(m + k)) : 0xFFFFFFFFu;
        }
    }
    asm volatile("" ::: "memory");

    if constexpr (SPEC && P_NZ != 0) {
        // the chain noise: table k row m_b*BW + b -> d_sgum{k} row b (whole row pages, staged through L1)
        // (P_NZ == 0: QWEN36_MTP_SPEC_SAMPLE_SSP, the chain's fs_cut makes its noise from the m page; no copy)
        for (uint32_t k = 0; k < K; ++k) {
            const auto tab = TensorAccessor(h_args, get_arg_val<uint32_t>(A_TAB + k), P_NZ);
            for (uint32_t b = 0; b < BW; ++b) {
                noc_async_read_page(mrow[b] * BW + b, tab, in + I_NZ + (k * BW + b) * P_NZ);
            }
        }
        noc_async_read_barrier();
        for (uint32_t k = 0; k < K; ++k) {
            const auto nz = TensorAccessor(h_args, get_arg_val<uint32_t>(A_NZ + k), P_NZ);
            for (uint32_t b = 0; b < BW; ++b) {
                noc_async_write_page(b, nz, in + I_NZ + (k * BW + b) * P_NZ);
            }
        }
    }

    noc_async_write_page(0, TensorAccessor(h_args, a_gslot, P_GSLOT), out + O_GSLOT);
    noc_async_write_page(0, TensorAccessor(h_args, a_anchor, P_ANCHOR), out + O_ANCHOR);
    noc_async_write_page(0, TensorAccessor(h_args, a_m, P_M), out + O_M);
    for (uint32_t k = 0; k < K; ++k) {
        noc_async_write_page(0, TensorAccessor(h_args, get_arg_val<uint32_t>(A_DPOS + k), P_DPOS), out + O_DPOS + k * OSLOT);
        noc_async_write_page(0, TensorAccessor(h_args, get_arg_val<uint32_t>(A_DRPOS + k), P_DPOS), out + O_DRPOS + k * OSLOT);
    }
    {
        const auto t0n = TensorAccessor(h_args, a_t0n, P_T0N);
        for (uint32_t b = 0; b < BW; ++b) {
            noc_async_write_page(b, t0n, out + O_T0N + b * SLOT);
        }
    }
    noc_async_write_barrier();
}

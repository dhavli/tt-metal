// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_DEVFEED (docs/MTP-BATCH.md section 31): the next verify's draft-dependent inputs, built on device by the
// last program of the sampled family's draft trace (cd), so a fast-path step's verify can be enqueued before the host
// has read the drafts back. Data movement and scalar integer work only; one chain row per core, one RISC.
//
// Chain row u = k*BW + b (k < K the draft depth, b < BW the row), core u. Inputs (ROW_MAJOR, one page per row):
//   t0n   [BW, 1]      uint32  the device accept's t0n (the token at the next step's base position)
//   dr[k] [BW, 1]      uint32  draft k's ids (the chain's emitted ids: FULL vocab ids)
//   qp[k] [BW, C2]     float32 draft k's candidate probs
//   qi[k] [BW, C2]     float32 draft k's candidate ids (MAP: draft-vocab SUB-ids, mapped here through TAB)
//   tab   [N/16, 16]   uint32  (MAP) the draft vocab's full ids, entry s = ids[s] (mtp_draft_vocab, unshifted)
// Outputs (the bucket's persistent verify inputs, M = (K+1)*BW rows), what MTPBatchHost.plan + host_inputs write:
//   v_tok [M, 1]       uint32  row b = t0n[b]; row (k+1)*BW + b = dr[k][b]
//   v_sqp [M, C2]      float32 row u = qp[k][b] (bits); rows >= K*BW zero
//   v_sqi [M, C2]      float32 row u = float(tab[min(trunc(qi), N-1)]) (MAP) or qi (bits); rows >= K*BW zero
//   v_sx  [M, 1]       float32 row u = float(dr[k][b]); rows >= K*BW zero
// Core u < BW also writes v_tok row u (t0n) and the zero rows K*BW + u.
//
// Conversions are integer code with the host's results: u32 -> f32 round-to-nearest-even (torch float(int)); the
// q id f32 -> index truncates toward zero, negatives / NaN / inf -> 0 (in range by construction), then clamps to N-1
// (mtp_batch._read_chain: tab[t.long().clamp_(0, N-1)]).
//
// MF = 1 (QWEN36_MTP_DEVFEED_MUSTFAIL): draft 0 of row 0 + 1 in v_tok and v_sx (the self-tests must REFUSE).
//
// All transfers are FULL aligned pages (sub-page NoC reads silently fail on this stack); staged output pages are zeroed
// first, so the padding past a page's payload is written as zeros. Every L1 offset is 64-byte aligned.
//
// Compile args: {BW, K, C2, MAP, MF, N_IDS, P_ID, P_Q, P_TB, CB}
//               + TensorAccessorArgs (one class: every tensor is DRAM interleaved, page sizes passed explicitly)
//   P_ID: the page of t0n, dr[k], v_tok, v_sx (<= 64 B); P_Q: the page of qp[k], qi[k], v_sqp, v_sqi; P_TB: tab's page.
// Runtime args: {u, t0n, v_tok, v_sqp, v_sqi, v_sx, tab, dr[K], qp[K], qi[K]}

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

static inline uint32_t clz32(uint32_t x) {
    uint32_t n = 0;
    if (x == 0) return 32;
    while ((x & 0x80000000u) == 0) {
        x <<= 1;
        ++n;
    }
    return n;
}

// u32 -> IEEE f32 bits, round to nearest even
static inline uint32_t u32_to_f32(uint32_t u) {
    if (u == 0) return 0;
    uint32_t e = 31 - clz32(u);
    uint32_t m;
    if (e <= 23) {
        m = u << (23 - e);
    } else {
        const uint32_t sh = e - 23;
        m = u >> sh;
        const uint32_t rem = u & ((1u << sh) - 1u);
        const uint32_t half = 1u << (sh - 1);
        if (rem > half || (rem == half && (m & 1u))) {
            ++m;
            if (m == (1u << 24)) {
                m >>= 1;
                ++e;
            }
        }
    }
    return ((127u + e) << 23) | (m & 0x7FFFFFu);
}

// f32 bits -> clamp(trunc(x), 0, n - 1); negatives, NaN and inf -> 0
static inline uint32_t f32_index(uint32_t bits, uint32_t n) {
    const uint32_t ex = (bits >> 23) & 0xFFu;
    if ((bits & 0x80000000u) || ex < 127u || ex == 255u) return 0;
    const uint32_t sh = ex - 127u;  // value in [2^sh, 2^(sh+1))
    if (sh >= 31u) return n - 1;
    const uint32_t mant = (bits & 0x7FFFFFu) | 0x800000u;
    const uint32_t v = sh <= 23u ? (mant >> (23u - sh)) : (mant << (sh - 23u));
    return v < n ? v : n - 1;
}

void kernel_main() {
    constexpr uint32_t BW = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t C2 = get_compile_time_arg_val(2);
    constexpr uint32_t MAP = get_compile_time_arg_val(3);
    constexpr uint32_t MF = get_compile_time_arg_val(4);
    constexpr uint32_t N_IDS = get_compile_time_arg_val(5);
    constexpr uint32_t P_ID = get_compile_time_arg_val(6);
    constexpr uint32_t P_Q = get_compile_time_arg_val(7);
    constexpr uint32_t P_TB = get_compile_time_arg_val(8);
    constexpr uint32_t CB = get_compile_time_arg_val(9);
    constexpr auto h_args = TensorAccessorArgs<10>();

    constexpr uint32_t SLOT = 64;
    constexpr uint32_t QS = (P_Q + 63) / 64 * 64;
    constexpr uint32_t TS = (P_TB + 63) / 64 * 64;
    static_assert(P_ID <= SLOT, "devfeed: id pages must fit a 64 B slot");
    static_assert(P_Q >= C2 * 4 && P_Q % 64 == 0, "devfeed: q row page");
    static_assert(MAP == 0 || (P_TB >= 64 && N_IDS >= 1), "devfeed: id table page holds 16 ids");
    static_assert(K >= 1 && BW * (K + 1) <= 32, "devfeed: (K+1)*BW <= 32");
    // L1 layout (one private CB)
    constexpr uint32_t L_DR = 0;            // draft id page (in), then the v_tok draft row (out)
    constexpr uint32_t L_X = L_DR + SLOT;   // v_sx row
    constexpr uint32_t L_T0 = L_X + SLOT;   // t0n page (u < BW): v_tok row u as is
    constexpr uint32_t L_Z = L_T0 + SLOT;   // a zero page (q rows and id rows)
    constexpr uint32_t L_QP = L_Z + QS;     // probs row
    constexpr uint32_t L_QI = L_QP + QS;    // q id row (in)
    constexpr uint32_t L_QO = L_QI + QS;    // mapped q id row (out)
    constexpr uint32_t L_TB = L_QO + QS;    // C2 table pages
    constexpr uint32_t L_END = L_TB + (MAP ? C2 * TS : 0);
    static_assert(L_END <= 96 * 1024, "devfeed: L1 scratch");

    uint32_t ai = 0;
    const uint32_t u = get_arg_val<uint32_t>(ai++);
    const uint32_t a_t0n = get_arg_val<uint32_t>(ai++);
    const uint32_t a_vtok = get_arg_val<uint32_t>(ai++);
    const uint32_t a_vqp = get_arg_val<uint32_t>(ai++);
    const uint32_t a_vqi = get_arg_val<uint32_t>(ai++);
    const uint32_t a_vx = get_arg_val<uint32_t>(ai++);
    const uint32_t a_tab = get_arg_val<uint32_t>(ai++);
    const uint32_t k = u / BW;
    const uint32_t b = u % BW;
    const uint32_t a_dr = get_arg_val<uint32_t>(ai + k);
    const uint32_t a_qp = get_arg_val<uint32_t>(ai + K + k);
    const uint32_t a_qi = get_arg_val<uint32_t>(ai + 2 * K + k);

    cb_reserve_back(CB, 1);
    const uint32_t s = get_write_ptr(CB);
    auto W = [&](uint32_t off) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(s + off); };

    const auto t0n = TensorAccessor(h_args, a_t0n, P_ID);
    const auto vtok = TensorAccessor(h_args, a_vtok, P_ID);
    const auto vx = TensorAccessor(h_args, a_vx, P_ID);
    const auto vqp = TensorAccessor(h_args, a_vqp, P_Q);
    const auto vqi = TensorAccessor(h_args, a_vqi, P_Q);
    const auto dr = TensorAccessor(h_args, a_dr, P_ID);
    const auto qp = TensorAccessor(h_args, a_qp, P_Q);
    const auto qi = TensorAccessor(h_args, a_qi, P_Q);

    noc_async_read_page(b, dr, s + L_DR);
    noc_async_read_page(b, qp, s + L_QP);
    noc_async_read_page(b, qi, s + L_QI);
    if (u < BW) {
        noc_async_read_page(u, t0n, s + L_T0);
    }
    {
        volatile tt_l1_ptr uint32_t* z = W(L_Z);
        for (uint32_t i = 0; i < QS / 4; ++i) z[i] = 0;
        volatile tt_l1_ptr uint32_t* x = W(L_X);
        for (uint32_t i = 0; i < SLOT / 4; ++i) x[i] = 0;
        volatile tt_l1_ptr uint32_t* o = W(L_QO);
        for (uint32_t i = 0; i < QS / 4; ++i) o[i] = 0;
    }
    noc_async_read_barrier();

    // the draft id: v_tok row (k+1)*BW + b as read (padding zeroed), v_sx row u its float
    volatile tt_l1_ptr uint32_t* d = W(L_DR);
    uint32_t id = d[0];
    if (MF && u == 0) id += 1;
    for (uint32_t i = 0; i < SLOT / 4; ++i) d[i] = 0;
    d[0] = id;
    W(L_X)[0] = u32_to_f32(id);
    if (u < BW) {
        volatile tt_l1_ptr uint32_t* t = W(L_T0);
        const uint32_t v = t[0];
        for (uint32_t i = 0; i < SLOT / 4; ++i) t[i] = 0;
        t[0] = v;
    }

    // the q ids
    volatile tt_l1_ptr uint32_t* qin = W(L_QI);
    volatile tt_l1_ptr uint32_t* qout = W(L_QO);
    if constexpr (MAP) {
        const auto tab = TensorAccessor(h_args, a_tab, P_TB);
        for (uint32_t c = 0; c < C2; ++c) {
            const uint32_t sidx = f32_index(qin[c], N_IDS);
            noc_async_read_page(sidx >> 4, tab, s + L_TB + c * TS);
        }
        noc_async_read_barrier();
        for (uint32_t c = 0; c < C2; ++c) {
            const uint32_t sidx = f32_index(qin[c], N_IDS);
            qout[c] = u32_to_f32(W(L_TB + c * TS)[sidx & 15u]);
        }
    } else {
        for (uint32_t c = 0; c < C2; ++c) qout[c] = qin[c];
    }
    // the probs row as read, padding zeroed
    {
        volatile tt_l1_ptr uint32_t* p = W(L_QP);
        for (uint32_t i = C2; i < QS / 4; ++i) p[i] = 0;
    }

    noc_async_write_page((k + 1) * BW + b, vtok, s + L_DR);
    noc_async_write_page(u, vx, s + L_X);
    noc_async_write_page(u, vqp, s + L_QP);
    noc_async_write_page(u, vqi, s + L_QO);
    if (u < BW) {
        noc_async_write_page(u, vtok, s + L_T0);
        noc_async_write_page(K * BW + u, vx, s + L_Z);
        noc_async_write_page(K * BW + u, vqp, s + L_Z);
        noc_async_write_page(K * BW + u, vqi, s + L_Z);
    }
    noc_async_write_barrier();
}

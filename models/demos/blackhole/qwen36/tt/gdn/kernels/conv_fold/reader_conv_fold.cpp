// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc. / Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_CONV_FOLD (gdn/conv_fold.py): reader of the MTP verify's batch-M gdn_decode_conv_gates with
// conv_multi.build_windows FOLDED IN. Derived from bench/probes/gdn_decode_chain/conv_gates.gdc.cpp (the served
// reader); only the window block differs, the taps and the gate instances are that file's code verbatim.
//
// CONV_MULTI today runs build_windows (rows_reader/rows_writer, 96 cores) to compose the window tensors
// P_1..P_3 ([1, M, C], row u = j*BW + b holds s_b[j+k]) in DRAM, then conv_gates reads P_1..P_3 and x = qkv page
// `inst`. Here the window tiles are composed in L1 straight from the live conv taps and the qkv tile:
//   s_b[i] = lives[i] row b            (i < 4; i >= 1 always, since k >= 1: lives[0] is never read)
//          = qkv row (i - 4)*BW + b    (i >= 4)
//   window slot k-1 (k = 1..3), row u = j*BW + b  <-  s_b[j + k]   (j < T, b < BW),  rows u >= M = T*BW are zero
//   window slot 3 = the qkv page itself with rows >= B zeroed (B = M: exactly the .gdc reader's x masking)
// which is, byte for byte, the page build_windows writes (whole 16-element face rows copied, zero rows elsewhere)
// followed by the .gdc reader's DMA of it. Same CB contents -> same compute, writer and outputs (bit-exact by
// construction; gated in ttsim by bench/probes/sim_mtp_conv_fold.py and on the card by conv_fold.selftest).
// MFJ < T (the must-fail): candidate MFJ's windows read s_b[j + k + 1] (conv_multi.window_map(mf=True)).
//
// Every row read (lives rows < BW <= 32, qkv rows < M <= 32) sits in tile-row 0, so page cc of tile-row 0 is page cc
// of the lives and page cc (+ bt*Wt, bt = 0) of qkv. All NoC reads are FULL pages (no sub-page transfer); the row
// copies are core-side 32-bit L1 copies, as rows_reader.cpp does them; pad rows are NoC zero spans, landing before
// the row copies (disjoint 32 B face rows) and barriered before the push.
//
// Compile args: {K, Ct, Nvt, B, xBt, Wt, GP, AWt, ACOL, BWt, BCOL, Nv, T, BW, MFJ} + accessor args (x, lv1, lv2, lv3,
// tap0..tap3, a, b, dt_bias, neg_exp_A). Runtime args: {inst_start, n_inst, g_n, x, lv1, lv2, lv3, tap0..tap3, a, b,
// dt_bias, neg_exp_A}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_win = 0, cb_tap = 1;
constexpr uint32_t cb_a = 6, cb_b = 7, cb_dtb = 8, cb_nega = 9, cb_gsrc = 13, cb_lv = 14;

void kernel_main() {
    constexpr uint32_t K = get_compile_time_arg_val(0);
    constexpr uint32_t Ct = get_compile_time_arg_val(1);
    constexpr uint32_t Nvt = get_compile_time_arg_val(2);
    constexpr uint32_t B = get_compile_time_arg_val(3);
    constexpr uint32_t xBt = get_compile_time_arg_val(4);
    constexpr uint32_t Wt = get_compile_time_arg_val(5);    // x's width in tiles (Ct when not wider)
    constexpr uint32_t GP = get_compile_time_arg_val(6);    // 1: gather a/b per element from windows
    constexpr uint32_t AWt = get_compile_time_arg_val(7);
    constexpr uint32_t ACOL = get_compile_time_arg_val(8);
    constexpr uint32_t BWt = get_compile_time_arg_val(9);
    constexpr uint32_t BCOL = get_compile_time_arg_val(10);
    constexpr uint32_t NV = get_compile_time_arg_val(11);   // gate width (columns >= NV are padding)
    constexpr uint32_t T = get_compile_time_arg_val(12);    // verify candidates
    constexpr uint32_t BW = get_compile_time_arg_val(13);   // bucket width
    constexpr uint32_t MFJ = get_compile_time_arg_val(14);  // must-fail candidate (>= T: none)
    constexpr uint32_t M = T * BW;
    static_assert(K == 4, "reader is written for K == 4 taps");
    static_assert(T >= 1 && BW >= 1 && M <= 32, "conv fold: every window row must sit in tile-row 0");
    static_assert(xBt == 1 && B == M, "conv fold: x is the [1, M, *] verify input, batch M");

    constexpr auto x_a = TensorAccessorArgs<15>();
    constexpr auto l1_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto l2_a = TensorAccessorArgs<l1_a.next_compile_time_args_offset()>();
    constexpr auto l3_a = TensorAccessorArgs<l2_a.next_compile_time_args_offset()>();
    constexpr auto t0_a = TensorAccessorArgs<l3_a.next_compile_time_args_offset()>();
    constexpr auto t1_a = TensorAccessorArgs<t0_a.next_compile_time_args_offset()>();
    constexpr auto t2_a = TensorAccessorArgs<t1_a.next_compile_time_args_offset()>();
    constexpr auto t3_a = TensorAccessorArgs<t2_a.next_compile_time_args_offset()>();
    constexpr auto a_a = TensorAccessorArgs<t3_a.next_compile_time_args_offset()>();
    constexpr auto b_a = TensorAccessorArgs<a_a.next_compile_time_args_offset()>();
    constexpr auto dtb_a = TensorAccessorArgs<b_a.next_compile_time_args_offset()>();
    constexpr auto nega_a = TensorAccessorArgs<dtb_a.next_compile_time_args_offset()>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t g_n = get_arg_val<uint32_t>(2);
    const uint32_t x_addr = get_arg_val<uint32_t>(3);
    const uint32_t l1_addr = get_arg_val<uint32_t>(4);
    const uint32_t l2_addr = get_arg_val<uint32_t>(5);
    const uint32_t l3_addr = get_arg_val<uint32_t>(6);
    const uint32_t t0_addr = get_arg_val<uint32_t>(7);
    const uint32_t t1_addr = get_arg_val<uint32_t>(8);
    const uint32_t t2_addr = get_arg_val<uint32_t>(9);
    const uint32_t t3_addr = get_arg_val<uint32_t>(10);
    const uint32_t a_addr = get_arg_val<uint32_t>(11);
    const uint32_t b_addr = get_arg_val<uint32_t>(12);
    const uint32_t dtb_addr = get_arg_val<uint32_t>(13);
    const uint32_t nega_addr = get_arg_val<uint32_t>(14);

    const uint32_t tb = get_tile_size(cb_win);
    const uint32_t elem = tb / 1024;
    const uint32_t row_words = 16 * elem / 4;  // one 16-element face row
    const auto x_acc = TensorAccessor(x_a, x_addr, tb);
    const auto l1_acc = TensorAccessor(l1_a, l1_addr, tb);
    const auto l2_acc = TensorAccessor(l2_a, l2_addr, tb);
    const auto l3_acc = TensorAccessor(l3_a, l3_addr, tb);
    const auto t0_acc = TensorAccessor(t0_a, t0_addr, tb);
    const auto t1_acc = TensorAccessor(t1_a, t1_addr, tb);
    const auto t2_acc = TensorAccessor(t2_a, t2_addr, tb);
    const auto t3_acc = TensorAccessor(t3_a, t3_addr, tb);
    const auto a_acc = TensorAccessor(a_a, a_addr, tb);
    const auto b_acc = TensorAccessor(b_a, b_addr, tb);
    const auto dtb_acc = TensorAccessor(dtb_a, dtb_addr, tb);
    const auto nega_acc = TensorAccessor(nega_a, nega_addr, tb);

    Noc noc;

    auto zero_words = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };
    // rows [m, 32) of the tile at `base` as NoC zero spans (the .fix reader's x-mask spans, in elements):
    //   m < 16 : [16m, 256) face 0 rows m..15, [256+16m, 1024) face 1 rows m..15 + faces 2,3
    //   m >= 16: [512+16k, 768), [768+16k, 1024), k = m-16   faces 2 and 3 rows k..15
    auto zero_rows_from = [&](uint32_t base, uint32_t m) {
        if (m >= 32) {
            return;
        }
        uint32_t s0, e0, s1;
        if (m < 16) {
            s0 = 16 * m;
            e0 = 256;
            s1 = 256 + 16 * m;
        } else {
            s0 = 512 + 16 * (m - 16);
            e0 = 768;
            s1 = 768 + 16 * (m - 16);
        }
        noc.async_write_zeros(CoreLocalMem<uint32_t>(base + s0 * elem), (e0 - s0) * elem);
        noc.async_write_zeros(CoreLocalMem<uint32_t>(base + s1 * elem), (1024 - s1) * elem);
    };
    // copy row sr of the tile at `src` to row dr of the tile at `dst` (both 16-element face rows of the row)
    auto copy_row = [&](uint32_t src, uint32_t sr, uint32_t dst, uint32_t dr) {
        const uint32_t so = (((sr / 16) * 2) * 256 + (sr % 16) * 16) * elem;
        const uint32_t dof = (((dr / 16) * 2) * 256 + (dr % 16) * 16) * elem;
        volatile tt_l1_ptr uint32_t* s0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + so);
        volatile tt_l1_ptr uint32_t* s1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(src + so + 256 * elem);
        volatile tt_l1_ptr uint32_t* d0 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + dof);
        volatile tt_l1_ptr uint32_t* d1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(dst + dof + 256 * elem);
        for (uint32_t w = 0; w < row_words; ++w) {
            d0[w] = s0[w];
            d1[w] = s1[w];
        }
    };

    // private scratch for the live taps lives[1..3] page cc (never pushed)
    CircularBuffer lv_cb(cb_lv);
    lv_cb.reserve_back(3);
    const uint32_t lv_base = lv_cb.get_write_ptr();

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t cc = inst % Ct;  // inst < Ct: one tile-row (M <= 32)

        // window = [W_1, W_2, W_3, x]
        {
            CircularBuffer cb(cb_win);
            cb.reserve_back(K);
            const uint32_t base = cb.get_write_ptr();
            const uint32_t xbase = base + 3 * tb;
            noc.async_read(l1_acc, lv_cb, tb, {.page_id = cc}, {.offset_bytes = 0 * tb});
            noc.async_read(l2_acc, lv_cb, tb, {.page_id = cc}, {.offset_bytes = 1 * tb});
            noc.async_read(l3_acc, lv_cb, tb, {.page_id = cc}, {.offset_bytes = 2 * tb});
            noc.async_read(x_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 3 * tb});
            // pad rows u >= M of W_1..W_3 (build_windows' zero rows); disjoint from the reads in flight
            if constexpr (M < 32) {
                for (uint32_t s = 0; s < 3; ++s) {
                    zero_rows_from(base + s * tb, M);
                }
            }
            noc.async_read_barrier();
            if constexpr (M < 32) {
                noc.write_zeros_l1_barrier();
            }
            asm volatile("" ::: "memory");
            for (uint32_t k = 1; k <= 3; ++k) {
                const uint32_t dst = base + (k - 1) * tb;
                for (uint32_t j = 0; j < T; ++j) {
                    const uint32_t i = j + k + ((j == MFJ) ? 1u : 0u);
                    const uint32_t src = (i < 4) ? lv_base + (i - 1) * tb : xbase;
                    const uint32_t sr0 = (i < 4) ? 0u : (i - 4) * BW;
                    for (uint32_t b = 0; b < BW; ++b) {
                        copy_row(src, sr0 + b, dst, j * BW + b);
                    }
                }
            }
            asm volatile("" ::: "memory");
            // x rows at or beyond the active batch enter the shift register as zeros (the .gdc reader's mask, B = M);
            // issued after the row copies that read x rows < M (disjoint bytes either way), barriered before the push
            if constexpr (B < 32) {
                zero_rows_from(xbase, B);
                noc.write_zeros_l1_barrier();
            }
            cb.push_back(K);
        }
        // taps: page cc of each [1,1,C] tap tensor
        {
            CircularBuffer cb(cb_tap);
            cb.reserve_back(K);
            noc.async_read(t0_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 0 * tb});
            noc.async_read(t1_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 1 * tb});
            noc.async_read(t2_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 2 * tb});
            noc.async_read(t3_acc, cb, tb, {.page_id = cc}, {.offset_bytes = 3 * tb});
            noc.async_read_barrier();
            cb.push_back(K);
        }
    }

    // Gather gate tile t (output cols h in [32t, 32t+32) ∩ [0,Nv)) from a window that starts
    // at element column `col0` of a tensor `wt` tiles wide: the source columns col0+h span one
    // or two source tiles. Full-page reads into the scratch CB, then per-element L1 copies
    // (16-bit at bf16, 32-bit at fp32) into the zeroed target tile.
    auto gather_gate = [&](const auto& acc, uint32_t cb_id, uint32_t wt, uint32_t col0, uint32_t bt_g, uint32_t t) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t dst = cb.get_write_ptr();
        zero_words(dst, tb / 4);
        const uint32_t h0 = t * 32;
        const uint32_t h1 = (h0 + 32 < NV) ? h0 + 32 : NV;  // exclusive: only real heads are gathered
        const uint32_t c0 = col0 + h0;
        const uint32_t c1 = col0 + h1 - 1;
        const uint32_t p0 = c0 / 32;
        const uint32_t p1 = c1 / 32;
        CircularBuffer scb(cb_gsrc);
        scb.reserve_back(2);
        const uint32_t sbase = scb.get_write_ptr();
        for (uint32_t p = p0; p <= p1; p++) {
            noc.async_read(acc, scb, tb, {.page_id = bt_g * wt + p}, {.offset_bytes = (p - p0) * tb});
        }
        noc.async_read_barrier();
        asm volatile("" ::: "memory");
        for (uint32_t h = h0; h < h1; h++) {
            const uint32_t c = col0 + h;
            const uint32_t sp = sbase + (c / 32 - p0) * tb;
            const uint32_t sc = c % 32;
            const uint32_t dc = h - h0;
            for (uint32_t r = 0; r < 32; r++) {
                const uint32_t se = ((r / 16) * 2 + (sc / 16)) * 256 + (r % 16) * 16 + (sc % 16);
                const uint32_t de = ((r / 16) * 2 + (dc / 16)) * 256 + (r % 16) * 16 + (dc % 16);
                if (elem == 2) {
                    auto sv = CoreLocalMem<volatile uint16_t>(sp + se * 2);
                    auto dv = CoreLocalMem<volatile uint16_t>(dst + de * 2);
                    dv[0] = sv[0];
                } else {
                    auto sv = CoreLocalMem<volatile uint32_t>(sp + se * 4);
                    auto dv = CoreLocalMem<volatile uint32_t>(dst + de * 4);
                    dv[0] = sv[0];
                }
            }
        }
        asm volatile("" ::: "memory");
        scb.push_back(2);
        scb.pop_front(2);
        cb.push_back(1);
    };

    // gates: one tile per instance gi = bt_g*Nvt + t
    for (uint32_t gi = 0; gi < g_n; ++gi) {
        const uint32_t t = gi % Nvt;
        const uint32_t bt_g = gi / Nvt;
        auto one = [&](const auto& acc, uint32_t cb_id, uint32_t page) {
            CircularBuffer cb(cb_id);
            cb.reserve_back(1);
            noc.async_read(acc, cb, tb, {.page_id = page}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        };
        if constexpr (GP && (ACOL % 32 == 0) && (BCOL % 32 == 0)) {
            // QWEN36_GDN_DECODE_CHAIN=direct: tile-aligned windows -> full-page reads (see the header)
            one(a_acc, cb_a, bt_g * AWt + ACOL / 32 + t);
            one(b_acc, cb_b, bt_g * BWt + BCOL / 32 + t);
        } else if constexpr (GP) {
            gather_gate(a_acc, cb_a, AWt, ACOL, bt_g, t);
            gather_gate(b_acc, cb_b, BWt, BCOL, bt_g, t);
        } else {
            one(a_acc, cb_a, gi);
            one(b_acc, cb_b, gi);
        }
        one(dtb_acc, cb_dtb, t);
        one(nega_acc, cb_nega, t);
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_GDN_STATE_ONCE reader (docs/MTP-BATCH.md section 16). Device 2.0 API, as the production reader.
//
// The gather helpers are the served packed-mode reader's (the arm's c90k overlay,
// bench/probes/gdn_reader_zero/reader.fix.cpp: full-page DMA, then row r moved to row 0 and the rest of the page
// zeroed by a NoC loopback from MEM_ZEROS_BASE), so every q/k/v/beta/g tile lands bit-identical to the production
// op's. What changes:
//   - MODE 0 (verify) reads the ONE batch-M conv_gates output of QWEN36_MTP_CONV_MULTI: conv_all TILE [1, M, C] and
//     beta_all / g_all TILE [1, M, H], row u = t*BW + b (candidate-major, M = T*BW <= 32); token t of row b gathers
//     row u, so the per-candidate split programs are not needed. The state of head instance bh = b*H + h is read
//     ONCE (rows < BW of the persistent [B*, H, K, V] state), then tokens 0..T-1 follow.
//     For the commit, each token's gathered k row, v row, beta and g of head (b, h) are also written as ONE
//     ROW_MAJOR stick of CIN_W elements [k (Dk) | v (Dv) | beta, g, 0 ...] to cin page u*H + h (DRAM). The commit
//     then reads CIN_W elements per token instead of re-gathering whole 32-row tiles, and conv_all can stay in L1.
//   - MODE 1 (commit): the idx page (ROW_MAJOR uint32 [1, 32], the QWEN36_MTP_STATE_SLOT page: entry b = the
//     accepted verify depth of row b; >= T means keep the row) gives n = idx[b] + 1 tokens (0 when idx[b] >= T).
//     n goes to the compute (cb_cnt) and to the writer (cb_wcnt) for every instance; for n > 0 the state and the
//     k/v/beta/g tiles of tokens 0..n-1 follow, rebuilt from the cin sticks exactly as the gather left them: row 0
//     = the stick's values, every other element of the k/v tiles zero, the scalar at [0,0] (the compute reads only
//     [0,0] of beta/g: scalar broadcasts). No q: the state update does not read it. n == 0 reads nothing.
//     MUSTFAIL=1: row BW-1 takes one token too many (n+1; n-1 when n == T, so it always differs).
//   - SO_T0=1 (QWEN36_MTP_GDN_SO_T0): the verify also writes each in-bucket row's state after token 0 over its own
//     rec_state row (token 0 is always committed: the slot page is one-hot for b < BW). MODE 0: the reader publishes
//     per instance the token after which the state leaves (WT = 0; T0_MUSTFAIL: 1 for row BW-1, the wrong depth) to
//     the compute (cb_cnt) and the writer (cb_wcnt), BEFORE the instance's state read; the instance's base state is
//     read (barrier) and pushed before the compute can produce token 0, so the in-place write of row bh follows the
//     read of row bh, and no other instance reads row bh. MODE 1: the commit starts from that updated row and feeds
//     tokens 1 .. n0-1 (n = n0 - 1 of them, n0 the count above, MUSTFAIL applied to n0); n == 0 reads nothing.
//
// SO_SPLITREAD (QWEN36_MTP_GDN_SO_SPLITREAD, a define prepended by state_once; MODE 0 only): the per-instance state
//   read and the ones tile move to the writer (BRISC, the other NoC), so this RISC only gathers; the compute's first
//   token no longer waits for ones fill (2.5 us) + state read (7.6 us) + the token-0 gathers in series
//   (op-anatomy-gdn-state-once-20261006T045611Z: compute starts at 18.5 us of a ~112 us window).
//
// SO_VSPLIT (QWEN36_MTP_GDN_SO_VSPLIT, MODE 0 with SO_QKH + SO_SPLITREAD; a define = P prepended by state_once): each
//   head instance is split over P cores by value-column tiles; this core runs part SO_VPART (last runtime arg) of
//   ONE instance: q and k as today, only v tiles [part*VP, (part+1)*VP) (VP = Vt / P). Its cin stick bytes are a
//   disjoint range of the instance's cin page: part 0 [k | v slice 0], part p [v slice p], part P-1 also the tail
//   [beta, g, 0...]; together exactly today's stick (all offsets 64 B aligned at Dk = Dv = 128).
//
// Compile args: {Kt, Vt, H, Ct, QOT, KOT, VOT, RF, NVT, T, BW, MODE, MUSTFAIL, IDX_PAGE, CIN_PAGE, SO_T0, T0_MUSTFAIL}
//               + TensorAccessorArgs(conv_all, beta_all, g_all, state, idx, cin) (unused slots repeat the state's).
// Runtime args: {bh_start, n_inst, conv_addr, beta_addr, g_addr, state_addr, idx_addr, cin_addr} (unused: 0).

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6, cb_scratch = 27;
constexpr uint32_t cb_cnt = 31, cb_wcnt = 32, cb_stick = 33;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t Ct = get_compile_time_arg_val(3);
    constexpr uint32_t QOT = get_compile_time_arg_val(4);
    constexpr uint32_t KOT = get_compile_time_arg_val(5);
    constexpr uint32_t VOT = get_compile_time_arg_val(6);
    constexpr uint32_t RF = get_compile_time_arg_val(7);
    constexpr uint32_t NVT = get_compile_time_arg_val(8);
    constexpr uint32_t T = get_compile_time_arg_val(9);
    constexpr uint32_t BW = get_compile_time_arg_val(10);
    constexpr uint32_t MODE = get_compile_time_arg_val(11);
    constexpr uint32_t MUSTFAIL = get_compile_time_arg_val(12);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(13);
    constexpr uint32_t CIN_PAGE = get_compile_time_arg_val(14);
    constexpr uint32_t SO_T0 = get_compile_time_arg_val(15);
    constexpr uint32_t T0_MUSTFAIL = get_compile_time_arg_val(16);
    constexpr bool VERIFY = MODE == 0;
    static_assert(SO_T0 == 0 || T >= 2, "SO_T0 needs T >= 2");

    constexpr auto x_a = TensorAccessorArgs<17>();
    constexpr auto beta_a = TensorAccessorArgs<x_a.next_compile_time_args_offset()>();
    constexpr auto g_a = TensorAccessorArgs<beta_a.next_compile_time_args_offset()>();
    constexpr auto s0_a = TensorAccessorArgs<g_a.next_compile_time_args_offset()>();
    constexpr auto ix_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto cin_a = TensorAccessorArgs<ix_a.next_compile_time_args_offset()>();

    const uint32_t bh_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);

    const uint32_t tb_io = get_tile_size(cb_k);  // conv/beta/g/state share one dtype (checked by the wrapper)
    const uint32_t elem = tb_io / 1024;
    const auto x_acc = TensorAccessor(x_a, get_arg_val<uint32_t>(2), tb_io);
    const auto beta_acc = TensorAccessor(beta_a, get_arg_val<uint32_t>(3), tb_io);
    const auto g_acc = TensorAccessor(g_a, get_arg_val<uint32_t>(4), tb_io);
    const auto s0_acc = TensorAccessor(s0_a, get_arg_val<uint32_t>(5), tb_io);
    const auto cin_acc = TensorAccessor(cin_a, get_arg_val<uint32_t>(7), CIN_PAGE);

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t Dk = Kt * 32, Dv = Vt * 32;
#if defined(SO_VSPLIT)
    static_assert(VERIFY, "SO_VSPLIT is verify-only");
    constexpr uint32_t VP_ = SO_VSPLIT;
    static_assert(VP_ >= 2 && Vt % VP_ == 0, "SO_VSPLIT: P must divide Vt");
    constexpr uint32_t VP = Vt / VP_;
    const uint32_t vpart = get_arg_val<uint32_t>(8);
#else
    constexpr uint32_t VP_ = 1;
    constexpr uint32_t VP = Vt;
    constexpr uint32_t vpart = 0;
#endif

    Noc noc;

    // The private stick page (reserved once, never pushed: the reader is its only user).
    CircularBuffer stcb(cb_stick);
    stcb.reserve_back(1);
    const uint32_t stick = stcb.get_write_ptr();

    // Copy row 0 of n_tiles TILE pages (faces 0 and 1: 16 + 16 elements each) to/from a ROW_MAJOR stick.
    auto row0_to_stick = [&](uint32_t base, uint32_t n_tiles, uint32_t dst) {
        const uint32_t cw = 16 * elem / 4;
        asm volatile("" ::: "memory");
        for (uint32_t t = 0; t < n_tiles; t++) {
            auto s0 = CoreLocalMem<volatile uint32_t>(base + t * tb_io);
            auto s1 = CoreLocalMem<volatile uint32_t>(base + t * tb_io + 256 * elem);
            auto d0 = CoreLocalMem<volatile uint32_t>(dst + (32 * t) * elem);
            auto d1 = CoreLocalMem<volatile uint32_t>(dst + (32 * t + 16) * elem);
            for (uint32_t w = 0; w < cw; w++) {
                d0[w] = s0[w];
                d1[w] = s1[w];
            }
        }
        asm volatile("" ::: "memory");
    };

    // reader.fix.cpp gather_row (c90k): full-page DMA into the target CB, save row r's two face chunks, zero the
    // pages by NoC loopback, put the chunks back as row 0. MODE 0: row 0 also goes to the stick at stick_off.
    auto gather_row = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t first_page, uint32_t r,
                          uint32_t stick_off) {
        const uint32_t frow = r % 16;
        const uint32_t fhalf = r / 16;
        const uint32_t src_e0 = (fhalf * 2 + 0) * 256 + frow * 16;
        const uint32_t src_e1 = (fhalf * 2 + 1) * 256 + frow * 16;
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_read(acc, cb, tb_io, {.page_id = first_page + t}, {.offset_bytes = t * tb_io});
        }
        noc.async_read_barrier();
        constexpr uint32_t kMaxT = Kt > Vt ? Kt : Vt, kCw = 16;
        uint32_t saved[kMaxT][2][kCw];
        const uint32_t cw = 16 * elem / 4;
        asm volatile("" ::: "memory");
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t p = base + t * tb_io;
            auto s0 = CoreLocalMem<volatile uint32_t>(p + src_e0 * elem);
            auto s1 = CoreLocalMem<volatile uint32_t>(p + src_e1 * elem);
            for (uint32_t w = 0; w < cw; w++) {
                saved[t][0][w] = s0[w];
                saved[t][1][w] = s1[w];
            }
        }
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(base + t * tb_io), tb_io);
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t t = 0; t < n_tiles; t++) {
            const uint32_t p = base + t * tb_io;
            auto d0 = CoreLocalMem<volatile uint32_t>(p);
            auto d1 = CoreLocalMem<volatile uint32_t>(p + 256 * elem);
            for (uint32_t w = 0; w < cw; w++) {
                d0[w] = saved[t][0][w];
                d1[w] = saved[t][1][w];
            }
        }
        asm volatile("" ::: "memory");
        if (stick_off != 0xFFFFFFFFu) {
            row0_to_stick(base, n_tiles, stick + stick_off * elem);
        }
        cb.push_back(n_tiles);
    };

    // reader.fix.cpp gather_scalar: the head scalar at (row r, col cc) of one page, into [0,0] of the same page.
    // Returns the word written to [0,0] (its low element is the scalar).
    auto gather_scalar = [&](const auto& acc, uint32_t cb_id, uint32_t page, uint32_t r, uint32_t cc) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        const uint32_t eoff = ((r / 16) * 2 + (cc / 16)) * 256 + (r % 16) * 16 + (cc % 16);
        const uint32_t byte = base + eoff * elem;
        asm volatile("" ::: "memory");
        auto s = CoreLocalMem<volatile uint32_t>(byte & ~3u);
        uint32_t w = s[0];
        if (elem == 2 && (byte & 2u)) {
            w >>= 16;
        }
        auto d = CoreLocalMem<volatile uint32_t>(base);
        d[0] = w;
        asm volatile("" ::: "memory");
        cb.push_back(1);
        return w;
    };

    // MODE 1: rebuild a gathered row tile set from the stick: zero the pages (NoC loopback), row 0 from the stick.
    auto tiles_from_stick = [&](uint32_t cb_id, uint32_t n_tiles, uint32_t stick_off) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(base + t * tb_io), tb_io);
        }
        noc.write_zeros_l1_barrier();
        const uint32_t cw = 16 * elem / 4;
        const uint32_t src = stick + stick_off * elem;
        asm volatile("" ::: "memory");
        for (uint32_t t = 0; t < n_tiles; t++) {
            auto s0 = CoreLocalMem<volatile uint32_t>(src + (32 * t) * elem);
            auto s1 = CoreLocalMem<volatile uint32_t>(src + (32 * t + 16) * elem);
            auto d0 = CoreLocalMem<volatile uint32_t>(base + t * tb_io);
            auto d1 = CoreLocalMem<volatile uint32_t>(base + t * tb_io + 256 * elem);
            for (uint32_t w = 0; w < cw; w++) {
                d0[w] = s0[w];
                d1[w] = s1[w];
            }
        }
        asm volatile("" ::: "memory");
        cb.push_back(n_tiles);
    };

    // MODE 1: a scalar tile with the stick's scalar at [0,0] (the rest zero; the compute reads only [0,0]).
    auto scalar_from_stick = [&](uint32_t cb_id, uint32_t stick_off) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        noc.async_write_zeros(CoreLocalMem<uint32_t>(base), tb_io);
        noc.write_zeros_l1_barrier();
        asm volatile("" ::: "memory");
        const uint32_t byte = stick + stick_off * elem;
        uint32_t w = CoreLocalMem<volatile uint32_t>(byte & ~3u)[0];
        if (elem == 2) {
            w = (byte & 2u) ? (w >> 16) : (w & 0xFFFFu);
        }
        auto d = CoreLocalMem<volatile uint32_t>(base);
        d[0] = w;
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };

#if defined(SO_QKH)
    static_assert(VERIFY, "SO_QKH is verify-only");
    static_assert(T >= 2 && T * BW <= 32, "SO_QKH: the T rows of one instance live in ONE batch-M row page");
    // Element offset of (row r, col c) inside a 32x32 TILE page (faces 16x16, row-major faces).
    auto tile_eoff = [](uint32_t r, uint32_t c) -> uint32_t {
        return ((r / 16) * 2 + (c / 16)) * 256 + (r % 16) * 16 + (c % 16);
    };
    // SO_QKH gather: zero n_tiles pages of cb_id by NoC loopback, then read ONLY the rows u_t = t*BW + b (t < T) of
    // pages first_page.. into the SAME rows of the zeroed pages (2 face chunks of 16 elements per row and tile). Every
    // other row is +0, exactly as the per-token gather leaves rows 1..31. Not pushed (the caller pushes).
    auto gather_rows_inplace = [&](const auto& acc, uint32_t cb_id, uint32_t n_tiles, uint32_t first_page,
                                   uint32_t b) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n_tiles);
        const uint32_t base = cb.get_write_ptr();
        for (uint32_t t = 0; t < n_tiles; t++) {
            noc.async_write_zeros(CoreLocalMem<uint32_t>(base + t * tb_io), tb_io);
        }
        noc.write_zeros_l1_barrier();
        for (uint32_t t = 0; t < n_tiles; t++) {
            for (uint32_t tok = 0; tok < T; ++tok) {
                const uint32_t r = tok * BW + b;
                for (uint32_t f = 0; f < 2; ++f) {
                    const uint32_t off = tile_eoff(r, 16 * f) * elem;
                    noc.async_read(acc, cb, 16 * elem, {.page_id = first_page + t, .offset_bytes = off},
                                   {.offset_bytes = t * tb_io + off});
                }
            }
        }
        noc.async_read_barrier();
        return base;
    };
    // Row r of n_tiles TILE pages at base -> a ROW_MAJOR stick (row0_to_stick at row r).
    auto rowr_to_stick = [&](uint32_t base, uint32_t n_tiles, uint32_t r, uint32_t dst) {
        const uint32_t cw = 16 * elem / 4;
        const uint32_t e0 = tile_eoff(r, 0), e1 = tile_eoff(r, 16);
        asm volatile("" ::: "memory");
        for (uint32_t t = 0; t < n_tiles; t++) {
            auto s0 = CoreLocalMem<volatile uint32_t>(base + t * tb_io + e0 * elem);
            auto s1 = CoreLocalMem<volatile uint32_t>(base + t * tb_io + e1 * elem);
            auto d0 = CoreLocalMem<volatile uint32_t>(dst + (32 * t) * elem);
            auto d1 = CoreLocalMem<volatile uint32_t>(dst + (32 * t + 16) * elem);
            for (uint32_t w = 0; w < cw; w++) {
                d0[w] = s0[w];
                d1[w] = s1[w];
            }
        }
        asm volatile("" ::: "memory");
    };
    // The word holding element (r, c) of the page at base, shifted so its low element is that element (as
    // gather_scalar returns it).
    auto word_at = [&](uint32_t base, uint32_t r, uint32_t c) -> uint32_t {
        const uint32_t byte = base + tile_eoff(r, c) * elem;
        uint32_t w = CoreLocalMem<volatile uint32_t>(byte & ~3u)[0];
        if (elem == 2 && (byte & 2u)) {
            w >>= 16;
        }
        return w;
    };
    // A zeroed scalar page with the scalar word's low element at (r, 0): the beta COLUMN tile. The compute multiplies
    // delta by it with a column broadcast, so row r keeps delta * beta (the product the scalar broadcast gave) and
    // every other row becomes 0: the rank-1 update then sees only token t's row, as the per-token gather's row 0.
    auto push_col_scalar = [&](uint32_t cb_id, uint32_t r, uint32_t w) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        noc.async_write_zeros(CoreLocalMem<uint32_t>(base), tb_io);
        noc.write_zeros_l1_barrier();
        asm volatile("" ::: "memory");
        // column 0: element index even, the low half (a named pointer: `CoreLocalMem<..>(x)[0] = v;` parses as a
        // declaration of an array x)
        auto dst = CoreLocalMem<volatile uint32_t>(base + tile_eoff(r, 0) * elem);
        dst[0] = (elem == 2) ? (w & 0xFFFFu) : w;
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };
    // A zeroed scalar page with the scalar at [0,0] (the g tile: only [0,0] is read, a scalar broadcast).
    auto push_scalar00 = [&](uint32_t cb_id, uint32_t w) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        noc.async_write_zeros(CoreLocalMem<uint32_t>(base), tb_io);
        noc.write_zeros_l1_barrier();
        asm volatile("" ::: "memory");
        auto dst = CoreLocalMem<volatile uint32_t>(base);
        dst[0] = (elem == 2) ? (w & 0xFFFFu) : w;
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };
    // One io page of a [1, M, H] scalar tensor into cb_id's write area (NOT pushed: read-only scratch for word_at).
    auto peek_page = [&](const auto& acc, uint32_t cb_id, uint32_t page) -> uint32_t {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        noc.async_read(acc, cb, tb_io, {.page_id = page}, {.offset_bytes = 0});
        noc.async_read_barrier();
        return base;
    };
#endif

    auto push_count = [&](uint32_t cb_id, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        auto p = CoreLocalMem<volatile uint32_t>(cb.get_write_ptr());
        p[0] = n;
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };

#if defined(SO_VSPLIT) && !(defined(SO_QKH) && defined(SO_SPLITREAD))
#error "SO_VSPLIT is built on the served verify reader: SO_QKH + SO_SPLITREAD"
#endif
#if defined(SO_SPLITREAD)
    static_assert(VERIFY, "SO_SPLITREAD is verify-only");
#else
    {
        CircularBuffer cb(cb_ones);
        cb.reserve_back(1);
        const uint32_t base = cb.get_write_ptr();
        auto ptr = CoreLocalMem<uint32_t>(base);
        for (uint32_t w = 0; w < 1024; w++) {
            ptr[w] = 0x3F800000u;
        }
        cb.push_back(1);
    }
#endif

    // MODE 1: the idx page, read once into this kernel's private scratch page (reserved, never pushed).
    uint32_t idx_l1 = 0;
    if constexpr (!VERIFY) {
        const auto ix_acc = TensorAccessor(ix_a, get_arg_val<uint32_t>(6), IDX_PAGE);
        CircularBuffer scb(cb_scratch);
        scb.reserve_back(1);
        idx_l1 = scb.get_write_ptr();
        noc_async_read_page(0, ix_acc, idx_l1);
        noc_async_read_barrier();
    }

    // stick layout (elements): [k: 0 .. Dk) [v: Dk .. Dk+Dv) [beta: Dk+Dv] [g: Dk+Dv+1] [0 ...]
    constexpr uint32_t ST_K = 0, ST_V = Dk, ST_B = Dk + Dv, ST_G = Dk + Dv + 1;

    for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;
        const uint32_t hq = h / RF;

        uint32_t n = T;
        uint32_t tok0 = 0;  // first verify token fed (SO_T0 commit: 1, token 0 is already in rec_state)
        if constexpr (VERIFY) {
            if constexpr (SO_T0 != 0) {
                // the token after which the state is written over rec_state row b (the writer and compute agree)
                const uint32_t wt = (T0_MUSTFAIL != 0 && b == BW - 1) ? 1u : 0u;
                push_count(cb_cnt, wt);
                push_count(cb_wcnt, wt);
            }
        } else {
            asm volatile("" ::: "memory");
            const uint32_t s = CoreLocalMem<volatile uint32_t>(idx_l1)[b];
            n = s < T ? s + 1 : 0;
            if constexpr (MUSTFAIL != 0) {
                if (b == BW - 1) {
                    n = n < T ? n + 1 : n - 1;
                }
            }
            if constexpr (SO_T0 != 0) {
                n = n > 0 ? n - 1 : 0;
                tok0 = 1;
            }
            push_count(cb_cnt, n);
            push_count(cb_wcnt, n);
            if (n == 0) {
                continue;  // keep this row: compute and writer skip it too
            }
        }

        // The pre-verify state, once per instance (SO_SPLITREAD: the writer reads it).
#if !defined(SO_SPLITREAD)
        {
            CircularBuffer cbs(cb_state);
            cbs.reserve_back(kv);
            const uint32_t base_page = bh * kv;
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});
            }
            noc.async_read_barrier();
            cbs.push_back(kv);
        }
#endif

#if defined(SO_QKH)
        // SO_QKH (QWEN36_MTP_GDN_SO_QKH): q, k and v of ALL T tokens of this instance at once, each token's row at its
        // own row u_t = t*BW + b of the zeroed q/k/v pages (M = T*BW <= 32: one row page). The compute normalises q and
        // k ONCE per instance (row-independent tile ops), reads token t's rows in place and the writer takes o from
        // row u_t. The commit's sticks are the per-token gather's bytes: [k row u_t | v row u_t | beta_t, g_t, 0...].
        {
            const uint32_t qb = gather_rows_inplace(x_acc, cb_q, Kt, QOT + hq * Kt, b);
            const uint32_t kb = gather_rows_inplace(x_acc, cb_k, Kt, KOT + hq * Kt, b);
            const uint32_t vb = gather_rows_inplace(x_acc, cb_v, VP, VOT + h * Vt + vpart * VP, b);
            (void)qb;
            const uint32_t spage = h / 32;  // (u / 32) * NVT + h / 32 with u < 32
            const uint32_t bpage = peek_page(beta_acc, cb_beta, spage);
            uint32_t wbs[T], wgs[T];
            for (uint32_t tok = 0; tok < T; ++tok) {
                wbs[tok] = word_at(bpage, tok * BW + b, h % 32);
            }
            const uint32_t gpage = peek_page(g_acc, cb_g, spage);
            for (uint32_t tok = 0; tok < T; ++tok) {
                wgs[tok] = word_at(gpage, tok * BW + b, h % 32);
            }
            for (uint32_t tok = 0; tok < T; ++tok) {
                const uint32_t u = tok * BW + b;
                if (vpart == 0) {
                    rowr_to_stick(kb, Kt, u, stick + ST_K * elem);
                }
                rowr_to_stick(vb, VP, u, stick + (ST_V + vpart * VP * 32) * elem);
                auto tail = CoreLocalMem<volatile uint32_t>(stick + ST_B * elem);
                const uint32_t tail_words = (CIN_PAGE - ST_B * elem) / 4;
                for (uint32_t w = 0; w < tail_words; w++) {
                    tail[w] = 0u;
                }
                if (elem == 2) {
                    tail[0] = (wbs[tok] & 0xFFFFu) | (wgs[tok] << 16);
                } else {
                    tail[0] = wbs[tok];
                    tail[1] = wgs[tok];
                }
                asm volatile("" ::: "memory");
                if constexpr (VP_ == 1) {
                    noc.async_write(CoreLocalMem<uint32_t>(stick), cin_acc, CIN_PAGE, {}, {.page_id = u * H + h});
                } else {
                    // this part's disjoint byte range of the instance's cin page (see SO_VSPLIT above)
                    const uint32_t lo = (vpart == 0) ? 0u : (ST_V + vpart * VP * 32) * elem;
                    const uint32_t hi = (vpart + 1 == VP_) ? CIN_PAGE : (ST_V + (vpart + 1) * VP * 32) * elem;
                    noc.async_write(CoreLocalMem<uint32_t>(stick + lo), cin_acc, hi - lo, {},
                                    {.page_id = u * H + h, .offset_bytes = lo});
                }
                noc.async_write_barrier();
            }
            CircularBuffer(cb_q).push_back(Kt);
            CircularBuffer(cb_k).push_back(Kt);
            CircularBuffer(cb_v).push_back(VP);
            for (uint32_t tok = 0; tok < T; ++tok) {
#if defined(SO_QKH_MUSTFAIL)
                // must-fail: the last row's beta lands on the NEXT token's row (token t updates with the wrong row)
                const uint32_t br = (b == BW - 1) ? ((tok + 1) % T) * BW + b : tok * BW + b;
#else
                const uint32_t br = tok * BW + b;
#endif
                push_scalar00(cb_g, wgs[tok]);
                push_col_scalar(cb_beta, br, wbs[tok]);
            }
            continue;
        }
#endif

        for (uint32_t tok = tok0; tok < tok0 + n; ++tok) {
            const uint32_t u = tok * BW + b;  // candidate-major row of the batch-M conv_gates output
            if constexpr (VERIFY) {
                const uint32_t row_page0 = (u / 32) * Ct;
                const uint32_t r = u % 32;
                gather_row(x_acc, cb_q, Kt, row_page0 + QOT + hq * Kt, r, 0xFFFFFFFFu);
                gather_row(x_acc, cb_k, Kt, row_page0 + KOT + hq * Kt, r, ST_K);
                gather_row(x_acc, cb_v, Vt, row_page0 + VOT + h * Vt, r, ST_V);
                const uint32_t wb = gather_scalar(beta_acc, cb_beta, (u / 32) * NVT + h / 32, r, h % 32);
                const uint32_t wg = gather_scalar(g_acc, cb_g, (u / 32) * NVT + h / 32, r, h % 32);
                // the scalars and the zero tail of the stick (full words; the stick page is private)
                auto tail = CoreLocalMem<volatile uint32_t>(stick + ST_B * elem);
                const uint32_t tail_words = (CIN_PAGE - ST_B * elem) / 4;
                for (uint32_t w = 0; w < tail_words; w++) {
                    tail[w] = 0u;
                }
                if (elem == 2) {
                    tail[0] = (wb & 0xFFFFu) | (wg << 16);
                } else {
                    tail[0] = wb;
                    tail[1] = wg;
                }
                asm volatile("" ::: "memory");
                noc.async_write(CoreLocalMem<uint32_t>(stick), cin_acc, CIN_PAGE, {}, {.page_id = u * H + h});
                noc.async_write_barrier();
            } else {
                noc_async_read_page(u * H + h, cin_acc, stick);
                noc_async_read_barrier();
                tiles_from_stick(cb_k, Kt, ST_K);
                tiles_from_stick(cb_v, Vt, ST_V);
                scalar_from_stick(cb_beta, ST_B);
                scalar_from_stick(cb_g, ST_G);
            }
        }
    }
}

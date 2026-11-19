// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_GDN_STATE_ONCE writer (docs/MTP-BATCH.md section 16). Device 2.0 API, as the production writer.
//
// MODE 0 (verify): o only. The production writer's o-stick staging (row 0 of the Vt io tiles -> one ROW_MAJOR
//   stick of Dv elements), per token, into ONE ROW_MAJOR [T*BW, 1, H, V] tensor: token t of head instance bh is
//   page t*TOK + bh (TOK = BW*H), i.e. row u = t*BW + b, the order the sequential verify's concat gives and
//   gdn_decode_norm_gate reads. No state is written (the sequential verify wrote one per token).
//   SO_T0=1 (QWEN36_MTP_GDN_SO_T0): per instance the reader's WT (cb_wcnt); right after token WT's o, the KV state
//   pages of cb_sout go over rec_state pages [bh*kv, bh*kv + kv) (the commit writer's write, one token earlier).
// MODE 1 (commit): per instance the reader's token count n (cb_wcnt); n > 0: the KV committed state pages from
//   cb_sout straight over the live state's pages [bh*kv, bh*kv + kv) (full pages, as the production writer and the
//   slot commit write them). n == 0: the row is not touched.
//
// SO_QKH (QWEN36_MTP_GDN_SO_QKH, MODE 0; defines SO_QKH_BW / SO_QKH_H prepended by state_once): token tok's o is row
//   u = tok*BW + b of the o tiles (b = bh / H), not row 0: the compute keeps every token's q at its own row.
// SO_SPLITREAD_MUSTFAIL: every instance reads instance (bh ^ 1)'s state.
// SO_SPLITREAD (QWEN36_MTP_GDN_SO_SPLITREAD, MODE 0 only): this RISC also produces the compute's ones tile and every
//   instance's pre-verify state (cb_state), on its own NoC while the reader gathers: instance 0's state reads are issued
//   first and the ones tile is filled under them; instance i+1's state is read as soon as the compute frees cb_state
//   (during instance i's token 0), before instance i's o pages are drained. SO_T0: row bh is read here before this RISC
//   writes the state after token wt over it (same RISC, a barrier in between), so the in-place order is kept.
//
// SO_VSPLIT (QWEN36_MTP_GDN_SO_VSPLIT, MODE 0 with SO_SPLITREAD; a define = P prepended by state_once): this core
//   runs part SO_VPART (runtime arg 6) of ONE head instance: it reads only the state tiles of value columns
//   [part*VP, (part+1)*VP) (VP = Vt / P; page kt*Vt + part*VP + j, pushed in kt-major order as a [Kt, VP] state),
//   writes only that slice of every o stick (VP*32 elements at the same byte offset in the page and in the staging
//   page, so the L1 and DRAM addresses keep today's relative alignment) and, under SO_T0, only those state pages.
//   SO_VSPLIT_MUSTFAIL: part 1 reads part 0's state columns (a wrong-input branch the self-test must catch).
//
// Compile args: {Kt, Vt, T, MODE, SO_T0} + TensorAccessorArgs(o or state) + TensorAccessorArgs(state).
// Runtime args: MODE 0 {bh_start, n_inst, o_addr, o_page_bytes, TOK, state_addr}; MODE 1 {bh_start, n_inst, state_addr}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/page.h"

constexpr uint32_t cb_out = 19, cb_sout = 18, cb_scratch = 27, cb_wcnt = 32;
#if defined(SO_SPLITREAD)
constexpr uint32_t cb_state = 5, cb_ones = 6;
#endif

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t T = get_compile_time_arg_val(2);
    constexpr uint32_t MODE = get_compile_time_arg_val(3);
    constexpr uint32_t SO_T0 = get_compile_time_arg_val(4);
    constexpr auto d_a = TensorAccessorArgs<5>();
    constexpr auto st_a = TensorAccessorArgs<d_a.next_compile_time_args_offset()>();

    const uint32_t bh_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t d_addr = get_arg_val<uint32_t>(2);

    const uint32_t tb_io = get_tile_size(cb_sout);
    const uint32_t elem = tb_io / 1024;
    constexpr uint32_t kv = Kt * Vt;
#if defined(SO_VSPLIT)
    static_assert(MODE == 0, "SO_VSPLIT is verify-only");
#if !defined(SO_SPLITREAD)
#error "SO_VSPLIT needs SO_SPLITREAD (the state is read here)"
#endif
    constexpr uint32_t VP_ = SO_VSPLIT;
    static_assert(VP_ >= 2 && Vt % VP_ == 0, "SO_VSPLIT: P must divide Vt");
    constexpr uint32_t VP = Vt / VP_;
    const uint32_t vpart = get_arg_val<uint32_t>(6);
#else
    constexpr uint32_t VP = Vt;
    constexpr uint32_t vpart = 0;
#endif
    constexpr uint32_t kvp = Kt * VP;  // this core's state tiles per instance (kv without SO_VSPLIT)
    // page of this core's state tile i (kt-major over [Kt, VP]) inside instance bh's [Kt, Vt] tile block
    auto spage = [&](uint32_t bh, uint32_t i, uint32_t part) -> uint32_t {
        return bh * kv + (i / VP) * Vt + part * VP + (i % VP);
    };

    Noc noc;

    if constexpr (MODE == 0) {
        const uint32_t o_page = get_arg_val<uint32_t>(3);
        const uint32_t tok_stride = get_arg_val<uint32_t>(4);
        const auto o_acc = TensorAccessor(d_a, d_addr, o_page);
        const auto st_acc = TensorAccessor(st_a, get_arg_val<uint32_t>(5), tb_io);

        auto zero = [&](uint32_t base, uint32_t n_words) {
            auto ptr = CoreLocalMem<volatile uint32_t>(base);
            for (uint32_t w = 0; w < n_words; w++) {
                ptr[w] = 0u;
            }
            asm volatile("" ::: "memory");
        };
        auto copy_words = [&](uint32_t src_bytes, uint32_t dst_bytes, uint32_t n_words) {
            asm volatile("" ::: "memory");
            auto s = CoreLocalMem<volatile uint32_t>(src_bytes);
            auto d = CoreLocalMem<volatile uint32_t>(dst_bytes);
            for (uint32_t w = 0; w < n_words; w++) {
                d[w] = s[w];
            }
            asm volatile("" ::: "memory");
        };

#if defined(SO_SPLITREAD)
        // issue the state reads of instance bh into cb_state (blocks until the compute has freed it)
        auto issue_state = [&](uint32_t bh) {
            CircularBuffer cbs(cb_state);
            cbs.reserve_back(kvp);
#if defined(SO_SPLITREAD_MUSTFAIL)
            const uint32_t src_bh = bh ^ 1u;  // must-fail: the neighbouring instance's state
#else
            const uint32_t src_bh = bh;
#endif
#if defined(SO_VSPLIT_MUSTFAIL)
            const uint32_t src_part = vpart == 1 ? 0u : vpart;  // must-fail: part 1 reads part 0's columns
#else
            const uint32_t src_part = vpart;
#endif
            for (uint32_t t = 0; t < kvp; t++) {
                noc.async_read(st_acc, cbs, tb_io, {.page_id = spage(src_bh, t, src_part)}, {.offset_bytes = t * tb_io});
            }
        };
        if (n_inst > 0) {
            issue_state(bh_start);
        }
        {
            CircularBuffer cb(cb_ones);
            cb.reserve_back(1);
            auto ptr = CoreLocalMem<uint32_t>(cb.get_write_ptr());
            for (uint32_t w = 0; w < 1024; w++) {
                ptr[w] = 0x3F800000u;
            }
            cb.push_back(1);
        }
        if (n_inst > 0) {
            noc.async_read_barrier();
            CircularBuffer(cb_state).push_back(kvp);
        }
#endif
        for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
            uint32_t wt = 0xFFFFFFFFu;
            if constexpr (SO_T0 != 0) {
                CircularBuffer ccb(cb_wcnt);
                ccb.wait_front(1);
                asm volatile("" ::: "memory");
                wt = CoreLocalMem<volatile uint32_t>(ccb.get_read_ptr())[0];
                ccb.pop_front(1);
            }
            for (uint32_t tok = 0; tok < T; ++tok) {
#if defined(SO_SPLITREAD)
                if (tok == 0 && bh + 1 < bh_start + n_inst) {
                    issue_state(bh + 1);
                    noc.async_read_barrier();
                    CircularBuffer(cb_state).push_back(kvp);
                }
#endif
                CircularBuffer cb(cb_out);
                cb.wait_front(VP);
                const uint32_t src = cb.get_read_ptr();
                CircularBuffer scb(cb_scratch);
                scb.reserve_back(1);
                const uint32_t stage = scb.get_write_ptr();
                zero(stage, (o_page + 3) / 4);
                const uint32_t cw = 16 * elem / 4;
#if defined(SO_QKH)
                // SO_QKH: token tok's o is row u = tok*BW + b of the o tiles (q of every token sits at its own row)
                const uint32_t orow = tok * SO_QKH_BW + bh / SO_QKH_H;
                const uint32_t oe0 = ((orow / 16) * 2) * 256 + (orow % 16) * 16;
#else
                constexpr uint32_t oe0 = 0;
#endif
                const uint32_t ooff = vpart * VP * 32 * elem;  // this part's byte offset in the o stick
                for (uint32_t t = 0; t < VP; t++) {
                    const uint32_t s = src + t * tb_io + oe0 * elem;
                    copy_words(s, stage + ooff + (32 * t) * elem, cw);
                    copy_words(s + 256 * elem, stage + ooff + (32 * t + 16) * elem, cw);
                }
                if constexpr (VP == Vt) {
                    noc.async_write(CoreLocalMem<uint32_t>(stage), o_acc, o_page, {}, {.page_id = tok * tok_stride + bh});
                } else {
                    noc.async_write(CoreLocalMem<uint32_t>(stage + ooff), o_acc, VP * 32 * elem, {},
                                    {.page_id = tok * tok_stride + bh, .offset_bytes = ooff});
                }
                noc.async_write_barrier();
                scb.pop_front(1);
                cb.pop_front(VP);
                if (tok == wt) {
                    // the state after token wt over this instance's own rec_state row (its base was read and pushed
                    // by the reader before the compute could produce token 0)
                    CircularBuffer sb(cb_sout);
                    sb.wait_front(kvp);
                    auto ssrc = use<CircularBuffer::AddrSelector::READ_PTR>(sb);
                    for (uint32_t t = 0; t < kvp; t++) {
                        noc.async_write(ssrc, st_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = spage(bh, t, vpart)});
                    }
                    noc.async_write_barrier();
                    sb.pop_front(kvp);
                }
            }
        }
    } else {
        const auto s_acc = TensorAccessor(d_a, d_addr, tb_io);
        for (uint32_t bh = bh_start; bh < bh_start + n_inst; ++bh) {
            CircularBuffer ccb(cb_wcnt);
            ccb.wait_front(1);
            asm volatile("" ::: "memory");
            const uint32_t n = CoreLocalMem<volatile uint32_t>(ccb.get_read_ptr())[0];
            ccb.pop_front(1);
            if (n == 0) {
                continue;
            }
            CircularBuffer cb(cb_sout);
            cb.wait_front(kv);
            const uint32_t base_page = bh * kv;
            auto src = use<CircularBuffer::AddrSelector::READ_PTR>(cb);
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_write(src, s_acc, tb_io, {.offset_bytes = t * tb_io}, {.page_id = base_page + t});
            }
            noc.async_write_barrier();
            cb.pop_front(kv);
        }
    }
}

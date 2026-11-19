// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_COMMIT_PACK (docs/MTP-BATCH.md section 19): the STATE_ONCE commit (so_reader.cpp MODE 1) of ALL GDN
// layers in ONE program. Reader half.
//
// Why: with QWEN36_MTP_GDN_SO_T0 the per-layer commit costs 3.74 / 3.29 / 3.93 / 4.98 ms per step at Bw 1 / 2 / 4 / 8
// (tm-so1-t0, [source validation record]): almost flat in Bw. Per layer the
// commit has only Bw x H head instances (24 at Bw=1) on 24 of the grid's cores, each recomputing its tokens serially,
// and the 48 layers run one after another. The layers' commits are independent (disjoint rec_state and cin buffers,
// one shared slot page), so this program spreads the L x Bw x H instances over the whole grid.
//
// Per instance it does EXACTLY what so_reader.cpp MODE 1 does (same idx read, same token count rule incl. the
// STATE_ONCE MUSTFAIL and SO_T0, same cb_cnt / cb_wcnt counts, same state page reads, same cin stick -> tile rebuild),
// so the compute kernel (so_compute.cpp MODE 1, unchanged) runs the same op sequence on the same bytes. The only
// difference: instance gi = l * BH + bh carries its layer l, whose state and cin buffer addresses come from the runtime
// args (a core's contiguous instance range spans at most a few layers; the wrapper passes only those).
//
// Compile args: {Kt, Vt, H, T, BW, MUSTFAIL, IDX_PAGE, CIN_PAGE, SO_T0}
//               + TensorAccessorArgs(state of layer 0) + TensorAccessorArgs(idx) + TensorAccessorArgs(cin of layer 0)
//               (every layer's state / cin has the same accessor args: the wrapper checks)
// Runtime args: {gi_start, n_inst, idx_addr, BH, l_first, nl, state_addr[nl], cin_addr[nl]}

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6, cb_scratch = 27;
constexpr uint32_t cb_cnt = 31, cb_wcnt = 32, cb_stick = 33;

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t H = get_compile_time_arg_val(2);
    constexpr uint32_t T = get_compile_time_arg_val(3);
    constexpr uint32_t BW = get_compile_time_arg_val(4);
    constexpr uint32_t MUSTFAIL = get_compile_time_arg_val(5);
    constexpr uint32_t IDX_PAGE = get_compile_time_arg_val(6);
    constexpr uint32_t CIN_PAGE = get_compile_time_arg_val(7);
    constexpr uint32_t SO_T0 = get_compile_time_arg_val(8);
    static_assert(SO_T0 == 0 || T >= 2, "SO_T0 needs T >= 2");

    constexpr auto s0_a = TensorAccessorArgs<9>();
    constexpr auto ix_a = TensorAccessorArgs<s0_a.next_compile_time_args_offset()>();
    constexpr auto cin_a = TensorAccessorArgs<ix_a.next_compile_time_args_offset()>();

    const uint32_t gi_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_arg_val<uint32_t>(2);
    const uint32_t BH = get_arg_val<uint32_t>(3);
    const uint32_t l_first = get_arg_val<uint32_t>(4);
    const uint32_t nl = get_arg_val<uint32_t>(5);
    constexpr uint32_t ARG_ADDR = 6;

    const uint32_t tb_io = get_tile_size(cb_k);  // state / cin share one dtype (checked by the wrapper)
    const uint32_t elem = tb_io / 1024;

    constexpr uint32_t kv = Kt * Vt;
    constexpr uint32_t Dk = Kt * 32, Dv = Vt * 32;

    Noc noc;

    // The private stick page (reserved once, never pushed: the reader is its only user).
    CircularBuffer stcb(cb_stick);
    stcb.reserve_back(1);
    const uint32_t stick = stcb.get_write_ptr();

    // so_reader.cpp MODE 1: rebuild a gathered row tile set from the stick: zero the pages (NoC loopback), row 0 from
    // the stick.
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

    // so_reader.cpp MODE 1: a scalar tile with the stick's scalar at [0,0] (the rest zero; the compute reads only
    // [0,0]).
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

    auto push_count = [&](uint32_t cb_id, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(1);
        auto p = CoreLocalMem<volatile uint32_t>(cb.get_write_ptr());
        p[0] = n;
        asm volatile("" ::: "memory");
        cb.push_back(1);
    };

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

    // The idx page (shared by every layer), read once into this kernel's private scratch page (reserved, never pushed).
    uint32_t idx_l1 = 0;
    {
        const auto ix_acc = TensorAccessor(ix_a, idx_addr, IDX_PAGE);
        CircularBuffer scb(cb_scratch);
        scb.reserve_back(1);
        idx_l1 = scb.get_write_ptr();
        noc_async_read_page(0, ix_acc, idx_l1);
        noc_async_read_barrier();
    }

    // stick layout (elements): [k: 0 .. Dk) [v: Dk .. Dk+Dv) [beta: Dk+Dv] [g: Dk+Dv+1] [0 ...]
    constexpr uint32_t ST_K = 0, ST_V = Dk, ST_B = Dk + Dv, ST_G = Dk + Dv + 1;

    for (uint32_t gi = gi_start; gi < gi_start + n_inst; ++gi) {
        const uint32_t l = gi / BH;
        const uint32_t bh = gi % BH;
        const uint32_t lo = l - l_first;  // < nl by construction (the wrapper's layer window)
        const uint32_t s_addr = get_arg_val<uint32_t>(ARG_ADDR + lo);
        const uint32_t c_addr = get_arg_val<uint32_t>(ARG_ADDR + nl + lo);
        const uint32_t b = bh / H;
        const uint32_t h = bh % H;

        asm volatile("" ::: "memory");
        const uint32_t s = CoreLocalMem<volatile uint32_t>(idx_l1)[b];
        uint32_t n = s < T ? s + 1 : 0;
        uint32_t tok0 = 0;
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

        // The base state of this layer's row b, head h, once per instance.
        {
            const auto s0_acc = TensorAccessor(s0_a, s_addr, tb_io);
            CircularBuffer cbs(cb_state);
            cbs.reserve_back(kv);
            const uint32_t base_page = bh * kv;
            for (uint32_t t = 0; t < kv; t++) {
                noc.async_read(s0_acc, cbs, tb_io, {.page_id = base_page + t}, {.offset_bytes = t * tb_io});
            }
            noc.async_read_barrier();
            cbs.push_back(kv);
        }

        const auto cin_acc = TensorAccessor(cin_a, c_addr, CIN_PAGE);
        for (uint32_t tok = tok0; tok < tok0 + n; ++tok) {
            const uint32_t u = tok * BW + b;  // candidate-major row of the verify
            noc_async_read_page(u * H + h, cin_acc, stick);
            noc_async_read_barrier();
            tiles_from_stick(cb_k, Kt, ST_K);
            tiles_from_stick(cb_v, Vt, ST_V);
            scalar_from_stick(cb_beta, ST_B);
            scalar_from_stick(cb_g, ST_G);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// decode_reader_risc NOFAN VARIANT (bisect): stock ones tile; pad rows by noc.async_write_zeros.
// Reader for the fused attention decode prologue. Device 2.0 API.
//
// Instance inst = b*4 + kind (kind 0 q, 1 k, 2 v, 3 gate). The kind's block is 32 rows (heads)
// x HD: for head h < n_heads(kind), the source is row b of the HDt projection tiles at tile
// column off(kind) + h*HDt -- DMA those HDt full pages into the staging ring, then copy row
// (b % 32)'s two 16-element chunks of each tile into row h of the block's tile. Rows at or
// beyond the head count are zeroed (two chunks per tile per row). q/k blocks go to cb_blk for
// the compute; v/gate blocks go straight to cb_out (no math). q/k also get this row's cos and
// sin (page b*RDt + t, row 0 is the data). q_norm_w / k_norm_w row-0 tiles are read once.
//
// Compile args: {HDt, RDt, NH, NKV, Wt, q_off_t, k_off_t, v_off_t, g_off_t} + accessors
// (qkv, cos, sin, wq, wk). Runtime args: {inst_start, n_inst, qkv, cos, sin, wq, wk}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"

constexpr uint32_t cb_src = 0, cb_blk = 1, cb_out = 2, cb_cos = 3, cb_sin = 4, cb_wq = 5, cb_wk = 6, cb_ones = 9;

void kernel_main() {
    constexpr uint32_t HDt = get_compile_time_arg_val(0);
    constexpr uint32_t RDt = get_compile_time_arg_val(1);
    constexpr uint32_t NH = get_compile_time_arg_val(2);
    constexpr uint32_t NKV = get_compile_time_arg_val(3);
    constexpr uint32_t Wt = get_compile_time_arg_val(4);
    constexpr uint32_t QOT = get_compile_time_arg_val(5);
    constexpr uint32_t KOT = get_compile_time_arg_val(6);
    constexpr uint32_t VOT = get_compile_time_arg_val(7);
    constexpr uint32_t GOT = get_compile_time_arg_val(8);
    constexpr uint32_t HG = 4;  // heads gathered per barrier (staging ring = HDt*HG pages)

    constexpr auto qkv_a = TensorAccessorArgs<9>();
    constexpr auto cos_a = TensorAccessorArgs<qkv_a.next_compile_time_args_offset()>();
    constexpr auto sin_a = TensorAccessorArgs<cos_a.next_compile_time_args_offset()>();
    constexpr auto wq_a = TensorAccessorArgs<sin_a.next_compile_time_args_offset()>();
    constexpr auto wk_a = TensorAccessorArgs<wq_a.next_compile_time_args_offset()>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t qkv_addr = get_arg_val<uint32_t>(2);
    const uint32_t cos_addr = get_arg_val<uint32_t>(3);
    const uint32_t sin_addr = get_arg_val<uint32_t>(4);
    const uint32_t wq_addr = get_arg_val<uint32_t>(5);
    const uint32_t wk_addr = get_arg_val<uint32_t>(6);

    const uint32_t tb = get_tile_size(cb_src);
    const uint32_t elem = tb / 1024;
    const auto qkv_acc = TensorAccessor(qkv_a, qkv_addr, tb);
    const auto cos_acc = TensorAccessor(cos_a, cos_addr, tb);
    const auto sin_acc = TensorAccessor(sin_a, sin_addr, tb);
    const auto wq_acc = TensorAccessor(wq_a, wq_addr, tb);
    const auto wk_acc = TensorAccessor(wk_a, wk_addr, tb);

    Noc noc;

    auto zero_words = [&](uint32_t base, uint32_t n_words) {
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
    // element offset of row r's first face chunk (cols 0-15); +256 is the second (cols 16-31)
    auto row_e0 = [](uint32_t r) { return ((r / 16) * 2) * 256 + (r % 16) * 16; };
    const uint32_t cw = 16 * elem / 4;  // words per 16-element chunk

    // ones tile (row-sum contraction) and the two norm weights, once per core
    // decode_reader_risc fix: stock wrote all 1024 fp32 words with RISC stores. Now: a 256 B seed
    // (64 volatile stores), then 15 NoC loopback L1->L1 reads copy it to bytes [256, 4096) of the
    // same page (sources never rewritten, destinations disjoint), one barrier, then the push.
    {
        CircularBuffer cb(cb_ones);
        cb.reserve_back(1);
        auto ptr = CoreLocalMem<uint32_t>(cb.get_write_ptr());
        for (uint32_t w = 0; w < 1024; w++) {
            ptr[w] = 0x3F800000u;
        }
        cb.push_back(1);
    }
    auto read_row0_tiles = [&](const auto& acc, uint32_t cb_id, uint32_t first_page, uint32_t n) {
        CircularBuffer cb(cb_id);
        cb.reserve_back(n);
        for (uint32_t t = 0; t < n; t++) {
            noc.async_read(acc, cb, tb, {.page_id = first_page + t}, {.offset_bytes = t * tb});
        }
        noc.async_read_barrier();
        cb.push_back(n);
    };
    read_row0_tiles(wq_acc, cb_wq, 0, HDt);
    read_row0_tiles(wk_acc, cb_wk, 0, HDt);

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t b = inst / 4;
        const uint32_t kind = inst % 4;
        const uint32_t n_rows = (kind == 0 || kind == 3) ? NH : NKV;
        const uint32_t off_t = (kind == 0) ? QOT : (kind == 1) ? KOT : (kind == 2) ? VOT : GOT;
        const bool math = (kind == 0 || kind == 1);
        const uint32_t target = math ? cb_blk : cb_out;
        const uint32_t r = b % 32;
        const uint32_t row_page0 = (b / 32) * Wt;

        CircularBuffer tcb(target);
        tcb.reserve_back(HDt);
        const uint32_t tbase = tcb.get_write_ptr();
        // rows beyond the head count: zero (two chunks per tile)
        // decode_reader_risc fix: rows [n_rows, 32) of a tile are, in elements (face = 256):
        //   n_rows < 16 : [16n, 256)                  face 0 rows n..15
        //                 [256+16n, 1024)             face 1 rows n..15 + faces 2,3 (contiguous)
        //   n_rows >= 16: [512+16m, 768), [768+16m, 1024), m = n-16   faces 2 and 3 rows m..15
        // exactly the chunks stock zeroed. Byte offsets are 16*elem*k (elem 2 or 4): 16 B aligned,
        // sizes multiples of 32 B. Zeroed by NoC loopback from MEM_ZEROS_BASE; the barrier comes
        // after the first head group's DMA barrier and before any copy_words into this block.
        bool zeros_pending = false;
        if (n_rows < 32) {
            uint32_t s0, e0, s1;  // span A [s0, e0), span B [s1, 1024), in elements
            if (n_rows < 16) {
                s0 = 16 * n_rows;
                e0 = 256;
                s1 = 256 + 16 * n_rows;
            } else {
                s0 = 512 + 16 * (n_rows - 16);
                e0 = 768;
                s1 = 768 + 16 * (n_rows - 16);
            }
            for (uint32_t t = 0; t < HDt; t++) {
                const uint32_t p = tbase + t * tb;
                noc.async_write_zeros(CoreLocalMem<uint32_t>(p + s0 * elem), (e0 - s0) * elem);
                noc.async_write_zeros(CoreLocalMem<uint32_t>(p + s1 * elem), (1024 - s1) * elem);
            }
            zeros_pending = true;
        }
        // head rows: DMA the heads' HDt tiles (row b lives at row r of each) in groups of HG heads
        // behind one barrier, then scatter row r of each tile into row h of the block's tile.
        const uint32_t se0 = row_e0(r);
        for (uint32_t h0 = 0; h0 < n_rows; h0 += HG) {
            const uint32_t hn = (n_rows - h0 < HG) ? (n_rows - h0) : HG;
            CircularBuffer scb(cb_src);
            scb.reserve_back(HDt * HG);
            const uint32_t sbase = scb.get_write_ptr();
            for (uint32_t hh = 0; hh < hn; hh++) {
                const uint32_t first_page = row_page0 + off_t + (h0 + hh) * HDt;
                for (uint32_t t = 0; t < HDt; t++) {
                    noc.async_read(qkv_acc, scb, tb, {.page_id = first_page + t}, {.offset_bytes = (hh * HDt + t) * tb});
                }
            }
            noc.async_read_barrier();
            if (zeros_pending) {
                noc.write_zeros_l1_barrier();  // pad-row zeros complete before any RISC copy into the block
                zeros_pending = false;
            }
            for (uint32_t hh = 0; hh < hn; hh++) {
                const uint32_t de0 = row_e0(h0 + hh);
                const uint32_t hb = sbase + hh * HDt * tb;
                for (uint32_t t = 0; t < HDt; t++) {
                    copy_words(hb + t * tb + se0 * elem, tbase + t * tb + de0 * elem, cw);
                    copy_words(hb + t * tb + (se0 + 256) * elem, tbase + t * tb + (de0 + 256) * elem, cw);
                }
            }
            scb.push_back(HDt * HG);
            scb.pop_front(HDt * HG);  // staging recycled by this kernel alone
        }
        if (zeros_pending) {
            noc.write_zeros_l1_barrier();  // n_rows == 0 guard: never push un-barriered zeros
        }
        tcb.push_back(HDt);

        if (math) {
            // cos / sin: [1,B,1,RD] TILE -> page b*RDt + t, row 0 holds the values
            read_row0_tiles(cos_acc, cb_cos, b * RDt, RDt);
            read_row0_tiles(sin_acc, cb_sin, b * RDt, RDt);
        }
    }
}

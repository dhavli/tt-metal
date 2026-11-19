// SPDX-FileCopyrightText: © 2026 Thatch Cloud
// SPDX-License-Identifier: Apache-2.0
//
// decode_reader_risc NOFAN VARIANT (bisect): stock ones tile; pad rows by noc.async_write_zeros.
// Reader for the fused GDN decode output norm + gate. Device 2.0 API.
//
// Per instance (bt, h, ct): DMA the 32 sticks o[bt*32 + r, h, :] (full ROW_MAJOR pages)
// contiguously into the row-major block CB -- stick r at byte offset r*o_page -- so the block
// is a 32 x V row-major matrix the compute tilizes. Rows at or beyond B are zeroed instead
// of read. Then one z tile (bt, z_off_t + h*Vt + ct) and one norm_w tile (ct). All reads
// are issued back to back with one barrier.
//
// Compile args: {Vt, H, B, Wt, z_off_t, o_page} + accessor args (o, z, w).
// Runtime args: {inst_start, n_inst, o, z, w}.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"

constexpr uint32_t cb_orm = 0, cb_z = 1, cb_w = 2, cb_ones = 3;

void kernel_main() {
    constexpr uint32_t Vt = get_compile_time_arg_val(0);
    constexpr uint32_t H = get_compile_time_arg_val(1);
    constexpr uint32_t B = get_compile_time_arg_val(2);
    constexpr uint32_t Wt = get_compile_time_arg_val(3);
    constexpr uint32_t ZOT = get_compile_time_arg_val(4);
    constexpr uint32_t O_PAGE = get_compile_time_arg_val(5);

    constexpr auto o_a = TensorAccessorArgs<6>();
    constexpr auto z_a = TensorAccessorArgs<o_a.next_compile_time_args_offset()>();
    constexpr auto w_a = TensorAccessorArgs<z_a.next_compile_time_args_offset()>();

    const uint32_t inst_start = get_arg_val<uint32_t>(0);
    const uint32_t n_inst = get_arg_val<uint32_t>(1);
    const uint32_t o_addr = get_arg_val<uint32_t>(2);
    const uint32_t z_addr = get_arg_val<uint32_t>(3);
    const uint32_t w_addr = get_arg_val<uint32_t>(4);

    const uint32_t tb_z = get_tile_size(cb_z);
    const uint32_t tb_w = get_tile_size(cb_w);
    const auto o_acc = TensorAccessor(o_a, o_addr, O_PAGE);
    const auto z_acc = TensorAccessor(z_a, z_addr, tb_z);
    const auto w_acc = TensorAccessor(w_a, w_addr, tb_w);

    Noc noc;

    auto zero_words = [&](uint32_t base, uint32_t n_words) {
        auto ptr = CoreLocalMem<volatile uint32_t>(base);
        for (uint32_t w = 0; w < n_words; w++) {
            ptr[w] = 0u;
        }
        asm volatile("" ::: "memory");
    };

    // fp32 all-ones tile (row-sum contraction operand), once.
    // decode_reader_risc fix: stock wrote all 1024 words with RISC stores (a narrow L1 store is a
    // read-modify-write). Now: a 256 B seed (64 volatile stores), then 15 NoC loopback L1->L1 reads
    // copy the seed to bytes [256, 4096) of the same page. Sources [0,256) are never written after
    // the seed; destinations are disjoint from them. One barrier, then the push. Same 4096 bytes.
    {
        CircularBuffer cb(cb_ones);
        cb.reserve_back(1);
        auto ptr = CoreLocalMem<uint32_t>(cb.get_write_ptr());
        for (uint32_t w = 0; w < 1024; w++) {
            ptr[w] = 0x3F800000u;
        }
        cb.push_back(1);
    }

    for (uint32_t inst = inst_start; inst < inst_start + n_inst; ++inst) {
        const uint32_t ct = inst % Vt;
        const uint32_t hh = (inst / Vt) % H;
        const uint32_t bt = inst / (Vt * H);

        // ---- o: 32 sticks -> row-major block ----
        {
            CircularBuffer cb(cb_orm);
            cb.reserve_back(Vt);
            const uint32_t base = cb.get_write_ptr();
            // decode_reader_risc fix: rows r < nread are DMA'd exactly as stock; rows [nread, 32) --
            // the ones stock zeroed with (32-nread)*O_PAGE/4 RISC stores -- are one contiguous byte span
            // [nread*O_PAGE, 32*O_PAGE) of the block, zeroed by a NoC loopback from MEM_ZEROS_BASE,
            // issued after the o reads so both are in flight together. Both barriers complete before
            // the push. Disjoint bytes from every o read.
            const uint32_t row0 = bt * 32;
            const uint32_t nread = (B > row0) ? ((B - row0 < 32) ? (B - row0) : 32) : 0;
            for (uint32_t r = 0; r < nread; r++) {
                noc.async_read(o_acc, cb, O_PAGE, {.page_id = (row0 + r) * H + hh}, {.offset_bytes = r * O_PAGE});
            }
            if (nread < 32) {
                if constexpr (O_PAGE % 16 == 0) {
                    noc.async_write_zeros(CoreLocalMem<uint32_t>(base + nread * O_PAGE), (32 - nread) * O_PAGE);
                    noc.write_zeros_l1_barrier();
                } else {
                    // span start/size not 16 B aligned: keep the stock RISC stores. Barrier the o reads
                    // first: a narrow RISC store is a 128-bit read-modify-write and an unaligned boundary
                    // word could be shared with the last DMA'd row.
                    noc.async_read_barrier();
                    for (uint32_t r = nread; r < 32; r++) {
                        zero_words(base + r * O_PAGE, O_PAGE / 4);
                    }
                }
            }
            noc.async_read_barrier();
            cb.push_back(Vt);
        }
        // ---- z tile and norm_w tile for this column ----
        {
            CircularBuffer cb(cb_z);
            cb.reserve_back(1);
            noc.async_read(z_acc, cb, tb_z, {.page_id = bt * Wt + ZOT + hh * Vt + ct}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        }
        {
            CircularBuffer cb(cb_w);
            cb.reserve_back(1);
            noc.async_read(w_acc, cb, tb_w, {.page_id = ct}, {.offset_bytes = 0});
            noc.async_read_barrier();
            cb.push_back(1);
        }
    }
}

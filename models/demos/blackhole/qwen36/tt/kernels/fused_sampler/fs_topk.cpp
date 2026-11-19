// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_SPEC_SAMPLE_FUSED (fused_sampler.py): the per-chip top-K of the vocab-shard logits, straight from the
// TILE layout (no untilize), in ONE multi-core program.  Replaces the served untilize + topk_large_indices + gather
// + typecasts (and, because the order below is total, the full-vocab greedy argmax as well).
//
// Input:  logits [1, 1, M, PER] bf16 TILE, DRAM interleaved (one tile row: M <= 32), page = one 32x32 bf16 tile.
// Output: block  [1, 1, M, 2K] float32 ROW_MAJOR, DRAM interleaved, page = one row:
//           cols 0..K-1   the K largest values of the row, as float32 BITS of the bf16 (h << 16), descending
//           cols K..2K-1  their LOCAL (in-shard) column ids, as raw uint32 bits
//         The block only ever travels (one all-gather) to fs_cut.cpp, which reads it as bits.
//
// THE ORDER is total: value descending, then local id ascending (the served greedy tie rule: the lowest id among the
// ids tied at the max; +0 == -0, compared as floats).  So the top-K SET is unique even with ties (the served wide
// top-K's tie order is unspecified), and entry 0 of the merged list is exactly the served greedy argmax.
//
// Two phases, every core of the grid, both data-movement RISCs:
//   scan    core s owns tiles [TILE0, TILE0 + NTILES) of the row; this RISC owns rows [ROW0, ROW1).  It reads ONLY the
//           face rows it needs (64-byte aligned spans of faces 0/1 for rows < 16, faces 2/3 for rows >= 16), then per
//           row keeps a K-entry min-heap of composites (ordered key << 16 | (0xFFFF - rel col)), with a 2-instruction
//           sign-extended compare against the heap root once the root is positive.  Each row's list is written sorted
//           (descending) into this core's L1 list area and the row's reducer is signalled (one semaphore increment).
//   reduce  core m (m < M), on the RISC that owns row m: waits for all NC increments, NoC-reads row m's list from every
//           core, merges the NC sorted lists (a max-heap of the list heads, K pops) and writes the output row.
//
// Ordered key of a bf16 bit pattern h (int16 view hs): hs >= 0 -> hs; hs == -0 -> 0; else hs ^ 0x7FFF.  ukey = key +
// 0x8000 in 0..0xFFFF.  NaN orders above +inf (logits are never NaN).  A list entry is (ukey << 16 | h, local id); a
// short list (a core with fewer than K columns) is padded with (0, 0xFFFFFFFF), the smallest entry.
//
// Compile args: {CB_DATA, CB_LIST, CB_RED, M, K, NC, ROW0, ROW1, SEM, DATA_OFF, GX, GY, TILE_PAGE, OUT_PAGE, MF}
//                + TensorAccessorArgs(logits) + (out)
// Runtime args: {logits_addr, out_addr, s (core index), tile0, ntiles, PX[0..GX-1], PY[0..GY-1]} (PX/PY: the NoC
//               coordinates of the grid's logical columns / rows; core q is at logical (q % GX, q / GX))
// MF (QWEN36_MTP_SPEC_SAMPLE_FUSED_MUSTFAIL=topk): the scan drops the LAST column of every tile: the top-K misses
// candidates, so the self-test must REFUSE.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

namespace {

FORCE_INLINE int32_t okey(int32_t hs) {
    // hs: the bf16 bits sign-extended to 32 bits
    if (hs >= 0) {
        return hs;
    }
    if (hs == -32768) {
        return 0;  // -0 == +0
    }
    return hs ^ 0x7FFF;
}

FORCE_INLINE uint32_t h_of_key(int32_t k) {
    if (k >= 0) {
        return static_cast<uint32_t>(k) & 0xFFFFu;
    }
    return static_cast<uint32_t>(k ^ 0x7FFF) & 0xFFFFu;
}

template <uint32_t N>
FORCE_INLINE void heap_down(uint32_t* h, uint32_t n, uint32_t i) {
    // min-heap sift-down
    const uint32_t v = h[i];
    while (true) {
        uint32_t c = 2 * i + 1;
        if (c >= n) {
            break;
        }
        if (c + 1 < n && h[c + 1] < h[c]) {
            ++c;
        }
        if (h[c] >= v) {
            break;
        }
        h[i] = h[c];
        i = c;
    }
    h[i] = v;
}

FORCE_INLINE void heap_up(uint32_t* h, uint32_t i) {
    const uint32_t v = h[i];
    while (i > 0) {
        const uint32_t p = (i - 1) >> 1;
        if (h[p] <= v) {
            break;
        }
        h[i] = h[p];
        i = p;
    }
    h[i] = v;
}

}  // namespace

void kernel_main() {
    constexpr uint32_t CB_DATA = get_compile_time_arg_val(0);
    constexpr uint32_t CB_LIST = get_compile_time_arg_val(1);
    constexpr uint32_t CB_RED = get_compile_time_arg_val(2);
    constexpr uint32_t M = get_compile_time_arg_val(3);
    constexpr uint32_t K = get_compile_time_arg_val(4);
    constexpr uint32_t NC = get_compile_time_arg_val(5);
    constexpr uint32_t ROW0 = get_compile_time_arg_val(6);
    constexpr uint32_t ROW1 = get_compile_time_arg_val(7);
    constexpr uint32_t SEM = get_compile_time_arg_val(8);
    constexpr uint32_t DATA_OFF = get_compile_time_arg_val(9);
    constexpr uint32_t GX = get_compile_time_arg_val(10);
    constexpr uint32_t GY = get_compile_time_arg_val(11);
    constexpr uint32_t TILE_PAGE = get_compile_time_arg_val(12);
    constexpr uint32_t OUT_PAGE = get_compile_time_arg_val(13);
    constexpr uint32_t MF = get_compile_time_arg_val(14);
    constexpr auto l_args = TensorAccessorArgs<15>();
    constexpr auto o_args = TensorAccessorArgs<l_args.next_compile_time_args_offset()>();
    static_assert(K >= 1 && K <= 255, "K");
    static_assert(NC >= 1 && NC <= 255, "NC");
    static_assert(M >= 1 && M <= 32, "M");

    // the face-row spans this RISC reads: half f covers rows [16f, 16f + 16); a span is 64-byte aligned (2 rows)
    constexpr uint32_t A0 = ROW0 < 16 ? ROW0 : 16;
    constexpr uint32_t B0 = ROW1 < 16 ? ROW1 : 16;
    constexpr bool P0 = A0 < B0;
    constexpr uint32_t A0A = P0 ? (A0 & ~1u) : 0;
    constexpr uint32_t SPAN0 = P0 ? (((B0 + 1) & ~1u) - A0A) * 32 : 0;
    constexpr uint32_t A1 = (ROW0 > 16 ? ROW0 : 16) - 16;
    constexpr uint32_t B1 = (ROW1 > 16 ? ROW1 : 16) - 16;
    constexpr bool P1 = A1 < B1;
    constexpr uint32_t A1A = P1 ? (A1 & ~1u) : 0;
    constexpr uint32_t SPAN1 = P1 ? (((B1 + 1) & ~1u) - A1A) * 32 : 0;
    constexpr uint32_t TS = 2 * (SPAN0 + SPAN1);  // L1 bytes per tile: [face0|face1 span][face2|face3 span]
    constexpr uint32_t FACE = 512;
    constexpr uint32_t LIST_ROW = K * 8;

    const uint32_t logits_addr = get_arg_val<uint32_t>(0);
    const uint32_t out_addr = get_arg_val<uint32_t>(1);
    const uint32_t s = get_arg_val<uint32_t>(2);
    const uint32_t tile0 = get_arg_val<uint32_t>(3);
    const uint32_t ntiles = get_arg_val<uint32_t>(4);

    const auto lg = TensorAccessor(l_args, logits_addr, TILE_PAGE);
    const auto out = TensorAccessor(o_args, out_addr, OUT_PAGE);
    const uint32_t data = get_write_ptr(CB_DATA) + DATA_OFF;
    const uint32_t lists = get_write_ptr(CB_LIST);

    if constexpr (ROW0 < ROW1) {
        // ---------------------------------------------------------------- read this RISC's face rows
        for (uint32_t t = 0; t < ntiles; ++t) {
            const uint32_t page = tile0 + t;
            uint32_t dst = data + t * TS;
            if constexpr (P0) {
                noc_async_read(lg.get_noc_addr(page, 0 * FACE + A0A * 32), dst, SPAN0);
                noc_async_read(lg.get_noc_addr(page, 1 * FACE + A0A * 32), dst + SPAN0, SPAN0);
                dst += 2 * SPAN0;
            }
            if constexpr (P1) {
                noc_async_read(lg.get_noc_addr(page, 2 * FACE + A1A * 32), dst, SPAN1);
                noc_async_read(lg.get_noc_addr(page, 3 * FACE + A1A * 32), dst + SPAN1, SPAN1);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();

        // ---------------------------------------------------------------- scan: one row at a time, heap in local RAM
        uint32_t heap[K];
        for (uint32_t r = ROW0; r < ROW1; ++r) {
            const bool hi_half = r >= 16;
            const uint32_t rr = hi_half ? r - 16 : r;
            const uint32_t lo_off = hi_half ? (2 * SPAN0 + (rr - A1A) * 32) : ((rr - A0A) * 32);
            const uint32_t span = hi_half ? SPAN1 : SPAN0;
            uint32_t n = 0;
            int32_t thr = -2147483647 - 1;  // the root's ordered key once the heap is full
            bool fast = false;        // thr >= 0: a sign-extended compare is exact

            auto put = [&](int32_t hs, uint32_t rel) {
                const uint32_t uk = static_cast<uint32_t>(okey(hs) + 0x8000);
                const uint32_t c = (uk << 16) | (0xFFFFu - rel);
                if (n < K) {
                    heap[n] = c;
                    heap_up(heap, n);
                    ++n;
                    if (n < K) {
                        return;
                    }
                } else {
                    heap[0] = c;
                    heap_down<K>(heap, K, 0);
                }
                thr = static_cast<int32_t>(heap[0] >> 16) - 0x8000;
                fast = thr >= 0;
            };
            auto seg = [&](uint32_t addr, uint32_t rel0) {
                volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(addr);
                for (uint32_t w = 0; w < 8; ++w) {
                    const uint32_t v = p[w];
                    const int32_t h0 = static_cast<int32_t>(v << 16) >> 16;
                    const int32_t h1 = static_cast<int32_t>(v) >> 16;
                    const uint32_t rel = rel0 + 2 * w;
                    if (fast) {
                        if (h0 > thr) {
                            put(h0, rel);
                        }
                        if (h1 > thr && !(MF && w == 7 && (rel0 & 16u))) {  // MF: column 31 never a candidate
                            put(h1, rel + 1);
                        }
                    } else {
                        if (okey(h0) > thr) {
                            put(h0, rel);
                        }
                        if (okey(h1) > thr && !(MF && w == 7 && (rel0 & 16u))) {
                            put(h1, rel + 1);
                        }
                    }
                }
            };
            for (uint32_t t = 0; t < ntiles; ++t) {
                const uint32_t base = data + t * TS + lo_off;
                seg(base, t * 32);              // columns 0..15  (face 0 or 2)
                seg(base + span, t * 32 + 16);  // columns 16..31 (face 1 or 3)
            }
            // heapsort: the min-heap's minimum moves to the back, so the array ends DESCENDING
            for (uint32_t i = n; i > 1; --i) {
                const uint32_t top = heap[0];
                heap[0] = heap[i - 1];
                heap[i - 1] = top;
                heap_down<K>(heap, i - 1, 0);
            }
            volatile tt_l1_ptr uint32_t* L = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(lists + r * LIST_ROW);
            const uint32_t col0 = tile0 * 32;
            for (uint32_t j = 0; j < K; ++j) {
                if (j < n) {
                    const uint32_t c = heap[j];
                    const uint32_t uk = c >> 16;
                    const uint32_t h = h_of_key(static_cast<int32_t>(uk) - 0x8000);
                    L[2 * j] = (uk << 16) | h;
                    L[2 * j + 1] = col0 + (0xFFFFu - (c & 0xFFFFu));
                } else {
                    L[2 * j] = 0;
                    L[2 * j + 1] = 0xFFFFFFFFu;
                }
            }
            // the stores must land before the reducer's NoC read: read the last word back, then fence
            (void)L[2 * K - 1];
            asm volatile("fence" ::: "memory");
            if (r < M) {
                // signal row r's reducer (core r)
                const uint32_t px = get_arg_val<uint32_t>(5 + (r % GX));
                const uint32_t py = get_arg_val<uint32_t>(5 + GX + (r / GX));
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(SEM)), 1);
            }
        }
        noc_async_atomic_barrier();

        // ---------------------------------------------------------------- reduce: row s, on the RISC that owns it
        if (s < M && s >= ROW0 && s < ROW1) {
            const uint32_t m = s;
            volatile tt_l1_ptr uint32_t* sem = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(SEM));
            noc_semaphore_wait(sem, NC);
            noc_semaphore_set(sem, 0);
            const uint32_t red = get_write_ptr(CB_RED);
            for (uint32_t q = 0; q < NC; ++q) {
                const uint32_t px = get_arg_val<uint32_t>(5 + (q % GX));
                const uint32_t py = get_arg_val<uint32_t>(5 + GX + (q / GX));
                noc_async_read(get_noc_addr(px, py, lists + m * LIST_ROW), red + q * LIST_ROW, LIST_ROW);
            }
            noc_async_read_barrier();
            invalidate_l1_cache();
            // a max-heap of the NC list heads; entry (w0, w1) beats (v0, v1) iff w0 > v0 or (w0 == v0 and w1 < v1)
            // the merge state lives in L1 (NC <= 255 bytes each), not on the RISC's small local stack
            uint8_t* cur = reinterpret_cast<uint8_t*>(red + NC * LIST_ROW + 2 * K * 4);
            uint8_t* hp = cur + NC;
            auto w0 = [&](uint32_t q) -> uint32_t {
                return cur[q] < K ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red + q * LIST_ROW)[2 * cur[q]] : 0u;
            };
            auto w1 = [&](uint32_t q) -> uint32_t {
                return cur[q] < K ? reinterpret_cast<volatile tt_l1_ptr uint32_t*>(red + q * LIST_ROW)[2 * cur[q] + 1]
                                  : 0xFFFFFFFFu;
            };
            auto gt = [&](uint32_t a, uint32_t b) -> bool {
                const uint32_t a0 = w0(a), b0 = w0(b);
                return a0 > b0 || (a0 == b0 && w1(a) < w1(b));
            };
            auto down = [&](uint32_t i, uint32_t n) {
                const uint8_t v = hp[i];
                while (true) {
                    uint32_t c = 2 * i + 1;
                    if (c >= n) {
                        break;
                    }
                    if (c + 1 < n && gt(hp[c + 1], hp[c])) {
                        ++c;
                    }
                    if (!gt(hp[c], v)) {
                        break;
                    }
                    hp[i] = hp[c];
                    i = c;
                }
                hp[i] = v;
            };
            for (uint32_t q = 0; q < NC; ++q) {
                cur[q] = 0;
                hp[q] = static_cast<uint8_t>(q);
            }
            for (uint32_t i = NC / 2; i-- > 0;) {
                down(i, NC);
            }
            const uint32_t orow = red + NC * LIST_ROW;
            volatile tt_l1_ptr uint32_t* O = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(orow);
            for (uint32_t j = 0; j < K; ++j) {
                const uint32_t q = hp[0];
                const uint32_t e0 = w0(q);
                O[j] = (e0 & 0xFFFFu) << 16;  // the bf16 value as float32 bits
                O[K + j] = w1(q);             // the local id
                cur[q] = static_cast<uint8_t>(cur[q] + 1);
                down(0, NC);
            }
            (void)O[2 * K - 1];
            asm volatile("fence" ::: "memory");
            noc_async_write(orow, out.get_noc_addr(m), 2 * K * 4);
            noc_async_write_barrier();
        }
    }
}

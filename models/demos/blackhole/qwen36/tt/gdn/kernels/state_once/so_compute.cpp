// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_GDN_STATE_ONCE (docs/MTP-BATCH.md section 16): the MTP verify's GDN recurrence (MODE 0) and the per-row
// state commit by masked recompute (MODE 1), each ONE program per layer, the state read from DRAM once per row.
//
// Derived from QWEN36_GDN_MULTITOKEN's mt_compute.cpp (branch gdn-multitoken), which is the production T=1 decode
// kernel (ttnn/.../decode_gated_delta_rule/device/kernels/compute/decode_gated_delta_rule.cpp, the served -gf arm
// adds only a HiFi2 MATH_FIDELITY define) run T times with the state carried in L1. Everything in the anonymous
// namespace below is that file's helpers VERBATIM and each token runs that file's step ops in the same order on the
// same CBs. Between tokens the new fp32 state is packed to the IO dtype (cb_carry) and unpacked back, which is the
// exact rounding the T-sequential verify applies when it writes hist_t to DRAM and the next launch reads it.
//
// MODE 0 (verify): tokens 0..T-1 of every head instance; the state comes from the reader ONCE (cb_state, the
//   pre-verify state); o of every token goes to the writer (cb_out). NO state leaves the core: the carry stays in
//   L1 and the last token's state is dropped. (MULTITOKEN wrote every per-token state; the sequential verify wrote
//   T states and read T.)
// MODE 1 (commit): per head instance the reader publishes the row's token count n (cb_cnt, one uint32: n = idx+1
//   for an accepted depth idx < T, 0 for SKIP / idx >= T) and feeds only tokens 0..n-1. The state after token n-1
//   is packed to cb_sout and written over the live state by the writer. The update of token t touches the state
//   only through tokens < n, so the committed bytes equal the sequential verify's hist_{n-1} (the slot commit's
//   selected candidate) by construction. The q chain and o are not computed: the state update never reads them.
//   n == 0: nothing is read, computed or written for that instance.
// SO_T0=1 (QWEN36_MTP_GDN_SO_T0), MODE 0: per instance the reader publishes WT (cb_cnt); after token WT the new state
//   is ALSO packed to cb_sout (the writer puts it over rec_state row b). It is packed from the same fp32 cb_snew by
//   the same copy_tiles as the carry, so the bytes in DRAM are the carry's bytes; the commit's token 1 then unpacks
//   them from cb_state exactly as the verify's token 1 unpacks cb_carry. MODE 1 is unchanged (the reader feeds
//   tokens 1..idx from the updated row).
//
// SO_SFPU_ROW0 (QWEN36_MTP_GDN_SO_SFPU_ROW0, a define prepended by state_once.compute_prefix; absent = today's text):
//   inv_rms and expc run their SFPU chain on iteration 0 of face 0 only (VectorMode::None, ITERATIONS=1: the 32 lanes
//   holding element [0,0]) instead of the whole tile. Their consumers read only [0,0] (bcast scalar) and column 0
//   (bcast cols) and multiply q/k tiles whose rows 1..31 are +0; the untouched rows of the inv_rms tile keep the row
//   sums of those zero rows (+0), so every output byte is today's (op-anatomy-gdn-state-once-20261006T045611Z: the
//   three full-tile passes were 4.5 of ~15.5 us per token on the math RISC). SO_SFPU_ROW0_MUSTFAIL: expc skips exp.
//
// SO_FUSE (QWEN36_MTP_GDN_SO_FUSE, defines prepended by state_once.compute_prefix; absent = today's text): the
//   per-token state chain keeps its intermediates in DST instead of packing them to an fp32 CB and unpacking them
//   for the next op (op-anatomy-gdn-state-once-20261006T045611Z: the compute is busy back to back, ~0.2 us fixed per
//   dependent op, fp32 pack ~64 cycles/tile). Every FPU op keeps its operands in the SAME source register (srcA/srcB)
//   with the same values and fidelity; only the transport changes (DST -> Src by MOVD2B, or MOVD2A of bf16-exact
//   values only, with the Src format pinned to tf32, instead of pack fp32 -> unpack fp32->tf32). Parts (each its own define, SO_FUSE sets all four):
//   SO_FUSE_DECAY  state (io) copied into DST and multiplied there (dest reuse, srcA <- DST) by a full tile of exp(g)
//                  (one unary_bcast of the scalar, so srcB holds the value the bcast-scalar mul gave it): no cb_sf.
//   SO_FUSE_DELTA  v_read = kn @ h into DST, delta = v - v_read (srcB <- DST): one pack of Vt tiles instead of two
//                  ops; no cb_vread. (beta stays the stock op: MOVD2A of fp32 values was not bitwise in ttsim.)
//   SO_FUSE_UPD    the rank-1 outer product into DST, then h + outer (srcB <- DST): no cb_outer.
//   SO_FUSE_PACK   new_h packs straight from DST to every consumer: fp32 cb_snew (verify: o), io cb_carry (t+1 < n),
//                  io cb_sout (commit last token / SO_T0 token wt) instead of copy_tiles(cb_snew -> io) (needs UPD).
//                  The commit (MODE 1) packs no fp32 new_h at all.
//   SO_FUSE_MUSTFAIL (needs DELTA): the subtraction is skipped (delta = kn @ h, not v - kn @ h).
//   Bitwise only if DST -> Src moves round as the unpacker's fp32 -> tf32 does and the io pack from fp32 DST gives the
//   bytes copy_tiles gave: checked by ttsim (sim_mtp_gdn_state_once --fuse) and the warmup self-test, not assumed.
//
// SO_IOSRC (QWEN36_MTP_GDN_SO_IOSRC, a define prepended by state_once.compute_prefix; absent = today's text):
//   the io-dtype operands the SFPU consumes are read straight from their io CB: the recurrent state (cb_state /
//   cb_carry) is the srcA of the decay's bcast_scalar_mul, and g (cb_g) is the input of the exp. Their fp32 mirror
//   copies (cb_sf: kv = 16 tiles per token; cb_gf: 1 tile) are not made: the unpacker converts io -> fp32 into the
//   DST exactly, so every product is bitwise the mirror's. The served path already does this for the delta
//   (bcast_scalar_mul(cb_delta (io), cb_betaf (fp32))), so the mixed io-srcA / fp32-srcB SFPU pair is a served,
//   bitwise-verified shape. The beta mirror (cb_betaf) stays: the SFPU reads srcB directly, so srcB must be fp32.
//   SO_IOSRC_MUSTFAIL: the decay's srcA tile index is rotated by one (a wrong-input branch the warmup self-test
//   must catch).
//
// SO_QKH (QWEN36_MTP_GDN_SO_QKH, MODE 0, on the served SO_FUSE delta,upd,pack + SO_IOSRC body): the reader gives q, k
//   and v of all T tokens of an instance at once, token t at row u_t = t*BW + b (M = T*BW <= 32, one row page), and a
//   beta COLUMN tile per token (beta_t at (u_t, 0), +0 elsewhere). The q and k L2-norm chains (5 ops each, ~4.5 us of
//   ~10.3 us per token on the card, op-anatomy-gdn-state-once-20261006T045611Z) run ONCE per instance, inv_rms on the
//   full tile; kn -> kcol is transposed once. Per token: decay, delta = v - kn @ h (all rows), delta *= beta column
//   (only row u_t survives), new_h = h + kcol @ delta, o = qn @ new_h (row u_t). Row-independent tile ops plus exact
//   zeros: bitwise the per-token chain except the sign of an exact-zero update (checked by ttsim and the self-test).
//   SO_QKH_MUSTFAIL (reader): the last row's beta goes to the next token's row.
//
// Compile args: {Kt, Vt, EPS_BITS, SCALE_BITS, T, MODE, SO_T0}. Runtime args: {n_inst}.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/bcast.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose.h"
#include "api/compute/reconfig_data_format.h"
#if defined(SO_FUSE_DECAY) || defined(SO_FUSE_DELTA) || defined(SO_FUSE_UPD)
#define SO_FUSE_ANY 1
#endif
#if defined(SO_FUSE_PACK) && !defined(SO_FUSE_UPD)
#error "SO_FUSE_PACK needs SO_FUSE_UPD (new_h must still be in DST)"
#endif
#if defined(SO_QKH) && !(defined(SO_FUSE_DELTA) && defined(SO_FUSE_UPD) && defined(SO_FUSE_PACK) && defined(SO_IOSRC) && \
                          !defined(SO_FUSE_DECAY))
#error "SO_QKH is built on the served verify body: SO_FUSE delta,upd,pack + SO_IOSRC, without SO_FUSE_DECAY"
#endif
#include "api/dataflow/circular_buffer.h"

namespace {

constexpr uint32_t cb_q = 0, cb_k = 1, cb_v = 2, cb_g = 3, cb_beta = 4;
constexpr uint32_t cb_state = 5, cb_ones = 6;
constexpr uint32_t cb_qsq = 7, cb_ksq = 8;
constexpr uint32_t cb_sc = 9;    // 1 tile fp32 rowsum of squares
constexpr uint32_t cb_sc2 = 28;  // 1 tile fp32 q-chain inv-rms factor (dedicated)
constexpr uint32_t cb_sc3 = 29;  // 1 tile fp32 k-chain inv-rms factor (dedicated)
constexpr uint32_t cb_qn = 10, cb_kn = 11, cb_kcol = 12, cb_gexp = 13;
constexpr uint32_t cb_sdec = 14, cb_vread = 15, cb_delta = 16;
constexpr uint32_t cb_outer = 17, cb_sout = 18, cb_out = 19;
// fp32 mirrors of the io-dtype inputs (every math operand must be fp32:
// mixed bf16-srcA x fp32-srcB pairs corrupt the fp32 side) + fp32 new state.
constexpr uint32_t cb_qf = 20, cb_kf = 21, cb_vf = 22, cb_gf = 23;
constexpr uint32_t cb_betaf = 24, cb_sf = 25, cb_snew = 26;
// The io-dtype state carried from token t to t+1 (compute is its only producer
// and only consumer; cb_state stays the reader's).
constexpr uint32_t cb_carry = 30;
// MODE 1: per-instance token count published by the reader (one uint32 in a 16-byte page).
constexpr uint32_t cb_cnt = 31;
// SO_FUSE_DECAY: a full tile of the decay factor exp(g) (every element = the scalar at [0,0]).
constexpr uint32_t cb_gbc = 34;

inline void WAIT(uint32_t cb, uint32_t n) { CircularBuffer(cb).wait_front(n); }
inline void POP(uint32_t cb, uint32_t n) { CircularBuffer(cb).pop_front(n); }

// out[Mt,Nt] = A[Mt,Kt] @ (tr ? B[Nt,Kt]^T : B[Kt,Nt]). Inputs must be available.
void mm(uint32_t a, uint32_t b, uint32_t o, uint32_t Mt, uint32_t Kt, uint32_t Nt, bool tr) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    // matmul_tiles(a,b): in0=a->srcA, in1=b->srcB (reconfig src formats to match).
    reconfig_data_format(a, b);
    matmul_init(a, b, tr ? 1 : 0);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            for (uint32_t ki = 0; ki < Kt; ki++) {
                uint32_t bi = tr ? (ni * Kt + ki) : (ki * Nt + ni);
                matmul_tiles(a, b, mi * Kt + ki, bi, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// out = A (op) B elementwise, n tiles. op: 0 add, 1 sub, 2 mul.
void ew(uint32_t a, uint32_t b, uint32_t o, uint32_t n, int op) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, b);  // binary(a,b): a->srcA, b->srcB
    if (op == 0) {
        add_init(a, b);
    } else if (op == 1) {
        sub_init(a, b);
    } else {
        mul_init(a, b);
    }
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        if (op == 0) {
            add_tiles(a, b, i, i, 0);
        } else if (op == 1) {
            sub_tiles(a, b, i, i, 0);
        } else {
            mul_tiles(a, b, i, i, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = exp(in), n tiles (copy to fp32 DST, SFPU exp).
void expc(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    exp_tile_init();
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
#if defined(SO_SFPU_ROW0_MUSTFAIL)
        // must-fail: no exp (the decay multiplies by g)
#elif defined(SO_SFPU_ROW0)
        // only [0,0] is read (mul_tiles_bcast_scalar): SFPU iteration 0 of face 0
        exp_tile<false, false, ckernel::InputClamping::ClampToNegative, 1>(0, ckernel::VectorMode::None);
#else
        exp_tile(0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out = S * scalar[0,0], n tiles.
void bcast_scalar_mul(uint32_t a, uint32_t scal, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, scal);  // bcast(a,scal): a->srcA, scal->srcB
    mul_bcast_scalar_init(a, scal);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
#if defined(SO_IOSRC_MUSTFAIL)
        // must-fail: the srcA tile index is rotated by one (the state's tiles are shifted) -- a wrong-input branch
        // the warmup self-test must catch. The offset is a COMPILE-time 1 so the served path keeps today's index
        // expression verbatim (a runtime offset would put a division in this loop).
        mul_tiles_bcast_scalar(a, scal, (i + 1) % n, 0, 0);
#else
        mul_tiles_bcast_scalar(a, scal, i, 0, 0);
#endif
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// out[Mt,Nt] = A[Mt,Nt] * col[Mt,1] (broadcast col's single column across N).
void bcast_cols_mul(uint32_t a, uint32_t col, uint32_t o, uint32_t Mt, uint32_t Nt) {
    cb_reserve_back(o, Mt * Nt);
    pack_reconfig_data_format(o);
    reconfig_data_format(a, col);  // bcast(a,col): a->srcA, col->srcB
    mul_bcast_cols_init(a, col);
    for (uint32_t mi = 0; mi < Mt; mi++) {
        for (uint32_t ni = 0; ni < Nt; ni++) {
            tile_regs_acquire();
            mul_tiles_bcast_cols(a, col, mi * Nt + ni, mi, 0);
            tile_regs_commit();
            tile_regs_wait();
            pack_tile(0, o, mi * Nt + ni);
            tile_regs_release();
        }
    }
    cb_push_back(o, Mt * Nt);
}

// rowsum_k: o[1 tile] = in[1,Kt] @ ones — every element [r,c] of the output
// holds the row-r sum (row 0 = this head's sum of squares of `in`). Reuses the
// reader-built fp32 ones tile as the contraction operand.
void rowsum_k(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format(in, cb_ones);  // matmul(in, ones): in->srcA, ones->srcB
    matmul_init(in, cb_ones, 0);
    tile_regs_acquire();
    for (uint32_t ki = 0; ki < Kt; ki++) {
        matmul_tiles(in, cb_ones, ki, 0, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

// inv_rms: o = rsqrt(in + eps) [* scale]. in holds per-row sums of squares
// (rowsum_k output); every column of a row carries the same factor, so the
// result feeds bcast_cols_mul directly. eps/scale arrive as fp32-bit-cast
// uint32 compile args. in (cb_sc) and o (cb_sc2) must be DISTINCT CBs: with
// in==out on the 2-page ring, the second norm chain's bcast consumed the
// first chain's factor (ttsim: gdn_decode_simdiag3.py swap experiment).
void inv_rms(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
#if defined(SO_SFPU_ROW0)
    // only column 0 is read (mul_tiles_bcast_cols) against q/k rows that are +0 below row 0: the SFPU chain on iteration
    // 0 of face 0 (the lanes of [0,0]); add_unary_tile / rsqrt_tile / mul_unary_tile with ITERATIONS=1, VectorMode::None
    binop_with_scalar_tile_init();
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
                         (APPROX, ckernel::ADD_UNARY, 1, DST_ACCUM_MODE), 0, ckernel::VectorMode::None, eps_bits));
    rsqrt_tile_init();
    MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_rsqrt, (APPROX, 1, DST_ACCUM_MODE, false, false), 0,
                         ckernel::VectorMode::None));
    if (do_scale) {
        binop_with_scalar_tile_init();
        MATH(SFPU_UNARY_CALL(DST_SYNC_MODE, DST_ACCUM_MODE, calculate_binop_with_scalar,
                             (APPROX, ckernel::MUL_UNARY, 1, DST_ACCUM_MODE), 0, ckernel::VectorMode::None, scale_bits));
    }
#else
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);  // + eps  (== python rms_norm(x, eps/K))
    rsqrt_tile_init();
    rsqrt_tile(0);  // 1/sqrt(sumsq + eps)
    if (do_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);  // * scale (q only; == python q*K**-0.5)
    }
#endif
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}

#if defined(SO_QKH)
// SO_QKH: inv_rms on the WHOLE tile (every row is a token's factor), the stock full-tile SFPU path. SO_SFPU_ROW0 runs
// the same SFPU code on the lanes of [0,0] only and was gated bitwise against this path, so row u's factor here is the
// factor row 0 gets today for the same row.
void inv_rms_full(uint32_t in, uint32_t o, uint32_t eps_bits, uint32_t scale_bits, bool do_scale) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    tile_regs_acquire();
    copy_tile(in, 0, 0);
    binop_with_scalar_tile_init();
    add_unary_tile(0, eps_bits);
    rsqrt_tile_init();
    rsqrt_tile(0);
    if (do_scale) {
        binop_with_scalar_tile_init();
        mul_unary_tile(0, scale_bits);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
#endif

// o = copy of the n tiles of `in` (format conversion happens on the copy:
// unpack converts io->fp32 dest, packer converts fp32 dest->io).
void copy_tiles(uint32_t in, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    copy_tile_to_dst_init_short(in);
    for (uint32_t i = 0; i < n; i++) {
        tile_regs_acquire();
        copy_tile(in, i, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, i);
        tile_regs_release();
    }
    cb_push_back(o, n);
}

// o[Kt,1] = transpose(in[1,Kt]): tile ki of o is the transposed tile ki of in
// (row 0 -> column 0), the column form needed for the rank-1 outer product.
void transpose_row(uint32_t in, uint32_t o, uint32_t Kt) {
    cb_reserve_back(o, Kt);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    transpose_init(in);
    for (uint32_t ki = 0; ki < Kt; ki++) {
        tile_regs_acquire();
        transpose_tile(in, ki, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, o, ki);
        tile_regs_release();
    }
    cb_push_back(o, Kt);
}


#if defined(SO_FUSE_ANY)
// SrcA/SrcB format pinned to TF32 around the dest-reuse ops (tt-metal #58974 src_format_enter/leave): with the implied
// per-bank format a MOVD2A/MOVD2B from 32-bit DST truncates to whatever format the bank last held (bf16 after a bf16
// unpack), not to tf32 as the unpacker's fp32 -> tf32 does. MATH thread only.
inline void src_fmt_pin() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Tf32));
    TTI_SETC16(DISABLE_IMPLIED_SRCB_FMT_Base_ADDR32, 1);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(to_underlying(DataFormat::Tf32));
#endif
}
inline void src_fmt_unpin() {
#ifdef TRISC_MATH
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG0_SrcA_RMW>(to_underlying(DataFormat::Float32));
    TTI_SETC16(DISABLE_IMPLIED_SRCA_FMT_Base_ADDR32, 0);
    cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG1_SrcB_RMW>(to_underlying(DataFormat::Float32));
    TTI_SETC16(DISABLE_IMPLIED_SRCB_FMT_Base_ADDR32, 0);
#endif
}
#endif

#if defined(SO_FUSE_DECAY)
// o = a full tile of in[0,0] (unary_bcast SCALAR: fp32 in -> unpack to DST, exact), 1 tile fp32.
void bcast_full(uint32_t in, uint32_t o) {
    cb_reserve_back(o, 1);
    pack_reconfig_data_format(o);
    reconfig_data_format_srca(in);
    unary_bcast_init<BroadcastType::SCALAR>(in);
    tile_regs_acquire();
    unary_bcast<BroadcastType::SCALAR>(in, 0, 0);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, o, 0);
    tile_regs_release();
    cb_push_back(o, 1);
}
#endif

#if defined(SO_FUSE_DECAY)
// o = s_in (io) * gbc, kv tiles, 4 per DST section: copy_tile io -> DST (exact), then ELWMUL with srcA <- DST and
// srcB <- gbc (the value bcast_scalar_mul put in srcB). Replaces copy_tiles(s_in -> cb_sf) + bcast_scalar_mul.
void decay_fused(uint32_t s_in, uint32_t gbc, uint32_t o, uint32_t n) {
    cb_reserve_back(o, n);
    pack_reconfig_data_format(o);
    for (uint32_t i0 = 0; i0 < n; i0 += 4) {
        tile_regs_acquire();
        reconfig_data_format_srca(s_in);
        copy_tile_to_dst_init_short(s_in);
        for (uint32_t j = 0; j < 4; j++) {
            copy_tile(s_in, i0 + j, j);
        }
        reconfig_data_format(gbc, gbc);  // srcA (from DST) and srcB as the fp32 operands of bcast_scalar_mul
        mul_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gbc);
        src_fmt_pin();
        for (uint32_t j = 0; j < 4; j++) {
            mul_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gbc, 0, j);
        }
        src_fmt_unpin();
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t j = 0; j < 4; j++) {
            pack_tile(j, o, i0 + j);
        }
        tile_regs_release();
    }
    cb_push_back(o, n);
}
#endif

#if defined(SO_FUSE_DELTA)
// o[1,Vt] = v - kn @ h in one DST section (Vt <= 4): mm as mm(kn, h, ., 1, Kt, Vt) per output tile, then ELWSUB
// srcA <- v, srcB <- DST (ew(v, vread, ., sub)). The beta multiply stays the stock bcast_scalar_mul: with beta in srcB
// it needs delta in srcA, and MOVD2A from fp32 DST did not reproduce the unpacker's tf32 operand (ttsim, 2026-10-06:
// the beta-in-DST variant differed in o; MOVD2B paths were bitwise), so delta goes through L1 for that one op.
void delta_fused(uint32_t kn, uint32_t h, uint32_t v, uint32_t o, uint32_t Kt, uint32_t Vt) {
    cb_reserve_back(o, Vt);
    pack_reconfig_data_format(o);
    tile_regs_acquire();
    reconfig_data_format(kn, h);
    matmul_init(kn, h, 0);
    for (uint32_t ni = 0; ni < Vt; ni++) {
        for (uint32_t ki = 0; ki < Kt; ki++) {
            matmul_tiles(kn, h, ki, ki * Vt + ni, ni);
        }
    }
#if !defined(SO_FUSE_MUSTFAIL)
    reconfig_data_format(v, v);
    sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(v);
    src_fmt_pin();
    for (uint32_t ni = 0; ni < Vt; ni++) {
        sub_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(v, ni, ni);
    }
    src_fmt_unpin();
#endif
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t ni = 0; ni < Vt; ni++) {
        pack_tile(ni, o, ni);
    }
    tile_regs_release();
    cb_push_back(o, Vt);
}
#endif

#if defined(SO_FUSE_UPD)
// new_h = h + kcol @ delta, one DST section per kcol tile (Vt <= 4 outputs): the outer tiles as mm(kcol, delta, .,
// Kt, 1, Vt) gives them, then ELWADD srcA <- h, srcB <- DST (ew(h, outer, ., add)). Packed to up to three CBs:
// o_f32 (fp32 new_h), o_io1 / o_io2 (io copies; copy_tiles(cb_snew -> io) without SO_FUSE_PACK). 0xFFFFFFFF = none.
void upd_fused(uint32_t kcol, uint32_t delta, uint32_t h, uint32_t o_f32, uint32_t o_io1, uint32_t o_io2, uint32_t Kt,
               uint32_t Vt) {
    constexpr uint32_t NONE = 0xFFFFFFFFu;
    const uint32_t kv = Kt * Vt;
    if (o_f32 != NONE) {
        cb_reserve_back(o_f32, kv);
    }
    if (o_io1 != NONE) {
        cb_reserve_back(o_io1, kv);
    }
    if (o_io2 != NONE) {
        cb_reserve_back(o_io2, kv);
    }
    for (uint32_t ki = 0; ki < Kt; ki++) {
        tile_regs_acquire();
        reconfig_data_format(kcol, delta);
        matmul_init(kcol, delta, 0);
        for (uint32_t ni = 0; ni < Vt; ni++) {
            matmul_tiles(kcol, delta, ki, ni, ni);
        }
        reconfig_data_format(h, h);
        add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(h);
        src_fmt_pin();
        for (uint32_t ni = 0; ni < Vt; ni++) {
            add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(h, ki * Vt + ni, ni);
        }
        src_fmt_unpin();
        tile_regs_commit();
        tile_regs_wait();
        if (o_f32 != NONE) {
            pack_reconfig_data_format(o_f32);
            for (uint32_t ni = 0; ni < Vt; ni++) {
                pack_tile(ni, o_f32, ki * Vt + ni);
            }
        }
        if (o_io1 != NONE) {
            pack_reconfig_data_format(o_io1);
            for (uint32_t ni = 0; ni < Vt; ni++) {
                pack_tile(ni, o_io1, ki * Vt + ni);
            }
        }
        if (o_io2 != NONE) {
            pack_reconfig_data_format(o_io2);
            for (uint32_t ni = 0; ni < Vt; ni++) {
                pack_tile(ni, o_io2, ki * Vt + ni);
            }
        }
        tile_regs_release();
    }
    if (o_f32 != NONE) {
        cb_push_back(o_f32, kv);
    }
    if (o_io1 != NONE) {
        cb_push_back(o_io1, kv);
    }
    if (o_io2 != NONE) {
        cb_push_back(o_io2, kv);
    }
}
#endif

}  // namespace

void kernel_main() {
    constexpr uint32_t Kt = get_compile_time_arg_val(0);
    constexpr uint32_t Vt = get_compile_time_arg_val(1);
    constexpr uint32_t EPS_BITS = get_compile_time_arg_val(2);
    constexpr uint32_t SCALE_BITS = get_compile_time_arg_val(3);
    constexpr uint32_t T = get_compile_time_arg_val(4);
    constexpr uint32_t MODE = get_compile_time_arg_val(5);  // 0 verify, 1 commit
    constexpr uint32_t SO_T0 = get_compile_time_arg_val(6);
    constexpr bool VERIFY = MODE == 0;

    constexpr uint32_t kv = Kt * Vt;

    const uint32_t n_inst = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(cb_q, cb_k, cb_out);

    for (uint32_t it = 0; it < n_inst; ++it) {
        uint32_t n = T;
        uint32_t wt = 0xFFFFFFFFu;  // SO_T0 verify: the token whose state goes to rec_state (none otherwise)
        if constexpr (!VERIFY) {
            cb_wait_front(cb_cnt, 1);
            n = read_tile_value(cb_cnt, 0, 0);  // unpack reads, math/pack get it by mailbox: one value on all three
            cb_pop_front(cb_cnt, 1);
        } else if constexpr (SO_T0 != 0) {
            cb_wait_front(cb_cnt, 1);
            wt = read_tile_value(cb_cnt, 0, 0);
            cb_pop_front(cb_cnt, 1);
        }
#if defined(SO_QKH)
        // ---- SO_QKH: q/k L2 norms ONCE per instance (every token's row at u_t = t*BW + b), then T state steps ----
        // Every tile op below is row-independent (FPU matmul/eltwise rows, full-tile SFPU), so token t's row of qn/kn
        // is the row 0 the per-token chain gives. delta row u_t = v_t - k_t @ h_t; the beta COLUMN (beta_t at row u_t,
        // +0 elsewhere) zeroes every other row, so the rank-1 update kcol @ delta adds only k_t (x) delta_t plus exact
        // zeros. o row u_t = q_t @ new_h_t; the writer reads that row.
        {
            static_assert(VERIFY, "SO_QKH is verify-only");
            constexpr uint32_t NONE = 0xFFFFFFFFu;
            WAIT(cb_q, Kt);
            WAIT(cb_k, Kt);
            WAIT(cb_v, Vt);
            WAIT(cb_ones, 1);
            copy_tiles(cb_q, cb_qf, Kt);
            copy_tiles(cb_k, cb_kf, Kt);
            copy_tiles(cb_v, cb_vf, Vt);
            POP(cb_q, Kt);
            POP(cb_k, Kt);
            POP(cb_v, Vt);
            WAIT(cb_qf, Kt);
            WAIT(cb_kf, Kt);
            WAIT(cb_vf, Vt);

            ew(cb_qf, cb_qf, cb_qsq, Kt, 2);
            WAIT(cb_qsq, Kt);
            rowsum_k(cb_qsq, cb_sc, Kt);
            WAIT(cb_sc, 1);
            POP(cb_qsq, Kt);
            inv_rms_full(cb_sc, cb_sc2, EPS_BITS, SCALE_BITS, true);
            POP(cb_sc, 1);
            WAIT(cb_sc2, 1);
            bcast_cols_mul(cb_qf, cb_sc2, cb_qn, 1, Kt);
            WAIT(cb_qn, Kt);
            POP(cb_qf, Kt);
            POP(cb_sc2, 1);

            ew(cb_kf, cb_kf, cb_ksq, Kt, 2);
            WAIT(cb_ksq, Kt);
            rowsum_k(cb_ksq, cb_sc, Kt);
            WAIT(cb_sc, 1);
            POP(cb_ksq, Kt);
            inv_rms_full(cb_sc, cb_sc3, EPS_BITS, SCALE_BITS, false);
            POP(cb_sc, 1);
            WAIT(cb_sc3, 1);
            bcast_cols_mul(cb_kf, cb_sc3, cb_kn, 1, Kt);
            WAIT(cb_kn, Kt);
            POP(cb_kf, Kt);
            POP(cb_sc3, 1);

            transpose_row(cb_kn, cb_kcol, Kt);  // column u of kcol = row u of kn (every token at once)
            WAIT(cb_kcol, Kt);

            for (uint32_t t = 0; t < T; ++t) {
                const uint32_t cb_s_in = (t == 0) ? cb_state : cb_carry;
                WAIT(cb_g, 1);
                WAIT(cb_beta, 1);
                copy_tiles(cb_beta, cb_betaf, 1);
                POP(cb_beta, 1);
                WAIT(cb_betaf, 1);

                expc(cb_g, cb_gexp, 1);
                WAIT(cb_gexp, 1);
                POP(cb_g, 1);
                WAIT(cb_s_in, kv);
                bcast_scalar_mul(cb_s_in, cb_gexp, cb_sdec, kv);
                WAIT(cb_sdec, kv);
                POP(cb_s_in, kv);
                POP(cb_gexp, 1);

                delta_fused(cb_kn, cb_sdec, cb_vf, cb_delta, Kt, Vt);
                WAIT(cb_delta, Vt);
                bcast_cols_mul(cb_delta, cb_betaf, cb_delta, 1, Vt);
                POP(cb_betaf, 1);
                POP(cb_delta, Vt);
                WAIT(cb_delta, Vt);

                {
                    const uint32_t o_io1 = (t + 1 < T) ? cb_carry : NONE;
                    uint32_t o_io2 = NONE;
                    if constexpr (SO_T0 != 0) {
                        if (t == wt) {
                            o_io2 = cb_sout;
                        }
                    }
                    upd_fused(cb_kcol, cb_delta, cb_sdec, cb_snew, o_io1, o_io2, Kt, Vt);
                }
                WAIT(cb_snew, kv);
                POP(cb_delta, Vt);
                POP(cb_sdec, kv);

                mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
                POP(cb_snew, kv);
            }
            POP(cb_qn, Kt);
            POP(cb_kn, Kt);
            POP(cb_kcol, Kt);
            POP(cb_vf, Vt);
        }
        continue;
#endif
        for (uint32_t t = 0; t < n; ++t) {
#if defined(SO_FUSE_ANY)
            constexpr uint32_t NONE = 0xFFFFFFFFu;
            const uint32_t cb_s_in = (t == 0) ? cb_state : cb_carry;
            if constexpr (VERIFY) {
                WAIT(cb_q, Kt);
            }
            WAIT(cb_k, Kt);
            WAIT(cb_v, Vt);
            WAIT(cb_g, 1);
            WAIT(cb_beta, 1);
            WAIT(cb_ones, 1);
#if !defined(SO_FUSE_DECAY)
            WAIT(cb_s_in, kv);
#endif

            // ---- io -> fp32 (stock; DECAY: the state goes io -> DST inside the decay; IOSRC: the SFPU reads the io
            // state and the io g directly, so their fp32 mirror copies are not made) ----
            if constexpr (VERIFY) {
                copy_tiles(cb_q, cb_qf, Kt);
            }
            copy_tiles(cb_k, cb_kf, Kt);
            copy_tiles(cb_v, cb_vf, Vt);
#if !defined(SO_IOSRC)
            copy_tiles(cb_g, cb_gf, 1);
#endif
            copy_tiles(cb_beta, cb_betaf, 1);
#if !defined(SO_FUSE_DECAY) && !defined(SO_IOSRC)
            copy_tiles(cb_s_in, cb_sf, kv);
#endif
            if constexpr (VERIFY) {
                POP(cb_q, Kt);
            }
            POP(cb_k, Kt);
            POP(cb_v, Vt);
#if !defined(SO_IOSRC)
            POP(cb_g, 1);
#endif
            POP(cb_beta, 1);
#if !defined(SO_FUSE_DECAY) && !defined(SO_IOSRC)
            POP(cb_s_in, kv);
#endif

            if constexpr (VERIFY) {
                WAIT(cb_qf, Kt);
            }
            WAIT(cb_kf, Kt);
            WAIT(cb_vf, Vt);
#if !defined(SO_IOSRC)
            WAIT(cb_gf, 1);
#endif
            WAIT(cb_betaf, 1);
#if !defined(SO_FUSE_DECAY) && !defined(SO_IOSRC)
            WAIT(cb_sf, kv);
#endif

            // ---- L2 norm q (scale folded): o only ----
            if constexpr (VERIFY) {
                ew(cb_qf, cb_qf, cb_qsq, Kt, 2);
                WAIT(cb_qsq, Kt);
                rowsum_k(cb_qsq, cb_sc, Kt);
                WAIT(cb_sc, 1);
                POP(cb_qsq, Kt);
                inv_rms(cb_sc, cb_sc2, EPS_BITS, SCALE_BITS, true);
                POP(cb_sc, 1);
                WAIT(cb_sc2, 1);
                bcast_cols_mul(cb_qf, cb_sc2, cb_qn, 1, Kt);
                WAIT(cb_qn, Kt);
                POP(cb_qf, Kt);
                POP(cb_sc2, 1);
            }

            // ---- L2 norm k (no scale) ----
            ew(cb_kf, cb_kf, cb_ksq, Kt, 2);
            WAIT(cb_ksq, Kt);
            rowsum_k(cb_ksq, cb_sc, Kt);
            WAIT(cb_sc, 1);
            POP(cb_ksq, Kt);
            inv_rms(cb_sc, cb_sc3, EPS_BITS, SCALE_BITS, false);
            POP(cb_sc, 1);
            WAIT(cb_sc3, 1);
            bcast_cols_mul(cb_kf, cb_sc3, cb_kn, 1, Kt);
            WAIT(cb_kn, Kt);
            POP(cb_kf, Kt);
            POP(cb_sc3, 1);

            // ---- decay: h = state * exp(g) ----
#if defined(SO_IOSRC)
            // the io g goes straight into the SFPU's DST (the unpacker converts io -> fp32 exactly)
            expc(cb_g, cb_gexp, 1);
            WAIT(cb_gexp, 1);
            POP(cb_g, 1);
#else
            expc(cb_gf, cb_gexp, 1);
            WAIT(cb_gexp, 1);
            POP(cb_gf, 1);
#endif
#if defined(SO_FUSE_DECAY)
            bcast_full(cb_gexp, cb_gbc);
            WAIT(cb_gbc, 1);
            POP(cb_gexp, 1);
            WAIT(cb_s_in, kv);
            decay_fused(cb_s_in, cb_gbc, cb_sdec, kv);
            WAIT(cb_sdec, kv);
            POP(cb_s_in, kv);
            POP(cb_gbc, 1);
#else
#if defined(SO_IOSRC)
            // the io state is the SFPU's srcA directly (the unpacker converts io -> fp32 exactly, so the product is
            // bitwise the fp32 mirror's); SO_IOSRC_MUSTFAIL rotates the state's tiles (a wrong-input branch)
            bcast_scalar_mul(cb_s_in, cb_gexp, cb_sdec, kv);
            WAIT(cb_sdec, kv);
            POP(cb_s_in, kv);
#else
            bcast_scalar_mul(cb_sf, cb_gexp, cb_sdec, kv);
            WAIT(cb_sdec, kv);
            POP(cb_sf, kv);
#endif
            POP(cb_gexp, 1);
#endif

            // ---- delta = (v - kn @ h) * beta ----
#if defined(SO_FUSE_DELTA)
            delta_fused(cb_kn, cb_sdec, cb_vf, cb_delta, Kt, Vt);
            WAIT(cb_delta, Vt);
            POP(cb_vf, Vt);
            bcast_scalar_mul(cb_delta, cb_betaf, cb_delta, Vt);
            POP(cb_betaf, 1);
            POP(cb_delta, Vt);
#else
            mm(cb_kn, cb_sdec, cb_vread, 1, Kt, Vt, false);
            WAIT(cb_vread, Vt);
            ew(cb_vf, cb_vread, cb_delta, Vt, 1);
            WAIT(cb_delta, Vt);
            POP(cb_vf, Vt);
            POP(cb_vread, Vt);
            bcast_scalar_mul(cb_delta, cb_betaf, cb_delta, Vt);
            POP(cb_betaf, 1);
            POP(cb_delta, Vt);
#endif

            // ---- rank-1 write: new_h = h + (kn)^T @ (beta*delta) ----
            transpose_row(cb_kn, cb_kcol, Kt);
            WAIT(cb_kcol, Kt);
            POP(cb_kn, Kt);
#if defined(SO_FUSE_UPD)
            {
                uint32_t o_f32 = cb_snew, o_io1 = NONE, o_io2 = NONE;
#if defined(SO_FUSE_PACK)
                if constexpr (!VERIFY) {
                    o_f32 = NONE;  // the commit never reads the fp32 new_h
                }
                if (t + 1 < n) {
                    o_io1 = cb_carry;
                } else if constexpr (!VERIFY) {
                    o_io1 = cb_sout;
                }
                if constexpr (VERIFY && SO_T0 != 0) {
                    if (t == wt) {
                        o_io2 = cb_sout;
                    }
                }
#endif
                upd_fused(cb_kcol, cb_delta, cb_sdec, o_f32, o_io1, o_io2, Kt, Vt);
            }
#if !defined(SO_FUSE_PACK)
            WAIT(cb_snew, kv);
#else
            if constexpr (VERIFY) {
                WAIT(cb_snew, kv);
            }
#endif
            POP(cb_delta, Vt);
            POP(cb_kcol, Kt);
            POP(cb_sdec, kv);
#else
            mm(cb_kcol, cb_delta, cb_outer, Kt, 1, Vt, false);
            WAIT(cb_outer, kv);
            POP(cb_delta, Vt);
            POP(cb_kcol, Kt);
            ew(cb_sdec, cb_outer, cb_snew, kv, 0);
            WAIT(cb_snew, kv);
            POP(cb_sdec, kv);
            POP(cb_outer, kv);
#endif

            // ---- o = qn @ new_h (verify only) ----
            if constexpr (VERIFY) {
                mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
                POP(cb_qn, Kt);
            }

#if defined(SO_FUSE_PACK)
            if constexpr (VERIFY) {
                POP(cb_snew, kv);
            }
#else
            if constexpr (VERIFY && SO_T0 != 0) {
                if (t == wt) {
                    copy_tiles(cb_snew, cb_sout, kv);
                }
            }
            if (t + 1 < n) {
                copy_tiles(cb_snew, cb_carry, kv);
            } else if constexpr (!VERIFY) {
                copy_tiles(cb_snew, cb_sout, kv);
            }
            POP(cb_snew, kv);
#endif
#else  // !SO_FUSE_ANY: today's token body
            const uint32_t cb_s_in = (t == 0) ? cb_state : cb_carry;
            if constexpr (VERIFY) {
                WAIT(cb_q, Kt);
            }
            WAIT(cb_k, Kt);
            WAIT(cb_v, Vt);
            WAIT(cb_g, 1);
            WAIT(cb_beta, 1);
            WAIT(cb_s_in, kv);
            WAIT(cb_ones, 1);

            // ---- io -> fp32 (stock; IOSRC: the SFPU reads the io state and the io g directly) ----
            if constexpr (VERIFY) {
                copy_tiles(cb_q, cb_qf, Kt);
            }
            copy_tiles(cb_k, cb_kf, Kt);
            copy_tiles(cb_v, cb_vf, Vt);
#if !defined(SO_IOSRC)
            copy_tiles(cb_g, cb_gf, 1);
#endif
            copy_tiles(cb_beta, cb_betaf, 1);
#if !defined(SO_IOSRC)
            copy_tiles(cb_s_in, cb_sf, kv);
#endif
            if constexpr (VERIFY) {
                POP(cb_q, Kt);
            }
            POP(cb_k, Kt);
            POP(cb_v, Vt);
#if !defined(SO_IOSRC)
            POP(cb_g, 1);
#endif
            POP(cb_beta, 1);
#if !defined(SO_IOSRC)
            POP(cb_s_in, kv);
#endif

            if constexpr (VERIFY) {
                WAIT(cb_qf, Kt);
            }
            WAIT(cb_kf, Kt);
            WAIT(cb_vf, Vt);
#if !defined(SO_IOSRC)
            WAIT(cb_gf, 1);
#endif
            WAIT(cb_betaf, 1);
#if !defined(SO_IOSRC)
            WAIT(cb_sf, kv);
#endif

            // ---- L2 norm q (scale folded): o only ----
            if constexpr (VERIFY) {
                ew(cb_qf, cb_qf, cb_qsq, Kt, 2);
                WAIT(cb_qsq, Kt);
                rowsum_k(cb_qsq, cb_sc, Kt);
                WAIT(cb_sc, 1);
                POP(cb_qsq, Kt);
                inv_rms(cb_sc, cb_sc2, EPS_BITS, SCALE_BITS, true);
                POP(cb_sc, 1);
                WAIT(cb_sc2, 1);
                bcast_cols_mul(cb_qf, cb_sc2, cb_qn, 1, Kt);
                WAIT(cb_qn, Kt);
                POP(cb_qf, Kt);
                POP(cb_sc2, 1);
            }

            // ---- L2 norm k (no scale) ----
            ew(cb_kf, cb_kf, cb_ksq, Kt, 2);
            WAIT(cb_ksq, Kt);
            rowsum_k(cb_ksq, cb_sc, Kt);
            WAIT(cb_sc, 1);
            POP(cb_ksq, Kt);
            inv_rms(cb_sc, cb_sc3, EPS_BITS, SCALE_BITS, false);
            POP(cb_sc, 1);
            WAIT(cb_sc3, 1);
            bcast_cols_mul(cb_kf, cb_sc3, cb_kn, 1, Kt);
            WAIT(cb_kn, Kt);
            POP(cb_kf, Kt);
            POP(cb_sc3, 1);

            // ---- decay: h = state * exp(g) ----
#if defined(SO_IOSRC)
            expc(cb_g, cb_gexp, 1);
            WAIT(cb_gexp, 1);
            POP(cb_g, 1);
#else
            expc(cb_gf, cb_gexp, 1);
            WAIT(cb_gexp, 1);
            POP(cb_gf, 1);
#endif
#if defined(SO_IOSRC)
            bcast_scalar_mul(cb_s_in, cb_gexp, cb_sdec, kv);
            WAIT(cb_sdec, kv);
            POP(cb_s_in, kv);
#else
            bcast_scalar_mul(cb_sf, cb_gexp, cb_sdec, kv);
            WAIT(cb_sdec, kv);
            POP(cb_sf, kv);
#endif
            POP(cb_gexp, 1);

            // ---- v_read = kn @ h; delta = v - v_read; delta *= beta ----
            mm(cb_kn, cb_sdec, cb_vread, 1, Kt, Vt, false);
            WAIT(cb_vread, Vt);
            ew(cb_vf, cb_vread, cb_delta, Vt, 1);
            WAIT(cb_delta, Vt);
            POP(cb_vf, Vt);
            POP(cb_vread, Vt);
            bcast_scalar_mul(cb_delta, cb_betaf, cb_delta, Vt);
            POP(cb_betaf, 1);
            POP(cb_delta, Vt);

            // ---- rank-1 write: new_h = h + (kn)^T @ (beta*delta) ----
            transpose_row(cb_kn, cb_kcol, Kt);
            WAIT(cb_kcol, Kt);
            POP(cb_kn, Kt);
            mm(cb_kcol, cb_delta, cb_outer, Kt, 1, Vt, false);
            WAIT(cb_outer, kv);
            POP(cb_delta, Vt);
            POP(cb_kcol, Kt);
            ew(cb_sdec, cb_outer, cb_snew, kv, 0);
            WAIT(cb_snew, kv);
            POP(cb_sdec, kv);
            POP(cb_outer, kv);

            // ---- o = qn @ new_h (verify only) ----
            if constexpr (VERIFY) {
                mm(cb_qn, cb_snew, cb_out, 1, Kt, Vt, false);
                POP(cb_qn, Kt);
            }

            // ---- SO_T0 verify: the state after token wt -> io for the writer (over rec_state row b) ----
            if constexpr (VERIFY && SO_T0 != 0) {
                if (t == wt) {
                    copy_tiles(cb_snew, cb_sout, kv);
                }
            }
            // ---- new state -> io: the carry for token t+1, or (commit, last token) the committed state ----
            if (t + 1 < n) {
                copy_tiles(cb_snew, cb_carry, kv);
            } else if constexpr (!VERIFY) {
                copy_tiles(cb_snew, cb_sout, kv);
            }
            POP(cb_snew, kv);
#endif  // SO_FUSE_ANY
        }
    }
}

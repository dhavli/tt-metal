// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_SPEC_SAMPLE_FUSED (fused_sampler.py): the cut, the pick and the greedy id of a sampled MTP step, one
// row per RISC, after the ONE all-gather of fs_topk.cpp's per-chip blocks.  Replaces the served merge (typecasts, a
// second all-gather, tilize, the 2K top-K, untilizes, the id gather), the cut (~30 programs), the pick (~10), the
// greedy argmax (~5) and the selects (~7 each).
//
// Input rows (ROW_MAJOR, one page per row):
//   g      [M, 4K]  float32 BITS: chip 0's (K values, K local ids) then chip 1's (fs_topk.cpp), each list descending
//   scale  [M, 2K]  float32 the per-row 1/T (column 0 is read)
//   rank   [M, 2K]  float32 the top_k prefix mask (1.0 for the first top_k columns; all ones = no top_k)
//   topp   [M, 1]   float32 the top_p threshold
//   noise  [M, 2K]  float32 the pre-scaled Gumbel noise T*g, in the MERGED column order (chip 0's K then chip 1's K)
//   flag   [M, 1]   float32 1.0 = emit the sample, 0.0 = emit the greedy id
//   x, xf  [M, 1]   float32 (OVR only, the verify) the draft id and its flag: xf != 0 emits x instead
// Output rows:
//   probs  [M, 2K]  float32 the post-cut probabilities in the sorted order (0 off the support)
//   ids    [M, 2K]  float32 the GLOBAL ids in the sorted order (chip 1's local id + PER)
//   sel    [M, 1]   uint32  the emitted id: OVR && xf ? x : (flag ? pick : greedy)
//   aux    [M, 4]   uint32  (DBG only) {sel, pick, greedy, kept | topk_kept << 16}
//
// THE ORDER (vLLM 0.26, the served spec_sample.cut): temperature -> top_k (ties at the k-th value kept) -> softmax over
// the kept set -> top_p (keep rank r iff the mass ABOVE it is < p, strictly) -> renormalised.  The merged order is
// value descending, then global id ascending, so entry 0 is the served greedy id (the lowest id among the ids tied at
// the max) and the zero-noise pick is that same id.
//
// NUMERICS: vt = fl(v * (1/T)) and d = fl(vt - vt_0) in IEEE float32 (libgcc soft-float, round to nearest even: the
// served path's own fp32 ops); exp(d) in 64-bit fixed point (Q56, |rel err| < 2e-9, the max element is exactly 2^56);
// the sums, the cumulative mass and the top_p test exact in integers (the test in double); the probability
// E_i / sum E as a correctly rounded division in double, rounded once to float32.  The served cut's ttnn.exp /
// sum / divide / cumsum chain is within ~1e-7 of fp64 (opus-specsample records), so the two agree to ~1e-7.
// The pick is the served Gumbel-max: argmax over the kept entries of fl(v + noise[merged col]) (first max wins).
//
// QWEN36_MTP_SPEC_SAMPLE_SSP (sampled-step parity) adds three things, each a compile switch, so the S7 program is the
// switches at 0:
//   RULE = 1 (the verify): the exact rule (spec_sample.rule, ~46 programs) on this row, in IEEE float32 like the host's
//     fp32 reference spec_sample.host_rule: q_at_p_i = sum_j [pi_i == qi_j] q_j, p(x), q(x), accept = fl(u q(x)) < p(x),
//     the residual race argmax_i -(nres_i / fl(r_i + 1e-30)) over r_i = fl(p_i - q_at_p_i) > 0 (else -3e38), first max
//     wins.  Inputs qp / qi [M, 2K] (the carried q block), u [M, 1], nres [M, 2K] (NZ_MODE 0); outputs accept [M, 1]
//     fp32 (0 / 1) and resid [M, 1] uint32.  Only the terms that can be non-zero are visited (p_i > 0 only for i < nk;
//     q_j = 0 adds nothing), which leaves every sum and the argmax unchanged.
//   NZ_MODE = 1 (the verify) / 2 (the chain): the noise is made HERE (fs_noise.h, the host's hash and the host's values)
//     instead of read from noise / u / nres: only for the columns the row reads (the kept entries j < nk; the residual's
//     i < nk with r_i > 0), from a per-call noise plan page NP (fused_sampler.NP_*): per verify row r the dev_sample key
//     (seed, position, T) of the target pick and the spec seed of u (stream 11) / nres (stream 12); per chain row b the
//     spec seed, the base position p_b + 2 and T, with m_b read from the device accept's m page (MP word b), so depth
//     DEPTH keys stream 20 + DEPTH at p_b + 2 + m_b + DEPTH (mtp_batch_host.da_noise_positions).  A T == 0 row gets no
//     pick noise (+0.0; its flag is 0, so the pick is never emitted).
//   MAP = 1 (the chain under the draft vocab): the emitted id is the draft-vocab sub-id looked up in the id table TAB
//     ([N / 16, 16] uint32 pages, the full token ids), which is what mtp_draft_vocab.map_ids computes in ~11 programs.
//     The ids block stays sub-ids (the host maps the q ids, as served).
//
// MF: 1 = every emitted id + 1 (the self-test must REFUSE); 2 = top_k ignored (the softmax and top_p run over all 2K
// candidates: the distribution bug the chi-square gate must catch); 3 = every accept flipped; 4 = the residual drawn from
// p, not (p - q)^+ (the rule's two must-fails, as spec_sample.rule); 5 = the made noise keyed one position late.
//
// Compile args: {CB_SCR, K, PER, MF, OVR, DBG, SCR_OFF, G_PAGE, SC_PAGE, RK_PAGE, TP_PAGE, NZ_PAGE, FL_PAGE, X_PAGE,
//                XF_PAGE, PR_PAGE, ID_PAGE, SEL_PAGE, AUX_PAGE, RULE, NZ_MODE, MAP, DEPTH, NP_PAGE, QP_PAGE, QI_PAGE,
//                U_PAGE, NR_PAGE, AC_PAGE, RS_PAGE, MP_PAGE, TB_PAGE, ND_PAGE}
//                + TensorAccessorArgs of g, scale, rank, topp, noise, flag, x, xf, probs, ids, sel, aux, np, qp, qi, u,
//                nres, accept, resid, mpage, table, ndump (any tensor when unused)
// Runtime args: {22 buffer addresses in that order, row0, row1}
// DBG with NZ_MODE != 0: ndump row m = the made pick noise of every merged column (C2), then the made nres of every
// sorted column (C2), then u: what the host would have uploaded, for the gate's bitwise comparison.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "fs_noise.h"

namespace {

union FU {
    float f;
    uint32_t u;
};
FORCE_INLINE float asf(uint32_t u) {
    FU x;
    x.u = u;
    return x.f;
}
FORCE_INLINE uint32_t asu(float f) {
    FU x;
    x.f = f;
    return x.u;
}

// ordered key of a bf16 bit pattern (+0 == -0): the same order as fs_topk.cpp
FORCE_INLINE int32_t okey16(uint32_t h) {
    const int32_t hs = static_cast<int32_t>(h << 16) >> 16;
    if (hs >= 0) {
        return hs;
    }
    if (hs == -32768) {
        return 0;
    }
    return hs ^ 0x7FFF;
}

constexpr uint32_t T64[64] = {
    2147483648u, 2170868212u, 2194507417u, 2218404036u, 2242560872u, 2266980759u, 2291666561u, 2316621173u,
    2341847524u, 2367348571u, 2393127307u, 2419186755u, 2445529972u, 2472160047u, 2499080105u, 2526293303u,
    2553802834u, 2581611923u, 2609723834u, 2638141863u, 2666869345u, 2695909648u, 2725266179u, 2754942382u,
    2784941738u, 2815267765u, 2845924021u, 2876914102u, 2908241642u, 2939910317u, 2971923842u, 3004285971u,
    3037000500u, 3070071267u, 3103502151u, 3137297074u, 3171459999u, 3205994934u, 3240905930u, 3276197082u,
    3311872529u, 3347936457u, 3384393094u, 3421246719u, 3458501653u, 3496162267u, 3534232978u, 3572718252u,
    3611622603u, 3650950594u, 3690706840u, 3730896002u, 3771522796u, 3812591987u, 3854108391u, 3896076880u,
    3938502376u, 3981389855u, 4024744348u, 4068570940u, 4112874773u, 4157661043u, 4202935003u, 4248701965u};
constexpr uint32_t LOG2E_HI = 1549082004u;  // log2(e) * 2^62 = LOG2E_HI * 2^32 + LOG2E_LO
constexpr uint32_t LOG2E_LO = 2920020062u;
constexpr uint32_t LN2_Q32 = 2977044472u;  // ln(2) * 2^32
constexpr uint32_t C6 = 357913941u;        // 2^31 / 6
constexpr uint32_t C24 = 89478485u;        // 2^31 / 24
constexpr uint32_t C2H = 1073741824u;      // 2^31 / 2

FORCE_INLINE uint32_t mul31(uint32_t a, uint32_t b) {
    return static_cast<uint32_t>((static_cast<uint64_t>(a) * b) >> 31);
}

// round(exp(d) * 2^56) for a float32 d <= 0 (bits), in integers
uint64_t exp_q56(uint32_t dbits) {
    const uint32_t mag = dbits & 0x7FFFFFFFu;
    const uint32_t e = mag >> 23;
    if (e == 0) {
        return 1ull << 56;  // 0 or a denormal: exp(d) rounds to 1
    }
    if (e > 132) {
        return 0;  // |d| >= 64: exp(d) * 2^56 < 1e-11
    }
    const uint64_t m = (mag & 0x7FFFFFu) | 0x800000u;
    // y = |d| * log2(e) in Q40: (m * L) >> (172 - e), L = log2(e) * 2^62, m * 2^(e - 150) = |d|
    const uint64_t A = m * LOG2E_HI;
    const uint64_t B = m * LOG2E_LO;
    const uint64_t t = A + (B >> 32);
    const uint32_t sh = 172 - e - 32;  // >= 8
    const uint64_t y = sh >= 64 ? 0 : (t >> sh);
    const uint32_t I = static_cast<uint32_t>(y >> 40);
    const uint64_t F = y & ((1ull << 40) - 1);
    if (F == 0) {
        return I <= 56 ? (1ull << (56 - I)) : 0;
    }
    // exp(d) = 2^-y = 2^-(I+1) * 2^phi, phi = 1 - F/2^40 = G/2^40 in (0, 1)
    const uint64_t G = (1ull << 40) - F;
    const uint32_t j = static_cast<uint32_t>(G >> 34);
    const uint32_t gl = static_cast<uint32_t>((G & ((1ull << 34) - 1)) >> 2);  // rho * 2^38, rho < 1/64
    const uint32_t z = static_cast<uint32_t>((static_cast<uint64_t>(gl) * LN2_Q32) >> 39);  // rho * ln2 * 2^31
    const uint32_t a = C6 + mul31(z, C24);
    const uint32_t b = C2H + mul31(z, a);
    const uint32_t c = 0x80000000u + mul31(z, b);
    const uint32_t P = 0x80000000u + mul31(z, c);  // 2^rho * 2^31
    const uint64_t g = (static_cast<uint64_t>(T64[j]) * P) >> 31;  // 2^phi * 2^31
    if (I <= 24) {
        return g << (24 - I);
    }
    const uint32_t r = I - 24;
    if (r >= 40) {
        return 0;
    }
    return (g + (1ull << (r - 1))) >> r;
}

constexpr uint32_t al64(uint32_t x) { return (x + 63u) & ~63u; }

}  // namespace

void kernel_main() {
    constexpr uint32_t CB_SCR = get_compile_time_arg_val(0);
    constexpr uint32_t K = get_compile_time_arg_val(1);
    constexpr uint32_t PER = get_compile_time_arg_val(2);
    constexpr uint32_t MF = get_compile_time_arg_val(3);
    constexpr uint32_t OVR = get_compile_time_arg_val(4);
    constexpr uint32_t DBG = get_compile_time_arg_val(5);
    constexpr uint32_t SCR_BYTES_MAX = 1u << 20;
    constexpr uint32_t SCR_OFF = get_compile_time_arg_val(6);
    constexpr uint32_t G_PAGE = get_compile_time_arg_val(7);
    constexpr uint32_t SC_PAGE = get_compile_time_arg_val(8);
    constexpr uint32_t RK_PAGE = get_compile_time_arg_val(9);
    constexpr uint32_t TP_PAGE = get_compile_time_arg_val(10);
    constexpr uint32_t NZ_PAGE = get_compile_time_arg_val(11);
    constexpr uint32_t FL_PAGE = get_compile_time_arg_val(12);
    constexpr uint32_t X_PAGE = get_compile_time_arg_val(13);
    constexpr uint32_t XF_PAGE = get_compile_time_arg_val(14);
    constexpr uint32_t PR_PAGE = get_compile_time_arg_val(15);
    constexpr uint32_t ID_PAGE = get_compile_time_arg_val(16);
    constexpr uint32_t SEL_PAGE = get_compile_time_arg_val(17);
    constexpr uint32_t AUX_PAGE = get_compile_time_arg_val(18);
    constexpr uint32_t RULE = get_compile_time_arg_val(19);
    constexpr uint32_t NZ_MODE = get_compile_time_arg_val(20);
    constexpr uint32_t MAP = get_compile_time_arg_val(21);
    constexpr uint32_t DEPTH = get_compile_time_arg_val(22);
    constexpr uint32_t NP_PAGE = get_compile_time_arg_val(23);
    constexpr uint32_t QP_PAGE = get_compile_time_arg_val(24);
    constexpr uint32_t QI_PAGE = get_compile_time_arg_val(25);
    constexpr uint32_t U_PAGE = get_compile_time_arg_val(26);
    constexpr uint32_t NR_PAGE = get_compile_time_arg_val(27);
    constexpr uint32_t AC_PAGE = get_compile_time_arg_val(28);
    constexpr uint32_t RS_PAGE = get_compile_time_arg_val(29);
    constexpr uint32_t MP_PAGE = get_compile_time_arg_val(30);
    constexpr uint32_t TB_PAGE = get_compile_time_arg_val(31);
    constexpr uint32_t ND_PAGE = get_compile_time_arg_val(32);
    constexpr auto a_g = TensorAccessorArgs<33>();
    constexpr auto a_sc = TensorAccessorArgs<a_g.next_compile_time_args_offset()>();
    constexpr auto a_rk = TensorAccessorArgs<a_sc.next_compile_time_args_offset()>();
    constexpr auto a_tp = TensorAccessorArgs<a_rk.next_compile_time_args_offset()>();
    constexpr auto a_nz = TensorAccessorArgs<a_tp.next_compile_time_args_offset()>();
    constexpr auto a_fl = TensorAccessorArgs<a_nz.next_compile_time_args_offset()>();
    constexpr auto a_x = TensorAccessorArgs<a_fl.next_compile_time_args_offset()>();
    constexpr auto a_xf = TensorAccessorArgs<a_x.next_compile_time_args_offset()>();
    constexpr auto a_pr = TensorAccessorArgs<a_xf.next_compile_time_args_offset()>();
    constexpr auto a_id = TensorAccessorArgs<a_pr.next_compile_time_args_offset()>();
    constexpr auto a_sel = TensorAccessorArgs<a_id.next_compile_time_args_offset()>();
    constexpr auto a_aux = TensorAccessorArgs<a_sel.next_compile_time_args_offset()>();
    constexpr auto a_np = TensorAccessorArgs<a_aux.next_compile_time_args_offset()>();
    constexpr auto a_qp = TensorAccessorArgs<a_np.next_compile_time_args_offset()>();
    constexpr auto a_qi = TensorAccessorArgs<a_qp.next_compile_time_args_offset()>();
    constexpr auto a_u = TensorAccessorArgs<a_qi.next_compile_time_args_offset()>();
    constexpr auto a_nr = TensorAccessorArgs<a_u.next_compile_time_args_offset()>();
    constexpr auto a_ac = TensorAccessorArgs<a_nr.next_compile_time_args_offset()>();
    constexpr auto a_rs = TensorAccessorArgs<a_ac.next_compile_time_args_offset()>();
    constexpr auto a_mp = TensorAccessorArgs<a_rs.next_compile_time_args_offset()>();
    constexpr auto a_tb = TensorAccessorArgs<a_mp.next_compile_time_args_offset()>();
    constexpr auto a_nd = TensorAccessorArgs<a_tb.next_compile_time_args_offset()>();
    constexpr uint32_t C2 = 2 * K;
    static_assert(K >= 1 && C2 <= 128, "K: the uint64 sum of up to 2K Q56 terms must not wrap");
    static_assert(RULE == 0 || OVR == 1, "the rule runs in the verify (it needs the draft x)");
    static_assert(NZ_MODE != 1 || OVR == 1, "NZ_MODE 1 is the verify's noise");
    static_assert(NZ_MODE != 2 || (OVR == 0 && RULE == 0), "NZ_MODE 2 is the chain's noise");
    static_assert(MAP == 0 || OVR == 0, "MAP is the chain's draft-vocab id map");
    // the noise plan page (fused_sampler.NP_*): per verify row r < 32, per chain row b < 16
    constexpr uint32_t NP_SEED_DS = 0, NP_VPOS = 32, NP_TDS_LO = 64, NP_TDS_HI = 96, NP_SEED_SP = 128;
    constexpr uint32_t NP_CSEED = 160, NP_CBASE = 176, NP_CT_LO = 192, NP_CT_HI = 208;
    static_assert(NZ_MODE == 0 || NP_PAGE >= 256 * 4, "noise plan page: 256 words");

    // L1 scratch (64-byte aligned pieces)
    constexpr uint32_t O_G = 0;
    constexpr uint32_t O_SC = O_G + al64(G_PAGE);
    constexpr uint32_t O_RK = O_SC + al64(SC_PAGE);
    constexpr uint32_t O_TP = O_RK + al64(RK_PAGE);
    constexpr uint32_t O_NZ = O_TP + al64(TP_PAGE);
    constexpr uint32_t O_FL = O_NZ + al64(NZ_PAGE);
    constexpr uint32_t O_X = O_FL + al64(FL_PAGE);
    constexpr uint32_t O_XF = O_X + al64(X_PAGE);
    constexpr uint32_t O_PR = O_XF + al64(XF_PAGE);
    constexpr uint32_t O_ID = O_PR + al64(PR_PAGE);
    constexpr uint32_t O_SEL = O_ID + al64(ID_PAGE);
    constexpr uint32_t O_AUX = O_SEL + al64(SEL_PAGE);
    constexpr uint32_t O_SV = O_AUX + al64(AUX_PAGE);
    constexpr uint32_t O_GID = O_SV + al64(C2 * 4);
    constexpr uint32_t O_COL = O_GID + al64(C2 * 4);
    constexpr uint32_t O_E = O_COL + al64(C2 * 4);
    constexpr uint32_t O_NP = O_E + al64(C2 * 8);
    constexpr uint32_t O_QP = O_NP + (NZ_MODE ? al64(NP_PAGE) : 0);
    constexpr uint32_t O_QI = O_QP + (RULE ? al64(QP_PAGE) : 0);
    constexpr uint32_t O_U = O_QI + (RULE ? al64(QI_PAGE) : 0);
    constexpr uint32_t O_NR = O_U + ((RULE && NZ_MODE == 0) ? al64(U_PAGE) : 0);
    constexpr uint32_t O_AC = O_NR + ((RULE && NZ_MODE == 0) ? al64(NR_PAGE) : 0);
    constexpr uint32_t O_RS = O_AC + (RULE ? al64(AC_PAGE) : 0);
    constexpr uint32_t O_MP = O_RS + (RULE ? al64(RS_PAGE) : 0);
    constexpr uint32_t O_TB = O_MP + (NZ_MODE == 2 ? al64(MP_PAGE) : 0);
    constexpr uint32_t O_ND = O_TB + (MAP ? al64(TB_PAGE) : 0);
    constexpr uint32_t O_NZ2 = O_ND + ((DBG && NZ_MODE) ? al64(ND_PAGE) : 0);  // the made pick noise per merged column
    constexpr uint32_t O_END = O_NZ2 + (NZ_MODE ? al64(C2 * 4) : 0);
    static_assert(O_END <= SCR_BYTES_MAX, "fs_cut scratch");

    uint32_t ai = 0;
    const auto g_acc = TensorAccessor(a_g, get_arg_val<uint32_t>(ai++), G_PAGE);
    const auto sc_acc = TensorAccessor(a_sc, get_arg_val<uint32_t>(ai++), SC_PAGE);
    const auto rk_acc = TensorAccessor(a_rk, get_arg_val<uint32_t>(ai++), RK_PAGE);
    const auto tp_acc = TensorAccessor(a_tp, get_arg_val<uint32_t>(ai++), TP_PAGE);
    const auto nz_acc = TensorAccessor(a_nz, get_arg_val<uint32_t>(ai++), NZ_PAGE);
    const auto fl_acc = TensorAccessor(a_fl, get_arg_val<uint32_t>(ai++), FL_PAGE);
    const auto x_acc = TensorAccessor(a_x, get_arg_val<uint32_t>(ai++), X_PAGE);
    const auto xf_acc = TensorAccessor(a_xf, get_arg_val<uint32_t>(ai++), XF_PAGE);
    const auto pr_acc = TensorAccessor(a_pr, get_arg_val<uint32_t>(ai++), PR_PAGE);
    const auto id_acc = TensorAccessor(a_id, get_arg_val<uint32_t>(ai++), ID_PAGE);
    const auto sel_acc = TensorAccessor(a_sel, get_arg_val<uint32_t>(ai++), SEL_PAGE);
    const auto aux_acc = TensorAccessor(a_aux, get_arg_val<uint32_t>(ai++), AUX_PAGE);
    const auto np_acc = TensorAccessor(a_np, get_arg_val<uint32_t>(ai++), NP_PAGE);
    const auto qp_acc = TensorAccessor(a_qp, get_arg_val<uint32_t>(ai++), QP_PAGE);
    const auto qi_acc = TensorAccessor(a_qi, get_arg_val<uint32_t>(ai++), QI_PAGE);
    const auto u_acc = TensorAccessor(a_u, get_arg_val<uint32_t>(ai++), U_PAGE);
    const auto nr_acc = TensorAccessor(a_nr, get_arg_val<uint32_t>(ai++), NR_PAGE);
    const auto ac_acc = TensorAccessor(a_ac, get_arg_val<uint32_t>(ai++), AC_PAGE);
    const auto rs_acc = TensorAccessor(a_rs, get_arg_val<uint32_t>(ai++), RS_PAGE);
    const auto mp_acc = TensorAccessor(a_mp, get_arg_val<uint32_t>(ai++), MP_PAGE);
    const auto tb_acc = TensorAccessor(a_tb, get_arg_val<uint32_t>(ai++), TB_PAGE);
    const auto nd_acc = TensorAccessor(a_nd, get_arg_val<uint32_t>(ai++), ND_PAGE);
    const uint32_t row0 = get_arg_val<uint32_t>(ai++);
    const uint32_t row1 = get_arg_val<uint32_t>(ai++);

    const uint32_t scr = get_write_ptr(CB_SCR) + SCR_OFF;
    auto P32 = [&](uint32_t off) { return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scr + off); };
    volatile tt_l1_ptr uint32_t* G = P32(O_G);
    volatile tt_l1_ptr uint32_t* SC = P32(O_SC);
    volatile tt_l1_ptr uint32_t* RK = P32(O_RK);
    volatile tt_l1_ptr uint32_t* TP = P32(O_TP);
    volatile tt_l1_ptr uint32_t* NZ = P32(O_NZ);
    volatile tt_l1_ptr uint32_t* FL = P32(O_FL);
    volatile tt_l1_ptr uint32_t* X = P32(O_X);
    volatile tt_l1_ptr uint32_t* XF = P32(O_XF);
    volatile tt_l1_ptr uint32_t* PR = P32(O_PR);
    volatile tt_l1_ptr uint32_t* ID = P32(O_ID);
    volatile tt_l1_ptr uint32_t* SEL = P32(O_SEL);
    volatile tt_l1_ptr uint32_t* AUX = P32(O_AUX);
    volatile tt_l1_ptr uint32_t* SV = P32(O_SV);
    volatile tt_l1_ptr uint32_t* GID = P32(O_GID);
    volatile tt_l1_ptr uint32_t* COL = P32(O_COL);
    volatile tt_l1_ptr uint64_t* E = reinterpret_cast<volatile tt_l1_ptr uint64_t*>(scr + O_E);
    volatile tt_l1_ptr uint32_t* NP = P32(O_NP);
    volatile tt_l1_ptr uint32_t* QP = P32(O_QP);
    volatile tt_l1_ptr uint32_t* QI = P32(O_QI);
    volatile tt_l1_ptr uint32_t* U = P32(O_U);
    volatile tt_l1_ptr uint32_t* NR = P32(O_NR);
    volatile tt_l1_ptr uint32_t* AC = P32(O_AC);
    volatile tt_l1_ptr uint32_t* RS = P32(O_RS);
    volatile tt_l1_ptr uint32_t* MP = P32(O_MP);
    volatile tt_l1_ptr uint32_t* TB = P32(O_TB);
    volatile tt_l1_ptr uint32_t* ND = P32(O_ND);
    volatile tt_l1_ptr uint32_t* NZ2 = P32(O_NZ2);

    for (uint32_t i = 0; i < al64(SEL_PAGE) / 4; ++i) {
        SEL[i] = 0;
    }
    if constexpr (DBG) {
        for (uint32_t i = 0; i < al64(AUX_PAGE) / 4; ++i) {
            AUX[i] = 0;
        }
    }
    if constexpr (RULE) {
        for (uint32_t i = 0; i < al64(AC_PAGE) / 4; ++i) {
            AC[i] = 0;
        }
        for (uint32_t i = 0; i < al64(RS_PAGE) / 4; ++i) {
            RS[i] = 0;
        }
    }
    if constexpr (DBG && NZ_MODE != 0) {
        for (uint32_t i = 0; i < al64(ND_PAGE) / 4; ++i) {
            ND[i] = 0;
        }
    }
    // the per-call pages: the noise plan, and (the chain) the device accept's m page
    if constexpr (NZ_MODE != 0) {
        noc_async_read_page(0, np_acc, scr + O_NP);
        if constexpr (NZ_MODE == 2) {
            noc_async_read_page(0, mp_acc, scr + O_MP);
        }
        noc_async_read_barrier();
        invalidate_l1_cache();
    }
    constexpr uint64_t NZ_SHIFT = MF == 5 ? 1u : 0u;

    for (uint32_t m = row0; m < row1; ++m) {
        noc_async_read_page(m, g_acc, scr + O_G);
        noc_async_read_page(m, sc_acc, scr + O_SC);
        noc_async_read_page(m, rk_acc, scr + O_RK);
        noc_async_read_page(m, tp_acc, scr + O_TP);
        if constexpr (NZ_MODE == 0) {
            noc_async_read_page(m, nz_acc, scr + O_NZ);
        }
        noc_async_read_page(m, fl_acc, scr + O_FL);
        if constexpr (OVR) {
            noc_async_read_page(m, x_acc, scr + O_X);
            noc_async_read_page(m, xf_acc, scr + O_XF);
        }
        if constexpr (RULE) {
            noc_async_read_page(m, qp_acc, scr + O_QP);
            noc_async_read_page(m, qi_acc, scr + O_QI);
            if constexpr (NZ_MODE == 0) {
                noc_async_read_page(m, u_acc, scr + O_U);
                noc_async_read_page(m, nr_acc, scr + O_NR);
            }
        }
        noc_async_read_barrier();
        invalidate_l1_cache();

        // ---- this row's noise keys (NZ_MODE 1 / 2)
        uint64_t pk_seed = 0, pk_pos = 0, pk_stream = 0, sp_seed = 0, sp_pos = 0;
        uint32_t t_hi = 0, t_lo = 0;
        bool pk_ds = false;
        if constexpr (NZ_MODE == 1) {
            pk_ds = true;
            pk_seed = NP[NP_SEED_DS + m];
            pk_pos = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(NP[NP_VPOS + m]))) + NZ_SHIFT;
            t_lo = NP[NP_TDS_LO + m];
            t_hi = NP[NP_TDS_HI + m];
            sp_seed = NP[NP_SEED_SP + m];
            sp_pos = pk_pos;
        } else if constexpr (NZ_MODE == 2) {
            pk_seed = NP[NP_CSEED + m];
            pk_pos = static_cast<uint64_t>(static_cast<int64_t>(static_cast<int32_t>(NP[NP_CBASE + m])) +
                                           static_cast<int64_t>(static_cast<int32_t>(MP[m])) + DEPTH) + NZ_SHIFT;
            pk_stream = 20 + DEPTH;
            t_lo = NP[NP_CT_LO + m];
            t_hi = NP[NP_CT_HI + m];
        }
        auto pick_noise = [&](uint32_t col) -> uint32_t {
            if (pk_ds) {
                return fsn::gumbel_f32(fsn::hash_ds(pk_seed, pk_pos, col), t_hi, t_lo);
            }
            return fsn::gumbel_f32(fsn::hash_sp(pk_seed, pk_pos, pk_stream, col), t_hi, t_lo);
        };
        (void)pick_noise;
        (void)sp_seed;
        (void)sp_pos;

        // ---- merge the two descending lists: value descending, global id ascending (chip 0's ids are the lower)
        {
            uint32_t a = 0, b = 0;
            for (uint32_t j = 0; j < C2; ++j) {
                bool take0;
                if (a >= K) {
                    take0 = false;
                } else if (b >= K) {
                    take0 = true;
                } else {
                    take0 = okey16(G[a] >> 16) >= okey16(G[2 * K + b] >> 16);
                }
                if (take0) {
                    SV[j] = G[a];
                    GID[j] = G[K + a];
                    COL[j] = a;
                    ++a;
                } else {
                    SV[j] = G[2 * K + b];
                    GID[j] = G[3 * K + b] + PER;
                    COL[j] = K + b;
                    ++b;
                }
            }
        }

        // ---- temperature + top_k (ties at the k-th value kept)
        const float s = asf(SC[0]);
        uint32_t k = 0;
        while (k < C2 && RK[k] == 0x3F800000u) {
            ++k;
        }
        if (k == 0) {
            k = C2;
        }
        uint32_t kk = k;
        if constexpr (MF == 2) {
            kk = C2;
        } else {
            const float kth = asf(SV[k - 1]) * s;
            while (kk < C2 && asf(SV[kk]) * s >= kth) {
                ++kk;
            }
        }

        // ---- softmax over the kept set: exp(vt - vt_0) in Q56, exact integer sums
        const float m0 = asf(SV[0]) * s;
        uint64_t S = 0;
        for (uint32_t j = 0; j < kk; ++j) {
            const float d = asf(SV[j]) * s - m0;
            const uint64_t ej = exp_q56(asu(d));
            E[j] = ej;
            S += ej;
        }

        // ---- top_p: keep rank j iff the mass above it is < p (strict)
        const double pS = static_cast<double>(asf(TP[0])) * static_cast<double>(S);
        uint32_t nk = 0;
        uint64_t Sk = 0;
        for (uint32_t j = 0; j < kk; ++j) {
            if (!(static_cast<double>(Sk) < pS)) {
                break;
            }
            Sk += E[j];
            nk = j + 1;
        }
        if (nk == 0) {
            nk = 1;  // the largest is never removed (p > 0 always keeps it; a defensive floor)
            Sk = E[0];
        }

        // ---- probabilities (renormalised over the kept set) and the ids, in the sorted order
        const double dS = static_cast<double>(Sk);
        for (uint32_t j = 0; j < C2; ++j) {
            PR[j] = j < nk ? asu(static_cast<float>(static_cast<double>(E[j]) / dS)) : 0u;
            ID[j] = asu(static_cast<float>(GID[j]));
        }

        // ---- the Gumbel-max pick over the kept entries (first max wins), the greedy id, the emitted id
        if constexpr (NZ_MODE != 0) {
            // the made noise, by merged column: every column under DBG (the dump), else only the kept ones
            const bool tz = !((t_hi & 0x7FFFFFFFu) | t_lo);
            for (uint32_t c = 0; c < C2; ++c) {
                NZ2[c] = 0;
            }
            if constexpr (DBG) {
                for (uint32_t c = 0; c < C2; ++c) {
                    NZ2[c] = tz ? 0u : pick_noise(c);
                }
            } else if (!tz) {
                for (uint32_t j = 0; j < nk; ++j) {
                    NZ2[COL[j]] = pick_noise(COL[j]);
                }
            }
        }
        volatile tt_l1_ptr uint32_t* NZP = NZ_MODE != 0 ? NZ2 : NZ;
        uint32_t r = 0;
        float best = asf(SV[0]) + asf(NZP[COL[0]]);
        for (uint32_t j = 1; j < nk; ++j) {
            const float v = asf(SV[j]) + asf(NZP[COL[j]]);
            if (v > best) {
                best = v;
                r = j;
            }
        }
        const uint32_t pick = GID[r];
        const uint32_t greedy = GID[0];
        uint32_t sel = (FL[0] & 0x7FFFFFFFu) ? pick : greedy;
        if constexpr (OVR) {
            if (XF[0] & 0x7FFFFFFFu) {
                sel = static_cast<uint32_t>(asf(X[0]));
            }
        }
        if constexpr (MF == 1) {
            sel += 1;
        }
        if constexpr (MAP) {
            // the draft vocab: sub-id -> full id (mtp_draft_vocab.map_ids), one 16-id page of the table
            noc_async_read_page(sel >> 4, tb_acc, scr + O_TB);
            noc_async_read_barrier();
            invalidate_l1_cache();
            sel = TB[sel & 15u];
        }
        SEL[0] = sel;

        // ---- the exact rule (RULE): spec_sample.host_rule's fp32 arithmetic on this row
        if constexpr (RULE) {
            const uint32_t xb = X[0] == 0x80000000u ? 0u : X[0];  // the draft id (fp32; -0 compares as +0)
            // u and the residual's Exp(1) noise: made (NZ_MODE 1) or read (NZ_MODE 0)
            const uint32_t ub = NZ_MODE == 1 ? fsn::u01_f32(fsn::hash_sp(sp_seed, sp_pos, 11, 0)) : U[0];
            float p_at_x = 0.0f, q_at_x = 0.0f;
            for (uint32_t i = 0; i < nk; ++i) {
                if (ID[i] == xb) {
                    p_at_x = p_at_x + asf(PR[i]);
                }
            }
            for (uint32_t j = 0; j < C2; ++j) {
                if (QP[j] != 0u && QI[j] == xb) {
                    q_at_x = q_at_x + asf(QP[j]);
                }
            }
            float acc = (asf(ub) * q_at_x < p_at_x) ? 1.0f : 0.0f;
            if constexpr (MF == 3) {
                acc = 1.0f - acc;
            }
            // the residual (p - q)^+ over the target's support (p_i > 0 only for i < nk), the exponential race:
            // argmax_i score_i, score_i = -(nres_i / fl(r_i + 1e-30)) where r_i = fl(p_i - q_at_p_i) > 0, else -3e38.
            // Every positive-r score is > -3e38 (nres <= 28, r + 1e-30 >= 1e-30), so the first max is the first
            // maximum among the positive-r entries, and index 0 when there is none (all -3e38).
            const float tiny = 1e-30f;
            uint32_t jbest = 0;
            float sbest = 0.0f;
            bool any = false;
            for (uint32_t i = 0; i < nk; ++i) {
                float qap = 0.0f;
                if constexpr (MF != 4) {
                    for (uint32_t j = 0; j < C2; ++j) {
                        if (QP[j] != 0u && QI[j] == ID[i]) {
                            qap = qap + asf(QP[j]);
                        }
                    }
                }
                const float rn = asf(PR[i]) - qap;
                if (!(rn > 0.0f)) {
                    continue;
                }
                const uint32_t nb = NZ_MODE == 1 ? fsn::expo_f32(fsn::hash_sp(sp_seed, sp_pos, 12, i)) : NR[i];
                const float sc = -(asf(nb) / (rn + tiny));
                if (!any || sc > sbest) {
                    sbest = sc;
                    jbest = i;
                    any = true;
                }
            }
            AC[0] = asu(acc);
            RS[0] = static_cast<uint32_t>(asf(ID[jbest]));
            if constexpr (DBG && NZ_MODE == 1) {
                for (uint32_t i = 0; i < C2; ++i) {
                    ND[C2 + i] = fsn::expo_f32(fsn::hash_sp(sp_seed, sp_pos, 12, i));
                }
                ND[2 * C2] = ub;
            }
        }
        if constexpr (DBG && NZ_MODE != 0) {
            for (uint32_t c = 0; c < C2; ++c) {
                ND[c] = NZ2[c];
            }
        }
        if constexpr (DBG) {
            AUX[0] = sel;
            AUX[1] = pick;
            AUX[2] = greedy;
            AUX[3] = nk | (kk << 16);
        }
        (void)SEL[0];
        asm volatile("fence" ::: "memory");
        noc_async_write_page(m, pr_acc, scr + O_PR);
        noc_async_write_page(m, id_acc, scr + O_ID);
        noc_async_write_page(m, sel_acc, scr + O_SEL);
        if constexpr (DBG) {
            noc_async_write_page(m, aux_acc, scr + O_AUX);
        }
        if constexpr (RULE) {
            noc_async_write_page(m, ac_acc, scr + O_AC);
            noc_async_write_page(m, rs_acc, scr + O_RS);
        }
        if constexpr (DBG && NZ_MODE != 0) {
            noc_async_write_page(m, nd_acc, scr + O_ND);
        }
        noc_async_write_barrier();
    }
}

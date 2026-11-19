// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// QWEN36_MTP_SPEC_SAMPLE_SSP (fused_sampler.py): the exact-sampling noise of one (seed, position, stream, column), made
// where it is used (fs_cut.cpp) instead of on the host.  PURE INTEGER C++: no float or double type anywhere, so the same
// source compiled for the Tensix RISC-V cores and for the host (bench/probes/fs_noise_host.py builds it with g++ for the
// large host gate) computes the same bits by construction.
//
// The host builders this replaces (all three keep serving when the flag is off):
//   dev_sample.noise_matrix / row_block_fast    the verify's target-sample noise v_gum = fl32(fl64(T * -ln(-ln u)))
//   spec_sample.uniform / expo / gumbel and noise_group (HOSTNOISE)
//                                               v_su = fl32(u) (stream 11), v_snr = fl32(-ln u) (stream 12),
//                                               d_sgum{k} = fl32(fl64(T * -ln(-ln u))) (stream 20 + k)
//   u = clamp((h >> 11 & (2^53 - 1)) * 2^-53, 1e-12, 1 - 1e-9) in float64, h = splitmix64-style hash of the key.
//
// THE HASH is the host's int64 arithmetic word for word: multiplies wrap, ">>" is torch/numpy's ARITHMETIC shift (the
// masks after the 27- and 31-bit shifts keep 4 and 8 sign bits, so a logical shift would differ), so the 53-bit u is
// bit-identical by construction.
//
// THE LOGS are computed in 64-bit fixed point / a 64-bit-mantissa float (XF): ln x = e ln2 + ln f, f in [1, 2), with a
// 128-entry reciprocal table (|r| <= 2^-8 after the reduction) and a Q62 series for ln(1 + r); within 2^-6 of 1 the
// series runs on r = x - 1 itself (exact), so ln keeps its RELATIVE precision there (u -> 1 gives L -> 1e-9).  Error
// ~2^-58 relative (typical), ~2^-51 in the worst regime; the result is rounded ONCE to float32 (nearest even).  The host
// rounds -ln u to float64, then (Gumbel) -ln of that, then T * g, each to float64, then to float32: the two float32s
// agree unless the real value sits within ~2^-28 ulp32 of a float32 rounding boundary (and the host's own float64 libm
// log is not correctly rounded there) -- an ESTIMATE of < 1e-8 per value.  The gate counts it on 1e8 values
// (bench/probes/sim_ssp_noise.py); the boot self-test compares the device noise with the served host builders bitwise.
//
// A row with T == 0 gets +0.0 here (the host's T * g is -0.0 when g < 0): fs_cut never reads the noise of a T == 0 row
// (its pick is discarded by the flag select), so this is not a difference anyone can observe.

#pragma once

#include <stdint.h>

namespace fsn {

struct U128 {
    uint64_t hi, lo;
};

inline U128 mul64(uint64_t a, uint64_t b) {
    const uint64_t a0 = static_cast<uint32_t>(a), a1 = a >> 32, b0 = static_cast<uint32_t>(b), b1 = b >> 32;
    const uint64_t p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const uint64_t mid = (p00 >> 32) + static_cast<uint32_t>(p01) + static_cast<uint32_t>(p10);
    U128 r;
    r.lo = (mid << 32) | static_cast<uint32_t>(p00);
    r.hi = p11 + (p01 >> 32) + (p10 >> 32) + (mid >> 32);
    return r;
}

// int64 arithmetic shift right (torch / numpy ">>" on int64), without relying on implementation-defined behaviour
inline uint64_t sra(uint64_t x, uint32_t n) {
    const uint64_t s = x >> n;
    return (x >> 63) ? (s | ~(~0ull >> n)) : s;
}

inline uint64_t mix(uint64_t x, uint64_t m1, uint64_t m2) {
    x ^= sra(x, 30) & 0x3FFFFFFFFull;
    x *= m1;
    x ^= sra(x, 27) & 0x1FFFFFFFFFFull;
    x *= m2;
    x ^= sra(x, 31) & 0x1FFFFFFFFFFull;
    return x;
}

// dev_sample.noise_matrix / row_block_fast (the verify's target-sample noise): base = s*M3 ^ p*GOLD, no stream
// (DS_M2 is dev_sample._M2 = -7697172588593208085 as uint64; its comment there says 0x94D049BB133111EB, which is not it)
constexpr uint64_t DS_M1 = 0xBF58476D1CE4E5B9ull, DS_M2 = 0x952E264FE20724EBull, DS_GOLD = 0x9E3779B97F4A7C15ull;
// spec_sample._u01 / noise_group: base = s*M3 ^ p*GOLD ^ ((stream*M1) & (2^63 - 1))
constexpr uint64_t SP_M1 = 0x3F58476D1CE4E5B9ull, SP_M2 = 0x0149D049BB133111ull, SP_GOLD = 0x1E3779B97F4A7C15ull;
constexpr uint64_t M3 = 0x2545F4914F6CDD1Dull;

inline uint64_t hash_ds(uint64_t seed, uint64_t pos, uint64_t col) {
    const uint64_t base = (seed * M3) ^ (pos * DS_GOLD);
    return mix(base + col * M3, DS_M1, DS_M2);
}

inline uint64_t hash_sp(uint64_t seed, uint64_t pos, uint64_t stream, uint64_t col) {
    const uint64_t base = (seed * M3) ^ (pos * SP_GOLD) ^ ((stream * SP_M1) & 0x7FFFFFFFFFFFFFFFull);
    return mix(base + col * M3, SP_M1, SP_M2);
}

inline uint64_t u53(uint64_t h) { return (h >> 11) & 0x1FFFFFFFFFFFFFull; }

// ------------------------------------------------------------------ XF: (-1)^s * m * 2^(e - 63), m normalised (bit 63)
struct XF {
    uint64_t m;
    int32_t e;
    uint32_t s;
};

inline uint32_t clz64(uint64_t x) {
    uint32_t n = 0;
    if (!(x >> 32)) { n += 32; x <<= 32; }
    if (!(x >> 48)) { n += 16; x <<= 16; }
    if (!(x >> 56)) { n += 8; x <<= 8; }
    if (!(x >> 60)) { n += 4; x <<= 4; }
    if (!(x >> 62)) { n += 2; x <<= 2; }
    if (!(x >> 63)) { n += 1; }
    return n;
}

// value m * 2^(e - 63) for any m (0 = zero)
inline XF xf_norm(uint64_t m, int32_t e, uint32_t s) {
    XF r;
    r.s = s;
    if (!m) {
        r.m = 0;
        r.e = 0;
        return r;
    }
    const uint32_t sh = clz64(m);
    r.m = m << sh;
    r.e = e - static_cast<int32_t>(sh);
    return r;
}

inline XF xf_mul(XF a, XF b) {
    XF r;
    r.s = a.s ^ b.s;
    if (!a.m || !b.m) {
        r.m = 0;
        r.e = 0;
        return r;
    }
    const U128 p = mul64(a.m, b.m);  // value p * 2^(a.e + b.e - 126)
    uint64_t m, rnd;
    int32_t e;
    if (p.hi >> 63) {
        m = p.hi;
        rnd = p.lo >> 63;
        e = a.e + b.e + 1;
    } else {
        m = (p.hi << 1) | (p.lo >> 63);
        rnd = (p.lo >> 62) & 1u;
        e = a.e + b.e;
    }
    m += rnd;
    if (!m) {  // all ones rounded up
        m = 1ull << 63;
        e += 1;
    }
    r.m = m;
    r.e = e;
    return r;
}

inline XF xf_add(XF a, XF b) {
    if (!a.m) {
        return b;
    }
    if (!b.m) {
        return a;
    }
    if (a.e < b.e || (a.e == b.e && a.m < b.m)) {
        const XF t = a;
        a = b;
        b = t;
    }
    const uint32_t d = static_cast<uint32_t>(a.e - b.e);
    const uint64_t am = a.m >> 1;
    const uint64_t bm = d >= 63 ? 0 : ((b.m >> 1) >> d);
    const uint64_t r = (a.s == b.s) ? am + bm : am - bm;  // value r * 2^(a.e - 62)
    return xf_norm(r, a.e + 1, a.s);
}

// signed Q62 * Q62 -> Q62 (magnitude truncated)
inline int64_t mulq62(int64_t a, int64_t b) {
    const bool neg = (a < 0) != (b < 0);
    const uint64_t ua = a < 0 ? 0ull - static_cast<uint64_t>(a) : static_cast<uint64_t>(a);
    const uint64_t ub = b < 0 ? 0ull - static_cast<uint64_t>(b) : static_cast<uint64_t>(b);
    const U128 p = mul64(ua, ub);
    const uint64_t r = (p.hi << 2) | (p.lo >> 62);
    return neg ? -static_cast<int64_t>(r) : static_cast<int64_t>(r);
}

inline XF xf_q62(int64_t v) {  // value v * 2^-62
    return v < 0 ? xf_norm(0ull - static_cast<uint64_t>(v), 1, 1) : xf_norm(static_cast<uint64_t>(v), 1, 0);
}

constexpr uint32_t INV[128] = {
    65281, 64777, 64281, 63792, 63310, 62836, 62369, 61909, 61455, 61008, 60568, 60133, 59705, 59283, 58867, 58457,
    58053, 57654, 57260, 56872, 56489, 56111, 55738, 55370, 55007, 54649, 54295, 53946, 53601, 53261, 52925, 52593,
    52265, 51942, 51622, 51306, 50995, 50686, 50382, 50081, 49784, 49490, 49200, 48913, 48630, 48349, 48072, 47798,
    47528, 47260, 46995, 46733, 46474, 46218, 45965, 45714, 45467, 45222, 44979, 44739, 44502, 44267, 44035, 43805,
    43577, 43352, 43129, 42908, 42690, 42474, 42260, 42048, 41838, 41631, 41425, 41222, 41020, 40820, 40623, 40427,
    40233, 40041, 39851, 39662, 39476, 39291, 39108, 38926, 38746, 38568, 38392, 38217, 38044, 37872, 37702, 37533,
    37366, 37200, 37036, 36873, 36712, 36552, 36393, 36236, 36080, 35926, 35772, 35620, 35470, 35320, 35172, 35026,
    34880, 34735, 34592, 34450, 34309, 34169, 34031, 33893, 33757, 33622, 33487, 33354, 33222, 33091, 32961, 32832};
constexpr int64_t LNINV[128] = {
    17979030617864065ll, 53721566841893809ll, 89169311688258771ll, 124385562132551897ll, 159362876675174486ll,
    194020328528059026ll, 228422586551868754ll, 262561939555911652ll, 296505633089862732ll, 330171847459276224ll,
    363552625038834140ll, 396793312413045394ll, 429734616644506846ll, 462446144856591826ll, 494921291898391171ll,
    527153383619098516ll, 559135679021566025ll, 590941360872044003ll, 622565208468182596ll, 653920850503099834ll,
    685082933859715454ll, 716046057544234220ll, 746804763687088968ll, 777353538942321773ll, 807686815984830748ll,
    837798975107428958ll, 867769282719454645ll, 897508180140171349ll, 927095934962885391ll, 956441792932719713ll,
    985627027780727351ll, 1014647380906639480ll, 1043498550060978821ll, 1072087404060262024ll,
    1100586577950519378ll, 1128903408364953640ll, 1156942992926259548ll, 1184972130452083048ll,
    1212714972451425057ll, 1240349457688350988ll, 1267779984612355943ll, 1295095085307753489ll,
    1322197989243369968ll, 1349178260205028570ll, 1375937962185196011ll, 1402663075824565448ll,
    1429160219421642853ll, 1455521029051160012ll, 1481645246047701701ll, 1507723128190822638ll,
    1533654909316137680ll, 1559437277101490695ll, 1585066891895395618ll, 1610540387396121397ll,
    1635854371367312777ll, 1661106306549464322ll, 1686091538841660722ll, 1711008911912876969ll,
    1735856583524529628ll, 1760529609773203711ll, 1785024453083204214ll, 1809441732675919244ll,
    1833674787830593093ll, 1857825293483475296ll, 1881891274998396377ll, 1905764359269902870ll,
    1929547809079034906ll, 1953239589397095388ll, 1976729617800355261ll, 2000122749074639426ll,
    2023416890888693421ll, 2046609932986349629ll, 2069699747424457001ll, 2092573412206805905ll,
    2115449767144301760ll, 2138104531247288524ll, 2160758709137421648ll, 2183298763873390016ll,
    2205608941973345450ll, 2227913500186711094ll, 2250097204940798979ll, 2272157782562693693ll,
    2294092944917893524ll, 2316016663027227881ll, 2337694622512870839ll, 2359357590055245823ll,
    2380886948188003004ll, 2402398811664295218ll, 2423773437510154656ll, 2445008441821560272ll,
    2466101429197205411ll, 2487170662759573955ll, 2508094150512874388ll, 2528991230243507057ll,
    2549738787484579629ll, 2570457236291266051ll, 2591022339732739585ll, 2611555589303424965ll,
    2631931626196854471ll, 2652273020065931162ll, 2672453291418779479ll, 2692596088317760962ll,
    2712700529697381233ll, 2732638456218684461ll, 2752535135648535612ll, 2772261289848969955ll,
    2792072183848846372ll, 2811709603559311122ll, 2831170960412668427ll, 2850714792537888865ll,
    2870079558287146368ll, 2889262644116734241ll, 2908525858652257796ll, 2927737104495755410ll,
    2946762074158881667ll, 2965731986328031060ll, 2984645834407807243ll, 3003502604235256188ll,
    3022165757875122346ll, 3040904746927371352ll, 3059446953680836948ll, 3077926849362793745ll,
    3096481095508447313ll, 3114833754802454675ll, 3133120925918390204ll, 3151341535883061135ll,
    3169494504952779133ll, 3187578746702450052ll};
constexpr int64_t CS[14] = {
    4611686018427387904ll, -2305843009213693952ll, 1537228672809129301ll, -1152921504606846976ll,
    922337203685477581ll, -768614336404564651ll, 658812288346769701ll, -576460752303423488ll, 512409557603043100ll,
    -461168601842738790ll, 419244183493398900ll, -384307168202282325ll, 354745078340568300ll,
    -329406144173384850ll};

// ln(1 + r) / r = sum_k (-r)^k / (k + 1), Horner, n terms, Q62
inline int64_t ln1p_ratio(int64_t r62, uint32_t n) {
    int64_t s = CS[n - 1];
    for (uint32_t k = n - 1; k-- > 0;) {
        s = CS[k] + mulq62(r62, s);
    }
    return s;
}

constexpr uint64_t LN2_M = 0xB17217F7D1CF79ACull;  // ln 2 * 2^64, e = -1

// ln x, x > 0
inline XF xf_ln(XF x) {
    const uint64_t m = x.m;
    const int32_t e = x.e;
    if ((e == 0 && m < (1ull << 63) + (1ull << 57)) || (e == -1 && m >= 0ull - (1ull << 58))) {
        // x in [1 - 2^-6, 1 + 2^-6): r = x - 1 exactly, ln x = r * (ln(1 + r) / r)
        XF r;
        int64_t r62;
        if (e == 0) {
            const uint64_t d = m - (1ull << 63);  // r = d * 2^-63
            if (!d) {
                return xf_norm(0, 0, 0);
            }
            r = xf_norm(d, 0, 0);
            r62 = static_cast<int64_t>(d >> 1);
        } else {
            const uint64_t d = 0ull - m;  // r = -d * 2^-64
            r = xf_norm(d, -1, 1);
            r62 = -static_cast<int64_t>(d >> 2);
        }
        return xf_mul(r, xf_q62(ln1p_ratio(r62, 12)));
    }
    // f = m / 2^63 in [1, 2); c = INV[j] / 2^16 ~ 1 / f; ln f = -ln c + ln(f c), |f c - 1| <= 2^-8 + 2^-16
    const uint32_t j = static_cast<uint32_t>(m >> 56) & 0x7Fu;
    const U128 t = mul64(m, INV[j]);  // f c = t * 2^-79
    const uint64_t q = (t.hi << 47) | (t.lo >> 17);  // Q62
    const int64_t r62 = static_cast<int64_t>(q - (1ull << 62));
    const int64_t lnf = LNINV[j] + mulq62(r62, ln1p_ratio(r62, 10));
    const XF a = xf_q62(lnf);
    if (e == 0) {
        return a;
    }
    const uint64_t ae = static_cast<uint64_t>(e < 0 ? -e : e);
    XF ln2;
    ln2.m = LN2_M;
    ln2.e = -1;
    ln2.s = 0;
    return xf_add(xf_mul(xf_norm(ae, 63, e < 0 ? 1u : 0u), ln2), a);
}

// float32 bits of x, round to nearest even (normal range only: |x| in [2^-126, 2^128))
inline uint32_t xf_f32(XF x) {
    if (!x.m) {
        return x.s << 31;
    }
    uint64_t mant = x.m >> 40;
    const uint64_t rem = x.m & ((1ull << 40) - 1);
    if (rem > (1ull << 39) || (rem == (1ull << 39) && (mant & 1u))) {
        ++mant;
    }
    int32_t e = x.e;
    if (mant >> 24) {
        mant >>= 1;
        ++e;
    }
    const int32_t be = e + 127;
    if (be <= 0) {
        return x.s << 31;
    }
    if (be >= 255) {
        return (x.s << 31) | 0x7F800000u;
    }
    return (x.s << 31) | (static_cast<uint32_t>(be) << 23) | (static_cast<uint32_t>(mant) & 0x7FFFFFu);
}

// the host's clamp(u, 1e-12, 1 - 1e-9) on u = n * 2^-53 (float64 compares, exact here)
constexpr uint64_t N_LO = 9008ull;              // n < N_LO  <=>  n * 2^-53 < double(1e-12)
constexpr uint64_t N_HI = 9007199245733793ull;  // n > N_HI  <=>  n * 2^-53 > double(1 - 1e-9)
constexpr uint64_t ULO_M = 0x8CBCCC096F508800ull;  // double(1e-12) = ULO_M * 2^(-40 - 63)
constexpr int32_t ULO_E = -40;
constexpr uint64_t UHI_M = 0xFFFFFFFBB47D0800ull;  // double(1 - 1e-9) = UHI_M * 2^(-1 - 63)
constexpr int32_t UHI_E = -1;

inline XF u_xf(uint64_t h) {
    const uint64_t n = u53(h);
    if (n < N_LO) {
        return xf_norm(ULO_M, ULO_E, 0);
    }
    if (n > N_HI) {
        return xf_norm(UHI_M, UHI_E, 0);
    }
    return xf_norm(n, 10, 0);  // n * 2^-53
}

// the three noise kinds, float32 bits
inline uint32_t u01_f32(uint64_t h) { return xf_f32(u_xf(h)); }

inline uint32_t expo_f32(uint64_t h) {
    XF l = xf_ln(u_xf(h));
    l.s = 0;  // -ln u > 0
    return xf_f32(l);
}

// T * -ln(-ln u); t_hi:t_lo the float64 bits of T (>= 0); T == 0 -> +0.0
inline uint32_t gumbel_f32(uint64_t h, uint32_t t_hi, uint32_t t_lo) {
    if (!((t_hi & 0x7FFFFFFFu) | t_lo)) {
        return 0u;
    }
    XF l = xf_ln(u_xf(h));
    l.s = 0;
    XF g = xf_ln(l);
    g.s ^= 1u;
    XF t;
    t.m = ((static_cast<uint64_t>(t_hi & 0xFFFFFu) << 32 | t_lo) | (1ull << 52)) << 11;
    t.e = static_cast<int32_t>((t_hi >> 20) & 0x7FFu) - 1023;
    t.s = 0;
    return xf_f32(xf_mul(g, t));
}

}  // namespace fsn

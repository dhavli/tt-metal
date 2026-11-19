// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// cfx (tt-stack patches/tt-metal/0049): silu(gate) * up in one SFPU pass over a gate / up tile pair in DST, driven
// from the PACK thread by the fused-SwiGLU AGMM (QWEN36_AGMM_SWIGLU_PACK), so it overlaps the math thread's next
// subblock. Port of upstream tenstorrent/tt-metal #59325 (open, 10-05) minimal_matmul/device/kernels/swiglu_sfpu.hpp,
// WITHOUT its block-float sigmoid (that path needs fp32 dest off; our prefill AGMM runs fp32 dest).
//
// The arithmetic is exactly silu_tile + mul_binary_tile's on this pin:
// - fp32 dest: calculate_silu<true> computes x * _sfpu_sigmoid_<true>(x) (no rounding) and calculate_sfpu_binary_mul
//   computes in0 * in1 (no rounding); the product is commutative, so for the same gate / up in DST this is bitwise
//   the epilogue's result.
// - bf16 dest: silu rounded to bf16 (RNE), then the product rounded to bf16 (float32_to_bf16_rne, with the FPU's
//   0 * x = 0 rule), as calculate_silu<false> + calculate_sfpu_binary_mul<..., false>.
// - AGMM_SWPK_FAST_SIGMOID (defined by a kernel that includes this header, e.g. tt-stack's c97 served compute.cpp,
//   which runs silu_tile<false> in an fp32-dest epilogue): silu = bf16_rne(x * _sfpu_sigmoid_<false>(x)) (exp_21f,
//   ~1 ULP bf16), then silu * up unrounded -- exactly calculate_silu<false> + an fp32-dest mul_binary_tile.

#if defined(TRISC_PACK) || defined(TRISC_MATH)

#include "ckernel_sfpu_sigmoid.h"
#include "ckernel_sfpu_binary.h"
#include "llk_math_eltwise_binary_sfpu_macros.h"

namespace ckernel::sfpu {

#ifdef AGMM_SWPK_FAST_SIGMOID
constexpr bool kAgmmSwpkFastSigmoid = true;
#else
constexpr bool kAgmmSwpkFastSigmoid = false;
#endif

template <bool is_fp32_dest_acc_en, bool FAST_SIGMOID, int ITERATIONS = 8>
inline void calculate_agmm_swiglu_pack(const uint gate_tile_idx, const uint up_tile_idx, const uint out_tile_idx) {
    constexpr uint dst_tile_size = 32;  // 32 rows per tile in SFPU addressing
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat gate = sfpi::dst_reg[gate_tile_idx * dst_tile_size];
        sfpi::vFloat up = sfpi::dst_reg[up_tile_idx * dst_tile_size];
        sfpi::vFloat silu = gate * _sfpu_sigmoid_<is_fp32_dest_acc_en && !FAST_SIGMOID>(gate);
        sfpi::vFloat result;
        if constexpr (is_fp32_dest_acc_en && FAST_SIGMOID) {
            silu = sfpi::convert<sfpi::vFloat16b>(silu, sfpi::RoundMode::Nearest);
            result = silu * up;
        } else if constexpr (is_fp32_dest_acc_en) {
            result = silu * up;
        } else {
            silu = sfpi::convert<sfpi::vFloat16b>(silu, sfpi::RoundMode::Nearest);
            result = float32_to_bf16_rne(silu * up);
            v_if(silu == 0 || up == 0) { result = 0.0f; }
            v_endif;
        }
        sfpi::dst_reg[out_tile_idx * dst_tile_size] = result;
        sfpi::dst_reg++;
    }
}

// silu_init's constants: _sfpu_sigmoid_'s reciprocal (sfpu_reciprocal_iter) needs vConstFloatPrgm0 = 2.0f, which
// sigmoid_init<false> programs. Nothing else on the pack thread reprograms the SFPU.
inline void agmm_swiglu_pack_init() { sigmoid_init</*APPROXIMATION_MODE=*/false>(); }

}  // namespace ckernel::sfpu

namespace ckernel {

inline void llk_agmm_swiglu_pack_init() {
    llk_math_eltwise_binary_sfpu_init<SfpuType::unused>(ckernel::sfpu::agmm_swiglu_pack_init);
}

inline void llk_agmm_swiglu_pack(uint gate_tile, uint32_t up_tile, uint32_t out_tile) {
    SFPU_BINARY_CALL(
        DST_SYNC_MODE,
        DST_ACCUM_MODE,
        calculate_agmm_swiglu_pack,
        (DST_ACCUM_MODE, ckernel::sfpu::kAgmmSwpkFastSigmoid, 8 /* ITERATIONS */),
        gate_tile,
        up_tile,
        out_tile,
        VectorMode::RC);
}

}  // namespace ckernel

#endif  // TRISC_PACK || TRISC_MATH

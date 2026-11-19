# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP helpers for Qwen3.5/3.6 on Blackhole (9B single-device + 27B TP=4 / TP=8).

Used only when num_devices > 1. DRAM-sharded matmul cfgs, prefill progcfgs,
mesh shard/replicate, FP8 dequant, HF weight reorder for per-device sharding.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import math
import os

import torch
from loguru import logger

import ttnn
from loguru import logger
from models.common.utility_functions import is_blackhole

# Hardware constants
TILE_SIZE = 32
DRAM_CORES = 8
DRAM_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(DRAM_CORES - 1, 0))})


# --------------------------------------------------------------------------------
# Paged KV cache dtype policy (K and V are independent)
# --------------------------------------------------------------------------------
# Only the 16 full-attention layers carry a KV cache; the 48 GDN layers do not.
# Per token, per device, at TP=2 the pair holds 16 layers x 2 local KV heads x 256
# head_dim x 2 (K,V) = 16,384 values. Block-float carries one shared exponent per
# 16 values, so bf8 is 17/16 B/value (measured 17,408 B/token/device,
# [source validation record]) and bf4 is 9/16 B/value.
# K8/V4 is therefore 8,704 + 4,608 = 13,312 B/token/device = 26.00 KiB/token over
# the pair, against 34.00 KiB/token at bf8/bf8 -- exactly 4,096 B/token/device
# saved, i.e. 4.0 GiB/card at a 1,048,576-token pool. That arithmetic is the
# motivation for the knob; nothing here is a measurement.
KV_DTYPE_ALIASES = {
    "bf16": ttnn.bfloat16,
    "bfloat16": ttnn.bfloat16,
    "bf8": ttnn.bfloat8_b,
    "bfp8": ttnn.bfloat8_b,
    "bfp8_b": ttnn.bfloat8_b,
    "bfloat8_b": ttnn.bfloat8_b,
    "bf4": ttnn.bfloat4_b,
    "bfp4": ttnn.bfloat4_b,
    "bfp4_b": ttnn.bfloat4_b,
    "bfloat4_b": ttnn.bfloat4_b,
}


def resolve_kv_dtypes(default_dtype):
    """-> (k_dtype, v_dtype, source). Reads QWEN36_KV_DTYPE, then QWEN_SDPA_BF8.

    QWEN36_KV_DTYPE="<K>/<V>" (e.g. "bf8/bf4") sets the two paged-cache dtypes
    independently and WINS over QWEN_SDPA_BF8; a single token ("bf8") means both.
    QWEN_SDPA_BF8=1 remains the legacy symmetric switch and means bf8/bf8, so every
    historical arm keeps its exact meaning. Unset -> the caller's dtype for both.

    Raises on an unparseable spec: a typo must not silently serve bf16 KV and be
    recorded as a bf4 arm.
    """
    spec = _tt_public_getenv("QWEN36_KV_DTYPE", "").strip()
    if spec:
        parts = [p.strip().lower() for p in spec.split("/")]
        if len(parts) == 1:
            parts = parts * 2
        if len(parts) != 2 or any(p not in KV_DTYPE_ALIASES for p in parts):
            raise ValueError(
                f"QWEN36_KV_DTYPE={spec!r} is not '<K>/<V>' over {sorted(set(KV_DTYPE_ALIASES))}"
            )
        return KV_DTYPE_ALIASES[parts[0]], KV_DTYPE_ALIASES[parts[1]], f"QWEN36_KV_DTYPE={spec}"
    if _tt_public_getenv("QWEN_SDPA_BF8", "0") == "1":
        return ttnn.bfloat8_b, ttnn.bfloat8_b, "QWEN_SDPA_BF8=1"
    return default_dtype, default_dtype, "default"


def log_kv_dtype_banner(k_dtype, v_dtype, source, default_dtype, shape):
    """The runtime evidence of the KV arm. Logs nothing NEW unless QWEN36_KV_DTYPE is
    set, so the shipped path (the flag unset) is byte-identical to the old logs.

    vLLM's "GPU KV cache size" is a token count derived from max_model_len x
    max_num_seqs and is IDENTICAL across KV dtypes -- measured 2026-09-13, both arms
    logging 278,784 at the same shape -- so an A/B that trusts it cannot tell its own
    arms apart. scripts/verify-arm-banners.sh keys off the exact line below.
    """
    if source.startswith("QWEN36_KV_DTYPE"):
        logger.info(
            f"QWEN36_KV_DTYPE engaged: paged KV dtypes K={k_dtype} V={v_dtype} "
            f"(source={source}, caller asked for {default_dtype}), shape {list(shape)}"
        )
        # A bf8/bf8 pair set through the new flag is the same arm production serves, so
        # keep the legacy line the verifier and the serving nix module grep for.
        if k_dtype == ttnn.bfloat8_b and v_dtype == ttnn.bfloat8_b:
            logger.info(
                f"QWEN_SDPA_BF8 engaged: paged KV dtype {ttnn.bfloat8_b} "
                f"(caller asked for {default_dtype}), shape {list(shape)}"
            )
        return
    if source == "QWEN_SDPA_BF8=1":
        # Byte-identical to the pre-port line.
        logger.info(
            f"QWEN_SDPA_BF8 engaged: paged KV dtype {ttnn.bfloat8_b} "
            f"(caller asked for {default_dtype}), shape {list(shape)}"
        )
        return
    logger.info(f"QWEN_SDPA_BF8 not set: paged KV dtype {default_dtype}")


# Compute kernel configs
COMPUTE_HIFI2 = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    math_approx_mode=True,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)

# QWEN36_PROJ_LOFI=1: attention + GDN projection matmuls at LoFi instead of HiFi2. Their weights
# are bfp4 (3 mantissa bits) under G2, which a single LoFi pass already covers, so the second
# HiFi2 pass is pure compute cost (the MLP has run LoFi all along, mlp.py in the image). Scope:
# the linear projections only -- SDPA keeps its own configs (the LoFi+fp32-acc SDPA trap).
if _tt_public_getenv("QWEN36_PROJ_LOFI", "0") == "1":
    COMPUTE_PROJ = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    logger.info("QWEN36_PROJ_LOFI engaged: attention/GDN projection matmuls at LoFi (fp32 acc)")
else:
    COMPUTE_PROJ = COMPUTE_HIFI2


# QWEN36_PREFILL_FS8=1: the row-parallel prefill projections (attn_wo, gdn_out, mlp_down) take upstream's
# tuned-GEMM recipe (tech_reports/GEMM_FLOPS) in its bit-exact form: fp32 dest + dst_full_sync_en, which
# doubles the fp32 DST to 8 tiles, so the output subblock can reach 8 (today <= 4), plus K blocks up to 16 tiles
# (today 4) on the full device-width grid. ttsim: bitwise equal to the production config on all three shapes
# ([source validation record]); one card at M=2048: gdn_out and
# attn_wo 276/268 TF standalone ([source validation record]). The in-model A/B decides.
# Value: "1"/"all" = all three roles, else a comma list of roles (gdn_out, attn_wo, mlp_down). Read per call (not at
# import) so the workbench can flip it in a live process (`ttx wb set --knob QWEN36_PREFILL_FS8=...`, registered in
# tools/ttx/clients/wb_knobs.py); the trace re-capture that `set` forces picks the new configs up.
_FS8_ROLES_ALL = ("gdn_out", "attn_wo", "mlp_down")
_FS8_L1_BUDGET_DRAM_OUT = int(_tt_public_getenv("QWEN36_PREFILL_FS8_L1_BUDGET", "1250000"))
_FS8_L1_BUDGET_L1_OUT = int(_tt_public_getenv("QWEN36_PREFILL_FS8_L1_BUDGET_L1OUT", "900000"))
_FS8_PARSED = {}
_FS8_BANNERED = set()


def _fs8_roles():
    v = _tt_public_getenv("QWEN36_PREFILL_FS8", "0").strip()
    roles = _FS8_PARSED.get(v)
    if roles is None:
        roles = (set(_FS8_ROLES_ALL) if v in ("1", "all") else set() if v in ("", "0")
                 else {r.strip() for r in v.split(",") if r.strip()})
        bad = roles - set(_FS8_ROLES_ALL)
        if bad:
            raise ValueError(f"QWEN36_PREFILL_FS8: unknown role(s) {sorted(bad)}")
        _FS8_PARSED[v] = roles
    return roles


def prefill_fs8(role):
    return role in _fs8_roles()


def prefill_fs8_compute(packer_l1_acc):
    """The row-parallel prefill compute config under QWEN36_PREFILL_FS8 (LoFi, fp32 dest, full-sync DST)."""
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=True,
        packer_l1_acc=packer_l1_acc,
        dst_full_sync_en=True,
    )


# ---------------------------------------------------------------- QWEN36_FAST_MUL (c81)
# binary_ng routes a bf16 ttnn.multiply to the SFPU (RNE) unless fast_and_approximate_mode=True, which
# routes it to the FPU. T2 decode profile (c80): the MLP gate*up mul costs 3.8 us and the GDN out-gate
# mul 2.6 us on the SFPU; an FPU binary costs ~0.9 us. Rounding can differ, so this lever is tier 2.
# Values: 1|decode = decode rows only (M <= 32); all = also prefill.
_FAST_MUL_LOGGED = set()


def fast_mul(site, m):
    """{"fast_and_approximate_mode": True} for `site` at M=m rows when QWEN36_FAST_MUL covers it, else {}."""
    v = _tt_public_getenv("QWEN36_FAST_MUL", "0")
    if v in ("0", ""):
        return {}
    if v not in ("1", "decode", "all"):
        raise ValueError(f"QWEN36_FAST_MUL={v!r}: expected 0|1|decode|all")
    if v != "all" and m > TILE_SIZE:
        return {}
    if site not in _FAST_MUL_LOGGED:
        _FAST_MUL_LOGGED.add(site)
        logger.info(f"QWEN36_FAST_MUL engaged: {site} bf16 multiply on the FPU (fast_and_approximate_mode), scope {v}")
    return {"fast_and_approximate_mode": True}


def create_prefill_fs8_program_config(m, k, n, grid_w, role):
    """2D mcast progcfg for QWEN36_PREFILL_FS8: grid (grid_w, prefill rows), subblock volume <= 8, K block <= 16."""
    gy = prefill_grid_default()[1]
    pm = max(1, math.ceil(m / TILE_SIZE / gy))
    pn = max(1, math.ceil(n / TILE_SIZE / grid_w))
    best = (1, 1)
    for h in range(1, 9):
        for w in range(1, 9):
            if h * w <= 8 and pm % h == 0 and pn % w == 0 and h * w > best[0] * best[1]:
                best = (h, w)
    # L1 fit (c77 wb: bw=16 overflowed, "circular buffers ... grow to 1735680 B ... max L1 size of 1572864 B").
    # Conservative per-core CB estimate: in0 bf16 + in1 (bf8 upper bound) double-buffered, bf16 out, fp32
    # packer-L1-acc interm. Roles whose OUTPUT lives in L1 (attn_wo, mlp_down) also need room for that tensor
    # and the resident activations (the _PREFILL_TUNING note below: cap 8 clashed in the full model), so they get
    # a smaller budget. Largest divisor of K tiles <= 16 that fits.
    k_tiles = math.ceil(k / TILE_SIZE)
    budget = _FS8_L1_BUDGET_DRAM_OUT if role == "gdn_out" else _FS8_L1_BUDGET_L1_OUT
    fixed = pm * pn * (2048 + 4096)
    bw = 1
    for cand in range(min(16, k_tiles), 0, -1):
        if k_tiles % cand == 0 and fixed + 2 * cand * (pm * 2048 + pn * 1088) <= budget:
            bw = cand
            break
    if (role, pm, pn, bw) not in _FS8_BANNERED:
        _FS8_BANNERED.add((role, pm, pn, bw))
        logger.info(
            f"QWEN36_PREFILL_FS8 engaged: role={role} grid={grid_w}x{gy} in0_block_w={bw} "
            f"subblock={best[0]}x{best[1]} per_core={pm}x{pn} (fp32 dest, dst_full_sync)"
        )
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(grid_w, gy),
        in0_block_w=bw,
        out_subblock_h=best[0],
        out_subblock_w=best[1],
        per_core_M=pm,
        per_core_N=pn,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )

# Grid helpers
def prefill_grid_default():
    """BH P150: (8,10); WH: (8,8). y capped at 10 on BH (grid_x=10 breaks matmul)."""
    return (8, 10) if is_blackhole() else (8, 8)


# Max grid COLUMNS a tuned prefill config may use. A Blackhole galaxy reports a 12-wide worker
# grid, but harvested P150s expose only 11, so tuning to 12 would not port. 11 x 10 = 110 cores.
PREFILL_MAX_COLS_PORTABLE = 11

# Why TP=8 wants different values (measured at S=2048, 27B, 1x8 Ring):
#   * widest_cols -- `_best_prefill_cols` ranks candidate widths by (out_subblock_w, cols), i.e.
#     subblock first. At TP=8 the halved N makes wide grids yield a small per_core_N and hence a
#     narrow subblock, so that ranking retreats to fewer columns and leaves cores idle. Measured
#     device time is monotonically decreasing in column count instead: attn_wo went 1944us @ 60
#     cores -> 700us @ 110, and mlp_gate 2943us @ 60 -> 1935us @ 110. So take the width.
#   * in0_block_w_divisor -- `min(cap, k_tiles // grid_x)` is a function of the per-device K, which
#     halves. attn_wo/gdn_out go k_tiles 48 -> 24 and `24 // 11 = 2`, but in0_block_w only has to
#     DIVIDE k_tiles, so a larger block is legal and much faster (attn_wo @ 11 cols, from the sweep:
#     bw2 786us, bw4 719us, bw6 700us, bw8 705us).
#
# in0_block_w_cap is L1-BOUND, NOT just a legality bound. in0_block_w sizes the in0 circular
# buffer, and `_wo_proj` / the MLP prefill arm write their OUTPUT to L1 (attention/tp.py:246,
# mlp.py:284) -- so the CBs and a resident L1 output tensor compete for the same 1536 KB. Measured
# on the real model: cap=8 overflows and test_model_tp_long_prefill dies with
#   "Statically allocated circular buffers in program N clash with L1 buffers on core range
#    [0-0 - 10-8]. L1 buffer allocated at 1314560 and static circular buffer region ends at 1372032"
# from attention/tp.py:241. A standalone per-op sweep CANNOT see this: in isolation the only L1
# tenant is the op under test, so it reports a win that the full model has no room for. Any future
# raise of this cap must be validated by test_model_tp_long_prefill, not by the sweep alone.
_PREFILL_TUNING = {
    4: dict(widest_cols=False, in0_block_w_divisor=False, in0_block_w_cap=4),
    8: dict(widest_cols=True, in0_block_w_divisor=True, in0_block_w_cap=4),
}


def prefill_tuning(num_devices):
    """Prefill matmul tuning for this TP; unknown TP falls back to the frozen TP=4 values."""
    return _PREFILL_TUNING.get(num_devices, _PREFILL_TUNING[4])


def _roundup(a, b):
    return b * math.ceil(a / b)


def _find_largest_divisor(n, max_div=8):
    for d in range(max_div, 0, -1):
        if n % d == 0:
            return d
    return 1


def _find_grid(n_tiles, target=32):
    max_r, max_c = 8, 8
    possible = [k for k in range(1, max_r * max_c + 1) if n_tiles % k == 0]
    possible.sort(key=lambda x: abs(x - target))
    for cores in possible:
        for rows in range(1, max_r + 1):
            if cores % rows == 0:
                cols = cores // rows
                if cols <= max_c:
                    return rows, cols
    raise ValueError(f"Cannot find grid for {n_tiles} tiles")


# DRAM-sharded config builders
def create_dram_sharded_mem_config(k, n):
    """WIDTH_SHARDED DRAM memory config for a weight matrix [k, n]."""
    padded_n = _roundup(n, TILE_SIZE * DRAM_CORES)
    shard_spec = ttnn.ShardSpec(
        DRAM_GRID,
        (k, padded_n // DRAM_CORES),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        shard_spec,
    )


def create_dram_sharded_matmul_program_config(m, k, n, num_cores=None):
    """DRAM-sharded matmul program config (decode, small M)."""
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_padded = _roundup(n, TILE_SIZE * DRAM_CORES)
    n_tiles = n_padded // TILE_SIZE

    if num_cores is None:
        rows, cols = _find_grid(k_tiles)
        num_cores = rows * cols

    k_tiles_per_core = k_tiles // num_cores
    if k_tiles_per_core == 0:
        k_tiles_per_core = k_tiles
        num_cores = 1
    in0_block_w = _find_largest_divisor(k_tiles_per_core)
    per_core_N = n_tiles // num_cores if n_tiles >= num_cores else 1

    return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
        in0_block_w=in0_block_w,
        per_core_M=m_tiles,
        per_core_N=per_core_N,
        fused_activation=None,
    )


def create_matmul_1d_decode_progcfg(
    m, k, n, num_cores, fused_activation=None, fp32_acc=True, grid_w=8, max_in0_block_w=8
):
    """Explicit-grid 1D (mcast_in0) decode matmul progcfg on ~`num_cores` cores — small grids beat
    the ~80-core DRAM-sharded grid on the bandwidth-bound skinny decode matmuls. Weight must be interleaved.

    Grid is shaped WIDE-first (cols up to `grid_w`, the device worker-grid width — 11 on BH P150, 8 on
    WH): for a fixed core budget a wide-short grid shortens the in0 multicast column and beats a
    tall-narrow one (~2% on this matmul; see test_mlp_matmul_sweep wide1d_* vs forced1d_*). Default
    grid_w=8 preserves the legacy shaping for callers that don't pass the device width."""
    cols = min(grid_w, num_cores)
    rows = math.ceil(num_cores / cols)
    m_tiles = math.ceil(m / TILE_SIZE)
    k_tiles = math.ceil(k / TILE_SIZE)
    n_tiles = math.ceil(n / TILE_SIZE)
    # mcast_in0: every core streams the full K, so in0_block_w must divide the full k_tiles.
    # max_in0_block_w: in1 CB per core is in0_block_w x per_core_N tiles (double-buffered); a very wide
    # N (lm_head: 3880 tiles) needs a small K block to fit L1. Default 8 keeps every existing config.
    per_core_k = _find_largest_divisor(k_tiles, max_div=max_in0_block_w)
    per_core_n = math.ceil(n_tiles / (cols * rows))
    cap = 4 if fp32_acc else 8  # fp32_dest_acc caps subblock area at 4
    sub_w = max(i for i in range(1, cap + 1) if per_core_n % i == 0)
    sub_h = max(i for i in range(1, cap + 1) if m_tiles % i == 0 and i * sub_w <= cap)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(cols, rows),
        in0_block_w=per_core_k,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        per_core_M=m_tiles,
        per_core_N=per_core_n,
        fuse_batch=True,
        fused_activation=fused_activation,
        mcast_in0=True,
    )


def matmul_1d_decode(x, weight, decode_1d_progcfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """Small-grid 1D (mcast_in0) decode matmul on an interleaved weight; interleaves the K-sharded
    activation first since mcast_in0 needs the full K per core. See test_mlp_matmul_sweep."""
    x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.linear(
        x_il,
        weight,
        compute_kernel_config=compute_cfg,
        program_config=decode_1d_progcfg,
        memory_config=out_memory_config,
    )
    if x_il is not x:
        ttnn.deallocate(x_il)
    return out


def create_activation_shard_config(k):
    """WIDTH_SHARDED L1 activation config for a [*, k] activation."""
    k_tiles = k // TILE_SIZE
    rows, cols = _find_grid(k_tiles)
    num_cores = rows * cols
    width_per_core = k // num_cores
    return ttnn.create_sharded_memory_config(
        shape=(TILE_SIZE, width_per_core),
        core_grid=ttnn.CoreGrid(x=cols, y=rows),
        strategy=ttnn.ShardStrategy.WIDTH,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


# 2D prefill matmul config
def _get_out_subblock_w(per_core_n, out_subblock_h):
    for w in range(min(per_core_n, 4 // out_subblock_h), 0, -1):
        if per_core_n % w == 0:
            return w
    return 1


def _full_grid_crs(grid):
    """Full-grid allowed_worker_cores for CCL-fused matmuls, which bypass ttnn::prim::matmul()'s normalize_program_config()."""
    gx, gy = grid
    return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})


def create_prefill_matmul_program_config(m, k, n, grid_size=None, fused_activation=None, tuning=None):
    """2D prefill matmul progcfg (DRAM-interleaved).

    fused_activation in packer; sharded kernel rejects ttnn.linear(activation=...) with progcfg.
    tuning: a `_PREFILL_TUNING` entry (see `prefill_tuning`); None = the frozen TP=4 behavior."""
    if grid_size is None:
        grid_size = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    per_core_M = max(1, math.ceil(m / TILE_SIZE / grid_size[1]))
    per_core_N = max(1, math.ceil(n / TILE_SIZE / grid_size[0]))

    out_subblock_h = 1
    out_subblock_w = _get_out_subblock_w(per_core_N, out_subblock_h)
    ov = prefill_subblock_override()
    if ov is not None:
        ow, oh = ov
        if oh * ow <= 8 and per_core_M % oh == 0 and per_core_N % ow == 0:
            out_subblock_h, out_subblock_w = oh, ow
        else:
            logger.info(f"QWEN36_PREFILL_SUBBLOCK refused: {ow}x{oh} illegal for per_core {per_core_M}x{per_core_N} (reg/divisibility); picker kept")

    k_tiles = math.ceil(k / TILE_SIZE)
    cap = tuning["in0_block_w_cap"]
    if tuning["in0_block_w_divisor"]:
        # in0_block_w only has to divide k_tiles (no K tail in the 2D mcast kernel), so take the
        # largest legal block rather than scaling with grid width -- see _PREFILL_TUNING.
        in0_block_w = _find_largest_divisor(k_tiles, cap)
    else:
        in0_block_w = min(cap, max(1, k_tiles // grid_size[0]))

    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid_size,
        in0_block_w=in0_block_w,
        out_subblock_h=out_subblock_h,
        out_subblock_w=out_subblock_w,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=fused_activation,
        fuse_batch=False,
    )


def _widest_prefill_cols(n, max_cols, subblock_slack=1):
    """Widest grid whose output subblock stays within `subblock_slack` of the best achievable.

    The TP=8 counterpart to `_best_prefill_cols`. More columns is usually a win at TP=8 (the halved
    per-device N leaves cores idle), but NOT when the extra width collapses the subblock: measured
    at S=2048, mlp_gate (N=2176 -> 68 tiles) goes cols 9 -> 11, per_core_N 8 -> 7, and 7 is prime so
    out_subblock_w drops 4 -> 1 -- a 2058us -> 2118us REGRESSION, i.e. the subblock-first ranking
    was right for that shape. Guarding on the subblock keeps the wide grid exactly where it pays:

        matmul     default        this rule       measured
        attn_wo    c10_bw2_sw4    c11_bw4_sw3     803.5 -> 718.7us
        gdn_out    c10_bw2_sw4    c11_bw4_sw3     802.3 -> 719.9us
        mlp_down   c10_bw4_sw4    c11_bw4_sw3    1787.4 -> 1724.9us
        mlp_gate   c9_bw4_sw4     c9_bw4_sw4     2058.1us (unchanged -- already optimal)
    """
    n_tiles = math.ceil(n / TILE_SIZE)
    sw = {cols: _get_out_subblock_w(math.ceil(n_tiles / cols), 1) for cols in range(1, max_cols + 1)}
    floor = max(sw.values()) - subblock_slack
    return max((cols for cols, w in sw.items() if w >= floor), default=1)


def _best_prefill_cols(n, max_cols):
    """Grid width (<=max_cols) maximizing the output subblock, tie-broken to more cores — avoids the
    1x1-subblock stall (e.g. gate/up N=4352 -> 7-wide -> 1x4) the default full width can force."""
    n_tiles = math.ceil(n / TILE_SIZE)
    best_cols, best_key = 1, None
    for cols in range(1, max_cols + 1):
        sw = _get_out_subblock_w(math.ceil(n_tiles / cols), 1)
        key = (sw, cols)  # prefer wider subblock, then more columns (more compute cores)
        if best_key is None or key > best_key:
            best_key, best_cols = key, cols
    return best_cols


def create_prefill_mlp_matmul_program_config(m, k, n, fused_activation=None, max_cols=None, tuning=None):
    """FPU-tuned 2D prefill progcfg for MLP matmuls: picks the grid width that maximizes the output
    subblock (drives prefill FPU) instead of the default full width.

    max_cols caps the grid width. Default = prefill_grid_default()[0] (8). Pass the device worker-grid
    width (11 on BH P150) to let the subblock heuristic go wide -> the measured prefill winners
    (gate 9-wide, down/wo 10-wide, gdn_qkvz 11-wide; test_mlp_matmul_sweep_prefill). Fused AG/RS paths
    pin 8-wide separately and are unaffected.

    tuning: a `_PREFILL_TUNING` entry. With `widest_cols` (TP=8) the subblock-first width heuristic
    is replaced by "take the width, clamped to PREFILL_MAX_COLS_PORTABLE" -- measured device time at
    TP=8 falls monotonically with column count, so trading cores for a wider subblock loses."""
    grid = prefill_grid_default()
    tuning = tuning or _PREFILL_TUNING[4]
    limit = max_cols or grid[0]
    if tuning["widest_cols"]:
        # Cap the width at PREFILL_MAX_COLS_PORTABLE (harvested parts expose 11, not 12) and never
        # exceed the output tile count -- columns beyond it get per_core_N=1 with nothing to compute,
        # paying mcast cost for no work.
        cols = _widest_prefill_cols(n, max(1, min(limit, PREFILL_MAX_COLS_PORTABLE, math.ceil(n / TILE_SIZE))))
    else:
        cols = _best_prefill_cols(n, limit)
    return create_prefill_matmul_program_config(
        m, k, n, grid_size=(cols, grid[1]), fused_activation=fused_activation, tuning=tuning
    )


# Mesh tensor helpers
def shard_w(torch_tensor, mesh, dim, memory_config, cache_path, dtype=ttnn.bfloat8_b):
    """Torch weight [out,in] -> sharded mesh tensor. Transpose to [in,out]; dim=-1 column, dim=0 row."""
    w = torch_tensor.to(torch.bfloat16).T.contiguous()
    return ttnn.as_tensor(
        w,
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=memory_config,
        cache_file_name=cache_path,
    )


_AGMM_KB_LOGGED = set()


def agmm_k_block_size(k_local, default=8):
    """Largest power-of-2 K_block_size <= `default` that divides K_tiles/device (AGMM Ring has no tail).

    TP=4: 1280->40 tiles->8; TP=8: 640->20 tiles->4. Odd divisors (e.g. 5|20) are unsafe on Ring.
    c101 QWEN36_AGMM_KBLOCK=N (power of 2), read per call (wb set screens it): the cap `default` for every prefill AGMM.
    Unset = 8, the shipped config. K accumulation order is unchanged per output tile at fp32 dest (blocks are summed in
    order), so it should be bit-exact; L1 per block grows with N.
    """
    kb = _tt_public_getenv("QWEN36_AGMM_KBLOCK", "")
    if kb:
        default = int(kb)
        if default & (default - 1):
            raise ValueError(f"QWEN36_AGMM_KBLOCK={kb}: must be a power of 2")
        if kb not in _AGMM_KB_LOGGED:
            _AGMM_KB_LOGGED.add(kb)
            logger.info(f"QWEN36_AGMM_KBLOCK engaged: prefill AGMM K_block_size cap {default} (K_local={k_local})")
    k_tiles = k_local // TILE_SIZE
    b = 1 << (min(default, max(1, k_tiles)).bit_length() - 1)
    while b > 1 and k_tiles % b:
        b //= 2
    return b


_LOGGED_AGMM_GRID = False


def _agmm_prefill_grid(default=(8, 9)):
    """Prefill AGMM grid override via QWEN36_AGMM_PREFILL_GRID=COLSxROWS (default 8x9).

    Default is (8, 9) because Blackhole P150 worker grid has 10 rows (0..9) where row 9
    is reserved for 2-link CCL Fabric Mux workers. A 10-row matmul grid collides on row 9
    with the mux kernels on RISCV_0.
    """
    global _LOGGED_AGMM_GRID
    override = _tt_public_getenv("QWEN36_AGMM_PREFILL_GRID")
    if override:
        try:
            gx, gy = (int(v) for v in override.lower().split("x"))
            if not _LOGGED_AGMM_GRID:
                print(f"QWEN36_AGMM_PREFILL_GRID engaged: grid=({gx}, {gy})", flush=True)
                _LOGGED_AGMM_GRID = True
            return (gx, gy)
        except Exception:
            pass
    return default

_LOGGED_AGMM_LAYOUT = set()


def _agmm_nt11x8(n_tiles, m_rows, role):
    """QWEN36_AGMM_LAYOUT=nt11x8 (upstream tt-metal#56119 `_agmm_layout`, QB2 production layout).

    Default AGMM: TRANSPOSED 8x9 grid, gather (in0) axis = the 8 columns = 2 links x 4 workers, N over
    9 rows, 72 compute cores. nt11x8: UNTRANSPOSED 11x8 grid, gather axis = the 8 ROWS = 2 links x 4
    workers exactly as before (so the op's `(in0_axis + workers - 1) / workers == num_links` check that
    refused every *transposed* 11-column grid in c19 is satisfied), N over 11 columns, fabric muxes stay
    on the free row 9, 88 compute cores. Per chip, intra-chip only: nothing here depends on the mesh
    size. The untransposed orientation needs N >= M (the op auto-transposes otherwise), so it engages
    only when N > S (all TP=2 in-projections and the packed gate|up at S=2048 qualify).

    Returns (grid, force_transpose, workers_per_link) or None when not engaged. Third-party (QB2 TP=4,
    standalone): GDN in-proj 587->457 us, attention in-proj 603->450 us, MLP gate|up 751->673 us."""
    if _tt_public_getenv("QWEN36_AGMM_LAYOUT", "t8x9") != "nt11x8":
        return None
    if m_rows is not None and n_tiles * TILE_SIZE <= m_rows:
        return None
    grid = (11, 8)
    if role not in _LOGGED_AGMM_LAYOUT:
        _LOGGED_AGMM_LAYOUT.add(role)
        logger.info(
            f"QWEN36_AGMM_LAYOUT engaged: nt11x8 role={role} grid={grid} untransposed, 2 links x 4 workers, "
            f"N={n_tiles} tiles -> {math.ceil(n_tiles / grid[0])}/core, S={m_rows}"
        )
    return grid, False, grid[1] // 2


# ---------------------------------------------------------------- QWEN36_AGMM_GRID9 (c134, branch agmm-grid)
# nt11x8 computes on 88 of the 110 Tensix cores: the untransposed grid splits M (the chunk's tile rows) over its ROWS
# and row 9 holds the fabric muxes (all_gather_minimal_matmul_async_program_factory.cpp:545-556: in0 muxes sit on
# full_grid_size.y - 1 when the grid is not transposed and spans every column). Row 8 is idle only because 64 tile
# rows (S=2048) do not split over 9 rows. At S=2304 (72 tile rows, QWEN36_PREFILL_CHUNK=2304) they do, 8 per row as
# today, so 11x9 = 99 cores does 12.5% more tokens per call at the same per-core work. The gather axis becomes 9 rows
# = 2 links x 5 workers (5 + 4): the factory's own group check (:494-500) wants ceil(9/5) == 2 links, and it clamps
# the short group's mux client count (:1316-1321). N stays over the same 11 columns, so N_tiles_per_core, the SwiGLU
# pair split, every block size and every CB are unchanged; so is the K order (bit-exact per output tile by
# construction: the K loop and blocks do not depend on the M split).
#   QWEN36_AGMM_GRID9           comma list of roles {mlp, gdn_in, attn_in, gdn_out, attn_out} or all. Read per call.
#                               Engages only with nt11x8 engaged and M_tiles % 9 == 0; otherwise that call keeps
#                               nt11x8 and a fallback banner says why (masked buckets 128..2048 never qualify).
#   QWEN36_AGMM_GRID9_MUSTFAIL  1: inside an engaged 11x9 call, swap the input's first and last M row-groups
#                               (tile rows 0-7 and 64-71) before the op and do not swap back: a wrong M-tile
#                               assignment that must FAIL PCC (proves the 11x9 branch ran).
AGMM_GRID9_ROLES = ("mlp", "gdn_in", "attn_in", "gdn_out", "attn_out")
_AGMM_GRID9_LOGGED = set()


def _agmm_grid9_roles():
    v = _tt_public_getenv("QWEN36_AGMM_GRID9", "").strip()
    if not v or v == "0":
        return frozenset()
    roles = set(AGMM_GRID9_ROLES) if v in ("1", "all") else {r.strip() for r in v.split(",") if r.strip()}
    bad = roles - set(AGMM_GRID9_ROLES)
    if bad:
        raise ValueError(f"QWEN36_AGMM_GRID9={v!r}: unknown roles {sorted(bad)} (expected {AGMM_GRID9_ROLES} or all)")
    return frozenset(roles)


def agmm_grid9_mustfail():
    return _tt_public_getenv("QWEN36_AGMM_GRID9_MUSTFAIL", "0") == "1"


def agmm_grid9(role, S, nt):
    """(grid, force_transpose, workers_per_link) for an engaged 11x9 call, else None (the caller keeps `nt`)."""
    roles = _agmm_grid9_roles()
    if role not in roles:
        return None
    m_tiles = -(-int(S) // TILE_SIZE)
    if nt is None or m_tiles % 9:
        key = ("fallback", role, int(S), nt is None)
        if key not in _AGMM_GRID9_LOGGED:
            _AGMM_GRID9_LOGGED.add(key)
            why = "needs QWEN36_AGMM_LAYOUT=nt11x8 with N > S" if nt is None else f"M={m_tiles} tiles % 9 != 0"
            logger.info(f"QWEN36_AGMM_GRID9 fallback: role={role} S={S} keeps the stock grid ({why})")
        return None
    grid, workers = (11, 9), 5
    key = ("engaged", role, int(S))
    if key not in _AGMM_GRID9_LOGGED:
        _AGMM_GRID9_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_GRID9 engaged: role={role} grid={grid} untransposed, 2 links x {workers}+"
                    f"{grid[1] - workers} workers, M={m_tiles} tiles -> {m_tiles // grid[1]}/row, S={S}")
    return grid, False, workers


def agmm_grid9_mustfail_swap(x4, role):
    """QWEN36_AGMM_GRID9_MUSTFAIL: x4 [1,1,S,K] with its first and last 1/9 of the rows swapped (new tensor)."""
    S, K = x4.shape[-2], x4.shape[-1]
    r = S // 9
    key = ("mustfail", role)
    if key not in _AGMM_GRID9_LOGGED:
        _AGMM_GRID9_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_GRID9_MUSTFAIL engaged: role={role} M row-groups 0 and 8 ({r} rows each) swapped "
                    f"before the op and not restored (must FAIL PCC)")
    head = ttnn.slice(x4, (0, 0, 0, 0), (1, 1, r, K))
    mid = ttnn.slice(x4, (0, 0, r, 0), (1, 1, S - r, K))
    tail = ttnn.slice(x4, (0, 0, S - r, 0), (1, 1, S, K))
    out = ttnn.concat([tail, mid, head], dim=2)
    for t in (head, mid, tail):
        ttnn.deallocate(t)
    return out


# ---------------------------------------------------------------- QWEN36_AGMM_T11X9 (10-06, prefill 8k PR)
# nt11x8 splits M over its 8 ROWS (row 8 idle: 64 tile rows at the 2048 chunk do not split over 9, and row 9 holds the
# fabric muxes), so the critical core does 8 tile rows. TRANSPOSED 11x9 (force_transpose: M over the 11 COLUMNS, N over
# 9 rows, muxes stay on the free row 9) gives each core ceil(64/11) = 6 tile rows (66 padded) on 99 cores. The gather
# axis becomes the 11 columns = 2 links x 6 workers (6 + 5): the factory's group check ceil(11/6) == 2 holds. (c19 and
# an internal record refused "11-column transposed" with 5 workers per link, which fails that check.) Per-core work
# vs nt11x8 [estimate, code-read]: swiglu 6 x 62 vs 8 x 50 tiles (-7%), gdn_in 6 x 29 vs 8 x 24 (-9%), attn_in 6 x 25
# vs 8 x 21 (-11%), out-projs 6 x 9 vs 8 x 8 (-16%). Upstream factory :311-319 (transpose), :494 (group check).
# Bit-exact by construction: the K loop, K_block and the N block cap do not depend on the M split or the orientation.
#   QWEN36_AGMM_T11X9           comma list of roles {mlp, gdn_in, attn_in, gdn_out, attn_out} or all. Read per call.
#                               Engages only with nt11x8 engaged, GRID9 not engaged for the role, and when it lowers the
#                               tile rows per core (ceil(M/11) < ceil(M/8): buckets >= 1024); else nt11x8, with a banner.
#   QWEN36_AGMM_T11X9_MUSTFAIL  1: the warmup self-test's candidate runs on a zeroed-row input -> the boot is REFUSED.
# The first engaged call per role is self-tested (prefill_selftest): nt11x8 vs t11x9 bitwise on the live inputs, plus a
# wrong-input rung that must differ.
_AGMM_T11X9_LOGGED = set()
_AGMM_T11X9_FORCE_OFF = [False]


def _agmm_t11x9_roles():
    v = _tt_public_getenv("QWEN36_AGMM_T11X9", "").strip()
    mf = _tt_public_getenv("QWEN36_AGMM_T11X9_MUSTFAIL", "").strip()
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_AGMM_T11X9_MUSTFAIL={mf!r}: expected unset, 0 or 1")
    if not v or v == "0":
        if mf == "1":
            raise ValueError("QWEN36_AGMM_T11X9_MUSTFAIL=1 needs QWEN36_AGMM_T11X9")
        return frozenset()
    roles = set(AGMM_GRID9_ROLES) if v in ("1", "all") else {r.strip() for r in v.split(",") if r.strip()}
    bad = roles - set(AGMM_GRID9_ROLES)
    if bad:
        raise ValueError(f"QWEN36_AGMM_T11X9={v!r}: unknown roles {sorted(bad)} (expected {AGMM_GRID9_ROLES} or all)")
    return frozenset(roles)


def agmm_t11x9_mustfail():
    return _tt_public_getenv("QWEN36_AGMM_T11X9_MUSTFAIL", "") == "1"


def agmm_t11x9(role, S, nt, g9):
    """(grid, force_transpose, workers_per_link) for an engaged transposed 11x9 call, else None (keep nt/g9)."""
    if _AGMM_T11X9_FORCE_OFF[0] or role not in _agmm_t11x9_roles():
        return None
    m_tiles = -(-int(S) // TILE_SIZE)
    why = None
    if nt is None:
        why = "needs QWEN36_AGMM_LAYOUT=nt11x8 with N > S"
    elif g9 is not None:
        why = "QWEN36_AGMM_GRID9 engaged for this role"
    elif -(-m_tiles // 11) >= -(-m_tiles // 8):
        why = f"M={m_tiles} tiles: no fewer rows per core than nt11x8"
    if why:
        key = ("fallback", role, int(S))
        if key not in _AGMM_T11X9_LOGGED:
            _AGMM_T11X9_LOGGED.add(key)
            logger.info(f"QWEN36_AGMM_T11X9 fallback: role={role} S={S} keeps nt11x8 ({why})")
        return None
    grid, workers = (11, 9), 6
    key = ("engaged", role, int(S))
    if key not in _AGMM_T11X9_LOGGED:
        _AGMM_T11X9_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_T11X9 engaged: role={role} grid={grid} transposed, 2 links x {workers}+"
                    f"{grid[0] - workers} workers, M={m_tiles} tiles -> {-(-m_tiles // 11)}/column (was "
                    f"{-(-m_tiles // 8)}/row on nt11x8), S={S}"
                    + (" MUSTFAIL: self-test candidate on a zeroed-row input" if agmm_t11x9_mustfail() else ""))
    return grid, True, workers


_AGMM_T11X9_ACTIVE = set()


def agmm_t11x9_selftest(role, call, x):
    """Once per role and process, on the first engaged call: `call(x_in)` (the AGMM wrapper, re-entered) as nt11x8
    (ref), as t11x9 (cand) and as t11x9 on x with its first 32 rows zeroed (wrong). Raises on a mismatch."""
    def ref():
        _AGMM_T11X9_FORCE_OFF[0] = True
        try:
            return call(x)
        finally:
            _AGMM_T11X9_FORCE_OFF[0] = False

    def zeroed():
        xz = zero_first_rows(x)
        try:
            return call(xz)
        finally:
            ttnn.deallocate(xz)

    _AGMM_T11X9_ACTIVE.add(role)
    try:
        prefill_selftest(f"QWEN36_AGMM_T11X9[{role}]", ref=ref,
                         cand=zeroed if agmm_t11x9_mustfail() else (lambda: call(x)),
                         wrong=zeroed, desc=f"role={role} S={x.shape[-2]} K_local={x.shape[-1]}",
                         ok_banner=f"QWEN36_AGMM_T11X9[{role}] self-test exact")
    finally:
        _AGMM_T11X9_ACTIVE.discard(role)


def agmm_t11x9_needs_selftest(role):
    return role not in _AGMM_T11X9_ACTIVE and f"QWEN36_AGMM_T11X9[{role}]" not in _PREFILL_SELFTEST


_SWPK_MODES = {"1": "pack-thread SwiGLU + injector write deferral", "nodefer": "pack-thread SwiGLU",
               "defer": "injector write deferral"}
_SWPK_LOGGED = set()
_SWPK_ACTIVE = set()
# Bound for the tier-2 (pack-thread SwiGLU) self-test: relative Frobenius error of the hidden vs the stock epilogue.
# The only difference is the rounding of the last K block's partial-sum add (FPU dest-reuse vs packer L1 accumulation
# + unpack), which flips ~3.7% of the bf8 hidden by one step: ttsim S=1024 measured 5.33e-3 vs stock
# ([source validation record]), so 1e-2 (~1.9x margin). The zeroed-row wrong rung sits near
# sqrt(32/S) (0.125 at S=2048, the largest prefill chunk), above 10 x this.
AGMM_SWPK_SELFTEST_TOL = 1e-2


# ---------------------------------------------------------------- QWEN36_AGMM_SWIGLU_PACK (10-06, patch 0049, #59325)
# The fused SwiGLU AGMM (all_gather_swiglu_prefill) runs silu(gate)*up as a serial MATH-thread epilogue after every
# output block (copy each gate/up pair back from the fp32 intermediate, silu_tile, mul_binary_tile, pack). Patch 0049
# ports upstream tt-metal #59325: in the last K block each subblock adds its partial sums with one dest-reuse add and
# the PACK thread runs the SFPU silu*up on its DST half and packs the half-width output, so the SFPU overlaps the math
# thread's next subblock. Its second commit makes injector cores defer their output writes too (bit-exact, every AGMM).
#   QWEN36_AGMM_SWIGLU_PACK  1 (pack-thread SwiGLU + deferral) | nodefer (pack only) | defer (deferral only, every
#                            role). Read by the C++ op on EVERY call (hashed); unset/0 = the stock programs.
#   QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL  1: the self-test candidate runs the C++ must-fail mode (no partial-sum add) ->
#                            the boot is REFUSED. Never set QWEN36_AGMM_SWIGLU_PACK=mustfail directly.
# Numerics: deferral is bit-exact; the pack-thread SwiGLU is tier 2 (SFPU arithmetic identical, the partial-sum add
# rounds differently). First engaged call per role: prefill_selftest_tol (mlp, pack modes; tol above) or the bitwise
# prefill_selftest (deferral only; the matmul roles), each with a zeroed-row wrong rung.
def agmm_swiglu_pack():
    v = _tt_public_getenv("QWEN36_AGMM_SWIGLU_PACK", "").strip()
    mf = _tt_public_getenv("QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL", "").strip()
    if v == "mustfail" and _SWPK_ACTIVE:  # the self-test's own must-fail candidate (set below, restored after)
        return v
    if v not in ("", "0") and v not in _SWPK_MODES:
        raise ValueError(f"QWEN36_AGMM_SWIGLU_PACK={v!r}: expected unset, 0, 1, nodefer or defer "
                         "(the must-fail is QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL=1)")
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL={mf!r}: expected unset, 0 or 1")
    if mf == "1" and v not in ("1", "nodefer") and not (v == "0" and _SWPK_ACTIVE):  # "0": the self-test's ref
        raise ValueError("QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL=1 needs QWEN36_AGMM_SWIGLU_PACK=1 or nodefer")
    agmm_swpk_exact(v)  # QWEN36_AGMM_SWPK_EXACT validation (raises on a bad combination)
    return v if v in _SWPK_MODES else None


# ---------------------------------------------------------------- QWEN36_AGMM_SWPK_EXACT (agmm-eff, 2026-10-15)
# Bit-exact pack-thread SwiGLU ("swpk-x"). 0049's pack mode is tier 2 only because its last-K-block partial-sum add is an
# FPU dest-reuse add (both operands through TF32 SrcA/SrcB). The swpk-x kernel (bench/probes/agmm_swiglu_pack/
# compute.exact.cpp, mounted over the AGMM compute.cpp by the arm; marker SWPK_EXACT_V1) keeps stock's packer fp32 L1
# accumulation for the last K block, reloads each subblock's final sums through SrcA exactly as stock's swiglu_block
# copy_tile does, and runs 0049's SFPU silu*up (c97 sigmoid) on the PACK thread. Same values, same arithmetic: tier 1.
# No C++ change: it rides 0049's C++ pack mode (QWEN36_AGMM_SWIGLU_PACK=1 or nodefer; SWIGLU_PACK define) and 0049's
# must-fail mode (SWIGLU_PACK_MUSTFAIL define: the exact kernel then packs the last K block WITHOUT L1 accumulation).
#   QWEN36_AGMM_SWPK_EXACT=1           engage: needs QWEN36_AGMM_SWIGLU_PACK=1|nodefer and the exact kernel mounted
#                                      (refuses when the AGMM compute.cpp on disk lacks the SWPK_EXACT_V1 marker); the
#                                      mlp warmup self-test turns BITWISE (prefill_selftest) on both chips.
#   QWEN36_AGMM_SWPK_EXACT_MUSTFAIL=1  the self-test candidate runs 0049's must-fail C++ mode -> the boot is REFUSED.
AGMM_COMPUTE_KERNEL_REL = ("ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_minimal_matmul_async/device/kernels/"
                           "compute.cpp")
SWPK_EXACT_MARKER = "SWPK_EXACT_V1"
_SWPK_EXACT_LOGGED = set()


def _agmm_compute_kernel_path():
    for root in (_tt_public_getenv("TT_METAL_RUNTIME_ROOT", ""), _tt_public_getenv("TT_METAL_HOME", ""), "/opt/tt-metal-src"):
        if root and os.path.isfile(os.path.join(root, AGMM_COMPUTE_KERNEL_REL)):
            return os.path.join(root, AGMM_COMPUTE_KERNEL_REL)
    return None


def agmm_swpk_exact(mode=None):
    """True when QWEN36_AGMM_SWPK_EXACT=1 is engaged (validated against the 0049 mode and the mounted kernel)."""
    v = _tt_public_getenv("QWEN36_AGMM_SWPK_EXACT", "").strip()
    mf = _tt_public_getenv("QWEN36_AGMM_SWPK_EXACT_MUSTFAIL", "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"QWEN36_AGMM_SWPK_EXACT={v!r}: expected unset, 0 or 1")
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_AGMM_SWPK_EXACT_MUSTFAIL={mf!r}: expected unset, 0 or 1")
    if mf == "1" and v != "1":
        raise ValueError("QWEN36_AGMM_SWPK_EXACT_MUSTFAIL=1 needs QWEN36_AGMM_SWPK_EXACT=1")
    if v != "1":
        return False
    if mode is None:
        mode = _tt_public_getenv("QWEN36_AGMM_SWIGLU_PACK", "").strip()
    if mode not in ("1", "nodefer", "0", "mustfail"):  # "0"/"mustfail": inside the self-test's ref/cand
        raise ValueError(f"QWEN36_AGMM_SWPK_EXACT=1 needs QWEN36_AGMM_SWIGLU_PACK=1 or nodefer (got {mode!r})")
    if _tt_public_getenv("QWEN36_AGMM_SUBBLOCK", "").strip():
        raise ValueError("QWEN36_AGMM_SWPK_EXACT=1 refuses QWEN36_AGMM_SUBBLOCK (the exact kernel needs subblock_h 1)")
    if "kernel" not in _SWPK_EXACT_LOGGED:
        path = _agmm_compute_kernel_path()
        if path is None:
            raise RuntimeError("QWEN36_AGMM_SWPK_EXACT=1: AGMM compute.cpp not found under TT_METAL_RUNTIME_ROOT/"
                               "TT_METAL_HOME//opt/tt-metal-src")
        with open(path, "rb") as f:
            src = f.read()
        if SWPK_EXACT_MARKER.encode() not in src:
            raise RuntimeError(f"QWEN36_AGMM_SWPK_EXACT=1: {path} lacks {SWPK_EXACT_MARKER} (mount "
                               "bench/probes/agmm_swiglu_pack/compute.exact.cpp over it)")
        _SWPK_EXACT_LOGGED.add("kernel")
        import hashlib
        logger.info(f"QWEN36_AGMM_SWPK_EXACT engaged: role=mlp bit-exact pack-thread SwiGLU (stock L1-acc last K "
                    f"block + SrcA reload + 0049 SFPU on PACK) kernel={path} sha256={hashlib.sha256(src).hexdigest()[:16]}")
    if mf == "1" and "mf" not in _SWPK_EXACT_LOGGED:
        _SWPK_EXACT_LOGGED.add("mf")
        logger.info("QWEN36_AGMM_SWPK_EXACT_MUSTFAIL engaged: role=mlp self-test candidate packs the last K block "
                    "without L1 accumulation (must be REFUSED)")
    return True


def agmm_swiglu_pack_engaged(role):
    """The engaged mode for this AGMM call (None = stock): 'mlp' (the fused SwiGLU) takes every mode, the matmul
    roles only the deferral. Banner once per role and mode."""
    v = agmm_swiglu_pack()
    if v is None or (role != "mlp" and v == "nodefer"):
        return None
    key = (role, v)
    if key not in _SWPK_LOGGED and v != "mustfail":
        _SWPK_LOGGED.add(key)
        what = _SWPK_MODES[v] if role == "mlp" else "injector write deferral"
        mf = _tt_public_getenv("QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL", "").strip() == "1" and role == "mlp"
        logger.info(f"QWEN36_AGMM_SWIGLU_PACK engaged: mode={v} role={role} ({what}; patch 0049, upstream #59325)")
        if mf:
            logger.info("QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL engaged: role=mlp self-test candidate without the "
                        "partial-sum add (must be REFUSED)")
    return v


def agmm_swiglu_pack_needs_selftest(role):
    return role not in _SWPK_ACTIVE and f"QWEN36_AGMM_SWIGLU_PACK[{role}]" not in _PREFILL_SELFTEST


def agmm_swiglu_pack_selftest(role, mode, call, x):
    """Once per role and process, on the first engaged call: `call(x_in)` (the AGMM wrapper, re-entered) with the env
    forced to 0 (ref), as engaged (cand; the C++ must-fail mode under _MUSTFAIL) and as engaged on x with its first 32
    rows zeroed (wrong). mlp under a pack mode: within AGMM_SWPK_SELFTEST_TOL; deferral only: bitwise."""
    env = "QWEN36_AGMM_SWIGLU_PACK"
    pack = role == "mlp" and mode in ("1", "nodefer")
    exact = pack and agmm_swpk_exact(mode)  # QWEN36_AGMM_SWPK_EXACT: tier 1, bitwise self-test
    mf = pack and (_tt_public_getenv("QWEN36_AGMM_SWIGLU_PACK_MUSTFAIL", "").strip() == "1"
                   or (exact and _tt_public_getenv("QWEN36_AGMM_SWPK_EXACT_MUSTFAIL", "").strip() == "1"))

    def with_env(val, xi):
        old = _tt_public_getenv(env)
        os.environ[env] = val
        try:
            return call(xi)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old

    def zeroed():
        xz = zero_first_rows(x)
        try:
            return with_env(mode, xz)
        finally:
            ttnn.deallocate(xz)

    flag = f"QWEN36_AGMM_SWIGLU_PACK[{role}]"
    desc = f"role={role} mode={mode} S={x.shape[-2]} K_local={x.shape[-1]}" + (" MUSTFAIL" if mf else "")
    _SWPK_ACTIVE.add(role)
    try:
        ref = lambda: with_env("0", x)  # noqa: E731
        cand = (lambda: with_env("mustfail", x)) if mf else (lambda: with_env(mode, x))
        if exact:
            prefill_selftest(flag, ref=ref, cand=cand, wrong=zeroed, desc=desc + " SWPK_EXACT",
                             ok_banner="QWEN36_AGMM_SWPK_EXACT[mlp] self-test exact")
        elif pack:
            prefill_selftest_tol(flag, ref=ref, cand=cand, wrong=zeroed, desc=desc, tol=AGMM_SWPK_SELFTEST_TOL,
                                 ok_banner=f"QWEN36_AGMM_SWIGLU_PACK[{role}] self-test passed")
        else:
            prefill_selftest(flag, ref=ref, cand=cand, wrong=zeroed, desc=desc,
                             ok_banner=f"QWEN36_AGMM_SWIGLU_PACK[{role}] self-test passed")
    finally:
        _SWPK_ACTIVE.discard(role)


def agmm_t11x9_nblock(n_tiles, rows):
    """N_block for a transposed 11x9 matmul-role call: per-core N = ceil(N/9) (rows). Above 12 tiles keep the nt11x8
    cap (12, QWEN36_AGMM_NT_NBLOCK); at or below it, round up to 2 only (out-projs: 80 tiles -> 9/core -> 10, not 12)."""
    per_core = max(1, math.ceil(n_tiles / rows))
    cap = int(_tt_public_getenv("QWEN36_AGMM_NT_NBLOCK", "12"))
    if per_core > cap:
        return cap
    return min(cap, ((per_core + 1) // 2) * 2)


# ---------------------------------------------------------------- QWEN36_AGMM_MBLOCK (c134c, branch agmm-grid)
# The fused AGMM gives each core M_tiles_per_core = ceil(M_tiles / M-axis cores) tile rows and walks them in M blocks
# of M_block_size (factory: M_blocks_per_core = div_up(M_tiles_per_core, M_block)); for EVERY M block it streams the
# whole K x N_per_core weight slice again (dm_in1_sender_out, per (m, n) block), and only compute shrinks for a partial
# block. At the 2048 chunk on nt11x8, 64 rows / 8 = 8 per core = 2 blocks of 4. At a 2304 chunk with the grid OFF,
# 72 / 8 = 9 per core = 4 + 4 + 1: a third weight pass for one tile row. The c134 hw probe (S=2304, standalone) has the
# 11x9 grid (8 rows/core, 2 blocks) at -17..-18% for gdn_in/mlp against -11% from the core count alone; the excess is
# the ragged block ([source validation record]). Bit-exact by construction: the K order per output
# tile does not depend on the M blocking.
#   QWEN36_AGMM_MBLOCK      M_block_size of the four matmul roles (gdn_in, attn_in, gdn_out, attn_out). Default 4.
#   QWEN36_AGMM_MLP_MBLOCK  (existing, c99) the same for the fused SwiGLU gate|up. Default 4.
# Both read per call (a wb `set` screens them). The banner names the per-core split so a rung self-identifies.
_AGMM_MBLOCK_LOGGED = set()


def agmm_m_block(role, S, grid, force_transpose, env="QWEN36_AGMM_MBLOCK"):
    """M_block_size for one prefill AGMM call: `env` (default 4) and a banner with the per-core block split when set."""
    v = _tt_public_getenv(env, "").strip()
    if not v:
        return 4
    mb = int(v)
    if mb < 1:
        raise ValueError(f"{env}={v}: must be a positive tile count")
    m_tiles = -(-int(S) // TILE_SIZE)
    m_axis = grid[0] if force_transpose else grid[1]
    per_core = -(-m_tiles // m_axis)
    key = (env, role, int(S), mb, tuple(grid))
    if key not in _AGMM_MBLOCK_LOGGED:
        _AGMM_MBLOCK_LOGGED.add(key)
        full, rem = divmod(per_core, mb)
        split = "+".join([str(mb)] * full + ([str(rem)] if rem else []))
        logger.info(f"{env} engaged: role={role} M_block={mb} S={S} grid={tuple(grid)} M_tiles/core={per_core} "
                    f"-> blocks {split} ({full + (1 if rem else 0)} weight passes)")
    return mb


# ---------------------------------------------------------------- QWEN36_AGMM_BLOCKS (f-swiglu, 2026-10-20)
# The prefill AGMM's compute kernel (device/kernels/compute.cpp:460-521) loops
#   for m_block { for n_block { for k_block { ... in1_cb.wait_front(K_block x N_block) ... } } }
# and the injector (dm_in1_sender_out, row 0 of every column) fills the in1 (weight) CB in exactly that order. A core
# with M_blocks_per_core > 1 therefore reads EVERY weight tile once per M block: with the shipped M_block=4 and 8
# M-tiles per core (nt11x8, S=2048 -> 64 tile rows over 8 rows) the MLP gate|up weight is read TWICE per core, twice
# from DRAM (~180 MB per call per card, ~11.5 GB per chunk) and twice over the column multicast. The per-RISC anatomy
# sees the supply as 68 us of in1 unpack wait + 74 us of block-start in1 restarts per call
# ([source validation record]).
# M_block=8 (one M block per core) reads the weight ONCE. M8 was never shipped: with the shipped N_block=28 the fp32
# intermediate (M_block x N_block) plus the in1 CB need 2160 KiB and the op TT_FATALs on L1
# ("L1 overflow 1913856 > 1572864 B", c111). M8 x N16 needs 1344 KiB of the 1536 KiB and fits; the shipped
# 4 x 28 x 16 already sits at 1528 KiB (99.5 % of the L1), so M8 x N16 also FREES 184 KiB.
# Numerics: TIER 1, bit-exact. The block split changes only the GROUPING of the output tiles; the order in which one
# output tile accumulates its K blocks is unchanged, so every tile gets the same sum in the same order.
#   QWEN36_AGMM_BLOCKS          comma list of <role>=MxN, role in {swiglu}; M and N in tiles.
#                               Unset / "0" = the shipped split per role (byte-identical). Read per call.
#                               Only the swiglu role is wired in this branch: the same mechanism (one weight pass per
#                               core) applies to the other four AGMM roles (qkvzab, attn qkv, gdn out, attn out:
#                               M_block 8 with their N_block unchanged keeps every CB inside the 1536 KiB L1),
#                               but a 'matmul=' item is REFUSED here rather than silently ignored.
#   QWEN36_AGMM_BLOCKS_MUSTFAIL 1: the self-test's candidate runs on a zeroed-row input -> the boot is REFUSED.
# The first engaged call per role self-tests: the shipped split (ref) vs the engaged split (cand) must be bitwise
# equal, and the engaged split on a zeroed-row input (wrong) must differ.
_AGMM_BLOCKS_ROLES = ("swiglu",)
_AGMM_BLOCKS_LOGGED = set()
_AGMM_BLOCKS_ACTIVE = set()
_AGMM_BLOCKS_FORCE_OFF = [False]


def agmm_blocks_override(role):
    """The QWEN36_AGMM_BLOCKS override for `role` as (M_block, N_block) in tiles, or None = the shipped split."""
    if _AGMM_BLOCKS_FORCE_OFF[0]:
        return None
    v = _tt_public_getenv("QWEN36_AGMM_BLOCKS", "").strip()
    if not v or v == "0":
        return None
    table = {}
    for item in (i.strip() for i in v.split(",")):
        if not item:
            continue
        r, sep, d = item.partition("=")
        r, d = r.strip(), d.strip().lower()
        if not sep or r not in _AGMM_BLOCKS_ROLES or "x" not in d:
            raise ValueError(f"QWEN36_AGMM_BLOCKS={v!r}: expected <role>=MxN items, role in {_AGMM_BLOCKS_ROLES}"
                             + ("" if r in _AGMM_BLOCKS_ROLES else
                                f" ('{r}' is not wired in this branch; only the swiglu role is honoured)"))
        mi, ni = d.split("x", 1)
        try:
            m, n = int(mi), int(ni)
        except ValueError:
            raise ValueError(f"QWEN36_AGMM_BLOCKS={v!r}: {d!r} is not MxN")
        if m < 1 or n < 1:
            raise ValueError(f"QWEN36_AGMM_BLOCKS={v!r}: M and N must be at least 1 tile")
        if role == "swiglu" and n % 2:
            raise ValueError(f"QWEN36_AGMM_BLOCKS={v!r}: the SwiGLU role needs an even N_block (gate/up pairs)")
        table[r] = (m, n)
    return table.get(role)


def agmm_blocks_mustfail():
    return _tt_public_getenv("QWEN36_AGMM_BLOCKS_MUSTFAIL", "") == "1"


def agmm_blocks_needs_selftest(role):
    return role not in _AGMM_BLOCKS_ACTIVE and f"QWEN36_AGMM_BLOCKS[{role}]" not in _PREFILL_SELFTEST


def agmm_blocks_selftest(role, call, x, m_new, n_new):
    """Once per role and process, on the first engaged call: `call(x)` with the SHIPPED block split (ref) and with the
    engaged split (cand) must be bitwise equal (the block split only groups tiles), and the engaged split on a
    zeroed-row input (wrong) must differ."""
    def ref():
        _AGMM_BLOCKS_FORCE_OFF[0] = True
        try:
            return call(x)
        finally:
            _AGMM_BLOCKS_FORCE_OFF[0] = False

    def cand():
        return call(x)

    def wrong():
        xz = zero_first_rows(x)
        try:
            return call(xz)
        finally:
            ttnn.deallocate(xz)

    _AGMM_BLOCKS_ACTIVE.add(role)
    try:
        prefill_selftest(f"QWEN36_AGMM_BLOCKS[{role}]", ref=ref,
                         cand=wrong if agmm_blocks_mustfail() else cand,
                         wrong=wrong, desc=f"role={role} S={x.shape[-2]} K_local={x.shape[-1]} "
                                           f"M_block={m_new} N_block={n_new}",
                         ok_banner=f"QWEN36_AGMM_BLOCKS[{role}] self-test exact")
    finally:
        _AGMM_BLOCKS_ACTIVE.discard(role)


def agmm_blocks_apply(role, m_block, n_block, call=None, x=None, m_tiles_per_core=None):
    """The block split for one AGMM role: the shipped (m_block, n_block), or the QWEN36_AGMM_BLOCKS override (with
    its banner and, on the first engaged call, the bitwise self-test).

    m_tiles_per_core: when the shipped M_block already covers a core's M tiles (one weight pass, e.g. S <= 1024 on
    nt11x8), the override would only split N finer and re-stream in0 once per extra N block, so it is not applied."""
    ov = agmm_blocks_override(role)
    if ov is None:
        return m_block, n_block
    m, n = ov
    if m_tiles_per_core is not None and int(m_tiles_per_core) <= int(m_block):
        key = (role, "skip", int(m_tiles_per_core))
        if key not in _AGMM_BLOCKS_LOGGED:
            _AGMM_BLOCKS_LOGGED.add(key)
            logger.info(f"QWEN36_AGMM_BLOCKS skipped: role={role} M_tiles/core={m_tiles_per_core} <= shipped "
                        f"M_block={m_block} (already one weight pass); shipped {m_block}x{n_block} serves it")
        return m_block, n_block
    key = (role, m, n)
    if key not in _AGMM_BLOCKS_LOGGED:
        _AGMM_BLOCKS_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_BLOCKS engaged: role={role} M_block={m} N_block={n} tiles (shipped "
                    f"{m_block}x{n_block})" + (" MUSTFAIL: self-test candidate on a zeroed-row input"
                                               if agmm_blocks_mustfail() else ""))
    if call is not None and agmm_blocks_needs_selftest(role):
        agmm_blocks_selftest(role, call, x, m, n)
    return m, n


# ---------------------------------------------------------------- QWEN36_AGMM_IN_L1FIT (k-gdnpre, 2026-10-21)
# The GDN in-projection (the fused [qkv|z|a|b] all-gather matmul, role gdn_in) is 28.7 ms of the 294.7 ms 32k
# prefill chunk (an internal record; [source validation record] on origin/pp/s3prof:
# qkvzab 28.74 ms, 48 calls, 598.8 us/call, 61.3 % of the LoFi peak). The device anatomy of the five AGMM roles
# ([source validation record]) puts the gdn_in call at 525 us against a 364.1 us FPU floor
# (69.3 %): of the 161 us above the floor, ~70 us is the L1 wait inside the K-block loop and ~47 us is the four
# K-block "restarts" (the first K block of every group costs ~20 us against a 9.1 us floor). The cause is the BLOCKING:
# the compute kernel walks m_block -> n_block -> k_block (device/kernels/compute.cpp:460-520), so with M_block 4 and
# 8 M tile rows per core (S=2048 on the nt11x8 grid) every core runs TWO M blocks, and dm_in1_sender_out streams the
# whole K x N_per_core weight slice AGAIN for the second one (the in1 reread factor 2.0x) while the in0 chain walks
# the K blocks once per M block as well.
#
# f-qkvz (branch free/f-qkvz, an internal record) built exactly that lever (M_block 8 for the in-projections)
# and ttsim-gated it bitwise, but the CARD screen REFUSED the boot on 10-01 with an L1 CB clash
# ([source validation record] line 8) and the coordinator's review said:
# "Reopen with a factory-derived L1 cut (N_block 8 or qkvzab output in DRAM). Record
# [source validation record], commit 87e46b3e." THIS is that reopening: the same one-M-block
# program with the factory-derived L1 cut that makes it fit.
#
# The CB sizes come from the factory (all_gather_minimal_matmul_async_program_factory.cpp:380-397):
#   in0 = M_block * K_block * 2 tiles, in1 = K_block * N_block * 2 tiles, out = M_block * N_block (single-buffered),
#   interm = M_block * N_block (single, fp32 with fp32_dest_acc_en).
# Tile sizes on Blackhole carry a 64 B per-tile face/exponent overhead: bf8 = 1088 B, bf4 = 576 B, bf16 = 2112 B,
# fp32 = 4160 B (the review of f-qkvz counted 1088/576 and that is what made its L1 arithmetic 31 KiB short).
# K_block 16, Blackhole 1427 KiB of usable L1 per core (docs/BLACKHOLE-PERF.md:52-61):
#   shipped  M4/N12: in0 128x1088 + in1 384x576 + out 48x2112 + interm 48x4160 =  639 KiB of CBs
#   M8/N12   (f-qkvz): in0 256x1088 + in1 384x576 + out 96x2112 + interm 96x4160 = 1064 KiB of CBs  -> the refused boot
#   M8/N8    (here):  in0 256x1088 + in1 256x576 + out 64x2112 + interm 64x4160 =  808 KiB of CBs  -> 256 KiB LESS
# The qkvzab output tensor is L1-resident on top (2048 x 8256 x 2 B = 33.8 MB over 88 cores = 384 KiB/core), so the
# refused M8/N12 arm needed ~1448 KiB of the 1427 KiB and M8/N8 needs ~1192 KiB: ~235 KiB of headroom. This is an
# ESTIMATE (ttsim has no L1 model); an L1 clash would refuse the boot, which is a safe failure with no speed number.
# (The other cut the review named - the qkvzab output in DRAM - would add ~68 MB of DRAM traffic per GDN layer, i.e.
# ~3.3 GB per 32k chunk, ~+7 ms/chunk of DRAM time: it would eat the whole win, so it is NOT used.)
# The out-projection roles keep M_block 4 (branch free/agmm2 owns their own knob).
#
# Numerics: TIER 1, bit-exact by construction -- the K accumulation order per output tile does not depend on the M or
# the N blocking (the same argument as QWEN36_AGMM_MBLOCK, c134c). The in-boot self-test compares the candidate
# program against the shipped one bitwise on every chip and refuses the boot if they differ OR if the wrong-input
# rung does not differ.
#   QWEN36_AGMM_IN_L1FIT = 1           engage (prefill AGMM in-projections only: gdn_in, attn_in)
#   QWEN36_AGMM_IN_L1FIT = mustfail    ttsim only: the self-test candidate runs on a wrong input -> the boot REFUSES
_AGMM_IN_L1FIT_LOGGED = set()
_AGMM_IN_L1FIT_ACTIVE = set()
_AGMM_IN_L1FIT_SPLIT = set()


def agmm_in_l1fit():
    """'1' | 'mustfail' when QWEN36_AGMM_IN_L1FIT is engaged, else None. Read on every call (the model's warmup
    self-test runs the shipped and the candidate program in ONE process, so the env has to be read per call)."""
    v = _tt_public_getenv("QWEN36_AGMM_IN_L1FIT", "").strip()
    if v in ("", "0"):
        return None
    if v not in ("1", "mustfail"):
        raise ValueError(f"QWEN36_AGMM_IN_L1FIT={v!r}: expected unset, 0, 1 or mustfail")
    if v not in _AGMM_IN_L1FIT_LOGGED:
        _AGMM_IN_L1FIT_LOGGED.add(v)
        logger.info(f"QWEN36_AGMM_IN_L1FIT engaged: mode={v} prefill AGMM IN-projection roles (gdn_in, attn_in) take M_block=8 and N_block=8: one M block per core (the weight slice and the in0 chain are streamed once per call instead of twice) with the factory-derived L1 cut (N_block 12 -> 8) that makes the CBs fit next to the L1 output buffer")
        if v == "mustfail":
            logger.info("QWEN36_AGMM_IN_L1FIT_MUSTFAIL engaged: the self-test candidate is called on a wrong input (must be REFUSED)")
    return v


def agmm_in_l1fit_roles():
    """The roles QWEN36_AGMM_IN_L1FIT applies to: the in-projections (gdn_in, attn_in)."""
    return ("gdn_in", "attn_in")


def agmm_in_l1fit_needs_selftest(role):
    """True until this role's in-boot self-test has run in this process (the GBS/MB8 pattern)."""
    return role not in _AGMM_IN_L1FIT_ACTIVE and f"QWEN36_AGMM_IN_L1FIT[{role}]" not in _PREFILL_SELFTEST


def agmm_in_l1fit_selftest(role, mode, call, x):
    """Once per role and process, on the first engaged call: `call(x)` with the env forced to 0 (ref: the shipped
    M_block 4 / N_block 12 program), as engaged (cand) and as engaged on x with its first 32 rows zeroed (wrong).
    Bitwise, on every chip. `mode` 'mustfail' is the ttsim-only wrong-result rung: the candidate is called on the
    ZEROED input while the reference runs on x, so the two must differ and the self-test REFUSES the boot. This lever
    has no device-side wrong-result mode (it only changes the blocking), so the wrong result is injected on the host.
    """
    env = "QWEN36_AGMM_IN_L1FIT"

    def with_env(val, xi):
        old = _tt_public_getenv(env)
        os.environ[env] = val
        try:
            return call(xi)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old

    def zeroed():
        xz = zero_first_rows(x)
        try:
            return with_env("1", xz)
        finally:
            ttnn.deallocate(xz)

    def cand():
        if mode == "mustfail":
            return zeroed()
        return with_env("1", x)

    _AGMM_IN_L1FIT_ACTIVE.add(role)
    try:
        prefill_selftest(f"QWEN36_AGMM_IN_L1FIT[{role}]", ref=lambda: with_env("0", x), cand=cand, wrong=zeroed,
                         desc=f"role={role} mode={mode} S={x.shape[-2]} K_local={x.shape[-1]}",
                         ok_banner=f"QWEN36_AGMM_IN_L1FIT[{role}] self-test exact")
    finally:
        _AGMM_IN_L1FIT_ACTIVE.discard(role)


# ---------------------------------------------------------------- QWEN36_AGMM_GBS (pp/agmm, 2026-10-19; patch 0057)
# The chain-end core of every nt11x8 row forwards each local in0 K block to the peer as 16 four-tile scatter packets
# (4352 B: the fabric payload cap, 4 NoC addresses per scatter packet) at 12.9 us per K block, inside its receive loop,
# so the row's in0 chain runs at the forward pace for the forwarded half ([source validation record]).
# With the gather buffer DRAM height-sharded (full-width shards) a K-block row is contiguous on both chips: the chain
# end forwards it as contiguous unicast packets of payload/tile tiles (8 with QWEN36_FABRIC_PAYLOAD=8704) and the
# injector reads a gathered row with one NoC read. Dataflow only: bit-identical output (tier 1, bitwise self-test).
#   QWEN36_AGMM_GBS=1         engage (needs the 0057 _ttnncpp.so: the op refuses an unsharded buffer only when built)
#   QWEN36_AGMM_GBS=mustfail  ttsim only: the chain end drops the last tile of every forwarded row (must be REFUSED)
_GBS_LOGGED = set()
_GBS_ACTIVE = set()


def agmm_gbs():
    """'1' | 'mustfail' when QWEN36_AGMM_GBS is engaged, else None. Banner once per mode."""
    v = _tt_public_getenv("QWEN36_AGMM_GBS", "").strip()
    if v in ("", "0"):
        return None
    if v not in ("1", "mustfail"):
        raise ValueError(f"QWEN36_AGMM_GBS={v!r}: expected unset, 0, 1 or mustfail")
    if v not in _GBS_LOGGED:
        _GBS_LOGGED.add(v)
        logger.info(f"QWEN36_AGMM_GBS engaged: mode={v} prefill AGMM gather buffer DRAM height-sharded (full-width "
                    f"shards): one NoC read per gathered K-block row, contiguous fabric packets (patch 0057)")
        if v == "mustfail":
            logger.info("QWEN36_AGMM_GBS_MUSTFAIL engaged: self-test candidate drops the last tile of every forwarded "
                        "row (must be REFUSED)")
    return v


def agmm_gbs_memcfg(S, K_total):
    """DRAM HEIGHT_SHARDED over the 8 banks, each shard whole tile rows x the full K (what 0057 validates)."""
    shard_h = -(-S // (DRAM_CORES * TILE_SIZE)) * TILE_SIZE
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(DRAM_GRID, [shard_h, K_total], ttnn.ShardOrientation.ROW_MAJOR),
    )


def agmm_gbs_needs_selftest(role):
    return role not in _GBS_ACTIVE and f"QWEN36_AGMM_GBS[{role}]" not in _PREFILL_SELFTEST


def agmm_gbs_selftest(role, mode, call, x):
    """Once per role and process, on the first engaged call: `call(x_in)` (the AGMM wrapper, re-entered) with the env
    forced to 0 (ref: interleaved buffer, stock kernels), as engaged (cand; `mustfail` drops a tile per forwarded row)
    and as engaged on x with its first 32 rows zeroed (wrong). Bitwise."""
    env = "QWEN36_AGMM_GBS"

    def with_env(val, xi):
        old = _tt_public_getenv(env)
        os.environ[env] = val
        try:
            return call(xi)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old

    def zeroed():
        xz = zero_first_rows(x)
        try:
            return with_env(mode, xz)
        finally:
            ttnn.deallocate(xz)

    _GBS_ACTIVE.add(role)
    try:
        prefill_selftest(f"QWEN36_AGMM_GBS[{role}]", ref=lambda: with_env("0", x), cand=lambda: with_env(mode, x),
                         wrong=zeroed, desc=f"role={role} mode={mode} S={x.shape[-2]} K_local={x.shape[-1]}",
                         ok_banner=f"QWEN36_AGMM_GBS[{role}] self-test exact")
    finally:
        _GBS_ACTIVE.discard(role)


# ---------------------------------------------------------------- QWEN36_AGMM_FWO (pp/agmm-fwo, 2026-10-20; patch 0063)
# In the nt11x8 AGMM one chain core per row forwards each LOCAL in0 K block to the peer inside its receive loop (12.9 us
# per block vs 6.7-10 us of compute), so during the local half (consumed first on Linear) the whole row chain is held to
# the forward pace, while the remote half never waits (I0-SEMW ~0.03 us; [source validation record]).
# With QWEN36_AGMM_FWO the op adds one forwarder core per row on the free row 8 (beside the row's mux on row 9): it reads
# the local K blocks from the local input and forwards them, the chain cores no longer forward. Same bytes, same tiles,
# same K order: bit-identical output (tier 1, bitwise self-test). Needs the untransposed nt11x8 grid with row 8 free
# (no QWEN36_AGMM_GRID9 / T11X9 on that call); anything else is refused here before the op would TT_FATAL.
#   QWEN36_AGMM_FWO=1         engage (needs the 0063 _ttnncpp.so + dm_in0_forwarder.cpp)
#   QWEN36_AGMM_FWO=mustfail  ttsim only: the forwarder zeroes the last tile of every forwarded block (must be REFUSED)
_FWO_LOGGED = set()
_FWO_ACTIVE = set()


def agmm_fwo():
    """'1' | 'mustfail' when QWEN36_AGMM_FWO is engaged, else None. Read per call (the self-test flips it)."""
    v = _tt_public_getenv("QWEN36_AGMM_FWO", "").strip()
    if v in ("", "0"):
        return None
    if v not in ("1", "mustfail"):
        raise ValueError(f"QWEN36_AGMM_FWO={v!r}: expected unset, 0, 1 or mustfail")
    if v == "mustfail" and "mustfail" not in _FWO_LOGGED:
        _FWO_LOGGED.add("mustfail")
        logger.info("QWEN36_AGMM_FWO_MUSTFAIL engaged: the forwarder zeroes the last tile of every forwarded block "
                    "(the self-test must REFUSE)")
    return v


def agmm_fwo_check(role, grid, force_transpose):
    """Refuse (raise) an FWO call whose grid leaves no free forwarder row; log the per-role banner once."""
    v = agmm_fwo()
    if v is None:
        return None
    if force_transpose or grid[1] > 8:
        raise ValueError(f"QWEN36_AGMM_FWO={v}: role={role} needs the untransposed nt11x8 grid with row 8 free for the "
                         f"forwarders (got grid={tuple(grid)} force_transpose={force_transpose}; QWEN36_AGMM_LAYOUT="
                         f"nt11x8 without QWEN36_AGMM_GRID9 / QWEN36_AGMM_T11X9 on this role)")
    if role not in _FWO_LOGGED:
        _FWO_LOGGED.add(role)
        logger.info(f"QWEN36_AGMM_FWO engaged: mode={v} role={role} grid={tuple(grid)}: in0 peer forward on one "
                    f"forwarder core per row (row 8), chain cores no longer forward (patch 0063)")
    return v


def agmm_fwo_needs_selftest(role):
    return role not in _FWO_ACTIVE and f"QWEN36_AGMM_FWO[{role}]" not in _PREFILL_SELFTEST


def agmm_fwo_selftest(role, mode, call, x):
    """Once per role and process, on the first engaged call: `call(x_in)` (the AGMM wrapper, re-entered) with the env
    forced to 0 (ref: the stock forward on the chain), as engaged (cand; `mustfail` zeroes a tile per forwarded block)
    and as engaged on x with its first 32 rows zeroed (wrong). Bitwise, on both devices (each device's output needs the
    peer's forwarded half)."""
    env = "QWEN36_AGMM_FWO"

    def with_env(val, xi):
        old = _tt_public_getenv(env)
        os.environ[env] = val
        try:
            return call(xi)
        finally:
            if old is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = old

    def zeroed():
        xz = zero_first_rows(x)
        try:
            return with_env(mode, xz)
        finally:
            ttnn.deallocate(xz)

    _FWO_ACTIVE.add(role)
    try:
        prefill_selftest(f"QWEN36_AGMM_FWO[{role}]", ref=lambda: with_env("0", x), cand=lambda: with_env(mode, x),
                         wrong=zeroed, desc=f"role={role} mode={mode} gbs={agmm_gbs() or 0} S={x.shape[-2]} "
                                            f"K_local={x.shape[-1]}",
                         ok_banner=f"QWEN36_AGMM_FWO[{role}] self-test exact")
    finally:
        _FWO_ACTIVE.discard(role)


def agmm_gather_buffer(tt_ccl, x, cluster_axis=1):
    """Persistent gather buffer for all_gather_minimal_matmul_async on activation x [.,S,K/tp].

    TODO(#57458): once all_gather_minimal_matmul_async honours barrier_semaphore (an in-kernel receiver-ready
    handshake), drop this buffer and pass barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
    instead; test_gdn_out_agmm_deterministic_under_device_skew must keep passing."""
    cache = tt_ccl.__dict__.setdefault("_agmm_gather_buffers", {})
    mesh = tt_ccl.mesh_device
    S, K_local = x.shape[-2], x.shape[-1]
    gbs = agmm_gbs()  # QWEN36_AGMM_GBS (0057): DRAM height-sharded, full-width shards; read per call (self-test)
    key = (S, K_local, cluster_axis, x.dtype, bool(gbs))
    if key not in cache:
        K_total = K_local * mesh.shape[cluster_axis]
        shape = ttnn.Shape([1, 1, S, K_total])
        memcfg = agmm_gbs_memcfg(S, K_total) if gbs else ttnn.DRAM_MEMORY_CONFIG
        pair = [
            ttnn.allocate_tensor_on_device(shape, x.dtype, ttnn.TILE_LAYOUT, mesh, memcfg)
            for _ in range(2)
        ]
        cache[key] = [pair, 0]
    pair, idx = cache[key]
    cache[key][1] = idx ^ 1
    return pair[idx]


# ---------------------------------------------------------------- QWEN36_AGMM_CKC / QWEN36_AGMM_BUFS (c88)
# Prefill AGMM (all_gather_minimal_matmul_async) is 32% of chip-0 prefill busy time at 30k on c86 (tt_fabric_mux
# 1846 of 5726 ms; [source validation record]). Every call site passes a LoFi,
# fp32-dest, packer_l1_acc=False config. Both knobs are read per call, so the workbench can screen them.
#   QWEN36_AGMM_CKC: comma list of l1acc (packer L1 accumulation across K blocks) and dest16 (bf16 dest:
#     tier 2, K=5120 accumulates in bf16 halves). Empty = the call site's config, unchanged.
#   QWEN36_AGMM_BUFS: num_buffers_per_channel for the gather (default 8).
_AGMM_CKC_LOGGED = set()


def agmm_compute_cfg(cfg):
    v = _tt_public_getenv("QWEN36_AGMM_CKC", "")
    if not v:
        return cfg
    opts = set(v.split(","))
    bad = opts - {"l1acc", "dest16"}
    if bad:
        raise ValueError(f"QWEN36_AGMM_CKC={v!r}: expected a comma list of l1acc,dest16")
    if v not in _AGMM_CKC_LOGGED:
        _AGMM_CKC_LOGGED.add(v)
        logger.info(f"QWEN36_AGMM_CKC engaged: prefill AGMM LoFi packer_l1_acc={'l1acc' in opts} "
                    f"fp32_dest_acc_en={'dest16' not in opts}")
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en="dest16" not in opts,
        packer_l1_acc="l1acc" in opts,
    )


def attn_outproj_colpar():
    """QWEN36_ATTN_OUTPROJ_COLPAR=1 (c90): attention prefill out-proj column-parallel. Adds wo.colpar (weight-affecting)."""
    return _tt_public_getenv("QWEN36_ATTN_OUTPROJ_COLPAR", "0") == "1"

_PREFILL_SUBBLOCK_LOGGED = set()


def prefill_subblock_override():
    """QWEN36_PREFILL_SUBBLOCK=WxH (Fable 10k): force the 2D prefill matmul out_subblock to W
    columns x H rows, clamped to reg-file legal (H*W<=8) and per_core divisibility (falls back to
    the picker with a banner saying why). Wide subblocks reuse the bf16 in0 (reuse_a = ct>=rt in
    llk_unpack_AB_matmul.h) instead of re-unpacking it per column. Bit-exact (same math, new order).
    MUSTFAIL: QWEN36_PREFILL_SUBBLOCK_MUSTFAIL=1 forces 7x7 (illegal) -> boot REFUSED."""
    v = _tt_public_getenv("QWEN36_PREFILL_SUBBLOCK", "").strip()
    mf = _tt_public_getenv("QWEN36_PREFILL_SUBBLOCK_MUSTFAIL", "") == "1"
    if mf:
        v = "7x7"
    if not v:
        return None
    try:
        w, h = v.lower().split("x")
        w, h = int(w), int(h)
    except ValueError:
        raise ValueError(f"QWEN36_PREFILL_SUBBLOCK={v!r}: expected WxH, e.g. 8x1")
    if (w, h) not in _PREFILL_SUBBLOCK_LOGGED:
        _PREFILL_SUBBLOCK_LOGGED.add((w, h))
        logger.info(f"QWEN36_PREFILL_SUBBLOCK engaged: forcing subblock {w}x{h}")
    return (w, h)



_SDPA_PF_LOGGED = set()


def sdpa_prefill_knobs(default_cfg, streaming):
    """c95 prefill SDPA compute knobs, read per call (wb set screens them without a reload):
    QWEN36_SDPA_PREFILL_LOFI=1        HiFi2 -> LoFi, streaming kernel only (fp32_dest_acc_en stays off: the
                                      LoFi + fp32 acc silent-corruption trap is on the legacy kernel, AGENTS.md)
    QWEN36_SDPA_PREFILL_EXP_APPROX=1  SDPAProgramConfig exp_approx_mode=True
    Returns (compute_kernel_config, exp_approx_mode). Numerics move (tier 2)."""
    lofi = _tt_public_getenv("QWEN36_SDPA_PREFILL_LOFI", "0") == "1"
    exp = _tt_public_getenv("QWEN36_SDPA_PREFILL_EXP_APPROX", "0") == "1"
    cfg = default_cfg
    if lofi:
        if not streaming:
            if "lofi-refused" not in _SDPA_PF_LOGGED:
                _SDPA_PF_LOGGED.add("lofi-refused")
                logger.warning("QWEN36_SDPA_PREFILL_LOFI NOT engaged: it needs QWEN36_SDPA_STREAMING=1 (legacy kernel trap)")
        else:
            cfg = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=True
            )
            if "lofi" not in _SDPA_PF_LOGGED:
                _SDPA_PF_LOGGED.add("lofi")
                logger.info("QWEN36_SDPA_PREFILL_LOFI engaged: chunked prefill SDPA at LoFi (streaming kernel, no fp32 acc)")
    if exp and "exp" not in _SDPA_PF_LOGGED:
        _SDPA_PF_LOGGED.add("exp")
        logger.info("QWEN36_SDPA_PREFILL_EXP_APPROX engaged: chunked prefill SDPA exp_approx_mode=True")
    return cfg, exp



_SDPA_DEC_LOGGED = set()


def sdpa_decode_knobs(grid):
    """c102 decode SDPA knobs, read per call (wb set re-captures the decode trace, so it screens them in one process):
    QWEN36_SDPA_DEC_FID=hifi2|lofi   math fidelity; the op default is HiFi2, math_approx on, fp32 dest off,
                                     packer_l1_acc off (ttnn sdpa_decode.cpp). fp32 dest stays off, so the LoFi +
                                     fp32-acc SDPA trap (AGENTS.md) does not apply.
    QWEN36_SDPA_DEC_L1ACC=1          packer_l1_acc on.
    QWEN36_SDPA_DEC_KCHUNK=N         SDPAProgramConfig k_chunk_size (power of 2, multiple of 32); 0/unset = the op picks.
    QWEN36_SDPA_DEC_EXP_APPROX=1     exp_approx_mode.
    QWEN36_SDPA_DEC_CORES_PER_HEAD=N SDPAProgramConfig max_cores_per_head_batch (1..64); unset = 16, the nanobind
                                     default. At B=8 on 11x10 the factory caps it at floor(110/8/2)=6 (96 cores); at
                                     B=1 16 is the binding cap (32 of 110 cores). Changes the k-chunk split and the
                                     reduction tree, so numerics may move (tier 2). N=1 at B=8 is the must-fail rung.
    Returns (program_config, extra call kwargs). Unset = exactly the shipped config (no compute_kernel_config passed)."""
    fid = _tt_public_getenv("QWEN36_SDPA_DEC_FID", "").lower()
    l1acc = _tt_public_getenv("QWEN36_SDPA_DEC_L1ACC", "0") == "1"
    kc = int(_tt_public_getenv("QWEN36_SDPA_DEC_KCHUNK", "0") or 0)
    exp = _tt_public_getenv("QWEN36_SDPA_DEC_EXP_APPROX", "0") == "1"
    cph = int(_tt_public_getenv("QWEN36_SDPA_DEC_CORES_PER_HEAD", "0") or 0)
    if cph and not 1 <= cph <= 64:
        raise ValueError(f"QWEN36_SDPA_DEC_CORES_PER_HEAD={cph}: expected 1..64 (MAX_TREE_REDUCTION_ROUNDS=6)")
    if fid not in ("", "hifi2", "lofi"):
        raise ValueError(f"QWEN36_SDPA_DEC_FID={fid!r}: expected hifi2 or lofi")
    if kc and (kc & (kc - 1) or kc % 32):
        raise ValueError(f"QWEN36_SDPA_DEC_KCHUNK={kc}: must be a power of 2 and a multiple of 32")
    cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(grid.x, grid.y), exp_approx_mode=exp,
                                 q_chunk_size=0, k_chunk_size=kc)
    if cph:
        cfg.max_cores_per_head_batch = cph
    kw = {}
    if fid or l1acc:
        kw["compute_kernel_config"] = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi if fid == "lofi" else ttnn.MathFidelity.HiFi2,
            math_approx_mode=True, fp32_dest_acc_en=False, packer_l1_acc=l1acc)
    key = (fid, l1acc, kc, exp, cph)
    if any(key) and key not in _SDPA_DEC_LOGGED:
        _SDPA_DEC_LOGGED.add(key)
        logger.info(f"QWEN36_SDPA_DEC knobs engaged: fid={fid or 'hifi2(default)'} packer_l1_acc={l1acc} "
                    f"k_chunk={kc or 'auto'} exp_approx={exp} cores_per_head_cap={cph or 16}")
    return cfg, kw

_KDA_FID_LOGGED = set()

# ---------------------------------------------------------------- QWEN36_GDN_CONV_KDA2 (patches/tt-metal/0035)
# The prefill kda conv (qkv_causal_conv1d_silu, 24.3 ms per 2048-token chunk on c127) reads every activation row 4x
# and a channel block's 64 tap tiles once per item: ~168 MB of reads for a 21 MB output per call. Patch 0035 (upstream
# #58493's work order and tap reuse, without its own-core NoC loopback copies) reads each row once and each block's
# taps once per core. The C++ op reads the env on EVERY call (the mode is part of the program hash), so the warmup
# self-test below can run the stock and the read-once programs back to back in this process:
#   QWEN36_GDN_CONV_KDA2 = unset/0 (stock) | 1 (read-once, stock silu_tile) | fast (read-once, silu_tile<false>: the
#     arithmetic of the c98 compute overlay bench/probes/kda_conv_fid/compute.fast.cpp, i.e. the served arms)
#   QWEN36_GDN_CONV_KDA2_MUSTFAIL=1: the read-once reader stages every row one row late -> the self-test REFUSES.
# Needs QWEN36_GDN_CONV_KDA=1 and a _ttnncpp.so with 0035 (bench/overlays/pk3r+0033+0034+0035.json): without it the
# env is ignored by the op, the C++ banner "QWEN36_GDN_CONV_KDA2 engaged mode=" never appears and the arm's identity
# check refuses the server.
_KDA2_STATE = {"logged": False, "selftest": None}


def gdn_conv_kda2_mode():
    """The validated QWEN36_GDN_CONV_KDA2 value ('1' | 'fast'), or None when off."""
    v = _tt_public_getenv("QWEN36_GDN_CONV_KDA2", "").strip()
    mf = _tt_public_getenv("QWEN36_GDN_CONV_KDA2_MUSTFAIL", "").strip()
    if v not in ("", "0", "1", "fast"):
        raise ValueError(f"QWEN36_GDN_CONV_KDA2={v!r}: expected unset, 0, 1 or fast")
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_GDN_CONV_KDA2_MUSTFAIL={mf!r}: expected unset, 0 or 1")
    on = v in ("1", "fast")
    if mf == "1" and not on:
        raise ValueError("QWEN36_GDN_CONV_KDA2_MUSTFAIL=1 needs QWEN36_GDN_CONV_KDA2=1|fast")
    if on and not _KDA2_STATE["logged"]:
        _KDA2_STATE["logged"] = True
        logger.info(f"QWEN36_GDN_CONV_KDA2 engaged: mode={v} (kda conv reads once, patch 0035; warmup self-test "
                    f"bitwise vs the stock program)" + (" MUSTFAIL: rows staged one late" if mf == "1" else ""))
    return v if on else None


def _kda2_readback(ts):
    """Per-device host copies of a list of mesh tensors (bitwise compare, no composer arithmetic)."""
    return [[ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)] for t in ts]


def kda2_selftest(call, cases, desc):
    """QWEN36_GDN_CONV_KDA2 warmup self-test, fail closed, once per process. `call(case)` runs the op on the live
    inputs of one case and returns its (q, k, v); for each case it runs with QWEN36_GDN_CONV_KDA2 forced to 0 (the
    stock program) and with the arm's value, and compares every output on every device bitwise. Runs eagerly on the
    first conv call (the chunk-prefill warmup pass, before begin_trace_capture). Raises -> the boot is refused."""
    if _KDA2_STATE["selftest"] is not None:
        return
    mode = _tt_public_getenv("QWEN36_GDN_CONV_KDA2", "")
    results = []
    try:
        for name in cases:
            os.environ["QWEN36_GDN_CONV_KDA2"] = "0"
            try:
                ref = call(name)
            finally:
                os.environ["QWEN36_GDN_CONV_KDA2"] = mode
            cand = call(name)
            r, c = _kda2_readback(ref), _kda2_readback(cand)
            for t in list(ref) + list(cand):
                ttnn.deallocate(t)
            diff = 0
            for rt, ct in zip(r, c):
                for rd, cd in zip(rt, ct):
                    diff += int((rd.view(torch.int16) != cd.view(torch.int16)).sum()) if rd.shape == cd.shape else rd.numel()
            results.append((name, diff))
    finally:
        os.environ["QWEN36_GDN_CONV_KDA2"] = mode
    bad = [(n, d) for n, d in results if d]
    if bad:
        _KDA2_STATE["selftest"] = False
        raise RuntimeError(f"QWEN36_GDN_CONV_KDA2 self-test REFUSED ({desc}): differing output elements per case "
                           f"{bad} vs the stock program")
    _KDA2_STATE["selftest"] = True
    logger.info(f"QWEN36_GDN_CONV_KDA2 self-test exact ({desc}): q/k/v bitwise equal to the stock program on every "
                f"device, cases {[n for n, _ in results]}")


def kda_conv_cfg(default_cfg):
    """c98 QWEN36_GDN_CONV_KDA_FID=hifi4|hifi3|hifi2|lofi, read per call (wb set screens it without a reload): the
    math fidelity of the prefill kda conv1d+silu (qkv_causal_conv1d_silu). Its FPU work is mul_tiles_bcast_rows of a
    bf16 activation by bf16 taps plus dest-reuse adds; fp32 dest and no approx mode stay (the op refuses approx).
    Default hifi4 = the shipped config. Numerics may move (tier 2)."""
    v = _tt_public_getenv("QWEN36_GDN_CONV_KDA_FID", "hifi4").lower()
    fids = {"hifi4": ttnn.MathFidelity.HiFi4, "hifi3": ttnn.MathFidelity.HiFi3,
            "hifi2": ttnn.MathFidelity.HiFi2, "lofi": ttnn.MathFidelity.LoFi}
    if v not in fids:
        raise ValueError(f"QWEN36_GDN_CONV_KDA_FID={v!r}: expected one of {sorted(fids)}")
    if v == "hifi4":
        return default_cfg
    if v not in _KDA_FID_LOGGED:
        _KDA_FID_LOGGED.add(v)
        logger.info(f"QWEN36_GDN_CONV_KDA_FID engaged: prefill kda conv1d+silu at {v} (fp32 dest, no approx)")
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fids[v], math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )


_AGMM_SUB_LOGGED = set()


def agmm_subblock(role, m_block, n_block, default_w):
    """c99 QWEN36_AGMM_SUBBLOCK=HxW, read per call (wb set screens it): the prefill AGMM output subblock for both roles
    (matmul, swiglu). Default (unset) = subblock 1 x default_w, the shipped config. The fused AGMM runs at ~55% of its
    cores' LoFi peak vs ~71% for the plain w2 bmm (advisor est); a taller subblock reuses each in1 tile H times.
    H*W is the dest tile count: 4 at fp32 dest (half sync), 8 with QWEN36_AGMM_CKC=dest16. Refuses shapes that do not
    divide the block. Bit-exact at fp32 dest (same accumulation order per tile); numerics move with dest16."""
    v = _tt_public_getenv("QWEN36_AGMM_SUBBLOCK", "")
    if not v:
        return 1, default_w
    h, w = (int(t) for t in v.lower().split("x"))
    if m_block % h or n_block % w:
        raise ValueError(f"QWEN36_AGMM_SUBBLOCK={v}: must divide role={role} M_block={m_block} N_block={n_block}")
    if (v, role) not in _AGMM_SUB_LOGGED:
        _AGMM_SUB_LOGGED.add((v, role))
        logger.info(f"QWEN36_AGMM_SUBBLOCK engaged: role={role} subblock={h}x{w} M_block={m_block} N_block={n_block}")
    return h, w


def agmm_bufs():
    n = int(_tt_public_getenv("QWEN36_AGMM_BUFS", "8"))
    if n != 8 and n not in _AGMM_CKC_LOGGED:
        _AGMM_CKC_LOGGED.add(n)
        logger.info(f"QWEN36_AGMM_BUFS engaged: prefill AGMM num_buffers_per_channel={n}")
    return n


def agmm_persistent_default(tt_ccl, x4, role):
    """QWEN36_AGMM_PERSISTENT=1 (upstream tt-metal#56119 flag of the same name; issue #57458): give every
    prefill AGMM a persistent gather buffer. all_gather_minimal_matmul_async needs EITHER a persistent
    output buffer OR a barrier semaphore, and implements no barrier: without one, a chip that runs ahead
    writes its next call's shard into a peer's gather-buffer address before the peer allocated it
    (upstream f7736cf5d6 names this as the root cause of non-deterministic logits). Today only the opt-in
    column-parallel out-proj passes one (gdn/tp.py); the attention/GDN in-projections and the MLP gate|up
    do not. Correctness hardening first; any speed change is a side effect. Buffers are allocated on the
    first (warmup, pre-capture) call per (S, K, dtype) and live for the process: at S=2048, K=5120 bf16
    that is 2 x 20 MiB per key per device. Returns the buffer or None when off."""
    if _tt_public_getenv("QWEN36_AGMM_PERSISTENT") != "1":
        return None
    if role not in _LOGGED_AGMM_PERSIST:
        _LOGGED_AGMM_PERSIST.add(role)
        logger.info(f"QWEN36_AGMM_PERSISTENT engaged: role={role} prefill AGMM uses a persistent gather buffer")
    return agmm_gather_buffer(tt_ccl, x4)


_LOGGED_AGMM_PERSIST = set()


# ---------------------------------------------------------------- QWEN36_AGMM_UNFUSED (c103)
# The fused prefill AGMM is 1160 of 2682 ms chip-0 busy at 16k (43%) at ~55% of its 88 cores' LoFi peak, against
# ~71% for the plain w2 bmm ([source validation record], rung 'stock prof');
# every AGMM config knob was inert. Hypothesis: the gather is small (~21 MB/chunk at S=2048, K=5120 bf16) and a
# separate all_gather_async + a full-grid matmul beats the fused op. All knobs are read per call (wb set screens them):
#   QWEN36_AGMM_UNFUSED       comma list of roles {matmul, swiglu}, or all. Unset = the fused op, byte-identical.
#   QWEN36_AGMM_UNFUSED_GRID  XxY override of the matmul grid (default: the w2 recipe for bmm, the full compute grid
#                             for minimal_matmul; no row is reserved for fabric muxes: the gather is its own op).
#   QWEN36_AGMM_UNFUSED_MM    auto (default) | minimal. matmul role, auto: the 2D-mcast bmm with the prefill w2 recipe
#                             (create_prefill_mlp_matmul_program_config, max_cols = device width, as mlp.py's down-proj)
#                             when the weight is DRAM-interleaved; minimal_matmul when it is DRAM-sharded, because the
#                             2D-mcast kernel misreads a width-sharded in1 with no error (n6 probe) and the default MR
#                             roles re-lay the GDN qkvzab / attention qkv in-proj weights out in place. minimal: always
#                             minimal_matmul (the AGMM's own compute kernel, minus the gather).
#   QWEN36_AGMM_UNFUSED_AG    links,chunks_per_sync,workers_per_link,buffers_per_channel of the gather (default 2,10,2,8:
#                             2 links as the AGMM, tt_all_gather's chunks/workers, the AGMM's 8 buffers).
# swiglu role: minimal_matmul(fuse_swiglu=True) on the same packed w_gate_up (the fused AGMM's epilogue, same numerics
# class; w1/w3 are re-laid out DRAM-sharded by the MR w13 role, so a bmm gate/up pair is not an option). Numerics move
# by K accumulation order only (fused ring order vs 0..K); gathered x and every full-width tensor stay in DRAM.
_AGMM_UNFUSED_ROLES = ("matmul", "swiglu")
_AGMM_UNFUSED_LOGGED = set()


def agmm_unfused(role):
    v = _tt_public_getenv("QWEN36_AGMM_UNFUSED", "").strip()
    if not v or v == "0":
        return False
    roles = set(_AGMM_UNFUSED_ROLES) if v in ("1", "all") else {r.strip() for r in v.split(",") if r.strip()}
    bad = roles - set(_AGMM_UNFUSED_ROLES)
    if bad:
        raise ValueError(f"QWEN36_AGMM_UNFUSED={v!r}: unknown roles {sorted(bad)} (expected {_AGMM_UNFUSED_ROLES} or all)")
    return role in roles


# ---------------------------------------------------------------- QWEN36_AGMM_IN0_DTYPE (c106)
# The fused AGMM is 43% of 16k prefill busy and every config knob is inert (LEVERS); the gather and the in0 CB carry
# bf16 activations against bf4 weights. FS8 held its upstream rate only with low-precision activations.
#   QWEN36_AGMM_IN0_DTYPE  comma list of <role>=<dtype>, role in {matmul, swiglu, all}, dtype in {bf8, bf4}
#                          (later entries win). Unset / "0" = the shipped bf16 activation, byte-identical.
# Engaged: x is typecast (TILE, same memory config) to bfloat8_b / bfloat4_b before all_gather_minimal_matmul_async;
# the op validates BFLOAT16/BFLOAT8_B/BFLOAT4_B/FLOAT32 act dtypes. Output stays bf16. The gather buffer MUST match the
# input dtype: the op sizes the local-slice/fabric pages from the input (in3_tile_size) and the in0 CB/gathered reads
# from the gather buffer (in0_tile_size), with no check between them, and an op-allocated gather buffer takes the
# OUTPUT dtype (bf16). So an engaged call always passes a persistent gather buffer allocated in the in0 dtype
# (agmm_gather_buffer, keyed by dtype). Numerics: activation quantized to bfp8/bfp4 (shared exponent per 16), not
# bit-exact; needs the served equiv/gsm8k gate.
_AGMM_IN0_ROLES = ("matmul", "swiglu")
_AGMM_IN0_DTYPES = {"bf8": ttnn.bfloat8_b, "bf4": ttnn.bfloat4_b}
_AGMM_IN0_LOGGED = set()


def agmm_in0_dtype(role):
    """-> 'bf8' | 'bf4' | None for this AGMM role; read per call (wb set screens it)."""
    v = _tt_public_getenv("QWEN36_AGMM_IN0_DTYPE", "").strip()
    if v in ("", "0"):
        return None
    table = {}
    for item in (i.strip() for i in v.split(",")):
        if not item:
            continue
        r, sep, d = (s.strip() for s in item.partition("="))
        if not sep or d not in _AGMM_IN0_DTYPES or r not in (*_AGMM_IN0_ROLES, "all"):
            raise ValueError(f"QWEN36_AGMM_IN0_DTYPE={v!r}: expected <role>=<dtype> items, role in "
                             f"{_AGMM_IN0_ROLES} or all, dtype in {tuple(_AGMM_IN0_DTYPES)}")
        for rr in (_AGMM_IN0_ROLES if r == "all" else (r,)):
            table[rr] = d
    d = table.get(role)
    if d is not None and (role, d) not in _AGMM_IN0_LOGGED:
        _AGMM_IN0_LOGGED.add((role, d))
        logger.info(f"QWEN36_AGMM_IN0_DTYPE engaged: role={role} dtype={d}")
    return d


_PREFILL_MLP_LOWP_SITES = ("hidden", "partial")
_PREFILL_MLP_LOWP_LOGGED = set()


def prefill_mlp_lowp(site):
    """QWEN36_PREFILL_MLP_LOWP=<site>=<dtype>[,...] (c113), read per call (wb set screens it): the dtype of a prefill
    MLP tensor. site hidden = the fused AGMM SwiGLU output (w2's in0), partial = the w2 output (the reduce-scatter
    payload and its sum). dtype bf8 | bf4 (bf4 is the must-fail rung). Unset = bf16 at both sites, as shipped."""
    v = _tt_public_getenv("QWEN36_PREFILL_MLP_LOWP", "").strip()
    if v in ("", "0"):
        return None
    table = {}
    for item in (i.strip() for i in v.split(",")):
        if not item:
            continue
        k, sep, d = (t.strip() for t in item.partition("="))
        if not sep or d not in _AGMM_IN0_DTYPES or k not in _PREFILL_MLP_LOWP_SITES:
            raise ValueError(f"QWEN36_PREFILL_MLP_LOWP={v!r}: expected <site>=<dtype> items, site in "
                             f"{_PREFILL_MLP_LOWP_SITES}, dtype in {tuple(_AGMM_IN0_DTYPES)}")
        table[k] = d
    d = table.get(site)
    if d is not None and (site, d) not in _PREFILL_MLP_LOWP_LOGGED:
        _PREFILL_MLP_LOWP_LOGGED.add((site, d))
        logger.info(f"QWEN36_PREFILL_MLP_LOWP engaged: site={site} dtype={d}")
    return _AGMM_IN0_DTYPES[d] if d is not None else None


# ---------------------------------------------------------------- shared warmup self-test (prefill levers, 10-06)
# One fail-closed bitwise self-test per lever and process, run on the first (eager warmup) call of the op, before any
# trace capture. Three programs on the LIVE inputs of that call:
#   ref   = the shipped program (the lever forced off),
#   cand  = the lever's program,
#   wrong = the lever's program on a deliberately wrong input (a must-differ rung).
# PASS needs cand == ref bitwise on every device AND wrong != ref somewhere. A wrong rung that matches means the
# compare is blind (e.g. the readback does not see the output), so the boot is REFUSED either way. Each real boot thus
# proves its own test can fail (owner 10-03: no separate must-fail boot). <FLAG>_MUSTFAIL=1 corrupts the candidate
# itself, so the test REFUSES (proven in ttsim, never by a card boot).
_PREFILL_SELFTEST = {}


def _bitwise_diff(a, b):
    """Differing elements between two lists of per-device host tensors (int view: NaN-safe, -0 != +0)."""
    n = 0
    for x, y in zip(a, b):
        if x.shape != y.shape or x.dtype != y.dtype:
            n += max(x.numel(), y.numel())
            continue
        if x.dtype in (torch.float32, torch.int32, torch.uint32):
            n += int((x.view(torch.int32) != y.view(torch.int32)).sum())
        elif x.element_size() == 2:
            n += int((x.view(torch.int16) != y.view(torch.int16)).sum())
        else:
            n += int((x != y).sum())
    return n


def _device_readback(t):
    """Per-device host copies of one mesh tensor or of a list of them (flattened in order)."""
    ts = t if isinstance(t, (list, tuple)) else [t]
    return [ttnn.to_torch(d) for x in ts for d in ttnn.get_device_tensors(x)]


def _dealloc(t):
    for x in (t if isinstance(t, (list, tuple)) else [t]):
        ttnn.deallocate(x)


def prefill_selftest(flag, ref, cand, wrong, desc, ok_banner=None):
    """Run the shared self-test once per `flag` and process. ref/cand/wrong: zero-arg callables that each return a
    new device tensor or a list of them (deallocated here). Raises RuntimeError (boot REFUSED) on a mismatch or a
    blind wrong rung; logs '<flag> self-test exact (...)' on a pass. No-op after the first call.
    ok_banner: the literal pass-banner prefix (default '<flag> self-test exact'); pass it as a literal so ttx lint
    can find the arm's expected banner in the source.
    """
    if flag in _PREFILL_SELFTEST:
        if _PREFILL_SELFTEST[flag] is False:
            raise RuntimeError(f"{flag} self-test REFUSED earlier in this process")
        return
    outs = {}
    for name, fn in (("ref", ref), ("cand", cand), ("wrong", wrong)):
        t = fn()
        outs[name] = _device_readback(t)
        _dealloc(t)
    d_cand = _bitwise_diff(outs["ref"], outs["cand"])
    d_wrong = _bitwise_diff(outs["ref"], outs["wrong"])
    if d_cand or not d_wrong:
        _PREFILL_SELFTEST[flag] = False
        why = (f"{d_cand} output elements differ from the shipped program" if d_cand
               else "the wrong-input rung matched the shipped program (blind compare)")
        raise RuntimeError(f"{flag} self-test REFUSED ({desc}): {why}")
    _PREFILL_SELFTEST[flag] = True
    logger.info(f"{ok_banner or flag + ' self-test exact'} ({desc}): bitwise equal to the shipped program on "
                f"{len(outs['ref'])} device(s); wrong-input rung differs in {d_wrong} elements")


def _rel_err(a, b):
    """||b - a|| / ||a|| over two lists of per-device host tensors (fp64; inf when a shape differs or a is zero)."""
    num = den = 0.0
    for x, y in zip(a, b):
        if x.shape != y.shape:
            return float("inf")
        x64, y64 = x.double(), y.double()
        num += float(((y64 - x64) ** 2).sum())
        den += float((x64 ** 2).sum())
    return (num / den) ** 0.5 if den > 0 else float("inf")


def prefill_selftest_tol(flag, ref, cand, wrong, desc, tol, ok_banner):
    """prefill_selftest for a tier-2 lever (not bit-exact by construction): PASS needs cand within `tol` of ref
    (relative Frobenius error over every device, or bitwise equal) AND the wrong-input rung beyond 10 x tol. Same
    once-per-flag bookkeeping and fail-closed RuntimeError. The pass banner names exact/within and both errors."""
    if flag in _PREFILL_SELFTEST:
        if _PREFILL_SELFTEST[flag] is False:
            raise RuntimeError(f"{flag} self-test REFUSED earlier in this process")
        return
    outs = {}
    for name, fn in (("ref", ref), ("cand", cand), ("wrong", wrong)):
        t = fn()
        outs[name] = _device_readback(t)
        _dealloc(t)
    d_cand = _bitwise_diff(outs["ref"], outs["cand"])
    e_cand = 0.0 if not d_cand else _rel_err(outs["ref"], outs["cand"])
    e_wrong = _rel_err(outs["ref"], outs["wrong"])
    if not (e_cand <= tol) or not (e_wrong > 10 * tol):
        _PREFILL_SELFTEST[flag] = False
        why = (f"rel_err {e_cand:.3e} > tol {tol:.1e} ({d_cand} elements differ)" if not (e_cand <= tol)
               else f"the wrong-input rung is within 10 x tol of the shipped program (rel_err {e_wrong:.3e}: blind)")
        raise RuntimeError(f"{flag} self-test REFUSED ({desc}): {why}")
    _PREFILL_SELFTEST[flag] = True
    logger.info(f"{ok_banner} ({desc}): {'exact' if not d_cand else 'within tol'} vs the shipped program on "
                f"{len(outs['ref'])} device(s): rel_err {e_cand:.3e} (tol {tol:.1e}, {d_cand} elements differ); "
                f"wrong-input rung rel_err {e_wrong:.3e}")


# ---------------------------------------------------------------- QWEN36_PREFILL_W2_T11 (prefill w2 on 110 cores)
# The prefill down-projection (ttnn.linear, 2D mcast, [2048,8704] bf8 x [8704,5120] bf4 per chip, 8.96% of a 32k
# prefill: 29.4 ms per 2048 chunk at 72.5% of its 100-core LoFi peak, prefill-opmark-20261003T135353Z) runs on the
# 10x10 grid that _best_prefill_cols picks (per_core_M 7, per_core_N 16). M = 64 tile rows split 7 per row puts 9 full
# rows plus one row with a single tile row, and column 10 is idle: the critical core does 7 x 16 = 112 output tiles.
# Transposed (transpose_mcast: M over the 11 COLUMNS, N over the 10 ROWS) the same op is per_core_M 6 (64 = 10 x 6 + 4)
# x per_core_N 16 = 96 output tiles on the critical core, all 110 cores busy: -14% of the FPU work per core
# [estimate]. in0_block_w (the K blocking) and the output subblock (1 x 4) are kept, so every output tile sees the
# same K accumulation order as today: bit-exact by construction, and the warmup self-test checks it bitwise.
#   QWEN36_PREFILL_W2_T11 = 1          engaged (prefill w2 only; decode M <= 32 never takes this path)
#   QWEN36_PREFILL_W2_T11_MUSTFAIL = 1 the candidate runs on an input whose first 32 rows are zeroed -> the self-test
#                                      REFUSES (ttsim proves it; never booted)
_W2T_LOGGED = set()


def prefill_w2_t11():
    v = _tt_public_getenv("QWEN36_PREFILL_W2_T11", "").strip()
    mf = _tt_public_getenv("QWEN36_PREFILL_W2_T11_MUSTFAIL", "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"QWEN36_PREFILL_W2_T11={v!r}: expected unset, 0 or 1")
    if mf not in ("", "0", "1"):
        raise ValueError(f"QWEN36_PREFILL_W2_T11_MUSTFAIL={mf!r}: expected unset, 0 or 1")
    if mf == "1" and v != "1":
        raise ValueError("QWEN36_PREFILL_W2_T11_MUSTFAIL=1 needs QWEN36_PREFILL_W2_T11=1")
    return v == "1"


def prefill_w2_t11_mustfail():
    return _tt_public_getenv("QWEN36_PREFILL_W2_T11_MUSTFAIL", "") == "1"


def create_prefill_w2_t11_program_config(m, n, base_cfg, grid=(11, 10)):
    """The transposed 11x10 2D-mcast config for the prefill w2, built from the shipped `base_cfg` (same in0_block_w
    and subblock, so the K order per output tile is unchanged). None (the caller keeps base_cfg) when the shape does not
    fit (ceil(Mt/per_core_M) <= grid.x, ceil(Nt/per_core_N) <= grid.y), when it does not lower per_core_M (buckets
    <= 512 rows: 1-2 tile rows per core either way) or when per_core_N would change."""
    gx, gy = grid
    mt, nt = -(-m // TILE_SIZE), -(-n // TILE_SIZE)
    pcm, pcn = -(-mt // gx), -(-nt // gy)
    if pcn % base_cfg.out_subblock_w or pcm % base_cfg.out_subblock_h:
        return None
    if -(-mt // pcm) > gx or -(-nt // pcn) > gy:
        return None
    if pcm >= base_cfg.per_core_M or pcn != base_cfg.per_core_N:  # no fewer tile rows per core (small buckets)
        return None
    key = (m, n)
    if key not in _W2T_LOGGED:
        _W2T_LOGGED.add(key)
        logger.info(f"QWEN36_PREFILL_W2_T11 engaged: prefill w2 grid=({gx}, {gy}) transpose_mcast, per_core_M={pcm} "
                    f"per_core_N={pcn} (was {base_cfg.per_core_M}x{base_cfg.per_core_N} on "
                    f"{getattr(base_cfg, 'compute_with_storage_grid_size', '?')}), in0_block_w={base_cfg.in0_block_w} "
                    f"subblock {base_cfg.out_subblock_h}x{base_cfg.out_subblock_w}, M={m} N={n}"
                    + (" MUSTFAIL: first 32 input rows zeroed" if prefill_w2_t11_mustfail() else ""))
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=base_cfg.in0_block_w,
        out_subblock_h=base_cfg.out_subblock_h,
        out_subblock_w=base_cfg.out_subblock_w,
        per_core_M=pcm,
        per_core_N=pcn,
        transpose_mcast=True,
        fused_activation=None,
        fuse_batch=False,
    )


def zero_first_rows(x, rows=TILE_SIZE):
    """A new tensor equal to x [.., M, K] with its first `rows` rows zeroed (a must-fail / wrong-input rung)."""
    shp = list(x.shape)
    k = shp[-1]
    lead = [0] * (len(shp) - 2)
    head = ttnn.slice(x, lead + [0, 0], shp[:-2] + [rows, k])
    tail = ttnn.slice(x, lead + [rows, 0], shp)
    z = ttnn.zeros_like(head)
    out = ttnn.concat([z, tail], dim=len(shp) - 2)
    for t in (head, tail, z):
        ttnn.deallocate(t)
    return out


_PREFILL_NORM_DTYPE_LOGGED = set()


def prefill_norm_dtype():
    """QWEN36_PREFILL_NORM_DTYPE=bf8 (c118), read per call: the TP prefill attention/ff norms emit their distributed
    (K-sharded) output in bf8, so a bf8-in0 AGMM (QWEN36_AGMM_IN0_DTYPE=all=bf8) needs no typecast. Refused unless
    every AGMM role takes bf8 in0: a bf8 activation into a bf16 gather buffer is the gather-dtype trap. Unset = None."""
    v = _tt_public_getenv("QWEN36_PREFILL_NORM_DTYPE", "").strip()
    if v in ("", "0"):
        return None
    if v != "bf8":
        raise ValueError(f"QWEN36_PREFILL_NORM_DTYPE={v!r}: only bf8")
    if any(agmm_in0_dtype(r) != "bf8" for r in _AGMM_IN0_ROLES):
        raise ValueError("QWEN36_PREFILL_NORM_DTYPE=bf8 needs QWEN36_AGMM_IN0_DTYPE=all=bf8 (gather buffer dtype)")
    if v not in _PREFILL_NORM_DTYPE_LOGGED:
        _PREFILL_NORM_DTYPE_LOGGED.add(v)
        logger.info("QWEN36_PREFILL_NORM_DTYPE engaged: prefill attn/ff distributed norm output bf8")
    return ttnn.bfloat8_b


def _agmm_in0_prepare(x4, role, tt_ccl, persistent_output_buffer):
    """-> (x4, persistent_output_buffer, owned). Unset: returned unchanged (owned False). Engaged: x4 cast to the in0
    dtype (owned True unless it already was that dtype) and a gather buffer in that dtype replaces any other."""
    # QWEN36_AGMM_GBS (0057): the C++ op reads the env per call, so a caller's gather buffer must match the CURRENT
    # mode (the self-test's ref runs with GBS=0). A sharded buffer under GBS=0 compiles dm_in0_sender without AGMM_GBS
    # against a sharded accessor (no `page_size`): the 2026-10-20 attn_out boot failure (its x is already bf8, so the
    # dtype swap below never replaced the caller's buffer). Replace a mismatched buffer with one for the current mode.
    if persistent_output_buffer is not None and \
            persistent_output_buffer.memory_config().is_sharded() != bool(agmm_gbs()):
        persistent_output_buffer = agmm_gather_buffer(tt_ccl, x4)
    d = agmm_in0_dtype(role)
    if d is None:
        return x4, persistent_output_buffer, False
    owned = x4.dtype != _AGMM_IN0_DTYPES[d]
    if owned:
        x4 = ttnn.typecast(x4, _AGMM_IN0_DTYPES[d], memory_config=x4.memory_config())
    if persistent_output_buffer is None or persistent_output_buffer.dtype != x4.dtype:
        persistent_output_buffer = agmm_gather_buffer(tt_ccl, x4)
    return x4, persistent_output_buffer, owned


def _agmm_unfused_grid(device):
    """-> ((gx, gy), overridden). Default = the device compute grid (11x10 on P150)."""
    g = device.compute_with_storage_grid_size()
    v = _tt_public_getenv("QWEN36_AGMM_UNFUSED_GRID", "").strip()
    if not v:
        return (g.x, g.y), False
    gx, gy = (int(t) for t in v.lower().split("x"))
    if not (1 <= gx <= g.x and 1 <= gy <= g.y):
        raise ValueError(f"QWEN36_AGMM_UNFUSED_GRID={v}: outside the {g.x}x{g.y} compute grid")
    return (gx, gy), True


def _agmm_unfused_ag_cfg():
    v = _tt_public_getenv("QWEN36_AGMM_UNFUSED_AG", "").strip()
    if not v:
        return 2, 10, 2, 8
    links, chunks, workers, buffers = (int(t) for t in v.split(","))
    return links, chunks, workers, buffers


def _agmm_unfused_gather(x4, tt_ccl, topology, cluster_axis):
    """all_gather_async of the K-sharded x4 [1,1,S,K/tp] along dim 3 into DRAM (at S=2048, K=5120 bf16 that is 20 MiB:
    never L1). Semaphores from tt_ccl on the same axis as the AGMM, plus the barrier the AGMM lacks (#57458)."""
    links, chunks, workers, buffers = _agmm_unfused_ag_cfg()
    src = ttnn.sharded_to_interleaved(x4, ttnn.DRAM_MEMORY_CONFIG) if x4.is_sharded() else x4
    gathered = ttnn.experimental.all_gather_async(
        src,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=links,
        cluster_axis=cluster_axis,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
        chunks_per_sync=chunks,
        num_workers_per_link=workers,
        num_buffers_per_channel=buffers,
    )
    if src is not x4:
        ttnn.deallocate(src)
    return gathered, (links, chunks, workers, buffers)


def _agmm_unfused_minimal_cfg(role, S, K, n_tiles, grid, m_block, n_block_cap, pair):
    """MinimalMatmulConfig for the gathered x [S,K] @ w [K, n_tiles*32] on `grid`. minimal_matmul puts N on grid x and
    M on grid y unless M > N (it transposes). Block sizes follow the nt11x8 AGMM recipe: N block = per-core N rounded
    up to 4 (to 2 for swiglu pairs), capped; K block = agmm_k_block_size of the FULL K."""
    m_tiles = math.ceil(S / TILE_SIZE)
    n_axis = grid[1] if m_tiles > n_tiles else grid[0]
    if pair:
        per_core = 2 * math.ceil(n_tiles // 2 / n_axis)
        n_block = min(per_core + (per_core % 2), n_block_cap)
        default_w = max(d for d in (4, 2) if n_block % d == 0)
    else:
        per_core = max(1, math.ceil(n_tiles / n_axis))
        n_block = min(((per_core + 3) // 4) * 4, n_block_cap)
        default_w = max(d for d in (4, 2, 1) if n_block % d == 0)
    sub_h, sub_w = agmm_subblock(role, m_block, n_block, default_w)
    return ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=agmm_k_block_size(K),
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )


def _agmm_unfused_matmul(x4, weight, tt_ccl, compute_cfg, topology, cluster_axis, fused_activation, out_memory_config):
    """Unfused all_gather_matmul_prefill: gather, then bmm (w2 recipe) or minimal_matmul. Same return contract:
    bf16 [1,1,S,N_local] in out_memory_config."""
    S = x4.shape[-2]
    x_full, ag = _agmm_unfused_gather(x4, tt_ccl, topology, cluster_axis)
    K, N = x_full.shape[-1], weight.shape[-1]
    mm = _tt_public_getenv("QWEN36_AGMM_UNFUSED_MM", "auto").strip() or "auto"
    if mm not in ("auto", "minimal"):
        raise ValueError(f"QWEN36_AGMM_UNFUSED_MM={mm!r}: expected auto or minimal")
    grid, overridden = _agmm_unfused_grid(tt_ccl.mesh_device)
    ckc = agmm_compute_cfg(compute_cfg)
    if mm == "auto" and not weight.is_sharded():
        kind = "bmm"
        if overridden:
            pc = create_prefill_matmul_program_config(S, K, N, grid_size=grid, fused_activation=fused_activation)
        else:  # mlp.py prefill down-proj recipe: subblock-first width up to the device width, prefill_grid_default rows
            pc = create_prefill_mlp_matmul_program_config(S, K, N, fused_activation=fused_activation, max_cols=grid[0])
        _g = pc.compute_with_storage_grid_size
        grid = (_g.x, _g.y) if hasattr(_g, "x") else tuple(_g)
        detail = f"in0_block_w={pc.in0_block_w} per_core={pc.per_core_M}x{pc.per_core_N} subblock=1x{pc.out_subblock_w}"
        out = ttnn.linear(x_full, weight, compute_kernel_config=ckc, program_config=pc, memory_config=out_memory_config,
                          dtype=ttnn.bfloat16)
    else:
        kind = "minimal"
        n_cap = int(_tt_public_getenv("QWEN36_AGMM_NT_NBLOCK", "12"))
        cfg = _agmm_unfused_minimal_cfg("matmul", S, K, math.ceil(N / TILE_SIZE), grid, 4, n_cap, pair=False)
        detail = f"M_block=4 K_block={cfg.K_block_size} N_block={cfg.N_block_size} subblock={cfg.subblock_h}x{cfg.subblock_w}"
        out = ttnn.experimental.minimal_matmul(x_full, weight, fused_activation=fused_activation, config=cfg,
                                               compute_kernel_config=ckc, memory_config=out_memory_config,
                                               dtype=ttnn.bfloat16)
    ttnn.deallocate(x_full)
    key = ("matmul", kind, S, K, N)
    if key not in _AGMM_UNFUSED_LOGGED:
        _AGMM_UNFUSED_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_UNFUSED engaged: role=matmul grid={grid[0]}x{grid[1]} mm={kind} S={S} K={K} N={N} "
                    f"{detail} weight_sharded={weight.is_sharded()} ag=links{ag[0]},chunks{ag[1]},workers{ag[2]},"
                    f"bufs{ag[3]} gather=DRAM")
    return out


def _agmm_unfused_swiglu(x4, weight, tt_ccl, compute_cfg, topology, cluster_axis, out_memory_config):
    """Unfused all_gather_swiglu_prefill: gather, then minimal_matmul(fuse_swiglu) on the packed w_gate_up.
    Output bf16 [1,1,S,N_packed/2] in out_memory_config; the full-width gate/up never exists as a tensor."""
    S = x4.shape[-2]
    x_full, ag = _agmm_unfused_gather(x4, tt_ccl, topology, cluster_axis)
    K, N = x_full.shape[-1], weight.shape[-1]
    grid, _ = _agmm_unfused_grid(tt_ccl.mesh_device)
    m_block = int(_tt_public_getenv("QWEN36_AGMM_MLP_MBLOCK", "4"))
    n_cap = int(_tt_public_getenv("QWEN36_AGMM_MLP_NBLOCK", "28"))
    cfg = _agmm_unfused_minimal_cfg("swiglu", S, K, math.ceil(N / TILE_SIZE), grid, m_block, n_cap, pair=True)
    out = ttnn.experimental.minimal_matmul(x_full, weight, config=cfg, compute_kernel_config=agmm_compute_cfg(compute_cfg),
                                           memory_config=out_memory_config, dtype=ttnn.bfloat16, fuse_swiglu=True)
    ttnn.deallocate(x_full)
    key = ("swiglu", S, K, N)
    if key not in _AGMM_UNFUSED_LOGGED:
        _AGMM_UNFUSED_LOGGED.add(key)
        logger.info(f"QWEN36_AGMM_UNFUSED engaged: role=swiglu grid={grid[0]}x{grid[1]} mm=minimal_fuse_swiglu S={S} "
                    f"K={K} N_packed={N} M_block={m_block} K_block={cfg.K_block_size} N_block={cfg.N_block_size} "
                    f"subblock={cfg.subblock_h}x{cfg.subblock_w} ag=links{ag[0]},chunks{ag[1]},workers{ag[2]},"
                    f"bufs{ag[3]} gather=DRAM")
    return out


def all_gather_matmul_prefill(
    x,
    weight,
    tt_ccl,
    compute_cfg,
    topology,
    grid=(7, 9),
    cluster_axis=1,
    fused_activation=None,
    out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
    persistent_output_buffer=None,
    chunk_sizes=None,
    role="matmul",
):
    """Fused all-gather(dim=3) + column-parallel matmul for prefill (all_gather_minimal_matmul_async).

    role (c134): which call site this is, for the per-role grid choice QWEN36_AGMM_GRID9 (gdn_in, attn_in, gdn_out,
    attn_out). The default "matmul" matches no GRID9 role, so an untagged caller keeps today's grid.

    chunk_sizes (QWEN36_AGMM_CHUNKS, c105): per-chunk output widths (tile multiples summing to N); the op
    then writes each column block to its own tensor (one memory config for all) and a LIST is returned.
    None (the default) = the single-tensor return, byte-identical to before. The caller gates it with
    agmm_chunks(role), which refuses it under QWEN36_AGMM_UNFUSED=matmul.

    x: K-sharded activation [.,S,K/tp]; weight: [K,N] col-sharded (K full). Gathers x to full K and
    matmuls in one op, replacing a separate all_gather + linear. fused_activation applied per tile
    before pack (non-parametrized op, e.g. ttnn.UnaryOpType.SILU). out_memory_config places the result
    (default DRAM; L1 keeps it resident for downstream slices). persistent_output_buffer: the gather buffer
    (agmm_gather_buffer); None lets the op allocate one per call."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    if agmm_unfused("matmul"):  # QWEN36_AGMM_UNFUSED (c103): separate gather + full-grid matmul
        if chunk_sizes is not None:
            raise ValueError("chunk_sizes with QWEN36_AGMM_UNFUSED=matmul: gate it with agmm_chunks(role)")
        return _agmm_unfused_matmul(x4, weight, tt_ccl, compute_cfg, topology, cluster_axis, fused_activation,
                                    out_memory_config)
    # AG-bound: 2 ethernet links parallelize the gather (P150x4 max; traced_8k TTFT win). grid.x must
    # = num_links*workers, and the 7-wide default (prime) forces 1 link -> widen to 8 (2 links, 4 workers).
    num_links = 2
    grid = _agmm_prefill_grid((8, grid[1]))
    workers = max(1, grid[0] // num_links)
    force_transpose = True
    n_block = 8
    n_tiles = math.ceil(weight.shape[-1] / TILE_SIZE)
    _nt = _agmm_nt11x8(n_tiles, S, "matmul")
    if _nt is not None:
        grid, force_transpose, workers = _nt
        # Per-core N doubles at TP=2 vs upstream's TP=4 (GDN 258 tiles -> 24/core, attention 224 -> 21/core).
        # Cap the N block at upstream's validated TP=4 size (12) so the fp32 intermediate fits L1; the op
        # iterates the remaining N blocks. QWEN36_AGMM_NT_NBLOCK overrides (multiple of 4).
        per_core = max(1, math.ceil(n_tiles / grid[0]))
        n_block = min(((per_core + 3) // 4) * 4, int(_tt_public_getenv("QWEN36_AGMM_NT_NBLOCK", "12")))
    _g9 = agmm_grid9(role, S, _nt)  # QWEN36_AGMM_GRID9 (c134): same 11 columns, M over 9 rows; n_block unchanged
    if _g9 is not None:
        grid, force_transpose, workers = _g9
    _t9 = agmm_t11x9(role, S, _nt, _g9)  # QWEN36_AGMM_T11X9 (10-06): transposed 11x9, M over 11 columns
    if _t9 is not None:
        if agmm_t11x9_needs_selftest(role):
            agmm_t11x9_selftest(role, lambda xi: all_gather_matmul_prefill(
                xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis, fused_activation=fused_activation,
                out_memory_config=out_memory_config, persistent_output_buffer=persistent_output_buffer,
                chunk_sizes=chunk_sizes, role=role), x)
        grid, force_transpose, workers = _t9
        n_block = agmm_t11x9_nblock(n_tiles, grid[1])
    # QWEN36_AGMM_FWO (0063): the C++ op reads the env per call. Its self-test runs FIRST (before swpk / gbs), so its
    # bitwise compare is made in the served context (swpk deferral and gbs engaged); the nested ones run under FWO=0.
    _fwo = agmm_fwo_check(role, grid, force_transpose)
    if _fwo is not None and agmm_fwo_needs_selftest(role):
        agmm_fwo_selftest(role, _fwo, lambda xi: all_gather_matmul_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis, fused_activation=fused_activation,
            out_memory_config=out_memory_config, persistent_output_buffer=persistent_output_buffer,
            chunk_sizes=chunk_sizes, role=role), x)
    _swpk = agmm_swiglu_pack_engaged(role)  # QWEN36_AGMM_SWIGLU_PACK (0049): matmul roles take only the deferral
    if _swpk is not None and agmm_swiglu_pack_needs_selftest(role):
        agmm_swiglu_pack_selftest(role, _swpk, lambda xi: all_gather_matmul_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis, fused_activation=fused_activation,
            out_memory_config=out_memory_config, persistent_output_buffer=persistent_output_buffer,
            chunk_sizes=chunk_sizes, role=role), x)
    _gbs = agmm_gbs()  # QWEN36_AGMM_GBS (0057): sharded gather buffer; the C++ op reads the env per call
    if _gbs is not None and agmm_gbs_needs_selftest(role):
        agmm_gbs_selftest(role, _gbs, lambda xi: all_gather_matmul_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis, fused_activation=fused_activation,
            out_memory_config=out_memory_config, persistent_output_buffer=persistent_output_buffer,
            chunk_sizes=chunk_sizes, role=role), x)
    # QWEN36_AGMM_IN_L1FIT (k-gdnpre): the in-projection roles take ONE M block per core (M_block 8) with the
    # factory-derived L1 cut (N_block 12 -> 8). The in-boot self-test runs here, after the FWO / swpk / gbs
    # self-tests, so the bitwise compare is made in the served context (fwo, swpk and gbs engaged).
    _l1fit = agmm_in_l1fit()
    if _l1fit is not None and role in agmm_in_l1fit_roles() and agmm_in_l1fit_needs_selftest(role):
        agmm_in_l1fit_selftest(role, _l1fit, lambda xi: all_gather_matmul_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis, fused_activation=fused_activation,
            out_memory_config=out_memory_config, persistent_output_buffer=persistent_output_buffer,
            chunk_sizes=chunk_sizes, role=role), x)
    m_block = agmm_m_block(role, S, grid, force_transpose)  # QWEN36_AGMM_MBLOCK (c134c); unset = 4, as shipped
    if _l1fit is not None and role in agmm_in_l1fit_roles():
        # ONE M block per core: the in1 (weight) slice and the in0 (gathered activation) chain are streamed once per
        # call instead of twice (the in1 reread factor 2.0x -> 1.0x). N_block 12 -> 8 is the L1 cut (the CB total
        # 1024 -> 768 KiB, see the block comment above): without it the boot refuses with an L1 CB clash.
        m_block = 8
        n_block = 8
        _key = (role, int(S), tuple(grid))
        if _key not in _AGMM_IN_L1FIT_SPLIT:
            _AGMM_IN_L1FIT_SPLIT.add(_key)
            _m_tiles = -(-int(S) // TILE_SIZE)
            _m_axis = grid[0] if force_transpose else grid[1]
            _per_core = -(-_m_tiles // _m_axis)
            logger.info(f"QWEN36_AGMM_IN_L1FIT engaged: role={role} M_block=8 N_block=8 S={S} grid={tuple(grid)} "
                        f"M_tiles/core={_per_core} N_tiles/core={max(1, math.ceil(n_tiles / grid[0]))} "
                        f"-> one M block per core (1 weight pass), N_block 12->8 (the L1 cut: 808 KiB of CBs)")
    sub_h, sub_w = agmm_subblock("matmul", m_block, n_block, max(d for d in (4, 2, 1) if n_block % d == 0))
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    if persistent_output_buffer is None:
        persistent_output_buffer = agmm_persistent_default(tt_ccl, x4, "matmul")
    _mf_x4 = None
    if _g9 is not None and agmm_grid9_mustfail():
        x4 = _mf_x4 = agmm_grid9_mustfail_swap(x4, role)
    x4, persistent_output_buffer, _in0_owned = _agmm_in0_prepare(x4, "matmul", tt_ccl, persistent_output_buffer)
    _chunk_kw = {} if chunk_sizes is None else {"chunks": len(chunk_sizes), "chunk_sizes": [int(c) for c in chunk_sizes]}
    outs = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        config=cfg,
        fused_activation=fused_activation,
        compute_kernel_config=agmm_compute_cfg(compute_cfg),
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        force_transpose=force_transpose,
        num_workers_per_link=workers,
        num_buffers_per_channel=agmm_bufs(),
        persistent_output_buffer=persistent_output_buffer,
        **_chunk_kw,
    )
    if _in0_owned:
        ttnn.deallocate(x4)
    if _mf_x4 is not None and not (_in0_owned and x4 is _mf_x4):
        ttnn.deallocate(_mf_x4)
    if chunk_sizes is not None:
        return list(outs)
    out = outs[0]

    return out


# ---------------------------------------------------------------- c105 prefill small-op removals
# Four independent knobs, prefill only (decode paths never read them); each unset = the shipped op sequence.
#   QWEN36_GDN_CONV_STATE_RM=1     gdn/tp.py _conv_kda_prefill: the conv carry (last K-1 rows) is sliced from the
#                                  row-major x_rm the kda op already needs, instead of a slice at the unaligned row
#                                  T-3 of the TILE qkv (which untilizes the whole [T, qkv_dim_tp] block). Per call.
#   QWEN36_AGMM_CHUNKS=gdn,attn    the fused in-projection AGMM writes its column blocks as separate tensors
#                                  (chunks/chunk_sizes of all_gather_minimal_matmul_async) instead of one tensor that
#                                  is then sliced. Per call. Falls back to slicing under QWEN36_AGMM_UNFUSED=matmul.
#   QWEN36_GDN_CONV_WINDOW_ONCE=1  the decode conv window is written once per prefill (model.prefill_traced_chunked
#                                  finalize) instead of in every chunk inside the chunk trace. Boot time (the chunk
#                                  trace is captured with it).
#   QWEN36_QK_NORM_W=1             attention prefill q/k rms_norm takes weight= (gamma in-kernel) instead of a
#                                  separate row-broadcast multiply. Per call. Numerics move (tier 2).
_SMALLOPS_LOGGED = set()


def _smallops_banner(key, msg):
    if key not in _SMALLOPS_LOGGED:
        _SMALLOPS_LOGGED.add(key)
        logger.info(msg)


def gdn_conv_state_rm():
    return _tt_public_getenv("QWEN36_GDN_CONV_STATE_RM", "0") == "1"


_AGMM_CHUNK_ROLES = ("gdn", "attn")


def agmm_chunks(role):
    """QWEN36_AGMM_CHUNKS: comma list of {gdn, attn}, or 1/all. Read per call. False when the role is off or when
    QWEN36_AGMM_UNFUSED=matmul replaces the fused op for that role (logged once; the slicing path is kept)."""
    v = _tt_public_getenv("QWEN36_AGMM_CHUNKS", "").strip()
    if not v or v == "0":
        return False
    roles = set(_AGMM_CHUNK_ROLES) if v in ("1", "all") else {r.strip() for r in v.split(",") if r.strip()}
    bad = roles - set(_AGMM_CHUNK_ROLES)
    if bad:
        raise ValueError(f"QWEN36_AGMM_CHUNKS={v!r}: unknown roles {sorted(bad)} (expected {_AGMM_CHUNK_ROLES} or all)")
    if role not in roles:
        return False
    if agmm_unfused("matmul"):
        _smallops_banner(("agmm_chunks_fallback", role),
                         f"QWEN36_AGMM_CHUNKS fallback: role={role} keeps one output + slices because "
                         f"QWEN36_AGMM_UNFUSED=matmul replaces the fused AGMM")
        return False
    return True


def agmm_chunks_banner(role, chunk_sizes, mem):
    _smallops_banner(("agmm_chunks", role, tuple(chunk_sizes)),
                     f"QWEN36_AGMM_CHUNKS engaged: role={role} chunk_sizes={list(chunk_sizes)} mem={mem}")


def gdn_conv_window_once():
    return _tt_public_getenv("QWEN36_GDN_CONV_WINDOW_ONCE", "0") == "1"


def qk_norm_w():
    on = _tt_public_getenv("QWEN36_QK_NORM_W", "0") == "1"
    if on:
        _smallops_banner("qk_norm_w", "QWEN36_QK_NORM_W engaged: attention prefill q/k rms_norm applies q_norm/k_norm "
                                      "as weight= (separate row-bcast multiply dropped)")
    return on


def mlp_gateup_agmm_enabled(num_devices):
    """Fuse the ff_norm all-gather into the MLP gate/up matmul (prefill). TP-only (needs the gather)."""
    return num_devices > 1


def all_gather_swiglu_prefill(
    x, weight, tt_ccl, compute_cfg, topology, grid=(7, 9), cluster_axis=1, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
):
    """Fused all-gather + col-parallel gate/up matmul + SwiGLU for prefill (packing gate+up lets ff_norm's AG fuse in).

    x: K-sharded [.,S,K/tp]; weight: tile-pair-interleaved [gate|up] [K, 2N/tp]. Emits silu(gate)*up of width N/tp."""
    S, K_local = x.shape[-2], x.shape[-1]
    x4 = ttnn.reshape(x, (1, 1, S, K_local))
    if agmm_unfused("swiglu"):  # QWEN36_AGMM_UNFUSED (c103): separate gather + minimal_matmul(fuse_swiglu)
        return _agmm_unfused_swiglu(x4, weight, tt_ccl, compute_cfg, topology, cluster_axis, out_memory_config)
    num_links = 2
    grid = _agmm_prefill_grid((8, grid[1]))
    workers = max(1, grid[0] // num_links)
    force_transpose = True
    m_block, n_block = 8, 16
    _nt = _agmm_nt11x8(math.ceil(weight.shape[-1] / TILE_SIZE), S, "swiglu")
    if _nt is not None:
        grid, force_transpose, workers = _nt
        # TP=2 gate|up: 544 interleaved tiles -> 50/core over 11 columns. Upstream (TP=4, 272 -> 25/core)
        # measured M8/N28 overflowing L1 and M4/N28 fitting; N_block must be even (gate/up pairs).
        # Default = upstream's validated block (M4/N28, subblock 1x4; the gathered K=5120 is the same at
        # TP=2 and TP=4, so the per-block L1 is identical): 2 blocks/core, 12% padding. N26 (4% padding,
        # subblock 1x2) is the obvious sweep point. QWEN36_AGMM_MLP_MBLOCK / _NBLOCK override.
        m_block = int(_tt_public_getenv("QWEN36_AGMM_MLP_MBLOCK", "4"))
        n_block = int(_tt_public_getenv("QWEN36_AGMM_MLP_NBLOCK", "28"))
    _g9 = agmm_grid9("mlp", S, _nt)  # QWEN36_AGMM_GRID9 (c134): gate|up keeps its 11-column N split and blocks
    if _g9 is not None:
        grid, force_transpose, workers = _g9
    _t9 = agmm_t11x9("mlp", S, _nt, _g9)  # QWEN36_AGMM_T11X9: 272 pairs over 9 rows -> 31 pairs (62 tiles)/core;
    if _t9 is not None:                    # the 28-tile N block (pairs kept whole) and every CB size are unchanged
        if agmm_t11x9_needs_selftest("mlp"):
            agmm_t11x9_selftest("mlp", lambda xi: all_gather_swiglu_prefill(
                xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis,
                out_memory_config=out_memory_config), x)
        grid, force_transpose, workers = _t9
    _fwo = agmm_fwo_check("mlp", grid, force_transpose)  # QWEN36_AGMM_FWO (0063): self-test first (served context)
    if _fwo is not None and agmm_fwo_needs_selftest("mlp"):
        agmm_fwo_selftest("mlp", _fwo, lambda xi: all_gather_swiglu_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis,
            out_memory_config=out_memory_config), x)
    _swpk = agmm_swiglu_pack_engaged("mlp")  # QWEN36_AGMM_SWIGLU_PACK (0049): the C++ op reads the env per call
    if _swpk is not None and agmm_swiglu_pack_needs_selftest("mlp"):
        agmm_swiglu_pack_selftest("mlp", _swpk, lambda xi: all_gather_swiglu_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis,
            out_memory_config=out_memory_config), x)
    _gbs = agmm_gbs()  # QWEN36_AGMM_GBS (0057): sharded gather buffer; the C++ op reads the env per call
    if _gbs is not None and agmm_gbs_needs_selftest("mlp"):
        agmm_gbs_selftest("mlp", _gbs, lambda xi: all_gather_swiglu_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis,
            out_memory_config=out_memory_config), x)
    if _nt is not None:  # same value as above (unset = 4); now with the per-core split banner (c134c)
        m_block = agmm_m_block("mlp", S, grid, force_transpose, env="QWEN36_AGMM_MLP_MBLOCK")
    # QWEN36_AGMM_BLOCKS (f-swiglu): the block split override (M8 x N16 = one weight pass per core) + its self-test.
    m_block, n_block = agmm_blocks_apply(
        "swiglu", m_block, n_block,
        call=lambda xi: all_gather_swiglu_prefill(
            xi, weight, tt_ccl, compute_cfg, topology, cluster_axis=cluster_axis,
            out_memory_config=out_memory_config),
        x=x, m_tiles_per_core=-(-(-(-int(S) // TILE_SIZE)) // (grid[0] if force_transpose else grid[1])))
    sub_h, sub_w = agmm_subblock("swiglu", m_block, n_block, max(d for d in (4, 2) if n_block % d == 0))
    cfg = ttnn.MinimalMatmulConfig(
        M_block_size=m_block,
        K_block_size=agmm_k_block_size(K_local),
        N_block_size=n_block,
        subblock_h=sub_h,
        subblock_w=sub_w,
        compute_with_storage_grid_size=ttnn.CoreCoord(grid[0], grid[1]),
    )
    _mf_x4 = None
    if _g9 is not None and agmm_grid9_mustfail():
        x4 = _mf_x4 = agmm_grid9_mustfail_swap(x4, "mlp")
    _persist = agmm_persistent_default(tt_ccl, x4, "swiglu")
    x4, _persist, _in0_owned = _agmm_in0_prepare(x4, "swiglu", tt_ccl, _persist)
    out = ttnn.experimental.all_gather_minimal_matmul_async(
        input_tensor=x4,
        weight_tensor=weight,
        persistent_output_buffer=_persist,
        config=cfg,
        compute_kernel_config=agmm_compute_cfg(compute_cfg),
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=num_links,
        topology=topology,
        cluster_axis=cluster_axis,
        memory_config=out_memory_config,
        dtype=prefill_mlp_lowp("hidden") or ttnn.bfloat16,
        force_transpose=force_transpose,
        num_workers_per_link=workers,
        num_buffers_per_channel=agmm_bufs(),
        fuse_swiglu=True,
    )[0]
    if _in0_owned:
        ttnn.deallocate(x4)
    if _mf_x4 is not None and not (_in0_owned and x4 is _mf_x4):
        ttnn.deallocate(_mf_x4)
    return out


_LOGGED_GATEUP_DECODE = False


def decode_gateup_fused_enabled(num_devices, grid_w=11):
    """QWEN36_DECODE_GATEUP_FUSED=1: decode gate+up as ONE matmul on the prefill AGMM's packed
    w_gate_up (already resident; no new tensor, no cache change) with SwiGLU fused in-kernel,
    instead of the w1 (silu) and w3 1D matmuls plus a mul. TP-only: w_gate_up exists only there."""
    global _LOGGED_GATEUP_DECODE
    on = _tt_public_getenv("QWEN36_DECODE_GATEUP_FUSED") == "1" and mlp_gateup_agmm_enabled(num_devices)
    if on and not _LOGGED_GATEUP_DECODE:
        _LOGGED_GATEUP_DECODE = True
        logger.info(
            f"QWEN36_DECODE_GATEUP_FUSED engaged: decode gate/up = one minimal_matmul(fuse_swiglu) on w_gate_up, "
            f"grid={gateup_swiglu_decode_grid(grid_w)} (w1/w3 decode matmuls and the mul skipped)"
        )
    return on


def gateup_swiglu_decode_grid(grid_w=11):
    """(cols, rows) for the decode gate/up minimal_matmul. minimal_matmul splits M over rows and N over
    cols, so at M=1 tile only the `cols` row-0 cores read the weight (rows>1 get mcast + zero-padded M);
    the op refuses a grid under 2x2. QWEN36_DECODE_GATEUP_GRID=CxR overrides (A/B instrument)."""
    val = _tt_public_getenv("QWEN36_DECODE_GATEUP_GRID")
    if val:
        cols, rows = (int(v) for v in val.lower().split("x"))
        return cols, rows
    return grid_w, 2


def create_gateup_swiglu_decode_config(k, n_packed, grid_w=11):
    """MinimalMatmulConfig for x[M<=32,k] @ w_gate_up[k,n_packed] with fuse_swiglu (output n_packed/2).
    fuse_swiglu partitions N on gate/up PAIRS, so N_block must be even; subblock_w=2 keeps a pair in DST."""
    cols, rows = gateup_swiglu_decode_grid(grid_w)
    pairs = n_packed // TILE_SIZE // 2
    cols = min(cols, pairs)
    n_per_core = 2 * math.ceil(pairs / cols)
    n_block = max(b for b in range(2, min(16, n_per_core) + 1, 2) if n_per_core % b == 0)
    return ttnn.MinimalMatmulConfig(
        M_block_size=1,
        K_block_size=_find_largest_divisor(math.ceil(k / TILE_SIZE)),
        N_block_size=n_block,
        subblock_h=1,
        subblock_w=2,
        compute_with_storage_grid_size=ttnn.CoreCoord(cols, rows),
    )


def gateup_swiglu_decode(x, w_gate_up, cfg, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG):
    """silu(x @ gate) * (x @ up) in ONE op on the tile-pair-interleaved w_gate_up (see
    all_gather_swiglu_prefill). Interleaves the K-sharded decode activation first, as matmul_1d_decode."""
    x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
    out = ttnn.experimental.minimal_matmul(
        x_il,
        w_gate_up,
        config=cfg,
        compute_kernel_config=compute_cfg,
        memory_config=out_memory_config,
        dtype=ttnn.bfloat16,
        fuse_swiglu=True,
    )
    if x_il is not x:
        ttnn.deallocate(x_il)
    return out


# ---------------------------------------------------------------- QWEN36_DECODE_DRAM_SHARDED_MR (D1)
# Decode (M<=32) projections as DRAM-WIDTH-sharded matmuls with num_workers_per_dram_bank > 1
# (tt-metal #54242, in our base; the 1x2-mesh reader placement needs #55922 = image vllm-tt:k2-mr,
# patches/tt-metal/0024). With readers > 1 the in1 reader issues one DMA per k-row of the bank shard
# instead of one 576 B bf4 tile per request -- the request-rate wall our 1D decode matmuls sit on.
# One card, M=32, bf4, TP=2 per-chip shapes ([source validation record]):
#   w1/w3 5120x8704 89.7 -> 64.7 us (3 readers, in0 40)    qkvzab 5120x8256 82.4 -> 59.6 us (3, 40)
#   w2    8704x5120 87.6 -> 72.9 us (2 readers, in0 8)     out    3072x5120 33.1 -> 29.2 us (2, 8)
# (standalone op times incl. the activation reshard, NOT an in-model number; attn_qkv was not probed).
#
# Weight layout per role -- no tensor-cache change: the interleaved weight is loaded exactly as today
# and re-laid out ON DEVICE (ttnn.to_memory_config interleaved -> DRAM width-sharded; lossless, uneven
# last shard allowed -- bench/probes/n1n2/n6_mr_relayout_equiv.py on ttsim):
#   relayout  the sharded tensor REPLACES the interleaved one (0 extra DRAM). Only where every other
#             consumer reads a sharded weight correctly: w1/w3 (prefill uses the packed w_gate_up) and
#             qkvzab / attn_qkv (prefill = all_gather_minimal_matmul_async; on ttsim it and
#             minimal_matmul are bit-identical on the sharded vs the interleaved weight).
#   copy      a decode-only sharded copy NEXT TO the interleaved weight. Forced for w2 and out: the 2D
#             mcast prefill matmul reads a DRAM width-sharded in1 WRONG with no error (ttsim PCC ~0.1
#             vs bit-identical interleaved, n6 probe), so their prefill keeps the interleaved tensor.
#             ESTIMATE +1.6 GiB/chip for w2 (bf4, 64 layers) and +0.57 GiB for attn wo + GDN out.
# Env (all optional besides the flag):
#   QWEN36_DECODE_MR_ROLES=w13,w2,qkvzab,attn_qkv,out   subset to convert (drop w2,out to save DRAM)
#   QWEN36_DECODE_MR_READERS=w13=3,w2=2,...             readers per DRAM bank, per role
#   QWEN36_DECODE_MR_IN0=w13=40,w2=8,...                activation (in0) cores; must divide K tiles
#   QWEN36_DECODE_MR_LAYOUT=qkvzab=copy,...             copy instead of relayout (relayout of w2/out refused)
# Fail closed: a bad config raises at boot; an op the device rejects raises at warmup/capture. There
# is no fallback to the 1D path once the flag is set.
MR_ROLES = ("w13", "w2", "qkvzab", "attn_qkv", "out")
_MR_DEFAULT = {"w13": (3, 40), "w2": (2, 8), "qkvzab": (3, 40), "attn_qkv": (2, 40), "out": (2, 8)}
_MR_LAYOUT_DEFAULT = {"w13": "relayout", "w2": "copy", "qkvzab": "relayout", "attn_qkv": "relayout", "out": "copy"}
_MR_RELAYOUT_FORBIDDEN = {"w2", "out"}  # their prefill 2D matmul misreads a sharded in1 (n6 probe)


def decode_mr_enabled():
    return _tt_public_getenv("QWEN36_DECODE_DRAM_SHARDED_MR") == "1"


def _mr_kv(env_name, cast):
    out = {}
    val = _tt_public_getenv(env_name, "").strip()
    for item in filter(None, (s.strip() for s in val.split(","))):
        if "=" not in item:
            raise ValueError(f"{env_name}={val!r}: expected role=value[,role=value], roles {MR_ROLES}")
        role, v = (s.strip() for s in item.split("=", 1))
        if role not in MR_ROLES:
            raise ValueError(f"{env_name}: unknown role {role!r} (roles {MR_ROLES})")
        out[role] = cast(v)
    return out


class DecodeMRRole:
    """One decode projection role: DRAM width-sharded weight memcfg + matmul progcfg + activation shard."""

    def __init__(self, role, k, n, readers, in0_cores, layout, grid, tag=None):
        k_tiles, n_tiles = k // TILE_SIZE, math.ceil(n / TILE_SIZE)
        tag = tag or f"QWEN36_DECODE_DRAM_SHARDED_MR {role}"
        # QWEN36_DECODE_MR_CPB (patches/tt-metal/0040, upstream #58922/#58923): 0 = today's MR program.
        self.cpb = mr_cpb_for(role, n_tiles)
        self.base_readers = readers  # today's MR readers for this role (the self-test's control program)
        if self.cpb:
            # #58923's Blackhole opt-in: one reader slot per bank (the weight shard = ceil(N/8) tiles, no MR
            # padding), in0_block_w = K tiles per streamed weight block = 2. The re-layout is on device at load
            # (decode_mr_weight), so no tensor-cache key moves.
            readers = 1
        if k % TILE_SIZE or k_tiles % in0_cores:
            raise ValueError(f"{tag}: in0 cores {in0_cores} must divide K={k_tiles} tiles")
        if not 1 <= readers <= 8:
            raise ValueError(f"{tag}: readers per bank {readers} out of range 1..8")
        if in0_cores > grid.x * grid.y:
            raise ValueError(f"{tag}: in0 cores {in0_cores} exceed the {grid.x}x{grid.y} worker grid")
        if layout not in ("relayout", "copy"):
            raise ValueError(f"{tag}: layout {layout!r} is not relayout|copy")
        if layout == "relayout" and role in _MR_RELAYOUT_FORBIDDEN:
            raise ValueError(
                f"{tag}: relayout refused -- the prefill 2D matmul reads a DRAM width-sharded weight "
                f"wrong without an error (bench/probes/n1n2/n6_mr_relayout_equiv.py); use copy"
            )
        self.role, self.k, self.n, self.readers, self.in0_cores, self.layout = role, k, n, readers, in0_cores, layout
        # Shard width: N rounded up to 8 banks x readers tiles (each reader owns shard/readers columns).
        self.shard_w = math.ceil(n_tiles / (DRAM_CORES * readers)) * readers * TILE_SIZE
        self.weight_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.DRAM,
            ttnn.ShardSpec(DRAM_GRID, [k, self.shard_w], ttnn.ShardOrientation.ROW_MAJOR),
        )
        self.act_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                ttnn.num_cores_to_corerangeset(in0_cores, grid, row_wise=True),
                [TILE_SIZE, (k_tiles // in0_cores) * TILE_SIZE],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        def pc(act):
            if self.cpb:
                if act is not None:
                    return None  # refused in decode_mr_matmul: the #58922 variant has no fused activation
                return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=_MR_CPB_IN0_BLOCK_W,
                    per_core_M=1,
                    per_core_N=math.ceil(self.shard_w * DRAM_CORES / TILE_SIZE / in0_cores),
                    fused_activation=None,
                    num_workers_per_dram_bank=1,
                    cores_per_bank=self.cpb,
                )
            return ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                in0_block_w=k_tiles // in0_cores,
                per_core_M=1,
                per_core_N=math.ceil(self.shard_w * DRAM_CORES / TILE_SIZE / in0_cores),
                fused_activation=act,
                num_workers_per_dram_bank=readers,
            )

        self.progcfg = pc(None)
        # MLP gate: SiLU fused in the packer, as the 1D path fuses it (w1 only).
        self.progcfg_silu = pc(ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU))

    def describe(self):
        cpb = f"/cpb {self.cpb}" if self.cpb else ""
        return f"{self.role}=K{self.k}xN{self.n}/r{self.readers}/in0 {self.in0_cores}/shard {self.shard_w}/{self.layout}{cpb}"


# ---------------------------------------------------------------- QWEN36_DECODE_MR_CPB (patches/tt-metal/0040)
# Upstream tt-metal #58922 (multi-core DRAM-sharded decode matmul, `cores_per_bank`) with #58923's Blackhole rule,
# lifted as patch 0040 (overlay bench/overlays/pk3r+0033+0034+0035+0040.json). Per role it replaces today's MR
# program (#55922 multi-reader, NOC0 only) with cores_per_bank cores next to each DRAM bank that split the bank's
# columns, stream the weight on BOTH data-movement RISCs (one per NoC), read the activation straight from its L1
# shards and keep the whole K in fp32 Dest. Third-party: -39% median per op vs main's SINGLE reader on a BH P300 die;
# vs our MR it is unmeasured (bench/probes/dsmc/hw1card.sh first).
#   QWEN36_DECODE_MR_CPB = unset | 0       today's MR programs, nothing changes (no banner)
#                        = auto            every MR role + the MR lm_head: clamp(ceil(Nbt / 8), 2, 4) cores per
#                                          bank (#58923's Blackhole rule), Nbt = ceil(N tiles / 8)
#                        = role=N[,...]    explicit per role (w13 w2 qkvzab attn_qkv out lm_head; 0 = off); may
#                                          follow `auto`, e.g. "auto,lm_head=0"
# With it a role's weight is re-laid out with ONE reader slot per bank (shard ceil(N/8) tiles, on device at load,
# no tensor-cache change), in0_block_w = 2, num_workers_per_dram_bank = 1. Refused: a fused SiLU (needs
# QWEN36_MR_SILU_UNFUSED=1), cores_per_bank > the per-bank shard width (K row groups are not lifted).
# Numerics move (whole K in fp32 Dest instead of in0_block_w partials): tier 2. Warmup self-test (fail closed, at
# model build, before any trace): per engaged role, random bf16 x [32, K] @ a random bf4 [K, N] weight at the served
# shape, both programs, error vs an fp64 reference of the same quantised operands; REFUSED unless
# err(cpb) <= 1.05 * err(MR) + 1e-4 on every device. TT_MM_DSMC_MUSTFAIL=1 (C++, drops the last K block) must be
# REFUSED. Banners: "QWEN36_DECODE_MR_CPB engaged: ..." and "QWEN36_DECODE_MR_CPB self-test pass ..." here, the C++
# "TT_MM_DSMC engaged cores_per_bank=" from the factory.
_MR_CPB_ROLES = MR_ROLES + ("lm_head",)
_MR_CPB_IN0_BLOCK_W = 2
_MR_CPB_STATE = {"parsed": None, "logged": set(), "tested": set()}


def _mr_cpb_table():
    """{"auto": bool, role: int} from QWEN36_DECODE_MR_CPB (parsed once)."""
    if _MR_CPB_STATE["parsed"] is not None:
        return _MR_CPB_STATE["parsed"]
    val = _tt_public_getenv("QWEN36_DECODE_MR_CPB", "").strip()
    out = {"auto": False}
    if val not in ("", "0"):
        for item in filter(None, (t.strip() for t in val.split(","))):
            if item == "auto":
                out["auto"] = True
                continue
            if "=" not in item:
                raise ValueError(f"QWEN36_DECODE_MR_CPB={val!r}: expected auto and/or role=N (roles {_MR_CPB_ROLES})")
            role, v = (t.strip() for t in item.split("=", 1))
            if role not in _MR_CPB_ROLES:
                raise ValueError(f"QWEN36_DECODE_MR_CPB: unknown role {role!r} (roles {_MR_CPB_ROLES})")
            n = int(v)
            if not 0 <= n <= 8:
                raise ValueError(f"QWEN36_DECODE_MR_CPB {role}={n}: expected 0..8")
            out[role] = n
    _MR_CPB_STATE["parsed"] = out
    return out


def mr_cpb_for(role, n_tiles):
    """cores_per_bank for this MR role (0 = today's MR program)."""
    t = _mr_cpb_table()
    nbt = math.ceil(n_tiles / DRAM_CORES)
    if role in t:
        n = t[role]
    elif t["auto"]:
        n = min(max(math.ceil(nbt / 8), 2), 4)  # #58923, Blackhole
    else:
        return 0
    if n and n > nbt:
        raise ValueError(f"QWEN36_DECODE_MR_CPB {role}={n} > the per-bank shard width {nbt} tiles "
                         f"(K row groups of #58922 are not lifted in 0040)")
    return n


def _mr_cpb_err(out_dev, ref):
    o = out_dev.double().reshape(ref.shape[0], -1)[:, : ref.shape[1]]
    return float((o - ref).norm() / (ref.norm() + 1e-30)), bool(torch.isfinite(o).all())


def mr_cpb_selftest(mesh_device, role_cfg, compute_cfg):
    """Fail-closed warmup self-test of one cpb role at its served shape (see the block comment). Raises -> boot refused."""
    k, n = role_cfg.k, role_cfg.n
    torch.manual_seed(1234)
    x = torch.randn(1, 1, TILE_SIZE, k, dtype=torch.float32).to(torch.bfloat16)
    w = torch.randn(1, 1, k, n, dtype=torch.float32) * (1.0 / math.sqrt(k))
    rep = ttnn.ReplicateTensorToMesh(mesh_device)
    w_int = ttnn.from_torch(w, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    # The quantised operands the device sees: bf16 x, the bf4 weight as read back from device 0.
    wq = ttnn.to_torch(ttnn.get_device_tensors(w_int)[0]).reshape(k, n)
    ref = x.double().reshape(TILE_SIZE, k) @ wq.double()
    x_dev = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG, mesh_mapper=rep)
    # Today's MR program for the same role (its own readers / in0_block_w / layout), built with the flag ignored.
    saved = _MR_CPB_STATE["parsed"]
    _MR_CPB_STATE["parsed"] = {"auto": False}
    try:
        base = DecodeMRRole(role_cfg.role, k, n, role_cfg.base_readers, role_cfg.in0_cores, "copy",
                            mesh_device.compute_with_storage_grid_size(), tag=f"QWEN36_DECODE_MR_CPB selftest {role_cfg.role}")
    finally:
        _MR_CPB_STATE["parsed"] = saved
    errs = {}
    for name, rc in (("mr", base), ("cpb", role_cfg)):
        w_ds = ttnn.to_memory_config(w_int, rc.weight_memcfg)
        x_sh = ttnn.to_memory_config(x_dev, rc.act_memcfg)
        o = ttnn.linear(x_sh, w_ds, compute_kernel_config=compute_cfg, program_config=rc.progcfg,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG, dtype=ttnn.bfloat16)
        o_il = ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)
        errs[name] = [_mr_cpb_err(ttnn.to_torch(d), ref) for d in ttnn.get_device_tensors(o_il)]
        for t in (o_il, o, x_sh, w_ds):
            ttnn.deallocate(t)
    ttnn.deallocate(w_int)
    ttnn.deallocate(x_dev)
    ok = all(fc and fm and ec <= 1.05 * em + 1e-4 for (em, fm), (ec, fc) in zip(errs["mr"], errs["cpb"]))
    desc = (f"{role_cfg.role} K{k}xN{n} cpb {role_cfg.cpb} vs MR r{base.readers}: rel err vs fp64 "
            f"cpb {[round(e, 6) for e, _ in errs['cpb']]} mr {[round(e, 6) for e, _ in errs['mr']]} per device")
    if not ok:
        raise RuntimeError(f"QWEN36_DECODE_MR_CPB self-test REFUSED: {desc}")
    logger.info(f"QWEN36_DECODE_MR_CPB self-test pass: {desc}")


# The served compute config the MR projections run with (QWEN36_PROJ_LOFI: LoFi, fp32 acc, packer L1 acc); the
# self-test uses it for every role but the lm_head, which passes its own (lmhead_compute_cfg).
_MR_CPB_TEST_CFG = dict(math_fidelity=ttnn.MathFidelity.LoFi, math_approx_mode=False, fp32_dest_acc_en=True,
                        packer_l1_acc=True)


def mr_cpb_engage(mesh_device, cfg, compute_cfg=None):
    """Banner + warmup self-test for the cpb roles of `cfg` ({role: DecodeMRRole}); a no-op when none is engaged."""
    roles = {r: c for r, c in cfg.items() if getattr(c, "cpb", 0)}
    if not roles:
        return
    if "w13" in roles and _tt_public_getenv("QWEN36_MR_SILU_UNFUSED", "0") != "1":
        raise RuntimeError("QWEN36_DECODE_MR_CPB w13 needs QWEN36_MR_SILU_UNFUSED=1 (the #58922 variant has no fused "
                           "SiLU); or exclude it with w13=0")
    desc = ", ".join(f"{r}=cpb {c.cpb}/shard {c.shard_w}/in0_block_w {_MR_CPB_IN0_BLOCK_W}" for r, c in roles.items())
    if desc not in _MR_CPB_STATE["logged"]:
        _MR_CPB_STATE["logged"].add(desc)
        mf = _tt_public_getenv("TT_MM_DSMC_MUSTFAIL", "").strip()
        logger.info(f"QWEN36_DECODE_MR_CPB engaged: {desc} (patch 0040, tt-metal #58922/#58923)"
                    + (" MUSTFAIL: TT_MM_DSMC_MUSTFAIL=1 drops the last K block" if mf == "1" else ""))
    ckc = compute_cfg or ttnn.WormholeComputeKernelConfig(**_MR_CPB_TEST_CFG)
    for r, c in roles.items():
        key = (r, c.k, c.n, c.cpb)
        if key in _MR_CPB_STATE["tested"]:
            continue
        mr_cpb_selftest(mesh_device, c, ckc)
        _MR_CPB_STATE["tested"].add(key)


def build_decode_mr(mesh_device, shapes):
    """{role: DecodeMRRole} for QWEN36_DECODE_DRAM_SHARDED_MR, or {} when the flag is off.
    shapes: {role: (K, N)} per chip. Logs the engaged banner (the per-role config it will run)."""
    if not decode_mr_enabled():
        return {}
    roles_env = _tt_public_getenv("QWEN36_DECODE_MR_ROLES", ",".join(MR_ROLES))
    roles = [r.strip() for r in roles_env.split(",") if r.strip()]
    bad = [r for r in roles if r not in MR_ROLES]
    if bad or not roles:
        raise ValueError(f"QWEN36_DECODE_MR_ROLES={roles_env!r}: unknown {bad} (roles {MR_ROLES})")
    readers = _mr_kv("QWEN36_DECODE_MR_READERS", int)
    in0 = _mr_kv("QWEN36_DECODE_MR_IN0", int)
    layout = _mr_kv("QWEN36_DECODE_MR_LAYOUT", str)
    grid = mesh_device.compute_with_storage_grid_size()
    cfg = {}
    for r in roles:
        k, n = shapes[r]
        dr, dc = _MR_DEFAULT[r]
        cfg[r] = DecodeMRRole(
            r, k, n, readers.get(r, dr), in0.get(r, dc), layout.get(r, _MR_LAYOUT_DEFAULT[r]), grid
        )
    logger.info("QWEN36_DECODE_DRAM_SHARDED_MR engaged: " + ", ".join(c.describe() for c in cfg.values()))
    mr_cpb_engage(mesh_device, cfg)
    return cfg


def decode_mr_for(args, mesh_device):
    """The process's {role: DecodeMRRole} table for these ModelArgs, built once and memoised on `args`
    ({} when QWEN36_DECODE_DRAM_SHARDED_MR is off). Built here rather than in model_config.py so the
    lever changes no file ttx keys tensor caches on (tools/ttx WEIGHT_FILE_RE)."""
    cfg = getattr(args, "_decode_mr", None)
    if cfg is not None:
        return cfg
    if not decode_mr_enabled():
        cfg = {}
    else:
        tp = getattr(args, "num_devices", 1)
        if tp < 2:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR is a TP (mesh) lever; num_devices=%d" % tp)
        cfg = build_decode_mr(
            mesh_device,
            {
                "w13": (args.dim, args.hidden_dim // tp),
                "w2": (args.hidden_dim // tp, args.dim),
                "qkvzab": (args.dim, args.gdn_qkvzab_dim_tp),
                "attn_qkv": (args.dim, args.attn_qkv_fused_dim_tp),
                "out": (args.attn_out_dim_tp, args.dim),
            },
        )
    args._decode_mr = cfg
    return cfg


def decode_mr_weight(w, mr):
    """Re-lay the interleaved weight `w` out as mr's DRAM width-sharded tensor. relayout: frees `w` and
    returns the sharded tensor for EVERY consumer; copy: keeps `w` (prefill) and returns the decode copy."""
    w_ds = ttnn.to_memory_config(w, mr.weight_memcfg)
    if mr.layout == "relayout":
        ttnn.deallocate(w)
    return w_ds


def decode_mr_matmul(x, w_ds, mr, compute_cfg, out_memory_config=ttnn.L1_MEMORY_CONFIG, silu=False):
    """Decode (M<=32) x @ w_ds on the multi-reader DRAM-sharded kernel. The activation is resharded to
    mr's in0 width shard; the width-sharded L1 result is returned in `out_memory_config`, so downstream
    code sees exactly what the 1D path handed it. No try/except: a rejected config must raise."""
    if silu and mr.cpb:
        raise RuntimeError(f"QWEN36_DECODE_MR_CPB {mr.role}: the #58922 variant has no fused SiLU; "
                           f"set QWEN36_MR_SILU_UNFUSED=1 or exclude the role ({mr.role}=0)")
    already = x.memory_config() == mr.act_memcfg
    x_sh = x if already else ttnn.to_memory_config(x, mr.act_memcfg)
    if isinstance(out_memory_config, MRDirect):  # QWEN36_DECODE_MR_OUT_DIRECT: write the AR input in place
        if silu:
            raise RuntimeError(f"QWEN36_DECODE_MR_OUT_DIRECT {mr.role}: a fused SiLU is not an all-reduce input")
        res = mr_out_direct_matmul(x_sh, w_ds, mr, compute_cfg, out_memory_config)
        if not already:
            ttnn.deallocate(x_sh)
        return res
    out = ttnn.linear(
        x_sh,
        w_ds,
        compute_kernel_config=compute_cfg,
        program_config=mr.progcfg_silu if silu else mr.progcfg,
        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
    )
    if not already:
        ttnn.deallocate(x_sh)
    if out_memory_config is None:  # QWEN36_DECODE_MR_MLP_SHARDED: the caller consumes the L1 width shard as is
        return out
    res = ttnn.to_memory_config(out, out_memory_config)
    ttnn.deallocate(out)
    return res


# ---------------------------------------------------------------- QWEN36_MTP_FC_BLOCK (docs/MTP-BATCH.md section 30)
# The MTP head's fc ([enorm(embed), hnorm(hidden)] [M, 2*dim] -> [M, dim/tp], bf8, COMPUTE_HIFI2) runs 4x per MTP step
# (3 draft depths + the reseed) as an interleaved-DRAM ttnn.linear with NO program config: ttnn's auto config for this
# all-DRAM-interleaved wide shape is the 1D in0-mcast program on the full grid, per_core_N 1 (80 cores at N 2560), and
# in0_block_w = 2 (matmul_program_config.cpp get_mcast_1d_config) -> 160 K blocks of 2 weight tiles per core, one in0
# mcast + one small DRAM read + a CB handshake each. Card: ~0.18-0.22 ms per call vs a 0.061 ms bf8 DRAM floor
# (s10-optrace-20261024T095517Z, DIAGNOSTIC, by difference). QWEN36_MTP_FC_BLOCK=<b> runs the decode-shaped calls
# (M <= 32) with the SAME program, grid and per-core split and in0_block_w = b (K/32 = 320 tiles: b divides 320).
# TIER 2 (not bitwise): a different in0_block_w re-associates the fp32 K sum at block boundaries (dest-accumulated block,
# then packer L1 accumulation), which flips the bf16 rounding of a rare output by 1 ulp: ttsim 0..1 of 81920 outputs
# per call at b 8..160 (sim_mtp_fc_blocking.py, sim_mtp_fc_block.py). Only b = 2 is the served order. The MR DRAM-sharded
# decode matmul is far off (~8% of outputs 1 ulp, sim_mtp_fc_mr.py) -> not built. The fc feeds only the drafter, so a
# flip can move a draft token (acceptance), never an emitted token (the verify decides; exact spec rule).
#   QWEN36_MTP_FC_BLOCK=<b>            engage (banner "QWEN36_MTP_FC_BLOCK engaged: ..."); unset/0 = the auto config
#   QWEN36_MTP_FC_BLOCK_MUSTFAIL=1     the wrong-input branch: the activation's last K tile zeroed before the matmul
#                                      -> the boot self-test must REFUSE.
# Boot self-test (fail closed, before any trace), per device: the served fc weight, a random bf16 [1,1,32,K]
# activation, the blocked program vs the served auto-config linear: at most MTP_FC_BLOCK_MAX_FLIP_FRAC of the outputs
# differ and each by at most 1 bf16 ulp (the re-association class, nothing else); and the auto-config linear on the
# wrong input must differ in at least 1% of the outputs (the compare is live). Banner "QWEN36_MTP_FC_BLOCK self-test
# pass".
_MTP_FC_BLOCK_STATE = {"logged": False, "tested": set()}
MTP_FC_BLOCK_MAX_FLIP_FRAC = 1e-4


def mtp_fc_block():
    v = _tt_public_getenv("QWEN36_MTP_FC_BLOCK", "0").strip() or "0"
    try:
        b = int(v)
    except ValueError:
        raise ValueError(f"QWEN36_MTP_FC_BLOCK={v!r}: expected an integer in0_block_w (0 = off)")
    if b < 0:
        raise ValueError(f"QWEN36_MTP_FC_BLOCK={b}: negative")
    return b


def mtp_fc_block_mustfail():
    return _tt_public_getenv("QWEN36_MTP_FC_BLOCK_MUSTFAIL", "0") == "1"


def mtp_fc_block_progcfg(mesh_device, k, n, block):
    """The auto config's 1D in0-mcast program (full grid, per_core_M 1, per_core_N ceil(N tiles / cores)) with
    in0_block_w = block. Valid for M <= 32 (one tile row)."""
    k_tiles = k // TILE_SIZE
    if k % TILE_SIZE or k_tiles % block:
        raise ValueError(f"QWEN36_MTP_FC_BLOCK={block}: must divide K={k_tiles} tiles")
    g = mesh_device.compute_with_storage_grid_size()
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=(g.x, g.y), in0_block_w=block, out_subblock_h=1, out_subblock_w=1,
        per_core_M=1, per_core_N=math.ceil(math.ceil(n / TILE_SIZE) / (g.x * g.y)), fuse_batch=False,
        fused_activation=None, mcast_in0=True)


def _mtp_fc_zero_last_ktile(x):
    """x [1,1,M,K] with its last TILE_SIZE K columns zeroed (the must-fail / wrong-input branch)."""
    s = list(x.shape)
    k = s[-1]
    head = ttnn.slice(x, [0, 0, 0, 0], [s[0], s[1], s[2], k - TILE_SIZE], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    tail = ttnn.slice(x, [0, 0, 0, k - TILE_SIZE], [s[0], s[1], s[2], k], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    zero = ttnn.multiply(tail, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.concat([head, zero], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    for t in (head, tail, zero):
        ttnn.deallocate(t)
    return out


def mtp_fc_block_matmul(x, w, progcfg, mustfail=None):
    """The MTP fc with the explicit blocked program: x [1,1,M<=32,K] DRAM -> [1,1,M,N] DRAM interleaved."""
    if mustfail is None:
        mustfail = mtp_fc_block_mustfail()
    xx = _mtp_fc_zero_last_ktile(x) if mustfail else x
    out = ttnn.linear(xx, w, compute_kernel_config=COMPUTE_HIFI2, program_config=progcfg,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if mustfail:
        ttnn.deallocate(xx)
    return out


def _dev_tensors(t):
    try:
        return ttnn.get_device_tensors(t)
    except Exception:  # noqa: BLE001 -- a single-device tensor
        return [t]


def _bf16_ulp(t):
    a = t.float().abs().clamp_min(2.0 ** -126)
    return torch.pow(2.0, torch.floor(torch.log2(a)) - 7)


def mtp_fc_block_compare(ref, got, bad):
    """The self-test's rule on host tensors (ref = served op, got = blocked op, bad = served op on the wrong input)."""
    n = ref.numel()
    d = ref != got
    ulp = _bf16_ulp(torch.maximum(ref.float().abs(), got.float().abs()))
    dist = ((ref.float() - got.float()).abs() / ulp)[d]
    max_ulp = float(dist.max()) if dist.numel() else 0.0
    ndiff, wrong = int(d.sum()), int((ref != bad).sum())
    return {"n": n, "ndiff": ndiff, "max_ulp": round(max_ulp, 3), "wrong_ndiff": wrong,
            "max_abs": float((ref.float() - got.float()).abs().max()),
            "ok": ndiff <= MTP_FC_BLOCK_MAX_FLIP_FRAC * n and max_ulp <= 1.0 and wrong >= 0.01 * n}


def mtp_fc_block_selftest(mesh_device, w, progcfg, k, n, block):
    """Fail-closed boot self-test (block comment above). Returns a summary dict; raises RuntimeError -> boot refused."""
    g = torch.Generator().manual_seed(4321)
    x = torch.randn(1, 1, TILE_SIZE, k, generator=g).to(torch.bfloat16)
    kw = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if hasattr(mesh_device, "get_num_devices"):
        kw["mesh_mapper"] = ttnn.ReplicateTensorToMesh(mesh_device)
    x_dev = ttnn.from_torch(x, **kw)
    ref = ttnn.linear(x_dev, w, compute_kernel_config=COMPUTE_HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    got = mtp_fc_block_matmul(x_dev, w, progcfg)  # honours QWEN36_MTP_FC_BLOCK_MUSTFAIL
    x_bad = _mtp_fc_zero_last_ktile(x_dev)
    bad = ttnn.linear(x_bad, w, compute_kernel_config=COMPUTE_HIFI2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    per = [mtp_fc_block_compare(ttnn.to_torch(r)[..., :TILE_SIZE, :n], ttnn.to_torch(o)[..., :TILE_SIZE, :n],
                                ttnn.to_torch(b)[..., :TILE_SIZE, :n])
           for r, o, b in zip(_dev_tensors(ref), _dev_tensors(got), _dev_tensors(bad))]
    for t in (ref, got, bad, x_bad, x_dev):
        ttnn.deallocate(t)
    ok = bool(per) and all(p["ok"] for p in per)
    desc = (f"K{k}xN{n} in0_block_w {block} vs auto (2) M{TILE_SIZE} tier 2: per device ndiff "
            f"{[p['ndiff'] for p in per]} of {per[0]['n'] if per else 0} (max {MTP_FC_BLOCK_MAX_FLIP_FRAC:g}) max ulp "
            f"{[p['max_ulp'] for p in per]} (max 1) max_abs {[round(p['max_abs'], 6) for p in per]} wrong-input ndiff "
            f"{[p['wrong_ndiff'] for p in per]}" + (" MUSTFAIL" if mtp_fc_block_mustfail() else ""))
    if not ok:
        raise RuntimeError(f"QWEN36_MTP_FC_BLOCK self-test REFUSED: {desc}")
    logger.info(f"QWEN36_MTP_FC_BLOCK self-test pass: {desc}")
    return {"ok": True, "per_device": per}


def mtp_fc_block_build(mesh_device, w, k, n):
    """The decode fc program config after the banner + boot self-test; None when QWEN36_MTP_FC_BLOCK is off."""
    block = mtp_fc_block()
    if not block:
        return None
    progcfg = mtp_fc_block_progcfg(mesh_device, k, n, block)
    if not _MTP_FC_BLOCK_STATE["logged"]:
        _MTP_FC_BLOCK_STATE["logged"] = True
        logger.info(f"QWEN36_MTP_FC_BLOCK engaged: in0_block_w {block} (auto 2), K{k}xN{n} bf8 HiFi2, decode calls "
                    f"M<=32 (draft depths + reseed)"
                    + (" MUSTFAIL: last K tile of the activation zeroed" if mtp_fc_block_mustfail() else ""))
    key = (k, n, block)
    if key not in _MTP_FC_BLOCK_STATE["tested"]:
        mtp_fc_block_selftest(mesh_device, w, progcfg, k, n, block)
        _MTP_FC_BLOCK_STATE["tested"].add(key)
    return progcfg


_AB_GAP_LOGGED = []


def gdn_ab_gap(nv_tp):
    """QWEN36_GDN_AB_PAD=1 (c82): zero columns inserted between a and b in the fused [qkv|z|a|b] in-projection, so b
    starts on a tile boundary. T2 decode profile (c80): with Nv=24 per chip, b starts mid-tile, and its slice runs
    untilize + stick slice + tilize (~7 us/layer, 48 layers). NOT bit-exact at bf4: the fused weight packs a shared
    exponent per 16 output columns, and a/b are RTN (not grid-snapped, QUANT.md), so moving b by the gap regroups
    a's and b's exponent blocks. Tier 2. Per chip the weight grows 8240 -> 8248 columns; b starts at 8224 (tile 257)."""
    if _tt_public_getenv("QWEN36_GDN_AB_PAD", "0") != "1":
        return 0
    gap = (-nv_tp) % TILE_SIZE
    if not _AB_GAP_LOGGED:
        _AB_GAP_LOGGED.append(1)
        logger.info(f"QWEN36_GDN_AB_PAD engaged: {gap} zero columns between a and b (Nv/chip={nv_tp}); b starts tile-aligned")
    return gap


_MR_MLP_SHARDED_LOGGED = []


def mr_mlp_sharded():
    """QWEN36_DECODE_MR_MLP_SHARDED=1 (c81): the decode MLP keeps the MR gate/up outputs L1 width-sharded, runs the
    gate*up mul on that shard, and hands it to the w2 matmul, whose activation reshard then goes sharded -> sharded.
    T2 profile (c80): each MR output is copied to interleaved L1 (~1.5 us, writer_unary_sharded_blocks) and w2
    re-shards it (~1.9 us). This removes the two gate/up copies per layer. Data movement only: bit-exact expected."""
    on = _tt_public_getenv("QWEN36_DECODE_MR_MLP_SHARDED", "0") == "1"
    if on and not _MR_MLP_SHARDED_LOGGED:
        _MR_MLP_SHARDED_LOGGED.append(1)
        logger.info("QWEN36_DECODE_MR_MLP_SHARDED engaged: decode MLP gate/up stay L1 width-sharded into mul and w2")
    return on


_MR_SILU_UNFUSED_LOGGED = []


def mr_silu_unfused():
    """QWEN36_MR_SILU_UNFUSED=1 (c84): the MR decode gate matmul (w1) runs WITHOUT the packer-fused SiLU and the
    SiLU moves into the gate*up mul as an input_tensor_a activation (upstream llama32 mlp_1d does the same).
    Workbench profiles (c77..c82pre2, B=8 4k, 5 sessions): w1 76.6 us vs w3 61.4 us per layer for the same
    [5120, 8704] bf4 bytes and progcfg; SiLU is the only difference, ~15 us x 64 layers. The mul is 3.8 us.
    SiLU then runs on the bf16-packed gate instead of the fp32 dest: tier 2 (rounding moves)."""
    on = _tt_public_getenv("QWEN36_MR_SILU_UNFUSED", "0") == "1"
    if on and not _MR_SILU_UNFUSED_LOGGED:
        _MR_SILU_UNFUSED_LOGGED.append(1)
        logger.info("QWEN36_MR_SILU_UNFUSED engaged: decode MR w1 without fused SiLU; SiLU runs as the mul's input_a activation")
    return on


# ---------------------------------------------------------------- QWEN36_DECODE_MR_OUT_DIRECT (kx-matmul A1)
# The row-parallel decode projections that feed the one-shot all-reduce (GDN out-proj, attention wo, MLP w2) end
# today in THREE programs: the MR matmul (its output width shard sits on ceil(N/per_core_N) row-major cores), a
# sharded -> L1/DRAM-interleaved copy (decode_mr_matmul's out_memory_config), then all_reduce_async_decode's
# to_memory_config into the AR input layout (the decoder-norm shard, CoreGrid 8x4 = 32 cores x [32,160]) with the
# bf8 payload typecast. The MR factory already writes its result back over the NoC to whatever storage cores the
# OUTPUT tensor's shard spec names (in1 writer's write-back loop; per_core_N in the program config is the storage
# width), so handing it a preallocated output tensor in the AR layout (optional_output_tensor, per_core_N = 160/32 = 5
# tiles) makes the matmul write the AR input directly and both copies disappear (rz3 wb profile, B=8 4k, chip 0:
# out copy 1.47-1.85 us + AR reshard 1.23-1.72 us per call, 128 calls per step). No C++ change.
#   QWEN36_DECODE_MR_OUT_DIRECT = 0 | unset   today's path, nothing changes (no banner)
#                               = 1 | out     the out role (GDN out-proj + attention wo, 64 calls/step) packs the AR
#                                             payload dtype (bf8 under QWEN36_AR_BF8) straight from its fp32
#                                             accumulator: both copies go
#                               = out,w2      the same for the MLP down-proj too (128 calls/step)
# NOT bit-exact vs today, which rounds to bf16 first and re-packs bf8: tier 2, fp64 gate. One card
# ([source validation record], M=1..32): the out chain's relative Frobenius error vs
# fp64 is 3-4% LOWER than today's in every M; w2 alone (the self-test) is lower too, but the whole MLP chain
# (w1/w3 LoFi -> silu*up -> w2) ends 11-12% HIGHER (0.0122 vs 0.0109) in every M, so w2 is opt-in, not the default.
# (A dtype-preserving variant, the MR landing bf16 in the AR layout + a same-grid typecast, is not bit-exact either
# (ttsim, M=1: the typecast rounds differently from the reshard's dtype conversion) and keeps one of the two
# programs, so it is not offered.)
# Requires QWEN36_AR_ASYNC (the persistent workspace + AR layout exist only on the replicated one-shot seam);
# refused otherwise. Roles: out (GDN + attention), w2. Warmup self-test (eager, before any trace, once per role on
# its first call): today's path vs this one on the same device tensors; relative Frobenius error vs an fp64 reference of the
# device operands <= 1.05 x today's + 1e-4. Every self-test also runs a wrong-input branch (x with row 0 negated
# through the candidate) and REFUSES the boot if that branch passes. QWEN36_DECODE_MR_OUT_DIRECT_MUSTFAIL=1 packs
# bfloat4_b instead of the payload dtype (bf8): the self-test must REFUSE.
# Banners: "QWEN36_DECODE_MR_OUT_DIRECT engaged: ..." and "QWEN36_DECODE_MR_OUT_DIRECT self-test pass ...".
_MR_OUT_DIRECT_STATE = {"logged": False, "tested": set(), "progcfg": {}}


class MRDirect:
    """out_memory_config marker for decode_mr_matmul: land the result in `memcfg` (the AR input layout), dtype
    `dtype` (None = the matmul's own output dtype = in0's)."""

    def __init__(self, memcfg, dtype, mode):
        self.memcfg, self.dtype, self.mode = memcfg, dtype, mode


def mr_out_direct_mode():
    v = _tt_public_getenv("QWEN36_DECODE_MR_OUT_DIRECT", "0").strip()
    if v in ("", "0"):
        return None
    roles = ("out",) if v in ("1", "bf8") else tuple(r.strip() for r in v.split(",") if r.strip())
    if not roles or any(r not in ("out", "w2") for r in roles):
        raise ValueError(f"QWEN36_DECODE_MR_OUT_DIRECT={v!r}: expected 0|1|out|out,w2|w2")
    return roles


def mr_ar_target(tt_ccl, fallback, role="out"):
    """The out_memory_config a row-parallel decode MR call (role out | w2) should pass: an MRDirect into the
    all_reduce_async input layout when QWEN36_DECODE_MR_OUT_DIRECT covers the role, else `fallback` (today's)."""
    roles = mr_out_direct_mode()
    if roles is None or role not in roles:
        return fallback
    mode = "bf8"
    cfg = getattr(tt_ccl, "_qwen_decode_ar_memcfg", None)
    if cfg is None or getattr(tt_ccl, "_qwen_ar_workspace", None) is None:
        raise RuntimeError("QWEN36_DECODE_MR_OUT_DIRECT needs QWEN36_AR_ASYNC=1 (+ QWEN36_ONESHOT_AR=1): the AR input "
                           "layout and workspace exist only on the replicated one-shot seam")
    pay = getattr(tt_ccl, "_qwen_ar_dtype", ttnn.bfloat16)
    if not _MR_OUT_DIRECT_STATE["logged"]:
        _MR_OUT_DIRECT_STATE["logged"] = True
        sh = cfg.shard_spec
        logger.info(f"QWEN36_DECODE_MR_OUT_DIRECT engaged: roles={','.join(roles)}, MR writes the AR input directly "
                    f"({sh.grid.num_cores()} cores x {list(sh.shape)}, dtype {pay}); out copy + AR reshard removed")
    return MRDirect(cfg, pay, mode)


def _mr_direct_progcfg(mr, per_core_n):
    key = (mr.role, mr.k, mr.n, mr.readers, mr.in0_cores, per_core_n)
    pc = _MR_OUT_DIRECT_STATE["progcfg"].get(key)
    if pc is None:
        if mr.cpb:
            raise RuntimeError(f"QWEN36_DECODE_MR_OUT_DIRECT {mr.role}: not validated with QWEN36_DECODE_MR_CPB")
        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
            in0_block_w=(mr.k // TILE_SIZE) // mr.in0_cores,
            per_core_M=1,
            per_core_N=per_core_n,
            fused_activation=None,
            num_workers_per_dram_bank=mr.readers,
        )
        _MR_OUT_DIRECT_STATE["progcfg"][key] = pc
    return pc


def _mr_direct_call(x_sh, w_ds, mr, compute_cfg, d, dtype):
    """One MR matmul whose output tensor is preallocated in d.memcfg (the MR factory writes back to its storage
    cores in shard order; per_core_N = the shard width)."""
    sh = d.memcfg.shard_spec
    n_tiles = math.ceil(mr.n / TILE_SIZE)
    pcn = sh.shape[1] // TILE_SIZE
    if sh.shape[0] != TILE_SIZE or pcn * sh.grid.num_cores() != n_tiles:
        raise RuntimeError(f"QWEN36_DECODE_MR_OUT_DIRECT {mr.role}: AR layout {sh.grid.num_cores()} x {list(sh.shape)} "
                           f"does not tile N={mr.n}")
    shape = list(x_sh.shape)[:-1] + [mr.n]
    y = ttnn.empty(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=x_sh.device(), memory_config=d.memcfg)
    return ttnn.linear(
        x_sh,
        w_ds,
        compute_kernel_config=compute_cfg,
        program_config=_mr_direct_progcfg(mr, pcn),
        memory_config=d.memcfg,
        dtype=dtype,
        optional_output_tensor=y,
    )


def _mr_direct_reference(x_sh, w_ds, mr, compute_cfg, d):
    """Today's path for the same operands: MR into its own shard, interleaved copy, AR to_memory_config."""
    out = ttnn.linear(x_sh, w_ds, compute_kernel_config=compute_cfg, program_config=mr.progcfg,
                      memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG)
    il = ttnn.to_memory_config(out, ttnn.L1_MEMORY_CONFIG)
    ttnn.deallocate(out)
    res = ttnn.to_memory_config(il, d.memcfg, d.dtype)
    ttnn.deallocate(il)
    return res


def _dev_torch(t):
    """Device tensor -> per-device float64 torch tensors (a mesh tensor yields one per device)."""
    try:
        parts = ttnn.get_device_tensors(t)
    except Exception:
        parts = [t]
    return [ttnn.to_torch(p).double() for p in parts]


def _mr_direct_selftest(x_sh, w_ds, mr, compute_cfg, d, cand_dtype):
    """Warmup self-test of one engaged role at this M (eager; fail closed). Inputs are the live device operands."""
    ref_t = _mr_direct_reference(x_sh, w_ds, mr, compute_cfg, d)
    cand_t = _mr_direct_call(x_sh, w_ds, mr, compute_cfg, d, cand_dtype)
    ref = _dev_torch(ref_t)
    cand = _dev_torch(cand_t)
    ttnn.deallocate(ref_t)
    ttnn.deallocate(cand_t)
    x_il = ttnn.to_memory_config(x_sh, ttnn.L1_MEMORY_CONFIG)
    xs = _dev_torch(x_il)
    ttnn.deallocate(x_il)
    ws = _dev_torch(w_ds)
    # wrong-input branch: the candidate's row 0 recomputed from x with row 0 negated must be REFUSED by the same rule
    wrong = []
    for xi in xs:
        xn = xi.clone()
        xn[..., 0, :] = -xn[..., 0, :]
        wrong.append(xn)
    rows = x_sh.shape[-2]
    verdict = {"role": mr.role, "mode": d.mode, "M": rows, "devices": len(ref)}
    ok = True
    errs = []
    for i, (r, c, xi, wi) in enumerate(zip(ref, cand, xs, ws)):
        r2, c2 = r.reshape(-1, r.shape[-1])[:rows], c.reshape(-1, c.shape[-1])[:rows]
        exact = xi.reshape(-1, xi.shape[-1])[:rows] @ wi.reshape(-1, wi.shape[-1])
        scale = exact.norm().item() + 1e-30  # relative Frobenius error: a max-abs metric flips on one element
        e_ref = (r2 - exact).norm().item() / scale
        e_cand = (c2 - exact).norm().item() / scale
        bitwise = bool(torch.equal(r2, c2))
        dev_ok = e_cand <= 1.05 * e_ref + 1e-4
        # wrong-input branch: the candidate on x with row 0 negated vs today's output must fail the same rule
        xn = wrong[i].reshape(-1, xi.shape[-1])[:rows]
        wrong_out = c2.clone()
        wrong_out[0] = (xn @ wi.reshape(-1, wi.shape[-1]))[0].to(c2.dtype)
        e_wrong = (wrong_out - exact).norm().item() / scale
        wrong_pass = e_wrong <= 1.05 * e_ref + 1e-4
        if wrong_pass:
            raise RuntimeError(f"QWEN36_DECODE_MR_OUT_DIRECT self-test {mr.role}: the wrong-input branch PASSED "
                               f"(the test cannot fail); refusing")
        errs.append({"dev": i, "bitwise": bitwise, "err_today": e_ref, "err_direct": e_cand, "err_wrong": e_wrong})
        ok = ok and dev_ok
    verdict["per_device"] = errs
    if not ok:
        raise RuntimeError(f"QWEN36_DECODE_MR_OUT_DIRECT self-test REFUSED: {verdict}")
    logger.info(f"QWEN36_DECODE_MR_OUT_DIRECT self-test pass {mr.role} mode={d.mode} M={rows}: "
                + ", ".join(f"dev{e['dev']} bitwise={e['bitwise']} err {e['err_direct']:.3e} vs today "
                            f"{e['err_today']:.3e}" for e in errs))
    return verdict


def mr_out_direct_matmul(x_sh, w_ds, mr, compute_cfg, d):
    """The QWEN36_DECODE_MR_OUT_DIRECT matmul (see the block comment). x_sh is already in mr.act_memcfg."""
    mustfail = _tt_public_getenv("QWEN36_DECODE_MR_OUT_DIRECT_MUSTFAIL") == "1"
    dtype = d.dtype
    if mustfail:
        dtype = ttnn.bfloat4_b
        if "mustfail" not in _MR_OUT_DIRECT_STATE["tested"]:
            _MR_OUT_DIRECT_STATE["tested"].add("mustfail")
            logger.warning("QWEN36_DECODE_MR_OUT_DIRECT_MUSTFAIL engaged: the direct MR packs bfloat4_b; the self-test must REFUSE")
    # Once per role, on its first (eager, pre-capture) call: host reads cannot run inside a trace capture. The M
    # dependence (rows 1..32) is gated in ttsim and on one card (bench/probes/mr_seam).
    key = mr.role
    if key not in _MR_OUT_DIRECT_STATE["tested"]:
        _mr_direct_selftest(x_sh, w_ds, mr, compute_cfg, d, dtype)
        _MR_OUT_DIRECT_STATE["tested"].add(key)
    return _mr_direct_call(x_sh, w_ds, mr, compute_cfg, d, dtype)


# ---------------------------------------------------------------- QWEN36_DECODE_MR_LMHEAD (D2)
# The decode LM head on the same multi-reader DRAM-sharded kernel as D1, split into chunks the QB2 way
# (#57914 runs its lm_head as 8 DRAM-sharded matmuls). Facts it is built on (model.py): the lm_head is
# ONE bf8 (bfloat8_b) TILE tensor [dim=5120, vocab/TP] per chip, DRAM-interleaved, vocab-sharded over
# the mesh (ShardTensorToMesh dim=-1, cache output.weight.vshard); 248320/2 = 124160 = 3880 tiles per
# chip = 675 MB of bf8 read per decode step (the largest single weight a step reads). Decode calls it
# through Qwen36Model._lm_head_matmul (ttnn-auto: HiFi2, no fp32 acc, packer L1 acc) and hands the
# vocab-SHARDED [1,1,B,124160] bf16 TILE logits to _greedy_argmax_device (QWEN36_DEVICE_ARGMAX) or to
# _lm_head's gather (QWEN36_LM_HEAD_GATHER_MODE); this lever returns exactly that tensor (DRAM, TILE,
# bf16, same logical shape) so both consumers run unchanged.
#
# Why chunks: the DRAM-sharded kernel's in1 CB is reader_width x in0_block_w tiles, TRIPLE-buffered, on
# each of the 8 x readers worker cores. The whole per-chip head at 2 readers is 243 tiles per reader ->
# 243 x 4 x 3 x 1088 B = 3.2 MB of in1 CB per core; BH L1 is 1.5 MB. 8 chunks of 485 tiles at 2
# readers -> 31 tiles per reader -> ~0.5 MB. A reader's k-row is then 31 x 1088 B = 33.7 KB > the 16 KB
# NoC burst: exactly the #55922 k-row burst bound (patches/tt-metal/0024), so image vllm-tt:k2-mr is
# REQUIRED (without it readers > 1 TT_FATALs on the mesh first, so a wrong image fails closed at warmup).
#
# Layout: always a decode-only COPY. Prefill keeps reading the interleaved weight (_lm_head on x_last,
# PREFILL_LOGITS_FAST, the bucket path's direct ttnn.linear), and ttsim showed the prefill-side 2D
# matmul reads a DRAM width-sharded in1 wrong without an error (n6 probe), so the interleaved tensor
# stays. Cost: chunks x (8 banks x shard_w) x 160 k-tiles x 1088 B = +0.65 GiB/chip at the defaults
# (the banner prints the figure for the config). Chunks are sliced from the interleaved weight ON
# DEVICE (no tensor-cache change, no host re-quantisation).
# Env: QWEN36_DECODE_MR_LMHEAD=1 (needs QWEN36_DECODE_DRAM_SHARDED_MR=1, TP>=2, vocab-sharded head)
#      QWEN36_DECODE_MR_LMHEAD_CHUNKS=8   must divide the per-chip vocab tiles (3880 = 2^3 x 5 x 97)
#      QWEN36_DECODE_MR_LMHEAD_READERS=2  readers per DRAM bank (QB2 plan D2: 2)
#      QWEN36_DECODE_MR_LMHEAD_IN0=40     activation cores; must divide K tiles (160)
# Defaults are placeholders until bench/probes/n1n2/n5_dram_sharded_multireader.py --shapes lm_head
# picks them on a card. Fail closed: bad config / estimated L1 overflow / missing prerequisite raise at
# boot; a config the device rejects raises at warmup. No fallback to the auto lm_head once set.
_LMHEAD_DEFAULT = {"chunks": 8, "readers": 2, "in0": 40}
_BF8_TILE_BYTES = 1088  # bfloat8_b 32x32 tile (1024 mantissas + 64 exponents), already 64 B aligned
_BF16_TILE_BYTES = 2048
_BF4_TILE_BYTES = 576  # bfloat4_b 32x32 tile (512 B mantissas + 64 exponents)
# Budget for the per-core CB estimate below: BH L1 is 1.5 MB, minus firmware, l1_small (24 KB) and the
# decode activations that live in L1. The device is the real judge; this refuses the obvious overflow.
_LMHEAD_L1_BUDGET = 1_200_000


def decode_mr_lmhead_enabled():
    return _tt_public_getenv("QWEN36_DECODE_MR_LMHEAD") == "1"


def _env_int(name, default):
    val = _tt_public_getenv(name, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        raise ValueError(f"{name}={val!r}: expected an integer") from None


class DecodeMRLMHead:
    """Chunked multi-reader DRAM-sharded decode LM head: one DecodeMRRole shared by `chunks` equal
    column chunks of the per-chip [k, n] head (the same memcfg / progcfg serves every chunk)."""

    TAG = "QWEN36_DECODE_MR_LMHEAD"

    def __init__(self, k, n, chunks, readers, in0_cores, grid):
        tag = self.TAG
        if n % TILE_SIZE:
            raise ValueError(f"{tag}: per-chip vocab {n} is not tile aligned")
        n_tiles = n // TILE_SIZE
        if chunks < 1 or n_tiles % chunks:
            raise ValueError(
                f"{tag}: chunks={chunks} must divide the per-chip vocab {n_tiles} tiles "
                f"(for 3880: 4, 5, 8, 10, 20, 40, ...)"
            )
        self.k, self.n, self.chunks = k, n, chunks
        self.chunk_n = n // chunks
        self.role = DecodeMRRole("lm_head", k, self.chunk_n, readers, in0_cores, "copy", grid, tag=tag)
        # role.readers, not the env's: QWEN36_DECODE_MR_CPB forces one reader slot per bank (the L1 estimate below
        # models the MR program's CBs, not the cores_per_bank one's).
        self.readers, self.in0_cores = self.role.readers, in0_cores
        k_tiles = k // TILE_SIZE
        self.reader_w = self.role.shard_w // TILE_SIZE // self.readers  # tiles per reader per k-row
        self.in0_block_w = self.role.progcfg.in0_block_w
        # The head's dtype follows QWEN36_LMHEAD_DTYPE (model.py); the CB/copy sizing uses its tile bytes.
        self.w_dtype = _tt_public_getenv("QWEN36_LMHEAD_DTYPE", "bf8").strip() or "bf8"
        self.w_tile_bytes = _BF4_TILE_BYTES if self.w_dtype == "bf4" else _BF8_TILE_BYTES
        self.k_row_bytes = self.reader_w * self.w_tile_bytes
        # Per-core L1 estimate from the DRAM-sharded factory's CB sizing: in1 triple-buffered, in0
        # double-buffered, out + packer-L1-acc interm (subblock padding bounded by 8 tiles), the
        # width-sharded output storage and the in0 shard (a worker may also be a storage core).
        out_w = self.reader_w + 8
        self.l1_est = (
            3 * self.reader_w * self.in0_block_w * self.w_tile_bytes
            + 2 * self.in0_block_w * _BF16_TILE_BYTES
            + 2 * out_w * _BF16_TILE_BYTES
            + self.role.progcfg.per_core_N * _BF16_TILE_BYTES
            + (k_tiles // in0_cores) * _BF16_TILE_BYTES
        )
        if self.role.cpb:
            # 0040's DFBs instead (factory create_program_dram_sharded_multicore_spec): x = the whole K row in bf16,
            # two weight streams of depth 3 x in0_block_w x columns-per-core, the owned output columns; plus the in0
            # and output shards a work core may also store.
            cols = math.ceil(self.role.shard_w // TILE_SIZE / self.role.cpb)
            self.l1_est = (
                k_tiles * _BF16_TILE_BYTES
                + 2 * 3 * self.in0_block_w * cols * self.w_tile_bytes
                + cols * _BF16_TILE_BYTES
                + self.role.progcfg.per_core_N * _BF16_TILE_BYTES
                + (k_tiles // in0_cores) * _BF16_TILE_BYTES
            )
        if self.l1_est > _LMHEAD_L1_BUDGET:
            raise ValueError(
                f"{tag}: estimated {self.l1_est / 1e6:.2f} MB of CBs per core (> {_LMHEAD_L1_BUDGET / 1e6:.1f} MB): "
                f"{chunks} chunk(s) of {self.chunk_n} at {readers} readers = {self.reader_w} tiles per reader "
                f"x in0_block_w {self.in0_block_w} x 3 buffers; use more chunks, readers or in0 cores"
            )
        # The decode-only copy: every chunk padded to 8 banks x shard_w.
        self.copy_bytes = chunks * DRAM_CORES * (self.role.shard_w // TILE_SIZE) * k_tiles * self.w_tile_bytes
        # ttnn-auto for a bf16 x bf8 linear with no program config (matmul_device_operation.cpp
        # create_matmul_attributes): HiFi2, no fp32 dest acc, packer L1 acc. Passing a program config
        # without this would drop the fidelity to LoFi. QWEN36_LMHEAD_FID (lmhead_compute_cfg) overrides the
        # fidelity per call; unset = this config, unchanged.
        self.compute_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def describe(self):
        return (
            f"K{self.k}xN{self.n} {self.w_dtype} as {self.chunks} chunks of N{self.chunk_n} "
            f"({self.chunk_n // TILE_SIZE} tiles), r{self.readers}/in0 {self.in0_cores}/shard {self.role.shard_w}/"
            f"in0_block_w {self.in0_block_w}/k-row {self.k_row_bytes} B/L1 est {self.l1_est // 1024} KiB, "
            f"decode copy +{self.copy_bytes / 2**20:.0f} MiB/chip, HiFi2 l1acc"
        )


# ---------------------------------------------------------------- QWEN36_LMHEAD_FID (decode MR lm_head fidelity)
# The decode lm_head MR matmul (K5120 x N124160 bf4 per chip, 4 chunks) runs HiFi2, copied from ttnn-auto's bf8
# default (DecodeMRLMHead.compute_cfg). Its ~1.46 ms per decode/verify step is close to compute (HiFi2: 26,880 tile
# matmuls x 32 cycles ~ 0.64 ms) plus the DRAM read (357.6 MB ~ 0.79 ms) in series (docs/OP-DATASHEETS.md D1').
# The weight is bf4, which is full precision in fidelity phase 0 (LoFi); the second HiFi2 phase only adds the
# activation's (srcB) low mantissa bits (docs/BLACKHOLE-PERF.md section 4). Numerics move: tier 2.
#   QWEN36_LMHEAD_FID = unset | hifi2  the shipped config, returned unchanged (no banner)
#                     = lofi           LoFi, same fp32_dest_acc_en=False / packer_l1_acc=True
#                     = hifi3 | hifi4  more phases; hifi4 is the timing must-fail rung (must be SLOWER than hifi2,
#                                      or the knob does not reach the op)
# Read per call (the workbench screens it with `wb set` and a fresh trace capture); a traced decode bakes the value
# it was captured with. Applies to every decode_mr_lmhead_matmul call: plain decode, the MTP verify and the MTP draft
# head when it shares the MR head.
_LMHEAD_FID_LOGGED = set()


def lmhead_compute_cfg(default_cfg):
    v = _tt_public_getenv("QWEN36_LMHEAD_FID", "").strip().lower()
    if v in ("", "hifi2"):
        return default_cfg
    fids = {"lofi": ttnn.MathFidelity.LoFi, "hifi3": ttnn.MathFidelity.HiFi3, "hifi4": ttnn.MathFidelity.HiFi4}
    if v not in fids:
        raise ValueError(f"QWEN36_LMHEAD_FID={v!r}: expected unset, lofi, hifi2, hifi3 or hifi4")
    if v not in _LMHEAD_FID_LOGGED:
        _LMHEAD_FID_LOGGED.add(v)
        note = " (MUSTFAIL timing rung: must be slower than hifi2)" if v == "hifi4" else ""
        logger.info(f"QWEN36_LMHEAD_FID engaged: decode MR lm_head at {v} (fp32_dest_acc_en=False, packer_l1_acc=True; "
                    f"default hifi2){note}")
    return ttnn.WormholeComputeKernelConfig(
        math_fidelity=fids[v],
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )


def build_decode_mr_lmhead(mesh_device, k, n):
    """DecodeMRLMHead for QWEN36_DECODE_MR_LMHEAD, or None when the flag is off. k, n: per-chip head."""
    if not decode_mr_lmhead_enabled():
        return None
    cfg = DecodeMRLMHead(
        k,
        n,
        _env_int("QWEN36_DECODE_MR_LMHEAD_CHUNKS", _LMHEAD_DEFAULT["chunks"]),
        _env_int("QWEN36_DECODE_MR_LMHEAD_READERS", _LMHEAD_DEFAULT["readers"]),
        _env_int("QWEN36_DECODE_MR_LMHEAD_IN0", _LMHEAD_DEFAULT["in0"]),
        mesh_device.compute_with_storage_grid_size(),
    )
    logger.info(f"QWEN36_DECODE_MR_LMHEAD engaged: {cfg.describe()}")
    mr_cpb_engage(mesh_device, {"lm_head": cfg.role}, compute_cfg=lmhead_compute_cfg(cfg.compute_cfg))
    return cfg


def decode_mr_lmhead_for(args, mesh_device, vocab_sharded):
    """The model's DecodeMRLMHead (None when QWEN36_DECODE_MR_LMHEAD is unset), after the prerequisite
    checks. Raises when the flag is set but cannot engage -- never a silent fallback."""
    if not decode_mr_lmhead_enabled():
        return None
    tag = DecodeMRLMHead.TAG
    if not decode_mr_enabled():
        raise RuntimeError(f"{tag}=1 needs QWEN36_DECODE_DRAM_SHARDED_MR=1 (and image vllm-tt:k2-mr)")
    tp = getattr(args, "num_devices", 1)
    if tp < 2 or not vocab_sharded:
        raise RuntimeError(f"{tag} needs the vocab-sharded TP lm_head (num_devices={tp}, vocab_sharded={vocab_sharded})")
    if _tt_public_getenv("QWEN36_1D_GRID_LM_HEAD"):
        raise RuntimeError(f"{tag} and QWEN36_1D_GRID_LM_HEAD both claim the decode lm_head; unset one")
    return build_decode_mr_lmhead(mesh_device, args.dim, args.vocab_size // tp)


def decode_mr_lmhead_weights(w, cfg):
    """Slice the interleaved [.., k, n] lm_head into cfg.chunks column chunks and re-lay each out as a
    DRAM width-sharded tensor. `w` is NOT freed (prefill reads it). Transient: one interleaved chunk."""
    if cfg.chunks == 1:
        # A full-range ttnn.slice hands back the input's buffer: deallocating it freed the prefill weight
        # (caught by the n5 lm_head smoke on ttsim). One chunk = relayout the whole tensor, keep `w`.
        return [ttnn.to_memory_config(w, cfg.role.weight_memcfg)]
    shape = list(w.shape)
    lead = [0] * (len(shape) - 1)
    out = []
    for c in range(cfg.chunks):
        s = ttnn.slice(w, lead + [c * cfg.chunk_n], shape[:-1] + [(c + 1) * cfg.chunk_n])
        out.append(ttnn.to_memory_config(s, cfg.role.weight_memcfg))
        ttnn.deallocate(s)
    return out


def decode_mr_lmhead_matmul(x, chunks_w, cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Decode (M<=32) logits = x @ lm_head as cfg.chunks multi-reader DRAM-sharded matmuls on ONE
    resharded activation, each result moved to DRAM and concatenated along vocab: the same
    [.., M, n] bf16 TILE tensor the auto path returns. No try/except: a rejected config must raise."""
    role = cfg.role
    already = x.memory_config() == role.act_memcfg
    x_sh = x if already else ttnn.to_memory_config(x, role.act_memcfg)
    parts = []
    ckc = lmhead_compute_cfg(cfg.compute_cfg)
    for w in chunks_w:
        o = ttnn.linear(
            x_sh,
            w,
            compute_kernel_config=ckc,
            program_config=role.progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        parts.append(ttnn.to_memory_config(o, out_memory_config))
        ttnn.deallocate(o)
    if not already:
        ttnn.deallocate(x_sh)
    if len(parts) == 1:
        return parts[0]
    out = ttnn.concat(parts, dim=-1, memory_config=out_memory_config)
    for p in parts:
        ttnn.deallocate(p)
    return out


def build_mmrs_decode_state(mesh_device, M, K_local, N, nd, dtype=ttnn.bfloat16):
    """Build (progcfg, intermediate_buffer, output_buffer) for a decode matmul_reduce_scatter out-proj.

    M = LOGICAL decode batch (max_batch_size) — the op returns the persistent buffer with its logical
    shape, so an oversized (tile-padded) M leaks into the residual stream. TILE layout pads M<32.
    dtype MUST match the out-proj input activation (bf16 for MLP/attn; FLOAT32 for GDN, which keeps
    fp32 for stability) — the op's default output dtype is the input's, and writing it into a
    mismatched buffer corrupts the output. Matmul on reduced grid (8,6); RS workers at offset (0,6).
    interm [1,1,M,N], out [1,1,M,N/nd]."""
    cg = (8, 6)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / cg[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=cg,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // cg[0])),
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / cg[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(cg),
    )
    mk = lambda w: ttnn.from_torch(
        torch.zeros(1, 1, M, w),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=dtype,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    return pc, mk(N), mk(N // nd)


def matmul_reduce_scatter_decode(
    x, weight, tt_ccl, interm_buf, out_buf, progcfg, compute_cfg, topology, rs_offset=(0, 6)
):
    """Fused row-parallel matmul + reduce-scatter(dim=3) for decode (matmul_reduce_scatter_async).

    x: K-sharded [.,M,K_local]; weight: [K_local,N] K-sharded. Matmul runs on progcfg's (reduced)
    grid; RS workers land at rs_offset (disjoint rows) to avoid the collision that deadlocks a
    full-grid fused CCL. Persistent buffers are caller-owned. Returns [.,M,N/nd] (fractured, DRAM)."""
    _, rs_out = ttnn.experimental.matmul_reduce_scatter_async(
        x,
        weight,
        persistent_intermediate_buffer=interm_buf,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=1,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=progcfg,
        compute_kernel_config=compute_cfg,
    )
    # rs_out IS the persistent output buffer; clone so the caller can deallocate its copy while the
    # persistent buffer survives for the next token (else layer.py's deallocate frees it -> corruption).
    return ttnn.clone(rs_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)


_LOGGED_ONESHOT_AR = False


def oneshot_ar_enabled(mesh_device):
    """QWEN36_ONESHOT_AR=1 (1x2 mesh only): decode keeps the residual REPLICATED. Each out-proj
    all-reduce becomes one all-gather of the two partials + a local add, and the decoder norms
    skip their pre-norm all-gather: 2 CCLs per layer instead of 4 (RS + AG, twice)."""
    global _LOGGED_ONESHOT_AR
    on = _tt_public_getenv("QWEN36_ONESHOT_AR") == "1" and tuple(mesh_device.shape) == (1, 2)
    if on and not _LOGGED_ONESHOT_AR:
        _LOGGED_ONESHOT_AR = True
        logger.info(
            f"QWEN36_ONESHOT_AR engaged: decode residual replicated on mesh {tuple(mesh_device.shape)}; "
            f"out-proj all-reduce = all_gather + local add, decoder norms skip their all-gather"
        )
    return on


def all_reduce_oneshot_decode(partial, tt_ccl, topology, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """All-reduce of a [1,1,B,N] partial on a 1x2 mesh as ONE CCL: all-gather the two partials on
    dim 0 ([2,1,B,N], every device holds both) then sum them locally. Output is replicated [1,1,B,N]
    (what reduce-scatter + the next norm's all-gather produced in two CCLs).
    QWEN36_AR_ASYNC routes to all_reduce_async_decode instead; QWEN36_RESID_L1 returns the result in the
    L1 residual layout (configure_decode_residual) instead of memory_config."""
    if getattr(tt_ccl, "_qwen_ar_workspace", None) is not None:
        return all_reduce_async_decode(partial, tt_ccl, topology, memory_config)
    if partial.is_sharded():
        partial = ttnn.sharded_to_interleaved(partial, ttnn.L1_MEMORY_CONFIG)
    shape = ttnn.Shape(list(partial.shape))
    links, chunks, workers = decode_ag_cfg(tt_ccl)
    gathered = ttnn.experimental.all_gather_async(
        partial,
        persistent_output_buffer=None,
        dim=0,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=links,
        topology=topology,
        memory_config=memory_config,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=chunks,
        num_workers_per_link=workers,
        num_buffers_per_channel=2,
    )
    ttnn.deallocate(partial)
    out = ttnn.experimental.fast_reduce_nc(gathered, dims=[0], output=None, memory_config=memory_config)
    ttnn.deallocate(gathered)
    if out.shape != shape:  # fast_reduce_nc returns the tile-padded rows (B=8 -> 32): restore B as a view
        out = ttnn.reshape(out, shape, out.padded_shape)
    rcfg = resid_l1_memcfg(tt_ccl)
    if rcfg is not None:
        out_l1 = ttnn.to_memory_config(out, rcfg)
        ttnn.deallocate(out)
        out = out_l1
    return out


def norm_replicated_decode(dist_norm, x, norm_config):
    """DistributedNorm.forward's DECODE path minus its pre-norm all-gather, for a residual that is
    already replicated at full width (QWEN36_ONESHOT_AR). Same input shard config, same norm call."""
    x = ttnn.to_memory_config(x, norm_config.get("sharded_output_config"))
    return dist_norm.norm(x, mode="decode", in_sharded=True, out_sharded=True, norm_config=norm_config)


_LOGGED_DECODE_LEVERS = set()


def _banner_once(key, msg):
    if key not in _LOGGED_DECODE_LEVERS:
        _LOGGED_DECODE_LEVERS.add(key)
        logger.info(msg)


def _ccl_cfg_env(name):
    """'links,chunks,workers' (the #56119 QWEN36_RS_CFG / QWEN36_NORM_AG_CFG format) -> int tuple, or None."""
    v = _tt_public_getenv(name, "").strip()
    if not v:
        return None
    links, chunks, workers = (int(t) for t in v.split(","))
    return links, chunks, workers


def decode_rs_kwargs(x):
    """QWEN36_RS_CFG=links,chunks,workers (#56119; QB2 value 2,1,1): tt_all_reduce kwargs for a DECODE
    (<= 32 rows) reduce-scatter. workers=1 drops the per-link mux core, chunks_per_sync=1 syncs every chunk
    (the payload is a few tiles). Unset, or a prefill-sized x: {} -> the image defaults (auto, 10, 2)."""
    cfg = _ccl_cfg_env("QWEN36_RS_CFG")
    if cfg is None or x.shape[-2] > TILE_SIZE:
        return {}
    _banner_once(
        "rs",
        f"QWEN36_RS_CFG engaged: decode reduce-scatter num_links={cfg[0]} chunks_per_sync={cfg[1]} "
        f"num_workers_per_link={cfg[2]}",
    )
    return {"num_reduce_scatter_links": cfg[0], "chunks_per_sync": cfg[1], "num_workers_per_link": cfg[2]}


def _prefill_rs_cfg():
    """QWEN36_PREFILL_RS_CFG=links,chunks,workers,buffers (T7, c69 probe bench/probes/prefill_rs) or None."""
    v = _tt_public_getenv("QWEN36_PREFILL_RS_CFG", "").strip()
    if not v:
        return None
    links, chunks, workers, buffers = (int(t) for t in v.split(","))
    return links, chunks, workers, buffers


def prefill_all_reduce(partial, mesh, tt_ccl, topology, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Row-parallel prefill all-reduce (attn wo, GDN out, MLP down). Unset QWEN36_PREFILL_RS_CFG, a decode-
    sized partial, or a mesh without a unit axis: exactly tt_all_reduce(cluster_axis=0, dim=3). Set on 1xN:
    the same single reduce_scatter_minimal_async that tt_all_reduce issues there (ccl.py 1-in-mesh branch,
    DRAM intermediate) with the RS knobs tt_all_reduce does not expose (num_buffers_per_channel is fixed
    at 2 there, against 8 on the AGMMs). One fp add per element either way, so the output is bit-identical
    to the default config (the c69 probe asserts it per config)."""
    from models.tt_transformers.tt.ccl import tt_all_reduce

    cfg = _prefill_rs_cfg()
    if cfg is None or partial.shape[-2] <= TILE_SIZE or 1 not in list(mesh.shape):
        return tt_all_reduce(partial, mesh, tt_ccl, cluster_axis=0, dim=3, topology=topology, memory_config=memory_config)
    links, chunks, workers, buffers = cfg
    _banner_once(
        "prefill_rs",
        f"QWEN36_PREFILL_RS_CFG engaged: prefill reduce-scatter num_links={links} chunks_per_sync={chunks} "
        f"num_workers_per_link={workers} num_buffers_per_channel={buffers}",
    )
    s = partial.shape
    x = partial if (s[0] == 1 and s[1] == 1) else ttnn.reshape(partial, (1, 1, s[-4] * s[-3] * s[-2], s[-1]))
    out = ttnn.experimental.reduce_scatter_minimal_async(
        x,
        persistent_output_buffers=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=links,
        memory_config=memory_config,
        intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        chunks_per_sync=chunks,
        num_workers_per_link=workers,
        num_buffers_per_channel=buffers,
    )
    x.deallocate(True)
    return out


def decode_ag_cfg(tt_ccl):
    """(num_links, chunks_per_sync, num_workers_per_link) for the DECODE all-gathers we issue (the norm
    pre-gather via norm_gathered_decode, the one-shot all-reduce's gather). QWEN36_NORM_AG_CFG (#56119; QB2
    value 2,4,1) overrides; unset = the image DistributedNorm's effective decode values (get_num_links(1), 10, 2):
    its model_config branch compares Mode to "decode" and is dead, so those are what production runs."""
    cfg = _ccl_cfg_env("QWEN36_NORM_AG_CFG")
    if cfg is None:
        return tt_ccl.get_num_links(1), 10, 2
    _banner_once(
        "ag",
        f"QWEN36_NORM_AG_CFG engaged: decode all-gather num_links={cfg[0]} chunks_per_sync={cfg[1]} "
        f"num_workers_per_link={cfg[2]}",
    )
    return cfg


def norm_ag_cfg_enabled():
    return _ccl_cfg_env("QWEN36_NORM_AG_CFG") is not None


def norm_gathered_decode(dist_norm, x, norm_config, tt_ccl, topology):
    """DistributedNorm.forward's DECODE path (pre-norm all-gather straight into the norm's sharded input
    layout, then the sharded norm) with decode_ag_cfg's CCL knobs. Same semaphore calls in the same order,
    so the traced semaphore cycle is unchanged."""
    links, chunks, workers = decode_ag_cfg(tt_ccl)
    x = ttnn.experimental.all_gather_async(
        x,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=links,
        topology=topology,
        memory_config=norm_config.get("sharded_output_config"),
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        chunks_per_sync=chunks,
        num_workers_per_link=workers,
        num_buffers_per_channel=2,
    )
    return dist_norm.norm(x, mode="decode", in_sharded=True, out_sharded=True, norm_config=norm_config)


def ar_async_enabled():
    """QWEN36_AR_ASYNC=1 (QB2 #57914 direct_allreduce; needs QWEN36_ONESHOT_AR): each decode out-proj
    all-reduce is ONE ttnn.experimental.all_reduce_async (the minimal fabric kernel: every device sends its
    width-sharded partial into a persistent L1 workspace on its peers and reduces locally) instead of the
    one-shot all-gather + fast_reduce_nc pair."""
    return _tt_public_getenv("QWEN36_AR_ASYNC") == "1"


def ar_bf8_out8():
    """QWEN36_AR_BF8=out8 (CB-fit probe): bf8 payload + workspace AND bf8 op output
    (no `dtype` extra); the result is typecast to bf16 after the op. Tests whether
    the run-#8059 CB-overflow TT_FATAL comes from the mixed-dtype `dtype`=bf16 path."""
    return _tt_public_getenv("QWEN36_AR_BF8") == "out8"

def ar_bf8_enabled():
    """QWEN36_AR_BF8=1 (c101; needs QWEN36_AR_ASYNC): the all_reduce_async payload and workspace are bfloat8_b
    (1088 B/tile vs 2048) and the reduction packs its output as bfloat16 (the op's `dtype`), so the residual stays
    bf16 and no typecast op is added. The payload is a full 32-row tile at every B; the c101 ar_links probe put
    ~6.4 us of the ~12.2 us op on the wire at 2 links (1 link: 18.6 us), so halving its bytes targets that part.
    NOT bit-exact: each partial is rounded to a shared exponent per 16 values before the sum. Tier 2."""
    return _tt_public_getenv("QWEN36_AR_BF8") in ("1", "out8")


def resid_l1_enabled():
    """QWEN36_RESID_L1=1 (QB2 #57914 carry_residual; needs QWEN36_ONESHOT_AR): the replicated decode residual
    stays WIDTH-sharded in L1 in the decoder norms' input layout, so every residual add runs sharded and the
    norms' input reshard is a no-op (no DRAM round trip of the residual per layer)."""
    return _tt_public_getenv("QWEN36_RESID_L1") == "1"


def configure_decode_residual(args, tt_ccl, mesh_device):
    """Once at model init, before any warmup or trace capture (Qwen36Model.__init__, replicated seam only).
    Records the residual layout on tt_ccl and allocates the all_reduce_async workspace: ONE persistent
    buffer shared by every layer (ops run in order), so its address is fixed for every captured trace.
    Layout = the decode norm input shard config (WIDTH_SHARDED L1; 32 cores x 160 at dim 5120)."""
    from models.tt_transformers.tt.common import Mode

    cfg = args.get_norm_config("attn", Mode.DECODE)["sharded_output_config"]
    if not (ar_async_enabled() or resid_l1_enabled()):
        return
    assert cfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED and cfg.buffer_type == ttnn.BufferType.L1, cfg
    tt_ccl._qwen_decode_ar_memcfg = cfg
    shard = cfg.shard_spec
    nd = mesh_device.get_num_devices()
    if ar_async_enabled():
        ws_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(shard.grid, [shard.shape[0], shard.shape[1] * nd], ttnn.ShardOrientation.ROW_MAJOR),
        )
        tt_ccl._qwen_ar_dtype = ttnn.bfloat8_b if ar_bf8_enabled() else ttnn.bfloat16
        tt_ccl._qwen_ar_workspace = ttnn.empty(
            [1, 1, shard.shape[0], args.dim * nd],
            dtype=tt_ccl._qwen_ar_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ws_cfg,
        )
        tt_ccl._qwen_ar_links = int(_tt_public_getenv("QWEN36_AR_ASYNC_LINKS", "0")) or tt_ccl.get_num_links(1)
        logger.info(
            f"QWEN36_AR_ASYNC engaged: decode out-proj all-reduce = all_reduce_async, persistent L1 workspace "
            f"{list(tt_ccl._qwen_ar_workspace.shape)} on {shard.grid.num_cores()} cores, "
            f"num_links={tt_ccl._qwen_ar_links}, {'bf8 payload, bf8 out + typecast' if ar_bf8_out8() else ('bf8 payload, bf16 out' if ar_bf8_enabled() else 'bf16')}"
        )
        if ar_bf8_enabled():
            logger.info("QWEN36_AR_BF8 engaged: all_reduce_async payload + workspace bfloat8_b, " +
                        ("output bfloat8_b + host typecast bf16" if ar_bf8_out8() else "output bfloat16"))
    if resid_l1_enabled():
        tt_ccl._qwen_decode_resid_memcfg = cfg
        logger.info(
            f"QWEN36_RESID_L1 engaged: decode residual WIDTH-sharded in L1, {shard.grid.num_cores()} cores x "
            f"{list(shard.shape)} (the decoder norm input layout)"
        )


def resid_l1_memcfg(tt_ccl):
    return getattr(tt_ccl, "_qwen_decode_resid_memcfg", None)


def all_reduce_async_decode(partial, tt_ccl, topology, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """[1,1,B,dim] partial -> replicated sum in ONE all_reduce_async (see ar_async_enabled). The partial is
    resharded onto the workspace grid in bf16 (QB2 ccl_dtype; the GDN out-proj partial is fp32, as in the
    default path it reaches the residual add in bf16 anyway). Output: the L1 residual layout under
    QWEN36_RESID_L1, else memory_config (the one-shot path's DRAM default)."""
    cfg = tt_ccl._qwen_decode_ar_memcfg
    pay = getattr(tt_ccl, "_qwen_ar_dtype", ttnn.bfloat16)
    if partial.memory_config() == cfg and partial.dtype == pay:
        x = partial
    else:
        x = ttnn.to_memory_config(partial, cfg, pay)
        ttnn.deallocate(partial)
    out8 = ar_bf8_out8()
    extra = {} if out8 else ({"dtype": ttnn.bfloat16} if pay != ttnn.bfloat16 else {})  # QWEN36_AR_BF8: reduce packs bf16
    out = ttnn.experimental.all_reduce_async(
        x,
        tt_ccl._qwen_ar_workspace,
        cluster_axis=1,
        mesh_device=tt_ccl.mesh_device,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(1),
        topology=topology,
        num_links=tt_ccl._qwen_ar_links,
        memory_config=cfg,
        **extra,
    )
    ttnn.deallocate(x)
    if out8:
        cast = ttnn.typecast(out, dtype=ttnn.bfloat16)
        ttnn.deallocate(out)
        out = cast
    if resid_l1_memcfg(tt_ccl) is not None:
        return out
    out_mc = ttnn.to_memory_config(out, memory_config)
    ttnn.deallocate(out)
    return out_mc


def _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype):
    """Lazily allocate (and cache on tt_ccl) shared persistent buffers for the prefill fused out-proj.

    Prefill M (=chunk seq, e.g. 2048) makes per-layer buffers huge (fp32 [1,1,2048,5120]≈42MB × 64
    layers = infeasible). Prefill runs layers sequentially and each op's output is cloned before the
    next layer reuses the buffer, so ONE shared set per (M,N,nd,dtype) is safe. Allocated during the
    pre-capture warmup forward (eager), reused inside the trace. Keyed so variable M/dtype coexist."""
    cache = getattr(tt_ccl, "_qwen36_mmrs_prefill_bufs", None)
    if cache is None:
        cache = {}
        tt_ccl._qwen36_mmrs_prefill_bufs = cache
    key = (M, N, nd, str(dtype))
    if key not in cache:
        mesh = tt_ccl.mesh_device
        mk = lambda w: ttnn.from_torch(
            torch.zeros(1, 1, M, w),
            device=mesh,
            layout=ttnn.TILE_LAYOUT,
            dtype=dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        cache[key] = (mk(N), mk(N // nd))
    return cache[key]


def _mmrs_prefill_placement(mesh_device, num_links, want_grid=(8, 8)):
    """Topology-aware (matmul grid, RS core offset) for the fused prefill MMRS.

    ccl choose_worker_cores (ccl_common.cpp) selects the first num_links * 2 * (1 + workers_per_dir)
    worker cores row-major from (0,0) and then SHIFTS them by reduce_scatter_core_grid_offset; the
    shifted set must stay on the worker grid AND off the matmul rows (a collision deadlocks the
    fused CCL). workers_per_dir is data-size dependent (reduce_scatter_default_workers: up to 8 on
    Linear topologies, which a 1x2 mesh uses, for our ~40MB fp32 GDN-out intermediates), so a
    2-link call selects 2*2*(1+8)=36 cores. On a p150a 1x2 worker grid the old fixed (8,8)/(0,8)
    pushes 14 of those 36 cores off the grid (rows 10-11 don't exist). Instead, reserve the bottom
    ceil(36/W) rows for the RS workers and cap the matmul grid above them, so the two footprints
    are always disjoint and in-grid on any worker-grid shape.
    """
    try:
        gs = mesh_device.compute_with_storage_grid_size()
        W, H = int(gs.x), int(gs.y)
    except Exception:
        return want_grid, (0, want_grid[1])
    n_rs = num_links * 2 * 9  # worst case: 8 workers + 1 mux core per direction, 2 directions/link
    rs_rows = math.ceil(n_rs / W)
    gx = min(want_grid[0], W)
    gy = want_grid[1] if want_grid[1] + rs_rows <= H else max(1, H - rs_rows)
    return (gx, gy), (0, gy)


def matmul_reduce_scatter_prefill(x, weight, tt_ccl, compute_cfg, topology, nd, dtype, grid=None, rs_offset=None):
    """Fused row-parallel out-proj matmul + reduce-scatter for PREFILL (matmul_reduce_scatter_async).

    Unlike decode (M=1, where the 2D matmul collapses to ~8 cores and this loses), at prefill M>>1 the
    2D matmul fills the grid, so overlapping the RS with the matmul is a WIN (biggest for the fp32
    GDN-out with its large RS). Matmul rows 0..gy-1; RS workers the bottom ceil(36/W) rows
    (_mmrs_prefill_placement derives both from the device grid). x: K-sharded
    [.,M,K_local]; weight [K_local,N]. Returns [1,1,M,N/nd] (cloned; shared buffer survives)."""
    M, K_local = x.shape[-2], x.shape[-1]
    N = weight.shape[-1]
    interm, out_buf = _mmrs_prefill_shared_bufs(tt_ccl, M, N, nd, dtype)
    x4 = ttnn.reshape(x, (1, 1, M, K_local))
    # RS-bound: 2 ethernet links parallelize the fp32 cross-device reduce (traced_8k win).
    num_links = 2
    if grid is None or rs_offset is None:
        grid, rs_offset = _mmrs_prefill_placement(tt_ccl.mesh_device, num_links)
    per_core_N = max(1, math.ceil(N / TILE_SIZE / grid[0]))
    pc = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=grid,
        in0_block_w=min(4, max(1, K_local // TILE_SIZE // grid[0])),
        out_subblock_h=1,
        # Keep 1x1: op242 is RS-bound and this op is pipelined to overlap the matmul with the RS.
        # Widening the subblock desyncs that overlap and measured net-negative on traced_8k TTFT.
        out_subblock_w=1,
        per_core_M=max(1, math.ceil(M / TILE_SIZE / grid[1])),
        per_core_N=per_core_N,
        out_block_w=max(1, per_core_N // 2),
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
        allowed_worker_cores=_full_grid_crs(grid),
    )
    _, rs = ttnn.experimental.matmul_reduce_scatter_async(
        x4,
        weight,
        persistent_intermediate_buffer=interm,
        persistent_output_buffer=out_buf,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
        reduce_scatter_core_grid_offset=rs_offset,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
        num_links=num_links,
        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
        topology=topology,
        subdevice_id=None,
        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
        program_config=pc,
        compute_kernel_config=compute_cfg,
    )
    return ttnn.clone(rs, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def sharded_decode_matmul(
    x,
    weight,
    compute_cfg,
    decode_progcfg,
    act_shard_cfg,
    prefill_progcfg_fn,
    prefill_k,
    decode_out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    """DRAM-WIDTH_SHARDED weight matmul; branches on M (decode vs prefill).

    Decode (M<=32): L1-sharded act + DRAM-sharded kernel. Prefill: 2D matmul.
    Gate on x.shape[-2] (seq/M), not x.shape[1] (Z=1 in both modes). Decode result placement is
    `decode_out_memory_config` (default DRAM-interleaved; pass L1 to keep the small decode
    activation resident). Prefill result is always DRAM-interleaved."""
    seq = x.shape[-2]
    if seq <= TILE_SIZE:
        # Reshard act to L1 if needed; skip dealloc when x already sharded (GDN reuses x).
        already_sharded = x.memory_config() == act_shard_cfg
        x_sh = x if already_sharded else ttnn.to_memory_config(x, act_shard_cfg)
        out = ttnn.linear(
            x_sh,
            weight,
            compute_kernel_config=compute_cfg,
            program_config=decode_progcfg,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )
        if not already_sharded:
            ttnn.deallocate(x_sh)
        return ttnn.to_memory_config(out, decode_out_memory_config)
    pc = prefill_progcfg_fn(seq, prefill_k, weight.shape[-1])
    return ttnn.linear(
        x, weight, compute_kernel_config=compute_cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )


def replicate(torch_tensor, mesh, cache_path, dtype=ttnn.bfloat16):
    """Small tensor (norm/bias) -> replicated on every device."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def shard_small(torch_tensor, mesh, cache_path, dim=-1, dtype=ttnn.bfloat16):
    """Small per-head tensor (conv taps, A_log, dt_bias) -> sharded."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.bfloat16),
        dtype=dtype,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def replicate_kv_weight(weight, n_kv_heads, tp, head_dim):
    """Replicate KV weight so each device gets >=1 head. No-op when tp <= n_kv_heads."""
    if tp <= n_kv_heads:
        return weight
    chunks = weight.reshape(n_kv_heads, head_dim, -1)
    parts = []
    for d in range(tp):
        kv_idx = (d * n_kv_heads) // tp
        parts.append(chunks[kv_idx])
    return torch.cat(parts, dim=0).reshape(tp * head_dim, -1)


# FP8 dequantization
def dequant_fp8_block(weight_fp8, scale_inv, block_size=128):
    """Dequantize a block-wise FP8 weight tensor to bfloat16."""
    out_f, in_f = weight_fp8.shape
    weight_bf16 = weight_fp8.to(torch.bfloat16).reshape(out_f // block_size, block_size, in_f // block_size, block_size)
    weight_bf16 = weight_bf16 * scale_inv[:, None, :, None].to(torch.bfloat16)
    return weight_bf16.reshape(out_f, in_f)


# Weight-prep (reorder HF weights for per-device sharding)
def prepare_attn_qkv(q_w, k_w, v_w, qg_per, kv_per, tp):
    """Fuse attn q+gate/k/v for column-parallel shard: each device gets [qg_d|k_d|v_d].

    q_w: [n_heads*head_dim*2, in]; k_w/v_w: [n_kv_heads*head_dim, in].
    qg_per/kv_per: per-device out block sizes."""
    parts = []
    for d in range(tp):
        parts.append(q_w[d * qg_per : (d + 1) * qg_per, :])
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
    return torch.cat(parts, dim=0)


def prepare_attn_qkv_deint(q_w, k_w, v_w, nh_local, hd, kv_per, tp):
    """Like prepare_attn_qkv but de-interleaves [q,g] per head -> [all_q|all_gate|k|v] per device.

    Avoids prefill relayout in _make_heads (column perm only; numerically identical).
    q_w: [nh_total*hd*2, in]; nh_local/kv_per: per-device block sizes."""
    hd2 = hd * 2
    parts = []
    for d in range(tp):
        base = d * nh_local * hd2
        q_rows = [q_w[base + h * hd2 : base + h * hd2 + hd, :] for h in range(nh_local)]
        g_rows = [q_w[base + h * hd2 + hd : base + h * hd2 + hd2, :] for h in range(nh_local)]
        # Per-device layout [all_q | k | v | all_gate]: q/k/v contiguous so _make_heads* can hand
        # the fused q|k|v block straight to nlp_create_qkv_heads (no re-concat); gate trails, applied
        # post-SDPA. (Column perm only; numerically identical to [q|gate|k|v].)
        parts.append(torch.cat(q_rows, dim=0))  # all_q
        parts.append(k_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(v_w[d * kv_per : (d + 1) * kv_per, :])
        parts.append(torch.cat(g_rows, dim=0))  # all_gate (last)
    return torch.cat(parts, dim=0)


def prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp):
    """Interleave GDN Q/K/V heads for row-parallel shard (contiguous q/k/v block per device).

    qkv_w: [key_dim*2 + value_dim, hidden]."""
    q_part = qkv_w[:key_dim, :]
    k_part = qkv_w[key_dim : 2 * key_dim, :]
    v_part = qkv_w[2 * key_dim :, :]

    q_per = nk // tp
    v_per = nv // tp
    shards = []
    for s in range(tp):
        q_s = q_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        k_s = k_part[s * q_per * dk : (s + 1) * q_per * dk, :]
        v_s = v_part[s * v_per * dv : (s + 1) * v_per * dv, :]
        shards.append(torch.cat([q_s, k_s, v_s], dim=0))
    return torch.cat(shards, dim=0)


def prepare_conv_taps(conv_w, key_dim, nk, dk, nv, dv, kernel_size, tp):
    """Split fused conv1d into kernel taps, reordered to match prepare_gdn_qkv grouping."""
    cw = conv_w.float()
    q_per = nk // tp
    v_per = nv // tp
    taps = []
    for j in range(kernel_size):
        tap = cw[:, 0, j]
        q_tap = tap[:key_dim]
        k_tap = tap[key_dim : 2 * key_dim]
        v_tap = tap[2 * key_dim :]
        shards = []
        for s in range(tp):
            q_s = q_tap[s * q_per * dk : (s + 1) * q_per * dk]
            k_s = k_tap[s * q_per * dk : (s + 1) * q_per * dk]
            v_s = v_tap[s * v_per * dv : (s + 1) * v_per * dv]
            shards.append(torch.cat([q_s, k_s, v_s]))
        taps.append(torch.cat(shards))
    return taps


# ---------------------------------------------------------------------------------------------
# Prefill-kernel levers (branch prefill-kernels; image vllm-tt:k2-mr-pk, patches/tt-metal/0025).
# Both need C++ that only that image carries. Each lever checks the image at engagement and
# raises if the op surface is missing, so an arm can never run the default kernels while
# believing it runs the new ones. With the env unset nothing here is reached.
# ---------------------------------------------------------------------------------------------


def sdpa_gqa_mcast_enabled():
    """QWEN36_SDPA_GQA_MCAST=1: causal chunked-prefill SDPA with GQA K/V multicast (tt-metal #56119)."""
    return _tt_public_getenv("QWEN36_SDPA_GQA_MCAST", "0") == "1"


_SDPA_GQA_CHECKED = False


def sdpa_gqa_mcast_check_image():
    """Fail closed: the per-call gate SDPAProgramConfig(gqa_kv_mcast=...) must exist in this ttnn."""
    global _SDPA_GQA_CHECKED
    if _SDPA_GQA_CHECKED:
        return
    try:
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(1, 1), q_chunk_size=32, k_chunk_size=32, gqa_kv_mcast=True
        )
        ok = getattr(cfg, "gqa_kv_mcast", False) is True
    except TypeError:
        ok = False
    if not ok:
        raise RuntimeError(
            "QWEN36_SDPA_GQA_MCAST=1 but this ttnn has no SDPAProgramConfig.gqa_kv_mcast "
            "(needs image vllm-tt:k2-mr-pk / patches/tt-metal/0025); refusing to run the default schedule"
        )
    _SDPA_GQA_CHECKED = True
    logger.info(
        "QWEN36_SDPA_GQA_MCAST engaged: causal chunked-prefill SDPA requests the GQA K/V multicast "
        "schedule per call (q_chunk=k_chunk chosen so every (head, q_chunk) unit gets its own core); "
        "the op TT_FATALs if a requested call cannot take it"
    )


def sdpa_gqa_mcast_chunk(S, chunk_start, n_q_heads, num_cores, n_kv_heads=1, max_chunk=256):
    """Chunk size for a GQA-multicast SDPA call, or None when the call cannot take the schedule.

    The schedule (sdpa_program_factory.cpp, #56119) gives every (head, q_chunk) unit its own core:
    n_q_heads * ceil(S / q) <= num_cores. q must divide S (a whole number of chunks), divide
    chunk_start (chunk_start % q_chunk == 0), be a multiple of 32, and at most max_chunk (256 is what
    upstream runs; larger K/V CBs are untested in L1). Largest eligible q wins: fewer, longer
    K/V streams per injector. Groups need n_q_heads % n_kv_heads == 0 with >= 2 q heads per kv head.
    """
    if n_kv_heads <= 0 or n_q_heads % n_kv_heads != 0 or n_q_heads // n_kv_heads < 2:
        return None
    q = max_chunk
    while q >= 32:
        if S % q == 0 and (chunk_start % q == 0) and n_q_heads * (S // q) <= num_cores:
            return q
        q //= 2
    return None


def sdpa_gqa_mcast_q128_enabled():
    """QWEN36_SDPA_GQA_MCAST_Q128=1: GQA K/V multicast at production's own chunking (patches/tt-metal/0026).

    Unlike QWEN36_SDPA_GQA_MCAST (which switches the call to q=k=256 and therefore changes the
    accumulation order), this keeps the q/k chunk the control picks and asks the op for the
    zigzag-pair placement: every core runs exactly the default schedule's work at that chunk size,
    only re-placed so the 6 q heads of a kv head share one K/V multicast stream. The two levers are
    mutually exclusive (they pick different chunk sizes).
    """
    on = _tt_public_getenv("QWEN36_SDPA_GQA_MCAST_Q128", "0") == "1"
    if on and sdpa_gqa_mcast_enabled():
        raise RuntimeError(
            "QWEN36_SDPA_GQA_MCAST_Q128=1 and QWEN36_SDPA_GQA_MCAST=1 are mutually exclusive "
            "(q=k=128 bit-exact placement vs q=k=256 schedule); set one"
        )
    return on


_SDPA_GQA_Q128_CHECKED = False
_SDPA_GQA_Q128_SEEN = set()


def sdpa_gqa_mcast_q128_check_image():
    """Fail closed: SDPAProgramConfig(gqa_kv_mcast=..., gqa_kv_mcast_pairs=...) must exist (0026)."""
    global _SDPA_GQA_Q128_CHECKED
    if _SDPA_GQA_Q128_CHECKED:
        return
    try:
        cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(1, 1),
            q_chunk_size=32,
            k_chunk_size=32,
            gqa_kv_mcast=True,
            gqa_kv_mcast_pairs=True,
        )
        ok = getattr(cfg, "gqa_kv_mcast", False) is True and getattr(cfg, "gqa_kv_mcast_pairs", False) is True
    except TypeError:
        ok = False
    if not ok:
        raise RuntimeError(
            "QWEN36_SDPA_GQA_MCAST_Q128=1 but this ttnn has no SDPAProgramConfig.gqa_kv_mcast_pairs "
            "(needs image vllm-tt:k2-mr-pk2 / patches/tt-metal/0026); refusing to run the default schedule"
        )
    _SDPA_GQA_Q128_CHECKED = True
    logger.info(
        "QWEN36_SDPA_GQA_MCAST_Q128 engaged: causal chunked-prefill SDPA keeps production's q/k chunk "
        "(128 for 2048-token chunks) and requests the GQA K/V multicast with the default schedule's "
        "zigzag-pair work per core (gqa_kv_mcast_pairs); the op TT_FATALs if a requested call cannot take it"
    )


def sdpa_gqa_mcast_q128_kwargs(S, qk_chunk, n_q_heads, num_cores, n_kv_heads=1):
    """SDPAProgramConfig kwargs for a Q128-lever call at the control's chunk size, or None.

    Mirrors sdpa_program_factory.cpp (0025 + 0026) without changing qk_chunk:
      one-unit mode  n_q_heads * n_chunks <= num_cores (every (head, q_chunk) on its own core);
      pair mode      n_chunks even and n_q_heads * n_chunks / 2 <= num_cores (the default schedule
                     then gives every busy core one causal zigzag pair; the op keeps that work).
    None (the call keeps the default schedule, same chunk size) when neither holds or the grouping
    is not GQA (n_q_heads % n_kv_heads, >= 2 q heads per kv head).
    """
    if n_kv_heads <= 0 or n_q_heads % n_kv_heads != 0 or n_q_heads // n_kv_heads < 2:
        return None
    if qk_chunk <= 0 or S % qk_chunk != 0:
        return None
    n_chunks = S // qk_chunk
    units = n_q_heads * n_chunks
    if units <= num_cores or (n_chunks % 2 == 0 and units // 2 <= num_cores):
        return {"gqa_kv_mcast": True, "gqa_kv_mcast_pairs": True}
    return None


def gdn_fused_enabled():
    """QWEN36_GDN_FUSED=1: prefill GDN on the fused prep->scan prim (tt-metal #53790 + #57440)."""
    return _tt_public_getenv("QWEN36_GDN_FUSED", "0") == "1"


# Heads per device above which the fused prim is not requested. The upstream cost model
# (#57440/#57912) predicts fused wins at BH <= 32 on an 11x10 grid; BH=48 (two users batched at
# TP=2) is predicted a loss, and there the call keeps the default phased program.
GDN_FUSED_MAX_BH = int(_tt_public_getenv("QWEN36_GDN_FUSED_MAX_BH", "32"))
_GDN_FUSED_INSTALLED = False
_GDN_FUSED_SEEN_BH = set()


def gdn_fused_bh(args, kwargs):
    """B*HV of a chunk_gated_delta_rule call: beta is [B, T, HV] (5th positional or kwarg)."""
    beta = kwargs.get("beta", args[4] if len(args) > 4 else None)
    if beta is None:
        return None
    shp = list(beta.shape)
    if len(shp) != 3:
        return None
    return int(shp[0]) * int(shp[2])


def gdn_fused_install():
    """Route prefill chunk_gated_delta_rule calls with BH <= GDN_FUSED_MAX_BH to the fused prim.

    The call site is the image's fused_chunk.py adapter, which is not in the production mount set,
    so the op is wrapped once here: a call that passes no program_config gets
    ttnn.ChunkGdnFusedProgramConfig() (geometry from the upstream cost model). The fused prim
    TT_FATALs when no geometry fits (it never falls back), so a requested call fails closed.
    Fails closed at install when the image lacks the fused prim.
    """
    global _GDN_FUSED_INSTALLED
    if _GDN_FUSED_INSTALLED:
        return
    cfg_cls = getattr(ttnn, "ChunkGdnFusedProgramConfig", None)
    if cfg_cls is None:
        cfg_cls = getattr(getattr(ttnn._ttnn.operations, "transformer", None), "ChunkGdnFusedProgramConfig", None)
    orig = ttnn.transformer.chunk_gated_delta_rule
    if cfg_cls is None or "program_config" not in (getattr(orig, "__doc__", "") or ""):
        raise RuntimeError(
            "QWEN36_GDN_FUSED=1 but this ttnn has no ChunkGdnFusedProgramConfig / program_config kwarg "
            "(needs image vllm-tt:k2-mr-pk / patches/tt-metal/0025); refusing to run the phased path"
        )

    def _chunk_gdr_fused(*args, **kwargs):
        if kwargs.get("program_config") is None:
            bh = gdn_fused_bh(args, kwargs)
            if bh is not None and bh <= GDN_FUSED_MAX_BH:
                kwargs["program_config"] = cfg_cls()
                if bh not in _GDN_FUSED_SEEN_BH:
                    _GDN_FUSED_SEEN_BH.add(bh)
                    logger.info(f"QWEN36_GDN_FUSED path: BH={bh} -> fused prep->scan prim")
            elif bh not in _GDN_FUSED_SEEN_BH:
                _GDN_FUSED_SEEN_BH.add(bh)
                logger.info(f"QWEN36_GDN_FUSED path: BH={bh} > {GDN_FUSED_MAX_BH} -> default phased program")
        return orig(*args, **kwargs)

    _chunk_gdr_fused.__wrapped__ = orig
    _chunk_gdr_fused.__doc__ = orig.__doc__
    ttnn.transformer.chunk_gated_delta_rule = _chunk_gdr_fused
    _GDN_FUSED_INSTALLED = True
    logger.info(
        f"QWEN36_GDN_FUSED engaged: prefill chunk_gated_delta_rule with BH <= {GDN_FUSED_MAX_BH} runs the "
        "fused prep->scan prim (tt-metal #57440; no DRAM intermediates); decode and larger BH unchanged"
    )


# ---------------------------------------------------------------------------------------------
# Prefill bundle H (branch prefill-bundle-h; image vllm-tt:k2-mr-pk3, patches/tt-metal/0027 + 0028).
# Two data-movement-only levers: the bytes each kernel consumes, and the order it consumes them in,
# are unchanged, so outputs are bit-identical by construction (ttsim: [source validation record]).
# Each fails closed on an image without its op surface and logs one '<FLAG> engaged' banner.
# ---------------------------------------------------------------------------------------------

# T3 defaults: one read barrier per K/V chunk at production's k=128, d=256 (4 x 8 = 32 tiles) instead
# of every 4 tiles, and a third K/V slot so the reader can run one chunk further ahead of compute.
SDPA_READER_BARRIER_DEFAULT = 32
SDPA_READER_KV_SLOTS_DEFAULT = 3


def sdpa_reader_depth_enabled():
    """QWEN36_SDPA_READER_DEPTH=1: deeper SDPA prefill reader (patches/tt-metal/0027)."""
    return _tt_public_getenv("QWEN36_SDPA_READER_DEPTH", "0") == "1"


def sdpa_reader_depth_kwargs():
    """SDPAProgramConfig kwargs of the T3 lever: {reader_barrier_tiles, kv_cb_slots}.

    QWEN36_SDPA_READER_BARRIER (tiles between partial read barriers, >= 1) and
    QWEN36_SDPA_READER_KV_SLOTS (K/V CB depth in chunks, 2..4) override the defaults.
    """
    barrier = int(_tt_public_getenv("QWEN36_SDPA_READER_BARRIER", str(SDPA_READER_BARRIER_DEFAULT)))
    slots = int(_tt_public_getenv("QWEN36_SDPA_READER_KV_SLOTS", str(SDPA_READER_KV_SLOTS_DEFAULT)))
    if barrier < 1:
        raise ValueError(f"QWEN36_SDPA_READER_BARRIER must be >= 1, got {barrier}")
    if not 2 <= slots <= 4:
        raise ValueError(f"QWEN36_SDPA_READER_KV_SLOTS must be 2..4, got {slots}")
    return {"reader_barrier_tiles": barrier, "kv_cb_slots": slots}


_SDPA_READER_DEPTH_CHECKED = False


def sdpa_reader_depth_check_image():
    """Fail closed: SDPAProgramConfig(reader_barrier_tiles=..., kv_cb_slots=...) must exist (0027)."""
    global _SDPA_READER_DEPTH_CHECKED
    if _SDPA_READER_DEPTH_CHECKED:
        return
    kw = sdpa_reader_depth_kwargs()
    try:
        cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(1, 1), q_chunk_size=32, k_chunk_size=32, **kw)
        ok = (
            getattr(cfg, "reader_barrier_tiles", None) == kw["reader_barrier_tiles"]
            and getattr(cfg, "kv_cb_slots", None) == kw["kv_cb_slots"]
        )
    except TypeError:
        ok = False
    if not ok:
        raise RuntimeError(
            "QWEN36_SDPA_READER_DEPTH=1 but this ttnn has no SDPAProgramConfig.reader_barrier_tiles/kv_cb_slots "
            "(needs image vllm-tt:k2-mr-pk3 / patches/tt-metal/0027); refusing to run the default reader"
        )
    _SDPA_READER_DEPTH_CHECKED = True
    logger.info(
        f"QWEN36_SDPA_READER_DEPTH engaged: chunked-prefill SDPA reader issues {kw['reader_barrier_tiles']} tiles "
        f"between read barriers (default 4; 8 on GQA-mcast injectors) with {kw['kv_cb_slots']} K/V CB slots "
        "(default 2); data movement only, same q/k chunk and schedule"
    )


_SDPA_PACK2_CHECKED = False


def sdpa_stream_pack2_enabled():
    """QWEN36_SDPA_STREAM_PACK2=1: streaming SDPA blocked-pack threshold 4 -> 2 on BH (patches/tt-metal/0052, upstream #60138)."""
    return _tt_public_getenv("QWEN36_SDPA_STREAM_PACK2", "0") == "1"


def sdpa_stream_pack2_check_image():
    """Fail closed: the 0052 factory getenv must exist; the C++ banner proves the define path."""
    global _SDPA_PACK2_CHECKED
    if _SDPA_PACK2_CHECKED:
        return
    _SDPA_PACK2_CHECKED = True
    logger.info("QWEN36_SDPA_STREAM_PACK2 engaged: streaming SDPA blocked-pack threshold 4 -> 2 (output-transparent pack choice)")

def gdn_scan_mcast_enabled():
    """QWEN36_GDN_SCAN_MCAST=1: phased GDN scan shared-tile multicast (patches/tt-metal/0028)."""
    return _tt_public_getenv("QWEN36_GDN_SCAN_MCAST", "0") == "1"


def gdn_scan_mcast_geometry(bh, vt, grid_x=11, grid_y=10):
    """(NV, w, h) the 0028 scan factory will use for BH heads of Vt V-tiles, or None (not applicable).

    Mirrors chunk_gdn_phased_program_factory.cpp: NV is the default schedule's (largest NV | Vt with
    BH*NV <= cores); each head's NV cores go on a w x h rectangle, the 1xNV row first, then the most
    square shape, whichever fits BH times on the grid. NV < 2 has nothing to share.
    """
    if not bh or not vt or bh <= 0 or vt <= 0 or bh > grid_x * grid_y:
        return None
    nv = 1
    for cand in range(vt, 0, -1):
        if vt % cand == 0 and bh * cand <= grid_x * grid_y:
            nv = cand
            break
    if nv < 2:
        return None
    rest = [(nv // h, h) for h in range(2, nv + 1) if nv % h == 0]
    rest.sort(key=lambda s: (abs(s[0] - s[1]), -s[0]))
    for w, h in [(nv, 1)] + rest:
        if w <= grid_x and h <= grid_y and bh <= (grid_x // w) * (grid_y // h):
            return nv, w, h
    return None


def gdn_scan_vt(args, kwargs, hv):
    """V in tiles of a chunk_gated_delta_rule call: v is [B, T, HV, V] or flat [B, T, HV*V]."""
    v = kwargs.get("v", args[2] if len(args) > 2 else None)
    if v is None or not hv:
        return None
    shp = list(v.shape)
    if len(shp) == 4:
        V = int(shp[3])
    elif len(shp) == 3:
        V = int(shp[2]) // int(hv)
    else:
        return None
    return V // 32 if V % 32 == 0 else None


_GDN_SCAN_MCAST_INSTALLED = False
_GDN_SCAN_MCAST_SEEN = set()


def gdn_scan_mcast_install(grid_x=11, grid_y=10):
    """Route PHASED prefill chunk_gated_delta_rule calls to ChunkGdnPhasedProgramConfig(use_mcast=True).

    A call gets the multicast only when it would otherwise run the default phased program: no
    program_config from the caller, not taken by QWEN36_GDN_FUSED (BH <= QWEN36_GDN_FUSED_MAX_BH),
    and a geometry the factory can place (gdn_scan_mcast_geometry). Under QWEN36_GDN_FUSED that means
    only the batched BH > 32 calls. Order-independent with gdn_fused_install. Fails closed at install
    when the image lacks 0028 (its ChunkGdnPhasedProgramConfig doc names "cfx 0028"; k2-mr-pk/-pk2 refuse use_mcast).
    """
    global _GDN_SCAN_MCAST_INSTALLED
    if _GDN_SCAN_MCAST_INSTALLED:
        return
    cfg_cls = getattr(ttnn, "ChunkGdnPhasedProgramConfig", None)
    if cfg_cls is None:
        cfg_cls = getattr(getattr(ttnn._ttnn.operations, "transformer", None), "ChunkGdnPhasedProgramConfig", None)
    orig = ttnn.transformer.chunk_gated_delta_rule
    doc = (getattr(cfg_cls, "__doc__", "") or "") if cfg_cls is not None else ""
    if cfg_cls is None or "program_config" not in (getattr(orig, "__doc__", "") or "") or "0028" not in doc:
        raise RuntimeError(
            "QWEN36_GDN_SCAN_MCAST=1 but this ttnn does not carry ChunkGdnPhasedProgramConfig(use_mcast=True) "
            "(needs image vllm-tt:k2-mr-pk3 / patches/tt-metal/0028); refusing to run the default scan"
        )

    def _chunk_gdr_scan_mcast(*args, **kwargs):
        if kwargs.get("program_config") is None:
            beta = kwargs.get("beta", args[4] if len(args) > 4 else None)
            hv = int(beta.shape[2]) if beta is not None and len(list(beta.shape)) == 3 else None
            bh = gdn_fused_bh(args, kwargs)
            fused_takes = gdn_fused_enabled() and bh is not None and bh <= GDN_FUSED_MAX_BH
            if bh is not None and not fused_takes:
                vt = gdn_scan_vt(args, kwargs, hv)
                geo = gdn_scan_mcast_geometry(bh, vt, grid_x, grid_y)
                if geo is not None:
                    kwargs["program_config"] = cfg_cls(use_mcast=True)
                key = (bh, vt, geo)
                if key not in _GDN_SCAN_MCAST_SEEN:
                    _GDN_SCAN_MCAST_SEEN.add(key)
                    if geo is not None:
                        logger.info(f"QWEN36_GDN_SCAN_MCAST path: BH={bh} Vt={vt} -> NV={geo[0]} on {geo[1]}x{geo[2]}")
                    else:
                        logger.info(f"QWEN36_GDN_SCAN_MCAST path: BH={bh} Vt={vt} not applicable, default scan")
        return orig(*args, **kwargs)

    _chunk_gdr_scan_mcast.__wrapped__ = orig
    _chunk_gdr_scan_mcast.__doc__ = orig.__doc__
    ttnn.transformer.chunk_gated_delta_rule = _chunk_gdr_scan_mcast
    _GDN_SCAN_MCAST_INSTALLED = True
    logger.info(
        "QWEN36_GDN_SCAN_MCAST engaged: phased prefill chunk_gated_delta_rule scans multicast the six shared "
        "per-chunk tiles from one core per head to its V-block siblings (tt-metal 5f02a6718f4, 2x2 rectangles "
        "keep NV=4 at BH=24); calls taken by QWEN36_GDN_FUSED are unchanged"
    )


# Experimental, decode-only MLP column stream. Off: no import or function patch.
# Candidate arms mount the launcher/kernels at /opt/mkern. Repacking is from
# actual device weights at load, so no tensor-cache format or host-placeholder
# contract changes. Import-time dispatch/weight layout makes this REBOOT class.
if (_tt_public_getenv("QWEN36_DECODE_COLUMN_STREAM", "0") not in ("", "0", "off")
        or _tt_public_getenv("QWEN36_DECODE_COLUMN_STREAM_MUSTFAIL", "0") == "1"):
    import sys as _column_sys
    _column_sys.path.insert(0, "/opt/mkern")
    from column_port import install as _column_install
    _column_install(globals())

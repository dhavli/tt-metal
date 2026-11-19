# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel Gated DeltaNet for Qwen3.5.

Recurrence is per value-head (no cross-device comms inside); all-reduce after row-parallel out.
Reuses `recurrent_gated_delta_rule_decode_ttnn`; weights interleaved. GDN norm uses raw weight
(no +1) + SiLU(z) gate — distinct from QK/layer norms.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
from loguru import logger
import os

import torch

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt import gdn_qrm, gdn_tail
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
    recurrent_gated_delta_rule_decode_ttnn,
)
try:
    from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
        recurrent_gated_delta_rule_decode_packed_ttnn,
    )
except ImportError:
    recurrent_gated_delta_rule_decode_packed_ttnn = None
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_seq import (
    chunk_gated_delta_rule_seq_adapter,
    create_chunk_masks_seq,
)
from models.experimental.gated_attention_gated_deltanet.tt.ttnn_gated_deltanet import _causal_conv1d_fir
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce


def _causal_conv1d_fir_conv_sel(x, kernel_size, device, memory_config, conv_state, weight_taps, conv_sel):
    """FIR conv1d for the traced masked-bucket prefill (QWEN36_PREFILL_BUCKET_TRACE).

    The eager masked path passes an int valid_len to _causal_conv1d_fir, which builds the
    decode-window one-hot with ttnn.from_torch -- a host write, fatal inside trace capture. Here the
    conv output itself takes the trace-safe valid_len=None path (padding rows only feed padding
    rows, causal), and the new conv state [1, K-1, D] (the last K-1 REAL inputs) is selected by a
    matmul against ``conv_sel`` [1, K-1, T+K-1], a persistent device one-hot refreshed per request.
    Only reached when gdn_masks is supplied; the eager path never calls this."""
    assert conv_state is not None, "conv_sel requires persistent conv_state"
    x_padded = ttnn.concat([conv_state, x], dim=1, memory_config=memory_config)
    if x_padded.layout != ttnn.TILE_LAYOUT:
        xp = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_padded)
    else:
        xp = x_padded
    # HiFi4 + fp32 dest: a one-hot select must return x exactly (a lower fidelity drops mantissa bits of x).
    _cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    new_state = ttnn.matmul(
        conv_sel, xp, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=conv_state.dtype, compute_kernel_config=_cc
    )
    ttnn.deallocate(xp)
    conv, _static_state = _causal_conv1d_fir(
        x,
        None,
        None,
        kernel_size,
        device,
        memory_config=memory_config,
        conv_state=conv_state,
        weight_taps=weight_taps,
        bias_dev=None,
        valid_len=None,
    )
    ttnn.deallocate(_static_state)  # static last-(K-1) slice: wrong rows for a padded bucket
    return conv, new_state


def _conv_sel_state(x, device, memory_config, conv_state, conv_sel):
    """QWEN36_PREFILL_BUCKET_KDA: ONLY the decode-window half of _causal_conv1d_fir_conv_sel -- the same concat,
    to_layout and HiFi4/fp32-dest one-hot matmul, op for op, so the conv state is bitwise the shipped one. The conv
    output itself comes from the kda op (_conv_kda_prefill), so the MAC FIR (a second concat, three unaligned TILE
    slices + re-tilizes, the K multiply/addcmul chain, a SiLU and a discarded static-state slice) never runs."""
    x_padded = ttnn.concat([conv_state, x], dim=1, memory_config=memory_config)
    if x_padded.layout != ttnn.TILE_LAYOUT:
        xp = ttnn.to_layout(x_padded, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_padded)
    else:
        xp = x_padded
    _cc = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
    )
    new_state = ttnn.matmul(
        conv_sel, xp, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=conv_state.dtype, compute_kernel_config=_cc
    )
    ttnn.deallocate(xp)
    return new_state


def _mul_mask(t, mask):
    """t * mask for the traced masked bucket. The mask buffers are fp32 (beta/g) and bf16 (q/k/v);
    GDN_G_FP32 / AB_BF8 move the gate dtypes, so match the mask to t on device (typecast is a
    device op, legal under trace) rather than assume a dtype pairing."""
    m = mask if mask.dtype == t.dtype else ttnn.typecast(mask, t.dtype)
    out = ttnn.multiply(t, m, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    if m is not mask:
        ttnn.deallocate(m)
    ttnn.deallocate(t)  # the unmasked q/k/v/beta/g are dead once masked (never the persistent mask)
    return out


def _is_gdn_outproj_colpar_enabled():
    """Column-parallel GDN prefill out-projection (PR #57377). Default OFF."""
    return _tt_public_getenv("QWEN36_GDN_OUTPROJ_COLPAR", "0") == "1" or _tt_public_getenv("QWEN36_GDN_OUT_COLPAR", "0") == "1"


def _shard_small_f32(torch_tensor, mesh, cache_path, dim=-1):
    """Small per-head tensor -> fp32, sharded. tp_common.shard_small casts to bfloat16 on the HOST
    whatever device dtype is requested, which would silently throw away the precision this is for."""
    if torch_tensor.dim() == 1:
        torch_tensor = torch_tensor.unsqueeze(0).unsqueeze(0)
    elif torch_tensor.dim() == 2:
        torch_tensor = torch_tensor.unsqueeze(0)
    return ttnn.as_tensor(
        torch_tensor.to(torch.float32),
        dtype=ttnn.float32,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=dim),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def _softplus_add(a, bias):
    """g-gate: softplus(a + bias) fused into one op (softplus as a post-activation on the add)."""
    return ttnn.add(a, bias, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)])


def _silu_mul(x, z, memory_config, dtype=None, fast=None):
    """out-gate: x * silu(z). NOT fused into one op: fusing silu via input_tensor_b_activations
    overflows to NaN in the real layer for large-magnitude z (op-level PCC hid it — small inputs).
    dtype: optional output dtype (bf16 for the column-parallel prefill out-proj; default = x's)."""
    s = ttnn.silu(z, memory_config=memory_config)
    kw = tpc.fast_mul("gdn_gate", x.shape[-2]) if fast is None else fast
    if dtype is None:
        return ttnn.multiply(x, s, memory_config=memory_config, **kw)
    return ttnn.multiply(x, s, memory_config=memory_config, dtype=dtype, **kw)


def verify_double():
    """QWEN36_MTP_VERIFY_DOUBLE: comma set of {proj, cmulti, rec, normgate, out}. A timing diagnostic for the MTP
    verify's GDN layers: the named op runs a second time on the same inputs and the copy is freed, so the outputs stay
    CORRECT and the verify p50 delta is that op's marginal cost (cmulti also clones its three inputs, which it frees).
    Never ship it."""
    v = _tt_public_getenv("QWEN36_MTP_VERIFY_DOUBLE", "")
    s = {x.strip() for x in v.split(",") if x.strip() and x.strip() != "0"}
    bad = s - {"proj", "cmulti", "rec", "normgate", "out"}
    if bad:
        raise RuntimeError(f"QWEN36_MTP_VERIFY_DOUBLE: unknown {sorted(bad)} (known: proj, cmulti, rec, normgate, out)")
    return s


# The gap the bare ".abpad" key has always meant: (-24) % 32 with Nv/chip = 24 (27B, TP=2). Kept so the served
# tensor cache stays warm; every other gap carries its value in the key.
_ABPAD_LEGACY_GAP = 8


def qkvzab_cache_name(proj1d, ab_gap):
    """Tensor-cache name of the fused [qkv|z|a|(gap zeros)|b] GDN in-projection. The zero rows change the weight's
    layout, so the name must change with the gap, or a warm cache serves a weight padded for a different gap."""
    name = "qkvzab" + (".il" if proj1d else ".dramshard")
    ab_gap = int(ab_gap or 0)
    if ab_gap == _ABPAD_LEGACY_GAP:
        name += ".abpad"
    elif ab_gap:
        name += ".abpad%d" % ab_gap
    return name


def load_gdn_weights_tp(mesh, sd, args, cache_dir=None):
    """Shard one GDN layer's linear_attn.* weights across the mesh."""
    # QWEN36_ATTN_GDN_BF4=1 stores the attention and GDN projection weights as
    # bfloat4_b. Default bfloat8_b, i.e. this file is inert unless the flag is set. WEIGHTS ONLY --
    # activations, the KV cache and the GDN state dtypes are untouched.
    #
    # This is a weight-FORMAT switch, not a quantiser: it says how the loader packs whatever the
    # checkpoint holds. Over a checkpoint that was NOT rounded onto the bfp4_b grid it is plain
    # round-to-nearest bf4 and measurably worse than production (arm R2: KL 0.176 vs prod 0.111).
    # Set it only with a grid-snapped checkpoint; the GPTQ "G2" one is the measured pairing
    # (KL 0.085, top-1 0.944, GSM8K 58/60). See docs/QUANT.md.
    import os as _bfpq_os  # the loader imports os locally further down, which shadows the global
    _wdt = ttnn.bfloat4_b if _tt_public_getenv("QWEN36_ATTN_GDN_BF4") == "1" else ttnn.bfloat8_b
    if _tt_public_getenv("QWEN36_ATTN_GDN_BF4") == "1":
        print("QWEN36_ATTN_GDN_BF4 engaged: %s weights BFLOAT4_B" % __name__, flush=True)
    tp = args.num_devices
    nk, dk, nv, dv = args.gdn_nk, args.gdn_dk, args.gdn_nv, args.gdn_dv
    key_dim, value_dim = args.gdn_key_dim, args.gdn_value_dim
    qkv_per = args.gdn_qkv_dim_tp
    z_per = args.gdn_z_dim_tp
    nv_per = args.gdn_nv_tp

    if cache_dir is not None:
        import os

        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    # State-dict keys vary by loader: optional linear_attn. prefix; conv1d may be fused or q/k/v split.
    P = "linear_attn." if any(k.startswith("linear_attn.") for k in sd) else ""

    def first_key(*names):
        for n in names:
            if (P + n) in sd:
                return sd[P + n]
        raise KeyError(f"none of {[P + n for n in names]} found in GDN state dict")

    # Fused QKV+Z (column-parallel)
    qkv_w = first_key("in_proj_qkv.weight", "qkv_proj.weight")
    if (P + "conv1d.weight") in sd:
        conv1d_w = sd[P + "conv1d.weight"]
    else:  # bf16 remap: reassemble fused conv1d from q/k/v streams
        conv1d_w = torch.cat([sd[P + "q_conv.weight"], sd[P + "k_conv.weight"], sd[P + "v_conv.weight"]], dim=0)
    qkv_re = tpc.prepare_gdn_qkv(qkv_w, key_dim, value_dim, nk, dk, nv, dv, tp)
    z_w = sd[P + "in_proj_z.weight"]
    a_w, b_w = sd[P + "in_proj_a.weight"], sd[P + "in_proj_b.weight"]
    tw = {}
    # Column-parallel qkvz (DRAM-sharded decode matmul when enabled); distinct .dramshard cache
    qkvz_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
    # Fold a/b into qkvz → one matmul outputs [qkv|z|a|b] (default when DRAM-sharded)
    fuse_ab = qkvz_sharded
    if fuse_ab:
        fused = torch.cat(
            [
                torch.cat(
                    [
                        qkv_re[d * qkv_per : (d + 1) * qkv_per],
                        z_w[d * z_per : (d + 1) * z_per],
                        a_w[d * nv_per : (d + 1) * nv_per],
                        # QWEN36_GDN_AB_PAD: zero rows so b starts tile-aligned (empty when off)
                        a_w.new_zeros((getattr(args, "gdn_ab_gap", 0), a_w.shape[1])),
                        b_w[d * nv_per : (d + 1) * nv_per],
                    ],
                    dim=0,
                )
                for d in range(tp)
            ],
            dim=0,
        )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.gdn_qkvzab_weight_memcfg,
            cache_path=c(qkvzab_cache_name(_proj1d, getattr(args, "gdn_ab_gap", 0))),
            dtype=_wdt,
        )
    else:
        fused = torch.cat(
            [
                torch.cat([qkv_re[d * qkv_per : (d + 1) * qkv_per], z_w[d * z_per : (d + 1) * z_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        qkvz_mc = args.gdn_qkvz_weight_memcfg if qkvz_sharded else ttnn.DRAM_MEMORY_CONFIG
        tw["qkvz"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=qkvz_mc,
            cache_path=c("qkvz" + (".dramshard" if qkvz_sharded else "")),
            dtype=_wdt,
        )
        # Separate A+B projection (column-parallel fallback)
        ab = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        tw["ab"] = tpc.shard_w(
            ab, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab"), dtype=_wdt
        )
    # Row-parallel out projection: DRAM-width-sharded (like the in-proj) — decode tput win.
    _out_sharded = getattr(args, "gdn_out_weight_memcfg", None) is not None
    tw["out"] = tpc.shard_w(
        sd[P + "out_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.gdn_out_weight_memcfg if _out_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("out.dramshard" if _out_sharded else "out"),
        dtype=_wdt,
    )
    # QWEN36_DECODE_DRAM_SHARDED_MR (tp_common): decode-only DRAM width-sharded views of the fused
    # qkvzab in-proj (relayout by default: prefill's AGMM reads the same tensor) and of the out-proj
    # (always a copy: the unfused prefill 2D matmul and line ~1209 keep the interleaved tw["out"]).
    mr = tpc.decode_mr_for(args, mesh)
    if "qkvzab" in mr:
        if not fuse_ab:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR qkvzab needs the folded [qkv|z|a|b] weight")
        tw["qkvz_mr"] = tpc.decode_mr_weight(tw["qkvz"], mr["qkvzab"])
        if mr["qkvzab"].layout == "relayout":
            tw["qkvz"] = tw["qkvz_mr"]
    if "out" in mr:
        if _out_sharded:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR out expects the interleaved GDN out-proj")
        if mr["out"].k != args.gdn_value_dim_tp:
            raise RuntimeError(
                f"QWEN36_DECODE_DRAM_SHARDED_MR out: role K={mr['out'].k} (attention) != GDN value dim "
                f"{args.gdn_value_dim_tp}; the shared 'out' role needs both out-projections the same shape"
            )
        tw["out_mr"] = tpc.decode_mr_weight(tw["out"], mr["out"])
    if mr:
        logger.info(f"QWEN36_DECODE_MR_LAYER engaged: gdn roles={sorted(k for k in mr if k in ('qkvzab', 'out'))}")
    if getattr(args, "num_devices", 1) > 1 and _is_gdn_outproj_colpar_enabled():
        # COLUMN-parallel copy of the out-proj for prefill (PR #57377).
        # Decode keeps the row-sharded tw["out"] (matmul + all-reduce).
        tw["out_colpar"] = tpc.shard_w(
            sd[P + "out_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("out.colpar"),
            dtype=_wdt,
        )
    # Per-head params
    tw["dt_bias"] = tpc.shard_small(sd[P + "dt_bias"].float(), mesh, c("dt_bias"))
    A_log = tpc.shard_small(sd[P + "A_log"].float(), mesh, c("A_log"))
    tw["neg_exp_A"] = ttnn.neg(ttnn.exp(A_log))
    # ---- GDN prefill gate precision -------------------------------------------------
    # g = -exp(A_log) * softplus(a + dt_bias), and PREFILL sums g over thousands of rows, so a
    # slow-decay head integrates any relative error in g into a large cumulative-decay error.
    # Measured at TP=1 on the G2 bf4 arm: GDN prefill layers 8-19% off fp32 in L1-30 (recurrent
    # state up to 48%), vs 2-3.5% for the attention layers.
    #
    # QWEN36_GDN_G_FP32=1 keeps A_log/dt_bias in fp32 and runs the prefill gate in fp32. The bf16
    # tw["dt_bias"] / tw["neg_exp_A"] above are LEFT AS THEY ARE because the decode kernels take
    # them directly; these are separate fp32 copies used only by the prefill gate. 48 values/layer.
    if _tt_public_getenv("QWEN36_GDN_G_FP32", "0") == "1":
        tw["dt_bias_f32"] = _shard_small_f32(sd[P + "dt_bias"].float(), mesh, c("dt_bias.f32"))
        # -exp(A_log) on the HOST in fp32: the device bf16 neg+exp above is itself 0.26% off.
        tw["neg_exp_A_f32"] = _shard_small_f32(-sd[P + "A_log"].float().exp(), mesh, c("neg_exp_A.f32"))
        print("QWEN36_GDN_G_FP32 engaged: A_log/dt_bias and the prefill g gate in FLOAT32", flush=True)
    # QWEN36_GDN_AB_BF8=prefill: GPTQ put in_proj_qkv/z/out on the bfp4 grid but left in_proj_a and
    # in_proj_b on the 8-bit grid, and QWEN36_ATTN_GDN_BF4 packs the whole fused [qkv|z|a|b] weight
    # as bfloat4_b -- so a/b get a plain RTN to 4 bits (measured 9.7% / 6.4% off at L6). a/b are the
    # entire input to the gates, so this is where that error enters. A separate bfloat8_b [a|b]
    # weight is exact for the checkpoint's values. PREFILL ONLY: at decode the extra per-layer
    # matmul costs ~+7.7 ms/token and the error does not integrate, so "1" is refused.
    _ab8 = _tt_public_getenv("QWEN36_GDN_AB_BF8", "0").lower()
    if _ab8 not in ("0", "", "prefill"):
        raise RuntimeError(
            f"QWEN36_GDN_AB_BF8={_ab8!r} unsupported in production; only 'prefill' is implemented "
            "(applying the 8-bit a/b at decode too costs ~+7.7 ms/token and buys nothing: the error "
            "only integrates over the prefill)."
        )
    if _ab8 == "prefill" and _tt_public_getenv("QWEN36_ATTN_GDN_BF4") == "1":
        _ab_f = torch.cat(
            [
                torch.cat([a_w[d * nv_per : (d + 1) * nv_per], b_w[d * nv_per : (d + 1) * nv_per]], dim=0)
                for d in range(tp)
            ],
            dim=0,
        )
        # Column-sharded exactly like the a/b slice of the fused weight, so each device keeps its
        # own [a_d|b_d]; ~0.25 MB per layer at bfloat8_b.
        tw["ab8_col"] = tpc.shard_w(
            _ab_f, mesh, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG, cache_path=c("ab8.col"), dtype=ttnn.bfloat8_b
        )
        print("QWEN36_GDN_AB_BF8 engaged: in_proj_a/b BFLOAT8_B, mode=prefill", flush=True)
    tw["norm_w"] = tpc.replicate(sd[P + "norm.weight"].float(), mesh, c("norm_w"))
    # Conv taps (4), sharded per Q/K/V head grouping
    taps = tpc.prepare_conv_taps(conv1d_w, key_dim, nk, dk, nv, dv, args.gdn_conv_kernel_size, tp)
    tw["conv_taps"] = [tpc.shard_small(taps[j], mesh, c(f"tap{j}")) for j in range(args.gdn_conv_kernel_size)]
    # Depthwise conv1d weight [qkv_dim, 1, K], host-held mesh-sharded (dim=0) for prepare_conv_weights / _conv1d_prefill.
    W1d = torch.stack(taps, dim=-1).reshape(args.gdn_qkv_dim, 1, args.gdn_conv_kernel_size).contiguous()
    tw["conv_w1d"] = ttnn.from_torch(
        W1d,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
    )
    return tw


class TPGatedDeltaNet:
    """Standalone TP GDN decode (per-device value-head recurrence + all-reduce)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        # QWEN36_GDN_FUSED=1 (image vllm-tt:k2-mr-pk): prefill chunk_gated_delta_rule on the fused
        # prep->scan prim (tt-metal #53790 + #57440). Installed once; fails closed without the image.
        if tpc.gdn_fused_enabled():
            tpc.gdn_fused_install()
        # QWEN36_GDN_SCAN_MCAST=1 (image vllm-tt:k2-mr-pk3, 0028): phased prefill scans multicast their
        # shared per-chunk tiles (5f02a6718f4, 2x2 placement keeps NV=4). Calls QWEN36_GDN_FUSED takes stay
        # fused; with both on only the batched BH > 32 calls reach it. Fails closed without the image.
        if tpc.gdn_scan_mcast_enabled():
            _g = mesh.compute_with_storage_grid_size()
            tpc.gdn_scan_mcast_install(_g.x, _g.y)
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        # DRAM-shard the row-parallel out projection (decode tput win; matches loader gate).
        self._out_sharded = getattr(self.args, "gdn_out_weight_memcfg", None) is not None
        # QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded decode in-/out-proj (None = 1D path).
        _mr = tpc.decode_mr_for(args, mesh)
        self._mr_qkvzab = _mr.get("qkvzab")
        self._mr_out = _mr.get("out")
        self.B = args.max_batch_size
        self.Nk = args.gdn_nk_tp
        self.Nv = args.gdn_nv_tp
        self.Dk = args.gdn_dk
        self.Dv = args.gdn_dv
        self.qkv_dim_tp = args.gdn_qkv_dim_tp
        self.qkvz_dim_tp = args.gdn_qkvz_dim_tp
        self.key_dim_tp = args.gdn_key_dim_tp
        self.value_dim_tp = args.gdn_value_dim_tp
        # Flat q/k/v into adapter (skips prefill head-split reshapes)
        self._gdn_flat_qkv = True
        # Fuse adapter output relayout with rms_norm + head-flatten
        self._gdn_fuse_out = True
        self.K = args.gdn_conv_kernel_size
        self.scale = self.Dk**-0.5
        self.cfg = tpc.COMPUTE_PROJ
        # Must match load_gdn_weights_tp gates
        self._dram_sharded = getattr(args, "gdn_qkvz_weight_memcfg", None) is not None
        self._fuse_ab = self._dram_sharded
        # Fuse prefill norm-allgather + qkvzab in-proj into all_gather_minimal_matmul_async.
        # Requires the folded qkvzab weight; norm's post-AG is disabled in layer.py (GDN, prefill).
        self._fuse_agmm = self._fuse_ab
        # PREFILL out-proj fusion (matmul_reduce_scatter, (8,8) grid). Slight TTFT cost at small ISL
        # (~13k crossover from a fixed warmup/compile overhead) but a large win at long ISL (e.g.
        # 128k ~-2s); overlaps the fp32 GDN-out reduce-scatter with the matmul.
        # 2-device meshes with a unit dimension (e.g. P300 1x2 p150a): the fused experimental op
        # deadlocks there in every offset/links/topology config probed (gdn-silicon-mmrs-probe{,2,3}
        # .log), while the unfused _row_proj + tt_all_reduce path (reduce_scatter_minimal_async)
        # is silicon-proven on the same mesh (gdn-silicon-mmrs-probe4.log PASS pcc 1.0007). Keep the
        # fusion only on the topologies it was validated on (P150x4).
        _mesh_shape = list(mesh.shape)
        self._fuse_out_mmrs_prefill = (
            not self._out_sharded and args.num_devices > 1 and not (args.num_devices == 2 and 1 in _mesh_shape)
        )
        # PREFILL out-proj as column-parallel AG+matmul (takes precedence over the MMRS arm when the
        # col-sharded weight was loaded, PR #57377).
        self._out_colpar_prefill = "out_colpar" in tw and _is_gdn_outproj_colpar_enabled()
        if self._out_colpar_prefill and not getattr(TPGatedDeltaNet, "_out_colpar_banner", False):
            TPGatedDeltaNet._out_colpar_banner = True
            logger.info(
                "QWEN36_GDN_OUTPROJ_COLPAR engaged: GDN prefill out-proj uses column-parallel "
                "AG+matmul (all_gather_minimal_matmul_async, PR #57377)"
            )
        # set only when the fp32 gate copies / the bf8 a|b weight were actually built.
        self._g_fp32 = "neg_exp_A_f32" in tw
        self._ab8 = "ab8_col" in tw
        # Pre-build chunk masks once (trace-safe; avoids from_torch inside captured trace)
        self.chunk_seq_masks = create_chunk_masks_seq(args.gdn_chunk_size, mesh)
        # Prefill fused-op constant tiles, owned by this layer (avoids process-lifetime C++ cache vs device lifetime).
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import _FUSED_CHUNK_SIZE, build_fused_const_tiles

        self._fused_const_tiles = build_fused_const_tiles(mesh, _FUSED_CHUNK_SIZE)
        self.conv_states = None
        self.rec_state = None
        # In-place state updates for decode/prefill traces (set by model allocate_kv_caches)
        self._stable_state = False
        self.conv_carry = None  # cross-chunk prefill conv carry [1, K-1, qkv_dim_tp]
        # Native ttnn.conv1d depthwise prefill; L1_FULL slice keeps it trace-safe.
        # Only used when valid_len is None (masked buckets keep the MAC FIR).
        self._gdn_conv1d = True
        self._conv1d_wprep = None  # prepared depthwise weight (populated on first prefill call)
        # QWEN36_GDN_CONV_KDA=1 (c74): prefill causal conv1d + SiLU + q/k/v split in ONE upstream op,
        # ttnn.experimental.kda.qkv_causal_conv1d_silu (4 taps, [1,3,C] history), in place of the conv2d
        # machinery chain (c71 seq profile: untilize/stick-slice/tilize/untilize, 2x halo + depthwise conv2d +
        # untilize + slice_write, tilize, silu, 3 q/k/v slice copies = ~1.4 ms per GDN layer per 2k chunk, of which
        # the conv itself is 0.15 ms). Numerics differ from the conv2d path (FPU mul/add vs the conv's fp32
        # accumulation), so it is gated as tier 2/3 (an internal record).
        # QWEN36_GDN_CONV_KDA_CHUNK: channel_chunk_size (tile aligned, divides qkv_dim_tp; default 512).
        # QWEN36_GDN_CONV_SHIFT=concat: decode conv shift-register via one concat +
        # slice-shift + split (3 programs vs K copies). Same buffer addresses (trace-safe).
        self._gdn_conv_kda = _tt_public_getenv("QWEN36_GDN_CONV_KDA", "0") == "1"
        self._gdn_conv_shift = _tt_public_getenv("QWEN36_GDN_CONV_SHIFT", "0")
        if self._gdn_conv_shift == "concat":
            logger.info("QWEN36_GDN_CONV_SHIFT engaged: concat shift-register (3 programs vs K copies)")
        elif self._gdn_conv_shift not in ("0", ""):
            raise ValueError(f"QWEN36_GDN_CONV_SHIFT={self._gdn_conv_shift!r}: expected concat or 0")
        if self._gdn_conv_kda:
            _kda = getattr(ttnn.experimental, "kda", None)
            if _kda is None or not hasattr(_kda, "qkv_causal_conv1d_silu"):
                raise RuntimeError("QWEN36_GDN_CONV_KDA=1 but this ttnn has no ttnn.experimental.kda.qkv_causal_conv1d_silu")
            self._kda_chunk = int(_tt_public_getenv("QWEN36_GDN_CONV_KDA_CHUNK", "512"))
            if self._kda_chunk % 32 or self.qkv_dim_tp % self._kda_chunk:
                raise ValueError(f"QWEN36_GDN_CONV_KDA_CHUNK={self._kda_chunk} must be tile aligned and divide {self.qkv_dim_tp}")
            self._kda_cc = ttnn.init_device_compute_kernel_config(
                mesh.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True
            )
            logger.info(
                f"QWEN36_GDN_CONV_KDA engaged: prefill conv1d+silu+split via kda.qkv_causal_conv1d_silu "
                f"(C={self.qkv_dim_tp}, chunk={self._kda_chunk})"
            )
        # Persistent zero sources for trace-safe reset_state_inplace (alloc before any trace)
        self._zero_conv0 = None
        self._zero_conv_carry = None
        self._zero_rec = None
        self._pending = []  # per-user (rec, conv) states collected during batched per-user prefill
        # QWEN36_GDN_CONV_WINDOW_ONCE (c105, boot time): while the model sets _defer_conv_window (only inside
        # prefill_traced_chunked and the chunk-trace capture), carry-path chunks skip the decode conv-window
        # writes; the model then calls finalize_conv_window() once after the last chunk.
        self._conv_window_once = tpc.gdn_conv_window_once()
        self._defer_conv_window = False
        # QWEN36_PREFILL_BUCKET_TRACE: (mask_f32, mask_q, conv_sel) persistent device buffers, set by the model
        # ONLY around the traced masked-bucket body (warm pass + capture) and None otherwise, so every other
        # forward_prefill call runs exactly as before. An attribute, not a layer.py kwarg: layer.py is image/overlay
        # owned and forwards valid_len only.
        self._mb_gdn_masks = None
        # QWEN36_PREFILL_BUCKET_KDA (boot time, tier 2): inside the traced masked bucket (gdn_masks set) the conv
        # output comes from the kda op the full chunks already use, instead of the MAC FIR; the decode conv window
        # keeps the conv_sel one-hot (bitwise). Off = the masked branch byte for byte. Needs QWEN36_GDN_CONV_KDA=1.
        self._mb_kda = _tt_public_getenv("QWEN36_PREFILL_BUCKET_KDA", "0").strip() == "1"
        if self._mb_kda and not (self._gdn_conv_kda and self.K == 4):
            raise ValueError(
                "QWEN36_PREFILL_BUCKET_KDA=1 needs QWEN36_GDN_CONV_KDA=1 and a K=4 conv "
                f"(QWEN36_GDN_CONV_KDA={int(self._gdn_conv_kda)}, K={self.K}): refusing"
            )

    def reset_state(self):
        def z(shape):
            return ttnn.from_torch(
                torch.zeros(*shape, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.conv_states = [z((1, self.B, self.qkv_dim_tp)) for _ in range(self.K)]
        # fp32 recurrent state by default (QWEN35_GDN_STATE_BF16=1 reverts)
        if _tt_public_getenv("QWEN35_GDN_STATE_BF16") != "1":
            self.rec_state = ttnn.from_torch(
                torch.zeros(self.B, self.Nv, self.Dk, self.Dv, dtype=torch.float32),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        else:
            self.rec_state = z((self.B, self.Nv, self.Dk, self.Dv))
        # Cross-chunk conv carry + persistent zero sources (created before any trace)
        self.conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_conv0 = z((1, self.B, self.qkv_dim_tp))
        self._zero_conv_carry = z((1, self.K - 1, self.qkv_dim_tp))
        self._zero_rec = z((self.B, self.Nv, self.Dk, self.Dv))
        # Chunk-outer batched-prefill conv left-context (allocated lazily by forward_prefill_batched).
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
        self._batched_conv_carry = None

    def reset_state_inplace(self):
        """Zero conv + recurrent state in place (preserves trace buffer addresses).

        Copies from preallocated _zero_* buffers only — never allocates during an active trace.
        """
        # Drop any chunk-outer batched-prefill conv left-context so the next sequence starts clean.
        if getattr(self, "_batched_conv_carry", None) is not None:
            ttnn.deallocate(self._batched_conv_carry)
            self._batched_conv_carry = None
        if self.conv_states is None:
            self.reset_state()
            return
        # Zero sources must exist (reset_state runs first; no lazy alloc during trace)
        assert (
            self._zero_conv0 is not None and self._zero_conv_carry is not None and self._zero_rec is not None
        ), "zero sources missing; reset_state must run before reset_state_inplace"
        for cs in self.conv_states:
            ttnn.copy(self._zero_conv0, cs)
        ttnn.copy(self._zero_rec, self.rec_state)
        # Zero cross-chunk conv carry for new sequence
        ttnn.copy(self._zero_conv_carry, self.conv_carry)

    def _col_proj(self, x, weight, decode_progcfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG):
        """Column-parallel qkvz projection; DRAM-sharded decode matmul when enabled.
        out_memory_config: decode result placement (default DRAM; L1 keeps it resident)."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=out_memory_config)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
            decode_out_memory_config=out_memory_config,
        )

    def _conv1d_prefill(self, qkv, T, conv_state):
        """Depthwise causal conv1d + SiLU via ttnn.conv1d. Returns (out [1,T,C], new_state [1,K-1,C]) DRAM TILE.

        Prepends K-1 carry rows with padding=0 so one program serves every chunk (native pad only zeros,
        so it can't inject cross-chunk carry into a shared trace).
        """
        dev, K, C = self.mesh, self.K, self.qkv_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        Lin = (K - 1) + T
        # new_state: last K-1 real input tokens (for the next chunk's carry), TILE/DRAM.
        new_state = ttnn.slice(qkv, (0, T - (K - 1), 0), (1, T, C))
        new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
        if conv_state is None:
            pad = ttnn.zeros(
                [1, K - 1, C], device=dev, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, memory_config=_dram
            )
            xin = ttnn.concat([pad, qkv], dim=1, memory_config=_dram)
            ttnn.deallocate(pad)
        else:
            xin = ttnn.concat([conv_state, qkv], dim=1, memory_config=_dram)
        xin = ttnn.to_layout(xin, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        xin = ttnn.reshape(xin, (1, Lin, 1, C))
        cc = ttnn.init_device_compute_kernel_config(
            dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        # Needs l1_small_size on the device (prefill/demo set 24576); matches the validated A/B config.
        conv_cfg = ttnn.Conv1dConfig(
            weights_dtype=ttnn.bfloat16,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        )
        # Prepare conv weight once (warmup); avoids host reprocess + keeps traced replay device-only.
        if self._conv1d_wprep is None:
            self._conv1d_wprep = ttnn.prepare_conv_weights(
                weight_tensor=self.tw["conv_w1d"],
                input_memory_config=_dram,
                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                weights_format="OIHW",
                in_channels=C,
                out_channels=C,
                batch_size=1,
                input_height=1,
                input_width=Lin,
                kernel_size=(1, K),
                stride=(1, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=False,
                groups=C,
                device=dev,
                input_dtype=ttnn.bfloat16,
                conv_config=conv_cfg,
                compute_config=cc,
            )
        out = ttnn.conv1d(
            input_tensor=xin,
            weight_tensor=self._conv1d_wprep,
            device=dev,
            in_channels=C,
            out_channels=C,
            batch_size=1,
            input_length=Lin,
            kernel_size=K,
            stride=1,
            padding=0,
            dilation=1,
            groups=C,
            dtype=ttnn.bfloat16,
            conv_config=conv_cfg,
            compute_config=cc,
            # L1_FULL slice: keep the conv in L1 instead of DRAM-width-slicing. The DRAM-slice path does
            # host reads that begin_trace_capture rejects (see uniad); L1_FULL is trace-safe (as UNet).
            slice_config=ttnn.Conv2dL1FullSliceConfig,
            return_output_dim=False,
            return_weights_and_bias=False,
        )
        ttnn.deallocate(xin)
        out = ttnn.sharded_to_interleaved(out, _dram)
        out = ttnn.reshape(out, (1, T, C))
        out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=_dram)
        # SiLU stays separate (folding via conv_config.activation drops PCC to ~0.84 on this depthwise).
        return ttnn.silu(out, memory_config=_dram), new_state

    def _conv_kda_prefill(self, qkv, T, conv_state, want_state=True):
        """QWEN36_GDN_CONV_KDA: (q, k, v, new_state). q/k/v flat [1,T,width] TILE DRAM with SiLU applied;
        new_state = the last K-1 raw inputs [1,K-1,C] TILE DRAM, exactly as _conv1d_prefill builds it.
        want_state=False (QWEN36_PREFILL_BUCKET_KDA, padded bucket: the static last rows are the wrong ones): no
        new_state ops at all, new_state is None. Every other caller passes the default, unchanged."""
        K, C, kd = self.K, self.qkv_dim_tp, self.key_dim_tp
        _dram = ttnn.DRAM_MEMORY_CONFIG
        if qkv.layout == ttnn.ROW_MAJOR_LAYOUT:
            # QWEN36_PREFILL_GDN_QKV_RM: qkv already IS x_rm (same bytes as to_layout of the TILE slice, gdn_qrm
            # self-test); the carry comes from it as under QWEN36_GDN_CONV_STATE_RM (bitwise, that flag's note).
            x_rm = qkv
            new_state = None
            if want_state:
                new_state = ttnn.slice(x_rm, (0, T - (K - 1), 0), (1, T, C), memory_config=_dram)
                new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
                if tpc.gdn_conv_state_rm():
                    # the same carry ops as the QWEN36_GDN_CONV_STATE_RM branch below, so its banner holds here too
                    tpc._smallops_banner("gdn_conv_state_rm", "QWEN36_GDN_CONV_STATE_RM engaged: kda conv carry = "
                                         "last K-1 rows of the row-major x_rm (no untilize of the full TILE qkv)")
        elif not want_state:
            new_state = None
            x_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        elif tpc.gdn_conv_state_rm():
            # QWEN36_GDN_CONV_STATE_RM (c105): row T-(K-1) is not tile aligned, so slicing the TILE qkv falls back
            # to untilizing all of it. x_rm is the same values already row major (bf16 untilize is lossless), so
            # the last K-1 rows come from there: identical [1,K-1,C] values, one small tilize, no second untilize.
            x_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
            new_state = ttnn.slice(x_rm, (0, T - (K - 1), 0), (1, T, C), memory_config=_dram)
            new_state = ttnn.to_layout(new_state, ttnn.TILE_LAYOUT, memory_config=_dram)
            tpc._smallops_banner("gdn_conv_state_rm", "QWEN36_GDN_CONV_STATE_RM engaged: kda conv carry = last "
                                 "K-1 rows of the row-major x_rm (no untilize of the full TILE qkv)")
        else:
            new_state = ttnn.slice(qkv, (0, T - (K - 1), 0), (1, T, C))
            new_state = ttnn.to_memory_config(ttnn.to_layout(new_state, ttnn.TILE_LAYOUT), _dram)
            x_rm = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        if conv_state is None:
            hist = ttnn.zeros(
                [1, K - 1, C], device=self.mesh, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram
            )
        else:
            hist = ttnn.to_layout(conv_state, ttnn.ROW_MAJOR_LAYOUT, memory_config=_dram)
        taps = self.tw["conv_taps"]
        # Read per call (workbench knob, c87): the post-kda 30k profile has the op at ~0.52 ms/layer/chunk, 3.5x the
        # old depthwise conv alone; channel_chunk_size is its only tiling knob. Invalid values fall back to the init one.
        _ch = int(_tt_public_getenv("QWEN36_GDN_CONV_KDA_CHUNK", str(self._kda_chunk)))
        if _ch % 32 or C % _ch:
            _ch = self._kda_chunk

        def _kda(h):
            return ttnn.experimental.kda.qkv_causal_conv1d_silu(
                x_rm,
                h,
                taps[0],
                taps[1],
                taps[2],
                taps[3],
                kd,
                kd,
                C - 2 * kd,
                program_config=ttnn.QkvCausalConv1dSiluProgramConfig(channel_chunk_size=_ch),
                memory_config=_dram,
                compute_kernel_config=tpc.kda_conv_cfg(self._kda_cc),
            )

        if tpc.gdn_conv_kda2_mode() is not None:
            # QWEN36_GDN_CONV_KDA2 (patch 0035): once per process, on this first (eager warmup) call, the read-once
            # program is compared bitwise with the stock one on these live inputs, with this call's history and
            # with the first K-1 rows of x as a nonzero history (exercises the history rows of every block).
            def _case(name):
                if name == "hist":
                    return _kda(hist)
                h2 = ttnn.slice(x_rm, (0, 0, 0), (1, K - 1, C), memory_config=_dram)
                try:
                    return _kda(h2)
                finally:
                    ttnn.deallocate(h2)

            tpc.kda2_selftest(_case, ("hist", "hist_x0"), f"T={T} C={C} chunk={_ch}")
        q, k, v = _kda(hist)
        if x_rm is not qkv:  # QWEN36_PREFILL_GDN_QKV_RM: the caller frees qkv (= x_rm)
            ttnn.deallocate(x_rm)
        ttnn.deallocate(hist)
        return q, k, v, new_state

    def mb_kda_selftest(self, T, mustfail=False):
        """QWEN36_PREFILL_BUCKET_KDA in-boot self-test, eager, BEFORE any bucket trace is captured; raises -> the
        boot is refused. On THIS layer's device taps (never host weights: a warm-cache boot has placeholders there)
        and synthetic distinct-row activations, on every device of the mesh:
          state   _conv_sel_state == the shipped _causal_conv1d_fir_conv_sel state, bitwise, at actual_len 1, 2,
                  T//2+5 and T (1 and 2 take rows from the history); a one-hot shifted by one row must DIFFER.
          conv    kda vs the MAC FIR vs an fp64 reference (masked_bucket_trace.kda_conv_gate): kda's max error at
                  most 2x the FIR's; the in-boot wrong-input branch (kda with a zero history) must FAIL that gate.
          causal  kda rows < a are bitwise unchanged when every row >= a is replaced (padding never leaks).
        mustfail (QWEN36_PREFILL_BUCKET_KDA_MUSTFAIL=1): the candidate runs with the zero history, so the conv gate
        fails and the boot is refused. Returns the per-device summary."""
        import importlib

        mbt = importlib.import_module("models.demos.blackhole.qwen36.tt.masked_bucket_trace")
        K, C = self.K, self.qkv_dim_tp
        mesh, rep = self.mesh, ttnn.ReplicateTensorToMesh(self.mesh)
        _dram, _l1, _tile = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG, ttnn.TILE_LAYOUT
        gen = torch.Generator().manual_seed(20261015 + T)
        x_h = (torch.randn(1, T, C, generator=gen) * 0.5).to(torch.bfloat16)
        x2_h = x_h.clone()
        a_causal = min(T, T // 2 + 5)
        x2_h[:, a_causal:] = (torch.randn(1, T - a_causal, C, generator=gen) * 0.5).to(torch.bfloat16)
        st_h = (torch.randn(1, K - 1, C, generator=gen) * 0.5).to(torch.bfloat16)

        def up(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=_tile, device=mesh, mesh_mapper=rep,
                                   memory_config=_dram)

        def dev(t):
            return [ttnn.to_torch(d) for d in ttnn.get_device_tensors(t)]

        def bits_differ(a, b):
            return sum(int((x.reshape(-1).view(torch.int16) != y.reshape(-1).view(torch.int16)).sum())
                       if x.shape == y.shape else x.numel() for x, y in zip(a, b))

        x, x2, st = up(x_h), up(x2_h), up(st_h)
        st0 = up(torch.zeros_like(st_h))
        taps = [torch.cat([w.reshape(1, -1) for w in ws], 0) for ws in zip(*[dev(t) for t in self.tw["conv_taps"]])]
        out = dict(T=T, devices=len(taps), state={}, conv=[], causal_diff=None, mustfail=bool(mustfail))
        try:
            ref_conv = None
            for a in sorted({1, 2, a_causal, T}):
                sel, wrong = (up(mbt.host_conv_sel(a, T, K)),
                              up(mbt.host_conv_sel(a + 1 if a < T else a - 1, T, K)))
                fir_conv, ref_state = _causal_conv1d_fir_conv_sel(x, K, mesh, _l1, st, self.tw["conv_taps"], sel)
                cand_state = _conv_sel_state(x, mesh, _l1, st, sel)
                wrong_state = _conv_sel_state(x, mesh, _l1, st, wrong)
                r, c, w = dev(ref_state), dev(cand_state), dev(wrong_state)
                out["state"][a] = dict(diff=bits_differ(r, c), wrong_diff=bits_differ(r, w))
                for t in (sel, wrong, ref_state, cand_state, wrong_state):
                    ttnn.deallocate(t)
                if ref_conv is None:
                    ref_conv = dev(fir_conv)
                ttnn.deallocate(fir_conv)
            hist_c, hist_w = (st0, st) if mustfail else (st, st0)
            cq = self._conv_kda_prefill(x, T, hist_c, want_state=False)[:3]
            wq = self._conv_kda_prefill(x, T, hist_w, want_state=False)[:3]
            pq = self._conv_kda_prefill(x2, T, hist_c, want_state=False)[:3]
            cand = [torch.cat(p, dim=-1) for p in zip(*[dev(t) for t in cq])]
            wrongc = [torch.cat(p, dim=-1) for p in zip(*[dev(t) for t in wq])]
            pad = [torch.cat(p, dim=-1) for p in zip(*[dev(t) for t in pq])]
            for t in list(cq) + list(wq) + list(pq):
                ttnn.deallocate(t)
            out["causal_diff"] = bits_differ([c[..., :a_causal, :] for c in cand], [p[..., :a_causal, :] for p in pad])
            for d in range(len(taps)):
                ref64 = mbt.conv_ref64(x_h[0], st_h[0], taps[d].reshape(K, -1))
                g = mbt.kda_conv_gate(ref64, ref_conv[d].reshape(T, -1), cand[d].reshape(T, -1),
                                      wrongc[d].reshape(T, -1))
                out["conv"].append(g)
        finally:
            for t in (x, x2, st, st0):
                ttnn.deallocate(t)
        bad = [a for a, s in out["state"].items() if s["diff"] != 0 or s["wrong_diff"] == 0]
        conv_bad = [d for d, g in enumerate(out["conv"]) if not g["ok"]]
        if bad or conv_bad or out["causal_diff"] != 0:
            raise RuntimeError(
                f"QWEN36_PREFILL_BUCKET_KDA self-test REFUSED (T={T}, mustfail={bool(mustfail)}): state actual_len "
                f"{bad} (diff/wrong_diff {out['state']}), conv devices {conv_bad} ({out['conv']}), causal diff "
                f"{out['causal_diff']}: refusing the boot"
            )
        logger.info(
            f"QWEN36_PREFILL_BUCKET_KDA self-test pass: T={T} on {len(taps)} devices; state bitwise at actual_len "
            f"{sorted(out['state'])}, shifted one-hot differs; kda max err "
            f"{max(g['err_cand'] for g in out['conv']):.3g} vs FIR {max(g['err_fir'] for g in out['conv']):.3g} "
            f"(fp64 ref); wrong-input branch (zero history) refused, err {min(g['err_wrong'] for g in out['conv']):.3g}"
            f"; padding rows never reach rows < {a_causal}"
        )
        return out

    def _row_proj(self, x, weight):
        """Row-parallel out projection: DRAM-sharded decode/prefill matmul (K=gdn_value_dim_tp),
        matching the in-proj. Falls back to plain interleaved on single device (no sharded memcfg)."""
        if self._mr_out is not None and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode, QWEN36_DECODE_DRAM_SHARDED_MR: the decode-only sharded copy -> DRAM for the reduce.
            # QWEN36_DECODE_MR_OUT_DIRECT: straight into the all_reduce_async input layout (else DRAM, as before)
            return tpc.decode_mr_matmul(
                x, self.tw["out_mr"], self._mr_out, self.cfg,
                out_memory_config=tpc.mr_ar_target(self.tt_ccl, ttnn.DRAM_MEMORY_CONFIG),
            )
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x, weight, self.args.gdn_out_decode_1d_progcfg, self.cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if not self._out_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill non-fused arm (single device, or out-sharded): tuned 2D config vs ttnn-auto.
                # fp32 [seq,dim] output too big for L1 (42MB) -> DRAM out; separate tt_all_reduce does the RS.
                # max_cols = device width (11 on BH): wide grid (~10-wide), fp32-neutral.
                if tpc.prefill_fs8("gdn_out"):
                    return ttnn.linear(
                        x,
                        weight,
                        compute_kernel_config=tpc.prefill_fs8_compute(packer_l1_acc=True),
                        program_config=tpc.create_prefill_fs8_program_config(
                            x.shape[-2], weight.shape[-2], weight.shape[-1],
                            getattr(self.args, "decode_grid_w", 8), "gdn_out",
                        ),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x, weight, compute_kernel_config=self.cfg, program_config=pc, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.cfg,
            self.args.gdn_out_progcfg,
            self.args.act_shard_gdn_value,
            self.args.prefill_progcfg,
            self.args.gdn_value_dim_tp,
        )

    def _gate_prefill_g(self, a):
        """g = -exp(A_log) * softplus(a + dt_bias) for the PREFILL rows.

        Under QWEN36_GDN_G_FP32 the gate runs in fp32 against the fp32 A_log/dt_bias copies; the
        fused chunk op already wants fp32 g ("op contract -- load-bearing", fused_chunk.py). The
        decode paths keep the bf16 tensors and do not call this."""
        if not self._g_fp32:
            return ttnn.multiply(self.tw["neg_exp_A"], _softplus_add(a, self.tw["dt_bias"]))
        a32 = ttnn.typecast(a, ttnn.float32)
        sp = _softplus_add(a32, self.tw["dt_bias_f32"])
        ttnn.deallocate(a32)
        g = ttnn.multiply(self.tw["neg_exp_A_f32"], sp)
        ttnn.deallocate(sp)
        return g

    def _project_ab8(self, x, S, out_mc, kshard):
        """a, b from the separate bfloat8_b [a|b] weight. Prefill only.

        kshard: x is K-sharded (the fused-AGMM prefill path, where the norm skipped its
        all-gather), so it is gathered to full width first. Prod imports only tt_all_reduce from
        models.tt_transformers.tt.ccl, so tt_all_gather is added to that SAME import."""
        Nv = self.Nv
        xin = x
        if kshard:
            # The norm skipped its all-gather on this path, so x is [.., dim/tp]. Gather it to full
            # width and use the column-sharded weight -- each device then produces exactly its OWN
            # a/b columns, matching the per-device [a_d|b_d] layout the fused weight hands back.
            # (A row-parallel matmul + all_reduce would instead give every device ALL devices' a/b.)
            # tt_all_gather force-frees its INPUT; gather a clone so the caller's x (still owned by
            # layer.py) is not freed under it ("Tensor is not allocated" in the ttsim run).
            xin = tt_all_gather(
                ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                self.mesh,
                self.tt_ccl,
                cluster_axis=None,
                dim=-1,
                topology=self.args.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        ab = ttnn.linear(
            xin, self.tw["ab8_col"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        if xin is not x:
            ttnn.deallocate(xin)
        ab = ttnn.reshape(ab, (1, S, ab.shape[-1]))
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv), memory_config=out_mc)
        ttnn.deallocate(ab)
        return a, b

    def finalize_conv_window(self):
        """QWEN36_GDN_CONV_WINDOW_ONCE (c105): write the decode conv window from the cross-chunk carry, once, after
        the last prefill chunk. conv_carry is a ttnn.copy of the last chunk's conv_new_state (same dtype/shape), so
        this is the same op sequence the carry-path chunk ran with the same source values: bit-exact. No-op unless
        the carry path is live (_stable_state) — the only path that skips the per-chunk write."""
        if not (self._conv_window_once and self._stable_state and self.conv_carry is not None
                and self.conv_states is not None):
            return
        B, D = 1, self.qkv_dim_tp
        if self._zero_conv0 is not None:
            ttnn.copy(self._zero_conv0, self.conv_states[0])
        else:
            zero = ttnn.from_torch(
                torch.zeros(1, B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
            ttnn.copy(zero, self.conv_states[0])
            ttnn.deallocate(zero)
        for j in range(self.K - 1):
            src = ttnn.reshape(ttnn.slice(self.conv_carry, (0, j, 0), (1, j + 1, D)), (1, B, D))
            ttnn.copy(src, self.conv_states[j + 1])

    def _project_qkvzab_chunked(self, x, S, out_mc, proj_mc, chunk_sizes, ab8):
        """QWEN36_AGMM_CHUNKS=gdn (c105), prefill fused-AGMM path only: the AGMM writes [qkv | z | ab] as three
        tensors (one memory config for all = proj_mc, L1 here), so the qkv and ab slice copies go away. z keeps its
        DRAM placement (L1 z clashes with the scan kernel's CBs, see _project_qkvzab) through one to_memory_config,
        which replaces the z slice copy. a/b are split from the ab chunk exactly as from the tile-aligned ab block
        before. Same matmul program (grid, blocks, K order): only the writer's destination differs, so bit-exact.
        Peak L1 drops: the full qkvzab and the qkv slice no longer coexist."""
        Nv = self.Nv
        qkv, z_l1, ab = tpc.all_gather_matmul_prefill(
            x,
            self.tw["qkvz"],
            self.tt_ccl,
            self.cfg,
            self.args.ccl_topology(),
            out_memory_config=proj_mc,
            chunk_sizes=chunk_sizes,
            role="gdn_in",
        )
        tpc.agmm_chunks_banner("gdn", chunk_sizes, "L1 (z to DRAM)")
        qkv = ttnn.reshape(qkv, (1, S, chunk_sizes[0]))
        z = ttnn.to_memory_config(ttnn.reshape(z_l1, (1, S, chunk_sizes[1])), ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(z_l1)
        ab = ttnn.reshape(ab, (1, S, chunk_sizes[2]))
        _gap = getattr(self.args, "gdn_ab_gap", 0)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
        b = ttnn.slice(ab, (0, 0, Nv + _gap), (1, S, 2 * Nv + _gap), memory_config=out_mc)
        ttnn.deallocate(ab)
        if ab8 and self._ab8:  # 8-bit a/b, prefill only (opt-in per call site)
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            a, b = self._project_ab8(x, S, out_mc, kshard=True)
        return qkv, z, a, b

    def _split_qkvzab(self, qkvzab, S, out_mc, z_mc, free=False, qrm=False):
        """The fused projection's five slice copies -> (qkv, z, a, b). The served path (free=True: qkvzab is freed
        right after the ab slice, as before this refactor -- same ops, same order, same allocations) and the
        QWEN36_GDN_DECODE_CHAIN self-test's reference (free=False).
        qrm (QWEN36_PREFILL_GDN_QKV_RM, full-chunk kda prefill only): qkv comes back ROW_MAJOR in DRAM straight from
        qkvzab (gdn_qrm: slice + untilize in one program); _conv_kda_prefill takes it as its x_rm."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        if qrm and gdn_qrm.usable(qkvzab, qz):
            qkv = gdn_qrm.run(qkvzab, qz)
        else:
            qkv = ttnn.slice(qkvzab, (0, 0, 0), (1, S, qz), memory_config=out_mc)
        z = ttnn.slice(qkvzab, (0, 0, qz), (1, S, az), memory_config=z_mc)
        # a,b end mid-tile; slicing straight from qkvzab untilizes the full 4120-wide tensor.
        # Grab the enclosing tile-aligned block once (no untilize), then split a/b from it (test_gdn_slice_opt).
        _gap = getattr(self.args, "gdn_ab_gap", 0)  # QWEN36_GDN_AB_PAD: b starts at Nv + gap (tile-aligned)
        _ab_end = min(az + -(-(2 * Nv + _gap) // tpc.TILE_SIZE) * tpc.TILE_SIZE, qkvzab.shape[-1])  # up to a tile
        ab = ttnn.slice(qkvzab, (0, 0, az), (1, S, _ab_end), memory_config=out_mc)
        if free:
            ttnn.deallocate(qkvzab)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv), memory_config=out_mc)
        b = ttnn.slice(ab, (0, 0, Nv + _gap), (1, S, 2 * Nv + _gap), memory_config=out_mc)
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def _project_qkvzab(self, x, S, out_mc=None, ab8=False, raw=False, qrm=False):
        """Project x → (qkv, z, a, b). Fused path: one [qkv|z|a|b] matmul then slice.
        out_mc: placement of the qkvzab matmul + slices. None → DRAM; prefill+decode now pass L1 to
        keep qkvzab + q/k/v/z/a/b resident (was DRAM to spare NoC traffic — re-measure if reverting).
        raw=True (QWEN36_GDN_DECODE_CHAIN=direct, decode only): return the unsliced qkvzab [1,S,W] itself."""
        Nv, qz, az = self.Nv, self.qkv_dim_tp, self.qkvz_dim_tp
        _proj_mc = out_mc if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG
        if self._fuse_ab:
            # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + qkvzab matmul.
            if self._fuse_agmm and S > tpc.TILE_SIZE:
                _N = self.tw["qkvz"].shape[-1]
                _cs = [qz, az - qz, _N - az]
                if tpc.agmm_chunks("gdn"):
                    if all(c > 0 and c % tpc.TILE_SIZE == 0 for c in _cs):
                        return self._project_qkvzab_chunked(x, S, out_mc, _proj_mc, _cs, ab8)
                    tpc._smallops_banner(("agmm_chunks_fallback", "gdn"), f"QWEN36_AGMM_CHUNKS fallback: role=gdn "
                                         f"chunk_sizes={_cs} not tile aligned (needs QWEN36_GDN_AB_PAD); slicing kept")
                qkvzab = tpc.all_gather_matmul_prefill(
                    x,
                    self.tw["qkvz"],
                    self.tt_ccl,
                    self.cfg,
                    self.args.ccl_topology(),
                    out_memory_config=_proj_mc,
                    role="gdn_in",
                )
                qkvzab = ttnn.reshape(qkvzab, (1, S, qkvzab.shape[-1]))
            elif self._mr_qkvzab is not None and S <= tpc.TILE_SIZE:
                # Decode, QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded matmul, same placement.
                qkvzab = tpc.decode_mr_matmul(
                    x,
                    self.tw["qkvz_mr"],
                    self._mr_qkvzab,
                    self.cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            elif getattr(self.args, "proj_1d_decode", False) and S <= tpc.TILE_SIZE:
                # Decode: small-grid 1D matmul on the interleaved fused weight (beats the DRAM-sharded grid).
                qkvzab = tpc.matmul_1d_decode(
                    x,
                    self.tw["qkvz"],
                    self.args.gdn_qkvz_decode_1d_progcfg,
                    self.cfg,
                    out_memory_config=ttnn.L1_MEMORY_CONFIG if out_mc is not None else ttnn.DRAM_MEMORY_CONFIG,
                )
            else:
                qkvzab = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvzab_progcfg, out_memory_config=_proj_mc)
            if raw:
                if S > tpc.TILE_SIZE or ab8:
                    raise RuntimeError(f"_project_qkvzab(raw=True) is decode only (S={S} <= 32, no ab8)")
                return qkvzab
            # z (output gate) lives across the chunk kernel (gated = out_f * silu(z)); L1 z (6MB@S=2048)
            # clashes with the scan kernel CBs -> keep DRAM in chunk-prefill; decode (small S) keeps out_mc.
            _z_mc = ttnn.DRAM_MEMORY_CONFIG if (self._fuse_agmm and S > tpc.TILE_SIZE) else out_mc
            qkv, z, a, b = self._split_qkvzab(qkvzab, S, out_mc, _z_mc, free=True, qrm=qrm)
            if ab8 and self._ab8:  # 8-bit a/b, prefill only (opt-in per call site)
                ttnn.deallocate(a)
                ttnn.deallocate(b)
                a, b = self._project_ab8(x, S, out_mc, kshard=bool(self._fuse_agmm and S > tpc.TILE_SIZE))
            return qkv, z, a, b
        qkvz = self._col_proj(x, self.tw["qkvz"], self.args.gdn_qkvz_progcfg)
        qkv = ttnn.slice(qkvz, (0, 0, 0), (1, S, qz))
        z = ttnn.slice(qkvz, (0, 0, qz), (1, S, az))
        ttnn.deallocate(qkvz)
        ab = ttnn.linear(x, self.tw["ab"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        a = ttnn.slice(ab, (0, 0, 0), (1, S, Nv))
        b = ttnn.slice(ab, (0, 0, Nv), (1, S, 2 * Nv))
        ttnn.deallocate(ab)
        return qkv, z, a, b

    def forward_prefill(
        self, x, chunk_size=128, valid_len=None, capture_state=False, return_state=False, gdn_masks=None
    ):
        """Causal chunk-prefill from scratch. x [1,1,T,dim]: K-sharded (dim/tp per device) when the
        fused in-proj AG-matmul path is active (``_fuse_agmm`` and T>TILE — the norm skips its
        post-AG); replicated otherwise. Output reduce-scattered.

        valid_len: real token count (rest is padding). capture_state: save rec/conv state for decode.
        return_state: when True (per-user batched prefill), return
        ``(output, final_state, conv_new_state)`` for one user's from-scratch B=1
        pass and skip all self.* writeback; the caller stitches per-user states via
        assemble_batched_state(). Single-sequence behavior is unchanged when False.

        gdn_masks: None (every eager caller; falls back to self._mb_gdn_masks, also None outside the traced
        bucket body) or (mask_f32, mask_q, conv_sel) persistent device tensors for the traced masked-bucket
        prefill (QWEN36_PREFILL_BUCKET_TRACE). When supplied, validity masking is a device multiply and
        the conv-state select a device matmul, so no int valid_len reaches library code (whose masking does
        host writes). When None, every line below runs exactly as before this parameter existed.
        """
        if gdn_masks is None:
            gdn_masks = self._mb_gdn_masks
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        T = x.shape[1]
        # Pass the RAW valid_len (may be None) to the conv-FIR / seq kernels below — NOT a
        # `valid_len or T` coercion. A full chunk (valid_len is None) must take the kernels'
        # valid_len-None path (a static last-(K-1) slice for the conv state), which is trace-safe;
        # the valid_len-set path builds a one-hot via ttnn.from_torch (a host write) that TT_FATALs
        # ("Writes are not supported during trace capture") inside the captured chunk-outer trace.
        # Masked buckets still pass a real valid_len (< T) so their exact masking is unchanged, and
        # for a full chunk the None slice and the valid_len==T one-hot select the identical rows.

        # Cross-chunk carry (chunk-outer prefill): when _stable_state, the recurrent + conv
        # state continue from the persistent buffers (zeroed at sequence start by
        # reset_state_inplace, so a from-scratch single pass reads zeros == None). The demo
        # path (_stable_state False) is unchanged: no carry, reassign state.
        # Per-user prefill (return_state) is always from scratch: must not carry the shared
        # batched buffer (other users' state) as its initial recurrent/conv state.
        carry = self._stable_state and not return_state
        if carry and self.conv_carry is None:
            self.reset_state()

        if gdn_masks is not None:
            _mask_f32, _mask_q, _conv_sel = gdn_masks
            assert not return_state, "gdn_masks is the single-sequence traced path (no per-user collect)"
            assert carry and self.conv_carry is not None, "gdn_masks needs persistent conv carry (_stable_state)"
            assert self._zero_conv0 is not None, "gdn_masks needs persistent zero conv row"
            assert tw["conv_taps"] is not None, "gdn_masks needs pre-built conv taps"
            valid_len = None  # shapes only from here on; values come from the masks

        # Prefill qkvzab in L1: keeps proj + q/k/v/z/a/b resident for conv+gate prep.
        # QWEN36_PREFILL_GDN_QKV_RM: the full-chunk kda conv reads only a row-major copy of qkv, so qkv is produced
        # row major (DRAM) by one program instead of slice (L1 TILE) + to_layout. Masked buckets read the TILE qkv.
        _qrm = (gdn_masks is None and self._gdn_conv_kda and valid_len is None and self.K == 4
                and gdn_qrm.enabled())
        qkv, z, a, b = self._project_qkvzab(x, T, out_mc=ttnn.L1_MEMORY_CONFIG, ab8=True, qrm=_qrm)

        # FIR conv1d; conv_state = previous chunk's last K-1 inputs (None/zero from scratch)
        _cstate = self.conv_carry if carry else None
        _kda_qkv = None
        if gdn_masks is not None and self._mb_kda:
            # QWEN36_PREFILL_BUCKET_KDA: conv output from the kda op (causal, so the bucket's padding rows never reach
            # a real row; they are masked below like the FIR's), decode window from the conv_sel one-hot (bitwise).
            conv_new_state = _conv_sel_state(qkv, self.mesh, ttnn.L1_MEMORY_CONFIG, _cstate, _conv_sel)
            *_kda_qkv, _unused = self._conv_kda_prefill(qkv, T, _cstate, want_state=False)
            conv = None
        elif gdn_masks is not None:
            # Traced masked bucket: the MAC FIR the eager masked bucket also uses (never kda / conv1d, whose
            # conv state is a static last-(K-1) slice), with the state picked by the conv_sel one-hot matmul.
            conv, conv_new_state = _causal_conv1d_fir_conv_sel(
                qkv, self.K, self.mesh, ttnn.L1_MEMORY_CONFIG, _cstate, tw["conv_taps"], _conv_sel
            )
        elif self._gdn_conv_kda and valid_len is None and self.K == 4:
            *_kda_qkv, conv_new_state = self._conv_kda_prefill(qkv, T, _cstate)
            conv = None
        elif self._gdn_conv1d and valid_len is None:
            # Native depthwise ttnn.conv1d (masked buckets keep the MAC FIR: valid_len new_state differs)
            conv, conv_new_state = self._conv1d_prefill(qkv, T, _cstate)
        else:
            conv, conv_new_state = _causal_conv1d_fir(
                qkv,
                None,
                None,
                self.K,
                self.mesh,
                # Conv in L1 (output freed before chunk kernel; new_state lands in DRAM internally)
                memory_config=ttnn.L1_MEMORY_CONFIG,
                conv_state=_cstate,
                weight_taps=tw["conv_taps"],
                bias_dev=None,
                valid_len=valid_len,
            )
        ttnn.deallocate(qkv)

        # q/k/v/beta/g stay DRAM — alive across chunk kernel; L1 crashes it.
        kd = self.key_dim_tp
        if _kda_qkv is not None:
            q, k, v = _kda_qkv
            if self._gdn_flat_qkv:
                _qkv_head_dims = (Nk, Dk, Nv, Dv)
            else:
                q = ttnn.reshape(q, (1, T, Nk, Dk))
                k = ttnn.reshape(k, (1, T, Nk, Dk))
                v = ttnn.reshape(v, (1, T, Nv, Dv))
                _qkv_head_dims = None
        elif self._gdn_flat_qkv:
            # Flat q/k/v: adapter splits heads inside untilize
            q = ttnn.slice(conv, (0, 0, 0), (1, T, kd))
            k = ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd))
            v = ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp))
            _qkv_head_dims = (Nk, Dk, Nv, Dv)
        else:
            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, T, kd)), (1, T, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, T, 2 * kd)), (1, T, Nk, Dk))
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, T, self.qkv_dim_tp)), (1, T, Nv, Dv))
            _qkv_head_dims = None
        if conv is not None:
            ttnn.deallocate(conv)
        # GQA late-expand: adapter L2-norms at Nk, expands to Nv after
        beta = ttnn.reshape(ttnn.sigmoid(b), (1, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(self._gate_prefill_g(a), (1, T, Nv))
        ttnn.deallocate(a)

        if gdn_masks is not None:
            beta, g = _mul_mask(beta, _mask_f32), _mul_mask(g, _mask_f32)
            q, k, v = _mul_mask(q, _mask_q), _mul_mask(k, _mask_q), _mul_mask(v, _mask_q)

        # Fused chunk_gated_delta_rule; also used for masked valid_len.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        # const_tiles only applies to the fused op; the seq adapter has no such param.
        _extra = {"const_tiles": self._fused_const_tiles} if _use_fused else {}
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_len,
            qkv_head_dims=_qkv_head_dims,
            return_o_bh=self._gdn_fuse_out,
            **_extra,
        )
        B, D = 1, self.qkv_dim_tp
        captured = None
        if return_state:
            # Per-user prefill: return this user's state for assemble_batched_state to stitch
            # into the batched buffers. No self.* writeback; tensors are not deallocated here.
            captured = (final_state, conv_new_state)
        else:
            # ---- Carry recurrent + conv state for the NEXT chunk (chunk-outer prefill). ----
            # In place (ttnn.copy) when _stable_state so the addresses the prefill/decode traces
            # baked in stay valid across execute_trace replays and across sequences.
            if carry:
                ttnn.copy(final_state, self.rec_state)
                ttnn.deallocate(final_state)
                ttnn.copy(conv_new_state, self.conv_carry)  # [1, K-1, D] last-K-1 conv inputs
            else:
                self.rec_state = final_state
            # ---- Finalize the decode conv window (last chunk / short prompt). ----
            # conv_states[1..K-1] = the last K-1 real conv inputs; [0] is the (shifted-out) zero.
            # Harmless to refresh every chunk — the last chunk's values are the ones decode reads.
            # QWEN36_GDN_CONV_WINDOW_ONCE: skipped on the carry path while the model defers it; conv_carry (written
            # above every chunk) then holds exactly these rows and finalize_conv_window() writes them once.
            if capture_state and not (carry and self._conv_window_once and self._defer_conv_window):
                if self.conv_states is None:
                    self.reset_state()
                if self._zero_conv0 is not None:
                    ttnn.copy(self._zero_conv0, self.conv_states[0])
                else:
                    zero = ttnn.from_torch(
                        torch.zeros(1, B, D, dtype=torch.bfloat16),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
                    )
                    ttnn.copy(zero, self.conv_states[0])
                    ttnn.deallocate(zero)
                for j in range(self.K - 1):
                    src = ttnn.reshape(ttnn.slice(conv_new_state, (0, j, 0), (1, j + 1, D)), (1, B, D))
                    ttnn.copy(src, self.conv_states[j + 1])
            ttnn.deallocate(conv_new_state)
        # Gated RMSNorm + SiLU(z); norm/flatten in L1, gated output in DRAM for out-proj
        _L1 = ttnn.L1_MEMORY_CONFIG
        _gtl = self._gdn_fuse_out and self._out_colpar_prefill and gdn_tail.usable(o, Nv)
        if _gtl:
            # QWEN36_PREFILL_GDN_TAIL: per-head rms_norm + nlp_concat_heads in ONE program (the writer places each
            # head's tile row at nlp_concat_heads' token-major position), then silu(z) * out in ONE program below.
            # Both bitwise vs the ops they replace (in-boot self-test, both chips).
            gdn_tail.log_engaged(T, Nv, Dv)
            n = gdn_tail.norm_concat(o, tw["norm_w"], 1e-6, Nv, _L1)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        elif self._gdn_fuse_out:
            # Fuse adapter relayout with per-head rms_norm + head-flatten.
            # TILE-native head->token relayout (transpose + fold), dropping the
            # TILE->ROW_MAJOR->TILE round-trip. o is head-major (1,Nv,T,Dv).
            n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            n = ttnn.reshape(n, (1, Nv, T, Dv))
            # Fused head->token relayout: [1,Nv,T,Dv] -> [1,1,T,Nv*Dv].
            n = ttnn.experimental.nlp_concat_heads(n, memory_config=_L1)
            out_f = ttnn.reshape(n, (1, T, self.value_dim_tp))
        else:
            out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o)
            out_f = ttnn.reshape(out_n, (1, T, self.value_dim_tp), memory_config=_L1)
            ttnn.deallocate(out_n)
        if self._out_colpar_prefill:
            # Column-parallel out-proj: the gate multiply emits the AGMM input directly as bf16 (the
            # only numerics change vs the fp32 MMRS arm: activation quantized to bf16 before the
            # matmul, as every other projection in the model already does).
            if _gtl and gdn_tail.gate_usable(out_f, z):
                gated = gdn_tail.gate(out_f, z, _L1)
            else:
                gated = _silu_mul(out_f, z, _L1, dtype=ttnn.bfloat16)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)
            # TODO(#57458): switch to the op's barrier_semaphore once it is wired up (see tpc.agmm_gather_buffer).
            out = tpc.all_gather_matmul_prefill(
                gated,
                tw["out_colpar"],
                self.tt_ccl,
                self.cfg,
                self.args.ccl_topology(),
                out_memory_config=_L1,
                persistent_output_buffer=tpc.agmm_gather_buffer(self.tt_ccl, gated),
                role="gdn_out",
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        gated = _silu_mul(out_f, z, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        # Prefill: fused out-proj matmul + reduce-scatter (matmul_reduce_scatter_async), flag-gated.
        if self._fuse_out_mmrs_prefill:
            x_out = ttnn.reshape(gated, (1, 1, T, gated.shape[-1]))
            # fp32 output is load-bearing: o_proj is row-parallel, so the RS SUMS 4 per-device partials
            # across devices — bf16 there tanks PCC to ~0.69 even at ISL 2048 (test_oproj_dtype_isl). Keep fp32.
            out = tpc.matmul_reduce_scatter_prefill(
                x_out, tw["out"], self.tt_ccl, self.cfg, self.args.ccl_topology(), self.args.num_devices, ttnn.float32
            )
            ttnn.deallocate(gated)
            if return_state:
                return out, captured[0], captured[1]
            return out
        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, T, partial.shape[-1]))
        out = tpc.prefill_all_reduce(partial, self.mesh, self.tt_ccl, self.args.ccl_topology())
        if return_state:
            return out, captured[0], captured[1]
        return out

    def forward_prefill_collect(self, x, chunk_size=128, valid_len=None):
        """Per-user prefill that stashes this user's B=1 state for later assembly.

        Called once per user; finalize_pending() then stitches the collected states into the
        batched decode buffers. Returns the user's prefill output (needed for residual + MLP)."""
        out, rec, conv = self.forward_prefill(x, chunk_size=chunk_size, valid_len=valid_len, return_state=True)
        self._pending.append((rec, conv))
        return out

    def finalize_pending(self):
        """Assemble the per-user states collected by forward_prefill_collect into the batched
        decode buffers (row u = user u), then clear the accumulator."""
        assert self._pending, "finalize_pending called with no collected per-user states"
        rec_list = [r for (r, _) in self._pending]
        conv_list = [c for (_, c) in self._pending]
        self.assemble_batched_state(rec_list, conv_list)
        self._pending = []

    def assemble_batched_state(self, rec_list, conv_new_list):
        """Stitch B per-user prefill states (from forward_prefill(return_state=True)) into the
        batched decode buffers.

        rec_list[u]: [1, Nv, Dk, Dv] recurrent state; conv_new_list[u]: [1, K-1, qkv_dim_tp]
        last-(K-1) conv inputs. Row u of rec_state and conv_states[1..K-1] becomes user u's state;
        conv_states[0] is zeroed (shifted-out tap). ttnn has no in-place row write, so buffers are
        built by concat along the batch dim (rec: dim 0; conv: dim 1).

        Under _stable_state (decode-trace path) the result is copied into the fixed-address
        buffers; otherwise (demo/standalone) it is assigned.
        """
        assert len(rec_list) == self.B and len(conv_new_list) == self.B, "need one state per batch row"
        D = self.qkv_dim_tp
        rec_batched = ttnn.concat(rec_list, dim=0)  # [B, Nv, Dk, Dv]
        conv_states = [
            ttnn.from_torch(
                torch.zeros(1, self.B, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        ]
        for m in range(1, self.K):  # conv_states[m] row u = conv_new_list[u][:, m-1]
            rows = [
                ttnn.reshape(ttnn.slice(conv_new_list[u], (0, m - 1, 0), (1, m, D)), (1, 1, D)) for u in range(self.B)
            ]
            cs = ttnn.concat(rows, dim=1)  # [1, B, D]
            for r in rows:
                ttnn.deallocate(r)
            conv_states.append(cs)

        if self._stable_state and self.rec_state is not None:
            rec_src = (
                rec_batched
                if rec_batched.dtype == self.rec_state.dtype
                else ttnn.typecast(rec_batched, self.rec_state.dtype)
            )
            ttnn.copy(rec_src, self.rec_state)
            if rec_src is not rec_batched:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(rec_batched)
            for m in range(self.K):
                ttnn.copy(conv_states[m], self.conv_states[m])
                ttnn.deallocate(conv_states[m])
        else:
            self.rec_state = rec_batched
            self.conv_states = conv_states
        for t in rec_list:
            ttnn.deallocate(t)
        for t in conv_new_list:
            ttnn.deallocate(t)

    # ------------------------------------------------------------------ #
    # Per-slot state edits for vLLM continuous batching.
    # ------------------------------------------------------------------ #
    # The demo prefills all B users up front and assembles the whole batch at
    # once (assemble_batched_state). vLLM instead prefills ONE user at a time
    # into its decode slot while the other rows are mid-decode, and condenses
    # the batch when a request finishes. GDN's recurrent+conv state is a fixed
    # [B,...] buffer indexed by physical slot (not paged), so both events need a
    # single-row edit that preserves the other (live) rows. ttnn has no in-place
    # row write, so — exactly like assemble_batched_state — these rebuild the
    # buffer by slice+concat and ttnn.copy the result back (the copy preserves
    # the decode trace's baked buffer address).
    def _slice_along(self, buf, dim, lo, hi):
        """ttnn.slice of buf along `dim` for indices [lo, hi), other dims kept full."""
        start = [0] * len(buf.shape)
        end = list(buf.shape)
        start[dim] = lo
        end[dim] = hi
        return ttnn.slice(buf, tuple(start), tuple(end))

    def _write_recurrent_state_prefix(self, new_rec, B):
        """Write active rows [0:B] without reading or copying idle rows."""
        grid_size = self.mesh.compute_with_storage_grid_size()
        assert (
            grid_size.x >= 8 and grid_size.y >= 6
        ), f"GDN prefix state write needs an 8x6 core rectangle, got {grid_size.x}x{grid_size.y}"
        nhw = B * self.Nv * self.Dk
        assert (
            nhw % ttnn.TILE_SIZE == 0
        ), f"GDN prefix state rows B={B}, Nv={self.Nv}, Dk={self.Dk} -> {nhw} is not tile-aligned"
        n_tiles = nhw // ttnn.TILE_SIZE

        # Prefer the tuned 8x6=48-core rectangle, which every TP=4 shape hits (Nv=12 -> nhw=B*1536
        # -> 48*B tiles). At TP=8 Nv halves to 6, so B=1 gives only 24 tiles and cannot fill 48
        # cores with tile-aligned shards — fall back to the largest core count that divides evenly.
        if n_tiles % 48 == 0:
            num_cores = 48
            grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 5))})
        else:
            num_cores = max(c for c in range(1, min(48, grid_size.x * grid_size.y) + 1) if n_tiles % c == 0)
            grid = ttnn.num_cores_to_corerangeset(num_cores, grid_size, row_wise=True)

        shard_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                grid,
                (nhw // num_cores, self.Dv),
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        src = (
            new_rec
            if new_rec.dtype == self.rec_state.dtype
            else ttnn.typecast(new_rec, self.rec_state.dtype, memory_config=ttnn.L1_MEMORY_CONFIG)
        )
        sharded = ttnn.to_memory_config(src, shard_memcfg)
        ttnn.experimental.slice_write(
            sharded,
            self.rec_state,
            [0, 0, 0, 0],
            [B, self.Nv, self.Dk, self.Dv],
            [1, 1, 1, 1],
        )
        ttnn.deallocate(sharded)
        if src is not new_rec:
            ttnn.deallocate(src)
        ttnn.deallocate(new_rec)

    def _write_index(self, buf, src, idx, dim):
        """Replace slice `idx` of `buf` along `dim` with `src` (extent 1 along `dim`), preserving
        the other slices, via an in-place copy into `buf`. Consumes `src` (and the temporary
        slices). `src` must already match `buf`'s dtype."""
        n = buf.shape[dim]
        if n == 1:
            ttnn.copy(src, buf)
            ttnn.deallocate(src)
            return
        parts = []
        if idx > 0:
            parts.append(self._slice_along(buf, dim, 0, idx))
        parts.append(src)
        if idx < n - 1:
            parts.append(self._slice_along(buf, dim, idx + 1, n))
        new = ttnn.concat(parts, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for p in parts:
            ttnn.deallocate(p)

    def write_slot(self, slot, rec, convs):
        """Write one user's B=1 prefill state into decode `slot`, preserving every other (live)
        row. The per-slot analogue of assemble_batched_state for vLLM continuous batching.

        rec:   [1, Nv, Dk, Dv] the user's recurrent state.
        convs: list of K [1, 1, qkv_dim_tp] the user's conv taps (conv_states[m] column). Unlike
               assemble_batched_state (which zeroes tap 0), every tap is written straight from the
               user's B=1 prefill state, so decode continues from exactly the produced shift register.
        Consumes rec and convs. Requires the batched buffers (allocate_kv_caches(batch_size=B))."""
        assert self.rec_state is not None and self.conv_states is not None, "batched GDN state not allocated"
        assert 0 <= slot < self.B, f"slot {slot} out of range [0,{self.B})"
        rec_src = rec if rec.dtype == self.rec_state.dtype else ttnn.typecast(rec, self.rec_state.dtype)
        if rec_src is not rec:
            ttnn.deallocate(rec)
        self._write_index(self.rec_state, rec_src, slot, dim=0)
        for m in range(self.K):
            c = convs[m]
            c_src = c if c.dtype == self.conv_states[m].dtype else ttnn.typecast(c, self.conv_states[m].dtype)
            if c_src is not c:
                ttnn.deallocate(c)
            self._write_index(self.conv_states[m], c_src, slot, dim=1)

    def _write_index_keep(self, buf, src, idx, dim):
        """Non-consuming twin of _write_index: `src` is left alive (it is a persistent B=1 scratch buffer, read
        by every replay of the slot-copy trace). Same slice/concat/copy op sequence as _write_index, so the row
        lands bit for bit as in the eager path; only the temporaries it makes are freed."""
        if src.dtype != buf.dtype:
            self._write_index(buf, ttnn.typecast(src, buf.dtype), idx, dim)  # consumes the typecast temp
            return
        n = buf.shape[dim]
        if n == 1:
            ttnn.copy(src, buf)
            return
        slices = []
        if idx > 0:
            slices.append(self._slice_along(buf, dim, 0, idx))
        parts = slices + [src]
        if idx < n - 1:
            tail = self._slice_along(buf, dim, idx + 1, n)
            slices.append(tail)
            parts.append(tail)
        new = ttnn.concat(parts, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for p in slices:
            ttnn.deallocate(p)

    def copy_scratch_to_slot(self, slot, rec_dst, conv_dst, fold_conv_window):
        """QWEN36_GDN_SLOT_COPY_TRACE (traced body, one per decode slot): write the BOUND B=1 prefill scratch
        (self.rec_state / conv_states / conv_carry) into row `slot` of the batched decode buffers rec_dst
        [B,Nv,Dk,Dv] and conv_dst[m] [1,B,D], without the ttnn.clone the eager _write_gdn_slot_from_scratch makes.
        Reads only persistent buffers, allocates only temporaries it frees, no host read/write, shapes fixed by
        `slot`: trace safe.

        fold_conv_window (QWEN36_GDN_CONV_WINDOW_ONCE on): finalize_conv_window fused in. The decode window is
        written straight from conv_carry into the decode row (tap 0 <- _zero_conv0, tap j+1 <- conv_carry[:, j]),
        the same values finalize_conv_window would have copied into the scratch's conv_states and the eager copy
        would then have moved on (ttnn.copy / slice / concat are exact). The scratch's own conv_states are left
        unfinalized; nothing reads them before the next request's in-place reset (prefill_paged_slots only)."""
        assert 0 <= slot < int(rec_dst.shape[0]), f"slot {slot} out of range [0,{int(rec_dst.shape[0])})"
        self._write_index_keep(rec_dst, self.rec_state, slot, dim=0)
        if not fold_conv_window:
            for m in range(self.K):
                self._write_index_keep(conv_dst[m], self.conv_states[m], slot, dim=1)
            return
        assert self._conv_window_once and self._stable_state and self.conv_carry is not None, (
            "fold_conv_window needs the QWEN36_GDN_CONV_WINDOW_ONCE carry path (the finalize it replaces)"
        )
        assert self._zero_conv0 is not None, "persistent _zero_conv0 missing (a from_torch zero is not trace safe)"
        D = self.qkv_dim_tp
        self._write_index_keep(conv_dst[0], self._zero_conv0, slot, dim=1)
        for j in range(self.K - 1):
            src = ttnn.slice(self.conv_carry, (0, j, 0), (1, j + 1, D))  # [1, 1, D], as finalize_conv_window
            self._write_index_keep(conv_dst[j + 1], src, slot, dim=1)
            ttnn.deallocate(src)

    def remap_slots(self, remap):
        """Reindex the batched decode state after a vLLM batch condense: slot i takes the state
        previously at slot remap[i] (identity entries are no-ops). Mirrors
        seed_manager.apply_slot_remap for GDN's per-slot recurrent+conv state, which the plugin's
        slot_remap does not itself move. In-place copy into the fixed buffers (preserves the decode
        trace's baked addresses)."""
        idx = [int(remap[i]) for i in range(self.B)]
        if all(idx[i] == i for i in range(self.B)):
            return
        self._gather_indices(self.rec_state, idx, dim=0)
        for m in range(self.K):
            self._gather_indices(self.conv_states[m], idx, dim=1)

    def _gather_indices(self, buf, idx, dim):
        """Rebuild `buf` so slice i along `dim` becomes old slice idx[i], then copy back in place.
        `new` is fully materialized before the copy, so gathering from `buf` into itself is safe."""
        rows = [self._slice_along(buf, dim, idx[i], idx[i] + 1) for i in range(len(idx))]
        new = ttnn.concat(rows, dim=dim)
        ttnn.copy(new, buf)
        ttnn.deallocate(new)
        for r in rows:
            ttnn.deallocate(r)

    def forward_prefill_batched(self, x, chunk_size=128, valid_lens=None, carry=False):
        """Batched prefill: all B users in one pass (no per-user Python loop).

        The chunk-seq GDN kernel scans a leading BH = B*H batch dim, each (user, head) row an
        independent causal scan, so B is a true batch dim (not a time concat). Runs projection /
        conv-FIR / chunk-parallel recurrence over [B, T, *] and writes straight into the batched
        decode buffers (rec_state[B,Nv,Dk,Dv], conv_states[*][1,B,D]); row u == user u.

        x:          [B, T, dim] replicated (all users padded to a common bucket length T).
        valid_lens: optional list of B real token counts (< T => right-padding masked per row);
                    None => every row is full length T.
        carry:      False (default) => from scratch (single-shot). True => CHUNK-OUTER carry: read
                    the recurrent state (self.rec_state) and conv left-context (self._batched_conv_carry)
                    from the previous chunk and write the updated ones back, so a long prompt can be
                    prefilled chunk-by-chunk over the batch. Mirrors the B=1 forward_prefill carry;
                    the caller zeroes rec_state (reset_state_inplace) + _batched_conv_carry at
                    sequence start, so the first chunk reads zeros (== from scratch). Requires
                    _stable_state (the batched decode buffers).

        KERNEL CAP: gated_delta_attn_seq maps one BH = B*Nv_tp row per core and is L1-bound, so BH
        must stay <= ~32 (at TP=4, Nv_tp=8 => B <= 4). Larger B trips an L1 clash (B=8) or the
        kernel's `BH <= compute_grid` assert (B=32); B>4 would need grouped launches (groups <=4).
        The model currently prefills per-user instead (see prefill_paged_peruser).
        """
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (x.shape[-3], x.shape[-2], x.shape[-1]))  # [.,B,T,dim] -> [B,T,dim]
        B, T = x.shape[0], x.shape[1]
        D = self.qkv_dim_tp

        # Route through the shared per-token projection (handles _fuse_ab/_fuse_agmm — required
        # when the caller's norm skipped its post-AG and x arrives K-sharded, e.g. prefill_paged_
        # grouped). A plain ttnn.linear(x, tw["qkvz"]) here would (a) mismatch the K-sharded width
        # against the fused-weight's full-K height, and (b) KeyError on tw["ab"], which doesn't
        # exist when _fuse_ab folds a/b into tw["qkvz"]. Flatten the batch dim into the token dim
        # (the projection is per-token; user boundaries don't matter to a linear layer) since
        # _project_qkvzab's slicing assumes a leading dim of 1.
        x_flat = ttnn.reshape(x, (1, B * T, x.shape[-1]))
        # ab8=True: batched serving prefill is the path the vLLM server actually runs ("TP batched
        # masked-bucket"); without it QWEN36_GDN_AB_BF8 loaded its weight and banner but a/b stayed
        # bf4 here (found by the ttsim pre-validation, [source validation record]).
        qkv_flat, z_flat, a_flat, b_flat = self._project_qkvzab(
            x_flat, B * T, out_mc=ttnn.DRAM_MEMORY_CONFIG, ab8=True
        )
        qkv = ttnn.reshape(qkv_flat, (B, T, D))
        z = ttnn.reshape(z_flat, (B, T, self.qkvz_dim_tp - D))
        a = ttnn.reshape(a_flat, (B, T, Nv))
        b = ttnn.reshape(b_flat, (B, T, Nv))

        # FIR causal conv1d + SiLU over each user's sequence (per-row valid_len picks each user's
        # decode conv window). Chunk-outer carry: left-context = previous chunk's last K-1 inputs.
        if carry and getattr(self, "_batched_conv_carry", None) is None:
            # First chunk of a chunk-outer prefill: zeroed left-context (== from scratch).
            self._batched_conv_carry = ttnn.from_torch(
                torch.zeros(B, self.K - 1, D, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )
        conv_carry_in = self._batched_conv_carry if carry else None
        conv, conv_new_state = _causal_conv1d_fir(
            qkv,
            None,
            None,
            self.K,
            self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            conv_state=conv_carry_in,
            weight_taps=tw["conv_taps"],
            bias_dev=None,
            valid_len=valid_lens,
        )
        ttnn.deallocate(qkv)

        kd = self.key_dim_tp
        # Flat token-major q/k/v (no host head-split / GQA): the fused op does in-kernel L2-norm and
        # GQA (Nk->Nv) from qkv_head_dims, matching the single-user forward_prefill fused path.
        q = ttnn.slice(conv, (0, 0, 0), (B, T, kd))
        k = ttnn.slice(conv, (0, 0, kd), (B, T, 2 * kd))
        v = ttnn.slice(conv, (0, 0, 2 * kd), (B, T, D))
        ttnn.deallocate(conv)

        beta = ttnn.reshape(ttnn.sigmoid(b), (B, T, Nv))
        ttnn.deallocate(b)
        g = ttnn.reshape(self._gate_prefill_g(a), (B, T, Nv))
        ttnn.deallocate(a)

        # Chunk-parallel recurrence over the BH = B*Nv batch (each row an independent scan). Fused
        # chunk_gated_delta_rule (same op as single-user prefill); per-row valid_lens mask each user.
        from models.demos.blackhole.qwen36.tt.gdn.fused_chunk import (
            chunk_gated_delta_rule_fused_adapter,
            fused_chunk_enabled,
        )

        _use_fused = fused_chunk_enabled()
        _delta_fn = chunk_gated_delta_rule_fused_adapter if _use_fused else chunk_gated_delta_rule_seq_adapter
        _extra = {"const_tiles": self._fused_const_tiles} if _use_fused else {}
        o, final_state = _delta_fn(
            q,
            k,
            v,
            beta,
            g,
            chunk_size=chunk_size,
            scale=self.scale,
            initial_state=self.rec_state if carry else None,
            device=self.mesh,
            cached_masks=self.chunk_seq_masks,
            valid_len=valid_lens,
            qkv_head_dims=(Nk, Dk, Nv, Dv),
            **_extra,
        )

        # ---- write the batched decode state directly (row u == user u) ----
        if self._stable_state and self.rec_state is not None:
            rec_src = (
                final_state
                if final_state.dtype == self.rec_state.dtype
                else ttnn.typecast(final_state, self.rec_state.dtype)
            )
            ttnn.copy(rec_src, self.rec_state)
            if rec_src is not final_state:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(final_state)
        else:
            self.rec_state = final_state  # [B, Nv, Dk, Dv]
        # conv_states[0] = shifted-out zero; conv_states[m] row u = conv_new_state[u, m-1].
        zero0 = ttnn.from_torch(
            torch.zeros(1, B, D, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        new_conv = [zero0]
        for m in range(1, self.K):
            cs = ttnn.reshape(ttnn.slice(conv_new_state, (0, m - 1, 0), (B, m, D)), (1, B, D))  # [1,B,D]
            new_conv.append(cs)
        if carry:
            # Preserve this chunk's last K-1 inputs as the next chunk's left-context (replace the
            # buffer just consumed by the FIR above).
            if conv_carry_in is not None:
                ttnn.deallocate(conv_carry_in)
            self._batched_conv_carry = conv_new_state  # [B, K-1, D]
        else:
            ttnn.deallocate(conv_new_state)
        if self._stable_state and self.conv_states is not None:
            for m in range(self.K):
                ttnn.copy(new_conv[m], self.conv_states[m])
                ttnn.deallocate(new_conv[m])
        else:
            self.conv_states = new_conv

        # ---- output (gated RMSNorm + SiLU(z) gate + row-parallel out proj + all-reduce) ----
        out_n = ttnn.rms_norm(o, weight=tw["norm_w"], epsilon=1e-6)
        ttnn.deallocate(o)
        out_f = ttnn.reshape(out_n, (B, T, self.value_dim_tp))
        ttnn.deallocate(out_n)
        gated = ttnn.multiply(out_f, ttnn.silu(z))
        ttnn.deallocate(out_f)
        ttnn.deallocate(z)
        partial = ttnn.linear(gated, tw["out"], compute_kernel_config=self.cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, B, T, partial.shape[-1]))
        return tpc.prefill_all_reduce(partial, self.mesh, self.tt_ccl, self.args.ccl_topology())

    def _conv_gates_enabled(self):
        return _tt_public_getenv("QWEN_GDN_CONV_GATES", "0") == "1" and self.K == 4 and hasattr(
            getattr(ttnn, "transformer", None), "gdn_decode_conv_gates"
        )

    def _conv_composed(self, qkv, B, Bmax, _L1):
        # The original conv shift-register + FIR + SiLU (the control arm's path).
        tw = self.tw
        # Conv1d shift-register + weighted sum + SiLU
        st = self.conv_states
        if B < Bmax:
            # Bucketed decode: active requests occupy a contiguous prefix [0:B]; idle rows [B:Bmax]
            # hold no live request (a slot is re-initialized by prefill/write_slot when reused), so
            # they are don't-care. Pad the width-B new input up to Bmax and run the SAME full-width
            # shift-register as below -- the conv sum's active rows [0:B] are exact and the downstream
            # q/k/v slices take [0:B]. This keeps the op COUNT identical to the baseline path (just a
            # single pad), vs a per-row slice/concat that added ~4*K ops/layer and erased the width win.
            qkv_p = ttnn.pad(qkv, [(0, 0), (0, Bmax - B), (0, 0)], value=0.0, memory_config=_L1)
            ttnn.deallocate(qkv)
            qkv = qkv_p
        if getattr(self, "_gdn_conv_shift", "0") == "concat":
            # Concat shift: [st1..stK-1, qkv] in one concat, slice back into the same
            # buffers (addresses preserved for traces). QWEN36_GDN_CONV_SHIFT_MUSTFAIL=1
            # shifts the wrong direction (must refuse equiv).
            _cat = ttnn.concat([st[j + 1] for j in range(self.K - 1)] + [qkv], dim=1,
                               memory_config=_L1)
            if _tt_public_getenv("QWEN36_GDN_CONV_SHIFT_MUSTFAIL") == "1":
                logger.warning("QWEN36_GDN_CONV_SHIFT_MUSTFAIL engaged: shift direction reversed")
                _cat = ttnn.concat([qkv] + [st[j + 1] for j in range(self.K - 1)], dim=1,
                                   memory_config=_L1)
            for j in range(self.K):
                ttnn.copy(ttnn.slice(_cat, (0, j * B, 0), (1, (j + 1) * B, qkv.shape[-1])), st[j])
            ttnn.deallocate(_cat)
            ttnn.deallocate(qkv)
        else:
            for j in range(self.K - 1):
                ttnn.copy(st[j + 1], st[j])
            ttnn.copy(qkv, st[self.K - 1])
            ttnn.deallocate(qkv)
        conv = ttnn.multiply(st[0], tw["conv_taps"][0], memory_config=_L1)
        for j in range(1, self.K):
            conv = ttnn.mac(st[j], tw["conv_taps"][j], conv)
        conv = ttnn.silu(conv, memory_config=_L1)

        return conv

    def _decode_chain(self):
        """QWEN36_GDN_DECODE_CHAIN (gdn/decode_chain.py): None when off, else the projection's column Layout. The mode
        is read on every call (runtime class: a served arm switch re-captures the traces); the Layout depends only on
        the layer's dims and is cached. Refuses (RuntimeError) a configuration the mode does not cover."""
        from models.demos.blackhole.qwen36.tt.gdn import decode_chain

        if decode_chain.mode() is None:
            return None
        decode_chain.check_env()
        lay = getattr(self, "_decode_chain_lay", None)
        if lay is None:
            if not (self._conv_gates_enabled() and hasattr(ttnn.transformer, "gdn_decode_norm_gate")
                    and recurrent_gated_delta_rule_decode_packed_ttnn is not None):
                raise RuntimeError(f"{decode_chain.ENV}=direct needs the image's gdn_decode_conv_gates, "
                                   "gdn_decode_norm_gate and decode_gated_delta_rule_packed")
            lay = decode_chain.layout_for(self)
            self._decode_chain_lay = lay
        return lay

    def forward_decode(self, x, oneshot_ar=False):
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        Bmax = self.B
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode conv→recurrence→norm/gate chain L1-resident
        if self.conv_states is None:
            self.reset_state()
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))

        # Active decode width, taken from the input. Normally == Bmax. BUCKETED decode: a request
        # feeds B<Bmax tokens and the whole step runs on state rows [0:B]; idle rows [B:Bmax] are
        # preserved. Conv taps are per-channel (broadcast over batch), so the conv weighted-sum
        # works at any width. The B==Bmax path is byte-identical to before.
        B = x.shape[-2]

        # QWEN36_GDN_DECODE_CHAIN=direct (gdn/decode_chain.py): conv_gates and norm_gate read qkvzab directly; the
        # five slice copies are not made. Unset: _chain is None and everything below is today's path.
        _chain = self._decode_chain()
        qkvzab = None
        if _chain is not None:
            from models.demos.blackhole.qwen36.tt.gdn import decode_chain

            qkvzab = self._project_qkvzab(x, B, out_mc=_L1, raw=True)
            if int(qkvzab.shape[-1]) != _chain.W:
                raise RuntimeError(f"{decode_chain.ENV}: qkvzab width {qkvzab.shape[-1]} != {_chain}")
            qkv = z = a = b = None
            decode_chain.banner("decode", B, _chain, logger)
        else:
            qkv, z, a, b = self._project_qkvzab(x, B, out_mc=_L1)

        # K's first kernel (QWEN_GDN_CONV_GATES=1): conv shift-register + FIR + SiLU + both
        # gates in ONE op, replacing the K copies, multiply + (K-1) mac + silu, the B<Bmax pad,
        # sigmoid, add+softplus and multiply -- 12 ops/layer. Conv states advance in place
        # (same buffers: the decode trace's address discipline holds).
        _cg_beta = _cg_g = None
        if qkvzab is not None:
            conv, _cg_beta, _cg_g = decode_chain.conv_gates(
                qkvzab, self.conv_states, [tw["conv_taps"][j] for j in range(self.K)], tw["dt_bias"],
                tw["neg_exp_A"], _chain, B, _L1,
            )
        elif self._conv_gates_enabled():
            conv, _cg_beta, _cg_g = ttnn.transformer.gdn_decode_conv_gates(
                qkv,
                self.conv_states,
                [tw["conv_taps"][j] for j in range(self.K)],
                a,
                b,
                tw["dt_bias"],
                tw["neg_exp_A"],
                batch=B,
                memory_config=_L1,
            )
            ttnn.deallocate(qkv)
            if not getattr(self, "_conv_gates_logged", False):
                self._conv_gates_logged = True
                print(
                    f"QWEN_GDN_CONV_GATES engaged: ttnn.transformer.gdn_decode_conv_gates B={B} Bmax={Bmax}"
                    f" C={self.qkv_dim_tp} Nv={Nv} K={self.K} -> conv {list(conv.shape)} beta {list(_cg_beta.shape)}",
                    flush=True,
                )
        else:
            conv = self._conv_composed(qkv, B, Bmax, _L1)

        # Milestone 2 (QWEN_GDN_PACKED_QKV=1, needs the conv+gates kernel engaged): the recurrence
        # op's reader takes conv [1,B,C] and beta/g [1,B,Nv] directly, GQA included -- the 3
        # slices, 8 reshapes and 2 repeat_interleaves below disappear.
        _packed = (
            _cg_beta is not None
            and _tt_public_getenv("QWEN_GDN_PACKED_QKV", "0") == "1"
            and hasattr(getattr(ttnn, "transformer", None), "decode_gated_delta_rule_packed")
            # The wrapper import is optional so this module imports under the simulator, where
            # the op does not exist. Without this term, a tree that has the ttnn kernel but not
            # the Python wrapper would take the packed branch and die on a None call.
            and recurrent_gated_delta_rule_decode_packed_ttnn is not None
        )
        # K3 (QWEN36_GDN_NORM_GATE=1, c82): the grafted ttnn.transformer.gdn_decode_norm_gate takes the recurrence
        # op's ROW_MAJOR o and z, and writes the gated [1,B,H*V] TILE output directly. It replaces the o tilize,
        # rms_norm, [B,H,V]->[1,B,H*V] reshape, silu and multiply (T2 profile: ~19 us/layer over 5 programs).
        # The kernel runs in fp32 (HiFi4, fp32 dest) where the chain packs bf16 between ops: tier 2, not bit-exact.
        _norm_gate = (
            _packed
            and _tt_public_getenv("QWEN36_GDN_NORM_GATE", "0") == "1"
            and hasattr(ttnn.transformer, "gdn_decode_norm_gate")
        )
        if not _packed:
            kd = self.key_dim_tp
            q = ttnn.reshape(ttnn.slice(conv, (0, 0, 0), (1, B, kd)), (B, Nk, Dk))
            k = ttnn.reshape(ttnn.slice(conv, (0, 0, kd), (1, B, 2 * kd)), (B, Nk, Dk))
            v = ttnn.reshape(ttnn.slice(conv, (0, 0, 2 * kd), (1, B, self.qkv_dim_tp)), (B, Nv, Dv))
            ttnn.deallocate(conv)

            # GQA expand Q/K Nk→Nv; recurrence L2-norms + scales internally
            rf = Nv // Nk
            q = ttnn.repeat_interleave(q, rf, dim=1)
            k = ttnn.repeat_interleave(k, rf, dim=1)
            # Decode: hand q/k/v to the recurrent kernel in L1. The kernel typecasts + does a LOCAL
            # l2-norm (no cross-device gather), so placement is output-neutral here (unlike SDPA-q,
            # which hard-requires DRAM, and unlike the residual→DistributedNorm all-gather).
            q = ttnn.reshape(q, (B, 1, Nv, Dk), memory_config=_L1)
            k = ttnn.reshape(k, (B, 1, Nv, Dk), memory_config=_L1)
            v = ttnn.reshape(v, (B, 1, Nv, Dv), memory_config=_L1)

        if qkvzab is not None and not (_packed and _norm_gate):
            raise RuntimeError("QWEN36_GDN_DECODE_CHAIN=direct: the packed recurrence + norm_gate path is not engaged")
        if _packed:
            if qkvzab is None:
                ttnn.deallocate(a)
                ttnn.deallocate(b)
            beta, g = _cg_beta, _cg_g  # [1,B,Nv]: the packed reader's own layout
        elif _cg_beta is not None:
            ttnn.deallocate(a)
            ttnn.deallocate(b)
            beta = ttnn.reshape(_cg_beta, (B, 1, Nv))
            g = ttnn.reshape(_cg_g, (B, 1, Nv))
        else:
            beta = ttnn.reshape(ttnn.sigmoid(b, memory_config=_L1), (B, 1, Nv))
            ttnn.deallocate(b)
            g = ttnn.multiply(tw["neg_exp_A"], _softplus_add(a, tw["dt_bias"]), memory_config=_L1)
            ttnn.deallocate(a)
            g = ttnn.reshape(g, (B, 1, Nv))

        # fp32 decode step by default (QWEN35_GDN_DECODE_BF16=1 reverts)
        init_state = self.rec_state if B == Bmax else self._slice_along(self.rec_state, 0, 0, B)
        # QWEN_GDN_FUSED_INPLACE=1: ask the fused op to write the new state straight into
        # rec_state (its _inplace variant) instead of allocating h and copying it back.
        # Legal only when init_state IS the persistent buffer (B == Bmax, no slice). The
        # composed path ignores the flag, so the caller decides by IDENTITY below, not env.
        _want_inplace = (
            _tt_public_getenv("QWEN_GDN_FUSED_INPLACE") == "1"
            and self._stable_state and B == Bmax and init_state is self.rec_state
        )
        if _packed:
            o, new_rec = recurrent_gated_delta_rule_decode_packed_ttnn(
                conv,
                beta,
                g,
                Nk,
                Nv,
                Dk,
                Dv,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(_tt_public_getenv("QWEN35_GDN_DECODE_BF16") != "1"),
                inplace_state=_want_inplace,
                return_row_major=_norm_gate,
            )
            ttnn.deallocate(conv)
        else:
            o, new_rec = recurrent_gated_delta_rule_decode_ttnn(
                q,
                k,
                v,
                beta,
                g,
                scale=self.scale,
                initial_state=init_state,
                device=self.mesh,
                high_precision=(_tt_public_getenv("QWEN35_GDN_DECODE_BF16") != "1"),
                inplace_state=_want_inplace,
            )
        # The op returns *in.initial_state, but nanobind hands back a NEW Python wrapper
        # around the same device buffer, so `is` sees two objects. Compare the buffer.
        try:
            _did_inplace = _want_inplace and new_rec.buffer_address() == init_state.buffer_address()
        except Exception:
            _did_inplace = False
        if _want_inplace and not getattr(self, "_inplace_logged", False):
            self._inplace_logged = True
            print(f"QWEN_GDN_FUSED_INPLACE requested; in-place happened={_did_inplace}"
                  f" (same buffer={new_rec.buffer_address() == init_state.buffer_address()})", flush=True)
        if init_state is not self.rec_state:
            ttnn.deallocate(init_state)
        if self._stable_state:
            # In-place update preserves rec_state address for decode trace replay
            if B == Bmax:
                if _did_inplace:
                    pass  # new_rec IS rec_state: nothing to copy, and freeing it would free the state
                else:
                    ttnn.copy(new_rec, self.rec_state)
                    ttnn.deallocate(new_rec)
            else:
                self._write_recurrent_state_prefix(new_rec, B)
        else:
            self.rec_state = new_rec

        if _norm_gate:
            if qkvzab is not None:
                gated = decode_chain.norm_gate(o, qkvzab, tw["norm_w"], Nv, _chain, B, _L1)
                z = qkvzab  # freed below, after the gated output exists
            else:
                gated = ttnn.transformer.gdn_decode_norm_gate(
                    o, z, tw["norm_w"], Nv, batch=B, epsilon=1e-6, memory_config=_L1
                )
            if not getattr(self, "_norm_gate_logged", False):
                self._norm_gate_logged = True
                logger.info(
                    f"QWEN36_GDN_NORM_GATE engaged: ttnn.transformer.gdn_decode_norm_gate B={B} Nv={Nv} Dv={Dv} "
                    f"o {list(o.shape)} {o.layout} -> gated {list(gated.shape)}"
                )
            ttnn.deallocate(o)
            ttnn.deallocate(z)
        else:
            out_r = ttnn.reshape(o, (B, Nv, Dv))
            out_n = ttnn.rms_norm(out_r, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)  # gated norm (no +1)
            ttnn.deallocate(out_r)
            out_f = ttnn.reshape(out_n, (1, B, self.value_dim_tp))
            ttnn.deallocate(out_n)
            gated = _silu_mul(out_f, z, _L1)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)

        partial = self._row_proj(gated, tw["out"])
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, B, partial.shape[-1]))
        if oneshot_ar:  # QWEN36_ONESHOT_AR: replicated residual -> one all-gather + add
            return tpc.all_reduce_oneshot_decode(partial, self.tt_ccl, self.args.ccl_topology())
        out = tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(partial),
        )
        return out

    # ------------------------------------------------------------------------------------------ #
    # QWEN36_MTP_BATCH (docs/MTP-BATCH.md): batched speculative verify + per-row state commit.
    #
    # forward_verify runs the T = K+1 verify rows of every user as ONE decode-width pass for the
    # projections / norm / gate / out-proj (M = Bw*T <= 32 rows, row u = j*Bw + b), and the
    # recurrence as T sequential sub-steps of the SAME production decode kernels at width Bw
    # (gdn_decode_conv_gates + decode_gated_delta_rule_packed), so row (b, j) sees exactly the
    # state plain decode would have after j more tokens. The pre-verify recurrent state is NOT
    # overwritten (inplace_state=False); every sub-step's state is kept (hist_j) and conv taps
    # are snapshotted, so mtp_commit can select, per row, the state after that row's accepted
    # length m_b. Nothing here runs unless the MTP batch lever is engaged.
    # ------------------------------------------------------------------------------------------ #
    def _conv_multi_check(self, T):
        """QWEN36_MTP_CONV_MULTI gate: False when unset; refuses (RuntimeError) the combinations it does not cover."""
        from models.demos.blackhole.qwen36.tt.gdn import conv_multi, state_slot

        if not conv_multi.enabled():
            return False
        if not state_slot.enabled():
            raise RuntimeError(
                "QWEN36_MTP_CONV_MULTI=1 needs QWEN36_MTP_STATE_SLOT=1: the where-chain commit selects among per-depth "
                "conv snapshots, which the one-call verify no longer takes"
            )
        if self.K != 4:
            raise RuntimeError(f"QWEN36_MTP_CONV_MULTI=1 needs K=4 conv taps (got {self.K})")
        return True

    def _conv_multi_verify(self, qkv, a, b, T, Bw, M, _L1, whole=False, chain=None):
        """QWEN36_MTP_CONV_MULTI: one gdn_decode_conv_gates at batch M = T*Bw. Returns the T per-candidate
        (conv, beta, g) blocks at width Bw for the sequential recurrence, and the four window tensors (kept until
        the commit: row j*Bw + b = row b's taps after j+1 tokens). Frees qkv, a and b.
        whole=True (QWEN36_MTP_GDN_STATE_ONCE): no split; ONE block, the batch-M (conv, beta, g), read directly by the
        state-once verify program.
        chain (QWEN36_GDN_DECODE_CHAIN=direct, whole=True only): qkv = a = b = the unsliced qkvzab; the windows and
        conv_gates read it directly and it is NOT freed here (the caller's norm_gate reads z from it)."""
        from models.demos.blackhole.qwen36.tt.gdn import conv_multi

        tw, C, Nv = self.tw, self.qkv_dim_tp, self.Nv
        mf = conv_multi.mustfail()
        split = "none" if whole else conv_multi.split_mode()
        if chain is not None and not whole:
            raise RuntimeError("QWEN36_GDN_DECODE_CHAIN=direct covers the state-once verify only (whole=True)")
        from models.demos.blackhole.qwen36.tt.gdn import conv_fold

        fold = conv_fold.check_env()  # QWEN36_MTP_CONV_FOLD: refuses what it does not cover; False when unset
        if fold:
            # build_windows folded into the conv_gates reader: ONE program, same conv/beta/g and window bytes
            taps = [tw["conv_taps"][i] for i in range(self.K)]
            if chain is not None:
                from models.demos.blackhole.qwen36.tt.gdn import decode_chain

                b_col = chain.a_col if decode_chain.mustfail() else chain.b_col  # decode_chain.conv_gates' must-fail
                conv_all, beta_all, g_all, wins = conv_fold.conv_gates(
                    self.conv_states, qkv, taps, qkv, qkv, tw["dt_bias"], tw["neg_exp_A"], T, Bw, _L1,
                    channels=chain.C, a_col=chain.a_col, b_col=b_col, mf=conv_fold.mustfail())
            else:
                conv_all, beta_all, g_all, wins = conv_fold.conv_gates(
                    self.conv_states, qkv, taps, a, b, tw["dt_bias"], tw["neg_exp_A"], T, Bw, _L1,
                    mf=conv_fold.mustfail())
                for t in (qkv, a, b):
                    ttnn.deallocate(t)
            conv_fold.banner(T, Bw, logger)
        else:
            wins = conv_multi.build_windows(self.conv_states, qkv, T, Bw, mf=mf)
        if fold:
            pass  # conv_all, beta_all, g_all, wins come from the fold above
        elif chain is not None:
            from models.demos.blackhole.qwen36.tt.gdn import decode_chain

            conv_all, beta_all, g_all = decode_chain.conv_gates(
                qkv, wins, [tw["conv_taps"][i] for i in range(self.K)], tw["dt_bias"], tw["neg_exp_A"], chain, M, _L1
            )
        else:
            conv_all, beta_all, g_all = ttnn.transformer.gdn_decode_conv_gates(
                qkv,
                wins,
                [tw["conv_taps"][i] for i in range(self.K)],
                a,
                b,
                tw["dt_bias"],
                tw["neg_exp_A"],
                batch=M,
                memory_config=_L1,
            )
            for t in (qkv, a, b):
                ttnn.deallocate(t)
        if T == 1 or whole:
            blocks = [(conv_all, beta_all, g_all)]
        elif split == "gather":
            (convs,) = conv_multi.split_rows([conv_all], T, Bw, mc=_L1)
            betas, gs = conv_multi.split_rows([beta_all, g_all], T, Bw, mc=_L1)
            for t in (conv_all, beta_all, g_all):
                ttnn.deallocate(t)
            blocks = list(zip(convs, betas, gs))
        else:
            blocks = []
            for j in range(T):
                lo, hi = j * Bw, (j + 1) * Bw
                blocks.append((
                    ttnn.slice(conv_all, (0, lo, 0), (1, hi, C), memory_config=_L1),
                    ttnn.slice(beta_all, (0, lo, 0), (1, hi, Nv), memory_config=_L1),
                    ttnn.slice(g_all, (0, lo, 0), (1, hi, Nv), memory_config=_L1),
                ))
            for t in (conv_all, beta_all, g_all):
                ttnn.deallocate(t)
        seen = TPGatedDeltaNet.__dict__.get("_conv_multi_banners")
        if seen is None:
            seen = set()
            TPGatedDeltaNet._conv_multi_banners = seen
        if (T, Bw) not in seen:
            seen.add((T, Bw))
            n, m = conv_multi.op_counts(T, split) if not whole else (3, conv_multi.op_counts(T, "gather")[1])
            if fold:
                n -= 1  # QWEN36_MTP_CONV_FOLD: build_windows runs inside the conv_gates program
            logger.info(
                f"QWEN36_MTP_CONV_MULTI engaged: one conv_gates at batch M={M} (T={T} x Bw={Bw}), {n} ops/layer "
                f"(was {m}); split={split}, windows {wins[0].dtype} [1,{M},{C}]"
            )
            if mf:
                logger.info(
                    "QWEN36_MTP_CONV_MULTI_MUSTFAIL engaged: the last candidate's conv window is one token too far "
                    "(equivalence must FAIL)"
                )
        return blocks, wins

    def _state_once_check(self, T, norm_gate):
        """QWEN36_MTP_GDN_STATE_ONCE gate: False when unset; refuses (RuntimeError) what it does not cover."""
        from models.demos.blackhole.qwen36.tt.gdn import state_once

        if not state_once.check_env():  # raises without QWEN36_MTP_STATE_SLOT=1 / QWEN36_MTP_CONV_MULTI=1
            return False
        if not self._conv_multi_check(T):
            raise RuntimeError(f"{state_once.ENV}=1 needs QWEN36_MTP_CONV_MULTI engaged")
        if not norm_gate:
            raise RuntimeError(
                f"{state_once.ENV}=1 needs QWEN36_GDN_NORM_GATE=1 (the verify hands its ROW_MAJOR o straight to "
                "gdn_decode_norm_gate; the rms_norm tail is not covered by the self-test or ttsim)"
            )
        if _tt_public_getenv("QWEN35_GDN_DECODE_BF16") != "1":
            raise RuntimeError(
                f"{state_once.ENV}=1 needs QWEN35_GDN_DECODE_BF16=1 (one IO dtype through the program, as served; "
                "the packed wrapper's fp32 casts are not reproduced)"
            )
        return True

    def _forward_verify_state_once(self, qkv, z, a, b, T, Bw, M, _dbl, oneshot_ar, qkvzab=None, chain=None):
        """forward_verify under QWEN36_MTP_GDN_STATE_ONCE: CONV_MULTI's one batch-M conv_gates (no split), ONE verify
        program (state read once per row, nothing written to it, o for all T*Bw rows), the norm-gate tail. Keeps
        (conv_all, beta_all, g_all) and the conv windows alive in _mtp_verify for the commit (mtp_commit_slot)."""
        from models.demos.blackhole.qwen36.tt.gdn import state_once

        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        _L1 = ttnn.L1_MEMORY_CONFIG
        if chain is not None:  # QWEN36_GDN_DECODE_CHAIN=direct: qkv/a/b/z are all read from qkvzab
            from models.demos.blackhole.qwen36.tt.gdn import decode_chain

            if "cmulti" in _dbl:
                _q2 = ttnn.clone(qkvzab)
                _b2, _w2 = self._conv_multi_verify(_q2, _q2, _q2, T, Bw, M, _L1, whole=True, chain=chain)
                for t in [t for blk in _b2 for t in blk] + list(_w2) + [_q2]:
                    ttnn.deallocate(t)
            blocks, wins = self._conv_multi_verify(qkvzab, qkvzab, qkvzab, T, Bw, M, _L1, whole=True, chain=chain)
        elif "cmulti" in _dbl:
            _b2, _w2 = self._conv_multi_verify(ttnn.clone(qkv), ttnn.clone(a), ttnn.clone(b), T, Bw, M, _L1, whole=True)
            for t in [t for blk in _b2 for t in blk] + list(_w2):
                ttnn.deallocate(t)
        if chain is None:
            blocks, wins = self._conv_multi_verify(qkv, a, b, T, Bw, M, _L1, whole=True)
        conv_all, beta_all, g_all = blocks[0]
        # QWEN36_MTP_GDN_SO_T0: the verify also writes rows < Bw' state after token 0 over rec_state (the commit then
        # recomputes tokens 1..idx); the commit reads t0 from _mtp_verify so both programs agree.
        t0 = state_once.t0_enabled()
        o_all, cin, info = state_once.verify(conv_all, beta_all, g_all, self.rec_state, Nk, Nv, Dk, Dv, self.scale, T, Bw,
                                             t0=t0)
        if "rec" in _dbl:
            # the doubled (timing-only) verify must not apply token 0 a second time: t0=False
            for t in state_once.verify(conv_all, beta_all, g_all, self.rec_state, Nk, Nv, Dk, Dv, self.scale, T, Bw,
                                       t0=False)[:2]:
                ttnn.deallocate(t)
        for t in (conv_all, beta_all, g_all):
            ttnn.deallocate(t)
        if not hasattr(self, "_mtp_verify"):
            self._mtp_verify = {}
        # Kept alive until mtp_commit_slot (and, under trace capture, for the life of the trace): the per-(row, head,
        # token) commit inputs (cin) and CONV_MULTI's window tensors. No recurrent state is kept.
        self._mtp_verify[Bw] = ((state_once.ENV, cin, T, t0), wins)
        seen = TPGatedDeltaNet.__dict__.get("_state_once_banners")
        if seen is None:
            seen = set()
            TPGatedDeltaNet._state_once_banners = seen
        if (T, Bw) not in seen:
            seen.add((T, Bw))
            before, after, detail = state_once.state_units(T, Bw, self.B)
            n_once, n_cm = state_once.op_counts(T)
            logger.info(
                f"QWEN36_MTP_GDN_STATE_ONCE engaged: T={T} Bw={Bw}, recurrent state units/layer {before} -> {after} "
                f"({detail}); verify GDN conv+recurrence programs/layer {n_cm} -> {n_once}; verify {info['cores']} "
                f"cores x {info['per_core']}, CB {info['cb_kib']:.0f} KiB/core, fid {info['fid']}{', sfpu row0' if info.get('row0') else ''}{(', fuse ' + info['fuse']) if info.get('fuse') else ''}{', splitread' if info.get('split') else ''}{', qkh' if info.get('qkh') else ''}"
            )
            if info.get("vsplit", 1) > 1:
                logger.info(
                    f"QWEN36_MTP_GDN_SO_VSPLIT engaged: P={info['vsplit']} at Bw={Bw} ({info['cores']} cores), T={T}: each "
                    f"head instance's value columns split over {info['vsplit']} cores (bitwise the P=1 verify)"
                    f"{' [MUSTFAIL: part 1 reads part 0 state columns]' if info.get('vsplit_mf') else ''}"
                )
            if t0:
                logger.info(
                    f"{state_once.T0_BANNER}: T={T} Bw={Bw}, the verify writes rec_state rows < Bw after token 0 "
                    f"(token 0 is always committed), the commit recomputes tokens 1..idx"
                    f"{' [MUSTFAIL: row Bw-1 written after token 1]' if info['t0_mf'] else ''}"
                )
        if chain is not None:
            gated = decode_chain.norm_gate(o_all, qkvzab, tw["norm_w"], Nv, chain, M, _L1)
            if "normgate" in _dbl:
                ttnn.deallocate(decode_chain.norm_gate(o_all, qkvzab, tw["norm_w"], Nv, chain, M, _L1))
            z = qkvzab  # freed below with the norm-gate inputs
        else:
            gated = ttnn.transformer.gdn_decode_norm_gate(o_all, z, tw["norm_w"], Nv, batch=M, epsilon=1e-6, memory_config=_L1)
            if "normgate" in _dbl:
                ttnn.deallocate(
                    ttnn.transformer.gdn_decode_norm_gate(o_all, z, tw["norm_w"], Nv, batch=M, epsilon=1e-6, memory_config=_L1)
                )
        if not getattr(TPGatedDeltaNet, "_mtp_norm_gate_banner", False):
            TPGatedDeltaNet._mtp_norm_gate_banner = True
            logger.info(
                f"QWEN36_GDN_NORM_GATE engaged in MTP verify: gdn_decode_norm_gate batch=M={M} (T={T} x Bw={Bw}) "
                f"Nv={Nv} Dv={Dv} o {list(o_all.shape)} {o_all.layout} -> gated {list(gated.shape)}"
            )
        ttnn.deallocate(o_all)
        ttnn.deallocate(z)
        partial = self._row_proj(gated, tw["out"])
        if "out" in _dbl:
            ttnn.deallocate(self._row_proj(gated, tw["out"]))
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, M, partial.shape[-1]))
        if oneshot_ar:
            return tpc.all_reduce_oneshot_decode(partial, self.tt_ccl, self.args.ccl_topology())
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(partial),
        )

    def forward_verify(self, x, T, oneshot_ar=False):
        tw, Nk, Nv, Dk, Dv = self.tw, self.Nk, self.Nv, self.Dk, self.Dv
        Bmax = self.B
        _L1 = ttnn.L1_MEMORY_CONFIG
        if len(x.shape) == 4:
            x = ttnn.reshape(x, (1, x.shape[-2], x.shape[-1]))
        M = x.shape[-2]
        assert M % T == 0, f"verify rows {M} not a multiple of T={T}"
        Bw = M // T
        assert Bw <= Bmax and M <= tpc.TILE_SIZE, f"verify width Bw={Bw} T={T} (M={M}) Bmax={Bmax}"
        if not (
            self._conv_gates_enabled()
            and _tt_public_getenv("QWEN_GDN_PACKED_QKV", "0") == "1"
            and hasattr(ttnn.transformer, "decode_gated_delta_rule_packed")
            and recurrent_gated_delta_rule_decode_packed_ttnn is not None
        ):
            raise RuntimeError(
                "QWEN36_MTP_BATCH verify needs the production GDN decode kernels "
                "(QWEN_GDN_CONV_GATES=1, QWEN_GDN_PACKED_QKV=1 and an image with both ops)"
            )
        assert self._stable_state, "QWEN36_MTP_BATCH verify needs the batched fixed-address GDN state"

        # QWEN36_GDN_DECODE_CHAIN=direct on the state-once verify (the served MTP path): the windows, conv_gates and
        # norm_gate read the unsliced qkvzab; the five slice copies are not made. Other verify paths keep the slices.
        _chain = self._decode_chain()
        if _chain is not None:
            _ng_env = (_tt_public_getenv("QWEN36_GDN_NORM_GATE", "0") == "1"
                       and hasattr(ttnn.transformer, "gdn_decode_norm_gate"))
            if self._state_once_check(T, _ng_env):
                from models.demos.blackhole.qwen36.tt.gdn import decode_chain

                _dbl = verify_double()
                qkvzab = self._project_qkvzab(x, M, out_mc=_L1, raw=True)
                if int(qkvzab.shape[-1]) != _chain.W:
                    raise RuntimeError(f"{decode_chain.ENV}: qkvzab width {qkvzab.shape[-1]} != {_chain}")
                if "proj" in _dbl:
                    ttnn.deallocate(self._project_qkvzab(x, M, out_mc=_L1, raw=True))
                decode_chain.banner(f"verify T={T} Bw={Bw}", M, _chain, logger)
                return self._forward_verify_state_once(None, None, None, None, T, Bw, M, _dbl, oneshot_ar,
                                                       qkvzab=qkvzab, chain=_chain)
            if not getattr(TPGatedDeltaNet, "_decode_chain_verify_logged", False):
                TPGatedDeltaNet._decode_chain_verify_logged = True
                logger.warning("QWEN36_GDN_DECODE_CHAIN=direct: this verify path is not the state-once verify; "
                               "it keeps the five qkvzab slices")

        qkv, z, a, b = self._project_qkvzab(x, M, out_mc=_L1)
        _dbl = verify_double()
        if _dbl and not getattr(TPGatedDeltaNet, "_mtp_double_banner", False):
            TPGatedDeltaNet._mtp_double_banner = True
            logger.warning(
                f"QWEN36_MTP_VERIFY_DOUBLE engaged: {sorted(_dbl)} run twice in the GDN verify T={T} Bw={Bw} "
                f"(timing diagnostic: outputs correct, never a speed number)"
            )
        if "proj" in _dbl:
            for t in self._project_qkvzab(x, M, out_mc=_L1):
                ttnn.deallocate(t)
        C = self.qkv_dim_tp
        # QWEN36_GDN_NORM_GATE ported to the verify (same condition as forward_decode's _norm_gate): each
        # sub-step's recurrence keeps o ROW_MAJOR, the T o's concatenate along their batch dim (row u =
        # j*Bw + b, z's row order), and ONE gdn_decode_norm_gate at batch=M replaces the concat + rms_norm +
        # reshape + silu*z chain below -- the kernel plain decode runs on the same arm. Unset: unchanged.
        _norm_gate = (
            _tt_public_getenv("QWEN36_GDN_NORM_GATE", "0") == "1"
            and hasattr(ttnn.transformer, "gdn_decode_norm_gate")
        )
        # QWEN36_MTP_GDN_STATE_ONCE (docs/MTP-BATCH.md section 16, gdn/state_once.py): the recurrence reads each row's
        # state once and writes none; the commit recomputes the accepted state. Unset: everything below, unchanged.
        if self._state_once_check(T, _norm_gate):
            return self._forward_verify_state_once(qkv, z, a, b, T, Bw, M, _dbl, oneshot_ar)
        state = self.rec_state if Bw == Bmax else self._slice_along(self.rec_state, 0, 0, Bw)
        state0 = state
        hists, snaps, outs = [], [], []
        high = _tt_public_getenv("QWEN35_GDN_DECODE_BF16") != "1"
        # QWEN36_MTP_CONV_MULTI (docs/MTP-BATCH.md section 14, gdn/conv_multi.py): ONE conv_gates at batch M over
        # exact window tensors instead of T (slices + conv_gates) + 4(T-1) clones; the live taps stay untouched and
        # the slot commit writes the selected window rows. Unset: the sequential loop below, unchanged.
        _cmulti = self._conv_multi_check(T)
        if _cmulti:
            if "cmulti" in _dbl:
                _b2, _w2 = self._conv_multi_verify(ttnn.clone(qkv), ttnn.clone(a), ttnn.clone(b), T, Bw, M, _L1)
                for t in [t for blk in _b2 for t in blk] + list(_w2):
                    ttnn.deallocate(t)
            blocks, snaps = self._conv_multi_verify(qkv, a, b, T, Bw, M, _L1)
        for j in range(T):
            if _cmulti:
                conv, beta, g = blocks[j]
            else:
                lo, hi = j * Bw, (j + 1) * Bw
                qkv_j = ttnn.slice(qkv, (0, lo, 0), (1, hi, C), memory_config=_L1)
                a_j = ttnn.slice(a, (0, lo, 0), (1, hi, Nv), memory_config=_L1)
                b_j = ttnn.slice(b, (0, lo, 0), (1, hi, Nv), memory_config=_L1)
                conv, beta, g = ttnn.transformer.gdn_decode_conv_gates(
                    qkv_j,
                    self.conv_states,
                    [tw["conv_taps"][i] for i in range(self.K)],
                    a_j,
                    b_j,
                    tw["dt_bias"],
                    tw["neg_exp_A"],
                    batch=Bw,
                    memory_config=_L1,
                )
                for t in (qkv_j, a_j, b_j):
                    ttnn.deallocate(t)
            o, h = recurrent_gated_delta_rule_decode_packed_ttnn(
                conv,
                beta,
                g,
                Nk,
                Nv,
                Dk,
                Dv,
                scale=self.scale,
                initial_state=state,
                device=self.mesh,
                high_precision=high,
                inplace_state=False,
                **({"return_row_major": True} if _norm_gate else {}),
            )
            if "rec" in _dbl:
                for t in recurrent_gated_delta_rule_decode_packed_ttnn(
                    conv, beta, g, Nk, Nv, Dk, Dv, scale=self.scale, initial_state=state, device=self.mesh,
                    high_precision=high, inplace_state=False, **({"return_row_major": True} if _norm_gate else {}),
                ):
                    ttnn.deallocate(t)
            ttnn.deallocate(conv)
            ttnn.deallocate(beta)
            ttnn.deallocate(g)
            hists.append(h)
            state = h
            outs.append(o if _norm_gate else ttnn.reshape(o, (Bw, Nv, Dv)))
            if j < T - 1 and not _cmulti:
                # Conv shift-register after j+1 tokens (the live buffers keep advancing in place).
                snaps.append([ttnn.clone(cs) for cs in self.conv_states])
        if not _cmulti:  # the CONV_MULTI path freed them after its conv_gates
            for t in (qkv, a, b):
                ttnn.deallocate(t)
        if state0 is not self.rec_state:
            ttnn.deallocate(state0)
        # Kept alive until mtp_commit (and, under trace capture, for the life of the trace).
        if not hasattr(self, "_mtp_verify"):
            self._mtp_verify = {}
        self._mtp_verify[Bw] = (hists, snaps)
        if not getattr(TPGatedDeltaNet, "_mtp_verify_banner", False):
            TPGatedDeltaNet._mtp_verify_banner = True
            logger.info(
                f"QWEN36_MTP_BATCH verify: GDN T={T} sub-steps at width Bw={Bw} (M={M}), "
                f"conv_gates + packed recurrence, inplace_state=False, per-sub-step state kept"
            )

        if _norm_gate:
            # o is the packed op's ROW_MAJOR output, one page per (row, head): concatenating along the
            # batch dim (the first dim of size Bw ahead of the head dim) gives page order (u, h), u = j*Bw + b.
            shp = [int(d) for d in outs[0].shape]
            bdim = next((i for i, d in enumerate(shp) if d == Bw and Nv in shp[i + 1:]), None)
            if bdim is None:
                raise RuntimeError(f"QWEN36_GDN_NORM_GATE in MTP verify: no batch dim {Bw} before heads in o {shp}")
            o_all = ttnn.concat(outs, dim=bdim) if T > 1 else outs[0]
            if T > 1:
                for o in outs:
                    ttnn.deallocate(o)
            gated = ttnn.transformer.gdn_decode_norm_gate(
                o_all, z, tw["norm_w"], Nv, batch=M, epsilon=1e-6, memory_config=_L1
            )
            if "normgate" in _dbl:
                ttnn.deallocate(
                    ttnn.transformer.gdn_decode_norm_gate(o_all, z, tw["norm_w"], Nv, batch=M, epsilon=1e-6, memory_config=_L1)
                )
            if not getattr(TPGatedDeltaNet, "_mtp_norm_gate_banner", False):
                TPGatedDeltaNet._mtp_norm_gate_banner = True
                logger.info(
                    f"QWEN36_GDN_NORM_GATE engaged in MTP verify: gdn_decode_norm_gate batch=M={M} (T={T} x Bw={Bw}) "
                    f"Nv={Nv} Dv={Dv} o {list(o_all.shape)} {o_all.layout} -> gated {list(gated.shape)}"
                )
            ttnn.deallocate(o_all)
            ttnn.deallocate(z)
        else:
            o_all = ttnn.concat(outs, dim=0) if T > 1 else outs[0]  # [M, Nv, Dv], j-major rows
            if T > 1:
                for o in outs:
                    ttnn.deallocate(o)
            out_n = ttnn.rms_norm(o_all, weight=tw["norm_w"], epsilon=1e-6, memory_config=_L1)
            ttnn.deallocate(o_all)
            out_f = ttnn.reshape(out_n, (1, M, self.value_dim_tp))
            ttnn.deallocate(out_n)
            gated = _silu_mul(out_f, z, _L1)
            ttnn.deallocate(out_f)
            ttnn.deallocate(z)
        partial = self._row_proj(gated, tw["out"])
        if "out" in _dbl:
            ttnn.deallocate(self._row_proj(gated, tw["out"]))
        ttnn.deallocate(gated)
        partial = ttnn.reshape(partial, (1, 1, M, partial.shape[-1]))
        if oneshot_ar:
            return tpc.all_reduce_oneshot_decode(partial, self.tt_ccl, self.args.ccl_topology())
        return tt_all_reduce(
            partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(partial),
        )

    @staticmethod
    def mtp_select(cands, masks, method):
        """Row select across candidate tensors: out[b] = cands[j][b] where masks[j][b] == 1 (exactly
        one j per row). masks[j] broadcasts over every dim but the row dim. Both methods are exact:
        where picks, blend is 1*x + 0*y + ... with a one-hot mask. The where chain starts from the
        last candidate, so the last mask is not read. Never returns an input tensor."""
        if method == "where":
            sel = None
            for j in range(len(cands) - 2, -1, -1):
                nxt = ttnn.where(masks[j], cands[j], cands[-1] if sel is None else sel)
                if sel is not None:
                    ttnn.deallocate(sel)
                sel = nxt
            return ttnn.clone(cands[-1]) if sel is None else sel
        if method == "blend":
            acc = ttnn.multiply(cands[0], masks[0])
            for j in range(1, len(cands)):
                term = ttnn.multiply(cands[j], masks[j])
                nxt = ttnn.add(acc, term)
                ttnn.deallocate(acc)
                ttnn.deallocate(term)
                acc = nxt
            return acc
        raise ValueError(f"QWEN36_MTP_BATCH commit method {method!r}")

    def mtp_commit(self, Bw, rec_masks, conv_masks, method, release=False):
        """Per-row commit after acceptance: rec_state[b] <- hist_{m_b}[b], conv taps likewise.
        rec_masks: T one-hot [Bw,1,1,1] tensors; conv_masks: T one-hot [1,Bmax,1] tensors (rows
        >= Bw select the live buffer). release=True frees the verify leftovers (eager warmup);
        traced steps keep them, so the verify trace's outputs never change owner."""
        hists, snaps = self._mtp_verify[Bw]
        T = len(hists)
        if self._conv_multi_check(T):  # refuses without QWEN36_MTP_STATE_SLOT; never reached with it
            raise RuntimeError("QWEN36_MTP_CONV_MULTI=1: the where-chain commit has no conv snapshots to select")
        sel = self.mtp_select(hists, rec_masks, method)
        if not getattr(TPGatedDeltaNet, "_mtp_commit_banner", False):
            TPGatedDeltaNet._mtp_commit_banner = True
            logger.info(
                f"QWEN36_MTP_BATCH commit dtypes: hist {hists[0].dtype} select {sel.dtype} mask {rec_masks[0].dtype} "
                f"state {self.rec_state.dtype} (method {method})"
            )
        if sel.dtype != hists[0].dtype:
            # The select itself changed the dtype (e.g. an fp32 state through a bf16 where): the
            # committed state would be rounded every step, which plain decode never does.
            raise RuntimeError(
                f"QWEN36_MTP_BATCH commit select {method} returned {sel.dtype} for {hists[0].dtype} candidates"
            )
        if sel.dtype != self.rec_state.dtype:
            # The kernel's own output dtype differs from the persistent state: plain decode's
            # ttnn.copy(new_rec, rec_state) does the same conversion.
            cast = ttnn.typecast(sel, self.rec_state.dtype)
            ttnn.deallocate(sel)
            sel = cast
        if Bw == self.B:
            ttnn.copy(sel, self.rec_state)
            ttnn.deallocate(sel)
        else:
            self._write_recurrent_state_prefix(sel, Bw)  # consumes sel
        for i in range(self.K):
            cands = [snaps[j][i] for j in range(T - 1)] + [self.conv_states[i]]
            c = self.mtp_select(cands, conv_masks, method)
            ttnn.copy(c, self.conv_states[i])
            ttnn.deallocate(c)
        if release:
            for h in hists:
                ttnn.deallocate(h)
            for sn in snaps:
                for t in sn:
                    ttnn.deallocate(t)
            del self._mtp_verify[Bw]

    def mtp_commit_slot(self, Bw, idx, release=False):
        """QWEN36_MTP_STATE_SLOT (docs/MTP-BATCH.md section 10): the same per-row commit as mtp_commit,
        rec_state[b] <- hist_{idx[b]}[b] and conv taps likewise, as two index-driven programs that read
        only the selected candidate per row and write straight into the persistent buffers (no
        where-chain, no temporary, no copy). idx: the bucket's ROW_MAJOR uint32 [1, 32] commit-depth
        page (entry b = m_b; >= T keeps row b)."""
        from models.demos.blackhole.qwen36.tt.gdn import state_slot

        hists, snaps = self._mtp_verify[Bw]
        once = isinstance(hists, tuple)  # QWEN36_MTP_GDN_STATE_ONCE: (tag, cin, T), no states
        if once:
            from models.demos.blackhole.qwen36.tt.gdn import state_once

            _tag, cin, T, t0 = hists
            self._conv_multi_check(T)
            # rec_state[b] <- recompute of row b's first idx[b]+1 verify tokens from rec_state[b] (one read, one write);
            # QWEN36_MTP_GDN_SO_T0: rec_state[b] already holds token 0's state, tokens 1..idx[b] are recomputed
            state_once.commit(self.rec_state, cin, idx, self.Nk, self.Nv, self.Dk, self.Dv, self.scale, T, Bw, t0=t0)
            state_slot.commit_conv_multi(self.conv_states, snaps, idx, Bw, T)
            if not getattr(TPGatedDeltaNet, "_mtp_slot_banner", False):
                TPGatedDeltaNet._mtp_slot_banner = True
                logger.info(
                    f"QWEN36_MTP_STATE_SLOT engaged: GDN commit by slot index, rec {self.rec_state.dtype} "
                    f"[{Bw}x{self.Nv}x{self.Dk}x{self.Dv}] recomputed from the base state over the accepted tokens "
                    f"of T={T} (QWEN36_MTP_GDN_STATE_ONCE) + {self.K} conv taps {self.conv_states[0].dtype}, "
                    f"2 programs/layer"
                )
            if release:
                for t in (cin,) + tuple(snaps):
                    ttnn.deallocate(t)
                del self._mtp_verify[Bw]
            return
        cmulti = self._conv_multi_check(len(hists))
        state_slot.commit_rec(self.rec_state, hists, idx, Bw, self.Nv, self.Dk, self.Dv)
        if cmulti:
            # snaps = the four verify window tensors (QWEN36_MTP_CONV_MULTI): one program writes every live tap row.
            state_slot.commit_conv_multi(self.conv_states, snaps, idx, Bw, len(hists))
        else:
            state_slot.commit_conv(self.conv_states, snaps, idx, Bw)
        if not getattr(TPGatedDeltaNet, "_mtp_slot_banner", False):
            TPGatedDeltaNet._mtp_slot_banner = True
            logger.info(
                f"QWEN36_MTP_STATE_SLOT engaged: GDN commit by slot index, rec {self.rec_state.dtype} "
                f"[{Bw}x{self.Nv}x{self.Dk}x{self.Dv}] of T={len(hists)} candidates + {self.K} conv taps "
                f"{self.conv_states[0].dtype}, 2 programs/layer (was a where-chain + copy)"
            )
        if release:
            for h in hists:
                ttnn.deallocate(h)
            for sn in snaps:
                for t in ([sn] if cmulti else sn):
                    ttnn.deallocate(t)
            del self._mtp_verify[Bw]

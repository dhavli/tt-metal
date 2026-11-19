# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU MLP: down(silu(gate(x)) * up(x)).

9B (single device): dense matmuls, full weights.
27B TP (1,4 mesh): w1/w3 column-parallel, w2 row-parallel; tt_all_reduce
reduce-scatters on meshes with a dim-1 shape (e.g. P150x4), fracturing hidden.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import os
from dataclasses import dataclass, replace

import ttnn


@dataclass(frozen=True)
class MLPWeights:
    w1: ttnn.Tensor  # gate_proj [in, out], bfloat4_b
    w2: ttnn.Tensor  # down_proj [in, out], bfloat8_b (bfloat4_b under QWEN36_MLP_DOWN_BF4)
    w3: ttnn.Tensor  # up_proj [in, out], bfloat4_b
    w_gate_up: ttnn.Tensor = None  # TP prefill: tile-pair-interleaved packed [gate|up] for fused-swiglu AGMM
    w2_mr: ttnn.Tensor = None  # QWEN36_DECODE_DRAM_SHARDED_MR: decode-only DRAM width-sharded copy of w2


def _build_gate_up(gate_w, up_w, mesh, tp, cache_path):
    """Packed [gate|up] weight for all_gather_swiglu_prefill: prepare_for_fused_swiglu tile-pair
    interleave, then column-parallel shard on the 2N dim so each device holds its interleaved slice."""
    import torch

    from models.tt_dit.utils.tensor import prepare_for_fused_swiglu

    gk = gate_w.to(torch.bfloat16).T.contiguous()  # [K=dim, N=hidden]
    uk = up_w.to(torch.bfloat16).T.contiguous()
    packed = torch.cat([gk, uk], dim=-1)  # [dim, 2*hidden], gate first
    il = prepare_for_fused_swiglu(packed, ndev=tp, gate_is_first=True)  # [dim, 2*hidden]
    return ttnn.as_tensor(
        il,
        dtype=ttnn.bfloat4_b,
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        cache_file_name=cache_path,
    )


def load_mlp_weights(mesh_device, state_dict, tensor_cache_path=None, args=None) -> MLPWeights:
    """Per-layer MLP state: gate_proj, down_proj, up_proj weights."""
    tp = getattr(args, "num_devices", 1) if args is not None else 1

    # down_proj is the last tensor of MLP size still at bfloat8_b now that
    # gate/up are bfloat4_b, and decode reads the whole weight set once per
    # token -- so it is 2.65 GiB of the per-token read. QWEN36_MLP_DOWN_BF4=1
    # trades upstream's deliberate accuracy margin for that bandwidth.
    #
    # It has to be applied at ALL THREE call sites below (TP dram-sharded, TP
    # interleaved, single-device): a two-card mesh goes through tpc.shard_w,
    # not the local load(), so patching only the single-device path changes
    # nothing on TP=2 while looking correct. See docs/GOAL-VLLM-SERVING.md.
    _down_dtype = ttnn.bfloat4_b if _tt_public_getenv("QWEN36_MLP_DOWN_BF4") == "1" else ttnn.bfloat8_b

    if tp > 1:
        # TP: w1/w3 column-parallel (shard out dim), w2 row-parallel (shard in dim).
        # DRAM-sharded memcfgs from args.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        # w1/w3 DRAM-WIDTH_SHARDED for decode (M=1 tile, ~+10% tok/s); w2 interleaved.
        # Cache uses `.dramshard` suffix — layout incompatible with interleaved cache
        # (as_tensor ignores requested memcfg on reload). Fallback if memcfgs absent.
        # 1D-decode (default) uses interleaved weights (its mcast decode matmul needs them).
        dram_sharded = (
            args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not getattr(args, "mlp_1d_decode", False)
        )

        def cache(name, tag=""):
            return str(tensor_cache_path / f"mlp.{name}.weight{tag}.tp") if tensor_cache_path else None

        # Prefill-only packed [gate|up] AGMM weight (decode keeps w1/w3; extra DRAM ~w1+w3/layer).
        wgu = (
            _build_gate_up(
                state_dict["gate_proj.weight"],
                state_dict["up_proj.weight"],
                mesh_device,
                tp,
                cache("gate_up", ".swiglu"),
            )
            if tpc.mlp_gateup_agmm_enabled(tp)
            else None
        )

        if dram_sharded and tpc.decode_mr_for(args, mesh_device):
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR expects the interleaved (1D-decode) MLP weights")
        if dram_sharded:
            return MLPWeights(
                w1=tpc.shard_w(
                    state_dict["gate_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w1_weight_memcfg,
                    cache_path=cache("gate_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w3=tpc.shard_w(
                    state_dict["up_proj.weight"],
                    mesh_device,
                    dim=-1,
                    memory_config=args.mlp_w3_weight_memcfg,
                    cache_path=cache("up_proj", ".dramshard"),
                    dtype=ttnn.bfloat4_b,
                ),
                w2=tpc.shard_w(
                    state_dict["down_proj.weight"],
                    mesh_device,
                    dim=0,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    cache_path=cache("down_proj"),
                    dtype=_down_dtype,
                ),
                w_gate_up=wgu,
            )

        # Default: INTERLEAVED DRAM shards; ttnn.linear works for decode and prefill.
        return _decode_mr_mlp(mesh_device, args, MLPWeights(
            w1=tpc.shard_w(
                state_dict["gate_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("gate_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w3=tpc.shard_w(
                state_dict["up_proj.weight"],
                mesh_device,
                dim=-1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("up_proj"),
                dtype=ttnn.bfloat4_b,
            ),
            w2=tpc.shard_w(
                state_dict["down_proj.weight"],
                mesh_device,
                dim=0,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                cache_path=cache("down_proj"),
                dtype=_down_dtype,
            ),
            w_gate_up=wgu,
        ))

    def load(name, dtype):
        t = state_dict[f"{name}.weight"].T.contiguous()  # [in, out] for ttnn.linear
        return ttnn.as_tensor(
            t,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"mlp.{name}.weight") if tensor_cache_path else None,
        )

    # gate/up: bfloat4_b (bandwidth); down: bfloat8_b (accuracy).
    return MLPWeights(
        w1=load("gate_proj", ttnn.bfloat4_b),
        w2=load("down_proj", _down_dtype),
        w3=load("up_proj", ttnn.bfloat4_b),
    )


def _decode_mr_mlp(mesh_device, args, w):
    """QWEN36_DECODE_DRAM_SHARDED_MR (tp_common): w1/w3 are decode-only once prefill runs the packed
    w_gate_up, so they are RE-LAID OUT to DRAM width-sharded in place (no extra DRAM); w2 feeds the
    prefill 2D matmul, which misreads a sharded weight, so decode gets its own sharded COPY (w2_mr).
    One counted banner per converted layer. Identity when the flag is off."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc

    mr = tpc.decode_mr_for(args, mesh_device)
    if not mr:
        return w
    if "w13" in mr:
        if w.w_gate_up is None:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR w13 relayout needs the packed w_gate_up for prefill")
        if tpc.decode_gateup_fused_enabled(args.num_devices, grid_w=getattr(args, "decode_grid_w", 11)):
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR w13 and QWEN36_DECODE_GATEUP_FUSED both claim decode gate/up")
        w = replace(w, w1=tpc.decode_mr_weight(w.w1, mr["w13"]), w3=tpc.decode_mr_weight(w.w3, mr["w13"]))
    if "w2" in mr:
        w = replace(w, w2_mr=tpc.decode_mr_weight(w.w2, mr["w2"]))
        if mr["w2"].layout == "relayout":  # refused by DecodeMRRole today; kept consistent if ever allowed
            w = replace(w, w2=w.w2_mr)
    from loguru import logger

    logger.info(f"QWEN36_DECODE_MR_LAYER engaged: mlp roles={sorted(k for k in mr if k in ('w13', 'w2'))}")
    return w


class Qwen36MLP:
    """SwiGLU feed-forward network for Qwen3.5."""

    def __init__(self, mesh_device, state_dict, tensor_cache_path=None, args=None, tt_ccl=None):
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1) if args is not None else 1
        # 1D-decode (default): small-grid 1D matmuls beat the ~80-core DRAM-sharded grid on the
        # bandwidth-bound skinny decode MLP matmuls (see test_mlp_matmul_sweep). Forces interleaved weights.
        self._mlp_1d_decode = args is not None and getattr(args, "mlp_1d_decode", False)
        # Match load_mlp_weights dram_sharded condition for layout consistency.
        self._dram_sharded = (
            self.num_devices > 1
            and args is not None
            and getattr(args, "mlp_w1_weight_memcfg", None) is not None
            and not self._mlp_1d_decode
        )
        # Prefill fused-swiglu AGMM (ff_norm skips its AG; layer.py sets _fuse_ff_agmm to match).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_gateup_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        # QWEN36_DECODE_GATEUP_FUSED: decode reuses that packed w_gate_up (already loaded, same cache file)
        # for ONE gate/up matmul with SwiGLU fused, instead of the w1/w3 1D matmuls + mul.
        self._decode_gateup_fused = self._mlp_1d_decode and tpc.decode_gateup_fused_enabled(
            self.num_devices, grid_w=getattr(args, "decode_grid_w", 11)
        )
        if self._decode_gateup_fused:
            self._gateup_decode_cfg = tpc.create_gateup_swiglu_decode_config(
                args.dim, 2 * (args.hidden_dim // self.num_devices), grid_w=args.decode_grid_w
            )
        self.weights = load_mlp_weights(mesh_device, state_dict, tensor_cache_path, args=args)
        # QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded decode gate/up and down (None = 1D path).
        _mr = tpc.decode_mr_for(args, mesh_device) if self.num_devices > 1 else {}
        self._mr_w13 = _mr.get("w13")
        self._mr_w2 = _mr.get("w2")
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        # fuse_swiglu AGMM: fp32 acc (subblock_w=4) to match GDN/attn in-proj.
        self.compute_kernel_config_agmm = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        self.compute_kernel_config_decode = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, packer_l1_acc=True
        )

    def forward(self, x, oneshot_ar=False):
        if self.num_devices > 1:
            return self._forward_tp(x, oneshot_ar=oneshot_ar)
        w = self.weights
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config
        mc = ttnn.L1_MEMORY_CONFIG if T <= 512 else ttnn.DRAM_MEMORY_CONFIG
        w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
        w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
        hidden = ttnn.mul(w1_out, w3_out, memory_config=mc)
        ttnn.deallocate(w1_out)
        ttnn.deallocate(w3_out)
        down_pc = None
        if (
            T > 1
            and getattr(self.args, "prefill_progcfg", None) is not None
            and _tt_public_getenv("QWEN9B_MLP_DOWN_AUTO") != "1"
        ):
            down_pc = self.args.prefill_progcfg(T, hidden.shape[-1], w.w2.shape[-1])
        output = ttnn.linear(hidden, w.w2, compute_kernel_config=ckc, memory_config=mc, program_config=down_pc)
        ttnn.deallocate(hidden)
        return output

    def _forward_tp(self, x, oneshot_ar=False):
        """TP forward: replicated input; reduce-scatter output fractured on hidden dim
        (oneshot_ar: all-gather + add, output replicated full width -- QWEN36_ONESHOT_AR decode)."""
        from models.demos.blackhole.qwen36.tt import tp_common as tpc
        from models.tt_transformers.tt.ccl import tt_all_reduce

        w = self.weights
        args = self.args
        T = x.shape[1] if len(x.shape) >= 3 else 1
        ckc = self.compute_kernel_config_decode if T <= 1 else self.compute_kernel_config

        mc = ttnn.DRAM_MEMORY_CONFIG
        _silu_fused = False
        _silu_in_mul = False
        # Prefill: x is K-sharded (ff_norm skipped AG); fused AG + [gate|up] + SwiGLU
        _fused_gu = self._fuse_gateup_agmm and x.shape[-2] > ttnn.TILE_SIZE and w.w_gate_up is not None
        # Decode: same packed [gate|up] weight, one minimal_matmul with SwiGLU in-kernel (no AG: x is full-K).
        _fused_gu_decode = self._decode_gateup_fused and x.shape[-2] <= ttnn.TILE_SIZE and w.w_gate_up is not None
        if _fused_gu:
            hidden = tpc.all_gather_swiglu_prefill(
                x, w.w_gate_up, self.tt_ccl, self.compute_kernel_config_agmm, args.ccl_topology()
            )
            _silu_fused = True
        elif _fused_gu_decode:
            hidden = tpc.gateup_swiglu_decode(x, w.w_gate_up, self._gateup_decode_cfg, ckc)
            _silu_fused = True
        elif getattr(self, "_dram_sharded", False) and x.shape[-2] <= ttnn.TILE_SIZE:
            # DRAM-WIDTH_SHARDED w1/w3 decode (M=1 tile). Prefill uses fused AGMM above.
            x_sh = ttnn.to_memory_config(x, args.act_shard_hidden)
            w1_out = ttnn.linear(
                x_sh,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_sh,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_progcfg,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_sh)
            # Keep gate/up in L1 for mul → w2 (avoid L1→DRAM→L1).
            w1_out = ttnn.to_memory_config(w1_out, ttnn.L1_MEMORY_CONFIG)
            w3_out = ttnn.to_memory_config(w3_out, ttnn.L1_MEMORY_CONFIG)
        elif self._mr_w13 is not None and x.shape[-2] <= ttnn.TILE_SIZE:
            # QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded gate (SiLU fused) and up on the
            # re-laid-out w1/w3; one activation reshard shared by both. Outputs L1 like the 1D path.
            mr = self._mr_w13
            x_owned = x.memory_config() != mr.act_memcfg
            x_sh = ttnn.to_memory_config(x, mr.act_memcfg) if x_owned else x
            _mr_mc = None if tpc.mr_mlp_sharded() else ttnn.L1_MEMORY_CONFIG
            _silu_in_mul = tpc.mr_silu_unfused()
            w1_out = tpc.decode_mr_matmul(x_sh, w.w1, mr, ckc, out_memory_config=_mr_mc, silu=not _silu_in_mul)
            w3_out = tpc.decode_mr_matmul(x_sh, w.w3, mr, ckc, out_memory_config=_mr_mc)
            if x_owned:
                ttnn.deallocate(x_sh)
            _silu_fused = True
        elif self._mlp_1d_decode and x.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode matmuls on a small explicit grid, silu fused in the w1 progcfg.
            # mcast_in0 needs interleaved in0, but ff-norm hands us a width-shard -> interleave first.
            x_il = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG)
            w1_out = ttnn.linear(
                x_il,
                w.w1,
                compute_kernel_config=ckc,
                program_config=args.mlp_w1_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            w3_out = ttnn.linear(
                x_il,
                w.w3,
                compute_kernel_config=ckc,
                program_config=args.mlp_w3_decode_1d_progcfg,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(x_il)
            _silu_fused = True
        elif x.shape[-2] > ttnn.TILE_SIZE:
            # Prefill (M>1 tile, compute-bound): FPU-tuned 2D config (grid width -> 1x4 subblock,
            # in0_block_w=4) beats ttnn-auto's 1x1 stall ~2.7x (test_mlp_matmul_sweep_prefill). SILU fused.
            seq = x.shape[-2]
            # max_cols = device worker-grid width (11 on BH): wide grid (gate/up -> 9x10) vs old 8-wide.
            _gw = getattr(args, "decode_grid_w", 8)
            # TP-selected prefill tuning; absent (single-device 9B) => frozen TP=4 behavior.
            _pt = getattr(args, "prefill_tuning", None)
            pc_gate = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w1.shape[-1], fused_activation=ttnn.UnaryOpType.SILU, max_cols=_gw, tuning=_pt
            )
            pc_up = tpc.create_prefill_mlp_matmul_program_config(
                seq, args.dim, w.w3.shape[-1], max_cols=_gw, tuning=_pt
            )
            # L1 output (gate/up outputs; down output via mc_out below): +FPU, avoids the DRAM round-trip
            # (test_mlp_matmul_sweep_prefill *_outL1). The [seq,N] tensors fit L1 at the prefill chunk.
            w1_out = ttnn.linear(
                x, w.w1, compute_kernel_config=ckc, program_config=pc_gate, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            w3_out = ttnn.linear(
                x, w.w3, compute_kernel_config=ckc, program_config=pc_up, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            _silu_fused = True
        else:
            # Interleaved weights: auto matmul program for decode and prefill.
            w1_out = ttnn.linear(x, w.w1, activation="silu", compute_kernel_config=ckc, memory_config=mc)
            w3_out = ttnn.linear(x, w.w3, compute_kernel_config=ckc, memory_config=mc)
            _silu_fused = True

        # gated activation (down-proj INPUT): L1 in decode, DRAM in prefill. The L1 win is OUTPUT-only;
        # keeping both down input (hidden) and output (partial) in L1 at seq 2048 overflows L1.
        _prefill_tuned = x.shape[-2] > ttnn.TILE_SIZE and _silu_fused
        # gate * up (skipped when _fused_gu / _fused_gu_decode already produced `hidden` with SwiGLU in-kernel).
        if not (_fused_gu or _fused_gu_decode):
            mc_out = ttnn.L1_MEMORY_CONFIG if x.shape[-2] <= ttnn.TILE_SIZE else mc
            if w1_out.is_sharded():  # QWEN36_DECODE_MR_MLP_SHARDED: gate/up share one shard spec; keep it
                mc_out = w1_out.memory_config()
            # Standalone silu only on DRAM-sharded decode path (SILU not fused there).
            if _silu_fused and _silu_in_mul:  # QWEN36_MR_SILU_UNFUSED: SiLU as the mul's input_a activation
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
                                  **tpc.fast_mul("mlp", w3_out.shape[-2]))
                ttnn.deallocate(w1_out)
            elif _silu_fused:
                hidden = ttnn.mul(w1_out, w3_out, memory_config=mc_out, **tpc.fast_mul("mlp", w3_out.shape[-2]))
                ttnn.deallocate(w1_out)
            else:
                w1_act = ttnn.silu(w1_out, memory_config=mc_out)
                ttnn.deallocate(w1_out)
                hidden = ttnn.mul(w1_act, w3_out, memory_config=mc_out, **tpc.fast_mul("mlp", w3_out.shape[-2]))
                ttnn.deallocate(w1_act)
            ttnn.deallocate(w3_out)
        # Prefill w2: 2D progcfg on (8,10); decode (M<=32) keeps ttnn-auto.
        w2_pc = None
        if self._mlp_1d_decode and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # 1D mcast decode down-proj on a small explicit grid (~16 cores).
            w2_pc = args.mlp_w2_decode_1d_progcfg
        elif hidden.shape[-2] > ttnn.TILE_SIZE and tpc.prefill_fs8("mlp_down"):
            # QWEN36_PREFILL_FS8: full-sync fp32 DST (8-tile subblocks), K block <= 16 (packer_l1_acc kept off).
            w2_pc = tpc.create_prefill_fs8_program_config(
                hidden.shape[-2], hidden.shape[-1], w.w2.shape[-1], getattr(args, "decode_grid_w", 8), "mlp_down"
            )
            ckc = tpc.prefill_fs8_compute(packer_l1_acc=False)
        elif hidden.shape[-2] > ttnn.TILE_SIZE:
            # Prefill down-proj: subblock-tuned 2D config with the wide grid (max_cols=device width),
            # off the generic 8-wide prefill_progcfg. Output L1 via mc_w2_out below.
            w2_pc = tpc.create_prefill_mlp_matmul_program_config(
                hidden.shape[-2],
                hidden.shape[-1],
                w.w2.shape[-1],
                max_cols=getattr(args, "decode_grid_w", 8),
                tuning=getattr(args, "prefill_tuning", None),
            )
        # down-proj OUTPUT in L1 for the tuned prefill path (DRAM input `hidden` + L1 output = the
        # validated sweep outL1 config; tt_all_reduce already consumes an L1 partial).
        mc_w2_out = ttnn.L1_MEMORY_CONFIG if (x.shape[-2] <= ttnn.TILE_SIZE or _prefill_tuned) else mc
        if self._mr_w2 is not None and hidden.shape[-2] <= ttnn.TILE_SIZE:
            # QWEN36_DECODE_DRAM_SHARDED_MR: decode down-proj on the sharded copy (prefill keeps w.w2).
            # QWEN36_DECODE_MR_OUT_DIRECT (one-shot AR only): w2 writes the all_reduce_async input layout directly
            _w2_mc = tpc.mr_ar_target(self.tt_ccl, mc_w2_out, role="w2") if oneshot_ar else mc_w2_out
            partial = tpc.decode_mr_matmul(hidden, w.w2_mr, self._mr_w2, ckc, out_memory_config=_w2_mc)
        else:
            # QWEN36_PREFILL_MLP_LOWP partial=bf8 (c113): prefill w2 output, i.e. the reduce-scatter payload, in bf8.
            _w2_dt = tpc.prefill_mlp_lowp("partial") if hidden.shape[-2] > ttnn.TILE_SIZE else None
            _w2_kw = {"dtype": _w2_dt} if _w2_dt is not None else {}

            def _w2(pc, x_in=None):
                return ttnn.linear(
                    hidden if x_in is None else x_in, w.w2, compute_kernel_config=ckc, memory_config=mc_w2_out,
                    program_config=pc, **_w2_kw
                )

            # QWEN36_PREFILL_W2_T11: the same w2 on the transposed 11x10 grid (all 110 cores), bit-exact by construction;
            # the first (eager warmup) call self-tests it bitwise against the shipped 10x10 program.
            w2_t11_pc = None
            if (w2_pc is not None and hidden.shape[-2] > ttnn.TILE_SIZE and not tpc.prefill_fs8("mlp_down")
                    and tpc.prefill_w2_t11()):
                w2_t11_pc = tpc.create_prefill_w2_t11_program_config(hidden.shape[-2], w.w2.shape[-1], w2_pc)
            if w2_t11_pc is not None:
                def _zeroed():
                    xz = tpc.zero_first_rows(hidden)
                    try:
                        return _w2(w2_t11_pc, xz)
                    finally:
                        ttnn.deallocate(xz)

                tpc.prefill_selftest(
                    "QWEN36_PREFILL_W2_T11",
                    ref=lambda: _w2(w2_pc),
                    cand=_zeroed if tpc.prefill_w2_t11_mustfail() else (lambda: _w2(w2_t11_pc)),
                    wrong=_zeroed,
                    desc=f"w2 M={hidden.shape[-2]} K={hidden.shape[-1]} N={w.w2.shape[-1]}",
                    ok_banner="QWEN36_PREFILL_W2_T11 self-test exact",
                )
                w2_pc = w2_t11_pc
            partial = _w2(w2_pc)
        ttnn.deallocate(hidden)

        if oneshot_ar:
            return tpc.all_reduce_oneshot_decode(partial, self.tt_ccl, args.ccl_topology())
        if partial.shape[-2] > ttnn.TILE_SIZE:  # prefill: QWEN36_PREFILL_RS_CFG knobs (default = tt_all_reduce)
            return tpc.prefill_all_reduce(partial, self.device, self.tt_ccl, args.ccl_topology())
        # tt_all_reduce on (1,4) mesh reduce-scatters to hidden dim (dim=3).
        out = tt_all_reduce(
            partial,
            self.device,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(partial),
        )
        return out

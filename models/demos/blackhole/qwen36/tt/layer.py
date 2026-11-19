# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Hybrid TransformerBlock for Qwen3.5-9B.

Dispatches to either Gated DeltaNet (linear attention) or Gated Full Attention
based on the layer index. Both share the same RMSNorm + residual pattern and MLP.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import os

from loguru import logger

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt.attention import AttentionConfig, Qwen36GatedAttention
from models.demos.blackhole.qwen36.tt.gdn import GDNConfig, Qwen36GatedDeltaNet
from models.demos.blackhole.qwen36.tt.mlp import Qwen36MLP
from models.demos.blackhole.qwen36.tt.weight_cache import substate
from models.tt_transformers.tt.common import Mode

_PREFILL_DNORM_ADD = _tt_public_getenv("QWEN36_PREFILL_DNORM_ADD", "0") == "1"  # dnorm_fuse.py validates the value
_XADD_LOGGED = []


class Qwen36DecoderLayer:
    """Single transformer layer with hybrid attention dispatch.

    Pattern: x → attention_norm → attention → residual → ff_norm → MLP → residual
    Attention is either GatedAttention (full, with RoPE) or GatedDeltaNet (linear).
    """

    # QWEN36_MTP_BATCH (docs/MTP-BATCH.md) checks this marker and fails closed without it: forward()
    # accepts verify_T / group_io / kv_only (batched speculative verify and the drafter's KV reseed).
    MTP_BATCH_VERIFY = 1

    def __init__(self, mesh_device, args, state_dict, layer_num, tensor_cache_path=None, tt_ccl=None):
        self.layer_num = layer_num
        self.device = mesh_device
        self.args = args
        self.tt_ccl = tt_ccl
        self.num_devices = getattr(args, "num_devices", 1)
        self.is_full_attention = args.is_full_attention_layer(layer_num)

        prefix = f"layers.{layer_num}"

        # Zero-centered RMSNorm (Qwen3.5): output = x_normed * (1 + weight). The
        # framework RMSNorm applies the +1 internally via add_unit_offset=True and
        # is mesh-aware (replicates the weight across a MeshDevice).
        #
        # Single device: plain RMSNorm on the full hidden state (validated path).
        # TP (27B on a (1,4) mesh): the residual stream is fractured along the
        # hidden dim, so each norm is wrapped in the framework DistributedNorm,
        # which all-gathers (PREFILL: distributed rmsnorm + gather; DECODE:
        # gather-then-norm) to hand the modules a replicated full-dim input —
        # exactly as models/demos/qwen35_27b does via the framework decoder.
        # Prefill fuses the norm all-gather into the in-proj matmul (all_gather_minimal_matmul_async):
        # GDN qkvzab and full-attn QKV. attention_norm then skips its post-norm AG (prefill only;
        # decode gathers pre-norm). Gates must match the module-side _fuse_agmm gates.
        self._fuse_norm_agmm = self.num_devices > 1 and (
            (not self.is_full_attention and getattr(args, "gdn_qkvz_weight_memcfg", None) is not None)
            or (self.is_full_attention and getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None)
        )
        self.attention_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "input_layernorm",
            tensor_cache_path,
            tt_ccl,
            "attention_norm",
            enable_all_gather=not self._fuse_norm_agmm,
        )
        # Prefill: ff_norm skips AG (fused into gate/up AGMM); decode gathers pre-norm so this is a no-op there.
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        self._fuse_ff_agmm = tpc.mlp_gateup_agmm_enabled(self.num_devices)
        self.ffn_norm = self._make_norm(
            mesh_device,
            args,
            state_dict,
            layer_num,
            "post_attention_layernorm",
            tensor_cache_path,
            tt_ccl,
            "ff_norm",
            enable_all_gather=not self._fuse_ff_agmm,
        )

        if self.num_devices > 1:
            # Tensor-parallel modules (sharded weights from the raw substate).
            # Cache the sharded mesh weights to disk so re-runs skip the (slow,
            # single-threaded) reorder+shard of the full 27B.
            tp_cache = (tensor_cache_path / f"layers.{layer_num}" / "tp") if tensor_cache_path else None
            if self.is_full_attention:
                from models.demos.blackhole.qwen36.tt.attention.tp import TPAttention, load_attention_weights_tp

                tw = load_attention_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.self_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPAttention(mesh_device, args, tw, tt_ccl)
            else:
                from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet, load_gdn_weights_tp

                tw = load_gdn_weights_tp(
                    mesh_device, substate(state_dict, f"layers.{layer_num}.linear_attn"), args, cache_dir=tp_cache
                )
                self.attention = TPGatedDeltaNet(mesh_device, args, tw, tt_ccl)
        elif self.is_full_attention:
            attn_state = substate(state_dict, f"layers.{layer_num}.self_attn")
            attn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedAttention(mesh_device, AttentionConfig.from_args(args), attn_state, attn_cache)
        else:
            gdn_state = substate(state_dict, f"layers.{layer_num}.linear_attn")
            gdn_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
            self.attention = Qwen36GatedDeltaNet(mesh_device, GDNConfig.from_args(args), gdn_state, gdn_cache)

        mlp_state = substate(state_dict, f"layers.{layer_num}.mlp")
        mlp_cache = (tensor_cache_path / f"layers.{layer_num}") if tensor_cache_path else None
        self.feed_forward = Qwen36MLP(mesh_device, mlp_state, mlp_cache, args=args, tt_ccl=tt_ccl)

    def _make_norm(
        self,
        mesh_device,
        args,
        state_dict,
        layer_num,
        weight_key,
        tensor_cache_path,
        tt_ccl,
        ag_key,
        enable_all_gather=True,
    ):
        """Build the per-layer RMSNorm; wrap in DistributedNorm when TP>1.

        On a single device this returns the same plain RMSNorm the validated 9B
        path used. The DistributedNorm wrapper (TP>1) mirrors tt_transformers
        decoder.py and handles the fractured->replicated transition.
        """
        norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key=weight_key,
            state_dict_prefix=f"layers.{layer_num}.",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            return DistributedNorm(
                norm, args, tt_ccl=tt_ccl, TG=args.is_galaxy, ag_config_key=ag_key, enable_all_gather=enable_all_gather
            )
        return norm

    def _prefill_add_norm(self, x, attn_output, norm_config, dn=None):
        """QWEN36_PREFILL_DNORM_ADD (dnorm_fuse.run_add): h = x + attn_output folded into ffn_norm's fused PRE pass.
        Only where ffn_norm's prefill forward is exactly RMSNorm._distributed_rmsnorm (DistributedNorm with no post-norm
        all-gather, h already DRAM interleaved); returns (h, ff_input) or None (today's add + norm serve it).
        dn: the norm to fold into (default ffn_norm; QWEN36_PREFILL_DNORM_XADD passes the next layer's attention_norm)."""
        from models.demos.blackhole.qwen36.tt import dnorm_fuse

        dn = self.ffn_norm if dn is None else dn
        n = getattr(dn, "norm", None)
        if n is None or getattr(dn, "enable_all_gather", True) or not getattr(n, "is_distributed", None):
            return None
        if not n.is_distributed(Mode.PREFILL) or x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            return None
        nc = norm_config or {}
        od = nc.get("distributed_output_dtype")
        mc = nc.get("distributed_output_mem_config") or ttnn.DRAM_MEMORY_CONFIG
        if not dnorm_fuse.add_usable(x, attn_output, od if od is not None else x.dtype):
            return None
        return dnorm_fuse.run_add(n, x, attn_output, n.eps, n.weight_distributed, mc, od)

    def forward(
        self,
        x,
        cos=None,
        sin=None,
        mode="decode",
        chunk_size=128,  # = GDN long_prefill_chunk_size; the only size the chunk-seq prefill kernel supports
        position_tensor=None,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        valid_len=None,
        gdn_collect=False,
        oneshot_ar=False,
        verify_T=1,
        group_io=None,
        kv_only=False,
        fold_next=None,
    ):
        _norm_mode = Mode.PREFILL if mode == "prefill" else Mode.DECODE
        # QWEN36_PREFILL_DNORM_XADD: the previous layer folded x = h + ff_out into THIS layer's attention-norm PRE pass
        # and left (x, attn_input) here; used only when x IS that tensor, otherwise freed and the norm runs as today.
        _pre = self.__dict__.pop("_xadd_pre", None)
        if _pre is not None and not (mode == "prefill" and _pre[0] is x):
            ttnn.deallocate(_pre[1])
            _pre = None
        # QWEN36_ONESHOT_AR (Qwen36Model._forward_decode passes it): x is the REPLICATED full-width residual,
        # so the norms skip their pre-norm all-gather and every module all-reduces to a replicated output.
        oneshot_ar = oneshot_ar and mode == "decode" and self.num_devices > 1
        # QWEN36_NORM_AG_CFG: the fractured decode path's pre-norm all-gathers with QB2 CCL knobs.
        # QWEN36_RESID_L1 (replicated seam only): the residual lives WIDTH-sharded in L1 in the norms' input
        # layout, so the adds run sharded and the norms' reshard is a no-op (only layer 0 moves x from DRAM).
        _resid_cfg = None
        if self.num_devices > 1 and mode == "decode":
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            _knob_ag = not oneshot_ar and tpc.norm_ag_cfg_enabled()
            _resid_cfg = tpc.resid_l1_memcfg(self.tt_ccl) if oneshot_ar else None
            if _resid_cfg is not None and x.memory_config() != _resid_cfg:
                x = ttnn.to_memory_config(x, _resid_cfg)  # caller keeps (and frees) its DRAM x, as on every path
        if self.num_devices > 1:
            # TP: DistributedNorm uses the framework's per-norm memory configs.
            _attn_norm_config = self.args.get_norm_config("attn", _norm_mode)
            # PREFILL: distributed rmsnorm outputs in L1 so the fused in-proj AGMM gathers from L1, not DRAM.
            if _norm_mode == Mode.PREFILL:
                _attn_norm_config = {**_attn_norm_config, "distributed_output_mem_config": ttnn.L1_MEMORY_CONFIG}
            # DECODE ff_norm uses the attn_norm layout (act_shard_hidden, 32-core) so Qwen36MLP's input reshard is a no-op and the norm runs on 32 cores not 8; PREFILL keeps the framework ff config.
            if _norm_mode == Mode.DECODE:
                _ff_norm_config = self.args.get_norm_config("attn", _norm_mode)
            else:
                # ff_norm output stays DRAM: L1 keeps the full-width norm resident across the whole MLP,
                # clashing with each matmul's CBs (w1/w3/w2) for no gain. Verified dead end; keep DRAM.
                _ff_norm_config = self.args.get_norm_config("ff", _norm_mode)
            if _norm_mode == Mode.PREFILL:
                from models.demos.blackhole.qwen36.tt import tp_common as _tpc

                _nd = _tpc.prefill_norm_dtype()  # QWEN36_PREFILL_NORM_DTYPE (c118): norms emit bf8 into bf8-in0 AGMMs
                if _nd is not None:
                    _attn_norm_config = {**(_attn_norm_config or {}), "distributed_output_dtype": _nd}
                    _ff_norm_config = {**(_ff_norm_config or {}), "distributed_output_dtype": _nd}
        else:
            # In decode the norm output stays in L1 (as the old rms_norm_ttnn(memory_config=L1) did);
            # in prefill the framework RMSNorm returns interleaved DRAM (matches the old None default).
            _attn_norm_config = _ff_norm_config = (
                {"output_mem_config": ttnn.L1_MEMORY_CONFIG} if mode == "decode" else None
            )
        if oneshot_ar:
            attn_input = tpc.norm_replicated_decode(self.attention_norm, x, _attn_norm_config)
        elif self.num_devices > 1 and mode == "decode" and _knob_ag:
            attn_input = tpc.norm_gathered_decode(
                self.attention_norm, x, _attn_norm_config, self.tt_ccl, self.args.ccl_topology()
            )
        elif _pre is not None:
            attn_input = _pre[1]
        else:
            attn_input = self.attention_norm(x, mode=_norm_mode, norm_config=_attn_norm_config)

        if self.num_devices > 1:
            # TP modules: input is the gathered (full-dim) norm output [1,1,B/S,dim];
            # output is fractured along dim=3. cos/sin are in rope_tp format.
            if self.is_full_attention:
                if mode == "prefill":
                    # Contract/vLLM path supplies a page_table → paged KV prefill; the
                    # demo path (no page_table) uses the internal concat caches.
                    if page_table is not None:
                        attn_output = self.attention.forward_prefill_paged(
                            attn_input,
                            cos,
                            sin,
                            page_table,
                            chunk_page_table=chunk_page_table,
                            chunk_start_idx=chunk_start_idx if chunk_start_idx is not None else 0,
                            chunk_start_idx_tensor=chunk_start_idx_tensor,
                            **({"kv_only": True} if kv_only else {}),
                        )
                    else:
                        attn_output = self.attention.forward_prefill(attn_input, cos, sin)
                else:
                    attn_output = self.attention.forward_decode(
                        attn_input,
                        position_tensor,
                        cos,
                        sin,
                        page_table=page_table,
                        oneshot_ar=oneshot_ar,
                        **({"group_io": group_io, "kv_only": kv_only} if group_io is not None else {}),
                    )
            else:
                # GDN carries its recurrent/conv state internally (capture_state on
                # prefill, read on decode); it has no paged KV, so page_table is N/A.
                if mode == "prefill":
                    if gdn_collect:
                        # Batched per-user prefill: stash this user's from-scratch state for
                        # assembly into row u of the batched buffers (finalize_pending later).
                        attn_output = self.attention.forward_prefill_collect(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len
                        )
                    else:
                        attn_output = self.attention.forward_prefill(
                            attn_input, chunk_size=chunk_size, valid_len=valid_len, capture_state=True
                        )
                elif verify_T > 1:
                    attn_output = self.attention.forward_verify(attn_input, verify_T, oneshot_ar=oneshot_ar)
                else:
                    attn_output = self.attention.forward_decode(attn_input, oneshot_ar=oneshot_ar)
        elif self.is_full_attention:
            attn_output = self.attention.forward(
                attn_input,
                cos,
                sin,
                position_tensor=position_tensor,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
            )
        else:
            deltanet_mode = "chunk" if mode == "prefill" else "recurrent"
            attn_output = self.attention.forward(
                attn_input, mode=deltanet_mode, chunk_size=chunk_size, valid_len=valid_len
            )
        ttnn.deallocate(attn_input)
        if attn_output is None:  # QWEN36_MTP_BATCH kv_only: the KV write was the whole job
            return None

        _folded = None
        if _PREFILL_DNORM_ADD and mode == "prefill" and self.num_devices > 1 and not oneshot_ar and _resid_cfg is None:
            _folded = self._prefill_add_norm(x, attn_output, _ff_norm_config)
        if _folded is not None:
            h, ff_input = _folded
        else:
            h = ttnn.add(x, attn_output, **({"memory_config": _resid_cfg} if _resid_cfg is not None else {}))
        ttnn.deallocate(attn_output)

        if _folded is not None:
            ff_output = self.feed_forward.forward(ff_input)
        elif oneshot_ar:
            ff_input = tpc.norm_replicated_decode(self.ffn_norm, h, _ff_norm_config)
            ff_output = self.feed_forward.forward(ff_input, oneshot_ar=True)
        elif self.num_devices > 1 and mode == "decode" and _knob_ag:
            ff_input = tpc.norm_gathered_decode(self.ffn_norm, h, _ff_norm_config, self.tt_ccl, self.args.ccl_topology())
            ff_output = self.feed_forward.forward(ff_input)
        else:
            ff_input = self.ffn_norm(h, mode=_norm_mode, norm_config=_ff_norm_config)
            ff_output = self.feed_forward.forward(ff_input)
        ttnn.deallocate(ff_input)

        _xf = None
        if fold_next is not None and _folded is not None and mode == "prefill":
            from models.demos.blackhole.qwen36.tt import dnorm_fuse

            if dnorm_fuse.xadd_enabled():
                # the next layer's attention norm config is this layer's (both from args.get_norm_config("attn")
                # + the prefill L1 / norm-dtype overrides above)
                _xf = fold_next._prefill_add_norm(h, ff_output, _attn_norm_config, dn=fold_next.attention_norm)
        if _xf is not None:
            output, fold_next._xadd_pre = _xf[0], _xf
            if not _XADD_LOGGED:
                _XADD_LOGGED.append(1)
                logger.info("QWEN36_PREFILL_DNORM_XADD engaged: x = h + ff_out folded into the next layer's "
                            "attention-norm PRE pass")
        else:
            output = ttnn.add(h, ff_output, **({"memory_config": _resid_cfg} if _resid_cfg is not None else {}))
        ttnn.deallocate(h)
        ttnn.deallocate(ff_output)

        return output

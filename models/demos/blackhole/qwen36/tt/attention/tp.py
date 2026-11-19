# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel full-attention for Qwen3.5 (validated 64k+ on 27B).

Q/K-norm: HF-correct (1+weight) uniformly at prefill and decode.
Keep Q bf16 into SDPA unless the K cache is quantized (QWEN36_KV_DTYPE / QWEN_SDPA_BF8).
Weights interleaved per device; x replicated in, output reduce-scattered on dim=3.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem
import os

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt import tp_common as tpc
from models.demos.blackhole.qwen36.tt.attention import attn_post_fold, kv_fold
from models.demos.blackhole.qwen36.tt.attention.rope_tp import apply_partial_rope_decode, apply_partial_rope_prefill
from models.tt_transformers.tt.ccl import tt_all_reduce


# QWEN36_SDPA_SPEC_FOLD (tt-metal patch 0034, docs/analysis/sdpa-spec-fold.md): MTP verify issues ONE paged decode
# SDPA over all T candidate groups (KV read once per user and kv head, per-candidate causal mask) instead of T calls.
# "1" engages it, "mustfail" engages it with every candidate's causal bound one position too far (equiv must FAIL).
# The op reads TT_SDPA_DEC_SPEC_FOLD once per process, so it is set here, at import, before any decode SDPA runs.
_SPEC_FOLD = _tt_public_getenv("QWEN36_SDPA_SPEC_FOLD", "0")
if _SPEC_FOLD not in ("0", "1", "mustfail"):
    raise RuntimeError(f"QWEN36_SDPA_SPEC_FOLD must be 0, 1 or mustfail; got {_SPEC_FOLD!r}")
if _SPEC_FOLD != "0":
    _tt_fold = _tt_public_getenv("TT_SDPA_DEC_SPEC_FOLD")
    if _tt_fold not in (None, "", _SPEC_FOLD):
        raise RuntimeError(f"QWEN36_SDPA_SPEC_FOLD={_SPEC_FOLD} conflicts with TT_SDPA_DEC_SPEC_FOLD={_tt_fold}")
    os.environ["TT_SDPA_DEC_SPEC_FOLD"] = _SPEC_FOLD


def spec_fold_enabled():
    return _SPEC_FOLD != "0"


# Deepest T one folded SDPA call takes (verify-depth-gt4). 0034 multiplies the Q-row-sized CBs by T (PNHt = T row
# tiles): ttsim allocates 714,112 B at T=4 and 1,209,984 B at T=8 of a 1,461,248 B L1 (B=8, bf8/bf4,
# docs/analysis/sdpa-spec-fold.md "L1 budget"), so T > 8 would trip the op's L1 TT_FATAL. A verify deeper than this
# runs as ceil(T / SPEC_FOLD_TCHUNK) folded calls over balanced candidate chunks (each one KV read per user and kv
# head, the per-candidate causal mask as before), outputs concatenated in candidate order. T <= 8: one call, as today.
SPEC_FOLD_TCHUNK = 8


def spec_fold_chunks(T, cap=SPEC_FOLD_TCHUNK):
    """[(j0, j1)] candidate chunks of a T-candidate verify: one chunk when T <= cap, else ceil(T/cap) balanced
    chunks (sizes differ by at most one, every size >= 2 since T > cap >= 2)."""
    T, cap = int(T), int(cap)
    if T <= cap:
        return [(0, T)]
    n = -(-T // cap)
    base, rem = divmod(T, n)
    out, j = [], 0
    for c in range(n):
        size = base + (1 if c < rem else 0)
        out.append((j, j + size))
        j += size
    return out


class FoldGroupIO(list):
    """group_io [(pos_j, page_table), ...] plus fold_pos: the candidate-major concatenation of the pos_j ([T*Bw]
    int32, entry j*Bw + b), which the folded SDPA takes as its cur_pos."""

    fold_pos = None


def load_attention_weights_tp(mesh, state_dict, args, cache_dir=None):
    """Shard one full-attention layer's weights across the mesh."""
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
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    def c(n):
        return str(cache_dir / n) if cache_dir is not None else None

    tw = {}
    # Column-parallel q/k/v: fused [q+gate|k|v] per device, or separate DRAM-sharded weights.
    # Distinct cache names — as_tensor reload ignores requested memcfg.
    fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
    # De-interleave [q,gate] per head → contiguous q/gate slices (avoids ~5.3ms relayout).
    qg_deint = fused_qkv

    # TP > n_kv_heads (e.g. 27B's 4 KV heads on TP=8): there is no whole KV head per device, so
    # pre-expand K/V to tp*head_dim rows where device d holds the head its GQA query group maps
    # to (devices 2d, 2d+1 share head d at TP=8). The per-device slicing below is then uniform.
    # No-op when tp <= n_kv_heads, so TP=4 weights stay bit-identical.
    kv_rep = lambda w: tpc.replicate_kv_weight(w, args.n_kv_heads, args.num_devices, args.head_dim)
    k_proj, v_proj = kv_rep(state_dict["k_proj.weight"]), kv_rep(state_dict["v_proj.weight"])

    if fused_qkv:
        if qg_deint:
            fused = tpc.prepare_attn_qkv_deint(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads,
                args.head_dim,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        else:
            fused = tpc.prepare_attn_qkv(
                state_dict["q_proj.weight"],
                k_proj,
                v_proj,
                args.n_local_heads * args.head_dim * 2,
                args.n_local_kv_heads * args.head_dim,
                args.num_devices,
            )
        # proj_1d_decode: interleaved weight (fast small-grid 1D decode matmul; prefill AGMM verified
        # bit-identical on interleaved — test_agmm_accepts_interleaved_weight). Distinct cache suffix.
        _proj1d = getattr(args, "proj_1d_decode", False)
        _base = "wqkv_fused_qkvg" if qg_deint else "wqkv_fused"
        tw["wqkv_fused"] = tpc.shard_w(
            fused,
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if _proj1d else args.attn_qkv_fused_weight_memcfg,
            cache_path=c(_base + (".il" if _proj1d else ".dramshard")),
            dtype=_wdt,
        )
    else:
        qkv_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        qg_mc = args.attn_qg_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        k_mc = args.attn_k_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        v_mc = args.attn_v_weight_memcfg if qkv_sharded else ttnn.DRAM_MEMORY_CONFIG
        tag = ".dramshard" if qkv_sharded else ""
        tw["wqkv"] = tpc.shard_w(
            state_dict["q_proj.weight"],
            mesh,
            dim=-1,
            memory_config=qg_mc,
            cache_path=c("wqkv" + tag),
            dtype=_wdt,
        )
        # k_proj/v_proj are the KV-replicated weights: shard_w splits tp*head_dim rows evenly, so
        # each device lands on its GQA-assigned head instead of a fraction of one.
        tw["wk"] = tpc.shard_w(
            k_proj,
            mesh,
            dim=-1,
            memory_config=k_mc,
            cache_path=c("wk" + tag),
            dtype=_wdt,
        )
        tw["wv"] = tpc.shard_w(
            v_proj,
            mesh,
            dim=-1,
            memory_config=v_mc,
            cache_path=c("wv" + tag),
            dtype=_wdt,
        )
    # Row-parallel wo (reduce-scatter after): DRAM-width-sharded like the in-proj — decode tput win.
    wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
    tw["wo"] = tpc.shard_w(
        state_dict["o_proj.weight"],
        mesh,
        dim=0,
        memory_config=args.attn_wo_weight_memcfg if wo_sharded else ttnn.DRAM_MEMORY_CONFIG,
        cache_path=c("wo.dramshard" if wo_sharded else "wo"),
        dtype=_wdt,
    )
    if getattr(args, "num_devices", 1) > 1 and tpc.attn_outproj_colpar():
        # QWEN36_ATTN_OUTPROJ_COLPAR (c90): COLUMN-parallel copy of wo for prefill, the attention twin of the GDN
        # out-proj colpar (c88pre on c86: 32k makespan -3.0%). Decode keeps the row-sharded wo.
        tw["wo_colpar"] = tpc.shard_w(
            state_dict["o_proj.weight"],
            mesh,
            dim=-1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_path=c("wo.colpar"),
            dtype=_wdt,
        )
    # QWEN36_DECODE_DRAM_SHARDED_MR (tp_common): decode-only DRAM width-sharded views of the fused QKV
    # (relayout by default: prefill's AGMM reads the same tensor) and of wo (always a copy: the prefill
    # 2D matmul must keep the interleaved wo). Counted banner: one per converted layer.
    mr = tpc.decode_mr_for(args, mesh)
    if "attn_qkv" in mr:
        if not fused_qkv:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR attn_qkv needs the fused QKV weight")
        tw["wqkv_fused_mr"] = tpc.decode_mr_weight(tw["wqkv_fused"], mr["attn_qkv"])
        if mr["attn_qkv"].layout == "relayout":
            tw["wqkv_fused"] = tw["wqkv_fused_mr"]
    if "out" in mr:
        if wo_sharded:
            raise RuntimeError("QWEN36_DECODE_DRAM_SHARDED_MR out expects the interleaved wo")
        tw["wo_mr"] = tpc.decode_mr_weight(tw["wo"], mr["out"])
    if mr:
        logger.info(f"QWEN36_DECODE_MR_LAYER engaged: attention roles={sorted(k for k in mr if k in ('attn_qkv', 'out'))}")
    # QK norms: HF-correct zero-centered (1+weight), used uniformly at prefill AND decode
    tw["q_norm"] = tpc.replicate(state_dict["q_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    tw["k_norm"] = tpc.replicate(state_dict["k_norm.weight"].to(torch.float32) + 1.0, mesh, None)
    return tw


class TPAttention:
    """Standalone TP full-attention with internal per-head KV caches (decode)."""

    def __init__(self, mesh, args, tw, tt_ccl):
        self.mesh = mesh
        self.args = args
        self.tw = tw
        self.tt_ccl = tt_ccl
        self.B = args.max_batch_size
        self._kv_shard_cfg_cache = {}  # active-width B -> KV-update height shard cfg (bucketed decode)
        self.NH = args.n_local_heads
        self.NKV = args.n_local_kv_heads
        self.HD = args.head_dim
        self.scale = self.HD**-0.5
        self.rope_dim = args.rope_head_dim
        self.compute_cfg = tpc.COMPUTE_PROJ
        # QWEN36_SDPA_STREAMING=1: drop fp32 dest-accumulate for the CHUNKED PREFILL
        # SDPA only. sdpa_program_factory.cpp vetoes the streaming (v2) compute kernel
        # whenever fp32_dest_acc_en is set -- can_use_streaming_compute() is literally
        # `return !fp32_dest_acc_en` -- so our production prefill has always taken the
        # legacy kernel, which upstream measures ~24% slower on Blackhole.
        # The QKV/wo matmuls keep COMPUTE_HIFI2 untouched: this is SDPA-only on purpose.
        # It also makes the LoFi+fp32_dest_acc_en silent-corruption trap unreachable,
        # since that trap lives only on the legacy kernel.
        self._sdpa_streaming = _tt_public_getenv("QWEN36_SDPA_STREAMING", "0") == "1"
        self.sdpa_prefill_cfg = (
            ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=False,
                packer_l1_acc=True,
            )
            if self._sdpa_streaming
            else tpc.COMPUTE_HIFI2
        )
        if self._sdpa_streaming and not getattr(TPAttention, "_sdpa_streaming_banner", False):
            TPAttention._sdpa_streaming_banner = True
            logger.info(
                "QWEN36_SDPA_STREAMING engaged: chunked prefill SDPA uses "
                "fp32_dest_acc_en=False, so the streaming (v2) compute kernel is "
                "selected instead of the legacy one; QKV/wo matmuls unchanged"
            )
        # Paged KV cache dtypes, K and V INDEPENDENT (QWEN36_KV_DTYPE="<K>/<V>";
        # QWEN_SDPA_BF8=1 still means bf8/bf8). Must agree with what
        # Qwen36Model.allocate_kv_caches allocated -- both read the same resolver.
        # Q follows K down to bf8 but never to bf4: Q is one row per step, so its
        # precision is free, and dropping it would change the QK product for a reason
        # that buys no memory.
        self._kv_k_dtype, self._kv_v_dtype, self._kv_dtype_src = tpc.resolve_kv_dtypes(ttnn.bfloat16)
        self._sdpa_bf8 = self._kv_k_dtype != ttnn.bfloat16 or self._kv_v_dtype != ttnn.bfloat16
        # Self-identify the arm from this module too, and only when the KV flag set it
        # (the model's allocate_kv_caches banner is the primary witness; this is the
        # attention role's own line, once per process). A legacy QWEN_SDPA_BF8 arm keeps
        # its exact old log.
        if self._kv_dtype_src.startswith("QWEN36_KV_DTYPE") and not getattr(TPAttention, "_kv_dtype_banner", False):
            TPAttention._kv_dtype_banner = True
            logger.info(
                f"QWEN36_KV_DTYPE engaged (attention): K={self._kv_k_dtype} V={self._kv_v_dtype} "
                f"source={self._kv_dtype_src}; prefill casts K/V per cache before paged_fill_cache"
            )
        # QWEN36_SDPA_GQA_MCAST=1 (image vllm-tt:k2-mr-pk): chunked prefill SDPA takes the GQA K/V
        # multicast schedule (tt-metal #56119) -- one DRAM read per K/V chunk per kv head, multicast to
        # the 6 q-head cores sharing it. Fails closed on an image without it (tp_common).
        self._sdpa_gqa_mcast = tpc.sdpa_gqa_mcast_enabled()
        if self._sdpa_gqa_mcast:
            tpc.sdpa_gqa_mcast_check_image()
        # QWEN36_SDPA_GQA_MCAST_Q128=1 (image vllm-tt:k2-mr-pk2, 0026): same multicast, but at the
        # control's q/k chunk with the default schedule's per-core work (bit-exact placement).
        self._sdpa_gqa_mcast_q128 = tpc.sdpa_gqa_mcast_q128_enabled()
        if self._sdpa_gqa_mcast_q128:
            tpc.sdpa_gqa_mcast_q128_check_image()
        # QWEN36_SDPA_READER_DEPTH=1 (image vllm-tt:k2-mr-pk3, 0027): deeper chunked-prefill SDPA reader
        # (more K/V tiles in flight per read barrier, one more K/V CB slot). Data movement only; composes
        # with the GQA multicast levers (it deepens the injectors' reads there).
        self._sdpa_reader_depth_kw = {}
        if tpc.sdpa_reader_depth_enabled():
            tpc.sdpa_reader_depth_check_image()
            self._sdpa_reader_depth_kw = tpc.sdpa_reader_depth_kwargs()
        # QWEN36_SDPA_STREAM_PACK2=1 (0052, upstream #60138): streaming SDPA blocked-pack
        # threshold 4 -> 2 on BH. Output-transparent (same tiles, same indices); the C++
        # factory logs its own banner per process. Env-read, no program-config change.
        if tpc.sdpa_stream_pack2_enabled():
            tpc.sdpa_stream_pack2_check_image()
        # Must match load_attention_weights_tp gates
        self._dram_sharded = getattr(args, "attn_qg_weight_memcfg", None) is not None
        self._wo_sharded = getattr(args, "attn_wo_weight_memcfg", None) is not None
        self._fused_qkv = getattr(args, "attn_qkv_fused_weight_memcfg", None) is not None
        self._qg_deint = self._fused_qkv
        # Fuse prefill norm-allgather + fused-QKV in-proj (all_gather_minimal_matmul_async).
        # Norm's prefill post-AG disabled in layer.py; decode path unchanged.
        self._fuse_agmm = self._fused_qkv
        # Decode head split/merge via nlp_create/concat_heads_decode (the batched-decode idiom).
        self._use_nlp_decode_heads = True
        # QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded decode QKV / wo (None = 1D path).
        _mr = tpc.decode_mr_for(args, mesh)
        self._mr_qkv = _mr.get("attn_qkv")
        self._mr_out = _mr.get("out")
        # SA1 (docs/runbooks/104-decode-layout-sa.md), decode only, each off by default and each moving
        # bytes only (bit-exact by construction; gated by equiv regardless):
        #   QWEN36_DEC_KV_DIRECT=1      v straight from nlp_create_qkv_heads_decode's HEIGHT shard (1 user/core on
        #                               the first B cores == kv_update_shard_cfg) into paged_update_cache; k i2s'd to
        #                               the same config without ttnn.pad (the shard height is the padded 32 anyway)
        #   QWEN36_DEC_ROPE_NOCOPY=1    partial rope: cos/sin reshaped once per layer (shared by q and k), and the
        #                               pass-through slice written to DRAM directly instead of slice + copy
        #   QWEN36_DEC_ATTN_POST_SHARDED=gate   sigmoid gate applied flat after concat-heads (the prefill order),
        #                               dropping the gate reshape
        #   QWEN36_DEC_SA_FAULT=1       must-fail rung for KV_DIRECT: writes v rolled by one user
        self._sa_kv_direct = _tt_public_getenv("QWEN36_DEC_KV_DIRECT", "0") == "1"
        self._sa_rope_nocopy = _tt_public_getenv("QWEN36_DEC_ROPE_NOCOPY", "0") == "1"
        self._sa_post = _tt_public_getenv("QWEN36_DEC_ATTN_POST_SHARDED", "")
        if self._sa_post not in ("", "gate"):
            raise ValueError(f"QWEN36_DEC_ATTN_POST_SHARDED={self._sa_post!r}: only 'gate' is implemented")
        self._sa_fault = _tt_public_getenv("QWEN36_DEC_SA_FAULT", "0") == "1"
        self._sa_logged = set()
        self.k_caches = None
        self.v_caches = None
        # External paged KV cache (vLLM/contract path); internal caches kept for demo fallback
        self.paged_k = None
        self.paged_v = None
        self.use_paged = False

    def set_paged_kv_cache(self, k_cache, v_cache):
        """Attach an externally-allocated paged KV cache (one call after allocate_kv_caches)."""
        # Fail closed on a cache whose dtype is not the resolver's: the prefill fill path casts to
        # self._kv_{k,v}_dtype (paged_fill_cache does not cast), so a mismatch is a TT_FATAL after a cold
        # build at best. Covers the MTP head's own paged KV (QWEN36_MTP_BATCH) as well as the 16 layers.
        if (k_cache.dtype, v_cache.dtype) != (self._kv_k_dtype, self._kv_v_dtype):
            raise RuntimeError(
                f"paged KV dtypes K={k_cache.dtype} V={v_cache.dtype} differ from the resolver's "
                f"K={self._kv_k_dtype} V={self._kv_v_dtype} ({self._kv_dtype_src}); refusing to bind"
            )
        self.paged_k = k_cache
        self.paged_v = v_cache
        self.use_paged = True
        # QWEN36_KV_UPDATE_FOLD (kv_fold.py): self-test the one-program K+V update once per process (cache dtypes /
        # geometry) at every (T, Bw) it may serve, before any decode or trace; fails closed. Unset: no-op. The MTP
        # buckets' verify depths (K_b + 1, up to 16) join the base T 1..4 cells.
        if kv_fold.enabled():
            from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mbh

            kv_fold.ensure_selftest(k_cache, v_cache, self.B, self._kv_shard_cfg,
                                    served=_mbh.served_verify_cells(self.B))
        # QWEN36_ATTN_POST_FOLD (attn_post_fold.py): self-test the one-program sigmoid/multiply/concat-heads at every
        # decode width it may serve (plain, draft and verify widths T*Bw), before any decode or trace; fails closed.
        if attn_post_fold.enabled():
            from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mbh2

            attn_post_fold.ensure_selftest(self.mesh, self.NH, self.HD, self.B, served=_mbh2.served_verify_cells(self.B))

    def _qkv_raw_decode(self, x):
        """Decode: the fused [q|k|v|gate] projection output, unsliced (QWEN36_ATTN_PREP feeds it to attn_decode_prep)."""
        tw = self.tw
        if self._mr_qkv is not None:
            return tpc.decode_mr_matmul(
                x, tw["wqkv_fused_mr"], self._mr_qkv, self.compute_cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        if getattr(self.args, "proj_1d_decode", False):
            return tpc.matmul_1d_decode(
                x, tw["wqkv_fused"], self.args.attn_qkv_decode_1d_progcfg, self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)

    def _qkv(self, x):
        """Q+gate/K/V projections → (qg, kp, vp). Fused path: one matmul, then slice."""
        tw = self.tw
        if not self._fused_qkv:
            return (
                self._col_proj(x, tw["wqkv"], self.args.attn_qg_progcfg),
                self._col_proj(x, tw["wk"], self.args.attn_k_progcfg),
                self._col_proj(x, tw["wv"], self.args.attn_v_progcfg),
            )
        # Prefill: x is K-sharded (norm skipped its AG) -> fused all-gather + QKV matmul. Output stays
        # DRAM: L1 clashes with a downstream matmul's CBs (verified; full-attn has more L1 pressure here).
        if self._fuse_agmm and x.shape[-2] > tpc.TILE_SIZE:
            _cs = [self.NH * self.HD, 2 * self.NKV * self.HD, self.NH * self.HD]
            _chunk_ok = sum(_cs) == tw["wqkv_fused"].shape[-1] and all(c % tpc.TILE_SIZE == 0 for c in _cs)
            if tpc.agmm_chunks("attn") and not _chunk_ok:
                tpc._smallops_banner(("agmm_chunks_fallback", "attn"), f"QWEN36_AGMM_CHUNKS fallback: role=attn "
                                     f"chunk_sizes={_cs} vs N={tw['wqkv_fused'].shape[-1]}; slicing kept")
            elif tpc.agmm_chunks("attn"):
                # QWEN36_AGMM_CHUNKS=attn (c105): the AGMM writes [q | kv | gate] as three DRAM tensors (DRAM as the
                # single-tensor output: L1 clashes with a downstream matmul's CBs). Drops the qkv3 + gate slices here
                # and the q_flat + kv slices in _make_heads; nlp_create_qkv_heads reads q/kv from DRAM instead of L1.
                # Same matmul program, only the writer's destination differs: bit-exact. Sentinel for _make_heads:
                # qg is the (q_flat, kv) tuple, kp the gate, vp None.
                q_flat, kv, gate = tpc.all_gather_matmul_prefill(
                    x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology(), chunk_sizes=_cs,
                    role="attn_in",
                )
                tpc.agmm_chunks_banner("attn", _cs, "DRAM")
                return (q_flat, kv), gate, None
            qkv = tpc.all_gather_matmul_prefill(
                x, tw["wqkv_fused"], self.tt_ccl, self.compute_cfg, self.args.ccl_topology(), role="attn_in"
            )
        elif self._mr_qkv is not None and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode, QWEN36_DECODE_DRAM_SHARDED_MR: multi-reader DRAM-sharded matmul; DRAM output as below.
            qkv = tpc.decode_mr_matmul(
                x, tw["wqkv_fused_mr"], self._mr_qkv, self.compute_cfg, out_memory_config=ttnn.DRAM_MEMORY_CONFIG
            )
        elif getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: small-grid 1D matmul (interleaved weight). Output DRAM so _make_heads_decode's
            # to_memory_config(.,L1) stays a real copy before it deallocates the source.
            qkv = tpc.matmul_1d_decode(
                x,
                tw["wqkv_fused"],
                self.args.attn_qkv_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            qkv = self._col_proj(x, tw["wqkv_fused"], self.args.attn_qkv_fused_progcfg)
        # Fused weight is [q|k|v|gate] (prepare_attn_qkv_deint): the q|k|v block is contiguous, so
        # return it whole (no gate wedged between q and k → no re-concat in _make_heads*). Gate is
        # the trailing block. Sentinel: vp=None flags the fused/contiguous layout to _make_heads*.
        qkv3_dim = self.NH * self.HD + 2 * self.NKV * self.HD
        gate_dim = self.NH * self.HD
        sh = list(qkv.shape)
        # qkv3 short-lived (split by _make_heads then freed) -> L1 in PREFILL only; decode keeps DRAM
        # (L1 qkv3 breaks the decode trace). gate lives across SDPA (post-concat) -> always DRAM.
        _qkv3_mc = ttnn.L1_MEMORY_CONFIG if sh[2] > tpc.TILE_SIZE else ttnn.DRAM_MEMORY_CONFIG
        qkv3 = ttnn.slice(qkv, (0, 0, 0, 0), (sh[0], sh[1], sh[2], qkv3_dim), memory_config=_qkv3_mc)
        gate = ttnn.slice(qkv, (0, 0, 0, qkv3_dim), (sh[0], sh[1], sh[2], qkv3_dim + gate_dim))
        ttnn.deallocate(qkv)
        return qkv3, gate, None

    def _qk_norm_prefill(self, q, k):
        """Prefill q/k RMSNorm with the (1+w) q_norm/k_norm gains. QWEN36_QK_NORM_W=1 (c105, per call): gamma passed
        as weight= to ttnn.rms_norm ([1,1,HD] TILE, the format GDN's norm_w already uses this way), dropping the
        row-bcast multiply. Numerics move (gamma applied in-kernel before the bf16 pack): tier 2. Decode untouched."""
        tw = self.tw
        _L1 = ttnn.L1_MEMORY_CONFIG
        if tpc.qk_norm_w():
            q = ttnn.rms_norm(q, weight=tw["q_norm"], epsilon=1e-6, memory_config=_L1)
            k = ttnn.rms_norm(k, weight=tw["k_norm"], epsilon=1e-6, memory_config=_L1)
            return q, k
        q = ttnn.multiply(
            ttnn.rms_norm(q, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["q_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        k = ttnn.multiply(
            ttnn.rms_norm(k, epsilon=1e-6, memory_config=ttnn.L1_MEMORY_CONFIG),
            tw["k_norm"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        return q, k

    def _col_proj(self, x, weight, decode_progcfg):
        """Column-parallel projection; DRAM-sharded decode matmul when enabled."""
        if not self._dram_sharded:
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            decode_progcfg,
            self.args.act_shard_hidden,
            self.args.prefill_progcfg,
            self.args.dim,
        )

    def _wo_colpar_prefill(self, gated):
        """QWEN36_ATTN_OUTPROJ_COLPAR: prefill out-proj as all-gather + column-parallel matmul (the AGMM the
        in-projections use) instead of row-parallel matmul + reduce-scatter. The output is the same
        column-sharded [1,1,S,dim/tp] the reduce-scatter returns. Numerics: the bf16 gated activation is
        gathered instead of summing per-chip partials (tier 2, as the GDN colpar)."""
        if not getattr(TPAttention, "_wo_colpar_banner", False):
            TPAttention._wo_colpar_banner = True
            logger.info("QWEN36_ATTN_OUTPROJ_COLPAR engaged: attention prefill out-proj uses column-parallel "
                        "AG+matmul (all_gather_minimal_matmul_async)")
        s = list(gated.shape)
        rows = 1
        for d in s[:-1]:
            rows *= d
        x = gated if (len(s) == 4 and s[0] == 1 and s[1] == 1) else ttnn.reshape(gated, (1, 1, rows, s[-1]))
        out = tpc.all_gather_matmul_prefill(
            x,
            self.tw["wo_colpar"],
            self.tt_ccl,
            self.compute_cfg,
            self.args.ccl_topology(),
            persistent_output_buffer=tpc.agmm_gather_buffer(self.tt_ccl, x),
            role="attn_out",
        )
        ttnn.deallocate(gated)
        return out

    def _wo_proj(self, x, weight):
        """Row-parallel output projection: DRAM-sharded decode/prefill matmul (K=attn_out_dim_tp),
        matching the in-proj. Falls back to plain interleaved when no sharded memcfg."""
        if self._mr_out is not None and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode, QWEN36_DECODE_DRAM_SHARDED_MR: the decode-only sharded copy of wo -> DRAM for the RS.
            # QWEN36_DECODE_MR_OUT_DIRECT: straight into the all_reduce_async input layout (else DRAM, as before)
            return tpc.decode_mr_matmul(
                x, self.tw["wo_mr"], self._mr_out, self.compute_cfg,
                out_memory_config=tpc.mr_ar_target(self.tt_ccl, ttnn.DRAM_MEMORY_CONFIG),
            )
        if getattr(self.args, "proj_1d_decode", False) and x.shape[-2] <= tpc.TILE_SIZE:
            # Decode: tuned ~32-core 1D matmul (interleaved weight) -> DRAM for the reduce-scatter.
            return tpc.matmul_1d_decode(
                x,
                weight,
                self.args.attn_wo_decode_1d_progcfg,
                self.compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        if not self._wo_sharded:
            if x.shape[-2] > tpc.TILE_SIZE:
                # Prefill: FPU-tuned 2D config beats ttnn-auto's 1x1 stall; L1 output (gated stays DRAM)
                # feeds the separate RS. max_cols = device width (11 on BH): wide grid (~10-wide) + the
                # existing L1-out. See test_mlp_matmul_sweep_prefill.
                if tpc.prefill_fs8("attn_wo"):
                    return ttnn.linear(
                        x,
                        weight,
                        compute_kernel_config=tpc.prefill_fs8_compute(packer_l1_acc=True),
                        program_config=tpc.create_prefill_fs8_program_config(
                            x.shape[-2], weight.shape[-2], weight.shape[-1],
                            getattr(self.args, "decode_grid_w", 8), "attn_wo",
                        ),
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                pc = tpc.create_prefill_mlp_matmul_program_config(
                    x.shape[-2],
                    weight.shape[-2],
                    weight.shape[-1],
                    max_cols=getattr(self.args, "decode_grid_w", 8),
                    tuning=getattr(self.args, "prefill_tuning", None),
                )
                return ttnn.linear(
                    x,
                    weight,
                    compute_kernel_config=self.compute_cfg,
                    program_config=pc,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            return ttnn.linear(x, weight, compute_kernel_config=self.compute_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return tpc.sharded_decode_matmul(
            x,
            weight,
            self.compute_cfg,
            self.args.attn_wo_progcfg,
            self.args.act_shard_attn_out,
            self.args.prefill_progcfg,
            self.args.attn_out_dim_tp,
        )

    def _make_heads(self, qg, kp, vp, S):
        """Split qg into heads; returns (q, gate_flat, k, v) via fused nlp_create_qkv_heads.

        gate_flat stays flat [1,1,S,NH*HD] (col h*HD+d = head h, dim d), matching nlp_concat_heads'
        column order. Gate is applied AFTER concat_heads (see forward_prefill*), so no head-major
        reshape/transpose is needed; bit-identical to per-head gating, saves ~1 ms/attn-layer at S=2048.
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is the contiguous [q|k|v] block,
            # kp is the gate. Slice q and (already-contiguous) kv directly — no concat needed.
            gate_flat = kp
            if isinstance(qg, tuple):  # QWEN36_AGMM_CHUNKS=attn: the AGMM already wrote q and kv apart
                q_flat, kv = qg
            else:
                # q_flat, kv feed nlp_create_qkv_heads then free immediately -> L1 (short-lived, no clash).
                q_flat = ttnn.slice(qg, (0, 0, 0, 0), (1, 1, S, NH * HD), memory_config=ttnn.L1_MEMORY_CONFIG)
                kv = ttnn.slice(
                    qg, (0, 0, 0, NH * HD), (1, 1, S, NH * HD + 2 * NKV * HD), memory_config=ttnn.L1_MEMORY_CONFIG
                )
                ttnn.deallocate(qg)
            q, k, v = ttnn.experimental.nlp_create_qkv_heads(
                q_flat,
                kv,
                num_heads=NH,
                num_kv_heads=NKV,
                transpose_k_heads=False,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kv)
            return q, gate_flat, k, v
        # Interleaved qg: split [q;gate] per head; gate flattened to [1,1,S,NH*HD] (applied post-concat).
        qg = ttnn.reshape(qg, (1, S, NH, 2 * HD))
        q_part, gate_part = ttnn.chunk(qg, 2, dim=-1)
        ttnn.deallocate(qg)
        gate_flat = ttnn.reshape(gate_part, (1, 1, S, NH * HD))
        ttnn.deallocate(gate_part)
        q_flat = ttnn.reshape(q_part, (1, 1, S, NH * HD))
        ttnn.deallocate(q_part)
        kv = ttnn.concat([kp, vp], dim=-1)
        ttnn.deallocate(kp)
        ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            q_flat,
            kv,
            num_heads=NH,
            num_kv_heads=NKV,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(q_flat)
        ttnn.deallocate(kv)
        return q, gate_flat, k, v

    def _concat_heads(self, gated):
        """Prefill concat-heads via nlp_concat_heads (post-gate). L1 output: short-lived post-SDPA temp,
        no kernel-CB clash."""
        return ttnn.experimental.nlp_concat_heads(gated, memory_config=ttnn.L1_MEMORY_CONFIG)

    def _sa_banner(self, msg):
        # Once per layer instance per flag: one line per attention layer (16 per model).
        flag = msg.split()[0]
        if flag not in self._sa_logged:
            self._sa_logged.add(flag)
            logger.info(msg)

    def _rope_decode_nocopy(self, x, cos_p, sin_p, n_heads, B):
        """apply_partial_rope_decode (image rope_tp.py:308) with the same ops on the same operands, minus two
        conversions: cos/sin come pre-reshaped to [1,1,B,rope_dim] (once per layer), and the pass-through
        slice is written to DRAM by the slice itself instead of slice(L1) + to_memory_config(DRAM)."""
        rope_dim = self.rope_dim
        hd = x.shape[-1]
        x_rope = ttnn.slice(x, (0, 0, 0, 0), (1, B, n_heads, rope_dim))
        x_rope_t = ttnn.transpose(x_rope, 1, 2)
        ttnn.deallocate(x_rope)
        roped_t = ttnn.experimental.rotary_embedding_hf(
            x_rope_t, cos_p, sin_p, is_decode_mode=False, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(x_rope_t)
        roped = ttnn.to_memory_config(ttnn.transpose(roped_t, 1, 2), ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(roped_t)
        if rope_dim == hd:
            return roped
        x_pass = ttnn.slice(x, (0, 0, 0, rope_dim), (1, B, n_heads, hd), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.concat([roped, x_pass], dim=-1)
        ttnn.deallocate(roped)
        ttnn.deallocate(x_pass)
        return result

    def _make_heads_decode(self, qg, kp, vp, B, v_sharded=False, gate_flat_out=False):
        """Decode head-split via nlp_create_qkv_heads_decode (the batched-decode idiom).

        Returns (q, gate, k, v): q [1,B,NH,HD], gate [1,B,NH,HD], k/v [1,B,NKV,HD], all L1-interleaved.
        The kernel only shuffles a fused Q|K|V, so the gate half of qg is split off first and applied
        post-SDPA exactly like the reshape path. The fused tensor is kept in L1 to dodge the Blackhole
        interleaved-reader bug (tt-metal #16667: DRAM input zeros odd-indexed Q rows). The height-sharded
        output is returned to L1-interleaved so the existing rms_norm / partial-rope / SDPA-decode path
        is unchanged.
        """
        NH, NKV, HD = self.NH, self.NKV, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        if vp is None:
            # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is already the contiguous [q|k|v]
            # the decode head-split wants — feed it directly, no concat. kp is the gate. qkv must be
            # L1 (tt-metal #16667: DRAM input zeros odd Q rows); one to_memory_config replaces the
            # old 3-way concat (which had also served to land qkv in L1).
            qkv = ttnn.to_memory_config(qg, _L1)
            ttnn.deallocate(qg)
            gate_flat = kp
        else:
            # Interleaved qg: [q;gate] per head -> split then re-flatten to [1,1,B,NH*HD].
            qg_r = ttnn.reshape(qg, (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg)
            q_part = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
            gate_part = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, 2 * HD), memory_config=_L1)
            ttnn.deallocate(qg_r)
            q_flat = ttnn.reshape(q_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(q_part)
            gate_flat = ttnn.reshape(gate_part, (1, 1, B, NH * HD), memory_config=_L1)
            ttnn.deallocate(gate_part)
            qkv = ttnn.concat([q_flat, kp, vp], dim=-1, memory_config=_L1)
            ttnn.deallocate(q_flat)
            ttnn.deallocate(kp)
            ttnn.deallocate(vp)
        q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(
            qkv, num_heads=NH, num_kv_heads=NKV, memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
        )
        ttnn.deallocate(qkv)
        q = ttnn.sharded_to_interleaved(q, _L1)
        k = ttnn.sharded_to_interleaved(k, _L1)
        if not v_sharded:
            v = ttnn.sharded_to_interleaved(v, _L1)
        if gate_flat_out:
            return q, gate_flat, k, v  # [1,1,B,NH*HD], applied after concat-heads (QWEN36_DEC_ATTN_POST_SHARDED)
        gate = ttnn.reshape(gate_flat, (1, B, NH, HD), memory_config=_L1)
        ttnn.deallocate(gate_flat)
        return q, gate, k, v

    def _concat_heads_decode(self, gated, B, flat4=False):
        """Decode concat-heads via nlp_concat_heads_decode. gated [1,B,NH,HD] L1 -> [1,B,NH*HD] L1.

        The op wants a height-sharded input ([1,B,heads-padded-to-32,HD], one core per user), so the
        gated SDPA output is resharded across `B` cores first (a grid-width-aligned rectangle — a
        ragged core set is rejected by the height-sharded mem config). Output is width-sharded, then
        returned to L1-interleaved so the downstream o_proj matmul is unchanged.
        """
        from models.tt_transformers.tt.model_config import num_to_corerange

        NH, HD = self.NH, self.HD
        _L1 = ttnn.L1_MEMORY_CONFIG
        grid = self.mesh.compute_with_storage_grid_size()
        if B <= 32:
            gx = min(B, grid.x)
            if B >= gx and B % gx != 0:
                gx = max(x for x in range(gx, 0, -1) if B % x == 0 and B // x <= grid.y)
            core_grid = ttnn.CoreRangeSet({num_to_corerange(B, grid_x=gx, grid_y=grid.y)})
            shard_cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, HD),
                core_grid=core_grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            gated_sh = ttnn.to_memory_config(gated, shard_cfg)
            ttnn.deallocate(gated)
            out_sh = ttnn.experimental.nlp_concat_heads_decode(gated_sh, num_heads=NH)
            ttnn.deallocate(gated_sh)
            out = ttnn.sharded_to_interleaved(out_sh, _L1)  # [1, 1, 32, NH*HD] (batch padded to 32)
            ttnn.deallocate(out_sh)
            if out.shape[-2] != B:
                out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1)
        else:
            outs = []
            for b0 in range(0, B, 32):
                b1 = min(b0 + 32, B)
                bsz = b1 - b0
                part = ttnn.slice(gated, (0, b0, 0, 0), (1, b1, NH, HD), memory_config=_L1)
                gx = min(bsz, grid.x)
                if bsz >= gx and bsz % gx != 0:
                    gx = max(x for x in range(gx, 0, -1) if bsz % x == 0 and bsz // x <= grid.y)
                core_grid = ttnn.CoreRangeSet({num_to_corerange(bsz, grid_x=gx, grid_y=grid.y)})
                shard_cfg = ttnn.create_sharded_memory_config(
                    shape=(ttnn.TILE_SIZE, HD),
                    core_grid=core_grid,
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                part_sh = ttnn.to_memory_config(part, shard_cfg)
                ttnn.deallocate(part)
                part_out_sh = ttnn.experimental.nlp_concat_heads_decode(part_sh, num_heads=NH)
                ttnn.deallocate(part_sh)
                part_out = ttnn.sharded_to_interleaved(part_out_sh, _L1)
                ttnn.deallocate(part_out_sh)
                if part_out.shape[-2] != bsz:
                    part_out = ttnn.slice(part_out, (0, 0, 0, 0), (1, 1, bsz, NH * HD), memory_config=_L1)
                outs.append(part_out)
            ttnn.deallocate(gated)
            out = ttnn.concat(outs, dim=2, memory_config=_L1)
            for o in outs:
                ttnn.deallocate(o)
        if flat4:
            return out  # [1,1,B,NH*HD]
        return ttnn.reshape(out, (1, B, NH * HD), memory_config=_L1)

    def reset_state(self):
        def z():
            return ttnn.from_torch(
                torch.zeros(self.B, 1, self.args.max_seq_len, self.HD, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
            )

        self.k_caches = [z() for _ in range(self.NKV)]
        self.v_caches = [z() for _ in range(self.NKV)]

    def forward_prefill(self, x, cos_tt, sin_tt):
        """Causal prefill. x [1,1,S,dim]: K-sharded (dim/tp per device) when the fused in-proj
        AG-matmul path is active (``_fuse_agmm`` and S>TILE — the norm skips its post-AG); replicated
        otherwise. Output reduce-scattered on dim=3."""
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q, k = self._qk_norm_prefill(q, k)
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # Fill per-head KV cache for decode (stateful path only)
        if self.k_caches is not None:
            # Don't deallocate slices — for NKV==1 they alias k/v used by SDPA
            for h in range(NKV):
                ttnn.fill_cache(self.k_caches[h], ttnn.slice(k, (0, h, 0, 0), (1, h + 1, S, HD)), 0)
                ttnn.fill_cache(self.v_caches[h], ttnn.slice(v, (0, h, 0, 0), (1, h + 1, S, HD)), 0)

        q8, k8, v8 = q, k, v
        padded = max(32, ((S + 31) // 32) * 32)
        # SDPA flash chunk: 128 for S>=2048, 64 below. (256 wins in ISOLATION at S=3072/4096
        # -- test_sdpa_prefill_opt -- but in the full model its larger CBs clash with the resident
        # attn-input L1 buffer during a single-pass prefill of S>2048 (prefill_tp/generate_tp;
        # program.cpp "circular buffers ... clash with L1 buffers"). Production serving chunks
        # prefill at <=2048, so this path never sees S>2048 and 256 has no reachable win.)
        ch = min(128 if S >= 2048 else 64, padded)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=ch, k_chunk_size=ch
        )
        attn = ttnn.transformer.scaled_dot_product_attention(
            q8, k8, v8, is_causal=True, scale=self.scale, memory_config=ttnn.DRAM_MEMORY_CONFIG, program_config=sdpa_cfg
        )
        ttnn.deallocate(q8)
        ttnn.deallocate(k8)
        ttnn.deallocate(v8)

        # Concat heads first, then gate: concat col h*HD+d == gate_flat col h*HD+d, so this is
        # bit-identical to per-head gating but skips the gate reshape+transpose to head-major.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        if "wo_colpar" in tw and gated.shape[-2] > tpc.TILE_SIZE:
            return self._wo_colpar_prefill(gated)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tpc.prefill_all_reduce(partial, self.mesh, self.tt_ccl, self.args.ccl_topology())

    def _kv_shard_cfg(self, B):
        """Height shard for paged_update_cache (one user per core), sized to the ACTIVE width B.
        Returns the precomputed max-batch config unchanged when B==self.B (byte-identical prod path);
        builds a width-B config (B cores) for bucketed decode. Mirrors model_config.kv_update_shard_cfg."""
        if B == self.B:
            return self.args.kv_update_shard_cfg
        cfg = self._kv_shard_cfg_cache.get(B)
        if cfg is None:
            # today's rule (largest divisor <= 8) unless its rows overflow the grid (B = 11, 22: verify T = 11)
            _g = self.mesh.compute_with_storage_grid_size()
            cols, rows = kv_fold.kv_shard_grid(B, _g.x, _g.y)
            cfg = ttnn.create_sharded_memory_config(
                shape=(ttnn.TILE_SIZE, self.HD),
                core_grid=ttnn.CoreGrid(x=cols, y=rows),
                strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self._kv_shard_cfg_cache[B] = cfg
        return cfg

    def forward_decode(
        self, x, cur_pos_tt, cos_tt, sin_tt, page_table=None, oneshot_ar=False, group_io=None, kv_only=False
    ):
        # QWEN36_MTP_BATCH (docs/MTP-BATCH.md): group_io = [(pos_j, page_table_j), ...] splits the
        # B = G*Bw input rows (row u = j*Bw + b) into G groups of Bw users. The projections, norms,
        # RoPE, head split/merge, gate and wo still run once at width B; the paged KV update runs
        # group by group (rows of one user at consecutive positions share cache tiles, and a single
        # paged_update_cache over them would race), and SDPA runs per group with the SAME program
        # config as plain decode at width Bw. kv_only=True stops after the KV update and returns
        # None (the MTP drafter's KV-maintenance reseed). Both are unused by plain decode.
        if group_io is not None:
            return self._forward_decode_grouped(x, cos_tt, sin_tt, group_io, oneshot_ar, kv_only)
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        # Active decode width, taken from the input (x is [1,1,B,dim_frac]). Normally == self.B.
        # BUCKETED decode: a request feeds B<self.B users; every shape/reshape/rope/head-split and
        # the KV-update shard config below run at this width, and the paged SDPA reads only these B
        # users' pages via the width-B page_table. The B==self.B path is byte-identical to before.
        B = x.shape[-2]
        _L1 = ttnn.L1_MEMORY_CONFIG  # keep decode head-prep + attn output L1-resident
        use_paged = self.use_paged and page_table is not None
        if not use_paged and self.k_caches is None:
            self.reset_state()

        # QWEN36_ATTN_PREP=1 (c82, K5 grafted op): ttnn.transformer.attn_decode_prep takes the fused [q|k|v|gate]
        # projection output and returns q (rms_norm * (1+w), partial rotate-half rope), the head-split gate, and k/v
        # already padded to 32 heads in the KV update's height shard: one program instead of the ~29 between the QKV
        # matmul and paged_update_cache (T2 profile, 83 us/layer). Kernel numerics differ from the chain: tier 2.
        _prep = (
            _tt_public_getenv("QWEN36_ATTN_PREP", "0") == "1"
            and use_paged
            and self._fused_qkv
            and B <= tpc.TILE_SIZE
            and hasattr(ttnn.transformer, "attn_decode_prep")
        )
        if _prep:
            kv_direct, gate_after = False, False
            qkv_raw = self._qkv_raw_decode(x)
            # The op wants cos/sin [1,>=B,1,RD]; the on-device decode rope hands [1,1,B,RD] (the no-copy path's
            # reshape proves the volume). Two small reshapes per layer; hoist to once per step if PREP ships.
            _rd = self.rope_dim
            cos_p = cos_tt if list(cos_tt.shape) == [1, B, 1, _rd] else ttnn.reshape(cos_tt, (1, B, 1, _rd))
            sin_p = sin_tt if list(sin_tt.shape) == [1, B, 1, _rd] else ttnn.reshape(sin_tt, (1, B, 1, _rd))
            q, gate, k_sh, v_sh = ttnn.transformer.attn_decode_prep(
                qkv_raw, cos_p, sin_p, tw["q_norm"], tw["k_norm"], NH, NKV, HD, self.rope_dim,
                self._kv_shard_cfg(B), batch=B, memory_config=ttnn.DRAM_MEMORY_CONFIG,  # SDPA decode: interleaved Q must be DRAM
            )
            ttnn.deallocate(qkv_raw)
            self._sa_banner(
                f"QWEN36_ATTN_PREP engaged: attn_decode_prep B={B} NH={NH} NKV={NKV} HD={HD} RD={self.rope_dim} "
                f"-> q {list(q.shape)} gate {list(gate.shape)} k/v {list(k_sh.shape)}"
            )
        else:
            qg, kp, vp = self._qkv(x)

            kv_direct = self._sa_kv_direct and use_paged and self._use_nlp_decode_heads
            gate_after = self._sa_post == "gate" and self._use_nlp_decode_heads
            if self._use_nlp_decode_heads:
                q, gate, k, v = self._make_heads_decode(qg, kp, vp, B, v_sharded=kv_direct, gate_flat_out=gate_after)
            elif vp is None:
                # Fused [q|k|v|gate] weight (_qkv sentinel vp=None): qg is contiguous [q|k|v], kp is gate.
                # Slice q/k/v heads directly from qg; gate is the separate block.
                q = ttnn.reshape(
                    ttnn.slice(qg, (0, 0, 0, 0), (1, 1, B, NH * HD), memory_config=_L1), (1, B, NH, HD), memory_config=_L1
                )
                k = ttnn.reshape(
                    ttnn.slice(qg, (0, 0, 0, NH * HD), (1, 1, B, NH * HD + NKV * HD), memory_config=_L1),
                    (1, B, NKV, HD),
                    memory_config=_L1,
                )
                v = ttnn.reshape(
                    ttnn.slice(qg, (0, 0, 0, NH * HD + NKV * HD), (1, 1, B, NH * HD + 2 * NKV * HD), memory_config=_L1),
                    (1, B, NKV, HD),
                    memory_config=_L1,
                )
                ttnn.deallocate(qg)
                gate = ttnn.reshape(kp, (1, B, NH, HD), memory_config=_L1)
                ttnn.deallocate(kp)
            else:
                qg_r = ttnn.reshape(qg, (1, B, NH, HD * 2), memory_config=_L1)
                ttnn.deallocate(qg)
                q = ttnn.slice(qg_r, (0, 0, 0, 0), (1, B, NH, HD), memory_config=_L1)
                gate = ttnn.slice(qg_r, (0, 0, 0, HD), (1, B, NH, HD * 2), memory_config=_L1)
                ttnn.deallocate(qg_r)
                k = ttnn.reshape(kp, (1, B, NKV, HD), memory_config=_L1)
                ttnn.deallocate(kp)
                v = ttnn.reshape(vp, (1, B, NKV, HD), memory_config=_L1)
                ttnn.deallocate(vp)

            # QK norm — (1+w), matching prefill/HF (the prior "flat" no-+1 decode band-aided the reshape scramble).
            q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
            k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)

            if self._sa_rope_nocopy:
                self._sa_banner("QWEN36_DEC_ROPE_NOCOPY engaged: cos/sin reshaped once per layer, pass-through slice straight to DRAM")
                cos_p = ttnn.reshape(cos_tt, (1, 1, B, self.rope_dim))
                sin_p = ttnn.reshape(sin_tt, (1, 1, B, self.rope_dim))
                q = self._rope_decode_nocopy(q, cos_p, sin_p, NH, B)
                k = self._rope_decode_nocopy(k, cos_p, sin_p, NKV, B)
            else:
                q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
                k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)

        # SDPA-decode grid: use the real device grid (11x10=110 cores on P150x4), not a
        # hardcoded 64. cores_per_head = grid_total/B (sdpa_decode_program_factory.cpp), so a
        # bigger grid gives each batch row more parallel cores for its KV-reduction. At SHORT
        # context (~4k) the reduction is shallow enough that fixed per-core overhead dominates
        # and this makes ~no difference (B=1: flat; B=8: ~3% worse, both within noise). At LONG
        # context (~64k) the reduction is deep enough that the extra cores are a real win:
        # SdpaDecodeDeviceOperation duration B=8: 1569.9us -> 1396.2us (-11%); B=1: 220.8us ->
        # 215.5us (-2.4%, no regression). Using the full grid unconditionally since it never hurts
        # and helps significantly at long context, where batched decode is otherwise slowest.
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_dec_cfg, _sdpa_dec_kw = tpc.sdpa_decode_knobs(_sdpa_grid)  # unset = full grid, k_chunk auto, op-default compute (shipped)
        if use_paged:
            # External paged KV: update at cur_pos, then paged SDPA-decode
            keys, values = self.paged_k, self.paged_v
            _kv_cfg = self._kv_shard_cfg(B)
            if _prep:
                pass  # attn_decode_prep already emitted k_sh / v_sh in _kv_cfg
            elif kv_direct:
                # v is already HEIGHT [32(padded NKV), HD] per user on the first B cores; reshard only if the
                # producer's grid differs from the KV-update config (it does not at B<=8: both are row-major
                # from (0,0)).
                if v.memory_config().shard_spec.grid != _kv_cfg.shard_spec.grid:
                    v_re = ttnn.to_memory_config(v, _kv_cfg)
                    ttnn.deallocate(v)
                    v = v_re
                    self._sa_banner(f"QWEN36_DEC_KV_DIRECT engaged: B={B}: v resharded (producer grid != kv cfg), k unpadded")
                else:
                    self._sa_banner(f"QWEN36_DEC_KV_DIRECT engaged: B={B}: v from the head-split shard, k unpadded")
                if self._sa_fault:
                    # Must-fail rung: roll v's users by one through an interleaved copy (injected once per layer).
                    v_il = ttnn.sharded_to_interleaved(v, _L1)
                    ttnn.deallocate(v)
                    v_il2 = ttnn.concat([ttnn.slice(v_il, (0, 1, 0, 0), (1, B, NKV, HD)),
                                         ttnn.slice(v_il, (0, 0, 0, 0), (1, 1, NKV, HD))], dim=1) if B > 1 else v_il
                    v_p = ttnn.pad(v_il2, [1, B, 32, HD], [0, 0, 0, 0], 0.5, memory_config=_L1)
                    v = ttnn.to_memory_config(v_p, _kv_cfg)
                    ttnn.deallocate(v_p)
                    self._sa_banner("QWEN36_DEC_SA_FAULT engaged: v rolled by one user (must diverge)")
                k_sh = ttnn.to_memory_config(k, _kv_cfg)
                v_sh = v
                ttnn.deallocate(k)
            else:
                k_p = ttnn.pad(k, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                v_p = ttnn.pad(v, [1, B, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                ttnn.deallocate(k)
                ttnn.deallocate(v)
                k_sh = ttnn.to_memory_config(k_p, _kv_cfg)
                v_sh = ttnn.to_memory_config(v_p, _kv_cfg)
                ttnn.deallocate(k_p)
                ttnn.deallocate(v_p)
            # paged_update_cache takes bf16/fp32 and casts to bf8 cache; decode K/V stay bf16 (prefill fill needs bf8)
            if _prep and page_table is not None and kv_fold.usable(1, B, keys, values):
                # QWEN36_KV_UPDATE_FOLD at T=1: K and V in one program (kv_fold.py), same bytes as the two calls below
                kv_fold.update(keys, values, k_sh, v_sh, cur_pos_tt, page_table, 1, B)
            else:
                ttnn.experimental.paged_update_cache(keys, k_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
                ttnn.experimental.paged_update_cache(values, v_sh, update_idxs_tensor=cur_pos_tt, page_table=page_table)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
            attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=sdpa_dec_cfg,
                **_sdpa_dec_kw,
                # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                memory_config=_L1,
            )
            ttnn.deallocate(q)
        else:
            # Internal per-head KV caches; pad NKV head dim to 32 for tile-aligned update
            for h in range(NKV):
                k_h = ttnn.slice(k, (0, 0, h, 0), (1, B, h + 1, HD))
                v_h = ttnn.slice(v, (0, 0, h, 0), (1, B, h + 1, HD))
                k_hp = ttnn.pad(k_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                v_hp = ttnn.pad(v_h, [1, B, 32, HD], [0, 0, 0, 0], 0.0)
                ttnn.deallocate(k_h)
                ttnn.deallocate(v_h)
                _kv_cfg = self._kv_shard_cfg(B)
                k_sh = ttnn.to_memory_config(k_hp, _kv_cfg)
                v_sh = ttnn.to_memory_config(v_hp, _kv_cfg)
                ttnn.deallocate(k_hp)
                ttnn.deallocate(v_hp)
                ttnn.experimental.paged_update_cache(self.k_caches[h], k_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.experimental.paged_update_cache(self.v_caches[h], v_sh, update_idxs_tensor=cur_pos_tt)
                ttnn.deallocate(k_sh)
                ttnn.deallocate(v_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)

            if NKV == 1:
                k_full, v_full = self.k_caches[0], self.v_caches[0]
            else:
                k_full = ttnn.concat(self.k_caches, dim=1)
                v_full = ttnn.concat(self.v_caches, dim=1)

            # Non-paged oracle path (test/generate_tp only): the full-cache SDPA-decode's static CBs
            # grow with max_seq_len and, unbounded (k_chunk_size=0), overrun into the persistent CCL
            # semaphore buffers at the top of L1. Bound the K-chunk to cap the CB footprint (the paged
            # production path reads bounded blocks, so it keeps the auto config).
            nonpaged_sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8), exp_approx_mode=False, q_chunk_size=0, k_chunk_size=128
            )
            attn_out = ttnn.transformer.scaled_dot_product_attention_decode(
                q,
                k_full,
                v_full,
                cur_pos_tensor=cur_pos_tt,
                scale=self.scale,
                program_config=nonpaged_sdpa_cfg,
                # Emit to L1: consumed by the L1 sigmoid-gate multiply next (output-only, doesn't
                # change the SDPA reduction), before the wo matmul + all-reduce re-materialize to DRAM.
                memory_config=_L1,
            )
            ttnn.deallocate(q)

        if gate_after:
            # Same operand pairs as the per-head multiply, in the prefill order (gate after concat).
            self._sa_banner("QWEN36_DEC_ATTN_POST_SHARDED engaged: gate applied flat after concat-heads")
            attn_flat = self._concat_heads_decode(attn_out, B, flat4=True)  # [1,1,B,NH*HD]
            gated_flat = ttnn.multiply(attn_flat, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_flat)
            ttnn.deallocate(gate)
            gated_flat = ttnn.reshape(gated_flat, (1, B, NH * HD), memory_config=_L1)
        elif self._use_nlp_decode_heads and attn_post_fold.usable(attn_out, gate, B, NH, HD):
            # QWEN36_ATTN_POST_FOLD (attn_post_fold.py): the plain decode path's same six ops in ONE program
            flat4 = attn_post_fold.post(attn_out, gate, B, NH, HD)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = ttnn.reshape(flat4, (1, B, NH * HD), memory_config=_L1)
        elif self._use_nlp_decode_heads:
            gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = self._concat_heads_decode(gated, B)  # consumes + deallocates gated
        else:
            gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = ttnn.reshape(gated, (1, B, NH * HD))
            ttnn.deallocate(gated)
        wo_partial = self._wo_proj(gated_flat, tw["wo"])
        ttnn.deallocate(gated_flat)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
        if oneshot_ar:  # QWEN36_ONESHOT_AR: replicated residual -> one all-gather + add
            return tpc.all_reduce_oneshot_decode(wo_partial, self.tt_ccl, self.args.ccl_topology())
        return tt_all_reduce(
            wo_partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(wo_partial),
        )

    def _forward_decode_grouped(self, x, cos_tt, sin_tt, group_io, oneshot_ar, kv_only):
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        B = x.shape[-2]
        G = len(group_io)
        assert B % G == 0, f"grouped decode: {B} rows not divisible into {G} groups"
        Bw = B // G
        _L1 = ttnn.L1_MEMORY_CONFIG
        assert self.use_paged and self._use_nlp_decode_heads, "grouped decode needs the paged nlp-heads path"
        # QWEN36_ATTN_PREP ported to the grouped (MTP verify / reseed) pass, same condition as forward_decode's
        # _prep: one attn_decode_prep at width B = G*Bw replaces the head split, q/k norms and rope; its k/v come
        # back padded to 32 heads in the width-B KV-update height shard, so each group's rows are taken from an
        # interleaved copy and resharded at width Bw (the unported path pads then reshards per group). Unset:
        # unchanged.
        _prep = (
            _tt_public_getenv("QWEN36_ATTN_PREP", "0") == "1"
            and self._fused_qkv
            and B <= tpc.TILE_SIZE
            and hasattr(ttnn.transformer, "attn_decode_prep")
        )
        keys, values = self.paged_k, self.paged_v
        _kv_cfg = self._kv_shard_cfg(Bw)
        _kv_fold = False
        if _prep:
            qkv_raw = self._qkv_raw_decode(x)
            _rd = self.rope_dim
            cos_p = cos_tt if list(cos_tt.shape) == [1, B, 1, _rd] else ttnn.reshape(cos_tt, (1, B, 1, _rd))
            sin_p = sin_tt if list(sin_tt.shape) == [1, B, 1, _rd] else ttnn.reshape(sin_tt, (1, B, 1, _rd))
            q, gate, k_sh, v_sh = ttnn.transformer.attn_decode_prep(
                qkv_raw, cos_p, sin_p, tw["q_norm"], tw["k_norm"], NH, NKV, HD, self.rope_dim,
                self._kv_shard_cfg(B), batch=B, memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(qkv_raw)
            if "QWEN36_ATTN_PREP:grouped" not in self._sa_logged:
                self._sa_logged.add("QWEN36_ATTN_PREP:grouped")
                logger.info(
                    f"QWEN36_ATTN_PREP engaged in MTP verify: attn_decode_prep B={B} (G={G} x Bw={Bw}) NH={NH} "
                    f"NKV={NKV} HD={HD} RD={self.rope_dim} -> q {list(q.shape)} gate {list(gate.shape)} "
                    f"k/v {list(k_sh.shape)}, per-group reshard at width {Bw}"
                )
            kv_rows = int(k_sh.shape[2])  # heads padded to the tile height by the op
            _fold_pos = getattr(group_io, "fold_pos", None)
            _kv_fold = (
                spec_fold_enabled()
                and _fold_pos is not None
                and len({id(pt_j) for _, pt_j in group_io}) == 1
                and kv_fold.usable(G, Bw, keys, values)
            )
        if _prep and _kv_fold:
            # QWEN36_KV_UPDATE_FOLD: K and V of all G*Bw rows in ONE program on the width-B shards attn_decode_prep
            # emitted, index fold_pos (row j*Bw + b), the shared page table; rows of one user chained in candidate
            # order on the device (kv_fold.py). Replaces 2 * (1 + 3G) programs per layer.
            kv_fold.update(keys, values, k_sh, v_sh, _fold_pos, group_io[0][1], G, Bw)
            ttnn.deallocate(k_sh)
            ttnn.deallocate(v_sh)
        elif _prep:
            for sh, cache in ((k_sh, keys), (v_sh, values)):
                il = ttnn.sharded_to_interleaved(sh, _L1)
                ttnn.deallocate(sh)
                for j, (pos_j, pt_j) in enumerate(group_io):
                    lo, hi = j * Bw, (j + 1) * Bw
                    part = il if G == 1 else ttnn.slice(il, (0, lo, 0, 0), (1, hi, kv_rows, HD), memory_config=_L1)
                    part_sh = ttnn.to_memory_config(part, _kv_cfg)
                    if part is not il:
                        ttnn.deallocate(part)
                    ttnn.experimental.paged_update_cache(cache, part_sh, update_idxs_tensor=pos_j, page_table=pt_j)
                    ttnn.deallocate(part_sh)
                ttnn.deallocate(il)
        else:
            qg, kp, vp = self._qkv(x)
            q, gate, k, v = self._make_heads_decode(qg, kp, vp, B)
            q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6, memory_config=_L1), tw["q_norm"], memory_config=_L1)
            k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6, memory_config=_L1), tw["k_norm"], memory_config=_L1)
            q = apply_partial_rope_decode(q, cos_tt, sin_tt, NH, B, self.rope_dim)
            k = apply_partial_rope_decode(k, cos_tt, sin_tt, NKV, B, self.rope_dim)
            for j, (pos_j, pt_j) in enumerate(group_io):
                lo, hi = j * Bw, (j + 1) * Bw
                for src, cache in ((k, keys), (v, values)):
                    part = ttnn.slice(src, (0, lo, 0, 0), (1, hi, NKV, HD), memory_config=_L1)
                    part_p = ttnn.pad(part, [1, Bw, 32, HD], [0, 0, 0, 0], 0.0, memory_config=_L1)
                    ttnn.deallocate(part)
                    part_sh = ttnn.to_memory_config(part_p, _kv_cfg)
                    ttnn.deallocate(part_p)
                    ttnn.experimental.paged_update_cache(cache, part_sh, update_idxs_tensor=pos_j, page_table=pt_j)
                    ttnn.deallocate(part_sh)
            ttnn.deallocate(k)
            ttnn.deallocate(v)
        if kv_only:
            ttnn.deallocate(q)
            ttnn.deallocate(gate)
            return None
        _sdpa_grid = self.mesh.compute_with_storage_grid_size()
        sdpa_dec_cfg, _sdpa_dec_kw = tpc.sdpa_decode_knobs(_sdpa_grid)
        _pts = {id(pt_j) for _, pt_j in group_io}
        if spec_fold_enabled() and G >= 2 and len(_pts) == 1:
            # QWEN36_SDPA_SPEC_FOLD: q rows are already candidate-major (row j*Bw + b), the layout the folded op
            # takes; its cur_pos is the host-built concatenation of the pos_j (FoldGroupIO.fold_pos) or, without
            # one, a device concat. Output [1, G*Bw, NH, HD] = the concat of the per-group outputs.
            fold_pos = getattr(group_io, "fold_pos", None)
            own_pos = fold_pos is None
            if own_pos:
                fold_pos = ttnn.concat([p for p, _ in group_io], dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # A full-range ttnn.slice returns its INPUT (same buffer), so q must outlive the SDPA call: c130's four
            # EngineCore SIGSEGV boots were deallocate(q) here freeing q_f, then SDPA reading a null Q buffer
            # (reproduced on ttsim: bench/probes/sdpa_spec_fold/ttsim_alias_repro.py). Slice only when q is not
            # already the interleaved-DRAM [1, B, NH, HD] the op takes; free both after the call.
            q_ready = q.memory_config() == ttnn.DRAM_MEMORY_CONFIG and list(q.shape) == [1, B, NH, HD]
            chunks = spec_fold_chunks(G)
            if len(chunks) == 1:
                q_f = q if q_ready else ttnn.slice(q, (0, 0, 0, 0), (1, B, NH, HD), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                attn_out = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_f,
                    keys,
                    values,
                    page_table_tensor=group_io[0][1],
                    cur_pos_tensor=fold_pos,
                    scale=self.scale,
                    program_config=sdpa_dec_cfg,
                    memory_config=_L1,
                    **_sdpa_dec_kw,
                )
                ttnn.deallocate(q_f)
                if q_f is not q:
                    ttnn.deallocate(q)  # a no-op if the slice aliased q after all
            else:
                # T > SPEC_FOLD_TCHUNK (a drafter's K 8-15): one folded call per candidate chunk [j0, j1), its q rows
                # j0*Bw.. (candidate-major already) and its cur_pos the device concat of that chunk's pos_j.
                outs = []
                for j0, j1 in chunks:
                    q_c = ttnn.slice(q, (0, j0 * Bw, 0, 0), (1, j1 * Bw, NH, HD), memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    pos_c = ttnn.concat([group_io[j][0] for j in range(j0, j1)], dim=-1,
                                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    outs.append(ttnn.transformer.paged_scaled_dot_product_attention_decode(
                        q_c,
                        keys,
                        values,
                        page_table_tensor=group_io[0][1],
                        cur_pos_tensor=pos_c,
                        scale=self.scale,
                        program_config=sdpa_dec_cfg,
                        memory_config=_L1,
                        **_sdpa_dec_kw,
                    ))
                    ttnn.deallocate(q_c)
                    ttnn.deallocate(pos_c)
                ttnn.deallocate(q)
                attn_out = ttnn.concat(outs, dim=1, memory_config=_L1)
                for o in outs:
                    ttnn.deallocate(o)
            if own_pos:
                ttnn.deallocate(fold_pos)
            if "QWEN36_SDPA_SPEC_FOLD" not in self._sa_logged:
                self._sa_logged.add("QWEN36_SDPA_SPEC_FOLD")
                logger.info(
                    f"QWEN36_SDPA_SPEC_FOLD engaged mode={_SPEC_FOLD}: MTP verify SDPA folded, one call over "
                    f"T={G} candidates x Bw={Bw} users (was {G} calls), cur_pos "
                    f"{'host-built' if not own_pos else 'device concat'}"
                )
            if len(chunks) > 1 and f"QWEN36_SDPA_SPEC_FOLD:chunk{G}x{Bw}" not in self._sa_logged:
                self._sa_logged.add(f"QWEN36_SDPA_SPEC_FOLD:chunk{G}x{Bw}")
                logger.info(
                    f"QWEN36_SDPA_SPEC_FOLD chunked: T={G} x Bw={Bw} as {len(chunks)} folded calls {chunks} "
                    f"(T > {SPEC_FOLD_TCHUNK}: the 0034 L1 budget)"
                )
            return self._decode_grouped_post(attn_out, gate, B, oneshot_ar)
        outs = []
        for j, (pos_j, pt_j) in enumerate(group_io):
            lo, hi = j * Bw, (j + 1) * Bw
            # SDPA-decode rejects an interleaved L1 q (TT_FATAL: must be DRAM when not sharded; caught by
            # bench/probes/sim_mtp_batch_mechanics.py), so each group's q slice lands in DRAM.
            q_j = ttnn.slice(q, (0, lo, 0, 0), (1, hi, NH, HD), memory_config=ttnn.DRAM_MEMORY_CONFIG)
            # Decode Q stays bf16 whatever the cache dtypes (K8/V4 included), as in plain decode: the Q->bf8
            # cast is prefill-only, and a bf8 Q into the GQA decode SDPA is the c119 probe failure.
            if q_j.dtype != ttnn.bfloat16:
                raise RuntimeError(f"grouped decode SDPA: q is {q_j.dtype}, expected bf16 (cache K={keys.dtype} V={values.dtype})")
            outs.append(
                ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    q_j,
                    keys,
                    values,
                    page_table_tensor=pt_j,
                    cur_pos_tensor=pos_j,
                    scale=self.scale,
                    program_config=sdpa_dec_cfg,
                    memory_config=_L1,
                    **_sdpa_dec_kw,
                )
            )
            ttnn.deallocate(q_j)
        ttnn.deallocate(q)
        attn_out = ttnn.concat(outs, dim=1, memory_config=_L1) if G > 1 else outs[0]
        if G > 1:
            for o in outs:
                ttnn.deallocate(o)
        return self._decode_grouped_post(attn_out, gate, B, oneshot_ar)

    def _decode_grouped_post(self, attn_out, gate, B, oneshot_ar):
        tw = self.tw
        _L1 = ttnn.L1_MEMORY_CONFIG
        if attn_post_fold.usable(attn_out, gate, B, self.NH, self.HD):
            # QWEN36_ATTN_POST_FOLD (attn_post_fold.py): sigmoid, multiply, reshard, concat-heads, S2I, slice in ONE
            # program; same bytes (boot self-test). [1,1,B,NH*HD] L1, then the same reshape the served path ends with.
            flat4 = attn_post_fold.post(attn_out, gate, B, self.NH, self.HD)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = ttnn.reshape(flat4, (1, B, self.NH * self.HD), memory_config=_L1)
        else:
            gated = ttnn.multiply(attn_out, ttnn.sigmoid(gate, memory_config=_L1), memory_config=_L1)
            ttnn.deallocate(attn_out)
            ttnn.deallocate(gate)
            gated_flat = self._concat_heads_decode(gated, B)
        wo_partial = self._wo_proj(gated_flat, tw["wo"])
        ttnn.deallocate(gated_flat)
        wo_partial = ttnn.reshape(wo_partial, (1, 1, B, wo_partial.shape[-1]))
        if oneshot_ar:
            return tpc.all_reduce_oneshot_decode(wo_partial, self.tt_ccl, self.args.ccl_topology())
        return tt_all_reduce(
            wo_partial,
            self.mesh,
            self.tt_ccl,
            cluster_axis=0,
            dim=3,
            topology=self.args.ccl_topology(),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **tpc.decode_rs_kwargs(wo_partial),
        )

    def forward_prefill_paged(
        self,
        x,
        cos_tt,
        sin_tt,
        page_table,
        chunk_page_table=None,
        chunk_start_idx=0,
        chunk_start_idx_tensor=None,
        user_id=0,
        kv_only=False,
    ):
        """Paged-KV prefill for one chunk: fill cache + chunked SDPA over prior chunks.

        x is K-sharded when the fused in-proj path is active (same contract as ``forward_prefill``).
        chunk_start_idx_tensor: optional device offset for FLEXIBLE chunked SDPA (one program
        per trace/bucket). chunk_start_idx (int) still sizes the page table host-side.
        kv_only (QWEN36_MTP_WARM_TAIL, the MTP head's prompt-tail warm pass): return None right after the
        K/V fill -- no SDPA, gate or wo. The same ops up to the fill as the full call, so the cache contents
        are the full call's.
        """
        assert self.use_paged and self.paged_k is not None, "forward_prefill_paged requires a bound paged KV cache"
        tw, NH, NKV, HD = self.tw, self.NH, self.NKV, self.HD
        if chunk_start_idx is None:
            chunk_start_idx = 0
        S = x.shape[-2]

        qg, kp, vp = self._qkv(x)

        q, gate_flat, k, v = self._make_heads(qg, kp, vp, S)

        q, k = self._qk_norm_prefill(q, k)
        q = apply_partial_rope_prefill(q, cos_tt, sin_tt, NH, self.rope_dim)
        k = apply_partial_rope_prefill(k, cos_tt, sin_tt, NKV, self.rope_dim)

        # paged_fill_cache asserts input.dtype() == cache.dtype() and does NOT cast,
        # so K and V are each cast to THEIR OWN cache dtype (they may differ).
        if k.dtype != self._kv_k_dtype:
            _kc = ttnn.typecast(k, self._kv_k_dtype)
            ttnn.deallocate(k)
            k = _kc
        if v.dtype != self._kv_v_dtype:
            _vc = ttnn.typecast(v, self._kv_v_dtype)
            ttnn.deallocate(v)
            v = _vc

        # Fill this chunk into the paged cache
        k_paged, v_paged = self.paged_k, self.paged_v
        block_size = k_paged.shape[2]
        fill_page_table = chunk_page_table if chunk_page_table is not None else page_table
        page_len = fill_page_table.shape[1] * block_size
        if page_len < S:
            k_fill = ttnn.slice(k, (0, 0, 0, 0), (1, NKV, page_len, HD))
            v_fill = ttnn.slice(v, (0, 0, 0, 0), (1, NKV, page_len, HD))
        else:
            k_fill, v_fill = k, v
        ttnn.experimental.paged_fill_cache(k_paged, k_fill, fill_page_table, batch_idx=user_id)
        ttnn.experimental.paged_fill_cache(v_paged, v_fill, fill_page_table, batch_idx=user_id)
        if page_len < S:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        if kv_only:
            ttnn.deallocate(q)
            ttnn.deallocate(gate_flat)
            return None

        # Chunked SDPA over paged cache; keep Q bf16 unless the K cache is quantized, in which
        # case Q goes to bf8 (never bf4) -> block-float QK matmul. SDPA validates each of
        # Q/K/V independently against {bf16, bf8_b, bf4_b} and its program factory derives
        # q_df/k_df/v_df separately, so a mixed K8/V4 pair is accepted (CODE-FACT, tt-metal
        # 60053f1: sdpa/device/sdpa_device_operation.cpp:35-44,
        # sdpa/device/sdpa_program_factory.cpp:770-772,830-832,
        # sdpa_decode/device/sdpa_decode_program_factory.cpp:430-432).
        if self._kv_k_dtype != ttnn.bfloat16:
            q8 = ttnn.typecast(q, dtype=ttnn.bfloat8_b)
            ttnn.deallocate(q)
        else:
            q8 = q

        # chunk_start_idx % q_chunk_size == 0; FLEXIBLE path uses one program per trace.
        # q/k_chunk=128 is valid (chunk_start always divisible by 2048) and faster than 64/256.
        if chunk_start_idx_tensor is not None:
            qk_chunk = 128
        else:
            cap = 128 if S >= 2048 else 64  # 128 beats 256
            qk_chunk = cap if not chunk_start_idx else min(cap, chunk_start_idx & -chunk_start_idx)
        # Full BH grid for SDPA perf (bit-identical to 8×8; see test_tp_chunked_prefill_pcc_sweep)
        _grid = self.mesh.compute_with_storage_grid_size()
        _gqa_kw = {}
        if self._sdpa_gqa_mcast:
            # One core per (head, q_chunk) unit: q=k=256 for 12 heads x 2048 tokens (96 of 110 cores).
            # The traced path's chunk_start is a device tensor but always a multiple of 2048.
            _cs = 0 if chunk_start_idx_tensor is not None else (chunk_start_idx or 0)
            _gq = tpc.sdpa_gqa_mcast_chunk(S, _cs, NH, _grid.x * _grid.y, n_kv_heads=NKV)
            if _gq is not None:
                qk_chunk = _gq
                _gqa_kw = {"gqa_kv_mcast": True}
            else:
                logger.info(f"QWEN36_SDPA_GQA_MCAST: S={S} chunk_start={_cs} not eligible, default schedule")
        elif self._sdpa_gqa_mcast_q128:
            # qk_chunk stays the control's; only the core placement / K/V source changes.
            _q128 = tpc.sdpa_gqa_mcast_q128_kwargs(S, qk_chunk, NH, _grid.x * _grid.y, n_kv_heads=NKV)
            if _q128 is not None:
                _gqa_kw = _q128
            elif (S, qk_chunk) not in tpc._SDPA_GQA_Q128_SEEN:
                tpc._SDPA_GQA_Q128_SEEN.add((S, qk_chunk))
                logger.info(f"QWEN36_SDPA_GQA_MCAST_Q128: S={S} q=k={qk_chunk} not eligible, default schedule")
        _sdpa_cc, _sdpa_exp = tpc.sdpa_prefill_knobs(self.sdpa_prefill_cfg, self._sdpa_streaming)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=_grid,
            exp_approx_mode=_sdpa_exp,
            q_chunk_size=qk_chunk,
            k_chunk_size=qk_chunk,
            **_gqa_kw,
            **self._sdpa_reader_depth_kw,
        )

        # Pad page table to cover Q+offset and satisfy stick-size % 32 (extra blocks masked by causality)
        sdpa_page_table = page_table
        needed_blocks = (S + chunk_start_idx + block_size - 1) // block_size
        target_blocks = max(needed_blocks, page_table.shape[-1])
        target_blocks = ((target_blocks + 31) // 32) * 32
        if page_table.shape[-1] < target_blocks:
            zeros_pad = ttnn.zeros(
                (page_table.shape[0], target_blocks - page_table.shape[-1]),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sdpa_page_table = ttnn.concat([page_table, zeros_pad], dim=-1)
            ttnn.deallocate(zeros_pad)

        if chunk_start_idx_tensor is not None:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx_tensor=chunk_start_idx_tensor,
                compute_kernel_config=_sdpa_cc,
                program_config=sdpa_cfg,
            )
        else:
            attn = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q8,
                input_tensor_k=k_paged,
                input_tensor_v=v_paged,
                page_table_tensor=sdpa_page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=_sdpa_cc,
                program_config=sdpa_cfg,
            )
        if sdpa_page_table is not page_table:
            ttnn.deallocate(sdpa_page_table)
        ttnn.deallocate(q8)

        # Concat heads first, then gate (flat gate matches concat column order); see forward_prefill.
        attn = self._concat_heads(attn)
        # concat(attn)+sigmoid(gate) in L1; gated stays DRAM (feeds the wo matmul_reduce_scatter — an L1
        # CCL activation risks clashing with its CBs).
        gated = ttnn.multiply(
            attn, ttnn.sigmoid(gate_flat, memory_config=ttnn.L1_MEMORY_CONFIG), memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.deallocate(attn)
        ttnn.deallocate(gate_flat)
        if "wo_colpar" in tw and gated.shape[-2] > tpc.TILE_SIZE:
            return self._wo_colpar_prefill(gated)
        partial = self._wo_proj(gated, tw["wo"])
        ttnn.deallocate(gated)
        return tpc.prefill_all_reduce(partial, self.mesh, self.tt_ccl, self.args.ccl_topology())

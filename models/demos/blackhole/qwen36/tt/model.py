# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-9B text model for Blackhole P150.

tok_embeddings -> 32 x Qwen36DecoderLayer -> RMSNorm -> LM Head.
Hybrid state: KV cache (8 attn layers) + recurrent state (24 DeltaNet layers).
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import math
import os
import time
from dataclasses import dataclass, fields

import torch
from loguru import logger
from tqdm import tqdm

import ttnn
from models.common.rmsnorm import RMSNorm
from models.demos.blackhole.qwen36.tt import argmax_fold
from models.demos.blackhole.qwen36.tt import dev_sample
from models.demos.blackhole.qwen36.tt import masked_bucket_trace as mbt
from models.demos.blackhole.qwen36.tt import prefill_host as phl
from models.demos.blackhole.qwen36.tt import prefix_snap
from models.demos.blackhole.qwen36.tt.layer import Qwen36DecoderLayer
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs
from models.demos.blackhole.qwen36.tt.rope import Qwen36RoPESetup
from models.tt_transformers.tt.common import Mode, get_block_size, num_blocks_in_seq


def _match_cache_dtype(tensor, cache):
    """Return ``tensor`` cast to ``cache``'s dtype, or ``tensor`` itself if it matches.

    ``paged_fill_cache`` requires input.dtype() == cache.dtype() and will not cast
    (the TT_FATAL that kept bf8 KV off TP=1). The caller owns the result: deallocate
    it only when it is not the input.
    """
    if tensor.dtype == cache.dtype:
        return tensor
    return ttnn.typecast(tensor, cache.dtype)


@dataclass
class _MaskedBucketBufs:
    """Persistent replicated device inputs of ONE traced masked bucket (addresses baked into the
    trace; rewritten on EVERY replay with copy_host_to_device_tensor). See masked_bucket_trace.py."""

    token: object  # [1, bucket] uint32 ROW_MAJOR
    cos: object  # [1, 1, bucket, rope_head_dim] bf16 TILE
    sin: object  # [1, 1, bucket, rope_head_dim] bf16 TILE
    fill_pt: object  # [1, bucket//block_size] int32 ROW_MAJOR (fixed-width KV-fill page table)
    mask_f32: object  # [1, bucket, 1] float32 TILE (GDN beta/g validity mask, all layers)
    mask_q: object  # [1, bucket, 1] bf16 TILE (GDN q/k/v validity mask, all layers)
    conv_sel: object  # [1, K-1, bucket+K-1] bf16 TILE (FIR decode-window one-hot, all layers)


@dataclass
class _MaskedBucketTrace:
    """A captured masked-bucket prefill trace and the buffers its body reads."""

    bucket: int
    trace_id: object
    bufs: _MaskedBucketBufs
    output: object  # hidden [1, 1, bucket, dim]; owned by the trace, never cloned
    defer: bool  # QWEN36_GDN_CONV_WINDOW_ONCE skip state baked into the capture

# Fixed masked-prefill bucket set. Module level (mirrored by Qwen36Model._PREFILL_MASK_BUCKETS)
# so bench/fixtures/fixtures-prefix-chunk-plan-test.py can drive the selection rules with no
# device and no ttnn import. The SMALLEST bucket is 128 and that is load bearing: the masked
# forward passes bucket as SDPA's S, and tp.py caps qk_chunk at min(128, S). A bucket of 32
# would select k_chunk_size=32, which SILENTLY CORRUPTS the non-causal masked SDPA path
# (validation accepts it; the result is garbage -- [source validation record]).
PREFILL_MASK_BUCKETS = (128, 256, 512, 1024, 2048)
PREFILL_CHUNK_DEFAULT = 2048


def prefill_chunk_size():
    """QWEN36_PREFILL_CHUNK (c134, branch agmm-grid): the served prefill chunk, read per call (a wb `set` re-capture
    picks it up). Unset/"" = 2048, the shipped chunk. It must be a multiple of 128 (GDN sub-chunk, SDPA q/k chunk) and
    so of the 64-token KV block. 2304 is the chunk the 11x9 AGMM grid needs (72 tile rows = 9 x 8, QWEN36_AGMM_GRID9);
    it keeps the q256 GQA multicast (12 heads x 9 q chunks = 108 <= 110 cores, tp_common.sdpa_gqa_mcast_chunk).
    The prefix arm is refused at any other chunk: its blockSize is 2048 and a resume must land on a chunk boundary
    (the plan["lead"] assert in _prefill_traced_chunked_body)."""
    v = _tt_public_getenv("QWEN36_PREFILL_CHUNK", "").strip()
    if not v:
        return PREFILL_CHUNK_DEFAULT
    c = int(v)
    if c < 128 or c % 128:
        raise ValueError(f"QWEN36_PREFILL_CHUNK={v}: must be a positive multiple of 128")
    if c != PREFILL_CHUNK_DEFAULT and _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") == "1":
        raise ValueError(f"QWEN36_PREFILL_CHUNK={c} with QWEN36_PREFIX_CACHING=1: the prefix arm needs chunk == "
                         f"blockSize == {PREFILL_CHUNK_DEFAULT}")
    return c


def prefill_mask_buckets():
    """The masked-bucket set: PREFILL_MASK_BUCKETS, plus the chunk itself when QWEN36_PREFILL_CHUNK is larger than the
    largest bucket (a 2304 chunk leaves tails of 2049..2303 tokens; without a 2304 bucket they would fall back to an
    unwarmed 128-multiple that compiles at request time and clobbers the parked trace, #48536). Unset = unchanged."""
    c = prefill_chunk_size()
    if c <= PREFILL_MASK_BUCKETS[-1] or c in PREFILL_MASK_BUCKETS:
        return PREFILL_MASK_BUCKETS
    return PREFILL_MASK_BUCKETS + (c,)


def prefill_mask_bucket_for(length):
    """Smallest fixed bucket >= length (falls back to the next 128-multiple)."""
    for b in prefill_mask_buckets():
        if length <= b:
            return b
    return ((length + 127) // 128) * 128


def prefill_chunk_plan(start, end, chunk_size, block_size):
    """Split the absolute token range [start, end) into a traced/eager prefill plan.

    Returns a dict:
      reset   -- True iff this is a sequence start (start == 0) and the GDN recurrent/conv
                 state must be re-zeroed. A RESUME (start > 0) must NOT reset: the caller
                 has already restored the prefix snapshot and the conv carry into the same
                 buffers, and zeroing them throws that state away.
      lead    -- (segment_start, segment_len) or None. Present only when `start` is not a
                 multiple of chunk_size: the traced chunk replay can only begin on a chunk
                 boundary (the trace's page-table/token buffers are chunk_size wide), so the
                 partial run-up to the next boundary is run eagerly through the masked bucket.
      replays -- absolute start offsets of the FULL chunks to replay from the parked trace.
      tail    -- (segment_start, segment_len) or None; the final partial chunk, eager.

    Pure host arithmetic on purpose: every one of these decisions was previously implicit in
    two divergent loops (prefill_traced_chunked, which always assumed start == 0, and
    prefill_prefix_slots, which never replayed anything), and the divergence is the defect
    this function exists to close.
    """
    if type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError("prefill plan needs int 0 <= start < end")
    if chunk_size <= 0 or chunk_size % 128:
        raise ValueError("chunk_size must be a positive multiple of 128")
    if block_size <= 0 or chunk_size % block_size:
        raise ValueError("chunk_size must be a whole number of attention pages")
    if start % block_size:
        raise ValueError("a resume must start on an attention page boundary")
    cur = start
    lead = None
    if start % chunk_size:
        boundary = ((start // chunk_size) + 1) * chunk_size
        stop = min(boundary, end)
        lead = (start, stop - start)
        cur = stop
    replays = []
    while cur + chunk_size <= end:
        replays.append(cur)
        cur += chunk_size
    tail = (cur, end - cur) if cur < end else None
    return {"reset": start == 0, "lead": lead, "replays": replays, "tail": tail}


class Qwen36Model:
    """Qwen3.5-9B text LM on Blackhole P150. HF_MODEL env var selects checkpoint."""

    def __init__(self, mesh_device, args, state_dict, tensor_cache_path=None):
        self.args = args
        self.device = mesh_device
        self.mesh_device = mesh_device  # Generator reads model.mesh_device
        self.num_devices = mesh_device.get_num_devices()
        # CCL for multi-device all-reduce; None on single device (ops no-op).
        if self.num_devices > 1:
            from models.tt_transformers.tt.ccl import TT_CCL

            self.tt_ccl = TT_CCL(mesh_device)
        else:
            self.tt_ccl = None
        self.configuration = args  # Generator reads model.configuration.max_seq_len
        self.sampling_dp = 1
        # RoPE is host-recomputed each step, so refresh all decode trace inputs.
        self._tt_vllm_always_refresh_decode_trace_inputs = True
        # On-device sampling: allowlist 1x4/1x8 TP only — vocab/TP must fit Top-K's 64K shard limit (TP=2 does not).
        mesh_shape = tuple(int(dim) for dim in mesh_device.shape)
        self._supports_on_device_sampling = (
            mesh_shape in ((1, 4), (1, 8))
            and args.vocab_size % self.num_devices == 0
            and (args.vocab_size // self.num_devices <= 64 * 1024)
        )
        if self._supports_on_device_sampling:
            from models.common.sampling.generator import SamplingGenerator

            # vocab/num_devices isn't a power of 2; the multi-device TopK kernel needs it padded.
            args.pad_logits_to_power_of_2 = True
            # force_argmax (the cheap 1-all-gather greedy path) is enabled on the base
            # SAMPLING_AG_CONFIG in model_config.py and runs IN-TRACE (faster than eager). Decode
            # bucketing is made compatible with the in-trace sampler by namespacing the sampling
            # trace per bucket width (SamplingGenerator.set_trace_bucket, driven from
            # qwen36_vllm.decode_forward) — see generator._validate_trace_inputs.
            self.sampling = SamplingGenerator(args=args, mesh_device=mesh_device, tt_ccl=self.tt_ccl)
        else:
            self.sampling = None
        # QWEN36_DEVICE_ARGMAX=1 (2026-09-13): greedy device sampler for the 1x2 mesh. The TopK 64K
        # logits/device limit blocks the stock SamplingGenerator at TP=2, but a per-shard argmax is
        # not subject to it: per-shard ttnn.argmax (lowest index on ties, measured) + gather of the
        # max value, a 2-way all_gather of (idx, max), and an on-device select (shard 0 wins ties, so
        # the global winner is the lowest index -- vLLM greedy semantics). Runs INSIDE the decode
        # trace; the host reads B uint32 token ids instead of the B x 248320 logits row. Greedy only:
        # the plugin gate (_tt_device_sampling_greedy_only) sends any step with a temperature>0 row
        # to host sampling. Evidence: [source validation record]
        self._device_argmax = (
            _tt_public_getenv("QWEN36_DEVICE_ARGMAX") == "1" and self.sampling is None and self.num_devices == 2
        )
        self._device_argmax_logged = False
        # QWEN36_DEVICE_SAMPLE=1: the device-side SAMPLED decode (the device sampler over the
        # TP-sharded vocab).  The decode trace computes the greedy ids (unchanged) AND the sampled
        # ids, and a per-step select flag picks per row: 0.0 -> the greedy ids (byte-identical to
        # today's greedy), 1.0 -> the sampled ids.  The per-step Gumbel noise is a trace input
        # (see prepare_decode_inputs_host / bind_decode_trace_inputs).
        self._ds_sampling = None
        self._ds_step = 0
        self._ds_block = None
        # QWEN36_DEVICE_RESIDENT=1 (phase 2, 2026-09-13): keep the decode token / position chain ON
        # DEVICE between steps -- a port of tt_transformers' _increment_decode_positions_device +
        # RotarySetup.get_rot_mats(ttnn index): the trace writes its own argmax id back into the
        # token input, plus_one's the position input, and derives RoPE cos/sin in-trace from the
        # position via ttnn.embedding on a device table, so no host input is rebuilt per step.
        # Requires the device argmax (the device must own the sampled token). The tables are built
        # with the SAME torch formula as prepare_decode_inputs_host and are bitwise identical to it
        # (bench/probes/rope_table_bitwise_check.py). Text-only: rope_delta must be 0 (asserted).
        self._device_resident = self._device_argmax and _tt_public_getenv("QWEN36_DEVICE_RESIDENT") == "1"
        self._device_resident_logged = False
        if self._device_resident:
            self._tt_vllm_always_refresh_decode_trace_inputs = False
            if dev_sample.enabled():
                # The resident decode chains the token/position on DEVICE and does NOT refresh the
                # decode trace inputs every step, so the per-step Gumbel block would be STALE and the
                # sampled ids would repeat the previous step's draw. Refuse the boot.
                raise RuntimeError(
                    "QWEN36_DEVICE_SAMPLE is incompatible with QWEN36_DEVICE_RESIDENT: the resident "
                    "decode does not refresh the decode trace inputs every step, so the per-step Gumbel "
                    "block would be stale. Refusing to boot.")
        # QWEN36_ONESHOT_AR=1 (1x2 mesh): _forward_decode gathers the embedding ONCE and keeps the residual
        # replicated through every layer (tp_common.oneshot_ar_enabled; needs the decode-levers layer.py/mlp.py).
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        self._oneshot_ar = self.num_devices > 1 and _tpc.oneshot_ar_enabled(mesh_device)
        # QWEN36_AR_ASYNC / QWEN36_RESID_L1 ride the replicated seam: layout + persistent workspace now, before
        # any warmup or trace capture.
        if self._oneshot_ar:
            _tpc.configure_decode_residual(args, self.tt_ccl, mesh_device)
        elif _tpc.ar_async_enabled() or _tpc.resid_l1_enabled():
            logger.warning("QWEN36_AR_ASYNC / QWEN36_RESID_L1 set but NOT engaged: they need QWEN36_ONESHOT_AR=1")

        # Framework Embedding (mesh-aware; replicates on 1-device mesh).
        from models.tt_transformers.tt.embedding import Embedding

        self.embd = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=tensor_cache_path,
            state_dict=state_dict,
            dtype=ttnn.bfloat16,
        )

        # RoPE setup (for gated attention layers only)
        self.rope = Qwen36RoPESetup(mesh_device, args)
        if getattr(self, "_device_resident", False):
            self._build_resident_rope_tables()

        # layer_indices (from from_pretrained) picks checkpoint layers; else 0..n_layers-1.
        # Each layer uses its real checkpoint index for weights and type (DeltaNet vs attn).
        self.layer_indices = getattr(args, "layer_indices", None) or list(range(args.n_layers))

        # Per-request vision grid (t,h,w), stashed by get_image_features / get_video_features so the
        # prefill paths can build the multimodal 3D RoPE (M-RoPE) position ids without threading
        # grid_thw through every prefill signature. Exactly one is non-None for a multimodal request
        # (image XOR video); both None => text-only. The active one also selects which placeholder
        # token id (image_token_id vs video_token_id) the vision-splice paths look for.
        self._req_image_grid_thw = None
        self._req_video_grid_thw = None

        # Transformer layers
        logger.info(f"Loading {len(self.layer_indices)} transformer layers (indices={self.layer_indices})...")
        self.layers = []
        for i in tqdm(self.layer_indices, desc="Loading layers"):
            layer = Qwen36DecoderLayer(mesh_device, args, state_dict, i, tensor_cache_path, tt_ccl=self.tt_ccl)
            self.layers.append(layer)

        # Framework RMSNorm (add_unit_offset=True). Single device: is_distributed=None.
        # 27B TP: hidden is sharded -> pass is_distributed + tt_ccl or use DistributedNorm.
        self.norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            state_dict=state_dict,
            weight_key="norm",
            weight_cache_path=tensor_cache_path,
            weight_dtype=ttnn.bfloat16,
            add_unit_offset=True,
            eps=args.norm_eps,
            **(
                dict(is_distributed=args.is_distributed_norm, ccl_topology=args.ccl_topology(), tt_ccl=self.tt_ccl)
                if self.num_devices > 1
                else {}
            ),
        )
        if self.num_devices > 1:
            # TP: DistributedNorm all-gathers fractured hidden for LM head.
            from models.tt_transformers.tt.distributed_norm import DistributedNorm

            self.norm = DistributedNorm(self.norm, args, tt_ccl=self.tt_ccl, TG=args.is_galaxy)

        # LM head [in,out]. Mesh: vocab-sharded (dim=-1); _lm_head all-gathers logits.
        # M=1 decode is weight-read-bound (~1.3GB/token), so sharding cuts bandwidth;
        # gather moves only the logit row. REPLICATED fallback if vocab indivisible.
        lm_head_weight = state_dict["output.weight"].T.contiguous()  # [dim, vocab_size]
        self._lmhead_vocab_sharded = self.num_devices > 1 and lm_head_weight.shape[-1] % self.num_devices == 0
        if self.num_devices > 1 and not self._lmhead_vocab_sharded:
            logger.warning(
                f"LM-head vocab {lm_head_weight.shape[-1]} not divisible by num_devices "
                f"{self.num_devices}; falling back to replicated LM head."
            )
        # QWEN36_LMHEAD_DTYPE=bf4 (c79): the head is the largest weight a decode step reads (675 MB/chip at
        # bf8); bfloat4_b halves it (576 vs 1088 B/tile). Plain block-fp rounding, not GPTQ, so numerics move:
        # gate as a whole (equiv tier + gsm8k + retrieval). The cache name carries the dtype, because the
        # tensor cache is keyed by file name only (a bf8 cache would silently load as bf8).
        _lm_dt_name = _tt_public_getenv("QWEN36_LMHEAD_DTYPE", "bf8").strip() or "bf8"
        if _lm_dt_name not in ("bf8", "bf4"):
            raise ValueError(f"QWEN36_LMHEAD_DTYPE={_lm_dt_name!r}: expected bf8 or bf4")
        lm_dtype = ttnn.bfloat4_b if _lm_dt_name == "bf4" else ttnn.bfloat8_b
        _lm_sfx = ".bf4" if _lm_dt_name == "bf4" else ""
        if self._lmhead_vocab_sharded:
            # Separate cache (.vshard): as_tensor ignores mesh_mapper on reload.
            lm_mapper = ttnn.ShardTensorToMesh(mesh_device, dim=-1)
            lm_cache = tensor_cache_path / f"output.weight.vshard{_lm_sfx}" if tensor_cache_path else None
        else:
            lm_mapper = ttnn.ReplicateTensorToMesh(mesh_device) if self.num_devices > 1 else None
            lm_cache = tensor_cache_path / f"output.weight{_lm_sfx}" if tensor_cache_path else None
        if _lm_dt_name != "bf8":
            logger.info(f"QWEN36_LMHEAD_DTYPE engaged: lm_head {_lm_dt_name} (cache {lm_cache.name if lm_cache else None})")
        self.lm_head_weight = ttnn.as_tensor(
            lm_head_weight,
            dtype=lm_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=lm_cache,
            **(dict(mesh_mapper=lm_mapper) if lm_mapper is not None else {}),
        )
        # QWEN36_DECODE_MR_LMHEAD (tp_common, D2): a decode-only chunked multi-reader DRAM-sharded copy
        # of the head, sliced on device; the interleaved tensor above stays for prefill. None = inert.
        self._lm_head_mr = _tpc.decode_mr_lmhead_for(args, mesh_device, self._lmhead_vocab_sharded)
        self._lm_head_mr_chunks = (
            _tpc.decode_mr_lmhead_weights(self.lm_head_weight, self._lm_head_mr) if self._lm_head_mr else None
        )
        # QWEN36_MTP_DRAFT_VOCAB (mtp_draft_vocab.py): the MTP draft steps' reduced-vocab head, cut from the host
        # head while it is in memory, in the same dtype / sharding / MR layout. None = inert.
        from models.demos.blackhole.qwen36.tt import mtp_draft_vocab as _mdv

        self._draft_vocab = _mdv.build(self, lm_head_weight, tensor_cache_path, lm_dtype, _lm_dt_name)

        self.vocab_size = args.vocab_size
        # True: return pre-gather vocab-sharded logits for per-shard argmax + host combine.
        self._ondev_argmax = False
        self._paged_kv_caches = None
        # Positions in self.layers of full-attn layers (not checkpoint indices); drives KV cache bind.
        self._attention_layer_indices = [pos for pos, layer in enumerate(self.layers) if layer.is_full_attention]
        self._deltanet_external_states = None  # (recurrent, conv) tuples; set by allocate_kv_caches
        # Shared zero buffers for in-place DN reset between traced replays.
        self._dn_zero_recurrent = None
        self._dn_zero_conv = None
        # Chunk-outer trace: one all-layer chunk captured, replayed per chunk via DMA inputs.
        # Persistent buffers below; addresses baked into trace.
        self._chunked_trace_id = None
        self._chunked_trace_output = None
        self._chunked_chunk_size = None
        self._chunk_token_buf = None
        self._chunk_start_idx_tensor = None
        self._chunk_page_table_buf = None
        self._chunk_full_page_table_buf = None
        self._chunk_cos_buf = None
        self._chunk_sin_buf = None
        # Traced batched short-prompt (bucket) prefill: one B=1 full-bucket trace replayed
        # once per user (see capture_prefill_trace_bucket / prefill_traced_bucket_batched).
        self._bucket_trace_id = None
        self._bucket_trace_output = None
        self._bucket_size = None
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_page_table_buf = None
        self._bucket_full_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._gdn_batched_prev = None  # batched GDN bindings saved during bucket-trace capture
        # QWEN36_PREFILL_BUCKET_TRACE (boot time): per-bucket MASKED prefill traces (short prompts + every
        # long-prompt tail). "0"/unset = the eager masked path byte for byte (no buffers, no traces, no extra KV
        # block, gdn_masks stays None); "1" = every bucket; a comma list ("128,256,512,1024") = only those.
        self._mb_trace_buckets = mbt.parse_bucket_trace_gate(
            _tt_public_getenv("QWEN36_PREFILL_BUCKET_TRACE"), prefill_mask_buckets()
        )
        # QWEN36_PREFILL_BUCKET_TRACE_MUSTFAIL=stale (falsifier): replay WITHOUT rewriting the value inputs
        # (tokens, rope, GDN masks, conv one-hot), so a replay reads the previous request's values; equiv must FAIL.
        self._mb_mustfail = mbt.parse_bucket_trace_mustfail(_tt_public_getenv("QWEN36_PREFILL_BUCKET_TRACE_MUSTFAIL"))
        # QWEN36_PREFILL_BUCKET_KDA (boot time, tier 2; docs/analysis/prefill-tail-20261015.md): the traced masked
        # bucket's GDN conv runs the kda op the full chunks use instead of the MAC FIR (gdn/tp.py). Refused without
        # the bucket trace; _MUSTFAIL=1 feeds the self-test's candidate a zero history, so the boot is refused.
        self._mb_kda, self._mb_kda_mf = mbt.parse_bucket_kda(
            _tt_public_getenv("QWEN36_PREFILL_BUCKET_KDA"), _tt_public_getenv("QWEN36_PREFILL_BUCKET_KDA_MUSTFAIL"),
            bool(self._mb_trace_buckets),
        )
        # QWEN36_PREFILL_BUCKET_KDA_MAX_LEN=N (boot time; [source validation record]): keep BOTH
        # masked-bucket bodies per bucket and replay the kda one only for a tail ending at prompt position <= N; longer
        # prompts replay the shipped MAC FIR body, so they prefill exactly as without QWEN36_PREFILL_BUCKET_KDA. Unset =
        # QWEN36_PREFILL_BUCKET_KDA as before (kda for every tail, one trace set). _MUSTFAIL=1 captures the "kda" set
        # with the FIR body, so the init self-test (_bkda_len_selftest) must refuse the boot (ttsim rung only).
        self._mb_kda_max_len, self._mb_kda_len_mf = mbt.parse_bucket_kda_max_len(
            _tt_public_getenv("QWEN36_PREFILL_BUCKET_KDA_MAX_LEN"),
            _tt_public_getenv("QWEN36_PREFILL_BUCKET_KDA_MAX_LEN_MUSTFAIL"),
            self._mb_kda,
        )
        self._mb_traces_kda = {}  # bucket -> kda-body _MaskedBucketTrace (QWEN36_PREFILL_BUCKET_KDA_MAX_LEN only)
        self._mb_kda_len_replays = {"kda": 0, "fir": 0}
        self._mb_bufs = {}  # bucket -> _MaskedBucketBufs (allocated + eagerly warmed BEFORE the chunk trace parks)
        self._mb_traces = {}  # bucket -> _MaskedBucketTrace (empty unless the gate is on)
        self._mb_replays = 0  # replay counter, logged so a post-run grep proves the path was taken
        self._pad_kv_block = None  # scratch KV block for the fill page table's pad entries (allocate_kv_caches)
        if self._mb_trace_buckets:
            # Not the identity banner: that one is printed only once every trace is CAPTURED.
            logger.info(f"[prefill] masked-bucket trace enabled for buckets {self._mb_trace_buckets}")
        # QWEN36_GDN_SLOT_COPY_TRACE=1 (c121, boot time, TP batched serving): see capture_gdn_slot_copy_traces.
        self._slot_copy_trace_on = _tt_public_getenv("QWEN36_GDN_SLOT_COPY_TRACE", "0") == "1"
        self._slot_copy_mustfail = (_tt_public_getenv("QWEN36_GDN_SLOT_COPY_TRACE_MUSTFAIL") or "").strip() or None
        if self._slot_copy_mustfail not in (None, "0", "wrongslot"):
            raise ValueError(
                f"QWEN36_GDN_SLOT_COPY_TRACE_MUSTFAIL={self._slot_copy_mustfail!r}: only 'wrongslot' is defined"
            )
        if self._slot_copy_mustfail == "0":
            self._slot_copy_mustfail = None
        if self._slot_copy_trace_on and _tt_public_getenv("QWEN36_GDN_SLOT_DEVICE_COPY") != "1":
            raise RuntimeError(
                "QWEN36_GDN_SLOT_COPY_TRACE=1 requires QWEN36_GDN_SLOT_DEVICE_COPY=1 (it traces that path's device "
                "copy; the host-snapshot path has nothing to trace). Refusing to boot."
            )
        if self._slot_copy_mustfail and not self._slot_copy_trace_on:
            raise RuntimeError("QWEN36_GDN_SLOT_COPY_TRACE_MUSTFAIL needs QWEN36_GDN_SLOT_COPY_TRACE=1. Refusing to boot.")
        self._slot_copy_traces = {}  # decode slot -> trace id (empty unless the gate is on and capture ran)
        self._slot_copy_captured = None  # per-GDN-layer (scratch, batched) tensors the traces baked (identity guard)
        self._slot_copy_fold = False  # conv finalize folded into the slot traces (QWEN36_GDN_CONV_WINDOW_ONCE on)
        # QWEN36_PREFIX_CKPT_DEFER (prefix_snap.StagePool): allocated by prefix_ckpt_pool_alloc before trace capture.
        self._ckpt_pool = None
        self._ckpt_stats = {"staged": 0, "immediate": 0, "resolved": 0, "reads": 0, "read_s": 0.0,
                            "selftest": "pending", "armed": False}
        # QWEN36_PREFIX_HEAD_DEFER: an ONESTEP head row's MTP warm finish is parked behind its slot copy + checkpoint.
        self._head_defer = _tt_public_getenv("QWEN36_PREFIX_HEAD_DEFER", "0") == "1"
        self._gdn_bound_batched = None  # batched GDN bindings saved by _bind_gdn_prefill_scratch while bound
        self._gdn_conv_finalize_skip = False  # set ONLY by prefill_paged_slots around prefill_traced_chunked
        # pp/host levers (tt/prefill_host.py; boot time, each default off = the served path unchanged):
        # QWEN36_PREFILL_RESET_TRACE / _TAIL_NOSYNC / _DEFER_FINISH.
        self._ph = phl.parse_flags(bucket_trace=bool(self._mb_trace_buckets), num_devices=self.num_devices)
        self._reset_trace_id = None  # QWEN36_PREFILL_RESET_TRACE: the captured GDN-reset trace
        self._reset_trace_ident = None  # the (layer, targets, zero sources) tensors it baked (identity guard)
        self._reset_trace_replays = 0
        self._ph_req = False  # True only inside a request-time prefill_paged_slots (the levers' scope)
        self._ph_refs = phl.HostRefs()  # QWEN36_PREFILL_TAIL_NOSYNC: host inputs held until the logits readback
        self._ph_nosync_on = False  # QWEN36_PREFILL_TAIL_NOSYNC armed for the current request
        self._ph_deferred = None  # QWEN36_PREFILL_DEFER_FINISH: the parked last-user MTP warm finish
        self._ph_deferred_runs = 0
        self._ph_selftested = False
        if self._ph.reset_trace or self._ph.any_request_path:
            logger.info(
                f"[prefill] pp/host levers requested: reset_trace={self._ph.reset_trace} "
                f"tail_nosync={self._ph.tail_nosync} defer_finish={self._ph.defer_finish}"
            )
        # QWEN36_PREFILL_PHASE_TIMING=1 (read per request): the live per-request phase timer, else None.
        self._phase_timer = None
        # QWEN36_PREFILL_HOST_TIMING=1 (read per request): the live per-request HOST-ONLY timer,
        # else None. Its marks never synchronize the device (see _ht_mark): phases that enqueue
        # async work measure host CPU only, so one boot splits the prefill host gap without
        # perturbing it. Flag-off cost is one attribute check per mark site.
        self._host_timer = None
        self._host_timing_banner = False
        # QWEN36_PREFILL_GROUP=N (REBOOT class, read at init): overrides the group size for LONG
        # buckets (bucket > 2*gdn_chunk) in prefill_paged_grouped. Unset/"" = current behavior
        # (long buckets capped at 1, i.e. per-user). N must be in [1, 8] (the fused GDN op's SCAN
        # ceiling BH=Bg*Nv_tp <= cores). Scope: GDN + AGMM + w2 + norms + RS batched; SDPA stays
        # per-user loop (attention/tp.py scalar chunk_start_idx).
        _pg_raw = (_tt_public_getenv("QWEN36_PREFILL_GROUP") or "").strip()
        if not _pg_raw or _pg_raw == "0":
            self._prefill_group = None
        else:
            try:
                self._prefill_group = int(_pg_raw)
            except ValueError:
                raise ValueError(f"QWEN36_PREFILL_GROUP={_pg_raw!r}: expected an integer in [1, 8]")
            if not 1 <= self._prefill_group <= 8:
                raise ValueError(f"QWEN36_PREFILL_GROUP={_pg_raw!r}: expected an integer in [1, 8]")
        # QWEN36_PREFILL_GROUP_MUSTFAIL=1 (falsifier): swaps the page-table rows between the two
        # users of a Bg=2 group, so the batch reads the wrong KV blocks; a correctness gate MUST FAIL.
        self._prefill_group_mustfail = _tt_public_getenv("QWEN36_PREFILL_GROUP_MUSTFAIL") == "1"
        if self._prefill_group_mustfail and self._prefill_group is None:
            raise RuntimeError("QWEN36_PREFILL_GROUP_MUSTFAIL=1 needs QWEN36_PREFILL_GROUP=N. Refusing to boot.")
        if self._prefill_group is not None:
            logger.info(f"QWEN36_PREFILL_GROUP engaged: Bg={self._prefill_group} (long-bucket override, default 1)")
            if self._prefill_group_mustfail:
                logger.warning("QWEN36_PREFILL_GROUP_MUSTFAIL engaged: page-table rows swapped within each "
                               "Bg=2 group; a correctness gate MUST FAIL")
        # Persistent B=1 GDN prefill scratch (batched serving): allocated once at warmup, its buffer
        # addresses are baked into the chunk-prefill trace and reused by every prefill_paged_slots
        self._gdn_prefill_scratch = None
        self._scratch_resident = None  # QWEN36_PREFIX_SCRATCH_RESUME: (req_id, end) whose GDN state the scratch holds
        # QWEN36_GDN_CONV_WINDOW_ONCE (c105, boot time, TP only): see prefill_traced_chunked. The GDN layers read the
        # same env at init; single device keeps the per-chunk writes (its chunk trace is captured without the skip).
        self._gdn_conv_window_once = _tpc.gdn_conv_window_once() and self.num_devices > 1
        if self._gdn_conv_window_once:
            logger.info("QWEN36_GDN_CONV_WINDOW_ONCE engaged: GDN decode conv window written once per prefill "
                        "(after the last chunk, from conv_carry) instead of in every chunk")
        elif _tpc.gdn_conv_window_once():
            logger.warning("QWEN36_GDN_CONV_WINDOW_ONCE set but NOT active: single device keeps per-chunk writes")

        # Optional vision tower (DropInVisionTransformer), attached lazily by
        # init_vision_model() for the multimodal serving path. None on the text-only path.
        self.vision_model = None
        self.vision_args = None

        # Trace-safe vision splice (traced serving path). The chunk/masked-bucket forwards run a
        # FIXED-shape ttnn.where(mask, vision, text) over these persistent buffers — compiled once
        # at warmup, then updated per request via copy_host_to_device, so no per-request program
        # ever compiles to clobber a parked trace. Allocated (single device only) in
        # capture_prefill_trace_chunked; None means "no traced path" -> the where is skipped.
        self._vis_buf = None  # [1, chunk_size, dim] bf16, image rows placed at their positions
        self._vis_mask_buf = None  # [1, chunk_size, 1] bf16, 1 at image positions else 0
        self._vis_zero_mask_host = None  # cached host zero mask for the clear (text/tail) path

    def init_vision_model(self, reference_visual=None, vision_args=None, dtype=ttnn.bfloat8_b, debug=False):
        """Build and attach the TT vision tower (DropInVisionTransformer).

        The vision tower runs on the SAME mesh as the text model. It still needs the HF
        reference visual for the patch embed / positional-interpolation steps that are not
        ported to TT; if ``reference_visual`` is not supplied it is loaded here via
        ``VisionModelArgs.reference_vision_model``. Idempotent — returns the existing tower
        if already built.

        Args:
            reference_visual: HF ``model.model.visual`` to wrap. Loaded internally if None.
            vision_args (VisionModelArgs): vision config on this mesh. Built internally if None.
            dtype (ttnn.dtype): compute dtype for the vision weights.
            debug (bool): run the reference vision path alongside and log PCC.

        Returns:
            DropInVisionTransformer: the attached vision tower.
        """
        if self.vision_model is not None:
            return self.vision_model
        from models.demos.blackhole.qwen36.tt.vision.model import DropInVisionTransformer
        from models.demos.blackhole.qwen36.tt.vision.vision_model_config import VisionModelArgs

        if vision_args is None:
            vision_args = VisionModelArgs(
                self.mesh_device,
                max_batch_size=self.args.max_batch_size,
                max_seq_len=self.args.max_seq_len,
            )
        if reference_visual is None:
            reference_visual = vision_args.reference_vision_model(depth=vision_args.hf_config.vision_config.depth)
        self.vision_args = vision_args
        self.vision_model = DropInVisionTransformer(reference_visual, vision_args, dtype=dtype, debug=debug)
        return self.vision_model

    def get_image_features(self, pixel_values, image_grid_thw):
        """Run the vision tower over a single user's images.

        Mirrors the HF reference's ``get_image_features`` seam: pixel patches in, packed
        image embeddings out — one row per image-placeholder token, ready to be spliced
        into the text embeddings by ``_scatter_vision_tokens``.

        Args:
            pixel_values (torch.Tensor): patchified pixels ``[num_patches, patch_dim]``.
            image_grid_thw (torch.Tensor): per-image grid ``(t, h, w)``, ``[num_images, 3]``.

        Returns:
            ttnn.Tensor: ``[num_image_tokens, H]`` image embeddings, hidden-fractured along
            the last dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_image_features()"
        # Stash the grid (as an IMAGE grid) so the prefill paths can build M-RoPE position ids for
        # this request (the splice positions in input_ids + the (t,h,w) grid are all M-RoPE needs).
        # Clear any stale video grid so the modality (and thus the placeholder token id) is image.
        self._req_image_grid_thw = image_grid_thw
        self._req_video_grid_thw = None
        image_features = self.vision_model.forward(pixel_values, grid_thw=image_grid_thw)
        # The vision tower returns [1, B, S, H]; flatten the leading (batch/seq) dims to the
        # packed [num_image_tokens, H] rows the text-model splice (_scatter_vision_tokens /
        # _set_vision_merge) expects. The hidden dim is unchanged so the mesh hidden-fracture
        # is preserved. B == 1 for now.
        hidden = image_features.shape[-1]
        return ttnn.reshape(image_features, (-1, hidden))

    def get_video_features(self, pixel_values_videos, video_grid_thw):
        """Run the vision tower over a single user's video frames.

        Mirrors the HF reference's ``get_video_features`` seam, which is just ``get_image_features``
        on the video pixels/grid — the vision tower forward is identical for image and video. The
        only differences are downstream: M-RoPE treats the grid as a VIDEO grid (split per frame by
        timestamps, modality==2), and the embeddings splice into ``video_token_id`` placeholders
        rather than ``image_token_id``. Both are selected by stashing the grid here as a video grid.

        Args:
            pixel_values_videos (torch.Tensor): patchified video pixels ``[num_patches, patch_dim]``.
            video_grid_thw (torch.Tensor): per-video grid ``(t, h, w)``, ``[num_videos, 3]``.

        Returns:
            ttnn.Tensor: ``[num_video_tokens, H]`` video embeddings, hidden-fractured along the last
            dim on a mesh (same sharding as the text embeddings).
        """
        assert self.vision_model is not None, "init_vision_model() must be called before get_video_features()"
        # Stash the grid as a VIDEO grid; clear any stale image grid so the modality (and the
        # placeholder token id) is video.
        self._req_video_grid_thw = video_grid_thw
        self._req_image_grid_thw = None
        video_features = self.vision_model.forward(pixel_values_videos, grid_thw=video_grid_thw)
        hidden = video_features.shape[-1]
        return ttnn.reshape(video_features, (-1, hidden))

    def _vision_placeholder_token_id(self):
        """The input-id the current request's vision embeddings splice into: ``video_token_id`` for
        a video request (video grid stashed), else ``image_token_id``. The vision-splice paths
        (_scatter_vision_tokens / _set_vision_merge / _vis_row_offset_for) use this to locate the
        placeholder positions, mirroring HF's ``input_ids == image_token_id`` /
        ``input_ids == video_token_id`` masks."""
        if self._req_video_grid_thw is not None:
            return int(self.args.hf_config.video_token_id)
        return int(self.args.hf_config.image_token_id)

    def _build_request_rope(self, token_ids, vision_tokens):
        """Stage the per-request RoPE for this prefill: M-RoPE (3D position ids + rope_delta) when
        the request is multimodal (vision_tokens present -> use the grid stashed by
        get_image_features / get_video_features), else clear to ordinary 1D RoPE. Call once at a
        prefill entry point with the REAL token ids (token_ids[:, :actual_len]); the chunk/tail
        seams then slice the staged table by sequence position and decode offsets by rope_delta."""
        image_grid = self._req_image_grid_thw if vision_tokens is not None else None
        video_grid = self._req_video_grid_thw if vision_tokens is not None else None
        self.rope.build_request_rope(token_ids, image_grid_thw=image_grid, video_grid_thw=video_grid)

    def _alloc_vision_merge_buffers(self, device, chunk_size):
        """Allocate the persistent vision-splice buffers used by the traced prefill path.

        ``_vis_buf`` holds the image embeddings placed at their token positions (zeros elsewhere);
        ``_vis_mask_buf`` is the 0/1 image mask. Both are zero-initialised, so the ttnn.where baked
        into the captured forward is the identity until a real multimodal request stages them.
        Allocating before warmup means the where compiles in the warmup pass (and is then
        captured), never at request time.

        Shapes/sharding match the activations of the forward that consumes them:
          - single device: vis [1, chunk_size, dim], mask [1, chunk_size, 1] (the 3D embd output);
          - TP: vis [1, 1, chunk_size, dim] HIDDEN-SHARDED across the mesh exactly like embd
            fractures its output (ShardTensor2dMesh dims=(None, -1)), so each device's where sees
            its own [.., dim/TP] vision columns; mask [1, 1, chunk_size, 1] REPLICATED (it
            broadcasts over the sharded hidden dim). DropInVisionTransformer fractures its output
            the same way, so the per-device columns line up.
        """
        if self._vis_buf is not None:
            return
        H = self.args.dim
        if self.num_devices > 1:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            self._vis_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, H, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=shard,
            )
            self._vis_mask_buf = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=rep,
            )
            self._vis_zero_mask_host = ttnn.from_torch(
                torch.zeros(1, 1, chunk_size, 1, dtype=torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            return
        self._vis_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, H, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_mask_buf = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._vis_zero_mask_host = ttnn.from_torch(
            torch.zeros(1, chunk_size, 1, dtype=torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )

    def _apply_vision_merge(self, x, length):
        """Final text-vs-vision SELECTION of the trace-safe splice: where(mask, vision, x).

        This is NOT a scatter and is not a substitute for one. The scatter — placing the n
        (variable) vision rows at their n image positions — already happened on host in
        _set_vision_merge, which is why ``_vis_buf`` here is a FULL [1, chunk_size, dim] buffer
        (image rows at their positions, zeros elsewhere), the same length as ``x``, NOT the raw
        [n, dim] vision tensor. So n << seq_len (a few image tokens in a long text prompt) is fine:
        the [.,1] mask broadcasts over the hidden dim and is 1 only at those n positions, so
        out == vision there and out == x (text) everywhere else (mask==0 is the exact identity).

        The placement is forced onto the host because aligning a variable n to fixed positions is
        inherently variable-shape; any on-device form (ttnn.scatter / pad / slice-copy) recompiles
        per request and would clobber the parked trace. ``length`` selects the segment (chunk_size
        for a full chunk, bucket for the masked path); the buffers slice down to it. No-op when the
        buffers are unallocated (text-only / non-traced deployment, which uses the device scatter)."""
        if self._vis_buf is None:
            return x
        if self._vis_buf.shape.rank == 4:
            # TP: buffers are [1, 1, chunk_size, dim(/TP)], matching the 4D TP activations.
            full = self._vis_buf.shape[2] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :, :length, :]
        else:
            full = self._vis_buf.shape[1] == length
            mask = self._vis_mask_buf if full else self._vis_mask_buf[:, :length, :]
            vis = self._vis_buf if full else self._vis_buf[:, :length, :]
        out = ttnn.where(mask, vis, x)
        ttnn.deallocate(x)
        return out

    def _set_vision_merge(self, ids_host, vision_tokens, vis_row_offset=0):
        """Stage the persistent vision buffers for the next forward (host -> device copy only;
        no program compiles). ``vision_tokens`` None clears the mask (the where becomes identity,
        for text-only requests); otherwise the packed image rows are read back to host, placed at
        their token positions in a zero [1, chunk_size, dim] buffer, and uploaded along with the
        0/1 mask. ``ids_host`` is the segment's token ids (torch), used to locate the image
        placeholders (== hf_config.image_token_id).

        ``vis_row_offset`` is the number of image-placeholder tokens that appear in the prompt
        BEFORE this segment, i.e. the index of the first packed vision row belonging to it. A
        large image whose placeholders span multiple prefill chunks (or spill into the tail) is
        thus spliced correctly: each segment consumes its own slice
        ``vis_host[vis_row_offset : vis_row_offset + n]`` of the packed rows. A segment with no
        image placeholders (text-only chunk / tail) clears the mask (identity merge)."""
        if self._vis_buf is None:
            # Fail loudly rather than silently drop the image: a multimodal request must run on a
            # path with the trace-safe buffers (capture_prefill_trace_chunked, single device) or
            # the on-device scatter (non-traced prefill_paged).
            assert vision_tokens is None, "vision merge requested but the trace-safe buffers are not allocated"
            return
        if vision_tokens is None:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        tp = self.num_devices > 1
        cs = self._vis_buf.shape[-2]  # seq dim: dim 1 (3D single) / dim 2 (4D TP)
        Hg = self.args.dim  # global hidden (the buffer's last dim is dim/TP on a mesh)
        flat = ids_host.reshape(-1)
        pos = torch.nonzero(flat[:cs] == self._vision_placeholder_token_id(), as_tuple=False).reshape(-1)
        n = int(pos.numel())
        # No image placeholders in this segment (text-only chunk, or a tail that holds none of the
        # image rows): the merge is the identity, so just clear the mask.
        if n == 0:
            ttnn.copy_host_to_device_tensor(self._vis_zero_mask_host, self._vis_mask_buf)
            return
        assert vis_row_offset + n <= int(vision_tokens.shape[0]), (
            f"vision splice out of range: row offset {vis_row_offset} + {n} image positions in this "
            f"segment exceeds {int(vision_tokens.shape[0])} packed vision rows"
        )
        # Gather the (hidden-fractured on a mesh) vision rows to full [num_image_tokens, Hg] on
        # host, then take this segment's slice. The placement is along the SEQ dim, orthogonal to
        # the hidden fracture, so the round-trip gather->place->reshard preserves the per-device
        # columns. ConcatMeshToTensor(dim=1) over the 2D [rows, dim/TP] is the inverse of the
        # dims=(None,-1) hidden shard used on re-upload.
        if tp:
            vis_host = ttnn.to_torch(vision_tokens, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=1)).to(
                torch.bfloat16
            )
        else:
            vis_host = ttnn.to_torch(vision_tokens).to(torch.bfloat16)  # [num_image_tokens, Hg]
        seg = vis_host[vis_row_offset : vis_row_offset + n]
        if tp:
            shard = ttnn.ShardTensor2dMesh(self.mesh_device, dims=(None, -1), mesh_shape=self.args.cluster_shape)
            rep = ttnn.ReplicateTensorToMesh(self.mesh_device)
            vis_full = torch.zeros(1, 1, cs, Hg, dtype=torch.bfloat16)
            vis_full[0, 0, pos] = seg
            mask = torch.zeros(1, 1, cs, 1, dtype=torch.bfloat16)
            mask[0, 0, pos, 0] = 1.0
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=shard),
                self._vis_buf,
            )
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep),
                self._vis_mask_buf,
            )
            return
        vis_full = torch.zeros(1, cs, Hg, dtype=torch.bfloat16)
        vis_full[0, pos] = seg
        mask = torch.zeros(1, cs, 1, dtype=torch.bfloat16)
        mask[0, pos, 0] = 1.0
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(vis_full, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_buf
        )
        ttnn.copy_host_to_device_tensor(
            ttnn.from_torch(mask, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT), self._vis_mask_buf
        )

    def _vis_row_offset_for(self, token_ids, chunk_start):
        """Packed-vision-row offset for the prefill segment starting at absolute position
        ``chunk_start``: the number of image-placeholder tokens before it. The vision rows are
        packed in image-placeholder order, so this is the index of the first row this segment
        owns — used to splice a large image whose placeholders span multiple chunks / the tail."""
        if chunk_start <= 0:
            return 0
        return int((token_ids[:, :chunk_start] == self._vision_placeholder_token_id()).sum())

    def switch_mode(self, mode):
        """Generator mode-change hook; no-op (no prefetcher)."""
        return None

    def _lm_head_matmul(self, x, decode=False):
        """ttnn-auto lm_head matmul, or the explicit 1D decode grid when QWEN36_1D_GRID_LM_HEAD is set
        (decode-shaped input only; prefill/multi-row callers keep auto).

        decode=True (the traced decode step only) with QWEN36_DECODE_MR_LMHEAD: the chunked multi-reader
        DRAM-sharded copy; same [.., B, vocab/TP] bf16 TILE DRAM result, so argmax / gather are unchanged.
        Prefill never passes decode=True and keeps the interleaved weight."""
        mr = getattr(self, "_lm_head_mr", None)
        if decode and mr is not None:
            if x.shape[-2] > ttnn.TILE_SIZE:
                raise RuntimeError(f"QWEN36_DECODE_MR_LMHEAD: decode lm_head input has {x.shape[-2]} rows > 32")
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            return tpc.decode_mr_lmhead_matmul(x, self._lm_head_mr_chunks, mr)
        pc = getattr(self.args, "lm_head_decode_1d_progcfg", None)
        if pc is not None and x.shape[-2] <= ttnn.TILE_SIZE and not x.is_sharded():
            from models.demos.blackhole.qwen36.tt import tp_common as tpc

            return tpc.matmul_1d_decode(
                x, self.lm_head_weight, pc, self.args.lm_head_decode_compute_cfg,
                out_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return ttnn.linear(x, self.lm_head_weight)

    def _lm_head(self, x, decode=False):
        """LM-head matmul. Vocab-sharded mesh: partial logits + all-gather to full replicated.
        Single device: plain matmul.

        decode=True (host-sampling decode only) honours QWEN36_LM_HEAD_GATHER_MODE:
          tile (default)  all-gather the TILE logits, as always
          none            no all-gather: return the vocab SHARD in ROW_MAJOR; process_output_decode
                          reads both shards (ConcatMeshToTensor dim=3). One CCL op fewer per token,
                          and the gather's 32-row tile-padded traffic (7.9 MB/device) is gone."""
        logits = self._lm_head_matmul(x, decode=decode)
        if decode:
            # Decode runs as a captured trace: this Python executes once at capture, so the flag
            # process_output_decode reads must be set here and NEVER touched by prefill calls.
            self._decode_logits_sharded = False
        if self._lmhead_vocab_sharded:
            from models.tt_transformers.tt.ccl import tt_all_gather

            _gm = _tt_public_getenv("QWEN36_LM_HEAD_GATHER_MODE", "tile") if decode else "tile"
            if _gm == "none":
                logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
                self._decode_logits_sharded = True
                if not getattr(self, "_nogather_logged", False):
                    self._nogather_logged = True
                    logger.info(
                        f"QWEN36_LM_HEAD_GATHER_MODE=none engaged: decode logits stay vocab-sharded "
                        f"{list(logits.shape)} {logits.layout} per device; host concatenates the shards"
                    )
                return logits
            logits = tt_all_gather(
                logits,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=None,
                dim=len(logits.shape) - 1,
                topology=self.args.ccl_topology(),
            )
        return logits

    def _final_norm_decode(self, x, replicated=False):
        """Final RMSNorm before the LM head (TP decode).

        The bare `self.norm(x, DECODE)` runs plain ttnn.rms_norm on a DRAM-interleaved [32,dim]
        tensor -> single tile-row -> 1 core (~80us/token). Passing the framework's 'lm_head' norm
        config runs the sharded multi-core norm across lm_head_core_grid instead; output_mem_config
        is forced back to DRAM so the LM-head matmul input is byte-identical (layout-only change).
        """
        if self.num_devices > 1:
            nc = dict(self.args.get_norm_config("lm_head", Mode.DECODE))
            nc["output_mem_config"] = ttnn.DRAM_MEMORY_CONFIG
            if replicated:  # QWEN36_ONESHOT_AR: residual already full width, skip DistributedNorm's all-gather
                from models.demos.blackhole.qwen36.tt import tp_common as tpc

                return tpc.norm_replicated_decode(self.norm, x, nc)
            return self.norm(x, mode=Mode.DECODE, norm_config=nc)
        return self.norm(x, mode=Mode.DECODE)

    @classmethod
    def from_pretrained(
        cls,
        device,
        max_batch_size=1,
        max_seq_len=2048,
        n_layers=None,
        layer_indices=None,
        hf_model=None,
    ):
        # HF_MODEL env var (hub or local path) is canonical; hf_model sets it for back-compat.
        if hf_model is not None:
            import os

            os.environ["HF_MODEL"] = hf_model

        args = Qwen36ModelArgs(
            mesh_device=device,
            max_batch_size=max_batch_size,
            max_seq_len=max_seq_len,
        )

        # layer_indices: run only these checkpoint layers (e.g. [0,3,31]) for profiling.
        # Each keeps its real type via full attention_type_list. Overrides n_layers truncation.
        if layer_indices is not None:
            layer_indices = list(layer_indices)
            assert layer_indices, "layer_indices must be non-empty"
            assert all(
                0 <= i < len(args.attention_type_list) for i in layer_indices
            ), f"layer_indices {layer_indices} out of range [0, {len(args.attention_type_list)})"
            args.layer_indices = layer_indices
            args.n_layers = len(layer_indices)
        elif n_layers is not None:
            args.n_layers = n_layers
            args.attention_type_list = args.attention_type_list[:n_layers]

        # NOTE: the warm-ttnn-cache HF-load skip is DISABLED for qwen3.6.
        # Its Gated-DeltaNet loader consumes conv weights on the host without a cache_file_name --
        # gdn/weights.py::load_conv_weight does ttnn.from_torch(state_dict[name], ...) for q/k/v_conv in
        # every DeltaNet layer, and gdn/tp.py derives taps the same way -- so a dataless placeholder
        # feeds those layers garbage while the HF load is skipped. This is the same failure that made
        # the vision demo emit token soup, on the text path. Re-enabling needs those conv weights either
        # cache-backed or captured to the sidecar via an is_host_weight predicate. (#45400 review)
        cache_path = args.weight_cache_path()
        logger.info("Loading + remapping weights via Qwen36ModelArgs.load_state_dict()...")
        state_dict = args.load_state_dict()

        model = cls(device, args, state_dict, tensor_cache_path=cache_path)
        return model

    def prefill_tp(self, token_ids, valid_len=None, vision_tokens=None):
        """Tensor-parallel full-model prefill (num_devices>1). Stateless: runs the
        whole sequence from scratch through the fractured-residual TP layers and
        returns the next-token logits at position valid_len-1.

        token_ids: torch [1, T] (pad T to a multiple of 128 for the GDN chunk
        kernel; right-padding does not affect the causal logit at valid_len-1).
        Returns ttnn logits [1, 1, 1, vocab_size] (host).
        """
        B, T = token_ids.shape
        assert B == 1, "prefill_tp is single-sequence"
        valid_len = valid_len or T

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text), then build cos/sin from
        # that staged sequence table — same source as the traced TP path (_rope_tp_cos_sin_torch).
        self._build_request_rope(token_ids[:, :valid_len], vision_tokens)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1, T, dim_frac] (hidden dim sharded across mesh)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)

        # Last real position via one-hot matmul (not slice): bare slice breaks at long T (~49k+).
        sel = torch.zeros(1, 1, 1, T, dtype=torch.float32)
        sel[0, 0, 0, valid_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=x.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, x)  # [1,1,1,dim_frac]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)  # DistributedNorm on selected row
        logits = self._lm_head(x_last)
        # Replicated logits; read one replica -> torch [vocab_size].
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def reset_tp(self):
        """Reset TP layer KV cache / GDN state for a new sequence."""
        for layer in self.layers:
            layer.attention.reset_state()

    def decode_tp(self, token_id, pos):
        """Single-token TP decode at position `pos` (B=1). Uses KV + GDN from prefill/decode."""
        from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

        tok = ttnn.from_torch(
            torch.tensor([[int(token_id)]], dtype=torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)  # [1,1,dim_frac]
        x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))  # [1,1,B=1,dim_frac]
        # RoPE position offset by rope_delta for multimodal (KV position cur_pos_tt stays `pos`).
        cos, sin = rot_mats_decode(
            self.device,
            self.args.rope_head_dim,
            self.args.max_seq_len,
            self.args.rope_theta,
            torch.tensor([pos + self.rope.rope_delta], dtype=torch.int32),
        )
        cur_pos_tt = ttnn.from_torch(
            torch.tensor([pos], dtype=torch.int32),
            dtype=ttnn.int32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tt)
        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=0))
        return lt[0].reshape(-1)[: self.vocab_size]

    def generate_tp(self, prompt_ids, max_new_tokens=20):
        """TP greedy generation: prefill prompt, then decode. Returns new token ids."""
        import math as _math

        self.reset_tp()
        T = len(prompt_ids)
        T_pad = max(128, _math.ceil(T / 128) * 128)
        padded = prompt_ids + [0] * (T_pad - T)
        logits = self.prefill_tp(torch.tensor([padded], dtype=torch.long), valid_len=T)
        nxt = int(torch.argmax(logits).item())
        out = [nxt]
        for pos in range(T, T + max_new_tokens - 1):
            logits = self.decode_tp(nxt, pos)
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
        return out

    def _scatter_vision_tokens(self, x, token_ids, vision_tokens):
        """Splice vision-model embeddings into the text token embeddings, on device.

        On-device equivalent of the HF reference's
        ``inputs_embeds.masked_scatter(image_mask, image_embeds)``. The embedding is
        flattened to ``[rows, H]``; the packed ``vision_tokens`` are placed into a zero
        buffer at the image-placeholder rows with a dim-0 ``ttnn.scatter``, then merged
        with the text embeddings via
        ``ttnn.where(special_image_mask, vision, text)``. The embeddings/vision never
        leave the device. No-op when ``vision_tokens`` is None or the prompt has no
        image tokens (the text-only path).

        The image-placeholder mask and placement positions are computed on host from the
        token ids (which already live on host at every call site), mirroring the HF
        reference's ``special_image_mask = input_ids == self.config.image_token_id``, then
        uploaded. The two tiny derived tensors uploaded are the ``[n, H]`` scatter index
        (hidden-sharded on a mesh, like the embedding activations) and the ``[rows, 1]``
        where-predicate (replicated on a mesh — it broadcasts over the sharded hidden dim).

        Args:
            x (ttnn.Tensor): text embeddings from ``self.embd`` — ``[B, T, H]`` (the
                raw embedding output), hidden fractured along the last dim on a mesh.
            token_ids (torch.Tensor): the prefill token ids (``input_ids``), ``[B, T]`` on
                host; image placeholders are the entries equal to ``hf_config.image_token_id``.
            vision_tokens (ttnn.Tensor): ``[num_image_tokens, H]`` produced by the
                vision tower (fractured along hidden on a mesh, like ``x``), one row
                per image placeholder token.

        Returns:
            ttnn.Tensor: same logical shape / layout / sharding as ``x`` with the
            vision embeddings scattered in.
        """
        if vision_tokens is None:
            return x

        orig_shape = tuple(x.shape)
        hidden = orig_shape[-1]
        rows = 1
        for d in orig_shape[:-1]:
            rows *= d

        # special_image_mask = input_ids == image_token_id, computed on host from the
        # token ids and uploaded. torch.nonzero gives the placement positions directly.
        flat_ids = token_ids.reshape(-1)
        mask_bool = flat_ids == self._vision_placeholder_token_id()
        pos = torch.nonzero(mask_bool, as_tuple=False).reshape(-1)
        n = int(pos.numel())
        if n == 0:
            return x
        assert n == int(
            vision_tokens.shape[0]
        ), f"input_ids has {n} image-token positions but vision_tokens has {int(vision_tokens.shape[0])} rows"

        # Placement index: the dim-0 rows of the flattened [rows, H] embedding to fill,
        # repeated across the hidden dim so the whole hidden vector at each row is written.
        # ttnn.scatter mirrors torch.scatter: out[index[i, h], h] = src[i, h], with
        # index/src/input the same rank.
        index = pos.view(n, 1).expand(n, hidden).contiguous().to(torch.int32)
        # where-predicate: [rows, 1], broadcasts over hidden in ttnn.where.
        mask_col = mask_bool.view(rows, 1)

        if self.num_devices > 1:
            # Shard the index along hidden the same way the embedding shards its
            # activations, so each device's [n, H/TP] index matches its local x/vision
            # shard (the hidden columns are identical, so splitting is free). The predicate
            # broadcasts over the sharded hidden dim, so it is replicated.
            index_tt = ttnn.from_torch(
                index,
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(None, 1), mesh_shape=self.args.cluster_shape
                ),
            )
            mask_tt = ttnn.from_torch(
                mask_col,
                dtype=x.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        else:
            index_tt = ttnn.from_torch(index, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
            mask_tt = ttnn.from_torch(mask_col, dtype=x.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)

        # ttnn.scatter requires input.dtype == src.dtype.
        src = vision_tokens if vision_tokens.dtype == x.dtype else ttnn.typecast(vision_tokens, x.dtype)

        # Place the packed vision rows into a zero buffer, then select per row:
        # vision at image positions, original text embedding everywhere else.
        x_2d = ttnn.reshape(x, (rows, hidden))
        vision_placed = ttnn.scatter(ttnn.zeros_like(x_2d), 0, index_tt, src)
        ttnn.deallocate(index_tt)
        out = ttnn.where(mask_tt, vision_placed, x_2d)
        ttnn.deallocate(vision_placed)
        ttnn.deallocate(mask_tt)
        return ttnn.reshape(out, orig_shape)

    def prefill(self, token_ids, vision_tokens=None):
        B, T = token_ids.shape

        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam.
        self._build_request_rope(token_ids, vision_tokens)

        if T > 1024:
            return self.prefill_layer_chunked(token_ids, chunk_size=2048, vision_tokens=vision_tokens)

        # Short sequences (<=1024)
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)

        cos, sin = self.rope.get_prefill_rot_mats(0, T)

        for layer in self.layers:
            x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

        x = self.norm(x, mode=Mode.PREFILL)

        x_last = x[:, -1:, :]
        logits = self._lm_head(x_last)

        return logits

    def prefill_layer_chunked(self, token_ids, chunk_size=2048, page_table=None, vision_tokens=None):
        """Prefill long sequences using layer-at-a-time chunked processing.

        DeltaNet uses larger chunk_size (256 vs 64) to limit Neumann-series error
        (4096 tokens -> 16 sub-chunks, PCC >0.98). page_table enables paged prefill."""
        B, T = token_ids.shape
        self.reset_state(batch_size=B)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(token_ids_ttnn)

        # Attn layers: chunk_size>=4096 (no Neumann limit; fewer SDPA compilations).
        attn_chunk_size = max(chunk_size, 4096)

        page_table_tt = None
        if page_table is not None:
            page_table_tt = ttnn.from_torch(
                page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )

        for layer_idx, layer in enumerate(self.layers):
            layer_chunk_size = attn_chunk_size if layer.is_full_attention else chunk_size

            chunks_out = []
            for chunk_start in range(0, T, layer_chunk_size):
                chunk_end = min(chunk_start + layer_chunk_size, T)

                x_chunk = x[:, chunk_start:chunk_end, :]
                x_chunk = ttnn.to_layout(x_chunk, ttnn.TILE_LAYOUT)

                if layer.is_full_attention and page_table is not None:
                    # Paged prefill path. M-RoPE-aware cos/sin for sequence positions of this chunk
                    # (slices the staged per-request table for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)

                    block_size = get_block_size(self._paged_kv_caches)  # was 64; 2048 on the prefix arm
                    chunk_blocks_end = math.ceil(chunk_end / block_size)
                    chunk_page_table = page_table[:, chunk_start // block_size : chunk_blocks_end]
                    chunk_page_table_tt = ttnn.from_torch(
                        chunk_page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
                    )

                    x_chunk = layer.forward(
                        x_chunk,
                        cos=cos,
                        sin=sin,
                        mode="prefill",
                        page_table=page_table_tt,
                        chunk_page_table=chunk_page_table_tt,
                        chunk_start_idx=chunk_start,
                    )

                elif layer.is_full_attention:
                    # Original concat path (non-paged prefill). M-RoPE-aware cos/sin (per-request
                    # table slice for multimodal; 1D RoPE otherwise).
                    cos, sin = self.rope.get_prefill_rot_mats(chunk_start, chunk_end - chunk_start)
                    x_chunk = layer.forward(x_chunk, cos=cos, sin=sin, mode="prefill")
                else:
                    x_chunk = layer.forward(
                        x_chunk,
                        cos=None,
                        sin=None,
                        mode="prefill",
                        chunk_size=layer.attention.long_prefill_chunk_size,
                    )

                chunks_out.append(x_chunk)

            # Last layer: save last token from last chunk before concat (avoids L1 clash on long T).
            is_last_layer = layer_idx == len(self.layers) - 1
            if is_last_layer:
                x_last = chunks_out[-1][:, -1:, :]
                x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)

            if len(chunks_out) == 1:
                x_new = chunks_out[0]
            else:
                x_new = ttnn.concat(chunks_out, dim=1)
                for c in chunks_out:
                    ttnn.deallocate(c)
            x_new = ttnn.to_memory_config(x_new, ttnn.DRAM_MEMORY_CONFIG)

            ttnn.deallocate(x)
            x = x_new

        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)

        return logits

    def decode(self, token_ids, current_pos):
        B = token_ids.shape[0]

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)

        # RoPE position is offset by rope_delta for a multimodal request (image tokens compress the
        # position space); the KV/cache position (cur_pos_tensor below) stays the true sequence pos.
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos for SDPA decode + paged_update_cache ([B*n_kv] after cache reshape).
        n_kv = self.args.n_kv_heads
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B * n_kv,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for i, layer in enumerate(self.layers):
            x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=cur_pos_tensor)

        x = self._final_norm_decode(x)
        if self._ondev_argmax:
            # Pre-gather vocab-sharded logits; caller argmaxes shards, skips all-gather + readback.
            logits = ttnn.linear(x, self.lm_head_weight)
        else:
            logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    def _forward_decode(self, token_ids_buf, cos, sin, cur_pos_tensor, page_table, sharded_lm_head=False):
        """Trace-safe paged decode. All inputs are device tensors.

        sharded_lm_head=True: return the pre-gather vocab-sharded logits (no all-gather)
        for the on-device sampler, which does its own cross-device top-k + gather.
        """
        x = self.embd(token_ids_buf)
        if self.num_devices > 1:
            # TP expects [1,1,B,dim_frac]; embd yields [B,1,dim_frac].
            x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1], x.shape[-1]))
        oneshot = {}
        if self._oneshot_ar:
            # Replicate the residual once; every layer then all-reduces to a replicated output.
            from models.tt_transformers.tt.ccl import tt_all_gather

            x = tt_all_gather(x, self.mesh_device, self.tt_ccl, cluster_axis=None, dim=3, topology=self.args.ccl_topology())
            oneshot = {"oneshot_ar": True}
        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x, cos, sin, position_tensor=cur_pos_tensor, page_table=page_table, mode="decode", **oneshot
                )
            else:
                x = layer.forward(x, mode="decode", **oneshot)
        x = self._final_norm_decode(x, replicated=self._oneshot_ar)
        if sharded_lm_head or self._ondev_argmax:
            # Pre-gather vocab-sharded logits (on-device sampling / greedy argmax).
            logits = self._lm_head_matmul(x, decode=True)
        else:
            logits = self._lm_head(x, decode=True)
        ttnn.deallocate(x)
        return logits

    def _forward_prefill_chunk(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """Trace-safe single-chunk prefill. Updates paged KV + GDN state in place.
        Returns last-layer hidden [1, chunk_size, hidden_size]."""
        x = self.embd(token_buf)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over persistent buffers; identity when the
        # mask buffer is zero, which is the case for every text-only chunk and request). The caller
        # stages the buffers before replaying chunk 0 of a multimodal prompt; chunks>0 are cleared.
        x = self._apply_vision_merge(x, length=x.shape[1])
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    **self._xadd_kw(_li),
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, **self._xadd_kw(_li))
            ttnn.deallocate(x)
            x = x_new
        return x

    def _rope_tp_cos_sin_torch(self, start, length):
        """Torch cos/sin tables [1, 1, length, rope_head_dim] for SEQUENCE positions
        [start, start+length), in the rope_tp (HF split-halves) format consumed by
        apply_partial_rope_prefill. Single source of truth for the TP masked-bucket and
        traced chunk-outer prefill paths (so the captured trace's cos/sin are byte-identical
        to the eager path's). M-RoPE-aware: when a multimodal request staged a per-sequence
        table (build_request_rope) this slices it; otherwise it is ordinary 1D RoPE at
        positions [start, start+length) — byte-identical to the pre-M-RoPE behaviour."""
        rd = self.args.rope_head_dim
        cos_t, sin_t = self.rope.prefill_cos_sin_torch(start, length)  # [length, rd] bf16
        cos = cos_t.reshape(1, 1, length, rd)
        sin = sin_t.reshape(1, 1, length, rd)
        return cos, sin

    def _forward_prefill_chunk_tp(
        self, token_buf, cos_buf, sin_buf, chunk_start_idx_tensor, full_page_table, chunk_page_table
    ):
        """TP trace-safe single-chunk prefill (replicated persistent buffers).
        Full chunk (valid_len==chunk_size); flexible SDPA via device chunk_start_idx.
        Returns hidden [1,1,chunk_size,dim]."""
        chunk_size = self._chunked_chunk_size
        x = self.embd(token_buf)
        x = ttnn.reshape(x, (1, 1, chunk_size, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers;
        # identity when the mask is zero, i.e. every text-only chunk). The caller stages the
        # buffers before replaying chunk 0 of a multimodal prompt; later chunks are cleared.
        x = self._apply_vision_merge(x, length=chunk_size)
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos_buf,
                    sin=sin_buf,
                    mode="prefill",
                    page_table=full_page_table,
                    chunk_page_table=chunk_page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    **self._xadd_kw(_li),
                )
            else:
                # valid_len=None: no GDN mask; trace-safe static conv capture (matches valid_len==chunk_size).
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=None, **self._xadd_kw(_li))
            ttnn.deallocate(x)
            x = x_new
        return x

    def capture_prefill_trace_chunked(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """Capture one chunk's all-layer prefill as a trace; replayed per chunk.

        Chunk-outer prefill stays under the 4 GiB trace limit at long context.
        Flexible SDPA (runtime chunk_start) makes one trace serve all chunk positions.

        capture_chunk_trace=False warms the masked-bucket programs but skips parking the chunk trace.
        The batched (B>1) vLLM path passes capture_chunk_trace=True with the PERSISTENT B=1 prefill
        scratch bound (_bind_gdn_prefill_scratch), so the trace bakes that scratch's addresses and
        long prompts replay the traced chunk path per user (prefill_paged_slots rebinds the scratch)."""
        if self.num_devices > 1:
            return self._capture_prefill_trace_chunked_tp(
                device,
                page_table,
                chunk_size=chunk_size,
                warmup_masked_buckets=warmup_masked_buckets,
                capture_chunk_trace=capture_chunk_trace,
            )
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        B = 1
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None

        self._chunked_chunk_size = chunk_size

        # Allocate the vision-splice buffers BEFORE warmup so the fixed-shape ttnn.where in
        # _forward_prefill_chunk / _forward_prefill_chunk_masked compiles in the warmup pass (and
        # is captured), never at request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, chunk_size)

        # ---- Persistent per-chunk input buffers (addresses baked into the trace) ----
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
        )
        # TP handoff: add ReplicateTensorToMesh for cos/sin (parity with tt/rope.py).
        self._chunk_cos_buf = ttnn.from_torch(
            self.rope.cos_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )
        self._chunk_sin_buf = ttnn.from_torch(
            self.rope.sin_cpu[:chunk_size].unsqueeze(0).contiguous(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Bind GDN to persistent external state; enable in-place carry across replays.
        for layer, (ext_rec, ext_conv) in zip(
            (l for l in self.layers if not l.is_full_attention), self._deltanet_external_states
        ):
            dn = layer.attention
            dn.recurrent_state = ext_rec
            dn.fused_conv_state = ext_conv
            dn.conv_state_q = None
            dn.conv_state_k = None
            dn.conv_state_v = None
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None
            dn._chunk_inplace_state = True
        self._init_dn_zero_buffers()

        # Warmup outside trace: compile per-chunk programs.
        self._reset_dn_state_inplace()
        warmup_out = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Warmup masked-bucket programs outside trace (same GDN mode as serving).
        # Dummy prefills dirty state/KV; reset below before capture.
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)

        # Capture trace.
        self._reset_dn_state_inplace()
        self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._chunked_trace_output = self._forward_prefill_chunk(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        logger.info("Chunked prefill trace captured successfully!")

    def _capture_prefill_trace_chunked_tp(
        self, device, page_table, chunk_size=2048, warmup_masked_buckets=True, capture_chunk_trace=True
    ):
        """TP fork of capture_prefill_trace_chunked.

        Replicated persistent buffers; rope_tp cos/sin; GDN uses _stable_state (not external buffers).
        Trace replays _forward_prefill_chunk_tp."""
        assert self._deltanet_external_states is not None, "Call allocate_kv_caches first"
        assert chunk_size % 128 == 0, f"chunk_size {chunk_size} must be a multiple of 128"
        self._dnorm_fuse_selftest()  # QWEN36_PREFILL_DNORM_FUSE: before any warmup/capture (no-op once tested / off)
        self._gdn_tail_selftest()  # QWEN36_PREFILL_GDN_TAIL: same (no-op once tested / off)
        self._gdn_qrm_selftest()  # QWEN36_PREFILL_GDN_QKV_RM: same (no-op once tested / off)
        self._rs_pkt4_selftest()  # QWEN36_PREFILL_RS_PKT4: same (no-op once tested / off)
        self._rs_bank_selftest()  # QWEN36_PREFILL_RS_BANK: same (no-op once tested / off)
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size

        if self._chunked_trace_id is not None:
            ttnn.release_trace(device, self._chunked_trace_id)
            self._chunked_trace_id = None
        self._chunked_chunk_size = chunk_size

        # Allocate the hidden-sharded vision-splice buffers BEFORE warmup so the fixed-shape
        # ttnn.where in the TP forwards compiles in the warmup pass (and is captured), never at
        # request time. Zero-initialised -> identity for text-only.
        self._alloc_vision_merge_buffers(device, chunk_size)

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Persistent per-chunk inputs (replicated; addresses baked into trace).
        self._chunk_token_buf = ttnn.from_torch(
            torch.zeros(B, chunk_size, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._chunk_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_chunk].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, chunk_size)
        self._chunk_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._chunk_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup outside trace: compile per-chunk programs.
        self._reset_gdn_state_for_new_sequence()
        # QWEN36_GDN_CONV_WINDOW_ONCE: the chunk body is warmed and captured with the conv-window skip armed (every
        # replay runs inside prefill_traced_chunked, which finalizes); the finalize programs compile here, pre-park.
        self._gdn_conv_window_defer(getattr(self, "_gdn_conv_window_once", False))
        warmup_out = self._forward_prefill_chunk_tp(
            self._chunk_token_buf,
            self._chunk_cos_buf,
            self._chunk_sin_buf,
            self._chunk_start_idx_tensor,
            self._chunk_full_page_table_buf,
            self._chunk_page_table_buf,
        )
        self._gdn_conv_window_defer(False)
        # QWEN36_MTP_WARM_TAIL: compile the warm gather for the chunk-trace output's spec (same ops -> same spec),
        # and the warm pass itself, here, before any trace is parked.
        self._mtp_warm_compile(warmup_out, chunk_size)
        ttnn.deallocate(warmup_out)
        if getattr(self, "_gdn_conv_window_once", False):
            self._finalize_gdn_conv_windows()
        ttnn.synchronize_device(device)

        # Warmup masked-bucket/tail programs outside trace (same GDN mode; avoids trace clobber).
        if warmup_masked_buckets:
            self.warmup_prefill_masked_buckets(page_table)
        # QWEN36_PREFILL_BUCKET_TRACE: allocate each gated bucket's persistent inputs and compile its traced body
        # eagerly NOW, before the chunk trace parks (a post-park compile clobbers a parked trace, #48536). The
        # captures come after the chunk capture below. No-op when the gate is off.
        if capture_chunk_trace:
            self._warm_prefill_bucket_traces(device, page_table)
            # QWEN36_GDN_SLOT_COPY_TRACE: compile every slot's copy programs eagerly, also before the park.
            self._warm_gdn_slot_copy_traces(device)

        if not capture_chunk_trace:
            # Batched (B>1) vLLM path: masked-bucket programs are warmed above; skip parking the
            # chunk trace (it would bake the B=1 prefill scratch that is freed after warmup, and
            # batched serving handles short prompts only). num_full==0 prompts never need it.
            self._chunked_trace_id = None
            self._reset_gdn_state_for_new_sequence()
            logger.info("Masked-bucket prefill programs (TP) warmed; chunk trace skipped (batched path).")
            return

        # Capture trace.
        self._reset_gdn_state_for_new_sequence()
        self._gdn_conv_window_defer(getattr(self, "_gdn_conv_window_once", False))
        try:
            self._chunked_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            self._chunked_trace_output = self._forward_prefill_chunk_tp(
                self._chunk_token_buf,
                self._chunk_cos_buf,
                self._chunk_sin_buf,
                self._chunk_start_idx_tensor,
                self._chunk_full_page_table_buf,
                self._chunk_page_table_buf,
            )
            ttnn.end_trace_capture(device, self._chunked_trace_id, cq_id=0)
        finally:
            self._gdn_conv_window_defer(False)
        logger.info("Chunked prefill trace (TP) captured successfully!")
        # Per-bucket masked-prefill traces (short prompts + every long-prompt tail). No-op unless
        # QWEN36_PREFILL_BUCKET_TRACE lists a bucket. After the chunk trace so it gets its trace-region bytes first.
        self.capture_prefill_bucket_traces(device)
        # Per-decode-slot GDN scratch -> slot copy traces (QWEN36_GDN_SLOT_COPY_TRACE). Tiny; captured last.
        self.capture_gdn_slot_copy_traces(device)
        # QWEN36_PREFILL_RESET_TRACE: the per-request GDN reset as one trace (after every prefill trace). No-op off.
        self._capture_gdn_reset_trace(device)

    # ----------------------------------------------------------------------- #
    # Traced batched SHORT-prompt prefill (B=32 / ISL<=128)
    # ----------------------------------------------------------------------- #
    # The traced chunk body and GDN chunk-seq kernel are B=1 (the kernel caps
    # BH=B*Nv_tp at ~32 => B<=4 at TP=4). So capture ONE B=1 full-bucket(128) trace and
    # replay it once per user: each replay DMAs the user's token slice + page-table row
    # into persistent buffers, execute_trace, then copy the B=1 GDN state into row u of
    # the batched [B,...] decode buffer. Full bucket => valid_len=None (trace-safe; a
    # one-hot mask in-trace TT_FATALs). Avoids per-layer host dispatch and per-op from_torch.

    def _alloc_gdn_scratch_b1(self):
        """Allocate a dedicated B=1 GDN state set on every GDN layer, distinct from the
        batched [B,...] decode buffer. Returns the prior batched bindings for the caller to
        restore for decode. MUST run before trace capture (allocates buffers)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            prev.append(
                (
                    dn,
                    dn.B,
                    dn.rec_state,
                    dn.conv_states,
                    dn.conv_carry,
                    dn._zero_conv0,
                    dn._stable_state,
                )
            )
            # reset_state allocates against self.B, so set B=1 first.
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            dn._stable_state = True  # in-place carry so the trace's baked addresses survive replays
        return prev

    def _restore_gdn_batched(self, prev):
        """Restore the batched [B,...] GDN bindings saved by _alloc_gdn_scratch_b1 and free
        the B=1 scratch, so decode reads the assembled batched state."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            # Free the B=1 scratch allocated for the prefill trace.
            if dn.rec_state is not None:
                ttnn.deallocate(dn.rec_state)
            for cs in dn.conv_states or []:
                ttnn.deallocate(cs)
            if dn.conv_carry is not None:
                ttnn.deallocate(dn.conv_carry)
            if dn._zero_conv0 is not None:
                ttnn.deallocate(dn._zero_conv0)
            # Rebind the batched decode buffers.
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._stable_state = stable_b

    def _ensure_gdn_prefill_scratch(self):
        """Allocate the PERSISTENT B=1 GDN prefill scratch once (idempotent).

        Unlike _alloc_gdn_scratch_b1 (throwaway, freed by _restore_gdn_batched), this scratch lives
        for the server lifetime: the batched chunk-prefill trace bakes its buffer addresses at warmup
        and every prefill_paged_slots replay reuses them, so it must never be freed/reallocated (only
        zeroed in place via _reset_gdn_state_for_new_sequence). Allocate at warmup so no device buffer
        is allocated at request time (which would be unsafe under the parked decode trace)."""
        if self._gdn_prefill_scratch is not None:
            return
        scratch = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            # reset_state allocates fresh B=1 buffers and assigns them WITHOUT freeing the current
            # (batched) ones, so save+restore the batched bindings and keep the scratch handles alive.
            saved = (dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state)
            dn.B = 1
            dn.reset_state()  # builds rec_state [1,Nv,Dk,Dv], conv_states[*] [1,1,D], conv_carry, _zero_conv0
            scratch.append((dn, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0))
            dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state = saved
        self._gdn_prefill_scratch = scratch

    def _bind_gdn_prefill_scratch(self, keep_resident=False):
        """Bind the persistent B=1 prefill scratch onto every GDN layer (prefill runs B=1); returns the
        saved batched decode bindings for _unbind_gdn_prefill_scratch. Allocates the scratch on first use.
        The drop-in analogue of _alloc_gdn_scratch_b1 that reuses one persistent scratch instead of
        allocating a throwaway per call (so the chunk trace's baked addresses stay valid).

        keep_resident: only prefill_paged_slots passes True (it tracks the QWEN36_PREFIX_SCRATCH_RESUME tag per row
        itself); every other binder may write the scratch, so it drops the resident tag."""
        if not keep_resident:
            self._scratch_resident = None
        self._ensure_gdn_prefill_scratch()
        prev = []
        for dn, rec, conv, carry, zero0 in self._gdn_prefill_scratch:
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = 1
            dn.rec_state = rec
            dn.conv_states = conv
            dn.conv_carry = carry
            dn._zero_conv0 = zero0
            dn._stable_state = True  # in-place carry so the trace's baked addresses survive replays
        self._gdn_bound_batched = prev  # QWEN36_GDN_SLOT_COPY_TRACE capture needs the batched targets
        return prev

    def _unbind_gdn_prefill_scratch(self, prev):
        """Rebind the batched [B,...] decode buffers saved by _bind_gdn_prefill_scratch, WITHOUT freeing
        the persistent scratch (unlike _restore_gdn_batched, whose scratch is throwaway)."""
        self._gdn_bound_batched = None
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            dn.B = B_b
            dn.rec_state = rec_b
            dn.conv_states = conv_b
            dn.conv_carry = carry_b
            dn._zero_conv0 = zero0_b
            dn._stable_state = stable_b

    def _snapshot_gdn_scratch(self):
        """Snapshot the B=1 GDN scratch (host torch) to restore around the throwaway capture run."""
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        out = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            out.append(
                (
                    ttnn.to_torch(dn.rec_state, mesh_composer=comp).clone(),
                    [ttnn.to_torch(c, mesh_composer=comp).clone() for c in dn.conv_states],
                )
            )
        return out

    def _restore_gdn_scratch(self, snap):
        """Restore the B=1 GDN scratch in place (preserving the addresses the trace baked in)
        from a _snapshot_gdn_scratch result."""
        self._validate_gdn_snapshot(snap)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper)

        for layer, (rec, convs) in zip((l for l in self.layers if not l.is_full_attention), snap):
            dn = layer.attention
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)

    def _restore_gdn_scratch_fast(self, snap):
        """QWEN36_PREFIX_SNAP_FAST, the H2D half: _restore_gdn_scratch with each bf16 recurrent state uploaded row
        major and tilized on the device (prefix_snap.restore_rec). [Nv, Dk, Dv] = [24, 128, 128] is tile aligned, so
        the device tilize pads nothing and the tiles hold the same bytes the host tilize makes. The conv taps (small,
        [1, 1, D] padded to a tile row) keep the host tilize. The first restore of a boot also runs the legacy restore
        and compares the scratch bitwise (self-test, with a wrong-input branch)."""
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dns = self._gdn_layers()
        st = getattr(self, "_snap_fast_restore", None)
        if st is None:
            st = self._snap_fast_restore = {"restores": 0, "selftest": "pending"}
        legacy_digest = None
        if st["selftest"] == "pending":
            self._restore_gdn_scratch(snap)
            legacy_digest = prefix_snap.scratch_digest(ttnn, self.mesh_device,
                                                       [t for dn in dns for t in (dn.rec_state, *dn.conv_states)])
            for dn in dns:
                dn.reset_state_inplace()  # the fast restore below must write every byte itself
        for dn, (rec, convs) in zip(dns, snap):
            prefix_snap.restore_rec(ttnn, self.mesh_device, mapper, rec, dn.rec_state,
                                    wrong=prefix_snap.mustfail() == "wrongslot")
            for j, c in enumerate(convs):
                cc = ttnn.from_torch(c, dtype=dn.conv_states[j].dtype, layout=ttnn.TILE_LAYOUT,
                                     device=self.mesh_device, mesh_mapper=mapper)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)
        st["restores"] += 1
        if legacy_digest is not None:
            fast_digest = prefix_snap.scratch_digest(ttnn, self.mesh_device,
                                                     [t for dn in dns for t in (dn.rec_state, *dn.conv_states)])
            if not prefix_snap.digests_equal(legacy_digest, fast_digest):
                raise RuntimeError(f"{prefix_snap.ENV} restore self-test FAILED: fast restore != legacy: refusing")
            k = 1 + dns[0].K
            if prefix_snap.digests_equal(legacy_digest[k:] + legacy_digest[:k], fast_digest):
                raise RuntimeError(f"{prefix_snap.ENV} restore self-test wrong-input branch PASSED: refusing")
            st["selftest"] = "exact"
            logger.info(f"QWEN36_PREFIX_SNAP_FAST restore self-test exact: device-tilized restore == host-tilized restore "
                        f"({len(dns)} GDN layers); wrong-input branch refused")

    def _validate_gdn_snapshot(self, snap):
        """Validate the complete host B=1 snapshot before any upload or live write."""
        layers = [l.attention for l in self.layers if not l.is_full_attention]
        if not isinstance(snap, (tuple, list)) or len(snap) != len(layers):
            raise ValueError("incomplete GDN snapshot")
        dtypes = {ttnn.float32: torch.float32, ttnn.bfloat16: torch.bfloat16}
        for dn, state in zip(layers, snap):
            if (tuple(dn.rec_state.shape) != (dn.B, dn.Nv, dn.Dk, dn.Dv)
                    or len(dn.conv_states) != dn.K
                    or any(tuple(c.shape) != (1, dn.B, dn.qkv_dim_tp) for c in dn.conv_states)):
                raise ValueError("unsupported live GDN state layout")
            if not isinstance(state, (tuple, list)) or len(state) != 2:
                raise ValueError("GDN snapshot requires recurrent and convolution state")
            rec, convs = state
            if not isinstance(convs, list) or len(convs) != dn.K:
                raise ValueError("incomplete convolution snapshot")
            tensors = [(rec, (self.num_devices, dn.Nv, dn.Dk, dn.Dv), dn.rec_state.dtype)]
            tensors.extend((c, (self.num_devices, 1, dn.qkv_dim_tp), live.dtype)
                           for c, live in zip(convs, dn.conv_states))
            for tensor, shape, dtype in tensors:
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                        or tuple(tensor.shape) != shape or tensor.dtype != dtypes.get(dtype)):
                    raise ValueError(f"invalid host GDN tensor: expected {shape}, {dtypes.get(dtype)}")

    def _validate_prefix_slots(self, slots):
        layers = [l.attention for l in self.layers if not l.is_full_attention]
        if (not isinstance(slots, list) or any(type(s) is not int for s in slots)
                or len(set(slots)) != len(slots)
                or any(any(s < 0 or s >= dn.B for dn in layers) for s in slots)):
            raise ValueError("prefix slots must be distinct live decode slots")

    def prefix_cache_snapshot(self, slots):
        """Read live slots, preserving TP device order and owning all returned host bytes."""
        self._validate_prefix_slots(slots)
        # NOTHING TO SNAPSHOT MEANS NO READBACK. The loop below pulls rec_state and
        # every conv_state of all 48 GDN layers device->host with ttnn.to_torch, and
        # those calls sit OUTSIDE the per-slot loop -- so with slots == [] it used to
        # read the entire GDN state back and discard every byte.
        #
        # That is the common case on the decode path, not a corner: execute_model
        # resets _qwen_prefix_plans to [] each step and only prepare() (prefill)
        # repopulates it, so every decode step called this with an empty list.
        # MEASURED 2026-09-16: prefix-arm ITL 373.8 ms with this readback, against
        # 47.5 ms production. See [source validation record]
        if not slots:
            return []
        if self._ckpt_pool is not None:
            # QWEN36_PREFIX_CKPT_DEFER: stage what the pool holds, read the rest now.
            dn_states = [(dn.rec_state, dn.conv_states, dn.B, dn.Nv, dn.Dk, dn.Dv) for dn in self._gdn_layers()]
            out = [self._ckpt_stage(dn_states, s) for s in slots]
            rest = [i for i, r in enumerate(out) if r is None]
            if rest:
                for i, snap in zip(rest, self._prefix_cache_snapshot_fast([slots[i] for i in rest])):
                    out[i] = snap
            return out
        if prefix_snap.enabled():
            return self._prefix_cache_snapshot_fast(slots)
        result = [[] for _ in slots]
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            rec = ttnn.to_torch(dn.rec_state, mesh_composer=comp).reshape(
                self.num_devices, dn.B, dn.Nv, dn.Dk, dn.Dv)
            convs = [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states]
            for out, slot in zip(result, slots):
                out.append((rec[:, slot].clone(), [c[:, slot:slot + 1].clone() for c in convs]))
        for snap in result:
            self._validate_gdn_snapshot(snap)
        return result

    def _prefix_cache_snapshot_fast(self, slots, dn_states=None):
        """QWEN36_PREFIX_SNAP_FAST (prefix_snap.py): the wanted rows only, bf16 untilized on the device. The first
        snapshot that holds a non-empty state also runs the legacy readback and compares bitwise (the self-test, with
        its wrong-input branch), so every boot that serves a checkpoint proves the fast path on real traffic."""
        import time as _time

        st = getattr(self, "_snap_fast_stats", None)
        if st is None:
            st = self._snap_fast_stats = {"calls": 0, "slots": 0, "s": 0.0, "max_s": 0.0, "selftest": "pending",
                                          "_last_log": 0.0}
            rec0, _, b0 = (dn_states[0][:3] if dn_states is not None
                           else (self._gdn_layers()[0].rec_state, None, self._gdn_layers()[0].B))
            logger.info(
                f"QWEN36_PREFIX_SNAP_FAST engaged: GDN checkpoint readback of the wanted slots only (B={b0}, "
                f"rec {rec0.dtype}, device untilize for bf16)"
                + (f" MUSTFAIL={prefix_snap.mustfail()}" if prefix_snap.mustfail() else ""))
            if prefix_snap.nb_enabled():
                logger.info(f"QWEN36_PREFIX_SNAP_NB engaged: checkpoint reads non-blocking, one synchronize per "
                            f"{prefix_snap.nb_layers()} GDN layers")
            if prefix_snap.pack_enabled():
                logger.info(f"QWEN36_PREFIX_SNAP_PACK engaged: {prefix_snap.pack_layers()} GDN layers per device read"
                            + (f" MUSTFAIL={prefix_snap.pack_mustfail()}" if prefix_snap.pack_mustfail() else ""))
        if dn_states is None:
            dn_states = [(dn.rec_state, dn.conv_states, dn.B, dn.Nv, dn.Dk, dn.Dv) for dn in self._gdn_layers()]
        t0 = _time.perf_counter()
        result = prefix_snap.fast_snapshot(ttnn, self.mesh_device, self.num_devices, dn_states, slots,
                                           wrong_slot=prefix_snap.mustfail() == "wrongslot",
                                           nb=prefix_snap.nb_enabled(), group_layers=prefix_snap.nb_layers(),
                                           pack=prefix_snap.pack_enabled(), pack_layers=prefix_snap.pack_layers(),
                                           pack_swap=prefix_snap.pack_mustfail() == "swap")
        dt = _time.perf_counter() - t0
        st["calls"] += 1
        st["slots"] += len(slots)
        st["s"] += dt
        st["max_s"] = max(st["max_s"], dt)
        if st["selftest"] == "pending":
            legacy = prefix_snap.legacy_snapshot(ttnn, self.mesh_device, self.num_devices, dn_states, slots)
            ok, detail = prefix_snap.selftest_compare(result, legacy)
            if not ok:
                raise RuntimeError(f"{prefix_snap.ENV} self-test FAILED on a served checkpoint ({detail}): refusing")
            st["selftest"] = "exact"
            logger.info(f"QWEN36_PREFIX_SNAP_FAST self-test exact: {len(slots)} slot(s) x {len(dn_states)} GDN layers "
                        f"bitwise equal to the legacy readback; wrong-input branch refused")
        now = _time.monotonic()
        if now - st["_last_log"] >= 60.0:
            st["_last_log"] = now
            logger.info(f"{prefix_snap.ENV} stats: calls={st['calls']} slots={st['slots']} "
                        f"total_s={st['s']:.3f} mean_ms={1000 * st['s'] / st['calls']:.1f} "
                        f"max_ms={1000 * st['max_s']:.1f} last_ms={1000 * dt:.1f}")
        for snap in result:
            self._validate_gdn_snapshot(snap)
        return result

    def prefix_cache_restore(self, slots, snapshots):
        """Reject malformed batches before writes; device failures require engine abort."""
        self._validate_prefix_slots(slots)
        if not isinstance(snapshots, list) or len(slots) != len(snapshots):
            raise ValueError("one complete snapshot per slot required")
        for snap in snapshots:
            self._validate_gdn_snapshot(snap)
        for slot, snap in zip(slots, snapshots):
            self._write_gdn_slot(slot, [r for r, _ in snap], [c for _, c in snap])

    def prefill_prefix_slots(self, tokens, page_table, prompt_lens, slots, start_pos, states):
        """Prefill each suffix through the SAME chunk path production uses, resumed at `start`.

        Tokens contain the whole prefix and suffix. Starts must be attention-page
        aligned; end positions need not be. KV allocation/CoW belongs to the runner.

        WAS: an eager prefill_masked_bucket(bucket=2048) per 2048 tokens, on the stated ground
        that "offset-dependent compilation is not qualified alongside parked decode traces"
        (patch 0017 header). MEASURED 2026-09-17 that this is the arm's dominant cost and that
        the ground was wrong: production's traced chunk path compiles nothing per offset -- the
        chunk position is a runtime device tensor. Matched ~8-9k cold prefills, py-spy,
        [source validation record],prefix}-*.jsonl: production 3,014 tok/s and
        73.9% blocked in a native call (device bound); this arm 1,493 tok/s and 79.8% host
        bound, 61.8% of it raw ttnn op dispatch. One execute_trace per chunk replaces it.
        """
        self._validate_prefix_slots(slots)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.device.type != "cpu":
            raise ValueError("whole prefix+suffix CPU token rows required")
        n = tokens.shape[0]
        if (tokens.ndim != 2 or n == 0 or not isinstance(page_table, torch.Tensor)
                or page_table.ndim != 2 or page_table.shape[0] != n
                or any(not isinstance(x, list) or len(x) != n for x in (prompt_lens, slots, start_pos, states))):
            raise ValueError("one length, slot, start and state per token row required")
        block_size = get_block_size(self._paged_kv_caches)
        if 2048 % block_size:
            raise ValueError("prefix eager bucket must be page aligned")
        for end, start, snap in zip(prompt_lens, start_pos, states):
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= tokens.shape[1]
                    or start % block_size or end > page_table.shape[1] * block_size):
                raise ValueError("invalid absolute prefix/suffix bounds or page table")
            if (start == 0) != (snap is None):
                raise ValueError("cold miss requires None; positive start requires a complete snapshot")
            if snap is not None:
                self._validate_gdn_snapshot(snap)
        prev = self._bind_gdn_prefill_scratch()
        output, final_states = [], []
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        try:
            for u, (start, end, snap) in enumerate(zip(start_pos, prompt_lens, states)):
                # Idempotent: prefill_traced_chunked re-stages the same whole-prompt, position-
                # indexed table below. Kept here so the restore path is unchanged if that moves.
                self._build_request_rope(tokens[u:u + 1, :end], None)
                if start > 0:
                    self._restore_gdn_scratch(snap)
                    # Decode taps 1..K-1 are the chronological left context for prefill.
                    mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
                    for dn, (_, convs) in zip((l.attention for l in self.layers if not l.is_full_attention), snap):
                        carry = ttnn.from_torch(torch.cat(convs[1:], dim=1), dtype=dn.conv_carry.dtype,
                                                layout=ttnn.TILE_LAYOUT, device=self.mesh_device, mesh_mapper=mapper)
                        ttnn.copy(carry, dn.conv_carry)
                        ttnn.deallocate(carry)
                # ONE entry point shared with production (prefill_paged_slots), differing only
                # by `start`. prefill_traced_chunked owns the reset decision via the plan, so the
                # restored snapshot above is never zeroed; at start == 0 it resets exactly as the
                # old loop's first chunk_start == 0 bucket did.
                lg = self.prefill_traced_chunked(tokens[u:u + 1, :end], page_table[u:u + 1],
                                                 actual_len=end, start=start)
                output.append(ttnn.to_torch(lg, mesh_composer=comp).reshape(
                    -1, self.args.vocab_size)[:1].float().view(1, 1, -1))
                ttnn.deallocate(lg)
                final_states.append(self._snapshot_gdn_scratch())
        finally:
            self._unbind_gdn_prefill_scratch(prev)
        self.prefix_cache_restore(slots, final_states)
        return torch.cat(output, dim=0)

    def capture_prefill_trace_bucket(self, device, page_table, bucket=128):
        """Capture ONE B=1 full-bucket prefill trace (all-layer forward, valid_len=None) for
        batched serving of prompts whose length is EXACTLY the bucket. GDN points at a B=1
        scratch. Replay once per user via prefill_traced_bucket_batched.

        Only full-bucket prompts are traced: valid_len cannot be masked inside a trace, so a
        short prompt would pad through the GDN recurrence and corrupt the decode state. Callers
        route actual_len < bucket prompts to eager prefill_paged_peruser.

        Args:
          device: mesh device.
          page_table: torch.Tensor [1, bpu] int32 — one user's row (buffer width fixed across
            replays; each replay DMAs a different user's row in).
          bucket: fixed bucket length (128 for ISL==128; must be a multiple of 128).
        """
        assert self.num_devices > 1, "capture_prefill_trace_bucket is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        assert bucket % 128 == 0, f"bucket {bucket} must be a multiple of 128 (GDN sub-chunk)"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size

        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(device, self._bucket_trace_id)
            self._bucket_trace_id = None
        self._bucket_size = bucket
        # _forward_prefill_chunk_tp sizes its reshape/loop from _chunked_chunk_size; point it
        # at the bucket. Save the prior value so release_prefill_trace_bucket can restore it
        # (a later chunked prefill on the same model must not be left at 128).
        self._chunked_chunk_size_prebucket = self._chunked_chunk_size
        self._chunked_chunk_size = bucket

        # Swap GDN to a dedicated B=1 scratch for capture + replay (the trace writes B=1).
        self._gdn_batched_prev = self._alloc_gdn_scratch_b1()

        rep = ttnn.ReplicateTensorToMesh(device)
        B = 1
        # Full-page-table buffer width MUST be a 32-multiple >= the SDPA's target_blocks so
        # forward_prefill_paged's zero-PAD branch (ttnn.zeros + ttnn.concat — a host write that
        # TT_FATALs in a trace) never runs during replay. Replays' _fit_pt_row pad to this width.
        buf_blocks = max(32, ((page_table.shape[-1] + 31) // 32) * 32)
        if page_table.shape[-1] < buf_blocks:
            page_table = torch.cat(
                [page_table, torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[-1], dtype=torch.int32)],
                dim=1,
            )
        self._bucket_buf_blocks = buf_blocks
        # Persistent per-replay input buffers (replicated; addresses baked into the trace).
        self._bucket_token_buf = ttnn.from_torch(
            torch.zeros(B, bucket, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_start_idx_tensor = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        self._bucket_full_page_table_buf = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_page_table_buf = ttnn.from_torch(
            page_table[:, :blocks_per_bucket].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            mesh_mapper=rep,
        )
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        self._bucket_cos_buf = ttnn.from_torch(
            cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )
        self._bucket_sin_buf = ttnn.from_torch(
            sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=rep
        )

        # Warmup OUTSIDE the trace: compile every program the capture + replays run (a compile
        # during replay would clobber the parked trace). Two sets: (a) the full-bucket forward
        # (the trace body); (b) the logit-select run eagerly after each replay.
        self._reset_gdn_state_for_new_sequence()
        warmup_out = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        # Warm the logit-select at actual_len=bucket. Its program is fixed per bucket; actual_len
        # only changes the one-hot values (a host write), so one warmup covers every actual_len.
        warm_logits = self._masked_bucket_logits_tp(warmup_out, bucket, bucket)
        ttnn.deallocate(warm_logits)
        ttnn.deallocate(warmup_out)
        ttnn.synchronize_device(device)

        # Capture: snapshot the B=1 scratch, run the throwaway compile+capture passes, restore
        # so the addresses the trace baked in stay valid (both passes advance the in-place GDN
        # recurrence; KV at block 0 is harmlessly overwritten by the first real replay).
        self._reset_gdn_state_for_new_sequence()
        gdn_snap = self._snapshot_gdn_scratch()
        self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        self._bucket_trace_id = ttnn.begin_trace_capture(device, cq_id=0)
        self._bucket_trace_output = self._forward_prefill_chunk_tp(
            self._bucket_token_buf,
            self._bucket_cos_buf,
            self._bucket_sin_buf,
            self._bucket_start_idx_tensor,
            self._bucket_full_page_table_buf,
            self._bucket_page_table_buf,
        )
        ttnn.end_trace_capture(device, self._bucket_trace_id, cq_id=0)
        self._restore_gdn_scratch(gdn_snap)
        logger.info(f"Bucket({bucket}) prefill trace (TP) captured successfully!")

    def release_prefill_trace_bucket(self):
        """Release the captured bucket prefill trace + persistent buffers and restore the
        batched GDN bindings for decode. Called after prefill_traced_bucket_batched."""
        if getattr(self, "_bucket_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._bucket_trace_id)
            self._bucket_trace_id = None
        for buf in (
            getattr(self, "_bucket_token_buf", None),
            getattr(self, "_bucket_start_idx_tensor", None),
            getattr(self, "_bucket_full_page_table_buf", None),
            getattr(self, "_bucket_page_table_buf", None),
            getattr(self, "_bucket_cos_buf", None),
            getattr(self, "_bucket_sin_buf", None),
        ):
            if buf is not None:
                ttnn.deallocate(buf)
        self._bucket_token_buf = None
        self._bucket_start_idx_tensor = None
        self._bucket_full_page_table_buf = None
        self._bucket_page_table_buf = None
        self._bucket_cos_buf = None
        self._bucket_sin_buf = None
        self._bucket_trace_output = None
        # Restore _chunked_chunk_size (capture pointed it at the bucket) for a later chunked prefill.
        if hasattr(self, "_chunked_chunk_size_prebucket"):
            self._chunked_chunk_size = self._chunked_chunk_size_prebucket
            del self._chunked_chunk_size_prebucket
        # Restore the batched [B,...] GDN decode buffers the prefill assembled into.
        if getattr(self, "_gdn_batched_prev", None) is not None:
            self._restore_gdn_batched(self._gdn_batched_prev)
            self._gdn_batched_prev = None

    def prefill_traced_bucket_batched(self, token_ids_list, page_table, valid_lens=None):
        """Traced batched short-prompt prefill: replay the captured B=1 full-bucket trace once
        per user, stitching each replay's B=1 GDN state into row u of the batched [B,...] decode
        buffer. Attention fills each user's physical blocks via the per-user page-table row
        (batch_idx=0 baked into the trace). Returns a list of B device logits [1, 1, vocab].

        CORRECTNESS CONTRACT: every user's actual_len MUST equal the captured bucket. The trace
        runs valid_len=None (no GDN mask); for a short prompt padded to the bucket that would push
        padding tokens through the GDN recurrence and corrupt the decode state. Short prompts must
        be routed to eager prefill_paged_peruser. Asserts actual_len == bucket and never pads.
        Call capture_prefill_trace_bucket first; release_prefill_trace_bucket before decode.
        """
        assert self.num_devices > 1, "prefill_traced_bucket_batched is the TP (num_devices>1) path"
        assert getattr(self, "_bucket_trace_id", None) is not None, "Call capture_prefill_trace_bucket first"
        bucket = self._bucket_size
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_bucket = bucket // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Pad/clip each request's page-table row to the captured buffer width. Keep the
        # [1, buf_blocks] batch dim — copy_host_to_device_tensor requires identical logical shapes.
        buf_blocks = int(self._bucket_full_page_table_buf.shape[-1])

        def _fit_pt_row(row):
            row = row.reshape(1, -1)  # [1, bpu] -> ensure 2D
            if row.shape[1] < buf_blocks:
                row = torch.cat([row, torch.zeros(1, buf_blocks - row.shape[1], dtype=row.dtype)], dim=1)
            elif row.shape[1] > buf_blocks:
                row = row[:, :buf_blocks]
            return row.contiguous()

        # Per-user logits are read to HOST during the loop and re-uploaded at the end: each replay
        # overwrites the persistent trace output, and the post-loop assembly churns device memory.
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Collect each replay's B=1 GDN state to assemble into the batched decode buffer. Snapshot
        # via a host to_torch round trip (NOT ttnn.clone() — that allocates from the same general
        # device pool the captured trace's own baked-address intermediates draw from, so the next
        # user's execute_trace() silently overwrites the "cloned" snapshot; confirmed via
        # checksumming a live tensor changing value with nothing writing to it, ruled out as a race
        # since an added synchronize_device() didn't change the deterministic wrong result).
        per_user_rec = []
        per_user_conv = []
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                actual = valid_lens[u] if valid_lens is not None else toks.shape[1]
                # CORRECTNESS: traced path serves ONLY full-bucket prompts (see docstring); short
                # prompts must be routed to eager prefill_paged_peruser.
                assert actual == bucket, (
                    f"user {u}: actual_len {actual} != bucket {bucket}; the traced bucket prefill "
                    f"only serves full-bucket prompts — route short prompts to prefill_paged_peruser"
                )

                # Zero the B=1 GDN scratch before each user (address-stable; the trace baked these in).
                self._reset_gdn_state_for_new_sequence()

                # Full bucket, no padding (actual == bucket).
                token_buf = toks[:, :bucket].to(torch.int32)

                # DMA this user's inputs into the persistent buffers (addresses preserved).
                tok_host = ttnn.from_torch(
                    token_buf, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(tok_host, self._bucket_token_buf)

                row = _fit_pt_row(page_table_torch[u])
                pt_host = ttnn.from_torch(
                    row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
                )
                ttnn.copy_host_to_device_tensor(pt_host, self._bucket_full_page_table_buf)
                cpt_host = ttnn.from_torch(
                    row[:, :blocks_per_bucket].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=None,
                    mesh_mapper=rep,
                )
                ttnn.copy_host_to_device_tensor(cpt_host, self._bucket_page_table_buf)

                # chunk_start=0 and cos/sin for [0, bucket) are baked in; no per-replay DMA needed.
                ttnn.execute_trace(self.device, self._bucket_trace_id, cq_id=0, blocking=False)
                # Sync before reading this user's state/logit: the trace writes hidden/rec_state/
                # conv_states in place and the next replay would overwrite them.
                ttnn.synchronize_device(self.device)

                # Gather this replay's B=1 GDN state for assembly after the loop. The next replay
                # resets the scratch IN PLACE, so snapshot now via host to_torch (see comment above).
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                )

                # Logit at actual_len-1, read to HOST immediately (before the next replay overwrites
                # the trace output). Re-uploaded at the end.
                lg = self._masked_bucket_logits_tp(self._bucket_trace_output, actual, bucket)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

            ttnn.synchronize_device(self.device)
        finally:
            # Rebind the batched buffers regardless of success or failure (restore was deferred so
            # the loop could use the B=1 scratch) — otherwise a mid-loop assertion/exception (e.g. a
            # short prompt hitting the actual_len == bucket check) leaves every GDN layer pointed at
            # the B=1 scratch instead of the batched decode buffers.
            if getattr(self, "_gdn_batched_prev", None) is not None:
                self._restore_gdn_batched(self._gdn_batched_prev)
                self._gdn_batched_prev = None

        # Assemble the per-user states into row u in place (_stable_state path).
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)

        # Re-upload the per-user logits as stable device tensors after all allocations.
        return self._reupload_host_logits(host_logits)

    def _assemble_per_user_gdn(self, per_user_rec, per_user_conv):
        """Stitch B per-user B=1 GDN states (host torch) into row u of the batched [B,...] decode
        buffers via assemble_batched_state. The batched GDN bindings MUST already be rebound (writes
        in place under _stable_state). Shared by prefill_traced_bucket_batched/prefill_chunked_peruser.

        per_user_rec[u][li]:  host rec_state snapshot for user u, GDN layer li (mesh dim 0 = devices).
        per_user_conv[u][li]: list of K host conv_states snapshots for user u, GDN layer li.
        """
        B = len(per_user_rec)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for li, dn in enumerate(dn_layers):
            K = dn.K
            D = dn.qkv_dim_tp
            rec_list = [
                ttnn.from_torch(
                    per_user_rec[u][li],
                    dtype=dn.rec_state.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=mapper,
                )
                for u in range(B)
            ]
            # Rebuild the [1, K-1, D] carry tensor the assembler expects from the per-slot
            # conv_states snapshot (slots 1..K-1; slot 0 is the shifted-out zero). Each slot
            # snapshot is [1, 1, D]; concat along dim 1 -> [1, K-1, D].
            conv_carry_list = []
            staging = []  # (per-slot tensors) to deallocate after assembly
            for u in range(B):
                slots = [
                    ttnn.from_torch(
                        per_user_conv[u][li][m],
                        dtype=dn.conv_states[m].dtype,
                        layout=ttnn.TILE_LAYOUT,
                        device=self.mesh_device,
                        mesh_mapper=mapper,
                    )
                    for m in range(1, K)
                ]
                staging.extend(slots)
                conv_carry_list.append(ttnn.concat(slots, dim=1) if K - 1 > 1 else ttnn.reshape(slots[0], (1, 1, D)))
            # assemble_batched_state takes ownership of rec_list + conv_carry_list and deallocates
            # them (gdn/tp.py); we only free the per-slot staging tensors it never sees.
            dn.assemble_batched_state(rec_list, conv_carry_list)
            for t in staging:
                ttnn.deallocate(t)

    def _reupload_host_logits(self, host_logits):
        """Re-upload per-user host logits (read to host during a batched prefill loop) as stable
        replicated device tensors [1, 1, vocab] — the prefill_paged_peruser return contract.
        Done after all per-user execute/assembly allocations so the returned tensors are stable."""
        return [
            ttnn.from_torch(
                hl,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            for hl in host_logits
        ]

    # ----------------------------------------------------------------------- #
    # QWEN36_PREFILL_DEFER_FINISH (pp/host, tt/prefill_host.py)
    # ----------------------------------------------------------------------- #
    def _ph_defer_finish(self, mw, prompt_ids_t, logits, slot, pt_row):
        """Park the warm finish of the request begun last on `mw`. Clones the runner-owned inputs (the runner may reuse
        its token / block-table buffers before the flush)."""
        if self._ph_deferred is not None:
            self.ph_flush_deferred()  # a user whose prefill gathered nothing: its predecessor's finish runs now
        req, mw.req = mw.req, None  # prefill_paged_slots' abort() is now a no-op for this request
        if req is None:
            return
        if self._ph.defer_finish_mf:
            return  # MUSTFAIL: the parked finish is dropped (slot starts cold); the self-test must refuse
        self._ph_deferred = phl.DeferredFinish(
            req=req, args=(prompt_ids_t.clone(), logits, int(slot), pt_row.clone())
        )

    def ph_flush_deferred(self):
        """Run the parked warm finish (if any). Called at every model entry that can touch the warm window, MTP state
        or the device queue after a prefill step: prefill_forward / decode_forward / release_request (qwen36_vllm.py)
        and prefill_paged_slots itself."""
        d = self._ph_deferred
        if d is None:
            return False
        self._ph_deferred = None
        mw = self._mtp_warm()
        if mw is None:
            return False
        cur, mw.req = mw.req, d.req  # cur: the next user's open request (begun, nothing gathered yet) or None
        try:
            ok = mw.finish(d.args[0].tolist(), d.args[1], d.args[2], d.args[3])
        finally:
            mw.req = cur
        self._ph_deferred_runs += 1
        if self._ph_deferred_runs == 1 or self._ph_deferred_runs % 256 == 0:
            logger.info(f"QWEN36_PREFILL_DEFER_FINISH flushed: runs={self._ph_deferred_runs} slot={d.args[2]} warmed={ok}")
        return ok

    def prefill_host_selftest(self, num_blocks):
        """QWEN36_PREFILL_TAIL_NOSYNC / QWEN36_PREFILL_DEFER_FINISH in-boot self-test (fail closed), run once after the
        decode + MTP traces are captured (qwen36_vllm.warmup_model_decode), i.e. in the served state. Two synthetic
        prompts in ONE prefill_paged_slots call (user 0: 2 full chunks + a 1000-row tail into slot 0; user 1: 1 chunk +
        a 300-row tail into slot 1, so user 0's finish is the inner, hidden one and user 1's the parked one), levers OFF
        then ON. Both chips, bitwise: both users' host logits, the B=1 GDN scratch state the last slot copy read, the
        MTP warm window buffers each user's finish() READ (digested at the call) and after the run, and both MTP host
        slot records (present). Wrong-input branch: user 0's last prompt token changed, through the lever path, must
        change user 0's logits, else the comparison is blind and the boot is refused. Leaves slots 0/1's MTP host
        state reset (the first real request into a slot overwrites its GDN row and the KV blocks used here)."""
        if not self._ph.any_request_path or self._ph_selftested:
            return
        if not (self.num_devices > 1 and int(getattr(self.args, "max_batch_size", 0) or 0) > 1):
            logger.info("pp/host request-path levers idle: prefill_paged_slots serves only max_batch_size > 1 (TP)")
            return
        import copy as _copy
        import hashlib

        chunk = int(self._chunked_chunk_size or prefill_chunk_size())
        bs = int(get_block_size(self._paged_kv_caches))
        lens = [2 * chunk + 1000, chunk + 300]
        need = [-(-n // bs) for n in lens]
        free = [b for b in range(int(num_blocks)) if b != self._pad_kv_block]
        if len(free) < sum(need):
            raise RuntimeError(f"pp/host self-test REFUSED: {num_blocks} KV blocks cannot hold the {lens} probe")
        width = max(need)
        pt = torch.zeros(2, width, dtype=torch.int32)
        pt[0, : need[0]] = torch.tensor(free[: need[0]], dtype=torch.int32)
        pt[1, : need[1]] = torch.tensor(free[need[0] : need[0] + need[1]], dtype=torch.int32)
        g = torch.Generator().manual_seed(4321)
        hi = min(100000, int(self.args.vocab_size))
        toks = [torch.randint(1000, hi, (1, n), generator=g, dtype=torch.int32) for n in lens]
        toks_w = [t.clone() for t in toks]
        toks_w[0][0, -1] = toks_w[0][0, -1] + 7  # the wrong input: user 0's last prompt token changed
        slots = [0, 1]
        mtp = getattr(self, "_mtp_batch", None)
        mw = self._mtp_warm()
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)

        def _digest(tensors):
            h = hashlib.sha256()
            for t in tensors:
                v = ttnn.to_torch(t, mesh_composer=comp).contiguous()
                h.update(str((tuple(v.shape), str(v.dtype))).encode())
                h.update(v.view(torch.uint8).numpy().tobytes())
            return h.hexdigest()

        def _scratch_digest():
            prev = self._bind_gdn_prefill_scratch()
            try:
                return _digest([t for dn in self._gdn_layers() for t in [dn.rec_state, *dn.conv_states, dn.conv_carry]])
            finally:
                self._unbind_gdn_prefill_scratch(prev)

        def _wins():
            return _digest(mw.wins) if (mw is not None and getattr(mw, "wins", None)) else None

        def _run(levers, tk):
            saved = self._ph
            self._ph = levers
            seen = {}
            orig = mw.finish if mw is not None else None
            if mw is not None:
                def _spy(prompt_ids, logits, slot, pt_row):
                    seen[int(slot)] = _wins()  # what this finish reads
                    return orig(prompt_ids, logits, slot, pt_row)
                mw.finish = _spy
            try:
                if mtp is not None:
                    for sl in slots:
                        mtp.host.reset_slot(sl)
                lg = self.prefill_paged_slots(tk, pt, slots, valid_lens=lens)
                ttnn.synchronize_device(self.device)
                self.ph_flush_deferred()
                ttnn.synchronize_device(self.device)
                st = _scratch_digest()
                wins = _wins()
                hs = [_copy.deepcopy(mtp.host.slots[sl]) for sl in slots] if mtp is not None else None
            finally:
                self._ph = saved
                if mw is not None:
                    del mw.finish  # back to the class method
            return {"lg": lg, "st": st, "wins": wins, "seen": seen, "hs": hs}

        warm0 = (mw.stats.get("warmed", 0) if mw is not None else 0, mtp.host.stats.get("warm", 0) if mtp else 0)
        ref = _run(phl.HostFlags(), toks)
        lev = _run(self._ph, toks)
        wrong = _run(self._ph, toks_w)
        if mtp is not None:
            for sl in slots:
                mtp.host.reset_slot(sl)
            mtp.host.stats["warm"] = warm0[1]
        if mw is not None:
            mw.stats["warmed"] = warm0[0]
        fails = []
        for u in range(2):
            if not torch.equal(ref["lg"][u], lev["lg"][u]):
                fails.append(f"user {u} logits differ (max |d| {float((ref['lg'][u] - lev['lg'][u]).abs().max()):.3g})")
        if ref["st"] != lev["st"]:
            fails.append("GDN scratch state differs")
        if ref["wins"] != lev["wins"]:
            fails.append("MTP warm window buffers differ after the run")
        if ref["seen"] != lev["seen"]:
            fails.append(f"MTP warm windows read by finish() differ (slots ref {sorted(ref['seen'])} lev {sorted(lev['seen'])})")
        if ref["hs"] != lev["hs"]:
            fails.append(f"MTP host slot records differ ({ref['hs']} vs {lev['hs']})")
        if mw is not None and (not ref["hs"] or any(h is None for h in ref["hs"]) or len(ref["seen"]) != 2):
            fails.append("the reference prefill did not warm both MTP slots (the finish comparison is blind)")
        if torch.equal(ref["lg"][0], wrong["lg"][0]):
            raise RuntimeError(
                "pp/host self-test REFUSED: the wrong-input branch (last token changed) gave identical logits "
                "(the comparison is blind)"
            )
        if fails:
            raise RuntimeError(
                f"pp/host self-test REFUSED (tail_nosync={self._ph.tail_nosync} defer_finish={self._ph.defer_finish} "
                f"defer_mustfail={self._ph.defer_finish_mf}): " + "; ".join(fails)
            )
        self._ph_selftested = True
        self._ph_deferred_runs = 0
        logger.info(
            f"QWEN36_PREFILL_HOST self-test exact: 2 users {lens} in one call on {self.num_devices} devices; logits, "
            f"GDN scratch state ({len(self._gdn_layers())} layers), MTP warm windows (as each finish read them and after) "
            f"and both slot records identical levers off/on; wrong-input branch differs"
        )
        if self._ph.tail_nosync:
            logger.info("QWEN36_PREFILL_TAIL_NOSYNC engaged: no drain before the tail, no sync after its replay")
        if self._ph.defer_finish:
            logger.info(
                "QWEN36_PREFILL_DEFER_FINISH engaged: each user's MTP warm finish runs behind the next replay "
                "(inner users) or at the next model entry (last user)"
                + (" MUSTFAIL: dropped" if self._ph.defer_finish_mf else "")
            )

    def _prefix_restore_scratch(self, snap):
        """QWEN36_PREFIX_CACHING resume: write a host GDN snapshot (prefix_cache_snapshot's per-layer
        (rec [devices, Nv, Dk, Dv], conv taps K x [devices, 1, D])) into the BOUND B=1 prefill scratch in place, and
        its conv carry from taps 1..K-1 (the decode window is tap 0 = zero, tap j+1 = carry row j: copy_scratch_to_slot
        / finalize_conv_window), so the next chunk sees exactly the state a cold prefill would have carried there."""
        self._validate_gdn_snapshot(snap)
        if prefix_snap.enabled():
            self._restore_gdn_scratch_fast(snap)
        else:
            self._restore_gdn_scratch(snap)
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        for dn, (_, convs) in zip((l.attention for l in self.layers if not l.is_full_attention), snap):
            carry = ttnn.from_torch(torch.cat(convs[1:], dim=1), dtype=dn.conv_carry.dtype, layout=ttnn.TILE_LAYOUT,
                                    device=self.mesh_device, mesh_mapper=mapper)
            ttnn.copy(carry, dn.conv_carry)
            ttnn.deallocate(carry)

    def _prefix_scratch_resume(self, snap):
        """QWEN36_PREFIX_SCRATCH_RESUME: the bound B=1 scratch already holds this continuation's state (prefix_snap.py),
        so the host restore is skipped. Self-test on the first resident continuation of a boot: digest the scratch's
        rec_state + conv_carry, run the restore anyway, digest again; the two must be bitwise equal, and the
        wrong-input branch (the resident digest against the next layer's) must differ. A failure refuses the step."""
        st = getattr(self, "_scratch_resume_stats", None)
        if prefix_snap.is_staged(snap):
            # QWEN36_PREFIX_CKPT_DEFER: an ONESTEP head staged its checkpoint (only after this self-test passed); the
            # host snapshot is validated when the runner resolves it.
            if st is None or st["selftest"] != "exact":
                raise RuntimeError(f"{prefix_snap.ENV_DEFER}: staged checkpoint before the scratch-resume self-test")
            st["skipped"] += 1
            return
        self._validate_gdn_snapshot(snap)
        if st is None:
            st = self._scratch_resume_stats = {"skipped": 0, "selftest": "pending"}
        if prefix_snap.resume_mustfail() == "stale":
            for dn in self._gdn_layers():  # MUSTFAIL: the scratch no longer holds the state (zeroed in place)
                dn.reset_state_inplace()
        if st["selftest"] == "pending":
            dns = self._gdn_layers()
            tensors = [t for dn in dns for t in (dn.rec_state, dn.conv_carry)]
            before = prefix_snap.scratch_digest(ttnn, self.mesh_device, tensors)
            self._prefix_restore_scratch(snap)
            after = prefix_snap.scratch_digest(ttnn, self.mesh_device, tensors)
            if not prefix_snap.digests_equal(before, after):
                raise RuntimeError(f"{prefix_snap.ENV_RESUME} self-test FAILED: the resident scratch differs from the "
                                   "restored snapshot: refusing")
            shifted = before[2:] + before[:2]  # each layer's (rec, carry) against the next layer's
            if prefix_snap.digests_equal(shifted, after):
                raise RuntimeError(f"{prefix_snap.ENV_RESUME} self-test wrong-input branch PASSED: refusing")
            st["selftest"] = "exact"
            logger.info(f"QWEN36_PREFIX_SCRATCH_RESUME self-test exact: resident scratch == restored snapshot "
                        f"({len(dns)} GDN layers, rec_state + conv_carry); wrong-input branch refused")
        st["skipped"] += 1
        if st["skipped"] in (1, 10, 100) or st["skipped"] % 1000 == 0:
            logger.info(f"{prefix_snap.ENV_RESUME} stats: restores skipped={st['skipped']}")

    def _onestep_mid_snapshot(self, req_id, slot, prev):
        """QWEN36_PREFIX_ONESTEP: the GDN checkpoint of a head pass, read from its decode slot right after the slot copy
        (the same bytes the shipped head step's after-forward prefix_cache_snapshot reads). The scratch is bound
        during the loop, so the batched tensors come from `prev` (the saved decode bindings), in GDN layer order."""
        dn_states = [(rec_b, conv_b, B_b, dn.Nv, dn.Dk, dn.Dv) for dn, B_b, rec_b, conv_b, *_ in prev]
        # QWEN36_PREFIX_CKPT_DEFER: copy the slot into the staging pool now, read it after the step (runner). Only once
        # QWEN36_PREFIX_SCRATCH_RESUME's self-test has passed: that self-test needs this head's host snapshot.
        if (getattr(self, "_scratch_resume_stats", None) or {}).get("selftest") == "exact":
            ref = self._ckpt_stage(dn_states, int(slot))
            if ref is not None:
                self._prefix_mid_snaps[req_id] = ref
                return ref
        snap = self._prefix_cache_snapshot_fast([int(slot)], dn_states=dn_states)[0]
        self._prefix_mid_snaps[req_id] = snap
        return snap

    # ---- QWEN36_PREFIX_CKPT_DEFER (prefix_snap.py, lever L1) ---------------------------------------------------------
    def dram_ledger(self, tag):
        """QWEN36_DRAM_LEDGER=1 (diagnostic, no device work): one line of the device DRAM allocator view per bank
        (ttnn.get_memory_view): total / allocated / free / largest free block, MiB. Returns the view or None."""
        if _tt_public_getenv("QWEN36_DRAM_LEDGER", "0") != "1":
            return None
        try:
            v = ttnn.get_memory_view(self.mesh_device, ttnn.BufferType.DRAM)
        except Exception as e:  # noqa: BLE001 - a diagnostic never fails the boot
            logger.warning(f"QWEN36_DRAM_LEDGER {tag}: unavailable ({e!r})")
            return None
        mib = lambda b: b / 2**20
        logger.info(f"QWEN36_DRAM_LEDGER {tag}: banks={v.num_banks} per bank MiB total={mib(v.total_bytes_per_bank):.1f} "
                    f"alloc={mib(v.total_bytes_allocated_per_bank):.1f} free={mib(v.total_bytes_free_per_bank):.1f} "
                    f"largest_free={mib(v.largest_contiguous_bytes_free_per_bank):.1f}")
        return v

    def dram_ledger_tick(self):
        """QWEN36_DRAM_LEDGER: the ledger at the first served prefill (every trace captured, warmup done), then every
        QWEN36_DRAM_LEDGER_EVERY (256) prefill forwards; tracks the minimum largest-free block seen."""
        if _tt_public_getenv("QWEN36_DRAM_LEDGER", "0") != "1":
            return
        n = self._dram_ledger_n = getattr(self, "_dram_ledger_n", 0) + 1
        every = int(_tt_public_getenv("QWEN36_DRAM_LEDGER_EVERY", "256"))
        if n == 1 or n % every == 0:
            v = self.dram_ledger("post-warmup (first served prefill)" if n == 1 else f"serving prefill #{n}")
            if v is not None:
                lf = v.largest_contiguous_bytes_free_per_bank
                self._dram_ledger_min = min(getattr(self, "_dram_ledger_min", lf), lf)
                logger.info(f"QWEN36_DRAM_LEDGER min largest_free so far {self._dram_ledger_min / 2**20:.1f} MiB/bank")
                # fragmentation, not the total, killed w1 (8.0 MiB free per bank, largest block 3.75 < 5.0 asked):
                # an arm can require a floor on the largest free block (QWEN36_DRAM_LEDGER_MIN_LARGEST_MIB)
                floor = float(_tt_public_getenv("QWEN36_DRAM_LEDGER_MIN_LARGEST_MIB", "0") or 0)
                if floor and lf < floor * 2**20 and not getattr(self, "_dram_ledger_low", False):
                    self._dram_ledger_low = True
                    logger.warning(f"QWEN36_DRAM_LEDGER LOW HEADROOM: largest free block {lf / 2**20:.2f} MiB/bank "
                                   f"< {floor:.1f} MiB (total free {v.total_bytes_free_per_bank / 2**20:.2f})")

    def prefix_ckpt_pool_alloc(self):
        """Allocate the checkpoint staging pool. Called once before the first trace capture (warmup_model_prefill),
        with the batched decode GDN buffers bound. Halves the pool on an allocation failure; none fits -> lever off."""
        if self._ckpt_pool is not None or not prefix_snap.defer_enabled():
            return
        dns = self._gdn_layers()
        if any(dn.rec_state.dtype != ttnn.bfloat16 for dn in dns):
            raise RuntimeError(f"{prefix_snap.ENV_DEFER}=1 needs a bf16 recurrent state (QWEN35_GDN_STATE_BF16=1)")
        if any(c.dtype != ttnn.bfloat16 for dn in dns for c in dn.conv_states):
            raise RuntimeError(f"{prefix_snap.ENV_DEFER}=1 needs bf16 conv taps")
        layers = [(dn.Nv, dn.Dk, dn.Dv, dn.K, dn.qkv_dim_tp) for dn in dns]
        v0 = self.dram_ledger("before CKPT_DEFER pool")
        n, err = prefix_snap.defer_slots(), None
        while n >= 1:
            try:
                self._ckpt_pool = prefix_snap.StagePool(ttnn, self.mesh_device, self.num_devices, layers, n,
                                                        prefix_snap.pack_layers())
                break
            except Exception as e:  # device OOM: StagePool freed its partial allocation
                err, n = e, n // 2
        if self._ckpt_pool is None:
            logger.warning(f"QWEN36_PREFIX_CKPT_DEFER pool REFUSED (no DRAM for one entry: {err}); checkpoints are read "
                           f"in-step")
            return
        mib = self._ckpt_pool.bytes_per_entry * n / 2**20
        v1 = self.dram_ledger("after CKPT_DEFER pool")
        if v0 is not None and v1 is not None:
            used = (v1.total_bytes_allocated_per_bank - v0.total_bytes_allocated_per_bank) * v0.num_banks
            logger.info(f"QWEN36_DRAM_LEDGER CKPT_DEFER pool measured {used / 2**20:.1f} MiB/card "
                        f"(formula {mib:.1f} MiB/card)")
            formula = self._ckpt_pool.bytes_per_entry * n
            if abs(used - formula) > 0.02 * formula + 2**20:
                logger.warning(f"QWEN36_DRAM_LEDGER FORMULA MISMATCH: the pool took {used / 2**20:.1f} MiB/card, "
                               f"prefix_snap.entry_bytes says {formula / 2**20:.1f}")
            res = prefix_snap.reserved()
            if not res or used > res["kv_bytes"]:
                logger.warning(f"QWEN36_DRAM_LEDGER RESERVE SHORT: the pool took {used / 2**20:.1f} MiB/card, the KV "
                               f"sizer reserved {res.get('kv_bytes', 0) / 2**20:.1f} MiB/card")
            else:
                logger.info(f"QWEN36_DRAM_LEDGER reserve covers the pool: {used / 2**20:.1f} <= "
                            f"{res['kv_bytes'] / 2**20:.1f} MiB/card")
        logger.info(f"QWEN36_PREFIX_CKPT_DEFER engaged: GDN checkpoints staged on device, read back at the next engine "
                    f"step; pool {n} entries x {len(layers)} GDN layers = {mib:.1f} MiB per card"
                    + (f" (asked {prefix_snap.defer_slots()})" if n != prefix_snap.defer_slots() else "")
                    + (f" MUSTFAIL={prefix_snap.defer_mustfail()}" if prefix_snap.defer_mustfail() else ""))

    def _ckpt_stage(self, dn_states, slot):
        """Stage decode slot `slot` (dn_states: the batched GDN tensors) -> StagedRef, or None (lever off, the
        immediate path not yet self-tested on this boot, or the pool full: the caller reads now)."""
        pool = self._ckpt_pool
        if (pool is None or getattr(self, "_ckpt_pool_off", False)
                or (getattr(self, "_snap_fast_stats", None) or {}).get("selftest") != "exact"):
            return None
        st = self._ckpt_stats
        try:
            ref = pool.stage(dn_states, slot, wrong_slot=prefix_snap.defer_mustfail() == "wrongslot")
        except Exception as e:  # the entry went back to the free list; this checkpoint is read in-step
            logger.warning(f"QWEN36_PREFIX_CKPT_DEFER stage FAILED for slot {slot} ({type(e).__name__}: {e}); lever "
                           f"off for the rest of the boot, checkpoints read in-step")
            self._ckpt_pool_off = True
            return None
        if ref is None:
            st["immediate"] += 1
            return None
        if st["selftest"] == "pending" and not st["armed"]:
            # The served self-test: the first staged checkpoint of the boot is also read NOW (immediate packed path);
            # prefix_ckpt_resolve compares the two bitwise, with the wrong-input branch.
            ref.selftest_ref = prefix_snap.fast_snapshot(
                ttnn, self.mesh_device, self.num_devices, dn_states, [slot], pack=True,
                pack_layers=prefix_snap.pack_layers())[0]
            st["armed"] = True
        st["staged"] += 1
        return ref

    def prefix_ckpt_pending(self, items):
        return any(prefix_snap.is_staged(x) for x in items)

    def prefix_ckpt_resolve(self, items):
        """Replace every StagedRef in `items` (a list of snapshots / refs) by its host snapshot, in one packed read."""
        import time as _time

        refs = [x for x in items if prefix_snap.is_staged(x)]
        if not refs:
            return list(items)
        st = self._ckpt_stats
        t0 = _time.perf_counter()
        snaps = self._ckpt_pool.read(refs)
        dt = _time.perf_counter() - t0
        by_idx = {r.idx: s for r, s in zip(refs, snaps)}
        for r, snap in zip(refs, snaps):
            if r.selftest_ref is not None:
                ok, detail = prefix_snap.selftest_compare([snap], [r.selftest_ref])  # lists of per-slot snapshots
                if not ok:
                    raise RuntimeError(f"QWEN36_PREFIX_CKPT_DEFER self-test FAILED (staged vs in-step read of slot "
                                       f"{r.slot}: {detail}): refusing")
                st["selftest"] = "exact"
                logger.info(f"QWEN36_PREFIX_CKPT_DEFER self-test exact: staged checkpoint of slot {r.slot} read back "
                            f"after the step == the in-step packed read, bitwise; wrong-input branch refused")
            self._validate_gdn_snapshot(snap)
        st["resolved"] += len(refs)
        st["reads"] += 1
        st["read_s"] += dt
        if st["reads"] in (1, 10, 100) or st["reads"] % 500 == 0:
            logger.info(f"{prefix_snap.ENV_DEFER} stats: staged={st['staged']} immediate={st['immediate']} "
                        f"resolved={st['resolved']} reads={st['reads']} mean_read_ms="
                        f"{1000 * st['read_s'] / st['reads']:.1f} last_read_ms={1000 * dt:.1f} slots={len(refs)}")
        return [by_idx[x.idx] if prefix_snap.is_staged(x) else x for x in items]

    def prefill_paged_slots(self, token_ids_list, page_table, empty_slots, valid_lens=None, prefix=None):
        """vLLM continuous-batching TP prefill: prefill each new request into ITS decode slot.

        The per-slot analogue of prefill_paged_peruser for online serving. Under vLLM a new
        request is prefilled while the other decode slots are live, so each user's B=1 state must
        land in row empty_slots[u] WITHOUT disturbing the others (GDN state is a fixed [B,...]
        buffer indexed by slot, not paged). Mirrors prefill_traced_bucket_batched's machinery —
        bind a B=1 GDN scratch, run the trace-safe pre-warmed masked-bucket prefill per user,
        snapshot its B=1 state — but writes each snapshot into its slot via write_slot (preserving
        the live rows) instead of assembling the whole batch. Attention fills each user's physical
        blocks via its page-table row (the same blocks decode reads via the decode page table).

        token_ids_list: list of N torch.Tensor [1, T_u].
        page_table:     torch.Tensor [N, max_blocks] int32 — row u = request u's blocks.
        empty_slots:    list of N ints — request u's persistent decode slot.
        valid_lens:     optional list of N real token counts (defaults to each T_u).
        Returns:        list of N host torch logits [1, 1, vocab_size] (one per request, in call order).

        Call allocate_kv_caches(batch_size=B) + the batched warmup first. Any prompt length is served:
        prefill_traced_chunked chunks long prompts via pre-warmed programs (no post-park compile).

        prefix (QWEN36_PREFIX_CACHING, the S12 prefix arm; None everywhere else): per-request lists
          starts      absolute start of this step's chunk (0, a prefix-cache hit, or a chunked-prefill continuation;
                      always a multiple of the block, hence of the 2048 chunk)
          states      the host GDN snapshot to resume from when start > 0 (else None): restored into the B=1 scratch,
                      conv carry included, before prefill_traced_chunked(start=start), which does not reset it
          hits        the request's prefix-cache hit (its first chunk's start): the MTP head skips hits // block
                      page-table blocks (host.pt_shift) and the warm span starts at or past it
          totals      the request's token count (the final chunk's end); finals: this chunk ends at totals
          next_tokens the prompt token at this chunk's end (a head step's last warm pair), -1 on a final chunk
          req_ids     the request ids (the warm span's cross-step key)
        Everything the shipped path does (slot-copy trace, MTP warm, deferred finish, tail no-sync) runs unchanged.
        """
        assert self.num_devices > 1, "prefill_paged_slots is the TP (num_devices>1) path"
        N = len(token_ids_list)
        if prefix is not None:
            for _k in ("starts", "states", "hits", "totals", "finals", "next_tokens", "req_ids"):
                if len(prefix[_k]) != N:
                    raise ValueError(f"prefill_paged_slots: prefix[{_k!r}] has {len(prefix[_k])} rows, expected {N}")
        _mtp_b = getattr(self, "_mtp_batch", None)
        _bs = get_block_size(self._paged_kv_caches)
        assert len(empty_slots) == N, "one slot per request"
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt.shape[0] == N, "page_table must have one row per request"
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_states = [layer.attention for layer in self.layers if not layer.is_full_attention]

        # Bind the persistent B=1 GDN prefill scratch: prefill runs B=1 and its per-sequence reset
        # would otherwise zero the batched [B,...] decode buffer (clobbering the live rows). This is
        # the SAME scratch whose addresses the chunk-prefill trace baked at warmup, so the traced
        # long-prompt path replays correctly; short prompts use the masked bucket on the same scratch.
        prev = self._bind_gdn_prefill_scratch(keep_resident=True)
        host_logits = []
        per_user_rec = []
        per_user_conv = []
        _slot_dev_copy = self._gdn_slot_device_copy_enabled()
        _slot_trace = bool(self._slot_copy_traces) and _slot_dev_copy
        # QWEN36_PREFIX_SCRATCH_RESUME (prefix_snap.py): run the row whose state is still resident in the scratch
        # first and skip its host restore. Reordering needs the device slot copy (the host-snapshot fallback below
        # indexes per_user_* by loop position); without it only a resident row 0 benefits.
        _sr_on = prefix is not None and prefix_snap.scratch_resume_enabled()
        _resident = getattr(self, "_scratch_resident", None)
        self._scratch_resident = None
        _order, _sr_row = list(range(N)), None
        if _sr_on:
            _order, _sr_row = prefix_snap.resident_order(N, prefix["req_ids"], prefix["starts"], prefix["hits"],
                                                         _resident)
            if _sr_row is not None and _sr_row != 0 and not _slot_dev_copy:
                _order, _sr_row = list(range(N)), None
        _host_logits_by_row = [None] * N
        # QWEN36_PREFIX_ONESTEP (vllm_tt_plugin/prefix_onestep.py): a row with mids[u] > 0 is ONE scheduler chunk
        # [start, end) that crosses its prompt's last block boundary mid. It runs as two passes on the B=1 scratch,
        # exactly the two align steps of the shipped path back to back: head [start, mid) (non-final: slot copy, MTP warm
        # with the prompt token at mid), the GDN checkpoint read back from its decode slot (stored for PrefixCache), then
        # tail [mid, end) from the still-resident scratch (QWEN36_PREFIX_SCRATCH_RESUME's identity, self-tested there).
        _mids = [0] * N if prefix is None else [int(x) for x in prefix.get("mids", [0] * N)]
        if any(_mids):
            if not (_slot_dev_copy and _sr_on and prefix_snap.enabled()):
                raise ValueError("QWEN36_PREFIX_ONESTEP needs QWEN36_GDN_SLOT_DEVICE_COPY, QWEN36_PREFIX_SCRATCH_RESUME "
                                 "and QWEN36_PREFIX_SNAP_FAST")
            if self._head_defer and not self._ph.defer_finish:
                raise ValueError("QWEN36_PREFIX_HEAD_DEFER needs QWEN36_PREFILL_DEFER_FINISH")
            if not getattr(self, "_onestep_logged", False):
                self._onestep_logged = True
                if self._head_defer:
                    logger.info("QWEN36_PREFIX_HEAD_DEFER engaged: an ONESTEP head row's MTP warm finish runs after its "
                                "slot copy + checkpoint, right before its tail pass's begin_prefix")
                logger.info("QWEN36_PREFIX_ONESTEP engaged: in-step GDN checkpoint at the prompt's last block boundary "
                            "(head pass, slot copy, checkpoint readback, tail pass from the resident scratch)")
        self._prefix_mid_snaps = {}
        _passes = []
        for u in _order:
            if _mids[u]:
                _passes += [(u, "head"), (u, "tail")]
            else:
                _passes.append((u, None))
        _ht_own = self._ht_begin()  # QWEN36_PREFILL_HOST_TIMING (host clocks only, never syncs)
        self._ht_mark("bind")
        if self._host_timer is not None:
            self._host_timer.info["users"] = N  # phases accumulate over the N users (not additive per user)
            self._host_timer.info["prefix_starts"] = ",".join(str(int(x)) for x in
                                                            ([0] * N if prefix is None else prefix["starts"]))
        self.ph_flush_deferred()  # QWEN36_PREFILL_DEFER_FINISH: a parked finish always runs before new prefill work
        self._ph_req = True  # pp/host levers act only inside this request-time loop
        try:
            for u, _seg in _passes:
                self._scratch_resident = None  # this row is about to write the scratch
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"request {u}: token_ids must be [1, T_u]"
                actual = int(valid_lens[u]) if valid_lens is not None else toks.shape[1]
                assert actual >= 1, f"request {u}: empty prompt (actual_len={actual})"
                _mid = _mids[u]
                if _seg is not None and not (int(prefix["starts"][u]) < _mid < actual and _mid % _bs == 0
                                             and bool(prefix["finals"][u])):
                    raise ValueError(f"QWEN36_PREFIX_ONESTEP row {u}: mid {_mid} start {prefix['starts'][u]} "
                                     f"end {actual} final {prefix['finals'][u]} (block {_bs})")
                if _seg == "head":
                    actual = _mid
                # QWEN36_PREFILL_PHASE_TIMING=1: one "engaged" line per request (no-op when unset).
                _pt_own = self._pt_begin()
                _mw = self._mtp_warm()
                _start, _pf_final = 0, True
                if prefix is not None:
                    _start = int(prefix["starts"][u]) if _seg != "tail" else _mid
                    _hit, _total = int(prefix["hits"][u]), int(prefix["totals"][u])
                    _pf_final = bool(prefix["finals"][u]) and _seg != "head"
                    if _start % _bs or _hit % _bs or not 0 <= _hit <= _start < actual <= _total:
                        raise ValueError(f"prefix row {u}: hit {_hit} start {_start} end {actual} total {_total} "
                                         f"(block {_bs})")
                    if _pf_final != (actual == _total):
                        raise ValueError(f"prefix row {u}: final={_pf_final} but end {actual} vs total {_total}")
                    if _mtp_b is not None:
                        _mtp_b.host.set_pt_shift(int(empty_slots[u]), _hit // _bs)
                if _seg == "tail" and self._ph_deferred is not None:
                    # QWEN36_PREFIX_HEAD_DEFER: the head pass's parked warm finish ran behind its slot copy + checkpoint
                    # staging; it must run before this begin_prefix, which pops the head's issued passes.
                    self.ph_flush_deferred()
                    self._ht_mark("mtp_head_flush")
                if _mw is not None and prefix is not None:
                    _mw.begin_prefix(prefix["req_ids"][u], _total, _hit, _pf_final)
                elif _mw is not None:
                    # QWEN36_MTP_WARM_TAIL: the segments below report their hidden rows. The prompt and page-table row
                    # are read before the logits only by QWEN36_MTP_WARM_TAIL_SPAN_EARLY (an attribute test; no-op off).
                    if getattr(_mw, "early", False):
                        _mw.begin(actual, prompt_ids=toks[0, :actual].tolist(), page_table_row=pt[u : u + 1])
                    else:
                        _mw.begin(actual)
                try:
                    # Trace-safe prefill into the B=1 scratch: prefill_traced_chunked runs short prompts in
                    # one masked-bucket forward and chunks longer ones; GDN state carries + is snapshotted below.
                    # QWEN36_GDN_SLOT_COPY_TRACE with the conv finalize folded in: the slot trace below writes the
                    # decode conv window from conv_carry, so prefill_traced_chunked skips its eager finalize here
                    # (and ONLY here: every other caller still finalizes). Logits do not read conv_states.
                    self._gdn_conv_finalize_skip = _slot_trace and self._slot_copy_fold
                    # QWEN36_PREFILL_TAIL_NOSYNC: no drain before the tail, no sync after its replay; the host inputs
                    # are held (self._ph_refs) until the blocking readback below. Needs the span warm pass (it anchors
                    # its own writes) when the MTP warm tail is on.
                    self._ph_nosync_on = self._ph.tail_nosync and (_mw is None or bool(getattr(_mw, "span", 0)))
                    if _seg == "tail":
                        # QWEN36_PREFIX_ONESTEP: the head pass just left this request's state at `mid` in the scratch.
                        self._prefix_scratch_resume(self._prefix_mid_snaps[prefix["req_ids"][u]])
                        self._ht_mark("prefix_restore")
                        self._build_request_rope(toks[:, :actual], None)
                    elif _start > 0 and u == _sr_row:
                        # QWEN36_PREFIX_SCRATCH_RESUME: the scratch still holds this request's state at `start`.
                        self._prefix_scratch_resume(prefix["states"][u])
                        self._ht_mark("prefix_restore")
                        self._build_request_rope(toks[:, :actual], None)
                    elif _start > 0:
                        # QWEN36_PREFIX_CACHING resume: the snapshot (recurrent state + conv taps) into the bound B=1
                        # scratch and its conv carry, then the SAME traced chunk path from `start` (no reset).
                        # HOST_TIMING: this host->device restore used to be charged to the next mark, rope_stage
                        # (an internal record's B8 "+0.88 s rope_stage" is 7 rows of this restore).
                        self._prefix_restore_scratch(prefix["states"][u])
                        self._ht_mark("prefix_restore")
                        self._build_request_rope(toks[:, :actual], None)
                    elif prefix is not None and prefix["states"][u] is not None:
                        raise ValueError(f"prefix row {u}: a snapshot was supplied for a cold (start 0) chunk")
                    try:
                        lg = self.prefill_traced_chunked(toks[:, :actual], pt[u : u + 1], actual_len=actual,
                                                         start=_start)
                    finally:
                        self._gdn_conv_finalize_skip = False
                        self._ph_nosync_on = False
                    self._ht_mark("prefill_body")
                    _hl = self._prefill_host_logits(lg, comp)
                    if _seg != "head":  # a head pass's logits only feed its MTP warm finish
                        host_logits.append(_hl)
                        _host_logits_by_row[u] = _hl
                    self._ph_refs.release()  # the blocking readback drained every DMA these held
                    ttnn.deallocate(lg)
                    self._pt_mark("logits_read")
                    self._ht_mark("readback")
                    if _mw is not None and prefix is not None:
                        # QWEN36_PREFIX_CACHING: the prefix warm span (begin_prefix). The head step's pairs are all
                        # prompt tokens (its last row pairs with the next prompt token); the fill table is the
                        # slot's SHIFTED page-table row, so no drafter write lands in a shared cached block.
                        from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mbh_pf

                        _pt_m = _mbh_pf.shifted_page_table(pt[u : u + 1], [_hit // _bs])
                        if _pf_final:
                            _pids = toks[0, :actual]
                        else:
                            _pids = torch.zeros(_total, dtype=toks.dtype)
                            _pids[:actual] = toks[0, :actual]
                            _pids[actual] = (int(token_ids_list[u][0, actual]) if _seg == "head"
                                             else int(prefix["next_tokens"][u]))
                        if (_pf_final or (_seg == "head" and self._head_defer)) and self._ph.defer_finish:
                            self._ph_defer_finish(_mw, _pids, _hl, int(empty_slots[u]), _pt_m)
                            self._ht_mark("mtp_defer")
                        else:
                            _mw.finish(_pids.tolist(), _hl, int(empty_slots[u]), _pt_m)
                            self._pt_mark("mtp_warm")
                            self._ht_mark("mtp_finish")
                    elif _mw is not None and self._ph.defer_finish:
                        # QWEN36_PREFILL_DEFER_FINISH: this user's warm finish is parked. An inner user's runs from the
                        # next user's first warm gather (_mtp_warm_seg / _mtp_warm_span_seg flush first), i.e. right
                        # after that user's first replay is enqueued, so its host work hides behind the device. The
                        # last user's runs at the next model entry, after this step published its first tokens.
                        self._ph_defer_finish(_mw, toks[0, :actual], _hl, int(empty_slots[u]), pt[u : u + 1])
                        self._ht_mark("mtp_defer")
                    elif _mw is not None:
                        # QWEN36_MTP_WARM_TAIL: KV-only MTP pass over the prompt tail, before the next user's replay
                        # overwrites the segment buffers; seeds the slot's MTP origin. Never changes the logits.
                        _mw.finish(toks[0, :actual].tolist(), _hl, int(empty_slots[u]), pt[u : u + 1])
                        self._pt_mark("mtp_warm")
                        self._ht_mark("mtp_finish")
                    if _slot_trace:
                        # QWEN36_GDN_SLOT_COPY_TRACE: one execute_trace replaces ~1.2k eager ops (c117: slot_copy
                        # 112-115 ms + conv_finalize 31 ms of host dispatch per request).
                        self._replay_gdn_slot_copy_trace(int(empty_slots[u]), prev)
                        self._pt_mark("slot_copy")
                        self._ht_mark("slot_copy")
                        if _seg == "head":
                            self._onestep_mid_snapshot(prefix["req_ids"][u], int(empty_slots[u]), prev)
                            self._ht_mark("mid_ckpt")
                        if _sr_on and not _pf_final:
                            self._scratch_resident = (prefix["req_ids"][u], actual)
                        continue
                    if _slot_dev_copy:
                        # QWEN36_GDN_SLOT_DEVICE_COPY: copy the B=1 scratch straight into the decode
                        # slot on device, now, before the next user's reset overwrites it.
                        self._write_gdn_slot_from_scratch(int(empty_slots[u]), prev)
                        self._pt_mark("slot_copy")
                        self._ht_mark("slot_copy")
                        if _seg == "head":
                            self._onestep_mid_snapshot(prefix["req_ids"][u], int(empty_slots[u]), prev)
                            self._ht_mark("mid_ckpt")
                        if _sr_on and not _pf_final:
                            self._scratch_resident = (prefix["req_ids"][u], actual)
                        continue
                    # Snapshot this user's B=1 scratch state (host round trip — the next user's reset
                    # overwrites the scratch in place; see prefill_traced_bucket_batched for why not clone).
                    per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_states])
                    per_user_conv.append(
                        [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_states]
                    )
                    self._pt_mark("slot_snapshot")
                finally:
                    if _mw is not None:
                        _mw.abort()  # a request that raised (or finished) leaves no half-built window behind
                    self._pt_end(_pt_own)
        finally:
            self._ph_req = False
            if len(self._ph_refs):
                # QWEN36_PREFILL_TAIL_NOSYNC: a request that raised before its readback; drain before freeing inputs.
                ttnn.synchronize_device(self.device)
                self._ph_refs.release()
            # Always rebind the batched decode buffers (a mid-loop assert must not leave GDN on scratch).
            # Does NOT free the scratch — it persists for the next request and keeps the trace valid.
            self._unbind_gdn_prefill_scratch(prev)
            self._ht_mark("unbind")
            self._ht_end(_ht_own)

        # Write each user's snapshot into its decode slot, preserving the other live rows.
        # (Already done on device inside the loop under QWEN36_GDN_SLOT_DEVICE_COPY.)
        if not _slot_dev_copy:
            for u in range(N):
                self._write_gdn_slot(int(empty_slots[u]), per_user_rec[u], per_user_conv[u])
        if _order != list(range(N)):
            return _host_logits_by_row  # QWEN36_PREFIX_SCRATCH_RESUME ran the rows out of order: call order back
        return host_logits

    # --- QB2 backports (docs/QB2-BACKPORT-PLAN.md), both default OFF --------------------------------
    #
    # QWEN36_PREFILL_LOGITS_FAST=1 (upstream tt-metal#56119 of the same name): the prefill logits are a
    # replicated TILE [1,1,vocab] tensor. The default read concatenates BOTH replicas (2 x 32 padded
    # rows x 248320 x bf16 = ~32 MB over card 0's PCIe) and untilizes on the 4-core host, to keep one
    # row. Untilize on device and read one replica instead: ~0.5 MB, identical values.
    #
    # QWEN36_GDN_SLOT_DEVICE_COPY=1 (upstream #56119 of the same name): after each user's B=1 prefill,
    # prefill_paged_slots round-trips all 48 GDN layers' recurrent state (fp32 [2,24,128,128]) and conv
    # taps through the host (~0.2 GB down + ~0.2 GB up, host tilize/untilize) and then re-uploads them.
    # Instead clone the scratch on device and write it into the decode slot right away, before the next
    # user's replay. Why this is safe where ttnn.clone() snapshots were not
    # (prefill_traced_bucket_batched): those clones stayed alive ACROSS the next execute_trace, whose
    # baked intermediates can alias fresh allocations. Here every temporary is consumed by write_slot
    # and freed before the next replay is enqueued, the same lifetime as the default path's
    # post-loop from_torch uploads, and the command queue is in order.
    def _prefill_logits_fast_enabled(self):
        on = _tt_public_getenv("QWEN36_PREFILL_LOGITS_FAST") == "1" and self.num_devices > 1
        if on and not getattr(self, "_logits_fast_banner", False):
            self._logits_fast_banner = True
            logger.info(
                f"QWEN36_PREFILL_LOGITS_FAST engaged: prefill logits untilized on device, one replica read "
                f"(vocab={self.args.vocab_size}, mesh={tuple(self.mesh_device.shape)})"
            )
        return on

    def _prefill_host_logits(self, lg, comp):
        """Host [1,1,vocab] float logits of the (replicated) prefill logits `lg`."""
        if self._prefill_logits_fast_enabled():
            rm = lg if lg.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(lg, ttnn.ROW_MAJOR_LAYOUT)
            out = ttnn.to_torch(ttnn.get_device_tensors(rm)[0])
            if rm is not lg:
                ttnn.deallocate(rm)
        else:
            out = ttnn.to_torch(lg, mesh_composer=comp)
        return out.reshape(-1, self.args.vocab_size)[:1].float().view(1, 1, -1)

    def _gdn_slot_device_copy_enabled(self):
        on = _tt_public_getenv("QWEN36_GDN_SLOT_DEVICE_COPY") == "1" and self.num_devices > 1
        if on and not getattr(self, "_slot_dev_copy_banner", False):
            self._slot_dev_copy_banner = True
            n = sum(1 for layer in self.layers if not layer.is_full_attention)
            logger.info(
                f"QWEN36_GDN_SLOT_DEVICE_COPY engaged: prefill GDN state -> decode slot copied on device "
                f"({n} GDN layers, no host round trip)"
            )
        if on:
            self._slot_dev_copy_count = getattr(self, "_slot_dev_copy_count", 0) + 1
        return on

    def _write_gdn_slot_from_scratch(self, slot, prev):
        """Device-side twin of the host snapshot + _write_gdn_slot: for every GDN layer, clone the bound
        B=1 scratch (rec_state, conv taps) and write the clones into `slot` of the batched decode buffers
        saved in `prev` (from _bind_gdn_prefill_scratch). Leaves the scratch bound on return."""
        for dn, B_b, rec_b, conv_b, carry_b, zero0_b, stable_b in prev:
            s_B, s_rec, s_conv = dn.B, dn.rec_state, dn.conv_states
            rec = ttnn.clone(s_rec, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            convs = [ttnn.clone(c, memory_config=ttnn.DRAM_MEMORY_CONFIG) for c in s_conv]
            dn.B, dn.rec_state, dn.conv_states = B_b, rec_b, conv_b
            try:
                dn.write_slot(slot, rec, convs)  # consumes rec / convs
            finally:
                dn.B, dn.rec_state, dn.conv_states = s_B, s_rec, s_conv

    # --- QWEN36_GDN_SLOT_COPY_TRACE=1 (c121, boot time; requires QWEN36_GDN_SLOT_DEVICE_COPY=1) -------------------
    #
    # c117 ([source validation record]) split the per-request prefill tail on the
    # served TP=2 path: slot_copy 112-115 ms (_write_gdn_slot_from_scratch: 48 layers x clone + slice/concat/copy)
    # and conv_finalize 31 ms (48 x finalize_conv_window), both host op dispatch with tiny device time, paid once
    # per request whatever its length. This lever captures, at warmup, ONE trace per decode slot s that writes the
    # bound B=1 scratch into row s of the batched decode buffers (TPGatedDeltaNet.copy_scratch_to_slot), with the
    # conv finalize folded in when QWEN36_GDN_CONV_WINDOW_ONCE is on (it must precede the copy: the copy reads the
    # finalized window, so the fold writes the window from conv_carry straight into the row). Request time is then
    # one execute_trace per request.
    #
    # Trace safety:
    #  * Addresses. Every buffer the body reads or writes is persistent and never reallocated: the B=1 scratch
    #    (_gdn_prefill_scratch: rec_state, conv_states, conv_carry, _zero_conv0) is allocated once at warmup and
    #    already baked into the chunk + bucket traces; the batched rec_state / conv_states are the decode trace's
    #    fixed-address buffers (_stable_state: every writer, remap_slots included, copies in place). The replay
    #    checks by identity that the bound tensors are still the ones captured and fails closed otherwise.
    #  * Allocations. No clone: the scratch is read in place (_write_index_keep does not consume it). The only
    #    allocations are the slice/concat temporaries of the eager path, freed inside the body, so no trace-made
    #    buffer outlives the trace. Like every other trace here, its intermediates alias the free pool; nothing is
    #    allocated between the prefill forward and this replay that must survive it (lg is freed first).
    #  * Dtypes. Scratch and batched buffers come from the same reset_state (fp32 rec unless QWEN35_GDN_STATE_BF16,
    #    bf16 conv taps and carry); a mismatch is typecast inside the body (fixed shapes, trace safe).
    #  * Programs. The warm pass runs every slot's body eagerly BEFORE the chunk trace parks (post-park compile
    #    clobbers a parked trace, #48536); captures come after the chunk and bucket traces.
    # Falsifier QWEN36_GDN_SLOT_COPY_TRACE_MUSTFAIL=wrongslot replays slot (s+1)%B's trace: equiv MUST FAIL.

    def _slot_copy_identity(self, prev):
        return [
            (dn, dn.rec_state, tuple(dn.conv_states), dn.conv_carry, dn._zero_conv0, rec_b, tuple(conv_b))
            for dn, _B_b, rec_b, conv_b, _carry_b, _zero0_b, _stable_b in prev
        ]

    def _slot_copy_identity_matches(self, prev):
        cap = self._slot_copy_captured
        if cap is None or prev is None or len(cap) != len(prev):
            return False
        for c, n in zip(cap, self._slot_copy_identity(prev)):
            if c[0] is not n[0] or c[1] is not n[1] or c[3] is not n[3] or c[4] is not n[4] or c[5] is not n[5]:
                return False
            if len(c[2]) != len(n[2]) or len(c[6]) != len(n[6]):
                return False
            if any(a is not b for a, b in zip(c[2], n[2])) or any(a is not b for a, b in zip(c[6], n[6])):
                return False
        return True

    def _gdn_slot_copy_body(self, slot, prev):
        """The traced body for decode slot `slot` (also the eager warm pass): all GDN layers, scratch -> row."""
        for dn, _B_b, rec_b, conv_b, _carry_b, _zero0_b, _stable_b in prev:
            dn.copy_scratch_to_slot(slot, rec_b, conv_b, self._slot_copy_fold)

    def _warm_gdn_slot_copy_traces(self, device):
        """Eager pass of every slot's body (compiles the per-slot slice/concat programs) with the scratch bound.
        Called from _capture_prefill_trace_chunked_tp before the chunk trace parks. Writes the zeroed scratch into
        every decode row (boot time, before any request). No-op unless QWEN36_GDN_SLOT_COPY_TRACE=1."""
        if not self._slot_copy_trace_on:
            return
        prev = self._gdn_bound_batched
        if prev is None and int(getattr(self.args, "max_batch_size", 0) or 0) <= 1:
            # B=1 serving has no batched GDN slots, so there is nothing to copy: an expected no-op, not
            # the "set but NOT active" misconfiguration (whose banner arms list as [expect].absent and
            # which refused every B=1 boot of the tail arms, c145).
            logger.info("QWEN36_GDN_SLOT_COPY_TRACE idle at max_batch_size=1: single-user decode has no slot copy")
            return
        if prev is None or self.num_devices <= 1:
            logger.warning(
                "QWEN36_GDN_SLOT_COPY_TRACE set but NOT active: no batched GDN decode buffers bound at the "
                "chunk-trace warmup (TP batched serving only); the eager slot copy stays"
            )
            return
        nslots = {int(B_b) for _dn, B_b, *_rest in prev}
        assert len(nslots) == 1, f"GDN layers disagree on the decode batch: {sorted(nslots)}"
        nslots = nslots.pop()
        assert not (self._slot_copy_mustfail == "wrongslot" and nslots < 2), "wrongslot must-fail needs >= 2 slots"
        for dn, B_b, rec_b, conv_b, _carry_b, _zero0_b, _stable_b in prev:
            assert rec_b is not None and conv_b is not None, "batched GDN state not allocated"
            assert int(rec_b.shape[0]) == B_b and all(int(c.shape[1]) == B_b for c in conv_b), (
                f"batched GDN buffers are not [{B_b},...]"
            )
            assert dn.B == 1 and dn._stable_state and dn.rec_state is not rec_b, "B=1 prefill scratch not bound"
        self._slot_copy_fold = bool(getattr(self, "_gdn_conv_window_once", False))
        if self._slot_copy_fold:
            assert all(dn._conv_window_once for dn in self._gdn_layers()), "conv-window-once gate mismatch"
        self._reset_gdn_state_for_new_sequence()
        for s in range(nslots):
            self._gdn_slot_copy_body(s, prev)
        ttnn.synchronize_device(device)
        self._slot_copy_nslots = nslots
        self._slot_copy_captured = self._slot_copy_identity(prev)
        logger.info(f"[prefill] GDN slot-copy bodies warmed for {nslots} decode slots (fold={self._slot_copy_fold})")

    def capture_gdn_slot_copy_traces(self, device):
        """Capture one slot-copy trace per decode slot (after the chunk + bucket traces, so they keep their
        trace-region bytes first). Fails closed if the warm pass never ran against these same buffers."""
        if not self._slot_copy_trace_on or self._slot_copy_captured is None:
            return
        prev = self._gdn_bound_batched
        assert self._slot_copy_identity_matches(prev), "slot-copy capture: GDN bindings changed since the warm pass"
        before = self._trace_region_bytes(device)
        for s in range(self._slot_copy_nslots):
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            self._gdn_slot_copy_body(s, prev)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
            self._slot_copy_traces[s] = trace_id
        after = self._trace_region_bytes(device)
        delta = f"+{(after - before) / (1 << 20):.1f} MiB" if None not in (before, after) else "n/a"
        self._slot_copy_replays = 0
        logger.info(
            f"QWEN36_GDN_SLOT_COPY_TRACE engaged: {len(self._slot_copy_traces)} slot traces, "
            f"conv_finalize={'folded' if self._slot_copy_fold else 'eager'} "
            f"({len(prev)} GDN layers, TRACE region {delta})"
        )
        if self._slot_copy_mustfail:
            logger.warning(
                f"QWEN36_GDN_SLOT_COPY_TRACE_MUSTFAIL engaged: {self._slot_copy_mustfail} -- each request's GDN "
                f"state is written into decode slot (s+1)%{self._slot_copy_nslots}; a correctness gate MUST FAIL"
            )

    def _replay_gdn_slot_copy_trace(self, slot, prev):
        """Request-time twin of _write_gdn_slot_from_scratch: one non-blocking execute_trace (in-order CQ, so it
        runs after this request's prefill and before the next request's in-place scratch reset)."""
        if not self._slot_copy_identity_matches(prev):
            raise RuntimeError(
                "QWEN36_GDN_SLOT_COPY_TRACE: the bound GDN scratch / batched decode buffers are not the tensors the "
                "slot traces baked (reallocated after warmup?); refusing to replay into stale addresses"
            )
        n = self._slot_copy_nslots
        assert 0 <= slot < n, f"slot {slot} out of range [0,{n})"
        s = (slot + 1) % n if self._slot_copy_mustfail == "wrongslot" else slot
        ttnn.execute_trace(self.device, self._slot_copy_traces[s], cq_id=0, blocking=False)
        self._slot_copy_replays += 1
        if self._slot_copy_replays == 1:
            logger.info(f"QWEN36_GDN_SLOT_COPY_TRACE first replay: slot {slot} -> trace {s}")

    def _write_gdn_slot(self, slot, rec_snap, conv_snap):
        """Upload one request's B=1 GDN state snapshot (host torch, per GDN layer) and write it
        into decode `slot` of the batched buffers via TPGatedDeltaNet.write_slot (preserving the
        other live rows). Shapes/mappers mirror _assemble_per_user_gdn (mesh dim 0 = devices).

        rec_snap[li]:  host [num_devices, Nv, Dk, Dv]; conv_snap[li]: list of K host [num_devices, 1, D].
        """
        mapper = ttnn.ShardTensorToMesh(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        for li, dn in enumerate(dn_layers):
            rec = ttnn.from_torch(
                rec_snap[li],
                dtype=dn.rec_state.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=mapper,
            )
            convs = [
                ttnn.from_torch(
                    conv_snap[li][m],
                    dtype=dn.conv_states[m].dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.mesh_device,
                    mesh_mapper=mapper,
                )
                for m in range(dn.K)
            ]
            dn.write_slot(slot, rec, convs)

    def _remap_gdn_slots(self, remap):
        """Apply a vLLM batch-condense slot_remap to every GDN layer's batched decode state
        (device-side; slot i takes the state at slot remap[i]). Mirrors seed_manager.apply_slot_remap
        for GDN's per-slot recurrent+conv state, which the plugin's slot_remap does not itself move.
        No-op for an identity remap."""
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.remap_slots(remap)

    def prefill_chunked_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user LONG-prefill (TP, eager). Runs the single-user chunk-outer path
        (prefill_traced_chunked) for each user into a B=1 GDN scratch, then stitches each user's
        final GDN state into row u of the batched [B,...] decode buffers.

        Handles ANY prompt length with exact valid_len masking (no last-token padding through the
        GDN recurrence): short -> masked bucket; long -> chunk-outer + masked tail. The long-prompt
        counterpart to prefill_traced_bucket_batched (full-bucket-only) and prefill_paged_peruser
        (single-pass). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
                         IMPORTANT: max_blocks_per_seq MUST be a multiple of 8. The chunked SDPA
                         reads each row as a ROW_MAJOR int32 stick requiring stick_size
                         (width * 4 bytes) % 32 == 0, i.e. width % 8 == 0. A misaligned width
                         makes the SDPA read the wrong KV.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab] (replicated; at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_chunked_peruser is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        # The chunked path keys its chunk math on _chunked_chunk_size (default 2048); a parked
        # bucket trace leaves it at 128, breaking num_full/tail sizing. Require it released first.
        assert getattr(self, "_bucket_trace_id", None) is None, (
            "release the bucket prefill trace before prefill_chunked_peruser " "(_chunked_chunk_size would be wrong)"
        )
        assert self._chunked_chunk_size in (None, prefill_chunk_size()), (
            f"prefill_chunked_peruser expects the {prefill_chunk_size()}-token chunk; got _chunked_chunk_size="
            f"{self._chunked_chunk_size}"
        )

        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]

        # Swap every GDN layer to a B=1 scratch (per-user path is B=1); assembled into the batched
        # buffers after the loop. prev holds the batched bindings for restore.
        prev = self._alloc_gdn_scratch_b1()
        host_logits = []  # torch [1, 1, vocab] (one replica) per user
        # Snapshot each user's B=1 GDN state via a host to_torch round trip (NOT ttnn.clone() — see
        # prefill_traced_bucket_batched for why the device-side clone path is broken: it aliases
        # the captured trace's own baked-address intermediates and gets silently overwritten by
        # the next user's execute_trace()).
        per_user_rec = []  # per user, list over GDN layers of host rec_state snapshots
        per_user_conv = []  # per user, list over GDN layers of [list of K conv snapshots]
        try:
            for u in range(B):
                toks = token_ids_list[u]
                assert toks.shape[0] == 1, f"user {u}: token_ids must be [1, T_u]"
                vlen = valid_lens[u] if valid_lens is not None else toks.shape[1]
                assert 1 <= vlen <= toks.shape[1], f"user {u}: valid_len {vlen} not in [1, {toks.shape[1]}]"

                # Per-user long path (from scratch) into the B=1 scratch: carries state across
                # chunks, masks the tail exactly, writes user u's KV via the page-table row.
                lg = self.prefill_traced_chunked(toks, page_table_torch[u : u + 1].contiguous(), actual_len=vlen)
                # Read the logit to HOST immediately (the next prefill + post-loop assembly churn
                # device memory and would otherwise corrupt the returned tensor).
                ttnn.synchronize_device(self.device)
                host_logits.append(ttnn.to_torch(lg, mesh_composer=comp)[0:1].clone())  # [1,1,vocab] one replica
                ttnn.deallocate(lg)

                # Snapshot this user's B=1 GDN state for assembly after the loop (host round trip —
                # the B=1 scratch is reset IN PLACE for the next user). slot 0 of conv_states is
                # the zeroed shifted-out tap; only slots 1..K-1 carry state.
                per_user_rec.append([ttnn.to_torch(dn.rec_state, mesh_composer=comp) for dn in dn_layers])
                per_user_conv.append(
                    [[ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states] for dn in dn_layers]
                )
        finally:
            # Restore the batched [B,...] GDN decode buffers and free the B=1 scratch. The clones
            # are independent allocations, so freeing the scratch here does not touch them.
            self._restore_gdn_batched(prev)

        ttnn.synchronize_device(self.device)
        # Stitch the per-user states into row u of the (now-rebound) batched decode buffers.
        self._assemble_per_user_gdn(per_user_rec, per_user_conv)
        # Re-upload the per-user logits as stable device tensors (prefill_paged_peruser contract).
        return self._reupload_host_logits(host_logits)

    def _forward_prefill_chunk_eager(self, token_slice, chunk_start, page_table):
        """Eager final partial-chunk prefill (< chunk_size). GDN zero-pads to 128-multiple internally
        (not bucket padding). Returns hidden [1,T_tail_padded,hidden_size]."""
        T_tail = token_slice.shape[1]
        block_size = get_block_size(self._paged_kv_caches)  # was 64; 2048 on the prefix arm
        tok = ttnn.from_torch(
            token_slice.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, T_tail)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = math.ceil((chunk_start + T_tail) / block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                    **self._xadd_kw(_li),
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, **self._xadd_kw(_li))
            ttnn.deallocate(x)
            x = x_new
        return x

    # Fixed buckets for masked tail/short prefill. Lengths round up here -> bounded compile set.
    # All 128-multiples (GDN sub-chunk). Masked GDN in DRAM avoids L1 clash at bucket 512.
    # Diverges from get_padded_prefill_len: 256/512 for short TTFT; GDN needs exact valid_len mask.
    _PREFILL_MASK_BUCKETS = PREFILL_MASK_BUCKETS

    @classmethod
    def _mask_bucket_for(cls, length):
        """Smallest fixed bucket >= length (falls back to the next 128-multiple)."""
        return prefill_mask_bucket_for(length)

    def _forward_prefill_chunk_masked(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
    ):
        """Single masked fixed-bucket prefill forward over `bucket` positions.

        First valid_len tokens real; rest padded. Attn runs full bucket; GDN masks via valid_len.
        Returns hidden [1,bucket,hidden] or [1,1,bucket,hidden] (TP)."""
        if self.num_devices > 1:
            return self._forward_prefill_chunk_masked_tp(
                token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
            )
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        x = self.embd(tok)
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice: a fixed-shape ttnn.where over persistent buffers (compiled
        # at warmup; mask==0 -> identity, so text-only is unchanged). The caller stages the
        # buffers (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        cos, sin = self.rope.get_prefill_rot_mats(chunk_start, bucket)
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        # Fill K/V only for real blocks (ceil(valid_len/64)); padded writes would corrupt block 0.
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        # Flexible SDPA (device chunk_start): one program per bucket for any tail position.
        # Host-int chunk_start compiles per position and can clobber parked trace.
        csi_tensor = ttnn.from_torch(
            torch.tensor([chunk_start], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx_tensor=csi_tensor,
                    **self._xadd_kw(_li),
                )
            else:
                x_new = layer.forward(
                    x, mode="prefill", chunk_size=layer.attention.long_prefill_chunk_size, valid_len=valid_len,
                    **self._xadd_kw(_li),
                )
            ttnn.deallocate(x)
            x = x_new
        return x

    def _forward_prefill_chunk_masked_tp(
        self, token_buf, valid_len, chunk_start, page_table, bucket, flex_sdpa=True, vision_tokens=None
    ):
        """TP (num_devices>1) masked fixed-bucket single-chunk prefill forward.

        flex_sdpa=True: flexible chunked SDPA (serving). flex_sdpa=False: host-int path (debug).
        Fills K/V for real blocks only. Returns hidden [1,1,bucket,dim]."""
        block_size = get_block_size(self._paged_kv_caches)
        tok = ttnn.from_torch(
            token_buf.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(tok)
        # Trace-safe vision splice (fixed-shape where over the hidden-sharded persistent buffers,
        # sliced to bucket; identity when the mask is zero). The caller stages the buffers
        # (prefill_masked_bucket -> _set_vision_merge). No-op until a trace is captured.
        x = self._apply_vision_merge(x, length=bucket)
        # rope_tp cos/sin for absolute positions [chunk_start, chunk_start+bucket).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        cos = ttnn.from_torch(
            cos_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        sin = ttnn.from_torch(
            sin_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        full_pt = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        blk0 = chunk_start // block_size
        blkN = num_blocks_in_seq(chunk_start + valid_len, block_size)
        chunk_pt = ttnn.from_torch(
            page_table[:, blk0:blkN].contiguous(), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        csi_tensor = (
            ttnn.from_torch(
                torch.tensor([chunk_start], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
            )
            if flex_sdpa
            else None
        )
        for _li, layer in enumerate(self.layers):
            if layer.is_full_attention:
                x_new = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="prefill",
                    page_table=full_pt,
                    chunk_page_table=chunk_pt,
                    chunk_start_idx=chunk_start,
                    chunk_start_idx_tensor=csi_tensor,
                    valid_len=valid_len,  # unused by full attention
                    **self._xadd_kw(_li),
                )
            else:
                x_new = layer.forward(x, mode="prefill", chunk_size=self.args.gdn_chunk_size, valid_len=valid_len, **self._xadd_kw(_li))
            ttnn.deallocate(x)
            x = x_new
        # Deallocate per-chunk inputs; only hidden survives (avoids OOM in eager 64k loop).
        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(full_pt)
        ttnn.deallocate(chunk_pt)
        if csi_tensor is not None:
            ttnn.deallocate(csi_tensor)
        return x

    def prefill_masked_bucket(
        self,
        token_ids,
        page_table,
        actual_len,
        chunk_start=0,
        bucket=None,
        flex_sdpa=True,
        vision_tokens=None,
        vis_row_offset=0,
    ):
        """Masked fixed-bucket prefill for a segment of `actual_len` real tokens.

        Pads the segment up to a fixed bucket length, runs all layers ONCE, and masks the GDN
        recurrent + conv state so they reflect exactly `actual_len` real tokens — numerically
        equivalent to the eager exact-length path (prefill_paged) but using one of only a few
        bucket-sized programs instead of compiling a fresh program per prompt length. That
        bounded program set is what makes warmup able to compile every code path before a trace
        is parked, so a short request can never trigger the compile-clobbers-trace hang.

        `chunk_start` is the segment's absolute start position (0 for a from-scratch short
        prompt; num_full*chunk_size for the tail of a long prompt — the carried GDN/KV state
        must already be in place). `vis_row_offset` is the number of image-placeholder tokens
        before this segment (so a tail that holds the bottom of a large image splices the right
        slice of the packed vision rows). Returns ttnn.Tensor (host) [1, 1, vocab_size]: the
        logit after position actual_len-1.
        """
        B_batch, _ = token_ids.shape
        assert B_batch == 1, "masked-bucket prefill is single-sequence"
        if bucket is None:
            bucket = self._mask_bucket_for(actual_len)
        assert 1 <= actual_len <= bucket, f"actual_len {actual_len} not in [1, {bucket}]"

        if chunk_start == 0:
            # chunk_start==0: new sequence, re-zero GDN. chunk_start>0: tail, keep carried state.
            self._reset_gdn_state_for_new_sequence()
            self._ht_mark("gdn_reset")
            # Stage the per-request RoPE for this segment (M-RoPE for multimodal, 1D for text).
            # Only at the sequence start; a carried tail (chunk_start>0) keeps the table the
            # long-prompt entry (prefill_traced_chunked) already staged.
            self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
            self._ht_mark("rope_stage")
            self._pt_mark("tail_reset")

        real = token_ids[:, :actual_len].to(torch.int32)
        if bucket > actual_len:
            pad = torch.zeros(1, bucket - actual_len, dtype=torch.int32)
            token_buf = torch.cat([real, pad], dim=1)
        else:
            token_buf = real

        # Stage the trace-safe vision buffers (host->device copy only). A segment splices its own
        # slice of the packed vision rows (vis_row_offset); a segment with no image placeholders
        # (text-only prompt, or a tail past the image) clears the mask inside _set_vision_merge so
        # the where is the identity. No-op without buffers.
        self._set_vision_merge(token_buf, vision_tokens, vis_row_offset)
        self._pt_mark("tail_prep")
        self._ht_mark("tail_prep")

        # QWEN36_PREFILL_BUCKET_TRACE: one execute_trace + a few DMAs instead of the eager per-layer dispatch.
        # Empty dict unless the gate is on (dead by default); an un-traced bucket, a non-flex call, or a call whose
        # conv-window defer state differs from the capture's falls through to the eager path.
        _tr = self._mb_traces.get(bucket) if (self.num_devices > 1 and flex_sdpa) else None
        if _tr is not None and self._mb_kda_max_len is not None:
            _tr = self._bkda_len_pick(_tr, bucket, chunk_start + actual_len)
        if _tr is not None and _tr.defer == self._gdn_defer_state():
            if self._phase_timer is not None:
                self._phase_timer.info["path"] = "traced"
            lg = self._replay_prefill_bucket_trace_tp(_tr, token_buf, actual_len, chunk_start, page_table)
            self._pt_mark("tail_logits")
            self._ht_mark("tail_logits")
            self._mtp_warm_seg(_tr.output, chunk_start, actual_len, bucket)  # QWEN36_MTP_WARM_TAIL (no-op when off)
            self._ht_mark("mtp_seg")
            return lg
        if self._phase_timer is not None:
            self._phase_timer.info["path"] = "eager"

        hidden = self._forward_prefill_chunk_masked(
            token_buf, actual_len, chunk_start, page_table, bucket, flex_sdpa=flex_sdpa, vision_tokens=vision_tokens
        )
        self._pt_mark("tail_dispatch", sync=False)  # host time to enqueue the eager forward
        self._ht_mark("tail_dispatch")
        ttnn.synchronize_device(self.device)
        self._pt_mark("tail_device", sync=False)  # remaining device drain after the host finished enqueueing
        self._ht_mark("tail_device")

        if self.num_devices > 1:
            self._mtp_warm_seg(hidden, chunk_start, actual_len, bucket)  # QWEN36_MTP_WARM_TAIL (no-op when off)
            self._ht_mark("mtp_seg")
            if self._phase_timer is None and self._host_timer is None:
                return self._masked_bucket_logits_tp(hidden, actual_len, bucket)
            lg = self._masked_bucket_logits_tp(hidden, actual_len, bucket)
            self._pt_mark("tail_logits")
            self._ht_mark("tail_logits")
            return lg

        # One-hot matmul for last row (fixed program per bucket; slice would recompile per length).
        sel = torch.zeros(1, 1, bucket, dtype=torch.float32)
        sel[0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(sel, dtype=hidden.dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
        x_last = ttnn.matmul(sel_tt, hidden)
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        return logits.cpu()

    def _masked_bucket_logits_tp(self, hidden, actual_len, bucket):
        """TP: one-hot select row actual_len-1, norm, lm_head. Returns replicated [1,1,vocab]."""
        sel = torch.zeros(1, 1, 1, bucket, dtype=torch.float32)
        sel[0, 0, 0, actual_len - 1] = 1.0
        sel_tt = ttnn.from_torch(
            sel,
            dtype=hidden.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x_last = ttnn.matmul(sel_tt, hidden)  # [1, 1, 1, dim]
        ttnn.deallocate(sel_tt)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    # ----------------------------------------------------------------------- #
    # QWEN36_PREFILL_PHASE_TIMING (per request, host wall clock)
    # ----------------------------------------------------------------------- #
    def _phase_timing_enabled(self):
        return _tt_public_getenv("QWEN36_PREFILL_PHASE_TIMING", "0") == "1"

    def _pt_mark(self, name, sync=True):
        """Close phase `name` on the live timer. No timer (flag unset) -> returns at once: no sync, no op."""
        pt = self._phase_timer
        if pt is None:
            return
        if sync:
            ttnn.synchronize_device(self.device)
        pt.mark(name)

    def _pt_begin(self):
        """Start a per-request timer when the flag is set and none is live. Returns True if this call owns it."""
        if self._phase_timer is not None or not self._phase_timing_enabled():
            return False
        ttnn.synchronize_device(self.device)  # earlier queued work must not land in this request's first phase
        self._phase_timer = mbt.PrefillPhaseTimer()
        self._phase_timer.info["buckets_traced"] = ",".join(str(b) for b in sorted(self._mb_traces)) or "none"
        return True

    def _pt_end(self, owned):
        if not owned:
            return
        pt, self._phase_timer = self._phase_timer, None
        if pt is not None:
            logger.info(pt.line())

    # ----------------------------------------------------------------------- #
    # QWEN36_PREFILL_HOST_TIMING (per request, host wall clock only, no syncs)
    # ----------------------------------------------------------------------- #
    # Diagnostic twin of QWEN36_PREFILL_PHASE_TIMING for the prefill host gap
    # (docs/analysis/prefill-hostgap-20261015.md): per-request host time split into
    # bind / rope / reset / chunk_setup / chunk_prep / drain_sync / mtp_seg /
    # logits_sel / readback / mtp_finish / slot_copy / unbind. Unlike _pt_mark, these
    # marks MUST NEVER synchronize: _ht_mark is a bare timer mark, so a phase that
    # enqueues non-blocking device work accounts only its host CPU. The drain_sync
    # phase times the already-present chunk-loop syncs (host wait on busy device) and
    # is labelled as wait, not work. Measurement-only: no device behavior changes, so
    # no MUSTFAIL rung (cf. QWEN36_MTP_OVERLAP_TIMING precedent). Off = no timer, no
    # log lines, one attribute check per site. Nesting note: the paged_slots-level
    # prefill_body/readback phases bracket the inner chunked-prefill sub-phases, so the
    # line's phases are NOT all additive; and with N > 1 users they accumulate per call.
    def _host_timing_enabled(self):
        return _tt_public_getenv("QWEN36_PREFILL_HOST_TIMING", "0") == "1"

    def _ht_mark(self, name):
        """Close host phase `name` on the live timer. No timer (flag unset) -> returns
        at once: no sync, no op, no log. NEVER adds a synchronize (unit-guarded)."""
        ht = self._host_timer
        if ht is None:
            return
        ht.mark(name)

    def _ht_begin(self):
        """Start a per-request host timer when the flag is set and none is live."""
        if self._host_timer is not None or not self._host_timing_enabled():
            return False
        if not self._host_timing_banner:
            self._host_timing_banner = True
            logger.info(
                "QWEN36_PREFILL_HOST_TIMING engaged: host-only prefill wall clocks "
                "(no added device syncs; drain_sync phases time already-present syncs)"
            )
        self._host_timer = mbt.PrefillPhaseTimer(clock=time.monotonic)
        self._host_timer.info["syncs"] = "none-added"
        self._host_timer.info["start_mono"] = f"{self._host_timer._t0:.6f}"
        return True

    def _ht_end(self, owned):
        if not owned:
            return
        ht, self._host_timer = self._host_timer, None
        if ht is not None:
            ht.info["end_mono"] = f"{time.monotonic():.6f}"
            logger.info(ht.line(prefix="QWEN36_PREFILL_HOST_TIMING engaged:"))

    # ----------------------------------------------------------------------- #
    # Traced MASKED-bucket prefill (boot-time gate: QWEN36_PREFILL_BUCKET_TRACE)
    # ----------------------------------------------------------------------- #
    # Every prompt shorter than one 2048 chunk, and the tail of every long prompt, goes through the eager masked
    # bucket: 64 layers of host dispatch plus ~151 host->device uploads per request. What blocked tracing it was
    # that the GDN validity masks and the FIR decode-window one-hot are built INSIDE library code from an int
    # valid_len (a host write, illegal under trace) and that the KV-fill page table's WIDTH depends on valid_len.
    # Both are hoisted into per-bucket persistent buffers whose SHAPES depend only on the bucket, so one captured
    # program set serves every actual_len and only values move per request. Ported from port/issue-50-bucket-trace
    # (a83cace) with: gdn_masks reach GDN through TPGatedDeltaNet._mb_gdn_masks (no layer.py mount), the eager
    # compile pass runs BEFORE the chunk trace parks, the capture bakes the conv-window defer state, and the fill
    # page table's pad entries always point at the scratch block.

    def _gdn_defer_state(self):
        gl = self._gdn_layers()
        return bool(gl and gl[0]._defer_conv_window)

    def _set_gdn_masks(self, masks):
        for dn in self._gdn_layers():
            dn._mb_gdn_masks = masks

    def _assert_sdpa_pt_covers(self, chunk_start, length, block_size):
        """The SDPA page table must already be wide enough for [chunk_start, chunk_start+length).

        TPAttention.forward_prefill_paged pads a too-narrow table with ttnn.zeros + ttnn.concat (a device
        allocation): legal eagerly, fatal inside a captured trace. The shared full page-table buffer is sized
        from the WHOLE KV cache and 32-aligned; this replicates that function's arithmetic."""
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        needed_blocks = -(-(chunk_start + length) // block_size)
        target_blocks = ((max(needed_blocks, buf_blocks) + 31) // 32) * 32
        assert target_blocks <= buf_blocks, (
            f"full page-table buffer is {buf_blocks} blocks but chunk_start={chunk_start} + "
            f"{length} needs {target_blocks} (32-aligned); the SDPA zero-pad branch would run"
        )

    def _forward_prefill_bucket_body_tp(self, bucket, bufs, chunk_start_host):
        """The traced body: a device-only twin of _forward_prefill_chunk_masked_tp.

        Reads ONLY persistent buffers (no ttnn.from_torch), takes the masked GDN branch via gdn_masks, and fills
        the FULL bucket of K/V (fixed-width fill page table, so page_len == S and the per-width slice disappears).
        The captured op sequence depends only on `bucket`. Returns hidden [1, 1, bucket, dim]."""
        block_size = get_block_size(self._paged_kv_caches)
        self._assert_sdpa_pt_covers(chunk_start_host, bucket, block_size)
        # Block 2048 (the prefix arm): a bucket below one block fills ONE block from its start (page_len >= S, no slice;
        # ttsim [source validation record] fill_S*).
        assert int(bufs.fill_pt.shape[-1]) == -(-bucket // block_size), "fill page table must span the whole bucket"

        x = self.embd(bufs.token)
        x = ttnn.reshape(x, (1, 1, bucket, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        x = self._apply_vision_merge(x, length=bucket)
        self._set_gdn_masks((bufs.mask_f32, bufs.mask_q, bufs.conv_sel))
        try:
            for _li, layer in enumerate(self.layers):
                if layer.is_full_attention:
                    x_new = layer.forward(
                        x,
                        cos=bufs.cos,
                        sin=bufs.sin,
                        mode="prefill",
                        page_table=self._chunk_full_page_table_buf,
                        chunk_page_table=bufs.fill_pt,
                        # Host int: with the tensor form supplied it only sizes the (asserted-dead) SDPA pad.
                        chunk_start_idx=chunk_start_host,
                        chunk_start_idx_tensor=self._chunk_start_idx_tensor,
                        valid_len=bucket,  # unused by full attention
                        **self._xadd_kw(_li),
                    )
                else:
                    x_new = layer.forward(
                        x,
                        mode="prefill",
                        chunk_size=self.args.gdn_chunk_size,
                        valid_len=bucket,  # shapes only: gdn_masks sets valid_len=None inside, values come from masks
                        **self._xadd_kw(_li),
                    )
                ttnn.deallocate(x)
                x = x_new
        finally:
            self._set_gdn_masks(None)
        return x

    def _alloc_masked_bucket_bufs(self, device, bucket, page_table):
        """One bucket's persistent replicated inputs, primed at valid_len = bucket-1 (values do not matter to the
        captured programs, only shapes; bucket-1 keeps 0s and 1s in the masks)."""
        rep = ttnn.ReplicateTensorToMesh(device)
        block_size = get_block_size(self._paged_kv_caches)
        K = next(layer.attention.K for layer in self.layers if not layer.is_full_attention)
        warm_len = max(1, bucket - 1)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        m = mbt.host_masks(warm_len, bucket)

        def _up(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=device, mesh_mapper=rep)

        return _MaskedBucketBufs(
            token=_up(torch.zeros(1, bucket, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT),
            cos=_up(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            sin=_up(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            fill_pt=_up(
                mbt.fill_pt_row(page_table, 0, warm_len, bucket, self._pad_kv_block, block_size, mapped_pad=False),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            ),
            mask_f32=_up(m, ttnn.float32, ttnn.TILE_LAYOUT),
            mask_q=_up(m, ttnn.bfloat16, ttnn.TILE_LAYOUT),
            conv_sel=_up(mbt.host_conv_sel(warm_len, bucket, K), ttnn.bfloat16, ttnn.TILE_LAYOUT),
        )

    @staticmethod
    def _trace_region_bytes(device):
        """Bytes allocated in the device TRACE region, or None when the API is unavailable."""
        try:
            mv = ttnn.get_memory_view(device, ttnn.BufferType.TRACE)
            return int(mv.total_bytes_allocated_per_bank) * int(mv.num_banks)
        except Exception:  # older/other builds: skip the accounting, not the capture
            return None

    def _warm_prefill_bucket_traces(self, device, page_table):
        """Allocate every gated bucket's persistent inputs and run ONE eager pass of the traced body (compiles
        its programs and creates the lazily-allocated per-M CCL/matmul buffers). Called from
        _capture_prefill_trace_chunked_tp BEFORE the chunk trace is captured, with the conv-window defer state the
        capture will use. No-op when QWEN36_PREFILL_BUCKET_TRACE is off."""
        if not self._mb_trace_buckets:
            return
        assert self.num_devices > 1, "the traced masked bucket is the TP (num_devices>1) path"
        assert self._paged_kv_caches, "Call allocate_kv_caches first"
        block_size = get_block_size(self._paged_kv_caches)
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        num_kv_blocks = int(self._paged_kv_caches[0][0].shape[0])
        assert buf_blocks % 32 == 0, f"full page-table buffer width {buf_blocks} must be 32-aligned"
        if self._pad_kv_block is None:
            self._pad_kv_block = num_kv_blocks - 1
        assert 0 < self._pad_kv_block < num_kv_blocks, (
            f"pad KV block {self._pad_kv_block} out of range (1..{num_kv_blocks - 1}); the KV cache must carry one "
            f"spare block the scheduler never hands out (override with QWEN36_PREFILL_BUCKET_PAD_BLOCK)"
        )
        for dn in self._gdn_layers():
            assert dn._stable_state and dn.conv_carry is not None and dn._zero_conv0 is not None, (
                "GDN layers must own their persistent in-place state before capture (allocate_kv_caches -> reset_state)"
            )
        defer = getattr(self, "_gdn_conv_window_once", False)
        if self._mb_kda:
            # QWEN36_PREFILL_BUCKET_KDA: in-boot self-test on layer 0's device taps at the smallest and largest bucket,
            # eager and before any bucket trace is captured; it raises (boot refused) on a mismatch or a blind test.
            if self._mb_kda_mf:
                logger.warning("QWEN36_PREFILL_BUCKET_KDA_MUSTFAIL engaged: the self-test candidate gets a zero conv "
                               "history; this boot must be REFUSED")
            dn0 = self._gdn_layers()[0]
            for T in sorted({min(self._mb_trace_buckets), max(self._mb_trace_buckets)}):
                dn0.mb_kda_selftest(T, mustfail=self._mb_kda_mf)
            logger.info(
                f"QWEN36_PREFILL_BUCKET_KDA engaged: masked-bucket GDN conv via kda.qkv_causal_conv1d_silu (as the "
                f"full chunks; tier 2 vs the MAC FIR), decode conv window via the conv_sel one-hot (bitwise), buckets "
                f"{tuple(sorted(self._mb_trace_buckets))}"
            )
        logger.info(
            f"[prefill] bucket traces: page-table buffer {buf_blocks} blocks (covers {buf_blocks * block_size} "
            f"tokens), KV cache {num_kv_blocks} blocks, pad block {self._pad_kv_block}, conv-window defer {defer}"
        )
        for bucket in sorted(self._mb_trace_buckets, reverse=True):
            assert bucket % block_size == 0 or block_size % bucket == 0, (
                f"bucket {bucket} must be a multiple or a divisor of block size {block_size}")
            bufs = self._mb_bufs.get(bucket)
            if bufs is None:
                bufs = self._alloc_masked_bucket_bufs(device, bucket, page_table)
                self._mb_bufs[bucket] = bufs
            if self._mb_kda_max_len is not None:
                # QWEN36_PREFILL_BUCKET_KDA_MAX_LEN: compile the FIR body too (both bodies are captured); the pass below
                # then compiles the kda body as before.
                self._set_gdn_mb_kda(False)
                self._reset_gdn_state_for_new_sequence()
                self._gdn_conv_window_defer(defer)
                try:
                    warm_fir = self._forward_prefill_bucket_body_tp(bucket, bufs, 0)
                finally:
                    self._gdn_conv_window_defer(False)
                    self._set_gdn_mb_kda(True)
                if defer:
                    self._finalize_gdn_conv_windows()
                ttnn.deallocate(warm_fir)
            self._reset_gdn_state_for_new_sequence()
            self._gdn_conv_window_defer(defer)
            try:
                warm = self._forward_prefill_bucket_body_tp(bucket, bufs, 0)
            finally:
                self._gdn_conv_window_defer(False)
            if defer:
                self._finalize_gdn_conv_windows()
            self._mtp_warm_compile(warm, bucket)  # QWEN36_MTP_WARM_TAIL: this bucket's output spec, pre-park
            warm_logits = self._masked_bucket_logits_tp(warm, bucket, bucket)  # warms the post-replay select
            ttnn.deallocate(warm_logits)
            ttnn.deallocate(warm)
            ttnn.synchronize_device(device)
        self._reset_gdn_state_for_new_sequence()
        if self._mb_kda_max_len is not None:
            self._bkda_len_selftest(device)

    def capture_prefill_bucket_traces(self, device):
        """Capture one masked-prefill trace per gated bucket whose buffers _warm_prefill_bucket_traces made
        (largest first, so a trace-region overflow fails at warmup, not at request time). GDN state is re-zeroed
        before each capture and at the end; the garbage K/V the passes write into the warm blocks is overwritten
        by the first real request, as the chunk-trace capture already relies on."""
        if not self._mb_trace_buckets:
            return
        defer = getattr(self, "_gdn_conv_window_once", False)
        for bucket in sorted(self._mb_trace_buckets, reverse=True):
            bufs = self._mb_bufs[bucket]  # KeyError = the eager warm pass never ran: fail closed
            before = self._trace_region_bytes(device)
            self._reset_gdn_state_for_new_sequence()
            self._gdn_conv_window_defer(defer)
            if self._mb_kda_max_len is not None:
                self._set_gdn_mb_kda(False)  # QWEN36_PREFILL_BUCKET_KDA_MAX_LEN: the default set is the FIR body
            try:
                trace_id = ttnn.begin_trace_capture(device, cq_id=0)
                out = self._forward_prefill_bucket_body_tp(bucket, bufs, 0)
                ttnn.end_trace_capture(device, trace_id, cq_id=0)
            finally:
                self._gdn_conv_window_defer(False)
                if self._mb_kda_max_len is not None:
                    self._set_gdn_mb_kda(True)
            self._mb_traces[bucket] = _MaskedBucketTrace(bucket, trace_id, bufs, out, defer)
            after = self._trace_region_bytes(device)
            delta = f"+{(after - before) / (1 << 20):.1f} MiB" if None not in (before, after) else "n/a"
            logger.info(f"Masked-bucket({bucket}) prefill trace (TP) captured; TRACE region {delta}")
            if self._mb_kda_max_len is not None:
                self._capture_bkda_len_trace(device, bucket, bufs, defer)
        self._reset_gdn_state_for_new_sequence()
        if self._mb_kda_max_len is not None:
            logger.info(
                f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN engaged: N={self._mb_kda_max_len}; masked-bucket tails ending at "
                f"prompt position <= N replay the kda body, longer prompts the shipped MAC FIR body; both trace sets "
                f"captured for buckets {tuple(sorted(self._mb_traces_kda))}"
                + (" MUSTFAIL: the kda set holds the FIR body" if self._mb_kda_len_mf else "")
            )
        logger.info(
            f"QWEN36_PREFILL_BUCKET_TRACE engaged: masked-bucket prefill traces captured for buckets "
            f"{tuple(sorted(self._mb_traces))}, pad KV block {self._pad_kv_block}, conv-window defer {defer}, "
            f"inputs rewritten every replay"
        )
        if self._mb_mustfail:
            logger.warning(
                f"QWEN36_PREFILL_BUCKET_TRACE_MUSTFAIL engaged: {self._mb_mustfail} -- replays skip the token/rope/"
                f"mask/conv-one-hot rewrite (stale inputs); a correctness gate on this arm must FAIL"
            )

    def release_prefill_bucket_traces(self):
        """Release every captured masked-bucket trace and its persistent buffers."""
        for tr in list(self._mb_traces.values()) + list(self._mb_traces_kda.values()):
            ttnn.release_trace(self.device, tr.trace_id)
        self._mb_traces = {}
        self._mb_traces_kda = {}
        for bufs in self._mb_bufs.values():
            for f in fields(bufs):
                buf = getattr(bufs, f.name)
                if buf is not None:
                    ttnn.deallocate(buf)
        self._mb_bufs = {}

    def _replay_prefill_bucket_trace_tp(self, tr, token_buf, actual_len, chunk_start, page_table):
        """Rewrite one bucket trace's persistent inputs (ALL of them, every call: another trace's replay may have
        clobbered them since), replay it, then run the (already warmed) eager logits select.
        Returns replicated logits [1, 1, vocab] at position actual_len-1."""
        bucket = tr.bucket
        block_size = get_block_size(self._paged_kv_caches)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        assert tuple(token_buf.shape) == (1, bucket), f"token_buf {tuple(token_buf.shape)} != (1, {bucket})"
        self._assert_sdpa_pt_covers(chunk_start, bucket, block_size)

        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        pt = page_table.reshape(1, -1)
        if pt.shape[1] < buf_blocks:
            pt = torch.cat([pt, torch.zeros(1, buf_blocks - pt.shape[1], dtype=pt.dtype)], dim=1)
        elif pt.shape[1] > buf_blocks:
            pt = pt[:, :buf_blocks]
        pt = pt.contiguous().to(torch.int32)

        K = int(tr.bufs.conv_sel.shape[1]) + 1
        cos_t, sin_t = self._rope_tp_cos_sin_torch(chunk_start, bucket)
        m = mbt.host_masks(actual_len, bucket)
        conv = mbt.host_conv_sel(actual_len, bucket, K)
        fill = mbt.fill_pt_row(page_table, chunk_start, actual_len, bucket, self._pad_kv_block, block_size, mapped_pad=False)

        def _h(t, dtype, layout):
            return ttnn.from_torch(t, dtype=dtype, layout=layout, device=None, mesh_mapper=rep)

        # Host tensors stay referenced (the list) until after the synchronize: their DMAs must complete.
        value_inputs = [
            (_h(token_buf.to(torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT), tr.bufs.token),
            (_h(cos_t, ttnn.bfloat16, ttnn.TILE_LAYOUT), tr.bufs.cos),
            (_h(sin_t, ttnn.bfloat16, ttnn.TILE_LAYOUT), tr.bufs.sin),
            (_h(m, ttnn.float32, ttnn.TILE_LAYOUT), tr.bufs.mask_f32),
            (_h(m, ttnn.bfloat16, ttnn.TILE_LAYOUT), tr.bufs.mask_q),
            (_h(conv, ttnn.bfloat16, ttnn.TILE_LAYOUT), tr.bufs.conv_sel),
        ]
        # Address inputs are rewritten even under the must-fail rung: a stale fill/page table would write K/V into
        # another request's blocks, which is a different (and destructive) failure than the one being falsified.
        addr_inputs = [
            (_h(fill, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), tr.bufs.fill_pt),
            (_h(pt, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT), self._chunk_full_page_table_buf),
            (_h(torch.tensor([chunk_start], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
             self._chunk_start_idx_tensor),
        ]
        copies = addr_inputs if self._mb_mustfail == "stale" else value_inputs + addr_inputs
        for host_t, dev_t in copies:
            ttnn.copy_host_to_device_tensor(host_t, dev_t, cq_id=0)
        self._pt_mark("tail_dma", sync=False)
        self._ht_mark("tail_inputs")

        ttnn.execute_trace(self.device, tr.trace_id, cq_id=0, blocking=False)
        if self._ph_nosync_on:
            # QWEN36_PREFILL_TAIL_NOSYNC: the logits select + warm gathers are enqueued behind the replay; the inputs
            # are held until prefill_paged_slots' blocking readback.
            self._ph_refs.add([h for h, _d in copies])
        else:
            ttnn.synchronize_device(self.device)
        del value_inputs, addr_inputs, copies
        self._pt_mark("tail_replay", sync=False)
        self._ht_mark("tail_replay")
        self._mb_replays += 1
        if self._mb_replays == 1 or self._mb_replays % 256 == 0:
            logger.info(
                f"QWEN36_PREFILL_BUCKET_TRACE replay: bucket={bucket} actual_len={actual_len} "
                f"chunk_start={chunk_start} replays={self._mb_replays}"
                + (f" MUSTFAIL={self._mb_mustfail}" if self._mb_mustfail else "")
            )
        return self._masked_bucket_logits_tp(tr.output, actual_len, bucket)

    def warmup_prefill_masked_buckets(self, page_table, buckets=None):
        """Compile every masked-bucket prefill program up front, so a request never compiles after
        a trace is parked (a post-park compile clobbers the trace -> second-request hang, #48536).

        Two program kinds:
          * bucket-keyed (SDPA / GDN mask / norm / MLP-or-MoE): one per (bucket, is_full). Warmed
            by a dummy forward at each bucket, once masked (actual_len < bucket) and once full (==).
          * fill-width-keyed (paged_fill_cache): hashes on the fill shape, so it recompiles per
            fill width. Warmed directly by _warmup_paged_fill_widths (no full forward).

        MUST run in GDN serving state, before any trace is parked (capture_prefill_trace_chunked
        calls this just before begin_trace_capture). page_table must cover the largest bucket."""
        if buckets is None:
            buckets = prefill_mask_buckets()  # QWEN36_PREFILL_CHUNK > 2048 adds the chunk as a bucket (c134)
        block_size = get_block_size(self._paged_kv_caches)

        # Bucket-keyed programs: one masked + one no-mask forward per bucket.
        seen = set()
        for bucket in sorted(buckets):
            for actual_len in (max(1, bucket // 2), bucket):
                actual_len = max(1, min(actual_len, bucket))
                key = (bucket, actual_len == bucket)
                if key in seen:
                    continue
                seen.add(key)
                toks = torch.zeros(1, actual_len, dtype=torch.int32)
                self.prefill_masked_bucket(toks, page_table, actual_len=actual_len, bucket=bucket)
        # Fill-width-keyed programs: warm every width directly (no full forward).
        self._warmup_paged_fill_widths(page_table, buckets, block_size)
        ttnn.synchronize_device(self.device)

    def _warmup_paged_fill_widths(self, page_table, buckets, block_size):
        """Warm the per-fill-width programs in TPAttention.forward_prefill_paged's KV-fill sub-path
        (ttnn.slice + paged_fill_cache) without a full-model forward. Both hash on the fill shape
        (seq = fill_blocks * block_size), so each width is a fresh program; an un-warmed width would
        compile after the trace is parked and clobber it (hang).

        The ops are shape-keyed, so warming one layer's cache serves every layer -- far cheaper than
        the old per-width all-layer forward. Cover EVERY fill-width-dependent op here; a new one is
        caught by test_prefill_warmup_no_recompile (width sweep under misses-disallowed)."""
        if not self._paged_kv_caches:
            return
        k_cache, v_cache = self._paged_kv_caches[0]
        nkv, hd = k_cache.shape[1], k_cache.shape[3]
        v_needs_own_tensor = v_cache.dtype != k_cache.dtype
        mapper = ttnn.ReplicateTensorToMesh(self.device) if self.num_devices > 1 else None
        seen = set()
        for bucket in sorted(buckets):
            k_full = ttnn.from_torch(
                torch.zeros(1, nkv, bucket, hd, dtype=torch.bfloat16),
                dtype=k_cache.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                # L1 to match the request-time fill's L1 K/V (slice's cache key is buffer-type-specific).
                memory_config=ttnn.L1_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            # The V cache may carry a different dtype than K's (QWEN36_KV_DTYPE): a single
            # tensor cannot feed paged_fill_cache into both, so build a second one.
            v_full = None
            if v_needs_own_tensor:
                v_full = ttnn.from_torch(
                    torch.zeros(1, nkv, bucket, hd, dtype=torch.bfloat16),
                    dtype=v_cache.dtype,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                    mesh_mapper=mapper,
                )
            for w in range(1, num_blocks_in_seq(bucket, block_size) + 1):
                page_len = min(w * block_size, bucket)
                key = (bucket, page_len)
                if key in seen:
                    continue
                seen.add(key)
                pt = ttnn.from_torch(
                    page_table[:, :w].contiguous(),
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    device=self.device,
                    mesh_mapper=mapper,
                )
                if v_needs_own_tensor and page_len < bucket:
                    # K and V are different dtypes: one tensor cannot feed both caches, so
                    # slice each of its own source.
                    k_fill = ttnn.slice(k_full, (0, 0, 0, 0), (1, nkv, page_len, hd))
                    v_fill = ttnn.slice(v_full, (0, 0, 0, 0), (1, nkv, page_len, hd))
                elif v_needs_own_tensor:
                    k_fill, v_fill = k_full, v_full
                elif page_len < bucket:
                    fill = ttnn.slice(k_full, (0, 0, 0, 0), (1, nkv, page_len, hd))
                    k_fill, v_fill = fill, fill
                else:
                    k_fill, v_fill = k_full, k_full
                ttnn.experimental.paged_fill_cache(k_cache, k_fill, pt, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, v_fill, pt, batch_idx=0)
                if page_len < bucket:
                    if v_needs_own_tensor:
                        ttnn.deallocate(k_fill)
                        ttnn.deallocate(v_fill)
                    else:
                        ttnn.deallocate(fill)
                ttnn.deallocate(pt)
            ttnn.deallocate(k_full)
            if v_full is not None:
                ttnn.deallocate(v_full)

    # QWEN36_MTP_WARM_TAIL (tt/mtp_batch.py _MTPWarm, docs/MTP-BATCH.md section 13): the prefill paths hand the
    # MTP warm pass every segment hidden that may hold prompt-tail rows, while it is still on the device. Inert
    # (one getattr) unless the MTP batch object built a warm pass.
    def _mtp_warm(self):
        mb = getattr(self, "_mtp_batch", None)
        return getattr(mb, "warm", None) if mb is not None else None

    def _mtp_warm_seg(self, hidden, seg_start, valid, rows):
        if self._ph_deferred is not None:
            self.ph_flush_deferred()  # QWEN36_PREFILL_DEFER_FINISH: the parked finish reads the windows this overwrites
        w = self._mtp_warm()
        if w is not None and w.req is not None:
            w.segment(hidden, int(seg_start), int(valid), int(rows))

    def _mtp_warm_span_seg(self, hidden, seg_start, rows):
        if self._ph_deferred is not None:
            self.ph_flush_deferred()  # QWEN36_PREFILL_DEFER_FINISH: the parked finish reads the windows this overwrites
        w = self._mtp_warm()
        if w is not None and w.span and w.req is not None:
            w.segment(hidden, int(seg_start), int(rows), int(rows))

    def _mtp_warm_compile(self, hidden, rows):
        w = self._mtp_warm()
        if w is not None:
            w.compile_from(hidden, int(rows))

    def _gdn_layers(self):
        return [layer.attention for layer in self.layers if not layer.is_full_attention]

    def _set_gdn_mb_kda(self, on):
        """QWEN36_PREFILL_BUCKET_KDA_MAX_LEN: pick the masked-bucket GDN conv body (kda / MAC FIR) on every GDN layer
        for the next eager pass or capture. Only called with the max-len variant on; the layers start at True."""
        for dn in self._gdn_layers():
            dn._mb_kda = bool(on)

    def _capture_bkda_len_trace(self, device, bucket, bufs, defer):
        """QWEN36_PREFILL_BUCKET_KDA_MAX_LEN: capture the kda-body twin of one bucket's trace. It reads the SAME
        persistent input buffers (every replay rewrites all of them first) and owns its own output."""
        before = self._trace_region_bytes(device)
        self._reset_gdn_state_for_new_sequence()
        self._gdn_conv_window_defer(defer)
        self._set_gdn_mb_kda(not self._mb_kda_len_mf)  # MUSTFAIL: the FIR body again (the self-test refuses first)
        try:
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            out = self._forward_prefill_bucket_body_tp(bucket, bufs, 0)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        finally:
            self._gdn_conv_window_defer(False)
            self._set_gdn_mb_kda(True)
        self._mb_traces_kda[bucket] = _MaskedBucketTrace(bucket, trace_id, bufs, out, defer)
        after = self._trace_region_bytes(device)
        delta = f"+{(after - before) / (1 << 20):.1f} MiB" if None not in (before, after) else "n/a"
        logger.info(f"Masked-bucket({bucket}) prefill trace (TP, kda body) captured; TRACE region {delta}")

    def _bkda_len_pick(self, tr_fir, bucket, tail_end):
        """QWEN36_PREFILL_BUCKET_KDA_MAX_LEN: the trace to replay for a tail ending at prompt position tail_end."""
        use_kda = mbt.bucket_kda_selects(tail_end, self._mb_kda_max_len)
        key = "kda" if use_kda else "fir"
        self._mb_kda_len_replays[key] += 1
        n = self._mb_kda_len_replays[key]
        if n == 1 or n % 256 == 0:
            logger.info(
                f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN select: body={key} tail_end={tail_end} "
                f"max_len={self._mb_kda_max_len} bucket={bucket} replays={self._mb_kda_len_replays}"
            )
        return self._mb_traces_kda[bucket] if use_kda else tr_fir

    def _bkda_len_selftest(self, device):
        """QWEN36_PREFILL_BUCKET_KDA_MAX_LEN in-boot self-test, eager, after the bucket warm passes and BEFORE any
        trace is captured; raises -> the boot is refused. On every device of the mesh, the smallest bucket's eager
        body on distinct token ids: FIR, kda (the FIR body under _MUSTFAIL), FIR again; mbt.bucket_bodies_gate wants the
        two FIR passes bitwise equal (long prompts prefill exactly as without bkda), the kda pass different from the
        FIR one and within tier-2 rel-L2. Wrong-input branch: the gate fed the second FIR pass as the "kda" output
        must REFUSE (a blind gate refuses the boot). The tail selector is checked at N and N+1."""
        bucket = min(self._mb_trace_buckets)
        bufs = self._mb_bufs[bucket]
        defer = getattr(self, "_gdn_conv_window_once", False)
        rep = ttnn.ReplicateTensorToMesh(device)

        def _tok(t):
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(t.reshape(1, bucket).to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT,
                                device=None, mesh_mapper=rep),
                bufs.token, cq_id=0,
            )

        def _pass(kda_body):
            self._set_gdn_mb_kda(kda_body)
            self._reset_gdn_state_for_new_sequence()
            self._gdn_conv_window_defer(defer)
            try:
                h = self._forward_prefill_bucket_body_tp(bucket, bufs, 0)
            finally:
                self._gdn_conv_window_defer(False)
            if defer:
                self._finalize_gdn_conv_windows()
            out = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(h)]
            ttnn.deallocate(h)
            return out

        _tok((torch.arange(bucket, dtype=torch.int64) * 7919 + 1009) % min(int(self.vocab_size), 100000))
        try:
            fir = _pass(False)
            kda = _pass(not self._mb_kda_len_mf)
            fir2 = _pass(False)
        finally:
            self._set_gdn_mb_kda(True)
            _tok(torch.zeros(bucket, dtype=torch.int64))
            self._reset_gdn_state_for_new_sequence()
            ttnn.synchronize_device(device)
        gates = [mbt.bucket_bodies_gate(f, f2, k) for f, f2, k in zip(fir, fir2, kda)]
        wrong = [mbt.bucket_bodies_gate(f, f2, f2) for f, f2 in zip(fir, fir2)]
        n = self._mb_kda_max_len
        sel_ok = mbt.bucket_kda_selects(n, n) and not mbt.bucket_kda_selects(n + 1, n)
        bad = [d for d, g in enumerate(gates) if not g["ok"]]
        blind = [d for d, w in enumerate(wrong) if w["ok"]]
        if bad or blind or not sel_ok or not gates:
            raise RuntimeError(
                f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN self-test REFUSED (bucket={bucket}, N={n}, "
                f"mustfail={self._mb_kda_len_mf}): devices {bad} failed {gates}; wrong-input branch passed (blind) on "
                f"{blind}; selector ok={sel_ok}: refusing the boot"
            )
        logger.info(
            f"QWEN36_PREFILL_BUCKET_KDA_MAX_LEN self-test pass: bucket={bucket} on {len(gates)} devices; FIR body "
            f"bitwise repeatable, kda body differs in {min(g['diff_elems'] for g in gates)}+ elements at rel-L2 <= "
            f"{max(g['rel_l2'] for g in gates):.3g} (max {gates[0]['rel_max']}); wrong-input branch (FIR as kda) "
            f"refused; selector kda at tail_end={n}, FIR at {n + 1}"
        )

    def _gdn_conv_window_defer(self, on):
        """QWEN36_GDN_CONV_WINDOW_ONCE: arm/disarm the per-chunk decode conv-window skip on every GDN layer."""
        for dn in self._gdn_layers():
            dn._defer_conv_window = bool(on)

    def _finalize_gdn_conv_windows(self):
        """QWEN36_GDN_CONV_WINDOW_ONCE: write each GDN layer's decode conv window from its carry, once."""
        for dn in self._gdn_layers():
            dn.finalize_conv_window()

    def prefill_traced_chunked(self, token_ids, page_table, actual_len, vision_tokens=None, start=0):
        """See _prefill_traced_chunked_body. QWEN36_GDN_CONV_WINDOW_ONCE (c105, TP, boot time): every chunk of this
        prefill (the parked chunk trace, which was captured with the skip armed, the eager full chunks and the masked
        tail / short-prompt bucket) skips the decode conv-window write, and it is written once here after the last
        chunk from conv_carry, before any caller reads conv_states (slot copy, snapshot, prefix snapshot, decode).
        Every replay of the chunk trace goes through here, so no replay leaves the window stale."""
        _pt_own = self._pt_begin()  # QWEN36_PREFILL_PHASE_TIMING (B=1 callers; prefill_paged_slots owns it when batched)
        _ht_own = self._ht_begin()  # QWEN36_PREFILL_HOST_TIMING (same ownership; host clocks only, never syncs)
        try:
            if not getattr(self, "_gdn_conv_window_once", False):
                return self._prefill_traced_chunked_body(token_ids, page_table, actual_len, vision_tokens, start)
            self._gdn_conv_window_defer(True)
            try:
                out = self._prefill_traced_chunked_body(token_ids, page_table, actual_len, vision_tokens, start)
            finally:
                self._gdn_conv_window_defer(False)
            if not self._gdn_conv_finalize_skip:  # QWEN36_GDN_SLOT_COPY_TRACE folds it into the slot trace
                self._finalize_gdn_conv_windows()
                self._pt_mark("conv_finalize")
                self._ht_mark("conv_finalize")
            return out
        finally:
            self._pt_end(_pt_own)
            self._ht_end(_ht_own)

    def _prefill_traced_chunked_body(self, token_ids, page_table, actual_len, vision_tokens=None, start=0):
        """Prefill by replaying the captured per-chunk trace for each FULL 2048-token chunk,
        then processing the final partial chunk eagerly with minimal padding.

        Only the real prompt (token_ids[:, :actual_len]) is processed; any bucket padding in
        token_ids is ignored. Full chunks (num_full = actual_len // chunk_size) are replayed
        from the trace; the remaining tail (< chunk_size tokens) is run eagerly so the GDN
        kernel zero-pads it to the next multiple of 128 (matching the non-traced path) instead
        of repeating the bucket padding through the recurrence — which corrupts the decode
        state at long context. actual_len is the real prompt length; the next-token logit is
        extracted at actual_len-1. Returns ttnn.Tensor (host) [1, 1, vocab_size].

        vision_tokens (multimodal) are spliced trace-safely: the captured forward runs a
        FIXED-shape ttnn.where(mask, vision, text) over persistent buffers (_vis_buf /
        _vis_mask_buf) that this method stages per chunk via copy_host_to_device — no program
        compiles at request time, so the parked trace is never clobbered. Each chunk (and the tail)
        splices its own slice of the packed vision rows (vis_row_offset = image tokens before the
        chunk), so a large image whose placeholders span multiple chunks is handled; a segment with
        no image tokens clears the mask (the where becomes the identity). Works on both single
        device (3D buffers) and TP (4D hidden-sharded buffers; the vision rows are gathered to full
        hidden on host in _set_vision_merge, placed along seq, then re-sharded — see
        _alloc_vision_merge_buffers).

        `start` is the absolute position the prompt is RESUMED from (0 for a cold prefill).
        A prefix-cache hit has already written positions [0, start) into the paged KV and has
        restored the GDN recurrent/conv state for that prefix into the same B=1 scratch whose
        addresses the chunk trace baked, so this call must (a) NOT re-zero GDN and (b) begin
        replaying at the first chunk boundary at or after `start`, running the unaligned run-up
        eagerly. prefill_chunk_plan() owns that arithmetic; see its docstring.
        """
        # Default to the standard 2048-token chunk when no trace is captured (e.g. the TP MVP,
        # which serves <=2048 prompts entirely via the masked bucket below and so needs no chunk
        # trace). The chunk trace is only required once there is at least one full chunk to replay.
        chunk_size = self._chunked_chunk_size or prefill_chunk_size()
        B, T = token_ids.shape
        assert 1 <= actual_len <= T, f"actual_len {actual_len} not in [1, {T}]"
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        plan = prefill_chunk_plan(int(start), int(actual_len), chunk_size, block_size)
        assert plan["lead"] is None, (
            f"unaligned prefix resume (lead={plan['lead']}) reached in prefill_traced_chunked, "
            f"but the unaligned-resume path is unqualified and unreachable under valid configuration "
            f"(chunk_size={chunk_size}, block_size={block_size}, start={start});"
        )
        num_full = len(plan["replays"])
        tail_real = plan["tail"][1] if plan["tail"] else 0
        assert (
            num_full == 0 or self.num_devices > 1 or self._chunked_trace_id is not None
        ), "Call capture_prefill_trace_chunked first"

        # Stage the per-request RoPE once for the whole prompt (M-RoPE for multimodal, 1D for text).
        # The chunk-replay loops + the masked tail then slice this sequence-indexed table by chunk
        # position, and decode offsets by the stored rope_delta. (The num_full==0 short path below
        # re-stages it inside prefill_masked_bucket; that is idempotent.)
        self._build_request_rope(token_ids[:, :actual_len], vision_tokens)
        self._ht_mark("rope_stage")
        if self._phase_timer is not None:
            self._phase_timer.info.update(
                len=int(actual_len), start=int(start), chunks=num_full, tail=tail_real,
                bucket=self._mask_bucket_for(tail_real) if tail_real else 0,
            )
            self._pt_mark("rope_stage")
        if self._host_timer is not None:
            self._host_timer.info.update(len=int(actual_len), start=int(start), chunks=num_full, tail=tail_real)

        # Short prompt (no full chunks): route the whole prompt through the SAME masked
        # fixed-bucket path the long-prompt tail uses. chunk_start=0 makes prefill_masked_bucket
        # do the sequence-start GDN reset and run one masked forward — there is no trace to replay,
        # so the chunk-input plumbing below is skipped. This is the single bucketed+masked path
        # shared by short prompts and the long-prompt tail; prefill_dispatch routes every traced
        # prefill here so the short/long seam is defined once.
        if num_full == 0 and plan["lead"] is None:
            # Pad/clip the SDPA page table to the warmed/captured width so the short-prompt forward
            # REPLAYS the pre-warmed programs instead of recompiling at request time (which clobbers
            # parked decode/chunk traces -> second-request hang). vLLM pads to its own
            # max_num_blocks_per_req, which differs from the warmed width. Trailing entries index
            # blocks past the prompt and are never read by causal SDPA (as in the long-prompt branch
            # below). No-op when no chunk buffer was captured or the widths already match.
            buf = getattr(self, "_chunk_full_page_table_buf", None)
            if buf is not None:
                buf_blocks = int(buf.shape[-1])
                if page_table.shape[1] < buf_blocks:
                    page_table = torch.cat(
                        [
                            page_table,
                            torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                        ],
                        dim=1,
                    )
                elif page_table.shape[1] > buf_blocks:
                    page_table = page_table[:, :buf_blocks]
            seg_start, seg_len = plan["tail"]
            return self.prefill_masked_bucket(
                token_ids[:, seg_start:actual_len],
                page_table,
                actual_len=seg_len,
                chunk_start=seg_start,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, seg_start),
            )

        if self.num_devices > 1:
            # TP long prompt: traced replay preferred; eager masked-bucket fallback if no trace.
            if self._chunked_trace_id is not None:
                return self._prefill_traced_chunked_tp(
                    token_ids, page_table, actual_len, plan, chunk_size, vision_tokens=vision_tokens
                )
            # Eager fallback: flexible qk=64 SDPA matches traced path.
            return self._prefill_chunked_eager_tp(
                token_ids,
                page_table,
                actual_len,
                plan,
                chunk_size,
                flex_sdpa=True,
                vision_tokens=vision_tokens,
            )

        # Re-zero GDN once at a sequence START only; carries across replays + masked tail
        # (chunk_start>0 skips reset). A RESUME (plan["reset"] False) keeps the restored prefix state.
        if plan["reset"]:
            self._reset_gdn_state_for_new_sequence()
        # An unaligned resume runs its run-up to the first chunk boundary eagerly.
        if plan["lead"] is not None:
            assert False, (
                f"unaligned prefix resume (lead={plan['lead']}) reached in prefill_traced_chunked (single-device), "
                f"but the unaligned-resume path is unqualified and unreachable under valid configuration;"
            )
            lead_start, lead_len = plan["lead"]
            lead_logits = self.prefill_masked_bucket(
                token_ids[:, lead_start : lead_start + lead_len],
                page_table,
                actual_len=lead_len,
                chunk_start=lead_start,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, lead_start),
            )
            if num_full == 0 and tail_real == 0:
                return lead_logits
            ttnn.deallocate(lead_logits)
        # Pad/clip page_table to captured buffer width (vLLM may differ). Trailing blocks unused.
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if page_table.shape[1] < buf_blocks:
            page_table = torch.cat(
                [
                    page_table,
                    torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                ],
                dim=1,
            )
        elif page_table.shape[1] > buf_blocks:
            page_table = page_table[:, :buf_blocks]
        pt_host = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)

        # Replay trace for each full chunk (absolute offsets: a resume does not start at 0).
        for c, cs in enumerate(plan["replays"]):
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            # M-RoPE-aware per-chunk cos/sin (slices the staged per-request table for multimodal;
            # 1D RoPE for text). Updated into the persistent buffer per chunk via host->device copy,
            # so it stays trace-safe.
            cos_seq, sin_seq = self.rope.prefill_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            sin_host = ttnn.from_torch(
                sin_seq.unsqueeze(0).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)

            # Stage the trace-safe vision buffers: each chunk splices its own slice of the packed
            # vision rows (vis_row_offset = image tokens before cs); a chunk with no image tokens
            # clears the mask so the captured where is the identity. host->device copy only (no
            # compile), so the parked trace is untouched. Handles a large image whose placeholders
            # span multiple chunks.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)

        ttnn.synchronize_device(self.device)

        # Tail via masked bucket (or last full chunk hidden if exact multiple of chunk_size).
        if tail_real > 0:
            cs = plan["tail"][0]
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        hidden = self._chunked_trace_output  # last full chunk's hidden state
        pos_in_chunk = (actual_len - 1) - plan["replays"][-1]
        ttnn.synchronize_device(self.device)

        x_last = hidden[:, pos_in_chunk : pos_in_chunk + 1, :]
        x_last = ttnn.to_layout(x_last, ttnn.TILE_LAYOUT)
        x_last = ttnn.to_memory_config(x_last, ttnn.DRAM_MEMORY_CONFIG)
        x_last = self.norm(x_last, mode=Mode.PREFILL)
        logits = self._lm_head(x_last)
        return logits.cpu()

    def _prefill_chunked_eager_tp(
        self, token_ids, page_table, actual_len, plan, chunk_size, flex_sdpa=True, vision_tokens=None
    ):
        """TP eager long-prompt prefill via warmed bucket=chunk_size programs.
        Returns logits [1,1,vocab] at actual_len-1.

        `plan` comes from prefill_chunk_plan(): its "replays" are run eagerly here (there is no
        parked chunk trace on this path) but at the SAME absolute offsets, and its "reset" flag
        is what keeps a prefix resume from zeroing the state it just restored."""
        assert plan["lead"] is None, (
            f"unaligned prefix resume (lead={plan['lead']}) reached in _prefill_chunked_eager_tp, "
            f"but the unaligned-resume path is unqualified and unreachable under valid configuration;"
        )
        num_full = len(plan["replays"])
        tail_real = plan["tail"][1] if plan["tail"] else 0
        # Re-zero GDN at sequence START only; a resume keeps the restored prefix state.
        if plan["reset"]:
            self._reset_gdn_state_for_new_sequence()
        last_hidden = None
        if plan["lead"] is not None:
            assert False, (
                f"unaligned prefix resume (lead={plan['lead']}) reached in _prefill_chunked_eager_tp, "
                f"but the unaligned-resume path is unqualified and unreachable under valid configuration;"
            )
            lead_start, lead_len = plan["lead"]
            lead_logits = self.prefill_masked_bucket(
                token_ids[:, lead_start : lead_start + lead_len],
                page_table,
                actual_len=lead_len,
                chunk_start=lead_start,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, lead_start),
            )
            if num_full == 0 and tail_real == 0:
                return lead_logits
            ttnn.deallocate(lead_logits)
        for cs in plan["replays"]:
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            # Stage the vision buffers: each chunk splices its own slice of the packed vision rows
            # (vis_row_offset = image tokens before cs); a chunk with no image tokens clears the
            # mask so the where is the identity (host->device copy only, no compile).
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )
            # Full chunk: valid_len == bucket == chunk_size (no padding/masking).
            last_hidden = self._forward_prefill_chunk_masked_tp(
                token_ids[:, cs : cs + chunk_size], chunk_size, cs, page_table, chunk_size, flex_sdpa=flex_sdpa
            )
            ttnn.synchronize_device(self.device)
        if tail_real > 0:
            if last_hidden is not None:
                ttnn.deallocate(last_hidden)
            cs = plan["tail"][0]
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                flex_sdpa=flex_sdpa,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        # Exact multiple of chunk_size: logit from last full chunk.
        logits = self._masked_bucket_logits_tp(last_hidden, chunk_size, chunk_size)
        ttnn.deallocate(last_hidden)
        return logits

    def _prefill_traced_chunked_tp(
        self, token_ids, page_table, actual_len, plan, chunk_size, vision_tokens=None
    ):
        """TP traced chunk-outer prefill: replay the captured per-chunk trace
        (_forward_prefill_chunk_tp) for each FULL chunk, then run the partial tail through the
        masked bucket. The TP analog of the single-device loop in prefill_traced_chunked: each
        chunk's inputs are DMA'd into the REPLICATED persistent buffers via
        copy_host_to_device_tensor (no per-chunk program dispatch / device allocation — only one
        execute_trace per chunk), so GDN recurrent/conv + paged-KV state carry in place across
        replays and host pressure stays bounded at 128K. The tail's chunk_start>0 skips the GDN
        reset so the carried state continues. Returns logits [1, 1, vocab] at actual_len-1.

        `plan` (prefill_chunk_plan) carries the absolute chunk offsets, so a PREFIX RESUME
        replays the same parked trace starting at its own boundary. Nothing about the trace is
        offset-dependent: the chunk position enters the captured program as the runtime device
        tensor _chunk_start_idx_tensor (flexible SDPA), and tp.py pins q/k_chunk to 128 on that
        branch regardless of offset. Resuming compiles NOTHING -- which is what makes it safe
        beside a parked decode trace, and is the claim patch 0017's header got wrong."""
        assert plan["lead"] is None, (
            f"unaligned prefix resume (lead={plan['lead']}) reached in _prefill_traced_chunked_tp, "
            f"but the unaligned-resume path is unqualified and unreachable under valid configuration;"
        )
        block_size = get_block_size(self._paged_kv_caches)
        blocks_per_chunk = chunk_size // block_size
        rep = ttnn.ReplicateTensorToMesh(self.device)
        num_full = len(plan["replays"])
        tail_real = plan["tail"][1] if plan["tail"] else 0

        # Re-zero GDN once at a sequence START; carries across replays + tail (chunk_start>0
        # skips reset). A RESUME must not reset: prefill_prefix_slots has already restored the
        # snapshot and the conv carry into these exact buffers.
        if plan["reset"]:
            self._reset_gdn_state_for_new_sequence()
        self._ht_mark("gdn_reset")

        # Pad/clip page_table to captured width; write once (constant across chunks).
        buf_blocks = int(self._chunk_full_page_table_buf.shape[-1])
        if page_table.shape[1] < buf_blocks:
            page_table = torch.cat(
                [
                    page_table,
                    torch.zeros(page_table.shape[0], buf_blocks - page_table.shape[1], dtype=page_table.dtype),
                ],
                dim=1,
            )
        elif page_table.shape[1] > buf_blocks:
            page_table = page_table[:, :buf_blocks]
        pt_host = ttnn.from_torch(
            page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=None, mesh_mapper=rep
        )
        ttnn.copy_host_to_device_tensor(pt_host, self._chunk_full_page_table_buf)
        self._pt_mark("chunk_setup")
        self._ht_mark("chunk_setup")

        # Replay trace per full chunk. Host input-prep (from_torch tilize of cos/sin + DMA copies) and
        # device trace exec share cq_id=0, so the queue already ORDERS each chunk's copies AFTER the
        # prior chunk's trace reads them (no double-buffering needed). The old code did a full
        # synchronize_device every chunk, which forced the host to wait and serialized chunk N+1's CPU
        # prep behind chunk N's device exec. Instead we hold host-tensor refs alive (so their in-flight
        # DMAs aren't GC'd) and sync only every _SYNC_EVERY chunks — the host software-pipelines chunk
        # N+1's from_torch/tilize over chunk N's device exec. Periodic (not fully removed) sync bounds
        # in-flight queue depth so very long context (e.g. traced_128k = 64 chunks) can't overrun the
        # command queue. QWEN36_PREFILL_OVERLAP=0 restores the per-chunk sync.
        # An unaligned resume runs its run-up to the first chunk boundary eagerly (the trace's
        # token/page-table buffers are chunk_size wide, so replay can only begin on a boundary).
        if plan["lead"] is not None:
            assert False, (
                f"unaligned prefix resume (lead={plan['lead']}) reached in _prefill_traced_chunked_tp, "
                f"but the unaligned-resume path is unqualified and unreachable under valid configuration;"
            )
            lead_start, lead_len = plan["lead"]
            lead_logits = self.prefill_masked_bucket(
                token_ids[:, lead_start : lead_start + lead_len],
                page_table,
                actual_len=lead_len,
                chunk_start=lead_start,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, lead_start),
            )
            if num_full == 0 and tail_real == 0:
                return lead_logits
            ttnn.deallocate(lead_logits)

        _log_every = max(1, num_full // 4)
        _overlap = _tt_public_getenv("QWEN36_PREFILL_OVERLAP", "1") != "0"
        _SYNC_EVERY = 8 if _overlap else 1
        _host_refs = []  # keep host tensors alive until the next sync frees their DMAs
        for c, cs in enumerate(plan["replays"]):
            tok_host = ttnn.from_torch(
                token_ids[:, cs : cs + chunk_size].to(torch.int32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(tok_host, self._chunk_token_buf)

            csi_host = ttnn.from_torch(
                torch.tensor([cs], dtype=torch.int32),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(csi_host, self._chunk_start_idx_tensor)

            blk0 = cs // block_size
            cpt_host = ttnn.from_torch(
                page_table[:, blk0 : blk0 + blocks_per_chunk].contiguous(),
                dtype=ttnn.int32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=None,
                mesh_mapper=rep,
            )
            ttnn.copy_host_to_device_tensor(cpt_host, self._chunk_page_table_buf)

            cos_t, sin_t = self._rope_tp_cos_sin_torch(cs, chunk_size)
            cos_host = ttnn.from_torch(
                cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            sin_host = ttnn.from_torch(
                sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=None, mesh_mapper=rep
            )
            ttnn.copy_host_to_device_tensor(cos_host, self._chunk_cos_buf)
            ttnn.copy_host_to_device_tensor(sin_host, self._chunk_sin_buf)
            _host_refs += [tok_host, csi_host, cpt_host, cos_host, sin_host]

            # Stage the hidden-sharded vision buffers: each chunk splices its own slice of the
            # packed vision rows (vis_row_offset = image tokens before cs); a chunk with no image
            # tokens clears the mask so the captured where is the identity. host->device copy only
            # (no compile), so the parked trace is untouched.
            self._set_vision_merge(
                token_ids[:, cs : cs + chunk_size], vision_tokens, self._vis_row_offset_for(token_ids, cs)
            )

            ttnn.execute_trace(self.device, self._chunked_trace_id, cq_id=0, blocking=False)
            self._ht_mark("chunk_prep")
            if c + 1 < num_full:
                # QWEN36_MTP_WARM_TAIL_SPAN: an earlier chunk's window rows, gathered (same CQ) before the next replay
                # overwrites the trace output. The last chunk keeps its post-loop gather below. No-op when off.
                self._mtp_warm_span_seg(self._chunked_trace_output, cs, chunk_size)
                self._ht_mark("mtp_seg")

            # Bound in-flight depth; after a sync the completed DMAs' host tensors can be released.
            # QWEN36_PREFILL_TAIL_NOSYNC: not on the LAST chunk (its inputs move to self._ph_refs below), so the
            # post-loop warm gather and the tail's input prep overlap that chunk's replay at 8/16/... chunks too.
            if (c + 1) % _SYNC_EVERY == 0 and not (self._ph_nosync_on and c + 1 == num_full):
                self._ht_mark("pre_drain")
                ttnn.synchronize_device(self.device)
                self._ht_mark("drain_sync")
                _host_refs.clear()
            if (c + 1) % _log_every == 0:
                logger.info(f"[TP chunk-replay] {c + 1}/{num_full} chunks")

        # Drain any still-in-flight chunk DMAs before returning, so the loop's host input tensors
        # (_host_refs) are not GC'd while a non-blocking execute_trace is still reading them — a
        # use-after-free that hangs the device. When num_full < _SYNC_EVERY the loop never synced,
        # so the no-tail return below (which issues no further blocking work) would otherwise race.
        if _host_refs and self._ph_nosync_on:
            # QWEN36_PREFILL_TAIL_NOSYNC: no drain; the inputs stay referenced until prefill_paged_slots' readback.
            self._ph_refs.add(_host_refs)
            _host_refs = []
        if _host_refs:
            self._ht_mark("pre_drain")
            ttnn.synchronize_device(self.device)
            self._ht_mark("drain_sync")
            _host_refs.clear()
        self._pt_mark("chunks")
        self._ht_mark("chunks_drained")
        # QWEN36_MTP_WARM_TAIL: the chunk trace's output holds the LAST full chunk only, and the tail's replay below
        # may reuse its addresses, so its window rows are gathered into the warm pass's persistent buffer now.
        if plan["replays"]:
            self._mtp_warm_seg(self._chunked_trace_output, plan["replays"][-1], chunk_size, chunk_size)
            self._ht_mark("mtp_seg")

        # Tail via masked bucket, or _masked_bucket_logits_tp if no tail (TP 4D hidden).
        if tail_real > 0:
            cs = plan["tail"][0]
            return self.prefill_masked_bucket(
                token_ids[:, cs:actual_len],
                page_table,
                actual_len=tail_real,
                chunk_start=cs,
                vision_tokens=vision_tokens,
                vis_row_offset=self._vis_row_offset_for(token_ids, cs),
            )
        if self._phase_timer is None and self._host_timer is None:
            return self._masked_bucket_logits_tp(self._chunked_trace_output, chunk_size, chunk_size)
        lg = self._masked_bucket_logits_tp(self._chunked_trace_output, chunk_size, chunk_size)
        self._pt_mark("last_logits")
        self._ht_mark("logits_sel")
        return lg

    def reset_state(self, batch_size=None):
        """Reset layer state for a new sequence (eager/pre-trace path; trace uses _reset_dn_state_inplace)."""
        for layer in self.layers:
            if layer.is_full_attention:
                layer.attention.reset_cache()
            else:
                layer.attention.reset_state(batch_size)

    # ----------------------------------------------------------------------- #
    # QWEN36_PREFILL_RESET_TRACE (pp/host, tt/prefill_host.py)
    # ----------------------------------------------------------------------- #
    def _gdn_reset_targets(self, dn):
        """(zero source, target) pairs of one GDN layer's reset_state_inplace, in its order."""
        pairs = [(dn._zero_conv0, cs) for cs in dn.conv_states]
        pairs.append((dn._zero_rec, dn.rec_state))
        pairs.append((dn._zero_conv_carry, dn.conv_carry))
        return pairs

    def _gdn_reset_ident(self):
        return [(dn, tuple(t for pair in self._gdn_reset_targets(dn) for t in pair)) for dn in self._gdn_layers()]

    def _reset_trace_matches(self):
        cap = self._reset_trace_ident
        if cap is None:
            return False
        now = self._gdn_reset_ident()
        return len(cap) == len(now) and all(
            a[0] is b[0] and len(a[1]) == len(b[1]) and all(x is y for x, y in zip(a[1], b[1])) for a, b in zip(cap, now)
        )

    def _gdn_reset_trace_body(self, mustfail=False):
        dns = self._gdn_layers()
        for i, dn in enumerate(dns):
            for j, (z, t) in enumerate(self._gdn_reset_targets(dn)):
                if mustfail and i == len(dns) - 1 and t is dn.rec_state:
                    continue  # MUSTFAIL: the last layer's recurrent state is never zeroed
                ttnn.copy(z, t)

    def _replay_gdn_reset_trace(self):
        for dn in self._gdn_layers():
            # reset_state_inplace's host-side part: drop a chunk-outer batched conv carry (no device work).
            if getattr(dn, "_batched_conv_carry", None) is not None:
                ttnn.deallocate(dn._batched_conv_carry)
                dn._batched_conv_carry = None
        ttnn.execute_trace(self.device, self._reset_trace_id, cq_id=0, blocking=False)
        self._reset_trace_replays += 1
        if self._reset_trace_replays == 1 or self._reset_trace_replays % 256 == 0:
            logger.info(f"QWEN36_PREFILL_RESET_TRACE replay: replays={self._reset_trace_replays}")

    def _capture_gdn_reset_trace(self, device):
        """Capture the 240-copy GDN reset of the CURRENTLY BOUND state (the persistent B=1 prefill scratch at the
        warmup) as one trace, then self-test it on both chips (fail closed)."""
        if not self._ph.reset_trace:
            return
        dns = self._gdn_layers()
        for dn in dns:
            if dn.conv_states is None or dn._zero_conv0 is None or dn._zero_rec is None or dn._zero_conv_carry is None:
                raise RuntimeError("QWEN36_PREFILL_RESET_TRACE: GDN state / zero sources not allocated at capture; refusing")
        self._reset_gdn_state_for_new_sequence()  # eager: compiles (already compiled) and zeroes before the capture
        ttnn.synchronize_device(device)
        before = self._trace_region_bytes(device)
        tid = ttnn.begin_trace_capture(device, cq_id=0)
        self._gdn_reset_trace_body(mustfail=self._ph.reset_trace_mf)
        ttnn.end_trace_capture(device, tid, cq_id=0)
        after = self._trace_region_bytes(device)
        self._reset_trace_id = tid
        self._reset_trace_ident = self._gdn_reset_ident()
        n_ops = sum(len(self._gdn_reset_targets(dn)) for dn in dns)
        delta = f"+{(after - before) / (1 << 20):.2f} MiB" if None not in (before, after) else "n/a"
        self._gdn_reset_trace_selftest(device)
        logger.info(
            f"QWEN36_PREFILL_RESET_TRACE engaged: GDN reset ({len(dns)} layers, {n_ops} copies) replayed as one "
            f"trace per request (TRACE region {delta})"
            + (" MUSTFAIL: last layer's recurrent state not zeroed" if self._ph.reset_trace_mf else "")
        )

    def _gdn_reset_trace_selftest(self, device):
        """Both chips, fail closed: tt/prefill_host.reset_selftest over every GDN layer's reset targets."""
        pairs = [(li, z, t) for li, dn in enumerate(self._gdn_layers()) for z, t in self._gdn_reset_targets(dn)]

        def _replay():
            self._replay_gdn_reset_trace()
            self._reset_trace_replays = 0

        n_t = phl.reset_selftest(ttnn, self.mesh_device, device, pairs, _replay, self.num_devices,
                                 mustfail=self._ph.reset_trace_mf)
        logger.info(
            f"QWEN36_PREFILL_RESET_TRACE self-test exact: {n_t} targets on {self.num_devices} devices read junk before "
            f"the replay (wrong-input branch) and exactly zero after it"
        )

    def _reset_gdn_state_for_new_sequence(self):
        """Zero GDN recurrent+conv at sequence start.

        Trace capture runs forward twice; GDN state is non-idempotent. Must re-zero before each
        real sequence. In-place buffers (_chunk_inplace_state) use _reset_dn_state_inplace."""
        if self.num_devices > 1 and self._reset_trace_id is not None and self._ph_req and self._ph.reset_trace:
            # QWEN36_PREFILL_RESET_TRACE: request time only, and only on the tensors the trace baked.
            if self._reset_trace_matches():
                self._replay_gdn_reset_trace()
                return
        if self.num_devices > 1:
            # TP: reset_state_inplace preserves decode-trace baked addresses.
            for layer in self.layers:
                if not layer.is_full_attention:
                    layer.attention.reset_state_inplace()
            return
        inplace = any(
            (not l.is_full_attention) and getattr(l.attention, "_chunk_inplace_state", False) for l in self.layers
        )
        if inplace:
            self._reset_dn_state_inplace()
        else:
            self.reset_state(batch_size=1)

    def _reset_dn_state_inplace(self):
        """Zero DN state in place via pre-allocated zero buffers (trace addresses fixed)."""
        assert self._dn_zero_recurrent is not None, "Call _init_dn_zero_buffers first"
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            ttnn.copy(self._dn_zero_recurrent, dn.recurrent_state)
            ttnn.copy(self._dn_zero_conv, dn.fused_conv_state)
            # split_conv_state rebuilt lazily on first decode.
            if dn.split_conv_state is not None:
                for buf in dn.split_conv_state:
                    ttnn.deallocate(buf)
                dn.split_conv_state = None

    def _init_dn_zero_buffers(self):
        """Allocate shared zero buffers for DN recurrent and conv shapes."""
        if self._dn_zero_recurrent is not None:
            return
        # First DN layer defines shared zero-buffer shapes.
        first_dn = next(layer.attention for layer in self.layers if not layer.is_full_attention)
        rec_shape = list(first_dn.recurrent_state.shape)
        conv_shape = list(first_dn.fused_conv_state.shape)
        self._dn_zero_recurrent = ttnn.zeros(
            rec_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self._dn_zero_conv = ttnn.zeros(
            conv_shape,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def set_paged_kv_caches(self, kv_caches):
        """Attach paged KV caches to the 8 attention layers."""
        self._paged_kv_caches = kv_caches
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            k_cache, v_cache = kv_caches[cache_idx]
            self.layers[layer_idx].attention.set_paged_kv_cache(k_cache, v_cache)
        self._dnorm_fuse_selftest()
        self._gdn_tail_selftest()
        self._gdn_qrm_selftest()
        self._rs_pkt4_selftest()
        self._rs_bank_selftest()
        # QWEN36_ARGMAX_FOLD (argmax_fold.py): self-test the folded greedy combine at every (B, per) a served boot's
        # argmax calls run, BEFORE any decode (the compile warmup's plain-decode buckets B=1,2,4,8 at vocab/TP run before
        # the MTP compile; the 10-16 card boot refused on their fallback banner). Fails closed.
        if argmax_fold.enabled():
            from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mbh

            B_max = int(self.args.max_batch_size)
            dv = getattr(self, "_draft_vocab", None)
            cells = argmax_fold.served_cells(B_max, self.args.vocab_size // self.num_devices,
                                             draft_per=(int(dv.per) if dv is not None else None),
                                             served=_mbh.served_verify_cells(B_max))
            argmax_fold.ensure_selftest(self.mesh_device, self.tt_ccl, self.args.ccl_topology(), cells=cells)

    def _dnorm_fuse_selftest(self):
        """QWEN36_PREFILL_DNORM_FUSE (dnorm_fuse.py): self-test the fused prefill distributed RMSNorm, eagerly and before
        any prefill trace, at every row count a served prefill can hand the decoder norms (the chunk, the masked
        buckets, and Bg x bucket groups up to one tile row per core). Fails closed; untested shapes fall back."""
        from models.demos.blackhole.qwen36.tt import dnorm_fuse

        if not dnorm_fuse.enabled() or self.num_devices < 2:
            return
        cap = 32 * int(self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y)
        Ms = {prefill_chunk_size()} | {bg * b for b in prefill_mask_buckets() for bg in range(1, 9)}
        Ms = sorted(m for m in Ms if m <= cap)
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        od = _tpc.prefill_norm_dtype() or ttnn.bfloat16  # the attn/ff norms' output dtype (nhp: bf8)
        L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
        c = prefill_chunk_size()
        # every M at the attention norm's placement (L1); the chunk also at the ff norm's (DRAM) and in bf16 (final norm)
        cells = [(m, od, L1) for m in Ms] + [(c, od, DR)] + ([(c, ttnn.bfloat16, DR)] if od != ttnn.bfloat16 else [])
        n = self.layers[0].attention_norm
        n = getattr(n, "norm", n)  # DistributedNorm wraps the RMSNorm whose _distributed_rmsnorm is hooked
        dnorm_fuse.ensure_selftest(n, cells, self.args.dim // self.num_devices)
        # QWEN36_PREFILL_DNORM_ADD (layer.py _prefill_add_norm): h = x + attn_out folded into ffn_norm (DRAM output) at
        # every M, for both dtypes the attention output can arrive in.
        dnorm_fuse.ensure_selftest_add(n, [(m, od, DR, bd) for m in Ms for bd in (ttnn.bfloat16, ttnn.bfloat8_b)],
                                       self.args.dim // self.num_devices)

    def _xadd_next(self, i):
        """QWEN36_PREFILL_DNORM_XADD: the layer whose attention norm layer i folds its closing residual add into."""
        from models.demos.blackhole.qwen36.tt import dnorm_fuse

        if i + 1 >= len(self.layers) or not dnorm_fuse.xadd_enabled():
            return None
        return self.layers[i + 1]

    def _xadd_kw(self, i):
        """`fold_next=` only when xadd is on: layers without the decode-levers overlay (the `prod` arm) take no such
        argument."""
        nxt = self._xadd_next(i)
        return {} if nxt is None else {"fold_next": nxt}

    def _gdn_tail_selftest(self):
        """QWEN36_PREFILL_GDN_TAIL (gdn_tail.py): self-test the fused GDN prefill norm+relayout and out-gate on both
        chips, eagerly and before any prefill trace, at every T a served GDN prefill can run (the chunk and the masked
        bucket groups). Fails closed; an untested T falls back to today's ops (logged)."""
        from models.demos.blackhole.qwen36.tt import gdn_tail

        if not gdn_tail.enabled():
            return
        cap = 32 * int(self.mesh_device.compute_with_storage_grid_size().x * self.mesh_device.compute_with_storage_grid_size().y)
        Ts = {prefill_chunk_size()} | {bg * b for b in prefill_mask_buckets() for bg in range(1, 9)}
        Nv = int(self.args.gdn_nv_tp)
        Dv = int(self.args.gdn_value_dim_tp) // Nv
        gdn_tail.ensure_selftest(self.mesh_device, [(t, Nv, Dv) for t in sorted(Ts) if t <= max(cap, prefill_chunk_size())])

    def _rs_pkt4_selftest(self):
        """QWEN36_PREFILL_RS_PKT4 (rs_pkt4.py): the 4-tile-packet line reduce-scatter writer (a mounted kernel), flag and
        kernel checked against each other and self-tested on both chips, eagerly and before any prefill trace, through
        the served prefill_all_reduce at T = 32 / 128 / 160 / chunk. Fails closed. TP only."""
        if self.num_devices <= 1 or self.tt_ccl is None or _tt_public_getenv("QWEN36_PREFILL_RS_PKT4", "0") == "0":
            return
        from models.demos.blackhole.qwen36.tt import rs_pkt4

        rs_pkt4.ensure_selftest(self.mesh_device, self.tt_ccl, self.args.ccl_topology(),
                                [32, 128, 160, prefill_chunk_size()], int(self.args.dim))

    def _rs_bank_selftest(self):
        """QWEN36_PREFILL_RS_BANK (rs_bank.py): the bank-contiguous line reduce-scatter reader+writer (mounted kernels),
        flag and kernels checked against each other and self-tested on both chips, eagerly and before any prefill
        trace, through the served prefill_all_reduce at T = 32 / 128 / 160 / chunk. Fails closed. TP only; off = no-op
        (the rs_pkt4 pattern: nothing is imported or read)."""
        if self.num_devices <= 1 or self.tt_ccl is None or _tt_public_getenv("QWEN36_PREFILL_RS_BANK", "0") == "0":
            return
        from models.demos.blackhole.qwen36.tt import rs_bank

        rs_bank.ensure_selftest(self.mesh_device, self.tt_ccl, self.args.ccl_topology(),
                                [32, 128, 160, prefill_chunk_size()], int(self.args.dim))

    def _gdn_qrm_selftest(self):
        """QWEN36_PREFILL_GDN_QKV_RM (gdn_qrm.py): the fused qkv slice + untilize, self-tested on both chips, eagerly
        and before any prefill trace, at the T a full-chunk kda GDN prefill can run. Fails closed; an untested T
        falls back to slice + to_layout (logged)."""
        from models.demos.blackhole.qwen36.tt import gdn_qrm

        if not gdn_qrm.enabled():
            return
        Ts = {prefill_chunk_size()} | {bg * b for b in prefill_mask_buckets() for bg in range(1, 9)}
        C, W = int(self.args.gdn_qkv_dim_tp), int(self.args.gdn_qkvzab_dim_tp)
        gdn_qrm.ensure_selftest(self.mesh_device, [(t, C, W) for t in sorted(Ts) if t <= prefill_chunk_size()])

    def allocate_kv_caches(self, kv_cache_shape, dtype, batch_size=1):
        """Allocate caches for all 32 layers. Returns only the attention KV caches (for vLLM)."""
        assert self._deltanet_external_states is None, "allocate_kv_caches already called; deallocate first"
        # QWEN36_PREFILL_BUCKET_TRACE: scratch block for the traced bucket's fixed-width KV-fill page table = the
        # last physical block, which qwen36_vllm.allocate_kv_cache adds on top of the scheduler's pool.
        if self._mb_trace_buckets:
            self._pad_kv_block = int(_tt_public_getenv("QWEN36_PREFILL_BUCKET_PAD_BLOCK", int(kv_cache_shape[0]) - 1))
        # K and V get INDEPENDENT dtypes (QWEN36_KV_DTYPE="<K>/<V>"); QWEN_SDPA_BF8=1
        # still means bf8/bf8, so every historical arm keeps its exact meaning.
        # tp_common.log_kv_dtype_banner is the ONLY runtime evidence of the arm: vLLM's
        # "GPU KV cache size" is a token count derived from max_model_len x max_num_seqs
        # and is IDENTICAL across KV dtypes (measured 2026-09-13, both arms logging
        # 278,784 at the same shape).
        from models.demos.blackhole.qwen36.tt import tp_common as tpc

        k_dtype, v_dtype, kv_src = tpc.resolve_kv_dtypes(dtype)
        tpc.log_kv_dtype_banner(k_dtype, v_dtype, kv_src, dtype, kv_cache_shape)
        self._kv_k_dtype, self._kv_v_dtype = k_dtype, v_dtype
        if self.num_devices > 1:
            return self._allocate_kv_caches_tp(kv_cache_shape, (k_dtype, v_dtype), batch_size)

        kv_caches = []
        for idx in self._attention_layer_indices:
            k_cache = ttnn.zeros(kv_cache_shape, dtype=k_dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            v_cache = ttnn.zeros(kv_cache_shape, dtype=v_dtype, layout=ttnn.TILE_LAYOUT, device=self.device)
            kv_caches.append([k_cache, v_cache])
        self.set_paged_kv_caches(kv_caches)

        self._deltanet_external_states = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                rec = ttnn.from_torch(
                    torch.zeros(batch_size, dn.num_v_heads, dn.head_k_dim, dn.head_v_dim, dtype=torch.bfloat16),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                conv = ttnn.from_torch(
                    torch.zeros(
                        batch_size,
                        dn.conv_kernel_size - 1,
                        dn.cfg.q_dim + dn.cfg.k_dim + dn.cfg.v_dim,
                        dtype=torch.bfloat16,
                    ),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=self.device,
                )
                dn.set_external_state(rec, conv)
                self._deltanet_external_states.append((rec, conv))

        return kv_caches

    def free_kv_caches(self):
        """Release KV caches + GDN state for a fresh generation run."""
        if self._deltanet_external_states is None:
            return
        if getattr(self, "_chunked_trace_id", None) is not None:
            ttnn.release_trace(self.device, self._chunked_trace_id)
            self._chunked_trace_id = None
        for rec, conv in self._deltanet_external_states:
            ttnn.deallocate(rec)
            ttnn.deallocate(conv)
        self._deltanet_external_states = None
        if getattr(self, "_paged_kv_caches", None) is not None:
            for k_cache, v_cache in self._paged_kv_caches:
                ttnn.deallocate(k_cache)
                ttnn.deallocate(v_cache)
            self._paged_kv_caches = None

    def _allocate_kv_caches_tp(self, kv_cache_shape, dtype, batch_size):
        """TP paged KV allocation (B=1). Replicated per device; GDN self-manages state.

        ``dtype`` is either one dtype for both caches, or a ``(k_dtype, v_dtype)`` pair.
        """
        k_dtype, v_dtype = dtype if isinstance(dtype, tuple) else (dtype, dtype)
        from models.demos.blackhole.qwen36.tt import kv_devinit as kvd

        if kvd.enabled():
            # QWEN36_KV_DEVICE_INIT (kv_devinit.py): the same allocations in the same order, zero-filled on device
            # instead of torch.zeros + host bfp packing + upload (65 s of a BEST boot). Self-tested below, fail closed.
            kvd.log_engaged(k_dtype, v_dtype, kv_cache_shape, 2 * len(self._attention_layer_indices), "attention")
            kv_caches = [
                [
                    kvd.alloc_filled(kv_cache_shape, k_dtype, self.device, ttnn.DRAM_MEMORY_CONFIG),
                    kvd.alloc_filled(kv_cache_shape, v_dtype, self.device, ttnn.DRAM_MEMORY_CONFIG),
                ]
                for _ in self._attention_layer_indices
            ]
            n = len(kv_caches)
            picks = sorted({0, n // 2, n - 1})
            ids = self._attention_layer_indices
            sample = [(f"L{ids[i]}.{kv}", kv_caches[i][j]) for i in picks for j, kv in enumerate("KV")]
            logger.info(f"QWEN36_KV_DEVICE_INIT self-test exact (attention): {kvd.selftest(sample)}")
            return self._finish_kv_caches_tp(kv_caches, batch_size)

        def _mk(dt):
            return ttnn.as_tensor(
                torch.zeros(kv_cache_shape, dtype=torch.bfloat16),
                device=self.device,
                dtype=dt,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )

        kv_caches = [[_mk(k_dtype), _mk(v_dtype)] for _ in self._attention_layer_indices]
        return self._finish_kv_caches_tp(kv_caches, batch_size)

    def _finish_kv_caches_tp(self, kv_caches, batch_size):
        self.set_paged_kv_caches(kv_caches)  # binds via TPAttention.set_paged_kv_cache
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.B = batch_size
                layer.attention.reset_state()
                # Fixed-address GDN state for decode trace compatibility.
                layer.attention._stable_state = True
        # Marker for re-entry assert; TP GDN state lives in module, not external buffers.
        self._deltanet_external_states = []
        return kv_caches

    def _prefill_paged_tp(self, token_ids, page_table, valid_len=None, vision_tokens=None, gdn_collect=False):
        """TP (num_devices>1) paged prefill, B=1. Mirrors the demo prefill_tp but routes
        the full-attention layers through the paged KV cache (forward_prefill_paged) so
        decode can read it via page_table. GDN layers capture their recurrent/conv state
        as in the demo. Returns logits [1, 1, vocab] at position valid_len-1.
        """
        B, T = token_ids.shape
        assert B == 1, "TP prefill is single-sequence (B=1); batched serving prefills one user at a time"
        vlen = valid_len or T
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text).
        self._build_request_rope(token_ids[:, :vlen], vision_tokens)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        page_table_tt = ttnn.from_torch(pt_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
        tok = ttnn.from_torch(
            token_ids.to(torch.int32),
            dtype=ttnn.uint32,
            device=self.device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
        )
        x = self.embd(tok)
        x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
        x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, T)
        rep = ttnn.ReplicateTensorToMesh(self.device)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        for layer in self.layers:
            x = layer.forward(
                x,
                cos=cos,
                sin=sin,
                mode="prefill",
                chunk_size=self.args.gdn_chunk_size,
                valid_len=vlen,
                page_table=page_table_tt,
                chunk_page_table=page_table_tt,
                chunk_start_idx=0,
                gdn_collect=gdn_collect,
            )
        x = self.norm(x, mode=Mode.PREFILL)
        x_last = x[:, :, vlen - 1 : vlen, :]
        logits = self._lm_head(x_last)
        ttnn.deallocate(x)
        return ttnn.reshape(logits, (1, 1, logits.shape[-1]))

    def prefill_paged_peruser(self, token_ids_list, page_table, valid_lens=None):
        """Batched per-user TP prefill (the batched serving contract).

        Prefills B users into ONE shared paged KV cache + the batched GDN decode state, one user at
        a time. Each user's full-attention layers fill their own blocks via the per-user page-table
        row; each GDN layer collects that user's from-scratch state, stitched into row u of the
        batched decode buffers by finalize_pending(). Call allocate_kv_caches(batch_size=B) first.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, max_blocks_per_seq] int32 — row u = user u's blocks.
        valid_lens:      optional list of B ints (real token counts); defaults to each T_u.
        Returns:         list of B ttnn logits [1, 1, vocab_size] (one per user, at valid_len-1).
        """
        assert self.num_devices > 1, "prefill_paged_peruser is the TP (num_devices>1) path"
        B = len(token_ids_list)
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert page_table_torch.shape[0] == B, "page_table must have one row per user"

        # Fresh GDN per-user accumulators (cleared at the end by finalize_pending).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention._pending = []

        logits = []
        for u in range(B):
            vlen = valid_lens[u] if valid_lens is not None else None
            # Each user's prefill is the validated B=1 paged path, pointed at user u's blocks.
            lg = self._prefill_paged_tp(
                token_ids_list[u], page_table_torch[u : u + 1], valid_len=vlen, gdn_collect=True
            )
            logits.append(lg)

        # Stitch every user's collected GDN state into the batched decode buffers (row u = user u).
        for layer in self.layers:
            if not layer.is_full_attention:
                layer.attention.finalize_pending()
        return logits

    def _alloc_gdn_scratch_b(self, bg):
        """Like _alloc_gdn_scratch_b1 but for a group of `bg` users: allocate a dedicated
        [bg,...] GDN state on every GDN layer, distinct from the real batched [B,...] decode
        buffer. Returns the prior batched bindings for _restore_gdn_batched. Used by the grouped
        batched prefill (forward_prefill_batched writes [bg,...] into these in place)."""
        prev = []
        for layer in self.layers:
            if layer.is_full_attention:
                continue
            dn = layer.attention
            prev.append((dn, dn.B, dn.rec_state, dn.conv_states, dn.conv_carry, dn._zero_conv0, dn._stable_state))
            dn.B = bg
            dn.reset_state()  # builds rec_state [bg,Nv,Dk,Dv], conv_states[*] [1,bg,D], carry, zero0
            dn._stable_state = True  # forward_prefill_batched writes state in place under this flag
        return prev

    def _assemble_groups_gdn_dev(self, group_rec_dev, group_conv_dev):
        """Assemble per-GROUP GDN states (each already batched [bg,...] on device, from
        forward_prefill_batched) into the full [B,...] batched decode buffers via device-side
        concat — no host round-trip. rec: concat groups along dim 0 -> [B,Nv,Dk,Dv]; conv_states[m]:
        concat groups along dim 1 -> [1,B,D]. The batched GDN bindings MUST already be rebound
        (writes in place under _stable_state). Row u == user u because groups are contiguous
        (group g = users [g*group_size : ...])."""
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        ng = len(group_rec_dev)
        for li, dn in enumerate(dn_layers):
            rec_full = ttnn.concat([group_rec_dev[g][li] for g in range(ng)], dim=0)  # [B, Nv, Dk, Dv]
            rec_src = rec_full if rec_full.dtype == dn.rec_state.dtype else ttnn.typecast(rec_full, dn.rec_state.dtype)
            ttnn.copy(rec_src, dn.rec_state)
            if rec_src is not rec_full:
                ttnn.deallocate(rec_src)
            ttnn.deallocate(rec_full)
            for g in range(ng):
                ttnn.deallocate(group_rec_dev[g][li])
            for m in range(dn.K):
                conv_full = ttnn.concat([group_conv_dev[g][li][m] for g in range(ng)], dim=1)  # [1, B, D]
                ttnn.copy(conv_full, dn.conv_states[m])
                ttnn.deallocate(conv_full)
                for g in range(ng):
                    ttnn.deallocate(group_conv_dev[g][li][m])

    def prefill_paged_grouped(self, token_ids_list, page_table, valid_lens=None, group_size=4):
        """Grouped batched SHORT-prompt prefill (single-pass, every valid_len <= one GDN bucket):
        process users in groups of <= group_size through ONE hybrid forward per group instead of B
        sequential B=1 forwards. Within a group the GDN layers run BATCHED (forward_prefill_batched,
        per-row valid_len masking — bit-exact per user, see test_gdn_tp_batched_prefill) and the
        full-attention layers run PER-USER (attention prefill is B=1 only). Groups of <=4 respect the
        GDN kernel cap BH=B*Nv_tp<=32. Numerically the batched GDN + per-user attention is the same
        math as prefill_paged_peruser, so per-user output (incl. DIFFERENT prompts/lengths) is
        unchanged; it just amortizes the underutilized GDN over the group.

        token_ids_list: list of B torch.Tensor [1, T_u] (lengths may differ).
        page_table:      torch.Tensor [B, blocks_per_user] int32 (row u = user u's blocks).
        valid_lens:      optional list of B ints; defaults to each T_u. Every valid_len MUST be <=
                         the derived bucket (a single GDN chunk-set); callers route longer prompts
                         to the chunked path.
        Returns:         list of B ttnn logits [1, 1, vocab] (prefill_paged_peruser contract).
        """
        assert self.num_devices > 1, "prefill_paged_grouped is the TP (num_devices>1) path"
        assert self._paged_kv_caches is not None, "Call allocate_kv_caches first"
        B = len(token_ids_list)
        pt_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        assert pt_torch.shape[0] == B, "page_table must have one row per user"
        vlens = list(valid_lens) if valid_lens is not None else [int(t.shape[1]) for t in token_ids_list]

        gdn_chunk = self.args.gdn_chunk_size
        block_size = get_block_size(self._paged_kv_caches)
        # Common bucket for the group forward: round the longest prompt up to a GDN-chunk multiple.
        bucket = max(gdn_chunk, ((max(vlens) + gdn_chunk - 1) // gdn_chunk) * gdn_chunk)
        assert all(v <= bucket for v in vlens), "every valid_len must fit the single-pass bucket"

        # The fused chunk_gated_delta_rule op caps the group by its SCAN, which maps one (head,
        # v-block) row per core: BH = B*Nv_tp must stay <= the compute grid (~96-104 cores on P150).
        # With Nv_tp=12 that's B <= 8, and — unlike the old gated_delta_attn_seq kernel — it is
        # bucket-independent (SCAN L1 is state-sized, not chunk-count-sized). Validated bit-exact vs
        # per-user at B=8 for bucket 128 and 256 (test_gdn_fused_batch: ceiling + large-group). Buckets
        # >256 aren't produced here (callers route T>256 to per-user), so cap them at 1 defensively.
        # QWEN36_PREFILL_GROUP=N overrides the long-bucket cap (Bg=N for long buckets); default keeps it.
        gdn_max_bg = 8 if bucket <= 2 * gdn_chunk else 1
        _pg = getattr(self, "_prefill_group", None)
        if _pg is not None and bucket > 2 * gdn_chunk:
            gdn_max_bg = _pg
            logger.info(f"QWEN36_PREFILL_GROUP engaged: Bg={_pg} bucket={bucket}")
        group_size = max(1, min(group_size, gdn_max_bg))
        dn_layers = [layer.attention for layer in self.layers if not layer.is_full_attention]
        rep = ttnn.ReplicateTensorToMesh(self.device)
        # cos/sin for absolute positions [0, bucket) — shared by all users (single pass from pos 0).
        cos_t, sin_t = self._rope_tp_cos_sin_torch(0, bucket)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device, mesh_mapper=rep)
        csi = ttnn.from_torch(
            torch.tensor([0], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )

        group_rec_dev, group_conv_dev = [], []
        host_logits = [None] * B
        comp = ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)
        for g0 in range(0, B, group_size):
            grp = list(range(g0, min(g0 + group_size, B)))
            Bg = len(grp)
            prev = self._alloc_gdn_scratch_b(Bg)
            try:
                # Batched embedding: [1, Bg, bucket, dim] (pad each user's tokens to the bucket).
                tok_bg = torch.zeros(Bg, bucket, dtype=torch.int32)
                for i, u in enumerate(grp):
                    t = token_ids_list[u][0, : vlens[u]].to(torch.int32)
                    tok_bg[i, : t.shape[0]] = t
                tok = ttnn.from_torch(tok_bg, dtype=ttnn.uint32, device=self.device, mesh_mapper=rep)
                x = self.embd(tok)  # [Bg, bucket, d]
                d = x.shape[-1]
                # Canonical residual-stream shape [1, 1, Bg*bucket, d] (dim1==1) so the framework
                # norm / MLP / residual add see the SAME layout as the validated per-user path
                # (a [1,Bg,bucket,d] shape trips "invalid subtile broadcast" in the norm/residual).
                # Reshaped to [Bg, bucket, d] only for the batched GDN, and split to [1,Bg,bucket,d]
                # to slice each user for the per-user attention. Row order is user-major (user u owns
                # rows [u*bucket : (u+1)*bucket]), matching the group state assembly.
                x = ttnn.reshape(x, (1, 1, Bg * bucket, d))
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(tok)
                # QWEN36_PREFILL_GROUP_MUSTFAIL: swap the page-table rows between the two users of a
                # Bg=2 group (each user's KV lands in the other's blocks); logits MUST differ.
                _mf_rows = list(range(Bg))
                if getattr(self, "_prefill_group_mustfail", False) and Bg == 2:
                    _mf_rows = [1, 0]
                    logger.warning("QWEN36_PREFILL_GROUP_MUSTFAIL engaged: page-table rows swapped (logits must differ)")
                full_pts, chunk_pts = [], []
                for i, u in enumerate(grp):
                    row = pt_torch[grp[_mf_rows[i]] : grp[_mf_rows[i]] + 1].contiguous()
                    full_pts.append(
                        ttnn.from_torch(row, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
                    )
                    blkN = num_blocks_in_seq(vlens[u], block_size)
                    chunk_pts.append(
                        ttnn.from_torch(
                            row[:, :blkN].contiguous(),
                            dtype=ttnn.int32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                            device=self.device,
                        )
                    )

                for layer in self.layers:
                    # The DistributedNorm all-gathers to the FULL hidden dim for the module (dn), so
                    # attn_in's last dim is the full dim (!= d, the fractured residual-stream dim).
                    attn_in = layer.attention_norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                    full = attn_in.shape[-1]
                    if layer.is_full_attention:
                        attn_in_b = ttnn.reshape(attn_in, (1, Bg, bucket, full))  # split users for slicing
                        outs = []
                        for i, u in enumerate(grp):
                            xi = ttnn.reshape(attn_in_b[:, i : i + 1, :, :], (1, 1, bucket, full))
                            oi = layer.attention.forward_prefill_paged(
                                xi,
                                cos,
                                sin,
                                full_pts[i],
                                chunk_page_table=chunk_pts[i],
                                chunk_start_idx=0,
                                chunk_start_idx_tensor=csi,
                                user_id=0,  # per-user page tables are single-row; blocks route via values
                            )
                            ttnn.deallocate(xi)  # per-user slice copy
                            outs.append(ttnn.reshape(oi, (1, 1, bucket, oi.shape[-1])))
                        # Concat user outputs along the seq dim -> [1, 1, Bg*bucket, d_out] (user-major).
                        attn_out = ttnn.concat(outs, dim=2) if Bg > 1 else outs[0]
                        for o in outs:
                            if o is not attn_out:
                                ttnn.deallocate(o)
                    else:
                        # Batched GDN over the group (per-row valid_len masking, from scratch).
                        gdn_in = ttnn.reshape(attn_in, (Bg, bucket, full))
                        attn_out = layer.attention.forward_prefill_batched(
                            gdn_in, chunk_size=gdn_chunk, valid_lens=[vlens[u] for u in grp], carry=False
                        )  # [1, Bg, bucket, d_out]
                        attn_out = ttnn.reshape(attn_out, (1, 1, Bg * bucket, attn_out.shape[-1]))
                    ttnn.deallocate(attn_in)
                    h = ttnn.add(x, attn_out)  # both [1, 1, Bg*bucket, d]
                    ttnn.deallocate(x)
                    ttnn.deallocate(attn_out)
                    ff_in = layer.ffn_norm(h, mode=Mode.PREFILL)
                    ff_out = layer.feed_forward.forward(ff_in)
                    ttnn.deallocate(ff_in)
                    x = ttnn.add(h, ff_out)
                    ttnn.deallocate(h)
                    ttnn.deallocate(ff_out)

                # Final norm + per-user next-token logit at valid_len-1, read to host immediately.
                xn = self.norm(x, mode=Mode.PREFILL)  # [1, 1, Bg*bucket, full]
                ttnn.deallocate(x)
                xn_b = ttnn.reshape(xn, (1, Bg, bucket, xn.shape[-1]))
                for i, u in enumerate(grp):
                    x_last = xn_b[:, i : i + 1, vlens[u] - 1 : vlens[u], :]  # [1,1,1,full] (slice copy)
                    lg = ttnn.linear(x_last, self.lm_head_weight)
                    ttnn.deallocate(x_last)
                    host_logits[u] = (
                        ttnn.to_torch(lg, mesh_composer=comp).reshape(1, 1, -1)[:, :, : self.args.vocab_size].clone()
                    )
                    ttnn.deallocate(lg)
                ttnn.deallocate(xn)
                for t in full_pts + chunk_pts:
                    ttnn.deallocate(t)

                # Clone the group's batched GDN state (survives the next group's scratch reset).
                group_rec_dev.append([ttnn.clone(dn.rec_state) for dn in dn_layers])
                group_conv_dev.append([[ttnn.clone(dn.conv_states[m]) for m in range(dn.K)] for dn in dn_layers])
            finally:
                self._restore_gdn_batched(prev)

        ttnn.deallocate(cos)
        ttnn.deallocate(sin)
        ttnn.deallocate(csi)
        ttnn.synchronize_device(self.device)
        # Stitch the per-group states into the full [B,...] batched decode buffers (row u = user u).
        self._assemble_groups_gdn_dev(group_rec_dev, group_conv_dev)
        return self._reupload_host_logits(host_logits)

    def _fill_paged_cache_from_prefill(self, page_table):
        """Copy concat K/V into paged cache after prefill (one layer at a time to limit memory)."""
        for cache_idx, layer_idx in enumerate(self._attention_layer_indices):
            attn = self.layers[layer_idx].attention
            if attn.past_key is not None:
                k_cache, v_cache = self._paged_kv_caches[cache_idx]
                # paged_fill_cache asserts input.dtype() == cache.dtype(); it does NOT cast.
                # Cast per cache, so K and V may differ (QWEN36_KV_DTYPE).
                k_in = _match_cache_dtype(attn.past_key, k_cache)
                v_in = _match_cache_dtype(attn.past_value, v_cache)
                ttnn.experimental.paged_fill_cache(k_cache, k_in, page_table, batch_idx=0)
                ttnn.experimental.paged_fill_cache(v_cache, v_in, page_table, batch_idx=0)
                if k_in is not attn.past_key:
                    ttnn.deallocate(k_in)
                if v_in is not attn.past_value:
                    ttnn.deallocate(v_in)
                ttnn.deallocate(attn.past_key)
                ttnn.deallocate(attn.past_value)
                attn.past_key = None
                attn.past_value = None

    def prefill_paged(self, token_ids, page_table, valid_len=None, vision_tokens=None):
        """Prefill using paged attention for long sequences, concat for short.

        For T > 1024: uses paged prefill (paged_fill_cache + chunked_sdpa)
        via prefill_layer_chunked with page_table.
        For T <= 1024: uses direct concat prefill + post-hoc paged cache fill.

        Args:
            token_ids: torch.Tensor [B, T] token IDs
            page_table: torch.Tensor or ttnn.Tensor [B, max_blocks_per_seq] int32
        Returns:
            logits: ttnn.Tensor [B, 1, vocab_size]
        """
        if self.num_devices > 1:
            return self._prefill_paged_tp(token_ids, page_table, valid_len=valid_len, vision_tokens=vision_tokens)

        B, T = token_ids.shape
        # Stage the per-request RoPE (M-RoPE for multimodal, 1D for text) before any cos/sin seam;
        # prefill_layer_chunked (T>1024) inherits the staged table.
        self._build_request_rope(token_ids[:, :valid_len] if valid_len else token_ids, vision_tokens)
        # Keep page_table as torch.Tensor for CPU slicing in prefill_layer_chunked.
        page_table_torch = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        self.reset_state(batch_size=B)

        # Concat-based prefill for SDPA.
        if T > 1024:
            logits = self.prefill_layer_chunked(
                token_ids, chunk_size=2048, page_table=page_table_torch, vision_tokens=vision_tokens
            )
        else:
            token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
            x = self.embd(token_ids_ttnn)
            x = self._scatter_vision_tokens(x, token_ids, vision_tokens)
            ttnn.deallocate(token_ids_ttnn)

            cos, sin = self.rope.get_prefill_rot_mats(0, T)

            for layer in self.layers:
                x = layer.forward(x, cos=cos, sin=sin, mode="prefill")

            x = self.norm(x, mode=Mode.PREFILL)
            x_last = x[:, -1:, :]
            logits = self._lm_head(x_last)
            ttnn.deallocate(x)

        # Post-prefill: paged_fill no-op if already paged (T>1024); copies concat KV otherwise.
        page_table_device = ttnn.from_torch(
            page_table_torch, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        self._fill_paged_cache_from_prefill(page_table_device)

        # Fuse DeltaNet conv states for decode.
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                if dn.fused_conv_state is None and dn.conv_state_q is not None:
                    dn.fused_conv_state = ttnn.concat([dn.conv_state_q, dn.conv_state_k, dn.conv_state_v], dim=2)
                    dn.fused_conv_state = ttnn.to_layout(dn.fused_conv_state, ttnn.TILE_LAYOUT)

        # Copy DeltaNet state into external pre-allocated buffers.
        if self._deltanet_external_states is not None:
            dn_idx = 0
            for layer in self.layers:
                if not layer.is_full_attention:
                    dn = layer.attention
                    ext_rec, ext_conv = self._deltanet_external_states[dn_idx]
                    ttnn.copy(dn.recurrent_state, ext_rec)
                    if dn.fused_conv_state is not None:
                        ttnn.copy(dn.fused_conv_state, ext_conv)
                    dn_idx += 1

        return logits

    def decode_paged(self, token_ids, current_pos, page_table):
        """Single-token paged decode. Returns logits [B,1,vocab_size]."""
        B = token_ids.shape[0]
        # Accept torch or ttnn page_table.
        if isinstance(page_table, torch.Tensor):
            page_table = ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)

        token_ids_ttnn = ttnn.from_torch(token_ids, dtype=ttnn.uint32, device=self.device)
        x = self.embd(token_ids_ttnn)
        ttnn.deallocate(token_ids_ttnn)

        # RoPE position offset by rope_delta for multimodal (KV position stays the true seq pos).
        position_ids = torch.full((B, 1), current_pos + self.rope.rope_delta, dtype=torch.long)
        cos, sin = self.rope.get_rot_mats(position_ids)

        # cur_pos [B] for paged ops (not [B*n_kv] like non-paged decode).
        cur_pos_tensor = ttnn.from_torch(
            torch.full((B,), current_pos, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )

        for layer in self.layers:
            if layer.is_full_attention:
                x = layer.forward(
                    x,
                    cos=cos,
                    sin=sin,
                    mode="decode",
                    position_tensor=cur_pos_tensor,
                    page_table=page_table,
                )
            else:
                x = layer.forward(x, cos=cos, sin=sin, mode="decode")

        x = self._final_norm_decode(x)
        logits = self._lm_head(x)
        ttnn.deallocate(x)

        return logits

    # Generator contract — decode

    def prepare_decode_inputs_host(self, tokens, current_pos, page_table=None):
        """Build HOST decode inputs: (tokens_tt, cur_pos_tt, rope_packed, page_table_tt).

        QWEN36_DEVICE_SAMPLE=1 appends FIVE more: the per-step Gumbel block (the host's scaled
        noise T*g, keyed by the request's seed and the row's position), the per-row select flag
        (1.0 = sample this row, 0.0 = greedy), the top_k rank mask, the top_p threshold and the
        per-row 1/T (the temperature, for vLLM's order).  They are trace inputs like the others:
        the generator copies every host input into its device twin on EVERY step, so the noise
        follows the step.
        """
        from models.demos.blackhole.qwen36.tt.generator_interface import pack_rope_host

        B = tokens.shape[0]
        tokens_tt = ttnn.from_torch(tokens.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        # Per-user positions: current_pos may be a [B] tensor (each user at its own position) or a
        # scalar (lockstep). Build a [B] int32 vector so cur_pos and rope carry one rotation per user.
        if isinstance(current_pos, torch.Tensor):
            pos_vec = current_pos.to(torch.int32).reshape(-1)
            assert pos_vec.shape[0] == B, f"current_pos length {pos_vec.shape[0]} != batch {B}"
        else:
            pos_vec = torch.full((B,), int(current_pos), dtype=torch.int32)
        # RoPE position is the KV position offset by rope_delta (multimodal compresses the position
        # space; post-image text has t==h==w so 1D RoPE at rope_pos is correct). cur_pos_tt below
        # stays the true KV position. rope_delta is 0 for text, so this is a no-op there.
        rope_pos_vec = pos_vec + self.rope.rope_delta
        if self.num_devices > 1:
            # TP: rope_tp cos/sin [1,B,1,rope_dim] packed on host.
            rd = self.args.rope_head_dim
            inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
            freqs = torch.outer(rope_pos_vec.float(), inv_freq)  # [B, rd/2], per-user rotation
            emb = torch.cat([freqs, freqs], dim=-1)
            cos = emb.cos().reshape(1, B, 1, rd).to(torch.bfloat16)
            sin = emb.sin().reshape(1, B, 1, rd).to(torch.bfloat16)
            rope_packed = ttnn.from_torch(torch.cat([cos, sin], dim=0), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            if getattr(self, "_device_resident", False):
                # Resident decode derives cos/sin in-trace from cur_pos_tt; trace input 2 is not read.
                # Keep the slot (arity + DECODE_PAGE_TABLE_INPUT_IDX=3) with a tiny uint32 index.
                assert self.rope.rope_delta == 0, "QWEN36_DEVICE_RESIDENT is text-only (rope_delta must be 0)"
                rope_packed = ttnn.from_torch(
                    rope_pos_vec.clamp(min=0).to(torch.int32).reshape(1, B), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
                )
        else:
            # Single-device decode is B=1 in this port; per-user single-device rope is out of scope.
            cos_host, sin_host = self.rope.get_cos_sin_host(int(rope_pos_vec[0]))  # HOST ttnn [1,1,rope_head_dim]
            rope_packed = pack_rope_host(cos_host, sin_host)  # torch-based (host)
        cur_pos_tt = ttnn.from_torch(pos_vec, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        page_table_tt = (
            ttnn.from_torch(page_table, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            if page_table is not None
            else None
        )
        if not dev_sample.enabled():
            return tokens_tt, cur_pos_tt, rope_packed, page_table_tt
        blk = self._ds_block_host(pos_vec)
        return (tokens_tt, cur_pos_tt, rope_packed, page_table_tt, blk["noise"], blk["flag"],
                blk["rank"], blk["topp"], blk["scale"])

    # ------------------------------------------------------------------ QWEN36_DEVICE_SAMPLE (the per-step noise)
    def _ds_sampling_rows(self, B):
        """The per-row (seed, temperature, top_k, top_p) of the stashed sampling params
        (qwen36_vllm.decode_forward stashes them: the plugin only passes sampling_params when it
        samples on device)."""
        sp = getattr(self, "_ds_sampling", None)
        seeds = list(getattr(sp, "seed", None) or []) if sp is not None else []
        temps = list(getattr(sp, "temperature", None) or []) if sp is not None else []
        topks = list(getattr(sp, "top_k", None) or []) if sp is not None else []
        topps = list(getattr(sp, "top_p", None) or []) if sp is not None else []
        seeds = (seeds + [None] * B)[:B]
        temps = (temps + [0.0] * B)[:B]
        topks = (topks + [0] * B)[:B]
        topps = (topps + [1.0] * B)[:B]
        return (seeds,
                [float(t) if t is not None else 0.0 for t in temps],
                [int(t) if t is not None else 0 for t in topks],
                [float(t) if t is not None else 1.0 for t in topps])

    def _ds_block_host(self, pos_vec):
        """The host's per-step block (the Gumbel noise [1,1,B,2K], the select flag [1,1,B,1], the
        top_k rank mask [1,1,B,2K], the per-row 1/T [1,1,B,2K] and the top_p threshold [1,1,B,1])
        as ttnn host tensors.

        The noise is keyed by (the request's seed, the position of the token being predicted), so
        it is deterministic per request and per position.  An unseeded row gets a per-step random
        seed (the plugin draws a per-request one; this is the fallback so two requests never share
        noise)."""
        B = int(pos_vec.shape[0])
        K = dev_sample.k()
        seeds, temps, topks, topps = self._ds_sampling_rows(B)
        # the position of the token this step PREDICTS (the row's KV position + 1)
        pos = [int(p) + 1 for p in pos_vec.reshape(-1).tolist()]
        # QWEN36_DEVICE_SAMPLE (the s-sample3 review's blocker 1): the seed is an int32 on the
        # device path, so it is folded to 31 bits (a 63-bit value does not fit an int32 and the
        # overflow used to be swallowed, so every unseeded row shared seed 0).  The plugin writes
        # an ALREADY folded seed, so this is a no-op for it; it matters for a seed from anywhere
        # else.  The MTP path folds the same way (mtp_batch._ds_block), so the two paths give the
        # same noise for the same (seed, position).
        folded = []
        for s in seeds:
            if s is None:
                folded.append(dev_sample.random_seed31())
            else:
                si = int(s)
                folded.append(si if 0 <= si <= 0x7FFFFFFF else dev_sample.fold_seed31(si))
        seeds = folded
        block = dev_sample.row_block(seeds, pos, temps, topks=topks, topps=topps, K=K, M=B)
        self._ds_step += 1
        return {n: ttnn.from_torch(v.contiguous(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
                for n, v in block.items()}

    def bind_decode_trace_inputs(self, device_inputs):
        """QWEN36_DEVICE_SAMPLE: the decode trace reads its per-step Gumbel block and select flag
        through this stash (the same pattern Gemma4 uses for its extra host-precomputed input): the
        generator copies every host input into its device twin on every step, so the noise follows
        the step, and the trace's baked address stays valid."""
        if not dev_sample.enabled():
            return
        if len(device_inputs) < 9:
            raise RuntimeError("QWEN36_DEVICE_SAMPLE: the decode trace needs 9 device inputs "
                               f"(tokens, pos, rope, page table, noise, flag, rank, top_p, scale); "
                               f"got {len(device_inputs)}")
        self._ds_block = {"noise": device_inputs[4], "flag": device_inputs[5],
                          "rank": device_inputs[6], "topp": device_inputs[7],
                          "scale": device_inputs[8]}

    def _ds_block_device(self, B, K):
        """A REPLICATED block (noise, flag, rank, top_p) of the served shape, for the COMPILE pass.

        The compile pass runs BEFORE the capture (and before the trace's own inputs are bound), and
        it must compile every program the capture records -- a compile inside the capture would
        clobber a parked trace (docs/AGENT-ONBOARDING "a compile after a trace is parked (hang)").
        The temporary is built with the SAME copy path as the bound input, so the compiled program
        is the captured one."""
        from models.tt_transformers.tt.common import copy_host_to_device

        B = int(B)
        host = [torch.zeros((1, 1, B, 2 * int(K)), dtype=torch.float32),
                torch.zeros((1, 1, B, 1), dtype=torch.float32),
                torch.ones((1, 1, B, 2 * int(K)), dtype=torch.float32),
                torch.ones((1, 1, B, 1), dtype=torch.float32),
                torch.ones((1, 1, B, 2 * int(K)), dtype=torch.float32)]
        hs = [ttnn.from_torch(h, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT) for h in host]
        return copy_host_to_device(hs, mesh_device=self.mesh_device)

    def prepare_inputs_decode(self, tokens, current_pos, page_table=None):
        """Host-to-device transfer for decode inputs."""
        from models.tt_transformers.tt.common import copy_host_to_device

        host = self.prepare_decode_inputs_host(tokens, current_pos, page_table=page_table)
        return copy_host_to_device(host, mesh_device=self.mesh_device)

    def ttnn_decode_forward(
        self,
        tokens,
        current_pos,
        rot_mat_idxs=None,
        page_table=None,
        kv_cache=None,
        on_device_logits=False,
        **kwargs,
    ):
        """Generator decode forward. kv_cache accepted but unused (state is model-bound).

        on_device_logits=True: return the raw vocab-sharded shard for the on-device sampler.
        """
        from models.demos.blackhole.qwen36.tt.generator_interface import unpack_rope

        resident = getattr(self, "_device_resident", False)
        if resident:
            cos, sin = self._rope_from_positions_device(current_pos)
        else:
            cos, sin = unpack_rope(rot_mat_idxs)
        if on_device_logits and getattr(self, "_device_argmax", False):
            logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, sharded_lm_head=True)
            samp = None
            flagt = None
            if dev_sample.enabled():
                # QWEN36_DEVICE_SAMPLE: the sampled ids (the device sampler over the shard) and the
                # per-row select.  The sampler reads the per-step Gumbel block (the trace input the
                # generator refreshes every step); the greedy ids below are UNCHANGED, so a greedy
                # row (flag 0.0) is byte-identical to today's greedy decode.
                blk = getattr(self, "_ds_block", None)
                M = int(logits.shape[2])
                if blk is None or int(blk["noise"].shape[2]) != M:
                    # the COMPILE pass (before the capture binds the trace's own inputs): a
                    # temporary of the served shape, so the sampler is compiled OUTSIDE the capture.
                    d = self._ds_block_device(M, dev_sample.k())
                    blk = {"noise": d[0], "flag": d[1], "rank": d[2], "topp": d[3], "scale": d[4]}
                samp = dev_sample.sample(logits, int(logits.shape[3]), self.mesh_device, self.tt_ccl,
                                         self.args.ccl_topology(), blk, K=dev_sample.k(),
                                         deallocate=False)
            tok = self._greedy_argmax_device(logits)
            if samp is not None:
                tok = dev_sample.select_ids(tok, samp, blk["flag"])
            if resident:
                self._advance_decode_chain_device(tok, tokens, current_pos)
            if not self._device_argmax_logged:
                self._device_argmax_logged = True
                logger.info(
                    f"QWEN36_DEVICE_ARGMAX engaged: decode returns greedy token ids {list(tok.shape)} {tok.dtype} "
                    f"(per-shard argmax + on-device 2-way combine; logits never leave the device)"
                )
            # Bare tensor: the traced path stores it as the trace output; process_output_decode
            # reads device 0's [:B] ids (is_tokens=True).
            return tok
        if on_device_logits:
            assert self.sampling is not None, "on_device_logits=True but self.sampling is None"
            logits = self._forward_decode(tokens, cos, sin, current_pos, page_table, sharded_lm_head=True)
            # Sampler runs >=32-wide; pad B up to it (else shape mismatch). Extra slots unused.
            sampler_batch = self.sampling.tt_sampling.max_batch_size
            B = logits.shape[2]
            if B < sampler_batch:
                logits = ttnn.pad(logits, [(0, 0), (0, 0), (0, sampler_batch - B), (0, 0)], value=0.0)
            # Bare tensor (not a tuple): the traced path passes this straight to capture_trace().
            return logits
        logits = self._forward_decode(tokens, cos, sin, current_pos, page_table)
        if _tt_public_getenv("QWEN36_DECODE_LOGITS_RM") == "1" and logits.layout != ttnn.ROW_MAJOR_LAYOUT:
            # Host-sampling readback: the logits are logically [1,1,B,vocab] but TILE-padded to 32
            # rows, and .cpu() moves the PHYSICAL buffer (32 x 248320 x bf16 = 15.9 MB per token
            # over card 0's PCIe Gen3 x4) and then untilizes it on the 4-core host. Untilizing on
            # device inside the trace leaves a [1,1,B,vocab] row-major tensor: 0.5 MB, no host
            # untilize. Same values, same dtype -- only the layout the host reads changes.
            logits = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
            if not getattr(self, "_logits_rm_logged", False):
                self._logits_rm_logged = True
                logger.info(
                    f"QWEN36_DECODE_LOGITS_RM engaged: decode logits {list(logits.shape)} "
                    f"{logits.dtype} {logits.layout} (host readback is now the unpadded row)"
                )
        return logits, None

    def _build_resident_rope_tables(self):
        """Device cos/sin tables [max_seq_len, rope_head_dim] bf16 ROW_MAJOR (ttnn.embedding weights),
        replicated on the mesh. Same formula and same bf16 rounding as prepare_decode_inputs_host."""
        rd = self.args.rope_head_dim
        n = int(self.args.max_seq_len)
        inv_freq = 1.0 / (self.args.rope_theta ** (torch.arange(0, rd, 2).float() / rd))
        freqs = torch.outer(torch.arange(n, dtype=torch.int32).float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        mk = lambda t: ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        self._resident_cos_tab = mk(emb.cos())
        self._resident_sin_tab = mk(emb.sin())
        logger.info(f"QWEN36_DEVICE_RESIDENT: rope tables [{n}, {rd}] bf16 on device ({2 * n * rd * 2 / 2**20:.1f} MiB)")

    def _rope_from_positions_device(self, cur_pos_tt):
        """In-trace cos/sin [1,1,B,rd] bf16 TILE from the int32 [B] position input. Padded rows carry
        -1 (plus_one skips them); they are clamped to 0 so the table gather stays in bounds."""
        B = cur_pos_tt.shape[0]
        p_t = ttnn.to_layout(ttnn.reshape(cur_pos_tt, (1, 1, 1, B)), ttnn.TILE_LAYOUT)
        idx_t = ttnn.typecast(ttnn.relu(p_t), ttnn.uint32)
        idx = ttnn.reshape(ttnn.to_layout(idx_t, ttnn.ROW_MAJOR_LAYOUT), (1, B))
        cos = ttnn.embedding(idx, self._resident_cos_tab, layout=ttnn.TILE_LAYOUT)  # [1,B,rd]
        sin = ttnn.embedding(idx, self._resident_sin_tab, layout=ttnn.TILE_LAYOUT)
        cos = ttnn.unsqueeze_to_4D(cos)  # [1,1,B,rd]; apply_partial_rope_decode reshapes to this anyway
        sin = ttnn.unsqueeze_to_4D(sin)
        for x in (p_t, idx_t, idx):
            ttnn.deallocate(x)
        return cos, sin

    def _advance_decode_chain_device(self, tok_u, tokens, current_pos):
        """End-of-trace: feed the argmax id back into the token input and advance the positions
        (tt_transformers._increment_decode_positions_device). Padded rows (pos -1) do not advance."""
        tok_rm = ttnn.reshape(ttnn.to_layout(tok_u, ttnn.ROW_MAJOR_LAYOUT), tuple(tokens.shape))
        ttnn.copy(tok_rm, tokens)
        ttnn.deallocate(tok_rm)
        ttnn.plus_one(current_pos, skip_negative_entries=True)
        if not self._device_resident_logged:
            self._device_resident_logged = True
            logger.info(
                "QWEN36_DEVICE_RESIDENT engaged: decode trace writes its token back and advances "
                f"positions on device (tokens {list(tokens.shape)} {tokens.dtype}, pos {current_pos.dtype})"
            )

    def _greedy_argmax_device(self, logits, per=None):
        """Greedy token ids from the vocab-SHARDED decode logits, entirely on device (trace-safe).

        logits: mesh tensor, per-device shard [1,1,B,vocab/2] bf16 TILE (device d holds vocab range
        [d*per, (d+1)*per)). Returns a REPLICATED [1,1,B,1] uint32 tensor of global token ids.
        Tie rule = vLLM greedy (lowest index): ttnn.argmax is lowest-index within a shard (measured,
        [source validation record]) and the combine takes shard 1 only when its max is
        STRICTLY greater, so a cross-shard tie resolves to shard 0 = the lower global index.
        Cost measured in isolation: ~0.5 ms traced at B=8 (argmax 0.22, gather 0.04, 2 tiny CCLs).
        per: the per-shard width (default vocab/TP); QWEN36_MTP_DRAFT_VOCAB passes N/TP and gets sub-indices."""
        from models.tt_transformers.tt.ccl import tt_all_gather

        B = logits.shape[2]
        per = self.args.vocab_size // self.num_devices if per is None else int(per)
        if argmax_fold.usable(B, per, self.num_devices):
            # QWEN36_ARGMAX_FOLD (argmax_fold.py): pack + ONE all-gather + combine kernels, same ids (boot self-test)
            return argmax_fold.greedy(logits, per, self.mesh_device, self.tt_ccl, self.args.ccl_topology())
        rm = ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT)
        idx = ttnn.argmax(rm, dim=-1, keepdim=True)                       # [1,1,B,1] uint32, per shard
        val = ttnn.gather(rm, dim=-1, index=idx)                          # [1,1,B,1] bf16: the shard max
        idx_f = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)   # exact up to 2^24
        val_t = ttnn.to_layout(val, ttnn.TILE_LAYOUT)
        topo = self.args.ccl_topology()
        gi = tt_all_gather(idx_f, self.mesh_device, self.tt_ccl, cluster_axis=None, dim=3, topology=topo, dtype=ttnn.float32)
        gv = tt_all_gather(val_t, self.mesh_device, self.tt_ccl, cluster_axis=None, dim=3, topology=topo, dtype=ttnn.bfloat16)
        i0 = ttnn.slice(gi, [0, 0, 0, 0], [1, 1, B, 1]); i1 = ttnn.slice(gi, [0, 0, 0, 1], [1, 1, B, 2])
        v0 = ttnn.slice(gv, [0, 0, 0, 0], [1, 1, B, 1]); v1 = ttnn.slice(gv, [0, 0, 0, 1], [1, 1, B, 2])
        m = ttnn.typecast(ttnn.gt(v1, v0), ttnn.float32)                  # 1.0 where shard 1 strictly wins
        tok = ttnn.add(i0, ttnn.multiply(m, ttnn.add(ttnn.subtract(i1, i0), float(per))))
        tok_u = ttnn.typecast(tok, ttnn.uint32)
        for x in (rm, idx, val, idx_f, val_t, gi, gv, i0, i1, v0, v1, m, tok):
            ttnn.deallocate(x)
        ttnn.deallocate(logits)
        return tok_u

    def process_output_decode(self, tt_out, B, S=1, is_tokens=False, is_log_probs=False):
        """Convert decode output to host torch. Host-sampling returns logits [B,S,vocab];
        on-device sampling returns sampled token ids or sampled-token log-probs.
        """
        if is_tokens or is_log_probs:
            # Sampled ids and old-path sampled-token log-probs are replicated across devices.
            if self.num_devices > 1:
                return ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).reshape(-1)[:B]
            return ttnn.to_torch(tt_out).reshape(-1)[:B]
        if self.num_devices > 1:
            if getattr(self, "_decode_logits_sharded", False):
                # QWEN36_LM_HEAD_GATHER_MODE=none: each device holds vocab/2, so reading one
                # replica and reshaping to the full vocab fails (124160 vs 248320). Concat is
                # the contract _lm_head states when it sets this flag. Without this branch the
                # flag is set but never honoured, and the mode dies on the reshape.
                full = ttnn.to_torch(
                    tt_out, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=3)
                ).float()
            else:
                # TP: read one replica (get_device_tensors[0]), not ConcatMeshToTensor (~4x readback).
                full = ttnn.to_torch(ttnn.get_device_tensors(tt_out)[0]).float()
        else:
            full = ttnn.to_torch(tt_out).float()
        rows = full.reshape(-1, self.args.vocab_size)
        required_rows = B * S
        if rows.shape[0] < required_rows:
            # Decode bucketing returns only the active prefix. The shared
            # generator requests the fixed serving width, so add neutral rows
            # instead of splitting each vocabulary row during ``view(B, S, -1)``.
            rows = torch.nn.functional.pad(rows, (0, 0, 0, required_rows - rows.shape[0]))
        return rows[:required_rows].view(B, S, self.args.vocab_size)

    def _save_deltanet_states(self):
        """Snapshot GDN state to host (guard across decode-trace capture's double forward)."""
        saved = []
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved.append(
                    {
                        "recurrent": ttnn.to_torch(dn.recurrent_state),
                        "conv": ttnn.to_torch(dn.fused_conv_state) if dn.fused_conv_state is not None else None,
                    }
                )
        return saved

    def _restore_deltanet_states(self, saved_states, device):
        """Restore GDN state via ttnn.copy (preserves trace-baked buffer addresses)."""
        idx = 0
        for layer in self.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                saved = saved_states[idx]
                restored = ttnn.from_torch(
                    saved["recurrent"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                )
                ttnn.copy(restored, dn.recurrent_state)
                ttnn.deallocate(restored)
                if saved["conv"] is not None:
                    restored_conv = ttnn.from_torch(
                        saved["conv"], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
                    )
                    ttnn.copy(restored_conv, dn.fused_conv_state)
                    ttnn.deallocate(restored_conv)
                    dn._restore_split_conv_from_fused()
                idx += 1

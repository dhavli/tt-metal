# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""vLLM wrapper for Qwen3.5/3.6 on Blackhole — a thin tt_transformers Generator subclass.

Hybrid model: 8 paged-KV attention + 24 GDN recurrent-state layers. GDN forbids token-padding and
isn't position-general, so prefill is model-owned (masked-bucket for short prompts / chunk-outer
trace for long, via prefill_dispatch) while Generator drives decode only. GDN + KV state is
model-bound, so the kv_cache contract param is accepted but unused.
"""
from models.demos.blackhole.qwen36.public_defaults import getenv as _tt_public_getenv, contains as _tt_public_contains, getitem as _tt_public_getitem

import math
import os
from collections import defaultdict
from typing import Mapping, Optional

import torch
from loguru import logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.multimodal import MULTIMODAL_REGISTRY

import ttnn
from models.demos.blackhole.qwen36.tt import dev_sample
from models.demos.blackhole.qwen36.tt import spec_sample
from models.demos.blackhole.qwen36.tt import mtp_batch_host as _mtp_host
from models.demos.blackhole.qwen36.tt.common import create_tt_model
from models.demos.blackhole.qwen36.tt.generator_interface import prefill_dispatch, warmup_decode_buckets
from models.tt_transformers.tt.generator import Generator

_PREFILL_WARMUP_CHUNK = 2048  # the default; _prefill_warmup_chunk() is what the capture uses
_PREFILL_WARMUP_BUCKET = 4096
_BLOCK_SIZE = 64


def _prefill_warmup_chunk():
    """QWEN36_PREFILL_CHUNK (c134): the chunk the served prefill trace is captured at, read at each capture (so a wb
    `set` re-capture switches it). Unset = 2048, byte-identical. Any other value logs the banner the arm expects."""
    from models.demos.blackhole.qwen36.tt.model import prefill_chunk_size, prefill_mask_buckets

    chunk = prefill_chunk_size()
    if chunk % _BLOCK_SIZE:
        raise ValueError(f"QWEN36_PREFILL_CHUNK={chunk}: must be a multiple of the {_BLOCK_SIZE}-token KV block")
    if chunk != _PREFILL_WARMUP_CHUNK:
        logger.info(f"QWEN36_PREFILL_CHUNK engaged: chunk={chunk} (prefill chunk trace captured at {chunk} tokens; "
                    f"masked buckets {prefill_mask_buckets()})")
    return chunk


class TT_Qwen3_5ProcessingInfo(Qwen3_5ProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        # Serve a single visual item per request (B=1, max_concurrency=1). Image and video are both
        # supported, but only ONE modality per request: the model's vision splice keys off a single
        # placeholder token id (image_token_id XOR video_token_id), so a mixed image+video prompt
        # cannot be spliced correctly.
        return {"image": 1, "video": 1}


@MULTIMODAL_REGISTRY.register_processor(
    Qwen3VLMultiModalProcessor, info=TT_Qwen3_5ProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder
)
class Qwen36ForCausalLM(Generator, SupportsMultiModal):
    """vLLM-compatible wrapper for Qwen3.5-9B on Blackhole P150."""

    # Decode bucketing keeps several traces live and refreshes the selected
    # bucket's inputs before replay, so their I/O buffers may safely overlap.
    _tt_allow_decode_trace_buffer_reuse = True

    # supports_async_decode=False: async decode assumes on-device token/position continuity, which
    # corrupts Qwen's GDN scan. supports_sample_on_device=True: on-device sampling is decode-only.
    _RESIDENT = _tt_public_getenv("QWEN36_DEVICE_RESIDENT") == "1"
    _PREFIX = _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") == "1"
    model_capabilities = {
        "supports_prefix_caching": _PREFIX,
        "supports_chunked_prefill": _PREFIX,
        "supports_async_decode": _RESIDENT,
        "supports_sample_on_device": True,
    }
    decode_input_update_contract = 1 if _RESIDENT else 0

    # Opt-in requires the synchronous block-keyed consumer in vllm-tt.

    @classmethod
    def get_kv_cache_spec(cls, vllm_config):
        """Describe Qwen36's 16 attention caches and 48 GDN states."""
        if _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") != "1":
            return None

        from vllm.model_executor.layers.mamba.mamba_utils import (
            MambaStateDtypeCalculator,
            MambaStateShapeCalculator,
        )
        from dataclasses import replace
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            MambaSpec,
            MambaAttentionBackendEnum,
        )

        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        parallel_config = vllm_config.parallel_config
        hf_config = model_config.hf_text_config
        if parallel_config.tensor_parallel_size != 1 or parallel_config.pipeline_parallel_size != 1:
            raise ValueError("TT prefix caching requires vLLM TP=PP=1; physical TT mesh supplies TP2")
        # Quantized KV is a capacity requirement of the 1M-token pool, not a numeric choice:
        # accept any quantized pair, symmetric (QWEN_SDPA_BF8=1) or asymmetric
        # (QWEN36_KV_DTYPE="<K>/<V>", which wins over the symmetric switch).
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        _kd, _vd, _kvsrc = _tpc.resolve_kv_dtypes(ttnn.bfloat16)
        if _kd == ttnn.bfloat16 and _vd == ttnn.bfloat16:
            raise ValueError("QWEN36_PREFIX_CACHING requires quantized KV: QWEN36_KV_DTYPE=<K>/<V> or QWEN_SDPA_BF8=1")
        if vllm_config.speculative_config is not None:
            raise ValueError("QWEN36_PREFIX_CACHING does not support speculative decoding")
        if parallel_config.data_parallel_size != 1:
            raise ValueError("QWEN36_PREFIX_CACHING has not been validated with data parallelism")

        layer_types = list(hf_config.layer_types)
        if layer_types != ["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 16:
            raise ValueError("QWEN36_PREFIX_CACHING requires the exact 64-layer 27B topology")
        attention_layers = [i for i, kind in enumerate(layer_types) if kind == "full_attention"]
        gdn_layers = [i for i, kind in enumerate(layer_types) if kind != "full_attention"]
        if len(attention_layers) != 16 or len(gdn_layers) != 48:
            raise ValueError(
                "QWEN36_PREFIX_CACHING requires 16 attention and 48 GDN layers; "
                f"got {len(attention_layers)} and {len(gdn_layers)}"
            )

        mamba_mode = getattr(cache_config, "mamba_cache_mode", "align")
        if mamba_mode != "align":
            raise ValueError(f"QWEN36_PREFIX_CACHING requires mamba_cache_mode='align', got {mamba_mode!r}")
        mamba_shapes = MambaStateShapeCalculator.gated_delta_net_state_shape(
            2,  # physical TT TP; vLLM TP remains 1
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            0,
        )
        mamba_dtypes = MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            model_config.dtype,
            getattr(cache_config, "mamba_cache_dtype", "auto"),
            getattr(cache_config, "mamba_ssm_cache_dtype", "auto"),
        )
        mamba_template = MambaSpec(
            block_size=getattr(cache_config, "mamba_block_size", None) or cache_config.block_size,
            shapes=mamba_shapes,
            dtypes=mamba_dtypes,
            page_size_padded=getattr(cache_config, "mamba_page_size_padded", None),
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
            num_speculative_blocks=0,
        )
        head_dim = getattr(
            hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads
        )
        attention_template = FullAttentionSpec(
            block_size=cache_config.block_size,
            num_kv_heads=hf_config.num_key_value_heads,
            head_size=head_dim,
            dtype=torch.uint8,
        )
        # RECONCILE THE TWO PAGE SIZES, OR vLLM REFUSES TO CONSTRUCT.
        # MEASURED 2026-09-14, third prod-prefix boot:
        #   NotImplementedError: Layer model.layers.3.self_attn: page size is not divisible by
        #   the maximum page size and cannot be padded.
        # vllm/v1/core/kv_cache_utils.py::unify_kv_cache_spec_page_size takes the largest page
        # across all groups and then, for every other group, either (a) scales its block_size by
        # an exact integer ratio when its page DIVIDES the max, or (b) pads it -- but (b) is only
        # available to an AttentionSpec whose backend sets indexes_kv_by_block_stride, which ours
        # does not. A GDN state page is far larger than a 64-token attention page and is not a
        # multiple of it, so neither branch applied and the engine aborted.
        # The fix belongs on OUR side: round the Mamba page UP to a whole number of attention
        # pages. MambaSpec already supports page_size_padded and vLLM pads Mamba pages by this
        # same mechanism. The attention page then divides the max exactly and branch (a) applies.
        # Nothing is hardcoded: both sizes are read back off the constructed specs.
        attn_page = attention_template.page_size_bytes
        mamba_page = mamba_template.page_size_bytes
        explicit_pad = getattr(cache_config, "mamba_page_size_padded", None)
        if explicit_pad is not None:
            # An operator-supplied padding is honoured but must still satisfy the invariant,
            # otherwise the engine dies deep in init with the message above. Fail here instead.
            if explicit_pad % attn_page != 0:
                raise ValueError(
                    f"QWEN36_PREFIX_CACHING: mamba_page_size_padded={explicit_pad} is not a "
                    f"multiple of the {attn_page}-byte attention page; vLLM cannot unify these "
                    f"groups. Use a multiple of {attn_page}."
                )
        elif mamba_page % attn_page != 0:
            # Pad to a POWER-OF-TWO multiple of the attention page, not merely the next whole
            # multiple. MEASURED 2026-09-14: the plain ceiling gave ratio=13, and vLLM converts
            # the ratio directly into the attention block size (64 -> 832 tokens). 832 is not a
            # tile-friendly paged-attention block on this hardware and nothing else in this
            # stack has ever run at that width -- every bench and the block-keyed prefix
            # consumer use 64. A power-of-two ratio keeps the attention block at 64 * 2^k
            # (here 1024). The cost is extra slack in the GDN page, which is reported below;
            # a wrong-width attention kernel would be far more expensive than that slack.
            ratio = -(-mamba_page // attn_page)                # ceil in whole attention pages
            while ratio & (ratio - 1):                         # round up to a power of two
                ratio += 1
                ratio = 1 << (ratio - 1).bit_length() if ratio & (ratio - 1) else ratio
            padded = ratio * attn_page
            mamba_template = replace(mamba_template, page_size_padded=padded)
            _log = globals().get("logger")
            if _log is not None:
                _log.info(
                    f"QWEN36_PREFIX_PAGE_ALIGN: mamba page {mamba_page} B -> {padded} B "
                    f"({padded // attn_page} x the {attn_page} B attention page, "
                    f"+{padded - mamba_page} B slack per block)"
                )

        specs = {
            f"model.layers.{i}.self_attn": attention_template for i in attention_layers
        }
        specs.update({f"model.layers.{i}.linear_attn": mamba_template for i in gdn_layers})
        return specs

    @staticmethod
    def _gdn_state_dtypes():
        """The (conv, recurrent) dtypes the GDN sidecars PHYSICALLY carry, as torch and as ttnn.

        MEASURED 2026-09-15, two dead boots in a row. gdn/tp.py allocates the recurrent state
        float32 by default and bfloat16 under QWEN35_GDN_STATE_BF16=1, which the production
        profile sets. The prefix guards hardcoded float32, so every prefix run died on its first
        request ("prefix sidecars require fp32 recurrence and bf16 convolution"); forcing the
        state back to float32 instead killed the boot outright, because the fused decode kernel
        requires one dtype across its tensors:
            TT_FATAL decode_gated_delta_rule_device_operation.cpp:25: t.dtype() == dt
        So bf16 is not a preference here, it is what the cleared fused-decode path requires, and
        the guards must describe the state that exists rather than the one they expected. vLLM's
        declaration is then made to match with --mamba-ssm-cache-dtype; a declaration that
        disagrees with the physical state means the engine's block accounting is sized for a
        state that is not there, which is why the mismatch is still refused below.
        """
        import torch as _torch
        bf16 = _tt_public_getenv("QWEN35_GDN_STATE_BF16") == "1"
        rec_torch = _torch.bfloat16 if bf16 else _torch.float32
        rec_ttnn = ttnn.bfloat16 if bf16 else ttnn.float32
        return (_torch.bfloat16, rec_torch), (ttnn.bfloat16, rec_ttnn)

    def prefix_cache_initialize(self, vllm_config, kv_cache_config):
        """Validate the host-sidecar contract; block ownership stays in the runner.

        Runs before physical KV allocation; live tensor checks are deferred to use.
        No performance or device equivalence claim is made by this opt-in.
        """
        self._prefix_cache_initialized = False
        specs = self.get_kv_cache_spec(vllm_config)
        if specs is None:
            raise ValueError("prefix caching is disabled")
        if not vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("prefix caching requires scheduler prefix caching")
        _kt = getattr(vllm_config, "kv_transfer_config", None)
        # QWEN36_KV_HOST_TIER: the plugin's TT host-tier OffloadingConnector is the one connector admitted.
        _host_tier = (_tt_public_getenv("QWEN36_KV_HOST_TIER", "0") == "1" and _kt is not None
                      and _kt.kv_connector == "TTHostTierConnector")
        if ((_kt is not None and not _host_tier)
                or getattr(vllm_config, "ec_transfer_config", None) is not None
                or getattr(vllm_config.model_config, "is_encoder_decoder", False)
                or getattr(vllm_config, "kv_events_config", None) is not None
                or getattr(vllm_config.cache_config, "kv_offloading_size", None)
                or getattr(vllm_config.cache_config, "cpu_offload_gb", 0)
                or (getattr(vllm_config.scheduler_config, "async_scheduling", False) and not self._prefix_s12_async())):
            raise ValueError("prefix sidecars require synchronous local execution")
        model = self.model[0]
        if (model.num_devices != 2 or tuple(model.args.cluster_shape) != (1, 2)
                or model.layer_indices != list(range(64))
                or [l.is_full_attention for l in model.layers] != [False, False, False, True] * 16):
            raise ValueError("prefix caching requires the full TP2 27B model")
        observed = {}
        for group in kv_cache_config.kv_cache_groups:
            for name in group.layer_names:
                if name in observed:
                    raise ValueError("duplicate prefix cache layer")
                observed[name] = group.kv_cache_spec
        if set(observed) != set(specs):
            raise ValueError("scheduler cache groups do not cover the model exactly")
        for name, expected in specs.items():
            actual = observed[name]
            if type(actual) is not type(expected):
                raise ValueError(f"wrong cache spec for {name}")
            fields = ("shapes", "dtypes", "mamba_type", "mamba_cache_mode", "num_speculative_blocks") if hasattr(
                expected, "shapes") else ("num_kv_heads", "head_size", "dtype")
            if any(getattr(actual, field) != getattr(expected, field) for field in fields):
                raise ValueError(f"incompatible cache spec for {name}")
            # An error that does not say WHICH layer or WHAT the numbers were costs a whole
            # boot to diagnose. vLLM legitimately RESCALES block sizes during
            # unify_kv_cache_spec_page_size, so the observed size is expected to differ from
            # what we asked for; the invariant is only that it stays a whole number of
            # attention pages.
            # MEASURED 2026-09-14: this check was applied to EVERY group and is wrong for a
            # Mamba one. A MambaSpec block_size is a STATE-ALIGNMENT granularity in tokens
            # (16384), not a count of attention pages, and vLLM's unification rescales
            # cache_config.block_size itself (64 -> 832 here), so demanding
            # "16384 % 832 == 0" asserted a relationship that never had to hold. The documented
            # constraint for align mode is a floor of 2048 and divisibility by 2048
            # (docs/PREFIX-CACHING-IMPLEMENTATION.md), which is what is checked now.
            if hasattr(expected, "shapes"):
                if actual.block_size < 2048 or actual.block_size % 2048:
                    raise ValueError(
                        f"GDN state alignment must be >= 2048 and divisible by 2048: {name} "
                        f"has block_size={actual.block_size}"
                    )
            elif actual.block_size <= 0 or actual.block_size % vllm_config.cache_config.block_size:
                raise ValueError(
                    f"cache blocks must be attention-page aligned: {name} has "
                    f"block_size={actual.block_size} (expected a positive multiple of "
                    f"cache_config.block_size={vllm_config.cache_config.block_size}); "
                    f"page_size_bytes={getattr(actual, 'page_size_bytes', '?')}, "
                    f"requested block_size={expected.block_size}"
                )
        mamba = next(s for s in specs.values() if hasattr(s, "dtypes"))
        expected_torch, _ = self._gdn_state_dtypes()
        if tuple(mamba.dtypes) != expected_torch:
            raise ValueError(
                f"scheduler GDN dtypes {tuple(mamba.dtypes)} must match the physical state "
                f"{expected_torch} (QWEN35_GDN_STATE_BF16="
                f"{_tt_public_getenv('QWEN35_GDN_STATE_BF16', 'unset')}). Pass "
                "--mamba-ssm-cache-dtype to make vLLM's declaration match the state that exists; "
                "do not change the state to match the declaration, the fused decode kernel "
                "requires bf16.")
        self._prefix_cache_vllm_config = vllm_config
        self._prefix_cache_initialized = True
        _log = globals().get("logger")
        if _log is None:
            raise RuntimeError(
                "QWEN36_PREFIX_CACHING: no module logger, so the arm cannot identify "
                "itself at runtime. Refusing to serve unidentifiable results."
            )
        if _log is not None:  # guaranteed by the raise above
            _log.info("QWEN36_PREFIX_CACHING: model envelope verified (16 attention, 48 GDN, TP2 physical mesh)")
            # Cold-path levers (cold-pp, opt-in; off = the shipped path byte for byte). Banners at boot so ttx's
            # identity check sees them; their served self-tests log "... self-test exact" after traffic.
            from models.demos.blackhole.qwen36.tt import prefix_snap as _psnap

            if _psnap.enabled():
                _log.info(f"QWEN36_PREFIX_SNAP_FAST armed: checkpoint readback of the wanted GDN slots only, "
                          "bf16 untilized on device; self-test vs the legacy readback on the first served checkpoint"
                          + (f" MUSTFAIL={_psnap.mustfail()}" if _psnap.mustfail() else ""))
            if _psnap.nb_enabled():
                if not _psnap.enabled():
                    raise RuntimeError(f"{_psnap.ENV_NB}=1 needs {_psnap.ENV}=1 (it is that path's read schedule)")
                _log.info(f"QWEN36_PREFIX_SNAP_NB armed: GDN checkpoint reads enqueued non-blocking, one "
                          f"synchronize per {_psnap.nb_layers()} layers (same ops, same bytes; covered by the {_psnap.ENV} self-test)")
            if _psnap.pack_enabled():  # raises unless SNAP_FAST is on and SNAP_NB is off
                _log.info(f"QWEN36_PREFIX_SNAP_PACK armed: GDN checkpoint reads of {_psnap.pack_layers()} layers "
                          "concatenated on device, 2 blocking reads per group (same bytes; covered by the "
                          f"{_psnap.ENV} self-test)"
                          + (f" MUSTFAIL={_psnap.pack_mustfail()}" if _psnap.pack_mustfail() else ""))
            if _psnap.scratch_resume_enabled():
                _log.info(f"QWEN36_PREFIX_SCRATCH_RESUME armed: an align-mode continuation whose state is still in "
                          "the bound B=1 prefill scratch skips the host snapshot restore (resident row first); self-test on "
                          "the first resident continuation"
                          + (f" MUSTFAIL={_psnap.resume_mustfail()}" if _psnap.resume_mustfail() else ""))

    @staticmethod
    def _prefix_s12_async():
        """The S12 prefix arm keeps vLLM async scheduling: batched MTP under QWEN36_MTP_ASYNC needs it, and the prefix
        cache there only acts on PREFILL steps, which the runner executes synchronously (the snapshot is read right
        after the step's forward, before the next step is submitted). Decode steps never touch the prefix cache."""
        return _tt_public_getenv("QWEN36_MTP_BATCH") == "1" and _tt_public_getenv("QWEN36_MTP_ASYNC") == "1"

    def _validate_prefix_cache_model(self):
        model = self.model[0]
        caches = model._paged_kv_caches
        # Quantized paged KV: bf8 both (QWEN_SDPA_BF8) or the asymmetric QWEN36_KV_DTYPE pair (S12: K bf8 / V bf4);
        # each cache must carry the dtype the model resolved for it.
        from models.demos.blackhole.qwen36.tt import tp_common as _tpc

        _kd, _vd, _ = _tpc.resolve_kv_dtypes(ttnn.bfloat16)
        kv_dt = (_kd, _vd)
        quant = (ttnn.bfloat8_b, ttnn.bfloat4_b)
        if (caches is None or len(caches) != 16 or any(d not in quant for d in kv_dt)
                or any(len(pair) != 2 or pair[0].dtype != kv_dt[0] or pair[1].dtype != kv_dt[1] for pair in caches)):
            raise ValueError(f"prefix caching requires allocated quantized attention KV (K/V {kv_dt})")
        for layer in model.layers:
            if not layer.is_full_attention:
                dn = layer.attention
                _, (conv_dtype, rec_dtype) = self._gdn_state_dtypes()
                if (dn.rec_state is None or dn.conv_states is None or len(dn.conv_states) != dn.K
                        or dn.rec_state.dtype != rec_dtype
                        or any(c.dtype != conv_dtype for c in dn.conv_states)):
                    raise ValueError(
                        f"prefix sidecars must carry {rec_dtype} recurrence and {conv_dtype} "
                        f"convolution; found rec={getattr(dn.rec_state, 'dtype', None)}")
                hf = self._prefix_cache_vllm_config.model_config.hf_text_config
                if (dn.Nv != hf.linear_num_value_heads // 2 or dn.Dk != hf.linear_key_head_dim
                        or dn.Dv != hf.linear_value_head_dim or dn.K != hf.linear_conv_kernel_dim
                        or dn.qkv_dim_tp != (2 * hf.linear_num_key_heads * hf.linear_key_head_dim
                                            + hf.linear_num_value_heads * hf.linear_value_head_dim) // 2):
                    raise ValueError("physical GDN geometry does not match scheduler specs")
    def _require_prefix_cache(self):
        if _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") != "1" or not getattr(self, "_prefix_cache_initialized", False):
            raise ValueError("prefix caching was not initialized")
        self._validate_prefix_cache_model()

    def prefix_cache_snapshot(self, slots):
        self._require_prefix_cache()
        self._ph_flush()
        return self.model[0].prefix_cache_snapshot(slots)

    def prefix_ckpt_resolve(self, items):
        """QWEN36_PREFIX_CKPT_DEFER: read back the staged checkpoints among `items` (runner, next engine step)."""
        self._require_prefix_cache()
        self._ph_flush()
        return self.model[0].prefix_ckpt_resolve(items)

    def prefix_take_mid_snapshots(self):
        """QWEN36_PREFIX_ONESTEP: the in-step checkpoints of the last prefill (request id -> snapshot), cleared."""
        self._require_prefix_cache()
        m = self.model[0]
        out, m._prefix_mid_snaps = dict(getattr(m, "_prefix_mid_snaps", None) or {}), {}
        return out

    def prefix_cache_restore(self, slots, snapshots):
        self._require_prefix_cache()
        self._ph_flush()
        self.model[0].prefix_cache_restore(slots, snapshots)
    # Read by the plugin's check_perform_device_sampling: the 1x2-mesh device sampler is an ARGMAX
    # (QWEN36_DEVICE_ARGMAX=1), so a step is sampled on device only when EVERY resident row is
    # greedy (temperature == 0); any other row sends the whole step to host sampling.
    # QWEN36_DEVICE_SAMPLE=1 replaces the argmax with the device SAMPLER (a Gumbel-max pick over
    # the TP-sharded vocab): then a row with temperature > 0 is served on device too, and the
    # per-row select flag picks the greedy ids for the temperature == 0 rows (byte-identical).
    _tt_device_sampling_greedy_only = not dev_sample.enabled()
    # Read by the plugin's check_perform_device_sampling: the device SAMPLER implements the
    # temperature only, so a step that asks for top_k / top_p / a penalty must fall back to the
    # host sampler (which implements them).
    _tt_device_sampling_sampled_ok = dev_sample.enabled()
    # Read by the plugin's check_perform_device_sampling: the largest top_k the device sampler can
    # apply exactly (the number of CANDIDATES it holds: the per-shard top-K of both shards).
    _tt_device_sampling_k = (2 * dev_sample.k()) if dev_sample.enabled() else 0
    # Read by the plugin's check_perform_device_sampling: the largest top_k the device sampler can
    # apply EXACTLY.  The candidate set is the union of the two shards' per-shard top-K, so it
    # contains the GLOBAL top-k only when k <= the per-shard K (a k above that could put more than
    # K of the global top-k in one shard, and that shard's list would then miss some of them).
    # A row above this bound falls back to the host sampler.
    _tt_device_sampling_topk_max = dev_sample.k() if dev_sample.enabled() else 0
    # Read by the plugin's model runner (QWEN36_MTP_SPEC_SAMPLE): when True it resolves each
    # request's rule (vllm_xargs['tt_spec_rule'], else the arm's default) and hands it down per row
    # as sampling_params.spec_rule. False: the runner keeps and sends nothing.
    _tt_spec_sample_serve = spec_sample.serve()

    def _device_argmax_active(self):
        return all(getattr(m, "_device_argmax", False) for m in self.model)

    def sample_decode_on_device(self, tt_logits, sampling_params, *args, **kwargs):
        # QWEN36_DEVICE_ARGMAX: the decode trace already produced the greedy token ids (model.py
        # _greedy_argmax_device), so there is nothing to sample; hand them back per DP rank.
        if self._device_argmax_active():
            temps = getattr(sampling_params, "temperature", None)
            if temps is not None and not dev_sample.enabled():
                bad = [t for t in (temps if isinstance(temps, (list, tuple)) else [temps]) if t is not None and float(t) != 0.0]
                if bad:
                    raise RuntimeError(
                        "QWEN36_DEVICE_ARGMAX is greedy-only but this step carries temperature>0 rows "
                        f"{bad[:4]}; the plugin gate (_tt_device_sampling_greedy_only) must route it to host"
                    )
            return [t[0] if isinstance(t, tuple) else t for t in tt_logits]
        return super().sample_decode_on_device(tt_logits, sampling_params, *args, **kwargs)

    def _validate_device_sampling_request(self, requested):
        if not requested:
            return
        if self._device_argmax_active():
            logger.info("QWEN36_DEVICE_ARGMAX: on-device GREEDY sampling accepted on the 1x2 mesh (argmax, not TopK)")
            return
        for model in self.model:
            if model.sampling is not None:
                continue
            mesh_shape = tuple(int(dim) for dim in model.mesh_device.shape)
            logits_per_device = math.ceil(model.args.vocab_size / model.num_devices)
            raise RuntimeError(
                "Qwen3.6 on-device sampling requires a certified TP topology (1x4 or 1x8) "
                f"with at most 65536 logits/device; got mesh={mesh_shape}, "
                f"vocab={model.args.vocab_size}, logits/device={logits_per_device}. "
                "Unset sample_on_device_mode for host sampling."
            )

    @classmethod
    def _ckpt_pool_reserve(cls, model_name, num_devices, base) -> int:
        """QWEN36_PREFIX_CKPT_DEFER: the KV pool gives up the staging pool's DRAM (prefix_snap.kv_reserve_tokens), so
        device DRAM headroom after warmup equals the arm without the lever. 0 when the lever is off."""
        from models.demos.blackhole.qwen36.tt import prefix_snap as _ps
        if _tt_public_getenv(_ps.ENV_DEFER, "0") != "1":
            return 0
        import json as _json
        cfg = _json.load(open(os.path.join(model_name or _tt_public_getenv("HF_MODEL", ""), "config.json")))
        cfg = cfg.get("text_config", cfg)
        tok, pool, per_tok = _ps.kv_reserve_tokens(cfg, num_devices)
        logger.info(f"QWEN36_PREFIX_CKPT_DEFER KV reserve: pool {pool / 2**20:.1f} MiB/card ({_ps.defer_slots()} entries) "
                    f"= {tok} KV tokens (reserve {tok * per_tok / 2**20:.1f} MiB/card at {per_tok:.0f} B/token/card); "
                    f"KV pool {base} -> {base - tok} tokens")
        return tok

    @classmethod
    def get_max_tokens_all_users(
        cls,
        model_name: str = "",
        num_devices: int = 1,
        tt_data_parallel: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        **kwargs,
    ) -> int:
        """All-user KV capacity (the shared paged-KV token pool).

        QWEN36_MAX_TOKENS_ALL_USERS overrides it with a FIXED pool (set per device+model from the
        tt-inference-server spec's env_vars, mirroring GEMMA4_MAX_TOKENS_ALL_USERS). This decouples
        the pool from max_model_len × max_num_seqs so ONE config serves both a single long request
        (up to max_model_len) and a batch of shorter ones (sum of lengths ≤ pool) — e.g. 524288 =
        1×256K or 8×64K. Without the override, fall back to max_model_len × max_num_seqs (the old
        per-config product) so existing single-mode specs are unchanged."""
        override = _tt_public_getenv("QWEN36_MAX_TOKENS_ALL_USERS")
        if override or max_model_len is not None:
            base = int(override) if override else int(max_model_len) * int(max_num_seqs or 1)
            return base - cls._ckpt_pool_reserve(model_name, num_devices, base)
        return super().get_max_tokens_all_users(
            model_name=model_name,
            num_devices=num_devices,
            tt_data_parallel=tt_data_parallel,
            max_num_seqs=max_num_seqs,
            **kwargs,
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return "<|vision_start|><|image_pad|><|vision_end|>"
        if modality.startswith("video"):
            return "<|vision_start|><|video_pad|><|vision_end|>"
        raise ValueError("Only image or video modality is supported")

    @classmethod
    def initialize_vllm_model(
        cls,
        hf_config,
        mesh_device,
        max_batch_size,
        max_seq_len,
        tt_data_parallel=1,
        optimizations=None,
        **kwargs,
    ):
        # Weights dir: MODEL_WEIGHTS_DIR → HF_MODEL → hf_config._name_or_path; a hub id resolves to a local snapshot.
        name_or_path = _tt_public_getenv("MODEL_WEIGHTS_DIR") or _tt_public_getenv("HF_MODEL") or hf_config._name_or_path
        if name_or_path and not os.path.isdir(os.path.expanduser(name_or_path)):
            from huggingface_hub import snapshot_download

            # When offline/CI, resolve from the local cache only so snapshot_download
            # reads the cached refs instead of reaching the HF API (refused by HF_HUB_OFFLINE=1).
            offline = _tt_public_getenv("HF_HUB_OFFLINE") == "1" or _tt_public_getenv("CI") == "true"
            name_or_path = snapshot_download(name_or_path, local_files_only=offline)
        args, model, _ = create_tt_model(
            mesh_device, max_batch_size=max_batch_size, max_seq_len=max_seq_len, hf_model=name_or_path
        )
        # Attach the TT vision tower so prefill can splice image/video embeddings (multimodal path).
        # No-op cost for text-only requests; get_image_features / get_video_features are only invoked
        # when a request actually carries pixel_values / pixel_values_videos.
        if _tt_public_getenv("QWEN36_SKIP_VISION") == "1":
            logger.warning("QWEN36_SKIP_VISION=1: vision tower not built; image/video requests will be rejected.")
        else:
            model.init_vision_model()
        if cls.model_capabilities.get("supports_prefix_caching"):
            logger.info("QWEN36_PREFIX_CACHING engaged: supports_prefix_caching=True enabled for Qwen36")
        if cls._RESIDENT:
            logger.info("QWEN36_DEVICE_RESIDENT engaged: resident decode chain and async decode capability active")
        if _mtp_host.enabled():
            # QWEN36_MTP_BATCH (docs/MTP-BATCH.md): batched MTP speculative decode. Built after the base
            # model (it reuses the embedding, final norm, LM head and device argmax) and before any KV
            # allocation / warmup. Fails closed on a stack it does not support.
            from models.demos.blackhole.qwen36.tt.mtp_batch import Qwen36MTPBatch

            model._mtp_batch = Qwen36MTPBatch(model)
        if _tt_public_getenv("QWEN36_OPMARK", "") not in ("", "0"):
            # QWEN36_OPMARK: host-side op naming for profiler windows; unset = this branch never runs.
            from models.demos.blackhole.qwen36.tt import opmark as _opmark

            _opmark.install(model)
        return cls([model], [args], mesh_device)

    def allocate_kv_cache(self, kv_cache_shape, dtype, num_layers):
        """Allocate paged KV (8 attn layers) + external GDN state; returns the 8 KV pairs.

        batch_size = max_batch_size (vLLM's max_num_seqs, threaded through initialize_vllm_model):
        the paged KV blocks (kv_cache_shape) already cover all users, and this sizes the per-slot
        GDN recurrent/conv state [B,...] + the decode kv grid. B==1 is the single-sequence path."""
        batch_size = self.model[0].args.max_batch_size
        if getattr(self.model[0], "_mb_trace_buckets", None) and _tt_public_getenv("QWEN36_PREFILL_BUCKET_EXTRA_BLOCK", "1") == "1":
            # QWEN36_PREFILL_BUCKET_TRACE: the traced masked bucket fills the WHOLE bucket of K/V through a
            # fixed-width page table whose pad entries need one scratch block the scheduler never hands out.
            # Allocate num_blocks+1 physically; vLLM's page tables stay over 0..num_blocks-1 and the model takes
            # the last block as its pad block. Gate off: this branch is skipped and the call below is unchanged.
            shape = list(kv_cache_shape)
            shape[0] = int(shape[0]) + 1
            logger.info(
                f"[prefill] allocating {shape[0]} KV blocks ({kv_cache_shape[0]} scheduler + 1 pad block for bucket traces)"
            )
            kv = self.model[0].allocate_kv_caches(shape, ttnn.bfloat16, batch_size=batch_size)
        else:
            kv = self.model[0].allocate_kv_caches(kv_cache_shape, ttnn.bfloat16, batch_size=batch_size)
        mtp = getattr(self.model[0], "_mtp_batch", None)
        if mtp is not None:
            mtp.allocate_kv(kv_cache_shape)
        self.model[0].dram_ledger("after KV + MTP KV allocation")
        return kv

    # ---------------------------------------------------------------- QWEN36_MTP_BATCH plumbing
    def _ph_flush(self):
        """QWEN36_PREFILL_DEFER_FINISH (tt/prefill_host.py): run the last prefill's parked MTP warm finish before
        anything else touches the device or the MTP slot state. One attribute test when the lever is off."""
        m = self.model[0]
        if getattr(m, "_ph_deferred", None) is not None:
            m.ph_flush_deferred()

    def release_request(self, slot):
        """Plugin hook (model_runner._release_model_request): the slot's request finished or was
        preempted. Only the MTP host state is slot-owned here; GDN state is rewritten by prefill."""
        self._ph_flush()
        mtp = getattr(self.model[0], "_mtp_batch", None)
        if mtp is not None:
            mtp.drain()  # QWEN36_MTP_OVERLAP: the in-flight step's drafts land before the slot resets
            cap = mtp.host.pop_capture(int(slot), t=getattr(mtp, "_last_step_t", None))  # QWEN36_MTP_BATCH_CAPTURE=1 only
            if cap:
                logger.info(cap)
            mtp.host.reset_slot(int(slot))
            mtp.on_release()

    def set_mtp_row_req_ids(self, req_ids):
        """Plugin hook (model_runner, decode steps only): the request id served by each decode row
        this step, for QWEN36_MTP_BATCH_TRACE=<request-id substring>. Inert without the trace."""
        mtp = getattr(self.model[0], "_mtp_batch", None)
        if mtp is not None and not mtp.defers_drain():
            mtp.drain(steady=True)  # the in-flight step's trace lines must carry ITS row ids
        if mtp is not None and mtp.host.trace:
            mtp.host.set_row_ids(req_ids)

    def take_mtp_extra_tokens(self):
        """Plugin hook (model_runner._finish_front_packed_sync): the accepted drafts of the decode
        step that just ran, per row, or None when that step was a plain decode step."""
        extra = getattr(self, "_mtp_extra", None)
        self._mtp_extra = None
        return extra

    def _mtp_decode_step(self, model, kwargs):
        """Run one batched speculative step when possible; None -> the caller runs plain decode."""
        mtp = getattr(model, "_mtp_batch", None)
        self._mtp_extra = None
        if mtp is None:
            return None
        res = mtp.maybe_step(
            kwargs.get("tokens"),
            kwargs.get("start_pos"),
            kwargs.get("page_table"),
            kwargs.get("sampling_params"),
            kwargs.get("enable_trace", True),
            kwargs.get("slot_remap") is not None,
        )
        if res is None:
            return None
        first, extras = res
        self._mtp_extra = extras
        # The step consumed the host tokens for every row, like a plain step would have.
        if hasattr(self, "_slots_prefilled_since_decode"):
            self._slots_prefilled_since_decode = set()
        return first

    @staticmethod
    def _has_visual(kwargs, pixel_key):
        """True only when the request carries REAL visual data for this modality. vLLM attaches an
        empty pixel_values placeholder to text requests for a multimodal-registered model, so a
        plain ``is not None`` check misclassifies text as multimodal. Mirrors the emptiness test in
        _gather_user_visual (key absent / empty list / first item None => text-only)."""
        v = kwargs.get(pixel_key)
        return v is not None and len(v) > 0 and v[0] is not None

    @staticmethod
    def _gather_user_visual(kwargs, pixel_key, grid_key):
        """Pull this (B=1) user's patches + (t,h,w) grids for one modality out of the vLLM kwargs.

        Returns (pixel_values, grid_thw) or None when the request carries nothing for this modality.
        Multiple items for the user arrive as lists; concat the patches and stack the grids (same
        shape get_image_features / get_video_features expect: [num_patches, patch_dim] + [N, 3]).
        """
        if pixel_key not in kwargs or len(kwargs[pixel_key]) == 0 or kwargs[pixel_key][0] is None:
            return None
        pixel_values = kwargs[pixel_key][0]
        grid_thw = kwargs[grid_key][0]
        if isinstance(pixel_values, list) and len(pixel_values) > 0:
            pixel_values = torch.concat(pixel_values, dim=0)
            grid_thw = torch.stack([g.to(dtype=torch.int32) for g in grid_thw], dim=0)
        return pixel_values, grid_thw

    def _compute_vision_tokens(self, model, kwargs):
        """Run the vision tower for this (single-user, B=1) request, if it carries images or video.

        Mirrors the Qwen3-VL generator's multimodal check: pull this user's pixels + grid out of
        the vLLM kwargs and return the packed embeddings (ttnn [num_vision_tokens, H]) for prefill
        to splice in. Returns None for a text-only request, so the whole multimodal path is skipped.

        Image and video share the vision tower; dispatching to get_video_features (vs
        get_image_features) is what tells the model to splice into video_token_id placeholders and
        build the video M-RoPE. A request carries at most one visual modality (see
        get_supported_mm_limits); video takes precedence if both are somehow present.
        """
        video = self._gather_user_visual(kwargs, "pixel_values_videos", "video_grid_thw")
        if video is not None:
            return model.get_video_features(*video)

        image = self._gather_user_visual(kwargs, "pixel_values", "image_grid_thw")
        if image is not None:
            return model.get_image_features(*image)

        return None

    def prefill_forward(self, tokens, page_table, kv_cache, prompt_lens, **kwargs):
        """All prefill is model-owned (Generator drives decode only)."""
        self._ph_flush()
        model = self.model[0]
        model.dram_ledger_tick()  # QWEN36_DRAM_LEDGER: the post-warmup ledger on the first served prefill, then every N
        # Mirrors Generator.prefill_forward_text:1099-1103, the ONLY writer of this set in the
        # tree -- and every Qwen prefill branch below bypasses that method (grep -rn across
        # models/demos/blackhole/qwen36/ returns zero occurrences). The device-sampling merge at
        # Generator.decode_forward:2146 relies on this set to exclude freshly prefilled slots
        # from use_dev; with it always empty, a slot just handed to a NEW request can be fed the
        # PREVIOUS occupant's token at that slot's first decode step. Gated on on_device_sampling
        # upstream, which is why argmax-OFF arms never showed it.
        _slots = kwargs.get("empty_slots")
        if not hasattr(self, "_slots_prefilled_since_decode"):
            self._slots_prefilled_since_decode = set()
        _new = set(range(tokens.shape[0])) if _slots is None else set(int(s) for s in _slots)
        self._slots_prefilled_since_decode.update(_new)
        _mtp = getattr(model, "_mtp_batch", None)
        if _mtp is not None:
            _mtp.drain()  # QWEN36_MTP_OVERLAP: finish the in-flight step before its slots are reused
            for _s in _new:
                _cap = _mtp.host.pop_capture(_s, t=getattr(_mtp, "_last_step_t", None))  # a release that never came
                if _cap:
                    logger.info(_cap)
                _mtp.host.reset_slot(_s)
            # QWEN36_MTP_LOOKUP: seed each new request's id history with its prompt (row u -> slot _slots[u]).
            # A prompt that is not the full history (e.g. a prefix-cache hit prefills only the tail) breaks the
            # host's position check at the first decode step, which turns lookup off for that request.
            if _mtp.host.lookup_enabled():
                _order = list(range(tokens.shape[0])) if _slots is None else [int(s) for s in _slots]
                for _u, _s in enumerate(_order):
                    _n = int(prompt_lens[_u]) if prompt_lens is not None else int(tokens.shape[1])
                    _mtp.host.lookup_seed(_s, tokens[_u, :_n].tolist())
        if not getattr(self, "_prefilled_slots_fix_banner", False):
            self._prefilled_slots_fix_banner = True
            logger.info(
                f"QWEN36_PREFILLED_SLOTS_FIX engaged: recording prefilled slots {sorted(_new)} "
                "for the next decode reset (previously never written on the Qwen path)"
            )
        if hasattr(self, "_trace_buffer_last_step"):
            self._trace_buffer_last_step.clear()
        if _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") == "1" and kwargs.get("prefix_finals") is not None:
            # The S12 prefix arm's runner (prefill-chunk-end checkpoints, docs/SERVING.md section 5c): the shipped
            # batched path (prefill_paged_slots: slot-copy trace, MTP warm span, deferred finish) resumed per row at
            # its chunk start. The older prefix runner (patches/prefix-local) passes no prefix_finals and keeps
            # prefill_prefix_slots below.
            self._require_prefix_cache()
            if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
                raise ValueError("prefix caching is text-only")
            if not (model.num_devices > 1 and model.args.max_batch_size > 1):
                raise ValueError("the S12 prefix path is the TP batched path (num_devices > 1, max_num_seqs > 1)")
            prefix = {
                "starts": [int(x) for x in kwargs["start_pos"]],
                "states": list(kwargs["prefix_cache_states"]),
                "hits": [int(x) for x in kwargs["prefix_hits"]],
                "totals": [int(x) for x in kwargs["prefix_totals"]],
                "finals": [bool(x) for x in kwargs["prefix_finals"]],
                "next_tokens": [int(x) for x in kwargs["prefix_next_tokens"]],
                "req_ids": list(kwargs["prefix_req_ids"]),
            }
            if kwargs.get("prefix_mids") is not None:
                prefix["mids"] = [int(x) for x in kwargs["prefix_mids"]]  # QWEN36_PREFIX_ONESTEP
            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"),
                                                    prefix=prefix)
        if _tt_public_getenv("QWEN36_PREFIX_CACHING", "0") == "1":
            self._require_prefix_cache()
            if self._has_visual(kwargs, "pixel_values") or self._has_visual(kwargs, "pixel_values_videos"):
                raise ValueError("prefix caching is text-only")
            logits = model.prefill_prefix_slots(tokens, page_table, prompt_lens, kwargs.get("empty_slots"),
                                                kwargs.get("start_pos"), kwargs.get("prefix_cache_states"))
            return logits, torch.zeros(tokens.shape[0], dtype=torch.long)
        if kwargs.get("prefix_cache_states") is not None:
            raise ValueError("prefix state supplied while prefix caching is disabled")
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            # Batched text prefill into decode slots (MM is B=1). Require real visual data, not a
            # non-None empty pixel_values placeholder from vLLM on text requests.
            assert not self._has_visual(kwargs, "pixel_values") and not self._has_visual(
                kwargs, "pixel_values_videos"
            ), (
                "batched (max_num_seqs>1) serving is text-only; multimodal is single-sequence "
                "(max_concurrency=1). Run the model at max_num_seqs=1 for image/video requests."
            )
            return self._prefill_forward_tp_batched(model, tokens, page_table, prompt_lens, kwargs.get("empty_slots"))
        vision_tokens = self._compute_vision_tokens(model, kwargs)
        if model.num_devices > 1:
            return self._prefill_forward_tp(model, tokens, page_table, prompt_lens, vision_tokens=vision_tokens)
        seq_len = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        logger.info(f"Prefilling User 1 up to {seq_len} tokens")
        # Multimodal works WITH the captured trace here: prefill_dispatch routes to the traced
        # path, which splices the image/video rows via a fixed-shape ttnn.where over persistent
        # buffers (compiled at warmup, updated per request by copy_host_to_device — no request-time
        # compile).
        logits = prefill_dispatch(
            model,
            tokens,
            page_table,
            prompt_lens,
            use_trace=kwargs.get("enable_trace", False),
            vision_tokens=vision_tokens,
        )
        logits = ttnn.to_torch(logits)
        # The vLLM runner unpacks (logits, rope_deltas) because the HF config has mrope_section.
        # Zero deltas are returned for all modalities: the multimodal M-RoPE delta is applied
        # entirely model-side (build_request_rope stashes self.rope.rope_delta during prefill, and
        # every decode path offsets the rope position by it), so the value handed back to vLLM is
        # unused for device-side rope and stays zero.
        rope_deltas = torch.zeros(logits.shape[0], dtype=torch.long)
        logger.info(f"Finished prefill up to {seq_len} tokens, starting decode...")
        return logits, rope_deltas

    def _prefill_forward_tp(self, model, tokens, page_table, prompt_lens, vision_tokens=None):
        """TP (B=1) paged prefill via the model-owned masked fixed-bucket path.

        prefill_traced_chunked rounds the prompt up to a fixed bucket and masks the GDN to the
        EXACT valid_len, so prefill runs one of a bounded, pre-warmed program set (the
        compile-clobbers-trace fix) — for <=2048 prompts it is entirely the masked bucket (no
        chunk trace needed). Longer prompts replay the chunk-outer trace (Milestone B). Returns
        host logits [1, 1, vocab] gathered to a single replica."""
        T = int(prompt_lens[0]) if prompt_lens is not None else tokens.shape[1]
        if tokens.shape[1] > T:
            tokens = tokens[:, :T]
        logger.info(f"Prefilling User 1 up to {T} tokens (TP masked-bucket/chunked)")
        # Multimodal is supported on TP too: prefill_traced_chunked splices the image/video rows via
        # a fixed-shape ttnn.where over hidden-sharded persistent buffers (the vision rows are
        # gathered to full hidden on host, placed along seq, then re-sharded), so no request-time
        # compile clobbers the parked trace.
        logits = model.prefill_traced_chunked(
            tokens, page_table, actual_len=T, vision_tokens=vision_tokens
        )  # [1,1,vocab] replicated
        logits = (
            ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(model.mesh_device, dim=0))
            .reshape(-1, model.args.vocab_size)[:1]
            .float()
            .view(1, 1, -1)
        )
        logger.info(f"Finished prefill up to {T} tokens, starting decode...")
        return logits, torch.zeros(1, dtype=torch.long)

    def _prefill_forward_tp_batched(self, model, tokens, page_table, prompt_lens, empty_slots, prefix=None):
        """TP batched (max_num_seqs>1) prefill: prefill each request in this step into its decode slot.

        vLLM prefills new requests while other slots decode, so each user's B=1 state is written into
        row empty_slots[u] of the batched GDN buffers without disturbing the live rows (model-owned,
        via prefill_paged_slots). Attention fills each request's blocks via its page-table row.

        tokens:      torch [N, max_T] (rows are the N requests scheduled this prefill step).
        page_table:  torch [N, max_blocks] — row u = request u's blocks.
        prompt_lens: per-request real lengths (row u trimmed to prompt_lens[u]).
        empty_slots: per-request decode slot; defaults to range(N) (mirrors Generator.prefill_forward_text).
        Returns ([N, 1, vocab] host logits, [N] zero rope_deltas — text M-RoPE delta is 0, applied model-side).
        """
        N = tokens.shape[0]
        plens = [int(prompt_lens[u]) for u in range(N)] if prompt_lens is not None else [tokens.shape[1]] * N
        if empty_slots is None:
            empty_slots = list(range(N))
        empty_slots = [int(s) for s in empty_slots]
        token_ids_list = [tokens[u : u + 1, : plens[u]].to(torch.int32) for u in range(N)]
        pt = page_table if isinstance(page_table, torch.Tensor) else ttnn.to_torch(page_table)
        logger.info(f"Prefilling {N} user(s) into slots {empty_slots} (TP batched masked-bucket)")
        host_logits = model.prefill_paged_slots(token_ids_list, pt, empty_slots, valid_lens=plens, prefix=prefix)
        logits = torch.cat([hl.reshape(1, 1, -1) for hl in host_logits], dim=0)  # [N, 1, vocab]
        logger.info(f"Finished batched prefill of {N} user(s), starting decode...")
        return logits, torch.zeros(N, dtype=torch.long)

    def decode_forward(self, *args, **kwargs):
        # Traced decode (single-device and TP): trace captured at pos 0 in warmup, replayed here.
        # Valid for TP — GDN state is in fixed in-place buffers, and prefill only replays pre-warmed programs.
        self._ph_flush()
        if not getattr(self, "_decode_logged", False):
            self._decode_logged = True
            logger.info("Decode trace replay active (Qwen)")
        model = self.model[0]
        if self._RESIDENT and _tt_public_getenv("QWEN36_RESIDENT_IGNORE_RELOAD") == "1":
            # FAULT INJECTION, and it must witness itself. R2 (resident) and R3 (resident +
            # this flag) previously differed by an env var that printed NOTHING, so a passing
            # R3 could not be distinguished from "the flag never applied" -- the exact failure
            # the counted-banner rule exists to catch.
            #
            # The count is the diagnostic that matters, not the flag. Dropping reload_inputs can
            # only corrupt the chain on steps where vLLM actually ASKED for a reload. If
            # dropped=0 across a whole run, the schedule never invalidated the chain and R3 is
            # vacuous: it passed because there was nothing to break, not because the lever is safe.
            _had = kwargs.get("reload_inputs") is not None
            self._ignore_reload_seen = getattr(self, "_ignore_reload_seen", 0) + 1
            self._ignore_reload_dropped = getattr(self, "_ignore_reload_dropped", 0) + int(_had)
            if self._ignore_reload_seen == 1 or _had:
                # f-string: loguru does not %-interpolate (see generator.py).
                logger.info(
                    f"QWEN36_RESIDENT_IGNORE_RELOAD engaged: fault injected; decode steps "
                    f"seen={self._ignore_reload_seen}, "
                    f"reload_inputs actually dropped={self._ignore_reload_dropped}"
                )
            kwargs.pop("reload_inputs", None)
            kwargs["reset_batch"] = False
        # Batched serving: apply vLLM's condense slot_remap to the per-slot GDN recurrent/conv state
        # BEFORE the decode trace reads it. The plugin remaps its own buffers (and the seed RNG via
        # super().decode_forward), but GDN state is model-internal, so mirror the same reindex here.
        # slot_remap is passed through unchanged so the seed-RNG remap inside super() still runs.
        if getattr(model, "_mtp_batch", None) is not None:
            # QWEN36_MTP_OVERLAP: the previous speculative step's drafts land (and its slot state is final)
            # before a remap moves slots or this step plans from them.
            if kwargs.get("slot_remap") is not None or args or not model._mtp_batch.defers_drain():
                model._mtp_batch.drain(steady=kwargs.get("slot_remap") is None)
        if model.num_devices > 1 and model.args.max_batch_size > 1:
            slot_remap = kwargs.get("slot_remap")
            if slot_remap is not None:
                model._remap_gdn_slots(slot_remap)
                if getattr(model, "_mtp_batch", None) is not None:
                    model._mtp_batch.host.remap(slot_remap)
        if getattr(model, "_mtp_batch", None) is not None:
            # QWEN36_MTP_BATCH: the arm is sync + non-resident, so the host tokens are authoritative
            # every step. The Generator's reset_batch merge would otherwise read the plain decode
            # trace's input buffers, which are stale after speculative steps (they never ran).
            if "reset_batch" in kwargs:
                kwargs["reset_batch"] = False
            if not args:
                spec = self._mtp_decode_step(model, kwargs)
                if spec is not None:
                    return spec
        # Decode bucketing (default on; TT_DECODE_BUCKETING=0 off): slice host inputs to the
        # smallest power-of-2 width >= active prefix [0:num_active) before the base forward.
        # No runner edit / output re-pad — plugin reads unpadded_batch_size in slot order.
        # Each width keeps its own trace metadata, inputs, and output.
        args = list(args)

        def _read(name, pos):
            if name in kwargs:
                return kwargs[name]
            return args[pos] if pos < len(args) else None

        def _write(name, pos, val):
            if name in kwargs:
                kwargs[name] = val
            elif pos < len(args):
                args[pos] = val

        tokens = _read("tokens", 0)
        if _tt_public_getenv("TT_DECODE_BUCKETING", "1") == "1" and tokens is not None:
            start_pos = _read("start_pos", 1)
            width = int(tokens.shape[0])
            num_active = int((start_pos != -1).sum()) if start_pos is not None else width
            num_active = max(1, min(num_active, width))
            bucket = min(width, 1 << max(0, (num_active - 1).bit_length()))  # smallest pow2 >= num_active
            # Keep full width when slot_remap is set: remap indexes the full slot space (tokens /
            # GDN). Rare (row moves only); bucketing resumes next step. Check kw + positional #10.
            if _read("slot_remap", 10) is not None:
                bucket = width
            if bucket < width:
                _write("tokens", 0, tokens[:bucket])
                if start_pos is not None:
                    _write("start_pos", 1, start_pos[:bucket])
                page_table = _read("page_table", 2)
                if page_table is not None:
                    _write("page_table", 2, page_table[:bucket])
                tokens = tokens[:bucket]

        if tokens is not None:
            B = int(tokens.shape[0])
            store = getattr(self, "_bucket_trace_store", None)
            if store is None:
                store = self._bucket_trace_store = {}
            if B not in store:
                store[B] = (defaultdict(lambda: None), defaultdict(lambda: None), defaultdict(lambda: None))
            self.trace_ids_decode, self.trace_inputs_decode, self.trace_output_decode = store[B]
            # under resident decode every bucket's trace inputs chain on device on their own.
            # A bucket switch that vLLM does not see as a layout change (the slot_remap step above
            # forces full width, the next step drops back) would replay the new bucket's STALE
            # inputs -- garbage positions hang the device, old ones corrupt silently. The device
            # copy in the previous bucket is authoritative (sync and async), so carry its first B
            # token/position rows over and force a reload of this bucket's inputs.
            prev_B = getattr(self, "_last_decode_bucket", None)
            self._last_decode_bucket = B
            if (
                self._RESIDENT
                and prev_B is not None
                and prev_B != B
                and self.trace_ids_decode[True]  # captured: warmup's first visit is left alone
                and not kwargs.get("reload_inputs")
                and not kwargs.get("reset_batch")
                and _tt_public_getenv("QWEN36_RESIDENT_BUCKET_SYNC", "1") == "1"
            ):
                prev_in = store[prev_B][1][True]
                if prev_in is not None:
                    tr = prev_in[0]
                    n = min(B, prev_B)  # rows past the old width have no device copy: keep host's
                    d_tok = ttnn.to_torch(ttnn.get_device_tensors(tr[0])[0]).reshape(-1)[:n]
                    d_pos = ttnn.to_torch(ttnn.get_device_tensors(tr[1])[0]).reshape(-1)[:n]
                    tokens = tokens.clone()
                    tokens.view(-1)[:n] = d_tok.to(tokens.dtype)
                    _write("tokens", 0, tokens)
                    sp = _read("start_pos", 1).clone()
                    sp.view(-1)[:n] = d_pos.to(sp.dtype)
                    _write("start_pos", 1, sp)
                    kwargs["reload_inputs"] = True
                    self._bucket_sync_hits = getattr(self, "_bucket_sync_hits", 0) + 1
                    if self._bucket_sync_hits == 1 or self._bucket_sync_hits % 100 == 0:
                        logger.info(
                            f"QWEN36_RESIDENT_BUCKET_SYNC engaged: bucket {prev_B}->{B} carried device "
                            f"tok/pos rows over (hits={self._bucket_sync_hits})"
                        )
            # Key the sampling trace by bucket width too: Generator binds it to one logits tensor
            # by identity, and each decode-bucket width has its own.
            for _m in self.model:
                _sm = getattr(_m, "sampling", None)
                if _sm is not None and hasattr(_sm, "set_trace_bucket"):
                    _sm.set_trace_bucket(B)
        if dev_sample.enabled():
            # QWEN36_DEVICE_SAMPLE: the model needs the per-row seed and temperature to draw the
            # per-step Gumbel noise (the request's seed and the row's position key it, so the sample
            # is deterministic per request).  The plugin passes sampling_params only when it samples
            # on device; the host-sampling trace does not need them.
            for _m in self.model:
                _m._ds_sampling = kwargs.get("sampling_params")
        return super().decode_forward(*args, **kwargs)

    def warmup_model_prefill(self, kv_cache, enable_trace, *args, **kwargs):
        # Capture the chunk-prefill trace + warm the masked-bucket set so requests only replay
        # pre-compiled programs (compile-clobbers-trace fix). Guard name must match the plugin's reset.
        # QWEN36_PREFIX_CKPT_DEFER: the checkpoint staging pool must exist before any trace capture (a device buffer
        # allocated later and alive across a replay can alias the replay's baked intermediates). The runner's first
        # warmup call is this one (enable_trace=False), before every decode / prefill capture.
        self.model[0].prefix_ckpt_pool_alloc()
        if not enable_trace:
            return
        if getattr(self, "already_warmed_up_prefill", False):
            return
        self.already_warmed_up_prefill = True
        # Size the chunk-trace page table to the full KV cache (not a hardcoded 4096) so served ISL
        # isn't capped; still captures one chunk — just a bigger page-table tensor.
        if kv_cache:
            # Round to a multiple of 32: paged/chunked SDPA needs the page-table stick % 32 == 0.
            num_blocks = math.ceil(int(kv_cache[0][0].shape[0]) / 32) * 32
        else:
            num_blocks = math.ceil(_PREFILL_WARMUP_BUCKET / _BLOCK_SIZE)
        page_table = torch.arange(num_blocks, dtype=torch.int32).reshape(1, num_blocks)
        model = self.model[0]
        chunk = _prefill_warmup_chunk()
        # Batched serving (max_num_seqs>1): the decode buffers are [B,...], but prefill runs B=1. Bind
        # the PERSISTENT B=1 GDN prefill scratch and capture the chunk trace against IT, so long prompts
        # (>chunk_size) replay the traced chunk-outer path per user instead of the slower eager fallback.
        # The scratch is not freed (prefill_paged_slots rebinds it per request); the batched decode
        # buffers are restored before the decode-trace warmup captures at [B,...].
        batched = model.num_devices > 1 and model.args.max_batch_size > 1
        logger.info(
            f"Starting Qwen prefill warmup: chunk-prefill trace{' (batched, B=1 scratch)' if batched else ''} "
            f"(chunk={chunk}, page_table_blocks={num_blocks})..."
        )
        prev = model._bind_gdn_prefill_scratch() if batched else None
        try:
            model.capture_prefill_trace_chunked(
                self.mesh_device, page_table, chunk_size=chunk, capture_chunk_trace=True
            )
        finally:
            if prev is not None:
                model._unbind_gdn_prefill_scratch(prev)

    def warmup_model_decode(self, *args, **kwargs):
        # Defer to WarmupForwardMixin, which warms the paged-SDPA + GDN decode path at pos 0.
        # Drop stale `non_greedy_decoding_on_device` from the old vLLM plugin; no-op for Qwen.
        kwargs.pop("non_greedy_decoding_on_device", None)
        self._validate_device_sampling_request(kwargs.get("can_sample_on_device", False))
        if kwargs.get("can_sample_on_device") and self._device_argmax_active():
            # Only the greedy (temperature 0) and host (None) variants exist for the argmax sampler.
            kwargs["greedy_only"] = True
        # QWEN36_GDN_DECODE_CHAIN (gdn/decode_chain.py): fail-closed bitwise self-test before any decode compile.
        from models.demos.blackhole.qwen36.tt.gdn import decode_chain as _gdc

        _gdc.warmup_selftest(self.model[0], logger)
        result = warmup_decode_buckets(self, super().warmup_model_decode, *args, **kwargs)
        mtp = getattr(self.model[0], "_mtp_batch", None)
        if mtp is not None:
            # Phase 1 (enable_trace=False) compiles every speculative program; phase 2 captures the
            # verify + commit/draft traces AFTER the plain decode traces, with nothing left to compile.
            nblk = kwargs.get("num_blocks")
            if not isinstance(nblk, int) or nblk <= 0:
                raise RuntimeError(f"QWEN36_MTP_BATCH warmup needs the page-table width (num_blocks={nblk!r})")
            if kwargs.get("enable_trace", False):
                mtp.capture(nblk)
            else:
                mtp.compile(nblk)
        if kwargs.get("enable_trace", False) and self.model[0]._ph.any_request_path:
            # QWEN36_PREFILL_TAIL_NOSYNC / _DEFER_FINISH (tt/prefill_host.py): fail-closed equivalence self-test in
            # the served state (every trace captured), before the first request.
            nblk = kwargs.get("num_blocks")
            if not isinstance(nblk, int) or nblk <= 0:
                raise RuntimeError(f"pp/host self-test needs the page-table width (num_blocks={nblk!r})")
            self.model[0].prefill_host_selftest(nblk)
        return result
